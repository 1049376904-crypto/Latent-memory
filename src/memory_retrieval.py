#!/usr/bin/env python3
"""
最小检索接口参考实现（设计笔记"三层检索设计"的前两层）。

接 chunking 实验切好的块，实现 retrieve(query, topN) → [相关片段+元数据]：
  第一层 BM25 关键词：Okapi BM25，字符 bigram 当词（中文零依赖分词的常规做法）
  第二层 向量语义：bge-small-zh-v1.5（fastembed，--embed）；零依赖时退回 bigram 余弦代理
  第三层 关系图谱：零依赖词面代理版（任务卡"检索层关系图谱"）——共享显著词
    （低 df 的 bigram，"望远镜"这类专名的碎片）即相连，链接强度=Σidf；检索时
    把种子命中的关联块按强度排成第三路名次 append 进 RRF，融合逻辑不改。
    诚实标注：是词面代理不是实体识别/语义图谱，跟 bigram 当分词同档次的近似

两层排名用 RRF（Reciprocal Rank Fusion）融合——BM25 分和余弦分量纲完全不同，
RRF 只看名次不看绝对分，避开脆弱的跨层分数归一化（Elasticsearch 等 hybrid 检索同款做法）。

用进废退（README/设计笔记）：被 retrieve 命中（进 topN）的记忆权重 +0.05，
权重乘进融合分——越常被提起的越容易被搜到，符合人类记忆规律。

换窗/压缩召回（任务卡"换窗压缩记忆召回"）：retrieve 之外平级的第二种检索模式
recall_recent(topN)——换新窗口/context 压缩这两个场景没有 query 可用，语义相关性
无从算起，改用 时间新鲜度×用进废退权重 排序（设计结论已定，不引语义相似度）。
两个触发点的接线在同目录 session_recall.py。

零依赖优先：BM25 层、RRF、用进废退、时间戳解析全是 stdlib；只有真 embedding 走
fastembed。selftest 不需要装任何东西。

用法：
  python memory_retrieval.py --selftest                      # 零依赖自检
  uv run --with fastembed python memory_retrieval.py --selftest --embed
  uv run --with fastembed python memory_retrieval.py --embed --corpus <md目录> --query "..."

本仓库不含私人语料：真实验证在本地对 timeline 目录跑，语料不入库，
这里只带合成语料的 selftest。
"""

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 复用 chunking 实验里已验证的切块与 embedding（同目录模块，import 不触发它的 CLI）
from chunking_experiment import chunk_heading, bigram_counts, cosine, embed_texts

# recall_recent 的半衰期默认值（天）：经验值——timeline 语料按天/窗口推进，一周前的
# 上下文对"接上刚才"基本没用了。跟 +0.05/k=60 同一待遇，等真实评估再调。
RECALL_HALF_LIFE_DAYS = 7.0


def tokenize(text: str):
    """BM25 用的中文零依赖分词：清标点/空白后取字符 bigram 当词。"""
    s = re.sub(r"[^\w一-鿿]", "", text.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    """Okapi BM25。tf 饱和(k1)+长度归一(b)+平滑 idf(不产负值)。"""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / self.N if self.N else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        self.df = Counter()
        for d in docs_tokens:
            self.df.update(set(d))

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)  # +1 保证非负

    def scores(self, query_tokens):
        out = []
        for i in range(self.N):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in set(query_tokens):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def ranked(scores):
    """分数列表 → 索引名次（高分在前，同分按索引稳定）。"""
    return [i for i, _ in sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))]


def rrf_fuse(rank_lists, weights, k=60):
    """RRF 融合多路名次 + 用进废退权重。
    rank_lists: [[chunk_idx 从好到坏], ...]；weights: 每个 chunk 的权重乘子。
    返回 [(chunk_idx, 融合分), ...] 按分降序。"""
    fused = defaultdict(float)
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + rank + 1)
    for idx in fused:
        fused[idx] *= weights[idx]  # 用进废退：命中累积的权重乘进融合分
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class MemoryIndex:
    """记忆检索索引：add 片段 → build → retrieve。权重随命中浮沉（用进废退）。"""

    def __init__(self, embed=False, weight_boost=0.05, rrf_k=60,
                 graph_topK=5, graph_seeds=3):
        # graph_topK/graph_seeds 是经验值（每块存几个邻居/两路各取几个种子扩展），
        # 跟 +0.05/k=60 同一待遇，等真实检索质量评估再调
        self.embed = embed
        self.weight_boost = weight_boost
        self.rrf_k = rrf_k
        self.graph_topK = graph_topK
        self.graph_seeds = graph_seeds
        self.chunks, self.meta, self.weights = [], [], []
        self._bm25 = self._cvecs = self._ccounts = None
        self._neighbors = {}

    def add(self, text, meta=None):
        self.chunks.append(text)
        self.meta.append(meta or {})
        self.weights.append(1.0)

    def build(self):
        self._bm25 = BM25([tokenize(c) for c in self.chunks])
        if self.embed:
            self._cvecs = embed_texts(self.chunks)          # 单位化向量矩阵
        else:
            self._ccounts = [bigram_counts(c) for c in self.chunks]
        self._neighbors = self._build_graph()
        return self

    def _build_graph(self):
        """第三层·关系图谱（零依赖词面代理版）：共享"显著词"的块相连。

        显著词 = 出现在少数几个块里的 bigram（2 ≤ df ≤ max(3, N//5)——"望远镜"这类
        专名的碎片会同时出现在约定/兑现/后续几个块里，而"今天""然后"这类通用词
        df 高，被上限挡掉；上限是经验值，待真实评估再调）。链接强度 = 共享显著词
        的 Σidf——越稀有的共享词越说明两块讲的是同一个具体事物。

        每块只存强度前 graph_topK 的邻居。诚实标注局限：这是词面代理，"约定"和
        "兑现"若不共享任何字面（连专名都换了说法），这版连不上——那是实体识别/
        语义图谱的活，跟检索层 bigram 当分词一样，等真实评估再升级，现在不假装。"""
        docs = self._bm25.docs
        inv = defaultdict(list)
        for i, d in enumerate(docs):
            for t in set(d):
                inv[t].append(i)
        max_df = max(3, self._bm25.N // 5)
        link = defaultdict(float)
        for t, ds in inv.items():
            if not 2 <= len(ds) <= max_df:
                continue
            w = self._bm25.idf(t)
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    link[(ds[a], ds[b])] += w
        by_node = defaultdict(list)
        for (a, b), s in link.items():
            by_node[a].append((s, b))
            by_node[b].append((s, a))
        return {i: [j for _, j in sorted(v, key=lambda x: (-x[0], x[1]))][:self.graph_topK]
                for i, v in by_node.items()}

    def _vector_scores(self, query):
        if self.embed:
            qv = embed_texts([query], is_query=True)[0]
            return list(self._cvecs @ qv)                   # 已单位化，点积即余弦
        q = bigram_counts(query)
        return [cosine(q, cc) for cc in self._ccounts]

    def graph_neighbors(self, chunk_idx):
        """与 chunk_idx 关联的块索引，按链接强度降序（"望远镜"→"3.14约定"→
        "3.20兑现"这类：共享显著词即相连，见 _build_graph）。没有关联返回空。"""
        return self._neighbors.get(chunk_idx, [])

    def recall_recent(self, topN=5, half_life=None, now=None):
        """无 query 的主动召回：score = 时间新鲜度 × 用进废退权重。

        跟 retrieve() 平级的第二种检索模式——换新窗口/context 压缩这两个场景里
        没有 query，语义相关性无从算起（相关性是"对一个 query 而言相关"），所以
        这里刻意不碰 BM25/向量，只用两个不依赖 query 的信号（任务卡设计结论已定）：
          时间新鲜度：recency = 0.5 ** (距今天数 / half_life)，半衰期指数衰减
          用进废退权重：weights[idx]，被 retrieve 反复命中过的记忆本来就带"重要"信号

        half_life 单位天，默认 RECALL_HALF_LIFE_DAYS（经验值，跟 +0.05/k=60 一样等
        真实评估再调）。now 可注入固定时刻，selftest 确定性用。只读 meta/weights，
        不需要先 build()。

        兜底：meta 没有 timestamp 的块 recency 记 0——不崩、沉底但 topN 够大时仍会
        返回，缺时间戳的块之间按权重排（排序键里权重当第二关键字）。

        乘法 vs 加权和（任务卡测试第 3 条：不预设答案，实测后定，过程如实记）：
        极端对比实测（selftest 第 9 项）——刚发生但从未被命中的块（w=1.0, recency≈1）
        对 30 天前但权重 2.0 的块（recency≈0.05），乘法下新块 1.0 : 0.10 碾压。
        判断合理，保留乘法，理由：
          1. 这个场景要救的是"刚被换窗/压缩冲掉的最近上下文"，旧而重要的记忆本来
             就不是这一刻丢的东西——真需要它时自然有 query，走 retrieve() 主线；
          2. 乘法下权重仍在新鲜度相近的块之间起排序作用（selftest 第 8 项），正是
             "最近+曾经重要"的本意：权重是同龄块间的裁判，不是对抗新鲜度的杠杆；
          3. 改加权和需要把无上界的 weight 和 0~1 的 recency 跨量纲归一化——正是
             检索层当初选 RRF 就为避开的那类脆弱操作，不重蹈；
          4. 真想让旧记忆浮上来，调大 half_life 就够（selftest 第 9 项后半：
             half_life=1000 天时高权重旧块反超），信号配比有旋钮，不用换公式。

        召回命中不加 weight_boost：被系统自动带回≠被用户主动提起。若加，每次换窗
        都机械抬高最近块的权重，形成"最近的越来越重"的自激循环，污染 retrieve()
        的用进废退信号。

        返回 [{id, text, meta, score}]（任务卡定的形状）。"""
        half_life = RECALL_HALF_LIFE_DAYS if half_life is None else half_life
        now = time.time() if now is None else now
        scored = []
        for i, meta in enumerate(self.meta):
            ts = meta.get("timestamp")
            if ts is None:
                recency = 0.0  # 兜底：没时间戳当"无限旧"，不猜不崩
            else:
                recency = 0.5 ** (max(0.0, now - ts) / 86400.0 / half_life)
            scored.append((i, recency * self.weights[i]))
        scored.sort(key=lambda x: (-x[1], -self.weights[x[0]], x[0]))
        return [{"id": i, "text": self.chunks[i], "meta": self.meta[i], "score": s}
                for i, s in scored[:topN]]

    def retrieve(self, query, topN=5, reranker=None, coarse_topM=20):
        """query → 前 topN 个相关片段 [{id, text, meta, score, weight}]。
        命中的片段权重 +weight_boost（用进废退），作为副作用记在 index 上。

        二阶段检索（设计笔记 待验证问题 2，任务卡"粗筛重排序两阶段"）：
        reranker 可插拔（同 draft_extraction.llm_call 的先例）——不传行为不变
        （一阶段）；传入 (query, texts)->scores 回调时，三路 RRF 融合当粗筛取前
        coarse_topM（经验值待调），reranker 对这批候选精排后取 topN。真正的重排
        该用 cross-encoder，将来从同一个槽换进来。三条规则：
          精排分数乘用进废退权重——跟 RRF 融合处同一规则，重排不豁免；
          同分按粗筛名次——精排没意见时保留粗筛信号，不乱序；
          weight_boost 只加在最终 topN——粗筛池是机制内部产物，不算"被命中"。"""
        bm_scores = self._bm25.scores(tokenize(query))
        vec_scores = self._vector_scores(query)
        bm_ranks, vec_ranks = ranked(bm_scores), ranked(vec_scores)
        rank_lists = [bm_ranks, vec_ranks]
        # 第三层：按留口设计把关联块排成第三路 append 进 RRF，融合逻辑不改。
        # 两处实现细节是实测调出来的（过程见 设计笔记"关系图谱"一节）：
        #   种子只取真有分的命中——零分并列块不是命中，它们的邻居是纯噪声；
        #   第三路结构是"种子打头、关联块随后"——RRF 只看名次，短名次路里排头的
        #   贡献(≈1/(k+1))跟整路第一名一样大，纯邻居路会让弱关联反超真命中。
        # 已知局限：真 embedding 余弦几乎恒正，"有分才算命中"的过滤在 --embed
        # 路径基本不起作用，种子退化为固定取 top graph_seeds，待真实评估再调
        seeds = []
        for rank_list, scores in ((bm_ranks, bm_scores), (vec_ranks, vec_scores)):
            for i in rank_list[:self.graph_seeds]:
                if scores[i] > 0 and i not in seeds:
                    seeds.append(i)
        graph_route = list(seeds)
        for seed in seeds:
            for nb in self.graph_neighbors(seed):
                if nb not in graph_route:
                    graph_route.append(nb)
        if len(graph_route) > len(seeds):  # 带出了新关联块才追加，没有就退回两路融合
            rank_lists.append(graph_route)
        fused = rrf_fuse(rank_lists, self.weights, self.rrf_k)
        if reranker is not None:
            pool = fused[:coarse_topM]
            fine = reranker(query, [self.chunks[i] for i, _ in pool])
            order = sorted(range(len(pool)),
                           key=lambda j: (-fine[j] * self.weights[pool[j][0]], j))
            fused = [(pool[j][0], fine[j] * self.weights[pool[j][0]]) for j in order]
        fused = fused[:topN]

        results = [{
            "id": idx,
            "text": self.chunks[idx],
            "meta": self.meta[idx],
            "score": score,
            "weight": self.weights[idx],  # 本次排序所用的权重（+0.05 前）
        } for idx, score in fused]
        for idx, _ in fused:
            self.weights[idx] += self.weight_boost  # 用进废退：命中即加权
        return results

    # ---------- 权重持久化（任务卡"记忆写回与权重持久化"） ----------
    #
    # 用进废退权重原来只活在内存里，而 MCP server 是 stdio 进程、客户端每次会话
    # 都可能重启它——权重每次归 1.0，"被反复聊起的记忆更重"在生产形态下等于没有。
    #
    # **按内容哈希记，不按位置记**：chunk 下标随语料文件增删/重切而漂移，按下标
    # 存权重等于换一次语料就张冠李戴。按文本哈希存，文件改名、块顺序变动都不丢；
    # 正文被编辑过的块哈希变了、权重归 1 重新攒——内容都变了，旧的"被聊起次数"
    # 本来就不该继承。

    def save_weights(self, path):
        """权重落盘。只存 ≠1.0 的（稀疏）——没被聊起过的块不占地方。"""
        data = {_chunk_key(c): w
                for c, w in zip(self.chunks, self.weights) if w != 1.0}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return len(data)

    def load_weights(self, path):
        """从盘上把权重接回来，按内容哈希对号入座。文件不存在按零处理——
        第一次启动本来就没有历史。返回接上的块数。"""
        p = Path(path)
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for i, c in enumerate(self.chunks):
            w = data.get(_chunk_key(c))
            if w is not None:
                self.weights[i] = w
                n += 1
        return n


def _chunk_key(text):
    """权重持久化的键：chunk 文本的内容哈希（md5 前 16 位，非安全用途）。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ---------- chunk 时间戳（recall_recent 按新鲜度排序的前提，任务卡第 1 条） ----------

# 文件名里的完整日期：2026-07-29 / 2026.7.29 / 20260729 / 2026年7月29日 都认
_DATE_FULL_RE = re.compile(r"(20\d{2})[-._年]?(\d{1,2})[-._月]?(\d{1,2})")
# 正文/标题行里的完整日期：分隔符必带——紧凑 8 位（20260729）在文件名里是命名惯例，
# 在正文里更可能是单号/编号这类数字，不认，压误判
_DATE_FULL_TEXT_RE = re.compile(r"(20\d{2})[-._年](\d{1,2})[-._月](\d{1,2})")
# 标题行里的短日期："## 7.29" / "## 7月29日"，年份缺失需调用方补。
# 前后不吃数字和点，避免把 2026.7.29 的尾巴或 1.2.3 这类版本号误认成日期
_DATE_SHORT_RE = re.compile(r"(?<![\d.])(\d{1,2})\s*[.月]\s*(\d{1,2})(?![\d.])")


# 文件名里的窗口号：window_01.md / window_36_标题.md / Window-7.md 都认。
# 真实语料实测 36/36 可解析——比日期可靠得多，所以它同时兼任"呈现用的定位标签"
# （规格 §3.2：静态文件里用窗口号）和"没有日期时的时序信号"
_WINDOW_RE = re.compile(r"window[_\-]?(\d{1,4})", re.I)


def parse_window_no(filename):
    """文件名 → 窗口号（int）或 None。"""
    m = _WINDOW_RE.search(filename)
    return int(m.group(1)) if m else None


def _ymd_ts(year, month, day):
    """(y,m,d) → epoch 秒；无效日期（如 13.40）返回 None 不抛，交上层继续兜底。"""
    try:
        return datetime(year, month, day).timestamp()
    except ValueError:
        return None


def _first_valid_short_date(line, fallback_year):
    """扫一行里所有短日期候选，返回第一个**有效**的。

    必须遍历而不是只取第一个匹配（2026.07.31 真实语料实测出来的 bug）：标题行
    形如"window_35_某某4.0（7.25）"时，"4.0"先被匹到、判成 4 月 0 日无效，
    只取第一个匹配就会直接放弃，后面真正的"7.25"根本没机会。"""
    for m in _DATE_SHORT_RE.finditer(line):
        ts = _ymd_ts(fallback_year, int(m.group(1)), int(m.group(2)))
        if ts is not None:
            return ts
    return None


def _head_lines(text, n=3):
    """开头 n 个非空行——标题/日期都写在最前面，只扫这里，压正文里数字的误判概率。"""
    return [ln for ln in text.splitlines() if ln.strip()][:n]


def parse_chunk_timestamp(filename, chunk_text, fallback_year):
    """解析 chunk 时间戳 → (epoch秒, 来源) 或 (None, None)。

    来源按可信度排优先级：
      1. filename——按日期命名的语料，带年份，最可信
      2. chunk 开头几行里的完整日期（"chunk_head"）——真实 cloud window 语料文件名
         不带日期（window_36_xxx.md 这类），完整日期写在标题行，且分两种历史格式：
         早期"第十个窗口 · 2026.06.21深夜 · 标题"、后期"# window_30_标题（2026.07.15）"。
         2026.07.30 验收实测 95.6% 的块落 mtime 兜底后补的这条。只扫开头几行、且要求
         日期带分隔符（_DATE_FULL_TEXT_RE），都是为压正文里其它数字/日期的误判概率
      3. **任意级别标题行**里的短日期（"# window_35_某某（7.25）" / "## 7.29"）
         ——2026.07.31 拿真实语料实测放宽的：36 个窗口文件里 5 个的日期只写在
         `#` 一级标题行，而原来只扫 `## ` 开头的行，够不着。
         **只扫标题行、不扫正文**：正文里的"3.5 倍"这类数字会被读成 3 月 5 日，
         而标题行不会有这种用法，这是安全与召回率的折中点。
    短日期的年份缺失用 fallback_year 补（调用方给文件 mtime 的年份，跨年语料会有
    误差，接受：语料按时间推进命名，同文件跨年罕见）。
    都拿不到返回 (None, None)，由调用方兜底（load_corpus 先试文件级日期、再试同
    窗口号的邻层文件，最后退 mtime）。"""
    m = _DATE_FULL_RE.search(filename)
    if m:
        ts = _ymd_ts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if ts is not None:
            return ts, "filename"
    head = _head_lines(chunk_text)
    for line in head:
        m = _DATE_FULL_TEXT_RE.search(line)
        if m:
            ts = _ymd_ts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if ts is not None:
                return ts, "chunk_head"
    for line in chunk_text.splitlines():
        if not line.lstrip().startswith("#"):
            continue                      # 只认标题行；正文里的"3.5 倍"不该被读成日期
        ts = _first_valid_short_date(line, fallback_year)
        if ts is not None:
            return ts, "heading"
    return None, None


def timeline_chunks(text, filename, mtime):
    """单个 timeline md 文件 → [(chunk_text, meta)]。切块+时间戳解析的公共段：
    load_corpus 和 memory_import 的 markdown 翻译器共用，逻辑只此一份不复制。"""
    year = datetime.fromtimestamp(mtime).year
    # 文件级日期：日期常常只写在文件标题行、只属于第一个 chunk——先按文件解析一次，
    # 让同文件其余 chunk 继承，否则一个文件十几个块除首块外全落 mtime，
    # per-chunk 解析救不回兜底率
    file_ts, _ = parse_chunk_timestamp(filename, "\n".join(_head_lines(text)), year)
    out = []
    for i, chunk in enumerate(chunk_heading(text)):
        heading = next((ln for ln in chunk.splitlines() if ln.startswith("## ")), "")
        ts, ts_source = parse_chunk_timestamp(filename, chunk, year)
        if ts is None and file_ts is not None:
            # 继承文件级日期：文件内的先后粒度丢了，年月日还是真的
            ts, ts_source = file_ts, "file_head"
        elif ts is None:
            # 最后才兜 mtime：文件修改时间≠内容发生时间——每次全新 clone 会把
            # 全目录 mtime 刷成同一时刻，基本没有排序信号，timestamp_source 标出成色
            ts, ts_source = mtime, "mtime"
        out.append((chunk, {"source": filename, "heading": heading.lstrip("# ").strip(),
                            "chunk_index": i, "timestamp": ts, "timestamp_source": ts_source}))
    return out


def corpus_files(corpus_dirs, recursive=True):
    """一个或多个目录 → 排序后的 md 文件列表（默认递归子目录）。
    真实语料按 cloud/vps × timeline/index 分层放在多层子目录里，单层 glob 吃不到。"""
    if isinstance(corpus_dirs, (str, Path)):
        corpus_dirs = [corpus_dirs]
    out = []
    for d in corpus_dirs:
        out.extend(Path(d).glob("**/*.md" if recursive else "*.md"))
    return sorted(set(out))


def layer_of(path):
    """文件属于哪一层：父目录名叫 index 的是索引层，其余算叙事层。

    约定来自真实语料结构（`.../cloud/index/window_01.md` vs
    `.../cloud/timeline/window_01.md`）：index 层是每个会话一条高密度人写摘要，
    专门喂检索提高命中率；timeline 层是叙事正文。两层都要进库，但标出来，
    下游才能在命中摘要后指路到正文。"""
    return "index" if Path(path).parent.name.lower() == "index" else "timeline"


def load_corpus(corpus_dir, embed=False, recursive=True):
    """语料目录（可传多个）→ 建好的 MemoryIndex。

    每块 meta：source/heading/chunk_index/timestamp/timestamp_source
              + window（窗口号，解析不出为 None）+ layer（index/timeline）。

    时间戳兜底顺序（前三级在 parse_chunk_timestamp 里，后两级在这里）：
      文件名日期 > chunk 开头完整日期 > 开头短日期 > `## ` 短日期
      > 文件级日期继承（同文件其余块）
      > **同窗口号的邻层文件**（window_sibling）——index 层 36 个文件全是无标题
        纯段落、文件名也不带日期，本来 34/37 块落 mtime；但它跟 timeline 层同名
        同窗口号，是同一次会话的两种写法，借它的日期是有据可依的，不是猜
      > mtime（最后兜底，全新 clone 会把全目录刷成同一时刻，基本没有排序信号）"""
    index = MemoryIndex(embed=embed)
    files = corpus_files(corpus_dir, recursive=recursive)

    # 第一趟：解析每个文件的文件级日期与窗口号，供第二趟跨层继承用
    file_info = {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        mtime = p.stat().st_mtime
        ts, _ = parse_chunk_timestamp(p.name, "\n".join(_head_lines(text)),
                                      datetime.fromtimestamp(mtime).year)
        file_info[p] = {"text": text, "mtime": mtime, "file_ts": ts,
                        "window": parse_window_no(p.name), "layer": layer_of(p)}
    # 窗口号 → 该窗已知的日期（同一窗口号的不同层共享一次会话，取先解析出的那个）
    window_ts = {}
    for info in file_info.values():
        if info["window"] is not None and info["file_ts"] is not None:
            window_ts.setdefault(info["window"], info["file_ts"])

    for p in files:
        info = file_info[p]
        for chunk, meta in timeline_chunks(info["text"], p.name, info["mtime"]):
            meta["window"], meta["layer"] = info["window"], info["layer"]
            if meta["timestamp_source"] == "mtime":
                borrowed = window_ts.get(info["window"])
                if borrowed is not None:
                    meta["timestamp"], meta["timestamp_source"] = borrowed, "window_sibling"
            index.add(chunk, meta)
    return index.build()


# ---------- 写回：记忆库正文层的笔（任务卡"记忆写回与权重持久化"） ----------

def append_record(corpus_dir, text, current_state, window=None, now=None):
    """把"刚发生的值得记的事"落成 timeline 记录 → (path, chunk_text, meta)。

    这是记忆库自己生长的那半支笔（thread 是会话状态层，这里是正文层）。
    确认分级（规格 §7）：记忆库正文自动写，所以这里没有确认关卡；人格 md 任何
    改动必须用户确认，所以这支笔构造上就够不着它——写哪个文件由这里按窗口号+
    日期生成，调用方（MCP 工具）不传路径，永远只落在语料目录的 timeline 层。

    三条规则：
      当下状态必填（病灶迁移，同 close_thread）——没有状态的记录，未来重读时
        会把"最后读到的内容"误当成"正在发生的事"；
      文件名带日期——文件名日期是 parse_chunk_timestamp 的最高优先级来源，
        自己写的文件没理由掉 mtime 兜底；
      同一天的写回进同一个窗口文件，跨天开新窗口——窗口号取语料现有最大值 +1，
        "同天=同窗口"是近似（一天开两次会话会并进一个窗口），先用日期分组，
        等真机跑出问题再升级成显式传窗口号。window 参数可显式覆盖。"""
    if not (isinstance(text, str) and text.strip()):
        raise ValueError("写回内容不能为空")
    if not (isinstance(current_state, str) and current_state.strip()):
        raise ValueError("当下状态必填（病灶迁移）：这件事现在是什么状态？"
                         "不写的话，未来重读会把它当成正在发生的事")
    now = time.time() if now is None else now
    day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    timeline = Path(corpus_dir) / "timeline"
    timeline.mkdir(parents=True, exist_ok=True)

    known = [w for w in (parse_window_no(p.name) for p in corpus_files(corpus_dir))
             if w is not None]
    top = max(known, default=0)
    if window is None:
        # 现有最大窗口文件如果就是今天写回开的，继续用它；否则开新窗口
        today_file = next(iter(timeline.glob(f"window_{top:02d}_{day}.md")), None) \
            if top else None
        window, path = (top, today_file) if today_file \
            else (top + 1, timeline / f"window_{top + 1:02d}_{day}.md")
    else:
        window = int(window)
        existing = sorted(timeline.glob(f"window_{window:02d}_*.md"))
        path = existing[0] if existing else timeline / f"window_{window:02d}_{day}.md"

    chunk_text = f"## {day} 记\n{text.strip()}\n当下状态：{current_state.strip()}"
    if path.exists():
        path.write_text(path.read_text(encoding="utf-8").rstrip() +
                        "\n\n" + chunk_text + "\n", encoding="utf-8")
    else:
        path.write_text(f"# 第{window}个窗口 · {day}\n\n" + chunk_text + "\n",
                        encoding="utf-8")
    meta = {"source": path.name, "heading": f"{day} 记", "timestamp": now,
            "timestamp_source": "append", "window": window, "layer": "timeline"}
    return path, chunk_text, meta


# ---------- 合成语料 selftest（全部虚构，不含任何真实语料） ----------

SYNTH = [
    ("## 修理咖啡机\n家里那台老咖啡机又坏了，这次是加热管不工作。拆开后盖发现保险丝熔断，型号是10A250V。换上通电测试正常。",
     {"source": "a.md", "heading": "修理咖啡机", "chunk_index": 0}),
    ("## 一次远足\n上周六去了青云山，山路十二公里。半山腰凉亭遇到卖桂花糕的老人家，用的是山顶的野桂花，甜而不腻。",
     {"source": "a.md", "heading": "一次远足", "chunk_index": 1}),
    ("## 种薄荷失败\n四月阳台种的薄荷死了。复盘：花盆太小、浇水太勤、正午暴晒，盆底积水烂了根。",
     {"source": "b.md", "heading": "种薄荷失败", "chunk_index": 0}),
    ("## 读城市与狗\n略萨这本小说最震撼的是结构：四条叙事线交错，直到最后才拼出完整真相。",
     {"source": "b.md", "heading": "读城市与狗", "chunk_index": 1}),
]


def _build_synth(embed=False):
    idx = MemoryIndex(embed=embed)
    for text, meta in SYNTH:
        idx.add(text, meta)
    return idx.build()


def _selftest(embed=False):
    idx = _build_synth(embed=embed)

    # 1. 检索命中 + 元数据完整
    r = idx.retrieve("咖啡机坏了是什么原因", topN=2)
    assert "保险丝熔断" in r[0]["text"], f"BM25层该把咖啡机块排第一：{r[0]['meta']}"
    assert r[0]["meta"]["heading"] == "修理咖啡机"
    assert set(r[0]) == {"id", "text", "meta", "score", "weight"}, "结果字段：id/text/meta/score/weight"
    assert len(r) == 2, "topN 生效"

    # 2. BM25 纯层：idf 让稀有词区分度更高，命中块得分 > 无关块
    bm = idx._bm25.scores(tokenize("薄荷 盆底积水"))
    assert bm[2] == max(bm), "薄荷块 BM25 分最高"

    # 3. RRF 融合是纯函数：两路名次都把 0 排前 → 0 胜
    fused = rrf_fuse([[0, 1], [0, 1]], [1.0, 1.0])
    assert fused[0][0] == 0

    # 4. 用进废退改变排序（变异检查靶心之一：rrf_fuse 里 *= weights）：
    #    同名次下，权重被顶高的反超。1 的权重 2.0 压过 0 的 1.0
    fused_w = rrf_fuse([[0, 1], [0, 1]], [1.0, 2.0])
    assert fused_w[0][0] == 1, "权重高的被顶到前面（用进废退改排序）"

    # 5. 用进废退累加（靶心之二：retrieve 里 += weight_boost）：
    idx2 = _build_synth(embed=embed)
    before = list(idx2.weights)
    hit = [x["id"] for x in idx2.retrieve("咖啡机坏了", topN=2)]
    non_hit = [i for i in range(len(idx2.weights)) if i not in hit]
    for i in hit:
        assert abs(idx2.weights[i] - (before[i] + 0.05)) < 1e-9, "命中块 +0.05"
    for i in non_hit:
        assert idx2.weights[i] == before[i], "没命中的块权重不动"
    idx2.retrieve("咖啡机坏了", topN=2)  # 再命中一次
    for i in hit:
        assert abs(idx2.weights[i] - (before[i] + 0.10)) < 1e-9, "反复命中累加到 +0.10"

    # 6.【第三层·变异靶心：retrieve 里的 graph_route append + 种子零分过滤 + 强度排序】
    #    合成语料里"望远镜"这条实体线串起 约定(A)/兑现(B)/后续(C) 三块，D 无关。
    #    索引布局是特意排的：D 放 0——拆掉第三路后零分并列块按索引排序会让 D 挤进
    #    topN，6b 必红；强邻居 A 的索引(3)比弱邻居 C(1) 大——邻居若退化成按索引排
    #    而不是按强度排，6a 必红。不给"碰巧同序"留活路
    graph_synth = [
        "## 学做咖喱\n试了椰浆咖喱，小火慢炖差点糊锅。",                        # 0=D 无关
        "## 镜片保养\n给望远镜擦镜片，用了专用的软布。",                        # 1=C 只共享望远镜
        "## 挑设备\n看了三款设备，最后那台望远镜带脚架，山顶上说过的话应验了。",  # 2=B
        "## 山顶的约定\n那晚在山顶聊到以后，说好要买一台能看土星的望远镜。",      # 3=A 共享山顶+望远镜
    ]
    idx_g = MemoryIndex()
    for t in graph_synth:
        idx_g.add(t)
    idx_g.build()
    #    6a. B 的邻居按强度排：A（共享 山顶+望远镜）先于 C（只共享 望远镜）；关联对称；无关块孤立
    assert idx_g.graph_neighbors(2) == [3, 1], "邻居按链接强度降序，不是按索引"
    assert 2 in idx_g.graph_neighbors(3), "关联是对称的"
    assert idx_g.graph_neighbors(0) == [], "无共享显著词的块不相连"
    #    6b. 价值断言：query"设备"只有 B 有词面命中，A/C 与 query 零重叠，
    #        靠第三路进 topN，无关的 D 被挤出去
    r6 = idx_g.retrieve("设备", topN=3)
    ids6 = [x["id"] for x in r6]
    assert ids6[0] == 2 and set(ids6) == {2, 1, 3}, f"关联块该被图谱带进 topN，实际 {ids6}"

    # ---- recall_recent（换窗/压缩召回）----
    DAY = 86400.0
    now = 1_800_000_000.0  # 固定时刻，selftest 不依赖真实时钟

    # 7. 时间戳缺失兜底：不崩、沉底但仍返回，缺失块之间按权重排
    idx3 = MemoryIndex()
    idx3.add("有时间戳·昨天", {"timestamp": now - DAY})
    idx3.add("无时间戳·权重低", {})
    idx3.add("无时间戳·权重高", {})
    idx3.weights[2] = 5.0
    r = idx3.recall_recent(topN=3, now=now)
    assert [x["id"] for x in r] == [0, 2, 1], "缺时间戳沉底，缺失块之间按权重排"
    assert r[1]["score"] == 0.0 and r[2]["score"] == 0.0, "缺时间戳的块 recency 记 0"
    assert set(r[0]) == {"id", "text", "meta", "score"}, "结果字段按任务卡：id/text/meta/score"

    # 8.【变异检查靶心之三：recall 分数里的 * self.weights[i]】复合排序，不是单一维度：
    #    权重把稍旧的块顶过更新的块（1>0：0.673*1.5 > 0.743*1.0），但救不回旧太多的
    #    块（0>2）。注意 1>0 必须靠乘法本身成立——同龄+权重的构造会被排序键里的权重
    #    tiebreak 兜住，测不到乘法，这里故意让两块年龄错开
    idx4 = MemoryIndex()
    idx4.add("三天前·没被提过", {"timestamp": now - 3 * DAY})   # 0
    idx4.add("四天前·常被提起", {"timestamp": now - 4 * DAY})   # 1
    idx4.add("十天前·常被提起", {"timestamp": now - 10 * DAY})  # 2
    idx4.weights[1] = idx4.weights[2] = 1.5
    r = idx4.recall_recent(topN=3, now=now)
    assert [x["id"] for x in r] == [1, 0, 2], "权重顶过小年龄差（1>0）、顶不过大年龄差（0>2）"

    # 9.【任务卡不预设答案那条】极端对比：刚发生未被命中(w=1.0) vs 30天前高权重(w=2.0)
    #    默认 half_life=7 天下新块碾压（1.0 : ≈0.10）——实测后判断合理，保留乘法，
    #    理由记在 recall_recent docstring；这里断言住这个已定的行为当回归锚点
    idx5 = MemoryIndex()
    idx5.add("刚发生·从未被提起", {"timestamp": now})
    idx5.add("三十天前·权重高", {"timestamp": now - 30 * DAY})
    idx5.weights[1] = 2.0
    r = idx5.recall_recent(topN=2, now=now)
    assert r[0]["id"] == 0 and r[0]["score"] / r[1]["score"] > 5, "默认半衰期下新鲜度主导"
    #    信号配比的旋钮是 half_life 不是换公式：拉长到 1000 天，高权重旧块反超
    r = idx5.recall_recent(topN=2, half_life=1000.0, now=now)
    assert r[0]["id"] == 1, "half_life 拉长后权重信号占上风"

    # 10. 时间戳解析：文件名完整日期 > 标题短日期（年份用 fallback 补）> 解析不出
    ts_f, src_f = parse_chunk_timestamp("2026-07-29.md", "## 随便什么标题", 2000)
    assert src_f == "filename" and datetime.fromtimestamp(ts_f).strftime("%Y%m%d") == "20260729"
    ts_c, src_c = parse_chunk_timestamp("20260729.md", "", 2000)
    assert src_c == "filename" and ts_c == ts_f, "紧凑写法 20260729 也认"
    ts_h, src_h = parse_chunk_timestamp("window-12.md", "## 7.29 那天\n正文", 2026)
    assert src_h == "heading" and ts_h == ts_f, "标题短日期 + fallback 年份"
    assert parse_chunk_timestamp("window-12.md", "## 13.40 不是日期\n正文 7.29", 2026) == (None, None), \
        "无效日期不认、正文里的数字不扫"

    # 11.【验收返工回归】真实 cloud window 语料：文件名不带日期、完整日期写在标题行，
    #    两种历史格式（内容脱敏虚构）——首版 filename/短日期两条规则都吃不到，
    #    实测 95.6% 的块落 mtime，而全新 clone 会把全目录 mtime 刷成同一时刻，没有信号
    early = "第十个窗口 · 2026.06.21深夜 · 修补风筝\n\n聊了把旧风筝重新糊好的事。"
    ts_e, src_e = parse_chunk_timestamp("window_10_修补风筝.md", early, 2000)
    assert src_e == "chunk_head" and datetime.fromtimestamp(ts_e).strftime("%Y%m%d") == "20260621", \
        "早期标题格式：第N个窗口 · 2026.06.21深夜 · 标题"
    late = "# window_30_晒被子（2026.07.15）\n\n## 上午\n把被子都晒了。"
    ts_l, src_l = parse_chunk_timestamp("window_30_晒被子.md", late, 2000)
    assert src_l == "chunk_head" and datetime.fromtimestamp(ts_l).strftime("%Y%m%d") == "20260715", \
        "后期标题格式：# window_30_标题（2026.07.15）"
    #    正文里的紧凑 8 位数字不当日期认（正文匹配分隔符必带），单号/编号误判挡住
    assert parse_chunk_timestamp("window_11.md", "## 快递\n单号 20260721 已签收", 2026) == (None, None)

    # 12. 文件级日期继承：日期只写在文件标题行（只属于第一个 chunk）时，
    #     同文件其余 chunk 继承 file_head，不落 mtime
    with tempfile.TemporaryDirectory() as td:
        Path(td, "window_30_晒被子.md").write_text(
            "# window_30_晒被子（2026.07.15）\n\n## 上午\n把被子都晒了。\n\n"
            "## 傍晚\n收被子，带着太阳味。\n", encoding="utf-8")
        corpus = load_corpus(td)
        assert len(corpus.chunks) >= 2, "标题+两小节至少切出两块"
        for m in corpus.meta:
            assert m["timestamp_source"] in ("chunk_head", "file_head"), f"不该落 mtime：{m}"
            assert datetime.fromtimestamp(m["timestamp"]).strftime("%Y%m%d") == "20260715"
        assert any(m["timestamp_source"] == "file_head" for m in corpus.meta), \
            "无日期的 ## 小节继承文件级日期"

    # 13. 二阶段检索接口（机制部分；精排质量断言在 rerank_experiment.py）
    #  a) 常数精排分＝精排没意见 → 保持粗筛序，行为跟一阶段一致
    base13 = [x["id"] for x in _build_synth(embed=embed).retrieve("咖啡机坏了", topN=2)]
    same13 = [x["id"] for x in _build_synth(embed=embed).retrieve(
        "咖啡机坏了", topN=2, reranker=lambda q, ts: [1.0] * len(ts))]
    assert same13 == base13, "常数精排分该保持粗筛序"
    #  b) 精排接管排序：只给读书块打分 → 它被捞到第一
    r13 = _build_synth(embed=embed).retrieve(
        "咖啡机坏了", topN=2, reranker=lambda q, ts: [1.0 if "略萨" in t else 0.0 for t in ts])
    assert r13[0]["meta"]["heading"] == "读城市与狗", "精排该能接管排序"
    #  c)【变异靶心：boost 只加最终 topN】粗筛池其余候选不算"被命中"，权重不动
    idx13c = _build_synth(embed=embed)
    before13 = list(idx13c.weights)
    hit13 = [x["id"] for x in idx13c.retrieve(
        "咖啡机坏了", topN=1, reranker=lambda q, ts: [1.0] * len(ts), coarse_topM=4)]
    for i in range(4):
        if i in hit13:
            assert abs(idx13c.weights[i] - before13[i] - 0.05) < 1e-9, "最终 topN 加权"
        else:
            assert idx13c.weights[i] == before13[i], "粗筛池不算命中，不加权"
    #  d)【变异靶心：精排分乘权重】重排不豁免用进废退。构造上不能用"同精排分"——
    #     权重在粗筛 rrf_fuse 里已乘过一轮，常数精排分会靠"保持粗筛序"蒙混过关；
    #     这里让读书块精排分更低（0.6）但权重更高（2.0）：0.6×2.0 > 1.0×1.0，
    #     只有精排阶段真乘了权重它才能排第一
    idx13d = _build_synth(embed=embed)
    idx13d.weights[3] = 2.0
    r13d = idx13d.retrieve("咖啡机坏了", topN=2, coarse_topM=4,
                           reranker=lambda q, ts: [0.6 if "略萨" in t else 1.0 for t in ts])
    assert r13d[0]["id"] == 3, "精排分 0.6×权重 2.0 该压过 1.0×1.0"

    # 14. 真实语料适配（2026.07.31 拿私人 timeline 实测后补，内容脱敏虚构）
    #  a)【变异靶心：_first_valid_short_date 的 finditer】标题里版本号在前、日期在
    #     后时，必须跳过无效候选继续找——只取第一个匹配会被"4.0"吃掉整行
    ts_v, src_v = parse_chunk_timestamp("window_35_某某4.0.md", "# window_35_某某4.0（7.25）", 2026)
    assert src_v == "heading" and datetime.fromtimestamp(ts_v).strftime("%m%d") == "0725", \
        f"版本号 4.0 该被跳过、7.25 该被认出：{src_v}"
    #  b)【变异靶心：标题行放宽到任意级别】日期只写在 `#` 一级标题行也要认得
    ts_h1, _ = parse_chunk_timestamp("window_22_某某.md", "# window_22_某某（7.8）\n\n## 小节\n正文", 2026)
    assert ts_h1 is not None and datetime.fromtimestamp(ts_h1).strftime("%m%d") == "0708"
    #  c) 正文里的短日期一概不认——"3.5 倍"不是 3 月 5 日
    assert parse_chunk_timestamp("window_9.md", "# 无日期标题\n\n效率提升了 3.5 倍。", 2026) == (None, None)
    #  d) 窗口号与层标记
    assert parse_window_no("window_01.md") == 1 and parse_window_no("window_36_某某.md") == 36
    assert parse_window_no("Window-7.md") == 7 and parse_window_no("2026-07-15.md") is None
    assert layer_of("/a/cloud/index/window_01.md") == "index"
    assert layer_of("/a/cloud/timeline/window_01.md") == "timeline"

    # 15.【变异靶心：递归加载 + 同窗口号跨层继承】真实语料是多层子目录，且 index
    #     层是无标题纯段落、文件名也不带日期——只能从同窗口号的 timeline 层借日期
    with tempfile.TemporaryDirectory() as td:
        root = Path(td, "timeline")
        (root / "cloud" / "timeline").mkdir(parents=True)
        (root / "cloud" / "index").mkdir(parents=True)
        Path(root, "cloud", "timeline", "window_07_某某.md").write_text(
            "# window_07_某某（7.25）\n\n## 上午\n把被子晒了。\n", encoding="utf-8")
        Path(root, "cloud", "index", "window_07.md").write_text(
            "这一窗聊了晒被子和周末安排，没有标题也没有日期。", encoding="utf-8")
        corpus = load_corpus(root)
        assert len(corpus.chunks) >= 3, "递归该吃到两层子目录下的文件"
        by_layer = {}
        for m in corpus.meta:
            by_layer.setdefault(m["layer"], []).append(m)
        assert set(by_layer) == {"timeline", "index"}, f"两层都该进库并标出来：{set(by_layer)}"
        ix = by_layer["index"][0]
        assert ix["timestamp_source"] == "window_sibling", \
            f"index 块该从同窗口号的 timeline 借到日期，实际 {ix['timestamp_source']}"
        assert datetime.fromtimestamp(ix["timestamp"]).strftime("%m%d") == "0725"
        assert all(m["window"] == 7 for m in corpus.meta), "窗口号该进 meta"
        #   非递归模式下同一个根目录一个文件都吃不到（对照，证明递归确实在起作用）
        assert load_corpus(root, recursive=False).chunks == []

    # 16.【变异靶心：权重跟内容走，不跟位置走】用进废退权重的持久化
    #     server 是 stdio 进程、客户端每次会话都可能重启它——权重不落盘的话每次
    #     归 1.0，"被反复聊起的记忆更重"在生产形态下等于没有
    with tempfile.TemporaryDirectory() as td:
        wpath = Path(td, ".weights.json")
        idx_a = _build_synth()
        idx_a.retrieve("咖啡机 保险丝", topN=1)          # 命中咖啡机块 → 权重 +0.05
        assert idx_a.weights[0] > 1.0
        assert idx_a.save_weights(wpath) == 1, "只存 ≠1.0 的权重（稀疏）"
        #   模拟重启：新 index、块顺序打乱（原第 0 块挪到最后）——按位置记就张冠李戴
        idx_b = MemoryIndex()
        for text, meta in SYNTH[1:] + SYNTH[:1]:
            idx_b.add(text, dict(meta))
        idx_b.build()
        assert idx_b.load_weights(wpath) == 1
        assert idx_b.weights[-1] > 1.0 and all(w == 1.0 for w in idx_b.weights[:-1]), \
            "权重要按内容对号入座——咖啡机块换了位置，权重得跟着它走"
        #   正文被编辑过的块权重归 1 重新攒——内容变了，旧的命中次数不该继承
        idx_c = MemoryIndex()
        idx_c.add(SYNTH[0][0] + "（后来补记：其实是电源线的问题）", dict(SYNTH[0][1]))
        idx_c.build()
        assert idx_c.load_weights(wpath) == 0 and idx_c.weights[0] == 1.0
        #   文件不存在按零处理，不抛——第一次启动本来就没有历史
        assert MemoryIndex().build().load_weights(Path(td, "没有这个文件.json")) == 0

    # 17.【变异靶心：写回落盘可回读】append_record 是记忆库自己生长的那半支笔
    with tempfile.TemporaryDirectory() as td:
        t0 = datetime(2026, 7, 31, 21, 0).timestamp()
        #   当下状态必填（病灶迁移在写入口强制，同 close_thread）
        try:
            append_record(td, "她说了一件重要的事。", "", now=t0)
            assert False, "缺当下状态该拒绝"
        except ValueError as e:
            assert "当下状态" in str(e)
        #   文件名带日期 + 窗口号顺延
        p1, chunk1, meta1 = append_record(td, "约好周末去看她说过的那家店。",
                                          "约定成立，还没去。", now=t0)
        assert p1.name == "window_01_2026-07-31.md", f"文件名要带窗口号和日期：{p1.name}"
        assert chunk1.endswith("当下状态：约定成立，还没去。"), "记录必须以状态收尾"
        #   同一天第二笔进同一个窗口文件，不另开窗
        p2, _, meta2 = append_record(td, "她补了一句：想吃那儿的甜品。",
                                     "补进同一个约定。", now=t0 + 600)
        assert p2 == p1 and meta2["window"] == 1, "同天写回该进同一个窗口"
        #   跨天开新窗口
        p3, _, meta3 = append_record(td, "去过了，店已经搬走了。",
                                     "约定落空，她有点失落，说改天找新店。",
                                     now=t0 + 86400 * 2)
        assert meta3["window"] == 2 and "2026-08-02" in p3.name
        #   检索层回读：三笔都在、时间戳全部来自文件名、窗口号正确
        rt = load_corpus(td)
        assert sum("甜品" in c for c in rt.chunks) == 1
        assert all(m["timestamp_source"] != "mtime" for m in rt.meta), \
            "写回的文件不该有任何块落 mtime 兜底"
        assert {m["window"] for m in rt.meta} == {1, 2}

    print("selftest ok" + ("（含真embedding路径）" if embed else "（零依赖）"))


def run(corpus_dir, query, topN=5, embed=False):
    index = load_corpus(corpus_dir, embed=embed)
    retriever = "BM25 + bge-small-zh-v1.5" if embed else "BM25 + bigram代理"
    print(f"索引 {len(index.chunks)} 块，检索器={retriever}\n查询：{query}\n")
    for i, r in enumerate(index.retrieve(query, topN), 1):
        head = r["meta"].get("heading") or r["meta"].get("source", "")
        preview = r["text"].replace("\n", " ")[:60]
        print(f"{i}. [{head}] score={r['score']:.4f} w={r['weight']:.2f}  {preview}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embed", action="store_true", help="用真 embedding（需 fastembed）")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--query", help="检索词")
    ap.add_argument("--topN", type=int, default=5)
    args = ap.parse_args()
    if args.selftest:
        _selftest(embed=args.embed)
    elif args.corpus and args.query:
        run(args.corpus, args.query, args.topN, embed=args.embed)
    else:
        ap.print_help()
