#!/usr/bin/env python3
"""
最小检索接口参考实现（设计笔记"三层检索设计"）。

**关于"三层"这个说法的诚实修正**（2026.08.01 回归集消融实测）：三路是真的，但
**默认零依赖档下语义那一路没有独立贡献**——字符 bigram 余弦跟 BM25 用的是同一套
bigram，给 RRF 的排序与词面路高度重合，消融掉它两档所有分数一模一样。所以默认
配置实际是"词面 + 图谱"两路在干活。装了 fastembed 走 `--embed` 时，语义那一路
才名副其实（paraphrase R0.75→1.00）。别对外简单说"三层检索"。

接 chunking 实验切好的块，实现 retrieve(query, topN) → [相关片段+元数据]：
  第一层 BM25 关键词：Okapi BM25，字符 bigram 当词（中文零依赖分词的常规做法）
  第二层 向量语义：**提供方可插拔**（`embedding_provider`）——本地 fastembed
    或云端 HTTP 服务，`--embed` 开、`--embed-provider` 选哪一档；零依赖时退回
    bigram 余弦代理。检索层不认识任何一家，换服务商不改这里的代码
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
提供方（本地档需要 fastembed，云端档只用 stdlib 的 urllib）。selftest 不需要装任何东西。

**语料去哪儿，按档不一样，这件事不许含糊**：零依赖与本地模型档语料不出本机；
云端档**查询和被检索的内容都会发到你选的那家服务商**。所以它在初始化流程里是一个
显式选择点（memory_init 的 `--step route`），不是一个静默默认。

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

# 复用 chunking 实验里已验证的切块（同目录模块，import 不触发它的 CLI）
from chunking_experiment import chunk_heading, bigram_counts, cosine
# 真 embedding 走可插拔提供方层：本地 fastembed／云端 HTTP 都从这里进来，
# 检索层不认识任何一家（同 draft_extraction.llm_call 的口子）
from embedding_provider import (VectorCache, embed_with_cache, resolve_provider,
                                DEFAULT_LOCAL_MODEL, get_hit_floor)

# recall_recent 的半衰期默认值（天）：经验值——timeline 语料按天/窗口推进，一周前的
# 上下文对"接上刚才"基本没用了。跟 +0.05/k=60 同一待遇，等真实评估再调。
RECALL_HALF_LIFE_DAYS = 7.0

# 用进废退权重在"乘法路径"（recall_recent、精排）里的影响力上限。权重存储照旧
# 无上界累加（那是"被聊起多少次"的事实记录），但参与排序时钳到这个值——不封顶
# 的话，被频繁检索的块（哪怕只是撞上常见 query 词）权重一直涨，过期事实反而
# 压着新记录（2026.07.31 维护者指出的滚雪球陷阱）。2.0 是经验值，跟 +0.05/k=60
# 同一待遇：等真实检索质量评估再调，现在不拍脑袋。
WEIGHT_INFLUENCE_CAP = 2.0

# 图谱路专用的 RRF k（2026.08.01 回归集实测定的，理由见 rrf_fuse docstring）。
# **取值 5，2026.08.01 在真实私人语料（602 块、214 条自动构造的查询-答案对）上复核后
# 从 10 改过来的**。合成回归集上 3~15 是一整片平台、分不出高下，所以当初取了中值 10；
# 真实语料分得出来，而且差距不小：
#     k=3   hit@1=0.748  MRR=0.824      k=15  hit@1=0.575  MRR=0.732
#     k=5   hit@1=0.748  MRR=0.822      k=20  hit@1=0.547  MRR=0.711
#     k=10  hit@1=0.682  MRR=0.789      k=60  hit@1=0.374  MRR=0.560
# k=3 与 k=5 并列最好（真 embedding 档同结论），取 5 不取 3：两个值在真实语料上打平，
# 而 5 同时落在合成集那片平台里，两边都不吃亏；3 更贴边界。
# **这是本项目第一个"合成集调不出、真实语料才分得出"的参数**——合成集给的是
# 一片平台，真实语料给的是一条斜坡。记一笔：平台不代表"取哪都行"，可能只代表
# "这把尺子在这个维度上分辨率不够"。
GRAPH_RRF_K = 5

# 图谱建边的 df 绝对天花板（2026.08.02，《图谱建库耗时》任务卡，维护者拍的路线一）。
#
# **它不是判据，是成本上界**——判据仍然只有 `_distinctive_df()` 那一套，候选闸
# 这边一个字没动。分开的理由：候选闸的成本随 df 线性，图谱要两两配对、成本是 df²，
# 而 `N//5` 是占比，语料一大绝对值就上去，建库耗时随块数平方增长直到撞穿客户端的
# MCP 启动超时。完整论证与实测数在 `_graph_df_cap()` 的 docstring 里。
#
# 取 64 不是标定出来的最优值（这个位置没有能标定它的评测集，同 GRAPH_RRF_K 那次
# 的教训——先把它是什么写清楚，别假装它被调过）：它只需要够低到把平方项压住、
# 又够高到在 N ≤ 320 时完全不生效。
GRAPH_DF_CAP = 64

# 真 embedding 路径（--embed）的可靠命中门槛（2026.08.01 实测定的）。
#
# **这条不是优化，是修一个会让 --embed 变危险的缺陷。** 零依赖路径下门槛是
# "bm25 或向量层有分（>0）"，因为字符 bigram 余弦对不相干的文本真的会给 0；
# 但真 embedding 的余弦**几乎恒正**——库里根本没有的事，它照样给每块 0.2~0.4 分。
# 实测：开 --embed 后 absent 类的"正确空手率"从 1.00 **掉到 0.00**，两档都是。
# 也就是说模型问"去年体检血糖多少"，系统会端回四条毫不相干的记忆，而 instructions
# 又堵着"不许说我没有记录"——这是一条通往编造的流水线，比检索不准危险得多。
#
# 取值 0.45 原本的依据是合成集上的一道"干净间隙"（absent 最高 0.436、真实查询最低
# 0.485）。**2026.08.01 真实语料复核推翻了这个前提**：602 块真实语料上两个分布
# 明显重叠——真实查询最低 0.410、5% 分位 0.479；而"库里没有但像是会有"的日常问题
# （问牙医复诊、车险到期这类）中位就有 0.464、最高 0.517。**没有干净间隙，只有权衡点。**
#
# 更要紧的是：**端到端看，这道门槛几乎不起作用**。候选闸是"BM25 有分 **或** 向量
# 过槛"的或关系，而真实语料上任何查询都能跟十几到几十个块有 bigram 重叠、BM25 分
# 还有 9.8~15.4。实测门槛从 0.40 调到 0.55，无关查询的正确空手率纹丝不动（远主题
# 2/16、近生活 0/8）。**"查不到就说没有"这件事真正的短板在 BM25 那一路没有门槛，
# 不在这个常数上**——详见 regression_set 里"第四个语料伪影"一节与后续计划。
# 保留 0.45：在扫描里它是误挡真实查询（2.8%）与挡住无关查询之间最平衡的一档，
# 且它确实管住了向量路的候选资格；但别再宣称它"修好了查不到就说没有"。
#
# **这个数与 embedding 模型绑定**（当前是 bge-small-zh-v1.5，512 维）——换模型
# 余弦标度就变了，必须重新量，不能照抄。真实语料复核已做（2026.08.01）。零依赖路径**不适用**这个阈值（bigram
# 余弦是另一套标度），所以它只在 self.embed 为真时生效。
#
# **2026.08.02（云端档）：这个常数不再是"真 embedding 路径的门槛"，只是
# bge-small-zh-v1.5 这一档的标定值。** 提供方可插拔之后，门槛按模型查
# `embedding_provider.HIT_FLOOR_BY_MODEL`；查不到就是**未标定**（None），
# 走下面 `vec_floor is None` 那条保守路径，**绝不照抄这个 0.45**。
EMBED_HIT_FLOOR = get_hit_floor(DEFAULT_LOCAL_MODEL)

# 查询缺失率的标注阈值（2026.08.01，第二份外部反馈在其真实语料上标定，60 条样本）。
#
# **这个信号的定性比阈值重要得多：它是"专名缺席检测器"，不是"事件存在性检测器"。**
# 标定数据（present 30 条分三档 / absent_日常 30 条，全部穿透 bm25>0 的朴素闸）：
#     present 专名直问 4.2% / 日常问法 14.6% / 抽象归纳 23.1%（中位）
#     absent_日常 47.7%（中位）
#     两个分布重叠 27.5 个百分点——**不是干净分离**（测试者作废了自己第一份报告里
#     "3pp 重叠"的说法，那是 25 条小样本的假象；别再引用旧数）
# 按"误杀代价 = 3 × 漏判代价"加权的最优阈值是 35%：拦下 90% 的编造、误杀 6.7%。
#
# 两条形态结论决定了它**只能标注、不能硬拒**：
#   误杀全部来自**抽象归纳式**提问（"她最近在焦虑什么"），用户用自己的抽象词汇问
#     语料里用具体叙事写的事，词面天然不重合——而这恰恰是陪伴场景最有价值的一类
#     提问。把它拒掉，等于专门惩罚最该被答好的问题。
#   漏判全部来自**不含专名的日常编造**（"上个月搬家搬了几趟"12.5%），整句高频常用
#     词，原理上就挡不住，调阈值救不回来。
# 所以机制层的职责收窄成：**把明显的挡掉、把不确定的标出来**；"这五段跟你问的对不上"
# 这个判断只有读到内容的模型能下，留给行为层（引导指南里那句"检索返回了东西不等于
# 答案，别拿沾边的记录去圆"因此从止血升格为长期方案的主体）。
#
# **35% 是单一语料上的观测值，不是普适常数**——换语料（换语言、换记录风格、换规模）
# 必须重量，同 EMBED_HIT_FLOOR 换模型要重量的先例。
MISS_RATE_FLAG = 0.35


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


def rrf_fuse(rank_lists, k=60, ks=None):
    """RRF 融合多路名次（纯函数，只看名次不看分数）。
    rank_lists: [[chunk_idx 从好到坏], ...]。返回 [(chunk_idx, 融合分), ...] 按分降序。
    ks：可选，每路各自的 k（长度需与 rank_lists 一致）；不传则各路共用 k。
    **为什么要允许每路不同的 k**（2026.08.01 回归集实测后加）：k 决定这一路"名次
    差别值多少钱"。k=60 时 rank1(1/61) 与 rank10(1/70) 只差 15%，等于把名次抹平、
    让**进了几条路**压倒**排第几**。这对词面/向量两路是好事（它们本来就该互相印证），
    但对图谱路是致命的——图谱路存在的全部意义就是捞回那些**在词面和向量里都是零分**
    的块（同一件事换了说法），这种块天生只可能出现在一条路里，在 k=60 下永远赢不了
    任何一个在两条路里都混了个脸熟的闲词块。给图谱路一个更小的 k，它排头几名的块
    才有机会真的被端上来。

    2026.07.31 起权重不再乘进融合分。原来 fused *= weight 是无上界乘子叠在
    天生很平的 RRF 分上（rank1=1/61 与 rank10=1/70 只差 15%，命中三次的 ×1.15
    就能反超别处 rank1），"命中→加权→更靠前→更易命中"的正反馈没有任何东西
    封顶——维护者指出的滚雪球陷阱（详见 retrieve 的权重第四路注释）。
    现在权重的话语权由 retrieve 以"第四路名次"的形式并进来，影响力有界。"""
    fused = defaultdict(float)
    for li, ranks in enumerate(rank_lists):
        kk = k if ks is None else ks[li]
        for rank, idx in enumerate(ranks):
            fused[idx] += 1.0 / (kk + rank + 1)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class MemoryIndex:
    """记忆检索索引：add 片段 → build → retrieve。权重随命中浮沉（用进废退）。"""

    def __init__(self, embed=False, weight_boost=0.05, rrf_k=60,
                 graph_topK=5, graph_seeds=3, graph_rrf_k=None,
                 provider=None, cache_path=None):
        # graph_topK/graph_seeds 是经验值（每块存几个邻居/两路各取几个种子扩展），
        # 跟 +0.05/k=60 同一待遇，等真实检索质量评估再调
        self.embed = embed
        self.weight_boost = weight_boost
        self.rrf_k = rrf_k
        self.graph_rrf_k = GRAPH_RRF_K if graph_rrf_k is None else graph_rrf_k
        # embedding 提供方（本地 fastembed／云端 HTTP，见 embedding_provider）。
        # 只在 embed=True 时解析——零依赖档不该因为解析配置就报错。
        self.provider = (provider or resolve_provider()) if embed else None
        self.cache_path = cache_path
        # 向量层"算不算有信号"的门槛，**三种情形三套处理**：
        #   零依赖（embed=False）：bigram 余弦对不相干文本真会给 0，>0 就够 → 0.0
        #   真 embedding 且**该模型标定过**：用标定值（如 bge-small-zh-v1.5 的 0.45）
        #   真 embedding 但**没标定过**：None ＝ 未标定，**不许照抄别的模型的数字**
        # 在构造时定死成属性而不是在 retrieve 里现算，是为了能被直接断言——这条的
        # 失效形态（门槛误用到零依赖路径）**行为测试抓不到**：零依赖下 cosine>0
        # 必然 bm25>0（同一套 bigram），误用门槛今天是无害的，但等哪天语义代理
        # 换好了就会变成静默的坑
        self.vec_floor = (self.provider.hit_floor() if embed else 0.0)
        # 未标定档的保守取舍（2026.08.02 云端档）：**向量路不再有资格单独把一个块
        # 送进候选**，只能给词面路已经认下的候选排序。理由——没有标定就没法分辨
        # "0.3 分是弱相关还是纯噪声"，而真 embedding 的余弦几乎恒正；放它独立开闸
        # 等于把 absent 类的正确空手率直接送成 0（bge-small-zh 上实测过这个形态）。
        # 收窄成"只排序不开闸"是两害相权：漏掉一些只有语义路找得到的块，好过
        # 每个查询都端回四条不相干的记忆——后者接着的就是编造。
        # **这不是新的默认门槛，是"没量过"的诚实形态**；量过了就填进
        # HIT_FLOOR_BY_MODEL，这段自动失效。
        self.vec_calibrated = (not embed) or (self.vec_floor is not None)
        self.graph_topK = graph_topK
        self.graph_seeds = graph_seeds
        self.chunks, self.meta, self.weights = [], [], []
        self._bm25 = self._cvecs = self._ccounts = None
        self._df_cache = {}
        self._neighbors = {}
        # 撤回集（2026.07.31 验收反馈"记错以后怎么办"闭环）：被撤回的块不再被
        # retrieve/recall 返回，但原文件与 chunks 本体都不动——时间线是档案，
        # 不销毁；退出的只是检索可见性。撤回账本按内容哈希持久化（同权重先例）
        # 语料的日期顺序推断结论（load_corpus 填）："mdy"/"dmy"/None
        self.date_order = None
        self.retracted = set()          # chunk 下标
        self.retraction_log = {}        # 内容哈希 → {reason, time, source, heading}
        # 实体标注（图谱层可插拔升级路径，任务卡"图谱实体抽取可插拔"）：
        # chunk 下标 → 实体集合。默认空——图谱走零依赖词面代理；用户花 LLM 调用
        # 抽取过实体（load_entities 接入 .entities.json）后，实体边并进图谱
        self._entities = {}

    def add(self, text, meta=None):
        self.chunks.append(text)
        self.meta.append(meta or {})
        self.weights.append(1.0)

    def build(self):
        self._bm25 = BM25([tokenize(c) for c in self.chunks])
        if self.embed:
            # 块向量：缓存里有的不重算（见 embedding_provider.VectorCache）。
            # **云端档的成本模型全靠这一条**——不缓存的话每次起服务都是全库一遍。
            cache = (VectorCache(self.cache_path, self.provider.id)
                     if self.cache_path else None)
            self._cvecs = embed_with_cache(self.provider, self.chunks, cache)
        else:
            self._ccounts = [bigram_counts(c) for c in self.chunks]
        self._neighbors = self._build_graph()
        # 文档频次缓存，供 query_miss_rate 用（零依赖，建库时顺手算一次）
        self._df_cache = {}
        for d in self._bm25.docs:
            for t in set(d):
                self._df_cache[t] = self._df_cache.get(t, 0) + 1
        return self

    def _build_graph(self):
        """第三层·关系图谱（零依赖词面代理版）：共享"显著词"的块相连。

        显著词 = 出现在少数几个块里的 bigram（2 ≤ df ≤ `_graph_df_cap()`——"望远镜"
        这类专名的碎片会同时出现在约定/兑现/后续几个块里，而"今天""然后"这类通用词
        df 高，被上限挡掉）。链接强度 = 共享显著词的 Σidf——越稀有的共享词越说明
        两块讲的是同一个具体事物。

        **上限要问 `_graph_df_cap()`，别在这儿重算一遍**：它在区分性判据之上还压了
        一个成本上界（`GRAPH_DF_CAP`），理由见那个方法的 docstring。这里抄一份就会
        跟它漂移。

        每块只存强度前 graph_topK 的邻居。诚实标注局限：这是词面代理，"约定"和
        "兑现"若不共享任何字面（连专名都换了说法），这版连不上——那是实体识别/
        语义图谱的活，跟检索层 bigram 当分词一样，等真实评估再升级，现在不假装。"""
        docs = self._bm25.docs
        inv = defaultdict(list)
        for i, d in enumerate(docs):
            for t in set(d):
                inv[t].append(i)
        max_df = self._graph_df_cap()
        link = defaultdict(float)
        for t, ds in inv.items():
            if not 2 <= len(ds) <= max_df:
                continue
            w = self._bm25.idf(t)
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    link[(ds[a], ds[b])] += w
        # 实体边（2026.07.31，设计笔记候选 4 的可插拔槽）：词面代理的已知局限是
        # "同一件事换了说法就连不上"（没有共享字面词）；用户若花过 LLM 调用抽取
        # 实体（load_entities），共享同一实体的块在这里连边，强度用 idf 同款思路
        # （越少块共享的实体区分度越高）。**与词面边是并集不是替换**——实体标注
        # 可能只覆盖部分语料，丢掉词面边等于让没标注的块之间断联
        ent_chunks = defaultdict(list)
        for i in sorted(self._entities):
            for e in self._entities[i]:
                ent_chunks[e].append(i)
        for e, ds in ent_chunks.items():
            if len(ds) < 2:
                continue
            w = math.log(1.0 + self._bm25.N / len(ds))
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
            # **每次查询只算一个查询向量**：块那边已经缓存住了（build），查询这边
            # 按次算——云端档一次检索就是一个往返，不是全库一遍
            qv = self.provider.embed([query], is_query=True)[0]
            return [sum(a * b for a, b in zip(cv, qv)) for cv in self._cvecs]
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
             "最近+曾经重要"的本意：权重是同龄块间的裁判，不是对抗新鲜度的杠杆
             ——这句承诺要靠权重乘子钳到 WEIGHT_INFLUENCE_CAP 才真成立（2026.07.31
             滚雪球治理补的：不封顶的话，权重涨到 8 就能把 14 天前的块顶过刚发生的）；
          3. 改加权和需要把 weight 和 0~1 的 recency 跨量纲归一化——正是
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
            if i in self.retracted:   # 撤回的块不参与召回（同 retrieve）
                continue
            ts = meta.get("timestamp")
            if ts is None:
                recency = 0.0  # 兜底：没时间戳当"无限旧"，不猜不崩
            else:
                recency = 0.5 ** (max(0.0, now - ts) / 86400.0 / half_life)
            # 权重钳到 WEIGHT_INFLUENCE_CAP（2026.07.31 滚雪球治理）：不封顶的话
            # 无上界权重恰好推翻 docstring 第 2 条的承诺——"权重是同龄块间的裁判，
            # 不是对抗新鲜度的杠杆"只有在权重有上限时才真成立
            scored.append((i, recency * min(self.weights[i], WEIGHT_INFLUENCE_CAP)))
        scored.sort(key=lambda x: (-x[1], -self.weights[x[0]], x[0]))
        return [{"id": i, "text": self.chunks[i], "meta": self.meta[i], "score": s}
                for i, s in scored[:topN]]


    def _distinctive_df(self):
        """"区分性 token"的 df 上限：出现在超过这么多块里的词，不算区分性词面证据。

        **刻意复用图谱路早就在用的那条判据**（`max(3, N//5)`，即"最多出现在 20% 的
        块里"），不新造一个常数——`EMBED_HIT_FLOOR` 的教训是：跟外部标度绑死的手写
        数字换个模型就失灵。这个判据是**对语料自身分布定义的**（占比，不是绝对分数），
        换语料、换语言、换规模都跟着动，不需要重新标定。

        它在图谱路的语义是"这个词稀有到能说明两块讲同一件事"，在这里是"这个词稀有到
        能说明这个块跟查询讲的是同一件事"——同一个判据的两次使用，不是两套。

        ⚠ 图谱路在这条判据**之上**另压了一个成本上界（`_graph_df_cap()`），
        那是成本约束不是判据变了，两边为什么能分开见那个方法。**候选闸这边不封顶**：
        它是"库里没有就说没有"那道闸，封顶会动到这个项目最敏感的指标。"""
        return max(3, self._bm25.N // 5)

    def _graph_df_cap(self):
        """图谱建边的 df 上限：区分性判据，再压一个绝对成本上界。

        **判据仍然只有一套**（`_distinctive_df()`，对语料自身的占比），这里只是在
        它之上加了一层"最多枚举到这么宽"的成本闸。两边之所以能分开，是因为**它们
        约束的不是同一样东西**：

        - 候选闸是 per-query 的，一个词命中多少块，成本就是多少——**线性**；
        - 图谱要把共享该词的块**两两配对**连边，成本是 df²——**平方**。

        同一个 df 值，在候选闸那边是"多看几十个块"，在图谱这边是"多枚举几十万个
        块对"。而 `N//5` 是占比，语料越大这个绝对值越大，于是建库耗时随块数平方
        增长：真实语料 7237 块建库 44.65 秒，直接撞穿客户端的 MCP 启动超时——
        表现是"记忆服务连不上"，不是"慢一点"（见《图谱建库耗时》任务卡）。

        **代价是可控的，因为算掉的那些边本来就留不下**：每块只保留强度前
        `graph_topK` 个邻居，被天花板砍掉的恰恰是 df 高、idf 低、本来就排不进前几
        的弱边。实测（外部真实语料 7237 块）：建库 44.65 s → 3.57 s（约 12.5 倍），
        图谱边 36185 → 35504（−1.9%），**hit@1 不降、第一名 0 换人**；维护者
        755 块语料同样 0 换人。

        边界要写清：那是标题自监督尺，两份语料支持"这两份上未见回归"，
        **不能外推成普适无损**。天花板 64 也不是标定出来的最优值，是一个够低到
        把平方项压住、又高到（N ≤ 320 时）完全不生效的成本线——它不改变
        "什么算区分性词"，只限制建库愿意为此枚举多少块对。"""
        return min(self._distinctive_df(), GRAPH_DF_CAP)

    def lexical_admit(self, query_tokens):
        """词面路的候选资格：块必须与查询共享**至少一个区分性 token** → set(块下标)。

        为什么需要它（2026.08.02，P0"候选闸补词面门槛"）：原来的候选闸是
        「BM25 有分 **或** 向量过槛」，而 `bm_scores[i] > 0` 这一路**压根没有门槛**。
        真实语料 602 块词汇丰富，任意查询都能跟十几到几十个块撞上 bigram 重叠、
        BM25 分还有 9.8~15.4——于是"库里根本没有的事"照样端回五条沾边记录。
        实测：远主题无关查询只挡住 2/16、**贴近生活的日常问题 0/8**。

        **短板是"分从哪来"，不是"分够不够高"**：编造查询撞上的是「我们」「那次」
        「的事」这类功能词 bigram，它们在几乎每一块里都出现。所以门槛不设成分数阈值
        （那又是一个拍脑袋的常数，而且高频词堆起来照样能过），而是问一句结构性的话：
        **这个块跟查询共享的，有没有一个是区分性的词？** 一个都没有，就说明这块只是
        跟查询共用了些人人都有的功能词，它不构成"命中"。

        这条同时解释了为什么它对"专名缺席"以外的编造也有效：判据不是"有没有专名"，
        是"共享的词面里有没有一个在这份语料里是稀有的"。"""
        max_df = self._distinctive_df()
        distinctive = {t for t in set(query_tokens)
                       if 0 < self._bm25.df.get(t, 0) <= max_df}
        # 整句都是高频功能词、一个区分性 token 都没有：**词面路一个都不放行**。
        #
        # 这一格是这条判据最要紧的地方，也是我第一版写反了的地方：当时退回旧行为
        # （有分即可），理由是"怕误杀'她最近怎么样'这类合法提问"。但那恰恰把
        # "整句都是功能词"——也就是编造查询最典型的形态——原样放了过去。
        # **没有任何区分性词面证据时，正确的答案是"我这儿没有依据"，不是"都给你"。**
        #
        # 不是拍脑袋翻的：拿本回归集两档语料 + 留出集全量查询量过一遍，
        # **present 四类里零区分性 token 的查询是 0 条**（两档都是），
        # 命中这一格的全是 absent／越界提问。误杀风险在合成集上实测为零。
        #
        # ⚠⚠ **已标定，结论是这条闸在真实语料上不起作用**（2026.08.02）。
        # 外部实测（1144 条中文语料，报告人 `lu7899112-source`）：编造题正确空手
        # **0/9**；本地 791 块真实语料独立复现：**0/15**。
        # **病因不是这个 df 上限选错，是判据挂在了产不出内容词的那一层**——
        # 九道题每道都有一个 df=1 的**跨词边界碎片**开闸（「看极光是哪天」的「光是」、
        # 「学会说什么了」的「么了」、「摔伤的是哪条腿」的「伤的」）。报告人那句话
        # 就是病因本身：**在 bigram 这一层，"内容词"这个概念本身就产不出来。**
        # 所以把 `N//5` 收紧不解决问题（实测收到 N//100 仍只有 2/15）。
        #
        # 这条闸**留着不拆**：旧行为（bm25>0 即放行）同样是 0，而且连"哪里漏"都
        # 看不见——退回去不是修好，是把仪表盘砸了。
        # 修法的判定实验已做完，**结论是"还不能改"**：候选判据全都在 absent 与
        # 改述召回之间是一条权衡曲线，没有帕累托占优的格子，而且现有两把尺子分别
        # 落在待定阈值的两侧（真实语料 present 对 99.1% 共享 ≥5 字，合成集 present
        # 查询 100% 共享 ≤4 字）——**这道题现在没有一把尺子判得了**。
        # 全部数据、被证伪的三条路线、以及下一步该先补哪把尺子，见任务卡
        # 「区分性token离开bigram层」的「判定实验结果」一节。
        if not distinctive:
            return set()
        out = set()
        for i, tf in enumerate(self._bm25.tf):
            if any(t in tf for t in distinctive):
                out.add(i)
        return out

    def retrieve(self, query, topN=5, reranker=None, coarse_topM=20, routes=None):
        """query → 前 topN 个相关片段 [{id, text, meta, score, weight}]。
        命中的片段权重 +weight_boost（用进废退），作为副作用记在 index 上。

        二阶段检索（设计笔记 待验证问题 2，任务卡"粗筛重排序两阶段"）：
        reranker 可插拔（同 draft_extraction.llm_call 的先例）——不传行为不变
        （一阶段）；传入 (query, texts)->scores 回调时，三路 RRF 融合当粗筛取前
        coarse_topM（经验值待调），reranker 对这批候选精排后取 topN。真正的重排
        该用 cross-encoder，将来从同一个槽换进来。三条规则：
          精排分数乘用进废退权重（钳到 WEIGHT_INFLUENCE_CAP）——重排不豁免，
            但跟粗筛一样不给权重无上界话语权（滚雪球治理，见融合处注释）；
          同分按粗筛名次——精排没意见时保留粗筛信号，不乱序；
          weight_boost 只加在最终 topN——粗筛池是机制内部产物，不算"被命中"。

        routes：消融开关，默认 None＝四路全开（行为与不传完全一致）。传一个子集
        （如 {"bm25", "vector"}）可以只用其中几路——回归集靠它量"每一层各自贡献
        多少"，而且量的是**真实检索路径**，不是脚本自己拼一套 rank_lists 走捷径
        （同 e2e"走用户真实路径"那条纪律）。四个键：bm25 / vector / graph / weight。
        注意可靠命中门槛不受 routes 影响——门槛问的是"这个块跟 query 有没有关系"，
        跟"用哪几路排序"是两件事。"""
        routes = {"bm25", "vector", "graph", "weight"} if routes is None else set(routes)
        bm_scores = self._bm25.scores(tokenize(query))
        vec_scores = self._vector_scores(query)
        # 可靠命中门槛（2026.07.31 验收反馈：低相关也硬凑 topN，会让一次误召回
        # 越用越重）：只有词面或语义层真有分的块、以及被它们经图谱带出的关联块，
        # 才有资格进结果——RRF 只看名次不看分数，不设门槛的话零分块照样被排进
        # 前几名、照样被 +weight_boost，错误从第一次返回就开始积累。
        # 一个都没有就返回空列表，让上层明确说"没找到"，不硬凑。
        # 门槛就设在"有没有信号"（>0），不设更高的数值阈值——没有真实评估数据
        # 之前，任何非零阈值都是拍脑袋（同 MILESTONE_BODY_LIMIT 的教训）。
        # 诚实标注：真 embedding 余弦几乎恒正（同种子过滤那条已知局限），--embed
        # 路径下这道门槛基本不起作用，待真实评估再调。
        # 撤回的块（self.retracted）在这里一并出局：不进候选、不进种子、不被
        # 图谱带回、不加权。
        # 门槛未标定时（换了 embedding 模型、还没在这个模型上量过分布）：向量路
        # **只排序、不开闸**——见 __init__ 里 vec_calibrated 那段。开闸与排序在这里
        # 是两个判据，不是一个：能不能单独把一个块送进候选，跟已经进来的块之间怎么
        # 排先后，本来就是两件事。
        vec_admit = (lambda i: vec_scores[i] > self.vec_floor) if self.vec_calibrated \
            else (lambda i: False)
        vec_rank_ok = (lambda i: vec_scores[i] > self.vec_floor) if self.vec_calibrated \
            else (lambda i: True)
        # 词面路的资格判定（2026.08.02 P0）：BM25 有分**且**共享至少一个区分性
        # token，才算词面命中。整句都是功能词时 lexical_admit 返回空集——
        # 那时词面路一个都不放行（见那里的注释：这是"知道自己不知道"的那一半）。
        lex_ok = self.lexical_admit(tokenize(query))

        def bm_admit(i):
            return bm_scores[i] > 0 and i in lex_ok

        # **零依赖档的"语义"路也得受这条门槛管**（2026.08.02，做这一单时才发现）：
        # 那一路用的是**同一套字符 bigram** 的余弦（消融早就实测过它没有独立贡献），
        # 所以它跟 BM25 是同一种证据的两种算法，不是两种证据。而它的 floor 在零依赖
        # 档是 0.0——等于没有门槛。只关 BM25 那扇门，候选照样从这扇进来：
        # 我第一版就是只关了一扇，量出来"门槛毫无效果"，差点把结论写成"这条路不通"。
        # 真 embedding 档不在此列：那一路是真正独立的语义证据，有自己标定过的门槛。
        lexical_vector = not self.embed
        if lexical_vector:
            _vec_admit_raw = vec_admit
            vec_admit = lambda i: _vec_admit_raw(i) and i in lex_ok  # noqa: E731
        scored_ok = [i for i in range(len(self.chunks))
                     if i not in self.retracted
                     and (bm_admit(i) or vec_admit(i))]
        ok = set(scored_ok)
        bm_ranks = [i for i in ranked(bm_scores) if i in ok]
        vec_ranks = [i for i in ranked(vec_scores) if i in ok and vec_rank_ok(i)]
        rank_lists, route_ks = [], []
        for r, on in ((bm_ranks, "bm25" in routes), (vec_ranks, "vector" in routes)):
            if on:
                rank_lists.append(r)
                route_ks.append(self.rrf_k)
        # 第三层：按留口设计把关联块排成第三路 append 进 RRF，融合逻辑不改。
        # 两处实现细节是实测调出来的（过程见 设计笔记"关系图谱"一节）：
        #   种子只取真有分的命中——零分并列块不是命中，它们的邻居是纯噪声；
        #   第三路结构是"种子打头、关联块随后"——RRF 只看名次，短名次路里排头的
        #   贡献(≈1/(k+1))跟整路第一名一样大，纯邻居路会让弱关联反超真命中。
        # 已知局限：真 embedding 余弦几乎恒正，"有分才算命中"的过滤在 --embed
        # 路径基本不起作用，种子退化为固定取 top graph_seeds，待真实评估再调
        if not scored_ok:
            return []   # 没有可靠命中：明确空手而归，不硬凑、不加权
        seeds = []
        for rank_list, scores in ((bm_ranks, bm_scores), (vec_ranks, vec_scores)):
            for i in rank_list[:self.graph_seeds]:
                if scores[i] > 0 and i not in seeds:
                    seeds.append(i)
        graph_route = list(seeds)
        for seed in seeds:
            for nb in self.graph_neighbors(seed):
                if nb not in graph_route and nb not in self.retracted:
                    graph_route.append(nb)
        # 带出了新关联块才追加，没有就退回两路融合
        if "graph" in routes and len(graph_route) > len(seeds):
            rank_lists.append(graph_route)
            route_ks.append(self.graph_rrf_k)   # 图谱路用更小的 k，见 rrf_fuse docstring
        # 第四路：用进废退权重（2026.07.31，滚雪球治理——维护者指出的陷阱：
        # 权重当乘子时无上界增长×近乎持平的 RRF 分，被频繁命中的块会自激滚雪球，
        # "曾经对、现在错"的旧事实靠历史命中数一直占着前排）。两条治理规则：
        #   一票封顶：权重不再乘融合分，改成一路名次并进 RRF——权重再高也只值
        #     一路投票（≤1/(k+1)），压不过多路一致的相关性优势，挤占有了上限；
        #   只裁判相关块：权重路只收本次 query 里真有分的块（BM25 或向量层 >0），
        #     零分块不能靠权重买门票——同种子零分过滤的先例（"零分并列块不是
        #     命中"），权重是相关候选之间的裁判，不是无关块的通行证。
        # 权重存储照旧无上界累加（save/load 不变），动的只是排序话语权。
        weight_route = sorted(
            (i for i in scored_ok if self.weights[i] > 1.0),
            key=lambda i: (-self.weights[i], i))
        if "weight" in routes and weight_route:
            rank_lists.append(weight_route)
            route_ks.append(self.rrf_k)
        fused = rrf_fuse(rank_lists, self.rrf_k, ks=route_ks)
        if reranker is not None:
            # 精排是乘法路径，权重乘子钳到 WEIGHT_INFLUENCE_CAP（滚雪球治理的
            # 乘法侧：重排不豁免用进废退，但同样不给无上界话语权）
            pool = fused[:coarse_topM]
            fine = reranker(query, [self.chunks[i] for i, _ in pool])
            capped = [min(self.weights[i], WEIGHT_INFLUENCE_CAP) for i, _ in pool]
            order = sorted(range(len(pool)),
                           key=lambda j: (-fine[j] * capped[j], j))
            fused = [(pool[j][0], fine[j] * capped[j]) for j in order]
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

    # ---------- 撤回与追溯（任务卡"错误记忆治理闭环"） ----------
    #
    # 验收反馈（2026.07.31）指出的缺口：自动写回没有更正/撤回，模型理解错一次，
    # 在"只进不退"的设计下，错误记忆比正确记忆更难清掉。
    #
    # 设计判断：**文件只进不退，检索可进可退**。原 md 文件永远不改不删（时间线
    # 是档案，销毁历史跟"不废退"一样是产品底线），撤回改变的只是检索可见性；
    # 撤回账本（原文哈希+原因+时间+出处）单独落盘，错误怎么进来的、什么时候
    # 被谁以什么理由撤掉的，全程可追溯。更正内容走 memory_append 正常写回，
    # 新旧两条在账本里互相指认。

    def retract(self, quote, reason, now=None, replaced_by=None):
        """撤回一段记录：quote 必须唯一定位到一个块（从检索结果里逐字摘）。
        命中零条或多条都明确报错，不猜——歧义时替调用方挑一条，就是把"清错误"
        变成"造新错误"。返回 (chunk_idx, 块原文)。

        两件事跟着一起做（验收反馈 ③ 的两个子项）：

        **旧块权重归位**：被撤回的块权重重置回 1.0。它历史上攒的命中数是**错误
        信号**——一条记错的记录被反复召回过，恰恰说明它污染过多少次对话，那些
        命中不该继续算作"这条重要"。不归位的话，权重还会随块一起落盘、重启后
        回来，留下一个高权重的死块；万一将来支持人工改正文（哈希变、撤回失效），
        它会带着这份污染权重复活。

        **追溯链**：`replaced_by` 记下更正后那条记录的内容哈希，让账本能回答
        "哪条记录改了哪条"，而不只是"这条被撤了"。调用方先写更正、再把新块文本
        传进来（MCP 的 memory_correct 就是这个顺序）。只撤不补时为 None。"""
        if not (isinstance(quote, str) and quote.strip()):
            raise ValueError("quote 必填：从检索结果里逐字摘一段要撤回的原文")
        if not (isinstance(reason, str) and reason.strip()):
            raise ValueError("reason 必填（追溯用）：为什么撤回——记错了/已过时/对方更正了……")
        quote = quote.strip()
        matches = [i for i, c in enumerate(self.chunks)
                   if quote in c and i not in self.retracted]
        if not matches:
            raise ValueError("没有找到包含这段原文的记录（或它已被撤回）。"
                             "quote 要从 memory_search 返回的原文里逐字摘，别转述。")
        if len(matches) > 1:
            raise ValueError(f"这段原文命中了 {len(matches)} 条记录，定位不到唯一一条"
                             "——换一段更长、更独特的原文再试。")
        i = matches[0]
        now = time.time() if now is None else now
        self.retracted.add(i)
        self.weights[i] = 1.0        # 权重归位：误召回攒的命中数是错误信号，不留
        self.retraction_log[_chunk_key(self.chunks[i])] = {
            "reason": reason.strip(), "time": now,
            "source": self.meta[i].get("source"), "heading": self.meta[i].get("heading"),
            "replaced_by": _chunk_key(replaced_by) if replaced_by else None,
        }
        return i, self.chunks[i]

    def link_correction(self, chunk_idx, correction_text):
        """回填追溯链：把"这条被哪条更正了"记进账本。更正内容写完之后才调得了
        （新块的内容哈希这时才存在），所以跟 retract 分成两步，不合并。"""
        key = _chunk_key(self.chunks[chunk_idx])
        if key in self.retraction_log:
            self.retraction_log[key]["replaced_by"] = _chunk_key(correction_text)
        return key

    def save_retractions(self, path):
        """撤回账本落盘（内容哈希 → 原因/时间/出处），同权重的稀疏存法。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.retraction_log, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return len(self.retraction_log)

    def load_retractions(self, path):
        """从盘上把撤回账本接回来，按内容哈希对号入座——文件改名、块顺序变动
        都不影响；被撤回的块正文若被人工编辑过，哈希变了、撤回自动失效（内容
        都变了，旧的撤回判断本来就不该继承——同权重按内容哈希的理由）。"""
        p = Path(path)
        if not p.exists():
            return 0
        self.retraction_log = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for i, c in enumerate(self.chunks):
            if _chunk_key(c) in self.retraction_log:
                self.retracted.add(i)
                n += 1
        return n

    # ---------- 实体标注接入（图谱层可插拔升级路径） ----------

    def load_entities(self, path):
        """从 .entities.json（内容哈希 → 实体列表）接入实体标注，返回接上的块数。
        按内容哈希对号入座（同权重/撤回先例）；**载入后要重新 build() 才生效**——
        图谱边在 build 时算。文件不存在按零处理。

        文件怎么来：抽取要花 LLM 调用，这笔成本花不花由用户定（同"隐私二选一
        不给静默默认值"原则，设计笔记"自动提炼草稿流程"）——用
        entity_extraction_prompt() 导出任务书给用户自己的模型（路线 C，零密钥
        零 HTTP），答案经 entities_from_answer() 转成本文件格式落盘。"""
        p = Path(path)
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for i, c in enumerate(self.chunks):
            ents = data.get(_chunk_key(c))
            if ents:
                self._entities[i] = set(ents)
                n += 1
        return n


def entity_extraction_prompt(index, max_chunks=50):
    """给用户自己的模型的实体抽取任务书（路线 C：不内置 API，导出 prompt）。
    语料超过 max_chunks 块时只出前一批并明确说明——长语料分批抽，不静默截断。"""
    total = len(index.chunks)
    lines = [
        "下面是一批记忆片段。给每一段列出它提到的**具体实体**：人、物件、地点、",
        "事件名、约定名——要求是「换个说法仍指同一个东西」的那类词（如「那台望远镜」",
        "里的「望远镜」）。不要抽情绪词、形容词、日期。每段 0~5 个，没有就给空列表。",
        "输出 JSON：{\"1\": [\"实体A\", \"实体B\"], \"2\": [], ...}，键是片段编号，只输出 JSON。",
        "",
    ]
    for i, c in enumerate(index.chunks[:max_chunks], 1):
        lines.append(f"[{i}] {c}")
        lines.append("")
    if total > max_chunks:
        lines.append(f"（共 {total} 块，本批只含前 {max_chunks} 块；剩余分批再抽，别漏）")
    return "\n".join(lines)


def entities_from_answer(index, numbered, offset=0):
    """模型答案（{"1": [...], ...}，编号从 1 起）→ .entities.json 的内容
    （内容哈希 → 实体列表）。offset 用于分批：第二批 offset=50 时编号 1 对应
    第 51 块。编号越界明确报错，不静默丢。"""
    out = {}
    for k, ents in numbered.items():
        i = int(k) - 1 + offset
        if not 0 <= i < len(index.chunks):
            raise ValueError(f"编号 {k}（offset={offset}）超出语料范围（共 {len(index.chunks)} 块）")
        if ents:
            out[_chunk_key(index.chunks[i])] = sorted(set(ents))
    return out


def query_miss_rate(index, query):
    """查询里有多少比例的词，在这个记忆库里从未出现过（去重 token 中 df==0 的占比）。

    零依赖，只用现成的 BM25 文档频次。返回 0.0~1.0；查询切不出 token 时返回 1.0
    （切不出词＝没有任何可对照的信号，按"完全陌生"处理，偏保守）。

    **这是标注用的信号，不是门槛**——理由见 MISS_RATE_FLAG 的注释。调用方拿它
    生成一句可核对的话给模型，不拿它拒绝返回结果。"""
    tokens = set(tokenize(query))
    if not tokens:
        return 1.0
    docs = getattr(index, "_bm25", None)
    if docs is None:
        return 1.0
    df = index._df_cache
    missing = sum(1 for t in tokens if df.get(t, 0) == 0)
    return missing / len(tokens)


def miss_rate_note(rate, threshold=None):
    """把缺失率变成一句**可核对**的标注（模型能自己对着查询复核这句话对不对）。
    低于阈值返回 None——不给噪声。"""
    threshold = MISS_RATE_FLAG if threshold is None else threshold
    if rate < threshold:
        return None
    return (f"【核对提示】这次查询里约 {rate:.0%} 的词在这个记忆库里从未出现过。"
            f"这**不代表**这件事一定没发生过——你用的词跟记录里的写法不一样时也会这样"
            f"（尤其是问'她最近在焦虑什么'这类归纳性问题）。"
            f"但如果下面这些片段跟你问的对不上，**优先考虑库里确实没记过这件事**，"
            f"如实说没找到，不要拿沾边的记录去圆。")


def annotate_block(block, rate, threshold=None):
    """把检索结果块与缺失率标注拼在一起。**拼接这一步也放在库里，不放 MCP 外壳**——
    外壳的硬性约束是"返回值逐字等于底层库函数的返回值"，让它自己拼字符串就等于
    在适配层里长逻辑，真机出问题时又分不清是谁的锅了（同 mcp_server 顶部那条纪律）。
    低于阈值时原样返回，不加噪声。"""
    note = miss_rate_note(rate, threshold)
    return block + ("\n" + note if note else "")


def _chunk_key(text):
    """权重持久化的键：chunk 文本的内容哈希（md5 前 16 位，非安全用途）。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ---------- chunk 时间戳（recall_recent 按新鲜度排序的前提，任务卡第 1 条） ----------

# 文件名里的完整日期：2026-07-29 / 2026.7.29 / 20260729 / 2026年7月29日 都认
_DATE_FULL_RE = re.compile(r"(20\d{2})[-._年]?(\d{1,2})[-._月]?(\d{1,2})")
# 正文/标题行里的完整日期：分隔符必带——紧凑 8 位（20260729）在文件名里是命名惯例，
# 在正文里更可能是单号/编号这类数字，不认，压误判
_DATE_FULL_TEXT_RE = re.compile(r"(20\d{2})[-._年](\d{1,2})[-._月](\d{1,2})")
# Claude Exporter 插件时间戳格式：M/D/YYYY HH:MM:SS（在 > 引用行里，chunk 开头处）。
# **这段正则由内测用户贡献**（2026.08.02，他直接拿导出器的 md 建库时撞上的）。
# 只在 _head_lines 里查，不扫正文，压误判——实测正文里的"7/11 优惠"因此没被命中。
_DATE_MDY_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(20\d{2})")
# 斜杠 + 年在前："2026/7/15"。年份在最前面就没有歧义，跟上面那条分开写
_DATE_YMD_SLASH_RE = re.compile(r"(20\d{2})/(\d{1,2})/(\d{1,2})")

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


def _slash_date_ts(line, date_order=None):
    """一行文本里的斜杠日期 → epoch 秒；认不出、或**歧义且没有依据**时返回 None。

    两种形态分开处理，因为歧义程度完全不同：
      - `2026/7/15`（年在前）：没有歧义，直接认；
      - `7/22/2026` / `22/7/2026`（年在后）：**顺序本身是歧义的**。
        `7/22` 一眼能定（22 只能是日）；`3/5` 两解都通。

    歧义那一格的处理是这一单的核心：**不默认按美式解**。
    有决定性证据（同一份语料里出现过 次位>12 → 月/日序，或 首位>12 → 日/月序）
    就按 date_order 统一解释整份；**没有证据就返回 None，退到下一级来源**，
    宁可退 mtime 也不编一个可能差几个月的日期。同 absent 防线"宁可说不知道也不编"、
    缺失率"宁可漏报不可误伤"。

    date_order: "mdy" / "dmy" / None（未知）。由 infer_date_order 在语料范围内推断。"""
    m = _DATE_YMD_SLASH_RE.search(line)
    if m:
        return _ymd_ts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _DATE_MDY_RE.search(line)
    if not m:
        return None
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if b > 12:                      # 次位只能是日 → 月/日/年
        return _ymd_ts(y, a, b)
    if a > 12:                      # 首位只能是日 → 日/月/年（欧式）
        return _ymd_ts(y, b, a)
    if date_order == "mdy":
        return _ymd_ts(y, a, b)
    if date_order == "dmy":
        return _ymd_ts(y, b, a)
    return None                     # 歧义且无依据：不认，交给下一级兜底


def infer_date_order(texts):
    """一批文本（整份语料的开头几行）→ "mdy" / "dmy" / None。

    找**决定性证据**：`7/22/2026` 的 22 只能是日 → 这份语料是月/日/年；
    `22/7/2026` 的 22 在首位 → 日/月/年。两种证据都出现＝互相矛盾，返回 None
    （语料本身不自洽，猜哪一边都可能错半份）。一条证据都没有也返回 None。

    **推断放在语料范围、不放在单块**：单块只有一个日期时永远推不出来，而同一份
    导出里的日期格式是同一个导出器写的、必然一致——把范围放大到整份语料，
    才可能从别的块借到那条决定性证据。这也是为什么它属于 load_corpus 的第一趟。"""
    mdy = dmy = 0
    for text in texts:
        for line in _head_lines(text):
            if _DATE_YMD_SLASH_RE.search(line):
                continue            # 年在前的形态没有歧义，不构成顺序证据
            m = _DATE_MDY_RE.search(line)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if b > 12:
                mdy += 1
            elif a > 12:
                dmy += 1
    if mdy and dmy:
        return None                 # 自相矛盾，不选边
    if mdy:
        return "mdy"
    if dmy:
        return "dmy"
    return None


def parse_chunk_timestamp(filename, chunk_text, fallback_year, date_order=None):
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
        # 斜杠日期（Claude Exporter 的 `> 7/22/2026 12:54:09` 引用行、以及
        # `2026/7/15`）。**只在 head 里查，不扫正文**——正文里的"7/11 优惠"
        # 会被读成 7 月 11 日，这条约束不许放宽，要放宽先拿断言证明不引入误判。
        # 歧义且无依据时 _slash_date_ts 返回 None，于是这里不 return、继续往下
        # 一级来源走，而不是默认按美式解
        ts = _slash_date_ts(line, date_order)
        if ts is not None:
            #    单独一档来源：这样"这个块的日期是从斜杠格式来的、按哪种顺序解的"
            #    在 meta 里看得见，不跟 ISO 那档混在一起
            return ts, "chunk_head_slash"
    for line in chunk_text.splitlines():
        if not line.lstrip().startswith("#"):
            continue                      # 只认标题行；正文里的"3.5 倍"不该被读成日期
        ts = _first_valid_short_date(line, fallback_year)
        if ts is not None:
            return ts, "heading"
    return None, None


def timeline_chunks(text, filename, mtime, date_order=None):
    """单个 timeline md 文件 → [(chunk_text, meta)]。切块+时间戳解析的公共段：
    load_corpus 和 memory_import 的 markdown 翻译器共用，逻辑只此一份不复制。"""
    year = datetime.fromtimestamp(mtime).year
    # 调用方（load_corpus）会把**整份语料**推断出的顺序传进来；单文件导入
    # （memory_import 那条路）没有语料范围，就退而在本文件范围内推断——
    # 仍然是找决定性证据，不是猜
    if date_order is None:
        date_order = infer_date_order([text])
    # 文件级日期：日期常常只写在文件标题行、只属于第一个 chunk——先按文件解析一次，
    # 让同文件其余 chunk 继承，否则一个文件十几个块除首块外全落 mtime，
    # per-chunk 解析救不回兜底率
    file_ts, _ = parse_chunk_timestamp(filename, "\n".join(_head_lines(text)), year,
                                       date_order)
    out = []
    for i, chunk in enumerate(chunk_heading(text)):
        heading = next((ln for ln in chunk.splitlines() if ln.startswith("## ")), "")
        ts, ts_source = parse_chunk_timestamp(filename, chunk, year, date_order)
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


def load_corpus(corpus_dir, embed=False, recursive=True, provider=None, cache_path=None):
    """语料目录（可传多个）→ 建好的 MemoryIndex。

    embed 档下**默认把块向量缓存落在语料目录里**（`.embed_cache.json`，同
    `.weights.json` 的先例，已进 .gitignore）——不缓存的话每次起服务都要把全库
    重算一遍，云端档就是每次开机几百个往返。传 cache_path 可以改地方，传空字符串
    可以关掉。语料目录传的是多个时不猜落点，得显式给。

    每块 meta：source/heading/chunk_index/timestamp/timestamp_source
              + window（窗口号，解析不出为 None）+ layer（index/timeline）。

    时间戳兜底顺序（前三级在 parse_chunk_timestamp 里，后两级在这里）：
      文件名日期 > chunk 开头完整日期 > 开头短日期 > `## ` 短日期
      > 文件级日期继承（同文件其余块）
      > **同窗口号的邻层文件**（window_sibling）——index 层 36 个文件全是无标题
        纯段落、文件名也不带日期，本来 34/37 块落 mtime；但它跟 timeline 层同名
        同窗口号，是同一次会话的两种写法，借它的日期是有据可依的，不是猜
      > mtime（最后兜底。⚠ **它不是"基本没有排序信号"，是给出方向错误的信号**：
        全新 clone／刚落盘会把这些块刷成"刚刚"，于是最没有日期依据的内容拿到全库
        最新的时间戳，换窗召回按新鲜度排序时最旧的事实冒充最新的。2026.08.03 跨模型
        实测量到过这条：那么一份语料下两家模型 0/6 答对当下状态。召回层已对这一档
        改标「时间未知」不再打日期，`--doctor` 也有专门判据，但**根子仍是语料没日期**）"""
    if embed and cache_path is None and isinstance(corpus_dir, (str, Path)):
        cache_path = Path(corpus_dir) / ".embed_cache.json"
    index = MemoryIndex(embed=embed, provider=provider, cache_path=cache_path or None)
    files = corpus_files(corpus_dir, recursive=recursive)

    # 第一趟：解析每个文件的文件级日期与窗口号，供第二趟跨层继承用。
    # **顺序推断先于日期解析**：`3/5/2026` 这种歧义日期要靠同一份语料里别的块
    # （比如 `7/22/2026`）提供的决定性证据才敢解，所以先把所有文本读进来推一次，
    # 再拿结论去解析——这也是它属于第一趟的原因
    raw = {}
    for p in files:
        raw[p] = (p.read_text(encoding="utf-8"), p.stat().st_mtime)
    date_order = infer_date_order([t for t, _ in raw.values()])
    file_info = {}
    for p in files:
        text, mtime = raw[p]
        ts, _ = parse_chunk_timestamp(p.name, "\n".join(_head_lines(text)),
                                      datetime.fromtimestamp(mtime).year, date_order)
        file_info[p] = {"text": text, "mtime": mtime, "file_ts": ts,
                        "window": parse_window_no(p.name), "layer": layer_of(p)}
    # 窗口号 → 该窗已知的日期（同一窗口号的不同层共享一次会话，取先解析出的那个）
    window_ts = {}
    for info in file_info.values():
        if info["window"] is not None and info["file_ts"] is not None:
            window_ts.setdefault(info["window"], info["file_ts"])

    for p in files:
        info = file_info[p]
        for chunk, meta in timeline_chunks(info["text"], p.name, info["mtime"], date_order):
            meta["window"], meta["layer"] = info["window"], info["layer"]
            if meta["timestamp_source"] == "mtime":
                borrowed = window_ts.get(info["window"])
                if borrowed is not None:
                    meta["timestamp"], meta["timestamp_source"] = borrowed, "window_sibling"
            index.add(chunk, meta)
    # 推断结论挂在库上，**让它可见**：出了问题能一眼看出"这份语料被按哪种顺序解的"，
    # 而不是只能从时间戳倒推。None＝没找到决定性证据，那些歧义日期没被采信
    index.date_order = date_order
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

    # 1. 检索命中 + 元数据完整。topN 是上限不是保底（可靠命中门槛，见第 15 项）：
    #    这个 query 只有咖啡机块真有信号，topN=2 也只该回 1 条，不硬凑
    r = idx.retrieve("咖啡机坏了是什么原因", topN=2)
    assert "保险丝熔断" in r[0]["text"], f"BM25层该把咖啡机块排第一：{r[0]['meta']}"
    assert r[0]["meta"]["heading"] == "修理咖啡机"
    assert set(r[0]) == {"id", "text", "meta", "score", "weight"}, "结果字段：id/text/meta/score/weight"
    r1b = idx.retrieve("薄荷 咖啡机", topN=1)
    assert len(r1b) == 1, "两块都有信号时 topN 上限生效"

    # 2. BM25 纯层：idf 让稀有词区分度更高，命中块得分 > 无关块
    bm = idx._bm25.scores(tokenize("薄荷 盆底积水"))
    assert bm[2] == max(bm), "薄荷块 BM25 分最高"

    # 3. RRF 融合是纯函数：两路名次都把 0 排前 → 0 胜
    fused = rrf_fuse([[0, 1], [0, 1]])
    assert fused[0][0] == 0

    # 4.【滚雪球治理靶心（2026.07.31 维护者指出的陷阱）】权重是第四路投票，不是乘子
    #  a) 一票机制本身：权重路把相邻名次翻过来（1 多拿一路投票 → 反超 0）
    fused_w = rrf_fuse([[0, 1], [0, 1], [1]])
    assert fused_w[0][0] == 1, "权重路的一票该翻得动相邻名次"
    #  b) 一票封顶（变异靶心：改回 fused *= weights 必红）：天价权重 + 与 query
    #     零相关的块，既进不了权重路（零分过滤）也压不过真命中——乘法时代
    #     ×99 会把它直接抬到第一
    idx4b = _build_synth(embed=embed)
    idx4b.weights[3] = 99.0                       # 读书块，与咖啡机 query 零相关
    r4b = idx4b.retrieve("咖啡机坏了是什么原因", topN=2)
    assert r4b[0]["meta"]["heading"] == "修理咖啡机", \
        f"零相关块权重再高也不该压过真命中，实际第一是 {r4b[0]['meta']}"
    #  c) 相关块之间权重仍是裁判（变异靶心：去掉权重第四路必红）：两块都真命中
    #     query 时，被反复聊起的那块靠一票排到前面
    idx4c = MemoryIndex()
    idx4c.add("## 修理咖啡机\n加热管不工作，换保险丝后正常。")
    idx4c.add("## 咖啡机除垢\n给咖啡机做除垢保养，柠檬酸泡了两轮。")
    idx4c.add("## 一次远足\n上周六去了青云山，山路十二公里。")
    idx4c.build()
    base4c = [x["id"] for x in idx4c.retrieve("咖啡机", topN=2)]  # 先看无权重的名次
    idx4c.weights = [1.0, 1.0, 1.0]               # 清掉刚才那次命中的加权
    idx4c.weights[base4c[1]] = 2.0                # 把原本第二的相关块顶起来
    r4c = idx4c.retrieve("咖啡机", topN=2)
    assert r4c[0]["id"] == base4c[1], "同为相关块，权重高的该被一票顶到前面"

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

    # 9b.【滚雪球治理·recall 侧靶心（变异：去掉 min(...,CAP) 必红）】权重涨到 8
    #     也顶不过 14 天的年龄差——docstring"权重是同龄块间的裁判"的承诺靠钳制
    #     才成立：0.25×min(8,2)=0.5 < 1.0，不钳的话 0.25×8=2.0 反超刚发生的块
    idx5b = MemoryIndex()
    idx5b.add("刚发生·从未被提起", {"timestamp": now})
    idx5b.add("十四天前·被聊起过很多次", {"timestamp": now - 14 * DAY})
    idx5b.weights[1] = 8.0
    r5b = idx5b.recall_recent(topN=2, now=now)
    assert r5b[0]["id"] == 0, "recall 里权重影响力有上限，堆命中数买不来对抗新鲜度的杠杆"

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
    #  b) 精排接管排序（2026.07.31 随可靠命中门槛改写：零分块进不了粗筛池，
    #     精排只在真候选之间接管——原版让精排把与 query 零相关的读书块捞到第一，
    #     那正是门槛要拦的行为）。两个咖啡机块都是真候选，精排偏爱除垢块 → 反超
    def _build_two_coffee():
        i = MemoryIndex()
        i.add("## 修理咖啡机\n加热管不工作，换保险丝后通电正常。")      # 0
        i.add("## 咖啡机除垢\n给咖啡机做除垢保养，柠檬酸泡了两轮。")    # 1
        i.add("## 一次远足\n上周六去了青云山，山路十二公里。")          # 2
        return i.build()
    r13 = _build_two_coffee().retrieve(
        "咖啡机", topN=2, reranker=lambda q, ts: [1.0 if "除垢" in t else 0.5 for t in ts])
    assert r13[0]["id"] == 1, "精排该能在真候选之间接管排序"
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
    #     常数精排分会靠"保持粗筛序"蒙混过关；让除垢块精排分更低（0.6）但权重
    #     更高（2.0）：0.6×2.0 > 1.0×1.0，只有精排阶段真乘了权重它才能排第一
    idx13d = _build_two_coffee()
    idx13d.weights[1] = 2.0
    r13d = idx13d.retrieve("咖啡机", topN=2, coarse_topM=4,
                           reranker=lambda q, ts: [0.6 if "除垢" in t else 1.0 for t in ts])
    assert r13d[0]["id"] == 1, "精排分 0.6×权重 2.0 该压过 1.0×1.0"
    #  e)【滚雪球治理·精排侧靶心（变异：精排乘子不钳必红）】权重涨到 8 也只按
    #     上限 2.0 参与精排：0.3×min(8,2)=0.6 < 1.0×1.0，不钳的话 0.3×8=2.4 反超
    idx13e = _build_two_coffee()
    idx13e.weights[1] = 8.0
    r13e = idx13e.retrieve("咖啡机", topN=2, coarse_topM=4,
                           reranker=lambda q, ts: [0.3 if "除垢" in t else 1.0 for t in ts])
    assert r13e[0]["id"] != 1, "精排里权重影响力同样有上限，天价权重×低精排分不该赢"

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

    # 14b.【靶心：门槛的适用范围】两条路径的余弦是**两套标度**，门槛只归真
    #      embedding 用。这条只能做结构性断言：它的失效形态（把 0.45 套到零依赖
    #      路径上）**行为测试抓不到**——零依赖下 cosine>0 必然 bm25>0（两者用同一套
    #      bigram），误用今天不改变任何结果。但等语义代理换好了它就是个静默的坑，
    #      所以把决策定死成构造时的属性，直接断言
    assert MemoryIndex(embed=False).vec_floor == 0.0, \
        "零依赖路径不该有数值门槛——bigram 余弦是另一套标度，套上会掐死向量路"
    assert MemoryIndex(embed=True).vec_floor == EMBED_HIT_FLOOR, \
        "真 embedding 路径必须带数值门槛，否则 absent 类的正确空手率会归零"

    # 14b-2.【靶心：换了模型就得重新标定，不许照抄】（2026.08.02 云端档）
    #      失效形态跟 14b 是一对：14b 防的是"门槛套到不该用它的路径上"，这里防的是
    #      "门槛跟着到了不该用它的模型上"。后者更危险——0.45 是 bge-small-zh-v1.5
    #      的余弦标度，换个模型标度就变了，照抄不会报错、只会让"库里没有就说没有"
    #      无声失灵（那正是通往编造的那条流水线）。
    #      ⚠ 夹具用 `UNCALIBRATED_FIXTURE_MODEL`（一个永远不会进标定表的假模型名），
    #      不用真实模型名：bge-m3 在这儿当过一阵夹具，2026.08.03 它一被标定（外部
    #      PR #4，0.60），这条和下面整个索引的靶心当场全红。**修法是换夹具名，不是
    #      删断言**——它们守的是"未标定绝不退回 0.45"这个最坏方向。
    from embedding_provider import (HTTPCloudProvider, LocalProvider,
                                    _fake_transport, get_hit_floor,
                                    UNCALIBRATED_FIXTURE_MODEL)
    _env = {"MEMORY_EMBED_API_KEY": "k"}
    uncal = HTTPCloudProvider("https://api.example.com/v1/embeddings", UNCALIBRATED_FIXTURE_MODEL,
                              transport=_fake_transport(), env=_env)
    assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL) is None, "这条断言的前提：这个夹具模型没标定过"
    idx_uncal = MemoryIndex(embed=True, provider=uncal)
    assert idx_uncal.vec_floor is None and not idx_uncal.vec_calibrated, \
        "未标定的模型必须如实是 None，不许退回 0.45"
    assert MemoryIndex(embed=True, provider=LocalProvider()).vec_calibrated, \
        "标定过的模型该走标定值那条路"
    #      未标定档的行为约定：向量路只排序、不单独放行候选。**变异靶心**——
    #      把 vec_admit 写成 vec_rank_ok（即恢复"恒真开闸"）时这条必红：
    #      假向量对任何查询都给正分，词面零信号的查询会被向量路放进候选。
    idx_uncal2 = MemoryIndex(embed=True, provider=uncal)
    for text, meta in SYNTH:
        idx_uncal2.add(text, meta)
    idx_uncal2.build()
    assert idx_uncal2.retrieve("量子对撞机的运行日志", topN=3) == [], \
        "门槛未标定时向量路不该独立放行候选——否则每个查询都端回几条不相干的记忆"
    assert idx_uncal2.retrieve("咖啡机", topN=3), "词面有信号的查询仍该正常返回"

    # 14b-3.【靶心：块向量建库算一次，查询只付一个往返】（2026.08.02 云端档）
    #      这条决定了云端档的成本模型：不缓存的话每次起服务都是全库一遍，600 块
    #      就是 600 次云端调用，**而这件事在"单查耗时"那张表上完全看不见**。
    with tempfile.TemporaryDirectory() as td:
        cpath = Path(td, ".embed_cache.json")
        pv1 = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                                transport=_fake_transport(), env=_env)
        i1 = MemoryIndex(embed=True, provider=pv1, cache_path=cpath)
        for text, meta in SYNTH:
            i1.add(text, meta)
        i1.build()
        assert pv1.texts_embedded == len(SYNTH), "建库该把每块算一次"
        i1.retrieve("咖啡机", topN=3)
        assert pv1.texts_embedded == len(SYNTH) + 1, \
            "一次查询只该多算一个查询向量，不是把全库重算一遍"
        #   换窗开场召回（recall_recent）**一个 embedding 都不该花**：它按时间
        #   新鲜度×权重排序，压根没有 query 向量可算。这是云端档最该单独算账的
        #   地方（每开一次新会话都要付、用户正等着第一句话），实测代价是零。
        before = pv1.texts_embedded
        i1.recall_recent(topN=3)
        assert pv1.texts_embedded == before, "开场召回不该触发任何 embedding 调用"
        #   重启（重新建库）：块向量全部命中缓存，一次都不重算
        pv2 = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                                transport=_fake_transport(), env=_env)
        i2 = MemoryIndex(embed=True, provider=pv2, cache_path=cpath)
        for text, meta in SYNTH:
            i2.add(text, meta)
        i2.build()
        assert pv2.texts_embedded == 0 and pv2.calls == 0, \
            f"重启后块向量该全部走缓存，实际又算了 {pv2.texts_embedded} 条"
        #   缓存文件里只有提供方标识和向量——key 这类凭证一个字都不许落盘
        assert set(json.loads(cpath.read_text(encoding="utf-8"))) == {"provider", "vectors"}

    # 14c.【缺失率标注：**只标注、不硬拒**】这是本信号的核心纪律——它的误杀全部
    #      落在"抽象归纳式提问"那一档（陪伴场景最有价值的一类），拿它去拒绝返回
    #      等于专门惩罚最该被答好的问题。所以断言两件事：标注该出现时出现；
    #      **出现与否都不改变 retrieve 返回什么**
    idx14c = _build_synth(embed=embed)
    low = query_miss_rate(idx14c, "咖啡机 保险丝")
    high = query_miss_rate(idx14c, "量子对撞机的运行日志")
    assert low < high, f"库里有的词该比库里没有的词缺失率低（{low:.2f} vs {high:.2f}）"
    assert miss_rate_note(high) and "核对提示" in miss_rate_note(high), "高缺失率该给标注"
    assert miss_rate_note(0.0) is None, "低缺失率不该加噪声"
    #      标注里必须留一句"这不代表一定没发生"——信号是专名缺席检测器，
    #      不是事件存在性检测器，措辞不许把它说成后者
    assert "不代表" in miss_rate_note(high), \
        "标注必须写明高缺失率≠这件事没发生（它只是专名缺席检测器），否则会诱导模型误判"
    #      不硬拒：同一个查询，检索结果与标注无关
    r_before = [x["id"] for x in _build_synth(embed=embed).retrieve("量子对撞机的运行日志", topN=5)]
    r_after = [x["id"] for x in _build_synth(embed=embed).retrieve("量子对撞机的运行日志", topN=5)]
    assert r_before == r_after, "缺失率标注不该改变检索返回什么（只标注、不硬拒）"
    #      拼接放在库里（外壳不许自拼，见 mcp_server 薄适配层纪律）
    assert annotate_block("正文", 0.0) == "正文", "低缺失率原样返回"
    assert annotate_block("正文", high).startswith("正文\n"), "高缺失率把标注追加在后面"

    # 15.【可靠命中门槛靶心（2026.07.31 验收反馈；变异：去掉 scored_ok 过滤必红）】
    #     词面/语义都零信号的 query → 空手而归，且一个块都不加权——不设门槛的话
    #     RRF 照样排出 topN、照样 +boost，一次误召回从第一次返回就开始越用越重
    idx15 = _build_synth(embed=False)   # 门槛只在零依赖路径有效（embed 余弦恒正，已标注）
    w15 = list(idx15.weights)
    assert idx15.retrieve("量子对撞机的运行日志", topN=3) == [], \
        "零信号 query 该明确空手而归，不硬凑 topN"
    assert idx15.weights == w15, "没有可靠命中就不该有任何块被加权"

    # 16.【撤回与追溯靶心】文件只进不退，检索可进可退
    idx16 = _build_synth(embed=False)
    #  a) 歧义与未命中都明确报错，不猜（变异：歧义取第一个必红）
    try:
        idx16.retract("咖啡机", "测试")   # SYNTH 里只有一块含"咖啡机"——先造歧义
        pass
    except ValueError:
        raise AssertionError("单一命中不该报错")
    idx16b = MemoryIndex()
    idx16b.add("## 一\n咖啡机坏了")
    idx16b.add("## 二\n咖啡机修好了")
    idx16b.build()
    for bad_quote, why in (("咖啡机", "歧义"), ("烤箱", "未命中")):
        try:
            idx16b.retract(bad_quote, "测试")
            raise AssertionError(f"{why}的 quote 该报错，不该静默挑一条")
        except ValueError:
            pass
    #  b) 撤回后 retrieve 与 recall 都不再返回（变异：任一路不排除必红）
    idx16c = _build_synth(embed=False)
    for m in idx16c.meta:
        m["timestamp"] = 1_800_000_000.0
    i16, _ = idx16c.retract("保险丝熔断", "维修方案已过时", now=1_800_000_100.0)
    assert all("保险丝" not in x["text"]
               for x in idx16c.retrieve("咖啡机坏了是什么原因", topN=4)), \
        "撤回的块不该再被 retrieve 返回"
    assert all(x["id"] != i16 for x in idx16c.recall_recent(topN=4, now=1_800_000_200.0)), \
        "撤回的块不该再被 recall 召回"
    #  c) 账本落盘可追溯 + 重启后仍生效（变异：load_retractions 不接线必红）；
    #     chunks 本体一个不少——档案不销毁，退出的只是检索可见性
    assert len(idx16c.chunks) == len(SYNTH), "撤回不删块，原文是档案"
    with tempfile.TemporaryDirectory() as td16:
        rp = Path(td16) / ".retractions.json"
        idx16c.save_retractions(rp)
        log = json.loads(rp.read_text(encoding="utf-8"))
        entry = next(iter(log.values()))
        assert entry["reason"] == "维修方案已过时" and entry["source"] == "a.md", \
            "账本要记全：原因/时间/出处，错误怎么被清掉的要可追溯"
        idx16d = _build_synth(embed=False)          # "重启"：从头建库
        assert idx16d.load_retractions(rp) == 1
        assert all("保险丝" not in x["text"]
                   for x in idx16d.retrieve("咖啡机坏了是什么原因", topN=4)), \
            "撤回要撑过重启——不然'改过来了'只活一个进程（失败得像成功）"

    # 17.【图谱实体可插拔槽（设计笔记候选 4）】词面代理的已知局限："同一件事
    #     换了说法"没有共享字面词就连不上——实体标注接上后要连得上
    def _build_17():
        i17 = MemoryIndex()
        i17.add("## 山顶的约定\n那晚在山顶聊到以后，说好要买一台能看土星的家伙。")  # 0
        i17.add("## 到货\n快递终于送来了，装在阳台，晚上迫不及待试了试。")            # 1
        i17.add("## 学做咖喱\n试了椰浆咖喱，小火慢炖差点糊锅。")                      # 2
        return i17
    base17 = _build_17().build()
    r17 = base17.retrieve("山顶 土星", topN=3)
    assert all(x["id"] != 1 for x in r17), \
        "前提确认：无实体标注时换了说法的块连不上（词面代理已知局限）"
    #  a)【变异靶心：_build_graph 忽略实体边必红】实体标注接上 → 图谱带回换说法的块
    idx17 = _build_17()
    with tempfile.TemporaryDirectory() as td17:
        ep = Path(td17) / ".entities.json"
        ep.write_text(json.dumps(
            {_chunk_key(idx17.chunks[0]): ["望远镜"],
             _chunk_key(idx17.chunks[1]): ["望远镜"]}, ensure_ascii=False),
            encoding="utf-8")
        assert idx17.load_entities(ep) == 2
    idx17.build()
    assert 1 in idx17.graph_neighbors(0), "共享实体的块该连上边"
    r17b = idx17.retrieve("山顶 土星", topN=3)
    assert any(x["id"] == 1 for x in r17b), \
        "实体标注接上后，换了说法的关联块该被图谱带进结果"
    #  b)【变异靶心：实体边替换而非并集必红】只标注了一对块时，词面代理的边不能丢
    idxg2 = MemoryIndex()
    for t in ["## 镜片保养\n给望远镜擦镜片，用了专用的软布。",
              "## 挑设备\n看了三款设备，那台望远镜带脚架。",
              "## 无关\n中午吃了拌面。"]:
        idxg2.add(t)
    idxg2._entities = {0: {"软布"}, 2: {"软布"}}   # 只给 0/2 标实体
    idxg2.build()
    assert 1 in idxg2.graph_neighbors(0), "没被实体标注覆盖的词面边（共享'望远镜'）不能丢——并集不是替换"
    assert 2 in idxg2.graph_neighbors(0), "实体边也在"
    #  c) 任务书与答案回转（路线 C 闭环）：编号→哈希，越界报错不静默丢
    p17 = entity_extraction_prompt(base17)
    assert "[1]" in p17 and "只输出 JSON" in p17
    conv = entities_from_answer(base17, {"1": ["望远镜"], "2": []})
    assert conv == {_chunk_key(base17.chunks[0]): ["望远镜"]}, "空列表不占条目，编号从 1 起"
    try:
        entities_from_answer(base17, {"9": ["x"]})
        raise AssertionError("越界编号该报错")
    except ValueError:
        pass
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
        # 断"两个文件都被递归找到"，不断块数——块数是代理量，会随切块规则变动
        # （2026.08.03 切块加下界后，文档首行 `# 标题` 不再单独成块，这里原本
        # 写死的 >=3 就误红了。**测递归就断递归**，别拿块数当替身）
        assert {m["source"] for m in corpus.meta} == {"window_07_某某.md",
                                                      "window_07.md"}, \
            f"递归该吃到两层子目录下的两个文件：{ {m['source'] for m in corpus.meta} }"
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

    # 15b.【变异靶心：导出器原样格式的日期】**这一类语料从来没进过任何测试**：
    #      我们自己的 write_corpus 写出的是 ISO（`# 第1个窗口 · 2025-07-15`），
    #      所有自产自销的回归都走 ISO；memory_import 那边认的是 json 结构。
    #      而"直接拿导出器吐出来的 md 建库"是最省事、因此最多人走的一条路——
    #      2026.08.02 内测用户就是这么撞上的：头行 `> 7/22/2026 12:54:09` 一个都
    #      认不出来，整份语料的块全落 mtime，开场召回按新鲜度排序当场失效。
    #      语料样本是自己造的（用户明确说内容私人不给，我们也不要）。
    def _mk(td, name, text):
        Path(td, name).write_text(text, encoding="utf-8")

    #   a) `> M/D/YYYY HH:MM:SS`：22 只能是日，单块就能定，不依赖语料级推断
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "chat.md", "# 聊天记录\n> 7/22/2026 12:54:09\n\n## 晚上\n聊了买望远镜的事。\n")
        c = load_corpus(td)
        m = c.meta[0]
        assert m["timestamp_source"] == "chunk_head_slash", \
            f"导出器格式该被认出来，而不是落兜底：{m['timestamp_source']}"
        assert datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d") == "2026-07-22"

    #   b) `2026/7/15`：斜杠但年在前，没有歧义
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "a.md", "# 2026/7/15 那天\n\n## 下午\n修好了咖啡机。\n")
        m = load_corpus(td).meta[0]
        assert m["timestamp_source"] == "chunk_head_slash"
        assert datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d") == "2026-07-15"

    #   c) **歧义 + 语料里有决定性证据**：3/5 两解都通，但同一份语料里的 7/22
    #      证明这个导出器写的是月/日/年，于是 3/5 按 3 月 5 日解。
    #      这正是"推断放在语料范围、不放在单块"的理由——单看那个文件永远推不出来
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "a.md", "# 记录\n> 7/22/2026 12:54:09\n\n## 晚上\n买了望远镜。\n")
        _mk(td, "b.md", "# 记录\n> 3/5/2026 09:00:00\n\n## 早上\n煮了粥。\n")
        c = load_corpus(td)
        assert c.date_order == "mdy", f"语料里有 7/22 这条决定性证据：{c.date_order}"
        amb = [m for m in c.meta if m["source"] == "b.md"]
        #      **来源也要钉**：只断言最终日期的话，万一那条歧义日期是被别的来源解出来的
        #      （标题行短日期／文件名日期／同窗口号邻层继承），断言照样绿，而
        #      `date_order == "mdy"` 那半个分支还是裸的。
        #      （同一文件里后续块拿的是 file_head——继承的正是首块那个 slash 解，
        #      所以要求"至少有一块走了斜杠路"，不是每块都走。）
        assert any(m["timestamp_source"] == "chunk_head_slash" for m in amb), \
            f"歧义日期不是走斜杠那条路解出来的，这条断言就没在守它：" \
            f"{[m['timestamp_source'] for m in amb]}"
        got = {datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d") for m in amb}
        assert got == {"2026-03-05"}, f"有证据时歧义日期该按月/日序解：{got}"

    #   d) **歧义 + 没有任何证据**：必须退到下一级兜底，**不许默认按美式猜**。
    #      这一格是本单最要紧的一条：用户说了他们语料里没出现过这种，
    #      也就是说"默认美式"那条路一次都没被真实数据验过，不能假装它验过
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "b.md", "# 记录\n> 3/5/2026 09:00:00\n\n## 早上\n煮了粥。\n")
        c = load_corpus(td)
        assert c.date_order is None, "没有决定性证据时不许选边"
        assert all(m["timestamp_source"] == "mtime" for m in c.meta), \
            f"歧义又无依据时该退兜底、把成色标出来，不许猜一个日期：" \
            f"{[m['timestamp_source'] for m in c.meta]}"

    #   e) 欧式 `22/7/2026`：首位 >12 → 日/月序，同样单块就能定
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "eu.md", "# 记录\n> 22/7/2026 08:00:00\n\n## 早上\n出门散步。\n")
        c = load_corpus(td)
        assert c.date_order == "dmy"
        assert datetime.fromtimestamp(c.meta[0]["timestamp"]).strftime("%Y-%m-%d") == "2026-07-22"

    #   e2) **欧式 + 歧义**：`date_order == "dmy"` 那半个分支**唯一的用武之地**。
    #      前面 c) 只钉了美式那半边，而这两半是对称的——审查侧实测：把 dmy 那两行
    #      删掉，全量自检照样全绿，也就是说它一直是裸的。失效方向虽然安全
    #      （退到 mtime、不会判错），但静默，重构时很容易被顺手简化掉。
    #      **两份日期要分在两个文件里**：同文件的话，文件级日期继承那一级会把歧义
    #      那条盖过去（它会直接继承 22/7 的日期），靶子当场作废。
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "a.md", "# 记录\n> 22/7/2026 08:00:00\n\n## 早上\n出门散步。\n")
        _mk(td, "b.md", "# 记录\n> 3/5/2026 09:00:00\n\n## 早上\n煮了粥。\n")
        c = load_corpus(td)
        assert c.date_order == "dmy", \
            f"语料里有 22/7 这条决定性证据，该推断成日/月序：{c.date_order}"
        amb = [m for m in c.meta if m["source"] == "b.md"]
        assert any(m["timestamp_source"] == "chunk_head_slash" for m in amb), \
            f"歧义日期不是走斜杠那条路解出来的，这条断言就没在守它：" \
            f"{[m['timestamp_source'] for m in amb]}"
        got = {datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d") for m in amb}
        assert got == {"2026-05-03"}, \
            f"欧式语料里的 3/5 该解成 5 月 3 日（日/月序），不是 3 月 5 日：{got}"

    #   f) **正文里的 7/11 与 3.5 倍不许被命中**——只扫 head 那条约束的靶子。
    #      要放宽"只在 _head_lines 里查"必须先有断言证明不引入误判，这就是那条断言
    with tempfile.TemporaryDirectory() as td:
        #   正文里放三样东西，各自试一层防线：
        #     `7/11 优惠`  —— 挡它的其实是**正则要求年份**，不是 head 限制；
        #     `3.5 倍`     —— 挡它的是短日期只扫标题行；
        #     `3/14/2026`  —— **这条才是 head 限制的靶子**：完整的斜杠日期，
        #                     正则拦不住它，只有"不扫正文"这条能拦。
        #   第一版这里只有前两样，于是把 head 限制放宽成扫全文时自检照样全绿
        #   （变异④当场绿着过去了）——**防线要各自有各自的靶子**
        #   ⚠ `_head_lines` 取的是 chunk 的**前 3 个非空行**，所以要测"不扫正文"
        #     这条，正文里那个日期必须落在第 4 行之后——写在第 2 行的话它本来就在
        #     head 里，测的就不是这条防线了（第一版就是这么写的，基线当场红）
        _mk(td, "noise.md",
            "# 没有日期的标题\n\n## 聊天\n昨天下午出门。\n路上买了点东西。\n"
            "我们聊到 7/11 优惠，还有 3.5 倍的价格；他生日是 3/14/2026，记一下。\n")
        c = load_corpus(td)
        assert all(m["timestamp_source"] == "mtime" for m in c.meta), \
            f"正文里的日期被当成块时间戳了（7/11 / 3.5 / 3/14/2026）：" \
            f"{[m['timestamp_source'] for m in c.meta]}"
        assert c.date_order is None, \
            "正文里的 3/14/2026 不该被拿去当顺序证据——顺序推断同样只看 head"
        #   同一份正文，日期挪到 head 里就该认出来——证明上面那条不是因为解析器坏了。
        #   对照组的日期**必须是无歧义的**（22 只能是日）：第一版这里写的是
        #   `7/11/2026`，它本身两解都通、被正确拒掉，于是这条对照证明不了任何事
        _mk(td, "noise.md",
            "# 标题\n> 7/22/2026 10:00:00\n\n## 聊天\n我们聊到 7/11 优惠，还有 3.5 倍的价格。\n")
        c2 = load_corpus(td)
        assert c2.meta[0]["timestamp_source"] == "chunk_head_slash", \
            "同样的正文、日期写在 head 就该认出来——不然上面那条只是解析器坏了"

    #   g) 语料自相矛盾（两种证据都有）：不选边，退兜底
    with tempfile.TemporaryDirectory() as td:
        _mk(td, "a.md", "# 记录\n> 7/22/2026 1:00:00\n\n## 晚上\n甲。\n")
        _mk(td, "b.md", "# 记录\n> 22/7/2026 1:00:00\n\n## 晚上\n乙。\n")
        _mk(td, "c.md", "# 记录\n> 3/5/2026 1:00:00\n\n## 晚上\n丙。\n")
        c = load_corpus(td)
        assert c.date_order is None, "两种证据都有＝语料不自洽，猜哪边都可能错半份"
        amb = [m for m, ch in zip(c.meta, c.chunks) if "丙" in ch]
        assert all(m["timestamp_source"] == "mtime" for m in amb), \
            "自相矛盾时那条歧义日期不许被采信"

    # 15c.【变异靶心：词面路的候选门槛】P0"候选闸补词面门槛"。原来的闸是
    #      「BM25 有分 **或** 向量过槛」，而 `bm_scores[i] > 0` 那一路压根没门槛：
    #      真实语料词汇丰富，任意查询都能撞上一堆块的 bigram 重叠。
    #      **判据是"共享的词面里有没有一个是区分性的"，不是"分够不够高"**——
    #      分数阈值挡不住高频功能词堆出来的分，而且又会是一个拍脑袋的常数。
    with tempfile.TemporaryDirectory() as td:
        #   语料构造：每块都有那串功能词（模拟真实语料里人人都有的口语），
        #   而"保险丝"只出现在一块里（模拟真正有区分度的内容词）——**区分性判据是
        #   相对语料自身的 df，所以那个词在 fixture 里必须真的稀有**，
        #   第一版把它写进了每一块，于是它 df 超标、反被判成非区分性（断言当场红）
        for i in range(12):
            body = "我们那次说的那件事后来怎么样了，当时聊到很晚。"
            if i == 0:
                body += "拆开后盖发现保险丝熔断。"
            Path(td, f"w{i:02d}_2026-07-{i + 1:02d}.md").write_text(
                f"# 第{i}窗\n\n## 那天\n{body}\n", encoding="utf-8")
        idx_g = load_corpus(td)
        #   a) 整句都是高频功能词（语料里每块都有）→ **词面路一个都不放行**
        junk = "我们那次说的那件事后来怎么样了"
        assert idx_g.lexical_admit(tokenize(junk)) == set(), \
            "整句都是高频功能词却放行了——这正是编造查询最典型的形态"
        assert idx_g.retrieve(junk) == [], \
            "没有任何区分性词面证据时该明确空手，不该端回一堆沾边的块"
        #   b) 带一个区分性词 → 照常放行（**反向也要守**：把闸拧死成"永远空集"
        #      同样能让上面两条全过，那是把检索废掉，不是修好）
        real = "保险丝"
        admit = idx_g.lexical_admit(tokenize(real))
        assert admit, "带区分性词的查询必须还能进候选，否则等于把检索关了"
        assert idx_g.retrieve(real), "真实查询该照常有结果"
        #   c) 门槛的 df 上限仍是**对语料自身的占比**，不是跟外部标度绑死的绝对分数
        assert idx_g._distinctive_df() == max(3, idx_g._bm25.N // 5), \
            "区分性判据必须跟着语料规模走，不许另造一个跟外部标度绑死的常数"
        #   d) 图谱路在同一条判据**之上**压一个成本上界（2026.08.02 改，原断言钉的是
        #      "两边逐字同一个值"）。改它的理由是那条判据本身没变、变的是图谱肯为
        #      建边枚举多少块对（df² vs 候选闸的 df），完整论证在 _graph_df_cap()。
        #      **这三条要一起断**，少一条就漏掉一种把它改坏的方式：
        assert idx_g._graph_df_cap() == min(idx_g._distinctive_df(), GRAPH_DF_CAP), \
            "图谱上限必须是「同一条判据 ∧ 成本天花板」，不许另起一套判据"
        assert idx_g._distinctive_df() <= max(3, idx_g._bm25.N // 5), \
            "候选闸不许跟着封顶——那是「库里没有就说没有」那道闸，这一轮刻意没动它"
        #      规模一大两边才分岔：N ≤ 320 时 N//5 ≤ 64，天花板压根不生效
        #      语料要**造得让 df 正好落在两条线中间**（64 < df ≤ 80）：N=400 时判据是
        #      80，天花板是 64，所以 5 个主题各 80 块——主题词 df=80，判据放行、
        #      天花板挡掉，两条路线才真的分岔。整份都一样的话每个 bigram df 都是 400、
        #      两边都挡掉，图谱全空，下面那条对比断言会假绿（写这条时踩过）
        big_texts = [f"## 第 {i} 次聊{t}\n关于{t}的事，那天说定了。"
                     for i, t in enumerate(
                         ["咖啡机", "望远镜", "薄荷", "蛋糕", "单簧管"] * 80)]
        big = MemoryIndex()
        for t in big_texts:
            big.add(t)
        big.build()
        assert big._graph_df_cap() == GRAPH_DF_CAP < big._distinctive_df(), \
            "N=400 时天花板必须真的压住图谱（不生效就等于这条修法没接上）"
        #      而且天花板得**真的走进 _build_graph**：上面那条只断方法返回什么，
        #      _build_graph 里自己重算一遍 df 上限的话它照样绿（实测溜过一次）。
        #      拿一个不封顶的同款索引对比，图谱必须不一样
        uncapped = type("_NoCap", (MemoryIndex,),
                        {"_graph_df_cap": lambda self: self._distinctive_df()})()
        for t in big_texts:
            uncapped.add(t)
        uncapped.build()
        assert uncapped._neighbors != big._neighbors, \
            "天花板没走进 _build_graph——它多半在里面自己重算了一遍 df 上限"
        assert idx_g._graph_df_cap() == idx_g._distinctive_df(), \
            "小语料上两边必须仍然逐字相等——天花板只在大语料上才现身"

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


def run(corpus_dir, query, topN=5, embed=False, provider_spec=None):
    provider = resolve_provider(provider_spec) if embed else None
    index = load_corpus(corpus_dir, embed=embed, provider=provider)
    retriever = f"BM25 + {provider.id}" if embed else "BM25 + bigram代理"
    print(f"索引 {len(index.chunks)} 块，检索器={retriever}")
    if embed:
        # 语料去向按档不同，这句话每次都要说出来——它是用户选这条路时的代价
        print(f"提供方：{provider.describe()}")
        if not index.vec_calibrated:
            print("⚠ 这个模型的命中门槛**未标定**：向量路只参与排序、不单独放行候选"
                  "（不照抄别的模型的 0.45，见 embedding_provider）")
    print(f"查询：{query}\n")
    for i, r in enumerate(index.retrieve(query, topN), 1):
        head = r["meta"].get("heading") or r["meta"].get("source", "")
        preview = r["text"].replace("\n", " ")[:60]
        print(f"{i}. [{head}] score={r['score']:.4f} w={r['weight']:.2f}  {preview}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embed", action="store_true", help="用真 embedding")
    ap.add_argument("--embed-provider", dest="embed_provider",
                    help="embedding 提供方：local（默认，需 fastembed）/ local:<模型> / "
                         "cloud（云端，其余走 MEMORY_EMBED_* 环境变量；**语料会发到服务商**）")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--query", help="检索词")
    ap.add_argument("--topN", type=int, default=5)
    args = ap.parse_args()
    if args.selftest:
        _selftest(embed=args.embed)
    elif args.corpus and args.query:
        run(args.corpus, args.query, args.topN, embed=args.embed,
            provider_spec=args.embed_provider)
    else:
        ap.print_help()
