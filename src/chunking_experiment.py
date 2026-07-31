#!/usr/bin/env python3
"""
chunking策略对比实验（设计笔记"待验证的关键问题"第1条）

三种切法（同一检索器下对比，保证公平）：
  heading   按"## "标题自然边界切，超长小节再按空行段落细分
  fixed     固定字数滑窗切（size/overlap可调）
  semantic  相邻句窗相似度突变处断开（默认零依赖用字符bigram余弦做相似度代理；
            加--embed用真embedding，见下）

评估：(query, answer_substring)对，检索top-k个chunk，命中=answer_substring
出现在某个被检回的chunk里。指标：hit@1 / hit@3 / 平均注入字符数（top-3总长，token成本代理）。

--embed：检索器和semantic切法都换成真embedding（bge-small-zh-v1.5，
fastembed跑ONNX，本地CPU，首次运行自动下载约100MB模型）。默认不带
--embed时保持零依赖，selftest不需要装任何东西。

用法：
  python chunking_experiment.py --selftest
  python chunking_experiment.py --corpus <md目录> --queries <queries.json>
  uv run --with fastembed python chunking_experiment.py --embed --corpus ... --queries ...
queries.json格式：[{"query": "...", "answer": "语料里逐字存在的子串"}, ...]

本仓库不含私人语料：真实验证在本地对私人语料目录跑，语料和真实
查询集都不入库，这里只带合成示例。
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")


def sentences(text: str):
    return [s for s in (x.strip() for x in SENT_SPLIT_RE.split(text)) if s]


def bigram_counts(text: str) -> Counter:
    # 只留CJK和字母数字，去掉标点/空白，避免"，的。"这类高频噪声主导相似度
    s = re.sub(r"[^\w一-鿿]", "", text)
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------- 可选真embedding（--embed） ----------

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # bge-zh模型卡要求：query侧加指令，passage侧不加
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding("BAAI/bge-small-zh-v1.5")
    return _EMBEDDER


def embed_texts(texts, is_query=False):
    """返回单位化后的向量矩阵(numpy)。fastembed的query_embed对bge不加中文指令，前缀自己加"""
    import numpy as np
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    vecs = np.array(list(get_embedder().embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------- 三种切法 ----------

def chunk_heading(text: str, max_chars=1500):
    """按"## "标题切；没有标题或小节超长时按空行段落细分到max_chars以内"""
    sections = re.split(r"(?=^## )", text, flags=re.M)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
            continue
        # 超长小节：按段落攒，攒到接近max_chars就出一个chunk
        buf = ""
        for para in re.split(r"\n{2,}", sec):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def chunk_fixed(text: str, size=500, overlap=100):
    chunks = []
    step = size - overlap
    for i in range(0, max(len(text) - overlap, 1), step):
        c = text[i:i + size].strip()
        if c:
            chunks.append(c)
    return chunks


def chunk_semantic(text: str, win=3, min_chunk=150, max_chunk=1500, embed=False):
    """相邻句窗（各win句）余弦，低于(均值-0.5*标准差)处断开。
    相似度后端：默认bigram词面代理；embed=True用真embedding（每句嵌入一次，
    句窗向量取均值池化）。阈值是相对量（均值-0.5*std），换后端自适应，不用手调绝对值"""
    sents = sentences(text)
    if len(sents) <= win * 2:
        return [text.strip()] if text.strip() else []
    sims = []
    if embed:
        vecs = embed_texts(sents)  # 每句一个向量，窗口向量均值池化，比逐窗嵌入省win倍调用
        for i in range(win, len(sents) - win + 1):
            left = vecs[i - win:i].mean(axis=0)
            right = vecs[i:i + win].mean(axis=0)
            denom = (left @ left) ** 0.5 * (right @ right) ** 0.5
            sims.append(float(left @ right / denom) if denom else 0.0)
    else:
        for i in range(win, len(sents) - win + 1):
            left = bigram_counts("".join(sents[i - win:i]))
            right = bigram_counts("".join(sents[i:i + win]))
            sims.append(cosine(left, right))
    mean = sum(sims) / len(sims)
    std = math.sqrt(sum((s - mean) ** 2 for s in sims) / len(sims))
    threshold = mean - 0.5 * std

    chunks, buf = [], ""
    for idx, sent in enumerate(sents):
        buf += sent
        sim_idx = idx + 1 - win  # buf收完sents[idx]后，边界idx+1对应sims[idx+1-win]
        at_break = 0 <= sim_idx < len(sims) and sims[sim_idx] < threshold
        if (at_break and len(buf) >= min_chunk) or len(buf) >= max_chunk:
            chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


STRATEGIES = {
    "heading": chunk_heading,
    "fixed": chunk_fixed,
    "semantic": chunk_semantic,
}


# ---------- 检索与评估 ----------

def retrieve(query: str, chunk_vecs, k=3):
    """chunk_vecs: [(chunk文本, bigram Counter), ...]；返回top-k chunk文本"""
    q = bigram_counts(query)
    scored = sorted(chunk_vecs, key=lambda cv: cosine(q, cv[1]), reverse=True)
    return [c for c, _ in scored[:k]]


def retrieve_embed(queries, chunks, k=3):
    """真embedding检索：返回每条query的top-k chunk文本列表"""
    cvecs = embed_texts(chunks)
    qvecs = embed_texts([q["query"] for q in queries], is_query=True)
    scores = qvecs @ cvecs.T  # 已单位化，点积即余弦
    return [[chunks[j] for j in row.argsort()[::-1][:k]] for row in scores]


def evaluate(queries, chunks, k=3, embed=False):
    if embed:
        tops = retrieve_embed(queries, chunks, k)
    else:
        chunk_vecs = [(c, bigram_counts(c)) for c in chunks]
        tops = [retrieve(item["query"], chunk_vecs, k) for item in queries]
    hit1 = hit3 = 0
    injected = []
    for item, top in zip(queries, tops):
        if item["answer"] in top[0]:
            hit1 += 1
        if any(item["answer"] in c for c in top):
            hit3 += 1
        injected.append(sum(len(c) for c in top))
    n = len(queries)
    sizes = sorted(len(c) for c in chunks)
    return {
        "chunks": len(chunks),
        "median_size": sizes[len(sizes) // 2],
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "avg_injected_chars": sum(injected) // n,
    }


def run(corpus_dir: str, queries_path: str, k=3, embed=False):
    text = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(Path(corpus_dir).glob("*.md"))
    )
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    # 先验证每个answer子串真的在语料里，防止ground truth本身是错的
    missing = [q["answer"] for q in queries if q["answer"] not in text]
    if missing:
        sys.exit(f"这些answer子串不在语料里，查询集有错：{missing}")
    retriever = "bge-small-zh-v1.5" if embed else "bigram词面"
    print(f"语料 {len(text)} 字符，{len(queries)} 条查询，k={k}，检索器={retriever}\n")
    header = f"{'策略':<10}{'chunk数':>8}{'中位大小':>10}{'hit@1':>8}{'hit@3':>8}{'注入字符':>10}"
    print(header)
    for name, fn in STRATEGIES.items():
        chunks = fn(text, embed=embed) if name == "semantic" else fn(text)
        r = evaluate(queries, chunks, k, embed=embed)
        print(f"{name:<10}{r['chunks']:>8}{r['median_size']:>10}"
              f"{r['hit@1']:>8.0%}{r['hit@3']:>8.0%}{r['avg_injected_chars']:>10}")


# ---------- 合成语料selftest（全部虚构内容，不含任何真实语料） ----------

SYNTH = """## 一次远足的记录
上周六去了城郊的青云山。山路一共十二公里，走了五个小时。半山腰的凉亭里遇到一位卖桂花糕的老人家，他说这里的桂花糕用的是山顶的野桂花。买了两块，甜而不腻。
下山的时候下了小雨，石阶很滑。同行的阿岚差点摔倒，被树枝挂住了背包才稳住。回程的大巴上大家都睡着了。

## 修理咖啡机
家里那台老咖啡机又坏了，这次是加热管不工作。拆开后盖发现保险丝熔断，型号是10A250V。去电子市场花两块钱买了一根换上，通电测试正常。
顺手把水垢也清理了，用的柠檬酸溶液，泡了半小时冲干净。出水速度明显变快，咖啡味道也正了。

## 读《城市与狗》
这个月读完了略萨的《城市与狗》。军校的封闭环境把每个少年都磨成了另一副样子，"美洲豹"的强悍是装出来的壳。
最震撼的是结构：四条叙事线交错，直到最后才拼出完整真相。这种写法要求读者主动参与拼图。

## 种薄荷的失败经验
四月在阳台种的薄荷死了。原因复盘：花盆太小、浇水太勤、正午暴晒。薄荷喜湿但怕涝，盆底积水烂了根。
下次改用带排水孔的深盆，放东侧只晒上午的位置。剪下来的枝条水培了一周已经生根，算是抢救回来一支。
"""

SYNTH_QUERIES = [
    {"query": "咖啡机坏了是什么原因", "answer": "保险丝熔断"},
    {"query": "爬山遇到卖什么的老人", "answer": "桂花糕"},
    {"query": "薄荷为什么死了", "answer": "盆底积水烂了根"},
    {"query": "略萨那本小说的结构有什么特点", "answer": "四条叙事线交错"},
]


def _selftest(embed=False):
    # 三种切法都要能切出chunk，且heading切法边界落在标题处
    h = chunk_heading(SYNTH)
    assert len(h) == 4 and all(c.startswith("## ") for c in h), h
    f = chunk_fixed(SYNTH, size=300, overlap=60)
    assert len(f) > 1 and all(len(c) <= 300 for c in f)
    s = chunk_semantic(SYNTH, min_chunk=100)
    assert len(s) > 1, "semantic切法至少要切出两块"

    # 检索评估端到端：heading切法在合成语料上应全命中hit@3
    r = evaluate(SYNTH_QUERIES, h)
    assert r["hit@3"] == 1.0, r
    # 超长小节细分路径
    long_text = "## 超长节\n" + ("字" * 400 + "\n\n") * 6
    assert len(chunk_heading(long_text, max_chars=1000)) > 1
    # answer子串校验路径（evaluate之前由run做，这里直接验证判断逻辑）
    assert "保险丝熔断" in SYNTH

    if embed:  # 真embedding路径（需fastembed，仅--selftest --embed时跑）
        se = chunk_semantic(SYNTH, min_chunk=100, embed=True)
        assert len(se) > 1, "embed semantic切法至少要切出两块"
        re_ = evaluate(SYNTH_QUERIES, h, embed=True)
        assert re_["hit@3"] == 1.0, re_
    print("selftest ok" + ("（含embed路径）" if embed else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--embed", action="store_true", help="用真embedding（需fastembed）")
    ap.add_argument("--corpus", help="md语料目录")
    ap.add_argument("--queries", help="queries.json路径")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    if args.selftest:
        _selftest(embed=args.embed)
    elif args.corpus and args.queries:
        run(args.corpus, args.queries, args.k, embed=args.embed)
    else:
        ap.print_help()
