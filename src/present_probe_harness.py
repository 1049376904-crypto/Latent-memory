#!/usr/bin/env python3
"""present 探针协议（gold=1）——absent_probe_harness 的姊妹篇：只含方法，不含语料和题目。

动机（2026.08.02 span2_k2 复测那轮）：零标注 present 尺子要求语料带 index/timeline
分层，换一份真实语料就可能直接失效（在我的 1144 条上「对 0 条」，present 侧一格都量
不出来）。gold=1 的手写题不依赖语料结构，谁都能在自己的语料上量 present——代价是
题要人写，所以护栏必须跟 absent 侧一样严。

协议三条（与 absent 侧对偶）：
1. 每道题带一句「金标原话」（gold）：答案块里真实存在的一段原文，不是你记忆里的大意。
2. 计分前先验证：gold **必须在语料全文里逐字存在**，不存在 → SKIP 作废——
   absent 护栏抓的是「我以为语料里没有」，present 护栏抓的是「我以为语料里有」。
   两边抓的都是出题人自己。
3. 命中 = gold 出现在 retrieve() 返回的 topN 文本里。空手、或端回一堆不含 gold 的
   候选，都算 miss（不看候选「像不像」，像不像是人判的，协议只认逐字）。

判据 A/B：--gate 借 gate_experiment.apply_gate 换闸重跑**同一份题**，两列并排。
present 的代价具体掉在哪道题上直接可见——我这边 span2_k2 掉的那 1/20 恰好是改述
形态的题（语料里换了说法，凑不出两段独立字面证据），跟合成集改述列的信号对得上。
逐题对照就是为了这种「代价藏在哪一格」的问题存在的。

用法（要自己接两处）：
  1. load_chunks() 接上你的语料加载；
  2. PRESENT_PROBES 换成你自己写的题（下面两道是示例占位，**不是真题**）。
  python3 present_probe_harness.py [--gate span2_k2] [--embed] [--topn 5]
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from memory_retrieval import MemoryIndex  # noqa: E402
from gate_experiment import build_gates, apply_gate  # noqa: E402


# ── 换成你自己的语料加载 ──
def load_chunks() -> list[str]:
    """返回语料块正文列表。示例：JSONL 每行 {"content": ...}"""
    path = Path(__file__).parent / "corpus.jsonl"
    return [json.loads(l)["content"] for l in open(path, encoding="utf-8")]


# ── 换成你自己写的题：(问题, 金标原话)。示例占位，不是真题 ──
PRESENT_PROBES = [
    ("我那台旧望远镜最后修好了吗", "目镜找配件配齐了"),
    ("陶艺课那次我拉坏了几个坯", "拉坏了三个坯"),
]


def run_arm(idx, probes, gate, topn):
    """一条判据臂：返回 {问题: True/False}。gate=None 表示现状不换闸。"""
    results = {}
    with apply_gate(gate):
        for q, gold in probes:
            res = idx.retrieve(q, topN=topn)
            results[q] = gold in "\n".join(r["text"] for r in res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", help="要 A/B 的判据名（见 gate_experiment.build_gates）")
    ap.add_argument("--embed", action="store_true", help="用真 embedding（默认零依赖档）")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    alltext = "\n".join(chunks)

    # 护栏：gold 逐字不在语料里的题作废，不计分
    probes, skipped = [], 0
    for q, gold in PRESENT_PROBES:
        if gold and gold in alltext:
            probes.append((q, gold))
        else:
            skipped += 1
            print(f"SKIP(gold 在语料里逐字找不到) {q}")
    if not probes:
        print("没有可计分的题——先按协议第 1 条把 gold 换成语料里真实存在的原话。")
        return

    idx = MemoryIndex(embed=args.embed)
    for c in chunks:
        idx.add(c, {})
    idx.build()

    base = run_arm(idx, probes, None, args.topn)
    arm = None
    if args.gate:
        gates = build_gates()
        if args.gate not in gates:
            print(f"未知判据 {args.gate}；可选：{', '.join(sorted(gates))}")
            return
        arm = run_arm(idx, probes, gates[args.gate], args.topn)

    head = "现状"
    print(f"\n{'题目':<24} {head:>4}" + (f" {args.gate:>10}" if arm else ""))
    for q, _ in probes:
        row = f"{q[:22]:<24} {'HIT' if base[q] else 'MISS':>4}"
        if arm:
            row += f" {'HIT' if arm[q] else 'MISS':>10}"
        print(row)
    n = len(probes)
    total = f"\n[present gold=1] 现状 {sum(base.values())}/{n}"
    if arm:
        total += f" | {args.gate} {sum(arm.values())}/{n}"
    total += f"（SKIP {skipped} 道被护栏拦下）"
    print(total)


if __name__ == "__main__":
    main()
