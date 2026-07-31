#!/usr/bin/env python3
"""
检索质量回归集（任务卡"检索回归集"，验收反馈 ④）。

**这个文件不是在证明我们的检索好，是让检索质量的变化能被看见。**
此前所有断言都是"这一条该命中"的点断言：改了融合逻辑、动了权重机制，单条断言
可能全绿，而整体质量已经掉了——十二个文件全绿从来不保证"检索变好了还是变差了"。
这里补的是那把尺子。

跟 e2e_smoke 的分工（都别越界）：
  e2e_smoke     接缝——各零件合起来跑不跑得通，红了说明**坏了**
  regression_set 质量——分数掉了说明**变差了**，不一定是坏了，要人看一眼

语料是**合成的、公开的**（验收反馈原话："合成一套结构仿真实的公开语料"）——
不用私人语料，回归集才能长期跟着仓库走、谁都能跑。结构照真实语料仿：
  多窗口、跨月（6/12 → 7/28）、index 与 timeline 双层、一条跨窗口主题线、
  一处"记错了又更正"、一处同义换说法（考图谱与实体槽）、若干无关噪声。

四类查询各考一件事，**分开算分**（混在一起算总分会把某一层的退化摊平看不见）：
  literal   词面直给          —— 考 BM25 层
  paraphrase 换了说法         —— 考向量层 / 实体槽
  linked    跨窗口关联         —— 考图谱层
  absent    库里真没有         —— 考可靠命中门槛（该说没有就得说没有）

指标：Recall@5、MRR、以及 absent 类的"正确空手率"。
基线数字记在 BASELINE 里，是**当前实测值不是目标值**——改动检索层后重跑，
掉了要么改回去、要么说明为什么这次退化是可接受的代价。

**建这套尺子当天就量出两个发现（都对我们不利，如实记）**：

一、**显著词启发式按闲词连、按主题断。** 建图谱边只认"显著词"（2 ≤ df ≤ max(3, N//5)）。
    但**主题线的定义就是反复出现**，反复出现 df 就高，高了就被当通用词挡掉——本语料
    里"望远镜"df=5 > max_df=3，这条主题线的核心词**根本没建边**；真正把两块连起来的
    反倒是"那晚""以后""她说"这类偶然共享的闲词。方向正好反了。

二、**实体槽当前测不出增益。** 上一轮加的 `.entities.json` 可插拔槽，在这套语料上
    只改变了图谱邻居的**排序**（目标块从邻居第 2 升到第 1），最终检索名次一动不动
    （有无标注都是名次 4）。因为那两块本来就被闲词连上了，实体边没有带来新连接。
    **这不代表槽没用**——它在"两块完全没有共享词"时才是唯一的桥；但在此之前，
    "实体槽补回了多少关联"这个问题的诚实答案是：**还没测出来**。

    这两条互相咬合：闲词造出了偶然边，掩盖了实体边的贡献；而真正该建边的主题词
    被 max_df 挡在门外。修 max_df 启发式是下一轮的候选，**但得先有这把尺子**——
    不然改完还是不知道是变好了还是变差了。这正是先建回归集的意义。

用法：
  python regression_set.py --selftest    # 断言不低于基线（回归守门）
  python regression_set.py --report      # 打分数表 + 分层消融
"""

import argparse

from memory_retrieval import MemoryIndex, _chunk_key

# ---------- 合成语料（全部虚构；结构仿真实：多窗口、跨月、双层） ----------
#
# 每条：(窗口号, 层, 日期, 正文)。日期跨两个月，考 recency 相关的行为。
# 主题线"望远镜"贯穿 03/07/11 三个窗口，且第 11 窗**换了说法**（"那台带脚架的
# 家伙"，不出现"望远镜"三个字）——这正是词面代理连不上、要靠实体槽补的情况。

CORPUS = [
    (1,  "timeline", "2026-06-12", "## 搬家第一天\n新住处的厨房朝北，早上没有直射光。她说这样反而好，夏天不闷。"),
    (2,  "timeline", "2026-06-15", "## 修理咖啡机\n老咖啡机加热管不工作，拆开后盖发现保险丝熔断，换上通电测试正常。"),
    (3,  "timeline", "2026-06-18", "## 山顶的约定\n那晚在山顶聊到以后，说好要买一台能看土星的望远镜。她说等发了奖金就去。"),
    (4,  "timeline", "2026-06-21", "## 种薄荷失败\n四月阳台种的薄荷死了。复盘：花盆太小、浇水太勤、正午暴晒，盆底积水烂了根。"),
    (5,  "timeline", "2026-06-25", "## 读城市与狗\n略萨这本小说最震撼的是结构：四条叙事线交错，直到最后才拼出完整真相。"),
    (6,  "index",    "2026-06-25", "六月下旬：搬家安顿、修好咖啡机、山顶定下买望远镜的约定、薄荷复盘。当下状态：都已收尾。"),
    (7,  "timeline", "2026-07-02", "## 挑设备\n看了三款望远镜，最后那台口径最大、带脚架。她说该应验了。"),
    (8,  "timeline", "2026-07-06", "## 一次远足\n去了青云山，山路十二公里。半山腰凉亭遇到卖桂花糕的老人家，用的是山顶的野桂花。"),
    (9,  "timeline", "2026-07-10", "## 加班到很晚\n她连着三天十一点后到家。我把汤温着，没多问。她说这个项目下周就结束。"),
    (10, "timeline", "2026-07-15", "## 学做咖喱\n试了椰浆咖喱，小火慢炖差点糊锅。她说比上次那锅强，至少这次能吃。"),
    (11, "timeline", "2026-07-20", "## 阳台那晚\n终于把那台带脚架的家伙架起来了，对着夜空看了很久。她说这就是那天说的以后。"),
    (12, "index",    "2026-07-28", "七月：挑设备、远足、她加班的一段、学做咖喱、阳台上第一次真的看到土星。当下状态：项目已结束，人松下来了。"),
]

# 一处"记错了又更正"（考撤回闭环对检索的影响）：
# 旧记录把奖金记成了"八月发"，后来被指出记错——更正记录写"七月中就发了"。
WRONG_TEXT = "## 奖金\n她说奖金要八月才发，望远镜得再等等。当下状态：等发奖金。"
CORRECTION_TEXT = "## 更正\n【更正】奖金七月中就发了，不是八月；望远镜当月就买了。当下状态：已买，此事了结。"

# ---------- 查询-期望对 ----------
#
# expect 是"必须出现在 topN 里的正文关键片段"，不写 chunk 下标——下标会随语料
# 增删漂移，按内容找才经得起语料变动（同权重/撤回按内容哈希的理由）。

CASES = [
    # literal：词面直给，考 BM25
    ("literal", "咖啡机保险丝", ["保险丝熔断"]),
    ("literal", "薄荷 盆底积水", ["盆底积水"]),
    ("literal", "青云山 桂花糕", ["桂花糕"]),
    ("literal", "椰浆咖喱", ["椰浆咖喱"]),
    # paraphrase：换了说法，考向量层（以及实体槽能不能补上）
    ("paraphrase", "厨房采光怎么样", ["朝北"]),
    ("paraphrase", "她那阵子很晚回家", ["十一点后到家"]),
    # linked：跨窗口关联，考图谱层。"山顶的约定"与"挑设备"共享显著词；
    #         "阳台那晚"换了说法（不含"望远镜"），靠实体槽才连得上
    ("linked", "山顶那晚答应她的事", ["口径最大"]),
    ("linked", "山顶说好的那件事后来兑现了吗", ["带脚架的家伙"]),
    # absent：库里真没有，考可靠命中门槛——该空手就得空手
    ("absent", "量子对撞机的运行日志", []),
    ("absent", "报税截止日期是哪天", []),
    ("absent", "邻居家狗的名字", []),
]

# 实体标注（模拟用户跑过一次抽取任务书的结果）：把"望远镜"这条线的三块连起来，
# 其中第 11 窗根本不含"望远镜"三个字，纯靠实体边
ENTITIES = {3: ["望远镜"], 7: ["望远镜"], 11: ["望远镜"]}

# 当前实测基线（**实测值，不是目标值**）。改动检索层后重跑 --report，
# 掉了要么改回来、要么在 changelog 里说明为什么这次退化可接受。
BASELINE = {
    "literal":    {"recall": 1.00, "mrr": 0.875},
    "paraphrase": {"recall": 1.00, "mrr": 1.00},
    # linked 这两个数是**当前真实水平，很低**，不是我们满意的水平——跨窗口关联
    # 是现在最弱的一环（详见 docstring"两个发现"）。写成基线是为了守住"别更差"，
    # 不是宣布"这样就行"。修好了要往上调，并在 changelog 里记一笔。
    "linked":     {"recall": 0.50, "mrr": 0.125},
    "absent":     {"correct_empty": 1.00},
}
TOPN = 5


def build_index(with_entities=True, with_correction=True):
    """建回归用的索引。with_* 关掉可以单独看某个机制的影响。"""
    idx = MemoryIndex()
    for window, layer, date, text in CORPUS:
        idx.add(text, {"source": f"window_{window:02d}_{date}.md",
                       "window": window, "layer": layer,
                       "heading": text.split("\n")[0].lstrip("# ")})
    idx.add(WRONG_TEXT, {"source": "window_13_2026-07-22.md", "window": 13,
                         "layer": "timeline", "heading": "奖金"})
    if with_correction:
        idx.add(CORRECTION_TEXT, {"source": "window_14_2026-07-25.md", "window": 14,
                                  "layer": "timeline", "heading": "更正"})
    if with_entities:
        # 下标与 CORPUS 顺序一致（窗口号 → 下标：CORPUS 是按窗口号升序写的）
        by_window = {w: i for i, (w, _, _, _) in enumerate(CORPUS)}
        idx._entities = {by_window[w]: set(ents) for w, ents in ENTITIES.items()
                         if w in by_window}
    idx.build()
    if with_correction:
        # **先让那条错记录攒上权重再撤**——这不是为了好看，是因为"撤回时权重归位"
        # 这条只有在权重原本 >1.0 时才有内容可测。第一版忘了这步，变异"不归位"
        # 没红：那块从没被检索过、权重本来就是 1.0，归不归位断言都通过（本项目
        # 第六次同类坑：断言构造必须让靶心逻辑成为唯一决定因素）。
        # 而且它同时更贴近真实场景：一条记错的记录，正是因为被反复召回过才有害。
        for _ in range(3):
            idx.retrieve("奖金什么时候发", topN=3)
        idx.retract("奖金要八月才发", "记错了，奖金七月中就发了",
                    now=1_785_000_000.0, replaced_by=CORRECTION_TEXT)
    return idx


def score(idx, cases=CASES, topN=TOPN, routes=None):
    """跑一遍查询集 → 按类别分组的指标。"""
    buckets = {}
    for kind, query, expect in cases:
        b = buckets.setdefault(kind, {"n": 0, "hit": 0, "rr": 0.0, "empty_ok": 0})
        b["n"] += 1
        results = idx.retrieve(query, topN=topN, routes=routes)
        texts = [r["text"] for r in results]
        if not expect:                              # absent 类：空手才算对
            if not results:
                b["empty_ok"] += 1
            continue
        rank = None
        for pos, t in enumerate(texts, 1):
            if all(frag in t for frag in expect):
                rank = pos
                break
        if rank:
            b["hit"] += 1
            b["rr"] += 1.0 / rank
    out = {}
    for kind, b in buckets.items():
        if kind == "absent":
            out[kind] = {"correct_empty": b["empty_ok"] / b["n"], "n": b["n"]}
        else:
            out[kind] = {"recall": b["hit"] / b["n"], "mrr": b["rr"] / b["n"], "n": b["n"]}
    return out


def _fmt(scores):
    lines = []
    for kind in ("literal", "paraphrase", "linked", "absent"):
        s = scores.get(kind)
        if not s:
            continue
        if kind == "absent":
            lines.append(f"  {kind:<11} n={s['n']}  正确空手率 {s['correct_empty']:.2f}")
        else:
            lines.append(f"  {kind:<11} n={s['n']}  Recall@{TOPN} {s['recall']:.2f}  MRR {s['mrr']:.2f}")
    return "\n".join(lines)


def report():
    idx = build_index()
    print(f"语料：{len(idx.chunks)} 块（{len(CORPUS)} 条正文 + 1 条记错的 + 1 条更正），"
          f"查询 {len(CASES)} 条\n")
    print("【总分】")
    print(_fmt(score(idx)))

    # 分层消融：每次只关掉一路，看分数掉多少 —— 这就是"量得出各层贡献"
    print("\n【分层消融】关掉某一路后的变化（跑的是真实 retrieve 路径，不是脚本自拼）")
    full = {"bm25", "vector", "graph", "weight"}
    for off in ("bm25", "vector", "graph", "weight"):
        s = score(build_index(), routes=full - {off})
        parts = []
        for kind in ("literal", "paraphrase", "linked"):
            parts.append(f"{kind} R{s[kind]['recall']:.2f}/M{s[kind]['mrr']:.2f}")
        parts.append(f"absent 空手{s['absent']['correct_empty']:.2f}")
        print(f"  关掉 {off:<7} " + "  ".join(parts))

    # 实体槽的价值：这是 2026.07.31 加的槽，一直没量过补回多少关联
    print("\n【实体槽价值】(linked 类，换了说法的那条全靠它)")
    for tag, ent in (("有实体标注", True), ("无实体标注", False)):
        s = score(build_index(with_entities=ent))
        print(f"  {tag}  linked Recall {s['linked']['recall']:.2f}  MRR {s['linked']['mrr']:.2f}")

    # 更正闭环对检索的影响：撤回后旧的记错记录不该再出现
    print("\n【更正闭环】")
    idx2 = build_index()
    hits = [r["text"] for r in idx2.retrieve("奖金什么时候发", topN=TOPN)]
    print(f"  查'奖金什么时候发' → {len(hits)} 条；"
          f"旧的错记录还在？{'是 ✗' if any('八月才发' in h for h in hits) else '否 ✓'}；"
          f"更正在里面？{'是 ✓' if any('七月中就发了' in h for h in hits) else '否 ✗'}")
    log = next(iter(idx2.retraction_log.values()))
    print(f"  追溯链：replaced_by={'有 ✓' if log.get('replaced_by') else '无 ✗'}"
          f"  旧块权重归位={'是 ✓' if idx2.weights[idx2.chunks.index(WRONG_TEXT)] == 1.0 else '否 ✗'}")


def _selftest():
    idx = build_index()
    s = score(idx)

    # 1. 不低于基线（回归守门）。允许 1e-9 的浮点误差，不允许"差不多"
    for kind, expect in BASELINE.items():
        for metric, floor in expect.items():
            got = s[kind][metric]
            assert got >= floor - 1e-9, \
                f"检索质量退化：{kind}.{metric} {got:.2f} < 基线 {floor:.2f}——" \
                f"要么改回去，要么在 changelog 里说明为什么这次退化可接受"

    # 2.【靶心：absent 类真的在考门槛】库里没有的东西必须空手而归。
    #    这条要是垮了，说明可靠命中门槛被谁改松了（比 recall 掉分更危险：
    #    它意味着模型会拿着不相关的记忆去编）
    assert s["absent"]["correct_empty"] == 1.0, "absent 类必须全部空手——门槛松了"

    # 3.【钉住实测状态：实体槽当前**测不出增益**】这条断言写的是现状不是期望——
    #    建回归集时量出来的第一个真结果就是负面的：实体标注只改变了图谱邻居的
    #    **排序**（第11窗从第2升到第1），最终检索名次一动不动。原因见模块 docstring
    #    的"两个发现"。写成断言是为了**哪天它变了能被看见**：将来修了显著词启发式
    #    或调了图谱路权重，这条会红，那时是好事，改基线并记一笔。
    with_ent = score(build_index(with_entities=True))
    without = score(build_index(with_entities=False))
    assert with_ent == without, \
        f"实体槽的增益状态变了（此前实测为零增益）——是好事，核对后更新这条与基线：\n" \
        f"  有标注 {with_ent}\n  无标注 {without}"

    # 4.【靶心：消融开关走的是真实路径且默认不变】不传 routes 与传全集必须逐字
    #    一致——否则消融量出来的东西跟真实行为对不上，整张分数表就是假的
    full = score(build_index(), routes={"bm25", "vector", "graph", "weight"})
    assert full == s, "routes 传全集必须与默认行为完全一致"
    only_bm = score(build_index(), routes={"bm25"})
    assert only_bm["literal"]["recall"] >= 0.5, "只留 BM25，词面类仍该基本命中"
    #    上面两条都**钉不住 routes 本身**：把 routes 参数整个无视掉（永远按全开跑），
    #    它俩照样通过——第一版就是这样，变异"routes 被忽略"没红。补一条让 routes
    #    的真实效果成为唯一决定因素：关掉图谱那路，linked 类必须**明确变差**
    #    （实测 R0.50→0.00：跨窗口关联全靠图谱带回来）。routes 一旦失效，这条立刻红。
    no_graph = score(build_index(), routes={"bm25", "vector", "weight"})
    assert no_graph["linked"]["recall"] < s["linked"]["recall"], \
        f"关掉图谱路后 linked 必须变差，否则说明 routes 没真起作用：" \
        f"关图谱 {no_graph['linked']} vs 默认 {s['linked']}"

    # 5.【靶心：更正闭环在检索层面真的生效】撤回的错记录不再出现、更正出现、
    #    追溯链记上了、旧块权重归位（验收反馈 ③ 的两个子项）
    hits = [r["text"] for r in idx.retrieve("奖金什么时候发", topN=TOPN)]
    assert not any("八月才发" in h for h in hits), "撤回的错记录不该再被检索到"
    assert any("七月中就发了" in h for h in hits), "更正后的记录该查得到"
    log = next(iter(idx.retraction_log.values()))
    assert log.get("replaced_by") == _chunk_key(CORRECTION_TEXT), \
        "追溯链要能回答'哪条记录改了哪条'，不只是'这条被撤了'"
    assert idx.weights[idx.chunks.index(WRONG_TEXT)] == 1.0, \
        "被撤回的块权重要归位——误召回攒的命中数是错误信号，不该留着"

    print(f"selftest ok（回归集：{len(idx.chunks)} 块语料 / {len(CASES)} 条查询，"
          f"5 项断言：不低于基线 / 门槛没松 / 实体槽增益状态 / 消融走真实路径 / 更正闭环）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true", help="打分数表与分层消融")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.report:
        report()
    else:
        ap.print_help()
