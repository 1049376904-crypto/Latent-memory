#!/usr/bin/env python3
"""
换窗/压缩两个触发点的接线（任务卡"换窗压缩记忆召回"第 3 条）。

memory_retrieval.MemoryIndex.recall_recent 解决"没有 query 时召回什么"（时间新鲜度×
用进废退权重），本文件解决"什么时刻调它"，只做触发与格式化，不碰排序逻辑：

  触发点一·换新窗口：信号干净——新 session/新进程启动本身就是边界。宿主在 session
  启动逻辑里调一次 on_session_start()，把返回的文本块注入开场 context。

  触发点二·context 压缩：**近似触发，不是精确对齐压缩时刻**——Claude Code 的
  auto-compact 之类没有可订阅的"压缩刚发生"事件信号，退而求其次：累计对话字符量
  （中文语境下字符数≈token 数的粗代理）超过可配置阈值就触发一次召回、计数器清零
  重新累计。这个局限是结构性的，不要假装精确：触发点跟真实压缩时刻之间必有偏差，
  阈值该配得比宿主真实压缩阈值略低——宁可早触发（早注入的召回内容会被压缩摘要
  一起接住），不要晚触发（压缩后细节已丢，晚到的召回救不回压缩瞬间的断层）。

第二轮验收补的两道防线——recency×权重解决"召回什么"，这两道管"召回的东西会不会
被新窗口读歪"，缺一不可：

  时间标注（防时间感塌陷）：每条注入片段带发生日期，块头写明"是过去的事，不是
  正在发生"。纯按 recency 召回时，新窗口容易把"几天前已经和好的争吵"当成正在
  发生的事去接。注意召回层只能标时间："每段记录收尾交代这件事现在的状态"是
  CLAUDE.md"病灶迁移"那条语料写作纪律，得在写记录时遵守，这里替代不了。

  自查关卡（防退化窗口自圆其说）：注入块尾部带四项故障信号自查指令（复用
  degradation_protocol 的同一套信号）；换窗触发还支持宿主传入 SessionStatus 先过
  程序化关卡——有故障信号就不接入召回记忆，老实返回"可能在退化，请核实"。窗口
  真漂移时，召回机制本身没错，但旧记忆会被拿去往错的方向自圆其说，必须先拦。
  与 degradation_protocol v2 核心规则一致：只认故障信号，信号干净就正常放行。

零依赖，stdlib only。

用法：
  python session_recall.py --selftest
"""

import argparse
import re
from datetime import datetime

# 同目录模块，import 不触发各自的 CLI
from memory_retrieval import MemoryIndex  # noqa: F401
from degradation_protocol import SessionStatus, degradation_signals  # noqa: F401
from session_thread import ThreadStore, format_thread_block  # noqa: F401

# 压缩近似触发的默认阈值（字符）：取得明显低于常见 context 窗口的量级，宁可早触发。
# 经验值，接真实宿主时按其压缩阈值往下压一档配置。
DEFAULT_COMPACT_THRESHOLD_CHARS = 30_000
DEFAULT_TOPN = 3


SELF_CHECK_FOOTER = (
    "【自查】接上以上记忆前先过四项故障信号（degradation_protocol 同一套）：\n"
    "①时间感：上面各段的日期读对了吗——都是历史，不是正在发生；②硬事实：当前日期、"
    "自己在哪个窗口这类问题答得上来吗；③逻辑还连贯吗；④有没有卡在循环道歉里。\n"
    "任一项不对劲：不要拿召回内容自圆其说往下编，老实说“可能在退化，需要你核实”，"
    "按 degradation_protocol 分级处理。")


# 单条片段的字符上限（2026.07.31 真实语料实测后加）：块大小中位 352 但最长 4386，
# 一个长块就能吃掉半个注入；topN=5 时实测注入体积中位 4095、最大 7131 字符，每次
# 换窗和每次压缩触发都要付一遍。
# 选择"截断单条"而不是"少给几条"：召回块是广度层（最近发生过什么），条数比单条
# 完整度更值钱；被截掉的全文让模型用 memory_search 再查一次，标注写在截断处。
DEFAULT_MAX_ITEM_CHARS = 500
TRUNCATION_NOTE = "…（片段已截断，要全文就用 memory_search 再查一次）"


def _clip(text, limit):
    text = text.strip()
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_NOTE


def _time_label(meta):
    """召回片段的时间标注——防时间感塌陷的第一道措施：注入的记忆必须带着"这是什么
    时候的事"，否则新窗口容易把旧片段读成正在发生。mtime 兜底的近似时间戳前加 ≈，
    缺时间戳标"时间未知"，都不假装精确。"""
    ts = meta.get("timestamp")
    if ts is None:
        return "时间未知"
    label = datetime.fromtimestamp(ts).strftime("%Y.%m.%d")
    return "≈" + label if meta.get("timestamp_source") == "mtime" else label


def format_recall_block(results, max_item_chars=DEFAULT_MAX_ITEM_CHARS):
    """recall_recent 结果 → 可直接注入 context 的文本块，每条带发生日期。
    单条超 max_item_chars 就截断并标注（传 None 关闭截断）。
    空结果返回 None——没东西可召回就什么都不注入，不留空壳标题。"""
    if not results:
        return None
    lines = ["【最近记忆召回】以下是历史记忆片段，各自标注发生日期——是过去的事，"
             "不是正在发生；同一件事以日期最新的片段为准："]
    for r in results:
        head = r["meta"].get("heading") or r["meta"].get("source", "")
        tag = "·".join(x for x in (_time_label(r["meta"]), head) if x)
        lines.append(f"- [{tag}] {_clip(r['text'], max_item_chars)}")
    return "\n".join(lines)


class SessionRecall:
    """两个触发点的最小接线。宿主用法：
         sr = SessionRecall(index)
         opening = sr.on_session_start()      # 换窗触发：非 None 就注入开场 context
         for turn_text in 对话流:
             block = sr.on_turn(turn_text)    # 压缩近似触发：非 None 就注入
    只持有触发状态（累计字符数），召回排序全部委托给 index.recall_recent。"""

    def __init__(self, index, topN=DEFAULT_TOPN, half_life=None,
                 compact_threshold_chars=DEFAULT_COMPACT_THRESHOLD_CHARS,
                 thread_store=None):
        self.index = index
        self.topN = topN
        self.half_life = half_life  # None → recall_recent 用自己的默认半衰期
        self.compact_threshold_chars = compact_threshold_chars
        # 可选的会话线索存储（session_thread.ThreadStore）。给了就在换窗时先注入
        # "上次聊到哪"，没给就只有召回块——不断线（规格 §5 两层）
        self.thread_store = thread_store
        self._chars_since_recall = 0

    def _recall(self, now=None):
        block = format_recall_block(
            self.index.recall_recent(topN=self.topN, half_life=self.half_life, now=now))
        if block is None:
            return None
        return block + "\n" + SELF_CHECK_FOOTER  # 每次注入都带自查指令，换窗和压缩触发同待遇

    def on_session_start(self, now=None, status=None):
        """触发点一·换窗：session 启动时调一次。
        返回可注入开场 context 的文本块（无可召回内容时 None）。
        顺带把压缩计数器清零——新窗口的对话量从零累计。

        status: 可选的 degradation_protocol.SessionStatus。宿主启动时若能填出四项
        故障信号，先过程序化自查关卡：有信号就不接入召回记忆，返回"可能在退化，
        请核实"的提示——时间感塌陷的窗口拿着旧记忆只会往错的方向自圆其说，先拦。
        没给 status（宿主查不了）时照常注入，靠块尾自查指令让模型自检。
        与 degradation_protocol v2 核心规则一致：只认故障信号，信号干净正常放行，
        不做"疑神疑鬼"式拦截。

        两层注入（规格 §5）：有会话线索就先给 thread 块（精度："上次那条线聊到
        哪、什么状态、有什么没聊完"），再给召回块（广度："最近发生过什么、什么
        曾经重要"）；没有 thread 就只有召回块，不断线。退化关卡在两层之前——
        窗口真漂移时一个字都不注入。"""
        self._chars_since_recall = 0
        if status is not None:
            signals = degradation_signals(status)
            if signals:
                return ("【开场自查未通过】检测到" + "/".join(signals) +
                        "，本窗口可能在退化：先不接入召回记忆，请对方核实后再继续。")
        blocks = []
        if self.thread_store is not None:
            thread_block = format_thread_block(self.thread_store.latest(), now=now)
            if thread_block:
                blocks.append(thread_block)
        recall_block = self._recall(now=now)
        if recall_block:
            blocks.append(recall_block)
        elif blocks:
            blocks.append(SELF_CHECK_FOOTER)  # 只有 thread 没有召回时，自查指令也得跟上
        return "\n\n".join(blocks) if blocks else None

    def on_turn(self, turn_text, now=None):
        """触发点二·压缩近似：每轮对话文本喂进来，累计字符量过阈值即触发并清零。

        再强调一次局限（docstring 顶部详述）：这不是"压缩刚发生"的精确信号，
        只是"对话长到差不多该压缩了"的粗估。返回召回文本块或 None（未过阈值）。"""
        self._chars_since_recall += len(turn_text)
        if self._chars_since_recall < self.compact_threshold_chars:
            return None
        self._chars_since_recall = 0
        return self._recall(now=now)


# ---------- selftest（合成数据，全部虚构） ----------

def _selftest():
    DAY = 86400.0
    now = 1_800_000_000.0  # 固定时刻，不依赖真实时钟

    idx = MemoryIndex()
    idx.add("## 昨天\n昨天调好了咖啡机的研磨度。", {"heading": "昨天", "timestamp": now - DAY})
    idx.add("## 上周\n上周整理了阳台的花盆。", {"heading": "上周", "timestamp": now - 7 * DAY})
    idx.add("## 上月\n上月读完了一本小说。", {"heading": "上月", "timestamp": now - 30 * DAY})

    # 1. 换窗触发：启动即召回，最新的排最前，topN 截断生效
    sr = SessionRecall(idx, topN=2, compact_threshold_chars=100)
    block = sr.on_session_start(now=now)
    assert block is not None and "昨天" in block.splitlines()[1], "最新记忆排最前"
    assert "上月" not in block, "topN=2 截断"

    # 2. 压缩近似触发：累计没过阈值不触发
    assert sr.on_turn("x" * 40, now=now) is None
    assert sr.on_turn("x" * 40, now=now) is None

    # 3. 累计过阈值 → 触发一次
    compact_block = sr.on_turn("x" * 40, now=now)
    assert compact_block is not None and "昨天" in compact_block, "累计 120 ≥ 阈值 100，触发"

    # 4. 触发后计数器清零，不连发
    assert sr.on_turn("x" * 40, now=now) is None, "刚触发过，重新从零累计"

    # 5. 再次累计过阈值 → 可重复触发
    sr.on_turn("x" * 40, now=now)
    assert sr.on_turn("x" * 40, now=now) is not None, "第二轮累计过阈值，再次触发"

    # 6. 换窗把压缩计数器清零：窗口边界之前的累计不带进新窗口
    sr2 = SessionRecall(idx, topN=1, compact_threshold_chars=100)
    sr2.on_turn("x" * 90, now=now)
    sr2.on_session_start(now=now)
    assert sr2.on_turn("x" * 40, now=now) is None, "换窗后对话量从零累计"

    # 7. 空索引兜底：不崩，两个触发点都返回 None（不注入空壳标题也不注入自查指令）
    empty = SessionRecall(MemoryIndex(), compact_threshold_chars=10)
    assert empty.on_session_start(now=now) is None
    assert empty.on_turn("x" * 20, now=now) is None

    # 8.【防时间感塌陷·变异靶心】每条片段带发生日期，块头写明"不是正在发生"
    assert "不是正在发生" in block.splitlines()[0], "块头声明这是历史片段"
    assert re.search(r"- \[\d{4}\.\d{2}\.\d{2}·昨天\]", block), "片段标注发生日期"

    # 9. 时间标注不假装精确：mtime 近似戳前加 ≈，缺时间戳标"时间未知"
    idx9 = MemoryIndex()
    idx9.add("近似时间的事", {"heading": "近似", "timestamp": now - DAY, "timestamp_source": "mtime"})
    idx9.add("没有时间的事", {"heading": "没戳"})
    b9 = SessionRecall(idx9, topN=2).on_session_start(now=now)
    assert re.search(r"\[≈\d{4}\.\d{2}\.\d{2}·近似\]", b9), "mtime 兜底的时间戳带 ≈"
    assert "[时间未知·没戳]" in b9, "缺时间戳标时间未知，不崩"

    # 10. 自查指令跟着每次注入走：换窗和压缩触发的块尾都带
    assert block.endswith(SELF_CHECK_FOOTER) and compact_block.endswith(SELF_CHECK_FOOTER)

    # 11.【防退化窗口自圆其说·变异靶心】程序化自查关卡：
    #    信号干净 → 正常注入（不做"疑神疑鬼"式拦截，与 degradation_protocol v2 一致）
    b_clean = sr.on_session_start(now=now, status=SessionStatus())
    assert b_clean is not None and "昨天" in b_clean, "没有故障信号不拦截"
    #    有故障信号 → 不接入召回记忆，返回"请核实"，召回内容一个字都不出现
    b_broken = sr.on_session_start(now=now, status=SessionStatus(time_check_ok=False))
    assert "时间错乱" in b_broken and "核实" in b_broken, "带出检测到的信号并请求核实"
    assert "昨天" not in b_broken and SELF_CHECK_FOOTER not in b_broken, "退化窗口不注入召回记忆"

    # ---- 会话线索 threads 两层注入（规格 §5）----
    from session_thread import ThreadStore, close_thread

    # 12.【变异靶心：thread 块在前】两层顺序：thread（精度）在前、召回（广度）在后
    store = ThreadStore()
    store.append(close_thread(12, now - 2 * DAY, now - DAY, ("阳台的花",),
                              "花买好了，周末的事没定。", open_loops=("周末去哪还没定",)))
    sr_t = SessionRecall(idx, topN=2, thread_store=store)
    b12 = sr_t.on_session_start(now=now)
    assert b12.startswith("【上次会话】"), "thread 块该排在最前"
    assert b12.index("【上次会话】") < b12.index("【最近记忆召回】"), "thread 在召回之前"
    assert "周末去哪还没定" in b12 and b12.endswith(SELF_CHECK_FOOTER), "没聊完的事要带上，自查指令仍在末尾"

    # 13. 没有 thread_store 就只有召回块——不断线（退回原行为）
    b13 = SessionRecall(idx, topN=2).on_session_start(now=now)
    assert b13.startswith("【最近记忆召回】") and "【上次会话】" not in b13

    # 14. 空 store 同样退回召回块；索引空但有 thread 时，thread 也得能单独注入且带自查
    assert SessionRecall(idx, topN=2, thread_store=ThreadStore()
                         ).on_session_start(now=now).startswith("【最近记忆召回】")
    only_thread = SessionRecall(MemoryIndex(), thread_store=store).on_session_start(now=now)
    assert only_thread.startswith("【上次会话】") and only_thread.endswith(SELF_CHECK_FOOTER)

    # 15. 退化关卡优先于两层注入——窗口漂移时 thread 也不给
    b15 = sr_t.on_session_start(now=now, status=SessionStatus(time_check_ok=False))
    assert "【上次会话】" not in b15 and "核实" in b15, "退化窗口连 thread 都不该注入"

    # 16.【变异靶心：单条截断】长片段被截断并标注，短的原样不动
    long_idx = MemoryIndex()
    long_idx.add("长" * 2000, {"heading": "很长的一块", "timestamp": now - DAY})
    long_idx.add("短块", {"heading": "短的", "timestamp": now - DAY})
    b16 = format_recall_block(long_idx.recall_recent(topN=2, now=now))
    assert TRUNCATION_NOTE in b16 and "短块" in b16, "长片段该截断、短片段该原样保留"
    assert len(b16) < 1200, f"截断后整块该显著变小，实际 {len(b16)}"
    #    关掉截断时原样给全（旋钮真的可关）
    assert TRUNCATION_NOTE not in format_recall_block(
        long_idx.recall_recent(topN=2, now=now), max_item_chars=None)

    print("selftest ok（16项断言：触发时机 / 时间标注防塌陷 / 自查关卡 / threads 两层注入 / 单条截断）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
