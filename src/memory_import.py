#!/usr/bin/env python3
"""
通用导入翻译器层参考实现（README 核心想法第 1 条 + 设计笔记"通用导入"一节，
任务卡"通用导入翻译器层"）。

中间标准格式一个：MemoryEntry {时间戳, 说话人, 文本, 标签?}（设计笔记 原设想）。
翻译器六个（⚠ 原文写「四个」，Claude memories.json 与 projects/*.json 两支加进来时
这行没跟着改；selftest 打印那行写的一直是六个），各来源 → 统一中间格式：
  parse_chatgpt_export   ChatGPT conversations.json（mapping 树，沿 current_node
                         回溯取当前分支——编辑重发产生的废弃分支不混进来）
  parse_claude_export    Claude 导出 json（chat_messages 列表，ISO 时间戳）
  parse_claude_memories  Claude 导出包里的 memories.json——Claude 自己蒸馏的用户
                         记忆摘要，叙事体长文本，整条导入不切散
  parse_claude_project   Claude 导出包里的 projects/*.json——项目知识库文档
                         （docs 列表），每份文档整条导入
  parse_line_chat        通用逐行聊天文本：时间戳行式（"2026-07-15 21:03 说话人"+
                         内容行，微信第三方导出工具的常见形态）和"说话人：内容"式
  parse_markdown_timeline 我们自己的 timeline md——复用 memory_retrieval 的
                         timeline_chunks（切块+时间戳解析逻辑只有一份，不复制）

流水线两头接上现有消费方：
  entries_to_index → 建好的 MemoryIndex（切块走会话自然边界：时间间隔断开，呼应
                     chunking 实验"自然边界优于机械字数"；chunk meta 带时间戳，
                     recall_recent 对导入语料同样生效）
  entries_to_turns → [(speaker, text)]，draft_extraction 那套提炼函数直接可用
                     （它 docstring 里"留给翻译器层再补"的口从这里接上）
入口 load_any(path)：按扩展名+结构嗅探分发——"选个文件，点导入"的体验目标；
认不出的结构明确报错，不静默猜。

**诚实边界（核对状态分翻译器记，别混）**：
  1. Claude 全家翻译器**已与真实导出两轮核对**（2026.07.30，维护者提供官方导出，
     语料不入库、只核结构+聚合冒烟）：conversations.json 字段名/sender 枚举
     （human、assistant）/时间戳形状（ISO 微秒+Z）/空消息跳过全部对上，且全量
     样本（12 对话、2153 条非空消息）把正文提取和 entries_to_index→recall 整条
     管线真跑通（396 块全带时间戳、recall 按新鲜度降序）；content 块实测有
     text/thinking/tool_use/tool_result 四种类型，只取 text 是对的——thinking
     的正文键叫 thinking 不叫 text，工具块是过程记录，都不该进记忆；
     memories.json、projects/*.json 结构同批核对。
     另一条实测经验：官方导出选"最近 30 天"会把窗口外的对话全部框空（第一轮
     就踩了——最后一条 app 对话在窗口外 5 天，导出近乎全空），导记忆一律选全量。
  2. ChatGPT 翻译器字段名按公开已知结构写。**2026.08.03 维护者实测：ChatGPT、
     Kelivo、Operit 三家导出的 json 都能被 load_any 认出并正常建库起服务**——
     所以"认不认得出"这一格不再空着。
     ⚠ **但"起得来"不等于"字段语义对"，这两件事这里刻意分开记**：load_any 的分发
     是**按顶层键嗅探**（mapping / chat_messages / conversations_memory / docs），
     键匹配上就会进对应翻译器；说话人枚举归一得对不对、时间戳形状认得对不对、
     空消息该不该跳，嗅探一概不管。上面第 1 条 Claude 那家敢写"已核对"，
     是因为真做过两轮字段级比对并留了具体数字（12 对话、2153 条消息、396 块）。
     **这一档还没有那种证据，所以接真样本前仍然先跑 --stats 核对
     条数/说话人/时间范围再信。**
     ⚠ 还有一格没答：Kelivo／Operit 的 json 具体命中的是哪个翻译器（多半是
     ChatGPT 那支，因为不少 App 照抄了它的导出形状）。这决定要不要给它们单独写
     翻译器——**在答案出来之前，别把"它们能跑"写成"我们支持这两家的格式"**。
  3. 微信官方没有导出功能、第三方工具格式五花八门——不猜着写某个具体工具的
     解析器，parse_line_chat 按最常见形态覆盖，对不上的等拿到真实样本再补。

零依赖，stdlib only。
用法：
  python memory_import.py --selftest
  python memory_import.py --stats <导出文件>     # 接真实样本前的核对步骤
"""

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 同目录模块，import 不触发其 CLI
from memory_retrieval import MemoryIndex, timeline_chunks

# 会话切块参数：半小时没说话算一次会话边界、单块字符上限——都是经验值，
# 跟检索层 +0.05/k=60 同一待遇，等真实评估再调
DEFAULT_GAP_SECONDS = 1800
DEFAULT_MAX_CHARS = 800


@dataclass
class MemoryEntry:
    """中间标准格式（设计笔记"通用导入"：{时间戳, 说话人, 文本, 标签?}）。
    timestamp 是 epoch 秒或 None；speaker 叙事体语料（timeline）为空串；
    tags 第一个元素约定为来源（chatgpt/claude/chatlog/timeline）。"""
    text: str
    speaker: str = ""
    timestamp: float = None
    tags: tuple = ()


# ---------- 翻译器一：ChatGPT conversations.json ----------

def _chatgpt_conv_entries(conv):
    mapping = conv.get("mapping") or {}
    cur = conv.get("current_node")
    if cur in mapping:
        # 沿 current_node 回溯到根：这是当前对话分支——编辑重发会在 mapping 树里
        # 留下废弃分支，全收会把"被编辑掉的旧问法/旧回答"混进记忆
        node_ids = []
        while cur:
            node_ids.append(cur)
            cur = (mapping.get(cur) or {}).get("parent")
        node_ids.reverse()
    else:
        # 没有 current_node 的兜底：按 create_time 排。局限如实标：废弃分支会混进来
        node_ids = sorted(mapping, key=lambda k: ((mapping[k].get("message") or {}).get("create_time") or 0))
    title = conv.get("title") or ""
    out = []
    for nid in node_ids:
        msg = (mapping.get(nid) or {}).get("message")
        if not msg:
            continue
        if ((msg.get("author") or {}).get("role")) not in ("user", "assistant"):
            continue  # system/tool 节点不是对话内容
        parts = ((msg.get("content") or {}).get("parts")) or []
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        out.append(MemoryEntry(text=text, speaker=msg["author"]["role"], timestamp=msg.get("create_time"),
                               tags=("chatgpt", title) if title else ("chatgpt",)))
    return out


def parse_chatgpt_export(data):
    """conversations.json（单个对话 dict 或对话列表）→ [MemoryEntry]。"""
    entries = []
    for conv in (data if isinstance(data, list) else [data]):
        entries.extend(_chatgpt_conv_entries(conv))
    return entries


# ---------- 翻译器二：Claude 导出 json ----------

def _iso_ts(s):
    """ISO8601 → epoch 秒；解析不了返回 None 不抛（时间戳缺失下游本来就有兜底）。"""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def parse_claude_export(data):
    """Claude 导出（单个对话 dict 或列表，chat_messages 结构）→ [MemoryEntry]。
    sender 的 human 归一成 user，跟 ChatGPT 翻译器出口一致。"""
    entries = []
    for conv in (data if isinstance(data, list) else [data]):
        title = conv.get("name") or ""
        for m in conv.get("chat_messages") or []:
            text = (m.get("text") or "\n".join(
                b.get("text", "") for b in (m.get("content") or []) if b.get("type") == "text")).strip()
            if not text:
                continue
            speaker = {"human": "user"}.get(m.get("sender"), m.get("sender") or "")
            entries.append(MemoryEntry(text=text, speaker=speaker, timestamp=_iso_ts(m.get("created_at")),
                                       tags=("claude", title) if title else ("claude",)))
    return entries


def parse_claude_project(data):
    """Claude 导出包里的 projects/*.json：项目知识库（docs: [{filename, content,
    created_at}]），是用户主动整理上传的资料不是对话。每份文档整条导入（它天然
    是完整单元），时间戳取文档 created_at。结构与真实导出核对过（2026.07.30）。"""
    entries = []
    for proj in (data if isinstance(data, list) else [data]):
        pname = proj.get("name") or ""
        for doc in proj.get("docs") or []:
            text = (doc.get("content") or "").strip()
            if not text:
                continue
            entries.append(MemoryEntry(text=text, speaker="",
                                       timestamp=_iso_ts(doc.get("created_at")),
                                       tags=("claude_project", pname, doc.get("filename") or "")))
    return entries


def parse_claude_memories(data):
    """Claude 导出包里的 memories.json：[{conversations_memory: 长文本, account_uuid}]。
    是 Claude 自己蒸馏的用户记忆摘要——叙事体、无时间戳，整条当一个条目导入，
    不切散（它本身已是浓缩产物，再切只会打碎语义）。结构与真实导出核对过（2026.07.30）。"""
    entries = []
    for item in (data if isinstance(data, list) else [data]):
        text = (item.get("conversations_memory") or "").strip()
        if text:
            entries.append(MemoryEntry(text=text, speaker="", tags=("claude_memory",)))
    return entries


# ---------- 翻译器三：通用逐行聊天文本 ----------

_LINE_TS_RE = re.compile(
    r"^(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(.*)$")


def parse_line_chat(text):
    """通用逐行聊天文本 → [MemoryEntry]。两种形态：
      A) 时间戳行"2026-07-15 21:03[:22] 说话人"，后续行是内容，直到下一个时间戳行
      B) "说话人：内容"单行（draft_extraction.parse_transcript 同款），无时间戳
    微信第三方导出工具格式不一，这里按最常见形态写——接真实样本前先 --stats 核对。"""
    entries, cur = [], None  # cur = [speaker, ts, lines]

    def flush():
        nonlocal cur
        if cur and cur[2]:
            entries.append(MemoryEntry(text="\n".join(cur[2]).strip(), speaker=cur[0],
                                       timestamp=cur[1], tags=("chatlog",)))
        cur = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_TS_RE.match(line)
        if m:
            flush()
            y, mo, d, hh, mm, ss, speaker = m.groups()
            try:
                ts = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0)).timestamp()
            except ValueError:
                ts = None  # "2026-13-40 25:99"这类假日期行：当成时间未知，内容照收
            cur = [speaker.strip(), ts, []]
            continue
        if cur is not None:
            cur[2].append(line)
            continue
        sep = "：" if "：" in line else (":" if ":" in line else None)
        if sep:
            speaker, _, content = line.partition(sep)
            if speaker.strip() and content.strip():
                entries.append(MemoryEntry(text=content.strip(), speaker=speaker.strip(), tags=("chatlog",)))
    flush()
    return entries


# ---------- 翻译器四：markdown timeline ----------

def parse_markdown_timeline(text, filename="timeline.md", mtime=0.0):
    """timeline md → [MemoryEntry]。切块+时间戳解析全部复用 memory_retrieval 的
    timeline_chunks（chunk_head/file_head/mtime 那套优先级），这里只包一层中间格式。
    叙事体没有说话人，speaker 为空串——entries_to_turns 会把它筛掉，不进对话轮。"""
    entries = []
    for chunk, meta in timeline_chunks(text, filename, mtime):
        entries.append(MemoryEntry(text=chunk, speaker="", timestamp=meta["timestamp"],
                                   tags=("timeline", meta["heading"]) if meta["heading"] else ("timeline",)))
    return entries


# ---------- 流水线出口：两个消费方 ----------

def entries_to_index(entries, embed=False, gap_seconds=DEFAULT_GAP_SECONDS,
                     max_chars=DEFAULT_MAX_CHARS, source="import"):
    """中间格式 → 建好的 MemoryIndex（统一流水线的"切块→存库"段）。

    切块走会话自然边界（chunking 实验结论：自然边界优于机械字数）：相邻条目时间
    间隔超 gap_seconds 断开；块超 max_chars 也断开，但单条超长条目不硬切（timeline
    叙事块整条进，不重切）。chunk meta 带首条时间戳（timestamp_source="import"），
    recall_recent 对导入语料同样生效——换窗召回不区分语料来自哪个 app。"""
    index = MemoryIndex(embed=embed)
    buf, buf_ts, last_ts, size = [], None, None, 0

    def flush():
        nonlocal buf, buf_ts, size
        if buf:
            meta = {"source": source, "chunk_index": len(index.chunks)}
            if buf_ts is not None:
                meta["timestamp"], meta["timestamp_source"] = buf_ts, "import"
            index.add("\n".join(buf), meta)
        buf, buf_ts, size = [], None, 0

    for e in entries:
        line = f"{e.speaker}：{e.text}" if e.speaker else e.text
        gap = (e.timestamp is not None and last_ts is not None
               and e.timestamp - last_ts > gap_seconds)
        if buf and (gap or size + len(line) > max_chars):
            flush()
        buf.append(line)
        size += len(line)
        if buf_ts is None:
            buf_ts = e.timestamp
        if e.timestamp is not None:
            last_ts = e.timestamp
    flush()
    return index.build()


def entries_to_turns(entries):
    """中间格式 → [(speaker, text)] 对话轮，draft_extraction 的提炼函数直接可用。
    无说话人的叙事体条目（timeline）不是对话轮，筛掉。"""
    return [(e.speaker, e.text) for e in entries if e.speaker]


# ---------- 入口：选个文件，点导入 ----------

def load_any(path):
    """按扩展名+结构嗅探分发到对应翻译器 → [MemoryEntry]。
    认不出的结构明确报错，不静默猜成某种格式。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        sample = data[0] if isinstance(data, list) and data else data
        if isinstance(sample, dict) and "mapping" in sample:
            return parse_chatgpt_export(data)
        if isinstance(sample, dict) and "chat_messages" in sample:
            return parse_claude_export(data)
        if isinstance(sample, dict) and "conversations_memory" in sample:
            return parse_claude_memories(data)
        if isinstance(sample, dict) and isinstance(sample.get("docs"), list):
            return parse_claude_project(data)
        raise ValueError(f"认不出的 json 导出结构：{path}"
                         "（mapping / chat_messages / conversations_memory / docs 都没有）")
    if p.suffix.lower() in (".md", ".markdown"):
        return parse_markdown_timeline(raw, p.name, p.stat().st_mtime)
    return parse_line_chat(raw)


def stats(path):
    """接真实样本前的核对步骤：只打聚合统计，不打内容（语料不该进终端记录）。"""
    entries = load_any(path)
    speakers = sorted({e.speaker for e in entries if e.speaker})
    ts = [e.timestamp for e in entries if e.timestamp is not None]
    fmt = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    line = f"{len(entries)} 条；说话人：{'/'.join(speakers) or '（叙事体，无说话人）'}；带时间戳 {len(ts)} 条"
    if ts:
        line += f"；时间范围 {fmt(min(ts))} ~ {fmt(max(ts))}"
    print(line)


# ---------- selftest（合成 fixture，全部虚构，不含任何真实语料） ----------

_CHATGPT_FIXTURE = {
    "title": "修咖啡机",
    "current_node": "n3",
    "mapping": {
        "root": {"message": None, "parent": None, "children": ["sys"]},
        "sys": {"message": {"author": {"role": "system"}, "create_time": None,
                            "content": {"content_type": "text", "parts": [""]}},
                "parent": "root", "children": ["n1"]},
        "n1": {"message": {"author": {"role": "user"}, "create_time": 1752500000.0,
                           "content": {"content_type": "text", "parts": ["咖啡机又坏了，怎么办"]}},
               "parent": "sys", "children": ["n2", "n2_edited"]},
        "n2_edited": {"message": {"author": {"role": "assistant"}, "create_time": 1752500100.0,
                                  "content": {"content_type": "text", "parts": ["被编辑掉的旧回答"]}},
                      "parent": "n1", "children": []},
        "n2": {"message": {"author": {"role": "assistant"}, "create_time": 1752500060.0,
                           "content": {"content_type": "text", "parts": ["检查保险丝有没有熔断"]}},
               "parent": "n1", "children": ["n3"]},
        "n3": {"message": {"author": {"role": "user"}, "create_time": 1752500120.0,
                           "content": {"content_type": "text", "parts": ["果然是保险丝"]}},
               "parent": "n2", "children": []},
    },
}

# fixture 形状照 2026.07.30 真实导出核对结果构造（内容虚构）：时间戳 ISO 微秒+Z、
# 消息带 uuid/attachments/files 等我们不用的键（解析器要容忍）、含一条真实样本里
# 实际存在的空消息（text 空且 content 空）
_CLAUDE_FIXTURE = [{
    "uuid": "0" * 36,
    "name": "种薄荷",
    "summary": "",
    "created_at": "2026-07-15T21:03:00.000000Z",
    "chat_messages": [
        {"uuid": "1" * 36, "sender": "human", "text": "薄荷又蔫了", "content": [],
         "created_at": "2026-07-15T21:03:22.000000Z", "attachments": [], "files": []},
        {"uuid": "2" * 36, "sender": "assistant", "text": "",
         "content": [
             # 真实导出的 content 块有四种类型；thinking 的正文键叫 thinking、
             # 工具块是过程记录——只有 text 类型该进记忆，混在这里当回归守护
             {"type": "thinking", "thinking": "思考过程不该进记忆", "summaries": [], "cut_off": False},
             {"type": "tool_use", "name": "web_search", "input": {}, "message": ""},
             {"type": "tool_result", "name": "web_search", "content": [], "is_error": False},
             {"type": "text", "text": "盆底是不是积水了", "citations": [], "flags": []},
         ],
         "created_at": "2026-07-15T21:04:00.000000Z", "attachments": [], "files": []},
        {"uuid": "3" * 36, "sender": "human", "text": "", "content": [],
         "created_at": "2026-07-15T21:05:00.000000Z", "attachments": [], "files": []},
    ],
}]

_CLAUDE_MEMORIES_FIXTURE = [{"conversations_memory": "喜欢薄荷茶；阳台养了两盆植物；读小说偏爱结构复杂的。",
                             "account_uuid": "a" * 36}]

_CLAUDE_PROJECT_FIXTURE = {
    "uuid": "c" * 36, "name": "阳台种植笔记", "description": "", "is_private": True,
    "prompt_template": "", "created_at": "2026-07-01T10:00:00.000000+00:00",
    "docs": [
        {"uuid": "d" * 36, "filename": "浇水表.md", "content": "薄荷两天一浇，正午不浇。",
         "created_at": "2026-07-01T10:05:00.000000+00:00"},
        {"uuid": "e" * 36, "filename": "空文档.md", "content": "",
         "created_at": "2026-07-02T10:05:00.000000+00:00"},
    ],
}

_LINECHAT_FIXTURE = """2026-07-15 21:03 林岸
今天真的好累
2026-07-15 21:04:10 星回
先歇会儿，活儿明天还在
跑不掉的
"""

_TIMELINE_FIXTURE = "# window_30_晒被子（2026.07.15）\n\n## 上午\n把被子都晒了。\n\n## 傍晚\n收被子，带着太阳味。\n"


def _selftest():
    # 1.【变异靶心：current_node 回溯】ChatGPT 翻译器取当前分支：3 条、按序、
    #    编辑废弃分支不混入、system 节点被滤掉、epoch 时间戳原样带出
    e = parse_chatgpt_export(_CHATGPT_FIXTURE)
    assert [(x.speaker, x.text) for x in e] == [
        ("user", "咖啡机又坏了，怎么办"), ("assistant", "检查保险丝有没有熔断"), ("user", "果然是保险丝")], \
        f"当前分支该是 3 条对话，实际 {[(x.speaker, x.text) for x in e]}"
    assert all("被编辑掉" not in x.text for x in e), "废弃分支不该混进记忆"
    assert e[0].timestamp == 1752500000.0 and e[0].tags == ("chatgpt", "修咖啡机")

    # 2. 没有 current_node 的兜底：按 create_time 排——废弃分支会混进来，局限如实测出
    no_cur = {k: v for k, v in _CHATGPT_FIXTURE.items() if k != "current_node"}
    e2 = parse_chatgpt_export(no_cur)
    assert len(e2) == 4 and e2[2].text == "被编辑掉的旧回答", "兜底路径按时间排、含废弃分支（已知局限）"

    # 3. Claude 翻译器（fixture 形状照真实导出核对结果）：text 字段和 content 块两种
    #    形态都吃，human 归一成 user，微秒+Z 时间戳解析，空消息（真实样本里存在）跳过
    e3 = parse_claude_export(_CLAUDE_FIXTURE)
    assert [(x.speaker, x.text) for x in e3] == [("user", "薄荷又蔫了"), ("assistant", "盆底是不是积水了")], \
        "两条非空消息进、一条空消息跳过；thinking/tool_use/tool_result 块不进记忆"
    assert e3[0].timestamp is not None and e3[1].timestamp - e3[0].timestamp == 38.0, "ISO 时间戳差 38 秒"
    assert e3[0].tags == ("claude", "种薄荷")

    # 4. 逐行文本翻译器：时间戳行式（含多行内容归并、秒可省）和"说话人：内容"式
    e4 = parse_line_chat(_LINECHAT_FIXTURE)
    assert len(e4) == 2 and e4[0].speaker == "林岸" and e4[1].text == "先歇会儿，活儿明天还在\n跑不掉的"
    assert e4[1].timestamp - e4[0].timestamp == 70.0, "21:03 → 21:04:10 差 70 秒"
    e4b = parse_line_chat("林岸：不想歇\n星回：那就再撑十分钟")
    assert [(x.speaker, x.text) for x in e4b] == [("林岸", "不想歇"), ("星回", "那就再撑十分钟")]

    # 5. timeline 翻译器：时间戳走 timeline_chunks 那套（标题行完整日期），无说话人
    e5 = parse_markdown_timeline(_TIMELINE_FIXTURE, "window_30_晒被子.md", mtime=1752500000.0)
    assert len(e5) >= 2 and all(x.speaker == "" for x in e5)
    assert all(datetime.fromtimestamp(x.timestamp).strftime("%Y%m%d") == "20260715" for x in e5), \
        "timeline 条目时间戳该来自标题行日期，不是 mtime"

    # 6. 消费方一（draft_extraction）真接上：翻译器出口 → 对话轮 → 风格候选
    from draft_extraction import extract_style_candidates
    turns = entries_to_turns(e4 + e4b)
    assert ("星回", "那就再撑十分钟") in turns and all(who for who, _ in turns)
    assert entries_to_turns(e5) == [], "叙事体条目不进对话轮"
    cands = extract_style_candidates(turns, "星回", topN=2)
    assert cands and all(not c.confirmed for c in cands), "提炼函数直接吃翻译器出口"

    # 7.【变异靶心：会话间隙切块 + 时间戳传导】消费方二（MemoryIndex）真接上
    t0 = 1752500000.0
    convo = [MemoryEntry("今天真的好累", "林岸", t0),
             MemoryEntry("先歇会儿", "星回", t0 + 60),
             MemoryEntry("昨晚那杯薄荷茶不错", "林岸", t0 + 7200)]
    idx = entries_to_index(convo, gap_seconds=1800)
    assert len(idx.chunks) == 2, "隔两小时是新会话，该切成两块"
    assert idx.chunks[0] == "林岸：今天真的好累\n星回：先歇会儿", "同会话归并且带说话人前缀"
    assert idx.meta[0]["timestamp"] == t0 and idx.meta[1]["timestamp"] == t0 + 7200, "块时间戳=首条时间戳"
    assert idx.retrieve("薄荷茶", topN=1)[0]["id"] == 1, "导入后可检索"
    assert idx.recall_recent(topN=2, now=t0 + 7300)[0]["id"] == 1, "recall_recent 对导入语料生效"
    #    单条超长不硬切
    assert len(entries_to_index([MemoryEntry("长" * 2000, "林岸", t0)], max_chars=800).chunks) == 1

    # 8. load_any 分发：四种文件各进各的翻译器；认不出的 json 明确报错
    with tempfile.TemporaryDirectory() as td:
        cases = [("c.json", json.dumps([_CHATGPT_FIXTURE]), "chatgpt"),
                 ("k.json", json.dumps(_CLAUDE_FIXTURE), "claude"),
                 ("t.md", _TIMELINE_FIXTURE, "timeline"),
                 ("w.txt", _LINECHAT_FIXTURE, "chatlog")]
        for name, content, tag in cases:
            Path(td, name).write_text(content, encoding="utf-8")
            got = load_any(Path(td, name))
            assert got and got[0].tags[0] == tag, f"{name} 该分发到 {tag} 翻译器"
        Path(td, "bad.json").write_text('{"foo": 1}', encoding="utf-8")
        try:
            load_any(Path(td, "bad.json"))
            assert False, "认不出的结构该报错，不能静默猜"
        except ValueError:
            pass

    # 9. 空输入兜底：不崩，空进空出
    assert parse_line_chat("") == [] and entries_to_turns([]) == []
    empty = entries_to_index([])
    assert empty.retrieve("任意", topN=3) == [] and empty.recall_recent(topN=3) == []

    # 10. Claude memories.json 翻译器：整条导入不切散、无说话人不进对话轮、分发认得出
    e10 = parse_claude_memories(_CLAUDE_MEMORIES_FIXTURE)
    assert len(e10) == 1 and e10[0].tags == ("claude_memory",) and e10[0].speaker == ""
    assert "薄荷茶" in e10[0].text and entries_to_turns(e10) == []
    assert parse_claude_memories([{"conversations_memory": "", "account_uuid": "b" * 36}]) == [], "空摘要不产条目"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.json").write_text(json.dumps(_CLAUDE_MEMORIES_FIXTURE), encoding="utf-8")
        assert load_any(Path(td, "m.json"))[0].tags == ("claude_memory",), "load_any 认得 memories.json"

    # 11. Claude projects/*.json 翻译器：知识库文档整条导入、空文档跳过、时间戳解析、分发认得出
    e11 = parse_claude_project(_CLAUDE_PROJECT_FIXTURE)
    assert len(e11) == 1 and e11[0].tags == ("claude_project", "阳台种植笔记", "浇水表.md"), \
        "一份有内容的文档进、空文档跳过"
    assert e11[0].timestamp is not None and e11[0].speaker == ""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "p.json").write_text(json.dumps(_CLAUDE_PROJECT_FIXTURE), encoding="utf-8")
        assert load_any(Path(td, "p.json"))[0].tags[0] == "claude_project", "load_any 认得 projects/*.json"

    print("selftest ok（11项断言：六个翻译器 / 两个消费方真接上 / 分发与兜底）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", metavar="FILE", help="导入前核对：条数/说话人/时间范围（不打内容）")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.stats:
        stats(args.stats)
    else:
        ap.print_help()
