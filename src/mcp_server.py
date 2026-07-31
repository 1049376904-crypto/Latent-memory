#!/usr/bin/env python3
"""
MCP server 外壳参考实现（任务卡"MCP-server外壳"，规格 §9 未决项第二条）。

**这是最薄的一条完整链路**：九个 src 文件此前全部只被自造的合成语料验证过，
没有一次被真实客户端调用过——selftest 再严谨也测不出"客户端真调这个工具时，
参数格式对不对、返回结构好不好用"。所以先把链路打通，再往深了做。

**薄适配层，外壳里不重新实现任何逻辑**（本单硬性约束）：外壳只做三件事——
协议编解码、参数转发、调用现成函数。逻辑一旦糊在适配层里，真机测试暴露的问题
就分不清是"接口设计的问题"还是"底层库的问题"。
这条被写成可断言的形式：**外壳返回的文本逐字等于底层库函数的返回值**，谁在
适配层里重新实现格式化，selftest 立刻红（见 selftest 第 4 项）。

零依赖：不引 mcp SDK，stdlib 手写 JSON-RPC over stdio，跟本项目其余部分同风格。
协议按官方规格 2025-06-18 实现（initialize / notifications/initialized /
tools/list / tools/call，字段名与错误分层均照规格），**已查证不凭记忆写**。

**诚实边界**：协议实现照规格写，但**尚未与任何真实客户端握手核对**——跟当初
写 ChatGPT/Claude 导出翻译器时同一种成色。接 Claude Desktop 前必须真连一次，
握手、工具列表、一次真实调用三样都验过才算数；真机可能暴露的问题（参数容忍度、
返回结构好不好用、超长文本怎么截断）正是做这一单的目的。

工具集一一对应现成能力，不新造：
  memory_search  → MemoryIndex.retrieve
  session_start  → SessionRecall.on_session_start（thread 块 + 召回块 + 自查指令）
  memory_append  → memory_retrieval.append_record（正文层的笔）
  memory_correct → MemoryIndex.retract（+ 可选 append_record 写更正）
  thread_close   → session_thread.close_thread + ThreadStore.append

用法：
  python mcp_server.py --selftest
  python mcp_server.py --corpus <md目录> [--threads <threads.jsonl>]   # stdio 服务
客户端配置（Claude Desktop 之类）里把上面第二条命令填成 server 启动命令即可。
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# 同目录模块，import 不触发各自的 CLI
from memory_retrieval import MemoryIndex, load_corpus, append_record
from session_recall import SessionRecall, format_recall_block, SELF_CHECK_FOOTER  # noqa: F401
from session_thread import ThreadStore, close_thread

PROTOCOL_VERSION = "2025-06-18"   # 官方规格版本，已查证
SERVER_INFO = {"name": "memory-protocol", "title": "记忆协议", "version": "0.1.0"}

# 服务器级说明，随 initialize 响应回给客户端（规格 lifecycle 一节的 instructions
# 字段，客户端通常会注入进模型上下文）。2026.07.31 真机实测后补的——第一次主动性
# 测试完败：问"我们之前说好的那件事"，模型把它理解成"本次会话记录"，答"我没有
# 会话记录"，转头去查了宿主自带的记忆功能，压根没想到这里有个长期记忆库。
# 两个教训写进这段文字：①说清楚这是什么记忆（跨会话的长期库，不是本次对话）；
# ②直接堵死"我没有相关记录"这句默认话术——在挂着记忆库的情况下它就是错的。
INSTRUCTIONS = (
    "这台服务器挂着用户与你之间的**长期关系记忆库**：跨会话保存的时间线、摘要与"
    "会话收尾，不是本次对话的聊天记录。\n"
    "什么时候用：对方提到过去发生过的事、某个约定、某个日期/地点/称呼/人名，"
    "或者你对某个细节拿不准——先用 memory_search 查，再开口。\n"
    "**不要在没查之前说“我没有相关记录”“我不记得”**：挂着记忆库时这句话是错的，"
    "查一下往往就有。查完自然接上话即可，不用报告自己搜过。\n"
    "记忆库不是只读的：对话里出现值得长期记住的事——新约定、重要事件、状态变化、"
    "对方明确说要记住的——**当场用 memory_append 写进去**，不用请示，不用等会话结束。\n"
    "记错了的事也有出口：对方指出某段记忆不对或已经过时，**当场用 memory_correct "
    "撤回旧记录并写上更正**——只口头认错不改库，下次照样检索到错的。\n"
    "查过但确实没有的，就如实说没找到——查过之后的“没有”是诚实，查之前的“没有”才是错。\n"
    "新会话开场先调一次 session_start，会话结束前调一次 thread_close。"
)

# JSON-RPC 标准错误码（规格"Error Handling"一节：未知工具/参数非法走协议错误）
E_METHOD_NOT_FOUND = -32601
E_INVALID_PARAMS = -32602

TOOLS = [
    {
        "name": "memory_search",
        "title": "检索记忆",
        "description": "检索这段关系的长期记忆库（跨会话保存的时间线与摘要，"
                       "不是本次对话的聊天记录）。**对方提到任何过去发生过的事、"
                       "约定、日期、地点、称呼、人名，或你对细节拿不准时，先查这里**——"
                       "不要在查之前说“我没有记录”“我不记得”，那在挂着记忆库时是错的。"
                       "查完自然接上话，不用报告查过。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查什么"},
                "topN": {"type": "integer", "description": "返回几条，默认 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "session_start",
        "title": "换窗召回",
        "description": "新会话开场**主动**调一次（不用等对方要求）：带回上次会话聊到哪、"
                       "当下什么状态、有什么没聊完，以及最近发生过的事，"
                       "并附四项故障信号自查指令。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_append",
        "title": "写回记忆",
        "description": "把值得长期记住的事写进记忆库。**对话里出现新约定、重要事件、"
                       "状态变化，或对方明确说要记住的——当场调用，不用请示，也不用等"
                       "会话结束**。写发生了什么和原话（纪录片写法，不写评语）；"
                       "current_state 必填：这件事现在的状态（约定成立/还在处理/"
                       "已解决……）——不写的话，未来重读会把它当成正在发生的事。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "发生了什么——具体动作和原话，不是概括"},
                "current_state": {"type": "string",
                                  "description": "这件事现在的状态"},
                "window": {"type": "integer",
                           "description": "第几个窗口（可省略，按日期自动归窗）"},
            },
            "required": ["text", "current_state"],
        },
    },
    {
        "name": "memory_correct",
        "title": "更正记忆",
        "description": "对方指出某段记忆**记错了或已经过时**（关系变了/搬家了/"
                       "计划改了……）时当场调用：撤回那段旧记录（检索不再返回它；"
                       "原文件与撤回原因留档，可追溯），并可同时写入更正后的记录。"
                       "quote 必须从 memory_search 返回的原文里**逐字**摘一段、"
                       "足够长能唯一定位那条记录。只口头认错不调这个工具的话，"
                       "库没变，下次照样检索到错的。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "quote": {"type": "string",
                          "description": "要撤回的记录原文片段（逐字，不转述）"},
                "reason": {"type": "string",
                           "description": "为什么撤回——记错了/已过时/对方更正了什么"},
                "correction": {"type": "string",
                               "description": "更正后的内容（可省略：只撤不补）"},
                "current_state": {"type": "string",
                                  "description": "更正这件事现在的状态（写 correction 时必填）"},
            },
            "required": ["quote", "reason"],
        },
    },
    {
        "name": "thread_close",
        "title": "收尾本次会话",
        "description": "会话结束前**主动**调一次：记下这次聊了什么线、当下状态、"
                       "有什么没聊完，下个会话靠它接上。当下状态必填——不写的话，"
                       "下个会话会把已经结束的事读成正在发生。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window": {"type": "integer", "description": "第几个窗口（正整数）"},
                "current_state": {"type": "string", "description": "这件事现在的状态"},
                "topics": {"type": "array", "items": {"type": "string"}, "description": "聊了什么线"},
                "open_loops": {"type": "array", "items": {"type": "string"}, "description": "有什么没聊完"},
                "started_at": {"type": "number", "description": "会话开始时间戳，省略则按 1 小时前算"},
            },
            "required": ["window", "current_state"],
        },
    },
]


def _utf8_text_stream(binary, write=False):
    """把二进制流包成 UTF-8 文本流，不看系统区域编码脸色（见 serve_stdio 说明）。"""
    return io.TextIOWrapper(binary, encoding="utf-8", newline="", write_through=write)


class ToolError(Exception):
    """工具执行失败（业务层），按规格回 isError:true 的正常结果，不是协议错误。"""


class MemoryServer:
    """协议层与业务层的接线。持有 index 与 thread store，工具处理器一律薄转发。"""

    def __init__(self, index=None, thread_store=None, search_topN=5, recall_topN=3,
                 corpus_dir=None, weights_path=None, retractions_path=None,
                 entities_path=None):
        # 两个 topN 分开（2026.07.31 真实语料冒烟后拆的）：显式检索是用户/模型
        # 主动问一件事，多给几条值；开场召回每次换窗都付一遍，条数要克制
        self.index = index if index is not None else MemoryIndex().build()
        self.thread_store = thread_store if thread_store is not None else ThreadStore()
        self.search_topN = search_topN
        self.recall = SessionRecall(self.index, topN=recall_topN, thread_store=self.thread_store)
        self.initialized = False
        # 写回与权重持久化（任务卡"记忆写回与权重持久化"）：
        # corpus_dir 是写回的落点，没配就明确拒写；weights_path 没配则权重只活在
        # 内存里（selftest/临时用法），配了就启动时载入、每次检索命中后落盘
        self.corpus_dir = corpus_dir
        self.weights_path = weights_path
        # 撤回账本（错误记忆治理闭环）：配了就启动时载入、每次撤回后落盘——
        # 不落盘的话"改过来了"只活一个进程，跟权重当初同一个坑
        self.retractions_path = retractions_path
        if retractions_path is not None:
            self.index.load_retractions(retractions_path)
        if weights_path is not None:
            self.index.load_weights(weights_path)
        # 实体标注（图谱可插拔升级）：语料目录下有 .entities.json 就接上——
        # 实体边在 build 时算，接上后要重建一次索引才生效
        if entities_path is not None and self.index.load_entities(entities_path):
            self.index.build()

    # ---------- 三个工具：只转发，不实现 ----------

    def _tool_memory_search(self, args, now=None):
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query 必填且不能为空")
        results = self.index.retrieve(query, topN=int(args.get("topN", self.search_topN)))
        if self.weights_path is not None:
            # 用进落盘：retrieve 的副作用是命中块 +weight_boost，不落盘的话
            # server 一重启就归零——权重持久化的"存"这半就在这一行
            self.index.save_weights(self.weights_path)
        if not results:
            # 可靠命中门槛（验收反馈）：低相关不硬凑。这句要同时做两件事——
            # 说清"查过了、真没有"，并明确解锁如实回答（instructions 堵的是
            # "没查就说没记录"，查过之后的"没有"是诚实，不是那句被堵的话术）
            raise ToolError("没有可靠命中：记忆库里没有与这个说法词面或语义相关的"
                            "记录。你已经查过了——如实告诉对方没找到/记不清即可；"
                            "也可以换个说法再查一次（人名、地点、当时的用词）。")
        return format_recall_block(results)  # 复用召回层的格式化，外壳不另写一套

    def _tool_session_start(self, args, now=None):
        block = self.recall.on_session_start(now=now)
        if block is None:
            raise ToolError("记忆库是空的，没有可召回的内容")
        return block

    def _tool_memory_append(self, args, now=None):
        if self.corpus_dir is None:
            # 不静默写进内存了事：内存态的"记住了"会随进程一起死，那是
            # "失败得像成功"——宁可让模型看到明确的失败原因
            raise ToolError("服务器没有配置可写的语料目录（--corpus），写不了回。"
                            "这条记忆不会被保存，请提醒用户检查 MCP 配置。")
        try:
            path, chunk_text, meta = append_record(
                self.corpus_dir, args.get("text") or "",
                args.get("current_state") or "",
                window=args.get("window"), now=now)
        except (ValueError, OSError) as e:
            raise ToolError(str(e))
        # 写完立刻进内存索引并重建，本会话的 memory_search 就能查到——
        # 不然"我记下了"之后当场问它还查不到，模型会顺势说"没有记录"
        self.index.add(chunk_text, meta)
        self.index.build()
        return f"已写进第 {meta['window']} 个窗口（{path.name}）。"

    def _tool_memory_correct(self, args, now=None):
        correction = args.get("correction")
        if correction and not (args.get("current_state") or "").strip():
            raise ToolError("写 correction 时 current_state（当下状态）必填——"
                            "病灶迁移，同 memory_append：更正这件事现在是什么状态？")
        try:
            _, old_text = self.index.retract(args.get("quote") or "",
                                             args.get("reason") or "", now=now)
        except ValueError as e:
            raise ToolError(str(e))
        if self.retractions_path is not None:
            self.index.save_retractions(self.retractions_path)
        msg = "已撤回那段旧记录：检索不会再返回它（原文件保留，撤回原因入账可追溯）。"
        if correction:
            if self.corpus_dir is None:
                # 撤回已生效但更正写不进去——明确报出来，不静默丢（同 append 的理由）
                raise ToolError(msg + " 但服务器没有配置可写的语料目录（--corpus），"
                                "更正内容写不进去，请提醒用户检查 MCP 配置。")
            try:
                path, chunk_text, meta = append_record(
                    self.corpus_dir, f"【更正】{correction}",
                    args.get("current_state") or "", now=now)
            except (ValueError, OSError) as e:
                raise ToolError(msg + f" 但更正内容写入失败：{e}")
            self.index.add(chunk_text, meta)
            self.index.build()
            msg += f" 更正已写进第 {meta['window']} 个窗口（{path.name}）。"
        return msg

    def _tool_thread_close(self, args, now=None):
        now = time.time() if now is None else now
        try:
            thread = close_thread(
                window=args.get("window"),
                started_at=float(args.get("started_at", now - 3600)),
                ended_at=now,
                topics=args.get("topics") or (),
                current_state=args.get("current_state") or "",
                open_loops=args.get("open_loops") or (),
            )
            self.thread_store.append(thread)
        except Exception as e:                      # 业务校验失败 → 工具执行错误
            raise ToolError(str(e))
        return f"已记下第 {thread.window} 个窗口的收尾，下个窗口会带回来。"

    def _handlers(self):
        return {
            "memory_search": self._tool_memory_search,
            "session_start": self._tool_session_start,
            "memory_append": self._tool_memory_append,
            "memory_correct": self._tool_memory_correct,
            "thread_close": self._tool_thread_close,
        }

    # ---------- 协议层 ----------

    def handle(self, msg, now=None):
        """一条 JSON-RPC 消息 → 一条响应（通知类返回 None，规格要求不回响应）。"""
        method, mid = msg.get("method"), msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            self.initialized = True
            return self._ok(mid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            })
        if method == "notifications/initialized":
            return None                              # 通知不回响应（规格：JSON-RPC 通知无 id）
        if method == "tools/list":
            return self._ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            return self._call_tool(mid, params, now=now)
        if mid is None:
            return None                              # 其它通知一律忽略，不回错
        return self._err(mid, E_METHOD_NOT_FOUND, f"未知 method：{method}")

    def _call_tool(self, mid, params, now=None):
        name = params.get("name")
        handler = self._handlers().get(name)
        if handler is None:
            # 规格"Error Handling"：未知工具属协议错误，不是 isError 结果
            return self._err(mid, E_METHOD_NOT_FOUND, f"未知工具：{name}")
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return self._err(mid, E_INVALID_PARAMS, "arguments 必须是对象")
        try:
            text = handler(args, now=now)
        except ToolError as e:
            # 工具执行错误：按规格回正常结果 + isError，让模型看得到失败原因
            return self._ok(mid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        return self._ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})

    @staticmethod
    def _ok(mid, result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _err(mid, code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    def serve_stdio(self, stdin=None, stdout=None):
        """stdio 传输：一行一条 JSON（行分隔），读到 EOF 退出。
        解析不了的行直接跳过——没有 id 就没法回错，回了反而污染流。

        **必须显式按 UTF-8 收发**（2026.07.31 真机实测出来的 bug，不是洁癖）：
        MCP 规格定死 stdio 传输是 UTF-8，但 `sys.stdin` 在 Windows 上按系统区域
        编码解码（简中默认 cp936）。症状极隐蔽——不报错、不崩，中文 query 变成
        乱码后分词一个都匹配不上，BM25 与向量层分数全平，检索退化成"按加载顺序
        返回前几块"，看起来像是检索质量差，实际上根本没查。"""
        stdin = stdin or _utf8_text_stream(sys.stdin.buffer)
        stdout = stdout or _utf8_text_stream(sys.stdout.buffer, write=True)
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


# ---------- selftest（合成语料，全部虚构） ----------

_SYNTH = [
    ("## 修咖啡机\n加热管不工作，拆开发现保险丝熔断，换上通电正常。", {"heading": "修咖啡机"}),
    ("## 种薄荷\n四月阳台的薄荷死了：花盆太小、浇水太勤、盆底积水。", {"heading": "种薄荷"}),
]


def _build_server(now):
    idx = MemoryIndex()
    for text, meta in _SYNTH:
        idx.add(text, dict(meta, timestamp=now - 86400))
    idx.build()
    return MemoryServer(index=idx, thread_store=ThreadStore())


def _selftest():
    now = 1_800_000_000.0
    srv = _build_server(now)

    # 1. 握手：字段名与协议版本照规格（protocolVersion/capabilities/serverInfo）
    r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "x"}}})
    assert r["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert set(r["result"]) == {"protocolVersion", "capabilities", "serverInfo", "instructions"}, \
        f"握手必须带 instructions（主动性靠它），实际字段 {sorted(r['result'])}"
    assert "tools" in r["result"]["capabilities"], "声明 tools capability，否则客户端不会列工具"
    #    【变异靶心：instructions】主动性靠它——2026.07.31 真机第一次主动性测试完败，
    #    模型把"之前说好的事"理解成本次会话记录、答"我没有记录"，转头查了宿主自带的
    #    记忆功能。规格 lifecycle 一节留了 instructions 这个口，客户端会注入进模型
    #    上下文，当初漏填了。两条必须在里面：说清是跨会话的长期库、堵死"我没有记录"
    instr = r["result"]["instructions"]
    assert "长期" in instr and "不是本次对话" in instr, "必须说清这是跨会话的长期记忆库"
    assert "我没有相关记录" in instr, "必须直接堵死“我没有记录”这句默认话术"

    # 2b. 工具描述要写成触发条件，不是功能陈述——同样是真机反馈
    d = {t["name"]: t["description"] for t in
         srv.handle({"jsonrpc": "2.0", "id": 21, "method": "tools/list"})["result"]["tools"]}
    assert "不是本次对话" in d["memory_search"] and "我不记得" in d["memory_search"], \
        "memory_search 描述要说清记忆类型并堵死默认话术"
    assert "主动" in d["session_start"] and "主动" in d["thread_close"], \
        "两个生命周期工具要写明主动调用，不用等对方要求"
    #    initialized 是通知，不能回响应（回了客户端会当成野生响应）
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None, \
        "initialized 是通知，回响应会让客户端收到一条没人等的野生响应"

    # 2. tools/list：三个工具，schema 字段名照规格（name/inputSchema）
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert [t["name"] for t in tools] == ["memory_search", "session_start",
                                          "memory_append", "memory_correct",
                                          "thread_close"]
    for t in tools:
        assert set(t) >= {"name", "description", "inputSchema"} and t["inputSchema"]["type"] == "object"
    assert tools[4]["inputSchema"]["required"] == ["window", "current_state"], "当下状态必填要写进 schema"
    assert tools[3]["inputSchema"]["required"] == ["quote", "reason"], \
        "更正工具必填 quote+reason——没有原因的撤回不可追溯"
    assert tools[2]["inputSchema"]["required"] == ["text", "current_state"], \
        "写回的当下状态必填也要写进 schema（病灶迁移在写入口强制）"

    # 3. tools/call 正常往返：结果结构照规格（content 数组 + type:text + isError）
    def call(name, args=None, mid=9):
        return srv.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                           "params": {"name": name, "arguments": args or {}}}, now=now)
    res = call("memory_search", {"query": "咖啡机坏了"})["result"]
    assert res["isError"] is False and res["content"][0]["type"] == "text"
    assert "保险丝熔断" in res["content"][0]["text"]

    # 4.【变异靶心·薄适配层】外壳输出必须逐字等于底层库返回——谁在适配层里重新
    #    实现格式化，这条立刻红。这正是"分得清是接口问题还是底层库问题"的保证
    idx2 = _build_server(now).index
    expected_search = format_recall_block(idx2.retrieve("咖啡机坏了", topN=5))
    assert call("memory_search", {"query": "咖啡机坏了"}, mid=10)["result"]["content"][0]["text"] \
        == expected_search, "memory_search 必须原样返回 format_recall_block 的结果"
    srv2 = _build_server(now)
    expected_start = srv2.recall.on_session_start(now=now)
    assert srv2.handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                        "params": {"name": "session_start"}}, now=now
                       )["result"]["content"][0]["text"] == expected_start, \
        "session_start 必须原样返回 on_session_start 的结果"

    # 5.【变异靶心·错误分层】未知工具→协议错误；工具内部失败→isError 结果
    unknown = call("no_such_tool")
    assert "error" in unknown and "result" not in unknown, \
        f"未知工具是协议错误、不是 isError 结果：{unknown}"
    assert unknown["error"]["code"] == E_METHOD_NOT_FOUND
    empty_srv = MemoryServer(index=MemoryIndex().build())
    r5 = empty_srv.handle({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                           "params": {"name": "session_start"}}, now=now)
    assert "result" in r5 and r5["result"]["isError"] is True, \
        f"工具执行失败该回 isError 结果而不是协议错误——模型要看得到失败原因：{r5}"
    #    参数非法（缺 query）同样走 isError，不是崩
    assert call("memory_search", {})["result"]["isError"] is True
    #    arguments 不是对象 → 协议错误
    bad = srv.handle({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                      "params": {"name": "memory_search", "arguments": "字符串"}})
    assert bad["error"]["code"] == E_INVALID_PARAMS

    # 6. thread_close 真写进 store，且下一次 session_start 能带回来（一条完整链路）
    ok = call("thread_close", {"window": 7, "current_state": "花买好了，周末的事没定。",
                               "topics": ["阳台的花"], "open_loops": ["周末去哪还没定"],
                               "started_at": now - 3600})
    assert ok["result"]["isError"] is False and "第 7 个窗口" in ok["result"]["content"][0]["text"]
    assert srv.thread_store.latest().window == 7
    started = call("session_start")["result"]["content"][0]["text"]
    assert started.startswith("【上次会话】") and "周末去哪还没定" in started, \
        "thread_close 写的东西该被下一次 session_start 带回来"
    #    当下状态为空 → 业务校验拦下，走 isError（病灶迁移纪律穿透到协议层）
    assert call("thread_close", {"window": 8, "current_state": "  "})["result"]["isError"] is True

    # 7. 未知 method 走协议错误；未知通知（无 id）静默忽略，不回野生错误
    assert srv.handle({"jsonrpc": "2.0", "id": 14, "method": "resources/list"}
                      )["error"]["code"] == E_METHOD_NOT_FOUND
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None

    # 8. stdio 传输往返：坏行跳过不崩，好行逐条回
    import io
    srv3 = _build_server(now)
    inp = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
                      '这不是 json\n'
                      '\n'
                      '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
                      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
    out = io.StringIO()
    srv3.serve_stdio(stdin=inp, stdout=out)
    lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
    assert [m["id"] for m in lines] == [1, 2], f"坏行该跳过、通知不回响应：{lines}"

    # 9.【变异靶心：stdio 必须按 UTF-8 收发】2026.07.31 真机实测出来的 bug——
    #    Windows 上 sys.stdin 按系统区域编码（简中 cp936）解码，中文 query 变乱码，
    #    分词匹配不上 → 分数全平 → 检索退化成"按加载顺序返回前几块"，不报错不崩，
    #    看起来像检索质量差，实际上根本没查。这里用真实 UTF-8 字节流走一遍全程
    assert _utf8_text_stream(io.BytesIO(b"")).encoding == "utf-8", "stdio 必须锁死 UTF-8"
    srv4 = _build_server(now)
    payload = ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
               '{"name":"memory_search","arguments":{"query":"薄荷"}}}\n')
    out4 = io.BytesIO()
    #    两个包装流都要留引用：被 GC 掉时 TextIOWrapper 会顺手关掉底层 BytesIO
    in_s = _utf8_text_stream(io.BytesIO(payload.encode("utf-8")))
    out_s = _utf8_text_stream(out4, write=True)
    srv4.serve_stdio(stdin=in_s, stdout=out_s)
    out_s.flush()
    got = json.loads(out4.getvalue().decode("utf-8"))
    assert got["result"]["isError"] is False and "薄荷" in got["result"]["content"][0]["text"], \
        f"中文 query 该原样穿过 stdio 并命中，实际 {got}"

    # 9.【变异靶心：写回当场可查 + 没落点明确拒写】memory_append 是记忆库自己
    #    生长的那半支笔（thread 是会话状态层，这是正文层）
    import tempfile
    from pathlib import Path as _P
    call = lambda s, name, a, n: s.handle(
        {"jsonrpc": "2.0", "id": 77, "method": "tools/call",
         "params": {"name": name, "arguments": a}}, now=n)["result"]
    #    没配语料目录：明确 isError，不静默写进内存了事（内存态的"记住了"会随
    #    进程一起死，那是"失败得像成功"）
    r9 = call(srv, "memory_append",
              {"text": "x", "current_state": "y"}, now)
    assert r9["isError"] is True and "语料目录" in r9["content"][0]["text"]
    with tempfile.TemporaryDirectory() as td:
        srv9 = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore(),
                            corpus_dir=td, weights_path=_P(td) / ".weights.json")
        #    缺当下状态 → isError（病灶迁移在写入口强制），且没有文件落盘
        bad = call(srv9, "memory_append", {"text": "她说了件事"}, now)
        assert bad["isError"] is True and "当下状态" in bad["content"][0]["text"]
        assert not list(_P(td).glob("**/*.md")), "被拒的写回不该留下文件"
        #    正常写回 → 落盘 + 本会话立刻可查（不重建索引就查不到）
        ok9 = call(srv9, "memory_append",
                   {"text": "约好周末去看鲸头鹳，她念叨了一个月。",
                    "current_state": "约定成立，票还没买。"}, now)
        assert ok9["isError"] is False and "第 1 个窗口" in ok9["content"][0]["text"]
        hit = call(srv9, "memory_search", {"query": "鲸头鹳"}, now)
        assert hit["isError"] is False and "约定成立" in hit["content"][0]["text"], \
            "写回之后当场就要能查到——不然模型刚说'记下了'转头就'没有记录'"

    # 10.【变异靶心：用进撑过重启】权重持久化接线——同一份语料+权重文件，
    #     新起一个 server（模拟客户端重启 stdio 进程），命中过的块权重仍在
    with tempfile.TemporaryDirectory() as td:
        wp = _P(td) / ".weights.json"
        mk = lambda: MemoryServer(index=_build_server(now).index,
                                  thread_store=ThreadStore(),
                                  corpus_dir=td, weights_path=wp)
        s_a = mk()
        call(s_a, "memory_search", {"query": "薄荷"}, now)     # 命中 → 权重落盘
        s_b = mk()                                             # "重启"
        i = next(i for i, c in enumerate(s_b.index.chunks) if "薄荷" in c)
        assert s_b.index.weights[i] > 1.0, \
            "重启后命中过的块权重该还在——不落盘的话用进废退在生产形态下等于没有"

    # 11.【可靠命中门槛】零信号 query → isError + 明确"没有可靠命中"，且文案要
    #     解锁如实回答（instructions 堵的是"没查就说没记录"，查过之后的"没有"
    #     是诚实——两句话必须能共存，不然模型会在两条指令之间僵住）
    miss = call(_build_server(now), "memory_search", {"query": "量子对撞机的运行日志"}, now)
    assert miss["isError"] is True and "没有可靠命中" in miss["content"][0]["text"]
    assert "如实" in miss["content"][0]["text"], "没找到时要明确解锁'如实说没有'"

    # 12.【错误记忆治理闭环：撤回→更正→重启仍生效】
    with tempfile.TemporaryDirectory() as td:
        rp = _P(td) / ".retractions.json"
        mk12 = lambda extra: MemoryServer(
            index=extra, thread_store=ThreadStore(),
            corpus_dir=td, weights_path=_P(td) / ".weights.json",
            retractions_path=rp)
        s12 = mk12(_build_server(now).index)
        #    写 correction 但缺当下状态 → 拒（病灶迁移，更正也不豁免）
        bad12 = call(s12, "memory_correct",
                     {"quote": "保险丝熔断", "reason": "x", "correction": "y"}, now)
        assert bad12["isError"] is True and "当下状态" in bad12["content"][0]["text"]
        #    quote 定位不到 → 明确报错，不猜
        miss12 = call(s12, "memory_correct", {"quote": "烤箱", "reason": "x"}, now)
        assert miss12["isError"] is True
        #    正常更正：撤回旧记录 + 写入更正 → 旧的查不到、新的当场可查
        ok12 = call(s12, "memory_correct",
                    {"quote": "保险丝熔断", "reason": "维修方案已过时",
                     "correction": "咖啡机上月整机换新了，旧机的维修记录不再适用。",
                     "current_state": "新机运行正常。"}, now)
        assert ok12["isError"] is False and "已撤回" in ok12["content"][0]["text"]
        after = call(s12, "memory_search", {"query": "咖啡机"}, now)
        assert after["isError"] is False
        assert "保险丝熔断" not in after["content"][0]["text"] and \
               "整机换新" in after["content"][0]["text"], "旧的退出检索、更正当场可查"
        #    【变异靶心：撤回落盘】"重启"（语料从盘上重读 + 账本重载）后撤回仍生效
        s12b = mk12(load_corpus(td))     # 更正记录在盘上；合成旧块不在盘上没关系
        assert s12b.index.retraction_log, "重启后撤回账本该从盘上回来"
        assert json.loads(rp.read_text(encoding="utf-8")), "账本文件要真在盘上（可追溯）"

    # 13.【图谱实体可插拔·接线靶心（变异：__init__ 不接 entities_path 必红）】
    #     语料目录下有 .entities.json 时 server 要接上并重建——换了说法的关联块
    #     经图谱进结果
    from memory_retrieval import _chunk_key
    def mk13():
        i = MemoryIndex()
        i.add("## 山顶的约定\n那晚在山顶聊到以后，说好要买一台能看土星的家伙。")
        i.add("## 到货\n快递终于送来了，装在阳台，晚上迫不及待试了试。")
        return i.build()
    with tempfile.TemporaryDirectory() as td:
        ep = _P(td) / ".entities.json"
        idx13 = mk13()
        ep.write_text(json.dumps({_chunk_key(idx13.chunks[0]): ["望远镜"],
                                  _chunk_key(idx13.chunks[1]): ["望远镜"]},
                                 ensure_ascii=False), encoding="utf-8")
        s13 = MemoryServer(index=mk13(), thread_store=ThreadStore(), entities_path=ep)
        r13 = call(s13, "memory_search", {"query": "山顶 土星"}, now)
        assert r13["isError"] is False and "阳台" in r13["content"][0]["text"], \
            "server 接上 .entities.json 后，换了说法的关联块该被图谱带回"

    print("selftest ok（14项断言：握手 / 工具表 / 调用往返 / 薄适配层 / 错误分层 / "
          "完整链路 / stdio / UTF-8 / 写回当场可查 / 用进撑过重启 / "
          "无可靠命中明确说 / 撤回更正闭环 / 实体标注接线）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--threads", help="会话线索 jsonl 路径（省略则内存态）")
    ap.add_argument("--embed", action="store_true", help="用真 embedding（需 fastembed）")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.corpus:
        # 权重文件放语料目录下，起点号不带 .md——不会被 load_corpus 当语料吃进去
        MemoryServer(index=load_corpus(args.corpus, embed=args.embed),
                     thread_store=ThreadStore(args.threads),
                     corpus_dir=args.corpus,
                     weights_path=Path(args.corpus) / ".weights.json",
                     retractions_path=Path(args.corpus) / ".retractions.json",
                     entities_path=Path(args.corpus) / ".entities.json").serve_stdio()
    else:
        ap.print_help()
