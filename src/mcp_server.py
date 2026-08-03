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

**成色（2026.08.03 改准，此前写的是“尚未与任何真实客户端握手核对”，已过时）**：
已有一条真实客户端链路——Operit（Android）＋ proot Ubuntu，客户端按 stdio 拉起本进程，
握手、工具列表、真实调用三样都走过：五个工具全部接通，`memory_append` 写入后
**新开会话仍检索命中**。外部实测转录，本项目未复现。

⚠ **剩下的边界仍然作数，别读成“都验过了”**：Claude 桌面 App 那一格仍未连过；
超长文本怎么截断、参数容忍度这些真机问题只在一家客户端上过过手。
另有三条与宿主拉起有关的实测事实记在《快速上手》§3c「部署形态一」：
stdio 只能由宿主拉起（手动 `nohup` 必崩）、懒加载每次调用重读语料、
客户端的 `lastStartTime` 不能当跑通判据。

工具集一一对应现成能力，不新造：
  memory_search  → MemoryIndex.retrieve
  session_start  → SessionRecall.on_session_start（thread 块 + 召回块 + 自查指令）
  memory_append  → memory_retrieval.append_record（正文层的笔）
  memory_correct → MemoryIndex.retract（+ 可选 append_record 写更正）
  thread_close   → session_thread.close_thread + ThreadStore.append

用法：
  python mcp_server.py --selftest
  python mcp_server.py --corpus <md目录> [--threads <threads.jsonl>]   # stdio 服务
  python mcp_server.py --doctor --corpus <md目录> [--threads <threads.jsonl>]  # 部署体检
客户端配置（Claude Desktop 之类）里把上面第二条命令填成 server 启动命令即可；
配完接不上、或者不确定 --corpus 指对了没有时，把同一行参数换成 --doctor 跑一次
（体检只读，不往语料目录写任何东西）。
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 同目录模块，import 不触发各自的 CLI
from memory_retrieval import (MemoryIndex, load_corpus, append_record, corpus_files,
                              query_miss_rate, miss_rate_note, annotate_block,
                              _chunk_key)
from embedding_provider import resolve_provider
# 切块下界与体检的切块成色检查**共用同一个判据**，不各抄一份
from chunking_experiment import chunk_body as _chunk_body, chunk_heading
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
        note = miss_rate_note(query_miss_rate(self.index, query))
        if not results:
            # 可靠命中门槛（验收反馈）：低相关不硬凑。这句要同时做两件事——
            # 说清"查过了、真没有"，并明确解锁如实回答（instructions 堵的是
            # "没查就说没记录"，查过之后的"没有"是诚实，不是那句被堵的话术）
            raise ToolError("没有可靠命中：记忆库里没有与这个说法词面或语义相关的"
                            "记录。你已经查过了——如实告诉对方没找到/记不清即可；"
                            "也可以换个说法再查一次（人名、地点、当时的用词）。"
                            + (" " + note if note else ""))
        # 缺失率标注（2026.08.01，第二份外部反馈标定）：**不改变返回什么**，
        # 只在结果后面附一句可核对的话。真实威胁是"库里没有却返回了五条真记忆、
        # 模型拿去圆"，而这个判断只有读到内容的模型能下——机制层负责把不确定性
        # 摆到台面上，不负责替它拒绝。
        # 拼接走 annotate_block（库函数），外壳仍然只是转发+组合调用，不自己拼字符串
        return annotate_block(format_recall_block(results),
                              query_miss_rate(self.index, query))

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
        # 撤回先做——它同时是 quote 的校验关卡；quote 定位不到就该在写任何东西
        # 之前失败（否则更正落了盘、旧记录还在，比什么都没做更糟）
        try:
            old_idx, _ = self.index.retract(args.get("quote") or "",
                                            args.get("reason") or "", now=now)
        except ValueError as e:
            raise ToolError(str(e))
        msg = "已撤回那段旧记录：检索不会再返回它（原文件保留，撤回原因入账可追溯）。"
        if correction:
            if self.corpus_dir is None:
                # 撤回已生效但更正写不进去——明确报出来，不静默丢（同 append 的理由）
                if self.retractions_path is not None:
                    self.index.save_retractions(self.retractions_path)
                raise ToolError(msg + " 但服务器没有配置可写的语料目录（--corpus），"
                                "更正内容写不进去，请提醒用户检查 MCP 配置。")
            try:
                path, chunk_text, meta = append_record(
                    self.corpus_dir, f"【更正】{correction}",
                    args.get("current_state") or "", now=now)
            except (ValueError, OSError) as e:
                if self.retractions_path is not None:
                    self.index.save_retractions(self.retractions_path)
                raise ToolError(msg + f" 但更正内容写入失败：{e}")
            self.index.add(chunk_text, meta)
            self.index.build()
            # 追溯链补上：让账本能回答"哪条记录改了哪条"，不只是"这条被撤了"。
            # 只能在更正写完之后回填——新块的内容哈希这时才存在
            self.index.link_correction(old_idx, chunk_text)
            msg += f" 更正已写进第 {meta['window']} 个窗口（{path.name}）。"
        if self.retractions_path is not None:
            self.index.save_retractions(self.retractions_path)
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


# ---------- 部署体检（任务卡"部署体检命令"） ----------
#
# 挂在 mcp_server.py 上、不另开脚本文件：配 MCP 的人手里已经有这个文件的绝对路径
# （`claude mcp add` 那行就是它），体检命令跟着它走，等于零新增路径要记。
#
# **只读是硬约束，不是习惯**：体检最可能被在"看起来没接通"的时候跑，那时候人对
# 目录状态的信任最脆弱；跑一次体检往语料目录里掉一个 `.weights.json`，等于在
# 排查故障的现场留下一个新变量。所以这里一个字节都不写盘——包括 sidecar，
# 包括 embed 档的块向量缓存。这条由 selftest 的目录快照断言守着（第 14 项）。

OK, WARN, FAIL = "ok", "warn", "fail"
_DOCTOR_ICON = {OK: "✓", WARN: "⚠", FAIL: "✗"}

# 不该出现在语料目录里的 md：它们是产出目录那一层的东西。撞见就说明 --corpus
# 多指了一层（指到了产出目录本身，而不是里面的记忆库）——人格文件会被当成记忆
# 吃进库里，检索结果里冒出自己的人格设定，但不报任何错
_NOT_CORPUS_MD = {"claude.md", "agents.md", "persona.md", "readme.md",
                  "注入契约.md", "index_readme.md"}
# mtime 兜底占比超过这条线就报警：时间戳全落 mtime 时换窗召回的新鲜度排序整个失效
_MTIME_WARN_RATIO = 0.2
# 「只有标题行、没有正文」的块占到这个比例就报警——兜的是"第三方导出把每句消息
# 前面加 `##`"那类输入（见《导出格式把每句话切成一块》任务卡）。`chunk_heading`
# 有上界没有下界，这种语料建库成功、块数报得出来、检索也跑得动，只是每个块都是
# 一句话，**静默地没有检索价值**，还顺带把图谱那条平方曲线推过 MCP 启动超时。
#
# 判据**刻意不选"块长中位数低于 N 字"**：那是要拿真实语料标定的经验值（同
# MILESTONE_BODY_LIMIT 的教训），手上只有"正常"和"病态"两个极端、中间没有样本，
# 定不出来。"标题行就是整个块"是结构判据，不用标定：按小节切出来的块本该有正文。
# 这条线取"过半"是语义边界不是调出来的数——本仓三份真实 md 语料量到 0% / 3.6%
# / 11.6%，每句 `##` 的导出形态是 100%，两边离这条线都远得很。
_HEADING_ONLY_WARN_RATIO = 0.5


def diagnose(corpus_dir, threads_path=None, embed=False):
    """体检语料目录与接线，返回 [{level,title,detail}, ...]。**纯读，不写盘。**

    检的是"配好了没有"，不是"检索好不好"——后者归回归集（regression_set.py）。
    每一条都对应一种**不报错的失败**：指错目录、人格文件混进语料、时间戳全落
    mtime、sidecar 哈希对不上、写回落点只读、thread 没落点。这些的共同点是
    服务照常起、握手照常成功、模型照常回话，只是回得不对。"""
    out = []
    def add(level, title, detail):
        out.append({"level": level, "title": title, "detail": detail})

    # **相对路径必须解析成绝对路径**（判据 1，也是这一单的来源）：`.mcp.json` 里写的
    # 是 `--corpus corpus` 这种相对值，跑体检的人心里的问题正是"它到底去哪儿读了"。
    # 把他自己写的那个词原样回显给他，等于一个字没答——报告里必须出现真实绝对路径。
    # ⚠ 别把 .resolve() 去掉（selftest 第 15 项走真进程守着这条）
    root = Path(corpus_dir).resolve()
    if not root.exists():
        add(FAIL, "语料目录", f"{root} 不存在。服务端认的是 --corpus 传进去的这个路径，"
                              "跟目录叫什么名字无关——先确认 MCP 配置里那行路径写对了没有。")
        return out
    if not root.is_dir():
        add(FAIL, "语料目录", f"{root} 不是目录。--corpus 要指向记忆库那一层目录，不是单个文件。")
        return out

    files = corpus_files(root)          # 递归，子目录里的 md 也算
    if not files:
        # 典型是指到了 src/ 或路径写岔了一格。给候选时**只看目录里有没有 md，
        # 不看目录叫什么名字**——外层目录名本来就是自由的（叫 corpus/、我的记忆/
        # 都一样读得到），拿名字猜就是在教人一个错的判据
        near = [d for d in sorted(root.parent.iterdir())
                if d.is_dir() and d != root and corpus_files(d)] if root.parent.exists() else []
        hint = (f"同级的这些目录里有 md，你要指的多半是其中之一："
                f"{'、'.join(d.name for d in near[:5])}。" if near
                else "它的同级目录里也没有——确认一下记忆库到底建在哪儿。")
        add(FAIL, "语料文件", f"{root} 下（含子目录）没找到任何 .md 语料。{hint}")
        return out
    add(OK, "语料目录", f"{root}（{len(files)} 个 md 文件）")

    stray = [p.name for p in files if p.name.lower() in _NOT_CORPUS_MD]
    if stray:
        add(WARN, "混进来的文件",
            f"语料里有 {'、'.join(sorted(set(stray)))}——这些是产出目录那一层的文件，"
            "不是记忆。多半是 --corpus 多指了一层（应指向里面的记忆库目录）。"
            "它们会被当成记忆检索到，但不会报任何错。")

    index = load_corpus(root)          # embed=False：这一档不写任何缓存文件
    if not index.chunks:
        # "有 md、但一块都切不出来"是上面那道 `not files` 关卡挡不住的一档：
        # 文件全是空行/只有空白的话，语料目录看着满满当当，库却是空的，
        # 服务照样起、每次检索都空手。这条要显著地说，别让它一路往下崩在除法上
        add(FAIL, "建库", f"{len(files)} 个 md 文件里一块内容都切不出来——"
                          "文件是空的或只有空白行。这样的库起得来但查不到任何东西，"
                          "每次检索都会空手。先确认语料是不是真写进去了。")
        return out
    n_index = sum(1 for m in index.meta if m.get("layer") == "index")
    lens = sorted(len(c) for c in index.chunks)
    median_len = lens[len(lens) // 2]
    add(OK, "建库", f"{len(index.chunks)} 块（索引层 {n_index} / 叙事层 "
                    f"{len(index.chunks) - n_index}），中位块长 {median_len} 字")

    # 切块成色：块数漂亮、分层漂亮，但每个块只有一句话——体检以前只报块数，
    # 报告里那个虚高一个数量级的数字**没有任何一行说它不正常**（首发用户就是这么
    # 错过它的）。只加诊断、不改切块行为
    # ⚠ 这里必须看**合并前**的形态。2026.08.03 切块下界落地后，无正文块在建库阶段
    # 就被向上合并掉了，拿 index.chunks 数永远是 0——**两条防线会互相吃掉**，
    # 而被吃掉的那条恰恰是唯一会告诉用户"你的导出有毛病"的那条。修法治的是症状，
    # 诊断仍要照直说病因：导出该改，不然每次建库都要靠合并兜。
    raw = []
    for f in files:
        try:
            t = Path(f).read_text(encoding="utf-8")
        except OSError:
            continue          # 实际不可达（load_corpus 在前面先读过一遍），防御性保留
        raw += chunk_heading(t, merge_bodyless=False)
    n_heading_only = sum(1 for c in raw if not _chunk_body(c))
    # 有文件读不出来时 raw 会比实际建库的块少，差值可能变负——报负数比不报更糊涂，
    # 钳到 0（合并只会让块变少，负数在语义上不存在）
    merged_n = max(0, len(raw) - len(index.chunks))
    if raw and n_heading_only > len(raw) * _HEADING_ONLY_WARN_RATIO:
        add(WARN, "切块", f"切块前 {n_heading_only}/{len(raw)} 块只有一行标题、没有正文"
            f"——已在建库时合并 {merged_n} 块，现在是 {len(index.chunks)} 块、"
            f"中位块长 {median_len} 字。**库是能用了，但语料本身该改**：去看一眼是不是"
            "**每句话前面都有 `##`**（有的第三方聊天导出插件这么干）。切块按 `## ` 认"
            "小节，所以一句话就成了一个块。合并只是兜底，导出后把这些消息前缀去掉再"
            "建库，块的边界才落在你真正的话题上。")

    # 时间范围（判据 3）：兜的是《快速上手》第 0 步那个坑的**部署侧版本**——
    # 旧导出包建出来的库，条数漂亮、块数漂亮、什么都不报错，只是**整份停在了
    # 过去某一天**。`memory_import.py --stats` 只在导入那一刻能发现它；导完之后，
    # 这里是唯一还会把这个数摆到人眼前的地方。只读 index.meta 里现成的 timestamp，
    # 不开任何文件句柄
    ts = [m["timestamp"] for m in index.meta if m.get("timestamp")]
    if ts:
        span = (f"{datetime.fromtimestamp(min(ts)):%Y-%m-%d} ~ "
                f"{datetime.fromtimestamp(max(ts)):%Y-%m-%d}")
        add(OK, "时间范围", f"{span}——盯住后面那个日期，问自己一句：跟 TA 最近一次"
                            "聊天真的是这天吗？对不上说明这份语料是旧快照，"
                            "重新导一份再建库（数字再健康也救不了停在过去的语料）。")
    else:
        add(WARN, "时间范围", "一块都没有时间戳，算不出时间范围——"
                              "换窗召回按时间新鲜度排序，这种情况下它没有任何排序依据。")
    if n_index == 0:
        add(WARN, "分层", "没有索引层（父目录名叫 index 的才算）。不算错，但命中率会低一档："
                          "索引层是每会话一条高密度摘要，专门喂检索。")

    # 时间戳成色：mtime 兜底不是"不太准"，是**方向错误**——复制目录/重新 clone 会把
    # 全目录 mtime 刷成"刚刚"，于是最没有时间依据的块拿到全库最新的日期，
    # 换窗召回按新鲜度排序时**最旧的事实冒充最新的**（2026.08.03 跨模型实测：
    # 这么一份语料下两家模型 0/6 答对当下状态）。
    srcs = {}
    for m in index.meta:
        srcs[m.get("timestamp_source")] = srcs.get(m.get("timestamp_source"), 0) + 1
    detail = "、".join(f"{k} {v} 块" for k, v in sorted(srcs.items(), key=lambda x: -x[1]))
    n_mtime = srcs.get("mtime", 0)
    ratio = n_mtime / len(index.chunks)
    # 判据二（2026.08.03 补，**与比例无关**）：只要有 mtime 档的块比全库最新的
    # 真日期块还"新"，就报——**危险的恰恰是混合且占比低的情况**：绝大多数块有日期、
    # 少数几块落 mtime，那几块的时间戳就是"今天"，稳稳压过所有真日期，
    # 而占比 5% 连比例判据的门都够不着，于是静默。比例判据看的是"有多少块没日期"，
    # 这条看的是"没日期的块会不会骑到有日期的头上"，两件事。
    real_ts = [m["timestamp"] for m in index.meta
               if m.get("timestamp") is not None and m.get("timestamp_source") != "mtime"]
    mtime_ts = [m["timestamp"] for m in index.meta
                if m.get("timestamp") is not None and m.get("timestamp_source") == "mtime"]
    outranks = bool(real_ts) and bool(mtime_ts) and max(mtime_ts) > max(real_ts)
    if ratio > _MTIME_WARN_RATIO or outranks:
        bad = sorted({m["source"] for m in index.meta
                      if m.get("timestamp_source") == "mtime"})
        add(WARN, "时间戳来源",
            f"{detail}——{ratio:.0%} 的块只能退到文件修改时间。它不是“不太准”，"
            "**是方向错误**：复制一遍目录或重新 clone，这些块的时间会被刷成“刚刚”，"
            "于是最没有日期依据的内容拿到全库最新的时间戳，"
            "**换窗召回会把最旧的事实当成你现在的状态端出来**"
            + ("（**这份语料已经是这种形态**：落 mtime 的块比所有带真日期的块都新）"
               if outranks else "")
            + "。修法是把日期写进文件名"
            f"（window_04_2026-06-17.md 这种）或标题行。落 mtime 的文件："
            f"{'、'.join(bad[:5])}{' 等' if len(bad) > 5 else ''}")
    else:
        add(OK, "时间戳来源", detail)
    if index.date_order:
        add(OK, "日期顺序", f"语料里的 m/d/y 型日期按 {index.date_order} 解析（有决定性证据）")

    # sidecar：三个都是可选的，**没有不是错**；有、但一块都对不上才是错——
    # 那说明语料被编辑过或换过目录，哈希对不上号，等于这份 sidecar 静默失效了
    for name, loader, what in ((".retractions.json", index.load_retractions, "撤回账本"),
                               (".weights.json", index.load_weights, "命中权重"),
                               (".entities.json", index.load_entities, "实体标注")):
        p = root / name
        if not p.exists():
            add(OK, name, f"没有（正常：{what}第一次用到时才生成）")
            continue
        try:
            n = loader(p)
        except (ValueError, OSError) as e:
            add(FAIL, name, f"{what}读不出来：{e}")
            continue
        if n == 0:
            add(WARN, name, f"{what}在，但没有一条对得上当前语料——按内容哈希对号入座，"
                            "语料被编辑过或换了目录就会全部失效（文件还在，等于没有）。")
            continue
        # 孤儿条目（2026.08.03 外部缺陷报告逼出来的一格）：账本条目对不上任何现存块
        # 就是死账——**部分孤儿比全孤儿隐蔽得多**：接上的那几条让上面那格报 OK，
        # 孤儿那条静默失效。撤回账本的孤儿尤其要命：撤回看着成功了、落盘了、
        # 体检全绿，重启后被撤回的旧说法照常召回。此前 append 手拼块文本与
        # 重启重切不一致就是这么漏网的（那个根因已修，这格防的是同形状的下一个）。
        cur_keys = {_chunk_key(c) for c in index.chunks}
        orphans = [k for k in json.loads(p.read_text(encoding="utf-8"))
                   if k not in cur_keys]
        if orphans:
            add(WARN, name, f"{what}接上 {n} 块，但有 {len(orphans)} 条孤儿条目"
                            f"对不上任何现存块（键：{'、'.join(orphans[:3])}"
                            f"{' 等' if len(orphans) > 3 else ''}）——这些账等于已静默"
                            f"失效；若是撤回账本，对应的旧记录会照常被召回。多半是"
                            f"块正文被编辑过、或旧版本手拼块文本留下的死账。")
        else:
            add(OK, name, f"{what}接上 {n} 块，无孤儿条目")
    index.build()                       # 实体边在 build 时算，接上后要重建一次

    # 写回落点：memory_append/memory_correct 要往这里写。只用 os.access 判，
    # 不试写——试写就破了只读
    if os.access(root, os.W_OK):
        # 落点要报到层：写回永远进 timeline 层（append_record 构造上锁死的），
        # 人格文件里「按需读取指针」必须盖住这个目录——用户拿这行对自己的指针，
        # 指漏了的症状是"新长出来的记忆按需读不到"，而且不报错
        add(OK, "写回落点", f"{root}/timeline 可写（memory_append 按窗口号+日期"
            f"往这里加文件；人格文件的「按需读取指针」要盖住这个目录）")
    else:
        add(FAIL, "写回落点", f"{root} 不可写——模型会说“记下了”，但每一次写回都失败。")

    # thread 落点：没配 --threads 时 thread_close 只活在内存里，进程一退就没了，
    # 下个会话的开场召回接不上上一次聊到哪
    if not threads_path:
        add(WARN, "会话线索", "没配 --threads，会话收尾只在内存里、进程一退就没——"
                              "下个会话接不上“上次聊到哪”。MCP 配置里补一个 jsonl 路径。")
    else:
        tp = Path(threads_path)
        if not tp.exists():
            writable = tp.parent.exists() and os.access(tp.parent, os.W_OK)
            add(OK if writable else FAIL, "会话线索",
                f"{tp} 还没有（第一次 thread_close 时创建）"
                + ("" if writable else "，但它的父目录不存在或不可写，创建会失败。"))
        else:
            try:
                threads = ThreadStore(tp).all()
            except (ValueError, OSError) as e:
                add(FAIL, "会话线索", f"{tp} 读不出来：{e}")
                threads = None
            if threads is not None:
                latest = max(threads, key=lambda t: (t.window, t.ended_at)) if threads else None
                add(OK, "会话线索", f"{tp}：{len(threads)} 条"
                                    + (f"，最新是第 {latest.window} 个窗口" if latest else ""))

    if embed:
        # 明说不体检，而不是偷偷降级：embed 档下 load_corpus 会把块向量缓存
        # （.embed_cache.json）落在语料目录里，跑一次体检就落一个文件——跟只读冲突
        add(WARN, "检索路线", "体检只走零依赖档：embed 档建库会把块向量缓存"
                              "（.embed_cache.json）写进语料目录，跟“体检不落文件”冲突。"
                              "上面关于语料/时间戳/sidecar 的结论跟检索路线无关，照样作数；"
                              "embed 那一路通不通请直接起一次服务看。")

    # 接线本身：握手 + 工具表 + 一次真检索。前面全绿也可能死在这一步，
    # 而这是唯一一处能证明"这份语料真能被查到"的检查
    srv = MemoryServer(index=index, thread_store=ThreadStore(threads_path))
    #    ⚠ 故意不接 weights_path：接了的话下面这次检索会把权重落盘，体检就不再只读。
    #    要改这行之前先读 selftest 第 14 项——它就是守这个的。
    hs = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    add(OK, "MCP 接线", f"握手 {hs['result']['protocolVersion']}，工具 {len(tools)} 个："
                        + "、".join(t["name"] for t in tools))

    probe = _doctor_probe(index)
    res = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "memory_search",
                                 "arguments": {"query": probe}}})["result"]
    if res["isError"]:
        add(FAIL, "检索自查", f"拿语料自己的标题“{probe}”去查，反而查不到："
                              + res["content"][0]["text"].splitlines()[0])
    else:
        add(OK, "检索自查", f"拿语料自己的标题“{probe}”查得到")
    return out


def _doctor_probe(index):
    """从语料自己身上取一句探针 query——用最新那块的标题（没有标题就用首行）。
    拿库里确实存在的说法去查，查不到就说明接线坏了，而不是"这个问题库里没有"。"""
    i = max(range(len(index.meta)),
            key=lambda j: index.meta[j].get("timestamp") or 0)
    head = (index.meta[i].get("heading") or "").strip()
    if not head:
        head = next((ln.lstrip("# ").strip() for ln in index.chunks[i].splitlines()
                     if ln.strip()), "")
    return head[:30]


def format_doctor_report(checks):
    """体检结果 → 给人看的报告。结论一句话放最后，别让人自己数图标。"""
    lines = ["记忆库部署体检", ""]
    for c in checks:
        lines.append(f"{_DOCTOR_ICON[c['level']]} {c['title']}：{c['detail']}")
    n_fail = sum(1 for c in checks if c["level"] == FAIL)
    n_warn = sum(1 for c in checks if c["level"] == WARN)
    lines.append("")
    if n_fail:
        lines.append(f"结论：{n_fail} 项过不去" + (f"、{n_warn} 项要注意" if n_warn else "")
                     + "。上面标 ✗ 的先修，修完再起服务。")
    elif n_warn:
        lines.append(f"结论：能用，{n_warn} 项要注意——标 ⚠ 的都是"
                     "“不报错但会悄悄变差”的那类，值得看一眼。")
    else:
        lines.append("结论：全部通过。")
    lines.append("（体检只读，没有向语料目录写入任何文件。）")
    return "\n".join(lines)


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
    expected_search = annotate_block(
        format_recall_block(idx2.retrieve("咖啡机坏了", topN=5)),
        query_miss_rate(idx2, "咖啡机坏了"))
    #    2026.08.01 随缺失率标注放宽了一格，但**纪律没放松**：比对的仍是"底层库
    #    函数的组合结果"（annotate_block(format_recall_block(...), 缺失率)），
    #    外壳只要自己拼一个字都会红
    assert call("memory_search", {"query": "咖啡机坏了"}, mid=10)["result"]["content"][0]["text"] \
        == expected_search, "memory_search 必须原样返回底层库函数的组合结果，外壳不许自拼"
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

    # 12c.【同会话 append→撤回，重启仍生效·变异靶心】（2026.08.03 外部缺陷报告，
    #      真机 7232 块语料上抓到）：新建窗口文件的首块，手拼返回值与重启重切的
    #      文件态块不一致（H1 被向下并进首块），撤回账本键成孤儿，「已撤回」只活
    #      一个进程——触发的恰是最自然的用法：刚记完当场改。上面第 12 项盖不到
    #      这一格：它的旧块是合成的、不在盘上，只验账本回来了、验不了**重新对上号**。
    #      变异：append_record 改回手拼 chunk_text → 这段红
    with tempfile.TemporaryDirectory() as td:
        rp = _P(td) / ".retractions.json"
        mk12c = lambda: MemoryServer(index=load_corpus(td), thread_store=ThreadStore(),
                                     corpus_dir=td, retractions_path=rp)
        s_a12c = mk12c()
        call(s_a12c, "memory_append",
             {"text": "她想去的展览在下周三，先记着。", "current_state": "还没买票。"}, now)
        ok12c = call(s_a12c, "memory_correct",
                     {"quote": "下周三", "reason": "记错了，是下周四"}, now)
        assert ok12c["isError"] is False and "已撤回" in ok12c["content"][0]["text"]
        s_b12c = mk12c()                               # 重启：语料与账本都从盘上回来
        assert s_b12c.index.retracted, \
            "重启后撤回必须重新对上号（此前这里是空集：账本键是手拼块文本的孤儿）"
        r12c = call(s_b12c, "memory_search", {"query": "展览是哪天"}, now)
        assert "下周三" not in r12c["content"][0]["text"], \
            "被撤回的旧说法重启后不许再被召回——跟更正并排出现比撤回失败更糟"

    # 12b.【缺失率标注接线：两条路径都要带上】高缺失率查询无论"查到了"还是
    #      "没查到"，都该把那句可核对的话交给模型——空结果那条尤其重要，它是
    #      模型决定"如实说没找到"还是"拿沾边的记录去圆"的分水岭
    from memory_retrieval import query_miss_rate as _qmr0, MISS_RATE_FLAG as _F0
    srv12b = _build_server(now)
    hit12b = call(srv12b, "memory_search", {"query": "咖啡机 保险丝 熔断 通电"}, now)
    miss12b = call(srv12b, "memory_search", {"query": "量子对撞机的运行日志"}, now)
    assert miss12b["isError"] is True and "核对提示" in miss12b["content"][0]["text"], \
        "空结果那条必须带缺失率标注——它是'如实说没找到'与'拿沾边记录去圆'的分水岭"
    #      **最要紧的一条：高缺失率不许硬拒**。这是本信号定性（专名缺席检测器，
    #      不是事件存在性检测器）直接推出来的纪律——它的误杀全部落在"抽象归纳式
    #      提问"那一档，而那是陪伴场景最有价值的一类问题。有人把标注改成拒绝返回，
    #      行为上是"专门惩罚最该被答好的提问"，而在此之前没有任何断言守着这件事
    #      （变异检查抓出来的缺口）
    mixed = "咖啡机 量子对撞机 报税"          # 高缺失率，但确实有真命中
    assert _qmr0(srv12b.index, mixed) >= _F0, "测试前提：这条要真的触发标注"
    r_mixed = call(srv12b, "memory_search", {"query": mixed}, now)
    assert r_mixed["isError"] is False, \
        "高缺失率查询**不许硬拒**——它只该带标注，判断权留给读得到内容的模型"
    assert "保险丝" in r_mixed["content"][0]["text"], "真命中必须照常返回"
    assert "核对提示" in r_mixed["content"][0]["text"], "同时要带上那句可核对的标注"

    #      有结果时按缺失率决定带不带，不硬加噪声
    from memory_retrieval import query_miss_rate as _qmr, MISS_RATE_FLAG as _F
    if _qmr(srv12b.index, "咖啡机 保险丝 熔断 通电") < _F:
        assert "核对提示" not in hit12b["content"][0]["text"], "低缺失率不该加标注"

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

    # 14.【部署体检·靶心是"只读"】体检最常在"看起来没接通"的时候跑，那时候往
    #     语料目录里掉一个文件，等于在排查现场留下新变量。跑前跑后整棵目录树的
    #     快照（相对路径 + 大小 + mtime_ns）必须逐字相等——**顺手接一条
    #     weights_path 进 diagnose 里的 server，这条立刻红**，那是最容易被写出来的
    #     sidecar（检索命中就落盘）。
    #     靶子目录**故意叫 corpus/、不叫 memory/**：外层目录名是自由的，服务端认的
    #     是 --corpus 指向哪儿；判定必须靠目录内容，一旦有人拿名字做判断，这里就红
    def _snapshot(root):
        return {str(p.relative_to(root)): (p.is_dir(), p.stat().st_size if p.is_file() else 0,
                                           p.stat().st_mtime_ns)
                for p in sorted(_P(root).rglob("*"))}

    with tempfile.TemporaryDirectory() as td:
        corpus = _P(td) / "corpus"
        (corpus / "timeline").mkdir(parents=True)
        (corpus / "index").mkdir(parents=True)
        (corpus / "timeline" / "window_04_2026-06-17.md").write_text(
            "## 修咖啡机\n加热管不工作，拆开发现保险丝熔断，换上通电正常。\n",
            encoding="utf-8")
        (corpus / "index" / "window_04.md").write_text(
            "## 第4窗摘要\n修好了咖啡机。\n", encoding="utf-8")
        before = _snapshot(corpus)
        checks = diagnose(corpus, threads_path=_P(td) / "threads.jsonl")
        report = format_doctor_report(checks)
        assert _snapshot(corpus) == before, \
            "体检必须只读：跑完语料目录里多/少/改了东西（最常见的是 .weights.json）"
        assert not list(corpus.rglob(".*")), \
            f"体检不许留 sidecar：{[p.name for p in corpus.rglob('.*')]}"
        assert all(c["level"] != FAIL for c in checks), f"这份语料该全过：{checks}"
        by = {c["title"]: c for c in checks}
        assert "索引层 1" in by["建库"]["detail"], "index/ 那层要被认出来"
        assert by["时间戳来源"]["level"] == OK and "mtime" not in by["时间戳来源"]["detail"], \
            "文件名带日期 + 邻层继承，不该有块落 mtime"
        assert by["检索自查"]["level"] == OK, "拿语料自己的标题该查得到"
        assert "没有向语料目录写入任何文件" in report

        #    指错目录的三种典型都要给出**能照着改**的话，不是"失败"两个字
        gone = diagnose(_P(td) / "根本没有这个目录")
        assert gone[0]["level"] == FAIL and "--corpus" in gone[0]["detail"]
        #    指到了旁边一个没有语料的目录（典型是 src/）：报失败之外还要**按内容**
        #    把真正的候选找出来。这里的靶子目录叫 corpus/ 不叫 memory/，所以
        #    谁把候选判据写成"目录名叫 memory/"，这条立刻红
        (_P(td) / "src").mkdir()
        astray = diagnose(_P(td) / "src")
        assert astray[-1]["level"] == FAIL and "corpus" in astray[-1]["detail"], \
            f"该按内容指出“你要指的多半是 corpus/”：{astray[-1]}"
        #    指到了产出目录本身：人格文件被当成记忆吃进库，不报任何错——这条只能靠体检
        (corpus / "CLAUDE.md").write_text("# 人格文件\n你是……\n", encoding="utf-8")
        stray = {c["title"]: c for c in diagnose(corpus)}
        assert stray["混进来的文件"]["level"] == WARN and \
            "CLAUDE.md" in stray["混进来的文件"]["detail"], "人格文件混进语料要报出来"

    #     mtime 兜底的报警：文件名与正文都不带日期时，整批块拿到同一个假时间
    with tempfile.TemporaryDirectory() as td:
        c2 = _P(td) / "corpus"
        c2.mkdir()
        (c2 / "随手记.md").write_text("## 没写日期\n聊了点别的。\n", encoding="utf-8")
        m = {c["title"]: c for c in diagnose(c2)}
        assert m["时间戳来源"]["level"] == WARN and "mtime" in m["时间戳来源"]["detail"]
        assert m["会话线索"]["level"] == WARN, "没配 --threads 要提醒收尾不过夜"
        #     sidecar 在、但一块都对不上 = 静默失效（换过目录/改过语料），要报出来
        (c2 / ".weights.json").write_text('{"deadbeef": 2.0}', encoding="utf-8")
        w = {c["title"]: c for c in diagnose(c2)}
        assert w[".weights.json"]["level"] == WARN and "对得上" in w[".weights.json"]["detail"]
        #     部分孤儿比全孤儿隐蔽得多（2026.08.03 外部缺陷报告逼出来的一格）：
        #     接上的那几条让这格报 OK，孤儿那条静默失效——若是撤回账本，被撤回的
        #     旧说法照常召回。变异：把孤儿检查删掉（n>0 直接报 OK）→ 这条红
        from memory_retrieval import _chunk_key as _ck14
        good_key = _ck14(load_corpus(c2).chunks[0])
        (c2 / ".weights.json").write_text(
            json.dumps({good_key: 2.0, "deadbeef00000000": 3.0}), encoding="utf-8")
        o14 = {c["title"]: c for c in diagnose(c2)}
        assert o14[".weights.json"]["level"] == WARN and "孤儿" in o14[".weights.json"]["detail"], \
            "账本部分对得上时，孤儿条目必须点名报出来——OK 不许挡住死账"

    # 15.【部署体检·走真进程，从相对路径 cwd 起】上面第 14 项全是函数级断言，
    #     它有两个够不着的地方，而返工的三条缺陷恰好都藏在那里：
    #       ① 函数级断言喂的是 tempfile 给的**绝对**路径，于是"相对路径有没有被
    #          解析开"永远测不到——而 `.mcp.json` 里写的正是 `--corpus corpus`
    #          这种相对值，这一单的来源就是它；
    #       ② `__main__` 那段分派（缺 --corpus 的报错、按 FAIL 决定的退出码）
    #          一条断言都盖不到，全在裸奔。
    #     所以这一项起真进程、传相对值、断言 stdout 与退出码。
    #     **断的是 str(corpus.resolve()) 在不在输出里，不是字符串 "corpus" 在不在**
    #     ——后者用户本来就知道，回显给他等于一个字没答。
    #     变异：去掉 diagnose 里的 .resolve() / 把块数写死 / 空目录也报成功 → 各自红
    import subprocess
    here = _P(__file__).resolve().parent

    def run_doctor(cwd, *argv):
        p = subprocess.run([sys.executable, str(here / "mcp_server.py"), "--doctor", *argv],
                           cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
        return p.returncode, p.stdout + p.stderr

    with tempfile.TemporaryDirectory() as td:
        corpus = _P(td) / "corpus"
        (corpus / "timeline").mkdir(parents=True)
        (corpus / "index").mkdir(parents=True)
        (corpus / "timeline" / "window_04_2026-06-17.md").write_text(
            "## 修咖啡机\n加热管不工作，拆开发现保险丝熔断。\n", encoding="utf-8")
        (corpus / "index" / "window_04.md").write_text(
            "## 第4窗摘要\n修好了咖啡机。\n", encoding="utf-8")
        before = _snapshot(corpus)
        #    从父目录起、传相对值——**照 .mcp.json 里那行的形态跑**
        code, out = run_doctor(td, "--corpus", "corpus")
        assert code == 0, f"这份语料该全过，退出码 {code}：{out}"
        assert str(corpus.resolve()) in out, \
            f"报告里必须出现真实绝对路径（相对路径要解析开），实际输出：{out}"
        assert "建库：2 块" in out and "索引层 1" in out, f"块数与分层计数要在输出里：{out}"
        #    断在“建库：”那一行上，别只断“2 块”——时间戳来源那行也带块数，
        #    松着断的话把块数写死成常量的变异会从那儿溜过去（实测溜过一次）
        #    块数要真的数出来：**同一次 selftest 里再跑一份块数不同的语料**，
        #    不然把它写死成常量的变异测不出来（第一份恰好就是那个常量）
        bigger = _P(td) / "另一份"
        bigger.mkdir()
        (bigger / "window_05_2026-06-20.md").write_text(
            "## 换纱窗\n阳台的纱窗破了个洞，量好尺寸重新装了一扇。\n\n"
            "## 修水龙头\n厨房水龙头滴水，换掉里面的胶垫就好了。\n\n"
            "## 装晾衣杆\n阳台加了一根晾衣杆，位置挑在采光最好的一侧。\n",
            encoding="utf-8")
        code2, out2 = run_doctor(td, "--corpus", "另一份")
        assert code2 == 0 and "建库：3 块" in out2, f"块数要真数出来，不是写死的：{out2}"
        assert "时间范围" in out and "2026-06-17" in out, \
            f"时间范围（最早/最晚）要在输出里——旧快照那个坑靠它：{out}"
        assert "filename" in out, f"时间戳来源分布要在输出里：{out}"
        assert _snapshot(corpus) == before, "真进程跑一遍同样不许写盘"

        #    空目录：显著提示 + 退出码非零（自动化靠它，人靠那句话）
        empty = _P(td) / "空目录"
        empty.mkdir()
        code, out = run_doctor(td, "--corpus", "空目录")
        assert code != 0, f"空目录必须非零退出，实际 {code}：{out}"
        assert "✗" in out and "没找到任何 .md 语料" in out, f"提示要显著：{out}"
        assert str(empty.resolve()) in out, "失败路径同样要回答“读的是哪儿”"

        #    有 md、但切不出块（全空行）：不许崩成 traceback——正在排查故障的人
        #    要的是一句看得懂的话。旧版在这里 ZeroDivisionError
        blank = _P(td) / "空白语料"
        blank.mkdir()
        (blank / "a.md").write_text("\n   \n\n", encoding="utf-8")
        code, out = run_doctor(td, "--corpus", "空白语料")
        assert code != 0, f"切不出块的语料要非零退出，实际 {code}：{out}"
        assert "Traceback" not in out, f"不许崩，要出话：{out}"
        assert "一块内容都切不出来" in out and "✗" in out, f"要显著地说“一块都没有”：{out}"

        #    切块成色：正常语料不许报警，"每句一个 `##`"的导出必须报警。
        #    两个方向都断——只断报警那一半的话，"永远报警"的变异测不出来
        assert "中位块长" in out2 and "⚠ 切块" not in out2, \
            f"正常语料要报中位块长、且不该报切块警告：{out2}"
        每句一块 = _P(td) / "每句一块"
        每句一块.mkdir()
        (每句一块 / "chat_2026-06-21.md").write_text(
            "\n\n".join(["## 好的，我看一下", "## 这个改完了吗", "## 嗯",
                         "## 明天上午十点开会", "## 收到"]), encoding="utf-8")
        code, out3 = run_doctor(td, "--corpus", "每句一块")
        assert "⚠ 切块" in out3 and "只有一行标题" in out3, \
            f"每句话一个 `##` 的导出要被体检拦下来吭一声：{out3}"
        assert "`##`" in out3, f"要告诉人去看哪儿（语料里的 `##`）：{out3}"
        #    变异：去掉 _chunk_body 的首行判断（整块都当正文）/ 把阈值放到 1.0
        #    / 只报块数不报块长 → 各自红

        #    【mtime 骑到真日期头上·变异靶心】（2026.08.03 补）比例判据看的是
        #    "有多少块没日期"，这条看的是"没日期的块会不会骑到有日期的头上"。
        #    **危险的恰恰是占比低的情况**：绝大多数块有日期、少数几块落 mtime，
        #    那几块的时间戳就是"刚刚"，稳稳压过所有真日期，而占比连 20% 的门都
        #    够不着 → 静默。这里造的正是那种语料：4 块带日期、1 块落 mtime（20%
        #    不过阈值），必须靠新判据报出来。
        混合 = _P(td) / "混合"
        混合.mkdir()
        for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"):
            (混合 / f"window_01_{d}.md").write_text(
                f"# 窗口 · {d}\n\n那天做了些事，记一笔留着以后看。", encoding="utf-8")
        (混合 / "没有日期的备忘.md").write_text(
            "这条没有任何日期依据，落盘时间就是刚刚。", encoding="utf-8")
        code, out_mix = run_doctor(td, "--corpus", "混合")
        assert "⚠ 时间戳来源" in out_mix, \
            f"少数几块落 mtime 却比所有真日期都新时必须报警（比例判据够不着这一格）：{out_mix}"
        assert "最旧的事实当成你现在的状态" in out_mix or "方向错误" in out_mix, \
            f"WARN 正文要说清后果是方向错误，不是“没有排序信号”：{out_mix}"
        #    变异：把 outranks 那一支删掉（只留比例判据）→ 这条红

        #    缺 --corpus：argparse 拦下，同样是真进程才盖得到的一格
        code, out = run_doctor(td)
        assert code != 0 and "--corpus" in out, f"缺 --corpus 要被拦下：{code} {out}"

    print("selftest ok（17项断言：握手 / 工具表 / 调用往返 / 薄适配层 / 错误分层 / "
          "完整链路 / stdio / UTF-8 / 写回当场可查 / 用进撑过重启 / "
          "无可靠命中明确说 / 撤回更正闭环 / 缺失率标注接线 / 实体标注接线 / "
          "部署体检只读 / 部署体检走真进程（相对路径解析开、空语料非零退出、"
          "每句一个 `##` 的导出要报切块警告））")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doctor", action="store_true",
                    help="部署体检：查语料目录、时间戳成色、sidecar、写回落点与 MCP 接线，"
                         "只读不写盘。有一项过不去时退出码为 1")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--threads", help="会话线索 jsonl 路径（省略则内存态）")
    ap.add_argument("--embed", action="store_true", help="用真 embedding")
    ap.add_argument("--embed-provider", dest="embed_provider",
                    help="embedding 提供方：local（默认，需 fastembed）/ local:<模型> / "
                         "cloud（云端 HTTP，endpoint 与模型走 MEMORY_EMBED_* 环境变量，"
                         "key 只从环境变量读；**语料会发到那家服务商**）")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.doctor:
        if not args.corpus:
            ap.error("--doctor 要跟 --corpus 一起用：体检的就是它指向的那个目录")
        checks = diagnose(args.corpus, threads_path=args.threads, embed=args.embed)
        print(format_doctor_report(checks))
        # 退出码给自动化用：有 ✗ 就非零，⚠ 不算失败（那些是"能用但会悄悄变差"）
        sys.exit(1 if any(c["level"] == FAIL for c in checks) else 0)
    elif args.corpus:
        # 权重文件放语料目录下，起点号不带 .md——不会被 load_corpus 当语料吃进去
        # 块向量缓存（.embed_cache.json）同理，由 load_corpus 默认落在这里：
        # 没有它，云端档每次起服务都要把全库重算一遍
        MemoryServer(index=load_corpus(args.corpus, embed=args.embed,
                                       provider=(resolve_provider(args.embed_provider)
                                                 if args.embed else None)),
                     thread_store=ThreadStore(args.threads),
                     corpus_dir=args.corpus,
                     weights_path=Path(args.corpus) / ".weights.json",
                     retractions_path=Path(args.corpus) / ".retractions.json",
                     entities_path=Path(args.corpus) / ".entities.json").serve_stdio()
    else:
        ap.print_help()
