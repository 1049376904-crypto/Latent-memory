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

零依赖：不引 mcp SDK，stdlib 手写 JSON-RPC，跟本项目其余部分同风格。传输两条：
stdio（默认，宿主拉起）与 Streamable HTTP（--http，2026.08.03 加，给只认 HTTP 的
闭源前端直连，替掉 supergateway 桥——为什么与边界见 make_http_server 上面那段）。
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
  python mcp_server.py --corpus <md目录> --http 127.0.0.1:8765 --token <口令>  # HTTP 服务
  python mcp_server.py --doctor --corpus <md目录> [--threads <threads.jsonl>]  # 部署体检
客户端配置（Claude Desktop 之类）里把上面第二条命令填成 server 启动命令即可；
只认 HTTP 的客户端（Kelivo/Operit 这类）用第三条起服务、客户端里填地址与 token
（`--http` 的参数是 `[HOST:]PORT`，**省略 HOST 只绑回环、别的机器连不到**；
公网仍要域名 + 反代管 TLS，见《快速上手》§3c 部署形态二）；
配完接不上、或者不确定 --corpus 指对了没有时，把同一行参数换成 --doctor 跑一次
（体检只读，不往语料目录写任何东西）。
"""

import argparse
import hmac
import http.server
import io
import ipaddress
import json
import os
import sys
import threading
import time
import urllib.parse
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
                 entities_path=None, loader=None):
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
        self.entities_path = entities_path
        if entities_path is not None and self.index.load_entities(entities_path):
            self.index.build()
        # loader：怎么从盘上重建索引（常驻 HTTP 形态的重读用）。stdio 懒加载
        # 每次调用重读语料所以用不上；不传则常驻形态检测到语料变化时明确不重读
        self.loader = loader
        # 本进程自己写过哪些语料文件（常驻 HTTP 形态的指纹记账用）。
        # 为什么要精确到路径而不是"handle 之后重算一遍指纹"：那样会把**同一个请求
        # 窗口里用户手动上传的 md 一起吞进基线**，那次上传再也不触发重读——
        # 正好是自动重读这个特性要治的静默形态（2026.08.03 外部评审指出的竞态）
        self.written_paths = set()

    def _reload_from_disk(self):
        """语料目录有文件级变化时从盘上重建索引（常驻 HTTP 形态专用）。

        治的是 supergateway 时代那条实测过的静默坑：常驻 server 只在启动时读一次
        语料，手动上传进目录的 md 一条都看不到、不报错，直到重启。能这么重建的
        前提是**内存里没有只活在内存的账**：写回落盘（append_record）、撤回落盘
        （每次 correct 后 save_retractions）、权重落盘（每次 search 后 save_weights），
        重建后按各自的 sidecar 路径重新接上即可，无损。
        SessionRecall 只换 index 引用，会话内状态（距上次召回的字数）保留。"""
        self.index = self.loader()
        if self.retractions_path is not None:
            self.index.load_retractions(self.retractions_path)
        if self.weights_path is not None:
            self.index.load_weights(self.weights_path)
        if self.entities_path is not None and self.index.load_entities(self.entities_path):
            self.index.build()
        self.recall.index = self.index

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
        self.written_paths.add(str(path))    # 常驻形态的指纹记账（见 __init__ 注释）
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
            self.written_paths.add(str(path))    # 同 memory_append，见 __init__ 注释
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


# ---------- HTTP 传输（Streamable HTTP，2026.08.03） ----------
#
# 为什么要有这条：闭源前端（Kelivo/Operit 这类）只认 Streamable HTTP，此前用户
# 只能拿 npm 的 supergateway 把 stdio 桥一层。外部实测撞出来的四个坑全在桥上：
# ① supergateway 服务端模式**没有任何入站鉴权**——memory_search 读全库、
#    memory_append 可写入，端口被扫到记忆库就既可读又可写；
# ② 默认 SSE 输出与客户端的 Streamable HTTP 不匹配（`Transport disconnected`）；
# ③ 进程僵死（在但端口不通）要 pkill 重启；
# ④ 常驻只在启动时读一次语料，手动加的 md 静默看不到。
# 原生实现把四个一起治：鉴权内置且非回环裸跑直接拒绝起动、只说 Streamable HTTP、
# 少一个常驻进程、语料变化自动重读（_reload_from_disk）。
# 零依赖不破：http.server 是标准库。TLS 不在这做——闭源前端不认自签证书
# （两家独立实测），公网仍然要域名 + 反代（Caddy）那条路，反代顺手把 TLS 管了。
#
# 规格面（刻意做小，都是规格允许的服务器侧选择）：
#   POST 单条 JSON-RPC 消息 → application/json 单条响应；通知 → 202 空身；
#   GET（SSE 长流）→ 405 不提供，本 server 没有服务器主动推送的场景；
#   DELETE（会话终止）→ 200 空操作，本 server 无会话态、不发 Mcp-Session-Id；
#   JSON 数组（批量）→ 400（批量已从 2025-06 规格移除）。

_HTTP_BODY_LIMIT = 10 * 1024 * 1024     # 单请求上限：正常一条 tools/call 远小于此
# 被拒的请求最多吞多少 body 来保住 keep-alive（见 _drain）。超过就作废连接——
# 声明了超大长度的客户端多半根本没打算发完，等它等于把线程挂死
_HTTP_DRAIN_LIMIT = 64 * 1024
_HTTP_DRAIN_TIMEOUT = 5.0
# 被拒请求留痕的刷屏防线（见 _make_deny_logger）：一个窗口里最多逐条打多少行、
# 窗口有多长。绑了 0.0.0.0 就会被扫，401/404 能刷满终端、撑爆重定向到的文件
_HTTP_DENY_LOG_MAX = 60
_HTTP_DENY_LOG_WINDOW = 60.0


def _log_sanitize(s, limit):
    """要写进日志的那截字符串：不可打印字符换成 `?`，超长截断。

    客户端能塞进请求行的东西不受我们控制（路径里放 `\\r\\n` 就能伪造出一整行
    假日志，放几 KB 就能把终端顶满），而这行日志的用处正是让人一眼看清，
    所以进日志前一律过这道。"""
    s = "".join(ch if ch.isprintable() else "?" for ch in str(s))
    return s if len(s) <= limit else s[:limit] + "…"


def _make_deny_logger():
    """造一个「被拒请求留痕」的记录函数（每个 HTTP server 一份自己的计数器）。

    **为什么要有它**（2026.08.04 立卡）：八种拒绝（401/403/404/405/411/413/400/501）
    的原因 `_deny()` 每条都写得很清楚，但那句话只进**响应体**，而多数 MCP 客户端
    不显示响应体，用户看到的是「Transport disconnected」之类的通用失败；服务端这边
    `log_message` 整个静音，于是**两头都不说话，八种原因在用户那里长成同一个样子**。

    ⚠ **`log_message` 的静音不掀掉**：它静音的理由（个人部署逐请求刷屏是纯噪声）
    依然成立，这里只在它旁边开一个窄口子——**非 2xx 才留痕，正常请求照旧不打**。

    刷屏防线：一个窗口里最多逐条打 `_HTTP_DENY_LOG_MAX` 行，超出的压掉。
    ⚠ **压掉的条数必须说出来**——「压掉但不说」等于用一个新的静默去修一个旧的
    静默，那就白做了。所以窗口结束时补一行「过去一分钟另有 X 条被拒未逐条列出」，
    而且是**定时器兜底**：不能靠「下一个被拒请求来的时候顺手补」，那样一阵扫描
    打完就没下文了的话，那 X 条就真的没人说了。
    ⚠ ThreadingHTTPServer 是多线程的，计数器配自己的一把小锁（不复用 index 那把，
    那把锁的是整段检索，等它等于把留痕挂在业务后面）。"""
    lock = threading.Lock()
    st = {"t0": None, "n": 0, "dropped": 0, "timer": None}

    def _settle_locked():
        """窗口翻页：清计数，返回要补打的那行（没压掉东西就 None）。调用方持锁。"""
        if st["timer"] is not None:
            st["timer"].cancel()
            st["timer"] = None
        dropped, st["dropped"] = st["dropped"], 0
        st["t0"], st["n"] = None, 0
        return f"过去一分钟另有 {dropped} 条被拒未逐条列出" if dropped else None

    def _on_timeout():
        with lock:
            line = _settle_locked()
        if line:
            print(line, file=sys.stderr)

    def emit(line):
        out = []
        with lock:
            now = time.monotonic()
            if st["t0"] is None or now - st["t0"] >= _HTTP_DENY_LOG_WINDOW:
                closing = _settle_locked()
                if closing:
                    out.append(closing)
                st["t0"], st["n"] = now, 0
            if st["n"] < _HTTP_DENY_LOG_MAX:
                st["n"] += 1
                out.append(line)
            else:
                st["dropped"] += 1
                if st["timer"] is None:
                    t = threading.Timer(
                        max(0.0, st["t0"] + _HTTP_DENY_LOG_WINDOW - now), _on_timeout)
                    t.daemon = True      # 别让这只表拖住进程退出
                    st["timer"] = t
                    t.start()
        for x in out:
            print(x, file=sys.stderr)

    return emit


def _is_loopback(host):
    """绑定地址是不是回环。`localhost` 走名字判断，其余按 IP 解析；解析不了的
    （域名、空串）一律当非回环——**判不准时往严的那边倒**：那样守卫会要求 token、
    横幅会多打一行提醒，两个方向的错都是"多提醒一次"，不是"漏掉一次"。"""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost",)


def http_tls_notice(host):
    """非回环绑定时该多打的那一行（回环返回 None）。纯函数，起动横幅那边调。

    **为什么要有这一行**（2026.08.04 外部实测促成）：`http_bind_guard` 挡的是
    "非回环裸跑不配 token"，**它不挡"没有 TLS"**——配了 token 就照常起动，于是
    用户走在一条裸 HTTP 暴露公网的路上，token 与 `memory_search` 返回的记忆内容
    全程明文，而**没有任何人提醒过他**。那位用户自己提的建议是"检测到 token 已配
    就提示绑定地址"，方向要反过来：token 已配说明他知道要鉴权，**真正没被提醒的
    是 TLS**。

    ⚠ **只提示，不拒绝起动**：拒绝会把"临时绑 `0.0.0.0` 验证一下"这条路堵死，
    而那正是他这次定位根因的办法（怀疑过 SSE 格式、怀疑过桥接残留，最后是改绑定
    地址才连上）。
    ⚠ **不看 token**：token 与 TLS 是两件事，配了 token 也照样提醒——做成"有 token
    就不提示"等于把唯一一次提醒的机会送给最不需要它的人。"""
    if _is_loopback(host):
        return None
    return (f"⚠ 绑的是非回环地址（{host}）：这条链路没有 TLS，token 与 "
            f"memory_search 返回的记忆内容在网上都是明文，拿到 token 的人既能读"
            f"全库、也能写进去。推荐绑 127.0.0.1、公网那一跳交给反向代理"
            f"（Caddy／nginx）管 TLS；只是临时验证的话，至少上防火墙限制来源，"
            f"并把这个 token 当已经泄露过来处理。")


def http_bind_guard(host, token):
    """起动前的两道守卫：token 必须是 ASCII；非回环地址必须有 token。都是拒绝起动。

    **token 非 ASCII 也拒绝**（2026.08.03 外部实测）：HTTP 头是 latin-1，中文口令
    客户端根本发不出去（`UnicodeEncodeError`），就算发出去 `hmac.compare_digest`
    对非 ASCII str 直接抛 `TypeError` → 500。而失败形态是**服务看起来起来了、
    横幅也打了，就是连不上**——用户拿中文当口令是很可能的事，这正是本项目最怕的
    静默形态，所以在起动这一步就拦掉，跟下面绑定守卫同一个待遇。

    supergateway 那条路的头号坑就是"先跑通再说"地对公网开无鉴权端口——跑通那一刻
    记忆库已经暴露。这里把它做成出不了门的形态（同 guidance_text 超长拒绝出货的
    思路：失败要响，不静默）。回环上裸跑放行——本机 proot／反代在前的形态，
    鉴权在外层。"""
    if token:
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(
                "token 只能用 ASCII 字符（字母／数字／符号）：HTTP 头按 latin-1 编码，"
                "中文之类的口令客户端根本发不出去，而服务照样起得来——失败形态是"
                "「看起来起来了、就是连不上」。换一串英文数字口令再起。")
        return
    if not _is_loopback(host):
        raise ValueError(
            f"拒绝在非回环地址（{host}）上裸跑：没配 token 的 HTTP 端口等于把记忆库"
            f"公开成既可读又可写。加 --token（或环境变量 MEMORY_HTTP_TOKEN），"
            f"或绑回 127.0.0.1 由带鉴权的反向代理转发。")


def _corpus_signature(corpus_dir):
    """语料目录的文件级指纹（路径+mtime+大小）。只看 .md——sidecar（.weights.json
    每次检索都落盘）进指纹的话每个请求都会触发假重读。

    ⚠ glob 到 stat 之间文件可能已被删（并发整理语料），**漏掉它就行、不许抛**
    （2026.08.03 外部评审：这里抛 FileNotFoundError 会变成 500）——文件没了本身
    就是一次语料变化，指纹里少一条正好让下一次比较发现它。"""
    out = []
    for p in corpus_files(corpus_dir):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(out)


def make_http_server(server, host="127.0.0.1", port=8765, token=None):
    """MemoryServer → 绑好端口的 ThreadingHTTPServer（不启动；.serve_forever() 是
    调用方的事——selftest 要拿着实例开线程、取真端口、shutdown）。"""
    http_bind_guard(host, token)
    lock = threading.Lock()                  # index 的增删改建都不是线程安全的：串行化
    deny_log = _make_deny_logger()           # 被拒请求留痕（带刷屏上限，见那个函数）
    state = {"sig": _corpus_signature(server.corpus_dir)
             if (server.corpus_dir and server.loader) else None}

    class _Server(http.server.ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            """客户端中途断开不许喷 traceback（跟 log_message 静音同一个理由）。

            实测（2026.08.03 外部评审复验时量的）：客户端发到一半就走人——手机 App
            切后台、断网的常态——socketserver 默认往 stderr 打完整 Traceback，
            约 41 行一次。而 --http 的起动横幅也走 stderr，运维盯着终端会以为
            服务崩了。断连类只吞掉；**真异常保留一行摘要**，不静默成"什么都没发生"。"""
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionError, TimeoutError)):
                return
            print(f"HTTP 请求处理异常（{client_address[0]}）："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):           # 默认逐请求刷 stderr，个人部署只是噪声
            pass

        def _send(self, code, body=b"", ctype="application/json; charset=utf-8",
                  extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _drain(self):
            """把没读的请求体吞掉，返回是否吞干净了。

            **不吞就是下一个请求必坏**（2026.08.03 外部实测）：HTTP/1.1 默认
            keep-alive，401 直接返回时残留的 body 会被当成下一个请求的**请求行**
            解析，于是「先探一次拿 401 → 带 token 重试」这个再普通不过的姿势，
            第二发收到 400 + HTML 错误页。手机 App 复用连接是常态，而 selftest 里
            urllib 每次新连接、发 `Connection: close`——**这一格自检天然盖不住，
            判据必须走裸 socket**（见 18b）。

            ⚠ **吞不动的时候绝不能傻等**（第一版修法自己踩的坑）：413 那格客户端
            多半根本没打算发完那么多字节，`read(n)` 会把这个线程挂死；chunked、
            长度不合法同理。所以三种情况直接判连接作废：读不动、大到不值得读、
            读超时。作废走的是显式 `Connection: close` 响应头，不是只在服务端
            改个标志——客户端得知道这条连接不能再用了。"""
            te = self.headers.get("Transfer-Encoding")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if te or n < 0 or n > _HTTP_DRAIN_LIMIT:
                return False
            old = self.connection.gettimeout()
            try:
                self.connection.settimeout(_HTTP_DRAIN_TIMEOUT)
                while n > 0:
                    chunk = self.rfile.read(min(n, 8192))
                    if not chunk:
                        return False
                    n -= len(chunk)
            except OSError:                  # 超时／对端断了：连接作废，不挂线程
                return False
            finally:
                try:
                    self.connection.settimeout(old)
                except OSError:
                    pass
            return True

        def _log_deny(self, code, msg):
            """被拒的请求在 stderr 上留一行（八条拒绝全走 `_deny`，所以落点只有这一处）。

            形如：`被拒 404 GET /nope 来自 127.0.0.1：没有这个端点`
            ——状态码、方法、**剥掉 query 的**路径、客户端 IP、原因短语。

            ⚠ **三条硬边界（「为了让人能排查」而写进日志，正好是明文那张卡在防的
            事的另一个出口——日志会被重定向进文件、会被贴进 issue 找人帮看）**：

            1. **`Authorization` 的任何内容都不打**，一个字符都不打。401 只记
               「缺少或错误的 Bearer token」，不记收到的是什么——「只打前 4 位帮
               用户对一下」也不行。
            2. **不打请求体**。400「请求体不是合法的 UTF-8 JSON」时最想附上的就是
               那段 body，**恰恰最不能附**：那里面是 `memory_search` 的问句和
               `memory_append` 的正文，是用户最私密的记忆。413 同理。
            3. **路径剥掉 query string**：`_deny(404, ...)` 把整个 `self.path` 回给
               对方没问题（是他自己发的），但**写进日志就落盘了**，而有些客户端
               习惯把 token 塞 `?token=`。所以这里用 `urlsplit(self.path).path`，
               **不用 `self.path`**。

            原因短语取 `msg` 的**第一个分隔符之前**那截：各调用点把客户端来的值
            （路径、Origin、`Transfer-Encoding` 的值）都插在分隔符之后，切掉即可。
            ⚠ **以后新增 `_deny` 调用点，变量插值必须放在「：」「: 」「——」之后**，
            否则它会跟着原因短语进日志。"""
            reason = msg
            for sep in ("：", ": ", "——", "\n"):
                reason = reason.split(sep, 1)[0]
            path = urllib.parse.urlsplit(self.path).path
            ip = self.client_address[0] if self.client_address else "?"
            deny_log(f"被拒 {code} {_log_sanitize(self.command or '?', 16)} "
                     f"{_log_sanitize(path, 120)} 来自 {_log_sanitize(ip, 60)}："
                     f"{_log_sanitize(reason, 80)}")

        def _deny(self, code, msg, extra=()):
            self._log_deny(code, msg)
            extra = list(extra)
            if not self._drain():
                self.close_connection = True
                extra.append(("Connection", "close"))
            self._send(code, json.dumps({"error": msg}, ensure_ascii=False)
                       .encode("utf-8"), extra=extra)

        def _gate(self):
            """路径 + 鉴权 + Origin 三道闸。过了返回 True。"""
            # 路径校验（2026.08.03 外部评审：文档教人填 /mcp，实际任何路径都收，
            # 无害但跟文档不一致）。放行根路径是给"填地址时漏了 /mcp"留的余地——
            # 那是最常见的手滑，收下比回 404 让人排查半天强
            if urllib.parse.urlsplit(self.path).path.rstrip("/") not in ("", "/mcp"):
                self._deny(404, f"没有这个端点：{self.path}——本 server 只在 /mcp 上收")
                return False
            if token:
                got = self.headers.get("Authorization") or ""
                if not hmac.compare_digest(got, f"Bearer {token}"):
                    self._deny(401, "缺少或错误的 Bearer token",
                               extra=[("WWW-Authenticate", "Bearer")])
                    return False
            # Origin 校验（规格的 DNS rebinding 防线）：App 客户端不发 Origin，
            # 发了且不是本机的只会是浏览器页面在拿本地端口当跳板
            origin = self.headers.get("Origin")
            if origin:
                h = (urllib.parse.urlsplit(origin).hostname or "").lower()
                if h not in ("localhost", "127.0.0.1", "::1"):
                    self._deny(403, f"Origin 不被信任：{origin}")
                    return False
            return True

        def do_GET(self):
            if not self._gate():
                return
            self._deny(405, "本 server 不提供 SSE 长流，请用 POST（Streamable HTTP）",
                       extra=[("Allow", "POST, DELETE")])

        def do_DELETE(self):
            if not self._gate():
                return
            self._send(200)                  # 无会话态，终止是空操作

        def do_POST(self):
            if not self._gate():
                return
            # chunked 明确不支持，**且报错要指对方向**（2026.08.03 外部实测）：
            # 原先不带 Content-Length 时 length 取 0、读到空串，报的是
            # 「请求体不是合法的 UTF-8 JSON」——把人往"我的 JSON 写错了"引，
            # 而真因是传输编码不匹配。这跟 supergateway 时代那个「默认 SSE 与
            # 客户端不匹配」是同形物：传输层没对上，报错还指错地方。
            # 501 是规格给这种情况的码（服务器不支持该功能）。
            if self.headers.get("Transfer-Encoding"):
                return self._deny(
                    501, f"不支持 Transfer-Encoding: "
                         f"{self.headers.get('Transfer-Encoding')}——本 server 只收带 "
                         f"Content-Length 的整条 JSON-RPC 消息。这是传输编码不匹配，"
                         f"不是你的 JSON 写错了；客户端若有「分块传输/流式上传」开关，"
                         f"关掉即可。")
            if self.headers.get("Content-Length") is None:
                return self._deny(411, "缺 Content-Length：本 server 只收带长度的整条消息")
            try:
                length = int(self.headers.get("Content-Length"))
                if length < 0:
                    raise ValueError
            except ValueError:
                return self._deny(400, "Content-Length 不合法")
            if length > _HTTP_BODY_LIMIT:
                return self._deny(413, "请求体超限")
            try:
                msg = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._deny(400, "请求体不是合法的 UTF-8 JSON")
            if isinstance(msg, list):
                return self._deny(400, "不支持 JSON-RPC 批量（2025-06 规格已移除）")
            if not isinstance(msg, dict):
                return self._deny(400, "请求体要是一条 JSON-RPC 消息对象")
            with lock:
                # 常驻形态的重读：语料目录文件级变化（手动上传/删除 md）→ 从盘重建。
                # ⚠ **指纹要在 handle 之后重算、而且只认自己写的那次变化**
                # （2026.08.03 外部评审指出的竞态）：原先 handle 后无条件刷新指纹，
                # 于是「一次 memory_append 与用户手动上传落在同一个请求窗口」时，
                # 那次上传会被自己的刷新永久吞掉、再也不触发重读——正好是这个特性
                # 要治的静默形态。改法：handle 前后各取一次，**只把 server 自己写回
                # 造成的差异并进基线**（after 与 before 的差集里属于自己写回的部分
                # 无从区分，所以退一步取更安全的一侧：若 handle 期间指纹又变了，
                # 保留 before，让下一个请求去发现并重读——宁可多重读一次，
                # 不可漏掉一次）。
                if state["sig"] is not None:
                    sig = _corpus_signature(server.corpus_dir)
                    if sig != state["sig"]:
                        server._reload_from_disk()
                        state["sig"] = sig
                resp = server.handle(msg)
                if state["sig"] is not None and server.written_paths:
                    # **只折进 server 自己写的那几个文件**，其余差异留着让下一个
                    # 请求去发现——手动上传就落在"其余"里，不会被自己的刷新吞掉。
                    # 顺带：只读请求（memory_search 是绝大多数）written_paths 是空的，
                    # 这里整个跳过，不再每个请求 stat 全库两遍
                    base = {p: (m, s) for p, m, s in state["sig"]}
                    for p, m, s in _corpus_signature(server.corpus_dir):
                        if p in server.written_paths:
                            base[p] = (m, s)
                    state["sig"] = tuple(sorted((p, m, s) for p, (m, s) in base.items()))
                    server.written_paths = set()
            if resp is None:
                self._send(202)              # 通知：202 空身（规格）
            else:
                self._send(200, json.dumps(resp, ensure_ascii=False).encode("utf-8"))

    return _Server((host, port), _Handler)


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

    # ⚠ 下面 18 系列会**故意**撞出四种拒绝，而被拒请求 2026.08.04 起会在服务端留一行
    #    （18g 那张卡），于是这些行会跑到跑自检的人眼前。**先说一句它们不是出错**——
    #    否则用户看到一屏「被拒 …」根本分不清自检过没过（真机第一次跑就被这样问了）。
    #    ⚠ **刻意不做成「自检期间把它们吞掉」**：这张卡治的就是「压掉但不说」，
    #    在它自己的自检里加一个静默开关是反过来走；而且真有一天拒绝在不该触发时触发了，
    #    看得见比看不见强。所以是加一句说明，不是加一道闸。
    print("⚠ 下面几行「被拒 …」是自检故意撞出来的（18 系列要挨个验四种拒绝该不该挡住），"
          "不是出错——以最后一行 selftest ok 为准。", file=sys.stderr)

    # 18.【HTTP 传输·真端口往返】（2026.08.03「连不上 MCP 只好上 supergateway」
    #     那单）：supergateway 的四个实测坑逐个变成这里的一格——无鉴权（→401 两格
    #     ＋非回环裸跑拒绝）、传输不匹配（→握手/工具表/UTF-8 真 HTTP 往返）、
    #     常驻不重读语料（→手动放 md 下一个请求就查到）；外加 Origin 防线、
    #     通知 202、GET 405、批量 400 三条规格面。
    import urllib.request
    import urllib.error
    try:
        http_bind_guard("0.0.0.0", None)
        assert False, "非回环 + 无 token 必须拒绝起动——跑通的那一刻记忆库就暴露了"
    except ValueError as e18:
        assert "token" in str(e18), "拒绝的报错要指向修法（--token / 绑回环）"
    http_bind_guard("127.0.0.1", None)       # 回环裸跑放行：反代在前的形态
    http_bind_guard("0.0.0.0", "s3cret")     # 配了 token 的公开绑定放行
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        loader18 = lambda: load_corpus(td)
        srv18 = MemoryServer(index=loader18(), thread_store=ThreadStore(),
                             corpus_dir=td,
                             retractions_path=_P(td) / ".retractions.json",
                             loader=loader18)
        httpd = make_http_server(srv18, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base18 = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"

        def post18(payload, tok="s3cret", origin=None, method="POST"):
            req = urllib.request.Request(
                base18, data=(json.dumps(payload, ensure_ascii=False).encode("utf-8")
                              if payload is not None else None), method=method)
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            if origin:
                req.add_header("Origin", origin)
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8")

        lst18 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        #     鉴权两格（变异：把 _gate 的 token 检查删掉 → 这两条红）
        assert post18(lst18, tok=None)[0] == 401, "无 token 必须 401"
        assert post18(lst18, tok="wrong")[0] == 401, "错 token 必须 401"
        #     Origin 防线（DNS rebinding：浏览器页面拿本地端口当跳板）
        assert post18(lst18, origin="https://evil.example")[0] == 403
        #     握手 → 工具表 → 通知 202
        code18, body18 = post18({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                 "params": {}})
        assert code18 == 200 and "protocolVersion" in body18
        code18, body18 = post18(lst18)
        assert code18 == 200 and "memory_search" in body18
        code18, body18 = post18({"jsonrpc": "2.0",
                                 "method": "notifications/initialized"})
        assert code18 == 202 and not body18, "通知回 202 空身，不回 JSON-RPC 响应"
        #     中文 UTF-8 真 HTTP 往返：写回 → 当场检索命中（同 stdio 第 9 项的病灶）
        code18, body18 = post18({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "memory_append",
                                            "arguments": {"text": "绿萝换了大一号的盆。",
                                                          "current_state": "适应中。"}}})
        assert code18 == 200 and "已写进" in body18
        code18, body18 = post18({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                 "params": {"name": "memory_search",
                                            "arguments": {"query": "绿萝换盆"}}})
        assert code18 == 200 and "大一号的盆" in body18, \
            f"中文该原样穿过 HTTP 并命中：{body18}"
        #     常驻自动重读（变异：把 do_POST 里的指纹检查删掉 → 这条红）：
        #     手动上传 md 是 supergateway 时代实测过的静默坑——常驻只在启动读一次，
        #     新语料一条都查不到且不报错
        (_P(td) / "timeline" / "window_09_2026-08-02.md").write_text(
            "# 第9个窗口 · 2026-08-02\n\n## 2026-08-02 记\n手动上传：入手了一台"
            "折叠望远镜。\n当下状态：还没开箱。\n", encoding="utf-8")
        code18, body18 = post18({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                 "params": {"name": "memory_search",
                                            "arguments": {"query": "望远镜"}}})
        assert code18 == 200 and "折叠望远镜" in body18, \
            "语料目录手动加的 md 必须下一个请求就查得到——不重读是静默的"
        #     规格面三格：GET 无长流、DELETE 空操作、批量已移除
        assert post18(None, method="GET")[0] == 405
        assert post18(None, method="DELETE")[0] == 200
        code18, body18 = post18([lst18])
        assert code18 == 400 and "批量" in body18
        #     路径：文档教人填 /mcp，别的端点要明确 404（根路径放行是手滑余地）
        assert post18(lst18)[0] == 200
        req_bad = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/nope",
            data=json.dumps(lst18).encode(), method="POST")
        req_bad.add_header("Authorization", "Bearer s3cret")
        try:
            urllib.request.urlopen(req_bad, timeout=10)
            assert False, "未知端点要 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # 18b.【keep-alive 下被拒的请求必须吞掉 body·变异靶心】（2026.08.03 外部实测）
        #     ⚠ **这一格只能走裸 socket**：上面 urllib 每次新连接、发 Connection: close，
        #     所以它天然盖不住——而手机 App 复用连接是常态，「先探一次拿 401 →
        #     带 token 重试」这个最普通的姿势就撞这个坑。不吞 body 的话残留会被
        #     当成下一个请求的**请求行**，第二发收到 400 + HTML 错误页。
        #     变异：把 _deny 里的 _drain() 去掉 → 这条红
        import socket as _socket
        raw = json.dumps(lst18).encode()

        def _recv_http(sock):
            """收完整一条响应——**必须按 Content-Length 收满**：单发 recv 常常只
            拿到头部，body 还在路上，据此断言等于随机红绿。"""
            buf = b""
            while b"\r\n\r\n" not in buf:
                d = sock.recv(65536)
                if not d:
                    return buf.decode("utf-8", "replace")
                buf += d
            head, _, body = buf.partition(b"\r\n\r\n")
            want = 0
            for ln in head.decode("latin-1").split("\r\n")[1:]:
                if ln.lower().startswith("content-length:"):
                    want = int(ln.split(":", 1)[1])
            while len(body) < want:
                d = sock.recv(65536)
                if not d:
                    break
                body += d
            return (head + b"\r\n\r\n" + body).decode("utf-8", "replace")

        def _raw_post(sock, tok=None, extra_hdr=""):
            h = (f"POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(raw)}\r\n")
            if tok:
                h += f"Authorization: Bearer {tok}\r\n"
            sock.sendall((h + extra_hdr + "\r\n").encode() + raw)
            return _recv_http(sock)

        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        first = _raw_post(sk)                       # 无 token → 401
        assert "401" in first.split("\r\n")[0], first.split("\r\n")[0]
        second = _raw_post(sk, tok="s3cret")        # 同一条连接重试 → 必须是 200
        assert "200" in second.split("\r\n")[0], \
            f"401 之后同连接重试必须正常——被拒的请求没吞掉 body：{second.splitlines()[0]}"
        assert "memory_search" in second, "重试那发要真的拿到工具表"
        sk.close()
        #     413 分支同形（超限也要吞）：谎报一个超大 Content-Length 但不发那么多，
        #     server 该在读之前就拒掉，且连接不许被污染
        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        sk.sendall((f"POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n"
                    f"Content-Length: {_HTTP_BODY_LIMIT + 1}\r\n\r\n").encode())
        #     ⚠ 收响应一律走 `_recv_http`，不许 `sk.recv()` 一发了事——理由见 18c
        assert "413" in _recv_http(sk).split("\r\n")[0]
        sk.close()

        # 18c.【chunked 明确不支持，且报错指对方向·变异靶心】（2026.08.03 外部实测）
        #     原先走到"请求体不是合法的 UTF-8 JSON"，把人往「我 JSON 写错了」引，
        #     真因是传输编码不匹配——跟 supergateway 时代「默认 SSE 与客户端不匹配」
        #     同形物。变异：把 Transfer-Encoding 那一支删掉 → 这条红
        #     ⚠ **必须走 `_recv_http` 按 Content-Length 收满**（2026.08.04「自检偶发红」
        #     那单的根因）：`_send` 先冲响应头、再写 body 是两段，单发 `recv` 常常只
        #     拿到头——于是查状态行的断言恒绿、查 body 的那条随机红，看起来就是
        #     「偶发红、复跑即绿、与本轮改动无关」。同日撞了四次都记成噪音。
        #     ⚠ 这条规矩 18b 的 `_recv_http` docstring 里早写着，这里当时没照着用。
        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        sk.sendall((f"POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n"
                    f"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
                    ).encode() + f"{len(raw):x}\r\n".encode() + raw + b"\r\n0\r\n\r\n")
        ck = _recv_http(sk)
        assert "501" in ck.split("\r\n")[0], f"chunked 要回 501 不是 400：{ck.splitlines()[0]}"
        assert "Transfer-Encoding" in ck and "JSON 写错" in ck, \
            "报错必须指向传输编码，并明说不是 JSON 的问题——指错方向就是这条的病灶"
        sk.close()
        httpd.shutdown()

    # 18d.【非 ASCII token 拒绝起动·变异靶心】（2026.08.03 外部实测）：中文口令
    #     server 起得来、横幅照打，但 HTTP 头是 latin-1，客户端发不出去；
    #     compare_digest 对非 ASCII str 直接抛 TypeError → 500。失败形态是
    #     「看起来起来了、就是连不上」，所以在起动这一步拦。
    #     变异：把 token.encode("ascii") 那一支删掉 → 这条红
    try:
        http_bind_guard("127.0.0.1", "口令abc")
        assert False, "非 ASCII token 必须拒绝起动——不然失败形态是静默的"
    except ValueError as e18b:
        assert "ASCII" in str(e18b), "报错要说清是 ASCII 的问题并给修法"
    http_bind_guard("127.0.0.1", "s3cret-2026")     # ASCII 照常放行

    # 18f.【非回环绑定多打一行 TLS 提醒·真进程，两个绑法都跑】（2026.08.04 外部实测
    #     促成）：`http_bind_guard` 挡的是"非回环裸跑不配 token"，**不挡"没有 TLS"**，
    #     于是配了 token 的用户走在裸 HTTP 暴露公网的路上、全程没人提醒过他。
    #     ⚠ **必须起真进程**：提醒是在 `__main__` 里打的，只断纯函数就是"函数绿、
    #     生产路径死"那张脸——把 `__main__` 里那两行删掉，纯函数断言一条都不会红。
    #     ⚠ **两个绑法都要跑**：只跑非回环那次证明不了"回环时不吵"。
    #     ⚠ **非回环那次带着 token 跑**：提醒不许看 token（token 与 TLS 是两件事），
    #     做成"有 token 就不提示"这一格要红。
    #     ⚠ 两次都断"横幅在"：否则进程压根没起来也能让"不含 TLS"那条恒真（假绿）。
    #     变异：删掉 `__main__` 里打 notice 那两行 / 让 http_tls_notice 恒返回 None /
    #           改成 token 已配就不打 → 各自红
    def run_banner(corpus_dir, bind, *extra):
        """起真进程跑 `--http`，收横幅那几行，然后掐掉。返回 stderr 文本。"""
        proc = subprocess.Popen(
            [sys.executable, str(here / "mcp_server.py"), "--corpus", str(corpus_dir),
             "--http", bind, *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        got = []
        pump = threading.Thread(target=lambda: got.extend(proc.stderr), daemon=True)
        pump.start()
        deadline = time.time() + 20
        while time.time() < deadline and not any("语料变化自动重读" in x for x in got):
            time.sleep(0.05)
        time.sleep(0.5)          # 提醒行紧跟在横幅后面，给它落地的时间
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        pump.join(timeout=5)
        return "".join(got)

    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        out_pub = run_banner(td, "0.0.0.0:0", "--token", "s3cret-2026")
        assert "Streamable HTTP 服务在" in out_pub, \
            f"非回环 + token 必须照常起动（只提示、不拒绝）：{out_pub}"
        assert "TLS" in out_pub, \
            f"绑非回环要多打一行 TLS 提醒——配了 token 也照打：{out_pub}"
        assert "127.0.0.1" in out_pub and "反向代理" in out_pub, \
            f"提醒里要给出修法（绑回环 + 反代管 TLS），不能只说一句「有风险」：{out_pub}"
        out_lo = run_banner(td, "127.0.0.1:0")
        assert "Streamable HTTP 服务在" in out_lo, \
            f"回环裸跑照常起动（反代在前的形态）：{out_lo}"
        assert "TLS" not in out_lo, f"回环上不许吵这一句：{out_lo}"

    # 18e.【自动重读不许吞掉同窗口的手动上传·变异靶心】（2026.08.03 外部评审的竞态）：
    #     原先 handle 之后无条件重算指纹，于是一次 memory_append 与用户手动上传
    #     落在同一个请求窗口时，那次上传被自己的刷新吞掉、**再也不触发重读**——
    #     正好是这个特性要治的静默形态。改法是只把 server 自己写过的路径折进基线。
    #     变异：把 written_paths 那套换回 state["sig"] = _corpus_signature(...) → 这条红
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n养了绿萝。\n"
            "当下状态：正常。\n", encoding="utf-8")
        loader18e = lambda: load_corpus(td)
        srv18e = MemoryServer(index=loader18e(), thread_store=ThreadStore(),
                              corpus_dir=td, loader=loader18e)
        httpd18e = make_http_server(srv18e, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd18e.serve_forever, daemon=True).start()
        b18e = f"http://127.0.0.1:{httpd18e.server_address[1]}/mcp"

        def call18e(payload):
            rq = urllib.request.Request(b18e, data=json.dumps(payload).encode(),
                                        method="POST")
            rq.add_header("Authorization", "Bearer s3cret")
            with urllib.request.urlopen(rq, timeout=10) as r:
                return r.read().decode("utf-8")

        # ⚠ **上传必须落在 handle 期间**，这才是竞态窗口：写在请求之前的话，
        #   do_POST 的前置指纹检查当场就逮住它，反而测不到这一格（第一版夹具
        #   就这么造错了，变异不红——夹具错了跟钉子失效长得一模一样）。
        real_handle, done18e = srv18e.handle, []

        def handle_then_upload(msg, now=None):
            r = real_handle(msg, now=now)
            if not done18e:                  # 只插一次
                done18e.append(1)
                (_P(td) / "timeline" / "window_09_2026-08-02.md").write_text(
                    "# 第9个窗口 · 2026-08-02\n\n## 2026-08-02 记\n"
                    "手动上传：买了折叠望远镜。\n当下状态：还没开箱。\n",
                    encoding="utf-8")
            return r

        srv18e.handle = handle_then_upload
        call18e({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "memory_append",
                            "arguments": {"text": "顺手记一条别的。",
                                          "current_state": "无。"}}})
        srv18e.handle = real_handle
        got18e = call18e({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": "memory_search",
                                     "arguments": {"query": "望远镜"}}})
        assert "折叠望远镜" in got18e, \
            "同窗口里的手动上传不许被自己的写回吞掉——吞掉就再也不重读了（静默）"
        httpd18e.shutdown()

    # 18g.【被拒的请求在服务端要留一行·四个靶心】（2026.08.04 立卡）：八种拒绝的
    #     原因 `_deny` 每条都写清楚了，但那句话只进**响应体**，多数 MCP 客户端不显示；
    #     服务端 `log_message` 整个静音，于是**两头都不说话**，八种原因在用户那里
    #     长成同一个样子（「连不上」）。落点只有 `_deny` 一处（八条全走它）。
    #     ⚠ **靶心二必须跟靶心一一起跑**：只跑靶心一的话，「正确实现」与「把
    #     `log_message` 的静音整个掀掉」在证据上长得一模一样。
    #     ⚠ 取证靠把 `sys.stderr` 换成 StringIO（`print(..., file=sys.stderr)` 每次
    #     现查属性，替换全局有效）。这一格在进程内起 server，**不走起动横幅**，
    #     所以捕到的行就是请求行本身；靶心二据此断"一行都没有"。
    #     变异：① 删掉 `_deny` 里那行 `self._log_deny(...)` → 靶心一红；
    #           ② 把 `_log_deny` 里的 `urlsplit(...).path` 换回 `self.path` → 靶心三红；
    #           ③ 删掉 `_settle_locked` 里返回「另有 X 条」那一行（只保留上限压制）
    #              → 靶心四红。
    import socket as _socket18g
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        loader18g = lambda: load_corpus(td)
        srv18g = MemoryServer(index=loader18g(), thread_store=ThreadStore(),
                              corpus_dir=td,
                              retractions_path=_P(td) / ".retractions.json",
                              loader=loader18g)
        httpd18g = make_http_server(srv18g, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd18g.serve_forever, daemon=True).start()
        port18g = httpd18g.server_address[1]
        b18g = f"http://127.0.0.1:{port18g}"

        def hit18g(path="/mcp", tok="s3cret", origin=None, method="POST", body=None):
            req = urllib.request.Request(b18g + path, data=body, method=method)
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            if origin:
                req.add_header("Origin", origin)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                e.read()
                return e.code

        def cap18g(fn):
            """跑一段，返回它在 stderr 上留下的全部文本。"""
            buf, old = io.StringIO(), sys.stderr
            sys.stderr = buf
            try:
                fn()
            finally:
                sys.stderr = old
            return buf.getvalue()

        good18g = json.dumps({"jsonrpc": "2.0", "id": 1,
                              "method": "tools/list"}).encode("utf-8")

        #     靶心一（要说话）：401/403/404/405 各发一次 → **各出现一行**，
        #     且那行同时含状态码、方法、路径、客户端 IP。
        #     ⚠ 判的是四条各自出现，不是"stderr 非空"。
        def _four():
            assert hit18g(tok="wrong", body=good18g) == 401
            assert hit18g(origin="https://evil.example", body=good18g) == 403
            assert hit18g(path="/nope", method="GET") == 404
            assert hit18g(method="GET") == 405

        out18g = cap18g(_four)
        for code18g, meth18g, path18g in (("401", "POST", "/mcp"), ("403", "POST", "/mcp"),
                                          ("404", "GET", "/nope"), ("405", "GET", "/mcp")):
            hits = [ln for ln in out18g.splitlines()
                    if ln.startswith(f"被拒 {code18g} ")]
            assert len(hits) == 1, f"{code18g} 要在 stderr 上留且只留一行：{out18g!r}"
            assert meth18g in hits[0] and f" {path18g} " in hits[0] \
                and "127.0.0.1" in hits[0], \
                f"那一行要同时含方法、路径、客户端 IP：{hits[0]!r}"

        #     靶心二（不许多说）：一次正常往返 → stderr 上**不出现任何请求行**。
        #     断的是"一个字符都没有"：`log_message` 的静音一旦被掀掉，
        #     这里会冒出 `"POST /mcp HTTP/1.1" 200 -` 那种访问日志。
        def _ok():
            assert hit18g(body=json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_search",
                            "arguments": {"query": "绿萝"}}}).encode("utf-8")) == 200, \
                "正常的 tools/call 该 200——它不 200 的话下面那条恒真（假绿）"

        ok18g = cap18g(_ok)
        assert ok18g == "", \
            f"正常请求不许留痕——`log_message` 的静音不许被掀掉：{ok18g!r}"

        #     靶心三（不许说漏嘴）：三条硬边界各钉一枚哨兵。
        def _leak():
            assert hit18g(tok="wrong-SENTINELA", body=good18g) == 401
            assert hit18g(path="/nope?token=SENTINELB", method="GET") == 404
            assert hit18g(body=b"{not json SENTINELC") == 400

        leak18g = cap18g(_leak)
        assert "SENTINELA" not in leak18g, f"Authorization 一个字符都不许进日志：{leak18g!r}"
        assert "SENTINELB" not in leak18g, \
            f"路径要剥掉 query（有客户端把 token 塞 ?token=）：{leak18g!r}"
        assert "SENTINELC" not in leak18g, \
            f"请求体不许进日志（里面是 memory_search 的问句／memory_append 的正文）：{leak18g!r}"
        n404 = [ln for ln in leak18g.splitlines() if ln.startswith("被拒 404 ")]
        assert len(n404) == 1 and " /nope " in n404[0], \
            f"剥完 query 那行的路径要正好是 /nope：{n404!r}"

        #     靶心四（上限不静默）：一个窗口内连发 200 条被拒 → 逐条行 ≤ 60，
        #     **且出现「另有 140 条」那一行**（数目对得上）。压掉不说等于用一个
        #     新的静默去修一个旧的静默。
        #     ⚠ 窗口临时缩短到 5 秒，否则要等一分钟；同时**断这 200 发确实落在
        #     一个窗口里**——夹具太慢导致窗口翻页，跟"上限失效"长得一样。
        old_win = _HTTP_DENY_LOG_WINDOW
        globals()["_HTTP_DENY_LOG_WINDOW"] = 5.0
        try:
            buf18g, old_err = io.StringIO(), sys.stderr
            sys.stderr = buf18g
            try:
                t018g = time.monotonic()

                def burst_one():
                    """一条 keep-alive 连接上连发 20 个 404。
                    ⚠ **10 条连接并发发**：单连接串行跑 200 发要 8 秒多
                    （每发都吃一次 delayed-ACK，回环上实测 ~43ms），会跨出窗口。"""
                    sk = _socket18g.create_connection(("127.0.0.1", port18g))
                    sk.settimeout(10)
                    for _ in range(20):
                        sk.sendall(b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
                        _recv_http(sk)    # 按 Content-Length 收满，理由见 18b
                    sk.close()

                ths18g = [threading.Thread(target=burst_one) for _ in range(10)]
                for t in ths18g:
                    t.start()
                for t in ths18g:
                    t.join(timeout=20)
                burst18g = time.monotonic() - t018g
                # 溢出那行由定时器在窗口末尾补打，等它
                dead18g = time.monotonic() + 12
                while time.monotonic() < dead18g and "另有" not in buf18g.getvalue():
                    time.sleep(0.05)
            finally:
                sys.stderr = old_err
            burst_out = buf18g.getvalue()
        finally:
            globals()["_HTTP_DENY_LOG_WINDOW"] = old_win
        assert burst18g < 4.0, \
            f"夹具太慢：这 200 发没落在同一个 5 秒窗口里（用了 {burst18g:.1f}s），" \
            f"下面的 60／140 不作数——先修夹具，别当成上限失效"
        each18g = [ln for ln in burst_out.splitlines() if ln.startswith("被拒 ")]
        assert len(each18g) == _HTTP_DENY_LOG_MAX, \
            f"逐条行要正好压到上限 {_HTTP_DENY_LOG_MAX} 行，实得 {len(each18g)}"
        assert f"过去一分钟另有 {200 - _HTTP_DENY_LOG_MAX} 条被拒未逐条列出" in burst_out, \
            f"压掉的条数必须说出来，且数目对得上：{burst_out!r}"
        httpd18g.shutdown()

    # 19. CLI 入口必须把 stdout 锁成 UTF-8（`--doctor` 打的是中文报告）。第 8 项
    #     锁的是 stdio 传输那条流（serve_stdio 自己包 UTF-8），管不到 `print`。
    #     ⚠ **在 Linux／默认 UTF-8 的机器上这条恒真，在那儿跑不算验过**：变异要在
    #     `PYTHONIOENCODING=gbk` 下跑——删掉 `__main__` 里的
    #     `sys.stdout.reconfigure(...)` 必须转红，加回去复绿。
    assert (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8", \
        f"CLI 入口没把 stdout 锁成 UTF-8（当前 {sys.stdout.encoding}）：" \
        "中文 Windows（cp936）下 --doctor 遇到 emoji 会 UnicodeEncodeError"

    print("selftest ok（19项断言：握手 / 工具表 / 调用往返 / 薄适配层 / 错误分层 / "
          "完整链路 / stdio / UTF-8 / 写回当场可查 / 用进撑过重启 / "
          "无可靠命中明确说 / 撤回更正闭环 / 缺失率标注接线 / 实体标注接线 / "
          "部署体检只读 / 部署体检走真进程（相对路径解析开、空语料非零退出、"
          "每句一个 `##` 的导出要报切块警告）/ HTTP 传输真端口往返（鉴权 401、"
          "非回环裸跑拒绝、非 ASCII token 拒绝起动、Origin 403、路径 404、通知 202、"
          "GET 405、批量 400、超限 413、chunked 501、keep-alive 下被拒后同连接重试、"
          "UTF-8 写回即查、语料变化自动重读、同窗口手动上传不被写回吞掉、"
          "非回环绑定的起动横幅多打一行 TLS 提醒／回环时不打——真进程两个绑法都跑、"
          "被拒的请求在服务端留一行（码／方法／剥掉 query 的路径／客户端 IP／原因，"
          "正常请求照旧不打，token 与请求体不进日志，刷屏上限 60 行且溢出条数补一行）"
          "）/ "
          "CLI 入口把 stdout 锁成 UTF-8（⚠ 变异要在 PYTHONIOENCODING=gbk 下跑，"
          "默认 UTF-8 的机器上这条恒真））")


if __name__ == "__main__":
    # 中文 Windows（cp936/GBK）下 stdout 按系统区域编码写，`--doctor` 的中文报告
    # 一遇到 emoji 就 UnicodeEncodeError（缘由与判据见 memory_init.py 同一处注释）。
    # stdio 传输那条路本来就已经自己包了 UTF-8（见 serve_stdio），这里补的是
    # **面向控制台的那部分输出**；改的是编码，不是 ensure_ascii。
    # ⚠ 只放在 `__main__` 里：被 import 时不许改掉调用方进程的 stdout。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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
    ap.add_argument("--http", metavar="[HOST:]PORT",
                    help="改走 Streamable HTTP（省略 HOST 默认 127.0.0.1）——给只认 "
                         "HTTP 的客户端（Kelivo/Operit 这类）直连用，不再需要 "
                         "supergateway 桥。非回环地址必须配 token，否则拒绝起动；"
                         "绑非回环还会多打一行 TLS 提醒——那条链路是裸 HTTP")
    ap.add_argument("--token",
                    help="HTTP 传输的 Bearer token（也可用环境变量 MEMORY_HTTP_TOKEN；"
                         "客户端侧填进 bearerToken/Authorization 头）")
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
        loader = lambda: load_corpus(args.corpus, embed=args.embed,
                                     provider=(resolve_provider(args.embed_provider)
                                               if args.embed else None))
        srv = MemoryServer(index=loader(),
                           thread_store=ThreadStore(args.threads),
                           corpus_dir=args.corpus,
                           weights_path=Path(args.corpus) / ".weights.json",
                           retractions_path=Path(args.corpus) / ".retractions.json",
                           entities_path=Path(args.corpus) / ".entities.json",
                           loader=loader)
        if args.http:
            host, _, port = args.http.rpartition(":")
            host = host or "127.0.0.1"
            token = args.token or os.environ.get("MEMORY_HTTP_TOKEN") or None
            try:
                httpd = make_http_server(srv, host=host, port=int(port), token=token)
            except ValueError as e:
                ap.error(str(e))
            # 起动横幅进 stderr（stdio 传输里 stdout 是协议流，这里沿用习惯）
            print(f"Streamable HTTP 服务在 http://{host}:{httpd.server_address[1]} "
                  f"（鉴权：{'Bearer token' if token else '无——仅限回环+外层反代'}；"
                  f"语料变化自动重读）", file=sys.stderr)
            # 非回环绑定多打一行 TLS 提醒：只提示、不拒绝起动，也不看有没有配 token
            notice = http_tls_notice(host)
            if notice:
                print(notice, file=sys.stderr)
            httpd.serve_forever()
        else:
            srv.serve_stdio()
    else:
        ap.print_help()
