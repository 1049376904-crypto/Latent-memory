#!/usr/bin/env python3
"""
端到端冒烟（任务卡"端到端冒烟与快速上手"）。

十一个文件各自 selftest 全绿，防的是单元回归；这里防的是**接缝回归**——第二阶段
抓到的"三件套只出两件半"就是零件全绿、合起来漏的典型。整条链一口气跑：

  合成导出 json → 导入（load_any）→ 体检 → 出问卷 → 模拟模型答卷（文本）
  → parse_answer_sheet 解析 → 合并候选 → 逐条确认 → 三件套出货
  → load_corpus 回读 → MemoryServer 完整握手 → memory_search 命中导入原文
  → thread_close 往返

两条纪律：

  **走用户真实路径，不走测试捷径**——答案以文本清单喂 parse_answer_sheet，不直接
  构造 answers 字典。字典捷径绕过了用户真正会经过的解析层，那正是最容易
  "失败得像成功"的一段。

  **断言盯接缝，不重复单测**——mcp-config 的 --corpus 必须指向真写了语料的目录、
  出货语料回读零 mtime、检索命中的必须是导入原文。这些是跨文件约定，单测各自绿
  也可能合起来错。

写回也在链里（memory-writeback 合并后补上的一环）：memory_append → 当场可查 →
模拟重启 → 记录从盘上回来、命中过的块权重仍 >1.0（用进撑过重启）。

零依赖，stdlib only。合成语料全部虚构。
用法：
  python e2e_smoke.py --selftest
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from memory_import import load_any, entries_to_turns
from memory_init import (
    Persona, fill_protocol_defaults, coverage_report, questions_for,
    format_questionnaire, export_llm_prompt, parse_answer_sheet, answer_report,
    apply_answers, pending_confirmations, apply_confirmations, write_bundle,
)
from memory_retrieval import load_corpus
from mcp_server import MemoryServer, PROTOCOL_VERSION
from session_thread import ThreadStore

# ---------- 合成导出（全部虚构；结构照 Claude 导出的 chat_messages） ----------

_T0 = datetime(2026, 7, 20, 21, 0)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def _synthetic_export():
    """两个对话、隔天、各自成会话——含一个具体约定（去看鲸头鹳）和一次兑现，
    给检索层留下有名有姓的靶子。"""
    day2 = _T0.replace(day=22)
    return [
        {"name": "傍晚闲聊", "chat_messages": [
            {"sender": "human", "created_at": _iso(_T0),
             "text": "动物园新来了一只鲸头鹳，我看了直播，它一动不动站了十分钟。"},
            {"sender": "assistant", "created_at": _iso(_T0.replace(minute=2)),
             "text": "鲸头鹳就是这样的，站着不动是它的狩猎方式。想去现场看吗？"},
            {"sender": "human", "created_at": _iso(_T0.replace(minute=5)),
             "text": "想。那说好了，这个周末去看鲸头鹳。"},
        ]},
        {"name": "周末回来", "chat_messages": [
            {"sender": "human", "created_at": _iso(day2),
             "text": "看到鲸头鹳了！它冲我们缓慢鞠了一躬，饲养员说这是它打招呼。"},
            {"sender": "assistant", "created_at": _iso(day2.replace(minute=3)),
             "text": "约定兑现了。它鞠躬那下我记住了。"},
        ]},
    ]


def _fake_answer_sheet(questions):
    """模拟用户的模型整理回来的答案清单——纯文本，走真实解析路径。
    选择题一律选 A；pick/short 给一句极短原文。"""
    lines = []
    for i, q in enumerate(questions, 1):
        if q.kind in ("choice", "multi"):
            lines.append(f"{i}. A")
        else:
            lines.append(f"{i}. 你来，我就在。")
    return "\n".join(lines)


def _selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 1. 导入：合成导出落成文件，从文件读起——用户就是从文件开始的
        export_path = td / "conversations.json"
        export_path.write_text(json.dumps(_synthetic_export(), ensure_ascii=False),
                               encoding="utf-8")
        entries = load_any(export_path)
        assert len(entries) == 5 and entries_to_turns(entries), "导入翻译器该吃下合成导出"
        assert all(e.timestamp for e in entries), "ISO 时间戳该全部解析成功"

        # 2. 初始化：体检 → 问卷 → 模拟答卷 → 解析 → 合并 → 确认
        persona = Persona("partner")
        fill_protocol_defaults(persona)
        report = coverage_report(persona)
        qs = questions_for(report)
        assert qs, "冷启动该问出问题"
        assert format_questionnaire(qs) and export_llm_prompt(qs), "问卷两种导出都要有内容"
        sheet = _fake_answer_sheet(qs)
        answers, problems = parse_answer_sheet(sheet, qs)
        assert not problems, f"规范格式的答卷不该有读不懂的行：{problems}"
        assert answer_report(qs, answers, problems).startswith(f"读到 {len(qs)}/{len(qs)}")
        apply_answers(persona, qs, answers)
        pend = pending_confirmations(persona)
        assert pend, "问卷答案该以草稿形态等确认"
        apply_confirmations(persona, {p.key: "keep" for p in pend})
        assert not pending_confirmations(persona), "全部确认后不该有未决草稿"

        # 3. 出货三件套（含语料落盘）
        out = td / "bundle"
        got = write_bundle(out, persona, client="claude-code",
                           confirmed=True, entries=entries)
        assert got["persona"].name == "CLAUDE.md" and got["persona"].exists()
        assert got["corpus_files"], "语料该真的落进 memory/"
        md = got["persona"].read_text(encoding="utf-8")
        assert md.startswith("# ") and md.rstrip().endswith("你来，我就在。"), \
            "人格文件开头是标题、收尾是最终约定（顺序即权重）"
        #    第三轮真机实测（设计笔记）：人格文件里写明工具名的检索约定是主动性
        #    的主力杠杆——出货物必须带着它，且与被验证过的写法一致（点名工具）
        assert "memory_search" in md and "session_start" in md, \
            "检索约定与会话约定（主动性主力杠杆）必须在人格文件里，且点名工具"

        #    接缝断言①：mcp-config 的 --corpus 必须指向真写了语料的那个目录——
        #    config 指错目录时三件套各自看都正常，接上客户端才发现库是空的
        cfg = json.loads(got["mcp_config"].read_text(encoding="utf-8"))
        cfg_args = cfg["mcpServers"]["memory"]["args"]
        cfg_corpus = Path(cfg_args[cfg_args.index("--corpus") + 1])
        assert cfg_corpus == got["memory_dir"], \
            f"config 指向 {cfg_corpus}，语料实际在 {got['memory_dir']}"
        assert got["corpus_files"][0].is_relative_to(cfg_corpus), \
            "落盘的语料文件必须在 config 指向的目录下"

        # 4. 回读：接缝断言②——出货语料零 mtime 兜底（自己写的文件名都带日期）
        index = load_corpus(got["memory_dir"])
        assert index.chunks, "出货的记忆库回读不能是空的"
        assert all(m.get("timestamp_source") != "mtime" for m in index.meta), \
            "出货语料不该有任何块落 mtime 兜底"

        # 5. 上线：完整 MCP 握手 + 检索命中导入原文 + thread 往返。
        #    server 按生产接线配（corpus_dir + weights_path），跟 CLI 启动一致——
        #    冒烟测的就是用户真实路径，不配阉割版
        weights_path = got["memory_dir"] / ".weights.json"
        srv = MemoryServer(index=index, thread_store=ThreadStore(),
                           corpus_dir=got["memory_dir"], weights_path=weights_path)
        now = _T0.replace(day=23).timestamp()
        hs = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": PROTOCOL_VERSION,
                                    "clientInfo": {"name": "e2e"}}})
        assert hs["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        call = lambda name, a: srv.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": name, "arguments": a}}, now=now)["result"]
        hit = call("memory_search", {"query": "鲸头鹳的约定"})
        assert hit["isError"] is False, f"检索不该失败：{hit}"
        #    接缝断言③：命中的必须是导入语料的**原文**，不是任何中间加工物
        assert "这个周末去看鲸头鹳" in hit["content"][0]["text"], \
            "检索命中的该是导入对话的原话"
        closed = call("thread_close", {"window": 3, "current_state": "约定已兑现，这条线收尾。"})
        assert closed["isError"] is False and srv.thread_store.latest().window == 3

        # 6. 写回：memory_append → 当场可查 → "重启"后记录和权重都从盘上回来
        #    （memory-writeback 合并后补的一环——记忆库自己生长的那半支笔进链）
        wrote = call("memory_append",
                     {"text": "她说下次想去看企鹅漫步，最好挑个凉快的早上。",
                      "current_state": "新约定，档期还没定。"})
        assert wrote["isError"] is False, f"写回不该失败：{wrote}"
        hit2 = call("memory_search", {"query": "企鹅漫步"})
        assert hit2["isError"] is False and "凉快的早上" in hit2["content"][0]["text"], \
            "写回之后当场就要能查到原文"
        #    接缝断言④：模拟客户端重启 server（新进程 = 重新 load_corpus + 生产
        #    接线）——写回的记录该从盘上回来，检索命中过的块权重也该还在
        srv2 = MemoryServer(index=load_corpus(got["memory_dir"]),
                            thread_store=ThreadStore(),
                            corpus_dir=got["memory_dir"], weights_path=weights_path)
        #    权重检查必须在重启后的**第一次检索之前**——检索的副作用本身会加权，
        #    放后面的话断言会被本会话新加的权重救活，测不到"载入"这半（第一版
        #    就犯了这个错，变异"启动不载入权重"没红才发现）
        boosted = [w for w in srv2.index.weights if w > 1.0]
        assert boosted, "重启后、检索前，此前命中过的块权重就该 >1.0——用进要撑过重启"
        hit3 = srv2.handle(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "企鹅漫步"}}}, now=now)["result"]
        assert hit3["isError"] is False and "凉快的早上" in hit3["content"][0]["text"], \
            "重启后写回的记录该从盘上回来——不然'记住了'只活一个进程"

    print("selftest ok（端到端冒烟：导入 → 问卷 → 答卷解析 → 确认 → 三件套 → "
          "回读 → 握手 → 检索命中原文 → thread 往返 → 写回当场可查 → "
          "重启后记录与权重都在，接缝断言四处）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
