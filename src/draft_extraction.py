#!/usr/bin/env python3
"""
自动提炼草稿流程参考实现（设计笔记"自动提炼草稿流程"+"风格片段候选机制"两节）。

对话transcript → 候选草稿（Field/StyleExcerpt，全部confirmed=False）→ 交给
persona_template.py的用户确认关卡把关（section 用其十二节骨架的 key，2026.07.31
A-G 通用骨架退役后同步改的）。本文件只负责"生成候选"，不负责"写入生效"——
两件事分开是设计笔记反复强调的底线：自动提炼不能等于无声写入。

两级门限，参考设计笔记 prior art（KI-CO的Topic Gate思路）：
  本地规则先筛（zero-dep，免费）→ 只有规则筛出的候选，才值得再花一次便宜LLM调用去精炼
  没有直接每条消息都烧LLM，机械性初筛留给本地规则做。

隐私二选一（设计笔记"自动提炼草稿流程"）：便宜LLM是可插拔回调（llm_call参数），
不硬编码具体某个API——调用方（产品层）决定传云端便宜LLM的回调还是本地小模型的回调，
或者干脆不传（llm_call=None）就纯用本地规则出候选，语料不出本机。这个选择权在用户，
不在这份参考实现里替用户拍板。

零依赖，stdlib only。
用法：
  python draft_extraction.py --selftest
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from persona_template import Field, StyleExcerpt, PersonaValidationError
from persona_compiler import CompileIssue, PersonaItem, SECTION_KEYS

# 本地规则打分用的信号词——不是详尽列表，是"够不够格再花一次LLM调用"的粗筛。
# 标准来自设计笔记任务5："能看出个性——吐槽/拒绝/不顺从/接得住玩笑，不是平铺直叙的问答"
_PERSONALITY_MARKERS = ("不", "别", "才", "凭什么", "谁说", "才不", "哼", "！", "？")

EXTRACTION_PROMPT = """# 人格候选提取任务

候选只许从输入来源提取，不许创作。
找不到就返回空数组，不得用通用句补完整度。
风格必须给完整多轮对话；学风格，不抄台词。
里程碑必须同时给发生了什么、怎么读、当前状态。

逐个读取《人格候选输入清单.json》里的语料。每个候选必须填写真实 `source_ref`、
字符区间 `source_span` 和逐字 `evidence`。每个输入来源都要在 `source_accounting` 中
登记候选 ID，或明确写 `no_supported_candidate: true`。只返回符合 schema 的 JSON。

⚠ `evidence` 必须是**原始字符**，不许做任何加工：不要 HTML 转义（`>` 就写 `>`，
不要写成 `&gt;`；`<`、`&` 同理），不要把直引号改成弯引号，不要改空白与换行。
校验器是**逐字**比对原文的，改一个字符就整条不过——而报出来的错是
「evidence 不是原文逐字」，看不出是转义干的。
"""


@dataclass(frozen=True)
class ExtractionPackage:
    prompt_path: Path
    manifest_path: Path
    schema_path: Path


def _candidate_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["items", "source_accounting"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["item_id", "text", "section", "source_ref",
                                 "source_span", "candidate_kind", "evidence"],
                    "properties": {
                        "item_id": {"type": "string"},
                        "text": {"type": "string"},
                        "section": {"enum": list(SECTION_KEYS)},
                        "source_ref": {"type": "string"},
                        "source_span": {
                            "type": "array", "prefixItems": [
                                {"type": "integer"}, {"type": "integer"}],
                            "minItems": 2, "maxItems": 2},
                        "candidate_kind": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                },
            },
            "source_accounting": {"type": "array"},
        },
        "additionalProperties": False,
    }


def build_extraction_package(manifest, out_dir):
    """导出给用户当前模型的本地任务包；不调用网络、不读取 API key。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "人格候选提取提示.md"
    manifest_path = out / "人格候选输入清单.json"
    schema_path = out / "人格候选结果.schema.json"
    prompt_path.write_text(EXTRACTION_PROMPT, encoding="utf-8")
    sources = [{
        "source_ref": str(path),
        "sha256": manifest.source_hashes.get(str(path), ""),
        "kind": "corpus",
    } for path in manifest.corpus_files]
    manifest_path.write_text(json.dumps({"sources": sources}, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    schema_path.write_text(json.dumps(_candidate_schema(), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return ExtractionPackage(prompt_path, manifest_path, schema_path)


def issue_codes(candidate):
    """返回单条候选的结构错误码；不依赖模型或外部 schema 库。"""
    kind = candidate.get("candidate_kind")
    issues = set()
    if kind == "milestone" and not all(
            str(candidate.get(key, "")).strip()
            for key in ("event", "reading", "current_state")):
        issues.add("MILESTONE_INCOMPLETE")
    if kind == "naming" and not (
            candidate.get("user_to_ai") and candidate.get("ai_to_user")):
        issues.add("NAMING_NOT_BIDIRECTIONAL")
    if kind == "style_dialogue":
        turns = candidate.get("turns") or []
        speakers = {str(turn.get("speaker", "")).strip() for turn in turns}
        if len(turns) < 2 or "" in speakers or len(speakers) < 2:
            issues.add("STYLE_NOT_DIALOGUE")
    return issues


def _issue(code, message, item_id=None, severity="blocking"):
    return CompileIssue(code, severity, message, (item_id,) if item_id else ())


def validate_candidate_result(result, manifest):
    """验证模型结果的结构、逐字证据和来源交代，返回安全候选与问题。"""
    issues = list(manifest.issues)
    items = []
    known_sources = {str(path): path for path in manifest.corpus_files}
    seen_ids = set()
    raw_items = result.get("items")
    if not isinstance(raw_items, list):
        return [], issues + [_issue("CANDIDATE_ITEMS_INVALID", "items 必须是数组")]

    for raw in raw_items:
        item_id = str(raw.get("item_id", ""))
        if not item_id or item_id in seen_ids:
            issues.append(_issue("CANDIDATE_ID_INVALID", "候选 ID 缺失或重复", item_id or None))
            continue
        seen_ids.add(item_id)
        structural = issue_codes(raw)
        if structural:
            issues.extend(_issue(code, f"候选结构不完整：{code}", item_id)
                          for code in sorted(structural))
            continue
        section = raw.get("section")
        if section not in SECTION_KEYS:
            issues.append(_issue("SECTION_UNKNOWN", f"未知人格节：{section}", item_id))
            continue
        ref = str(raw.get("source_ref", ""))
        path = known_sources.get(ref)
        if path is None:
            issues.append(_issue("SOURCE_UNKNOWN", f"候选来源不在输入清单：{ref}", item_id))
            continue
        span = raw.get("source_span")
        if (not isinstance(span, list) or len(span) != 2
                or not all(isinstance(value, int) for value in span)):
            issues.append(_issue("SOURCE_SPAN_INVALID", "来源区间必须是两个整数", item_id))
            continue
        source = path.read_text(encoding="utf-8")
        start, end = span
        if start < 0 or end < start or end > len(source):
            issues.append(_issue("SOURCE_SPAN_INVALID", "来源区间越界", item_id))
            continue
        excerpt = source[start:end]
        evidence = str(raw.get("evidence", ""))
        if not evidence or evidence not in excerpt:
            issues.append(_issue("EVIDENCE_NOT_VERBATIM", "证据不能逐字回指来源区间", item_id))
            continue
        kind = raw.get("candidate_kind")
        text = str(raw.get("text", ""))
        if kind == "quote" and text not in excerpt:
            issues.append(_issue("QUOTE_NOT_VERBATIM", "原话不在声明的来源区间", item_id))
            continue
        if kind == "style_dialogue" and any(
                str(turn.get("text", "")) not in excerpt for turn in raw.get("turns", ())):
            issues.append(_issue("STYLE_NOT_VERBATIM", "风格对话有轮次无法逐字回指", item_id))
            continue
        items.append(PersonaItem(
            item_id=item_id, text=text, section=section, source_type="corpus",
            source_ref=ref, source_span=(start, end),
            source_hash=manifest.source_hashes.get(ref, ""), operation="add",
            original_text=evidence, proposed_text=text,
            confidence=str(raw.get("confidence", "model_candidate")), confirmed=False,
            conflicts_with=tuple(raw.get("conflicts_with", ())),
            group_id=str(raw.get("group_id", f"corpus:{item_id}"))))

    accounting = result.get("source_accounting")
    accounted = set()
    if isinstance(accounting, list):
        for record in accounting:
            ref = str(record.get("source_ref", ""))
            if ref not in known_sources:
                issues.append(_issue("SOURCE_UNKNOWN", f"来源交代表含未知来源：{ref}"))
                continue
            candidate_ids = record.get("candidate_item_ids") or []
            no_candidate = record.get("no_supported_candidate") is True
            if not candidate_ids and not no_candidate:
                continue
            accounted.add(ref)
    missing = sorted(set(known_sources) - accounted)
    if missing:
        issues.append(_issue(
            "SOURCE_ACCOUNTING_INCOMPLETE",
            "以下输入来源没有候选或无候选说明：" + "；".join(missing)))
    if known_sources and not raw_items and not missing:
        issues.append(_issue(
            "NO_SPECIFIC_CANDIDATES", "所有来源均已交代，但没有可支持的具体候选",
            severity="warning"))
    return items, issues


def load_candidate_result(path, manifest):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_candidate_result(data, manifest)


def parse_transcript(text):
    """最简对话transcript解析：每行"说话人：内容"格式 → [(speaker, text), ...]。
    是设计笔记"通用导入"设想的中间格式({时间戳,说话人,文本,标签?})的简化子集，
    时间戳/标签留给真正接入具体导出格式（微信/ChatGPT json）时的翻译器层再补，
    这里只做提炼逻辑本身的参考实现，不重做"通用导入"那节已经想清楚的格式适配。"""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "：" not in line and ":" not in line:
            continue
        sep = "：" if "：" in line else ":"
        speaker, _, content = line.partition(sep)
        speaker, content = speaker.strip(), content.strip()
        if speaker and content:
            turns.append((speaker, content))
    return turns


def style_candidate_score(text):
    """本地规则打分：零依赖，不调任何模型。分越高越像"体现个性"的片段。
    纯粹是初筛门限——够格的候选才值得再花一次便宜LLM调用去精炼/验证。"""
    score = sum(1 for m in _PERSONALITY_MARKERS if m in text)
    if 4 <= len(text) <= 40:  # 太短没内容，太长不像"一句话立住"的锚点
        score += 1
    return score


def extract_style_candidates(turns, speaker, pool="daily", topN=5, llm_call=None):
    """从transcript里挑speaker说的话，按style_candidate_score排序取topN，
    包成StyleExcerpt候选（confirmed=False，disclaimer走dataclass默认值不能省）。

    llm_call: 可选回调 str->str，传入本地规则筛出的候选文本，返回精炼后的版本
    （比如更完整的上下文、更准的措辞）。不传就是纯本地规则出候选——这是"隐私二选一"
    落到代码里的样子：调用方决定要不要引入LLM这一步，这里不替调用方做选择。"""
    speaker_turns = [t for who, t in turns if who == speaker]
    ranked = sorted(speaker_turns, key=style_candidate_score, reverse=True)[:topN]
    candidates = []
    for text in ranked:
        if llm_call is not None:
            text = llm_call(text)
        candidates.append(StyleExcerpt(text=text, pool=pool, confirmed=False))
    return candidates


def draft_field(field_id, section, label, value, size_limit=500, llm_call=None):
    """单个字段草稿。llm_call同上——可选的精炼步骤，不传就原样打包成草稿。
    confirmed强制False：这里只产出候选，不产出生效内容。"""
    if llm_call is not None:
        value = llm_call(value)
    return Field(id=field_id, section=section, label=label, value=value,
                 size_limit=size_limit, source="draft", confirmed=False)


def apply_confirmed(persona, drafts, confirmed_ids):
    """把用户勾选过的草稿（按id匹配）真正写进persona——用户确认关卡的落地点。
    drafts可以是Field或StyleExcerpt的混合列表；StyleExcerpt没有id，按对象身份匹配。
    未被confirmed_ids选中的草稿直接丢弃，不留在系统里"等下次自动生效"。"""
    applied = []
    for d in drafts:
        is_field = isinstance(d, Field)
        matched = (is_field and d.id in confirmed_ids) or (not is_field and id(d) in confirmed_ids)
        if not matched:
            continue
        d.confirmed = True
        try:
            if is_field:
                persona.add_field(d)
            else:
                persona.add_style_excerpt(d)
            applied.append(d)
        except PersonaValidationError:
            d.confirmed = False  # E-gating等校验没过，草稿撤回未确认状态，不静默吞掉
            raise
    return applied


# ---------- selftest（合成对话，不含任何真实语料） ----------

_SYNTH_TRANSCRIPT = """
林岸：今天工作好累
星回：辛苦了，先歇会儿
林岸：不想歇，还有一堆事
星回：那也得吃饭，饿肚子干不动活
林岸：哼，你才是那个该早点睡的
星回：说得对，但现在说的是你
林岸：谁说我不睡了
星回：我说的，你昨天四点才睡
"""


def _selftest():
    turns = parse_transcript(_SYNTH_TRANSCRIPT)
    assert len(turns) == 8, f"该解析出8轮对话，实际{len(turns)}"
    assert turns[0] == ("林岸", "今天工作好累")

    # 1. 本地规则打分：带个性标记的句子分更高
    assert style_candidate_score("哼，你才是那个该早点睡的") > style_candidate_score("好的")

    # 2. 候选全部confirmed=False（靶心：草稿不能一生成就算生效）
    candidates = extract_style_candidates(turns, "星回", pool="daily", topN=3)
    assert all(not c.confirmed for c in candidates), "候选必须是未确认状态"
    assert all(c.disclaimer for c in candidates), "候选必须带disclaimer"

    # 3. llm_call可插拔：传入回调时候选文本被回调处理过
    refined = extract_style_candidates(turns, "星回", pool="daily", topN=1,
                                        llm_call=lambda t: f"[精炼]{t}")
    assert refined[0].text.startswith("[精炼]"), "llm_call该被应用到候选文本上"

    # 4. 不传llm_call就是纯本地规则，候选文本原样（隐私二选一：不引入LLM这条路走得通）
    local_only = extract_style_candidates(turns, "星回", pool="daily", topN=1)
    assert not local_only[0].text.startswith("[精炼]")

    # 5. draft_field同样强制confirmed=False、source=draft
    f = draft_field("nickname", "naming", "称呼", "小名")
    assert f.confirmed is False and f.source == "draft"

    # 6. apply_confirmed：只有被选中的id才真正写入persona
    from persona_template import Persona
    p = Persona("partner")
    f1 = draft_field("nick1", "naming", "称呼1", "小名甲")
    f2 = draft_field("nick2", "naming", "称呼2", "小名乙")
    applied = apply_confirmed(p, [f1, f2], confirmed_ids={"nick1"})
    assert len(applied) == 1 and applied[0].id == "nick1"
    assert [x.id for x in p.active_fields()] == ["nick1"], "只有确认过的字段该生效"

    # 7. apply_confirmed对intimacy风格片段也遵守E-gating（assistant类型该在这里失败）
    from persona_template import Persona as P2
    assistant = P2("assistant")
    ex = StyleExcerpt(text="不该出现", pool="intimacy")
    try:
        apply_confirmed(assistant, [ex], confirmed_ids={id(ex)})
        assert False, "assistant类型不该允许intimacy片段通过确认关卡"
    except PersonaValidationError:
        assert ex.confirmed is False, "校验失败后草稿该撤回未确认状态，不能悬空显示已确认"

    # 8. 候选不能只写一个存在的文件名；原话必须落在它声明的精确 span 内。
    import tempfile
    from pathlib import Path
    from persona_compiler import build_source_manifest
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus = root / "window_01.md"
        corpus.write_text("林岸：真实证据。\n星回：接住了。\n", encoding="utf-8")
        manifest = build_source_manifest(None, corpus)
        ref = str(corpus.resolve())
        result = {
            "items": [{
                "item_id": "corpus:1", "text": "你来，我就在。",
                "section": "closing", "source_ref": ref,
                "source_span": [0, 8], "candidate_kind": "quote",
                "evidence": "林岸：真实证据"
            }],
            "source_accounting": [{"source_ref": ref,
                                    "candidate_item_ids": ["corpus:1"]}],
        }
        items, issues = validate_candidate_result(result, manifest)
        assert any(issue.code == "QUOTE_NOT_VERBATIM" for issue in issues)
        assert not items

        # 变异：删掉任一结构要素，都不能把说明书当成里程碑／称呼／风格片段。
        assert issue_codes({"candidate_kind": "milestone", "event": "发生了",
                            "reading": "", "current_state": "已经结束"}) == {
                                "MILESTONE_INCOMPLETE"}
        assert "NAMING_NOT_BIDIRECTIONAL" in issue_codes({
            "candidate_kind": "naming", "user_to_ai": ["哥哥"], "ai_to_user": []})
        assert "STYLE_NOT_DIALOGUE" in issue_codes({
            "candidate_kind": "style_dialogue",
            "turns": [{"speaker": "assistant", "text": "一句"}]})

        package = build_extraction_package(manifest, root / "task")
        assert package.prompt_path.exists() and package.manifest_path.exists()
        assert package.schema_path.exists()
        prompt = package.prompt_path.read_text(encoding="utf-8")
        assert "候选只许从输入来源提取，不许创作" in prompt
        assert "找不到就返回空数组" in prompt

    print("selftest ok（8组断言：旧候选兼容 + 来源回指 + 结构契约 + 零依赖任务包）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
