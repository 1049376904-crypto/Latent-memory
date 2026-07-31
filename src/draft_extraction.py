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
import re

from persona_template import Field, StyleExcerpt, PersonaValidationError

# 本地规则打分用的信号词——不是详尽列表，是"够不够格再花一次LLM调用"的粗筛。
# 标准来自设计笔记任务5："能看出个性——吐槽/拒绝/不顺从/接得住玩笑，不是平铺直叙的问答"
_PERSONALITY_MARKERS = ("不", "别", "才", "凭什么", "谁说", "才不", "哼", "！", "？")


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

    print("selftest ok（7项断言全绿）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
