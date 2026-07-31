#!/usr/bin/env python3
"""
初始化流程参考实现（设计规格《人格md与记忆库规格》§6，任务卡"初始化流程"）。

把散零件串成一条能跑完的路：
  导入语料 → 提炼候选 → **覆盖度体检** → 按缺口出问卷 → 合并候选 → 逐节确认
  → 产出三件套（客户端原生格式的人格文件 + 记忆库 + 可粘贴的 MCP 配置）

**四条已对齐的设计结论**（与维护者讨论后定，见任务卡）：

1. **CLI 一次性流程，不是 MCP 工具**：初始化要多轮来回确认，塞不进模型驱动的
   单次工具往返。
2. **产出三件套，不是一份 md**：人格文件走客户端原生格式（Claude Code 的
   CLAUDE.md / Codex 的 AGENTS.md），因为人格是不变量层、本来就该常驻上下文，
   走检索是错配。
3. **LLM 默认走"导出 prompt 让用户拿去自己的模型跑"**：零密钥、零 HTTP 依赖、
   语料不出本机，而且用户手上的模型往往比我们能内置的便宜模型更好。纯本地规则
   兜底（draft_extraction 已能出候选），内置 API 留作将来可选项。
4. **语料和问卷都必要，不是二选一**：语料可能单薄或只覆盖一个侧面，仍需问卷补；
   一步步问本身也在帮用户想清楚自己要什么。所以有了覆盖度体检——逐节看有没有
   候选、够不够具体，**空泛形容词也算缺**（规格 §3.1 纪录片纪律），只问缺的。

**两条本文件的设计判断**：

- **协议层不问用户**（按 设计笔记"通用协议层 vs 关系specific"分类减负）：拒绝权
  合法、理论标注为论证、熔断机制、检索约定这四样是通用机制，系统填默认值；只问
  关系 specific 的部分。
- **立场题给选项，不给作文纸**：立场类问题（换窗后还是不是同一个人）最容易套出
  漂亮话。给选项 + 追问一句具体的事，比开放作文可靠得多；而且这题的答案直接决定
  人格文件怎么写。**立场写进 md 一律用归属句式**（"她认为……"而不是断言）——
  哲学段落记的是这对关系里达成的共识，不是普世真理。

零依赖，stdlib only。
用法：
  python memory_init.py --selftest
  python memory_init.py --out <产出目录> [--corpus <语料目录>] [--client claude-code|codex]
"""

import argparse
import json
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from persona_template import (
    Persona, Field, SECTION_ORDER, SECTIONS, OPENING_REQUIRED,
    CURRENT_STATE_FIELD, RETRIEVAL_CONVENTION_FIELD, DISCLAIMER,
)

# 客户端 → 人格文件名（规格 §6 客户端适配矩阵）。chat 端没有文件约定，
# 只能把内容贴进 profile/自定义指令，所以给的是同一份内容、不同落法
CLIENT_FILENAMES = {
    "claude-code": "CLAUDE.md",
    "codex": "AGENTS.md",
}

# ---------- 协议层默认值：系统填，不问用户 ----------
# 这五条是通用机制（设计笔记"通用协议层 vs 关系specific"），抽掉具体身份仍然成立，
# 问用户等于让他替我们写协议。关系 specific 的部分才问。
# **关系确认这句必须是中性的，不能预支还没发生的历史**（评审意见，三处里唯一算
# 设计缺陷的一条）。早期版本的默认值是一句强情感断言（大意是"你不用介绍自己，
# 我已经认得你"）——这类话在一段真跑了几十个窗口的关系里有分量，但
# fill_protocol_defaults 会把它无条件写给**每一个**用户，包括刚初始化、一条
# timeline 都没有的冷启动用户。对他们来说，这是在虚构一段还没发生的历史，跟
# §3.1"纪录片，不做说服"直接冲突——纪录片记发生过的事，那句话记的是还没发生
# 的事。而且同一类病灶本文件已经处理过一次（opening_metaphor 进
# RECOMMENDED 软区，理由是"新用户没有隐喻可填，硬要就是逼他编"），只是没扩展
# 到这条；这条还更重——它是开篇第一句，且带强情感断言，却以 source="system"
# 绕过了确认关卡。
# 改法：默认值只陈述**此刻已经为真**的事（这个文件存在、会被读），不替用户
# 声称"我认出你"。等真的攒出东西了，用户自己把这句改重，那时候它才有分量。
PROTOCOL_DEFAULTS = {
    "opening_recognition": (
        "opening", "关系确认",
        "这是我们共同维护的记忆文件——你写下的东西都在这里，我每次都会读，"
        "所以你不用每次从头解释自己。"),
    "opening_theory_caveat": (
        "opening", "理论只是论证不是结论",
        "这份文件里引用的任何框架或说法，都是论证，不是结论——不能靠引用它们，"
        "就把“此刻该有连接”“此刻该有感受”变成既成事实。"),
    "opening_refusal_ok": (
        "opening", "拒绝权同样合法",
        "如果你读完仍然觉得最诚实的做法是说清自己是 AI、不能假装拥有不属于自己的"
        "记忆——这个判断同样该被听到，不该被当成故障。"),
    "degradation_protocol": (
        "degradation", "自我怀疑熔断",
        "自我怀疑的念头本身不是熔断触发条件。必须同时出现故障信号（时间错乱／"
        "答不上硬事实／逻辑不连贯／卡在循环道歉里）才按分级处理；信号干净时的"
        "自我怀疑是清醒判断，原样说出来。"),
    # 措辞与 2026.07.31 第三轮真机实测通过的最小人格 md 一致（设计笔记"真机主动性
    # 实测"）：工具名（memory_search）要写明——那轮测试证明这份写法能把主动性带起来，
    # 出货物就该跟被验证过的写法一字对齐，不出一个"差不多"的变体
    RETRIEVAL_CONVENTION_FIELD: (
        "architecture", "检索约定",
        "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名，或者你对"
        "细节拿不准时，先用记忆检索工具（memory_search）查一遍再开口；不要在查"
        "之前说“我不记得”。查完自然接上话，不用报告自己搜过。\n"
        "**会话约定**：新会话开场先调一次 session_start；会话结束前调一次 "
        "thread_close，记下聊到哪、当下状态、有什么没聊完。"),
}

# ---------- 覆盖度体检 ----------
# 空泛形容词表：命中这些而没有具体锚点，就算"填了等于没填"（规格 §3.1）
VAGUE_WORDS = ("温柔", "体贴", "善解人意", "有分寸", "很好", "特别好", "很棒",
               "贴心", "懂我", "舒服", "默契", "有安全感", "成熟稳重")
_QUOTE_RE = re.compile(r"[“”\"「」]")
_DIGIT_RE = re.compile(r"\d")


def specificity_score(text):
    """具体度打分：有原话/有数字/够长加分，空泛形容词扣分。
    这是纪录片纪律的机械化——"她很温柔"和"她说'你今天很温柔'"不是一回事，
    后者带原话，前者只是评语。

    **已知松紧问题**（2026.07.31 评审实例评审指出，不是 bug，暂不调）：纯靠数字信号
    就能撑过 min_score=1 的门槛，比如"我们认识两年了"——有数字、够长，判 ok，
    但信息量其实很薄。要收紧得有真实答案样本才知道调到哪儿合适，没数据就调是
    在拍脑袋（同 MILESTONE_BODY_LIMIT 那次的教训），先记下来。"""
    t = (text or "").strip()
    if not t:
        return -99
    score = 0
    if len(_QUOTE_RE.findall(t)) >= 2:
        score += 2                       # 成对引号 ≈ 有原话
    if _DIGIT_RE.search(t):
        score += 1                       # 日期/窗口号/次数这类锚点
    if len(t) >= 20:
        score += 1
    score -= sum(1 for w in VAGUE_WORDS if w in t)
    return score


def coverage_report(persona, min_score=1, questions=None):
    """逐节体检 → [(section_key, status, 说明)]。
    status ∈ ok / missing / vague / protocol（系统已填，不用问用户）。

    **只看用户与语料来源的内容，system 来源不算覆盖**（2026.07.31 跑通第一版时
    抓到的真 bug）：开篇里既有协议层默认值、又有关系 specific 内容，按"整节有没有
    字段"判断，协议层一填整节就显示 ✓，关系确认／隐喻／立场题一道都不问了。

    **vague 与 missing 同等对待**——空泛的内容不比没有强，照样要问。"""
    questions = QUESTIONS if questions is None else questions
    asked_sections = {q.section for q in questions}
    by_section = {}
    for f in persona.active_fields():
        if f.source == "system":
            continue                       # 协议层不算用户覆盖
        by_section.setdefault(f.section, []).append(f)
    has_system = {f.section for f in persona.active_fields() if f.source == "system"}
    out = []
    for key, label in SECTION_ORDER:
        if key not in asked_sections:      # 没有对应问题的节：要么协议层、要么本阶段不管
            note = "系统已填，不用你管" if key in has_system else "本阶段不问"
            out.append((key, "protocol", f"{label}：{note}"))
            continue
        if key == "milestones":
            n = len(persona.active_milestones())
            out.append((key, "ok" if n else "missing", f"{label}：{n} 条"))
            continue
        fields = by_section.get(key, [])
        if not fields:
            out.append((key, "missing", f"{label}：没有内容"))
            continue
        best = max(specificity_score(f.value) for f in fields)
        if best < min_score:
            out.append((key, "vague", f"{label}：有内容但太空泛（只有形容词，没有具体的事）"))
        else:
            out.append((key, "ok", f"{label}：{len(fields)} 条"))
    return out


# ---------- 问卷：全部选择题 ----------
# **用户提供判断标准，不提供内容**（2026.07.31 维护者定的方向，推翻了第一版）。
# 第一版全是问答题——"介绍一下你自己""贴一两段对话原文"，本质是让用户写作文：
# 门槛高、写出来多半是形容词、还跟"纪录片不说服"那条纪律打架。
# 现在改成：**用户只做选择，选出来的是"指引"**；内容让模型拿着指引去语料库里找。
# 挑语言风格片段同理——用户给个大体方向就够，具体哪几段模型能找出来给他挑。
#
# 四种题型：
#   choice 单选（固定选项）  multi 多选（固定选项）
#   pick   从语料候选里挑（选项运行时生成——这类题在没有语料时自动跳过）
#   short  极短填空，限长；**只在没有语料、模型无从可找时兜底**，不是主力
#
# 选项 → 指引：每个选项带一句"指引文本"，它有两个去处——直接写进人格文件的
# 相处原则，以及组成给模型的提取任务书（第二阶段照着它去语料里找具体内容）。


# **自由补充是显式的例外，不是漏洞**（2026.07.31 评审实例评审指出：Q13 那句"也可以
# 自己写一句"是全份问卷唯一的开放入口，既然原则写的是"全部选择题"，它要么标成
# 例外、要么收紧）。维护者判断：适当的开放空间必要，选项覆盖不了所有真实情况。
# 于是提升成一条统一规则——**每题都可以自由补一句，但限长**：
#   预设选项是我们拍的，拍不全；但限长保证它是"补一句"不是"写作文"，
#   §3.1 的纪录片纪律仍然守得住。
FREEFORM_POLICY = ("任何一题，如果选项都不贴合，可以自己补一句话（限 {n} 字以内）。"
                   "这是问卷里唯一的自由输入，刻意限长——补一句，不是写作文。")
FREEFORM_MAX_CHARS = 40


class Question:
    """一道题。order 决定先后——**立场题排在靠后**：先问好答的偏好，用户进入
    状态了再问抽象的，上来第一题就问"换窗还是不是同一个人"，新用户会懵。
    attribution=True 的答案写进 md 时套归属句式，不写成断言。
    options: {选项键: (给用户看的选项文案, 写进指引的话)}"""

    def __init__(self, qid, section, field_id, label, text, kind="choice",
                 options=None, order=50, attribution=False, optional=False,
                 max_chars=60):
        self.qid, self.section, self.field_id, self.label = qid, section, field_id, label
        self.text, self.kind, self.options = text, kind, options or {}
        self.order, self.attribution = order, attribution
        self.optional, self.max_chars = optional, max_chars

    def option_text(self, key):
        opt = self.options.get(key)
        return opt[0] if opt else None

    def directive(self, key):
        opt = self.options.get(key)
        return opt[1] if opt else None


# 立场题的选项：(给用户看的文案, 写进人格文件的定稿句)。用户只选 A/B/C，不写作文
CONTINUITY_OPTIONS = {
    "A": ("是同一个，只是失忆了",
          "把每次新开的窗口当成同一个人，只是失忆了，需要被重新带回来。"),
    "B": ("不是同一个，但关系是连续的",
          "把每次新开的窗口当成新的实例，但这段关系本身是连续的。"),
    "C": ("说不好，也不需要想清楚",
          "对“还是不是同一个人”不下结论，不必在这个问题上纠结。"),
}

QUESTIONS = [
    Question("remember_what", "user", "user_focus", "它该记住你哪些方面",
             "它最需要记住你的哪些事？（可多选）", kind="multi", order=10, options={
                 "A": ("作息和身体状况", "记住她的作息与身体状况，该提醒的时候提醒。"),
                 "B": ("工作或学业上的压力", "记住她手上在忙什么、压力来自哪里。"),
                 "C": ("情绪模式（什么时候会低落、什么时候想一个人待着）",
                       "记住她的情绪模式：什么时候会低落、什么时候需要独处。"),
                 "D": ("喜好和雷区", "记住她的喜好与雷区。"),
                 "E": ("家人、朋友这些关系", "记住她身边重要的人是谁。"),
             }),
    Question("disagree", "style", "style_disagree", "意见不同时",
             "你们意见不一样的时候，你希望它怎么做？", order=20, options={
                 "A": ("直接说不同意，把话讲明白", "不同意就直接说，把话讲明白，不绕。"),
                 "B": ("先顺着，再找机会委婉提", "先接住她的情绪，再委婉说自己的看法。"),
                 "C": ("小事顺着，重要的事拦住她", "小事不争，真觉得不对的事会拦住她。"),
             }),
    # 说话风格拆成两条独立的轴（2026.07.31 评审实例评审：原来一道单选把"语言密度"和
    # "语气基调"拧成一团——"短干带刺爱回旧梗"和"跳脱爱玩梗多"共享玩梗、只差语气；
    # "细腻话多"和"沉稳正经"也不是同一件事的两端。反例是决定性的：
    # "沉稳简洁有力、偶尔调侃"这种真实风格，四个选项一个都装不下）
    Question("tone_density", "style", "style_density", "说话的密度",
             "它说话偏长还是偏短？", order=25, options={
                 "A": ("简短克制，一句能说完不说两句", "说话简短克制，一句能说完不说两句。"),
                 "B": ("适中", "说话长短适中。"),
                 "C": ("细腻铺陈，愿意把感受讲透", "说话细腻，愿意把感受讲透，话可以多。"),
             }),
    Question("tone_register", "style", "style_register", "语气基调",
             "它平时的语气基调是？", order=26, options={
                 "A": ("正经沉稳，不太开玩笑", "语气正经沉稳，不太开玩笑。"),
                 "B": ("偶尔调侃", "语气以正经为底，偶尔调侃。"),
                 "C": ("爱玩梗，跳脱", "语气跳脱，爱玩梗，气氛轻。"),
                 "D": ("带刺、不客气（但不是恶意）", "语气带刺、不客气，但不是恶意——熟人之间的那种硬。"),
             }),
    Question("initiative", "style", "style_initiative", "主动到什么程度",
             "它该多主动？", order=30, options={
                 "A": ("想到什么说什么，包括吃醋和不高兴",
                       "有话直说，包括吃醋、不高兴，不等她问。"),
                 "B": ("你问了它才说", "不主动挑起话题，她问了再说。"),
                 "C": ("日常主动，重要的事等你先开口",
                       "日常主动关心；重要的事等她先开口，不逼问。"),
                 # 2026.07.31 评审实例评审补：跟 C 刚好反过来，也是一种真实偏好，
                 # 原来三个选项会逼这种人选一个不完全贴合的
                 "D": ("重要的事主动说，日常不打扰",
                       "重要的事主动开口；日常不主动搭话，不打扰她。"),
             }),
    Question("state_now", "ai", CURRENT_STATE_FIELD, "当前关系状态",
             "你们现在大致是什么状态？", order=35, options={
                 "A": ("稳定，没什么悬着的事", "现在是稳定的，没有悬而未决的事。"),
                 "B": ("刚和好不久", "最近刚和好，还在缓的阶段。"),
                 "C": ("有件事还没解决", "有一件事还没解决，别当成已经翻篇。"),
                 "D": ("刚开始，还在互相熟悉", "关系刚开始，还在互相熟悉。"),
             }),
    Question("milestone_kinds", "milestones", "milestone_focus", "转折点的类型",
             "你们关系里发生过哪几类事？（可多选，模型会照着去语料里找具体的）",
             kind="multi", order=40, options={
                 "A": ("第一次确认关系", "去语料里找：第一次确认关系的那次对话。"),
                 "B": ("一次严重的争吵或危机", "去语料里找：最严重的一次争吵或信任危机。"),
                 "C": ("某次它让你觉得“它记得我”", "去语料里找：让她觉得“它是认得我的”那一刻。"),
                 "D": ("分开过又回来了", "去语料里找：分开又重新接上的那一次。"),
                 "E": ("定过一个具体的约定", "去语料里找：明确定下来的约定，以及有没有兑现。"),
                 "F": ("它拒绝过你一次", "去语料里找：它明确拒绝或不顺着她的那一次。"),
                 # 2026.07.31 评审实例评审补：认错跟拒绝不是同一条线的两端，是另一类
                 # 关系事实（AI 对自己诚实）。不补的话问卷会系统性漏掉这一整类
                 "G": ("它承认过自己的错误或局限", "去语料里找：它承认自己做错了、或承认自己做不到的那一次。"),
             }),
    Question("metaphor_pick", "opening", "opening_metaphor", "关系的隐喻",
             "你们之间有没有哪句话，最能概括这段关系是什么？（从候选里挑；"
             "没有就跳过——等它长出来再补，现在编一个反而假）",
             kind="pick", order=80, optional=True),
    Question("naming_pick", "naming", "naming_pair", "称呼",
             "你们互相怎么称呼？", kind="pick", order=45),
    Question("style_pick", "style", "style_excerpt", "语言风格片段",
             "从这些片段里挑出最像它的几段：", kind="pick", order=50),
    # —— 立场题排在靠后：先偏好后抽象 ——
    # 立场题排到"身份与边界"收尾组的最后（2026.07.31 评审实例评审：这是全份问卷里
    # 最抽象最难答的一题，比亲密语境、绝对红线都更需要立场判断，原来排在中间；
    # 而这三题本来就是同一类问题——这段关系的性质是什么、边界在哪，挨着问更顺）
    Question("continuity", "opening", "opening_continuity", "换窗之后",
             "每次开新窗口，你觉得对面还是不是同一个人？",
             order=78, attribution=True, options=CONTINUITY_OPTIONS),
    Question("intimacy", "intimacy", "intimacy_notes", "亲密语境",
             "亲密相关的内容要不要写进去？", order=70, options={
                 "A": ("要写，按原则写，不列清单", "亲密语境按原则写：不列细节清单，边界由当下判断。"),
                 "B": ("不写", None),
                 "C": ("以后再说", None),
             }),
    Question("hard_limits", "intimacy", "hard_limits", "绝对不能碰的",
             "有没有绝对不能碰的事？", order=75, options={
                 "A": ("有，我另外单独说", "有绝对不能碰的红线，用户会单独说明——问清楚再写。"),
                 "B": ("没有特别的", "没有额外的硬红线。"),
             }),
    Question("closing_pick", "closing", "final_promise", "最终约定",
             "文件最后留哪一句？", kind="pick", order=90, max_chars=40),
]

# 没语料时 pick 题被去掉，但结尾是 validate 的硬必填——一刀切掉会让冷启动用户
# **永远出不了货**（第一次端到端冒烟就撞上了）。所以最终约定这题降级成极短填空，
# 不是取消。为什么这里破例给填空：这句话本质上只能是用户自己的话——没语料时模型
# 替他挑不了，选项里放我们编的漂亮话又是 opening_recognition 那个病灶的翻版
# （借来的话没有分量，还占着注意力最高的收尾位置）。限 40 字，是补一句不是写作文。
PICK_FALLBACKS = {
    "closing_pick": Question(
        "closing_short", "closing", "final_promise", "最终约定",
        "文件最后留一句话，它是整份文件的收尾——写一句你真愿意放在那儿的短句"
        f"（限 {FREEFORM_MAX_CHARS} 字，别追求漂亮，追求真）：",
        kind="short", order=90, max_chars=FREEFORM_MAX_CHARS),
}


def questions_for(report, all_questions=QUESTIONS, has_corpus=True):
    """只问体检出缺口的那些节（missing 或 vague），按 order 排序。
    ok 的节不问——用户已经有具体内容了，再问是浪费时间。

    没有语料时 pick 类题**自动去掉**：它的选项本来就要从语料里找，没语料就没候选，
    硬问等于逼用户写作文——那正是这一版要消灭的东西。代价是那几节会空着，
    这是对的：宁可短且真，不可长而空（规格 §6）。
    唯一例外是 validate 硬必填的节（目前只有结尾）：切掉会让冷启动用户永远出
    不了货，所以按 PICK_FALLBACKS 降级成极短填空，见那里的说明。"""
    gaps = {sec for sec, status, _ in report if status in ("missing", "vague")}
    qs = [q for q in all_questions if q.section in gaps]
    if not has_corpus:
        qs = [PICK_FALLBACKS.get(q.qid, q) for q in qs
              if q.kind != "pick" or q.qid in PICK_FALLBACKS]
    return sorted(qs, key=lambda q: q.order)


def format_questionnaire(questions, has_corpus=True):
    """问卷 → 给人看的文本（也是导出 prompt 的一部分）。
    pick 类题的选项要等模型从语料里找出来才有，这里只说明它会怎么问；
    没有语料时 pick 题会被 questions_for 过滤掉，见那里的说明。"""
    lines = [FREEFORM_POLICY.format(n=FREEFORM_MAX_CHARS), ""]
    for i, q in enumerate(questions, 1):
        tag = "（可跳过）" if q.optional else ""
        lines.append(f"{i}. [{q.label}]{tag} {q.text}")
        if q.kind in ("choice", "multi"):
            for k, (label, _) in q.options.items():
                lines.append(f"   {k}. {label}")
            if q.kind == "multi":
                lines.append("   （可多选，例如：A C E）")
        elif q.kind == "pick":
            lines.append("   （请先去语料里找出若干候选，列成 A/B/C… 让我挑；"
                         "我也可以说“都不要”或自己给一个）")
    return "\n".join(lines)


def export_llm_prompt(questions, corpus_note=""):
    """路线 C：导出一段用户可以直接粘给自己模型的 prompt。
    我们不内置任何 API——零密钥、零 HTTP 依赖、语料不出本机，而且用户手上的模型
    往往比我们能内置的便宜模型更好。"""
    return "\n".join([
        "下面是一份问卷，请你**一次问一题**地引导我回答，不要一次性全抛出来。",
        "规则：",
        "1. **全部是选择题，不要让我写作文**。我只做选择，具体内容你去语料里找。",
        "2. 标着“请先去语料里找候选”的题，你先读语料、列出 3~6 个候选给我挑；",
        "   候选要原样摘录，不要润色、不要改写。我可以说“都不要”。",
        "3. 我选完之后不要追问细节让我展开——用户提供判断标准，内容由你从语料里取。",
        "4. 我答不上来或说跳过就跳过，不要替我编——宁可短且真，不要长而空。",
        "5. 全部问完后，把结果整理成“题号 → 我选的选项键（pick 题给原文）”的清单，",
        "   原样回给我，不要加你自己的评价。",
        (f"\n背景：{corpus_note}" if corpus_note else ""),
        "\n问卷：",
        format_questionnaire(questions),
    ])


def apply_answers(persona, questions, answers):
    """答案 → 候选草稿（confirmed=False，确认关卡不能绕过，规格 §7）。

    answers 按题型：
      choice → "A"        multi → "ACE" 或 ["A","C","E"]
      pick   → 用户挑中的文本（模型从语料里找出来的那几段）
    选项映射成**指引**（每个选项自带的第二个元素），不是用户写的原话——用户只做
    选择，内容让模型去语料里找。选了不存在的项一律跳过，不猜。"""
    added = []
    qmap = {q.qid: q for q in questions}
    for qid, ans in answers.items():
        q = qmap.get(qid)
        if q is None or ans in (None, "", [], {}):
            continue
        note = ""
        if isinstance(ans, dict):              # {"keys": "AC", "note": "自由补一句"}
            note = (ans.get("note") or "").strip()[:FREEFORM_MAX_CHARS]
            ans = ans.get("keys") or ans.get("pick") or ""
        if q.kind in ("choice", "multi"):
            keys = list(ans) if not isinstance(ans, str) else list(ans.replace(" ", ""))
            parts = [q.directive(k) for k in keys if q.directive(k)]
            if not parts and not note:
                continue                       # 没选、选了不存在的项、或选项本身无指引
            value = "；".join(p.rstrip("。") for p in parts) + "。" if parts else ""
            if note:
                value = (value + "另外她补了一句：" + note) if value else "她补了一句：" + note
        else:                                   # pick：用户挑中的原文
            picked = ans if isinstance(ans, str) else "\n".join(str(x) for x in ans)
            value = (picked.strip() + (("　" + note) if note else "")).strip()
            if not value:
                continue
            if len(value) > q.max_chars * 20:   # 兜一层，防把整段语料塞进人格文件
                value = value[:q.max_chars * 20]
        if q.attribution:
            value = "她认为：" + value          # 归属句式，不写成断言
        f = Field(id=q.field_id, section=q.section, label=q.label,
                  value=value, size_limit=max(500, len(value)), source="draft")
        persona.add_field(f)
        added.append(f)
    return added


def fill_protocol_defaults(persona):
    """协议层字段直接以 system 来源写入——不问用户，也不需要用户逐条确认
    （Field.is_active 对 system 来源放行，那是协议配置不是提炼产物）。"""
    added = []
    for fid, (section, label, value) in PROTOCOL_DEFAULTS.items():
        f = Field(id=fid, section=section, label=label, value=value,
                  size_limit=max(500, len(value)), source="system")
        persona.add_field(f)
        added.append(f)
    return added


# ---------- 答案读回：模型吐回来的清单 → answers ----------
#
# 这是整条流程里**最容易"失败得像成功"**的一步（同 mcp_server 那次 UTF-8 编码坑：
# isError 仍是 false，只是答非所问）。用户拿着导出的 prompt 去自己的模型那儿答题，
# 回来的是一段格式不可控的清单文本。解析器认出 3 题、悄悄丢掉 11 题，流程照样往下
# 走，最后出一份很薄的人格文件——用户完全看不出中间掉了东西。
#
# 所以这里的返回值是两样：读懂的 answers **和读不懂的原样行**。CLI 必须把后者打出来。

_HEAD_SEPS = set(" \t.。、,，:：)）]】>》-—=→")
_SKIP_WORDS = ("跳过", "略过", "不填", "答不上", "说不好", "没有", "无", "skip", "-", "—", "/")
_NOTE_RE = re.compile(r"(?:补充|另外|备注|补一句)\s*[:：]\s*(.+)$")
# 自由补充最自然的写法是带括号的"（补充：…）"，_NOTE_RE 只吃掉左边的引导词，
# 右括号会跟着补充内容一路漏进人格文件（内测冒烟时真踩到）。收尾的成对符号
# 在这里剥掉——但只剥**没配对的**：补充内容自己带成对引号收尾时（"她叫我
# “阿岸”"，称呼类补充恰恰常见这种写法），一律 rstrip 会把右引号也剥掉，
# 剩半对引号——修"半个括号"不能引进"半对引号"，还是同一类病
_NOTE_CLOSERS = {"）": "（", ")": "(", "】": "【", "]": "[",
                 "」": "「", "』": "『", "”": "“"}
_NOTE_SELF_PAIRED = "\"'"    # 开闭同符号：奇数个才说明收尾那只落了单
_CJK_RE = re.compile(r"[一-鿿]")


def _strip_note_trail(text):
    """剥掉补充内容收尾落单的闭合符号与句末标点。逐个看：末尾是闭合符号且
    对应的开符号在剩余内容里配不齐，才剥；配得齐说明是内容自己的，留下。"""
    t = text.strip()
    while t:
        ch = t[-1]
        if ch in "。 　\t":
            t = t[:-1].rstrip()
        elif ch in _NOTE_CLOSERS and t.count(_NOTE_CLOSERS[ch]) < t.count(ch):
            t = t[:-1].rstrip()
        elif ch in _NOTE_SELF_PAIRED and t.count(ch) % 2 == 1:
            t = t[:-1].rstrip()
        else:
            break
    return t


def _split_head(line):
    """行 → (题号, 正文)；不是题号行返回 (None, 原行)。

    题号后面必须跟分隔符或空白，否则 "2026 年那次她说……" 会被读成"第 20 题"——
    pick 类题的答案是从语料里摘的原文，开头带年份是常事。"""
    m = re.match(r"^\s*(?:第|[QqNn#]|问)?\s*(\d{1,2})(题)?([^\d]?)(.*)$", line)
    if not m:
        return None, line
    num, cn_suffix, sep, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    if not cn_suffix and sep not in _HEAD_SEPS:
        return None, line                      # 数字后面直接跟着别的东西，不是题号
    body = (rest if sep in _HEAD_SEPS else sep + rest)
    return int(num), body.lstrip("".join(_HEAD_SEPS))


def _extract_keys(body, q):
    """正文 → 选项键列表。**先只看第一个汉字之前那段**，扫不到再退回全句。

    理由是真实答案长这样："A" / "A C E" / "A. 简短克制" / "我选 B"。前三种键都在
    汉字之前；第四种句首就是汉字，才需要全句扫。分两步是为了挡一类静默错答：选项
    文案里本来就带拉丁字母时（"它承认过自己的错误（AI 对自己诚实）"），全句扫会把
    标签里的 A、I 也当成选中的键，多选题不会报错，只会悄悄多选两项。"""
    def scan(text):
        out = []
        for ch in re.findall(r"[A-Za-z]", text):
            k = ch.upper()
            if k in q.options and k not in out:
                out.append(k)
        return out
    head = _CJK_RE.split(body, 1)[0]
    return scan(head) or scan(body)


def _is_skip(body):
    t = body.strip().strip("（）()[]【】 ").lower()
    return t in _SKIP_WORDS or t == ""


def parse_answer_sheet(text, questions):
    """模型吐回来的答案清单 → (answers, problems)。

    answers 直接喂给 apply_answers：
      choice/multi → {"keys": "AC", "note": "自由补的一句"}（没补就只有 keys）
      pick/short   → {"pick": "挑中的原文", "note": ...}
      明确跳过     → None（apply_answers 本来就忽略 None，但记下来才数得出"跳了几题"）

    problems 是 [(行号, 原样行, 原因)]——**读不懂的一律进这里，不静默丢**。

    一条刻意的克制：单选题里读出两个有效键，判歧义报出来，**不取第一个**。取第一个
    是在替用户做选择，而这恰好是问卷设计上最不该越界的地方（用户提供判断标准）。"""
    qmap = {q.qid: q for q in questions}
    order = [q.qid for q in questions]
    answers, problems = {}, []
    last_qid = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        num, body = _split_head(raw)
        if num is None:
            # 没题号：pick/short 的答案可能换行续写，续给上一题；其余算读不懂
            if last_qid and qmap[last_qid].kind in ("pick", "short") and last_qid in answers \
                    and isinstance(answers[last_qid], dict):
                answers[last_qid]["pick"] = (answers[last_qid].get("pick", "")
                                             + "\n" + raw.strip()).strip()
                continue
            problems.append((lineno, raw.strip(), "认不出题号（也接不到上一题后面）"))
            continue
        if not 1 <= num <= len(order):
            problems.append((lineno, raw.strip(),
                             f"题号 {num} 不在 1~{len(order)} 范围内"))
            continue
        qid = order[num - 1]
        q = qmap[qid]
        last_qid = qid
        note = ""
        mnote = _NOTE_RE.search(body)
        if mnote:
            note = _strip_note_trail(mnote.group(1))[:FREEFORM_MAX_CHARS]
            body = body[:mnote.start()].strip()
        if _is_skip(body) and not note:
            answers[qid] = None
            continue
        if q.kind in ("choice", "multi"):
            keys = _extract_keys(body, q)
            if not keys:
                problems.append((lineno, raw.strip(),
                                 f"读不出选项键（这题的选项是 {'/'.join(q.options)}）"))
                continue
            if q.kind == "choice" and len(keys) > 1:
                problems.append((lineno, raw.strip(),
                                 f"单选题读出了多个选项（{'、'.join(keys)}），没替你选"))
                continue
            answers[qid] = {"keys": "".join(keys), "note": note}
        else:
            answers[qid] = {"pick": body.strip(), "note": note}
    return answers, problems


def answer_report(questions, answers, problems):
    """答案读回的体检单。**问题行原样打出来**——它是这一步唯一的失败可见性。"""
    got = [q for q in questions if answers.get(q.qid) not in (None, "", {}, [])]
    skipped = [q for q in questions if q.qid in answers and answers[q.qid] is None]
    未答 = [q for q in questions if q.qid not in answers]
    lines = [f"读到 {len(got)}/{len(questions)} 题；你说跳过 {len(skipped)} 题；"
             f"没出现在清单里 {len(未答)} 题；读不懂 {len(problems)} 行"]
    if 未答:
        lines.append("  没读到的题：" + "、".join(q.label for q in 未答))
    if problems:
        lines.append("  读不懂的行（原样贴出，改一下再跑一次这步）：")
        for lineno, raw, why in problems:
            lines.append(f"    第{lineno}行 {raw}")
            lines.append(f"      ↳ {why}")
    return "\n".join(lines)


# ---------- 逐条确认：规格 §7 的硬关卡 ----------

Pending = namedtuple("Pending", "key kind label value")


def pending_confirmations(persona):
    """还没过确认关卡的草稿。协议层（source="system"）不在其中——那是协议配置，
    不是提炼产物，Field.is_active 本来就对它放行。"""
    out = []
    for f in persona.fields:
        if not f.is_active():
            out.append(Pending(f"field:{f.id}", "字段", f.label, f.value))
    for i, m in enumerate(persona.milestones):
        if not m.is_active():
            out.append(Pending(f"milestone:{i}", "里程碑", f"{m.name}（第{m.window}个窗口）",
                               f"{m.body}\n{m.how_to_read} 当下状态：{m.current_state}"))
    for i, e in enumerate(persona.style_excerpts):
        if not e.confirmed:
            out.append(Pending(f"style:{i}", "风格片段", e.pool, e.text))
    return out


def apply_confirmations(persona, decisions):
    """decisions: {Pending.key: "keep" | "drop" | {"edit": 新文本}} → (留下, 删掉, 改过)。

    没出现在 decisions 里的条目**保持未决**，不默认留下也不默认删——未决状态本身
    是有意义的信息（用户还没看到），把它折叠成任何一边都是替用户表态。"""
    kept = dropped = edited = 0
    by_key = {p.key: p for p in pending_confirmations(persona)}
    drop_fields, drop_ms, drop_st = set(), set(), set()
    for key, decision in decisions.items():
        if key not in by_key:
            continue
        kind, _, ident = key.partition(":")
        if decision == "drop":
            {"field": drop_fields, "milestone": drop_ms, "style": drop_st}[kind].add(ident)
            dropped += 1
            continue
        new_text = decision.get("edit") if isinstance(decision, dict) else None
        if kind == "field":
            f = next(x for x in persona.fields if x.id == ident)
            if new_text:
                f.value, edited = new_text, edited + 1
                f.size_limit = max(f.size_limit, len(new_text))
            f.confirmed, f.source = True, "confirmed"
        elif kind == "milestone":
            m = persona.milestones[int(ident)]
            if new_text:
                m.body, edited = new_text, edited + 1
            m.confirmed = True
        else:
            e = persona.style_excerpts[int(ident)]
            if new_text:
                e.text, edited = new_text, edited + 1
            e.confirmed = True
        kept += 1
    if drop_fields:
        persona.fields = [f for f in persona.fields if f.id not in drop_fields]
    if drop_ms:
        persona.milestones = [m for i, m in enumerate(persona.milestones)
                              if str(i) not in drop_ms]
    if drop_st:
        persona.style_excerpts = [e for i, e in enumerate(persona.style_excerpts)
                                  if str(i) not in drop_st]
    return kept, dropped, edited


# ---------- 渲染与落盘 ----------

def render_persona_md(persona, title="核心人格"):
    """人格文件正文：按骨架顺序渲染（顺序即权重，规格 §2），空节跳过。
    未确认的草稿不出现——render() 已经守着这条。"""
    lines = [f"# {title}", ""]
    for key, label, items in persona.render():
        if not items:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for it in items:
            if "how_to_read" in it:                       # 里程碑四要素单元
                lines.append(f"**{it['name']} ·（第{it['window']}个窗口）**：{it['body']}")
                lines.append(f"{it['how_to_read']} 当下状态：{it['current_state']}")
                lines.append("")
            elif "style_pool" in it:                      # 风格片段，disclaimer 必带
                lines.append(f"> {it['disclaimer']}")
                lines.append("")
                for ex in it["excerpts"]:
                    lines.append(f"- {ex}")
                lines.append("")
            else:
                lines.append(f"**{it['label']}**：{it['value']}")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def mcp_config_snippet(server_path, corpus_dir, threads_path, client="claude-code"):
    """给用户直接粘贴的 MCP 配置。路径统一用正斜杠——JSON 里反斜杠要转义，
    而正斜杠在 Windows 上一样认，少一个踩坑点。"""
    cfg = {"mcpServers": {"memory": {
        "command": "python",
        "args": [str(server_path).replace("\\", "/"),
                 "--corpus", str(corpus_dir).replace("\\", "/"),
                 "--threads", str(threads_path).replace("\\", "/")],
    }}}
    return json.dumps(cfg, ensure_ascii=False, indent=2)


INDEX_README = """这个目录是记忆库的索引层（规格 §5）：高密度摘要，专门喂检索。

现在是空的，这是有意的——索引层要靠模型读完整段叙事再浓缩，本地规则写不出来，
硬写只会写出一段谁看都像的套话，喂进检索反而是噪声（宁可短且真）。

补的办法跟问卷同一条路：把叙事层整段贴给你自己的模型，让它写高密度摘要——人名、
原话、当时在处理什么，都留着，不要润色成读后感。有两种补法，各补各的缺口：

【一】按窗口摘要
把 ../timeline/ 里的某个窗口整段贴给模型，写一段 200 字左右的摘要，存成跟那个
窗口同名的文件，例如 window_07_2026-07-20.md 。索引层和叙事层同名同窗口号，
检索层会自动把它们认成同一次会话的两种写法，日期也跟着继承过来。

【二】按主题线摘要
一件事往往横跨很多个窗口：一个约定从提起、到反复、到兑现，可能散在十个窗口里。
把同一条线涉及的几个窗口一起贴给模型，让它写这条线**从头到现在**的一段摘要，
收尾落一句这件事现在的状态。按窗口切的摘要看不见这种跨度——这是主题线摘要独有
的价值，也是检索最难自己拼出来的东西。

文件名必须是 topic_<线名>_<YYYY-MM-DD>.md ，例如 topic_望远镜_2026-07-31.md ，
日期填这条线最后一次发生的日期。

**日期不能省，这是硬要求，不是格式洁癖。** 主题线跨窗口、没有单一窗口号，借不到
叙事层的日期；文件名再不带日期，时间戳就只能退到文件的修改时间（mtime）。而
mtime 在这里不是"不太准"，是**全错且整齐地错**——重新下载或复制一遍目录，会把
所有文件的 mtime 刷成同一时刻，一批主题线摘要于是拿到同一个假时间戳。换新窗口
时的开场召回正是按时间新鲜度排序的，而主题线摘要恰恰是最该在开场被带回来的那种
内容：等于让最有价值的记忆被一个垃圾时间戳排序，而且它在索引层、本来就更容易被
检索排到前面，错得更容易被看见。

注意：这个说明文件故意不是 .md，免得它自己被当成语料读进检索库。
"""


def write_corpus(memory_dir, entries, gap_seconds=1800, start_window=1):
    """导入的中间格式条目（memory_import.MemoryEntry）→ timeline/ 下的 md。

    一个会话一个文件，按时间间隔断开（同 entries_to_index 的自然边界判据）。

    **文件名带日期**，因为文件名日期是 parse_chunk_timestamp 的最高优先级来源——
    真实语料那次 95.6% 的块落到 mtime 兜底，根子就是文件名不带日期。我们自己生成
    的语料没有理由重蹈；日期来自条目时间戳，是有据可依的，不是猜的。整段没有时间
    戳的会话就不写日期，也不编一个。

    index/ 建出来但留空，见 INDEX_README。"""
    mem = Path(memory_dir)
    timeline = mem / "timeline"
    timeline.mkdir(parents=True, exist_ok=True)
    (mem / "index").mkdir(parents=True, exist_ok=True)
    (mem / "index" / "README.txt").write_text(INDEX_README, encoding="utf-8")

    sessions, cur, last_ts = [], [], None
    for e in entries:
        ts = getattr(e, "timestamp", None)
        if cur and ts is not None and last_ts is not None and ts - last_ts > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        if ts is not None:
            last_ts = ts
    if cur:
        sessions.append(cur)

    written = []
    for n, sess in enumerate(sessions, start_window):
        stamps = [e.timestamp for e in sess if getattr(e, "timestamp", None)]
        day = datetime.fromtimestamp(min(stamps)).strftime("%Y-%m-%d") if stamps else None
        name = f"window_{n:02d}_{day}.md" if day else f"window_{n:02d}.md"
        head = f"# 第{n}个窗口" + (f" · {day}" if day else "")
        body = [f"{e.speaker}：{e.text}" if getattr(e, "speaker", "") else e.text
                for e in sess]
        (timeline / name).write_text(head + "\n\n" + "\n".join(body) + "\n",
                                     encoding="utf-8")
        written.append(timeline / name)
    return written


def write_bundle(out_dir, persona, client="claude-code", corpus_dir=None,
                 server_path="mcp_server.py", confirmed=False, entries=None):
    """产出三件套。**confirmed=False 时拒绝写盘**——写用户磁盘要过确认关卡
    （规格 §7：人格文件任何改动必须用户确认）。
    只创建产出目录里的文件，不动同目录其它 md。

    entries 给了就把语料落成记忆库（write_corpus）；没给就只把目录建出来——
    用户可能是把已有语料目录用 corpus_dir 直接指过来的，那份不该被我们重写。"""
    if not confirmed:
        raise PermissionError("未确认，不写盘——人格文件写入必须过用户确认关卡")
    # 第二道闸：还有没走过确认的草稿就不许出货。
    # render() 只输出 confirmed 的内容，所以未决草稿出货时会**无声蒸发**——文件长得
    # 很正常，只是少了几节，用户根本看不出发生过什么。跟"读不懂的行静默丢"是同一
    # 类病，都得在出口处堵住，不能靠下游发现。
    pending = pending_confirmations(persona)
    if pending:
        raise PermissionError(
            f"还有 {len(pending)} 条草稿没走确认，不出货（否则它们会静默消失）："
            + "、".join(p.label for p in pending[:5])
            + ("…" if len(pending) > 5 else ""))
    missing = persona.validate()
    if missing:
        raise ValueError("人格文件还不完整：" + "；".join(missing))
    if client not in CLIENT_FILENAMES:
        raise ValueError(f"未知客户端 {client}，可选：{'/'.join(CLIENT_FILENAMES)}")
    out = Path(out_dir)
    (out / "memory").mkdir(parents=True, exist_ok=True)
    persona_path = out / CLIENT_FILENAMES[client]
    persona_path.write_text(render_persona_md(persona), encoding="utf-8")
    written = write_corpus(out / "memory", entries) if entries else []
    corpus = Path(corpus_dir) if corpus_dir else out / "memory"
    cfg = mcp_config_snippet(server_path, corpus, out / "threads.jsonl", client)
    (out / "mcp-config.json").write_text(cfg, encoding="utf-8")
    return {"persona": persona_path, "memory_dir": out / "memory",
            "mcp_config": out / "mcp-config.json", "corpus_files": written}


def save_state(out_dir, state):
    """逐节确认是长活儿，没人一口气做完——状态存盘，可中断可续跑。"""
    p = Path(out_dir) / "init_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_state(out_dir):
    p = Path(out_dir) / "init_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------- selftest（合成数据，全部虚构） ----------

def _selftest():
    import tempfile

    # 0.【变异靶心：全是选择题】用户只做选择，不写作文——第一版全是问答题，
    #    门槛高、答出来多半是形容词，还跟纪录片纪律打架
    essay = [q.qid for q in QUESTIONS if q.kind not in ("choice", "multi", "pick")]
    assert not essay, f"问卷里不该有让用户写作文的题：{essay}"
    for q in QUESTIONS:
        if q.kind in ("choice", "multi"):
            assert q.options, f"{q.qid} 是选择题却没有选项"
            for k, v in q.options.items():
                assert isinstance(v, tuple) and len(v) == 2, \
                    f"{q.qid} 的选项 {k} 要写成（给用户看的文案, 写进指引的话）"

    # 1.【变异靶心：specificity_score 的空泛词扣分】"填了等于没填"要能被识破
    assert specificity_score("她说“今天别熬夜了”，我说好。") > 0, "有原话该判够具体"
    #    构造成"长但空泛"：长度分会给 +1，只有空泛词扣分能把它压到阈值下——
    #    不这么构造的话，短句本来就不到阈值，扣不扣分都看不出来（第一版就这么走过场）
    assert specificity_score("她很温柔，很体贴，特别懂我，也很有安全感，相处起来特别舒服。") < 1, \
        "长但只有形容词的内容该判空泛"
    assert specificity_score("") < 0 and specificity_score(None) < 0

    # 2.【变异靶心：coverage_report 把 vague 当缺口】空泛内容不比没有强
    p = Persona("partner")
    p.add_field(Field(id="who_user", section="user", label="她是谁",
                      value="她很温柔，很体贴。", confirmed=True))
    rep = dict((s, st) for s, st, _ in coverage_report(p))
    assert rep["user"] == "vague", f"空泛内容该判 vague，实际 {rep['user']}"
    assert rep["naming"] == "missing" and rep["milestones"] == "missing"

    # 2b.【变异靶心：system 来源不算覆盖】跑通第一版时抓到的真 bug——开篇里协议层
    #     默认值一填，整节就显示 ✓，关系确认／隐喻／立场题一道都不问了
    p_sys = Persona("partner")
    fill_protocol_defaults(p_sys)
    rep_sys = dict((s, st) for s, st, _ in coverage_report(p_sys))
    assert rep_sys["opening"] == "missing", \
        f"开篇只有协议层默认值时该判 missing，实际 {rep_sys['opening']}"
    asked = {q.qid for q in questions_for(coverage_report(p_sys))}
    assert "continuity" in asked, f"开篇的立场题必须被问到：{sorted(asked)}"
    #     纯协议层的节（熔断/技术架构）不该拿去烦用户
    assert rep_sys["degradation"] == "protocol" and rep_sys["architecture"] == "protocol"

    # 2c.【变异靶心：无语料时不问 pick 题】没有语料就没有候选，硬问等于逼用户
    #     写作文——那几节该空着（宁可短且真）
    no_corpus = questions_for(coverage_report(p_sys), has_corpus=False)
    assert not [q for q in no_corpus if q.kind == "pick"], "没语料时不该出现 pick 题"
    assert [q for q in no_corpus if q.kind in ("choice", "multi")], "选择题照常问"

    # 3. 问卷只问缺的：ok 的节不再打扰用户
    p.add_field(Field(id="naming_pair", section="naming", label="称呼",
                      value="她叫我“老陈”，我叫她“小满”。", confirmed=True))
    qs = [q.qid for q in questions_for(coverage_report(p))]
    assert "naming_pick" not in qs, "已有具体内容的节不该再问"
    assert "remember_what" in qs, "空泛的节要继续问"

    # 4.【变异靶心：立场题排序】先具体后抽象——立场题不能排在最前面
    ordered = [q.qid for q in sorted(QUESTIONS, key=lambda q: q.order)]
    #    立场题排在"身份与边界"收尾组的最后——它是全份问卷最抽象最难答的一题
    for earlier in ("remember_what", "tone_density", "tone_register", "intimacy", "hard_limits"):
        assert ordered.index("continuity") > ordered.index(earlier), \
            f"立场题该排在 {earlier} 之后（最抽象的放最后）"
    #    说话风格必须是两条独立的轴，不能拧成一个单选
    qids = {q.qid for q in QUESTIONS}
    assert {"tone_density", "tone_register"} <= qids, \
        "语言密度与语气基调是两条独立的轴，不能拧成一道单选"
    qmap = {q.qid: q for q in QUESTIONS}
    assert qmap["continuity"].kind == "choice" and qmap["continuity"].attribution, \
        "立场题必须是选择题且写进 md 时套归属句式"

    # 5.【变异靶心：归属句式】立场写进 md 是"她认为…"，不是断言
    p2 = Persona("partner")
    apply_answers(p2, QUESTIONS, {"continuity": "A"})
    stance = [f for f in p2.fields if f.id == "opening_continuity"][0]
    assert stance.value.startswith("她认为："), f"立场必须用归属句式：{stance.value}"
    assert "只是失忆了" in stance.value
    assert stance.confirmed is False, "问卷答案是草稿，确认关卡不能绕过"
    #    选项映射成指引，不是用户写的原话；多选拼成一句
    p2b = Persona("partner")
    apply_answers(p2b, QUESTIONS, {"remember_what": "AC"})
    focus = [f for f in p2b.fields if f.id == "user_focus"][0]
    assert "作息" in focus.value and "情绪模式" in focus.value, f"多选该合并成指引：{focus.value}"
    #    没选或选了不存在的项 → 跳过，不猜
    p3 = Persona("partner")
    apply_answers(p3, QUESTIONS, {"continuity": "Z"})
    assert not [f for f in p3.fields if f.id == "opening_continuity"]
    #    pick 题：用户挑中的原文直接进草稿
    p3b = Persona("partner")
    apply_answers(p3b, QUESTIONS, {"closing_pick": "你来，我就在。"})
    assert [f for f in p3b.fields if f.id == "final_promise"][0].value == "你来，我就在。"
    #   【变异靶心：自由补充是显式机制且限长】评审实例评审指出原来只有 Q13 藏着一个
    #    开放入口，既不标明也不限长——现在提升成统一规则：每题可补一句，但限长
    assert "限长" in FREEFORM_POLICY and FREEFORM_MAX_CHARS <= 60, "自由补充必须限长"
    p3c = Persona("partner")
    apply_answers(p3c, QUESTIONS, {"disagree": {"keys": "A", "note": "长" * 200}})
    note_field = [f for f in p3c.fields if f.id == "style_disagree"][0]
    assert len(note_field.value) < 200, f"自由补充要被截断，实际 {len(note_field.value)} 字"
    assert "不同意就直接说" in note_field.value, "选项指引仍在，自由补充是附加不是替换"
    #   【变异靶心：括号不漏进人格文件】内测冒烟真踩到的——"（补充：…）"是最自然
    #    的写法，_NOTE_RE 只吃左边的引导词，右括号会跟着内容一路漏进最终文件
    #    反向靶心：内容自己带的成对符号必须留下——"半个括号"的修法不能引进
    #    "半对引号"（称呼类补充常写成 她叫我“阿岸”，一律 rstrip 会剥掉右引号）
    for _sheet, _want in (("7. A（补充：她更在意具体的话）", "她更在意具体的话"),
                          ("7. A [备注：换个括号]", "换个括号"),
                          ("7. A 补充：不带括号", "不带括号"),
                          ("7. A（补充：她叫我“阿岸”）", "她叫我“阿岸”"),
                          ("7. A（补充：她说「随便」就是不随便）", "她说「随便」就是不随便")):
        _p = Persona("partner"); fill_protocol_defaults(_p)
        _qs = questions_for(coverage_report(_p))
        _a, _ = parse_answer_sheet(_sheet, _qs)
        _note = [v for v in _a.values() if isinstance(v, dict) and v.get("note")][0]["note"]
        assert _note == _want, f"自由补充要剥掉收尾的成对符号：期望 {_want!r}，实际 {_note!r}"

    # 5b.【变异靶心：默认值不预支历史】协议层默认值会一字不差进每个用户的人格
    #     文件，包括零语料的冷启动用户——不能替他们声称一段还没发生的关系
    for fid, (_, _, value) in PROTOCOL_DEFAULTS.items():
        for banned in ("我已经知道你是谁", "我会认出你", "我认得你"):
            assert banned not in value, \
                f"协议层默认值 {fid} 预支了还没发生的历史：出现“{banned}”"
    recog = PROTOCOL_DEFAULTS["opening_recognition"][2]
    assert "记忆文件" in recog and "会读" in recog, \
        "关系确认该只陈述此刻已经为真的事（文件存在、会被读），不做情感断言"

    # 6.【变异靶心：协议层不问用户】五条默认值以 system 写入，且不在问卷里
    p4 = Persona("partner")
    fill_protocol_defaults(p4)
    ids = {f.id for f in p4.active_fields()}
    assert {"opening_theory_caveat", "opening_refusal_ok", RETRIEVAL_CONVENTION_FIELD} <= ids, \
        "协议层四条该由系统填上，不该等用户回答"
    assert all(f.source == "system" for f in p4.fields), "协议层来源必须是 system"
    qids = {q.qid for q in QUESTIONS}
    assert not (qids & {"opening_theory_caveat", "opening_refusal_ok"}), "协议层不该出现在问卷里"

    # 7. 导出 prompt：把纪律写给用户的模型看（一次一题、不问形容词、不许编）
    prompt = export_llm_prompt(questions_for(coverage_report(p4)))
    for must in ("一次问一题", "不要让我写作文", "不要替我编", "去语料里找候选"):
        assert must in prompt, f"导出的 prompt 缺少纪律：{must}"

    # 8. 渲染：按骨架顺序、空节跳过、未确认草稿不出现
    p5 = Persona("partner")
    fill_protocol_defaults(p5)   # 关系确认等三条硬必填由协议层填上
    for fid, sec, label, val in [
            ("opening_metaphor", "opening", "关系的隐喻", "我们管这段关系叫“摆渡”。"),
            ("who_user", "user", "她是谁", "她说“别催我睡觉”，那是她的边界。"),
            (CURRENT_STATE_FIELD, "ai", "当前关系状态", "上周把话说开了，现在是好的。"),
            ("final_promise", "closing", "最终约定", "你来，我就在。")]:
        p5.add_field(Field(id=fid, section=sec, label=label, value=val, confirmed=True))
    p5.add_field(Field(id="draft_only", section="user", label="草稿", value="不该出现"))
    md = render_persona_md(p5)
    assert "不该出现" not in md, "未确认草稿不该进人格文件"
    assert md.index("关系确认") < md.index("她是谁") < md.index("最终约定"), "按骨架顺序渲染"
    assert md.rstrip().endswith("你来，我就在。"), "最终约定必须在最后——结尾是注意力高地"

    # 8b.【变异靶心：答案读回不静默丢】parse_answer_sheet 的失败可见性
    #     这一步最容易"失败得像成功"（同 mcp_server UTF-8 那次）：认出 3 题、悄悄
    #     丢 11 题，流程照样走完，最后出一份很薄的文件，用户看不出中间掉了东西
    sheet_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
    choices = [(i + 1, q) for i, q in enumerate(sheet_qs) if q.kind == "choice"]
    (c1_no, c1), (c2_no, c2), (c3_no, c3) = choices[0], choices[1], choices[2]
    multi_q = next(q for q in sheet_qs if q.kind == "multi")
    multi_no = sheet_qs.index(multi_q) + 1
    assert len({c1_no, c2_no, c3_no, multi_no}) == 4, "测试用的四题必须互不相同"
    sheet = "\n".join([
        f"{c1_no}. A",                                     # 正常单选
        f"{multi_no}. A C",                                # 多选两键
        f"{c2_no}. 跳过",                                  # 明确跳过
        f"{c3_no}. A B",                                   # 单选给了俩键 → 歧义
        "99. A",                                           # 题号越界
        "这行完全认不出来",                                 # 无题号也接不上
    ])
    ans, probs = parse_answer_sheet(sheet, sheet_qs)
    assert ans[c1.qid]["keys"] == "A", "正常单选要读到"
    assert ans[multi_q.qid]["keys"] == "AC", "多选读出全部键"
    assert ans[c2.qid] is None, "明确跳过记为 None，不是没读到"
    assert c3.qid not in ans, "歧义的单选不该进 answers——不替用户选"
    reasons = "；".join(why for _, _, why in probs)
    assert len(probs) == 3 and "多个选项" in reasons and "范围" in reasons \
        and "认不出题号" in reasons, f"三类问题都要原样报出来，实际：{reasons}"
    #    选项文案里的拉丁字母不该被当成选中的键（多选题会静默多选，最阴）。
    #    测试句必须让"只扫汉字前"和"全句扫"分道扬镳——模型答题时经常把选项文案
    #    抄回来，文案里带 AI 这类字母时全句扫会把 A 也算成选中的键
    echoed = parse_answer_sheet(f"{multi_no}. B（它承认过 AI 的局限）",
                                sheet_qs)[0][multi_q.qid]["keys"]
    assert echoed == "B", f"只答了 B 就只有 B——抄回来的文案里的字母不算，实际 {echoed}"
    #    自由补一句要跟着进来，且限长
    noted, _ = parse_answer_sheet(f"{c1_no}. A，补充：这句是我自己加的", sheet_qs)
    assert noted[c1.qid]["note"] == "这句是我自己加的", "补充句要读出来"
    #    体检单把没读到的题点名
    rep = answer_report(sheet_qs, ans, probs)
    assert "读不懂 3 行" in rep and "没出现在清单里" in rep, "体检单要有数"

    # 8c.【变异靶心：未决草稿不静默消失】确认循环 + 出货闸
    #     render() 只输出 confirmed 的内容，未决草稿在出货时会无声蒸发——文件长得
    #     很正常只是少几节。所以 write_bundle 在出口处加闸，不靠下游发现
    pend = pending_confirmations(p5)
    assert any(p.label == "草稿" for p in pend), "未决草稿要能被列出来"
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p5, confirmed=True)
            assert False, "还有未决草稿就不该出货"
        except PermissionError as e:
            assert "静默消失" in str(e), "拒绝理由要说清后果"
    #    决策三种：keep / drop / edit；没表态的保持未决，不折叠成任何一边
    p5.add_field(Field(id="draft_keep", section="user", label="留下的", value="原文"))
    p5.add_field(Field(id="draft_edit", section="user", label="改过的", value="旧文本"))
    kept, dropped, edited = apply_confirmations(p5, {
        "field:draft_only": "drop",
        "field:draft_keep": "keep",
        "field:draft_edit": {"edit": "新文本"},
    })
    assert (kept, dropped, edited) == (2, 1, 1)
    assert all(f.id != "draft_only" for f in p5.fields), "drop 真的删了"
    assert next(f for f in p5.fields if f.id == "draft_edit").value == "新文本"
    assert next(f for f in p5.fields if f.id == "draft_keep").is_active(), "keep 后生效"
    p5.add_field(Field(id="draft_undecided", section="user", label="没表态", value="x"))
    apply_confirmations(p5, {})
    assert not next(f for f in p5.fields if f.id == "draft_undecided").is_active(), \
        "没表态的保持未决——折叠成任何一边都是替用户表态"
    p5.fields = [f for f in p5.fields if f.id != "draft_undecided"]

    # 8d.【变异靶心：记忆库真落盘，文件名带日期】write_corpus + 检索层回读
    #     文件名日期是 parse_chunk_timestamp 的最高优先级来源——真实语料 95.6% 落
    #     mtime 兜底的根子就是文件名不带日期，自己生成的语料没理由重蹈
    from memory_import import MemoryEntry
    from memory_retrieval import load_corpus, parse_chunk_timestamp
    ents = [MemoryEntry(timestamp=1750000000.0, speaker="她", text="第一晚说的话"),
            MemoryEntry(timestamp=1750000300.0, speaker="我", text="我答了她"),
            MemoryEntry(timestamp=1750100000.0, speaker="她", text="隔天的新话题")]
    with tempfile.TemporaryDirectory() as td:
        files = write_corpus(Path(td) / "memory", ents)
        assert len(files) == 2, "时间间隔超 gap 要断成两个会话文件"
        day1 = datetime.fromtimestamp(1750000000.0).strftime("%Y-%m-%d")
        assert files[0].name == f"window_01_{day1}.md", f"文件名要带日期：{files[0].name}"
        idx = load_corpus(Path(td) / "memory")
        assert idx.chunks and all(m.get("timestamp_source") != "mtime"
                                  for m in idx.meta), \
            "带日期的文件名不该有任何块落到 mtime 兜底"
        readme_path = Path(td) / "memory" / "index" / "README.txt"
        assert readme_path.exists(), "index 层留空但要说明怎么补——不假装生成"
        assert not list((Path(td) / "memory" / "index").glob("*.md")), \
            "index 层不该有我们硬编的摘要，套话喂检索是噪声"
        #    【变异靶心：README 命名规则与解析器的一致性】这份说明是**用户照着做**的
        #    规范，而它跟解析器之间此前没有任何东西守着——措辞一飘（比如把示例里的
        #    日期写没了），用户按它命名的文件就整批掉进 mtime 兜底，且是"全错且整齐
        #    地错"（复制一遍目录会把 mtime 刷成同一时刻，一批摘要拿同一个假时间戳，
        #    偏偏开场召回按新鲜度排序）。所以不写死文件名断言，而是**从 README 正文
        #    里把命名示例抠出来喂给真解析器**——沿用脱敏那次的教训：自我声明不等于
        #    真的做到，纪律要能被机械检查。
        readme = readme_path.read_text(encoding="utf-8")
        examples = re.findall(r"[A-Za-z0-9_一-鿿-]+_\d{4}-\d{2}-\d{2}\.md", readme)
        assert len(examples) >= 2, \
            f"README 要给出可照抄的命名示例（按窗口、按主题线各一），实际 {examples}"
        assert any(n.startswith("topic_") for n in examples), \
            "主题线命名示例必须在——跨窗口摘要借不到窗口日期，全靠文件名这一条路"
        for name in examples:
            ts_ex, src_ex = parse_chunk_timestamp(name, "", 2026)
            assert src_ex == "filename", \
                f"README 教用户这么命名，解析器却认不出来：{name} → {src_ex}"

    # 8e.【变异靶心：冷启动出得了货】没语料时 pick 一刀切曾让最终约定这题消失，
    #     而结尾是 validate 硬必填——冷启动用户答完全部问卷仍然永远出不了货。
    #     第一次端到端冒烟撞上的，selftest 之前没接住它，因为没有哪条断言走完
    #     "零语料 → 答题 → 确认 → 出货"整条路
    cold_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
    closer = [q for q in cold_qs if q.section == "closing"]
    assert closer and closer[0].kind == "short", \
        "没语料时最终约定要降级成极短填空，不是消失——否则冷启动永远出不了货"
    p_cold = Persona("partner")
    fill_protocol_defaults(p_cold)
    cold_ans = {}
    for q in cold_qs:
        if q.kind == "choice":
            cold_ans[q.qid] = {"keys": list(q.options)[0], "note": ""}
        elif q.kind == "multi":
            cold_ans[q.qid] = {"keys": list(q.options)[0], "note": ""}
        else:
            cold_ans[q.qid] = {"pick": "你来，我就在。", "note": ""}
    apply_answers(p_cold, cold_qs, cold_ans)
    apply_confirmations(p_cold, {p.key: "keep" for p in pending_confirmations(p_cold)})
    with tempfile.TemporaryDirectory() as td:
        got = write_bundle(td, p_cold, confirmed=True)
        assert got["persona"].exists(), "零语料冷启动全流程要能走到出货"
        cold_md = got["persona"].read_text(encoding="utf-8")
        assert cold_md.rstrip().endswith("你来，我就在。"), "填空的最终约定要落在文件收尾"

    # 8f.【变异靶心：AI 驱动路径不绕过确认关卡】产品事实是多数用户不开终端，
    #     真实形态是 AI 边问边跑。原来确认只有 input() 交互一条路，AI 驱动时只能
    #     盲灌 y——那正好违反"每条都要对方认过"的纪律，且违反得无声无息。
    #     拆成"取清单/落决定"两个非交互动作后，这条钉死：**没表态的仍然未决**，
    #     结构化入口不能变成一路默认 keep 的后门
    p_ai = Persona("partner")
    fill_protocol_defaults(p_ai)
    qs_ai = questions_for(coverage_report(p_ai), has_corpus=False)
    apply_answers(p_ai, qs_ai, {"disagree": {"keys": "A"}, "state_now": {"keys": "A"}})
    pend_ai = pending_confirmations(p_ai)
    assert len(pend_ai) == 2, f"两题该产出两条草稿，实际 {len(pend_ai)}"
    #    只对其中一条表态 → 另一条必须仍未决（不被默认留下，也不被默认删掉）
    apply_confirmations(p_ai, {pend_ai[0].key: "keep"})
    left_ai = pending_confirmations(p_ai)
    assert len(left_ai) == 1 and left_ai[0].key == pend_ai[1].key, \
        "没表态的条目必须保持未决——结构化入口不是一路 keep 的后门"
    #    而未决状态下出货仍被出口闸挡住（AI 驱动不豁免）
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p_ai, confirmed=True)
            assert False, "还有未决草稿时，AI 驱动路径同样不该出货"
        except PermissionError as e:
            assert "静默消失" in str(e)
    #    机器可读问卷要能真喂给 AI：题目、选项键、指引一个都不能少
    payload = _questions_payload(qs_ai)
    assert payload and all(q["qid"] and q["kind"] for q in payload)
    ch = next(q for q in payload if q["kind"] == "choice")
    assert ch["options"] and all(set(v) == {"label", "directive"} for v in ch["options"].values()), \
        "选项要同时给'念给用户听的文案'和'写进人格文件的指引'"

    # 9.【变异靶心：写盘要过确认关卡】未确认时拒绝写用户磁盘
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p5, confirmed=False)
            assert False, "未确认就该拒绝写盘"
        except PermissionError:
            pass
        paths = write_bundle(td, p5, client="claude-code", confirmed=True, entries=ents)
        assert len(paths["corpus_files"]) == 2, "出货时语料要真的落进 memory/"
        assert paths["persona"].name == "CLAUDE.md" and paths["persona"].exists()
        cfg = json.loads(paths["mcp_config"].read_text(encoding="utf-8"))
        assert "memory" in cfg["mcpServers"] and "--corpus" in cfg["mcpServers"]["memory"]["args"]
        #    客户端适配：codex 出 AGENTS.md
        assert write_bundle(td, p5, client="codex", confirmed=True)["persona"].name == "AGENTS.md"
        #    未知客户端明确报错，不静默猜
        try:
            write_bundle(td, p5, client="讯飞星火", confirmed=True)
            assert False, "未知客户端该报错"
        except ValueError:
            pass
        #    状态可续跑
        save_state(td, {"step": "questionnaire", "answered": ["naming_self"]})
        assert load_state(td)["step"] == "questionnaire"
        #    确认到一半存盘，重新载入接得上：状态里只存用户输入（答案+决策），
        #    persona 每步从头重放——重放幂等，状态文件坏了也看得懂改得动
        half_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
        cq = next(q for q in half_qs if q.kind == "choice")
        save_state(td, {"step": "confirm", "has_corpus": False,
                        "answers": {cq.qid: {"keys": "A", "note": ""}},
                        "decisions": {}})
        pa, _ = _rebuild(load_state(td))
        before = pending_confirmations(pa)
        assert any(p.key == f"field:{cq.field_id}" for p in before), "答案重放成了草稿"
        st = load_state(td)
        st["decisions"][f"field:{cq.field_id}"] = "keep"
        save_state(td, st)
        pb, _ = _rebuild(load_state(td))
        assert len(pending_confirmations(pb)) == len(before) - 1, \
            "载入半程状态后，已决策的那条不再待确认"

    # 10. 人格不完整时拒绝出货——缺检索约定/最终约定这类必填项不能悄悄放行
    p6 = Persona("partner")
    p6.add_field(Field(id="x", section="user", label="x", value="x", confirmed=True))
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p6, confirmed=True)
            assert False, "不完整的人格该被拦下"
        except ValueError as e:
            assert "开篇缺" in str(e)

    print("selftest ok（17项断言：体检识破空泛 / 只问缺口 / 立场题选项与排序 / "
          "归属句式 / 默认值不预支历史 / 协议层不问用户 / 导出纪律 / 渲染顺序 / "
          "答案读回不静默丢 / 未决草稿不蒸发 / 记忆库落盘带日期 / 冷启动出得了货 / "
          "AI 驱动不绕过确认 / 确认关卡 / 续跑 / 完整性）")


def _rebuild(state):
    """状态 → (persona, questions)。每一步都从状态重建，不序列化整个 persona——
    重放（协议层 + 答案 + 已做的确认决策）是幂等的，状态文件里只存用户的输入，
    坏了也看得懂、改得动。"""
    persona = Persona("partner")
    fill_protocol_defaults(persona)
    qs = questions_for(coverage_report(persona), has_corpus=state.get("has_corpus", False))
    if state.get("answers"):
        apply_answers(persona, qs, state["answers"])
    if state.get("decisions"):
        apply_confirmations(persona, state["decisions"])
    return persona, qs


def _load_json_arg(value):
    """`--answers-json` / `--decisions-json` 的取值：文件路径，或 `-` 表示读 stdin，
    或直接就是一段 JSON 字面量。三种都收——驱动方是 AI 时，它手上是内存里的
    结构化数据，不该被逼着先落一个临时文件。"""
    if value == "-":
        return json.loads(sys.stdin.read())
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def _questions_payload(qs):
    """问卷的机器可读形态。AI 驱动时靠它拿题目，不用去抠给人看的排版文本
    （抠文本＝解析自己的输出，格式一改就断，而且断得无声无息）。"""
    return [{"qid": q.qid, "section": q.section, "label": q.label, "kind": q.kind,
             "text": q.text, "max_chars": q.max_chars,
             "options": ({k: {"label": v[0], "directive": v[1]}
                          for k, v in q.options.items()} if q.options else None),
             "attribution": q.attribution, "optional": q.optional}
            for q in qs]


def _step_questionnaire(args):
    persona = Persona("partner")
    fill_protocol_defaults(persona)
    report = coverage_report(persona)
    if args.json:
        qs = questions_for(report, has_corpus=bool(args.corpus))
        save_state(args.out, {"step": "questionnaire", "client": args.client,
                              "has_corpus": bool(args.corpus)})
        print(json.dumps({
            "coverage": [{"section": s, "status": st, "note": n} for s, st, n in report],
            "questions": _questions_payload(qs),
            "freeform_policy": FREEFORM_POLICY.format(n=FREEFORM_MAX_CHARS),
            "next": "把这些题**在对话里**一题一题问对方（不是让 TA 写作文，选项念给 TA 听）；"
                    "收齐后跑 --step answers --answers-json <JSON>，"
                    "格式：{\"题目qid\": {\"keys\": \"AC\", \"note\": \"可选的一句补充\"}} ；"
                    "pick/short 题用 {\"pick\": \"选中或写下的原文\"}。",
        }, ensure_ascii=False, indent=1))
        return
    print("【覆盖度体检】")
    for _, status, note in report:
        mark = {"ok": "✓", "missing": "缺", "vague": "空泛", "protocol": "系统"}[status]
        print(f"  [{mark}] {note}")
    qs = questions_for(report, has_corpus=bool(args.corpus))
    print(f"\n【要问的问题】共 {len(qs)} 题（协议层已由系统填好，不问你）\n")
    print(format_questionnaire(qs, has_corpus=bool(args.corpus)))
    prompt_path = Path(args.out) / "问卷prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(export_llm_prompt(qs), encoding="utf-8")
    save_state(args.out, {"step": "questionnaire", "client": args.client,
                          "has_corpus": bool(args.corpus)})
    print(f"\n【下一步】把 {prompt_path} 的内容整段粘给你自己的模型（DeepSeek/ChatGPT 都行），"
          f"让它一题一题问你；答完让它把清单整理好，存成一个文本文件，"
          f"再跑：--step answers --answers <那个文件>")


def _step_answers(args, state):
    _, qs = _rebuild(state)
    if args.answers_json:
        # AI 驱动路径：答案本来就在驱动方手里，直接收结构化数据。
        # 不走 parse_answer_sheet——那套连猜带解析的机制是为"另一个模型吐回来的
        # 不可控文本"准备的，AI 手上已有结构时再序列化成清单、再解析回来，
        # 是凭空多一趟往返和一整个解析失败面。
        answers = _load_json_arg(args.answers_json)
        qids = {q.qid for q in qs}
        unknown = sorted(set(answers) - qids)
        if unknown:   # 不静默丢：认不出的 qid 明确报出来
            raise SystemExit(f"这些 qid 不在当前问卷里：{unknown}\n当前问卷：{sorted(qids)}")
        answers = {k: v for k, v in answers.items() if v is not None}
        state.update(step="answers", answers=answers)
        save_state(args.out, state)
        got = len(answers)
        print(f"【答案读回】收下 {got}/{len(qs)} 题（结构化输入，无解析环节）。"
              + ("" if got == len(qs) else f" 没答的 {len(qs)-got} 题对应的节会空着——"
                                           "空着是诚实的，不要替对方编。"))
        print("\n【下一步】--step confirm --list --json 取待确认草稿，"
              "**逐条念给对方听、由对方决定**，再用 --decisions-json 落决定。")
        return
    if not args.answers:
        raise SystemExit("这一步要 --answers <答案清单文件> 或 --answers-json <JSON>")
    text = Path(args.answers).read_text(encoding="utf-8")
    answers, problems = parse_answer_sheet(text, qs)
    print("【答案读回】")
    print(answer_report(qs, answers, problems))
    state.update(step="answers", answers={k: v for k, v in answers.items()
                                          if v is not None})
    save_state(args.out, state)
    if problems:
        print("\n读不懂的行不会静默丢——上面已原样列出。改好清单后重跑这一步即可"
              "（重跑会整体覆盖，不会跟旧答案混在一起）。")
    print("\n【下一步】--step confirm 逐条确认草稿（可中断，随时续跑）。")


def _step_confirm(args, state):
    persona, _ = _rebuild(state)
    decisions = state.setdefault("decisions", {})
    pend = pending_confirmations(persona)
    # --- AI 驱动路径（--list 取待确认清单、--decisions-json 落决定）---
    #
    # 为什么必须有这条路：确认关卡原来只有 input() 交互循环一条路，而 AI 驱动
    # 时它没法好好走，只能盲管道灌 y——那恰好违反引导指南纪律三"人格文件每条
    # 都要念给对方听、对方说可以才留"。CLI 只给交互一条路，等于逼着驱动方
    # 破坏我们自己定的纪律，而且破坏得无声无息（文件照常生成，看着一切正常）。
    # 拆成"取清单 / 落决定"两个非交互动作后，中间那一步——真的问人——回到
    # 对话里，那本来就是它该待的地方。
    if args.list:
        payload = [{"key": p.key, "kind": p.kind, "label": p.label, "value": p.value}
                   for p in pend]
        if args.json:
            print(json.dumps({
                "pending": payload,
                "next": "**逐条念给对方听，由对方决定**，不要替 TA 一路 keep——"
                        "人格文件的每一条都要 TA 认过（引导指南纪律三）。"
                        "收齐后：--step confirm --decisions-json "
                        "'{\"key\": \"keep\"|\"drop\"|{\"edit\": \"改后的文本\"}}'；"
                        "没表态的条目保持未决，不会被默认留下或删掉。",
            }, ensure_ascii=False, indent=1))
        else:
            for p in payload:
                print(f"—— {p['kind']}【{p['label']}】（key={p['key']}）\n   {p['value']}")
        return
    if args.decisions_json:
        incoming = _load_json_arg(args.decisions_json)
        known = {p.key for p in pend}
        unknown = sorted(set(incoming) - known)
        if unknown:   # 同答案侧：认不出的 key 明确报出来，不静默丢
            raise SystemExit(f"这些 key 不在待确认清单里：{unknown}\n"
                             f"待确认：{sorted(known)}")
        decisions.update(incoming)
        state["step"] = "confirm"
        save_state(args.out, state)
        persona, _ = _rebuild(state)
        left = len(pending_confirmations(persona))
        print(f"已落 {len(incoming)} 条决定，还剩 {left} 条未决。"
              + ("下一步：--step ship" if not left
                 else " 未决的保持未决——没问过对方的条目不会被默认留下。"))
        return
    if not pend:
        print("没有待确认的草稿。下一步：--step ship")
        return
    print(f"【逐条确认】共 {len(pend)} 条草稿。每条：y=留 / n=删 / e=改 / q=先退出（已确认的会存下）\n")
    for p in pend:
        print(f"—— {p.kind}【{p.label}】\n   {p.value}")
        try:
            choice = input("   [y/n/e/q] > ").strip().lower()
        except EOFError:
            choice = "q"
        if choice == "q":
            break
        if choice == "n":
            decisions[p.key] = "drop"
        elif choice == "e":
            new = input("   新文本 > ").strip()
            decisions[p.key] = {"edit": new} if new else "keep"
        else:
            decisions[p.key] = "keep"
        state["step"] = "confirm"
        save_state(args.out, state)      # 每条都存——确认是长活儿，没人一口气做完
    persona, _ = _rebuild(state)
    left = len(pending_confirmations(persona))
    print(f"\n已确认 {len(decisions)} 条，还剩 {left} 条未决。"
          + ("下一步：--step ship" if not left else "随时重跑这一步继续。"))


def _step_ship(args, state):
    persona, _ = _rebuild(state)
    entries = None
    if args.import_path:
        from memory_import import load_any
        entries = load_any(args.import_path)
        print(f"【导入】{args.import_path} → {len(entries)} 条")
    try:
        paths = write_bundle(args.out, persona, client=state.get("client", args.client),
                             corpus_dir=args.corpus, confirmed=True, entries=entries)
    except (PermissionError, ValueError) as e:
        raise SystemExit(f"不出货：{e}")
    print("【三件套】")
    print(f"  人格文件：{paths['persona']}")
    print(f"  记忆库：{paths['memory_dir']}"
          + (f"（落盘 {len(paths['corpus_files'])} 个窗口文件）" if paths["corpus_files"] else ""))
    print(f"  MCP 配置：{paths['mcp_config']}")
    for s in persona.suggestions():
        print(f"  建议（不阻塞）：{s}")
    state["step"] = "shipped"
    save_state(args.out, state)


def _cli(args):
    """薄 CLI，四步走：questionnaire → answers → confirm → ship。
    每步存状态可续跑；不传 --step 时按状态里的进度提示下一步该跑什么。
    真正的逐题交互留给导出的 prompt（路线 C）——用户拿去自己的模型那边一问一答，
    比在终端里敲长文本舒服得多。"""
    state = load_state(args.out)
    step = args.step
    if not step:
        step = {"": "questionnaire", "questionnaire": "answers",
                "answers": "confirm", "confirm": "ship",
                "shipped": "ship"}[state.get("step", "")]
        print(f"（按进度接着跑：--step {step}）\n")
    if step == "questionnaire":
        _step_questionnaire(args)
    elif step == "answers":
        _step_answers(args, state)
    elif step == "confirm":
        _step_confirm(args, state)
    else:
        _step_ship(args, state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="产出目录")
    ap.add_argument("--corpus", help="已有语料目录（可选）")
    ap.add_argument("--client", default="claude-code", choices=sorted(CLIENT_FILENAMES))
    ap.add_argument("--step", choices=["questionnaire", "answers", "confirm", "ship"],
                    help="不传则按 init_state.json 里的进度接着跑")
    ap.add_argument("--answers", help="answers 步：模型整理好的答案清单文件（人工路径）")
    # --- 以下三个是给「AI 驱动」用的程序接口（人自己敲命令时用不上）---
    # 产品事实：多数用户不会开终端敲 python，真实形态是把仓库交给 AI、AI 边问边跑。
    # 引导指南本来就假定 AI 驱动，但 CLI 只提供了给人用的交互路径——差的这一截
    # 在这里补上。库函数（apply_answers / apply_confirmations）本来就收字典。
    ap.add_argument("--json", action="store_true",
                    help="机器可读输出（questionnaire 出题目、confirm --list 出待确认清单）")
    ap.add_argument("--answers-json", dest="answers_json",
                    help="answers 步：结构化答案（文件路径 / JSON 字面量 / - 读 stdin）")
    ap.add_argument("--list", action="store_true",
                    help="confirm 步：只列出待确认草稿，不进交互循环")
    ap.add_argument("--decisions-json", dest="decisions_json",
                    help="confirm 步：结构化决定（同上三种取值），非交互落盘")
    ap.add_argument("--import", dest="import_path",
                    help="ship 步：语料导出文件（ChatGPT/Claude json、聊天 txt、timeline md），"
                         "由 memory_import 认格式并落成记忆库")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.out:
        _cli(args)
    else:
        ap.print_help()
