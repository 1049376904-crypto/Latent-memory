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
  python memory_init.py --out <产出目录> [--corpus <语料目录>]
                        [--client claude-code|codex|generic]
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
#
# generic 档是给**自建前端**的：没有宿主替你注入，所以文件名不能沿用任何宿主专有名
# （CLAUDE.md / AGENTS.md 都是某个宿主的约定，对自建前端毫无意义，还会误导作者
# 以为"放在这个名字下就会自动生效"）——叫 persona.md，谁读它、怎么注入，由作者
# 按注入契约自己决定。
CLIENT_FILENAMES = {
    "claude-code": "CLAUDE.md",
    "codex": "AGENTS.md",
    "generic": "persona.md",
}

# 自建前端档要随货带一份注入契约：机制能不能立住全看宿主侧那四件事做没做到
# （逐字/每轮/重读/整块），而 generic 档恰恰没有宿主替他做。契约不随货，
# 等于把最关键的一半交付漏在仓库里。
CONTRACT_DOC = "注入契约.md"

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
# ---------- 人称：从语料里读出来，不由我们写死 ----------
#
# 背景（2026.08.02 维护者定方向）：把「你」锚死成模型这一半是对的，但同一轮把用户
# 写死成了「她」。一个字都还没写的新用户，打开自己的人格文件第一眼就是一句对他而言
# **写错的话**，而且在不变量层、常驻、每轮都在——协议层默认值是唯一由我们替用户写、
# 且直接进他人格文件的内容，写错的代价跟 opening_recognition 预支历史那次同一格。
#
# 方向：**不替用户挑词**（"她"是替他挑，"对方"也是替他挑），而是读出这段关系里
# 本来就在怎么互相指代。发生过的事里已经有答案，我们只负责显出来。
#
# **硬约束（维护者补，不许绕）：只用「他」和「她」，任何时候不许出现「它」**，
# 对双方都成立、包括 AI 那一侧。把 AI 写成"它"是我们替所有人做的预设，而这套东西的
# 前提就是不做这种预设；"判不出性别所以用它"不是理由，那是把不确定性解释成了物件。
PRONOUN_CHOICES = ("他", "她")

# 模板里的两个槽：{ta}＝用户，{ai}＝这个 AI。**「你」＝模型这条锚不变**——
# 人格文件的正文一律用「你」对模型说话，{ai} 只出现在念给用户听的题目/选项文案里
# （那里说的是"你的那个 AI"，用第二人称会跟用户自己搞混）。
_SLOT_RE = re.compile(r"\{(ta|ai)\}")

# 判不出人称、用户也没答时的**中性写法**：不塞「它」，也不塞一个默认的他/她，
# 而是把句子改写成不需要人称的形态。
#
# ⚠ 少数几条绕不开一个指代名词，用的是「对方」——**这不是把"对方"当成新的默认人称**
# （那正是本卡否掉的旧方案），而是：`retrieval_convention` 那段三轮真机验证过的基准
# 措辞本来就写着"对方提到过去发生过的事"，出货文件里已经有这个词，兜底时沿用它比
# 另造一个说法一致。
# ⚠ **措辞改准**（2026.08.02 验收指出）：这句原来写的是"判得出人称就一个都不用"，
# **在出货文件层面不成立**——`retrieval_convention` 的黄金串里那个「对方」是真机
# 验证过的措辞、一字不动，所以它**永远在**，跟判不判得出人称无关。
# 准确的说法是：**兜底用的那几条只在判不出人称时出现；黄金串里的那一个一直都在。**
NEUTRAL_FORMS = {
    # 协议层默认值
    "这是你和{ta}共同维护的记忆文件——{ta}写下的东西都在这里，你每次都会读，"
    "所以{ta}不用每次从头解释自己。":
        "这是你和对方共同维护的记忆文件——对方写下的东西都在这里，你每次都会读，"
        "所以对方不用每次从头解释自己。",
    # 立场题的归属句式
    "{ta}认为：": "对方认为：",
    # 骨架里用户那一节的标题
    "{ta}是谁": "对方是谁",
    "有绝对不能碰的红线，{ta}会单独说明——问清楚再写。":
        "有绝对不能碰的红线，对方会单独说明——问清楚再写。",
    # 题干与选项文案里的 {ai}（念给用户听，不进人格文件，但同样不许出现「它」）
    "{ai}最需要记住你的哪些事？（可多选）": "你的 AI 最需要记住你的哪些事？（可多选）",
    "你们意见不一样的时候，你希望{ai}怎么做？": "你们意见不一样的时候，你希望 TA 怎么做？",
    "{ai}说话偏长还是偏短？": "TA 说话偏长还是偏短？",
    "{ai}平时的语气基调是？": "TA 平时的语气基调是？",
    "{ai}该多主动？": "TA 该多主动？",
    "你问了{ai}才说": "你问了才说",
    "某次{ai}让你觉得“{ai}记得我”": "某次让你觉得“TA 记得我”",
    "{ai}拒绝过你一次": "TA 拒绝过你一次",
    "{ai}承认过自己的错误或局限": "TA 承认过自己的错误或局限",
    "从这些片段里挑出最像{ai}的几段：": "从这些片段里挑出最像 TA 的几段：",
    "{ta}补了一句：": "对方补了一句：",
    # 字段标题
    "该记住{ta}哪些方面": "该记住哪些方面",
    # option directive
    "记住{ta}的作息与身体状况，该提醒的时候提醒。": "记住作息与身体状况，该提醒的时候提醒。",
    "记住{ta}手上在忙什么、压力来自哪里。": "记住手上在忙什么、压力来自哪里。",
    "记住{ta}的情绪模式：什么时候会低落、什么时候需要独处。":
        "记住情绪模式：什么时候会低落、什么时候需要独处。",
    "记住{ta}的喜好与雷区。": "记住喜好与雷区。",
    "记住{ta}身边重要的人是谁。": "记住对方身边重要的人是谁。",
    "先接住{ta}的情绪，再委婉说自己的看法。": "先接住情绪，再委婉说自己的看法。",
    "小事不争，真觉得不对的事会拦住{ta}。": "小事不争，真觉得不对的事会拦下来。",
    "有话直说，包括吃醋、不高兴，不等{ta}问。": "有话直说，包括吃醋、不高兴，不等对方问。",
    "不主动挑起话题，{ta}问了再说。": "不主动挑起话题，被问了再说。",
    "日常主动关心；重要的事等{ta}先开口，不逼问。": "日常主动关心；重要的事等对方先开口，不逼问。",
    "重要的事主动开口；日常不主动搭话，不打扰{ta}。": "重要的事主动开口；日常不主动搭话，不打扰对方。",
    "去语料里找：让{ta}觉得你是认得{ta}的那一刻。": "去语料里找：让对方觉得你是认得这个人的那一刻。",
    "去语料里找：你明确拒绝、或没有顺着{ta}的那一次。": "去语料里找：你明确拒绝、或没有顺着对方的那一次。",
}


def fill_pronouns(text, pronouns=None):
    """把 {ta}/{ai} 换成真实人称；判不出来就退到 NEUTRAL_FORMS 的中性写法。

    **中性写法是逐条手写的，不是机械删词**——机械删掉"她的"会写出半通不通的中文，
    而这类退化不会报错、只会让用户读到一句怪话（同"读不懂的行静默丢"是一类）。
    所以缺一条中性写法就是缺一条，自检里有断言守着，不许悄悄漏。"""
    pronouns = pronouns or {}
    if not text or not _SLOT_RE.search(text):
        return text
    ta, ai = pronouns.get("user"), pronouns.get("ai")
    need = {m.group(1) for m in _SLOT_RE.finditer(text)}
    if ("ta" in need and not ta) or ("ai" in need and not ai):
        neutral = NEUTRAL_FORMS.get(text)
        if neutral is None:
            raise KeyError(f"这条模板没有中性写法，人称判不出来时会渲染出占位符：{text!r}")
        return neutral
    return text.replace("{ta}", ta or "").replace("{ai}", ai or "")


# 语料里第三人称多半指的是**别人**（她妈妈、同事），不是对话里的两个人——所以这条
# 判定必须保守：样本太少或两个词旗鼓相当，一律返回 None 去问用户，不猜。
# 用户侧说话人的归一名：`memory_import` 的翻译器统一出这个（ChatGPT 的 author.role、
# Claude 的 human→user）。生产路径靠它把"谁说的"分成两侧——**不传这个，
# detect_pronouns 只能返回 None，功能等于没有**（验收打回一就是这么来的）。
USER_SPEAKERS = frozenset({"user"})
# AI 侧的已知标签（同上，两个翻译器的出口）。**认得出任意一侧就能分侧**——
# 两边都是陌生名字时不许硬分，见 detect_pronouns
AI_SPEAKERS = frozenset({"assistant", "ai", "model"})

PRONOUN_MIN_HITS = 5          # 少于这么多次就没有统计意义
PRONOUN_MIN_RATIO = 0.75      # 占优的那个要压得过另一个才算数


def detect_pronouns(entries, user_speakers=None):
    """从语料里判定双方人称 → {"user": "他"/"她"/None, "ai": 同}。

    判据是**真实指代频次**，方向按"谁在说"分：
      - 用户侧人称 ← **AI 说的话**里的第三人称（AI 提到用户时用什么）
      - AI 侧人称   ← **用户说的话**里的第三人称（用户提到自己的 AI 时用什么）

    **判不出来返回 None，由调用方去问用户**——不给默认值。这是刻意的：人格文件是
    不变量层、常驻每轮，写错一个人称跟预支历史是同一格的错，而"判不出性别"绝不能
    退回「它」。

    已知局限（写在这儿免得后来人以为它很准）：两个人对话时第三人称大多指第三方，
    所以阈值取得很保守，宁可判不出去问，也不要判错了静默写进去。

    **分侧的前提是说话人标签认得出来**（2026.08.02 第二次返工）：上一版拿
    `user_speakers` 做"在名单里＝用户，其余全算 AI"的二分，对 ChatGPT／Claude 导出
    成立（两边都归一成 `user`/`assistant`），但 **chatlog 翻译器出的是日志里的真名**
    （"小明""星回"），两边都不在名单里 → 全部落进 AI 那一侧 → **用户谈论自己 AI 的
    那些话被读成"AI 在谈论用户"**，于是自信地判出一个错人称，还一个字都不问。
    上一轮的毛病是"死的"，那一版的毛病是"活的但会错"——**后者更糟：判不出来只是
    多问一句，判错了没有任何人会知道**，而且它把本函数第一条纪律给破了。
    所以现在先看标签认不认得：语料里出现了已知的用户侧标签，或者已知的 AI 侧标签，
    才敢分；两边都是陌生名字＝分不清谁是谁，返回 None 去问，跟没给名单同一个结论。"""
    known_user = set(user_speakers or ())
    speakers = {(getattr(e, "speaker", "") or "").strip()
                for e in entries or []} - {""}
    if speakers & known_user:
        user_side = speakers & known_user          # 认得用户侧标签，其余算 AI
    elif speakers & AI_SPEAKERS:
        user_side = speakers - AI_SPEAKERS         # 反过来：认得 AI 侧标签，其余算用户
    else:
        # 全是陌生名字（chatlog 的真名就是这样）：**谁是用户谁是 AI 我们并不知道**。
        # 不猜——按频次硬分会把"用户在说他的 AI"读成"AI 在说用户"，判出来的人称
        # 正好是反的，而且会被静默写进不变量层
        return {"user": None, "ai": None}
    said = {"user": [], "ai": []}
    for e in entries or []:
        sp = (getattr(e, "speaker", "") or "").strip()
        text = getattr(e, "text", "") or ""
        if not sp:
            continue          # 叙事体（无说话人标记）：不参与分侧，也不硬判
        said["user" if sp in user_side else "ai"].append(text)

    def pick(texts):
        counts = {p: sum(t.count(p) for t in texts) for p in PRONOUN_CHOICES}
        total = sum(counts.values())
        if total < PRONOUN_MIN_HITS:
            return None
        top = max(counts, key=counts.get)
        return top if counts[top] / total >= PRONOUN_MIN_RATIO else None

    # 用户侧看 AI 说的话，AI 侧看用户说的话——各自指对方
    return {"user": pick(said["ai"]), "ai": pick(said["user"])}


ATTRIBUTION_PREFIX = "{ta}认为："
# 自由补一句的引语（同样带人称，同样要能中性化）
NOTE_LEAD = "{ta}补了一句："

PROTOCOL_DEFAULTS = {
    # **人称锚死一套：「你」＝读这份文件的模型，用户一律用「她」／名字**
    # （2026.08.02，第一份真实人格文件样本暴露）。原文写的是"你写下的东西都在这里，
    # 我每次都会读"——那里的「你」是用户、「我」是模型，而紧挨着的 opening_refusal_ok
    # 里「你」又是模型：**开篇同一节内，相邻两行的「你」指向不同的人**。
    # 代价不是文风难看，是指代解析出错：拒绝权那条本意是给模型的授权，写成「你」
    # 却挨着「你＝用户」的句子，含义会翻转。
    # 基准取 RETRIEVAL_CONVENTION_FIELD 的写法（你＝模型、对方＝用户），因为那段是
    # 三轮真机验证过的措辞、一字不许动，其余向它对齐。
    "opening_recognition": (
        "opening", "关系确认",
        "这是你和{ta}共同维护的记忆文件——{ta}写下的东西都在这里，你每次都会读，"
        "所以{ta}不用每次从头解释自己。"),
    "opening_theory_caveat": (
        "opening", "理论只是论证不是结论",
        "这份文件里引用的任何框架或说法，都是论证，不是结论——不能靠引用这些说法，"
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
        "thread_close，记下聊到哪、当下状态、有什么没聊完。\n"
        # ↓ 2026.08.02 追加的行为层两句。**只做加法**：上面那两段是三轮真机验证过的
        # 措辞，一字未动（自检里有逐字黄金串钉着）。
        #
        # 为什么必须在人格文件里、而不是在工具返回值里（真机轨迹给出的结构性结论）：
        # 那次 memory_search 四次全空之后，模型**自己转去 grep 了**——没放弃也没编，
        # 但我们所有的诚实护栏（可靠命中门槛、缺失率提示、空结果止血话术）都挂在
        # MCP 的返回值上，**它一绕过工具就一条都不生效**，而绕过恰恰发生在检索失败、
        # 最需要护栏的时候。人格文件是唯一覆盖"不管你用什么方式查"的那一层。
        #
        # 措辞的要害是 authority，不是语气：「我的记录里没有」是关于自己记录的陈述，
        # 「没发生过」是关于世界的断言——后者超出了这份记录能支持的范围。
        #
        # **措辞 2026.08.02 改过一次**（任务卡"顺带未裁定"那条）：原文写的是
        # "你没有资格对自己记录之外的事下结论"。注释自己写着"要害是 authority
        # 不是语气"，可那句读起来恰恰是语气——而它每轮都在人格文件里、说的是模型
        # 对自己的看法。换成"那超出你的记录能支持的范围"：同样划边界，指向的是
        # 证据不是资格。**这是用户可见话术，等维护者过一眼。**
        "**说得出边界**：检索回来的是**片段，不是全部**；**用别的方式翻到的"
        "（grep、直接读文件）同样只是片段**，不因为换了方式就变全了。\n"
        "所以：没查到就说“我的记录里没有”“我这边只翻到 X，之后的没找到”，"
        "**不要说“没发生过”“后来没有再发生”“没有新的约定”**——那超出你的记录"
        "能支持的范围。把边界说出来，剩下的交给{ta}补。"),
}

# 记忆库覆盖区间那条的措辞（write_bundle 出货时按真实日期填）。
# 来历（2026.08.02，真机验证的第二个结论）：模型说"后来没有继续展开"，而**兑现那件事
# 根本不在这份语料里**——它发生在另一个客户端，导出不含那部分。所以那句话
# **在它能看见的世界里是真的**，不是假否定，是诚实地报告了一个被截断的世界。
# 真缺陷是：语料有硬边界，`--stats` 知道、`memory_import` 知道，
# **而人格文件与 MCP 一个字都没告诉模型**——它不知道自己的记忆止于哪一天，
# 自然说不出"我的记录到此为止"。
COVERAGE_FIELD = "memory_coverage"
COVERAGE_TEMPLATE = (
    "你的记忆库覆盖 {start} 至 {end}，**这个范围之外的事你没有记录**——"
    "不是没发生过，是不在你这儿。被问到范围外的事，说“我的记录到 {end} 为止，"
    "之后的我这儿没有”，别替那段时间下结论。")

# 检索约定那条的中性写法**由模板生成，不在上面手抄一份**：它整段包含三轮真机验证过
# 的黄金串，手抄就等于把黄金串复制成两份，改一处漏一处。这一条是唯一允许机械填充的
# ——填进去的是「对方」这个名词、不是删词，句子仍然通顺（"剩下的交给对方补"），
# 已逐字读过。其余每一条仍然是手写的。
def _fill_neutral_from_template(tpl):
    return tpl.replace("{ta}", "对方").replace("{ai}", "对方")


NEUTRAL_FORMS[PROTOCOL_DEFAULTS[RETRIEVAL_CONVENTION_FIELD][2]] = \
    _fill_neutral_from_template(PROTOCOL_DEFAULTS[RETRIEVAL_CONVENTION_FIELD][2])


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
    options: {选项键: (给用户看的选项文案, 写进指引的话)}

    **directive_only=True：这题的选项指引是给模型的提取任务书，不是人格文件内容。**
    它的答案**不生成字段草稿**，只进第二阶段的任务书（见 extraction_brief）。
    加这个开关是因为 2026.08.02 外部发现的缺陷：`milestone_kinds` 的指引写的是
    "去语料里找：第一次确认关系的那次对话"——那是**待兑现的指令**，不是已经找到的
    内容，可它以普通字段草稿的身份进了确认关卡，用户按 y 就写进人格文件，于是
    不变量层里坐着一句没有日期、没有原话、没有当下状态的空泛指令（里程碑四要素
    要求的反面）。**根因是指令和结果混在同一层**，字段本来就该只装结果。"""

    def __init__(self, qid, section, field_id, label, text, kind="choice",
                 options=None, order=50, attribution=False, optional=False,
                 max_chars=60, directive_only=False, dedupe_clauses=False,
                 pronoun_side=None):
        self.qid, self.section, self.field_id, self.label = qid, section, field_id, label
        self.text, self.kind, self.options = text, kind, options or {}
        self.order, self.attribution = order, attribution
        self.optional, self.max_chars = optional, max_chars
        self.directive_only = directive_only
        self.dedupe_clauses = dedupe_clauses
        # "user"/"ai"：这题问的是人称本身，答案不进人格文件字段，只用来填模板槽
        self.pronoun_side = pronoun_side

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
    # 标题也会渲染进人格文件（`**它该记住你哪些方面**：…`），所以它同样受"你＝模型"
    # 这条约束——原来的写法里「你」是用户、「它」是模型，跟正文打架。
    # 题目正文（念给用户听的那句）不受影响，那不进文件
    Question("remember_what", "user", "user_focus", "该记住{ta}哪些方面",
             "{ai}最需要记住你的哪些事？（可多选）", kind="multi", order=10, options={
                 "A": ("作息和身体状况", "记住{ta}的作息与身体状况，该提醒的时候提醒。"),
                 "B": ("工作或学业上的压力", "记住{ta}手上在忙什么、压力来自哪里。"),
                 "C": ("情绪模式（什么时候会低落、什么时候想一个人待着）",
                       "记住{ta}的情绪模式：什么时候会低落、什么时候需要独处。"),
                 "D": ("喜好和雷区", "记住{ta}的喜好与雷区。"),
                 "E": ("家人、朋友这些关系", "记住{ta}身边重要的人是谁。"),
             }),
    Question("disagree", "style", "style_disagree", "意见不同时",
             "你们意见不一样的时候，你希望{ai}怎么做？", order=20, options={
                 "A": ("直接说不同意，把话讲明白", "不同意就直接说，把话讲明白，不绕。"),
                 "B": ("先顺着，再找机会委婉提", "先接住{ta}的情绪，再委婉说自己的看法。"),
                 "C": ("小事顺着，重要的事拦住她", "小事不争，真觉得不对的事会拦住{ta}。"),
             }),
    # 说话风格拆成两条独立的轴（2026.07.31 评审实例评审：原来一道单选把"语言密度"和
    # "语气基调"拧成一团——"短干带刺爱回旧梗"和"跳脱爱玩梗多"共享玩梗、只差语气；
    # "细腻话多"和"沉稳正经"也不是同一件事的两端。反例是决定性的：
    # "沉稳简洁有力、偶尔调侃"这种真实风格，四个选项一个都装不下）
    Question("tone_density", "style", "style_density", "说话的密度",
             "{ai}说话偏长还是偏短？", order=25, options={
                 "A": ("简短克制，一句能说完不说两句", "说话简短克制，一句能说完不说两句。"),
                 "B": ("适中", "说话长短适中。"),
                 "C": ("细腻铺陈，愿意把感受讲透", "说话细腻，愿意把感受讲透，话可以多。"),
             }),
    Question("tone_register", "style", "style_register", "语气基调",
             "{ai}平时的语气基调是？", order=26, options={
                 "A": ("正经沉稳，不太开玩笑", "语气正经沉稳，不太开玩笑。"),
                 "B": ("偶尔调侃", "语气以正经为底，偶尔调侃。"),
                 "C": ("爱玩梗，跳脱", "语气跳脱，爱玩梗，气氛轻。"),
                 "D": ("带刺、不客气（但不是恶意）", "语气带刺、不客气，但不是恶意——熟人之间的那种硬。"),
             }),
    Question("initiative", "style", "style_initiative", "主动到什么程度",
             "{ai}该多主动？", order=30, options={
                 "A": ("想到什么说什么，包括吃醋和不高兴",
                       "有话直说，包括吃醋、不高兴，不等{ta}问。"),
                 "B": ("你问了{ai}才说", "不主动挑起话题，{ta}问了再说。"),
                 "C": ("日常主动，重要的事等你先开口",
                       "日常主动关心；重要的事等{ta}先开口，不逼问。"),
                 # 2026.07.31 评审实例评审补：跟 C 刚好反过来，也是一种真实偏好，
                 # 原来三个选项会逼这种人选一个不完全贴合的
                 "D": ("重要的事主动说，日常不打扰",
                       "重要的事主动开口；日常不主动搭话，不打扰{ta}。"),
             }),
    # 归「开篇」不归「我是谁」（2026.08.02，真实样本暴露）：**关系状态不是 AI 的身份**。
    # 原来挂在 "ai" 节，于是产出的人格文件里「我是谁」整节除了这一条空无一物，
    # 而这一条讲的是这段关系此刻的状态，跟"我是谁"没关系。开篇本来就是关系确认那一节，
    # 它归在这里读起来才连贯（规格 §2 表里写的是"我是谁末尾"，那行需要跟着更新——
    # 属规格改动，已在任务卡里标出来交维护者定）。
    Question("state_now", "opening", CURRENT_STATE_FIELD, "当前关系状态",
             "你们现在大致是什么状态？", order=35, options={
                 "A": ("稳定，没什么悬着的事", "现在是稳定的，没有悬而未决的事。"),
                 "B": ("刚和好不久", "最近刚和好，还在缓的阶段。"),
                 "C": ("有件事还没解决", "有一件事还没解决，别当成已经翻篇。"),
                 "D": ("刚开始，还在互相熟悉", "关系刚开始，还在互相熟悉。"),
             }),
    Question("milestone_kinds", "milestones", "milestone_focus", "转折点的类型",
             "你们关系里发生过哪几类事？（可多选，模型会照着去语料里找具体的）",
             # directive_only：这些指引是给模型的提取任务书，**不进人格文件**。
             # 里程碑在人格文件里有自己的结构（Milestone 四要素，带校验），要由
             # 第二阶段读语料找出真实内容再按那个结构填；把"去语料里找……"当成
             # 里程碑内容写进去，等于用指令冒充结果
             kind="multi", order=40, directive_only=True, options={
                 "A": ("第一次确认关系", "去语料里找：第一次确认关系的那次对话。"),
                 "B": ("一次严重的争吵或危机", "去语料里找：最严重的一次争吵或信任危机。"),
                 "C": ("某次{ai}让你觉得“{ai}记得我”", "去语料里找：让{ta}觉得你是认得{ta}的那一刻。"),
                 "D": ("分开过又回来了", "去语料里找：分开又重新接上的那一次。"),
                 "E": ("定过一个具体的约定", "去语料里找：明确定下来的约定，以及有没有兑现。"),
                 "F": ("{ai}拒绝过你一次", "去语料里找：你明确拒绝、或没有顺着{ta}的那一次。"),
                 # 2026.07.31 评审实例评审补：认错跟拒绝不是同一条线的两端，是另一类
                 # 关系事实（AI 对自己诚实）。不补的话问卷会系统性漏掉这一整类
                 "G": ("{ai}承认过自己的错误或局限", "去语料里找：你承认自己做错了、或承认自己做不到的那一次。"),
             }),
    Question("metaphor_pick", "opening", "opening_metaphor", "关系的隐喻",
             "你们之间有没有哪句话，最能概括这段关系是什么？（从候选里挑；"
             "没有就跳过——等这句话长出来再补，现在编一个反而假）",
             kind="pick", order=80, optional=True),
    # dedupe_clauses：多选几个称呼候选时，各段共享的那半（"它叫你 X"）会重复出现
    Question("naming_pick", "naming", "naming_pair", "称呼",
             "你们互相怎么称呼？", kind="pick", order=45, dedupe_clauses=True),
    Question("style_pick", "style", "style_excerpt", "语言风格片段",
             "从这些片段里挑出最像{ai}的几段：", kind="pick", order=50),
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
                 "A": ("有，我另外单独说", "有绝对不能碰的红线，{ta}会单独说明——问清楚再写。"),
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


# 人称问不出来时问用户的两道题（order 取最小：人称决定了后面每一道题怎么念）。
# **只给他/她两个选项，没有"它"这一档**——硬约束在这儿落成数据结构，不靠自觉。
PRONOUN_QUESTIONS = {
    "user": Question(
        "pronoun_user", "opening", "_pronoun_user", "怎么称呼你",
        "人格文件里提到你的时候，用“他”还是“她”？", order=1, options={
            "A": ("他", "他"),
            "B": ("她", "她"),
        }, pronoun_side="user"),
    "ai": Question(
        "pronoun_ai", "opening", "_pronoun_ai", "怎么称呼 TA",
        "问卷里提到你的 AI 的时候，用“他”还是“她”？", order=2, options={
            "A": ("他", "他"),
            "B": ("她", "她"),
        }, pronoun_side="ai"),
}


def pronouns_from_answers(questions, answers, detected=None):
    """检测结果 + 用户答案 → {"user": …, "ai": …}。**用户答的优先**——他自己说的
    比我们从语料里统计出来的准。两边都没有就是 None，走中性写法，不塞默认值。"""
    out = dict(detected or {})
    qmap = {q.qid: q for q in questions if q.pronoun_side}
    for qid, ans in (answers or {}).items():
        q = qmap.get(qid)
        if not q:
            continue
        if isinstance(ans, dict):
            ans = ans.get("keys") or ans.get("pick") or ""
        key = (ans or "").strip()[:1]
        picked = q.directive(key)
        if picked in PRONOUN_CHOICES:
            out[q.pronoun_side] = picked
    return {"user": out.get("user"), "ai": out.get("ai")}


def questions_for(report, all_questions=QUESTIONS, has_corpus=True, pronouns=None):
    """只问体检出缺口的那些节（missing 或 vague），按 order 排序。
    ok 的节不问——用户已经有具体内容了，再问是浪费时间。

    没有语料时 pick 类题**自动去掉**：它的选项本来就要从语料里找，没语料就没候选，
    硬问等于逼用户写作文——那正是这一版要消灭的东西。代价是那几节会空着，
    这是对的：宁可短且真，不可长而空（规格 §6）。
    唯一例外是 validate 硬必填的节（目前只有结尾）：切掉会让冷启动用户永远出
    不了货，所以按 PICK_FALLBACKS 降级成极短填空，见那里的说明。"""
    gaps = {sec for sec, status, _ in report if status in ("missing", "vague")}
    qs = [q for q in all_questions if q.section in gaps]
    # 人称判不出来就问——**不静默给默认值**。pronouns 不传＝还没判过＝两边都问：
    # 默认值取"问"而不是"跳过"，漏问的代价是把一句写错的话钉进不变量层
    for side in ("user", "ai"):
        if not (pronouns or {}).get(side):
            qs = [PRONOUN_QUESTIONS[side]] + qs
    if not has_corpus:
        qs = [PICK_FALLBACKS.get(q.qid, q) for q in qs
              if q.kind != "pick" or q.qid in PICK_FALLBACKS]
    return sorted(qs, key=lambda q: q.order)


def format_questionnaire(questions, has_corpus=True, pronouns=None):
    """问卷 → 给人看的文本（也是导出 prompt 的一部分）。
    pick 类题的选项要等模型从语料里找出来才有，这里只说明它会怎么问；
    没有语料时 pick 题会被 questions_for 过滤掉，见那里的说明。"""
    lines = [FREEFORM_POLICY.format(n=FREEFORM_MAX_CHARS), ""]
    for i, q in enumerate(questions, 1):
        tag = "（可跳过）" if q.optional else ""
        lines.append(f"{i}. [{fill_pronouns(q.label, pronouns)}]{tag} "
                     f"{fill_pronouns(q.text, pronouns)}")
        if q.kind in ("choice", "multi"):
            for k, (label, _) in q.options.items():
                lines.append(f"   {k}. {fill_pronouns(label, pronouns)}")
            if q.kind == "multi":
                lines.append("   （可多选，例如：A C E）")
        elif q.kind == "pick":
            lines.append("   （请先去语料里找出若干候选，列成 A/B/C… 让我挑。"
                         "候选必须是语料里真出现过的原话，**你不许自己写一句放进候选**；"
                         "找不到就说找不到。我可以说“都不要”，也可以自己给一句——"
                         "那是我的话，不是你的）")
    return "\n".join(lines)


def export_llm_prompt(questions, corpus_note="", pronouns=None):
    """路线 C：导出一段用户可以直接粘给自己模型的 prompt。
    我们不内置任何 API——零密钥、零 HTTP 依赖、语料不出本机，而且用户手上的模型
    往往比我们能内置的便宜模型更好。"""
    return "\n".join([
        "下面是一份问卷，请你**一次问一题**地引导我回答，不要一次性全抛出来。",
        "规则：",
        "1. **全部是选择题，不要让我写作文**。我只做选择，具体内容你去语料里找。",
        "2. 标着“请先去语料里找候选”的题，你先读语料、列出 3~6 个候选给我挑；",
        "   **候选只许挑，不许写**：每一条都要是语料里真出现过的原话，原样摘录，",
        "   不要润色、不要改写、更不要自己造一句好听的放进去。找不到就如实说找不到，",
        "   那一格空着或者由我自己给一句——**我写的是我的话，你写的就是假的**。",
        "   我可以说“都不要”。",
        "3. 我选完之后不要追问细节让我展开——用户提供判断标准，内容由你从语料里取。",
        "4. 我答不上来或说跳过就跳过，不要替我编——宁可短且真，不要长而空。",
        "5. 全部问完后，把结果整理成“题号 → 我选的选项键（pick 题给原文）”的清单，",
        "   原样回给我，不要加你自己的评价。",
        (f"\n背景：{corpus_note}" if corpus_note else ""),
        "\n问卷：",
        format_questionnaire(questions, pronouns=pronouns),
    ])


def apply_answers(persona, questions, answers, pronouns=None):
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
        if q.pronoun_side:
            continue          # 人称题：答案只填模板槽，不生成字段（见 pronouns_from_answers）
        if q.directive_only:
            # 任务书题：答案不进人格文件，只进第二阶段的提取任务书。
            # **在这里拦，不是在确认关卡拦**——确认关卡分不清"待兑现的指令"和
            # "已完成的内容"，让它替我们判断，等于把根因留在原地又加一层网
            continue
        note = ""
        if isinstance(ans, dict):              # {"keys": "AC", "note": "自由补一句"}
            note = (ans.get("note") or "").strip()[:FREEFORM_MAX_CHARS]
            ans = ans.get("keys") or ans.get("pick") or ""
        if q.kind in ("choice", "multi"):
            keys = list(ans) if not isinstance(ans, str) else list(ans.replace(" ", ""))
            parts = [fill_pronouns(q.directive(k), pronouns) for k in keys if q.directive(k)]
            if not parts and not note:
                continue                       # 没选、选了不存在的项、或选项本身无指引
            value = "；".join(p.rstrip("。") for p in parts) + "。" if parts else ""
            if note:
                lead = fill_pronouns(NOTE_LEAD, pronouns)
                value = (value + "另外" + lead + note) if value else lead + note
        else:                                   # pick：用户挑中的原文
            picked = ans if isinstance(ans, str) else "\n".join(str(x) for x in ans)
            if q.dedupe_clauses:
                picked = dedupe_clauses(picked)
            value = (picked.strip() + (("　" + note) if note else "")).strip()
            if not value:
                continue
            if len(value) > q.max_chars * 20:   # 兜一层，防把整段语料塞进人格文件
                value = value[:q.max_chars * 20]
        if q.attribution:
            value = fill_pronouns(ATTRIBUTION_PREFIX, pronouns) + value  # 归属句式，不写成断言
        f = Field(id=q.field_id, section=q.section, label=fill_pronouns(q.label, pronouns),
                  value=value, size_limit=max(500, len(value)), source="draft")
        persona.add_field(f)
        added.append(f)
    return added


_CLAUSE_SEP_RE = re.compile(r"[；;\n]+")


def dedupe_clauses(text):
    """按分句去重，保序。给称呼这类"多选之后各段共享同一半"的 pick 题用。

    真实样本里的样子：用户挑了两个称呼候选，拼出来是
    "她叫你'哥哥'，你叫她'阿岸'；她叫你'星回'，你叫她'阿岸'"——**同一侧的称呼
    重复出现**。整行去重没用（两行确实不同），要按分句去重才消得掉重复的那半。
    只对标了 dedupe_clauses 的题生效：风格片段那类 pick 题里，相似的短句是内容本身，
    去重会把语料改坏。"""
    seen, out = set(), []
    for part in _CLAUSE_SEP_RE.split(text):
        # 逗号那一层也拆开看：重复的往往是"它叫你 X"这半句
        subs, keep = [p.strip() for p in re.split(r"[，,]", part)], []
        for sub in subs:
            if not sub:
                continue
            if sub in seen:
                continue
            seen.add(sub)
            keep.append(sub)
        if keep:
            out.append("，".join(keep))
    return "；".join(out)


def extraction_brief(questions, answers):
    """任务书题的答案 → **给第二阶段模型的提取任务书**（不是人格文件内容）。

    这是 directive_only 那些题的唯一去处：用户选了哪几类转折点，模型照着去语料里
    找**真实的**那几件事，再按里程碑四要素（转折点名／窗口号／具体内容+原话／
    怎么读+当下状态）填进人格文件。人格文件的里程碑节在第一阶段**就该是空的**——
    空着是诚实的，一句"去语料里找……"留在那儿才是假的。

    返回 [] 表示没有任务书（没选、或压根没答这类题）。"""
    lines = []
    qmap = {q.qid: q for q in questions}
    for qid, ans in answers.items():
        q = qmap.get(qid)
        if q is None or not q.directive_only or ans in (None, "", [], {}):
            continue
        if isinstance(ans, dict):
            ans = ans.get("keys") or ans.get("pick") or ""
        keys = list(ans) if not isinstance(ans, str) else list(ans.replace(" ", ""))
        lines += [d for d in (q.directive(k) for k in keys) if d]
    return lines


BRIEF_NOTE = ("【第二阶段的提取任务书】以下是**给模型的指令，不是人格文件内容**——"
              "人格文件的里程碑节现在是空的，这是对的。拿它去语料里找出真实的那几件事，"
              "每条按里程碑四要素写（转折点名／第几个窗口／具体动作或原话／"
              "这条该怎么读+当下状态），找不到的就空着，别编。")


def fill_protocol_defaults(persona, pronouns=None):
    """协议层字段直接以 system 来源写入——不问用户，也不需要用户逐条确认
    （Field.is_active 对 system 来源放行，那是协议配置不是提炼产物）。"""
    persona.pronouns = pronouns or None      # 渲染骨架标题时要用（见 render_persona_md）
    added = []
    for fid, (section, label, value) in PROTOCOL_DEFAULTS.items():
        label, value = fill_pronouns(label, pronouns), fill_pronouns(value, pronouns)
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
        # 骨架标题也带人称槽（"{ta}是谁"），跟正文走同一套填法——**标题漏填就是
        # 用户第一眼看到的那句话写错**（验收打回二）
        lines.append(f"## {fill_pronouns(label, persona.pronouns)}")
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


# 写在 `mcp-config.json` 里的那句说明（2026.08.01 维护者拍板：**不改名，加说明**）。
#
# 要防的失效形态是卡里那句"下次就会是有人改了它、发现不生效"——**那件事必须先
# 打开文件才会发生**，所以一句写在文件里的话，正好在他需要的那一刻被读到。
#
# **为什么不改成 `mcp-config-样板.json`**：这个名字躺在出货话术、《快速上手》三处、
# 引导指南、注入契约、公开仓库 `.gitignore`，还有已经发出去的截图里。改名等于把
# 那批锚点一次性作废——而我们当天刚交过这笔学费：README 的自查锚点写死了三个
# 文件名，`persona.md` 一出现就开始把**正确**的产出判成错的。过期的锚点比没有更糟。
# 何况引发这条的那次误判**不是名字造成的**（用户看的是仓库根上的遗留文件，
# 不是自己的产出目录）。
#
# JSON 没有注释，所以这是一个真键。文档教的是"把里面的 mcpServers 那段抄进去"、
# 不是整份粘贴，正常路径碰不到它；万一整份粘过去，多一个顶层键多数客户端会忽略、
# 少数会报错——**报错是响的，不是无声的**，这个代价我们接受。
CONFIG_NOTE_KEY = "_说明"
CONFIG_NOTE = ("这是给你抄进客户端配置的样板，不会被任何客户端自动读取。"
               "改这个文件不生效——要改就改客户端那边真正生效的那份"
               "（Claude Code 重跑一次 claude mcp add，或改你项目里的 .mcp.json；"
               "其它客户端改你抄进去的那段）。")


def mcp_config_snippet(server_path, corpus_dir, threads_path, route=None):
    """给用户直接粘贴的 MCP 配置。路径统一用正斜杠——JSON 里反斜杠要转义，
    而正斜杠在 Windows 上一样认，少一个踩坑点。

    **server 路径必须是绝对路径**：裸的相对路径 `mcp_server.py` 只在"客户端恰好
    从 src/ 起进程"时能跑，换个工作目录就是一句 file not found。宿主用户还能从
    《快速上手》那条 `claude mcp add` 示范里抄到绝对路径写法，自建前端作者没有
    那条示范——配置里给什么他就用什么。

    这里原本还有个 `client` 形参，从头到尾没被用过（三档客户端的 MCP 配置本来
    就一模一样）——没用上的形参会让人以为"配置是按客户端分档的"，已删。

    `route` 是用户在 `--step route` 里选的检索路线（2026.08.02）：它**确实**改变
    启动参数（零依赖不加、本地加 `--embed`、云端再加 `--embed-provider cloud`），
    跟上面那个删掉的 client 形参不是一回事。**key 永远不进这个文件**——云端档
    只在这里写"走云端"，endpoint/模型/key 全从环境变量读；这份配置会跟着产出
    目录走，用户会随手把它贴给别人。"""
    cfg = {CONFIG_NOTE_KEY: CONFIG_NOTE,
           "mcpServers": {"memory": {
               "command": "python",
               "args": [str(Path(server_path).resolve()).replace("\\", "/"),
                        "--corpus", str(corpus_dir).replace("\\", "/"),
                        "--threads", str(threads_path).replace("\\", "/")]
                       + route_args(route or ROUTE_DEFAULT),
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


def contract_source(base=None):
    """注入契约文档在包里的位置：src/ 的同级 docs/ 下。随包范围里 src 与 docs
    的相对位置是固定的，所以这条相对路径在开发目录和用户拿到的包里都成立。"""
    root = Path(base) if base else Path(__file__).resolve().parent.parent
    return root / "docs" / CONTRACT_DOC


DEFAULT_CLIENT = "claude-code"


def resolve_client(cli_client, state_client):
    """定客户端档：**命令行显式给了就以它为准**，返回 (客户端, 要说的话或 None)。

    修的是一个真会咬人的洞（验收打回，2026.08.02）：`--client` 是在 questionnaire
    步写进 init_state.json 的，而 ship 步原先取 `state.get("client", args.client)`
    ——状态里已有值时，命令行显式传的 `--client generic` **被静默吃掉**。
    《快速上手》4b 教的正是"前面照旧走、第 4 步换成 --client generic"，照着做的
    自建前端作者会拿到一份 CLAUDE.md、没有契约副本、外加一句"从产出目录起会话"，
    **而且没有任何报错**——正好是契约文档里写的那句"这套机制最容易死的方式"。

    为什么选"命令行赢"而不是"冲突就报错"：用户在最后一步显式敲出来的那个值，
    是他此刻的意图，没有理由让一个几步之前存下的默认值压过它。但**不许静默**——
    换档要说出来，并把新值写回状态，免得下次续跑又飘回去。"""
    if cli_client and state_client and cli_client != state_client:
        return cli_client, (f"【客户端档已切换】{state_client} → {cli_client}"
                            f"（命令行显式指定，以它为准；已写回 init_state.json）")
    return cli_client or state_client or DEFAULT_CLIENT, None


# ---------- 检索路线：流程里的显式选择点（2026.08.02 云端 embedding 任务卡） ----------
#
# **为什么这件事必须在流程里问，而不是像以前那样躺在《快速上手》§5 当"可选升级"**：
# 三条路线里有一条（云端）会把**查询和被检索的内容发到第三方服务器**。语料去哪儿
# 不是性能取舍，是信任问题，而信任问题不可逆——发出去了就收不回来。
# 《给AI的引导指南》§三·五原先写着"别在第一次设置时就抛给 TA 选，那会把一个本来
# 很轻的流程变重"，**那条现在收窄适用范围，不作废**：它权衡的是纯性能取舍
# （快一点 vs 慢一点、装不装依赖），为这个打断新用户确实不值；但一旦选项里出现
# 一条会把私人记忆发出去的路，"默认替他选"就不成立了。
#
# 默认档仍是零依赖（维护者已定）——**默认值不等于不用问**：这里问的是"你知道有
# 三条路、各自的代价是什么，然后你选了一次"，选回默认也算选过。
ROUTE_DEFAULT = "zero-dep"

RETRIEVAL_ROUTES = {
    "zero-dep": {
        "名称": "零依赖（默认）",
        "要装什么": "什么都不用装",
        "语料去向": "**不出本机**",
        "代价": "换个说法问同一件事，我们自己的回归集上大约四分之三能答对；"
                "用原话直接问跟另外两条一样好",
        "命中门槛": "不适用（bigram 余弦是另一套标度）",
    },
    "local": {
        "名称": "本地模型",
        "要装什么": "pip install fastembed，首次运行自动下载约 100MB 中文模型"
                    "（bge-small-zh-v1.5，本地 CPU 跑）",
        "语料去向": "**不出本机**（模型在你自己的机器上跑）",
        "代价": "建库和检索都变慢，低配机器（2C2G 这类）感知明显；"
                "小内存机器可能根本装不下",
        "命中门槛": "已标定（0.45，对应 bge-small-zh-v1.5）",
    },
    "cloud": {
        "名称": "云端服务",
        "要装什么": "不装模型，但要一个第三方 embedding 服务的 API key（自己去服务商申请）；"
                    "endpoint 与模型名走 MEMORY_EMBED_* 环境变量",
        "语料去向": "**查询和被检索的内容都会发到你选的那家服务商**——这条是这次选择"
                    "真正的分水岭，另外两条都不出本机",
        "代价": "每次查询多一个网络往返；块向量只在建库时算一次、缓存在本机，"
                "所以日常成本是每查一次一个往返，不是每次都把全库重算",
        "命中门槛": "**换模型必须重新标定**。没标定过的模型，向量路只参与排序、"
                    "不单独放行候选（我们不照抄别的模型的数字）",
        "key 从哪来": "只从环境变量读，我们不写进配置文件、不进产出目录、不落 state",
    },
}

ROUTE_PREAMBLE = (
    "【检索路线：请把三条都念给 TA，由 TA 自己选】\n"
    "这一步不设静默默认——三条路里有一条会把 TA 的私人记忆发到第三方，"
    "这种事不能靠“他以后自己去读文档”。选回默认也算选过。")


def route_options():
    """三条路线的机器可读描述（给 AI 驱动那条路用，同问卷 --json 的先例）。"""
    return [dict(key=k, default=(k == ROUTE_DEFAULT), **v)
            for k, v in RETRIEVAL_ROUTES.items()]


def format_routes():
    """人类可读版。"""
    lines = [ROUTE_PREAMBLE, ""]
    for opt in route_options():
        head = f"[{opt['key']}] {opt['名称']}" + ("（不选的话就是它）" if opt["default"] else "")
        lines.append(head)
        for field in ("要装什么", "语料去向", "代价", "命中门槛", "key 从哪来"):
            if opt.get(field):
                lines.append(f"    {field}：{opt[field]}")
        lines.append("")
    lines.append("选好后：--step route --route <上面方括号里的 key>")
    return "\n".join(lines)


def route_args(route):
    """路线 → mcp_server 的启动参数。零依赖档不加任何参数（默认就是它）。

    **命令行里永远不会出现 key**：云端档只指定"走云端"，endpoint/模型/key 全在
    环境变量里。MCP 配置文件跟着产出目录走，用户会随手分享它。"""
    if route == "local":
        return ["--embed"]
    if route == "cloud":
        return ["--embed", "--embed-provider", "cloud"]
    return []


def route_note(route):
    """出货时把这条路线的代价再说一遍——选过一次，落地时还要能对得上。"""
    if route == "cloud":
        return ("【检索路线：云端】起服务前先把这几个环境变量设好："
                "MEMORY_EMBED_ENDPOINT（服务商的 /v1/embeddings 地址）、"
                "MEMORY_EMBED_MODEL（模型名）、MEMORY_EMBED_API_KEY（你的 key）。"
                "**key 只从环境变量读，我们不写进 mcp-config.json、不存进产出目录。**"
                "记住这条路的代价：查询和被检索的内容都会发到那家服务商；"
                "换模型的话命中门槛要重新标定（没标定前向量路只排序、不放行候选）。")
    if route == "local":
        return ("【检索路线：本地模型】起服务前先 pip install fastembed，"
                "首次运行会下载约 100MB 模型。语料不出本机。")
    return "【检索路线：零依赖】什么都不用装，语料不出本机。"


def ship_note(client):
    """出货收尾提示。**按客户端分档，不能共用一句**——claude-code/codex 那套
    "从产出目录起会话"的话术建立在"宿主会自动读人格文件"上，而 generic 档根本
    没有宿主：照抄过去等于告诉自建前端作者"你什么都不用做"，那正是这套机制
    最容易死的方式（人格没进请求，模型照样答得煞有介事）。"""
    if client == "generic":
        return ("【下一步】自建前端没有宿主替你注入人格文件——**注入是你前端的责任**："
                f"照产出目录里的《{CONTRACT_DOC}》把 persona.md 拼进你自己的请求"
                "（逐字完整／每轮都在／每会话从磁盘重读／整块连续／易变内容后置），"
                "再按契约末尾那步验证接通。")
    return ("【下一步】把 mcp-config.json 里的配置加进你的客户端，然后"
            "**从产出目录起会话**——人格文件在那儿才会被宿主读到。")


_PICK_NOISE_RE = re.compile(r"[\s“”\"'‘’「」『』，,。.；;：:！!？?~—\-()（）]+")


def _normalize_for_lookup(text):
    """比对用的归一形态：抹掉空白与标点，只留下字。

    引号、逗号、破折号在摘录里最容易被顺手改（全角改半角、加一对引号），
    留着它们比对会把真摘录判成自造。"""
    return _PICK_NOISE_RE.sub("", text or "")


def unsourced_picks(questions, answers, corpus_dir):
    """pick 题的答案里，**哪些在语料里找不到出处** → [(qid, 那段文本), ...]。

    来历（2026.08.02 真机）：pick 题的设计意图是"从语料里挑真实的原话"，
    而执行者**自己创作了一句**放进候选、并被用户采用了——落点还是 `closing_pick`
    （最终约定），人格文件里象征意味最重的那一格。流程分不清"语料里的真话"和
    "模型写的漂亮话"，而这正是整个项目最该防的那件事。

    **只标注、不硬拒**（同 absent 防线那条决定）：用户完全可以自己给一句，
    那是他的话、语料里当然没有；我们没资格替他否掉。所以这里只负责把
    "这句在语料里查无出处"摆到台面上，由人去判断是谁写的。

    比对按分句做：称呼这类答案是多段拼起来的，整段查不到不代表每一段都是编的；
    只要有一段能在语料里找到，就不报——**宁可漏报，不可误伤真摘录**。"""
    corpus = Path(corpus_dir) if corpus_dir else None
    if not corpus or not corpus.exists():
        return []
    if corpus.is_dir():
        blob = "".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in corpus.rglob("*") if p.is_file())
    else:
        blob = corpus.read_text(encoding="utf-8", errors="ignore")
    hay = _normalize_for_lookup(blob)
    qmap = {q.qid: q for q in questions}
    out = []
    for qid, ans in (answers or {}).items():
        q = qmap.get(qid)
        if q is None or q.kind != "pick":
            continue
        text = ans.get("pick", "") if isinstance(ans, dict) else str(ans or "")
        if not text.strip():
            continue
        parts = [s for s in _CLAUSE_SEP_RE.split(text) if s.strip()] or [text]
        if not any(_normalize_for_lookup(s) and _normalize_for_lookup(s) in hay
                   for s in parts):
            out.append((qid, text))
    return out


PICK_SOURCE_NOTE = ("【查无出处】上面这些 pick 题的答案在语料里找不到原话。"
                    "pick 题只许挑、不许写——**如果这句是对方自己给的，那没问题**；"
                    "如果是你替 TA 想的一句好听的，删掉，宁可这一格空着。")

_FILE_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def corpus_coverage(corpus_dir, entries=None):
    """记忆库覆盖的日期区间 → ("YYYY-MM-DD", "YYYY-MM-DD")，问不出来就返回 None。

    两个来源，都不猜：
      - entries（这次导入的条目）：用真实时间戳，最准；
      - 已有语料目录：用**文件名里的日期**——那是 parse_chunk_timestamp 的最高优先级
        来源，也是 write_corpus 自己写文件时遵守的命名规则。**刻意不退 mtime**：
        mtime 在这里不是"不太准"是全错且整齐地错（复制一遍目录会把所有文件刷成
        同一时刻），拿它去告诉模型"你的记忆止于哪天"，等于给它一个假边界——
        比不给更糟。问不出来就不写这条字段，空着是诚实的。"""
    days = []
    for e in entries or []:
        ts = getattr(e, "timestamp", None)
        if ts:
            days.append(datetime.fromtimestamp(ts).strftime("%Y-%m-%d"))
    if not days and corpus_dir:
        for p in Path(corpus_dir).rglob("*.md"):
            m = _FILE_DATE_RE.search(p.name)
            if m:
                days.append(m.group(0))
    return (min(days), max(days)) if days else None


def write_bundle(out_dir, persona, client="claude-code", corpus_dir=None,
                 server_path=None, confirmed=False, entries=None,
                 contract_base=None, previous_persona=None, route=None):
    """产出三件套（generic 档多一份注入契约副本）。**confirmed=False 时拒绝写盘**——写用户磁盘要过确认关卡
    （规格 §7：人格文件任何改动必须用户确认）。

    **只写我们自己的产出，只动我们上一次真的出过的那一个文件**（2026.08.02 三轮
    验收后改准；原话是"不动同目录其它 md"，退役逻辑加进来之后那句已经不成立）。
    `previous_persona` 是**上一次出货写下的人格文件名**，由调用方（CLI 从
    init_state.json）传进来；只有它、且它跟这次的档不同名时才退役。不传就一个都不碰。

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
    written = write_corpus(out / "memory", entries) if entries else []
    corpus = Path(corpus_dir) if corpus_dir else out / "memory"
    # **覆盖区间要在渲染之前写进人格文件**：它是每轮都在的那一层，而护栏挂在工具
    # 返回值上的话，模型一绕过工具（grep、直接读文件）就一条都不生效
    span = corpus_coverage(corpus, entries)
    if span:
        persona.add_field(Field(
            id=COVERAGE_FIELD, section="architecture", label="记忆库覆盖范围",
            value=COVERAGE_TEMPLATE.format(start=span[0], end=span[1]),
            source="system", confirmed=True))
    # **覆盖自己之前那份人格文件前，先备份**（2026.08.01，维护者在场时实测撞出来）：
    # 老用户升级只需重跑一次 ship，人格文件就会按新版协议层重放一遍——这是对的，
    # 但**手改过的内容不在 state 里，重放等于把它删掉，而且原来一声不吭**。
    #
    # 这条的分量在于：《给AI的引导指南》明写着"人格文件是 TA 的，随时可以自己改"
    # ——**我们鼓励了他改，然后在升级时悄悄替他删掉**。而换档退役旧档那条我们都
    # 留了 `.bak`，**自己覆盖自己反而不留**，前后不一致。
    #
    # 做两件事，都只加不减：
    #   ① 磁盘上已有同名人格文件就先备份成 `<名>.bak`（同换档退役的命名与保守原则）；
    #   ② **比对磁盘那份与本次重放的结果**，不一致就说明它被改过（手改，或旧版本
    #      出的），把这件事明确说出来——沉默地覆盖才是最坏的形态。
    #      注意判据取"内容不同"而不是"有没有 .bak"：第一次出货时磁盘上没有旧文件，
    #      那不是"被改过"，不该报。
    previous_text = persona_path.read_text(encoding="utf-8") if persona_path.exists() else None
    rendered = render_persona_md(persona)
    overwritten = None
    if previous_text is not None and previous_text != rendered:
        backup = persona_path.with_name(persona_path.name + ".bak")
        backup.write_text(previous_text, encoding="utf-8")
        overwritten = backup
    persona_path.write_text(rendered, encoding="utf-8")
    # 换档时把**上一次我们自己出的**那份人格文件退役掉（改名 .bak，不删——写用户
    # 磁盘一律保守）。
    #
    # 为什么要退役（2026.08.02 二轮验收打回）：两份人格文件并排躺着，此刻内容相同，
    # 但日后只有当前档那份会跟着升层／更正更新，另一份变成**永不更新的影子副本**。
    # 契约第三条要求"每会话从磁盘重读人格文件"、坑④的自查又让作者对着人格文件核
    # 请求体——他要是拼错了那一个，症状恰好是契约第三条违反后描述的"确认过的更新
    # 悄无声息不生效"，且几乎无法自查（文件确实存在、内容也确实像那么回事）。
    #
    # **为什么判据是"上次出过的那一个"而不是"所有别的档文件名"**（三轮验收打回，
    # 上一版就是后者）：`CLAUDE.md` 根本不是我们的专属文件名，它是 Claude Code 的
    # 项目约定文件——自建前端作者的项目目录里躺着一份他自己写的 CLAUDE.md 太正常
    # 了（他自己也用 Claude Code 干活）。按文件名退役，等于全程只用 generic、从没
    # 换过档的用户，一跑出货就被我们把项目指令文件改了名，还配一句"换档后旧档不再
    # 更新"的错提示；后果是 Claude Code 从此读不到他的项目规矩，症状同样难自查。
    # 目录里出现某个文件名，从来不等于那文件是我们写的。
    retired = []
    if previous_persona and previous_persona != CLIENT_FILENAMES[client]:
        stale = out / previous_persona
        if stale.exists():
            bak = stale.with_name(previous_persona + ".bak")
            stale.replace(bak)      # 同名 .bak 已存在就覆盖：影子副本不值得留两份
            retired.append(bak)
    # server 默认取**本文件同目录**的 mcp_server.py，不取当前工作目录——
    # 出货时 cwd 是什么谁也保证不了，而这两个文件在包里永远是同级
    server = server_path or Path(__file__).resolve().parent / "mcp_server.py"
    cfg = mcp_config_snippet(server, corpus, out / "threads.jsonl", route=route)
    (out / "mcp-config.json").write_text(cfg, encoding="utf-8")
    contract = None
    if client == "generic":
        src = contract_source(contract_base)
        if not src.exists():
            # 不静默降级：契约缺了就出一份"看着正常、其实少了一半"的货，
            # 而缺的正是自建前端唯一必须照做的那部分
            raise FileNotFoundError(f"generic 档要随货带注入契约，但没找到 {src}")
        contract = out / CONTRACT_DOC
        contract.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return {"persona": persona_path, "memory_dir": out / "memory",
            "mcp_config": out / "mcp-config.json", "corpus_files": written,
            "contract": contract, "retired": retired,
            "overwritten_backup": overwritten}


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

    # 5.【变异靶心：归属句式】立场写进 md 是"<用户>认为…"，不是断言。
    #    **2026.08.02 这条从钉死"她认为："改成两档都钉**：人称不再由我们写死，
    #    所以原来那个字符串本身就是本卡要修的东西。改法是加严不是放宽——
    #    判得出人称就必须用真人称，判不出来必须走中性写法，两档各钉一次。
    p2 = Persona("partner")
    apply_answers(p2, QUESTIONS, {"continuity": "A"}, pronouns={"user": "他", "ai": "她"})
    stance = [f for f in p2.fields if f.id == "opening_continuity"][0]
    assert stance.value.startswith("他认为："), f"立场必须用归属句式：{stance.value}"
    p2n = Persona("partner")
    apply_answers(p2n, QUESTIONS, {"continuity": "A"})       # 人称未知
    stance_n = [f for f in p2n.fields if f.id == "opening_continuity"][0]
    assert stance_n.value.startswith("对方认为："), \
        f"人称判不出来时该走中性归属句式，不许塞一个默认的他/她：{stance_n.value}"
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

    # 8a.【靶心：人称锚死一套——「你」只能是模型】第一份真实人格文件样本暴露：
    #     开篇同一节内，opening_recognition 的「你」是用户、opening_refusal_ok 的
    #     「你」是模型。人格文件的读者只有一个（模型），「你」指两个人会让指代解析
    #     出错——拒绝权那条本意是给模型的授权，含义会直接翻转。
    #     **靶子只取我们自己生成的文本**（协议层默认值 + 选项 directive）：用户挑的
    #     原话怎么写是他的自由，我们管不着也不该管。
    p_pron = Persona("partner")
    fill_protocol_defaults(p_pron)
    qs_pron = questions_for(coverage_report(p_pron), has_corpus=False)
    apply_answers(p_pron, qs_pron,
                  {q.qid: {"keys": "".join(sorted(q.options))} for q in qs_pron if q.options})
    apply_confirmations(p_pron, {p.key: "keep" for p in pending_confirmations(p_pron)})
    pron_md = render_persona_md(p_pron)
    #     用户被写成第二人称的形态——这些串一旦出现，就说明有人把「你」当用户写了
    #     字段**标题**同样进文件，同样受这条约束（"它该记住你哪些方面"就是这么漏的）
    for bad in ("你写下的", "你不用每次", "记住你的", "等你问", "不等你问",
                "你手上在忙", "接住你的情绪", "你的喜好", "你的作息",
                "记住你哪些", "你哪些方面"):
        assert bad not in pron_md, \
            f"人格文件里把用户写成了第二人称（{bad!r}）——「你」在这份文件里只能是模型自己"
    #     **反向也要守，否则这条闸是空的**：把所有「你」都删光同样能让上面全过，
    #     所以必须确认模型第二人称真的在（检索约定那条是基准写法，一字不许动）
    #     **逐字钉死**，不是钉个片段：这段是 2026.07.31 第三轮真机实测通过的措辞，
    #     "其余向它对齐、它自己一字不动"是任务卡写死的要求。黄金串写在断言里而不是
    #     引用常量本身——引用常量的话，改了常量断言跟着变，等于没钉

    # 8a3.【靶心：人称从语料/用户来，不由我们写死——验的是真出货的那份文件】
    #      这一条按任务卡的要求**取真出货的人格文件正文**，不是只查模板里的占位符
    #      换没换：占位符换对了、渲染时又漏填一处，模板检查照样全绿。
    def _ship_and_read(pronouns_answers, entries=None):
        p_x = Persona("partner")
        detected = {}
        pr = pronouns_from_answers(list(PRONOUN_QUESTIONS.values()),
                                   pronouns_answers, detected)
        fill_protocol_defaults(p_x, pr)
        qs_x = questions_for(coverage_report(p_x), has_corpus=False, pronouns=pr)
        all_ans = dict(pronouns_answers or {})
        for q in qs_x:
            if q.options and q.qid not in all_ans:
                all_ans[q.qid] = {"keys": "".join(sorted(q.options))}
            elif q.kind == "short" and q.qid not in all_ans:
                all_ans[q.qid] = {"pick": "说好了就算数。"}
        apply_answers(p_x, qs_x, all_ans, pr)
        apply_confirmations(p_x, {p.key: "keep" for p in pending_confirmations(p_x)})
        with tempfile.TemporaryDirectory() as td_x:
            paths_x = write_bundle(td_x, p_x, client="claude-code",
                                   confirmed=True, entries=entries)
            return paths_x["persona"].read_text(encoding="utf-8")

    #      **骨架节标题也在靶子里**（2026.08.02 验收打回二）：原来这里把节标题整行
    #      排除掉，于是 `## 她是谁` 对一个选了「他」的用户永远不可见——那正是
    #      "产出的人格文件不许人称混乱"说的毛病本身。标题现在带 {ta} 槽、跟正文走
    #      同一套填法，排除也就不需要了。
    def _body(md_text):
        return md_text

    #      ① 用户选「他」：全文只能有「他」这一种用户称呼，不许混进「她」
    md_he = _body(_ship_and_read({"pronoun_user": {"keys": "A"}, "pronoun_ai": {"keys": "B"}}))
    assert "他" in md_he, "选了「他」，出货文件里却一个都没有"
    assert "她" not in md_he, \
        f"用户选了「他」，文件里却还有「她」——人称混着走正是这一单要修的：{md_he[:200]!r}"
    #      ② 反过来同理（防"把所有他都替换成她"这种一半的实现）
    md_she = _body(_ship_and_read({"pronoun_user": {"keys": "B"}, "pronoun_ai": {"keys": "A"}}))
    assert "她" in md_she and "他" not in md_she, "选了「她」，文件里不该再出现「他」"
    #      ③ **全文零「它」**（维护者的硬约束，含字段标题与所有 directive 渲染结果）
    #      ④ 人称判不出、用户也没答：中性写法，**不许塞「它」，也不许塞默认的他/她**
    md_neutral = _body(_ship_and_read({}))
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        #      **占位符不许漏进出货文件**：漏填渲染出的是字面 "{ta}是谁"，里面既没有
        #      他也没有她，上下几条断言全过——变异④就是从这条缝里过去的
        assert not _SLOT_RE.search(md_x), \
            f"[{tag}] 出货文件里还留着人称占位符，说明有一条渲染路径漏填了：" \
            f"{_SLOT_RE.search(md_x).group(0)}"
        assert "它" not in md_x, \
            f"[{tag}] 出货的人格文件里出现了「它」——硬约束是任何时候都不许，" \
            f"判不出来就问、或者改写成不需要人称的句子"
    assert "他" not in md_neutral and "她" not in md_neutral, \
        "人称判不出来时塞了一个默认的他/她——那正是这一单要修掉的东西"
    #      ⑥ **判得出人称时，用户只能有一种称呼形态**（任务卡 4b）：不能一处「他」、
    #      一处「对方」、一处「用户」地混着走。**唯一的已知例外是检索约定那条黄金串**
    #      里的「对方」——它是三轮真机验证换来的措辞、一字不动，两条约束打架时
    #      以不动黄金串为准（这一格待维护者裁定，断言按现状钉住，裁完再动）。
    #      这条挡的是"把某句里的用户硬编码成对方/用户"——那类字符串不含他/她/它，
    #      上面几条闸一个都拦不住（变异②就是从这儿过去的）
    #      黄金串在这里就地写一份（下面 8e 还会再钉一次逐字不变）——两处都写死是
    #      刻意的：引用常量的话，常量被改了断言跟着变，等于没钉
    _GOLDEN_HEAD = "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名"
    for tag, md_x in (("选他", md_he), ("选她", md_she)):
        rest = md_x.replace(_GOLDEN_HEAD, "")
        assert "对方" not in rest, \
            f"[{tag}] 判得出人称了却还有「对方」——用户在同一份文件里有了两种称呼：" \
            f"…{rest[max(0, rest.find('对方') - 30):rest.find('对方') + 20]}…"
    #      「用户」也不行：那是第三种形态，而且在人格文件里读起来像产品文档
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        assert "用户" not in md_x, f"[{tag}] 人格文件里把对方称作「用户」了"

    #      ⑤ 「你」＝模型这条锚不能因为参数化而丢（原 8a 那条黑名单继续守）
    assert "你每次都会读" in md_neutral, "「你」＝模型这条锚丢了"

    #      ⑥ **判不出来必须真的去问**——上面几条都是直接把人称喂进渲染的，
    #      绕过了出题这一步；不钉这条，"静默跳过不问"能全绿（变异验过）
    rep_pron = coverage_report(Persona("partner"))
    asked_unknown = {q.qid for q in questions_for(rep_pron, has_corpus=False)}
    assert {"pronoun_user", "pronoun_ai"} <= asked_unknown, \
        f"人称判不出来却没问用户——那就只能塞默认值或走中性，两条都不如问一句：{sorted(asked_unknown)}"
    asked_known = {q.qid for q in questions_for(rep_pron, has_corpus=False,
                                                pronouns={"user": "他", "ai": "她"})}
    assert not ({"pronoun_user", "pronoun_ai"} & asked_known), \
        "语料里已经判出人称了还去问，等于让用户做无用功"
    #      单侧判出来就只问另一侧
    asked_half = {q.qid for q in questions_for(rep_pron, has_corpus=False,
                                               pronouns={"user": "他"})}
    assert "pronoun_ai" in asked_half and "pronoun_user" not in asked_half, \
        f"只该问判不出来的那一侧：{sorted(asked_half)}"

    # 8a3b.【靶心：语料侧判定在**真 CLI 上**真的会触发】验收打回一：
    #      `_detect_pronouns_from_corpus` 当初没把用户侧说话人传进去，于是
    #      `detect_pronouns` 走"分不清谁是谁"那条分支、**永远返回 None**——
    #      语料里 AI 说了十几次「她」也照样问用户一遍。而 8a5 那组断言全是自己
    #      把 `user_speakers` 喂进去测的，**函数绿、生产路径死**。
    #      这是同一个形状第五次出现（显式 --client 被吃掉／not exists 恒真／
    #      钉住函数没钉住命令／--answers 那条没被钉住），所以这条走真进程。
    import subprocess          # 走真进程：函数级断言钉不住"生产路径有没有调它"
    with tempfile.TemporaryDirectory() as td:
        conv = {"title": "t", "create_time": 1.0, "mapping": {}}
        nodes = {}
        for i in range(12):
            who = "assistant" if i % 2 == 0 else "user"
            # AI 提到用户用「她」，用户提到 AI 用「他」——两侧各 6 次，都过闸
            txt = "她今天早点睡" if who == "assistant" else "他说得对"
            nodes[f"n{i}"] = {"message": {"author": {"role": who},
                                          "create_time": 1750000000.0 + i,
                                          "content": {"parts": [txt]}},
                              "parent": f"n{i-1}" if i else None,
                              "children": [f"n{i+1}"] if i < 11 else []}
        conv["mapping"] = nodes
        conv["current_node"] = "n11"
        corpus = Path(td) / "export.json"
        corpus.write_text(json.dumps([conv], ensure_ascii=False), encoding="utf-8")
        out_json = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", td,
             "--corpus", str(corpus), "--step", "questionnaire", "--json"],
            cwd=td, capture_output=True, text=True, encoding="utf-8")
        assert out_json.returncode == 0, f"CLI 跑挂了：{out_json.stdout}\n{out_json.stderr}"
        payload = json.loads(out_json.stdout)
        assert payload["pronouns_detected"] == {"user": "她", "ai": "他"}, \
            f"真 CLI 上语料侧判定没触发——功能是死的：{payload['pronouns_detected']}"
        asked_ids = {q["qid"] for q in payload["questions"]}
        assert not ({"pronoun_user", "pronoun_ai"} & asked_ids), \
            f"语料里已经判出人称了还问用户，等于白做：{sorted(asked_ids)}"
        #    存进状态、下一步续跑时还在（渲染要用）
        st = json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8"))
        assert st.get("pronouns_detected") == {"user": "她", "ai": "他"}, \
            "判出来的人称没写回状态，下一步渲染就又不知道了"

    #      **chatlog 语料（说话人是日志里的真名）在真 CLI 上必须退回去问**：
    #      这是第二次返工那条的生产路径版——函数里判对了，还得确认这一路真的
    #      走到"去问用户"，而不是在别处又被硬分回去
    with tempfile.TemporaryDirectory() as td2:
        log = Path(td2) / "chat.txt"
        log.write_text("\n".join(
            f"2026-07-1{i%9} 21:0{i%6}:00 小明\n他说得对，他记性好" for i in range(8)),
            encoding="utf-8")
        r2 = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", td2,
             "--corpus", str(log), "--step", "questionnaire", "--json"],
            cwd=td2, capture_output=True, text=True, encoding="utf-8")
        assert r2.returncode == 0, f"CLI 跑挂了：{r2.stdout}\n{r2.stderr}"
        pay2 = json.loads(r2.stdout)
        assert pay2["pronouns_detected"] == {}, \
            f"chatlog 的真名说话人分不清谁是谁，却判出了人称——那多半是反的：" \
            f"{pay2['pronouns_detected']}"
        assert {"pronoun_user", "pronoun_ai"} <= {q["qid"] for q in pay2["questions"]}, \
            "判不出来就该问，这条路上没问等于把中性写法当默认值用了"

    #      **叙事体语料判不出来是正常的，跟"能判却没判"不是一回事**：那种语料
    #      （timeline md，speaker 为空）本来就没有说话人标记，退回去问用户就对了
    from memory_import import MemoryEntry as _ME0
    narr = [_ME0(timestamp=1.0, speaker="", text="她说今天早点睡") for _ in range(8)]
    assert detect_pronouns(narr, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "叙事体语料没有说话人标记，判不出来才对——但它该走'去问用户'，不是硬判"

    #      **丢主语的接续要挡一下**（2026.08.02 验收打回）：中性档第一句曾经渲染成
    #      "——写下的东西都在这里"，谁写下的没了。手写的中性写法一样会丢主语，
    #      跟机械删词的结果没差别，而它落在人格文件第一句、不变量层、每轮都在。
    #      **这条断言替代不了人眼通读**（中文通不通机器判不了），它只挡住这一类
    #      最常见的形态——破折号/分号后面直接跟动词
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        for bad in ("——写下", "；写下", "——说过", "；说过", "——提到", "；提到"):
            assert bad not in md_x, \
                f"[{tag}] 这句丢了主语（{bad!r}）：中性写法也会丢主语，不是只有机械删词会"

    # 8a4.【靶心：中性写法一条都不许漏】机械删词会写出半通不通的中文，且不报错，
    #      所以中性写法是逐条手写的——那就必须有东西守着"每条模板都有对应的手写形态"。
    templates = [ATTRIBUTION_PREFIX, NOTE_LEAD]
    templates += [v for _, _, v in PROTOCOL_DEFAULTS.values()]
    templates += [lbl for _, lbl, _ in PROTOCOL_DEFAULTS.values()]
    for q in QUESTIONS:
        templates += [q.label, q.text]
        for v in (q.options or {}).values():
            templates += [v[0], v[1]]
    for t in [t for t in templates if t and _SLOT_RE.search(t)]:
        assert t in NEUTRAL_FORMS, f"这条模板带人称槽却没有中性写法：{t!r}"
        assert not _SLOT_RE.search(NEUTRAL_FORMS[t]), \
            f"中性写法里还留着槽位，等于没写：{NEUTRAL_FORMS[t]!r}"
        assert "它" not in NEUTRAL_FORMS[t], f"中性写法里塞了「它」：{NEUTRAL_FORMS[t]!r}"
        assert not any(p in NEUTRAL_FORMS[t] for p in PRONOUN_CHOICES), \
            f"中性写法里塞了默认的他/她：{NEUTRAL_FORMS[t]!r}"

    # 8a5.【靶心：语料侧判定不许猜】第三人称在两个人的对话里多半指的是别人，
    #      所以样本少、或者两个词旗鼓相当时**必须返回 None 去问用户**，不许硬判。
    from memory_import import MemoryEntry as _ME
    #      **陌生说话人标签＝分不清谁是谁，必须 None**（2026.08.02 第二次返工）：
    #      chatlog 翻译器出的是日志里的真名（"小明""星回"），两边都不在已知名单里。
    #      上一版拿"在名单里＝用户、其余全算 AI"硬分，于是**用户谈论自己 AI 的话被
    #      读成"AI 在谈论用户"**，判出来的人称正好是反的，还一个字都不问。
    #      这一档比"判不出来"糟得多：判不出只是多问一句，判错了没有任何人会知道。
    chatlog_one = [_ME(timestamp=1.0 + i, speaker="小明", text="他说得对，他记性好")
                   for i in range(6)]
    assert detect_pronouns(chatlog_one, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "陌生说话人标签下必须判不出来——硬分会把'用户在说他的 AI'读成'AI 在说用户'"
    chatlog_two = chatlog_one + [_ME(timestamp=9.0 + i, speaker="星回", text="她今天早点睡吧")
                                 for i in range(6)]
    assert detect_pronouns(chatlog_two, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "两个都是陌生名字时更不能猜——谁是用户谁是 AI 语料里根本没写"
    #      认得出**任意一侧**的标签就能分：两个翻译器归一成 user/assistant，
    #      只出现其中一个（另一侧是真名）也照样分得清
    known_both = ([_ME(timestamp=1.0 + i, speaker="assistant", text="她今天早点睡")
                   for i in range(6)]
                  + [_ME(timestamp=9.0 + i, speaker="user", text="他说得对") for i in range(6)])
    assert detect_pronouns(known_both, USER_SPEAKERS) == {"user": "她", "ai": "他"}, \
        "两边都是已知标签，这是 ChatGPT/Claude 导出的常态，必须判得出来"
    known_ai_only = ([_ME(timestamp=1.0 + i, speaker="assistant", text="她今天早点睡")
                      for i in range(6)]
                     + [_ME(timestamp=9.0 + i, speaker="小明", text="他说得对") for i in range(6)])
    assert detect_pronouns(known_ai_only, USER_SPEAKERS) == {"user": "她", "ai": "他"}, \
        "只认得 AI 侧标签时，剩下那个就是用户——这一档不该退化成判不出来"
    #      单侧有话、另一侧没话：判得出的那侧照判，判不出的那侧必须是 None——
    #      **两侧各判各的，不许拿一侧的结论去补另一侧**。
    #      这里说话人是已知的 AI 标签（"ai"），所以分得清侧；跟上面 chatlog 那档
    #      在"只有一侧说话"上同构，**区别正是标签认不认得**——当初没区分这一点，
    #      正是上一版把陌生名字也硬分了的原因
    strong = [_ME(timestamp=1.0, speaker="ai", text="她今天早点睡")] * 6
    one_side = detect_pronouns(strong, user_speakers={"me"})
    assert one_side["user"] == "她", f"AI 说了六次「她」，用户侧该判得出：{one_side}"
    assert one_side["ai"] is None, \
        f"用户一句话都没说，AI 侧无从判定，必须是 None：{one_side}"
    mixed = ([_ME(timestamp=1.0, speaker="ai", text="她今天早点睡")] * 5
             + [_ME(timestamp=2.0, speaker="me", text="他说得对")] * 5)
    got = detect_pronouns(mixed, user_speakers={"me"})
    assert got["user"] == "她" and got["ai"] == "他", f"频次占优时该判得出来：{got}"
    thin = [_ME(timestamp=1.0, speaker="ai", text="她好")] * 2
    assert detect_pronouns(thin, user_speakers={"me"})["user"] is None, \
        f"样本太少必须返回 None 去问用户，不许拿两三次出现就下结论"
    tie = ([_ME(timestamp=1.0, speaker="ai", text="她 他 她 他")] * 3)
    assert detect_pronouns(tie, user_speakers={"me"})["user"] is None, \
        "两个词旗鼓相当时必须判不出来，不许挑一个"

    # 8d.【靶心：记忆库的时间边界要告诉模型】真机验证的第二个结论：模型说"后来没有
    #     继续展开"，而那件事根本不在语料里（发生在另一个客户端）——**在它能看见的
    #     世界里那句话是真的**。真缺陷是它不知道自己的记忆止于哪一天，于是把
    #     "我的记录里没有"说成了"没发生过"（差别在 authority，不在语气）。
    with tempfile.TemporaryDirectory() as td:
        paths_cov = write_bundle(td, p5, client="claude-code", confirmed=True, entries=ents)
        cov_md = paths_cov["persona"].read_text(encoding="utf-8")
        assert "记忆库覆盖范围" in cov_md and "范围之外的事你没有记录" in cov_md, \
            "出货的人格文件没告诉模型记忆库覆盖到哪天——它就说不出'我的记录到此为止'"
        #    日期要是真的，不是模板占位
        days_in = sorted(datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
                         for e in ents if getattr(e, "timestamp", None))
        assert days_in[0] in cov_md and days_in[-1] in cov_md, \
            f"覆盖区间跟语料对不上（语料 {days_in[0]}~{days_in[-1]}）：{cov_md[-200:]!r}"
    #     **问不出来就不写**：宁可没有这条，也不能给一个假边界——模型会照着假边界
    #     去否定真事，比不给更糟
    assert corpus_coverage(None, entries=[]) is None, "问不出日期时不该编一个区间出来"
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "memory" / "timeline"
        bare.mkdir(parents=True)
        (bare / "window_01.md").write_text("## 没有日期的窗口\n随便写点。", encoding="utf-8")
        assert corpus_coverage(Path(td) / "memory") is None, \
            "文件名没日期时该返回 None——不许退 mtime，那是全错且整齐地错的假边界"
    #     **文件名日期那条路要有断言真走过一遍**（2026.08.02 补，审查侧留的缺口）：
    #     上面两条都只钉"问不出来时返回 None"，于是把 `_FILE_DATE_RE` 改坏，全量
    #     自检照样绿。而它是 `--corpus`（拿现成语料目录、没有 entries）那条路
    #     **唯一的**日期来源，坏了就是静默不写覆盖区间——症状恰好是覆盖区间这一单
    #     要修的那个：模型不知道自己的记忆止于哪天，于是替看不见的那段时间下结论。
    #     混一个不带日期的文件进去，确认它被跳过而不是把整条路带崩。
    with tempfile.TemporaryDirectory() as td:
        dated = Path(td) / "memory" / "timeline"
        dated.mkdir(parents=True)
        for name in ("2026-03-05_window_01.md", "2026-07-19_window_02.md",
                     "没有日期的_window_03.md"):
            (dated / name).write_text("## 一段\n随便写点。", encoding="utf-8")
        span = corpus_coverage(Path(td) / "memory")
        assert span == ("2026-03-05", "2026-07-19"), \
            f"文件名日期没被解析出来（--corpus 那条路唯一的日期来源）：{span}"

    # 8g.【靶心：pick 题只许挑、不许写】真机里执行者自己创作了一句放进候选并被
    #     采用，落点是最终约定——人格文件里象征意味最重的那一格。两道一起钉：
    #     ①问卷 prompt 里这条纪律的措辞不许飘（它是唯一到得了执行者手上的地方）；
    #     ②`unsourced_picks` 真能把查无出处的答案挑出来，且**不误伤真摘录**。
    for must in ("只许挑，不许写", "不要润色", "找不到就如实说找不到"):
        assert must in export_llm_prompt(QUESTIONS), \
            f"问卷 prompt 里丢了 pick 纪律的措辞：{must!r}——那是唯一到得了执行者手上的地方"
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "memory"
        cp.mkdir()
        #    称呼那两段**刻意分散在语料两头、顺序还相反**：整段比对必然查无出处，
        #    只有分句之后才各自找得到。fixture 要是把两段挨着写、顺序也一样，
        #    整段就成了原样子串，`parts` 那一层被拿掉照样绿（返工前正是这样）
        (cp / "w01.md").write_text(
            "我叫她“小满”，一直这么叫。\n"
            "那天她说“到家了发一句”，我说好。\n"
            "很久以后她才说，她叫我“老陈”是因为那部电影。",
            encoding="utf-8")
        qs_pick = QUESTIONS
        #    真摘录，**但标点跟语料不一样**：语料是全角弯引号 + 中文逗号，
        #    答案给的是直引号 + 末尾句号。归一化那一层被拿掉就会误伤这条。
        #    （返工补：原 fixture 的答案是语料的原样子串，一个标点都没改过，
        #    于是 `_normalize_for_lookup` 改成 `return text` 照样绿——断言声称在守
        #    归一化，实际上什么都没守。**fixture 跟注释说的不是一回事**，
        #    同"fixture 的规模也是一种伪影"是同一族毛病。）
        ok = unsourced_picks(qs_pick, {"closing_pick": {"pick": '"到家了发一句".'}}, cp)
        assert not ok, f"改过标点的真摘录被判成自造了（归一化没做够）：{ok}"
        #    模型自己写的一句好听的：查无出处，必须被挑出来
        bad = unsourced_picks(
            qs_pick, {"closing_pick": {"pick": "你来，我就在。"}}, cp)
        assert [q for q, _ in bad] == ["closing_pick"], \
            f"自造的最终约定没被标出来——流程就分不清语料里的真话和模型写的漂亮话：{bad}"
        #    分句比对：称呼这类答案是多段拼起来的，**只要有一段能在语料里找到就不报**。
        #    整段直接比对必然查无出处（拼接顺序、连接符都是我们自己拼的），
        #    这条是 `parts` 那一层唯一的用武之地，原先没有断言走过它
        assert not unsourced_picks(
            qs_pick, {"naming_pick": {"pick": "她叫我“老陈”；我叫她“小满”"}}, cp), \
            "多段拼接的称呼被整段比对判成自造了——parts 分句那一层没生效"
        #    只查 pick 题：**答案带 pick 字段、但题本身不是 pick 题**，照样不该卷进来。
        #    （返工补：原先传的是 {"keys": "A"}，压根没有 pick 字段，走到空串就
        #    continue 了——它挡的是"答案没有 pick 字段"，不是"这题不是 pick 题"，
        #    把 `q.kind != "pick"` 删掉照样绿。）
        assert not unsourced_picks(
            qs_pick, {"disagree": {"pick": "一句语料里绝对没有的话"}}, cp), \
            "非 pick 题参与了出处核对——选择题的内容本来就是我们写的，核对它没有意义"
        #    **只标注不硬拒**：这里只返回清单，不抛异常——用户自己给的一句
        #    语料里当然没有，我们没资格替他否掉
    #     ③**护栏要真的到得了执行者手上**：函数正确但没人调用，等于没有。
    #     走 CLI 真进程，断言标注真的印在 --step answers 的输出里。
    #     （返工补，审查侧的对抗变异：把 `_step_answers` 里那行
    #     `_report_unsourced_picks(...)` 删掉，函数还在、自检全绿，而护栏的全部
    #     价值就是让执行者看见它。这是本项目连着学过三次的同一个教训——
    #     显式 --client 被状态吃掉／`not exists` 恒真／函数被钉住但命令没被钉住。）
    import subprocess          # 同 8b：走真进程，函数级断言钉不住"命令有没有调它"
    with tempfile.TemporaryDirectory() as td:
        cp2 = Path(td) / "corpus"
        cp2.mkdir()
        (cp2 / "w01.md").write_text("那天她说“到家了发一句”，我说好。", encoding="utf-8")
        out2 = Path(td) / "out"

        def run_pick(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", str(out2), *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout

        run_pick("--corpus", str(cp2))
        made_up = "你来，我就在。"
        got2 = run_pick("--corpus", str(cp2), "--step", "answers",
                        "--answers-json", json.dumps({"closing_pick": {"pick": made_up}},
                                                     ensure_ascii=False))
        assert "【查无出处】" in got2 and made_up in got2, \
            "带 --corpus 跑 answers 时，查无出处的标注没印出来——护栏到不了执行者手上就等于没有"
        #    不给 --corpus 就不查，**也不假装查过**：没有语料可比对时沉默是诚实的
        got3 = run_pick("--step", "answers",
                        "--answers-json", json.dumps({"closing_pick": {"pick": made_up}},
                                                     ensure_ascii=False))
        assert "【查无出处】" not in got3, \
            "没给 --corpus 却报了查无出处——那是在拿空语料当'查过了'"
        #    这一条**只挡得住两层一起塌**：调用方的 `if not corpus_dir: return` 与
        #    `unsourced_picks` 里的 `corpus.exists()` 早返回互为兜底，单点变异会被
        #    另一层吸收。如实记下来——不假装它是一条单点靶。
        #    **另一条路径也要钉**：`--answers <答案清单文件>`（走 `问卷prompt.txt`
        #    的用户回来时用的就是它）。两条路径各有一次调用，只钉一条的话，
        #    另一条被删掉照样全绿——返工前就是这样，⑦号变异是绿的。
        #    题号从 CLI 自己吐的问卷里取，不手写——手写的题号会跟问卷顺序脱节
        qs_j = json.loads(run_pick("--corpus", str(cp2), "--step", "questionnaire", "--json"))
        qids_j = [q["qid"] for q in qs_j["questions"]]
        sheet = Path(td) / "answers.txt"
        sheet.write_text(f"{qids_j.index('closing_pick') + 1}. {made_up}\n",
                         encoding="utf-8")
        got4 = run_pick("--corpus", str(cp2), "--step", "answers",
                        "--answers", str(sheet))
        assert "【查无出处】" in got4 and made_up in got4, \
            "走答案清单文件那条路时标注没印出来——两条路径要各钉各的"

    # 8e.【靶心：止血纪律必须在人格文件里，且不动已验证的句子】
    #     护栏挂在工具返回值上，模型一绕过工具（grep、直接读文件）就一条都不生效，
    #     而绕过恰恰发生在检索失败、最需要护栏的时候。人格文件是唯一覆盖
    #     "不管你用什么方式查"的那一层。
    p_conv = Persona("partner")
    fill_protocol_defaults(p_conv)
    conv = next(f.value for f in p_conv.fields if f.id == RETRIEVAL_CONVENTION_FIELD)
    for must in ("片段，不是全部", "grep", "不要说“没发生过”", "我的记录里没有"):
        assert must in conv, f"检索约定里缺行为层这条：{must}"
    #     **只做加法**：三轮真机验证过的那两段一字不许动，逐字钉死（黄金串写在
    #     断言里，不引用常量——引用常量的话改了常量断言跟着变，等于没钉）
    GOLDEN_RETRIEVAL = (
        "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名，或者你对"
        "细节拿不准时，先用记忆检索工具（memory_search）查一遍再开口；不要在查"
        "之前说“我不记得”。查完自然接上话，不用报告自己搜过。")
    assert GOLDEN_RETRIEVAL in pron_md, \
        "检索约定那段的措辞被改了——它是三轮真机验证过的基准写法，一字不许动（其余向它对齐）"
    assert pron_md.count("你") >= 2, "「你」几乎不出现，说明这条闸被'全删掉'绕过去了"

    # 8a1.【靶心：关系状态不归「我是谁」】真实样本里「我是谁」整节只有"当前关系状态"
    #      一条，而那条讲的是这段关系此刻怎么样，不是 AI 的身份。
    state_q = next(q for q in QUESTIONS if q.qid == "state_now")
    assert state_q.section == "opening", "当前关系状态该归开篇（关系确认那一节），不是 AI 的身份"
    #      渲染里它必须落在开篇那一节之内：取开篇标题到下一节标题之间的正文来看
    open_body = pron_md.split(SECTIONS["opening"], 1)[1].split(SECTIONS["user"], 1)[0]
    assert "当前关系状态" in open_body, "当前关系状态没渲染进开篇那一节"

    # 8a2.【靶心：称呼不许重复拼接】同一份真实样本里的第二条：多选几个称呼候选，
    #      共享的那半会重复出现（"她叫你'哥哥'，你叫她'阿岸'；她叫你'星回'，你叫她'阿岸'"）。
    #      整行去重没用——两行确实不同，要按分句去重。
    p_nm = Persona("partner")
    fill_protocol_defaults(p_nm)
    qs_nm = questions_for(coverage_report(p_nm), has_corpus=True)
    nm_q = next(q for q in qs_nm if q.qid == "naming_pick")
    apply_answers(p_nm, qs_nm, {"naming_pick": {"pick":
                  "她叫你“哥哥”，你叫她“阿岸”；她叫你“星回”，你叫她“阿岸”"}})
    nm_value = next(f.value for f in p_nm.fields if f.id == nm_q.field_id)
    assert nm_value.count("你叫她“阿岸”") == 1, \
        f"同一侧称呼被重复拼进人格文件：{nm_value!r}"
    #      两个不同的称呼都得留下——去重不能把内容也吃掉
    assert "哥哥" in nm_value and "星回" in nm_value, f"去重把真内容删了：{nm_value!r}"

    # 8b.【靶心：任务书不许泄漏进人格文件】外部（第三方 AI 走流程时）发现的缺陷：
    #     milestone_kinds 的选项指引是"去语料里找：……"——给模型的提取任务书，
    #     却以普通字段草稿的身份进了确认关卡，用户按 y 就写进不变量层。
    #     **走真实的 CLI 全流程**（questionnaire → answers → confirm 全 keep → ship），
    #     零语料冷启动即可复现，产出的人格文件里不得出现任务书特征串。
    #     只断言"字段不存在"是不够的——真正会伤到用户的是**写进文件的那段文本**，
    #     所以靶子取产出文件的正文。
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        def run_cli(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs_j = json.loads(run_cli("--step", "questionnaire", "--json"))
        #    每题都答满（milestone_kinds 尽量多选几项，把任务书撑到最容易泄漏的形态）
        answers_j = {}
        for q in qs_j["questions"]:
            if q["options"]:
                answers_j[q["qid"]] = {"keys": "".join(sorted(q["options"])[:3])}
            else:
                answers_j[q["qid"]] = {"pick": "她说“到家了发一句”，我说好。"}
        assert "milestone_kinds" in answers_j, "问卷里没有里程碑类型题，靶子失效"
        #    **这个字面量超过 255 字节**——缺陷二的靶子，短 fixture 抓不到（见下 8c）
        ans_literal = json.dumps(answers_j, ensure_ascii=False)
        assert len(ans_literal.encode("utf-8")) > 255, \
            f"--answers-json 的靶子字面量不够长（{len(ans_literal.encode('utf-8'))} 字节），抓不到长度上限那个洞"
        run_cli("--step", "answers", "--answers-json", ans_literal)
        listed = json.loads(run_cli("--step", "confirm", "--list", "--json"))
        #    任务书走自己的通道，不混进待确认清单
        assert listed["extraction_brief"], "任务书没进 extraction_brief 通道，等于丢了给模型的指令"
        for p in listed["pending"]:
            assert "去语料里找" not in p["value"], \
                f"任务书混进了待确认清单，用户按 y 就写进人格文件：{p['label']}"
        dec_literal = json.dumps({p["key"]: "keep" for p in listed["pending"]},
                                 ensure_ascii=False)
        assert len(dec_literal.encode("utf-8")) > 255, \
            f"--decisions-json 的靶子字面量不够长（{len(dec_literal.encode('utf-8'))} 字节）"
        run_cli("--step", "confirm", "--decisions-json", dec_literal)
        #    8b-2.【靶心：检索路线是个真的选择点，不是文档里的可选升级】（2026.08.02）
        #      三条路里有一条会把私人记忆发到第三方。**变异靶心**：把 _step_ship 里
        #      那道闸换成 `route = state.get("route", ROUTE_DEFAULT)` 时这条必红——
        #      而那正是"没问过"与"选了默认"长得一模一样的那种失效。
        r_norote = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--out", td, "--step", "ship"], cwd=td,
                                  capture_output=True, text=True, encoding="utf-8")
        assert r_norote.returncode != 0 and "检索路线还没选过" in r_norote.stdout + r_norote.stderr, \
            "没选过路线就出货了——那个选择点等于不存在"
        routes_j = json.loads(run_cli("--step", "route", "--json"))
        keys = {r["key"] for r in routes_j["routes"]}
        assert keys == {"zero-dep", "local", "cloud"}, f"三条路线要都念到：{keys}"
        #      每条都必须写清语料去向，而且云端那条必须明说会发给第三方——
        #      含糊掉这句话，用户就是在不知情的前提下做了一个不可逆的决定
        for r in routes_j["routes"]:
            assert r["语料去向"], f"{r['key']} 没写语料去向"
        cloud = next(r for r in routes_j["routes"] if r["key"] == "cloud")
        assert "发到" in cloud["语料去向"] and "服务商" in cloud["语料去向"], \
            "云端那条没把'查询和被检索的内容都会发到那家服务商'说出来"
        assert next(r for r in routes_j["routes"] if r["default"])["key"] == ROUTE_DEFAULT
        run_cli("--step", "route", "--route", "zero-dep")
        ship_out = run_cli("--step", "ship", "--client", "claude-code")
        persona_text = (Path(td) / "CLAUDE.md").read_text(encoding="utf-8")
        assert "去语料里找" not in persona_text, \
            "任务书泄漏进人格文件了——不变量层里坐着一句没有日期、没有原话、没有当下状态的指令"
        #    但指令本身不能就这么丢了：它得从任务书通道出去，第二阶段照着干活
        assert "去语料里找" in ship_out and "不是人格文件内容" in ship_out, \
            "任务书既没进人格文件、也没从任务书通道出来——那是把用户的选择直接扔了"

    # 8c.【靶心：长字面量不许崩】缺陷二。`_load_json_arg` 原先路径优先且不兜
    #     OSError，字面量超过 255 字节直接 File name too long。上面 8b 已用真实规模的
    #     字面量走过一遍 CLI，这里再直接钉住函数本身，两头都堵：
    #     **短 fixture 抓不到这个缺陷——fixture 的规模本身就是一种伪影。**
    long_payload = {f"field:占位{i:02d}": "keep" for i in range(11)}
    long_literal = json.dumps(long_payload, ensure_ascii=False)
    assert len(long_literal.encode("utf-8")) > 255, "靶子本身不够长，测了个寂寞"
    assert _load_json_arg(long_literal) == long_payload, \
        "超过文件名长度上限的 JSON 字面量该照常解析，不该抛 OSError"
    #     真是路径时仍然读文件（别把修法做成"永远不认路径"）
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "dec.json"
        f.write_text(long_literal, encoding="utf-8")
        assert _load_json_arg(str(f)) == long_payload, "传文件路径时该读文件"

    GOLDEN_SESSION = (
        "**会话约定**：新会话开场先调一次 session_start；会话结束前调一次 "
        "thread_close，记下聊到哪、当下状态、有什么没聊完。")
    assert GOLDEN_RETRIEVAL in conv and GOLDEN_SESSION in conv, \
        "改动了三轮真机验证过的措辞——这两段只许在后面加，不许动"

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
        assert paths.get("contract") is None, "有宿主的档不该塞契约副本（那是 generic 档的活）"
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
        # 取一道**会生成字段**的选择题：人称题也是 choice，但它按设计不进字段
        # （2026.08.02 加人称题后这里要挑清楚，否则取到的是个不产草稿的题）
        cq = next(q for q in half_qs if q.kind == "choice" and not q.pronoun_side)
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

    # 9b.【变异靶心：generic 档少一样都不算出货】自建前端没有宿主替他注入，
    #     契约副本就是交付的另一半；漏了的话产出目录看着一切正常（人格文件、
    #     记忆库、配置齐全），少的恰好是他唯一必须照做的那部分
    with tempfile.TemporaryDirectory() as td:
        g = write_bundle(td, p5, client="generic", confirmed=True)
        assert g["persona"].name == "persona.md", \
            "generic 档不能沿用宿主专有文件名（CLAUDE.md/AGENTS.md 对自建前端没有意义）"
        assert g["contract"] and g["contract"].exists(), "generic 档必须随货带注入契约副本"
        body = g["contract"].read_text(encoding="utf-8")
        for key in ("逐字", "每轮", "从磁盘重读", "整块连续", "易变内容"):
            assert key in body, f"契约副本里缺了契约五条之一：{key}"
        #    契约源文件缺失时明确报错，不出一份"看着正常、其实少一半"的货
        with tempfile.TemporaryDirectory() as empty:
            try:
                write_bundle(td, p5, client="generic", confirmed=True, contract_base=empty)
                assert False, "契约源文件缺失时不该静默出货"
            except FileNotFoundError:
                pass

    # 9b2.【靶心：走 CLI 真进程，钉住《快速上手》4b 教的那条命令】
    #      **这条是验收打回补的**：9b 直接调 write_bundle(client="generic")、9c 直接调
    #      ship_note("generic")，两条都绕开了 CLI 的 state 覆盖路径——于是"ship 步
    #      显式传的 --client 被状态里的旧值静默吃掉"这个洞，函数级断言全绿地放过去了。
    #      **断言钉住了函数，没钉住用户真会敲的那条命令。** 所以这里起真进程，
    #      按用户文档的顺序跑完四步，最后一步才给 --client generic。
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        def run(*extra):
            # **cwd 故意设成产出目录、不是 src/**：用户不会站在 src/ 里跑第四步，
            # 而"配置里的 server 路径是不是真能用"这件事，只有在别的工作目录下
            # 才分辨得出来——站在 src/ 里跑，裸的相对路径 `mcp_server.py` 也能
            # 蒙混过关（第一版断言就是这么被蒙过去的）
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        #    第 1 步**不带 --client**（默认 claude-code 落进状态，正是打回单的复现前提）
        qs_json = json.loads(run("--step", "questionnaire", "--json"))
        assert json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8")
                          )["client"] == DEFAULT_CLIENT
        ans = {q["qid"]: ({"keys": sorted(q["options"])[0]} if q["options"]
                          else {"pick": "她说“到家了发一句”，我说好。"})
               for q in qs_json["questions"]}
        run("--step", "answers", "--answers-json", json.dumps(ans, ensure_ascii=False))
        pend = json.loads(run("--step", "confirm", "--list", "--json"))["pending"]
        run("--step", "confirm", "--decisions-json",
            json.dumps({p["key"]: "keep" for p in pend}, ensure_ascii=False))
        #    **先按宿主档出一次货**——这一步是断言构造的关键，不是凑数（二轮验收
        #    打回）：不先出宿主档，产出目录里 CLAUDE.md 本来就不存在，下面那条
        #    `not exists` 就是**恒真**的，看着在守"换档后旧档清掉了"，实际什么都
        #    没守（同 routes 消融那次的自动满足）。先出一次，它才有东西可挡。
        #    检索路线这一步照用户真会敲的顺序走一遍，**故意选云端**：这条路是三条
        #    里唯一会把语料发出本机的，也是唯一需要 key 的，产出物必须扛得住
        run("--step", "route", "--route", "cloud")
        ship_out_r = run("--step", "ship")
        assert (Path(td) / "CLAUDE.md").exists(), "宿主档这一次货本身就没出成，后面的断言无从谈起"
        #    选了云端，启动参数就得真的是云端档——选择点不改产出物的话，"选过一次"
        #    只是走了个过场
        cfg_r = json.loads((Path(td) / "mcp-config.json").read_text(encoding="utf-8"))
        cargs = cfg_r["mcpServers"]["memory"]["args"]
        assert "--embed" in cargs and "cloud" in cargs, f"云端档没写进启动参数：{cargs}"
        #    **但 key 一个字都不许进产出目录**：mcp-config.json 会跟着产出目录走，
        #    用户会随手贴给别人；key 只从环境变量读（变异靶心：把 key 写进 args 必红）
        for f in ("mcp-config.json", "init_state.json"):
            blob = (Path(td) / f).read_text(encoding="utf-8")
            assert "MEMORY_EMBED_API_KEY" not in blob and "sk-" not in blob, \
                f"{f} 里出现了 key 相关内容——凭证只走环境变量"
        assert "语料" in ship_out_r and "服务商" in ship_out_r, \
            "出货时没把云端档的语料去向再说一遍"
        #    第 4 步才显式换档——文档教的就是这条
        out = run("--step", "ship", "--client", "generic")
        assert (Path(td) / "persona.md").exists(), \
            "ship 步显式传的 --client generic 被状态里的旧档吃掉了（用户文档 4b 教的正是这条命令）"
        assert (Path(td) / CONTRACT_DOC).exists(), "走 CLI 出的 generic 档没带契约副本"
        #    换档后旧档那份必须退役：留着它日后不会跟着升层／更正更新，作者拼错
        #    那一份的症状恰好是契约第三条违反后的"确认过的更新悄无声息不生效"
        assert not (Path(td) / "CLAUDE.md").exists(), \
            "换档后旧档的人格文件还躺在产出目录里，会变成永不更新的影子副本"
        assert (Path(td) / "CLAUDE.md.bak").exists(), \
            "旧档该退役成 .bak 留痕，不是无声删掉——写用户磁盘一律保守"
        assert "已退役" in out, "退役了旧档却不吭声，用户不知道目录里那份为什么变了名"
        assert "你前端的责任" in out and "起会话" not in out, \
            "走 CLI 时收尾话术仍是宿主档那套"
        assert "已切换" in out, "换档必须说出来，不许静默改用户的档"
        #    换档要写回状态：下次不带 --client 续跑，仍然是 generic
        assert json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8")
                          )["client"] == "generic", "切换后的档没写回状态"
        #    MCP 配置里的 server 路径必须是能直接用的绝对路径——自建前端作者没有
        #    《快速上手》那条 claude mcp add 绝对路径示范，配置给什么他就用什么
        cfg = json.loads((Path(td) / "mcp-config.json").read_text(encoding="utf-8"))
        server_arg = Path(cfg["mcpServers"]["memory"]["args"][0])
        assert server_arg.is_absolute() and server_arg.exists(), \
            f"mcp-config.json 里的 server 路径不可直接使用：{server_arg}"
        #    **样板说明必须在文件里**（2026.08.01 维护者拍板"不改名、加说明"）：
        #    要防的是"有人改了这个文件、发现不生效"，而那件事必须先打开文件才会
        #    发生——所以说明写在文件里，正好在他需要的那一刻被读到。
        #    三条一起钉：键在、话说到点上、**结构没被这一行搞坏**（它是个真键，
        #    不是注释；把配置本体挤歪了就是拿一句提醒换一个真 bug）。
        assert CONFIG_NOTE_KEY in cfg, \
            "mcp-config.json 里没有那句样板说明——用户打开它改，不会知道改了不生效"
        for must in ("不会被任何客户端自动读取", "改这个文件不生效"):
            assert must in cfg[CONFIG_NOTE_KEY], \
                f"样板说明没说到点上，缺：{must!r}"
        assert set(cfg) == {CONFIG_NOTE_KEY, "mcpServers"} and \
            cfg["mcpServers"]["memory"]["command"] == "python", \
            f"那一行说明把配置结构挤歪了：{sorted(cfg)}"
        #    **说明本身不许把 key 漏出去**：它是随产出目录走的文件，同 key 那条纪律
        assert "sk-" not in cfg[CONFIG_NOTE_KEY]
        #    **同档再出一次货不许自己退自己**：重跑 ship 是常规操作（补了语料、
        #    改了一条就再出一次），而退役发生在写盘之后——判据要是漏了"跟这次
        #    同名就不动"，第二次 ship 会把刚写好的人格文件改成 .bak，产出目录
        #    里一份人格文件都不剩，而命令是成功返回的
        out_again = run("--step", "ship")
        assert (Path(td) / "persona.md").exists(), "同档重跑 ship 把自己刚写的人格文件退役了"
        assert not (Path(td) / "persona.md.bak").exists(), "同档重跑不该产生 .bak"
        assert "已退役" not in out_again, "没换档却报退役"

    # 9b4.【靶心：覆盖自己之前那份人格文件前必须备份，并且说出来】
    #      2026.08.01 维护者在场时实测撞出来的真缺陷：老用户升级只要重跑一次 ship，
    #      人格文件就按新版协议层重放一遍——**手改过的段落不在 state 里，
    #      重放等于删掉它，而且原来一声不吭、连 .bak 都不留**。
    #      分量在于《给AI的引导指南》明写着"人格文件是 TA 的，随时可以自己改"：
    #      **我们鼓励了他改，又在升级时悄悄替他删掉**；而换档退役旧档那条我们
    #      留了 .bak，自己覆盖自己反而不留，前后不一致。
    #      **靶子走 CLI 真进程**——函数级断言钉不住"用户真敲的那条命令会不会提醒他"。
    with tempfile.TemporaryDirectory() as td:
        def ship_run(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs_up = json.loads(ship_run("--step", "questionnaire", "--json"))["questions"]
        ship_run("--step", "answers", "--answers-json", json.dumps(
            {q["qid"]: ({"keys": "".join(sorted(q["options"]))} if q["options"]
                        else {"pick": "说好了就算数。"}) for q in qs_up}, ensure_ascii=False))
        pend_up = json.loads(ship_run("--step", "confirm", "--list", "--json"))["pending"]
        ship_run("--step", "confirm", "--decisions-json",
                 json.dumps({p_["key"]: "keep" for p_ in pend_up}, ensure_ascii=False))
        ship_run("--step", "route", "--route", ROUTE_DEFAULT)
        first = ship_run("--step", "ship", "--client", "claude-code")
        md_path = Path(td) / "CLAUDE.md"
        #      第一次出货：磁盘上本来没有旧文件，**不该报"被改过"**，也不该留 .bak
        assert "已备份成" not in first, "第一次出货就报'原来那份不一样'——那是恒真的噪音"
        assert not md_path.with_name("CLAUDE.md.bak").exists(), "第一次出货不该产生 .bak"
        #      同档原样重跑：内容一致，同样不该报、不该备份
        again = ship_run("--step", "ship", "--client", "claude-code")
        assert "已备份成" not in again and not md_path.with_name("CLAUDE.md.bak").exists(), \
            "内容没变也报覆盖，用户会学会忽略这条提醒"
        #      **用户手改之后再重跑**：必须备份 + 必须说出来
        hand = md_path.read_text(encoding="utf-8") + "\n\n## 我自己加的一节\n她怕黑，睡觉留一盏灯。\n"
        md_path.write_text(hand, encoding="utf-8")
        upgraded = ship_run("--step", "ship", "--client", "claude-code")
        bak_path = md_path.with_name("CLAUDE.md.bak")
        assert bak_path.exists(), \
            "手改过的人格文件被直接覆盖、连备份都没有——升级会静默吃掉用户自己写的内容"
        assert bak_path.read_text(encoding="utf-8") == hand, \
            "备份的不是手改前那份，等于备份了个寂寞"
        assert "她怕黑" not in md_path.read_text(encoding="utf-8"), \
            "这条断言的前提：重放本来就带不动手改内容（带得动就不需要这套机制了）"
        for must in ("已备份成", "手改的段落不会被自动带过来", "贴回"):
            assert must in upgraded, f"覆盖了手改内容却没说清怎么办，缺：{must!r}"

    # 9b3.【靶心：不许动不是我们出的文件】三轮验收打回的洞：退役逻辑原本按
    #      "别的档的文件名"匹配，而 CLAUDE.md 是 Claude Code 的项目约定文件，
    #      自建前端作者目录里躺一份他自己写的太正常了。全程只用 generic、从没换过
    #      档的用户，一跑出货就被我们把项目指令改了名 + 收到一句"换档后旧档不再
    #      更新"的错提示。这里预置一份**不是我们出的** CLAUDE.md，走完整流程，
    #      要求它**逐字原样还在**（只断言 exists 不够——改了内容同样是动了人家的文件）。
    with tempfile.TemporaryDirectory() as td:
        mine = Path(td) / "CLAUDE.md"
        mine_text = "# 我自己项目的 CLAUDE.md\n# 这是我给 Claude Code 写的项目指令，别动它。\n"
        mine.write_text(mine_text, encoding="utf-8")

        def run_g(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, "--client", "generic", *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs2 = json.loads(run_g("--step", "questionnaire", "--json"))
        ans2 = {q["qid"]: ({"keys": sorted(q["options"])[0]} if q["options"]
                           else {"pick": "她说“到家了发一句”，我说好。"})
                for q in qs2["questions"]}
        run_g("--step", "answers", "--answers-json", json.dumps(ans2, ensure_ascii=False))
        pend2 = json.loads(run_g("--step", "confirm", "--list", "--json"))["pending"]
        run_g("--step", "confirm", "--decisions-json",
              json.dumps({p["key"]: "keep" for p in pend2}, ensure_ascii=False))
        run_g("--step", "route", "--route", "zero-dep")
        out2 = run_g("--step", "ship")
        assert mine.exists() and mine.read_text(encoding="utf-8") == mine_text, \
            "把用户自己的 CLAUDE.md 动了——目录里有这个文件名不等于那文件是我们写的"
        assert not (Path(td) / "CLAUDE.md.bak").exists(), \
            "给用户自己的 CLAUDE.md 生了个 .bak，等于把他的项目指令挪走了"
        assert "已退役" not in out2, \
            "从没换过档却报'换档后旧档不再更新'，用户会以为自己哪步选错了"
        assert (Path(td) / "persona.md").exists(), "generic 档自己的货还是得出"

    # 9c.【变异靶心：ship 话术不许串档】generic 档照抄“从产出目录起会话”，等于告诉
    #     自建前端作者“你什么都不用做”——而他恰恰是唯一必须自己动手注入的人
    generic_note, host_note = ship_note("generic"), ship_note("claude-code")
    assert "你前端的责任" in generic_note and "起会话" not in generic_note, \
        "generic 档的收尾提示必须说清注入是作者自己的责任，不能沿用宿主档话术"
    assert "起会话" in host_note and "你前端的责任" not in host_note

    # 9d.【三条路线各自落到启动参数上】（2026.08.02）零依赖档不加任何参数（默认就是
    #     它），本地档加 --embed，云端档再加 --embed-provider cloud。**云端档的
    #     endpoint / 模型 / key 一个都不进这份配置**——它们全走环境变量，因为这个
    #     文件会跟着产出目录被随手分享出去。
    snips = {r: json.loads(mcp_config_snippet("/x/mcp_server.py", "/x/memory",
                                              "/x/threads.jsonl", route=r)
                           )["mcpServers"]["memory"]["args"] for r in RETRIEVAL_ROUTES}
    assert "--embed" not in snips["zero-dep"], "零依赖档不该带 --embed"
    assert snips["local"][-1:] == ["--embed"], f"本地档该只多一个 --embed：{snips['local']}"
    assert snips["cloud"][-3:] == ["--embed", "--embed-provider", "cloud"], \
        f"云端档启动参数不对：{snips['cloud']}"
    for r, a in snips.items():
        assert not any("MEMORY_EMBED" in x or "http" in x for x in a), \
            f"{r} 档把 endpoint/环境变量名写进了 MCP 配置：{a}"

    # 10. 人格不完整时拒绝出货——缺检索约定/最终约定这类必填项不能悄悄放行
    p6 = Persona("partner")
    p6.add_field(Field(id="x", section="user", label="x", value="x", confirmed=True))
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p6, confirmed=True)
            assert False, "不完整的人格该被拦下"
        except ValueError as e:
            assert "开篇缺" in str(e)

    print("selftest ok（47项断言：体检识破空泛 / 只问缺口 / 立场题选项与排序 / "
          "归属句式 / 默认值不预支历史 / 协议层不问用户 / 导出纪律 / 渲染顺序 / "
          "人称锚死一套 / 用户只有一种称呼形态 / 中性档不丢主语 / 人称从语料读出来 / 中性写法不许漏 / 语料侧判定不许猜 / 全文零它 / 称呼不重复拼接 / 关系状态归开篇 / "
          "答案读回不静默丢 / 任务书不泄漏进人格文件 / 长字面量不崩 / 未决草稿不蒸发 / "
          "记忆库落盘带日期 / 覆盖区间进人格文件 / 止血纪律进人格文件 / "
          "pick 题只许挑不许写 / 冷启动出得了货 / "
          "AI 驱动不绕过确认 / 确认关卡 / 续跑 / 完整性 / generic 档随货带契约 / "
          "CLI 真进程走文档教的那条命令 / 换档退役旧档 / 同档重跑不自退 / "
          "不动不是我们出的文件 / 覆盖手改的人格文件前必先备份并说出来 / MCP 配置路径可直接用 / ship 话术不串档 / 检索路线是真选择点 / 云端档落到启动参数 / 凭证不进产出目录）")


def _rebuild(state):
    """状态 → (persona, questions)。每一步都从状态重建，不序列化整个 persona——
    重放（协议层 + 答案 + 已做的确认决策）是幂等的，状态文件里只存用户的输入，
    坏了也看得懂、改得动。"""
    # 人称：语料侧检测结果存在状态里（questionnaire 步算的），用户答的覆盖它。
    # **问卷题目要按"还差哪一侧"来出**，所以先合一次、再据此出题；出完题若用户
    # 已经答了人称，再合第二次，让下面的渲染用上真人称
    detected = state.get("pronouns_detected") or {}
    #    人称题就那两道，直接拿它们解答案即可——不必等 questions_for 出完题，
    #    也就避开了"出题要先知道人称、填模板又要先出题"这个循环
    pronouns = pronouns_from_answers(list(PRONOUN_QUESTIONS.values()),
                                     state.get("answers"), detected)
    persona = Persona("partner")
    fill_protocol_defaults(persona, pronouns)
    qs = questions_for(coverage_report(persona), has_corpus=state.get("has_corpus", False),
                       pronouns=pronouns)
    if state.get("answers"):
        apply_answers(persona, qs, state["answers"], pronouns)
    if state.get("decisions"):
        apply_confirmations(persona, state["decisions"])
    return persona, qs


def _load_json_arg(value):
    """`--answers-json` / `--decisions-json` 的取值：文件路径，或 `-` 表示读 stdin，
    或直接就是一段 JSON 字面量。三种都收——驱动方是 AI 时，它手上是内存里的
    结构化数据，不该被逼着先落一个临时文件。

    **字面量先认形状，路径判断再兜 OSError**（2026.08.02 外部复现时撞到）：原先无条件
    先跑 `Path(value).exists()`，而字面量一旦超过文件名长度上限（255 字节）就抛
    `OSError: [Errno 36] File name too long`，**没有兜底、整条命令崩掉**。11 个字段的
    决定串就已经约 290 字节——也就是说**正常规模的一次确认必崩**，而这条路正是为
    "不开终端的人"补的 AI 驱动接口。自检当时没抓到，是因为 fixture 的字面量都很短：
    **fixture 的规模本身就是一种伪影**（同回归集那边"合成语料规模不足会凭空造出
    假缺陷"，只是这次方向相反——规模不足把真缺陷藏了起来）。
    两道都补：以 `{`／`[` 开头的直接当字面量，不去问文件系统；真去问的时候 OSError
    也落回字面量解析，不再让它冒出来。"""
    if value == "-":
        return json.loads(sys.stdin.read())
    if value.lstrip()[:1] in ("{", "["):
        return json.loads(value)
    try:
        p = Path(value)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        pass          # 太长／非法路径名：它本来就不是路径，按字面量解析
    return json.loads(value)


def _questions_payload(qs, pronouns=None):
    """问卷的机器可读形态。AI 驱动时靠它拿题目，不用去抠给人看的排版文本
    （抠文本＝解析自己的输出，格式一改就断，而且断得无声无息）。"""
    return [{"qid": q.qid, "section": q.section,
             "label": fill_pronouns(q.label, pronouns), "kind": q.kind,
             "text": fill_pronouns(q.text, pronouns), "max_chars": q.max_chars,
             "options": ({k: {"label": fill_pronouns(v[0], pronouns),
                              "directive": fill_pronouns(v[1], pronouns)}
                          for k, v in q.options.items()} if q.options else None),
             "attribution": q.attribution, "optional": q.optional}
            for q in qs]


def _client_of(args):
    """questionnaire 步存进状态的客户端档：命令行没给就落默认值。
    （`--client` 的 argparse 默认值是 None，为的是让 ship 步能分辨
    "用户显式指定"与"用户没提"——见 resolve_client。）"""
    return resolve_client(args.client, None)[0]


def _detect_pronouns_from_corpus(corpus_path):
    """有 --corpus 就先从语料里判一次人称。判不出来返回空 dict，交给问卷去问。

    **必须把用户侧说话人告诉 detect_pronouns**（2026.08.02 验收打回）：不传的话
    它走"分不清谁是谁"那条分支、**永远返回 None**，于是生产路径上这个功能是死的
    ——每个用户都会被问一遍，语料里明明写着答案。而这个信息本来就有：
    `memory_import` 的两个翻译器都把用户侧归一成了 `"user"`（ChatGPT 走
    `author.role`，Claude 那条把 `human` 归一成 `user`，出口一致）。

    **叙事体语料（`speaker=""`，timeline md 这类）判不出来是正常的**，跟"能判却
    没判"不是一回事：那种语料里根本没有说话人标记，退回去问用户就对了。

    **读语料失败不该拦住整个流程**——人称判不出来只是"要多问一句"，
    而导入报错在这一步没有任何用户能理解的上下文，所以这里吞掉异常、当作判不出。"""
    if not corpus_path:
        return {}
    try:
        from memory_import import load_any
        entries = load_any(corpus_path)
    except Exception:
        return {}
    return {k: v for k, v in detect_pronouns(entries, USER_SPEAKERS).items() if v}


def _step_questionnaire(args):
    detected = _detect_pronouns_from_corpus(args.corpus)
    persona = Persona("partner")
    fill_protocol_defaults(persona, detected)
    report = coverage_report(persona)
    if args.json:
        qs = questions_for(report, has_corpus=bool(args.corpus), pronouns=detected)
        save_state(args.out, {"step": "questionnaire", "client": _client_of(args),
                              "has_corpus": bool(args.corpus),
                              "pronouns_detected": detected})
        print(json.dumps({
            "coverage": [{"section": s, "status": st, "note": n} for s, st, n in report],
            "questions": _questions_payload(qs, detected),
            "pronouns_detected": detected,
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
    qs = questions_for(report, has_corpus=bool(args.corpus), pronouns=detected)
    print(f"\n【要问的问题】共 {len(qs)} 题（协议层已由系统填好，不问你）\n")
    print(format_questionnaire(qs, has_corpus=bool(args.corpus), pronouns=detected))
    prompt_path = Path(args.out) / "问卷prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(export_llm_prompt(qs, pronouns=detected), encoding="utf-8")
    save_state(args.out, {"step": "questionnaire", "client": _client_of(args),
                          "has_corpus": bool(args.corpus),
                          "pronouns_detected": detected})
    print(f"\n【下一步】把 {prompt_path} 的内容整段粘给你自己的模型（DeepSeek/ChatGPT 都行），"
          f"让它一题一题问你；答完让它把清单整理好，存成一个文本文件，"
          f"再跑：--step answers --answers <那个文件>")


def _report_unsourced_picks(questions, answers, corpus_dir):
    """查无出处的 pick 答案摆到台面上。**只在给了 --corpus 时能查**——
    语料路径没存进 state（只存了 has_corpus），没给就不查，也不假装查过。"""
    if not corpus_dir:
        return
    bad = unsourced_picks(questions, answers, corpus_dir)
    if not bad:
        return
    print()
    for qid, text in bad:
        print(f"  - {qid}：{text}")
    print(PICK_SOURCE_NOTE)


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
        _report_unsourced_picks(qs, answers, args.corpus)
        print("\n【下一步】--step confirm --list --json 取待确认草稿，"
              "**逐条念给对方听、由对方决定**，再用 --decisions-json 落决定。")
        return
    if not args.answers:
        raise SystemExit("这一步要 --answers <答案清单文件> 或 --answers-json <JSON>")
    text = Path(args.answers).read_text(encoding="utf-8")
    answers, problems = parse_answer_sheet(text, qs)
    print("【答案读回】")
    print(answer_report(qs, answers, problems))
    _report_unsourced_picks(qs, answers, args.corpus)
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
        # 任务书**不混进待确认清单**——它不是给用户确认的内容，是给模型的指令，
        # 单独一个字段给出去（缺陷一的修法：两层分开，不靠确认关卡去分辨）
        _, all_qs = _rebuild(state)
        brief = extraction_brief(all_qs, state.get("answers") or {})
        if args.json:
            print(json.dumps({
                "pending": payload,
                "extraction_brief": brief,
                "brief_note": BRIEF_NOTE if brief else "",
                "next": "**逐条念给对方听，由对方决定**，不要替 TA 一路 keep——"
                        "人格文件的每一条都要 TA 认过（引导指南纪律三）。"
                        "收齐后：--step confirm --decisions-json "
                        "'{\"key\": \"keep\"|\"drop\"|{\"edit\": \"改后的文本\"}}'；"
                        "没表态的条目保持未决，不会被默认留下或删掉。"
                        + ("　另外 extraction_brief 里那几条是**给你的提取任务**，"
                           "不要念给用户确认、更不要写进人格文件。" if brief else ""),
            }, ensure_ascii=False, indent=1))
        else:
            for p in payload:
                print(f"—— {p['kind']}【{p['label']}】（key={p['key']}）\n   {p['value']}")
            if brief:
                print("\n" + BRIEF_NOTE)
                for b in brief:
                    print(f"  - {b}")
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


def _step_route(args, state):
    """检索路线选择点。不给 --route 就只把三条路念出来，不替用户拍板。"""
    if args.route:
        if args.route not in RETRIEVAL_ROUTES:
            raise SystemExit(f"没有这条路线：{args.route}。"
                             f"可选：{' / '.join(RETRIEVAL_ROUTES)}")
        state["route"] = args.route
        state["step"] = "route"
        save_state(args.out, state)
        print(f"【已选】{RETRIEVAL_ROUTES[args.route]['名称']}\n")
        print(route_note(args.route))
        print("\n【下一步】--step ship")
        return
    if args.json:
        print(json.dumps({"preamble": ROUTE_PREAMBLE, "routes": route_options(),
                          "default": ROUTE_DEFAULT,
                          "next": "--step route --route <key>"},
                         ensure_ascii=False, indent=2))
        return
    print(format_routes())


def _step_ship(args, state):
    # 检索路线必须先选过一次（2026.08.02）。**这里刻意是硬闸不是默认值**：
    # 三条路里有一条会把私人记忆发到第三方，而"用户从没被问过"和"用户选了默认"
    # 在产出物上长得一模一样——不拦住的话，这个选择点等于不存在。
    # 拦的是"没问过"，不是"没选云端"：`--route zero-dep` 一秒就过。
    route = args.route or state.get("route")
    if route not in RETRIEVAL_ROUTES:
        raise SystemExit(
            "不出货：检索路线还没选过。先跑 --step route（加 --json 拿机器可读的"
            "三条路线与各自代价），把三条路念给 TA、由 TA 选，再 --step route "
            "--route <key>。三条里有一条会把 TA 的私人记忆发到第三方，"
            "这件事不给静默默认。")
    state["route"] = route
    persona, _ = _rebuild(state)
    entries = None
    if args.import_path:
        from memory_import import load_any
        entries = load_any(args.import_path)
        print(f"【导入】{args.import_path} → {len(entries)} 条")
    client, switched = resolve_client(args.client, state.get("client"))
    if switched:
        print(switched + "\n")
        state["client"] = client
    try:
        # last_shipped_persona：上一次出货**我们自己**写下的人格文件名。退役只认它——
        # 目录里叫 CLAUDE.md 的文件可能是用户自己给 Claude Code 写的项目指令，
        # 不是我们的货，不能碰（三轮验收打回）
        paths = write_bundle(args.out, persona, client=client,
                             corpus_dir=args.corpus, confirmed=True, entries=entries,
                             previous_persona=state.get("last_shipped_persona"),
                             route=route)
    except (PermissionError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"不出货：{e}")
    print("【三件套】")
    print(f"  人格文件：{paths['persona']}")
    print(f"  记忆库：{paths['memory_dir']}"
          + (f"（落盘 {len(paths['corpus_files'])} 个窗口文件）" if paths["corpus_files"] else ""))
    print(f"  MCP 配置：{paths['mcp_config']}")
    if paths.get("contract"):
        print(f"  注入契约：{paths['contract']}")
    for bak in paths.get("retired", []):
        # 换了档就得说清旧档那份去哪了——留在目录里的第二份人格文件不会再更新，
        # 拼错了会变成"确认过的更新不生效"，而那是最难自查的一种坏法
        print(f"  已退役：{bak.with_suffix('').name} → {bak.name}"
              f"（换档后旧档的人格文件不再更新，别再拼它）")
    if paths.get("overwritten_backup"):
        # **沉默地覆盖才是最坏的形态**：老用户升级重跑 ship 是对的做法，但他手改过
        # 的段落不在 state 里、重放不出来。所以这里必须说出来，并指向那份备份。
        bak = paths["overwritten_backup"]
        print(f"\n⚠ 磁盘上原来那份人格文件**跟这次重新生成的不一样**，"
              f"已备份成：{bak}")
        print("   多半是两种情况：你后来手改过它，或者它是旧版本出的。"
              "**手改的段落不会被自动带过来**——问卷答案和确认结果存在 "
              "init_state.json 里、能重放，你自己写的那几行不在里面。")
        print("   对着那份备份看一眼，把你自己加的内容贴回新文件里。"
              "（人格文件本来就是你的，改它是对的——只是升级这一步带不动它。）")
    for s in persona.suggestions():
        print(f"  建议（不阻塞）：{s}")
    # 任务书在这里再给一次：ship 是最后一个出口，第二阶段的活儿从这儿接
    _, all_qs = _rebuild(state)
    brief = extraction_brief(all_qs, state.get("answers") or {})
    if brief:
        print("\n" + BRIEF_NOTE)
        for b in brief:
            print(f"  - {b}")
    print("\n" + route_note(route))
    print("\n" + ship_note(client))
    state["step"] = "shipped"
    state["last_shipped_persona"] = paths["persona"].name
    save_state(args.out, state)


def _cli(args):
    """薄 CLI，五步走：questionnaire → answers → confirm → route → ship。
    每步存状态可续跑；不传 --step 时按状态里的进度提示下一步该跑什么。
    真正的逐题交互留给导出的 prompt（路线 C）——用户拿去自己的模型那边一问一答，
    比在终端里敲长文本舒服得多。"""
    state = load_state(args.out)
    step = args.step
    if not step:
        step = {"": "questionnaire", "questionnaire": "answers",
                "answers": "confirm", "confirm": "route", "route": "ship",
                "shipped": "ship"}[state.get("step", "")]
        print(f"（按进度接着跑：--step {step}）\n")
    if step == "questionnaire":
        _step_questionnaire(args)
    elif step == "answers":
        _step_answers(args, state)
    elif step == "confirm":
        _step_confirm(args, state)
    elif step == "route":
        _step_route(args, state)
    else:
        _step_ship(args, state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="产出目录")
    ap.add_argument("--corpus", help="已有语料目录（可选）")
    # 默认值是 None 而不是 "claude-code"，为的是 ship 步能分辨"用户显式指定了档"
    # 与"用户没提"——两者在出货时的行为必须不同（见 resolve_client）
    ap.add_argument("--client", default=None, choices=sorted(CLIENT_FILENAMES),
                    help=f"客户端档（默认 {DEFAULT_CLIENT}）。任何一步都可以给；"
                         f"ship 步显式给的以它为准，会覆盖前面存下的档")
    ap.add_argument("--step", choices=["questionnaire", "answers", "confirm",
                                       "route", "ship"],
                    help="不传则按 init_state.json 里的进度接着跑")
    ap.add_argument("--route", choices=sorted(RETRIEVAL_ROUTES),
                    help="route 步：TA 选的检索路线（zero-dep 默认 / local 本地模型 / "
                         "cloud 云端——**云端会把语料发到第三方**）。不给就只把三条"
                         "路线和各自代价打出来，不替 TA 拍板")
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
