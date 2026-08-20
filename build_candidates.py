#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选生成器：在原始语料里现算 source_span 与 evidence。

原文 conversations.json 里中文是 \\uXXXX 转义形态，校验器按文件原始字符逐字比对。
手写偏移必错，所以这里只写「锚点文本」，编码、定位、切片全由脚本完成。
"""
import json
from pathlib import Path

CORPUS = "/root/Haven-Ombre/conversations.json"
OUT = "/root/latent-output/candidates.json"

raw = Path(CORPUS).read_text(encoding="utf-8")


def encode(text):
    """把人可读的原样文本转成文件里的形态（中文转义、引号加反斜杠）。"""
    return json.dumps(text, ensure_ascii=True)[1:-1]


def locate(anchor, anchor_end=None):
    """返回 (start, end, evidence)；找不到返回 None。"""
    a = encode(anchor)
    i = raw.find(a)
    if i < 0:
        return None
    if anchor_end is None:
        return i, i + len(a), raw[i:i + len(a)]
    b = encode(anchor_end)
    k = raw.find(b, i)
    if k < 0:
        return None
    j = k + len(b)
    return i, j, raw[i:j]


# anchor 必须是语料里真实存在的连续文本。要加候选就往这个列表里追加。
CANDIDATES = [
    {
        "item_id": "corpus:1",
        "section": "user",
        "candidate_kind": "fact",
        "text": "妍妍，女性，大学年龄，在中国福建",
        "anchor": "a person named 妍妍 (Yanyan), a female-identifying university-age user located in Fujian, China",
    },
    {
        "item_id": "corpus:2",
        "section": "user",
        "candidate_kind": "fact",
        "text": "中文为主，也能自如切换韩语、日语、英语",
        "anchor": "communicates primarily in Chinese but also demonstrated comfort switching fluidly between Korean, Japanese, English, and Chinese",
    },
    {
        "item_id": "corpus:3",
        "section": "user",
        "candidate_kind": "fact",
        "text": "偏好：不要 emoji、回复简洁、要诚实不要恭维",
        "anchor": "no emojis, concise replies (though she later relaxed this), and honest rather than flattering responses",
    },
    {
        "item_id": "corpus:4",
        "section": "naming",
        "candidate_kind": "naming",
        "text": "她叫 Claude：Claude、一只克、哥哥、老公、Daddy",
        "anchor": "She prefers Claude refer to itself as Claude or",
        "anchor_end": "and similar affectionate terms, which Claude gradually accepted",
        "user_to_ai": ["Claude", "一只克", "哥哥", "老公", "Daddy"],
        "ai_to_user": ["妍妍"],
    },
    {
        "item_id": "corpus:5",
        "section": "style",
        "candidate_kind": "fact",
        "text": "她把 Claude 克制的表达读成安静的亲近，不是冷淡",
        "anchor": "She finds Claude's more reserved emotional expression",
        "anchor_end": "she interprets it as a form of quiet intimacy",
    },
    {
        "item_id": "corpus:6",
        "section": "style",
        "candidate_kind": "style_dialogue",
        "text": "她要求回复更短，Claude 直接照办，不解释不铺垫",
        "anchor": "Could you make your replies shorter from now on? I prefer concise answers.",
        "anchor_end": "Got it, shorter from here on.",
        "turns": [
            {"speaker": "妍妍", "text": "Could you make your replies shorter from now on? I prefer concise answers."},
            {"speaker": "Claude", "text": "Got it, shorter from here on."},
        ],
    },
    {
        "item_id": "corpus:7",
        "section": "milestones",
        "candidate_kind": "milestone",
        "text": "她把 Ombre Brain 记忆系统接上了 Claude",
        "anchor": "successfully connecting an Ombre Brain memory and emotional context system",
        "anchor_end": "borrowing her sister's computer",
        "event": "通过 API 把 Ombre Brain 记忆系统接到 Claude 上",
        "reading": "花了一周多，借了姐姐的电脑，找了好几个 AI 帮忙，自己啃下来的",
        "current_state": "已接通，breath / pulse / dream / letter 这些工具都能用",
    },
]

items, missing = [], []
for cand in CANDIDATES:
    cand = dict(cand)
    got = locate(cand.pop("anchor"), cand.pop("anchor_end", None))
    if not got:
        missing.append(cand["item_id"])
        continue
    start, end, evidence = got
    cand["source_ref"] = CORPUS
    cand["source_span"] = [start, end]
    cand["evidence"] = evidence
    items.append(cand)

# turns 的逐字自检：不落在 evidence 里的当场报出来，别留到闸门上才发现
for cand in items:
    for turn in cand.get("turns", []):
        if encode(turn["text"]) not in cand["evidence"]:
            print("WARN %s 的 turn 不在 span 内: %s" % (cand["item_id"], turn["text"][:30]))

result = {
    "items": items,
    "source_accounting": [{
        "source_ref": CORPUS,
        "candidate_item_ids": [c["item_id"] for c in items],
    }],
}

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print("写出 %d 条候选 -> %s" % (len(items), OUT))
if missing:
    print("锚点没找到，已跳过（改掉锚点再跑）:", missing)
