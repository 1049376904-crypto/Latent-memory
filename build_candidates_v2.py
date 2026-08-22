#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选生成器 v2：填 source_anchor 挑锚号，不再手算 source_span。

新版 memory_init 会在输入清单里给出 anchors 锚表，模型填 anchor 号即可。
但本仓语料是 Claude 官方导出的单行 JSON（没有空行），分段算法退化成
「全文一条锚 A1」，所以 anchor 恒为 A1。

evidence 仍需是文件原始字符（中文为 \\uXXXX 转义形态），因此这里仍用
json.dumps 编码后 find 定位、原样切片，只是不再把偏移写进 source_span。
"""
import json
from pathlib import Path

CORPUS = "/root/Haven-Ombre/conversations.json"
OUT = "/root/latent-v2/candidates.json"
ANCHOR = "A1"

raw = Path(CORPUS).read_text(encoding="utf-8")


def encode(text):
    """把人可读的原样文本转成文件里的形态（中文转义、引号加反斜杠）。"""
    return json.dumps(text, ensure_ascii=True)[1:-1]


def slice_evidence(anchor, anchor_end=None):
    """在原文里切出 evidence（原始字符形态）；找不到返回 None。"""
    a = encode(anchor)
    i = raw.find(a)
    if i < 0:
        return None
    if anchor_end is None:
        return raw[i:i + len(a)]
    b = encode(anchor_end)
    k = raw.find(b, i)
    if k < 0:
        return None
    return raw[i:k + len(b)]


# anchor / anchor_end 是语料里真实存在的连续文本，用来定位 evidence。
# 要加候选就往这个列表里追加。
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
        "current_state": "已接通，latent_search 那一组工具都能用",
    },
]

items, missing = [], []
for cand in CANDIDATES:
    cand = dict(cand)
    evidence = slice_evidence(cand.pop("anchor"), cand.pop("anchor_end", None))
    if evidence is None:
        missing.append(cand["item_id"])
        continue
    cand["source_ref"] = CORPUS
    cand["source_anchor"] = ANCHOR
    cand["evidence"] = evidence
    items.append(cand)

# turns 的逐字自检：不落在 evidence 里的当场报出来，别留到闸门上才发现
for cand in items:
    for turn in cand.get("turns", []):
        if encode(turn["text"]) not in cand["evidence"]:
            print("WARN %s 的 turn 不在 evidence 内: %s" % (cand["item_id"], turn["text"][:30]))

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
    print("锚点文本没找到，已跳过:", missing)
