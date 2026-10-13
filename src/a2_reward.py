# -*- coding: utf-8 -*-
"""a2_reward.py — exact-match rewards for Packet A2 tasks, shared by the
Run 2b data filter and the GRPO trainer. Same parsers as the gates: the
first non-empty line is the answer; roots compare by root identity
(hamza carriers folded, ى -> ي), verbs by bake-off letters."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
import bakeoff_arabic as B  # noqa: E402
from build_packet_b import root_key  # noqa: E402

MARK = "الجواب:"


def gold_answer(item):
    """The string a completion is checked against."""
    c = item["completion"]
    if item["task"] == "reason":
        return B.norm_ar(c.split(MARK, 1)[1].strip().splitlines()[0])
    return c.strip()


def first_line(text):
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def correct(task, completion, answer):
    if task == "root":
        return root_key(B.parse_root_output(completion)) == root_key(answer)
    if task in ("morph", "conj"):
        return B.parse_verb_output(completion) == B.ar_letters(answer)
    if task == "derive":       # several masdars may be valid; list must be a subset
        gold = {B.ar_letters(x) for x in answer.split("،") if x.strip()}
        got = {B.ar_letters(x) for x in first_line(completion).split("،")
               if x.strip()}
        return bool(got) and got <= gold
    if task == "form":
        return B.norm_ar(first_line(completion)) == B.norm_ar(answer)
    if task == "reason":
        return B.parse_answer_output(completion) == answer
    raise ValueError(task)


def bucket(item):
    t = item["task"]
    if t in ("morph", "form"):
        return f"{t}/{item.get('form')}"
    if t == "root":
        return f"root/{item.get('cls')}"
    return t
