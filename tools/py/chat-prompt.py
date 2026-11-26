#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""chat-prompt.py — render a multi-turn conversation into the Mistral-Large-4
chat token stream (same template tools/py/chat.py drives), for offline A/B
runs via `fatcat-run --tokens`.

Usage:
  .venv/bin/python tools/py/chat-prompt.py CONV.json > CONV.tokens

CONV.json:
  {"system": str, "reasoning": "off|low|medium|high",
   "turns": [{"role": "user"|"assistant", "text": str}, ...]}

Template (matches chat.py): user turns are wrapped in [INST]..[/INST];
assistant turns are raw text followed by EOS. The LAST turn must be a user
turn, so `fatcat-run --tokens ... -n N` generates that assistant reply.

Special-token ids match chat.py: BOS=1 EOS=2 INST=3 INST_END=4 SYS 17/18
MODEL_SETTINGS 36/37.
"""
import json
import sys

from tokenizers import Tokenizer

BOS, EOS, INST, INST_END = 1, 2, 3, 4
SYS, SYS_END = 17, 18
MODEL_SETTINGS, MODEL_SETTINGS_END = 36, 37


def main():
    conv = json.load(open(sys.argv[1]))
    tk = Tokenizer.from_file("Manuals/model/tokenizer.json")

    def enc(t):
        return tk.encode(t, add_special_tokens=False).ids

    ids = [BOS]
    if conv.get("system"):
        ids += [SYS] + enc(conv["system"]) + [SYS_END]
    r = conv.get("reasoning", "low")
    if r != "off":
        ids += [MODEL_SETTINGS] + enc('{"reasoning_effort": "' + r + '"}') \
               + [MODEL_SETTINGS_END]
    turns = conv["turns"]
    assert turns and turns[-1]["role"] == "user", \
        "last turn must be 'user' (the reply is what gets generated)"
    for t in turns:
        if t["role"] == "user":
            ids += [INST] + enc(t["text"]) + [INST_END]
        elif t["role"] == "assistant":
            ids += enc(t["text"]) + [EOS]
        else:
            raise ValueError(f"bad role {t['role']!r}")
    print(" ".join(map(str, ids)))


if __name__ == "__main__":
    main()
