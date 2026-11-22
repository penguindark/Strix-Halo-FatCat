#!/usr/bin/env python3
"""chat.py — interactive chat REPL for the FatCat engine.

Spawns `fatcat-run MODEL.gguf --serve`, speaks its line protocol (see
tools/fatcat-run.c header), renders the Mistral-Large-4 chat format
(<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT][INST]..[/INST]reply</s>..) with the
HF tokenizer, and streams the reply token by token.

Engine env (FATCAT_GPU, FATCAT_WARM_MB, FATCAT_MEM_LIMIT_MB, FATCAT_KV_F16,
...) is inherited from your shell. Standalone GPU recipe (stop the assistant
LLM first — this box has 128 GB shared with it):

  FATCAT_GPU=1 FATCAT_WARM_MB=65536 FATCAT_MEM_LIMIT_MB=110000 \
  FATCAT_KV_F16=1 \
  .venv/bin/python tools/py/chat.py $M --ctx 4096

Context length is fixed at engine open: --ctx N sets FATCAT_KV_TARGET=N
(the engine may clamp it down to fit the memory budget; /info shows the
real capacity). The first turn is slow until the warm expert tier fills.

REPL:
  <text>          one user turn per line (end a line with a backslash for more)
  /reset          start a new conversation (KV rewind)
  /info           KV usage: seq / capacity
  /max N          max tokens per reply (default 512; thinking counts)
  /sys TEXT       system prompt (applies after /reset)
  /quit           exit
"""
import argparse
import os
import subprocess
import sys
import time

from tokenizers import Tokenizer

BOS, EOS, INST, INST_END = 1, 2, 3, 4
MODEL_SETTINGS, MODEL_SETTINGS_END = 36, 37


class Engine:
    """Line-protocol driver for `fatcat-run --serve`."""

    def __init__(self, cmd):
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                sys.exit("engine died before READY")
            if line.strip() == "READY":
                break
            # engine load chatter on stdout: echo it dimly
            print(f"  [engine] {line.rstrip()}", file=sys.stderr)

    def _send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def alive(self):
        return self.p.poll() is None

    def info(self):
        self._send("I")
        parts = self.p.stdout.readline().split()
        return (int(parts[1]), int(parts[2])) if len(parts) == 3 else (0, 0)

    def reset(self):
        self._send("R")
        self.p.stdout.readline()

    def prefill(self, ids):
        self._send("P " + " ".join(map(str, ids)))
        return self.p.stdout.readline().strip() == "OK"

    def generate(self, max_new, on_token):
        """Yield nothing; call on_token(id) per token. Returns (n, eos, loop)."""
        self._send(f"G {max_new} {EOS}")
        n, ended_eos, looped = 0, False, False
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died during generation")
            if line.startswith("T "):
                tid = int(line[2:])
                n += 1
                ended_eos = tid == EOS
                on_token(tid)
            elif line.startswith("GEN"):
                if "ERR" in line:
                    raise RuntimeError("engine generation error (KV full?)")
                return n, ended_eos, "LOOP" in line


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="path to the Mistral-Large-4 GGUF (first shard)")
    ap.add_argument("--engine", default="build/fatcat-run")
    ap.add_argument("--tokenizer", default="Manuals/model/tokenizer.json")
    ap.add_argument("--ctx", type=int, default=0,
                    help="context length in tokens (sets FATCAT_KV_TARGET; "
                         "0 = engine default)")
    ap.add_argument("--max", type=int, default=1024,
                    help="max tokens per reply (thinking tokens count; "
                         "medium/high reasoning can think for 500+)")
    ap.add_argument("--sys", default="", help="system prompt")
    ap.add_argument("--reasoning", default="low",
                    choices=["off", "low", "medium", "high"],
                    help="MODEL_SETTINGS reasoning_effort header; the model "
                         "thinks unless told otherwise (official default: "
                         "high). Model card: low = quick chat replies, "
                         "medium = day-to-day, high = complex")
    args = ap.parse_args()

    if args.ctx:
        os.environ["FATCAT_KV_TARGET"] = str(args.ctx)
    tk = Tokenizer.from_file(args.tokenizer)
    e = Engine([args.engine, args.model, "--serve"])

    seq, cap = e.info()
    note = "" if not args.ctx or not cap or cap >= args.ctx else \
        " (clamped by memory budget)"
    print(f"FatCat chat — reasoning={args.reasoning}, "
          f"KV capacity {cap or '?'} tokens{note}. "
          f"First turn is slow until the warm tier fills. /quit to exit.")

    sys_prompt = args.sys
    first_turn = True
    carry = []          # last generated token, not yet in engine KV
    max_new = args.max

    def encode(text):
        return tk.encode(text, add_special_tokens=False).ids

    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        while text.endswith("\\"):
            text = text[:-1] + "\n" + input("  ... ").strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            e.reset()
            first_turn, carry = True, []
            print("(conversation reset)")
            continue
        if text == "/info":
            seq, cap = e.info()
            print(f"KV: {seq}/{cap} tokens")
            continue
        if text.startswith("/max"):
            try:
                max_new = int(text.split()[1])
                print(f"max reply = {max_new} tokens")
            except (IndexError, ValueError):
                print(f"max reply = {max_new} tokens")
            continue
        if text.startswith("/sys"):
            sys_prompt = text[4:].strip()
            print(f"system prompt set ({len(sys_prompt)} chars) — "
                  "applies after /reset")
            continue

        # render one user turn in Mistral-Large-4 chat format
        turn = encode(text)
        if first_turn:
            head = [BOS]
            if sys_prompt:
                head += ([17] + encode(sys_prompt) + [18])
            if args.reasoning != "off":
                head += ([MODEL_SETTINGS]
                         + encode('{"reasoning_effort": "' + args.reasoning
                                  + '"}')
                         + [MODEL_SETTINGS_END])
            head += [INST]
            pre = carry + head + turn + [INST_END]
            first_turn = False
        else:
            pre = carry + [INST] + turn + [INST_END]

        t0 = time.time()
        if not e.prefill(pre):
            print("!! prefill failed — context full? try /reset",
                  file=sys.stderr)
            continue
        prefill_s = time.time() - t0
        if not cap:                      # KV allocates lazily at first prefill
            seq, cap = e.info()
            print(f"(KV capacity: {cap} tokens)")

        THINK, THINK_END = 34, 35
        state = {"mode": "answer", "ids": [], "printed": 0,
                 "n_think": 0, "n_answer": 0, "last": None, "prompted": False}

        def seg_flush():
            state["ids"], state["printed"] = [], 0

        def on_token(tid):
            state["last"] = tid
            if tid in (THINK, THINK_END):
                if (tid == THINK_END and state["mode"] == "answer"
                        and state["n_answer"]):
                    # reasoning=low quirk: some replies emit a bare
                    # [/THINK] with no opener - the text we just showed
                    # as the answer was actually the model's thinking
                    state["n_think"] += state["n_answer"]
                    state["n_answer"] = 0
                    print("\n\033[2m  (the text above was unmarked "
                          "thinking)\033[0m", flush=True)
                seg_flush()
                if tid == THINK:
                    state["mode"] = "think"
                    print("\033[2mthinking… \033[0m", end="", flush=True)
                else:
                    state["mode"] = "answer"
                    print("\n\nfat> ", end="", flush=True)
                    state["prompted"] = True
                return
            if state["mode"] == "answer":
                if not state["prompted"]:
                    print("fat> ", end="", flush=True)
                    state["prompted"] = True
                state["n_answer"] += 1
            else:
                state["n_think"] += 1
            state["ids"].append(tid)
            text = tk.decode(state["ids"], skip_special_tokens=True)
            delta = text[state["printed"]:]
            if delta:
                if state["mode"] == "think":
                    delta = "\033[2m" + delta + "\033[0m"
                sys.stdout.write(delta)
                sys.stdout.flush()
            state["printed"] = len(text)

        t1 = time.time()
        try:
            n, ended_eos, looped = e.generate(max_new, on_token)
        except RuntimeError as err:
            print(f"\n!! {err} — try /reset", file=sys.stderr)
            continue
        gen_s = time.time() - t1
        if looped:
            print("\n\033[33m …repetition loop detected, generation "
                  "stopped\033[0m", end="")
        elif not ended_eos:
            print(" …", end="")   # hit the /max cutoff
        if not state["prompted"] and not state["n_think"]:
            print("(no reply)")
        print()
        if n:
            carry = [state["last"]]   # last generated token is not in engine KV
            if not ended_eos and state["mode"] == "think":
                # cutoff landed INSIDE an open [THINK] block: leave the KV
                # mid-thought and the model keeps "thinking" as plain text
                # on every later turn. Close the block explicitly.
                carry.append(THINK_END)
                print("  !! reply cut mid-thinking: block closed, raise "
                      "/max or use --reasoning low")
            elif not ended_eos and state["n_answer"] == 0:
                print("  !! thinking used the whole budget, no answer "
                      "produced — raise /max or use --reasoning low")
            think_s = (f"{state['n_think']} thinking + "
                       if state["n_think"] else "")
            print(f"  [{n} tok ({think_s}{state['n_answer']} answer): "
                  f"prefill {len(pre)} tok in {prefill_s:.1f}s "
                  f"({len(pre)/max(prefill_s,1e-9):.1f} tok/s), "
                  f"decode {n} tok in {gen_s:.1f}s "
                  f"({n/max(gen_s,1e-9):.2f} tok/s)]")
        if not e.alive():
            sys.exit("engine exited")

    e._send("Q")
    e.p.wait(timeout=10)


if __name__ == "__main__":
    main()
