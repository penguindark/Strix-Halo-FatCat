#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""Inject the real Mistral-Large-4 tokenizer (from HF tokenizer.json) into a
GGUF whose tokenizer was stripped to a stub by the upstream quant repo.

Usage:
  .venv/bin/python tools/py/inject_tokenizer.py IN.gguf OUT.gguf \
      --hf-tokenizer tokenizer.json [--chat-template chat_template.jinja]

Rewrites the KV section; the tensor info + data sections are copied verbatim
(tensor offsets are relative to the data start, which may shift).
"""
import argparse
import struct
import json

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
# llama token types
TT_NORMAL, TT_CONTROL, TT_USER = 1, 3, 4


def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def rd_val(f, t):
    """Returns (type, value) with value in python; arrays as (elem_type, [vals])."""
    if t in (U8, I8, U16, I16, U32, I32, U64, I64):
        fmt = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
               U64: "<Q", I64: "<q"}[t]
        return t, struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == F32:
        return t, struct.unpack("<f", f.read(4))[0]
    if t == F64:
        return t, struct.unpack("<d", f.read(8))[0]
    if t == BOOL:
        return t, f.read(1)[0] != 0
    if t == STR:
        return t, rd_str(f)
    if t == ARR:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return t, (et, [rd_val(f, et)[1] for _ in range(n)])
    raise ValueError(f"type {t}")


def enc_val(t, v):
    if t in (U8, I8, U16, I16, U32, I32, U64, I64):
        fmt = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
               U64: "<Q", I64: "<q"}[t]
        return struct.pack(fmt, v)
    if t == F32:
        return struct.pack("<f", v)
    if t == F64:
        return struct.pack("<d", v)
    if t == BOOL:
        return struct.pack("<B", 1 if v else 0)
    if t == STR:
        b = v if isinstance(v, bytes) else v.encode("utf-8", "surrogateescape")
        return struct.pack("<Q", len(b)) + b
    if t == ARR:
        et, vals = v
        out = struct.pack("<I", et) + struct.pack("<Q", len(vals))
        return out + b"".join(enc_val(et, x) for x in vals)
    raise ValueError(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_gguf")
    ap.add_argument("out_gguf")
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--chat-template")
    args = ap.parse_args()

    # ---- load HF tokenizer vocab ----
    hf = json.load(open(args.hf_tokenizer))
    vocab = hf["model"]["vocab"]  # {token_str: id}
    n = max(vocab.values()) + 1
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = s
    assert all(t is not None for t in toks), "vocab has holes"
    specials = {a["content"] for a in hf.get("added_tokens", []) if a.get("special")}
    print(f"HF vocab: {n} tokens, {len(specials)} specials")

    # ---- read input GGUF header ----
    f = open(args.in_gguf, "rb")
    assert f.read(4) == b"GGUF"
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    kvs = []
    for _ in range(n_kv):
        k = rd_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        _, v = rd_val(f, t)
        kvs.append((k, t, v))
    tensor_infos = f.tell()  # tensor info section starts here
    # walk tensor infos to find where old data begins
    for _ in range(n_tensors):
        rd_str(f)                                   # name
        nd = struct.unpack("<I", f.read(4))[0]     # n_dims
        f.read(8 * nd)                             # dims
        f.read(4)                                  # type
        f.read(8)                                  # offset
    infos_end = f.tell()
    align = 32
    for k, t, v in kvs:
        if k == b"general.alignment":
            align = v
    old_data_start = (-(-infos_end // align)) * align
    f.seek(tensor_infos)
    infos_blob = f.read(infos_end - tensor_infos)  # tensor infos only (no padding)

    # ---- rebuild KV list ----
    drop = {b"tokenizer.ggml.tokens", b"tokenizer.ggml.scores", b"tokenizer.ggml.token_type",
            b"tokenizer.ggml.added_tokens", b"tokenizer.ggml.byte_fallback",
            b"tokenizer.ggml.add_bos_token", b"tokenizer.ggml.utf8_flush",
            b"tokenizer.chat_template"}
    out_kv = [(k, t, v) for (k, t, v) in kvs if k not in drop]

    def token_bytes(s):
        return s.encode("utf-8", "surrogateescape")

    tok_types = [TT_CONTROL if t in specials else TT_NORMAL for t in toks]
    new_kv = [
        (b"tokenizer.ggml.tokens", ARR, (STR, [token_bytes(t) for t in toks])),
        (b"tokenizer.ggml.scores", ARR, (F32, [0.0] * n)),
        (b"tokenizer.ggml.token_type", ARR, (I32, tok_types)),
        (b"tokenizer.ggml.added_tokens", ARR, (STR, [s.encode() for s in sorted(specials)])),
        (b"tokenizer.ggml.byte_fallback", BOOL, True),
        (b"tokenizer.ggml.add_bos_token", BOOL, False),
        (b"tokenizer.ggml.utf8_flush", BOOL, True),
    ]
    if args.chat_template:
        new_kv.append((b"tokenizer.chat_template", STR,
                       open(args.chat_template, "rb").read()))
    out_kv += new_kv

    # ---- write ----
    out = open(args.out_gguf, "wb")
    out.write(b"GGUF")
    out.write(struct.pack("<I", version))
    out.write(struct.pack("<Q", n_tensors))
    out.write(struct.pack("<Q", len(out_kv)))
    for k, t, v in out_kv:
        out.write(struct.pack("<Q", len(k)) + k)
        out.write(struct.pack("<I", t))
        out.write(enc_val(t, v))

    # tensor infos verbatim, then pad to alignment, then data verbatim
    out.write(infos_blob)
    pos = out.tell()
    pad = (-pos) % align
    out.write(b"\x00" * pad)

    f.seek(old_data_start)
    while True:
        chunk = f.read(1 << 22)
        if not chunk:
            break
        out.write(chunk)
    out.close()
    f.close()
    print(f"wrote {args.out_gguf}: {len(out_kv)} KVs, pad={pad}")


if __name__ == "__main__":
    main()
