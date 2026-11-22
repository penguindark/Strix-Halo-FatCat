#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""Compare two logit-dump .bin files row-wise: cosine sim + top-1 agreement.

Usage: cmp_logits.py A.bin B.bin [rows_to_show] [vocab]
logit-dump writes n_pos blocks of n_vocab f32 (no header).
"""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
n_vocab = int(sys.argv[4]) if len(sys.argv) > 4 else 204800   # Mistral-Large-4
rows = min(len(a), len(b)) // n_vocab
a = a[:rows * n_vocab].reshape(rows, n_vocab)
b = b[:rows * n_vocab].reshape(rows, n_vocab)
print(f"rows={rows}")
print(f"A: finite={np.isfinite(a).all()} min={a.min():.2f} max={a.max():.2f}")
print(f"B: finite={np.isfinite(b).all()} min={b.min():.2f} max={b.max():.2f}")
ca = a / np.linalg.norm(a, axis=1, keepdims=True)
cb = b / np.linalg.norm(b, axis=1, keepdims=True)
cs = (ca * cb).sum(axis=1)
ta = a.argmax(axis=1)
tb = b.argmax(axis=1)
top1 = int((ta == tb).sum())
show = int(sys.argv[3]) if len(sys.argv) > 3 else 8
for i in range(rows - min(show, rows), rows):
    m = "OK" if ta[i] == tb[i] else "MISMATCH"
    print(f"pos {i}: cos={cs[i]:.6f} top1 {ta[i]} vs {tb[i]} {m} "
          f"maxdiff={np.abs(a[i] - b[i]).max():.4f}")
print(f"SUMMARY: mean cos={cs.mean():.6f} min cos={cs.min():.6f} top1 {top1}/{rows}")
