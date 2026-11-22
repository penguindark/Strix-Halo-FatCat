#!/usr/bin/env python3
"""Generate golden dequantization test data using gguf-py (the reference).

Usage: .venv/bin/python tools/py/gen_dequant_golden.py <model.gguf> [...] <out.bin>

For each covered tensor type (Q8_0, Q6_K, IQ2_XXS, IQ1_S) takes the first N
blocks of a representative tensor and writes:
  per record: u32 ggml_type, u32 n_blocks, raw block bytes, f32 dequant values
The C test (tests/test_dequant.c) replays these through our kernels.
Multiple input files may be given (e.g. split-GGUF parts); tensors are looked
up across all of them.
"""
import sys
import struct
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

# (ggml_type_id, tensor name, n_blocks)
SAMPLES = [
    (8,  "blk.0.ffn_gate.weight",   8),   # Q8_0
    (14, "blk.0.attn_q.weight",     8),   # Q6_K
    (16, "blk.1.ffn_gate_exps.weight", 8), # IQ2_XXS
    (19, "blk.10.ffn_gate_exps.weight", 8), # IQ1_S
]

def main():
    in_paths, out_path = sys.argv[1:-1], sys.argv[-1]
    tensors = {}
    for p in in_paths:
        r = GGUFReader(p, 'r')
        for t in r.tensors:
            tensors.setdefault(t.name, t)
    out = open(out_path, "wb")
    total = 0
    for type_id, name, nblocks in SAMPLES:
        t = tensors[name]
        raw = np.ascontiguousarray(
            np.concatenate(t.data) if isinstance(t.data, list) else t.data
        ).astype(np.uint8).reshape(-1)
        # block sizes: Q8_0 32 elems/34 B, Q6_K 256/210, IQ2_XXS 256/66
        elems_per_block = {8: 32, 14: 256, 16: 256, 19: 256}[type_id]
        bytes_per_block = {8: 34, 14: 210, 16: 66, 19: 50}[type_id]
        nbytes = nblocks * bytes_per_block
        block = raw[:nbytes].reshape(nblocks, bytes_per_block)
        dq = dequantize(block, type_id).astype(np.float32)
        assert dq.size == nblocks * elems_per_block, (dq.size, nblocks)
        out.write(struct.pack("<II", type_id, nblocks))
        out.write(block.tobytes())
        out.write(dq.tobytes())
        total += 1
        print(f"{name}: type={type_id} blocks={nblocks} "
              f"raw={nbytes}B dq[:4]={dq[:4]}")
    out.close()
    print(f"wrote {total} records to {out_path}")

if __name__ == "__main__":
    main()
