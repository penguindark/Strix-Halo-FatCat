#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""Extract quantization codebook tables from ggml-common.h into a C source.

Usage: .venv/bin/python tools/py/gen_quant_tables.py ggml-common.h > src/model/quant_tables.c

Tables needed by FatCat's dequant kernels (subset of ggml-common.h):
  iq2xxs_grid (uint64[256]), ksigns_iq2xs (uint8[128]), kmask_iq2xs (uint8[8])
"""
import re
import sys

WANTED = {
    "iq2xxs_grid": "uint64_t",
    "iq1s_grid": "uint64_t",
    "ksigns_iq2xs": "uint8_t",
    "kmask_iq2xs": "uint8_t",
}

def main():
    src = open(sys.argv[1]).read()
    macros = dict(re.findall(r"#define\s+(\w+)\s+(\d+)", src))
    print("/* Auto-generated from ggml-common.h (llama.cpp) by tools/py/gen_quant_tables.py.")
    print(" * GGML is MIT licensed. Do not edit by hand. */")
    print("#include <stdint.h>\n")
    for name, ctype in WANTED.items():
        m = re.search(
            r"GGML_TABLE_BEGIN\(" + ctype + r",\s*" + name + r",\s*(\w+)\)(.*?)GGML_TABLE_END",
            src, re.S)
        if not m:
            sys.exit(f"table {name} not found")
        count = int(m.group(1)) if m.group(1).isdigit() else int(macros[m.group(1)])
        body = m.group(2)
        vals = re.findall(r"\b0[xX][0-9a-fA-F]+\b|\b\d+\b", body)
        assert len(vals) == count, (name, len(vals), count)
        print(f"const {ctype} fatcat_{name}[{count}] = {{")
        for i in range(0, len(vals), 8):
            print("    " + ", ".join(vals[i:i+8]) + ",")
        print("};\n")

if __name__ == "__main__":
    main()
