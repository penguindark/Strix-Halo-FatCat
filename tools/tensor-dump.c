/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* tensor-dump — dequantize one tensor to raw little-endian f32 (row-major,
 * dims[0] fastest) using the bit-exact fatcat dequantizers.
 *
 * Usage: tensor-dump MODEL.gguf TENSOR.NAME OUT.f32
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/gguf.h"
#include "fatcat/dequant.h"

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tensor out.f32\n", argv[0]); return 2; }
    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(argv[1], err, sizeof err);
    if (!g) { fprintf(stderr, "gguf: %s\n", err); return 1; }
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, argv[2]);
    if (!t) { fprintf(stderr, "tensor %s not found\n", argv[2]); return 1; }
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "oom (%llu elems)\n", (unsigned long long)n); return 1; }
    if (!fatcat_dequant(t->type, fatcat_gguf_tensor_ptr(g, t), buf, n)) {
        fprintf(stderr, "dequant failed (type %u)\n", t->type); return 1;
    }
    FILE *f = fopen(argv[3], "wb");
    if (!f) { perror("out"); return 1; }
    fwrite(buf, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "wrote %llu f32 (dims:", (unsigned long long)n);
    for (uint32_t i = 0; i < t->n_dims; i++) fprintf(stderr, " %llu", (unsigned long long)t->dims[i]);
    fprintf(stderr, ")\n");
    free(buf);
    fatcat_gguf_close(g);
    return 0;
}
