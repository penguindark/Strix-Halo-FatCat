/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* cpu-bench — benchmark the CPU gemv backend on real GGUF tensors.
 *
 * Usage: cpu-bench MODEL.gguf TENSOR.NAME [n_out_limit] [reps]
 * Runs fatcat_cpu_gemv on the (sub)matrix and reports ms + effective GB/s of
 * quantized weight bytes. Set FATCAT_CPU_SCALAR=1 to force the scalar
 * fallback for A/B; OMP_NUM_THREADS controls row parallelism.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"
#include "fatcat/gguf.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tensor [n_out_limit] [reps]\n", argv[0]); return 2; }
    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(argv[1], err, sizeof err);
    if (!g) { fprintf(stderr, "gguf: %s\n", err); return 1; }
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, argv[2]);
    if (!t) { fprintf(stderr, "tensor %s not found\n", argv[2]); return 1; }

    int64_t n_in = (int64_t)t->dims[0];
    int64_t n_out = t->n_dims > 1 ? (int64_t)t->dims[1] : 1;
    if (argc > 3) { int64_t lim = atoll(argv[3]); if (lim < n_out) n_out = lim; }
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    uint64_t row_bytes = fatcat_dequant_src_bytes(t->type, (uint64_t)n_in);
    if (!row_bytes) { fprintf(stderr, "unsupported type %u\n", t->type); return 1; }

    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    for (int64_t i = 0; i < n_in; i++) x[i] = (float)((i * 1103515245 + 12345) % 2000 - 1000) / 1000.0f;

    const char *base = (const char *)fatcat_gguf_tensor_ptr(g, t);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        fatcat_cpu_gemv(t->type, base, x, y, n_out, n_in);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    double bytes = (double)row_bytes * (double)n_out;
    printf("%-42s type=%2u  %6lld x %-6lld  %8.2f ms  %7.2f GB/s  %s\n",
           argv[2], t->type, (long long)n_out, (long long)n_in,
           best * 1e3, bytes / best / 1e9,
           fatcat_cpu_avx512() ? "avx512" : "scalar");
    fatcat_gguf_close(g);
    return 0;
}
