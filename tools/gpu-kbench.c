/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* gpu-kbench: microbenchmark the HIP fused dequant-GEMV kernels.
 *
 * Measures, on real expert slices from a GGUF:
 *   - practical device read-bandwidth ceiling (grid-stride sum kernel)
 *   - GPU GEMV achieved GB/s (sync staging path and async dev path)
 *   - CPU GEMV achieved GB/s (single thread, same slice) for comparison
 *
 * The GPU can only beat the CPU on this APU if its kernels approach the
 * read-bandwidth ceiling; this tool shows where each kernel stands.
 *
 * usage: gpu-kbench model.gguf [tensor.name ...] [--iters N]
 * Default tensors: one IQ1_S expert gate + down, one Q8_0 shared-expert
 * gate, one Q6_K attention matrix (dims/types as in Mistral-Large-4). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"
#include "fatcat/gguf.h"
#include "fatcat/gpu.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench_tensor(fatcat_gguf *g, const char *name, int iters)
{
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, name);
    if (!t || t->n_dims < 2) { fprintf(stderr, "kbench: %s not found\n", name); return; }
    const long long n_out = (long long)t->dims[0];
    const long long n_in  = (long long)t->dims[1];
    uint64_t elems = (uint64_t)n_out * (uint64_t)n_in;   /* one expert row block */
    size_t bytes = (size_t)fatcat_dequant_src_bytes(t->type, elems);
    if (!bytes) { fprintf(stderr, "kbench: %s bad size\n", name); return; }
    const char *src = (const char *)fatcat_gguf_tensor_ptr(g, t);  /* expert 0 */

    /* x/y must be device-accessible for the async path: use mapped pinned */
    float *x = fatcat_gpu_host_alloc((size_t)n_in * 4),
          *y = fatcat_gpu_host_alloc((size_t)n_out * 4);
    int xdev = 1;
    if (!x || !y) { free(x); free(y); x = malloc((size_t)n_in * 4); y = malloc((size_t)n_out * 4); xdev = 0; }
    if (!x || !y) { fprintf(stderr, "kbench: oom\n"); return; }
    for (long long i = 0; i < n_in; i++) x[i] = (float)((i * 37 % 101) - 50) / 50.0f;

    void *h = fatcat_gpu_upload(t->type, src, bytes);
    /* same bytes, but in mapped HOST memory (like a warm-tier slot) */
    void *hm = NULL; char *mh = NULL;
    {
        mh = fatcat_gpu_host_alloc(bytes);
        if (mh) { memcpy(mh, src, bytes); hm = fatcat_gpu_wrap(mh, t->type);
                  if (!hm) fatcat_gpu_host_free(mh); }
    }
    double gpu_sync = 0, gpu_dev = 0, gpu_hostw = 0;
    if (h) {
        if (fatcat_gpu_gemv(h, t->type, x, y, n_out, n_in)) {   /* warm */
            int ok = 1;
            double t0 = now_s();
            for (int i = 0; i < iters && ok; i++)
                if (!fatcat_gpu_gemv(h, t->type, x, y, n_out, n_in)) ok = 0;
            if (ok) gpu_sync = bytes * (double)iters / (now_s() - t0) / 1e9;
        }
        if (xdev && fatcat_gpu_gemv_dev(h, t->type, x, y, n_out, n_in) == 0 &&
            fatcat_gpu_wait() == 0) {                          /* warm */
            int ok = 1;
            double t0 = now_s();
            for (int i = 0; i < iters && ok; i++)
                if (fatcat_gpu_gemv_dev(h, t->type, x, y, n_out, n_in) != 0 ||
                    fatcat_gpu_wait() != 0) ok = 0;
            if (ok) gpu_dev = bytes * (double)iters / (now_s() - t0) / 1e9;
        }
        if (hm && fatcat_gpu_gemv(hm, t->type, x, y, n_out, n_in)) {  /* warm */
            int ok = 1;
            double t0 = now_s();
            for (int i = 0; i < iters && ok; i++)
                if (!fatcat_gpu_gemv(hm, t->type, x, y, n_out, n_in)) ok = 0;
            if (ok) gpu_hostw = bytes * (double)iters / (now_s() - t0) / 1e9;
        }
        fatcat_gpu_release(h);
        if (hm) fatcat_gpu_release(hm);
        if (mh) fatcat_gpu_host_free(mh);
    }
    fatcat_cpu_gemv(t->type, src, x, y, n_out, n_in);     /* warm */
    double t0 = now_s();
    for (int i = 0; i < iters; i++) fatcat_cpu_gemv(t->type, src, x, y, n_out, n_in);
    double cpu = bytes * (double)iters / (now_s() - t0) / 1e9;

    printf("%-34s %-8s %5lldx%-6lld %6.1f MB  gpu_sync %6.1f  gpu_dev %6.1f  gpu_hostw %6.1f  cpu1t %6.1f GB/s\n",
           name, fatcat_ggml_type_name(t->type), n_out, n_in, bytes / 1e6,
           gpu_sync, gpu_dev, gpu_hostw, cpu);
    if (xdev) { fatcat_gpu_host_free(x); fatcat_gpu_host_free(y); }
    else { free(x); free(y); }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [tensor.name ...] [--iters N]\n", argv[0]);
        return 2;
    }
    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(argv[1], err, sizeof err);
    if (!g) { fprintf(stderr, "open: %s\n", err); return 1; }
    char gerr[256];
    if (!fatcat_gpu_init(4, gerr, sizeof gerr)) {
        fprintf(stderr, "gpu unavailable: %s\n", gerr);
        return 1;
    }
    int iters = 0;
    const char *list[16]; int nlist = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (nlist < 16) list[nlist++] = argv[i];
    }
    printf("readbw  64 MB: %6.1f GB/s\n", fatcat_gpu_readbw( 64ull << 20, 60));
    printf("readbw 512 MB: %6.1f GB/s\n", fatcat_gpu_readbw(512ull << 20, 12));
    printf("readbw   2 GB: %6.1f GB/s\n", fatcat_gpu_readbw(  2ull << 30, 4));
    {   /* synthetic F32 GEMV: pure weight-streaming, no dequant math.
         * If this also sits far below readbw, the gap is contention or
         * launch overhead, not the quantization kernels. */
        const long long fo = 1024, fi = 10240;
        float *fw = malloc((size_t)fo * fi * 4);
        float *fx = fatcat_gpu_host_alloc((size_t)fi * 4);
        float *fy = fatcat_gpu_host_alloc((size_t)fo * 4);
        if (fw && fx && fy) {
            for (long long i = 0; i < fo * fi; i++) fw[i] = (float)((i % 251) - 125) / 125.0f;
            for (long long i = 0; i < fi; i++) fx[i] = (float)((i % 97) - 48) / 48.0f;
            void *fh = fatcat_gpu_upload(0 /*F32*/, fw, (size_t)fo * fi * 4);
            if (fh) {
                int fit = 10;
                fatcat_gpu_gemv(fh, 0, fx, fy, fo, fi);
                double t0 = now_s();
                int ok = 1;
                for (int i = 0; i < fit && ok; i++)
                    if (!fatcat_gpu_gemv(fh, 0, fx, fy, fo, fi)) ok = 0;
                if (ok)
                    printf("synthetic F32 41.9MB GEMV:          %6.1f GB/s\n",
                           (double)fo * fi * 4 * fit / (now_s() - t0) / 1e9);
                fatcat_gpu_release(fh);
            }
        }
        free(fw);
    }
    if (!nlist) {
        /* find a MoE layer (early layers may be dense) */
        static char names[4][48];
        int found = 0;
        for (int il = 0; il < 8 && !found; il++) {
            char probe[48];
            snprintf(probe, sizeof probe, "blk.%d.ffn_gate_exps.weight", il);
            if (fatcat_gguf_find_tensor(g, probe)) found = il;
        }
        if (!found) found = 3;
        snprintf(names[0], 48, "blk.%d.ffn_gate_exps.weight", found);
        snprintf(names[1], 48, "blk.%d.ffn_down_exps.weight", found);
        snprintf(names[2], 48, "blk.%d.ffn_gate_shexp.weight", found);
        snprintf(names[3], 48, "blk.%d.attn_q.weight", found);
        for (int i = 0; i < 4; i++) list[nlist++] = names[i];
    }
    for (int i = 0; i < nlist; i++) {
        const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, list[i]);
        int it = iters;
        if (!it && t) {   /* ~0.4 s of weight traffic per case */
            uint64_t b = fatcat_dequant_src_bytes(t->type, t->dims[0] * t->dims[1]);
            it = b ? (int)(0.4e9 / b) : 10;
            if (it < 3) it = 3; if (it > 300) it = 300;
        }
        if (!it) it = 10;
        bench_tensor(g, list[i], it);
    }
    fatcat_gpu_shutdown();
    fatcat_gguf_close(g);
    return 0;
}
