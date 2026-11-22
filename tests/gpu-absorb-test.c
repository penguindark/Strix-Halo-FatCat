/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* gpu-absorb-test — bit-identity check for the D5e attention kernels:
 *   fatcat_gpu_absorb_q8_0      vs CPU absorb (dequant_q8_0 + dot_f32)
 *   fatcat_gpu_gemv_q8_0_batched vs CPU gemv (fatcat_cpu_gemv_q8_0 per head)
 * RUN WITH THE ASSISTANT LLM STOPPED (hipMalloc fails otherwise).
 * Exit 0 = all rows bit-identical. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "fatcat/gpu.h"
#include "fatcat/cpu.h"
#include "fatcat/dequant.h"

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }

/* fill a Q8_0 buffer with safe random data: finite f16 scales, random int8 */
static void fill_q8(uint8_t *p, size_t rows, int n_in) {
    for (size_t r = 0; r < rows; r++)
        for (int b = 0; b < n_in / 32; b++) {
            uint8_t *blk = p + r * (size_t)(n_in / 32 * 34) + (size_t)b * 34;
            uint16_t h = (uint16_t)((rnd() & 0x8000) | ((14 + rnd() % 8) << 10) | (rnd() & 0x3ff));
            blk[0] = h & 0xff; blk[1] = h >> 8;
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t)rnd();
        }
}

int main(void)
{
    char err[256];
    if (!fatcat_gpu_init(16, err, sizeof(err))) { printf("gpu: %s\n", err); return 1; }
    int fails = 0;

    /* ---- absorb: 64 heads x (512 rows x 64 in), x_stride 128,
     * row_stride 136 (pad_rope layout as in NoPE layers: rows stored
     * over the full qk=128 elems, only the first 64 participate) ---- */
    {
        const int n_items = 64, rpi = 512, n_in = 64; const long long xs = 128;
        const long long rs = 136;   /* padded row bytes: 128 elems */
        const size_t rows = (size_t)n_items * rpi;
        uint8_t *W = malloc(rows * (size_t)rs);
        memset(W, 0, rows * (size_t)rs);
        float *x = fatcat_gpu_host_alloc((size_t)n_items * xs * sizeof(float));
        float *yg = fatcat_gpu_host_alloc(rows * sizeof(float));
        float *yc = malloc(rows * sizeof(float));
        if (!x || !yg) { printf("host_alloc failed (assistant stopped?)\n"); return 1; }
        for (size_t r = 0; r < rows; r++)
            fill_q8(W + r * (size_t)rs, 1, n_in);   /* first 64 elems valid */
        for (size_t i = 0; i < (size_t)n_items * xs; i++) x[i] = (float)((int)(rnd() % 2000) - 1000) / 1000.0f;
        void *h = fatcat_gpu_upload(8 /*Q8_0*/, W, rows * (size_t)rs);
        if (!h) { printf("upload failed\n"); return 1; }
        if (fatcat_gpu_absorb_q8_0(h, x, yg, rpi, n_items, n_in, xs, rs) != 0) {
            printf("absorb launch failed\n"); return 1;
        }
        fatcat_gpu_wait();
        float tmp[64];
        for (size_t r = 0; r < rows; r++) {
            fatcat_dequant(8, W + r * (size_t)rs, tmp, n_in);
            yc[r] = fatcat_cpu_dot_f32(tmp, x + (r / rpi) * xs, n_in);
        }
        size_t bad = 0;
        for (size_t r = 0; r < rows; r++) if (memcmp(&yg[r], &yc[r], 4)) bad++;
        printf("absorb 64x(512x64): %zu/%zu bit-mismatches %s\n", bad, rows, bad ? "FAIL" : "OK");
        fails += bad != 0;
        fatcat_gpu_release(h); free(W); free(yc);
        fatcat_gpu_host_free(x); fatcat_gpu_host_free(yg);
    }

    /* ---- wv: 64 heads x (128 rows x 512 in), x_stride 512 ---- */
    {
        const int n_items = 64, n_out = 128, n_in = 512;
        const size_t rb = (size_t)(n_in / 32 * 34);
        uint8_t *W = malloc((size_t)n_items * n_out * rb);
        float *x = fatcat_gpu_host_alloc((size_t)n_items * n_in * sizeof(float));
        float *yg = fatcat_gpu_host_alloc((size_t)n_items * n_out * sizeof(float));
        float *yv = fatcat_gpu_host_alloc((size_t)n_items * n_out * sizeof(float));
        float *yc = malloc((size_t)n_items * n_out * sizeof(float));
        if (!x || !yg || !yv) { printf("host_alloc failed (assistant stopped?)\n"); return 1; }
        fill_q8(W, (size_t)n_items * n_out, n_in);
        for (size_t i = 0; i < (size_t)n_items * n_in; i++) x[i] = (float)((int)(rnd() % 2000) - 1000) / 1000.0f;
        void *h = fatcat_gpu_upload(8, W, (size_t)n_items * n_out * rb);
        if (!h) { printf("upload failed\n"); return 1; }
        if (fatcat_gpu_gemv_q8_0_batched(h, (long long)n_out * rb, x, n_in, yg,
                                         n_out, n_out, n_in, n_items) != 0) {
            printf("batched launch failed\n"); return 1;
        }
        fatcat_gpu_wait();
        /* reference A: the EXISTING gpu gemv kernel, per head (bit-target
         * for the batched kernel: same per-row body must match exactly).
         * NOTE: a handle is anchored at its base pointer, so each head
         * needs its own upload — one handle cannot be offset per head. */
        for (int it = 0; it < n_items; it++) {
            void *hh = fatcat_gpu_upload(8, W + (size_t)it * n_out * rb,
                                         (size_t)n_out * rb);
            if (!hh || !fatcat_gpu_gemv(hh, 8, x + (size_t)it * n_in,
                                        yv + (size_t)it * n_out, n_out, n_in)) {
                printf("gemv failed\n"); return 1;
            }
            fatcat_gpu_release(hh);
        }
        /* reference B: CPU gemv (engine CPU fallback) */
        for (int it = 0; it < n_items; it++)
            fatcat_cpu_gemv_q8_0(W + (size_t)it * n_out * rb, x + (size_t)it * n_in,
                                 yc + (size_t)it * n_out, n_out, n_in);
        size_t bad_v = 0, bad_c = 0;
        int ulp_v = 0, ulp_c = 0;
        for (size_t r = 0; r < (size_t)n_items * n_out; r++) {
            if (memcmp(&yg[r], &yv[r], 4)) bad_v++;
            if (memcmp(&yg[r], &yc[r], 4)) bad_c++;
            int u, vc, uc;
            memcpy(&u, &yg[r], 4); memcpy(&vc, &yv[r], 4); memcpy(&uc, &yc[r], 4);
            /* integer ULP distance (same-sign region; fine for random data) */
            int dv = u - vc; dv = dv < 0 ? -dv : dv;
            int dc = u - uc; dc = dc < 0 ? -dc : dc;
            if (dv > ulp_v) ulp_v = dv;
            if (dc > ulp_c) ulp_c = dc;
        }
        printf("wv batched vs gpu-gemv: %zu/%zu bit-mismatches (max ULP %d) %s\n",
               bad_v, (size_t)n_items * n_out, ulp_v, bad_v ? "FAIL" : "OK");
        printf("wv batched vs cpu-gemv: %zu/%zu bit-mismatches (max ULP %d)\n",
               bad_c, (size_t)n_items * n_out, ulp_c);
        fails += bad_v != 0;
        fatcat_gpu_release(h); free(W); free(yc);
        fatcat_gpu_host_free(x); fatcat_gpu_host_free(yg); fatcat_gpu_host_free(yv);
    }

    fatcat_gpu_shutdown();
    printf("%s\n", fails ? "FAIL" : "ALL PASS");
    return fails != 0;
}
