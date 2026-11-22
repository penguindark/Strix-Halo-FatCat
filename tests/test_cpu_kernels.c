/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* test_cpu_kernels — AVX-512 fused dequant-GEMV kernels vs scalar reference.
 *
 * Reference = fatcat_dequant (bit-exact, validated vs gguf-py in
 * test_dequant) + naive dot. Random byte payloads are valid for all tested
 * formats (every bit pattern decodes). Tolerance covers f32 accumulation
 * order differences only. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"
#include "fatcat/gguf.h"

/* direct kernel entry points (bypass dispatch so the test is meaningful) */
void fatcat_gemv_q8_0_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_q6_k_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_iq1_s_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_iq2_xxs_avx512(const void *, const float *, float *, int64_t, int64_t);

static uint64_t rng_state = 0x12345678ABCDEFULL;
static uint32_t rnd(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static void fill_rand(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rnd();
}

/* keep the per-block f16 scale finite and ~O(1): exponent forced to 0b011110 */
static void patch_scales(uint8_t *p, size_t nblocks, size_t block_bytes, size_t scale_off)
{
    for (size_t b = 0; b < nblocks; b++) {
        uint16_t h = (uint16_t)((rnd() & 0x3FF) | 0x3C00);
        memcpy(p + b * block_bytes + scale_off, &h, 2);
    }
}

static void fill_x(float *x, int64_t n)
{
    for (int64_t i = 0; i < n; i++) x[i] = ((float)rnd() / 2147483648.0f) - 1.0f;
}

/* reference: dequant row + naive dot */
static void ref_gemv(uint32_t type, const uint8_t *W, const float *x, float *y,
                     int64_t n_out, int64_t n_in)
{
    const uint64_t rb = fatcat_dequant_src_bytes(type, (uint64_t)n_in);
    float *tmp = malloc((size_t)n_in * sizeof(float));
    for (int64_t i = 0; i < n_out; i++) {
        fatcat_dequant(type, W + i * (int64_t)rb, tmp, (uint64_t)n_in);
        float acc = 0.0f;
        for (int64_t k = 0; k < n_in; k++) acc += tmp[k] * x[k];
        y[i] = acc;
    }
    free(tmp);
}

static int check(const char *name, const float *a, const float *b, int64_t n)
{
    double num = 0.0, den = 0.0;
    int64_t worst = 0;
    double worst_rel = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
        double rel = fabs(b[i]) > 1e-9 ? fabs(d / b[i]) : 0.0;
        if (rel > worst_rel) { worst_rel = rel; worst = i; }
    }
    double rel = sqrt(num / (den + 1e-30));
    int ok = rel < 1e-4;
    printf("%-28s relerr=%.3e %s", name, rel, ok ? "PASS" : "FAIL");
    if (!ok) printf("  (worst elem %lld: %.6f vs %.6f)", (long long)worst, a[worst], b[worst]);
    printf("\n");
    return ok;
}

int main(void)
{
    int ok = 1;
    const int64_t n_out = 64;

    struct { uint32_t type; const char *name; int64_t n_in;
             size_t blk, scale_off; } cases[] = {
        { FATCAT_GGML_Q8_0,    "gemv_q8_0",    6144,  34,  0   },
        { FATCAT_GGML_Q8_0,    "gemv_q8_0_wide", 24576, 34, 0  },
        { FATCAT_GGML_Q6_K,    "gemv_q6_k",    6144,  210, 208 },
        { FATCAT_GGML_Q6_K,    "gemv_q6_k_wide", 24576, 210, 208 },
        { FATCAT_GGML_IQ1_S,   "gemv_iq1_s",   6144,  50,  0   },
        { FATCAT_GGML_IQ2_XXS, "gemv_iq2_xxs", 6144,  66,  0   },
    };

    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int64_t n_in = cases[c].n_in;
        uint64_t rb = fatcat_dequant_src_bytes(cases[c].type, (uint64_t)n_in);
        uint8_t *W = malloc((size_t)rb * (size_t)n_out);
        float *x = malloc((size_t)n_in * sizeof(float));
        float *ya = malloc((size_t)n_out * sizeof(float));
        float *yr = malloc((size_t)n_out * sizeof(float));
        fill_rand(W, (size_t)rb * (size_t)n_out);
        patch_scales(W, (size_t)rb * (size_t)n_out / cases[c].blk, cases[c].blk, cases[c].scale_off);
        fill_x(x, n_in);
        ref_gemv(cases[c].type, W, x, yr, n_out, n_in);

        /* force AVX-512 path explicitly (test must exercise the kernels) */
        switch (cases[c].type) {
        case FATCAT_GGML_Q8_0:    fatcat_gemv_q8_0_avx512(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_Q6_K:    fatcat_gemv_q6_k_avx512(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_IQ1_S:   fatcat_gemv_iq1_s_avx512(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_IQ2_XXS: fatcat_gemv_iq2_xxs_avx512(W, x, ya, n_out, n_in); break;
        }
        ok &= check(cases[c].name, ya, yr, n_out);

        /* and via the public dispatch API */
        switch (cases[c].type) {
        case FATCAT_GGML_Q8_0:    fatcat_cpu_gemv_q8_0(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_Q6_K:    fatcat_cpu_gemv_q6_k(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_IQ1_S:   fatcat_cpu_gemv_iq1_s(W, x, ya, n_out, n_in); break;
        case FATCAT_GGML_IQ2_XXS: fatcat_cpu_gemv_iq2_xxs(W, x, ya, n_out, n_in); break;
        }
        ok &= check(cases[c].name, ya, yr, n_out);
        free(W); free(x); free(ya); free(yr);
    }

    /* Batched GEMM (D1): must be BIT-identical to the per-token gemv loop
     * for every token count (tests the 8-token accumulator tiling). */
    {
        int64_t n_in = 512;
        struct { uint32_t type; const char *name; size_t blk, scale_off; } gc[] = {
            { FATCAT_GGML_IQ1_S,   "gemm_iq1_s_bitid",   50, 0 },
            { FATCAT_GGML_IQ2_XXS, "gemm_iq2_xxs_bitid", 66, 0 },
            { FATCAT_GGML_Q8_0,    "gemm_q8_0_bitid",    34, 0 },
            { FATCAT_GGML_Q6_K,    "gemm_q6_k_bitid",    210, 208 },
        };
        for (size_t c = 0; c < sizeof gc / sizeof gc[0]; c++) {
            uint64_t rb = fatcat_dequant_src_bytes(gc[c].type, (uint64_t)n_in);
            uint8_t *W = malloc((size_t)rb * (size_t)n_out);
            fill_rand(W, (size_t)rb * (size_t)n_out);
            patch_scales(W, (size_t)rb * (size_t)n_out / gc[c].blk, gc[c].blk, gc[c].scale_off);
            int bitid = 1;
            for (int nt = 1; nt <= 9; nt++) {
                float *X = malloc((size_t)nt * n_in * sizeof(float));
                float *y1 = malloc((size_t)nt * n_out * sizeof(float));
                float *y2 = malloc((size_t)nt * n_out * sizeof(float));
                for (int t = 0; t < nt; t++) fill_x(X + (int64_t)t * n_in, n_in);
                for (int t = 0; t < nt; t++)
                    fatcat_cpu_gemv(gc[c].type, W, X + (int64_t)t * n_in,
                                    y1 + (int64_t)t * n_out, n_out, n_in);
                fatcat_cpu_gemm(gc[c].type, W, X, y2, n_out, n_in, nt);
                if (memcmp(y1, y2, (size_t)nt * n_out * sizeof(float)) != 0) bitid = 0;
                free(X); free(y1); free(y2);
            }
            printf("%-28s %s\n", gc[c].name, bitid ? "PASS" : "FAIL");
            ok &= bitid;
            free(W);
        }
    }

    /* dot_f32 + rmsnorm */
    {
        rng_state = 0x5EED1234ULL;   /* reseed: this block's data must
                                      * not depend on how many earlier
                                      * cases consumed the stream */
        int64_t n = 6144;
        float *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float));
        float *w = malloc(n * sizeof(float)), *y1 = malloc(n * sizeof(float));
        float *y2 = malloc(n * sizeof(float));
        fill_x(a, n); fill_x(b, n); fill_x(w, n);
        double ref = 0.0;   /* double reference: an f32 naive sum has its
                             * own cancellation noise that made the 1e-5
                             * threshold data-dependent (flaky) */
        for (int64_t i = 0; i < n; i++) ref += (double)a[i] * (double)b[i];
        float got = fatcat_cpu_dot_f32(a, b, n);
        double rel = fabs((double)got - ref) / (fabs(ref) + 1e-9);
        printf("%-28s relerr=%.3e %s\n", "dot_f32", rel, rel < 1e-5 ? "PASS" : "FAIL");
        ok &= rel < 1e-5;

        fatcat_cpu_rmsnorm(a, w, 1e-6f, y1, n);
        float ss = 0.0f;
        for (int64_t i = 0; i < n; i++) ss += a[i] * a[i];
        float sc = 1.0f / sqrtf(ss / (float)n + 1e-6f);
        for (int64_t i = 0; i < n; i++) y2[i] = a[i] * sc * w[i];
        ok &= check("rmsnorm", y1, y2, n);
        free(a); free(b); free(w); free(y1); free(y2);
    }

    /* f16 KV read kernels: dot_f32_f16 and kv_accum_f16 vs scalar */
    {
        int64_t n = 512;
        float *a = malloc(n * sizeof(float));
        uint16_t *b = malloc(n * sizeof(uint16_t));
        fill_x(a, n);
        for (int64_t i = 0; i < n; i++)   /* normal-range f16 bit patterns */
            b[i] = (uint16_t)((rnd() & 0x8000u) | 0x3800u | (rnd() & 0x03FFu));
        float ref = 0.0f;
        for (int64_t i = 0; i < n; i++) ref += a[i] * fatcat_f16_to_f32(b[i]);
        float got = fatcat_cpu_dot_f32_f16(a, b, n);
        double rel = fabs((double)got - ref) / (fabs((double)ref) + 1e-9);
        printf("%-28s relerr=%.3e %s\n", "dot_f32_f16", rel, rel < 1e-5 ? "PASS" : "FAIL");
        ok &= rel < 1e-5;

        float corr = 0.7f, w = 0.4f;
        float *ac1 = malloc(n * sizeof(float)), *ac2 = malloc(n * sizeof(float));
        fill_x(ac1, n); memcpy(ac2, ac1, n * sizeof(float));
        fatcat_cpu_kv_accum_f16(ac1, corr, w, b, n);
        for (int64_t i = 0; i < n; i++) ac2[i] = ac2[i] * corr + w * fatcat_f16_to_f32(b[i]);
        ok &= check("kv_accum_f16", ac1, ac2, n);
        free(a); free(b); free(ac1); free(ac2);
    }

    printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
