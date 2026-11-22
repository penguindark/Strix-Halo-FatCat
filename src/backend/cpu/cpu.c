/* fatcat CPU backend — dispatch + scalar fallback (P2b).
 *
 * Public entry points route to the AVX-512 kernels (src/backend/cpu/avx512.c)
 * when the CPU supports them, else to a scalar reference path built on
 * fatcat_dequant (src/model/dequant.c). FATCAT_CPU_SCALAR=1 forces scalar for
 * A/B benchmarking. Row-level OpenMP parallelism stays with the caller. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"
#include "fatcat/gguf.h"

/* avx512.c */
void fatcat_gemv_q8_0_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_q6_k_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_iq1_s_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemv_iq2_xxs_avx512(const void *, const float *, float *, int64_t, int64_t);
void fatcat_gemm_iq1_s_avx512(const void *, const float *, float *, int64_t, int64_t, int);
void fatcat_gemm_iq2_xxs_avx512(const void *, const float *, float *, int64_t, int64_t, int);
void fatcat_gemm_q8_0_avx512(const void *, const float *, float *, int64_t, int64_t, int);
void fatcat_gemm_q6_k_avx512(const void *, const float *, float *, int64_t, int64_t, int);
void fatcat_gemv_f32_avx512(const float *, const float *, float *, int64_t, int64_t);
float fatcat_dot_avx512(const float *, const float *, int64_t);
float fatcat_dot_f32_f16_avx512(const float *, const uint16_t *, int64_t);
void fatcat_kv_accum_f16_avx512(float *, float, float, const uint16_t *, int64_t);
void fatcat_rmsnorm_avx512(const float *, const float *, float, float *, int64_t);

static int g_avx512 = -1;

static bool use_avx512(void)
{
    if (g_avx512 < 0) {
        int ok = __builtin_cpu_supports("avx512f")  && __builtin_cpu_supports("avx512bw")
              && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
        const char *force = getenv("FATCAT_CPU_SCALAR");
        if (force && atoi(force)) ok = 0;
        g_avx512 = ok;
    }
    return g_avx512 == 1;
}

bool fatcat_cpu_avx512(void) { return use_avx512(); }

/* ---------- scalar fallback (block-aligned rows only) ---------- */

static void gemv_scalar(uint32_t type, const void *W, const float *x, float *y,
                        int64_t n_out, int64_t n_in)
{
    const uint64_t row_bytes = fatcat_dequant_src_bytes(type, (uint64_t)n_in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++) {
        float tmp[1024];
        const char *row = (const char *)W + i * (int64_t)row_bytes;
        float acc = 0.0f;
        for (int64_t off = 0; off < n_in; off += 1024) {
            int64_t len = n_in - off < 1024 ? n_in - off : 1024;
            if (!fatcat_dequant(type, row + fatcat_dequant_src_bytes(type, (uint64_t)off),
                                tmp, (uint64_t)len)) { y[i] = 0.0f; acc = -1.0f; break; }
            for (int64_t k = 0; k < len; k++) acc += tmp[k] * x[off + k];
        }
        y[i] = acc;
    }
}

/* ---------- public API ---------- */

void fatcat_cpu_gemv_f32(const float *W, const float *x, float *y,
                         int64_t n_out, int64_t n_in)
{
    if (use_avx512()) { fatcat_gemv_f32_avx512(W, x, y, n_out, n_in); return; }
    for (int64_t i = 0; i < n_out; i++) {
        float acc = 0.0f;
        const float *row = W + i * n_in;
        for (int64_t k = 0; k < n_in; k++) acc += row[k] * x[k];
        y[i] = acc;
    }
}

void fatcat_cpu_gemv_q8_0(const void *W, const float *x, float *y,
                          int64_t n_out, int64_t n_in)
{
    if (n_in % 32) { gemv_scalar(FATCAT_GGML_Q8_0, W, x, y, n_out, n_in); return; }
    if (use_avx512()) {
        fatcat_gemv_q8_0_avx512(W, x, y, n_out, n_in);
        return;
    }
    gemv_scalar(FATCAT_GGML_Q8_0, W, x, y, n_out, n_in);
}

void fatcat_cpu_gemv_q6_k(const void *W, const float *x, float *y,
                          int64_t n_out, int64_t n_in)
{
    if (n_in % 256) { gemv_scalar(FATCAT_GGML_Q6_K, W, x, y, n_out, n_in); return; }
    if (use_avx512()) {
        fatcat_gemv_q6_k_avx512(W, x, y, n_out, n_in);
        return;
    }
    gemv_scalar(FATCAT_GGML_Q6_K, W, x, y, n_out, n_in);
}

void fatcat_cpu_gemv_iq1_s(const void *W, const float *x, float *y,
                           int64_t n_out, int64_t n_in)
{
    if (n_in % 256) { gemv_scalar(FATCAT_GGML_IQ1_S, W, x, y, n_out, n_in); return; }
    if (use_avx512()) {
        fatcat_gemv_iq1_s_avx512(W, x, y, n_out, n_in);
        return;
    }
    gemv_scalar(FATCAT_GGML_IQ1_S, W, x, y, n_out, n_in);
}

void fatcat_cpu_gemv_iq2_xxs(const void *W, const float *x, float *y,
                             int64_t n_out, int64_t n_in)
{
    if (n_in % 256) { gemv_scalar(FATCAT_GGML_IQ2_XXS, W, x, y, n_out, n_in); return; }
    if (use_avx512()) {
        fatcat_gemv_iq2_xxs_avx512(W, x, y, n_out, n_in);
        return;
    }
    gemv_scalar(FATCAT_GGML_IQ2_XXS, W, x, y, n_out, n_in);
}

void fatcat_cpu_gemv(uint32_t ggml_type, const void *W, const float *x,
                     float *y, int64_t n_out, int64_t n_in)
{
    switch (ggml_type) {
    case FATCAT_GGML_F32:     fatcat_cpu_gemv_f32(W, x, y, n_out, n_in); return;
    case FATCAT_GGML_Q8_0:    fatcat_cpu_gemv_q8_0(W, x, y, n_out, n_in); return;
    case FATCAT_GGML_Q6_K:    fatcat_cpu_gemv_q6_k(W, x, y, n_out, n_in); return;
    case FATCAT_GGML_IQ1_S:   fatcat_cpu_gemv_iq1_s(W, x, y, n_out, n_in); return;
    case FATCAT_GGML_IQ2_XXS: fatcat_cpu_gemv_iq2_xxs(W, x, y, n_out, n_in); return;
    default:                  gemv_scalar(ggml_type, W, x, y, n_out, n_in); return;
    }
}

/* scalar batched fallback: decode each row once per 1024-elem chunk, then
 * dot every token in the same order gemv_scalar uses -> bit-identical. */
static void gemm_scalar(uint32_t type, const void *W, const float *X, float *Y,
                        int64_t n_out, int64_t n_in, int n_tok)
{
    const uint64_t row_bytes = fatcat_dequant_src_bytes(type, (uint64_t)n_in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++) {
        const char *row = (const char *)W + i * (int64_t)row_bytes;
        for (int t0 = 0; t0 < n_tok; t0 += 8) {
            const int n = n_tok - t0 < 8 ? n_tok - t0 : 8;
            float acc[8] = {0};
            int bad = 0;
            for (int64_t off = 0; off < n_in && !bad; off += 1024) {
                float tmp[1024];
                int64_t len = n_in - off < 1024 ? n_in - off : 1024;
                if (!fatcat_dequant(type, row + fatcat_dequant_src_bytes(type, (uint64_t)off),
                                    tmp, (uint64_t)len)) { bad = 1; break; }
                for (int t = 0; t < n; t++) {
                    const float *xt = X + (int64_t)(t0 + t) * n_in + off;
                    float a = acc[t];
                    for (int64_t k = 0; k < len; k++) a += tmp[k] * xt[k];
                    acc[t] = a;
                }
            }
            for (int t = 0; t < n; t++)
                Y[(int64_t)(t0 + t) * n_out + i] = bad ? -1.0f : acc[t];
        }
    }
}

void fatcat_cpu_gemm(uint32_t ggml_type, const void *W, const float *X,
                     float *Y, int64_t n_out, int64_t n_in, int n_tok)
{
    if (n_tok <= 1) { fatcat_cpu_gemv(ggml_type, W, X, Y, n_out, n_in); return; }
    switch (ggml_type) {
    case FATCAT_GGML_IQ1_S:
        if (!(n_in % 256) && use_avx512()) {
            fatcat_gemm_iq1_s_avx512(W, X, Y, n_out, n_in, n_tok); return;
        }
        break;
    case FATCAT_GGML_IQ2_XXS:
        if (!(n_in % 256) && use_avx512()) {
            fatcat_gemm_iq2_xxs_avx512(W, X, Y, n_out, n_in, n_tok); return;
        }
        break;
    case FATCAT_GGML_Q8_0:
        if (!(n_in % 32) && use_avx512()) {
            fatcat_gemm_q8_0_avx512(W, X, Y, n_out, n_in, n_tok); return;
        }
        break;
    case FATCAT_GGML_Q6_K:
        if (!(n_in % 256) && use_avx512()) {
            fatcat_gemm_q6_k_avx512(W, X, Y, n_out, n_in, n_tok); return;
        }
        break;
    default:
        break;   /* F32/other: per-token gemv (no decode cost to amortize
                  * for F32; the expert path streams IQ1_S/IQ2_XXS, the
                  * dense path Q8_0/Q6_K) */
    }
    if (ggml_type == FATCAT_GGML_IQ1_S || ggml_type == FATCAT_GGML_IQ2_XXS ||
        ggml_type == FATCAT_GGML_Q8_0 || ggml_type == FATCAT_GGML_Q6_K) {
        gemm_scalar(ggml_type, W, X, Y, n_out, n_in, n_tok); return;
    }
    for (int t = 0; t < n_tok; t++)
        fatcat_cpu_gemv(ggml_type, W, X + (int64_t)t * n_in, Y + (int64_t)t * n_out,
                        n_out, n_in);
}

float fatcat_cpu_dot_f32(const float *a, const float *b, int64_t n)
{
    if (use_avx512()) return fatcat_dot_avx512(a, b, n);
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

float fatcat_cpu_dot_f32_f16(const float *a, const uint16_t *b, int64_t n)
{
    if (use_avx512()) return fatcat_dot_f32_f16_avx512(a, b, n);
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * fatcat_f16_to_f32(b[i]);
    return s;
}

void fatcat_cpu_kv_accum_f16(float *ac, float corr, float w,
                             const uint16_t *k, int64_t n)
{
    if (use_avx512()) { fatcat_kv_accum_f16_avx512(ac, corr, w, k, n); return; }
    for (int64_t i = 0; i < n; i++) ac[i] = ac[i] * corr + w * fatcat_f16_to_f32(k[i]);
}

void fatcat_cpu_rmsnorm(const float *x, const float *w, float eps,
                        float *y, int64_t n)
{
    if (use_avx512()) { fatcat_rmsnorm_avx512(x, w, eps, y, n); return; }
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int64_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}
