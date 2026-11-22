/* fatcat — CPU compute backend (P2b).
 *
 * Fused dequantize + GEMV kernels plus vector primitives for the always-
 * resident weight formats (Q8_0, Q6_K, F32) and the streamed expert formats
 * (IQ1_S, IQ2_XXS). Semantics match src/model/dequant.c bit-for-bit in the
 * decoded values; the dot-product accumulation order differs, so results are
 * numerically equivalent (f32 rounding noise), not bit-identical.
 *
 * All gemv kernels compute one row at a time: y[i] = dot(W_row_i, x).
 * Row-parallelism (OpenMP) stays with the caller. n_in must be a multiple of
 * the format's block size (always true for GGUF tensors).
 *
 * Dispatch: kernels_*() call the AVX-512 path when the CPU supports it
 * (Strix Halo: Zen 5, always), else the scalar reference path.
 * FATCAT_CPU_SCALAR=1 forces the scalar path (A/B benchmarking). */
#ifndef FATCAT_CPU_H
#define FATCAT_CPU_H

#include <stdbool.h>
#include <stdint.h>

/* true when the AVX-512 path is active (after first use). */
bool fatcat_cpu_avx512(void);

/* y[i] = dot_f32(a+i*n, b, n) for i in [0, n_out); W rows are n_in f32. */
void fatcat_cpu_gemv_f32(const float *W, const float *x, float *y,
                         int64_t n_out, int64_t n_in);

/* Generic dispatch by ggml type (F32/Q8_0/Q6_K/IQ1_S/IQ2_XXS, else scalar
 * dequant fallback). Row-level OpenMP parallelism lives inside these. */
void fatcat_cpu_gemv(uint32_t ggml_type, const void *W, const float *x,
                     float *y, int64_t n_out, int64_t n_in);

/* y[i] = dot(dequant(W_row_i), x). Row stride = quantized row bytes. */
void fatcat_cpu_gemv_q8_0(const void *W, const float *x, float *y,
                          int64_t n_out, int64_t n_in);
void fatcat_cpu_gemv_q6_k(const void *W, const float *x, float *y,
                          int64_t n_out, int64_t n_in);
void fatcat_cpu_gemv_iq1_s(const void *W, const float *x, float *y,
                           int64_t n_out, int64_t n_in);
void fatcat_cpu_gemv_iq2_xxs(const void *W, const float *x, float *y,
                             int64_t n_out, int64_t n_in);

/* Plain f32 dot product. */
float fatcat_cpu_dot_f32(const float *a, const float *b, int64_t n);

/* Batched GEMM (D1 kernels): Y[t*n_out + i] = dot(dequant(W_row_i), X + t*n_in)
 * for t in [0, n_tok). X is n_tok rows of n_in f32, Y is n_tok rows of n_out.
 * Each token's accumulation order matches fatcat_cpu_gemv exactly, so results
 * are BIT-IDENTICAL to calling gemv once per token; the win is that each
 * quantized weight row is decoded once for all n_tok tokens (IQ1_S/IQ2_XXS
 * dedicated kernels; other types loop gemv per token). */
void fatcat_cpu_gemm(uint32_t ggml_type, const void *W, const float *X,
                     float *Y, int64_t n_out, int64_t n_in, int n_tok);

/* f32 x f16 dot product (f16 KV cache read path). */
float fatcat_cpu_dot_f32_f16(const float *a, const uint16_t *b, int64_t n);

/* MLA value accumulation from an f16 key row: ac[i] = ac[i]*corr + w*k[i]. */
void fatcat_cpu_kv_accum_f16(float *ac, float corr, float w,
                             const uint16_t *k, int64_t n);

/* y = x/rms(x) * w (RMSNorm, eps inside the mean-square). */
void fatcat_cpu_rmsnorm(const float *x, const float *w, float eps,
                        float *y, int64_t n);

#endif /* FATCAT_CPU_H */
