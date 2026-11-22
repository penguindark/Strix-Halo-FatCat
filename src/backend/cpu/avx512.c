/* fatcat CPU backend — AVX-512 kernels (P2b).
 *
 * Fused dequant + GEMV row kernels and vector primitives. Zen 5 (Strix Halo)
 * provides AVX-512F/BW/VL/DQ; no AVX-512-VNNI int8 dots are used because the
 * design keeps weights in exact f32 after dequant (bit-exact with
 * src/model/dequant.c decode values; accumulation order differs from the
 * scalar reference, which is f32 rounding noise only).
 *
 * All kernels are single-row; row-level parallelism (OpenMP) is the caller's.
 * Decoded values land in 8-32 element register windows and are folded into
 * the dot immediately — no full-row f32 scratch buffer (the old scalar path
 * streamed a 32 KB tmp through L1/L2 per row). */
#include <immintrin.h>
#include <math.h>

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"

/* codebook grids, defined in src/model/quant_tables.c (ggml-identical) */
extern const uint64_t fatcat_iq1s_grid[2048];
extern const uint64_t fatcat_iq2xxs_grid[256];
extern const uint8_t  fatcat_ksigns_iq2xs[128];
extern const uint8_t  fatcat_kmask_iq2xs[8];

/* ---------- primitives ---------- */

static inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

float fatcat_dot_avx512(const float *a, const float *b, int64_t n)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), a3);
    }
    for (; i + 16 <= n; i += 16)
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* dot of f32 a with f16 b (f16 KV cache read path). F16C widens 16 halves. */
float fatcat_dot_f32_f16_avx512(const float *a, const uint16_t *b, int64_t n)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
             _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i))), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
             _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i + 16))), a1);
    }
    for (; i + 16 <= n; i += 16)
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
             _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i))), a0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; i < n; i++) s += a[i] * fatcat_f16_to_f32(b[i]);
    return s;
}

/* MLA value accumulation from an f16 key row: ac[i] = ac[i]*corr + w*k[i]. */
void fatcat_kv_accum_f16_avx512(float *ac, float corr, float w,
                                const uint16_t *k, int64_t n)
{
    const __m512 vc = _mm512_set1_ps(corr), vw = _mm512_set1_ps(w);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 kf = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(k + i)));
        __m512 a  = _mm512_loadu_ps(ac + i);
        a = _mm512_fmadd_ps(vw, kf, _mm512_mul_ps(a, vc));
        _mm512_storeu_ps(ac + i, a);
    }
    for (; i < n; i++) ac[i] = ac[i] * corr + w * fatcat_f16_to_f32(k[i]);
}

void fatcat_rmsnorm_avx512(const float *x, const float *w, float eps,
                           float *y, int64_t n)
{
    __m512 ss = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(x + i), v1 = _mm512_loadu_ps(x + i + 16);
        ss = _mm512_fmadd_ps(v0, v0, ss);
        ss = _mm512_fmadd_ps(v1, v1, ss);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        ss = _mm512_fmadd_ps(v, v, ss);
    }
    float scale = 1.0f / sqrtf(_mm512_reduce_add_ps(ss) / (float)n + eps);
    __m512 vs = _mm512_set1_ps(scale);
    int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        __m512 v = _mm512_loadu_ps(x + j), wv = _mm512_loadu_ps(w + j);
        _mm512_storeu_ps(y + j, _mm512_mul_ps(_mm512_mul_ps(v, vs), wv));
    }
    for (; j < n; j++) y[j] = x[j] * scale * w[j];
}

/* ---------- Q8_0: 32 elems / 34 B { f16 d; i8 q[32] } ---------- */

static inline float row_q8_0(const uint8_t *row, const float *x, int64_t n_in)
{
    const int64_t nblk = n_in / 32;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *p = row + b * 34;
        const __m512 vd = _mm512_set1_ps(fatcat_f16_to_f32(*(const uint16_t *)p));
        const int8_t *q = (const int8_t *)(p + 2);
        __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i_u *)q)));
        __m512 v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i_u *)(q + 16))));
        acc0 = _mm512_fmadd_ps(_mm512_mul_ps(v0, vd), _mm512_loadu_ps(x + b * 32),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_mul_ps(v1, vd), _mm512_loadu_ps(x + b * 32 + 16), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

void fatcat_gemv_q8_0_avx512(const void *W, const float *x, float *y,
                             int64_t n_out, int64_t n_in)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        y[i] = row_q8_0(w + i * (n_in / 32 * 34), x, n_in);
}

/* ---------- Q6_K: 256 elems / 210 B ----------
 * Layout (mirrors dequant.c): { u8 ql[128]; u8 qh[64]; i8 sc[16]; f16 d }.
 * Element e of half h (0..127), chunk c = e/32, e' = e%32 (see dequant_q6_k):
 *   ql byte h*64 + (c%2)*32 + e', nibble 0 (c<2) or 4 (c>=2)
 *   qh byte h*32 + e', 2-bit field shifted by {0,2,4,6}[c]
 *   v = ((ql_nib) | (qh2 << 4)) - 32, scaled by d * sc[h*8 + c*2 + e'/16] */
static inline float row_q6_k(const uint8_t *row, const float *x, int64_t n_in)
{
    __m512 acc = _mm512_setzero_ps();
    const int shifts[4] = { 0, 2, 4, 6 };
    for (int64_t blk = 0; blk < n_in / 256; blk++) {
        const uint8_t *p  = row + blk * 210;
        const uint8_t *ql = p, *qh = p + 128;
        const int8_t  *sc = (const int8_t *)(p + 192);
        const float d = fatcat_f16_to_f32(*(const uint16_t *)(p + 208));
        for (int half = 0; half < 2; half++) {
            for (int c = 0; c < 4; c++) {
                const uint8_t *qlc = ql + half * 64 + (c & 1) * 32;
                const uint8_t *qhc = qh + half * 32;
                /* 32 nibbles -> 2x16 int32 (shift-then-mask: neighbor leak
                 * lands above bit 3 and is masked away) */
                __m256i qbytes = _mm256_loadu_si256((const __m256i_u *)qlc);
                if (c >= 2) qbytes = _mm256_srli_epi16(qbytes, 4);
                const __m512i lomask = _mm512_set1_epi32(0x0F);
                __m512i n0 = _mm512_and_si512(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(qbytes)), lomask);
                __m512i n1 = _mm512_and_si512(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(qbytes, 1)), lomask);
                /* one qh byte per element; 2-bit field at shifts[c] (leak
                 * above bit 1 is masked) */
                __m512i qf = _mm512_loadu_si512((const __m512i *)qhc);
                qf = _mm512_and_si512(_mm512_srli_epi32(qf, shifts[c]), _mm512_set1_epi8(3));
                __m512i h0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(qf));
                __m512i h1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(qf, 1));
                __m512 v0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_or_si512(n0, _mm512_slli_epi32(h0, 4)), _mm512_set1_epi32(32)));
                __m512 v1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_or_si512(n1, _mm512_slli_epi32(h1, 4)), _mm512_set1_epi32(32)));
                const int64_t e = blk * 256 + half * 128 + c * 32;
                const __m512 s0 = _mm512_set1_ps(d * (float)sc[half * 8 + c * 2]);
                const __m512 s1 = _mm512_set1_ps(d * (float)sc[half * 8 + c * 2 + 1]);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(v0, s0), _mm512_loadu_ps(x + e),      acc);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(v1, s1), _mm512_loadu_ps(x + e + 16), acc);
            }
        }
    }
    return _mm512_reduce_add_ps(acc);
}

void fatcat_gemv_q6_k_avx512(const void *W, const float *x, float *y,
                             int64_t n_out, int64_t n_in)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        y[i] = row_q6_k(w + i * (n_in / 256 * 210), x, n_in);
}

/* ---------- batched GEMM (D4): dense Q8_0/Q6_K, N tokens per row decode ----------
 * Same bit-identity construction as the IQ1_S/IQ2_XXS batched kernels:
 * per-token accumulator order mirrors the single-x row kernels exactly. */
static void row_q8_0_nt(const uint8_t *row, const float *X, int nt,
                        float *y, int64_t ystride, int64_t n_in)
{
    const int64_t nblk = n_in / 32;
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int n = nt - t0 < 8 ? nt - t0 : 8;
        __m512 acc0[8], acc1[8];
        for (int t = 0; t < n; t++) {
            acc0[t] = _mm512_setzero_ps();
            acc1[t] = _mm512_setzero_ps();
        }
        for (int64_t b = 0; b < nblk; b++) {
            const uint8_t *p = row + b * 34;
            const __m512 vd = _mm512_set1_ps(fatcat_f16_to_f32(*(const uint16_t *)p));
            const int8_t *q = (const int8_t *)(p + 2);
            __m512 v0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i_u *)q)));
            __m512 v1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i_u *)(q + 16))));
            v0 = _mm512_mul_ps(v0, vd);
            v1 = _mm512_mul_ps(v1, vd);
            for (int t = 0; t < n; t++) {
                const float *x = X + (int64_t)(t0 + t) * n_in + b * 32;
                acc0[t] = _mm512_fmadd_ps(v0, _mm512_loadu_ps(x), acc0[t]);
                acc1[t] = _mm512_fmadd_ps(v1, _mm512_loadu_ps(x + 16), acc1[t]);
            }
        }
        for (int t = 0; t < n; t++)
            y[(int64_t)(t0 + t) * ystride] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[t], acc1[t]));
    }
}

void fatcat_gemm_q8_0_avx512(const void *W, const float *X, float *Y,
                             int64_t n_out, int64_t n_in, int n_tok)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        row_q8_0_nt(w + i * (n_in / 32 * 34), X, n_tok, Y + i, n_out, n_in);
}

static void row_q6_k_nt(const uint8_t *row, const float *X, int nt,
                        float *y, int64_t ystride, int64_t n_in)
{
    const int shifts[4] = { 0, 2, 4, 6 };
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int n = nt - t0 < 8 ? nt - t0 : 8;
        __m512 acc[8];
        for (int t = 0; t < n; t++) acc[t] = _mm512_setzero_ps();
        for (int64_t blk = 0; blk < n_in / 256; blk++) {
            const uint8_t *p  = row + blk * 210;
            const uint8_t *ql = p, *qh = p + 128;
            const int8_t  *sc = (const int8_t *)(p + 192);
            const float d = fatcat_f16_to_f32(*(const uint16_t *)(p + 208));
            for (int half = 0; half < 2; half++) {
                for (int c = 0; c < 4; c++) {
                    const uint8_t *qlc = ql + half * 64 + (c & 1) * 32;
                    const uint8_t *qhc = qh + half * 32;
                    __m256i qbytes = _mm256_loadu_si256((const __m256i_u *)qlc);
                    if (c >= 2) qbytes = _mm256_srli_epi16(qbytes, 4);
                    const __m512i lomask = _mm512_set1_epi32(0x0F);
                    __m512i n0 = _mm512_and_si512(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(qbytes)), lomask);
                    __m512i n1 = _mm512_and_si512(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(qbytes, 1)), lomask);
                    __m512i qf = _mm512_loadu_si512((const __m512i *)qhc);
                    qf = _mm512_and_si512(_mm512_srli_epi32(qf, shifts[c]), _mm512_set1_epi8(3));
                    __m512i h0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(qf));
                    __m512i h1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(qf, 1));
                    __m512 v0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                        _mm512_or_si512(n0, _mm512_slli_epi32(h0, 4)), _mm512_set1_epi32(32)));
                    __m512 v1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                        _mm512_or_si512(n1, _mm512_slli_epi32(h1, 4)), _mm512_set1_epi32(32)));
                    const int64_t e = blk * 256 + half * 128 + c * 32;
                    const __m512 s0 = _mm512_set1_ps(d * (float)sc[half * 8 + c * 2]);
                    const __m512 s1 = _mm512_set1_ps(d * (float)sc[half * 8 + c * 2 + 1]);
                    const __m512 w0 = _mm512_mul_ps(v0, s0);
                    const __m512 w1 = _mm512_mul_ps(v1, s1);
                    for (int t = 0; t < n; t++) {
                        const float *x = X + (int64_t)(t0 + t) * n_in;
                        acc[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + e), acc[t]);
                        acc[t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + e + 16), acc[t]);
                    }
                }
            }
        }
        for (int t = 0; t < n; t++) y[(int64_t)(t0 + t) * ystride] = _mm512_reduce_add_ps(acc[t]);
    }
}

void fatcat_gemm_q6_k_avx512(const void *W, const float *X, float *Y,
                             int64_t n_out, int64_t n_in, int n_tok)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        row_q6_k_nt(w + i * (n_in / 256 * 210), X, n_tok, Y + i, n_out, n_in);
}

/* ---------- IQ1_S: 256 elems / 50 B { f16 d; u8 qs[32]; u16 qh[8] } ----------
 * Per 32-sub-block ib: dl = d*(2*((qh[ib]>>12)&7)+1), delta = ±0.125;
 * 4 grid lookups idx = qs[l] | (((qh[ib]>>3l)&7)<<8), v = dl*(grid8[j]+delta). */
static inline float row_iq1_s(const uint8_t *row, const float *x, int64_t n_in)
{
    __m256 acc = _mm256_setzero_ps();
    for (int64_t blk = 0; blk < n_in / 256; blk++) {
        const uint8_t  *p  = row + blk * 50;
        const float d      = fatcat_f16_to_f32(*(const uint16_t *)p);
        const uint8_t  *qs = p + 2;
        const uint16_t *qh = (const uint16_t *)(p + 34);
        for (int ib = 0; ib < 8; ib++) {
            const __m256 vdl = _mm256_set1_ps(d * (float)(2 * ((qh[ib] >> 12) & 7) + 1));
            const __m256 vde = _mm256_set1_ps((qh[ib] & 0x8000) ? -0.125f : 0.125f);
            for (int l = 0; l < 4; l++) {
                const int idx = qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                /* 8 int8 grid values -> 8 f32; v = dl*(g + delta) */
                __m256 g = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i_u *)&fatcat_iq1s_grid[idx])));
                g = _mm256_fmadd_ps(g, vdl, _mm256_mul_ps(vdl, vde));
                const int64_t e = blk * 256 + ib * 32 + l * 8;
                acc = _mm256_fmadd_ps(g, _mm256_loadu_ps(x + e), acc);
            }
        }
    }
    return hsum256(acc);
}

void fatcat_gemv_iq1_s_avx512(const void *W, const float *x, float *y,
                              int64_t n_out, int64_t n_in)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        y[i] = row_iq1_s(w + i * (n_in / 256 * 50), x, n_in);
}

/* ---------- IQ2_XXS: 256 elems / 66 B { f16 d; u32 qs[16] } ----------
 * Per 32-sub-block ib: scale = 0.125*d*(2*(aux1>>28)+1); 4 lookups of an
 * 8-value grid (0..3) with per-element signs from ksigns[(aux1>>7l)&0x7F]. */
static inline float row_iq2_xxs(const uint8_t *row, const float *x, int64_t n_in)
{
    const __m256i kmask = _mm256_set_epi32(
        fatcat_kmask_iq2xs[7], fatcat_kmask_iq2xs[6],
        fatcat_kmask_iq2xs[5], fatcat_kmask_iq2xs[4],
        fatcat_kmask_iq2xs[3], fatcat_kmask_iq2xs[2],
        fatcat_kmask_iq2xs[1], fatcat_kmask_iq2xs[0]);
    __m256 acc = _mm256_setzero_ps();
    for (int64_t blk = 0; blk < n_in / 256; blk++) {
        const uint8_t  *p = row + blk * 66;
        const float d     = fatcat_f16_to_f32(*(const uint16_t *)p);
        const uint32_t *qs = (const uint32_t *)(p + 2);
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t aux0 = qs[2 * ib], aux1 = qs[2 * ib + 1];
            const __m256 vs = _mm256_set1_ps(0.125f * d * (float)(2 * (aux1 >> 28) + 1));
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = fatcat_ksigns_iq2xs[(aux1 >> (7 * l)) & 0x7F];
                __m256i g = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    (const __m128i_u *)&fatcat_iq2xxs_grid[(uint8_t)(aux0 >> (8 * l))]));
                __m256 v = _mm256_cvtepi32_ps(g);
                /* negate where (signs & kmask[j]) != 0 */
                __mmask8 sgn = _mm256_test_epi32_mask(_mm256_set1_epi32(sg), kmask);
                v = _mm256_mask_sub_ps(v, sgn, _mm256_setzero_ps(), v);
                const int64_t e = blk * 256 + ib * 32 + l * 8;
                acc = _mm256_fmadd_ps(_mm256_mul_ps(v, vs), _mm256_loadu_ps(x + e), acc);
            }
        }
    }
    return hsum256(acc);
}

void fatcat_gemv_iq2_xxs_avx512(const void *W, const float *x, float *y,
                                int64_t n_out, int64_t n_in)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        y[i] = row_iq2_xxs(w + i * (n_in / 256 * 66), x, n_in);
}

/* ---------- batched GEMM (D1/D4): one row decode, N token dots ----------
 * row_*_nt computes Y[t*n_out] for one row across nt tokens (<=8). Each
 * token keeps its own accumulator fed in the SAME order as the single-x
 * row kernel above, so every Y value is bit-identical to the gemv result;
 * the grid decode (the expensive part of IQ1_S/IQ2_XXS) runs once per row. */
static void row_iq1_s_nt(const uint8_t *row, const float *X, int nt,
                         float *y, int64_t ystride, int64_t n_in)
{
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int n = nt - t0 < 8 ? nt - t0 : 8;
        __m256 acc[8];
        for (int t = 0; t < n; t++) acc[t] = _mm256_setzero_ps();
        for (int64_t blk = 0; blk < n_in / 256; blk++) {
            const uint8_t  *p  = row + blk * 50;
            const float d      = fatcat_f16_to_f32(*(const uint16_t *)p);
            const uint8_t  *qs = p + 2;
            const uint16_t *qh = (const uint16_t *)(p + 34);
            for (int ib = 0; ib < 8; ib++) {
                const __m256 vdl = _mm256_set1_ps(d * (float)(2 * ((qh[ib] >> 12) & 7) + 1));
                const __m256 vde = _mm256_set1_ps((qh[ib] & 0x8000) ? -0.125f : 0.125f);
                for (int l = 0; l < 4; l++) {
                    const int idx = qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                    __m256 g = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i_u *)&fatcat_iq1s_grid[idx])));
                    g = _mm256_fmadd_ps(g, vdl, _mm256_mul_ps(vdl, vde));
                    const int64_t e = blk * 256 + ib * 32 + l * 8;
                    for (int t = 0; t < n; t++)
                        acc[t] = _mm256_fmadd_ps(g, _mm256_loadu_ps(X + (int64_t)(t0 + t) * n_in + e), acc[t]);
                }
            }
        }
        for (int t = 0; t < n; t++) y[(int64_t)(t0 + t) * ystride] = hsum256(acc[t]);
    }
}

void fatcat_gemm_iq1_s_avx512(const void *W, const float *X, float *Y,
                              int64_t n_out, int64_t n_in, int n_tok)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        row_iq1_s_nt(w + i * (n_in / 256 * 50), X, n_tok,
                     Y + i, n_out, n_in);
}

static void row_iq2_xxs_nt(const uint8_t *row, const float *X, int nt,
                           float *y, int64_t ystride, int64_t n_in)
{
    const __m256i kmask = _mm256_set_epi32(
        fatcat_kmask_iq2xs[7], fatcat_kmask_iq2xs[6],
        fatcat_kmask_iq2xs[5], fatcat_kmask_iq2xs[4],
        fatcat_kmask_iq2xs[3], fatcat_kmask_iq2xs[2],
        fatcat_kmask_iq2xs[1], fatcat_kmask_iq2xs[0]);
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int n = nt - t0 < 8 ? nt - t0 : 8;
        __m256 acc[8];
        for (int t = 0; t < n; t++) acc[t] = _mm256_setzero_ps();
        for (int64_t blk = 0; blk < n_in / 256; blk++) {
            const uint8_t  *p = row + blk * 66;
            const float d     = fatcat_f16_to_f32(*(const uint16_t *)p);
            const uint32_t *qs = (const uint32_t *)(p + 2);
            for (int ib = 0; ib < 8; ib++) {
                const uint32_t aux0 = qs[2 * ib], aux1 = qs[2 * ib + 1];
                const __m256 vs = _mm256_set1_ps(0.125f * d * (float)(2 * (aux1 >> 28) + 1));
                for (int l = 0; l < 4; l++) {
                    const uint8_t sg = fatcat_ksigns_iq2xs[(aux1 >> (7 * l)) & 0x7F];
                    __m256i g = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        (const __m128i_u *)&fatcat_iq2xxs_grid[(uint8_t)(aux0 >> (8 * l))]));
                    __m256 v = _mm256_cvtepi32_ps(g);
                    __mmask8 sgn = _mm256_test_epi32_mask(_mm256_set1_epi32(sg), kmask);
                    v = _mm256_mask_sub_ps(v, sgn, _mm256_setzero_ps(), v);
                    const int64_t e = blk * 256 + ib * 32 + l * 8;
                    const __m256 w8 = _mm256_mul_ps(v, vs);
                    for (int t = 0; t < n; t++)
                        acc[t] = _mm256_fmadd_ps(w8, _mm256_loadu_ps(X + (int64_t)(t0 + t) * n_in + e), acc[t]);
                }
            }
        }
        for (int t = 0; t < n; t++) y[(int64_t)(t0 + t) * ystride] = hsum256(acc[t]);
    }
}

void fatcat_gemm_iq2_xxs_avx512(const void *W, const float *X, float *Y,
                                int64_t n_out, int64_t n_in, int n_tok)
{
    const uint8_t *w = W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        row_iq2_xxs_nt(w + i * (n_in / 256 * 66), X, n_tok,
                       Y + i, n_out, n_in);
}

void fatcat_gemv_f32_avx512(const float *W, const float *x, float *y,
                            int64_t n_out, int64_t n_in)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n_out; i++)
        y[i] = fatcat_dot_avx512(W + i * n_in, x, n_in);
}
