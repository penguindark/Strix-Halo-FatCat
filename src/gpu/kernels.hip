/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat GPU kernels (HIP, gfx1151). Fused dequantize + GEMV.
 *
 * One thread block per output row; threads split the row's quantization
 * blocks, compute partial dots, and reduce in shared memory. Weights stay
 * quantized on device (memory-bound kernels dequantize in registers).
 *
 * Compiled by hipcc; exposes a C API (fatcat_gpu.h implements the wrapper).
 * Semantics mirror src/model/dequant.c (bit-layouts from ggml). */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK_K 256

extern "C" {
extern const uint64_t fatcat_iq2xxs_grid[256];
extern const uint64_t fatcat_iq1s_grid[2048];
extern const uint8_t  fatcat_ksigns_iq2xs[128];
extern const uint8_t  fatcat_kmask_iq2xs[8];
}

/* device copies of the codebook tables */
static __device__ int8_t  d_iq1s_grid[2048][8];
static __device__ uint8_t d_iq2xxs_grid[256][8];
static __device__ uint8_t d_ksigns[128];
static __device__ uint8_t d_kmask[8];

static inline __device__ float f16d(uint16_t h) { return __half2float(*(const __half *)&h); }

/* block-wide sum reduce (warpSize-aware: 64 lanes on RDNA) */
static inline __device__ float block_sum(float v, float *sh) {
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x / warpSize;
    const int nw   = (blockDim.x + warpSize - 1) / warpSize;
    for (int o = warpSize / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(~0ull, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nw) ? sh[lane] : 0.0f;
        for (int o = warpSize / 2; o > 0; o >>= 1)
            v += __shfl_down_sync(~0ull, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

/* ---------------- kernels: one block per output row ----------------
 * Perf notes (gfx1151, measured via tools/gpu-kbench):
 *  - x is read with float4 loads: thread g owns 8/16/32 CONTIGUOUS elements
 *    so a warp's access window is contiguous (the old per-element scalar
 *    loop with lane-strided bases touched one cache line per lane per op).
 *  - codebook rows are loaded as one u64 instead of 8 byte loads.
 *  - x/y must be 16-byte aligned for the vector kernels (checked in the
 *    launcher; callers fall back to CPU otherwise). Staging is aligned. */

__global__ void k_gemv_f32(const float *__restrict__ W, const float *__restrict__ x,
                           float *__restrict__ y, int n_in) {
    __shared__ float sh[8];
    const float4 *row = (const float4 *)(W + (size_t)blockIdx.x * n_in);
    const float4 *xv = (const float4 *)x;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_in / 4; i += blockDim.x) {
        const float4 w = row[i], v = xv[i];
        s += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

/* Q8_0 gemv/gemm bit-identity note: the batched k_gemm_q8_0 hoists the
 * quant decode out of the token loop, which let the compiler contract
 * mul/add into FMAs differently than in k_gemv_q8_0 (observed 1-2 ULP
 * drift). An explicit __fmaf_rn-chain pin was tried and MISCOMPILED by
 * this ROCm build (catastrophic logits) -> do NOT reintroduce it. The
 * contraction difference is instead removed build-wide with
 * -ffp-contract=off on kernels.hip (see CMakeLists.txt), so both
 * kernels evaluate exactly as written below, in identical order. */
__global__ void k_gemv_q8_0(const uint8_t *__restrict__ W, const float *__restrict__ x,
                            float *__restrict__ y, int n_in) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 32 * 34);
    float s = 0.0f;
    /* one thread per 32-elem block: 8 float4 x loads + 16 u16 q loads */
    for (int b = threadIdx.x; b < n_in / 32; b += blockDim.x) {
        const uint8_t *blk = row + (size_t)b * 34;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t *q16 = (const uint16_t *)(blk + 2);
        const float4 *xv = (const float4 *)(x + (size_t)b * 32);
        float acc = 0.0f;
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const float4 f = xv[v];
            const uint32_t q2 = (uint32_t)q16[2 * v] | ((uint32_t)q16[2 * v + 1] << 16);
            acc += (float)(int)(int8_t)(q2) * f.x
                 + (float)(int)(int8_t)(q2 >> 8) * f.y
                 + (float)(int)(int8_t)(q2 >> 16) * f.z
                 + (float)(int)(int8_t)(q2 >> 24) * f.w;
        }
        s += d * acc;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

/* D5e batched per-head gemv: for item i, y[i*y_stride + r] =
 * dot(dequant_q8_0(W + i*w_stride, row r), x + i*x_stride).
 * Per-row body identical to k_gemv_q8_0 => bit-identical to n_items
 * separate fatcat_gpu_gemv calls; one launch covers all items.
 * Used for the MLA absorbed-V loop (64 heads x 128x512). */
__global__ void k_gemv_q8_0_batched(const uint8_t *__restrict__ W,
                                    long long w_stride,
                                    const float *__restrict__ x,
                                    long long x_stride,
                                    float *__restrict__ y, long long y_stride,
                                    int n_out, int n_in) {
    const int item = blockIdx.x / n_out;
    const int row  = blockIdx.x % n_out;
    __shared__ float sh[8];
    const uint8_t *r0 = W + (size_t)item * (size_t)w_stride;
    const uint8_t *rw = r0 + (size_t)row * (n_in / 32 * 34);
    const float *xr = x + (size_t)item * (size_t)x_stride;
    float s = 0.0f;
    for (int b = threadIdx.x; b < n_in / 32; b += blockDim.x) {
        const uint8_t *blk = rw + (size_t)b * 34;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t *q16 = (const uint16_t *)(blk + 2);
        const float4 *xv = (const float4 *)(xr + (size_t)b * 32);
        float acc = 0.0f;
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const float4 f = xv[v];
            const uint32_t q2 = (uint32_t)q16[2 * v] | ((uint32_t)q16[2 * v + 1] << 16);
            acc += (float)(int)(int8_t)(q2) * f.x
                 + (float)(int)(int8_t)(q2 >> 8) * f.y
                 + (float)(int)(int8_t)(q2 >> 16) * f.z
                 + (float)(int)(int8_t)(q2 >> 24) * f.w;
        }
        s += d * acc;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[(size_t)item * (size_t)y_stride + row] = s;
}

/* D5e MLA k-absorption: q_abs[row] = dot(dequant_q8_0(W row), x + item*x_stride)
 * with item = row / rows_per_item. One thread per row. The arithmetic mirrors
 * the CPU absorb path (dequant_q8_0 + fatcat_dot_avx512) bit-for-bit: four
 * 16-lane fmaf accumulators over 64-elem steps, ((a0+a1)+(a2+a3)), then the
 * exact _mm512_reduce_add_ps tree (16->8->4->2->1). Verified bit-identical to
 * the CPU pattern by a standalone scalar mirror (0/256000 mismatches).
 * row_stride is the tensor's row_bytes: NoPE layers store wk_b rows over the
 * full qk (pad_rope), so stride != n_in/32*34 there. Requires n_in % 16 == 0.
 * Queued on the default stream; caller syncs. */
__global__ void k_absorb_q8_0(const uint8_t *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ y, int rows_per_item,
                              int n_in, long long x_stride,
                              long long row_stride, int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const uint8_t *rw = W + (size_t)row * (size_t)row_stride;
    const float *xr = x + (size_t)(row / rows_per_item) * (size_t)x_stride;
    float a0[16], a1[16], a2[16], a3[16];
#pragma unroll
    for (int j = 0; j < 16; j++) a0[j] = a1[j] = a2[j] = a3[j] = 0.0f;
    int i = 0;
    for (; i + 64 <= n_in; i += 64) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int off = i + j, b = off >> 5;
            const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
            a0[j] = fmaf(d * (float)(int8_t)rw[b * 34 + 2 + (off & 31)], xr[i + j], a0[j]);
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int off = i + 16 + j, b = off >> 5;
            const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
            a1[j] = fmaf(d * (float)(int8_t)rw[b * 34 + 2 + (off & 31)], xr[i + 16 + j], a1[j]);
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int off = i + 32 + j, b = off >> 5;
            const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
            a2[j] = fmaf(d * (float)(int8_t)rw[b * 34 + 2 + (off & 31)], xr[i + 32 + j], a2[j]);
        }
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int off = i + 48 + j, b = off >> 5;
            const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
            a3[j] = fmaf(d * (float)(int8_t)rw[b * 34 + 2 + (off & 31)], xr[i + 48 + j], a3[j]);
        }
    }
    for (; i + 16 <= n_in; i += 16)
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int off = i + j, b = off >> 5;
            const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
            a0[j] = fmaf(d * (float)(int8_t)rw[b * 34 + 2 + (off & 31)], xr[i + j], a0[j]);
        }
    float w[16], s8[8], s4[4];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = (a0[j] + a1[j]) + (a2[j] + a3[j]);
#pragma unroll
    for (int j = 0; j < 8; j++) s8[j] = w[j] + w[j + 8];
#pragma unroll
    for (int j = 0; j < 4; j++) s4[j] = s8[j] + s8[j + 4];
    float s = (s4[0] + s4[2]) + (s4[1] + s4[3]);
    for (; i < n_in; i++) {
        const int b = i >> 5;
        const float d = f16d(*(const uint16_t *)(rw + (size_t)b * 34));
        s += d * (float)(int8_t)rw[b * 34 + 2 + (i & 31)] * xr[i];
    }
    y[row] = s;
}

__global__ void k_gemv_q6_k(const uint8_t *__restrict__ W, const float *__restrict__ x,
                            float *__restrict__ y, int n_in) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 210);
    float s = 0.0f;
    /* one thread per 16-element scale group; x via 4 float4 loads */
    for (int g = threadIdx.x; g < n_in / 16; g += blockDim.x) {
        int b = g / 16;                       /* 256-elem block */
        int e = (g % 16) * 16;                /* elem within block */
        const uint8_t *blk = row + (size_t)b * 210;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        const float d = f16d(*(const uint16_t *)(blk + 208));
        const float ds = d * (float)sc[g % 16];
        float acc = 0.0f;
#pragma unroll
        for (int v = 0; v < 4; v++) {
            const float4 f = ((const float4 *)(x + b * 256 + e))[v];
            float q[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int el = e + v * 4 + k;       /* 0..255 */
                int half = el >> 7, r = el & 127;
                int l = r & 31, m = r >> 5;
                int qli = half * 64 + l + ((m == 1 || m == 3) ? 32 : 0);
                int qhi = half * 32 + l;
                int nib = (m >= 2) ? (ql[qli] >> 4) : (ql[qli] & 0xF);
                int bits = (qh[qhi] >> (2 * m)) & 3;
                q[k] = (float)((nib | (bits << 4)) - 32);
            }
            acc += q[0] * f.x + q[1] * f.y + q[2] * f.z + q[3] * f.w;
        }
        s += ds * acc;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

__global__ void k_gemv_iq1_s(const uint8_t *__restrict__ W, const float *__restrict__ x,
                             float *__restrict__ y, int n_in) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 50);
    float s = 0.0f;
    /* one thread per 32-elem sub-block (4 grid entries): 8 per 256-block.
     * Layout (mirrors dequant.c): { f16 d; u8 qs[32]; u16 qh[8] }, per
     * sub-block ib: dl = d*(2*((qh[ib]>>12)&7)+1),
     * delta = (qh[ib]&0x8000)?-1/8:+1/8, idx_l = qs[ib*4+l] | (((qh[ib]>>3l)&7)<<8).
     * The header (d, qh, dl, delta) is computed once per 32 elems and the
     * four grid rows load independently (4x ILP); x arrives as 8 float4. */
    for (int u = threadIdx.x; u < n_in / 32; u += blockDim.x) {
        int b = u >> 3, ib = u & 7;
        const uint8_t *blk = row + (size_t)b * 50;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t qh = ((const uint16_t *)(blk + 34))[ib];
        const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        const float4 *xv = (const float4 *)(x + (size_t)u * 32);
        float acc = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int idx = blk[2 + ib * 4 + l] | (((qh >> (3 * l)) & 7) << 8);
            const uint64_t gv = ((const uint64_t *)d_iq1s_grid)[idx];
            const float4 f0 = xv[2 * l], f1 = xv[2 * l + 1];
            acc += ((float)(int)(int8_t)(gv) + delta) * f0.x;
            acc += ((float)(int)(int8_t)(gv >> 8) + delta) * f0.y;
            acc += ((float)(int)(int8_t)(gv >> 16) + delta) * f0.z;
            acc += ((float)(int)(int8_t)(gv >> 24) + delta) * f0.w;
            acc += ((float)(int)(int8_t)(gv >> 32) + delta) * f1.x;
            acc += ((float)(int)(int8_t)(gv >> 40) + delta) * f1.y;
            acc += ((float)(int)(int8_t)(gv >> 48) + delta) * f1.z;
            acc += ((float)(int)(int8_t)(gv >> 56) + delta) * f1.w;
        }
        s += dl * acc;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

__global__ void k_gemv_iq2_xxs(const uint8_t *__restrict__ W, const float *__restrict__ x,
                               float *__restrict__ y, int n_in) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 66);
    float s = 0.0f;
    /* one thread per 32-elem sub-block: 8 per 256-block; x via 8 float4,
     * each grid row as one u64 */
    for (int u = threadIdx.x; u < n_in / 32; u += blockDim.x) {
        int b = u >> 3, ib = u & 7;
        const uint8_t *blk = row + (size_t)b * 66;
        const float d = f16d(*(const uint16_t *)blk);
        /* qs lives at offset 2: read as u16 pairs (u32 would be misaligned).
         * aux0/aux1 are the u32 pair at byte offset 2 + 8*ib. */
        const uint16_t *qs16 = (const uint16_t *)(blk + 2);
        const uint32_t aux0 = (uint32_t)qs16[4 * ib] | ((uint32_t)qs16[4 * ib + 1] << 16);
        const uint32_t aux1 = (uint32_t)qs16[4 * ib + 2] | ((uint32_t)qs16[4 * ib + 3] << 16);
        const float scale = 0.125f * d * (float)(2 * (aux1 >> 28) + 1);
        const float4 *xv = (const float4 *)(x + (size_t)b * 256 + ib * 32);
        float acc = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint64_t gv = ((const uint64_t *)d_iq2xxs_grid)[(aux0 >> (8 * l)) & 0xFF];
            const uint8_t signs = d_ksigns[(aux1 >> (7 * l)) & 0x7F];
            const float4 f0 = xv[2 * l], f1 = xv[2 * l + 1];
            const float xj[8] = { f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w };
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float sg = (signs & d_kmask[j]) ? -1.0f : 1.0f;
                acc += (float)(unsigned)(uint8_t)(gv >> (8 * j)) * sg * xj[j];
            }
        }
        s += scale * acc;
    }
    s = block_sum(s, sh);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

/* ---------------- batched GEMM (D1): one weight decode, N token dots ----------------
 * Same per-token op order as the GEMV kernels above (same thread mapping,
 * same u loop, same block_sum), so each token's output is BIT-identical to
 * the gemv result; the IQ codebook decode runs once per row and is reused
 * across nt tokens (nt <= 8 per launch; the host tiles larger batches).
 * X is nt rows of n_in f32, Y is nt rows of n_out (Y[t*n_out + row]). */
__global__ void k_gemm_iq1_s(const uint8_t *__restrict__ W, const float *__restrict__ X,
                             float *__restrict__ Y, int n_in, int n_out, int nt) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 50);
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.0f;
    for (int u = threadIdx.x; u < n_in / 32; u += blockDim.x) {
        int b = u >> 3, ib = u & 7;
        const uint8_t *blk = row + (size_t)b * 50;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t qh = ((const uint16_t *)(blk + 34))[ib];
        const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        float acc[8];
#pragma unroll
        for (int t = 0; t < 8; t++) acc[t] = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int idx = blk[2 + ib * 4 + l] | (((qh >> (3 * l)) & 7) << 8);
            const uint64_t gv = ((const uint64_t *)d_iq1s_grid)[idx];
            const float g0 = (float)(int)(int8_t)(gv) + delta;
            const float g1 = (float)(int)(int8_t)(gv >> 8) + delta;
            const float g2 = (float)(int)(int8_t)(gv >> 16) + delta;
            const float g3 = (float)(int)(int8_t)(gv >> 24) + delta;
            const float g4 = (float)(int)(int8_t)(gv >> 32) + delta;
            const float g5 = (float)(int)(int8_t)(gv >> 40) + delta;
            const float g6 = (float)(int)(int8_t)(gv >> 48) + delta;
            const float g7 = (float)(int)(int8_t)(gv >> 56) + delta;
            for (int t = 0; t < nt; t++) {
                const float4 *xv = (const float4 *)(X + (size_t)t * n_in + (size_t)u * 32);
                const float4 f0 = xv[2 * l], f1 = xv[2 * l + 1];
                acc[t] += g0 * f0.x;
                acc[t] += g1 * f0.y;
                acc[t] += g2 * f0.z;
                acc[t] += g3 * f0.w;
                acc[t] += g4 * f1.x;
                acc[t] += g5 * f1.y;
                acc[t] += g6 * f1.z;
                acc[t] += g7 * f1.w;
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) s[t] += dl * acc[t];
    }
    for (int t = 0; t < nt; t++) {
        float v = block_sum(s[t], sh);
        if (threadIdx.x == 0) Y[(size_t)t * n_out + blockIdx.x] = v;
    }
}

__global__ void k_gemm_iq2_xxs(const uint8_t *__restrict__ W, const float *__restrict__ X,
                               float *__restrict__ Y, int n_in, int n_out, int nt) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 66);
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.0f;
    for (int u = threadIdx.x; u < n_in / 32; u += blockDim.x) {
        int b = u >> 3, ib = u & 7;
        const uint8_t *blk = row + (size_t)b * 66;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t *qs16 = (const uint16_t *)(blk + 2);
        const uint32_t aux0 = (uint32_t)qs16[4 * ib] | ((uint32_t)qs16[4 * ib + 1] << 16);
        const uint32_t aux1 = (uint32_t)qs16[4 * ib + 2] | ((uint32_t)qs16[4 * ib + 3] << 16);
        const float scale = 0.125f * d * (float)(2 * (aux1 >> 28) + 1);
        float acc[8];
#pragma unroll
        for (int t = 0; t < 8; t++) acc[t] = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint64_t gv = ((const uint64_t *)d_iq2xxs_grid)[(aux0 >> (8 * l)) & 0xFF];
            const uint8_t signs = d_ksigns[(aux1 >> (7 * l)) & 0x7F];
            /* decoded, sign-applied weight values (reused for every token) */
            float wv[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float sg = (signs & d_kmask[j]) ? -1.0f : 1.0f;
                wv[j] = (float)(unsigned)(uint8_t)(gv >> (8 * j)) * sg;
            }
            for (int t = 0; t < nt; t++) {
                const float4 *xv = (const float4 *)(X + (size_t)t * n_in + (size_t)b * 256 + ib * 32);
                const float4 f0 = xv[2 * l], f1 = xv[2 * l + 1];
                const float xj[8] = { f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w };
#pragma unroll
                for (int j = 0; j < 8; j++) acc[t] += wv[j] * xj[j];
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) s[t] += scale * acc[t];
    }
    for (int t = 0; t < nt; t++) {
        float v = block_sum(s[t], sh);
        if (threadIdx.x == 0) Y[(size_t)t * n_out + blockIdx.x] = v;
    }
}

__global__ void k_gemm_q8_0(const uint8_t *__restrict__ W, const float *__restrict__ X,
                            float *__restrict__ Y, int n_in, int n_out, int nt) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 32 * 34);
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.0f;
    for (int b = threadIdx.x; b < n_in / 32; b += blockDim.x) {
        const uint8_t *blk = row + (size_t)b * 34;
        const float d = f16d(*(const uint16_t *)blk);
        const uint16_t *q16 = (const uint16_t *)(blk + 2);
        float acc[8];
#pragma unroll
        for (int t = 0; t < 8; t++) acc[t] = 0.0f;
#pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t q2 = (uint32_t)q16[2 * v] | ((uint32_t)q16[2 * v + 1] << 16);
            const float g0 = (float)(int)(int8_t)(q2);
            const float g1 = (float)(int)(int8_t)(q2 >> 8);
            const float g2 = (float)(int)(int8_t)(q2 >> 16);
            const float g3 = (float)(int)(int8_t)(q2 >> 24);
            for (int t = 0; t < nt; t++) {
                const float4 f = ((const float4 *)(X + (size_t)t * n_in + (size_t)b * 32))[v];
                /* same written order as k_gemv_q8_0; -ffp-contract=off
                 * (CMake) guarantees identical evaluation on both sides */
                acc[t] += g0 * f.x + g1 * f.y + g2 * f.z + g3 * f.w;
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) s[t] += d * acc[t];
    }
    for (int t = 0; t < nt; t++) {
        float v = block_sum(s[t], sh);
        if (threadIdx.x == 0) Y[(size_t)t * n_out + blockIdx.x] = v;
    }
}

__global__ void k_gemm_q6_k(const uint8_t *__restrict__ W, const float *__restrict__ X,
                            float *__restrict__ Y, int n_in, int n_out, int nt) {
    __shared__ float sh[8];
    const uint8_t *row = W + (size_t)blockIdx.x * (n_in / 256 * 210);
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; t++) s[t] = 0.0f;
    for (int g = threadIdx.x; g < n_in / 16; g += blockDim.x) {
        int b = g / 16;
        int e = (g % 16) * 16;
        const uint8_t *blk = row + (size_t)b * 210;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        const float d = f16d(*(const uint16_t *)(blk + 208));
        const float ds = d * (float)sc[g % 16];
        float acc[8];
#pragma unroll
        for (int t = 0; t < 8; t++) acc[t] = 0.0f;
#pragma unroll
        for (int v = 0; v < 4; v++) {
            float q[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int el = e + v * 4 + k;
                int half = el >> 7, r = el & 127;
                int l = r & 31, m = r >> 5;
                int qli = half * 64 + l + ((m == 1 || m == 3) ? 32 : 0);
                int qhi = half * 32 + l;
                int nib = (m >= 2) ? (ql[qli] >> 4) : (ql[qli] & 0xF);
                int bits = (qh[qhi] >> (2 * m)) & 3;
                q[k] = (float)((nib | (bits << 4)) - 32);
            }
            for (int t = 0; t < nt; t++) {
                const float4 f = ((const float4 *)(X + (size_t)t * n_in + b * 256 + e))[v];
                acc[t] += q[0] * f.x + q[1] * f.y + q[2] * f.z + q[3] * f.w;
            }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) s[t] += ds * acc[t];
    }
    for (int t = 0; t < nt; t++) {
        float v = block_sum(s[t], sh);
        if (threadIdx.x == 0) Y[(size_t)t * n_out + blockIdx.x] = v;
    }
}

/* ---------------- host-side C API ---------------- */

extern "C" {

typedef struct {
    uint32_t type;
    void *dev;
} ghandle;

static bool g_ok = false;
static bool g_unified = false;
static char g_devstr[128] = "";
static float *g_x = NULL, *g_y = NULL;        /* x: device mem; y: mapped host */
static float *g_ydev = NULL;                 /* device-side address of g_y */
/* batched-GEMM staging (D1): X device buffer + mapped-host Y, grow-on-demand */
static float *g_gx = NULL;     static size_t g_gx_cap = 0;
static float *g_gy = NULL, *g_gydev = NULL;  static size_t g_gy_cap = 0;
/* async x staging ring (see fatcat_gpu_xstage_impl) */
#define XRING 8
#define XRB_CAP (32768 * 4)
static char *g_xring[XRING]; static size_t g_xring_cap[XRING]; static int g_xring_i = 0;
#define G_XCAP (24576 * 4)
#define G_YCAP (24576 * 4)

/* staging slots for streamed experts */
typedef struct {
    void *dev; uint64_t tag; uint32_t type; size_t bytes; uint64_t last;
} slot_t;
static slot_t *g_slots = NULL;
static int g_nslots = 0;
static size_t g_slot_cap = 0;
static uint64_t g_use = 0;
static uint64_t g_copies = 0, g_copy_bytes = 0, g_stage_hits = 0;
static uint64_t g_upload_fails = 0, g_fail_bytes = 0;
static uint64_t g_gemm_calls = 0, g_gemm_fails = 0;   /* D1 gemm path */
static ghandle *g_handles = NULL; static int n_handles = 0, cap_handles = 0;

#define GGML_F32 0
#define GGML_F16 1
#define GGML_Q8_0 8
#define GGML_Q6_K 14
#define GGML_IQ2_XXS 16
#define GGML_IQ1_S 19

static bool hcheck(hipError_t e, const char *what, char *eb, size_t el) {
    if (e == hipSuccess) return true;
    snprintf(eb, el, "%s: %s", what, hipGetErrorString(e));
    return false;
}

int fatcat_gpu_init_impl(uint32_t slots, char *eb, size_t el) {
    /* D5b: spin-wait scheduling. The decode path syncs on the GPU hundreds
     * of times per token (dense gemvs); with the default blocking/yield
     * wait each sync pays scheduler latency. Spinning burns a core but cuts
     * sync latency to ~us. FATCAT_GPU_SPIN=0 restores blocking waits.
     * Must be set before the context is created (first HIP call). */
    { const char *s = getenv("FATCAT_GPU_SPIN");
      if (!s || atoi(s)) hipSetDeviceFlags(hipDeviceScheduleSpin); }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        snprintf(eb, el, "no HIP device"); return -1;
    }
    if (hipSetDevice(0) != hipSuccess) { snprintf(eb, el, "hipSetDevice"); return -1; }
    int ncu = 0; size_t mem = 0;
    hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, 0);
    hipDeviceTotalMem(&mem, 0);
    snprintf(g_devstr, sizeof(g_devstr), "gfx1151 (%d CUs, %.0f GB unified)", ncu, mem / 1e9);

    /* codebook tables into device module scope */
    if (hipMemcpyToSymbol(d_iq1s_grid, fatcat_iq1s_grid, sizeof(d_iq1s_grid)) != hipSuccess ||
        hipMemcpyToSymbol(d_iq2xxs_grid, fatcat_iq2xxs_grid, sizeof(d_iq2xxs_grid)) != hipSuccess ||
        hipMemcpyToSymbol(d_ksigns, fatcat_ksigns_iq2xs, 128) != hipSuccess ||
        hipMemcpyToSymbol(d_kmask, fatcat_kmask_iq2xs, 8) != hipSuccess) {
        snprintf(eb, el, "table upload failed"); return -1;
    }
    if (!hcheck(hipMalloc(&g_x, G_XCAP), "malloc x", eb, el)) return -1;
    if (!hcheck(hipHostMalloc((void **)&g_y, G_YCAP, hipHostMallocMapped), "host-malloc y", eb, el)) return -1;
    /* device-visible address for the mapped y staging (identity on an APU,
     * but ask rather than assume). x is plain device memory now: it is
     * re-read by every output row, and mapped-host reads proved slower. */
    if (hipHostGetDevicePointer((void **)&g_ydev, g_y, 0) != hipSuccess) {
        snprintf(eb, el, "host map failed"); return -1;
    }
    {   /* unified addressing probe: device ptr == host ptr? */
        void *tp = NULL, *td = NULL;
        g_unified = hipHostMalloc(&tp, 4096, hipHostMallocMapped) == hipSuccess &&
                    hipHostGetDevicePointer(&td, tp, 0) == hipSuccess && td == tp;
        if (tp) hipHostFree(tp);
    }

    g_nslots = (int)(slots ? slots : 16);
    g_slot_cap = 10 * 1024 * 1024;   /* >= IQ2_XXS expert (9.73 MB) */
    g_slots = (slot_t *)calloc(g_nslots, sizeof(slot_t));
    for (int i = 0; i < g_nslots; i++)
        if (hipMalloc(&g_slots[i].dev, g_slot_cap) != hipSuccess) {
            snprintf(eb, el, "slot alloc failed"); return -1;
        }
    g_ok = true;
    return 0;
}

void fatcat_gpu_shutdown_impl(void) {
    if (!g_ok) return;
    for (int i = 0; i < n_handles; i++) hipFree(g_handles[i].dev);
    free(g_handles); g_handles = NULL; n_handles = cap_handles = 0;
    for (int i = 0; i < g_nslots; i++) if (g_slots[i].dev) hipFree(g_slots[i].dev);
    free(g_slots); g_slots = NULL; g_nslots = 0;
    if (g_x) hipFree(g_x);
    if (g_y) hipHostFree(g_y);
    g_x = g_y = g_ydev = NULL;
    if (g_gx) (void)hipFree(g_gx);
    if (g_gy) (void)hipHostFree(g_gy);
    g_gx = g_gy = g_gydev = NULL; g_gx_cap = g_gy_cap = 0;
    for (int i = 0; i < XRING; i++) { if (g_xring[i]) hipFree(g_xring[i]); g_xring[i] = NULL; g_xring_cap[i] = 0; }
    g_ok = false;
}

void fatcat_gpu_stats_impl(uint64_t *copies, uint64_t *bytes, uint64_t *hits) {
    *copies = g_copies; *bytes = g_copy_bytes; *hits = g_stage_hits;
}

uint64_t fatcat_gpu_upload_fails_impl(void) { return g_upload_fails; }

const char *fatcat_gpu_devstr_impl(void) { return g_devstr; }

void *fatcat_gpu_upload_impl(uint32_t type, const void *host, size_t bytes) {
    if (!g_ok) return NULL;
    void *dev = NULL;
    if (hipMalloc(&dev, bytes) != hipSuccess) {
        g_upload_fails++; g_fail_bytes += bytes;
        fprintf(stderr, "gpu_upload: hipMalloc(%zu B) failed, CPU fallback\n", bytes);
        return NULL;
    }
    if (hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(dev); g_upload_fails++; g_fail_bytes += bytes;
        fprintf(stderr, "gpu_upload: hipMemcpy(%zu B) failed, CPU fallback\n", bytes);
        return NULL;
    }
    g_copies++; g_copy_bytes += bytes;
    ghandle *h = (ghandle *)malloc(sizeof(ghandle));
    h->type = type; h->dev = dev;
    if (n_handles == cap_handles) {
        cap_handles = cap_handles ? cap_handles * 2 : 256;
        g_handles = (ghandle *)realloc(g_handles, cap_handles * sizeof(ghandle));
    }
    g_handles[n_handles++] = *h;
    return h;
}

void *fatcat_gpu_stage_impl(uint32_t type, const void *host, size_t bytes, uint64_t tag) {
    if (!g_ok || bytes > g_slot_cap) {
        fprintf(stderr, "gpu_stage: reject (ok=%d bytes=%zu cap=%zu)\n", g_ok, bytes, g_slot_cap);
        return NULL;
    }
    /* hit? */
    for (int i = 0; i < g_nslots; i++) {
        if (g_slots[i].tag == tag && g_slots[i].type == type && g_slots[i].bytes == bytes) {
            g_slots[i].last = ++g_use; g_stage_hits++;
            ghandle *h = (ghandle *)malloc(sizeof(ghandle));
            h->type = type; h->dev = g_slots[i].dev;
            return h;
        }
    }
    /* LRU evict */
    int victim = 0; uint64_t oldest = ~0ull;
    for (int i = 0; i < g_nslots; i++)
        if (g_slots[i].last < oldest) { oldest = g_slots[i].last; victim = i; }
    if (hipMemcpy(g_slots[victim].dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess)
        return NULL;
    g_copies++; g_copy_bytes += bytes;
    g_slots[victim].tag = tag; g_slots[victim].type = type;
    g_slots[victim].bytes = bytes; g_slots[victim].last = ++g_use;
    ghandle *h = (ghandle *)malloc(sizeof(ghandle));
    h->type = type; h->dev = g_slots[victim].dev;
    return h;
}

void fatcat_gpu_free_handle(void *handle) { free(handle); }

/* ---- P4 zero-copy ---- */

void *fatcat_gpu_host_alloc_impl(size_t bytes) {
    if (!g_ok) return NULL;
    void *p = NULL;
    if (hipHostMalloc(&p, bytes, hipHostMallocMapped) != hipSuccess) return NULL;
    void *d = NULL;
    if (hipHostGetDevicePointer(&d, p, 0) != hipSuccess) { hipHostFree(p); return NULL; }
    if (d != p) {           /* not unified addressing; caller needs identity */
        hipHostFree(p); return NULL;
    }
    return p;
}

void fatcat_gpu_host_free_impl(void *p) { if (p) hipHostFree(p); }

int fatcat_gpu_unified_impl(void) {
    if (!g_ok) return 0;
    return g_unified;
}

void *fatcat_gpu_wrap_impl(void *dev_ptr, uint32_t type) {
    if (!g_ok || !g_unified) return NULL;
    ghandle *h = (ghandle *)malloc(sizeof(ghandle));
    h->type = type; h->dev = dev_ptr;
    return h;
}

/* Launch the fused dequant-GEMV for one weight matrix. x/y must be
 * device-accessible pointers (device memory or mapped host memory).
 * Does NOT synchronize: callers either sync via fatcat_gpu_wait_impl or
 * rely on stream ordering (kernels on the default stream serialize). */
static int launch_gemv(const void *wdev, uint32_t type, const float *x, float *y,
                       long long n_out, long long n_in) {
    /* vector kernels need 16-byte aligned x/y (checked here; callers fall
     * back to CPU on failure). Staging buffers are always aligned. */
    if ((((uintptr_t)x | (uintptr_t)y) & 15) != 0) return -1;
    (void)hipGetLastError();  /* clear any sticky error before launch */
    const uint8_t *w = (const uint8_t *)wdev;
    switch (type) {
    case GGML_F32:
        if (n_in & 3) return -1;
        k_gemv_f32<<<(int)n_out, 256>>>((const float *)wdev, x, y, (int)n_in); break;
    case GGML_Q8_0:
        k_gemv_q8_0<<<(int)n_out, 192>>>(w, x, y, (int)n_in); break;
    case GGML_Q6_K:
        k_gemv_q6_k<<<(int)n_out, 384>>>(w, x, y, (int)n_in); break;
    case GGML_IQ1_S:
        k_gemv_iq1_s<<<(int)n_out, 192>>>(w, x, y, (int)n_in); break;
    case GGML_IQ2_XXS:
        k_gemv_iq2_xxs<<<(int)n_out, 192>>>(w, x, y, (int)n_in); break;
    default: fprintf(stderr, "gpu_gemv: no kernel for type %u\n", type); return -1;
    }
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) { fprintf(stderr, "gpu_gemv: launch: %s\n", hipGetErrorString(e)); return -1; }
    return 0;
}

/* Sync-staging path: x/y are arbitrary host pointers; copy through the
 * persistent mapped staging buffers, launch, wait, copy back. */
int fatcat_gpu_gemv_impl(void *wh, uint32_t type, const float *x, float *y,
                         long long n_out, long long n_in) {
    if (!g_ok || !wh) return -1;
    const ghandle *h = (const ghandle *)wh;
    if (n_in * 4 > G_XCAP || n_out * 4 > G_YCAP) {
        fprintf(stderr, "gpu_gemv: caps (n_in=%lld n_out=%lld)\n", n_in, n_out); return -1; }
    /* x goes to device memory (re-read by every row: mapped-host reads are
     * slower), y is written once per row into mapped host memory. */
    if (hipMemcpy(g_x, x, (size_t)n_in * 4, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "gpu_gemv: memcpy x failed\n"); return -1; }
    if (launch_gemv(h->dev, type, g_x, g_ydev, n_out, n_in)) return -1;
    hipError_t e2 = hipDeviceSynchronize();
    if (e2 != hipSuccess) { fprintf(stderr, "gpu_gemv: sync: %s\n", hipGetErrorString(e2)); return -1; }
    memcpy(y, g_y, (size_t)n_out * 4);
    return 0;
}

/* Batched GEMM (D1): X is n_tok rows of n_in f32 (host), Y is n_tok rows of
 * n_out (host). Sync path: stage X to device, launch 8-token tiles, wait,
 * copy Y back. Bit-identical to per-token fatcat_gpu_gemv (see kernels).
 * Returns 0 on success; non-zero -> caller falls back to CPU. */
#define GEMM_CAP ((size_t)512 << 20)

int fatcat_gpu_gemm_impl(void *wh, uint32_t type, const float *X, float *Y,
                         long long n_out, long long n_in, int n_tok) {
    g_gemm_calls++;
    if (!g_ok || !wh || n_tok < 1) { g_gemm_fails++; return -1; }
    if (type != GGML_IQ1_S && type != GGML_IQ2_XXS &&
        type != GGML_Q8_0 && type != GGML_Q6_K) { g_gemm_fails++; return -1; }
    if ((n_in % 4) || (n_out % 4)) { g_gemm_fails++; return -1; }   /* float4 alignment */
    const size_t xb = (size_t)n_tok * n_in * 4, yb = (size_t)n_tok * n_out * 4;
    if (xb > GEMM_CAP || yb > GEMM_CAP) { g_gemm_fails++; return -1; }
    if (xb > g_gx_cap) {
        if (g_gx) (void)hipFree(g_gx);
        g_gx = NULL; g_gx_cap = 0;
        if (hipMalloc((void **)&g_gx, xb) != hipSuccess) { g_gemm_fails++; return -1; }
        g_gx_cap = xb;
    }
    if (yb > g_gy_cap) {
        if (g_gy) (void)hipHostFree(g_gy);
        g_gy = NULL; g_gydev = NULL; g_gy_cap = 0;
        if (hipHostMalloc((void **)&g_gy, yb, hipHostMallocMapped) != hipSuccess) { g_gemm_fails++; return -1; }
        if (hipHostGetDevicePointer((void **)&g_gydev, g_gy, 0) != hipSuccess) { g_gemm_fails++; return -1; }
        g_gy_cap = yb;
    }
    if (hipMemcpy(g_gx, X, xb, hipMemcpyHostToDevice) != hipSuccess) { g_gemm_fails++; return -1; }
    const ghandle *h = (const ghandle *)wh;
    (void)hipGetLastError();
    for (int t0 = 0; t0 < n_tok; t0 += 8) {
        const int nt = n_tok - t0 < 8 ? n_tok - t0 : 8;
        if (type == GGML_IQ1_S)
            k_gemm_iq1_s<<<(int)n_out, 192>>>((const uint8_t *)h->dev,
                g_gx + (size_t)t0 * n_in, g_gydev + (size_t)t0 * n_out,
                (int)n_in, (int)n_out, nt);
        else if (type == GGML_IQ2_XXS)
            k_gemm_iq2_xxs<<<(int)n_out, 192>>>((const uint8_t *)h->dev,
                g_gx + (size_t)t0 * n_in, g_gydev + (size_t)t0 * n_out,
                (int)n_in, (int)n_out, nt);
        else if (type == GGML_Q8_0)
            k_gemm_q8_0<<<(int)n_out, 192>>>((const uint8_t *)h->dev,
                g_gx + (size_t)t0 * n_in, g_gydev + (size_t)t0 * n_out,
                (int)n_in, (int)n_out, nt);
        else
            k_gemm_q6_k<<<(int)n_out, 384>>>((const uint8_t *)h->dev,
                g_gx + (size_t)t0 * n_in, g_gydev + (size_t)t0 * n_out,
                (int)n_in, (int)n_out, nt);
        if (hipGetLastError() != hipSuccess) {
            fprintf(stderr, "gpu_gemm: launch: %s\n", hipGetErrorString(hipGetLastError()));
            g_gemm_fails++;
            return -1;
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "gpu_gemm: sync failed\n"); g_gemm_fails++; return -1;
    }
    memcpy(Y, g_gy, yb);
    return 0;
}

void fatcat_gpu_gemm_stats_impl(uint64_t *calls, uint64_t *fails) {
    *calls = g_gemm_calls; *fails = g_gemm_fails;
}

/* Async path: x/y are already device-accessible (forward scratch). Launch
 * only; the result is visible after fatcat_gpu_wait_impl(). */
int fatcat_gpu_gemv_dev_impl(void *wh, uint32_t type, const float *x, float *y,
                             long long n_out, long long n_in) {
    if (!g_ok || !wh) return -1;
    if (n_out <= 0 || n_out > INT_MAX || n_in <= 0 || n_in > INT_MAX) return -1;
    const ghandle *h = (const ghandle *)wh;
    return launch_gemv(h->dev, type, x, y, n_out, n_in);
}

/* D5e: batched per-head Q8_0 gemv, one launch (see k_gemv_q8_0_batched).
 * Stream-ordered, no sync. Returns 0 if queued. */
int fatcat_gpu_gemv_q8_0_batched_impl(void *wh, long long w_stride,
                                      const float *x, long long x_stride,
                                      float *y, long long y_stride,
                                      long long n_out, long long n_in,
                                      long long n_items) {
    if (!g_ok || !wh) return -1;
    const ghandle *h = (const ghandle *)wh;
    if (h->type != GGML_Q8_0) return -1;
    if (n_out <= 0 || n_in <= 0 || n_in % 32 || n_items <= 0) return -1;
    if (n_items * n_out > INT_MAX) return -1;
    if ((((uintptr_t)x | (uintptr_t)y) & 15) != 0) return -1;
    (void)hipGetLastError();
    k_gemv_q8_0_batched<<<(int)(n_items * n_out), 192>>>(
        (const uint8_t *)h->dev, w_stride, x, x_stride, y, y_stride,
        (int)n_out, (int)n_in);
    return 0;
}

/* D5e: MLA k-absorption gemv (see k_absorb_q8_0). rows = n_items *
 * rows_per_item; x item i is at x + i*x_stride; weight row r at w + r*row_stride
 * (tensor row_bytes; NoPE layers pad rows to the full qk). Stream-ordered. */
int fatcat_gpu_absorb_q8_0_impl(void *wh, const float *x, float *y,
                                long long rows_per_item, long long n_items,
                                long long n_in, long long x_stride,
                                long long row_stride) {
    if (!g_ok || !wh) return -1;
    const ghandle *h = (const ghandle *)wh;
    if (h->type != GGML_Q8_0) return -1;
    if (rows_per_item <= 0 || n_items <= 0 || n_in <= 0 || n_in % 16) return -1;
    if (row_stride < n_in / 32 * 34) return -1;
    const long long rows = rows_per_item * n_items;
    if (rows > INT_MAX) return -1;
    if ((((uintptr_t)x | (uintptr_t)y) & 15) != 0) return -1;
    (void)hipGetLastError();
    const int thr = 128;
    k_absorb_q8_0<<<(int)((rows + thr - 1) / thr), thr>>>(
        (const uint8_t *)h->dev, x, y, (int)rows_per_item, (int)n_in,
        x_stride, row_stride, (int)rows);
    return 0;
}

/* ---- async x staging ring --------------------------------------------
 * Kernels read x much faster from device memory than from mapped host
 * memory (x is re-read by every output row; mapped reads bypass the
 * cache). xstage() copies x into the next ring slot with a STREAM-ORDERED
 * async memcpy: the copy runs after all previously queued kernels and
 * before the next launch, so it is safe to interleave stage/launch calls
 * on the default stream. Returns the device pointer, or NULL on failure. */

void *fatcat_gpu_xstage_impl(const void *x, size_t bytes) {
    if (!g_ok || !bytes || bytes > XRB_CAP) return NULL;
    int i = g_xring_i; g_xring_i = (g_xring_i + 1) % XRING;
    if (g_xring_cap[i] < bytes) {
        if (g_xring[i]) hipFree(g_xring[i]);
        g_xring[i] = NULL; g_xring_cap[i] = 0;
        if (hipMalloc((void **)&g_xring[i], bytes) != hipSuccess) return NULL;
        g_xring_cap[i] = bytes;
    }
    if (hipMemcpyAsync(g_xring[i], x, bytes, hipMemcpyHostToDevice, (hipStream_t)0)
        != hipSuccess) return NULL;
    return g_xring[i];
}

/* Block until all launched kernels are done. */
int fatcat_gpu_wait_impl(void) {
    if (!g_ok) return -1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : -1;
}

/* Sync path with x in device memory but y in mapped host memory. */
int fatcat_gpu_gemv_devx_impl(void *wh, uint32_t type, const float *xhost,
                              float *y, long long n_out, long long n_in) {
    if (!g_ok || !wh) return -1;
    const ghandle *h = (const ghandle *)wh;
    if (n_in * 4 > G_XCAP || n_out * 4 > G_YCAP) return -1;
    if (hipMemcpy(g_x, xhost, (size_t)n_in * 4, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (launch_gemv(h->dev, type, g_x, g_ydev, n_out, n_in) != 0) return -1;
    if (hipDeviceSynchronize() != hipSuccess) return -1;
    memcpy(y, g_y, (size_t)n_out * 4);
    return 0;
}

/* ---- microbenchmark: practical device read bandwidth ------------------
 * Allocates `bytes` of device memory, streams it with grid-stride float4
 * reads, returns GB/s. Measures the ceiling any GEMV kernel could reach. */
__global__ void k_readbw(const float4 *p, size_t n4, float *out) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0.0f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = p[i];
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 1234.567f) *out = s;   /* never true: keeps the loads alive */
}

double fatcat_gpu_readbw_impl(size_t bytes, int iters) {
    if (!g_ok || bytes < 16) return -1.0;
    void *d = NULL; float *out = NULL;
    if (hipMalloc(&d, bytes) != hipSuccess) return -1.0;
    if (hipMalloc((void **)&out, 4) != hipSuccess) { hipFree(d); return -1.0; }
    hipMemset(d, 0x3f, bytes);
    struct timespec t0, t1;
    k_readbw<<<2048, 256>>>((const float4 *)d, bytes / 16, out);
    hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        k_readbw<<<2048, 256>>>((const float4 *)d, bytes / 16, out);
    hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    hipFree(d); hipFree(out);
    return sec > 0 ? (double)bytes * iters / sec / 1e9 : -1.0;
}

} /* extern C */
