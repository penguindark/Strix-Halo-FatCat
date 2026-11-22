/* fatcat scalar reference dequant kernels. See include/fatcat/dequant.h.
 *
 * Format semantics follow ggml (validated against gguf-py golden data):
 *  - Q8_0:   32 elems/block: { f16 d; int8 qs[32]; }            v = d*q
 *  - Q6_K:  256 elems/block: { u8 ql[128]; u8 qh[64]; i8 sc[16]; f16 d; }
 *            6-bit q in [-32,31], 16 groups of 16 share d*sc[g]
 *  - IQ2_XXS:256 elems/block: { f16 d; u16 qs[32]; }
 *            per 32-val sub-block: 4 grid idx (8 B each from iq2xxs_grid),
 *            4x7 sign bits via ksigns_iq2xs, 4-bit scale sel in top nibble
 *            of the 4th u16: v = 0.125*d*(2*sel+1)*grid_byte*sign
 *  - IQ1_S:  256 elems/block: { f16 d; u8 qs[32]; u16 qh[8]; }
 *            per 32-val sub-block: dl = d*(2*((qh>>12)&7)+1), delta=±0.125,
 *            4 grid lookups (iq1s_grid, 11-bit idx): v = dl*(grid_byte+delta)
 */
#include <math.h>
#include <string.h>

#include "fatcat/dequant.h"
#include "fatcat/gguf.h"

extern const uint64_t fatcat_iq2xxs_grid[256];
extern const uint64_t fatcat_iq1s_grid[2048];
extern const uint8_t  fatcat_ksigns_iq2xs[128];
extern const uint8_t  fatcat_kmask_iq2xs[8];

/* IEEE-754 half -> float (subnormal-safe via integer assembly). */
float fatcat_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            /* subnormal: normalize */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void dequant_f32(const void *src, float *dst, uint64_t n)
{
    memcpy(dst, src, n * 4);
}

static void dequant_f16(const void *src, float *dst, uint64_t n)
{
    const uint16_t *s = src;
    for (uint64_t i = 0; i < n; i++) dst[i] = fatcat_f16_to_f32(s[i]);
}

static void dequant_q8_0(const void *src, float *dst, uint64_t n)
{
    const uint8_t *p = src;
    for (uint64_t i = 0; i < n; i += 32, p += 34) {
        float d = fatcat_f16_to_f32(*(const uint16_t *)p);
        const int8_t *q = (const int8_t *)(p + 2);
        for (int j = 0; j < 32; j++) dst[i + j] = d * q[j];
    }
}

static void dequant_q6_k(const void *src, float *dst, uint64_t n)
{
    const uint8_t *p = src;
    for (uint64_t i = 0; i < n; i += 256, p += 210) {
        const uint8_t *ql = p;          /* 128 B */
        const uint8_t *qh = p + 128;    /* 64 B  */
        const int8_t  *sc = (const int8_t *)(p + 192); /* 16 B */
        float d = fatcat_f16_to_f32(*(const uint16_t *)(p + 208));
        for (int j = 0; j < 256; j += 128) {
            const uint8_t *qhj = qh + j / 4;
            const uint8_t *qlj = ql + j / 2;
            float *yj = dst + i + j;
            for (int l = 0; l < 32; l++) {
                yj[l +  0] = (int)((qlj[l] & 0xF) | ((qhj[l] & 3) << 4)) - 32;
                yj[l + 32] = (int)((qlj[l + 32] & 0xF) | (((qhj[l] >> 2) & 3) << 4)) - 32;
                yj[l + 64] = (int)((qlj[l] >> 4) | (((qhj[l] >> 4) & 3) << 4)) - 32;
                yj[l + 96] = (int)((qlj[l + 32] >> 4) | (((qhj[l] >> 6) & 3) << 4)) - 32;
            }
            for (int g = 0; g < 8; g++) {
                float ds = d * sc[j / 16 + g];
                for (int l = 0; l < 16; l++)
                    yj[g * 16 + l] *= ds;
            }
        }
    }
}

static void dequant_iq2_xxs(const void *src, float *dst, uint64_t n)
{
    const uint8_t *p = src;
    for (uint64_t i = 0; i < n; i += 256, p += 66) {
        float d = fatcat_f16_to_f32(*(const uint16_t *)p);
        const uint32_t *qs = (const uint32_t *)(p + 2); /* 16 uint32 */
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux0 = qs[2 * ib], aux1 = qs[2 * ib + 1];
            float scale = 0.125f * d * (float)(2 * (aux1 >> 28) + 1);
            for (int l = 0; l < 4; l++) {
                uint8_t gidx = (uint8_t)(aux0 >> (8 * l));
                const uint8_t *grid = (const uint8_t *)&fatcat_iq2xxs_grid[gidx];
                uint8_t signs = fatcat_ksigns_iq2xs[(aux1 >> (7 * l)) & 0x7F];
                for (int j = 0; j < 8; j++) {
                    float sign = (signs & fatcat_kmask_iq2xs[j]) ? -1.0f : 1.0f;
                    dst[i + ib * 32 + l * 8 + j] = scale * grid[j] * sign;
                }
            }
        }
    }
}

/* IQ1_S: 256 elems / 50 B: { f16 d; u8 qs[32]; u16 qh[8]; }
 * Per 32-sub-block ib: dl = d*(2*((qh[ib]>>12)&7)+1),
 * delta = (qh[ib]&0x8000) ? -IQ1S_DELTA : +IQ1S_DELTA,
 * 4 grid lookups: idx = qs[l] | (((qh[ib]>>3l)&7)<<8), v = dl*(grid8[j]+delta). */
static void dequant_iq1_s(const void *src, float *dst, uint64_t n)
{
    const uint8_t *p = src;
    for (uint64_t i = 0; i < n; i += 256, p += 50) {
        float d = fatcat_f16_to_f32(*(const uint16_t *)p);
        const uint8_t  *qs = p + 2;
        const uint16_t *qh = (const uint16_t *)(p + 34);
        for (int ib = 0; ib < 8; ib++) {
            const float dl = d * (float)(2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = (qh[ib] & 0x8000) ? -0.125f : 0.125f;
            for (int l = 0; l < 4; l++) {
                const int8_t *grid = (const int8_t *)&fatcat_iq1s_grid[
                    qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8)];
                float *y = dst + i + ib * 32 + l * 8;
                for (int j = 0; j < 8; j++)
                    y[j] = dl * ((float)grid[j] + delta);
            }
        }
    }
}

bool fatcat_dequant(uint32_t type, const void *src, float *dst, uint64_t n)
{
    switch (type) {
    case FATCAT_GGML_F32:   dequant_f32(src, dst, n); return true;
    case FATCAT_GGML_F16:   dequant_f16(src, dst, n); return true;
    case FATCAT_GGML_Q8_0:  dequant_q8_0(src, dst, n); return true;
    case FATCAT_GGML_Q6_K:  dequant_q6_k(src, dst, n); return true;
    case FATCAT_GGML_IQ2_XXS: dequant_iq2_xxs(src, dst, n); return true;
    case FATCAT_GGML_IQ1_S: dequant_iq1_s(src, dst, n); return true;
    default: return false;
    }
}

uint64_t fatcat_dequant_src_bytes(uint32_t type, uint64_t n)
{
    uint32_t block, bytes;
    if (!fatcat_ggml_type_size(type, &block, &bytes)) return 0;
    if (n % block) return 0;
    return n / block * bytes;
}
