/* fatcat — scalar reference dequantization kernels.
 *
 * Correctness-first (scalar C, no SIMD) implementations of the GGML quant
 * formats used by FatCat. Validated against gguf-py golden data
 * (tests/test_dequant.c). SIMD/AVX-512 versions come later (P2 perf work).
 *
 * Supported: F32, F16, Q8_0, Q6_K, IQ2_XXS, IQ1_S. */
#ifndef FATCAT_DEQUANT_H
#define FATCAT_DEQUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Dequantize n contiguous elements from src (quant-packed) into dst[n] (f32).
 * n must be a multiple of the type's block size. Returns false on unsupported
 * type or bad n. */
bool fatcat_dequant(uint32_t ggml_type, const void *src, float *dst, uint64_t n);

/* Byte size of n quantized elements (0 if type unsupported / n misaligned). */
uint64_t fatcat_dequant_src_bytes(uint32_t ggml_type, uint64_t n);

/* IEEE half -> float. */
float fatcat_f16_to_f32(uint16_t h);

#endif /* FATCAT_DEQUANT_H */
