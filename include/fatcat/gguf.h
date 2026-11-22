/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat — GGUF v2/v3 file reader (single-file and split GGUF).
 *
 * mmap-based, bounds-checked parser for the GGUF container format:
 *   magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
 *   n_kv   × { string key, value }
 *   n_tensors × { string name, n_dims u32, dims u64[], type u32, offset u64 }
 *   padding to alignment (general.alignment, default 32)
 *   tensor data (tensor.offset is relative to the aligned data start)
 *
 * Split GGUFs follow llama.cpp semantics: each part is a self-contained
 * GGUF with its own header/KV/tensor table; part 0 ('-00001-of-NNNNN')
 * carries the global metadata plus split.count/split.no KVs, and each part
 * declares its own tensors with offsets relative to its own data start.
 * All parts are mmap'd; a tensor knows its part via fatcat_gguf_tensor.part.
 *
 * All metadata is little-endian on disk; this reader targets little-endian
 * hosts only (Strix Halo). */
#ifndef FATCAT_GGUF_H
#define FATCAT_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FATCAT_GGUF_MAGIC "GGUF"
#define FATCAT_GGUF_MAX_DIMS 4u
#define FATCAT_GGUF_DEFAULT_ALIGNMENT 32u

/* GGUF metadata value types. */
typedef enum fatcat_gguf_value_type {
    FATCAT_GGUF_U8 = 0,  FATCAT_GGUF_I8 = 1,
    FATCAT_GGUF_U16 = 2, FATCAT_GGUF_I16 = 3,
    FATCAT_GGUF_U32 = 4, FATCAT_GGUF_I32 = 5,
    FATCAT_GGUF_F32 = 6, FATCAT_GGUF_BOOL = 7,
    FATCAT_GGUF_STR = 8, FATCAT_GGUF_ARR = 9,
    FATCAT_GGUF_U64 = 10, FATCAT_GGUF_I64 = 11,
    FATCAT_GGUF_F64 = 12,
} fatcat_gguf_value_type;

/* GGML tensor element types (numeric values match ggml.h / GGUF spec). */
typedef enum fatcat_ggml_type {
    FATCAT_GGML_F32 = 0,     FATCAT_GGML_F16 = 1,
    FATCAT_GGML_Q4_0 = 2,    FATCAT_GGML_Q4_1 = 3,
    FATCAT_GGML_Q5_0 = 6,    FATCAT_GGML_Q5_1 = 7,
    FATCAT_GGML_Q8_0 = 8,    FATCAT_GGML_Q8_1 = 9,
    FATCAT_GGML_Q2_K = 10,   FATCAT_GGML_Q3_K = 11,
    FATCAT_GGML_Q4_K = 12,   FATCAT_GGML_Q5_K = 13,
    FATCAT_GGML_Q6_K = 14,   FATCAT_GGML_Q8_K = 15,
    FATCAT_GGML_IQ2_XXS = 16, FATCAT_GGML_IQ2_XS = 17,
    FATCAT_GGML_IQ3_XXS = 18, FATCAT_GGML_IQ1_S = 19,
    FATCAT_GGML_IQ4_NL = 20, FATCAT_GGML_IQ3_S = 21,
    FATCAT_GGML_IQ2_S = 22,  FATCAT_GGML_IQ4_XS = 23,
    FATCAT_GGML_I8 = 24,     FATCAT_GGML_I16 = 25,
    FATCAT_GGML_I32 = 26,    FATCAT_GGML_I64 = 27,
    FATCAT_GGML_F64 = 28,    FATCAT_GGML_IQ1_M = 29,
    FATCAT_GGML_BF16 = 30,   FATCAT_GGML_TQ1_0 = 34,
    FATCAT_GGML_TQ2_0 = 35,  FATCAT_GGML_MXFP4 = 39,
    FATCAT_GGML_NVFP4 = 40,  FATCAT_GGML_Q1_0 = 41,
    FATCAT_GGML_Q2_0 = 42,   FATCAT_GGML_F8_E4M3 = 43,
} fatcat_ggml_type;

/* Borrowed byte string (points into mmap'd data or malloc'd copy). */
typedef struct fatcat_str {
    const char *data;
    size_t len;
} fatcat_str;

/* Metadata value. Arrays store fixed-size elements packed in `elems`;
 * string arrays store fatcat_str[nelem]. */
typedef struct fatcat_gguf_value {
    fatcat_gguf_value_type type;
    union {
        uint8_t u8;   int8_t i8;
        uint16_t u16; int16_t i16;
        uint32_t u32; int32_t i32;
        uint64_t u64; int64_t i64;
        float f32;    double f64;
        bool boolean;
        fatcat_str str;
        struct {
            fatcat_gguf_value_type elem_type;
            uint64_t count;
            void *elems;
        } arr;
    };
} fatcat_gguf_value;

typedef struct fatcat_gguf_kv {
    fatcat_str key;
    fatcat_gguf_value value;
} fatcat_gguf_kv;

typedef struct fatcat_gguf_tensor {
    fatcat_str name;
    uint32_t part;          /* part index containing this tensor's data */
    uint32_t n_dims;
    uint64_t dims[FATCAT_GGUF_MAX_DIMS];
    uint32_t type;          /* fatcat_ggml_type */
    uint64_t offset;        /* relative to the part's aligned data start */
    uint64_t n_bytes;       /* computed from dims and type */
    uint64_t n_elems;
} fatcat_gguf_tensor;

/* Opaque handle. */
typedef struct fatcat_gguf fatcat_gguf;

/* Open a GGUF file. If `path` is part 1 of a split GGUF (metadata KV
 * split.count > 1), sibling parts are opened automatically; all parts must
 * be present. Returns NULL and fills errbuf on failure. */
fatcat_gguf *fatcat_gguf_open(const char *path, char *errbuf, size_t errlen);
void fatcat_gguf_close(fatcat_gguf *g);

/* Header info. */
uint32_t fatcat_gguf_version(const fatcat_gguf *g);
size_t fatcat_gguf_num_parts(const fatcat_gguf *g);
uint64_t fatcat_gguf_data_start(const fatcat_gguf *g);
uint32_t fatcat_gguf_alignment(const fatcat_gguf *g);

/* Metadata KV access. */
size_t fatcat_gguf_num_kv(const fatcat_gguf *g);
const fatcat_gguf_kv *fatcat_gguf_kv_at(const fatcat_gguf *g, size_t idx);
const fatcat_gguf_value *fatcat_gguf_find_kv(const fatcat_gguf *g, const char *key);
/* Convenience getters: return fallback if key missing or wrong type. */
uint64_t fatcat_gguf_kv_u64(const fatcat_gguf *g, const char *key, uint64_t fallback);
fatcat_str fatcat_gguf_kv_str(const fatcat_gguf *g, const char *key, fatcat_str fallback);

/* Tensor access. Tensors are also sorted by name for lookup. */
size_t fatcat_gguf_num_tensors(const fatcat_gguf *g);
const fatcat_gguf_tensor *fatcat_gguf_tensor_at(const fatcat_gguf *g, size_t idx);
const fatcat_gguf_tensor *fatcat_gguf_find_tensor(const fatcat_gguf *g, const char *name);

/* Direct pointer to tensor data inside its part's mapping, or NULL if the
 * part is not mapped or the range is out of bounds. */
const void *fatcat_gguf_tensor_ptr(const fatcat_gguf *g, const fatcat_gguf_tensor *t);

/* Streaming reads for pointers obtained from fatcat_gguf_tensor_ptr():
 * pread() copies bytes into dst without mapping them into process memory,
 * and drop_range() releases the kernel page cache behind [mapped, +bytes). */
int  fatcat_gguf_pread(const fatcat_gguf *g, const void *mapped, void *dst, size_t bytes);

/* Resolve a mapped pointer to its backing (fd, file offset). Returns false
 * if the pointer is not inside a mapped part. The fd remains owned by the
 * gguf handle (do not close). Used by the stream tier to pread slices. */
bool fatcat_gguf_resolve(const fatcat_gguf *g, const void *mapped,
                         int *fd, uint64_t *file_off);
/* Like fatcat_gguf_resolve but prefers the O_DIRECT descriptor (page-cache
 * bypass for stream fills; caller must keep offset/length/buffer 512-aligned
 * or use the buffered fallback). Falls back to the buffered fd if opening
 * O_DIRECT failed (e.g. filesystem without O_DIRECT support). */
bool fatcat_gguf_resolve_direct(const fatcat_gguf *g, const void *mapped,
                                int *fd, uint64_t *file_off);
void fatcat_gguf_drop_range(const fatcat_gguf *g, const void *mapped, size_t bytes);
/* Copy tensor data into dst. dst_len >= t->n_bytes. */
bool fatcat_gguf_tensor_read(const fatcat_gguf *g, const fatcat_gguf_tensor *t,
                             void *dst, size_t dst_len);

/* Quantization table: block size and per-block byte size for a ggml type.
 * Returns false for unknown types. */
bool fatcat_ggml_type_size(uint32_t type, uint32_t *block, uint32_t *bytes);
/* Human-readable name for a ggml type ("Q4_K", "IQ1_S", ...), or NULL. */
const char *fatcat_ggml_type_name(uint32_t type);

#ifdef __cplusplus
}
#endif

#endif /* FATCAT_GGUF_H */
