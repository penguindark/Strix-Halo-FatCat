/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat GGUF reader implementation. See include/fatcat/gguf.h.
 *
 * Split GGUF semantics follow llama.cpp: each part is a self-contained GGUF
 * file with its own header, KV metadata and tensor info table. Part 0 holds
 * the global model metadata (and split.count/split.no KVs); parts >0 hold
 * their share of the tensor tables and data. A tensor's offset is relative
 * to the data start of the part that declares it. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fatcat/gguf.h"

#define GGUF_VERSION_MIN 2u
#define GGUF_VERSION_MAX 3u
/* Sanity limits (mirror gguf-py / llama.cpp guards). */
#define GGUF_MAX_STRING_LEN (1024ull * 1024ull * 1024ull)
#define GGUF_MAX_ARRAY_LEN  (1024ull * 1024ull * 1024ull)
#define GGUF_MAX_TENSORS    100000000ull
#define GGUF_MAX_KV         10000000ull
#define GGUF_MAX_PARTS      64u

/* One mmap'd file part. */
typedef struct gguf_part {
    char *path;
    int fd;
    int fd_direct;        /* O_RDONLY|O_DIRECT, lazily opened (-1 if failed) */
    void *map;
    size_t size;
    uint64_t data_start;  /* absolute offset of tensor data within this part */
} gguf_part;

struct fatcat_gguf {
    gguf_part *parts;
    size_t n_parts;

    uint32_t version;
    uint32_t alignment;

    fatcat_gguf_kv *kvs;   /* global metadata (from part 0) */
    size_t n_kv;

    fatcat_gguf_tensor *tensors;     /* file order, all parts */
    uint32_t *by_name;               /* indices sorted by name for bsearch */
    size_t n_tensors;
};

/* ---------- streaming reads (bypass the process page tables) ---------- */

static const gguf_part *find_part_at(const fatcat_gguf *g, const void *p,
                                     uint64_t *file_off)
{
    for (size_t i = 0; i < g->n_parts; i++) {
        uintptr_t s = (uintptr_t)g->parts[i].map, e = s + g->parts[i].size;
        if (g->parts[i].map && (uintptr_t)p >= s && (uintptr_t)p < e) {
            *file_off = (uintptr_t)p - s;   /* mmap was whole-file at offset 0 */
            return &g->parts[i];
        }
    }
    return NULL;
}

/* Read bytes starting at a mapped tensor pointer via pread() into dst.
 * Pages land in the kernel page cache (reclaimable), not in process RSS. */
int fatcat_gguf_pread(const fatcat_gguf *g, const void *mapped, void *dst, size_t bytes)
{
    uint64_t off = 0;
    const gguf_part *pt = find_part_at(g, mapped, &off);
    if (!pt || pt->fd < 0) return -1;
    size_t done = 0;
    while (done < bytes) {
        ssize_t r = pread(pt->fd, (char *)dst + done, bytes - done, (off_t)(off + done));
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        done += (size_t)r;
    }
    return 0;
}

bool fatcat_gguf_resolve(const fatcat_gguf *g, const void *mapped,
                         int *fd, uint64_t *file_off)
{
    uint64_t off = 0;
    const gguf_part *pt = find_part_at(g, mapped, &off);
    if (!pt || pt->fd < 0) return false;
    *fd = pt->fd;
    *file_off = off;
    return true;
}

/* Like fatcat_gguf_resolve but returns the O_DIRECT descriptor for the
 * stream path: reads bypass the page cache, so streamed expert bytes never
 * evict the hot mmap'd weights (D5f: refault traffic saturated the NVMe).
 * Lazily opens the direct fd; falls back to the buffered fd on failure.
 * The caller must ensure 512-byte alignment of offset/length/buffer. */
bool fatcat_gguf_resolve_direct(const fatcat_gguf *g, const void *mapped,
                                int *fd, uint64_t *file_off)
{
    uint64_t off = 0;
    gguf_part *pt = (gguf_part *)find_part_at(g, mapped, &off);
    if (!pt || pt->fd < 0) return false;
    if (pt->fd_direct == 0) {          /* not yet attempted */
        int f = open(pt->path, O_RDONLY | O_DIRECT);
        pt->fd_direct = f >= 0 ? f : -1;
    }
    *fd = pt->fd_direct >= 0 ? pt->fd_direct : pt->fd;
    *file_off = off;
    return true;
}

/* Ask the kernel to drop cached pages for the file range backing
 * [mapped, mapped+bytes). */
void fatcat_gguf_drop_range(const fatcat_gguf *g, const void *mapped, size_t bytes)
{
    uint64_t off = 0;
    const gguf_part *pt = find_part_at(g, mapped, &off);
    if (!pt || pt->fd < 0) return;
    uint64_t s = off & ~4095ull;
    uint64_t e = (off + bytes + 4095ull) & ~4095ull;
    posix_fadvise(pt->fd, (off_t)s, (off_t)(e - s), POSIX_FADV_DONTNEED);
}

/* ---------- bounds-checked cursor over one mapped part ---------- */

typedef struct cursor {
    const uint8_t *p;
    const uint8_t *end;
    bool err;
} cursor;

static void cerr(cursor *c) { if (!c->err) c->err = true; }

static const uint8_t *cneed(cursor *c, size_t n)
{
    if (c->err || (size_t)(c->end - c->p) < n) { cerr(c); return NULL; }
    return c->p;
}

static bool rd_u32(cursor *c, uint32_t *out)
{
    const uint8_t *p = cneed(c, 4);
    if (!p) return false;
    memcpy(out, p, 4);
    c->p += 4;
    return true;
}

static bool rd_u64(cursor *c, uint64_t *out)
{
    const uint8_t *p = cneed(c, 8);
    if (!p) return false;
    memcpy(out, p, 8);
    c->p += 8;
    return true;
}

static bool rd_bytes(cursor *c, size_t n, const uint8_t **out)
{
    const uint8_t *p = cneed(c, n);
    if (!p) return false;
    *out = p;
    c->p += n;
    return true;
}

/* Read a GGUF string (u64 len + bytes) into a malloc'd, NUL-terminated copy. */
static bool rd_str(cursor *c, fatcat_str *out)
{
    uint64_t len;
    if (!rd_u64(c, &len)) return false;
    if (len > GGUF_MAX_STRING_LEN) { cerr(c); return false; }
    const uint8_t *bytes;
    if (!rd_bytes(c, (size_t)len, &bytes)) return false;
    char *buf = malloc((size_t)len + 1);
    if (!buf) { cerr(c); return false; }
    memcpy(buf, bytes, (size_t)len);
    buf[len] = '\0';
    out->data = buf;
    out->len = (size_t)len;
    return true;
}

static size_t value_scalar_size(fatcat_gguf_value_type t)
{
    switch (t) {
    case FATCAT_GGUF_U8: case FATCAT_GGUF_I8: case FATCAT_GGUF_BOOL: return 1;
    case FATCAT_GGUF_U16: case FATCAT_GGUF_I16: return 2;
    case FATCAT_GGUF_U32: case FATCAT_GGUF_I32: case FATCAT_GGUF_F32: return 4;
    case FATCAT_GGUF_U64: case FATCAT_GGUF_I64: case FATCAT_GGUF_F64: return 8;
    default: return 0; /* string/array: not fixed-size */
    }
}

static bool rd_value(cursor *c, fatcat_gguf_value *v);

static void free_value(fatcat_gguf_value *v);

static bool rd_array(cursor *c, fatcat_gguf_value *v)
{
    uint32_t raw_elem;
    uint64_t count;
    if (!rd_u32(c, &raw_elem)) return false;
    if (!rd_u64(c, &count)) return false;
    if (count > GGUF_MAX_ARRAY_LEN) { cerr(c); return false; }
    if (raw_elem > FATCAT_GGUF_F64 || raw_elem == FATCAT_GGUF_ARR) {
        cerr(c);
        return false;
    }
    v->type = FATCAT_GGUF_ARR;
    v->arr.elem_type = (fatcat_gguf_value_type)raw_elem;
    v->arr.count = count;
    v->arr.elems = NULL;

    if (raw_elem == FATCAT_GGUF_STR) {
        fatcat_str *strs = count ? calloc((size_t)count, sizeof *strs) : NULL;
        if (count && !strs) { cerr(c); return false; }
        v->arr.elems = strs;
        for (uint64_t i = 0; i < count; i++) {
            if (!rd_str(c, &strs[i])) {
                for (uint64_t j = 0; j < i; j++) free((void *)strs[j].data);
                free(strs);
                v->arr.elems = NULL;
                return false;
            }
        }
        return true;
    }

    size_t esz = value_scalar_size((fatcat_gguf_value_type)raw_elem);
    if (esz == 0) { cerr(c); return false; }
    size_t total = (size_t)count * esz;
    if ((uint64_t)(c->end - c->p) < total) { cerr(c); return false; }
    void *elems = malloc(total ? total : 1);
    if (!elems) { cerr(c); return false; }
    const uint8_t *p;
    if (!rd_bytes(c, total, &p)) { free(elems); v->arr.elems = NULL; return false; }
    memcpy(elems, p, total);
    v->arr.elems = elems;
    return true;
}

static bool rd_value(cursor *c, fatcat_gguf_value *v)
{
    uint32_t raw;
    if (!rd_u32(c, &raw)) return false;
    if (raw > FATCAT_GGUF_F64) { cerr(c); return false; }
    fatcat_gguf_value_type t = (fatcat_gguf_value_type)raw;
    v->type = t;
    switch (t) {
    case FATCAT_GGUF_U8: { const uint8_t *p; if (!rd_bytes(c,1,&p)) return false; memcpy(&v->u8,p,1); return true; }
    case FATCAT_GGUF_I8: { const uint8_t *p; if (!rd_bytes(c,1,&p)) return false; memcpy(&v->i8,p,1); return true; }
    case FATCAT_GGUF_U16:{ const uint8_t *p; if (!rd_bytes(c,2,&p)) return false; memcpy(&v->u16,p,2); return true; }
    case FATCAT_GGUF_I16:{ const uint8_t *p; if (!rd_bytes(c,2,&p)) return false; memcpy(&v->i16,p,2); return true; }
    case FATCAT_GGUF_U32: return rd_u32(c, &v->u32);
    case FATCAT_GGUF_I32: return rd_u32(c, &v->u32);  /* union type-pun */
    case FATCAT_GGUF_F32: return rd_u32(c, &v->u32);  /* union type-pun */
    case FATCAT_GGUF_BOOL: {
        uint8_t b;
        const uint8_t *p;
        if (!rd_bytes(c, 1, &p)) return false;
        b = *p;
        v->boolean = b != 0;
        return true;
    }
    case FATCAT_GGUF_U64: return rd_u64(c, &v->u64);
    case FATCAT_GGUF_I64: return rd_u64(c, &v->u64);  /* union type-pun */
    case FATCAT_GGUF_F64: return rd_u64(c, &v->u64);  /* union type-pun */
    case FATCAT_GGUF_STR: return rd_str(c, &v->str);
    case FATCAT_GGUF_ARR: return rd_array(c, v);
    }
    cerr(c);
    return false;
}

/* ---------- quantization table (values from ggml.h / GGUF spec) ---------- */

typedef struct ggml_type_info {
    const char *name;
    uint32_t block;
    uint32_t bytes;
} ggml_type_info;

#define QK_K 256u
static const ggml_type_info type_table[] = {
    [FATCAT_GGML_F32]     = { "F32", 1, 4 },
    [FATCAT_GGML_F16]     = { "F16", 1, 2 },
    [FATCAT_GGML_Q4_0]    = { "Q4_0", 32, 2 + 16 },
    [FATCAT_GGML_Q4_1]    = { "Q4_1", 32, 2 + 2 + 16 },
    [FATCAT_GGML_Q5_0]    = { "Q5_0", 32, 2 + 4 + 16 },
    [FATCAT_GGML_Q5_1]    = { "Q5_1", 32, 2 + 2 + 4 + 16 },
    [FATCAT_GGML_Q8_0]    = { "Q8_0", 32, 2 + 32 },
    [FATCAT_GGML_Q8_1]    = { "Q8_1", 32, 2 + 2 + 32 },
    [FATCAT_GGML_Q2_K]    = { "Q2_K", QK_K, 2 + 2 + QK_K / 16 + QK_K / 4 },
    [FATCAT_GGML_Q3_K]    = { "Q3_K", QK_K, 2 + QK_K / 4 + QK_K / 8 + 12 },
    [FATCAT_GGML_Q4_K]    = { "Q4_K", QK_K, 2 + 2 + QK_K / 2 + 12 },
    [FATCAT_GGML_Q5_K]    = { "Q5_K", QK_K, 2 + 2 + QK_K / 2 + QK_K / 8 + 12 },
    [FATCAT_GGML_Q6_K]    = { "Q6_K", QK_K, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16 },
    [FATCAT_GGML_Q8_K]    = { "Q8_K", QK_K, 4 + QK_K + QK_K / 8 },
    [FATCAT_GGML_IQ2_XXS] = { "IQ2_XXS", QK_K, 2 + QK_K / 4 },
    [FATCAT_GGML_IQ2_XS]  = { "IQ2_XS", QK_K, 2 + QK_K / 4 + QK_K / 32 },
    [FATCAT_GGML_IQ3_XXS] = { "IQ3_XXS", QK_K, 2 + QK_K / 4 + QK_K / 8 },
    [FATCAT_GGML_IQ1_S]   = { "IQ1_S", QK_K, 2 + QK_K / 8 + QK_K / 16 },
    [FATCAT_GGML_IQ4_NL]  = { "IQ4_NL", 32, 2 + 16 },
    [FATCAT_GGML_IQ3_S]   = { "IQ3_S", QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4 },
    [FATCAT_GGML_IQ2_S]   = { "IQ2_S", QK_K, 2 + QK_K / 4 + QK_K / 16 },
    [FATCAT_GGML_IQ4_XS]  = { "IQ4_XS", QK_K, 2 + 2 + QK_K / 2 + QK_K / 64 },
    [FATCAT_GGML_I8]      = { "I8", 1, 1 },
    [FATCAT_GGML_I16]     = { "I16", 1, 2 },
    [FATCAT_GGML_I32]     = { "I32", 1, 4 },
    [FATCAT_GGML_I64]     = { "I64", 1, 8 },
    [FATCAT_GGML_F64]     = { "F64", 1, 8 },
    [FATCAT_GGML_IQ1_M]   = { "IQ1_M", QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32 },
    [FATCAT_GGML_BF16]    = { "BF16", 1, 2 },
    [FATCAT_GGML_TQ1_0]   = { "TQ1_0", QK_K, 2 + 4 * 13 },
    [FATCAT_GGML_TQ2_0]   = { "TQ2_0", QK_K, 2 + 64 },
    [FATCAT_GGML_MXFP4]   = { "MXFP4", 32, 1 + 16 },
    [FATCAT_GGML_NVFP4]   = { "NVFP4", 64, 4 + 32 },
    [FATCAT_GGML_Q1_0]    = { "Q1_0", 128, 2 + 16 },
    [FATCAT_GGML_Q2_0]    = { "Q2_0", 64, 2 + 16 },
    [FATCAT_GGML_F8_E4M3] = { "F8_E4M3", 1, 1 },
};
#define TYPE_TABLE_SIZE (sizeof type_table / sizeof type_table[0])

bool fatcat_ggml_type_size(uint32_t type, uint32_t *block, uint32_t *bytes)
{
    if (type >= TYPE_TABLE_SIZE || type_table[type].name == NULL)
        return false;
    *block = type_table[type].block;
    *bytes = type_table[type].bytes;
    return true;
}

const char *fatcat_ggml_type_name(uint32_t type)
{
    if (type >= TYPE_TABLE_SIZE || type_table[type].name == NULL)
        return NULL;
    return type_table[type].name;
}

/* ---------- split GGUF part discovery ---------- */

/* Match "-NNNNN-of-MMMMM.gguf" at the end of path. */
static bool split_suffix(const char *path, char *stem, size_t stemlen,
                         int *index, int *total)
{
    size_t len = strlen(path);
    /* suffix "-NNNNN-of-MMMMM.gguf" is exactly 20 chars */
    if (len < 20 || strcmp(path + len - 5, ".gguf") != 0)
        return false;
    const char *tail = path + len - 20;
    if (tail[0] != '-' || tail[6] != '-' ||
        strncmp(tail + 6, "-of-", 4) != 0 || tail[15] != '.')
        return false;
    char ibuf[6] = {0}, buf[6] = {0};
    memcpy(ibuf, tail + 1, 5);
    memcpy(buf, tail + 10, 5);
    for (int i = 0; i < 5; i++)
        if (ibuf[i] < '0' || ibuf[i] > '9' || buf[i] < '0' || buf[i] > '9')
            return false;
    *index = atoi(ibuf);
    *total = atoi(buf);
    size_t stem_len = (size_t)(tail - path);
    if (stem_len + 16 > stemlen)
        return false;
    memcpy(stem, path, stem_len);
    stem[stem_len] = '\0';
    return true;
}

static bool map_part(gguf_part *part, bool required, char *errbuf, size_t errlen)
{
    part->fd = open(part->path, O_RDONLY);
    if (part->fd < 0) {
        if (required) {
            snprintf(errbuf, errlen, "cannot open part '%s'", part->path);
            return false;
        }
        return true; /* tolerated until this part's data is needed */
    }
    struct stat st;
    if (fstat(part->fd, &st) != 0) {
        snprintf(errbuf, errlen, "cannot stat '%s'", part->path);
        close(part->fd);
        part->fd = -1;
        return false;
    }
    part->size = (size_t)st.st_size;
    part->map = mmap(NULL, part->size, PROT_READ, MAP_PRIVATE, part->fd, 0);
    if (part->map == MAP_FAILED) {
        snprintf(errbuf, errlen, "mmap failed for '%s'", part->path);
        close(part->fd);
        part->fd = -1;
        part->size = 0;
        return false;
    }
    return true;
}

/* ---------- per-part header/KV/tensor-table parsing ---------- */

/* Parse one part's GGUF header + KV + tensor info table.
 * - part 0: KV goes to g->kvs (global metadata).
 * - part >0: KV is parsed and discarded (validated only).
 * Tensors are appended to g->tensors with their part tag.
 * Sets part->data_start. */
static bool parse_part(fatcat_gguf *g, size_t part_idx,
                       char *errbuf, size_t errlen)
{
    gguf_part *part = &g->parts[part_idx];
    if (!part->map) {
        snprintf(errbuf, errlen, "part %zu ('%s') not available",
                 part_idx, part->path);
        return false;
    }
    cursor c = { part->map, (uint8_t *)part->map + part->size, false };

    const uint8_t *magic;
    if (!rd_bytes(&c, 4, &magic) ||
        memcmp(magic, FATCAT_GGUF_MAGIC, 4) != 0) {
        snprintf(errbuf, errlen, "part %zu: not a GGUF (bad magic)", part_idx);
        return false;
    }
    uint32_t version;
    if (!rd_u32(&c, &version) ||
        version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        snprintf(errbuf, errlen, "part %zu: unsupported GGUF version", part_idx);
        return false;
    }
    if (part_idx == 0) g->version = version;

    uint64_t n_tensors, n_kv;
    if (!rd_u64(&c, &n_tensors) || !rd_u64(&c, &n_kv)) {
        snprintf(errbuf, errlen, "part %zu: truncated header", part_idx);
        return false;
    }
    if (n_tensors > GGUF_MAX_TENSORS || n_kv > GGUF_MAX_KV) {
        snprintf(errbuf, errlen, "part %zu: implausible counts", part_idx);
        return false;
    }

    /* KV metadata. */
    fatcat_gguf_kv *kvs = calloc((size_t)n_kv, sizeof *kvs);
    if (!kvs && n_kv) { snprintf(errbuf, errlen, "out of memory"); return false; }
    for (uint64_t i = 0; i < n_kv; i++) {
        if (!rd_str(&c, &kvs[i].key)) {
            snprintf(errbuf, errlen, "part %zu: truncated KV #%zu", part_idx, (size_t)i);
            goto kv_fail;
        }
        if (!rd_value(&c, &kvs[i].value)) {
            snprintf(errbuf, errlen, "part %zu: bad value for KV '%s'",
                     part_idx, kvs[i].key.data);
            goto kv_fail;
        }
    }

    /* Capture this part's alignment before KV ownership is resolved. */
    uint32_t part_align = FATCAT_GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t k = 0; k < n_kv; k++) {
        if (strcmp(kvs[k].key.data, "general.alignment") == 0 &&
            kvs[k].value.type == FATCAT_GGUF_U32)
            part_align = kvs[k].value.u32;
    }

    if (part_idx == 0) {
        g->kvs = kvs;
        g->n_kv = (size_t)n_kv;
        g->alignment = part_align;
    } else {
        for (uint64_t i = 0; i < n_kv; i++) free_value(&kvs[i].value);
        free(kvs);
    }

    /* Tensor info records. */
    if (g->n_tensors + n_tensors > GGUF_MAX_TENSORS) {
        snprintf(errbuf, errlen, "part %zu: tensor count overflow", part_idx);
        return false;
    }
    g->tensors = realloc(g->tensors,
                         (g->n_tensors + n_tensors) * sizeof *g->tensors);
    if (!g->tensors) { snprintf(errbuf, errlen, "out of memory"); return false; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        fatcat_gguf_tensor *t = &g->tensors[g->n_tensors + i];
        memset(t, 0, sizeof *t);
        t->part = (uint32_t)part_idx;
        if (!rd_str(&c, &t->name)) {
            snprintf(errbuf, errlen, "part %zu: truncated tensor info #%zu",
                     part_idx, (size_t)i);
            return false;
        }
        if (!rd_u32(&c, &t->n_dims) || t->n_dims > FATCAT_GGUF_MAX_DIMS) {
            snprintf(errbuf, errlen, "tensor '%s': bad n_dims", t->name.data);
            return false;
        }
        t->n_elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!rd_u64(&c, &t->dims[d])) {
                snprintf(errbuf, errlen, "tensor '%s': truncated dims", t->name.data);
                return false;
            }
            t->n_elems *= t->dims[d];
        }
        for (uint32_t d = t->n_dims; d < FATCAT_GGUF_MAX_DIMS; d++)
            t->dims[d] = 1;
        if (!rd_u32(&c, &t->type) || !rd_u64(&c, &t->offset)) {
            snprintf(errbuf, errlen, "tensor '%s': truncated record", t->name.data);
            return false;
        }
        uint32_t block, bytes;
        if (!fatcat_ggml_type_size(t->type, &block, &bytes)) {
            snprintf(errbuf, errlen, "tensor '%s': unknown ggml type %u",
                     t->name.data, t->type);
            return false;
        }
        if (t->n_elems % block != 0) {
            snprintf(errbuf, errlen, "tensor '%s': %" PRIu64
                     " elems not divisible by block %u",
                     t->name.data, t->n_elems, block);
            return false;
        }
        t->n_bytes = t->n_elems / block * bytes;
    }
    g->n_tensors += (size_t)n_tensors;

    /* Alignment and this part's data start. */
    uint32_t align = part_align;
    if (align == 0 || (align & (align - 1)) != 0) {
        snprintf(errbuf, errlen, "part %zu: invalid alignment %u", part_idx, align);
        return false;
    }
    uint64_t hdr = (uint64_t)(c.p - (const uint8_t *)part->map);
    part->data_start = (hdr + align - 1) / align * align;
    return true;

kv_fail:
    for (uint64_t i = 0; i < n_kv; i++) free_value(&kvs[i].value);
    free(kvs);
    return false;
}

/* ---------- open / close ---------- */

/* Active handle during qsort in fatcat_gguf_open (single-threaded parse). */
static const fatcat_gguf *g_sort_ctx;

static int cmp_tensor_name(const void *pa, const void *pb)
{
    const fatcat_gguf *g = g_sort_ctx;
    const fatcat_gguf_tensor *a = &g->tensors[*(const uint32_t *)pa];
    const fatcat_gguf_tensor *b = &g->tensors[*(const uint32_t *)pb];
    size_t n = a->name.len < b->name.len ? a->name.len : b->name.len;
    int r = memcmp(a->name.data, b->name.data, n);
    if (r != 0) return r;
    return (int)a->name.len - (int)b->name.len;
}

static uint64_t kv_u64_raw(const fatcat_gguf_kv *kvs, size_t n_kv,
                           const char *key, uint64_t fallback)
{
    size_t klen = strlen(key);
    for (size_t i = 0; i < n_kv; i++) {
        if (kvs[i].key.len == klen && memcmp(kvs[i].key.data, key, klen) == 0) {
            switch (kvs[i].value.type) {
            case FATCAT_GGUF_U8:  return kvs[i].value.u8;
            case FATCAT_GGUF_U16: return kvs[i].value.u16;
            case FATCAT_GGUF_U32: return kvs[i].value.u32;
            case FATCAT_GGUF_U64: return kvs[i].value.u64;
            default: return fallback;
            }
        }
    }
    return fallback;
}

fatcat_gguf *fatcat_gguf_open(const char *path, char *errbuf, size_t errlen)
{
    errbuf[0] = '\0';

    fatcat_gguf *g = calloc(1, sizeof *g);
    if (!g) { snprintf(errbuf, errlen, "out of memory"); return NULL; }
    g->alignment = FATCAT_GGUF_DEFAULT_ALIGNMENT;

    /* Open part 0 (or the single file). */
    g->parts = calloc(1, sizeof *g->parts);
    if (!g->parts) { snprintf(errbuf, errlen, "out of memory"); goto fail; }
    g->n_parts = 1;
    g->parts[0].fd = -1;
    g->parts[0].path = strdup(path);
    if (!g->parts[0].path) { snprintf(errbuf, errlen, "out of memory"); goto fail; }
    if (!map_part(&g->parts[0], true, errbuf, errlen)) goto fail;
    if (!parse_part(g, 0, errbuf, errlen)) goto fail;

    /* Split GGUF? (split.count KV in part 0) */
    uint64_t split_count = kv_u64_raw(g->kvs, g->n_kv, "split.count", 1);
    if (split_count > 1) {
        if (split_count > GGUF_MAX_PARTS) {
            snprintf(errbuf, errlen, "implausible split.count %" PRIu64, split_count);
            goto fail;
        }
        char stem[4096];
        int index = 1, total = 1;
        if (!split_suffix(path, stem, sizeof stem, &index, &total) ||
            (index != 0 && index != 1)) {
            snprintf(errbuf, errlen,
                     "split model but filename is not '-NNNNN-of-MMMMM.gguf'");
            goto fail;
        }
        if ((uint64_t)total != split_count) {
            snprintf(errbuf, errlen, "filename says %d parts, metadata says %" PRIu64,
                     total, split_count);
            goto fail;
        }
        gguf_part *parts = realloc(g->parts, (size_t)split_count * sizeof *parts);
        if (!parts) { snprintf(errbuf, errlen, "out of memory"); goto fail; }
        g->parts = parts;
        for (uint64_t i = 1; i < split_count; i++) {
            gguf_part *p = &g->parts[i];
            memset(p, 0, sizeof *p);
            p->fd = -1;
            size_t plen = (size_t)snprintf(NULL, 0, "%s-%05d-of-%05" PRIu64 ".gguf",
                                           stem, (int)i + 1, split_count) + 1;
            p->path = malloc(plen);
            if (!p->path) { snprintf(errbuf, errlen, "out of memory"); goto fail; }
            snprintf(p->path, plen, "%s-%05d-of-%05" PRIu64 ".gguf",
                     stem, (int)i + 1, split_count);
            if (!map_part(p, true, errbuf, errlen)) goto fail;
            if (!parse_part(g, i, errbuf, errlen)) goto fail;
        }
        g->n_parts = (size_t)split_count;
    }

    /* Name-sorted index for lookups. */
    g->by_name = calloc(g->n_tensors, sizeof *g->by_name);
    if (!g->by_name && g->n_tensors) { snprintf(errbuf, errlen, "out of memory"); goto fail; }
    for (size_t i = 0; i < g->n_tensors; i++)
        g->by_name[i] = (uint32_t)i;
    if (g->n_tensors > 0) {
        g_sort_ctx = g;
        qsort(g->by_name, g->n_tensors, sizeof *g->by_name, cmp_tensor_name);
        g_sort_ctx = NULL;
    }

    return g;

fail:
    fatcat_gguf_close(g);
    return NULL;
}

static void free_value(fatcat_gguf_value *v)
{
    switch (v->type) {
    case FATCAT_GGUF_STR:
        free((void *)v->str.data);
        break;
    case FATCAT_GGUF_ARR:
        if (v->arr.elem_type == FATCAT_GGUF_STR && v->arr.elems) {
            fatcat_str *strs = v->arr.elems;
            for (uint64_t i = 0; i < v->arr.count; i++)
                free((void *)strs[i].data);
        }
        free(v->arr.elems);
        break;
    default:
        break;
    }
}

void fatcat_gguf_close(fatcat_gguf *g)
{
    if (!g) return;
    for (size_t i = 0; i < g->n_kv; i++)
        free_value(&g->kvs[i].value);
    free(g->kvs);
    for (size_t i = 0; i < g->n_tensors; i++)
        free((void *)g->tensors[i].name.data);
    free(g->tensors);
    free(g->by_name);
    for (size_t i = 0; i < g->n_parts; i++) {
        if (g->parts[i].map && g->parts[i].map != MAP_FAILED)
            munmap(g->parts[i].map, g->parts[i].size);
        if (g->parts[i].fd >= 0)
            close(g->parts[i].fd);
        if (g->parts[i].fd_direct > 0)
            close(g->parts[i].fd_direct);
        free(g->parts[i].path);
    }
    free(g->parts);
    free(g);
}

/* ---------- accessors ---------- */

uint32_t fatcat_gguf_version(const fatcat_gguf *g) { return g->version; }
size_t fatcat_gguf_num_parts(const fatcat_gguf *g) { return g->n_parts; }
uint64_t fatcat_gguf_data_start(const fatcat_gguf *g) { return g->parts[0].data_start; }
uint32_t fatcat_gguf_alignment(const fatcat_gguf *g) { return g->alignment; }

size_t fatcat_gguf_num_kv(const fatcat_gguf *g) { return g->n_kv; }

const fatcat_gguf_kv *fatcat_gguf_kv_at(const fatcat_gguf *g, size_t idx)
{
    return idx < g->n_kv ? &g->kvs[idx] : NULL;
}

const fatcat_gguf_value *fatcat_gguf_find_kv(const fatcat_gguf *g, const char *key)
{
    size_t klen = strlen(key);
    for (size_t i = 0; i < g->n_kv; i++) {
        if (g->kvs[i].key.len == klen &&
            memcmp(g->kvs[i].key.data, key, klen) == 0)
            return &g->kvs[i].value;
    }
    return NULL;
}

uint64_t fatcat_gguf_kv_u64(const fatcat_gguf *g, const char *key, uint64_t fallback)
{
    return kv_u64_raw(g->kvs, g->n_kv, key, fallback);
}

fatcat_str fatcat_gguf_kv_str(const fatcat_gguf *g, const char *key, fatcat_str fallback)
{
    const fatcat_gguf_value *v = fatcat_gguf_find_kv(g, key);
    if (v && v->type == FATCAT_GGUF_STR) return v->str;
    return fallback;
}

size_t fatcat_gguf_num_tensors(const fatcat_gguf *g) { return g->n_tensors; }

const fatcat_gguf_tensor *fatcat_gguf_tensor_at(const fatcat_gguf *g, size_t idx)
{
    return idx < g->n_tensors ? &g->tensors[idx] : NULL;
}

const fatcat_gguf_tensor *fatcat_gguf_find_tensor(const fatcat_gguf *g, const char *name)
{
    size_t klen = strlen(name);
    size_t lo = 0, hi = g->n_tensors;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const fatcat_gguf_tensor *t = &g->tensors[g->by_name[mid]];
        size_t n = t->name.len < klen ? t->name.len : klen;
        int r = memcmp(t->name.data, name, n);
        if (r == 0) r = (int)t->name.len - (int)klen;
        if (r < 0) lo = mid + 1;
        else if (r > 0) hi = mid;
        else return t;
    }
    return NULL;
}

const void *fatcat_gguf_tensor_ptr(const fatcat_gguf *g, const fatcat_gguf_tensor *t)
{
    const gguf_part *p = &g->parts[t->part];
    if (!p->map) return NULL;
    uint64_t local = p->data_start + t->offset;
    if (local + t->n_bytes > p->size) return NULL;
    return (const uint8_t *)p->map + local;
}

bool fatcat_gguf_tensor_read(const fatcat_gguf *g, const fatcat_gguf_tensor *t,
                             void *dst, size_t dst_len)
{
    const void *p = fatcat_gguf_tensor_ptr(g, t);
    if (p && dst_len >= t->n_bytes) {
        memcpy(dst, p, t->n_bytes);
        return true;
    }
    return false;
}
