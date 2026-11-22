/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* gguf-slice — extract a leading layer range (+ embeddings / lm_head) from a
 * (possibly split) GGUF into a new single-file GGUF.
 *
 * Purpose: build small model slices so forward-pass and streaming tests run
 * on the shared dev machine within the RAM/disk budget (see project.md §3).
 *
 * Usage: gguf-slice <in.gguf> <out.gguf> [n_layers=3] [options]
 *   --keep SUBSTR       also keep tensors whose name contains SUBSTR
 *   --no-tokenizer      drop the big tokenizer arrays (tokens/scores/merges)
 *   --align N           output alignment (default: source alignment)
 *
 * Behaviour:
 *  - Keeps blk.0..blk.(n_layers-1) plus token_embd.weight, output.weight,
 *    output_norm.weight (the lm_head).
 *  - Copies all part-0 KV metadata, with two fixes:
 *      * <arch>.block_count is set to n_layers;
 *      * any array KV whose length == original block_count (per-layer arrays
 *        such as attention.rope_pattern, attention.kv_source_map,
 *        attention.indexer.types) is truncated to n_layers.
 *  - Adds provenance KVs: fatcat.slice.source, fatcat.slice.orig_block_count.
 *  - Tensor data is copied verbatim (quant types are preserved exactly).
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/gguf.h"

static int failures;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                      \
        }                                                                    \
    } while (0)

/* ---------- growable output buffer (header + tensor table) ---------- */

typedef struct {
    uint8_t *p;
    size_t len, cap;
} buf;

static void bput(buf *b, const void *src, size_t n)
{
    if (b->len + n > b->cap) {
        size_t cap = (b->len + n) * 2 + 65536;
        uint8_t *p = realloc(b->p, cap);
        if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void b_u32(buf *b, uint32_t v) { bput(b, &v, 4); }
static void b_u64(buf *b, uint64_t v) { bput(b, &v, 8); }
static void b_gguf_str(buf *b, fatcat_str s)
{
    b_u64(b, s.len);
    bput(b, s.data, s.len);
}

/* Scalar byte size of a GGUF value type (0 for str/arr). */
static size_t scalar_size(fatcat_gguf_value_type t)
{
    switch (t) {
    case FATCAT_GGUF_U8: case FATCAT_GGUF_I8: case FATCAT_GGUF_BOOL: return 1;
    case FATCAT_GGUF_U16: case FATCAT_GGUF_I16: return 2;
    case FATCAT_GGUF_U32: case FATCAT_GGUF_I32: case FATCAT_GGUF_F32: return 4;
    case FATCAT_GGUF_U64: case FATCAT_GGUF_I64: case FATCAT_GGUF_F64: return 8;
    default: return 0;
    }
}

/* Serialize one KV value back to GGUF wire format. */
static void b_value(buf *b, const fatcat_gguf_value *v)
{
    b_u32(b, (uint32_t)v->type);
    switch (v->type) {
    case FATCAT_GGUF_U8:  bput(b, &v->u8, 1); break;
    case FATCAT_GGUF_I8:  bput(b, &v->i8, 1); break;
    case FATCAT_GGUF_U16: bput(b, &v->u16, 2); break;
    case FATCAT_GGUF_I16: bput(b, &v->i16, 2); break;
    case FATCAT_GGUF_U32: bput(b, &v->u32, 4); break;
    case FATCAT_GGUF_I32: bput(b, &v->i32, 4); break;
    case FATCAT_GGUF_F32: bput(b, &v->f32, 4); break;
    case FATCAT_GGUF_BOOL: { uint8_t x = v->boolean ? 1 : 0; bput(b, &x, 1); } break;
    case FATCAT_GGUF_U64: bput(b, &v->u64, 8); break;
    case FATCAT_GGUF_I64: bput(b, &v->i64, 8); break;
    case FATCAT_GGUF_F64: bput(b, &v->f64, 8); break;
    case FATCAT_GGUF_STR: b_gguf_str(b, v->str); break;
    case FATCAT_GGUF_ARR: {
        b_u32(b, (uint32_t)v->arr.elem_type);
        b_u64(b, v->arr.count);
        if (v->arr.elem_type == FATCAT_GGUF_STR) {
            const fatcat_str *strs = v->arr.elems;
            for (uint64_t i = 0; i < v->arr.count; i++)
                b_gguf_str(b, strs[i]);
        } else {
            size_t esz = scalar_size(v->arr.elem_type);
            if (esz == 0) { failures++; return; }
            bput(b, v->arr.elems, v->arr.count * esz);
        }
    } break;
    }
}

/* Serialize an array KV holding only the first `count` elements of `v`. */
static void b_value_prefix(buf *b, const fatcat_gguf_value *v, uint64_t count)
{
    b_u32(b, (uint32_t)FATCAT_GGUF_ARR);
    b_u32(b, (uint32_t)v->arr.elem_type);
    b_u64(b, count);
    if (v->arr.elem_type == FATCAT_GGUF_STR) {
        const fatcat_str *strs = v->arr.elems;
        for (uint64_t i = 0; i < count; i++)
            b_gguf_str(b, strs[i]);
    } else {
        size_t esz = scalar_size(v->arr.elem_type);
        if (esz == 0) { failures++; return; }
        bput(b, v->arr.elems, count * esz);
    }
}

/* ---------- tensor selection ---------- */

typedef struct {
    const fatcat_gguf_tensor *t;
    uint64_t out_offset;
} kept_t;

typedef struct {
    unsigned n_layers;
    const char *keep_substr; /* NULL or single substring */
    bool no_tokenizer;
} sel_t;

static bool keep_tensor(const sel_t *sel, const char *name)
{
    if (strncmp(name, "blk.", 4) == 0) {
        unsigned l;
        return sscanf(name, "blk.%u.", &l) == 1 && l < sel->n_layers;
    }
    if (strcmp(name, "token_embd.weight") == 0 ||
        strcmp(name, "output.weight") == 0 ||
        strcmp(name, "output_norm.weight") == 0)
        return true;
    if (sel->keep_substr && strstr(name, sel->keep_substr))
        return true;
    return false;
}

/* Skip huge tokenizer metadata arrays when --no-tokenizer, and always drop
 * split-GGUF bookkeeping KVs (the output is a single file). */
static bool skip_kv(const sel_t *sel, const char *key)
{
    static const char *split_keys[] = {
        "split.count", "split.no", "split.tensors.count",
    };
    for (size_t i = 0; i < sizeof split_keys / sizeof *split_keys; i++)
        if (strcmp(key, split_keys[i]) == 0)
            return true;
    if (!sel->no_tokenizer) return false;
    static const char *big[] = {
        "tokenizer.ggml.tokens", "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
        "tokenizer.ggml.added_tokens", "tokenizer.chat_template",
    };
    for (size_t i = 0; i < sizeof big / sizeof *big; i++)
        if (strcmp(key, big[i]) == 0)
            return true;
    return false;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <in.gguf> <out.gguf> [n_layers=3]\n"
            "       [--keep SUBSTR] [--no-tokenizer] [--align N]\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    sel_t sel = { .n_layers = 3, .keep_substr = NULL, .no_tokenizer = false };
    uint32_t out_align = 0; /* 0 = inherit source */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--no-tokenizer")) sel.no_tokenizer = true;
        else if (!strcmp(argv[i], "--keep") && i + 1 < argc)
            sel.keep_substr = argv[++i];
        else if (!strcmp(argv[i], "--align") && i + 1 < argc)
            out_align = (uint32_t)atoi(argv[++i]);
        else
            sel.n_layers = (unsigned)atoi(argv[i]);
    }

    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(in_path, err, sizeof err);
    if (!g) { fprintf(stderr, "error: %s\n", err); return 1; }

    fatcat_str arch = fatcat_gguf_kv_str(g, "general.architecture",
                                         (fatcat_str){ "general", 7 });
    char archs[64];
    snprintf(archs, sizeof archs, "%.*s", (int)arch.len, arch.data);
    char key_block_count[128];
    snprintf(key_block_count, sizeof key_block_count, "%s.block_count", archs);
    uint64_t orig_block_count = fatcat_gguf_kv_u64(g, key_block_count, 0);
    if (orig_block_count == 0) {
        fprintf(stderr, "error: %s missing\n", key_block_count);
        return 1;
    }
    if (sel.n_layers == 0 || sel.n_layers > orig_block_count) {
        fprintf(stderr, "error: n_layers must be in [1, %" PRIu64 "]\n",
                orig_block_count);
        return 1;
    }
    uint32_t align = out_align ? out_align : fatcat_gguf_alignment(g);

    /* ---- pass 1: select tensors, assign output offsets ---- */
    size_t n_src = fatcat_gguf_num_tensors(g);
    kept_t *keep = calloc(n_src, sizeof *keep);
    if (!keep) { fprintf(stderr, "out of memory\n"); return 2; }

    size_t n_keep = 0;
    uint64_t data_size = 0, kept_bytes = 0;
    for (size_t i = 0; i < n_src; i++) {
        const fatcat_gguf_tensor *t = fatcat_gguf_tensor_at(g, i);
        char namebuf[512];
        snprintf(namebuf, sizeof namebuf, "%.*s", (int)t->name.len, t->name.data);
        if (!keep_tensor(&sel, namebuf)) continue;
        keep[n_keep].t = t;
        keep[n_keep].out_offset = data_size;
        n_keep++;
        data_size += t->n_bytes;
        data_size = (data_size + align - 1) / align * align;
        kept_bytes += t->n_bytes;
    }

    /* ---- pass 2: build output KV + tensor info in memory ---- */
    size_t out_kv_count = 2; /* fatcat.slice.* provenance */
    for (size_t i = 0; i < fatcat_gguf_num_kv(g); i++) {
        const fatcat_gguf_kv *kv = fatcat_gguf_kv_at(g, i);
        char key[512];
        snprintf(key, sizeof key, "%.*s", (int)kv->key.len, kv->key.data);
        if (!skip_kv(&sel, key)) out_kv_count++;
    }

    buf hdr = { 0 };
    bput(&hdr, "GGUF", 4);
    b_u32(&hdr, 3);
    b_u64(&hdr, n_keep);
    b_u64(&hdr, out_kv_count);

    for (size_t i = 0; i < fatcat_gguf_num_kv(g); i++) {
        const fatcat_gguf_kv *kv = fatcat_gguf_kv_at(g, i);
        char key[512];
        snprintf(key, sizeof key, "%.*s", (int)kv->key.len, kv->key.data);
        if (skip_kv(&sel, key)) continue;
        b_gguf_str(&hdr, kv->key);
        if (strcmp(key, key_block_count) == 0) {
            fatcat_gguf_value v;
            v.type = FATCAT_GGUF_U32;
            v.u32 = sel.n_layers;
            b_value(&hdr, &v);
        } else if (kv->value.type == FATCAT_GGUF_ARR &&
                   kv->value.arr.count == orig_block_count) {
            b_value_prefix(&hdr, &kv->value, sel.n_layers);
        } else {
            b_value(&hdr, &kv->value);
        }
    }
    {
        fatcat_str k1 = { "fatcat.slice.source", 19 };
        fatcat_str k2 = { "fatcat.slice.orig_block_count", 29 };
        fatcat_gguf_value v1;
        v1.type = FATCAT_GGUF_STR;
        v1.str = (fatcat_str){ in_path, strlen(in_path) };
        fatcat_gguf_value v2;
        v2.type = FATCAT_GGUF_U32;
        v2.u32 = (uint32_t)orig_block_count;
        b_gguf_str(&hdr, k1); b_value(&hdr, &v1);
        b_gguf_str(&hdr, k2); b_value(&hdr, &v2);
    }

    for (size_t i = 0; i < n_keep; i++) {
        const fatcat_gguf_tensor *t = keep[i].t;
        b_gguf_str(&hdr, t->name);
        b_u32(&hdr, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            b_u64(&hdr, t->dims[d]);
        b_u32(&hdr, t->type);
        b_u64(&hdr, keep[i].out_offset);
    }
    while (hdr.len % align != 0) { uint8_t z = 0; bput(&hdr, &z, 1); }

    /* ---- pass 3: stream tensor data to the output file ---- */
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", out_path);
        return 1;
    }
    fwrite(hdr.p, 1, hdr.len, f);
    uint64_t written = 0;
    static const uint8_t zeros[64] = {0};
    for (size_t i = 0; i < n_keep; i++) {
        const fatcat_gguf_tensor *t = keep[i].t;
        const void *p = fatcat_gguf_tensor_ptr(g, t);
        if (!p) {
            fprintf(stderr, "error: tensor '%.*s' data unavailable\n",
                    (int)t->name.len, t->name.data);
            fclose(f);
            return 1;
        }
        fwrite(p, 1, t->n_bytes, f);
        uint64_t pad = ((t->n_bytes + align - 1) / align * align) - t->n_bytes;
        while (pad) {
            size_t chunk = pad < 64 ? pad : 64;
            fwrite(zeros, 1, chunk, f);
            pad -= chunk;
        }
        written += t->n_bytes;
        if ((written >> 30) != ((written - t->n_bytes) >> 30))
            fprintf(stderr, "  %2" PRIu64 " / %.2f GB\r", written >> 30,
                    (double)kept_bytes / 1e9);
    }
    fclose(f);
    fprintf(stderr, "\n");

    printf("gguf-slice: wrote %s\n", out_path);
    printf("  layers kept: 0..%u of %" PRIu64 "\n",
           sel.n_layers - 1, orig_block_count);
    printf("  tensors:     %zu of %zu\n", n_keep, n_src);
    printf("  data:        %.2f GB\n", (double)written / 1e9);

    /* ---- self-check: reopen the slice with our own reader ---- */
    fatcat_gguf *s = fatcat_gguf_open(out_path, err, sizeof err);
    CHECK(s != NULL);
    if (s) {
        CHECK(fatcat_gguf_num_tensors(s) == n_keep);
        CHECK(fatcat_gguf_kv_u64(s, key_block_count, 0) == sel.n_layers);
        CHECK(fatcat_gguf_find_tensor(s, "token_embd.weight") != NULL);
        CHECK(fatcat_gguf_find_tensor(s, "output.weight") != NULL);
        CHECK(fatcat_gguf_find_tensor(s, "output_norm.weight") != NULL);
        char probe[64];
        snprintf(probe, sizeof probe, "blk.%u.attn_q.weight",
                 sel.n_layers - 1);
        CHECK(fatcat_gguf_find_tensor(s, probe) != NULL);
        snprintf(probe, sizeof probe, "blk.%u.attn_norm.weight", sel.n_layers);
        CHECK(fatcat_gguf_find_tensor(s, probe) == NULL);
        /* per-layer array KVs must be truncated to n_layers */
        char key_arr[128];
        snprintf(key_arr, sizeof key_arr, "%s.attention.kv_source_map", archs);
        const fatcat_gguf_value *va = fatcat_gguf_find_kv(s, key_arr);
        if (va)
            CHECK(va->type == FATCAT_GGUF_ARR && va->arr.count == sel.n_layers);
        fatcat_gguf_close(s);
    }

    free(hdr.p);
    free(keep);
    fatcat_gguf_close(g);

    if (failures) { printf("gguf-slice: %d FAILURE(S)\n", failures); return 1; }
    printf("gguf-slice: OK\n");
    return 0;
}
