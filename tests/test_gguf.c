/* test_gguf — unit tests for the GGUF reader using synthetic files.
 * Builds a small GGUF v3 in memory (single-file) and a proper llama.cpp
 * style split GGUF (each part a self-contained GGUF; part 0 holds global
 * metadata + split.* KVs, part 1 holds the tensor table and data).
 * Uses CHECK() (never compiled out) instead of assert(). */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fatcat/gguf.h"

static int failures;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                      \
        }                                                                    \
    } while (0)

/* ---------- tiny growable byte buffer ---------- */
typedef struct {
    uint8_t *p;
    size_t len, cap;
} buf;

static void bput(buf *b, const void *src, size_t n)
{
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 4096;
        b->p = realloc(b->p, b->cap);
        if (!b->p) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}
static void b_u16(buf *b, uint16_t v) { bput(b, &v, 2); }
static void b_u32(buf *b, uint32_t v) { bput(b, &v, 4); }
static void b_u64(buf *b, uint64_t v) { bput(b, &v, 8); }
static void b_str(buf *b, const char *s)
{
    b_u64(b, strlen(s));
    bput(b, s, strlen(s));
}
static void pad32(buf *b)
{
    while (b->len % 32 != 0) { uint8_t z = 0; bput(b, &z, 1); }
}

static void data_pattern(uint8_t *d, size_t n)
{
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)(i * 7 + 3);
}

#define N_DATA (32 + 144 + 100)

/* The three tensors and their data, shared by both builders. */
static void put_tensor_table(buf *b)
{
    b_str(b, "w_f32");  b_u32(b, 2); b_u64(b, 4); b_u64(b, 2);
    b_u32(b, FATCAT_GGML_F32);   b_u64(b, 0);
    b_str(b, "w_q4k");  b_u32(b, 1); b_u64(b, 256);
    b_u32(b, FATCAT_GGML_Q4_K);  b_u64(b, 32);
    b_str(b, "w_iq1s"); b_u32(b, 1); b_u64(b, 512);
    b_u32(b, FATCAT_GGML_IQ1_S); b_u64(b, 176);
}

/* Global model KV (part 0 of split, or the whole file when single). */
static void put_model_kv(buf *b)
{
    b_str(b, "general.architecture"); b_u32(b, FATCAT_GGUF_STR); b_str(b, "test");
    b_str(b, "test.n");   b_u32(b, FATCAT_GGUF_U32); b_u32(b, 7);
    b_str(b, "test.f");   b_u32(b, FATCAT_GGUF_F32); { float f = 1.5f; bput(b, &f, 4); }
    b_str(b, "test.list"); b_u32(b, FATCAT_GGUF_ARR);
    b_u32(b, FATCAT_GGUF_STR); b_u64(b, 2); b_str(b, "alpha"); b_str(b, "beta");
    b_str(b, "test.flag"); b_u32(b, FATCAT_GGUF_BOOL); { uint8_t one = 1; bput(b, &one, 1); }
    b_str(b, "test.flags"); b_u32(b, FATCAT_GGUF_ARR);
    b_u32(b, FATCAT_GGUF_BOOL); b_u64(b, 3);
    { uint8_t fl[3] = {1, 0, 1}; bput(b, fl, 3); }
}

/* ---------- single-file GGUF ---------- */
static void build_single(buf *out)
{
    buf *b = out;
    memset(b, 0, sizeof *b);
    bput(b, "GGUF", 4);
    b_u32(b, 3); b_u64(b, 3); b_u64(b, 6);
    put_model_kv(b);
    put_tensor_table(b);
    pad32(b);
    uint8_t data[N_DATA];
    data_pattern(data, sizeof data);
    bput(b, data, sizeof data);
}

/* ---------- split GGUF (llama.cpp semantics) ---------- */
/* part 0: metadata + split.* KVs, no tensors; part 1: tensor table + data */
static void build_split0(buf *out)
{
    buf *b = out;
    memset(b, 0, sizeof *b);
    bput(b, "GGUF", 4);
    b_u32(b, 3); b_u64(b, 0); b_u64(b, 8);
    put_model_kv(b);
    b_str(b, "split.count"); b_u32(b, FATCAT_GGUF_U16); b_u16(b, 2);
    b_str(b, "split.no");    b_u32(b, FATCAT_GGUF_U16); b_u16(b, 0);
    pad32(b);
}
static void build_split1(buf *out)
{
    buf *b = out;
    memset(b, 0, sizeof *b);
    bput(b, "GGUF", 4);
    b_u32(b, 3); b_u64(b, 3); b_u64(b, 2);
    b_str(b, "split.count"); b_u32(b, FATCAT_GGUF_U16); b_u16(b, 2);
    b_str(b, "split.no");    b_u32(b, FATCAT_GGUF_U16); b_u16(b, 1);
    put_tensor_table(b);
    pad32(b);
    uint8_t data[N_DATA];
    data_pattern(data, sizeof data);
    bput(b, data, sizeof data);
}

static void write_file(const char *path, const uint8_t *p, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fwrite(p, 1, n, f);
    fclose(f);
}

static void check_gguf(const fatcat_gguf *g)
{
    const fatcat_gguf_value *v;
    CHECK(fatcat_gguf_version(g) == 3);
    CHECK(fatcat_gguf_alignment(g) == 32);
    CHECK(fatcat_gguf_num_tensors(g) == 3);

    v = fatcat_gguf_find_kv(g, "general.architecture");
    CHECK(v && v->type == FATCAT_GGUF_STR);
    CHECK(v && v->str.len == 4 && memcmp(v->str.data, "test", 4) == 0);
    CHECK(fatcat_gguf_kv_u64(g, "test.n", 0) == 7);
    v = fatcat_gguf_find_kv(g, "test.f");
    CHECK(v && v->type == FATCAT_GGUF_F32 && v->f32 == 1.5f);
    v = fatcat_gguf_find_kv(g, "test.list");
    CHECK(v && v->type == FATCAT_GGUF_ARR && v->arr.count == 2);
    if (v && v->arr.elems) {
        const fatcat_str *strs = v->arr.elems;
        CHECK(strs[0].len == 5 && memcmp(strs[0].data, "alpha", 5) == 0);
        CHECK(strs[1].len == 4 && memcmp(strs[1].data, "beta", 4) == 0);
    }
    v = fatcat_gguf_find_kv(g, "test.flag");
    CHECK(v && v->type == FATCAT_GGUF_BOOL && v->boolean == true);
    v = fatcat_gguf_find_kv(g, "test.flags");
    CHECK(v && v->type == FATCAT_GGUF_ARR && v->arr.count == 3);
    if (v && v->arr.elems) {
        const uint8_t *fl = v->arr.elems;
        CHECK(fl[0] == 1 && fl[1] == 0 && fl[2] == 1);
    }
    CHECK(fatcat_gguf_find_kv(g, "nope") == NULL);

    const fatcat_gguf_tensor *t;
    t = fatcat_gguf_find_tensor(g, "w_f32");
    CHECK(t && t->type == FATCAT_GGML_F32 && t->n_bytes == 32);
    CHECK(t && t->dims[0] == 4 && t->dims[1] == 2 && t->n_elems == 8);
    t = fatcat_gguf_find_tensor(g, "w_q4k");
    CHECK(t && t->type == FATCAT_GGML_Q4_K && t->n_bytes == 144);
    t = fatcat_gguf_find_tensor(g, "w_iq1s");
    CHECK(t && t->type == FATCAT_GGML_IQ1_S && t->n_bytes == 100);
    CHECK(fatcat_gguf_find_tensor(g, "missing") == NULL);

    /* data integrity */
    uint8_t expect[N_DATA];
    data_pattern(expect, sizeof expect);
    uint8_t got[256];
    t = fatcat_gguf_find_tensor(g, "w_q4k");
    CHECK(t && fatcat_gguf_tensor_read(g, t, got, sizeof got));
    CHECK(memcmp(got, expect + 32, 144) == 0);
    t = fatcat_gguf_find_tensor(g, "w_iq1s");
    CHECK(t && fatcat_gguf_tensor_read(g, t, got, sizeof got));
    CHECK(memcmp(got, expect + 32 + 144, 100) == 0);
}

int main(void)
{
    char dir[] = "/tmp/fatcat-test-gguf-XXXXXX";
    if (!mkdtemp(dir)) { fprintf(stderr, "mkdtemp failed\n"); return 2; }
    char path[512];
    char err[256];

    /* --- single-file GGUF --- */
    buf single = {0};
    build_single(&single);
    snprintf(path, sizeof path, "%s/model.gguf", dir);
    write_file(path, single.p, single.len);
    fatcat_gguf *g = fatcat_gguf_open(path, err, sizeof err);
    CHECK(g != NULL);
    if (!g) { fprintf(stderr, "open error: %s\n", err); return 1; }
    CHECK(fatcat_gguf_num_parts(g) == 1);
    CHECK(fatcat_gguf_num_kv(g) == 6);
    check_gguf(g);
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, "w_f32");
    CHECK(t && fatcat_gguf_tensor_ptr(g, t) != NULL);
    fatcat_gguf_close(g);
    free(single.p);

    /* --- split GGUF: part 0 metadata, part 1 tensors+data --- */
    buf s0 = {0}, s1 = {0};
    build_split0(&s0);
    build_split1(&s1);
    snprintf(path, sizeof path, "%s/split-00001-of-00002.gguf", dir);
    write_file(path, s0.p, s0.len);
    char path2[512];
    snprintf(path2, sizeof path2, "%s/split-00002-of-00002.gguf", dir);
    write_file(path2, s1.p, s1.len);
    g = fatcat_gguf_open(path, err, sizeof err);
    CHECK(g != NULL);
    if (!g) { fprintf(stderr, "split open error: %s\n", err); return 1; }
    CHECK(fatcat_gguf_num_parts(g) == 2);
    CHECK(fatcat_gguf_num_kv(g) == 8);  /* global KV from part 0 only */
    check_gguf(g);
    /* tensors live in part 1 */
    t = fatcat_gguf_find_tensor(g, "w_iq1s");
    CHECK(t && t->part == 1);
    CHECK(t && fatcat_gguf_tensor_ptr(g, t) != NULL);
    fatcat_gguf_close(g);
    free(s0.p);
    free(s1.p);

    /* --- reject garbage --- */
    snprintf(path, sizeof path, "%s/bad.gguf", dir);
    write_file(path, (const uint8_t *)"NOTAGGUFDATA!!!!", 16);
    g = fatcat_gguf_open(path, err, sizeof err);
    CHECK(g == NULL);

    if (failures) {
        printf("test_gguf: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("test_gguf: PASS\n");
    return 0;
}
