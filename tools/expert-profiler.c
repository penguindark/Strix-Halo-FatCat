/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* expert-profiler — measure routed-expert usage of Mistral-Large-4 on real text.
 *
 * Runs the scalar ML4 forward pass (router-exact for quantized models: router
 * weights are F32) over a prompt and accumulates per-layer expert hotness.
 * Designed for low RAM: weights stay mmap'd; only histograms live in memory.
 *
 * Usage:
 *   expert-profiler MODEL.gguf [--tok MODEL_OR_FULL.gguf] [--text FILE]
 *                              [--random N] [--trace OUT.bin] [--quiet]
 *
 * --tok: GGUF providing the tokenizer (default: the model itself; required
 *        when MODEL is a slice whose tokenizer was stripped).
 * --text: UTF-8 text file to profile on (repeatable). --random N: also feed
 *        N random token ids (sanity/stress).
 *
 * Output: per-layer hotness table + coverage stats; optional binary trace:
 *   header "FCT1" u32 n_layers u32 n_experts, then f64 weighted[n_layers][n_experts],
 *   then u64 raw[n_layers][n_experts]. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "fatcat/dequant.h"
#include "fatcat/gguf.h"
#include "fatcat/gpu.h"
#include "fatcat/ml4.h"

#define MAX_TEXTS 64
#define TOK_HASH_BITS 19
#define MAX_TOK_LEN 48

typedef struct {
    fatcat_gguf *g;
    uint32_t n_tokens;
    const fatcat_str *tokens;   /* view into gguf arr elems */
    int32_t *hash;              /* open addressing: slot -> token idx */
    uint32_t hash_mask;
    uint32_t max_len;
} tokenizer;

static uint32_t str_hash(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

static bool tok_init(tokenizer *tk, fatcat_gguf *g)
{
    const fatcat_gguf_value *v = fatcat_gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!v || v->type != FATCAT_GGUF_ARR || v->arr.elem_type != FATCAT_GGUF_STR)
        return false;
    tk->g = g;
    tk->n_tokens = (uint32_t)v->arr.count;
    tk->tokens = v->arr.elems;
    uint32_t cap = 1u << TOK_HASH_BITS;
    tk->hash_mask = cap - 1;
    tk->hash = malloc(cap * sizeof(int32_t));
    if (!tk->hash) return false;
    memset(tk->hash, 0xFF, cap * sizeof(int32_t));
    tk->max_len = 1;
    for (uint32_t i = 0; i < tk->n_tokens; i++) {
        fatcat_str s = tk->tokens[i];
        if (s.len == 0 || s.len > MAX_TOK_LEN) continue;
        if (s.len > tk->max_len) tk->max_len = (uint32_t)s.len;
        uint32_t h = str_hash(s.data, s.len) & tk->hash_mask;
        while (tk->hash[h] >= 0) h = (h + 1) & tk->hash_mask;
        tk->hash[h] = (int32_t)i;
    }
    return true;
}

static int tok_lookup(const tokenizer *tk, const char *s, size_t n)
{
    uint32_t h = str_hash(s, n) & tk->hash_mask;
    while (tk->hash[h] >= 0) {
        fatcat_str t = tk->tokens[tk->hash[h]];
        if (t.len == n && memcmp(t.data, s, n) == 0) return tk->hash[h];
        h = (h + 1) & tk->hash_mask;
    }
    return -1;
}

/* Greedy longest-match tokenizer (tekken approximation). Spaces are mapped to
 * the GPT2 "Ġ" marker (0xC4 0xA0) before matching. Returns token count. */
static size_t tokenize(const tokenizer *tk, const char *text, size_t len,
                       uint32_t *out, size_t out_cap)
{
    /* expand spaces into Ġ marker */
    char *buf = malloc(len * 2 + 8);
    size_t bn = 0;
    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        if (c == '\r' || c == '\t') c = ' ';
        if (c == ' ') { buf[bn++] = (char)0xC4; buf[bn++] = (char)0xA0; }
        else buf[bn++] = c;
    }
    size_t n = 0;
    size_t i = 0;
    while (i < bn && n < out_cap) {
        int best = -1; size_t best_len = 0;
        for (size_t l = tk->max_len; l >= 1 && best < 0; l--) {
            if (i + l > bn) continue;
            int id = tok_lookup(tk, buf + i, l);
            if (id >= 0) { best = id; best_len = l; }
        }
        if (best >= 0) { out[n++] = (uint32_t)best; i += best_len; }
        else i++; /* drop unmatchable byte */
    }
    free(buf);
    return n;
}

/* ---------- profiling ---------- */

typedef struct {
    uint32_t n_layers, n_experts;
    double *wsum;   /* [n_layers * n_experts] */
    uint64_t *raw;  /* [n_layers * n_experts] */
    const char *trace_path;      /* if set, re-dump every 64 tokens */
    uint32_t first_moe;          /* first MoE layer (dedupe per-token) */
    uint64_t tok_seen;
} hist_t;

static void trace_dump(const char *path, const hist_t *h);

static void router_cb(void *ctx, uint32_t layer, uint32_t token,
                      const uint16_t *ids, const float *weights, uint32_t k)
{
    (void)token;
    hist_t *h = ctx;
    for (uint32_t i = 0; i < k; i++) {
        h->wsum[(size_t)layer * h->n_experts + ids[i]] += weights[i];
        h->raw[(size_t)layer * h->n_experts + ids[i]]++;
    }
    /* incremental dump so long runs always have usable data */
    if (h->trace_path && layer == h->first_moe && ++h->tok_seen % 64 == 0)
        trace_dump(h->trace_path, h);
}

static int cmp_desc(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

static void report(const hist_t *h, uint64_t n_tokens)
{
    printf("\n=== expert hotness (%u layers x %u experts, %llu tokens) ===\n",
           h->n_layers, h->n_experts, (unsigned long long)n_tokens);
    printf("%-6s %8s %8s %7s %7s %7s %7s %8s\n",
           "layer", "act/tok", "uniq", "top1%", "cov50", "cov80", "cov95", "cov99");
    for (uint32_t l = 0; l < h->n_layers; l++) {
        const double *row = h->wsum + (size_t)l * h->n_experts;
        double total = 0; uint32_t uniq = 0;
        double *sorted = malloc(h->n_experts * sizeof(double));
        for (uint32_t e = 0; e < h->n_experts; e++) {
            total += row[e];
            if (row[e] > 0) uniq++;
            sorted[e] = row[e];
        }
        if (total <= 0) { free(sorted); continue; }
        qsort(sorted, h->n_experts, sizeof(double), cmp_desc);
        double top1 = 100.0 * sorted[0] / total;
        uint32_t c50 = 0, c80 = 0, c95 = 0, c99 = 0;
        double acc = 0;
        for (uint32_t e = 0; e < h->n_experts; e++) {
            acc += sorted[e];
            double f = acc / total;
            if (!c50 && f >= 0.50) c50 = e + 1;
            if (!c80 && f >= 0.80) c80 = e + 1;
            if (!c95 && f >= 0.95) c95 = e + 1;
            if (!c99 && f >= 0.99) c99 = e + 1;
        }
        printf("%-6u %8.2f %8u %7.1f %7u %7u %7u %8u\n",
               l, total / (double)(n_tokens ? n_tokens : 1), uniq, top1,
               c50, c80, c95, c99);
        free(sorted);
    }
}

static void trace_dump(const char *path, const hist_t *h)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write trace %s\n", path); return; }
    fwrite("FCT1", 1, 4, f);
    fwrite(&h->n_layers, 4, 1, f);
    fwrite(&h->n_experts, 4, 1, f);
    fwrite(h->wsum, sizeof(double), (size_t)h->n_layers * h->n_experts, f);
    fwrite(h->raw, sizeof(uint64_t), (size_t)h->n_layers * h->n_experts, f);
    fclose(f);
    printf("trace written: %s\n", path);
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    buf[sz] = 0;
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- live progress (one line per token; monitor with tail -f) ---- */

static double g_prog_t0;
static fatcat_ml4 *g_prog_m;   /* model handle, for the per-token health check */

/* ---------- startup self-test: GPU gemv kernels vs CPU scalar path -----
 * Runs one fused dequant+GEMV per quant type on a slice of a real tensor
 * and compares against the (bit-validated) CPU dequant. A failure here
 * means every GPU number in this run would be garbage, so we abort. */
static double selftest_relerr(uint32_t type, const void *w,
                              int64_t n_out, int64_t n_in)
{
    float *x = malloc(n_in * 4), *ga = malloc(n_out * 4), *gb = malloc(n_out * 4);
    float *tmp = malloc(n_in * 4);
    if (!x || !ga || !gb || !tmp) { free(x); free(ga); free(gb); free(tmp); return -1; }
    for (int64_t i = 0; i < n_in; i++) x[i] = (float)((i * 37 % 101) - 50) / 50.0f;
    const int64_t rb = fatcat_dequant_src_bytes(type, n_in);
    double num = 0, den = 0;
    for (int64_t i = 0; i < n_out; i++) {
        fatcat_dequant(type, (const char *)w + i * rb, tmp, n_in);
        double s = 0;
        for (int64_t j = 0; j < n_in; j++) s += (double)tmp[j] * x[j];
        gb[i] = (float)s;
    }
    void *h = fatcat_gpu_stage(type, w, (size_t)(rb * n_out), 991 + type);
    int ok = h && fatcat_gpu_gemv(h, type, x, ga, n_out, n_in);
    if (h) fatcat_gpu_release(h);
    if (!ok) { free(x); free(ga); free(gb); free(tmp); return -1; }
    for (int64_t i = 0; i < n_out; i++) {
        double d = (double)ga[i] - gb[i];
        num += d * d; den += (double)gb[i] * gb[i];
    }
    free(x); free(ga); free(gb); free(tmp);
    return den > 0 ? sqrt(num / den) : (num > 0 ? 1e9 : 0);
}

static bool gpu_selftest(fatcat_gguf *g)
{
    static const char *names[] = {
        "blk.1.attn_q.weight",         /* Q6_K    */
        "blk.1.ffn_down_shexp.weight", /* Q8_0    */
        "blk.1.ffn_gate_exps.weight",  /* IQ2_XXS */
        "blk.10.ffn_gate_exps.weight", /* IQ1_S   */
    };
    bool ok = true;
    printf("selftest: GPU gemv kernels vs CPU on real tensors (128 rows each)\n");
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, names[i]);
        if (!t) { printf("selftest: %-30s missing (skipped)\n", names[i]); continue; }
        int64_t n_in = (int64_t)t->dims[0];
        int64_t n_out = t->n_dims > 1 ? (int64_t)t->dims[1] : 1;
        if (n_out > 128) n_out = 128;
        double e = selftest_relerr(t->type, fatcat_gguf_tensor_ptr(g, t), n_out, n_in);
        bool pass = e >= 0 && e < 1e-3;
        printf("selftest: %-30s type=%2u relerr=%.2g %s\n",
               names[i], t->type, e, pass ? "ok" : "FAIL");
        if (!pass) ok = false;
    }
    printf("selftest: %s\n", ok ? "ALL KERNELS OK" : "FAILED - GPU results unreliable");
    fflush(stdout);
    return ok;
}

static void fmt_dur(char *buf, size_t n, double s)
{
    if (s < 0 || s > 3600 * 100) { snprintf(buf, n, "  --:--:--"); return; }
    int h = (int)s / 3600, m = (int)s % 3600 / 60, sec = (int)s % 60;
    snprintf(buf, n, "%02d:%02d:%02d", h, m, sec);
}

static void token_progress(void *ctx, size_t done, size_t total)
{
    (void)ctx;
    double el = now_s() - g_prog_t0;
    double eta = done ? el / (double)done * (double)(total - done) : -1;
    char e[16], t[16];
    fmt_dur(e, sizeof(e), el);
    fmt_dur(t, sizeof(t), eta);
    uint64_t ht = 0, hb = 0;
    if (g_prog_m) fatcat_ml4_get_health(g_prog_m, &ht, &hb);
    long used = 0, mt = 0, ma = 0;
    { FILE *f = fopen("/proc/meminfo", "r");
      if (f) { char nm[64], u[16]; long v;
               while (fscanf(f, "%63s %ld %15s", nm, &v, u) == 3) {
                   if (!strcmp(nm, "MemTotal:")) mt = v;
                   else if (!strcmp(nm, "MemAvailable:")) ma = v; }
               fclose(f); used = (mt - ma) / 1024; } }
    printf("[%5zu/%zu] %5.1f%%  elapsed %s  eta %s  %.2f tok/s  health=%s  mem=%ldMB\n",
           done, total, 100.0 * (double)done / (double)total, e, t,
           el > 0 ? done / el : 0.0, hb ? "BAD" : "ok", used);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);  /* live progress even when piped to tee */
    const char *model_path = NULL, *tok_path = NULL, *trace_path = NULL;
    const char *tokens_file = NULL;
    const char *text_files[MAX_TEXTS]; int n_text_files = 0;
    uint32_t n_random = 0;
    bool quiet = false, check = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tok") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) {
            if (n_text_files < MAX_TEXTS) text_files[n_text_files++] = argv[++i];
        }
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_file = argv[++i];
        else if (!strcmp(argv[i], "--random") && i + 1 < argc) n_random = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = true;
        else if (!strcmp(argv[i], "--check")) check = true;
        else if (argv[i][0] != '-') model_path = argv[i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!model_path) {
        fprintf(stderr, "usage: expert-profiler MODEL.gguf [--tok GGUF] [--text FILE]"
                        " [--random N] [--trace OUT.bin] [--quiet]\n");
        return 2;
    }

    char err[256];
    char gpuerr[256] = "not attempted";
    bool gpu = false;

    /* structured log header (parseable when reviewing after the run) */
    time_t tm0 = time(NULL);
    printf("=== fatcat expert-profiler ===\n");
    printf("build: %s %s\n", __DATE__, __TIME__);
    printf("start: %s", ctime(&tm0));
    printf("model: %s\n", model_path);
    if (tokens_file) printf("tokens-file: %s\n", tokens_file);
    for (int i = 0; i < n_text_files; i++) printf("text-file: %s\n", text_files[i]);
    fflush(stdout);

    gpu = fatcat_gpu_init(16, gpuerr, sizeof(gpuerr));
    printf("gpu: %s\n", gpu ? fatcat_gpu_device_string() : gpuerr);
    fflush(stdout);

    fatcat_gguf *g = fatcat_gguf_open(model_path, err, sizeof(err));
    if (!g) { fprintf(stderr, "open %s: %s\n", model_path, err); return 1; }
    printf("gguf opened, loading model (uploads fixed weights to GPU)...\n");
    fflush(stdout);
    fatcat_ml4 *m = fatcat_ml4_open(g, err, sizeof(err));
    if (!m) { fprintf(stderr, "ml4: %s\n", err); fatcat_gguf_close(g); return 1; }
    fatcat_ml4_info info = fatcat_ml4_get_info(m);
    printf("model: %u layers (%u dense), %u experts top-%u, embd %u, vocab %u\n",
           info.n_layers, info.n_dense_layers, info.n_experts,
           info.n_experts_used, info.n_embd, info.n_vocab);
    fatcat_ml4_set_token_cb(m, token_progress, NULL);
    g_prog_m = m;
    fflush(stdout);

    if (gpu && !gpu_selftest(g)) {
        fprintf(stderr, "selftest failed - aborting (results would be invalid)\n");
        return 3;
    }

    /* tokenizer: from --tok, else from the model itself. Only needed for
     * --text input and --check decoding. */
    tokenizer tk = {0};
    fatcat_gguf *tg = g;
    bool need_tok = n_text_files > 0 || check;
    if (need_tok) {
        if (tok_path) {
            tg = fatcat_gguf_open(tok_path, err, sizeof(err));
            if (!tg) { fprintf(stderr, "open %s: %s\n", tok_path, err); return 1; }
        }
        if (!tok_init(&tk, tg)) {
            fprintf(stderr, "no usable tokenizer in %s (use --tokens with pre-tokenized ids)\n",
                    tok_path ? tok_path : model_path);
            return 1;
        }
        if (tk.n_tokens < info.n_vocab && !quiet)
            fprintf(stderr, "note: tokenizer has %u tokens, vocab is %u (partial coverage)\n",
                    tk.n_tokens, info.n_vocab);
    }

    hist_t hist = { .n_layers = info.n_layers, .n_experts = info.n_experts,
                    .trace_path = trace_path, .first_moe = info.n_dense_layers };
    hist.wsum = calloc((size_t)info.n_layers * info.n_experts, sizeof(double));
    hist.raw = calloc((size_t)info.n_layers * info.n_experts, sizeof(uint64_t));

    size_t tokcap = (size_t)1 << 20;
    uint32_t *tokbuf = malloc(tokcap * sizeof(uint32_t));
    uint64_t total_tokens = 0;
    double t0 = now_s();

    for (int i = 0; i < n_text_files; i++) {
        size_t len;
        char *text = read_file(text_files[i], &len);
        if (!text) { fprintf(stderr, "cannot read %s\n", text_files[i]); continue; }
        size_t n = tokenize(&tk, text, len, tokbuf, tokcap);
        printf("text %s: %zu bytes -> %zu tokens\n", text_files[i], len, n);
        free(text);
        if (!n) continue;
        double t = now_s();
        g_prog_t0 = t;
        fatcat_ml4_reset(m);   /* each input file is an independent sequence */
        if (fatcat_ml4_forward(m, tokbuf, n, router_cb, &hist)) {
            fprintf(stderr, "forward failed\n"); return 1;
        }
        total_tokens += n;
        printf("  forward %.1fs (%.2f tok/s)\n", now_s() - t, n / (now_s() - t));
    }

    if (tokens_file) {
        size_t len;
        char *ids = read_file(tokens_file, &len);
        if (!ids) { fprintf(stderr, "cannot read %s\n", tokens_file); return 1; }
        size_t n = 0;
        char *p = ids;
        while (*p && n < tokcap) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            if (v >= 0 && (uint32_t)v < info.n_vocab) tokbuf[n++] = (uint32_t)v;
            p = end;
        }
        free(ids);
        printf("tokens %s: %zu ids\n", tokens_file, n);
        if (n) {
            double t = now_s();
            g_prog_t0 = t;
            fatcat_ml4_reset(m);   /* independent sequence */
            if (fatcat_ml4_forward(m, tokbuf, n, router_cb, &hist)) {
                fprintf(stderr, "forward failed\n"); return 1;
            }
            total_tokens += n;
            printf("  forward %.1fs (%.2f tok/s)\n", now_s() - t, n / (now_s() - t));
        }
    }

    if (n_random) {
        for (uint32_t i = 0; i < n_random && i < tokcap; i++)
            tokbuf[i] = (uint32_t)(random() % info.n_vocab);
        size_t n = n_random;
        g_prog_t0 = now_s();
        fatcat_ml4_reset(m);   /* independent sequence */
        if (fatcat_ml4_forward(m, tokbuf, n, router_cb, &hist)) {
            fprintf(stderr, "forward failed\n"); return 1;
        }
        total_tokens += n;
        printf("random %zu tokens done\n", n);
    }

    if (check && total_tokens > 0) {
        size_t n = 0;
        if (tokens_file) {
            size_t len;
            char *ids = read_file(tokens_file, &len);
            if (ids) {
                char *p = ids;
                while (*p && n < tokcap) {
                    char *end;
                    long v = strtol(p, &end, 10);
                    if (end == p) break;
                    if (v >= 0 && (uint32_t)v < info.n_vocab) tokbuf[n++] = (uint32_t)v;
                    p = end;
                }
                free(ids);
            }
        } else if (n_text_files > 0) {
            size_t len;
            char *text = read_file(text_files[0], &len);
            if (text) { n = tokenize(&tk, text, len, tokbuf, tokcap); free(text); }
        }
        if (n) {
            float *lg = malloc((size_t)info.n_vocab * sizeof(float));
            fatcat_ml4_reset(m);   /* re-run the sequence from position 0 */
            if (lg && fatcat_ml4_last_logits(m, tokbuf, n, lg) == 0) {
                { FILE * fa = fopen("/tmp/opencode/fatcat_logits_full.bin", "wb");
                  if (fa) { fwrite(lg, 4, info.n_vocab, fa); fclose(fa); } }
                uint32_t top_id[32]; float top_v[32];
                for (int j = 0; j < 32; j++) {
                    uint32_t best = 0;
                    for (uint32_t e = 1; e < info.n_vocab; e++)
                        if (lg[e] > lg[best]) best = e;
                    top_id[j] = best; top_v[j] = lg[best];
                    lg[best] = -1e30f;
                }
                printf("\nnext-token prediction after %zu tokens:\n", n);
                for (int j = 0; j < 5; j++) {
                    if (need_tok) {
                        fatcat_str s = tk.tokens[top_id[j]];
                        printf("  %5u  %6.2f  '%.*s'\n", top_id[j], top_v[j],
                               (int)s.len, s.data);
                    } else {
                        printf("  %5u  %6.2f\n", top_id[j], top_v[j]);
                    }
                }
                fflush(stdout);
                {   /* machine-readable dump for cross-checking */
                    FILE *f = fopen("/tmp/opencode/fatcat_logits_top.bin", "wb");
                    if (f) {
                        for (int j = 0; j < 32; j++) {
                            fwrite(&top_id[j], 4, 1, f);
                            fwrite(&top_v[j], 4, 1, f);
                        }
                        fclose(f);
                    }
                }
            }
            free(lg);
        }
    }

    double dt = now_s() - t0;
    uint64_t ht = 0, hb = 0;
    fatcat_ml4_get_health(m, &ht, &hb);
    printf("\ntotal: %llu tokens in %.1fs (%.2f tok/s)\n",
           (unsigned long long)total_tokens, dt, total_tokens / (dt > 0 ? dt : 1));
    if (gpu) {
        uint64_t c = 0, b = 0, h = 0;
        fatcat_gpu_stats(&c, &b, &h);
        printf("gpu stats: copies=%llu copied=%.1f GB stage_hits=%llu upload_fails=%llu\n",
               (unsigned long long)c, (double)b / 1e9, (unsigned long long)h,
               (unsigned long long)fatcat_gpu_upload_fails());
    }
    printf("health: tokens=%llu nonfinite_tokens=%llu -> %s\n",
           (unsigned long long)ht, (unsigned long long)hb, hb ? "FAIL" : "OK");
    report(&hist, total_tokens);
    if (trace_path) trace_dump(trace_path, &hist);
    int rc = hb ? 4 : 0;
    printf("DONE rc=%d total_tokens=%llu elapsed_s=%.1f\n",
           rc, (unsigned long long)total_tokens, dt);
    fflush(stdout);

    free(tokbuf); free(hist.wsum); free(hist.raw); free(tk.hash);
    fatcat_ml4_close(m);
    fatcat_gguf_close(g);
    if (tg != g) fatcat_gguf_close(tg);
    fatcat_gpu_shutdown();
    return 0;
}
