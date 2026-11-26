/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat engine core implementation. See include/fatcat/engine.h.
 *
 * The engine owns the GGUF file, the ML4 model and a logits scratch row.
 * Generation is greedy (argmax); temperature/top-p sampling arrives with
 * the sampler work. Prefill and decode are timed separately so benchmarks
 * can report both rates. */
#define _GNU_SOURCE
#include "fatcat/engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fatcat/gguf.h"
#include "fatcat/ml4.h"
#include "fatcat/stream.h"

struct fatcat_engine {
    fatcat_gguf *g;
    fatcat_ml4 *m;
    float *logits;              /* n_vocab scratch */
    double prefill_s, decode_s;
    size_t n_prefill, n_decode;

    /* D3: prompt-lookup speculative decoding (greedy, lossless).
     * Drafts are the tokens following the LAST earlier occurrence of the
     * longest recent suffix of the history. Verification feeds
     * [true_next, draft...] in ONE batched forward and accepts the
     * longest prefix matching the model's own argmax rows, so the emitted
     * sequence is identical to plain greedy decode; rejected positions
     * are dropped by rewinding the position-indexed KV.
     * hist[] mirrors the fed token stream (hist.n == seq_len invariant).
     * pend[] holds accepted tokens not yet returned to the caller; the
     * LAST entry is never fed to the model (it is the correction/bonus
     * token the next real step will feed), the rest occupy KV slots. */
    int spec_on;
    size_t spec_draft_max, spec_ngram_min, spec_ngram_max;
    uint32_t *hist; size_t n_hist, hist_cap;
    uint32_t *pend; size_t pend_n, pend_i;
    uint32_t last_out;          /* last token returned by step (echo key) */
    int spec_fail, spec_cooldown; /* adaptive gate: back off after repeated
                                   * full draft rejections (prose has no
                                   * lookup signal; don't pay verification) */
    uint32_t *vbuf;             /* verification tokens (draft_max) */
    float *vrows;               /* verification logits (draft_max * n_vocab) */
    uint64_t spec_steps, spec_fed, spec_emitted;

    /* decode-window I/O probe: /proc/diskstats whole-disk sectors read,
     * sampled around the decode loop (ground truth NVMe throughput) */
    uint64_t io_sectors0;
    uint64_t io_majflt0, io_majflt;
    double io_t0;
    uint64_t io_bytes;
    uint64_t io_self0, io_self;
    double io_secs;
};

/* Sum 512B sectors read across whole-disk block devices (nvmeXnY, sdX,
 * mmcblkX; partitions excluded so nothing is double counted). */
static uint64_t diskstats_sectors(void)
{
    FILE *f = fopen("/proc/diskstats", "r");
    if (!f) return 0;
    char line[256];
    uint64_t tot = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned maj, min;
        char name[64];
        unsigned long long f1, f2, f3;
        if (sscanf(line, "%u %u %63s %llu %llu %llu",
                   &maj, &min, name, &f1, &f2, &f3) != 6) continue;
        size_t n = strlen(name);
        int whole = 0;
        if (strncmp(name, "nvme", 4) == 0) {          /* nvmeXnY */
            char *p = name + 4;
            while (*p >= '0' && *p <= '9') p++;
            if (*p == 'n') {
                p++;
                while (*p >= '0' && *p <= '9') p++;
                whole = (*p == '\0');
            }
        } else if (strncmp(name, "mmcblk", 6) == 0) { /* mmcblkX */
            char *p = name + 6;
            while (*p >= '0' && *p <= '9') p++;
            whole = (*p == '\0');
        } else if (name[0] == 's' && n <= 6) {        /* sdX */
            char *p = name + 2;
            whole = (name[1] == 'd');
            while (*p >= 'a' && *p <= 'z') p++;
            whole = whole && (*p == '\0');
        }
        if (whole) tot += f3;
    }
    fclose(f);
    return tot;
}

/* Device bytes read by THIS process only (block layer, not page cache):
 * separates engine I/O from background system noise in diskstats. */
static uint64_t self_io_read_bytes(void)
{
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[128];
    uint64_t v = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "read_bytes: %llu", (unsigned long long *)&v) == 1)
            break;
    fclose(f);
    return v;
}

/* Major faults = pages read FROM THE DEVICE outside read()/io_uring syscalls
 * (mmap page-cache refaults). The discriminator for the non-ring device
 * bytes: ring bytes + majflt*4KB should account for all device reads. */
static uint64_t self_majflt(void)
{
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    /* field 12 (majflt) after the comm field: parse past the last ')' */
    char *p = strrchr(buf, ')');
    if (!p) return 0;
    /* Empirically verified on this kernel (getrusage cross-check): after
     * comm, minflt is the 8th token and majflt the 9th. */
    unsigned long long minflt = 0, majflt = 0; (void)minflt;
    int got = sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %llu %llu",
                     &minflt, &majflt);
    if (got == 2)
        return majflt;
    return 0;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- D3: prompt-lookup speculation ---------------- */

static uint32_t argmax_row(const float *v, uint32_t n)
{
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static void hist_push(fatcat_engine *e, const uint32_t *t, size_t n)
{
    if (!e->spec_on) return;
    if (e->n_hist + n > e->hist_cap) {
        size_t cap = (e->n_hist + n) * 2 + 64;
        uint32_t *nh = realloc(e->hist, cap * sizeof *nh);
        if (!nh) return;      /* history lost: speculation degrades to off */
        e->hist = nh; e->hist_cap = cap;
    }
    memcpy(e->hist + e->n_hist, t, n * sizeof *t);
    e->n_hist += n;
}

/* Discard accepted-but-unemitted tokens. The last pending entry is the
 * unfed correction/bonus token (not in KV); everything after pend_i that
 * was fed must be rewound out of the KV. */
static void spec_drop_pending(fatcat_engine *e)
{
    if (e->pend_n <= e->pend_i) { e->pend_n = e->pend_i = 0; return; }
    size_t fed = e->pend_n - e->pend_i - 1;
    size_t seq = fatcat_ml4_seq_len(e->m);
    if (fed <= seq) fatcat_ml4_rewind(e->m, seq - fed);
    if (fed <= e->n_hist) e->n_hist -= fed;
    e->pend_n = e->pend_i = 0;
}

/* Longest suffix of hist with length in [ngram_min, ngram_max] that also
 * occurs earlier (overlap allowed: periodic output self-matches); copies
 * up to dmax tokens FOLLOWING that occurrence into D, returns the count. */
static size_t spec_lookup(const fatcat_engine *e, uint32_t *D, size_t dmax)
{
    const uint32_t *H = e->hist;
    size_t n = e->n_hist;
    if (n <= e->spec_ngram_min || !dmax) return 0;
    size_t gmax = e->spec_ngram_max < n - 1 ? e->spec_ngram_max : n - 1;
    for (size_t g = gmax; g >= e->spec_ngram_min; g--) {
        for (size_t q = n - g; q-- > 0; ) {
            if (memcmp(H + q, H + n - g, g * sizeof *H) == 0) {
                size_t m = n - (q + g);
                if (m > dmax) m = dmax;
                memcpy(D, H + q + g, m * sizeof *D);
                return m;
            }
        }
    }
    return 0;
}

static int plain_step(fatcat_engine *e, uint32_t token, uint32_t *next_out)
{
    int rc = fatcat_ml4_last_logits(e->m, &token, 1, e->logits);
    if (rc) return rc;
    hist_push(e, &token, 1);
    fatcat_ml4_info info = fatcat_ml4_get_info(e->m);
    *next_out = argmax_row(e->logits, info.n_vocab);
    if (e->spec_on) e->spec_emitted++;
    return 0;
}

static int spec_step(fatcat_engine *e, uint32_t token, uint32_t *next_out)
{
    fatcat_ml4_info info = fatcat_ml4_get_info(e->m);
    int rc = fatcat_ml4_last_logits(e->m, &token, 1, e->logits);
    if (rc) return rc;
    hist_push(e, &token, 1);
    uint32_t y1 = argmax_row(e->logits, info.n_vocab);

    /* draft from history; cap so the verification fits the KV capacity */
    size_t seq = fatcat_ml4_seq_len(e->m);
    size_t cap = fatcat_ml4_kv_cap(e->m);
    size_t room = cap > seq + 1 ? cap - seq - 1 : 0;
    size_t dmax = e->spec_draft_max < room ? e->spec_draft_max : room;
    size_t m = spec_lookup(e, e->vbuf, dmax);
    e->spec_steps++;
    /* adaptive gate: after SPEC_FAIL_GATE consecutive fully-rejected
     * verifications, stop drafting for SPEC_COOLDOWN steps (prose output
     * has no lookup signal; verification would be pure overhead), then
     * probe again. */
    if (m >= 2 && e->spec_cooldown > 0) { e->spec_cooldown--; m = 0; }
    if (m < 2) { *next_out = y1; e->spec_emitted++; return 0; }

    /* verify: feed [y1, D1..D_{m-1}] in ONE batched pass; row i = logits
     * after vbuf[i], bit-identical to sequential decode at that position */
    e->vbuf[0] = y1;
    rc = fatcat_ml4_forward_rows(e->m, e->vbuf, m, e->vrows);
    if (rc) return rc;
    hist_push(e, e->vbuf, m);
    e->spec_fed += m;

    /* accept the longest prefix; pend = accepted-and-fed matches plus the
     * final correction/bonus token (never fed, next step feeds it) */
    e->pend_n = e->pend_i = 0;
    int all = 1;
    for (size_t i = 1; i < m; i++) {
        uint32_t g = argmax_row(e->vrows + (i - 1) * (size_t)info.n_vocab,
                                info.n_vocab);
        e->pend[e->pend_n++] = g;
        if (g != e->vbuf[i]) { all = 0; break; }
    }
    if (all)
        e->pend[e->pend_n++] = argmax_row(e->vrows + (m - 1) * (size_t)info.n_vocab,
                                          info.n_vocab);

    /* gate bookkeeping: pend_n == 1 means every draft was wrong */
    if (e->pend_n == 1) {
        if (++e->spec_fail >= 4) { e->spec_cooldown = 8; e->spec_fail = 0; }
    } else {
        e->spec_fail = 0;
    }

    /* fed m at positions seq..seq+m-1; kept fed = pend_n (pend tail is
     * the unfed one); drop the rest */
    fatcat_ml4_rewind(e->m, seq + e->pend_n);
    e->n_hist = seq + e->pend_n;

    *next_out = y1;
    e->spec_emitted++;
    return 0;
}

/* ---------------- H2: oracle spec-decode benchmark ---------------- */

int fatcat_engine_spec_oracle(fatcat_engine *e, const uint32_t *prompt,
                              size_t n_prompt, size_t n,
                              const int *ks, int n_ks)
{
    if (!e || !prompt || !n || !ks || n_ks <= 0) return -1;
    fatcat_ml4_info info = fatcat_ml4_get_info(e->m);
    size_t cap = fatcat_ml4_kv_cap(e->m);
    if (n_prompt + n + 8 > cap) {
        fprintf(stderr, "spec-oracle: KV cap %zu < prompt %zu + n %zu\n",
                cap, n_prompt, n);
        return -1;
    }

    /* Phase A: greedy reference generation (warms the tier like a real run) */
    uint32_t *seq = malloc((n_prompt + n + 1) * sizeof *seq);
    if (!seq) return -1;
    memcpy(seq, prompt, n_prompt * sizeof *seq);
    size_t len = n_prompt;
    for (size_t i = 0; i < n; i++) {
        if (fatcat_ml4_last_logits(e->m, &seq[len - 1], 1, e->logits)) {
            free(seq); return -1;
        }
        seq[len++] = argmax_row(e->logits, info.n_vocab);
    }
    size_t n_gen = len - n_prompt;

    /* measurement window helper: wall time + device bytes */
    uint64_t sec0; double t0;
    double t_single = 0; double gb_single = 0; size_t tok_single = 0;

    /* Phase B1: single-token forwards (decode baseline), same KV start */
    fatcat_ml4_rewind(e->m, n_prompt);
    sec0 = diskstats_sectors(); t0 = now_s();
    for (size_t i = 0; i < n_gen; i++) {
        if (fatcat_ml4_forward(e->m, &seq[n_prompt + i], 1, NULL, NULL)) {
            free(seq); return -1;
        }
    }
    t_single = now_s() - t0; tok_single = n_gen;
    gb_single = (diskstats_sectors() - sec0) * 512.0 / 1e9;

    /* Phase B2: batched verification forwards of K tokens */
    int kmax = 0;
    for (int qi = 0; qi < n_ks; qi++)
        if (ks[qi] > kmax && ks[qi] <= 32) kmax = ks[qi];
    float *rows = malloc((size_t)kmax * (size_t)info.n_vocab * sizeof(float));
    if (!rows) { free(seq); return -1; }
    printf("spec-oracle: %zu ref tok, prompt %zu, vocab %u, Kmax %d\n",
           n_gen, n_prompt, info.n_vocab, ks[n_ks - 1]);
    printf("single-forward baseline: %.3f s/fwd  %.2f GB/fwd  -> %.2f tok/s\n",
           t_single / tok_single, gb_single / tok_single, tok_single / t_single);
    printf("     K   fwd_s   GB/fwd   R=t/T    E(a=.5) E(a=.7) E(a=.9)   tok/s a=.5/.7/.9\n");
    for (int qi = 0; qi < n_ks; qi++) {
        int K = ks[qi];
        if (K < 2 || K > 32) continue;
        fatcat_ml4_rewind(e->m, n_prompt);
        sec0 = diskstats_sectors(); t0 = now_s();
        int rc = 0; size_t fed = 0;
        while (fed + 1 < n_gen) {
            size_t k = n_gen - fed > (size_t)K ? (size_t)K : n_gen - fed;
            if (fatcat_ml4_forward_rows(e->m, &seq[n_prompt + fed], k, rows)) {
                rc = -1; break;
            }
            fed += k;
        }
        double t = now_s() - t0;
        if (rc) { free(rows); free(seq); return -1; }
        double T = t / (double)((fed + K - 1) / K);   /* avg time per fwd */
        double gb = (diskstats_sectors() - sec0) * 512.0 / 1e9;
        double R = T / (t_single / tok_single);
        printf("    %3d  %6.3f  %6.2f  %6.2f", K, T, gb / ((fed + K - 1) / K), R);
        double E3[3]; const double alph[3] = { 0.5, 0.7, 0.9 };
        for (int j = 0; j < 3; j++) {
            double a = alph[j];
            E3[j] = (1.0 - pow(a, K + 1)) / (1.0 - a);
            printf("  %8.2f", E3[j]);
        }
        printf("  ");
        for (int j = 0; j < 3; j++)
            printf(" %6.2f", E3[j] / T);
        printf("\n");
        fflush(stdout);
    }
    free(rows);
    fatcat_ml4_rewind(e->m, n_prompt);
    free(seq);
    return 0;
}

fatcat_engine *fatcat_engine_open(const fatcat_engine_config *cfg,
                                  char *errbuf, size_t errlen)
{
    if (!cfg || !cfg->model_path) {
        snprintf(errbuf, errlen, "engine: model_path required");
        return NULL;
    }
    fatcat_engine *e = calloc(1, sizeof *e);
    if (!e) { snprintf(errbuf, errlen, "engine: oom"); return NULL; }

    e->g = fatcat_gguf_open(cfg->model_path, errbuf, errlen);
    if (!e->g) { free(e); return NULL; }

    if (cfg->use_gpu >= 0) {
        /* translate the explicit choice onto the env the gpu layer reads */
        setenv("FATCAT_GPU", cfg->use_gpu ? "1" : "0", 1);
    }
    if (cfg->mem_limit_mb > 0) {
        char b[32]; snprintf(b, sizeof b, "%ld", cfg->mem_limit_mb);
        setenv("FATCAT_MEM_LIMIT_MB", b, 1);
    }
    if (cfg->mem_budget_mb > 0) {
        char b[32]; snprintf(b, sizeof b, "%ld", cfg->mem_budget_mb);
        setenv("FATCAT_MEM_BUDGET_MB", b, 1);
    }

    e->m = fatcat_ml4_open(e->g, errbuf, errlen);
    if (!e->m) { fatcat_gguf_close(e->g); free(e); return NULL; }

    fatcat_ml4_info info = fatcat_ml4_get_info(e->m);
    e->logits = malloc((size_t)info.n_vocab * sizeof(float));
    if (!e->logits) {
        snprintf(errbuf, errlen, "engine: oom logits");
        fatcat_ml4_close(e->m); fatcat_gguf_close(e->g); free(e);
        return NULL;
    }

    /* D3 prompt-lookup speculation (off by default until validated) */
    const char *sv = getenv("FATCAT_SPEC");
    e->spec_on = sv ? atoi(sv) != 0 : 0;
    const char *dv = getenv("FATCAT_SPEC_DRAFT");
    e->spec_draft_max = dv ? (size_t)atol(dv) : 8;
    const char *nv = getenv("FATCAT_SPEC_NGRAM_MIN");
    e->spec_ngram_min = nv ? (size_t)atol(nv) : 2;
    const char *xv = getenv("FATCAT_SPEC_NGRAM_MAX");
    e->spec_ngram_max = xv ? (size_t)atol(xv) : 8;
    if (e->spec_ngram_min < 1) e->spec_ngram_min = 1;
    if (e->spec_ngram_max < e->spec_ngram_min) e->spec_ngram_max = e->spec_ngram_min;
    if (e->spec_draft_max > 2) {
        e->pend = malloc((e->spec_draft_max + 1) * sizeof *e->pend);
        e->vbuf = malloc(e->spec_draft_max * sizeof *e->vbuf);
        e->vrows = malloc(e->spec_draft_max * (size_t)info.n_vocab * sizeof(float));
        if (!e->pend || !e->vbuf || !e->vrows) e->spec_on = 0;
    } else {
        e->spec_on = 0;
    }
    return e;
}

void fatcat_engine_close(fatcat_engine *e)
{
    if (!e) return;
    free(e->logits);
    free(e->hist); free(e->pend); free(e->vbuf); free(e->vrows);
    fatcat_ml4_close(e->m);
    fatcat_gguf_close(e->g);
    free(e);
}

void fatcat_engine_reset(fatcat_engine *e)
{
    fatcat_ml4_reset(e->m);
    e->n_hist = 0;
    e->pend_n = e->pend_i = 0;
    e->spec_fail = e->spec_cooldown = 0;
}

void fatcat_engine_context(const fatcat_engine *e, size_t *seq, size_t *cap)
{
    if (seq) *seq = fatcat_ml4_seq_len(e->m);
    if (cap) *cap = fatcat_ml4_kv_cap(e->m);
}

void fatcat_engine_set_router_cb(fatcat_engine *e, fatcat_router_cb cb, void *ctx)
{
    fatcat_ml4_set_router_cb(e->m, cb, ctx);
}

fatcat_ml4_info fatcat_engine_info(const fatcat_engine *e)
{
    return fatcat_ml4_get_info(e->m);
}

struct fatcat_stream *fatcat_engine_stream(const fatcat_engine *e)
{
    return fatcat_ml4_stream(e->m);
}

void fatcat_engine_prof(const fatcat_engine *e, double *attn, double *router,
                        double *shffn, double *routed, double *dense,
                        double *tok_total, uint64_t *tokens,
                        double *attn_core, double *attn_wout,
                        double *absorb, double *scan, double *wv,
                        double *r_slice, double *r_queue, double *r_wait,
                        double *r_act, double *r_pre)
{
    fatcat_ml4_get_prof(e->m, attn, router, shffn, routed, dense,
                        tok_total, tokens, attn_core, attn_wout,
                        absorb, scan, wv, r_slice, r_queue, r_wait,
                        r_act, r_pre);
}

int fatcat_engine_prefill(fatcat_engine *e, const uint32_t *tokens, size_t n)
{
    if (!n) return 0;
    if (e->spec_on) spec_drop_pending(e);   /* stale speculation, if any */
    double t0 = now_s();
    int rc = fatcat_ml4_forward(e->m, tokens, n, NULL, NULL);
    e->prefill_s += now_s() - t0;
    e->n_prefill += n;
    if (rc == 0) hist_push(e, tokens, n);
    return rc;
}

int fatcat_engine_step(fatcat_engine *e, uint32_t token, uint32_t *next_out)
{
    /* Echo contract: the caller feeds back the token step() last returned.
     * While a verification pass left accepted tokens in the KV, those are
     * served from the pending buffer without any forward pass. */
    if (e->spec_on && e->pend_n > e->pend_i && token == e->last_out) {
        *next_out = e->pend[e->pend_i++];
        e->last_out = *next_out;
        e->spec_emitted++;
        return 0;
    }
    if (e->spec_on && e->pend_n > e->pend_i)
        spec_drop_pending(e);   /* caller diverged mid-buffer */
    double t0 = now_s();
    int rc = e->spec_on ? spec_step(e, token, next_out)
                        : plain_step(e, token, next_out);
    if (rc == 0) e->last_out = *next_out;
    e->decode_s += now_s() - t0;
    e->n_decode++;
    return rc;
}

void fatcat_engine_flush(fatcat_engine *e)
{
    if (e->spec_on) spec_drop_pending(e);
}

void fatcat_engine_spec_stats(const fatcat_engine *e, int *on,
                              uint64_t *steps, uint64_t *fed, uint64_t *emitted)
{
    if (on) *on = e->spec_on;
    if (steps) *steps = e->spec_steps;
    if (fed) *fed = e->spec_fed;
    if (emitted) *emitted = e->spec_emitted;
}

/* expert-routing speculation accuracy (next-layer prefetch hit rate) */
void fatcat_engine_espec_stats(const fatcat_engine *e, uint64_t *total,
                               uint64_t *hit)
{
    fatcat_ml4_get_spec(e->m, total, hit);
}

/* H3 cache-aware tail-skip counters (see fatcat_ml4_get_skip). */
void fatcat_engine_skip_stats(const fatcat_engine *e, uint64_t *n, uint64_t *w)
{
    fatcat_ml4_get_skip(e->m, n, w);
}

/* Pista A staging counters (see fatcat_ml4_get_stage). */
void fatcat_engine_stage_stats(const fatcat_engine *e, uint64_t *slice_n,
                               uint64_t *slice_bytes, uint64_t *stage_n,
                               uint64_t *stage_bytes)
{
    fatcat_ml4_get_stage(e->m, slice_n, slice_bytes, stage_n, stage_bytes);
}

/* decode-window block-device reads (bytes, seconds) — ground-truth NVMe */
void fatcat_engine_io_stats(const fatcat_engine *e, uint64_t *bytes,
                            double *secs, uint64_t *self_bytes)
{
    if (bytes) *bytes = e->io_bytes;
    if (secs) *secs = e->io_secs;
    if (self_bytes) *self_bytes = e->io_self;
}

uint64_t fatcat_engine_majflt_bytes(const fatcat_engine *e)
{
    return e ? e->io_majflt : 0;
}

int fatcat_engine_generate(fatcat_engine *e, const uint32_t *prompt, size_t n_prompt,
                           size_t n_new, fatcat_gen_cb cb, void *ctx)
{
    if (!n_prompt || !n_new) { fprintf(stderr, "engine: empty prompt/n_new\n"); return -1; }
    fatcat_engine_reset(e);
    /* prefill all but the last prompt token; the last one is fed through the
     * normal decode step so its logits come out for free */
    if (n_prompt > 1 && fatcat_engine_prefill(e, prompt, n_prompt - 1) != 0)
        return -1;
    /* decode-only measurement window: stream counters and block-device
     * reads are sampled here so warm-tier stats and NVMe throughput
     * describe steady-state decode, not prefill's streaming churn */
    fatcat_stream_reset_stats(fatcat_ml4_stream(e->m));
    e->io_sectors0 = diskstats_sectors();
    e->io_self0 = self_io_read_bytes();
    e->io_majflt0 = self_majflt();
    e->io_t0 = now_s();

    uint32_t cur = prompt[n_prompt - 1];
    for (size_t i = 0; i < n_new; i++) {
        uint32_t next;
        if (fatcat_engine_step(e, cur, &next) != 0) return -1;
        if (cb) cb(ctx, next, i, n_new);
        cur = next;
    }
    e->io_bytes = (diskstats_sectors() - e->io_sectors0) * 512ull;
    e->io_self = self_io_read_bytes() - e->io_self0;
    e->io_majflt = (self_majflt() - e->io_majflt0) * 4096ull;
    e->io_secs = now_s() - e->io_t0;
    fatcat_engine_flush(e);   /* n_new cutoff: drop unemitted speculation */
    return 0;
}

void fatcat_engine_stats(const fatcat_engine *e, double *prefill_s,
                         double *decode_s, size_t *n_prefill, size_t *n_decode)
{
    if (prefill_s) *prefill_s = e->prefill_s;
    if (decode_s) *decode_s = e->decode_s;
    if (n_prefill) *n_prefill = e->n_prefill;
    if (n_decode) *n_decode = e->n_decode;
}
