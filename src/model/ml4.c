/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat ML4 forward pass. See include/fatcat/ml4.h.
 *
 * Memory model: weights are read through the GGUF reader's mmap pointers only;
 * nothing is bulk-copied into RAM. Per-token scratch is small; the KV cache is
 * seq_len * ~51 KB per KV-owning layer. OpenMP parallelizes the GEMVs.
 *
 * Numerics: scalar f32 throughout (quant weights dequantized on the fly).
 * This is the correctness reference; fast kernels come in P2/P4.
 *
 * RoPE follows ggml exactly: per (pos, pair i): theta_extrap = pos *
 * freq_base^(-2i/n_rot) / freq_factor[i], then YaRN(freq_scale=1/scaling_factor,
 * corr dims from original_context_length, ext_factor=1, mscale=attn_factor=1).
 *
 * DSA indexer (P5): full layers cache a 128-dim indexer key per token and,
 * once the sequence exceeds indexer.top_k, attend only to the top-k scored
 * positions (below that threshold selection = all = dense). Shared layers
 * (indexer.types[]==0) reuse the previous full layer's selection.
 * NoPE layers (rope_pattern=0) are not supported yet (slice is all-RoPE). */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fatcat/cpu.h"
#include "fatcat/dequant.h"
#include "fatcat/gguf.h"
#include "fatcat/gpu.h"
#include "fatcat/ml4.h"
#include "fatcat/stream.h"

#define ML4_MAX_HEADS   64
#define ML4_MAX_TOPK    8
#define ML4_MAX_EXPERTS 256

typedef struct {
    uint32_t type;
    const void *p;
    int64_t n_out;       /* output rows (dims[1]) */
    int64_t n_in;        /* quantized input elems per row (dims[0]) */
    int64_t row_bytes;   /* bytes per output row (row = n_in quantized elems) */
    void *dev;           /* device handle (GPU path) or NULL */
} wt_t;

typedef struct {
    /* attention (MLA) */
    wt_t wq;             /* [n_embd -> n_head*(qk_nope+qk_rope)] */
    wt_t wkv_a;          /* [n_embd -> kv_lora + qk_rope] */
    wt_t wk_b;           /* per head: [kv_lora -> qk_nope] */
    wt_t wv_b;           /* per head: [kv_lora -> v_head] */
    wt_t wout;           /* [n_head*v_head -> n_embd] */
    const float *norm_attn, *norm_kv_a;
    const float *ssmax_s, *ssmax_b;
    /* dense FFN (leading layers) */
    wt_t ffn_gate, ffn_up, ffn_down;
    /* MoE */
    bool is_moe;
    const float *router; /* F32 [n_embd -> n_experts] */
    wt_t sh_gate, sh_up, sh_down;
    wt_t ex_gate, ex_up, ex_down;   /* stacked experts; expert e at e*stride */
    int64_t ex_gate_stride;        /* bytes between experts (per tensor: */
    int64_t ex_up_stride;          /*  quant types may differ between */
    int64_t ex_down_stride;        /*  gate/up/down within a layer) */
    const float *norm_ffn;
    int kv_src;                    /* layer providing the KV cache */
    int kv_src_idx;                /* packed index of that layer among owners */
    bool rope;                     /* false = NoPE layer */
    /* P5 DSA lightning indexer (present on "full" indexer layers,
     * indexer.types[] == 1; shared layers reuse the previous selection) */
    bool idx_on;
    wt_t idx_wq, idx_wk, idx_proj;
    const float *idx_qn_w, *idx_qn_b;   /* LayerNorm(n_embd) on q input */
    const float *idx_kn_w, *idx_kn_b;   /* LayerNorm(idx_dk) on k output */
    const float *idx_ss_s, *idx_ss_b;   /* per-head SSMax */
    int idx_slot;                       /* packed index in the indexer k cache */
} ml4_layer;

struct fatcat_ml4 {
    fatcat_gguf *g;
    fatcat_ml4_info info;
    ml4_layer *layers;

    wt_t tok_embd;
    wt_t output;                   /* may be unused if logits not requested */
    const float *output_norm;
    const float *rope_freqs;       /* learned LeRope multipliers (n_rot/2) or NULL */
    float freq_base, freq_scale, eps;
    float logit_softcap;           /* final_logit_softcapping; 0 = disabled */
    bool kv_f16;                   /* store MLA KV cache as f16 (halves memory) */
    bool kv_hugepage;              /* madvise KV/indexer cache with MADV_HUGEPAGE */
    size_t kv_target;              /* pre-allocate KV to this many tokens (0 = grow by doubling) */
    float yarn_beta_fast, yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    int n_rot;                     /* rope dims per head (64) */
    uint32_t n_head, qk_nope, qk_rope, v_head, kv_lora;
    uint32_t n_ff_exp;             /* expert intermediate size */
    uint32_t n_ff;                 /* dense intermediate size */
    uint32_t indexer_top_k;

    /* P5 DSA lightning indexer state */
    uint32_t idx_nh, idx_dk;       /* indexer heads / head dim (32, 128) */
    float idx_eps;                 /* LayerNorm eps (layer_norm_epsilon) */
    int64_t idx_topk_eff;          /* effective top-k (FATCAT_IDX_TOPK dev override) */
    int idx_on;                    /* 0 = indexer disabled (FATCAT_IDX=0) */
    int n_idx_own;                 /* layers with their own indexer */
    float *ikv;                    /* indexer k cache: n_idx_own * ikv_cap * idx_dk */
    size_t ikv_cap;                /* kept equal to kv_cap */
    float *idx_scores;             /* selection scratch: kv_cap floats */
    int64_t *idx_sel;              /* selected positions (sorted asc): kv_cap */
    float *idx_qtmp;               /* idx_nh * idx_dk */
    float *idx_lntmp;              /* n_embd */
    float *idx_ktmp;               /* idx_dk */
    float *idx_wtmp;               /* idx_nh */

    /* KV cache (per KV-owning layer), grown on demand */
    fatcat_router_cb router_cb;      /* model-level default routing trace cb */
    void *router_cb_ctx;

    size_t seq_len;                /* positions already committed to KV */
    size_t kv_cap;
    float *kv;                     /* f32 KV storage: kv_cap * kv_stride per owner */
    uint16_t *kv_h;                /* f16 KV storage when kv_f16 (kv is NULL) */
    size_t kv_stride;              /* elements per position (f32 or f16 units) */
    size_t kv_off_ckv, kv_off_kpe, kv_off_kn, kv_off_v;

    /* progress hook (called once per processed token) */
    fatcat_token_cb token_cb;
    void *token_ctx;

    /* health counters: tokens processed + tokens that produced non-finite
     * (NaN/Inf) hidden states or logits. health_bad == 0 => math is healthy. */
    uint64_t health_tokens, health_bad;

    /* P3 1A router speculation: counters + enable flag (FATCAT_SPEC=0 off) */
    int spec_on;
    uint64_t spec_total, spec_hit;

    /* D1 batched prefill: chunk size in tokens (FATCAT_BATCH, 0 = off).
     * Layer-outer processing loads each expert once per chunk for all
     * tokens routed to it -> weight bytes/token drop by the reuse factor. */
    size_t batch_n;

    /* memory cap: once system memory used exceeds mem_limit_mb, streamed
     * expert pages are released (fadvise DONTNEED) after each use. 0 = no cap.
     * Configurable via FATCAT_MEM_LIMIT_MB (default 80000). */
    long mem_limit_mb;
    int drop_experts;

    /* hard cap on the engine's OWN anonymous memory (warm tier + KV/indexer
     * + reserve), FATCAT_MEM_BUDGET_GB / _MB; 0 = off. At open the warm
     * tier and KV target are planned to fit; at runtime KV growth shrinks
     * the warm tier and refuses growth that would exceed the cap. */
    long mem_budget_mb;
    size_t kv_tok_bytes;       /* KV+indexer bytes per token (all owners) */

    /* P4 v2: persistent forward scratch. When the GPU is up with unified
     * addressing it is device-mapped pinned memory, so async GEMV kernels
     * read x / write y in place (no staging copies). */
    float *scratch;
    size_t scratch_n;              /* floats */
    int scratch_dev;               /* 1 = device-mapped */
    int gpu_pipe_on;               /* FATCAT_GPU_PIPE=0 forces sync GEMVs */

    /* warm tier: expert slices cached in stream-managed slots (LFU+recency,
     * budget FATCAT_WARM_MB). Replaces the old fixed 24x10 MB LRU pool. */
    fatcat_stream *stream;

    /* D5a phase profiler (FATCAT_PROF=1): per-token seconds spent in the
     * sequential decode path's phase buckets; "other" = p_tok - sum. */
    int prof;
    double p_attn, p_router, p_shffn, p_routed, p_dense, p_tok;
    double p_core, p_wout;   /* attn split: attention_core / wout gemv */
    double p_absorb, p_scan, p_wv;  /* core split */
    double p_r_slice, p_r_queue, p_r_wait, p_r_act, p_r_pre; /* routed split:
                                                    * expert_slice stream waits,
                                                    * gemv launch/staging, pipe
                                                    * waits, CPU silu+combine,
                                                    * prefetch issue */
    int gpu_att;             /* D5e: GPU absorb/wv kernels on (FATCAT_GPU_ATT=0 off,
                              * 2 = GPU + CPU cross-check for validation) */
    int odirect;             /* D5f: O_DIRECT stream fills (FATCAT_ODIRECT=0 off) */
    int specpf;              /* D5f: speculative expert prefetch depth (top-k by
                              * router weight; FATCAT_SPECP, default 0). At
                              * 64 GB tiers speculation was speed-neutral and
                              * cost ~1 GB/token of NVMe (ring bytes were
                              * ~74% speculative at depth 4), so it is off
                              * by default; 1-2 may still help small tiers. */
    int pin_on;              /* D5f: pin hot per-token weights in RAM (FATCAT_PIN=0 off) */
    void **pins; size_t n_pins, pins_cap;   /* anonymous copies backing pinned weights */
    long pinned_mb;          /* total pinned bytes / MB (charged to the budget) */
    float *att_chk;              /* scratch for FATCAT_GPU_ATT=2 verification */
    int att_chk_logs;            /* mismatch lines printed so far */
    uint64_t att_chk_bad;        /* tokens*layers with any difference */
    uint64_t p_tokens;
};

/* ---------- resident-memory management ---------- */

/* Fixed reserve inside FATCAT_MEM_BUDGET_GB: forward scratch, logits,
 * indexer selection arrays, allocator slack. Measured baseline anonymous
 * memory (no warm tier, no KV) is ~0.9 GB on this model. */
#define MEM_RESERVE_MB 1024

/* system memory actually in use (MemTotal - MemAvailable), in MB. This
 * counts page cache + GPU (TTM) allocations, i.e. everything that matters
 * for keeping the machine responsive. */
static double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long mem_used_mb(void){
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    long total = 0, avail = 0, v;
    char nm[64], u[16];
    while (fscanf(f, "%63s %ld %15s", nm, &v, u) == 3) {
        if (!strcmp(nm, "MemTotal:")) total = v;
        else if (!strcmp(nm, "MemAvailable:")) avail = v;
    }
    fclose(f);
    return (total - avail) / 1024;
}

/* Release fully covered pages of [p, p+bytes) from the page cache. These are
 * clean file-backed pages: the next access simply re-faults them from NVMe. */
static void mem_drop(const void *p, size_t bytes)
{
    static long pg = 0;
    if (!pg) pg = sysconf(_SC_PAGESIZE);
    uintptr_t s = ((uintptr_t)p + pg - 1) & ~(uintptr_t)(pg - 1);
    uintptr_t e = ((uintptr_t)p + bytes) & ~(uintptr_t)(pg - 1);
    if (e > s) madvise((void *)s, e - s, MADV_DONTNEED);
}

/* ---------- small helpers ---------- */

static bool wt_load(fatcat_gguf *g, const char *name, wt_t *out, bool optional)
{
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, name);
    if (!t) return !optional ? false : true;
    out->type = t->type;
    out->p = fatcat_gguf_tensor_ptr(g, t);
    out->dev = NULL;
    {   /* output rows = product of all dims except dims[0] (the input dim);
         * 3-D tensors (attn_k_b/v_b: [kv_lora, per-head, n_head]) need the
         * full product, not just dims[1] */
        int64_t no = 1;
        for (uint32_t d = 1; d < t->n_dims; d++) no *= (int64_t)t->dims[d];
        out->n_out = no;
    }
    out->n_in = (int64_t)t->dims[0];   /* quantized input elems per row */
    out->row_bytes = (int64_t)fatcat_dequant_src_bytes(t->type, t->dims[0]);
    if (getenv("FATCAT_PINDBG") &&
        (uint64_t)out->n_out * (uint64_t)out->row_bytes != t->n_bytes)
        fprintf(stderr, "wtsize %s: n_out %lld rb %lld -> %llu != n_bytes %llu\n",
                name, (long long)out->n_out, (long long)out->row_bytes,
                (unsigned long long)(out->n_out * out->row_bytes),
                (unsigned long long)t->n_bytes);
    return out->p != NULL;
}

/* D5f: pin a weight into anonymous RAM. The per-token hot set (attention,
 * shared FFN, indexer, router, lm head) is read every token; keeping it in
 * anonymous pages makes it immune to the page-cache churn the expert
 * stream causes (those mmap'd pages were refaulting ~1.8 GB/token while
 * the NVMe ran at line rate). Copies are tracked in m->pins, freed at
 * close, and charged against the memory budget (warm tier shrinks). */
static void pin_wt(fatcat_ml4 *m, wt_t *w)
{
    if (!m->pin_on || !w->p || w->n_out <= 0 || w->row_bytes <= 0) return;
    size_t bytes = (size_t)w->n_out * (size_t)w->row_bytes;
    void *p = malloc(bytes);
    if (!p) return;                      /* OOM: leave on mmap, still correct */
    memcpy(p, w->p, bytes);
    if (m->n_pins == m->pins_cap) {
        size_t nc = m->pins_cap ? m->pins_cap * 2 : 64;
        void **np = realloc(m->pins, nc * sizeof *np);
        if (!np) { free(p); return; }
        m->pins = np; m->pins_cap = nc;
    }
    m->pins[m->n_pins++] = p;
    m->pinned_mb += (long)(bytes >> 20);
    if (getenv("FATCAT_PINDBG")) fprintf(stderr, "pinDBG %p %zu MB (n_out %lld row_bytes %lld)\n",
                                         (void*)w->p, bytes>>20, (long long)w->n_out, (long long)w->row_bytes);
    w->p = p;
}

static void pin_f32(fatcat_ml4 *m, const float **pp, size_t n)
{
    if (!m->pin_on || !*pp || !n) return;
    float *p = malloc(n * sizeof(float));
    if (!p) return;
    memcpy(p, *pp, n * sizeof(float));
    if (m->n_pins == m->pins_cap) {
        size_t nc = m->pins_cap ? m->pins_cap * 2 : 64;
        void **np = realloc(m->pins, nc * sizeof *np);
        if (!np) { free(p); return; }
        m->pins = np; m->pins_cap = nc;
    }
    m->pins[m->n_pins++] = p;
    m->pinned_mb += (long)(n * sizeof(float) >> 20);
    *pp = p;
}

/* Upload a loaded weight to the GPU (no-op when GPU is off or weight missing). */
static void wt_upload(fatcat_ml4 *m, wt_t *w, const char *name)
{
    if (!fatcat_gpu_enabled() || !w->p) return;
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(m->g, name);
    if (!t) return;
    uint64_t elems = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) elems *= t->dims[d];
    size_t bytes = (size_t)fatcat_dequant_src_bytes(t->type, elems);
    if (!bytes) return;
    w->dev = fatcat_gpu_upload(t->type, w->p, bytes);
    if (!w->dev && bytes >= (16u << 20))
        fprintf(stderr, "gpu_upload: %s (%.2f GB, type %u) failed -> CPU\n",
                name, bytes / 1e9, t->type);
    /* the mmap'd copy is dead weight once the device copy exists */
    if (w->dev) mem_drop(w->p, bytes);
}

static const float *f32_tensor(fatcat_gguf *g, const char *name)
{
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, name);
    if (!t || t->type != FATCAT_GGML_F32) return NULL;
    return fatcat_gguf_tensor_ptr(g, t);
}

static float kv_f32(fatcat_gguf *g, const char *key, float dflt)
{
    const fatcat_gguf_value *v = fatcat_gguf_find_kv(g, key);
    return (v && v->type == FATCAT_GGUF_F32) ? v->f32 : dflt;
}

static uint32_t kv_u32(fatcat_gguf *g, const char *key, uint32_t dflt)
{
    return (uint32_t)fatcat_gguf_kv_u64(g, key, dflt);
}

static inline float dot_f32(const float *a, const float *b, int64_t n)
{
#ifdef FATCAT_DOT_F64ACC
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
#else
    return fatcat_cpu_dot_f32(a, b, n);
#endif
}

static void rmsnorm(const float *x, const float *w, float eps, float *y, int64_t n)
{
#ifdef FATCAT_DOT_F64ACC
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n + eps));
    for (int64_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
#else
    fatcat_cpu_rmsnorm(x, w, eps, y, n);
#endif
}

/* LayerNorm (ggml LLM_NORM): (x-mean)/sqrt(var+eps)*w + b. Used by the DSA
 * indexer only, so plain scalar code. */
static void layernorm(const float *x, const float *w, const float *b, float eps,
                      float *y, int64_t n)
{
    double mean = 0.0, var = 0.0;
    for (int64_t i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    for (int64_t i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
    var /= (double)n;
    const float sc = 1.0f / sqrtf((float)var + eps);
    const float mu = (float)mean;
    for (int64_t i = 0; i < n; i++) y[i] = (x[i] - mu) * sc * w[i] + b[i];
}

/* indices of the k largest values of s[0..n) into out[], sorted ascending by
 * index. Deterministic quickselect (Hoare partition, middle pivot); ties are
 * resolved arbitrarily but deterministically. */
static int topk_cmp_idx_asc(const void *a, const void *b)
{
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}
void fatcat_topk_select(const float *s, int64_t n, int64_t k, int64_t *out)
{
    for (int64_t i = 0; i < n; i++) out[i] = i;
    if (k >= n) return;
    int64_t lo = 0, hi = n - 1;
    while (hi > lo) {
        const float piv = s[out[(lo + hi) / 2]];
        int64_t i = lo, j = hi;
        while (i <= j) {
            while (s[out[i]] > piv) i++;
            while (s[out[j]] < piv) j--;
            if (i <= j) { int64_t tmp = out[i]; out[i] = out[j]; out[j] = tmp; i++; j--; }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
    qsort(out, (size_t)k, sizeof(int64_t), topk_cmp_idx_asc);
}

/* y[out] = W @ x. W rows are quantized rows of n_in elems. GPU if uploaded,
 * else the CPU backend (AVX-512 fused dequant-GEMV, OpenMP inside). */
static void gemv(const wt_t *w, const float *x, float *y, int64_t n_out, int64_t n_in)
{
    if (w->dev && fatcat_gpu_gemv(w->dev, w->type, x, y, n_out, n_in))
        return;
    fatcat_cpu_gemv(w->type, w->p, x, y, n_out, n_in);
}

/* D4: batched GEMM for resident (uploaded) weights: one weight decode
 * serves n_tok token rows. Bit-identical to n_tok gemv() calls. */
static void gemm_res(const fatcat_ml4 *m, const wt_t *w, const float *X, float *Y,
                     int64_t n_out, int64_t n_in, int n_tok)
{
    (void)m;
    if (w->dev && fatcat_gpu_gemm(w->dev, w->type, X, Y, n_out, n_in, n_tok) == 0)
        return;
    fatcat_cpu_gemm(w->type, w->p, X, Y, n_out, n_in, n_tok);
}

/* Same but reading expert e from a stacked tensor at byte offset e*stride.
 * src points at the expert bytes (pool buffer from parallel prefetch, or the
 * mmap as fallback). GPU path stages the slice (tag = unique id for LRU).
 * Under the memory cap the expert's page-cache range is released after use. */
/* P4 zero-copy slot allocator for the warm tier (device-mapped pinned). */
static void *gpu_slot_alloc(size_t bytes, void *ctx)
{ (void)ctx; return fatcat_gpu_host_alloc(bytes); }
static void gpu_slot_free(void *p, size_t bytes, void *ctx)
{ (void)bytes; (void)ctx; fatcat_gpu_host_free(p); }

static void gemv_expert(fatcat_ml4 *m, const wt_t *w, const char *src, int64_t stride,
                        uint32_t e, uint64_t tag, const float *x, float *y,
                        int64_t n_out, int64_t n_in, int src_mapped, int async,
                        const float *xdev)
{
    int done = 0;
    if (fatcat_gpu_enabled()) {
        void *h = NULL;
        if (src_mapped)               /* P4 zero-copy: warm-tier slot is
                                       * device-mapped pinned memory */
            h = fatcat_gpu_wrap((void *)src, w->type);
        if (h) {
            done = async ? fatcat_gpu_gemv_dev(h, w->type, xdev ? xdev : x, y,
                                               n_out, n_in) == 0
                         : fatcat_gpu_gemv(h, w->type, x, y, n_out, n_in);
            fatcat_gpu_release(h);
        }
        /* Staging fallback: only when nothing is queued on the device,
         * because stage() may evict a slot a pending kernel is reading.
         * (async is only set when src_mapped, so no pending launches
         * from this path; the engine waits before any other GPU use.) */
        if (!done && !async) {
            h = fatcat_gpu_stage(w->type, src, (size_t)stride, tag);
            if (h) {
                done = fatcat_gpu_gemv(h, w->type, x, y, n_out, n_in);
                fatcat_gpu_release(h);
            }
        }
    }
    if (!done)
        fatcat_cpu_gemv(w->type, src, x, y, n_out, n_in);
    if (m->drop_experts)
        fatcat_gguf_drop_range(m->g, (const char *)w->p + (int64_t)e * stride,
                               (size_t)stride);
}

/* D1 batched expert GEMM: same GPU wrap->stage->CPU fallback ladder as
 * gemv_expert, but one weight decode serves n_tok token rows. Bit-identical
 * to n_tok gemv_expert calls (kernels preserve per-token op order). */
static void expert_gemm(fatcat_ml4 *m, const wt_t *w, const char *src, int64_t stride,
                        uint32_t e, uint64_t tag, const float *X, float *Y,
                        int64_t n_out, int64_t n_in, int src_mapped, int n_tok)
{
    int done = 0;
    if (fatcat_gpu_enabled()) {
        void *h = NULL;
        if (src_mapped)
            h = fatcat_gpu_wrap((void *)src, w->type);
        if (h) {
            done = fatcat_gpu_gemm(h, w->type, X, Y, n_out, n_in, n_tok) == 0;
            fatcat_gpu_release(h);
        }
        if (!done) {
            h = fatcat_gpu_stage(w->type, src, (size_t)stride, tag);
            if (h) {
                done = fatcat_gpu_gemm(h, w->type, X, Y, n_out, n_in, n_tok) == 0;
                fatcat_gpu_release(h);
            }
        }
    }
    if (!done)
        fatcat_cpu_gemm(w->type, src, X, Y, n_out, n_in, n_tok);
    if (m->drop_experts)
        fatcat_gguf_drop_range(m->g, (const char *)w->p + (int64_t)e * stride,
                               (size_t)stride);
}

/* f32 -> f16 (round-to-nearest-even), used for the f16 KV cache storage. */
static uint16_t f32_to_f16(float v)
{
    uint16_t h;
    int32_t bits; memcpy(&bits, &v, 4);
    uint32_t sign = ((uint32_t)bits >> 16) & 0x8000u;   /* already in f16 position */
    int32_t exp  = ((bits >> 23) & 0xFF) - (127 - 15);
    int32_t man  = bits & 0x7FFFFF;
    if (exp <= 0) {
        h = (uint16_t)sign;   /* flush subnormals to zero */
    } else if (exp >= 31) {
        h = (uint16_t)(sign | 0x7C00u);
    } else {
        uint32_t round = (uint32_t)man + 0xFFFu + ((man >> 13) & 1u);
        h = (uint16_t)(sign | ((uint32_t)exp << 10) | ((round >> 13) & 0x3FFu));
    }
    return h;
}

/* Opt-in transparent hugepages for the large KV/indexer caches: cuts TLB
 * pressure on the long-context attention scan. Off by default because under
 * memory pressure (defrag=madvise) fault-time compaction can stall. */
static void advise_hugepage(void *p, size_t bytes)
{
#ifdef MADV_HUGEPAGE
    if (!p || bytes < (1u << 20)) return;
    uintptr_t s = ((uintptr_t)p + 4095u) & ~(uintptr_t)4095u;
    uintptr_t e = ((uintptr_t)p + bytes) & ~(uintptr_t)4095u;
    if (e > s) madvise((void *)s, e - s, MADV_HUGEPAGE);
#else
    (void)p; (void)bytes;
#endif
}

/* Per-position rope cache: cos/sin pairs for n_rot dims (n_rot/2 pairs). */
static void rope_cache_pos(const fatcat_ml4 *m, int64_t pos, float *cs /*[n_rot]*/)
{
    const int n_rot = m->n_rot;
    const float theta_scale = powf(m->freq_base, -2.0f / (float)n_rot);
    /* YaRN correction dims (ggml_rope_yarn_corr_dims) */
    float corr[2];
    {
        double lb = log((double)m->freq_base);
        float d_fast = (float)(n_rot * log((double)m->yarn_orig_ctx / (2.0 * m->yarn_beta_fast * M_PI)) / (2.0 * lb));
        float d_slow = (float)(n_rot * log((double)m->yarn_orig_ctx / (2.0 * m->yarn_beta_slow * M_PI)) / (2.0 * lb));
        corr[0] = fmaxf(0.0f, floorf(d_fast));
        corr[1] = fminf((float)(n_rot - 1), ceilf(d_slow));
    }
    const float ext_factor = 1.0f;
    /* llama.cpp cancels the yarn magnitude mscale for this model:
     * yarn_attn_factor = 1/(1+0.1*log(factor)) (log_mul == 1 case, see
     * llama-context.cpp TAG_DEEPSEEK2_YARN_LOG_MUL_FIX) and rope_yarn()
     * multiplies it back by (1+0.1*log(1/freq_scale)). Net mscale = 1.0:
     * YaRN here is frequency-interpolation only, no magnitude scaling. */
    const float mscale = 1.0f;

    float theta = (float)pos;
    for (int i0 = 0; i0 < n_rot; i0 += 2) {
        float ff = m->rope_freqs ? m->rope_freqs[i0 / 2] : 1.0f;
        float theta_extrap = theta / ff;
        float t = m->freq_scale * theta_extrap;
        /* yarn ramp */
        float y = 1.0f - fminf(1.0f, fmaxf(0.0f, ((i0 / 2.0f) - corr[0]) / fmaxf(0.001f, corr[1] - corr[0])));
        float ramp = y * ext_factor;
        if (ramp != 0.0f)
            t = t * (1.0f - ramp) + theta_extrap * ramp;
        cs[i0 + 0] = cosf(t) * mscale;
        cs[i0 + 1] = sinf(t) * mscale;
        theta *= theta_scale;
    }
}

/* Rotate pairs in-place: vector of head_dim dims, rope on last n_rot dims. */
static void rope_apply(float *v, int64_t head_dim, int64_t n_rot, const float *cs)
{
    const float *x = v + head_dim - n_rot;
    float *dst = v + head_dim - n_rot;
    /* GGML_ROPE_TYPE_NORM: rotate adjacent pairs (2j, 2j+1) with freq j
     * (ggml rotate_pairs(n, n_offset=1, scale=1)), NOT the NEOX half-split. */
    for (int64_t i = 0; i < n_rot; i += 2) {
        float x0 = x[i], x1 = x[i + 1];
        dst[i]      = x0 * cs[i] - x1 * cs[i + 1];
        dst[i + 1]  = x0 * cs[i + 1] + x1 * cs[i];
    }
    (void)head_dim;
}
/* ---------- open / close ---------- */

static bool layer_load(fatcat_ml4 *m, int il)
{
    fatcat_gguf *g = m->g;
    ml4_layer *L = &m->layers[il];
    char nm[128];
#define TN(buf, fmt, name) snprintf(buf, sizeof(buf), "blk.%d." name, il)
    TN(nm, "", "attn_norm.weight");      L->norm_attn = f32_tensor(g, nm);
    TN(nm, "", "attn_q.weight");         if (!wt_load(g, nm, &L->wq, false)) return false;
    wt_upload(m, &L->wq, nm);
    /* attn_kv_a_mqa/attn_kv_a_norm exist only on KV-owning layers (checked later) */
    TN(nm, "", "attn_kv_a_mqa.weight");  wt_load(g, nm, &L->wkv_a, true);
    wt_upload(m, &L->wkv_a, nm);
    TN(nm, "", "attn_k_b.weight");       if (!wt_load(g, nm, &L->wk_b, false)) return false;
    wt_upload(m, &L->wk_b, nm);
    TN(nm, "", "attn_v_b.weight");       if (!wt_load(g, nm, &L->wv_b, false)) return false;
    wt_upload(m, &L->wv_b, nm);
    TN(nm, "", "attn_output.weight");    if (!wt_load(g, nm, &L->wout, false)) return false;
    wt_upload(m, &L->wout, nm);
    TN(nm, "", "attn_kv_a_norm.weight"); L->norm_kv_a = f32_tensor(g, nm);
    TN(nm, "", "attn_ssmax.weight");     L->ssmax_s = f32_tensor(g, nm);
    TN(nm, "", "attn_ssmax.bias");       L->ssmax_b = f32_tensor(g, nm);
    TN(nm, "", "ffn_norm.weight");       L->norm_ffn = f32_tensor(g, nm);
    if (!L->norm_attn || !L->ssmax_s || !L->ssmax_b || !L->norm_ffn)
        return false;

    /* P5 DSA indexer tensors: present exactly on the "full" indexer layers */
    L->idx_on = false;
    L->idx_slot = -1;
    {
        TN(nm, "", "indexer.attn_q_b.weight");
        if (fatcat_gguf_find_tensor(g, nm)) {
            if (!wt_load(g, nm, &L->idx_wq, false)) return false;
            TN(nm, "", "indexer.attn_k.weight");
            if (!wt_load(g, nm, &L->idx_wk, false)) return false;
            TN(nm, "", "indexer.proj.weight");
            if (!wt_load(g, nm, &L->idx_proj, false)) return false;
            TN(nm, "", "indexer.q_norm.weight"); L->idx_qn_w = f32_tensor(g, nm);
            TN(nm, "", "indexer.q_norm.bias");   L->idx_qn_b = f32_tensor(g, nm);
            TN(nm, "", "indexer.k_norm.weight"); L->idx_kn_w = f32_tensor(g, nm);
            TN(nm, "", "indexer.k_norm.bias");   L->idx_kn_b = f32_tensor(g, nm);
            TN(nm, "", "indexer.ssmax.weight");  L->idx_ss_s = f32_tensor(g, nm);
            TN(nm, "", "indexer.ssmax.bias");    L->idx_ss_b = f32_tensor(g, nm);
            if (!L->idx_qn_w || !L->idx_qn_b || !L->idx_kn_w || !L->idx_kn_b ||
                !L->idx_ss_s || !L->idx_ss_b) return false;
            L->idx_on = true;
        }
    }

    if ((uint32_t)il < m->info.n_dense_layers) {
        L->is_moe = false;
        TN(nm, "", "ffn_gate.weight"); if (!wt_load(g, nm, &L->ffn_gate, false)) return false;
        wt_upload(m, &L->ffn_gate, nm);
        TN(nm, "", "ffn_up.weight");   if (!wt_load(g, nm, &L->ffn_up, false)) return false;
        wt_upload(m, &L->ffn_up, nm);
        TN(nm, "", "ffn_down.weight"); if (!wt_load(g, nm, &L->ffn_down, false)) return false;
        wt_upload(m, &L->ffn_down, nm);
    } else {
        L->is_moe = true;
        TN(nm, "", "ffn_gate_inp.weight");  L->router = f32_tensor(g, nm);
        TN(nm, "", "ffn_gate_shexp.weight"); if (!wt_load(g, nm, &L->sh_gate, false)) return false;
        wt_upload(m, &L->sh_gate, nm);
        TN(nm, "", "ffn_up_shexp.weight");   if (!wt_load(g, nm, &L->sh_up, false)) return false;
        wt_upload(m, &L->sh_up, nm);
        TN(nm, "", "ffn_down_shexp.weight"); if (!wt_load(g, nm, &L->sh_down, false)) return false;
        wt_upload(m, &L->sh_down, nm);
        const fatcat_gguf_tensor *te;
#define LOAD_EXPS(field, stride, name)                                        \
        TN(nm, "", name); te = fatcat_gguf_find_tensor(g, nm);                \
        if (!te) return false;                                                \
        L->field.type = te->type; L->field.p = fatcat_gguf_tensor_ptr(g, te); \
        L->field.row_bytes =                                                  \
            (int64_t)fatcat_dequant_src_bytes(te->type, te->dims[0]);         \
        stride = (int64_t)fatcat_dequant_src_bytes(te->type,                  \
                                                   te->dims[0] * te->dims[1]);
        if (!L->router) return false;
        LOAD_EXPS(ex_gate, L->ex_gate_stride, "ffn_gate_exps.weight")
        LOAD_EXPS(ex_up,   L->ex_up_stride,   "ffn_up_exps.weight")
        LOAD_EXPS(ex_down, L->ex_down_stride, "ffn_down_exps.weight")
#undef LOAD_EXPS
    }
    return true;
#undef TN
}

fatcat_ml4 *fatcat_ml4_open(fatcat_gguf *g, char *errbuf, size_t errlen)
{
    fatcat_ml4 *m = calloc(1, sizeof(*m));
    m->mem_limit_mb = 80000;   /* keep total system usage under ~90 GB */
    { const char *env = getenv("FATCAT_MEM_LIMIT_MB");
      if (env) m->mem_limit_mb = atol(env); }
    m->mem_budget_mb = 0;
    { const char *e = getenv("FATCAT_MEM_BUDGET_MB");
      const char *g = getenv("FATCAT_MEM_BUDGET_GB");
      if (e) m->mem_budget_mb = atol(e);
      else if (g) m->mem_budget_mb = (long)(atof(g) * 1024.0); }
    m->spec_on = 1;
    { const char *sp = getenv("FATCAT_SPEC"); if (sp) m->spec_on = atoi(sp) != 0; }
    /* D1 batched prefill: chunk size in tokens (FATCAT_BATCH, 0 = off to
     * restore the sequential path). DEFAULT ON (Session 38): layer-outer
     * processing loads each expert once per chunk for all tokens routed
     * to it. Standalone GPU A/B @ 18 GB tier: chunk 512 = 3.41 tok/s,
     * chunk 128 = 2.94, sequential ~1.0; tier hit 100% vs 78.3%; logits
     * bit-identical. 256 balances the small-tier optimum (~128, where
     * the expert union thrashes tiny tiers) with the large-tier optimum
     * (>=512). */
    m->batch_n = 256;
    { const char *b = getenv("FATCAT_BATCH"); if (b) m->batch_n = (size_t)atol(b); }
    m->gpu_pipe_on = 1;
    { const char *pp = getenv("FATCAT_GPU_PIPE"); if (pp) m->gpu_pipe_on = atoi(pp) != 0; }
    { const char *pp = getenv("FATCAT_PROF"); m->prof = pp ? atoi(pp) != 0 : 0; }
    { const char *ga = getenv("FATCAT_GPU_ATT");    /* D5e GPU attention kernels */
      m->gpu_att = ga ? atoi(ga) : 1; }
    { const char *od = getenv("FATCAT_ODIRECT");    /* D5f page-cache bypass */
      m->odirect = od ? atoi(od) != 0 : 1; }
    { const char *sp = getenv("FATCAT_SPECP");      /* D5f spec-prefetch depth */
      m->specpf = sp ? atoi(sp) : 0;
      if (m->specpf < 0) m->specpf = 0; }   /* upper bound applied at call */
    { const char *pn = getenv("FATCAT_PIN");        /* D5f hot-weight pinning */
      if (pn) m->pin_on = atoi(pn) != 0;
      else m->pin_on = 0;   /* measured FLAT at 40 GB tier (refault theory
                             * disproved: the extra device traffic was not
                             * hot-weight refaults); costs 16 GB RAM + page
                             * cache pressure, so opt-in only */ }
    m->stream = NULL;
    if (!m) { snprintf(errbuf, errlen, "oom"); return NULL; }
    m->g = g;
    m->kv_f16 = getenv("FATCAT_KV_F16") != NULL;
    m->kv_hugepage = getenv("FATCAT_KV_HUGEPAGE") != NULL;
    /* Pre-allocate the KV/indexer caches to a fixed token capacity instead of
     * lazy doubling, which overshoots up to 2x the sequence (at 256K f32 that
     * is ~57 GB instead of 38 GB).  The GGUF context_length is unusable here
     * (this model declares 1M tokens = 154 GB f32), so the target must be
     * explicit: FATCAT_KV_TARGET=<tokens>.  calloc is lazy, so untouched
     * capacity costs no RSS. */
    m->kv_target = 0;
    { const char *env = getenv("FATCAT_KV_TARGET");
      if (env) { long t = atol(env); if (t > 0) m->kv_target = (size_t)t; } }

    fatcat_str archv = fatcat_gguf_kv_str(g, "general.architecture",
                                          (fatcat_str){ "mistral-large4", 14 });
    char arch[64];
    snprintf(arch, sizeof(arch), "%.*s", (int)archv.len, archv.data);
#define KA(field, fmt) snprintf(field, sizeof(field), "%s." fmt, arch)
    char k[128];
    KA(k, "block_count");                m->info.n_layers = kv_u32(g, k, 0);
    KA(k, "leading_dense_block_count");  m->info.n_dense_layers = kv_u32(g, k, 0);
    KA(k, "expert_count");               m->info.n_experts = kv_u32(g, k, 0);
    KA(k, "expert_used_count");          m->info.n_experts_used = kv_u32(g, k, 0);
    KA(k, "embedding_length");           m->info.n_embd = kv_u32(g, k, 0);
    KA(k, "vocab_size");                 m->info.n_vocab = kv_u32(g, k, 0);
    KA(k, "feed_forward_length");        m->n_ff = kv_u32(g, k, 0);
    KA(k, "expert_feed_forward_length"); m->n_ff_exp = kv_u32(g, k, 0);
    KA(k, "attention.head_count");       m->n_head = kv_u32(g, k, 1);
    KA(k, "attention.key_length_mla");   m->qk_nope = 0; m->qk_rope = 0;
    {
        uint32_t klen = kv_u32(g, k, 128); /* qk_nope + qk_rope */
        KA(k, "rope.dimension_count");
        m->n_rot = (int)kv_u32(g, k, 64);
        m->qk_rope = m->n_rot;
        m->qk_nope = klen - m->qk_rope;
    }
    KA(k, "attention.value_length_mla"); m->v_head = kv_u32(g, k, 128);
    KA(k, "attention.kv_lora_rank");     m->kv_lora = kv_u32(g, k, 512);
    KA(k, "rope.freq_base");             m->freq_base = kv_f32(g, k, 10000.0f);
    /* RMS norms use layer_norm_rms_epsilon (1e-5); the indexer LayerNorms use
     * layer_norm_epsilon (1e-6) — see llama.cpp mistral-large4.cpp lines 8-9. */
    KA(k, "attention.layer_norm_rms_epsilon"); m->eps = kv_f32(g, k, 1e-5f);
    KA(k, "rope.scaling.factor");        m->freq_scale = 1.0f / kv_f32(g, k, 1.0f);
    KA(k, "rope.scaling.original_context_length"); m->yarn_orig_ctx = kv_u32(g, k, m->info.n_layers ? 4096 : 4096);
    KA(k, "rope.scaling.yarn_beta_fast"); m->yarn_beta_fast = kv_f32(g, k, 32.0f);
    KA(k, "rope.scaling.yarn_beta_slow"); m->yarn_beta_slow = kv_f32(g, k, 1.0f);
    KA(k, "attention.indexer.top_k");    m->indexer_top_k = kv_u32(g, k, 2048);
    KA(k, "attention.indexer.head_count"); m->idx_nh = kv_u32(g, k, 0);
    KA(k, "attention.indexer.key_length"); m->idx_dk = kv_u32(g, k, 0);
    KA(k, "attention.layer_norm_epsilon"); m->idx_eps = kv_f32(g, k, 1e-6f);
    KA(k, "final_logit_softcapping");    m->logit_softcap = kv_f32(g, k, 0.0f);
#undef KA
    if (!m->info.n_layers || !m->info.n_embd) {
        snprintf(errbuf, errlen, "missing ML4 KVs (arch='%s'?)", arch);
        free(m); return NULL;
    }
    if (m->info.n_experts_used > ML4_MAX_TOPK || m->info.n_experts > ML4_MAX_EXPERTS ||
        m->n_head > ML4_MAX_HEADS) {
        snprintf(errbuf, errlen, "ML4 dims exceed compile-time caps"); free(m); return NULL;
    }

    m->rope_freqs = f32_tensor(g, "rope_freqs.weight");
    m->output_norm = f32_tensor(g, "output_norm.weight");
    if (!m->output_norm) { snprintf(errbuf, errlen, "missing output_norm"); free(m); return NULL; }

    /* P4: GPU acceleration (gfx1151). Must init before weight uploads.
     * FATCAT_GPU_SLOTS = device staging slots for streamed experts
     * (10 MB each, GTT); 0 or FATCAT_GPU=0 keeps everything on CPU. */
    {
        long slots = 48;
        const char *env = getenv("FATCAT_GPU_SLOTS");
        if (env) slots = atol(env);
        if (slots > 0) {
            char gerr[256] = "";
            if (!fatcat_gpu_init((uint32_t)slots, gerr, sizeof gerr))
                fprintf(stderr, "ml4: GPU unavailable (%s); CPU path\n", gerr);
        }
    }

    if (!wt_load(g, "token_embd.weight", &m->tok_embd, false)) {
        snprintf(errbuf, errlen, "missing token_embd"); free(m); return NULL;
    }
    wt_load(g, "output.weight", &m->output, true);  /* optional: only for logits */

    m->layers = calloc(m->info.n_layers, sizeof(ml4_layer));
    if (!m->layers) { snprintf(errbuf, errlen, "oom"); free(m); return NULL; }

    /* per-layer rope flag + KV source map */
    const fatcat_gguf_value *v_src;
    snprintf(k, sizeof(k), "%s.attention.kv_source_map", arch);
    v_src = fatcat_gguf_find_kv(g, k);
    snprintf(k, sizeof(k), "%s.attention.rope_pattern", arch);
    const fatcat_gguf_value *v_rope = fatcat_gguf_find_kv(g, k);

    for (uint32_t il = 0; il < m->info.n_layers; il++) {
        if (!layer_load(m, (int)il)) {
            snprintf(errbuf, errlen, "layer %u tensors missing", il);
            free(m->layers); free(m); return NULL;
        }
        ml4_layer *L = &m->layers[il];
        L->kv_src = (int)il;
        if (v_src && v_src->type == FATCAT_GGUF_ARR && v_src->arr.count > il &&
            (v_src->arr.elem_type == FATCAT_GGUF_U32 || v_src->arr.elem_type == FATCAT_GGUF_I32)) {
            const uint32_t *arr = v_src->arr.elems; /* U32/I32 array */
            L->kv_src = (int)arr[il];
        }
        L->rope = true;
        if (v_rope && v_rope->type == FATCAT_GGUF_ARR && v_rope->arr.count > il &&
            v_rope->arr.elem_type == FATCAT_GGUF_BOOL) {
            const uint8_t *arr = v_rope->arr.elems;
            L->rope = arr[il] != 0;
        }
        if (L->kv_src > (int)il) {
            snprintf(errbuf, errlen, "layer %u: KV source %d must be <= self", il, L->kv_src);
            free(m->layers); free(m); return NULL;
        }
        if (L->kv_src == (int)il && (!L->wkv_a.p || !L->norm_kv_a)) {
            snprintf(errbuf, errlen, "layer %u: KV owner missing attn_kv_a tensors", il);
            free(m->layers); free(m); return NULL;
        }
    }
    /* packed KV slot index per layer */
    {
        int *omap = malloc(m->info.n_layers * sizeof(int));
        if (!omap) { snprintf(errbuf, errlen, "oom"); free(m->layers); free(m); return NULL; }
        int n_own = 0;
        for (uint32_t il = 0; il < m->info.n_layers; il++)
            if (m->layers[il].kv_src == (int)il) omap[il] = n_own++;
        for (uint32_t il = 0; il < m->info.n_layers; il++)
            m->layers[il].kv_src_idx = omap[m->layers[il].kv_src];
        free(omap);
    }
    /* P5 DSA indexer: enable when config + tensors exist; packed slot per
     * full indexer layer. FATCAT_IDX=0 disables (A/B), FATCAT_IDX_TOPK
     * overrides top-k (dev: exercise selection on short sequences). */
    m->idx_on = (m->idx_nh && m->idx_dk && m->indexer_top_k) ? 1 : 0;
    { const char *e = getenv("FATCAT_IDX"); if (e) m->idx_on = atoi(e) != 0; }
    m->idx_topk_eff = (int64_t)m->indexer_top_k;
    { const char *e = getenv("FATCAT_IDX_TOPK"); if (e) m->idx_topk_eff = atoll(e); }
    if (m->idx_on) {
        int n = 0;
        for (uint32_t il = 0; il < m->info.n_layers; il++)
            if (m->layers[il].idx_on) m->layers[il].idx_slot = n++;
        if (!n) m->idx_on = 0;
        m->n_idx_own = n;
        m->idx_qtmp = malloc((size_t)m->idx_nh * m->idx_dk * sizeof(float));
        m->idx_lntmp = malloc((size_t)m->info.n_embd * sizeof(float));
        m->idx_ktmp = malloc((size_t)m->idx_dk * sizeof(float));
        m->idx_wtmp = malloc((size_t)m->idx_nh * sizeof(float));
        if (!m->idx_qtmp || !m->idx_lntmp || !m->idx_ktmp || !m->idx_wtmp)
            m->idx_on = 0;   /* graceful degradation: dense attention */
    }

    /* KV+indexer bytes per token (all owners), known only now that the
     * layer geometry is final. Used by the FATCAT_MEM_BUDGET_GB planner
     * and the runtime growth guard. */
    {
        size_t n_own = 0;
        for (uint32_t il = 0; il < m->info.n_layers; il++)
            if (m->layers[il].kv_src == (int)il) n_own++;
        m->kv_tok_bytes = n_own * (m->kv_lora + m->qk_rope)
                          * (size_t)(m->kv_f16 ? 2 : 4);
        if (m->idx_on)
            m->kv_tok_bytes += (size_t)m->n_idx_own * m->idx_dk * 4 + 12;
    }

    /* D5f: pin the per-token hot set into anonymous RAM (see pin_wt). */
    if (m->pin_on) {
        for (uint32_t il = 0; il < m->info.n_layers; il++) {
            ml4_layer *L = &m->layers[il];
            pin_wt(m, &L->wq); pin_wt(m, &L->wkv_a);
            pin_wt(m, &L->wk_b); pin_wt(m, &L->wv_b); pin_wt(m, &L->wout);
            if (L->is_moe) {
                pin_wt(m, &L->sh_gate); pin_wt(m, &L->sh_up); pin_wt(m, &L->sh_down);
            } else {
                pin_wt(m, &L->ffn_gate); pin_wt(m, &L->ffn_up); pin_wt(m, &L->ffn_down);
            }
            if (L->idx_on) { pin_wt(m, &L->idx_wq); pin_wt(m, &L->idx_wk);
                             pin_wt(m, &L->idx_proj); }
            if (L->router)
                pin_f32(m, &L->router, (size_t)m->info.n_embd * m->info.n_experts);
        }
        pin_wt(m, &m->output);
        fprintf(stderr, "pin: %.2f GB hot weights pinned to RAM (FATCAT_PIN=0 off)\n",
                m->pinned_mb / 1024.0);
    }

    {   /* warm tier budget: 32 GB default (policy doc: ~62% hit rate);
         * FATCAT_WARM_MB=0 disables (mmap fallback, page-cache dependent).
         * Under FATCAT_MEM_BUDGET_GB (hard cap on the engine's own memory,
         * distinct from the system-wide FATCAT_MEM_LIMIT_MB) the KV target
         * is clamped to fit and the tier gets what remains; KV growth at
         * runtime then shrinks the tier (forward_int) to stay inside. */
        long warm_mb = 32768;
        const char *env = getenv("FATCAT_WARM_MB");
        if (env) warm_mb = atol(env);
        if (m->mem_budget_mb > 0) {
            long kv_need_mb = 0;
            if (m->kv_target && m->kv_tok_bytes)
                kv_need_mb = (long)((double)m->kv_target * m->kv_tok_bytes
                                    / (1024.0 * 1024.0)) + 1;
            long kv_room = m->mem_budget_mb - MEM_RESERVE_MB;
            if (kv_room < 0) kv_room = 0;
            if (kv_need_mb > kv_room && m->kv_tok_bytes) {
                size_t t = (size_t)((double)kv_room * 1024.0 * 1024.0
                                    / (double)m->kv_tok_bytes);
                fprintf(stderr, "ml4: budget %ld MB: KV target %zu -> %zu tokens\n",
                        m->mem_budget_mb, m->kv_target, t);
                m->kv_target = t;
                kv_need_mb = kv_room;
            }
            long warm_room = m->mem_budget_mb - MEM_RESERVE_MB - kv_need_mb
                             - m->pinned_mb;
            if (warm_mb > warm_room) {
                fprintf(stderr, "ml4: budget %ld MB: warm tier %ld -> %ld MB\n",
                        m->mem_budget_mb, warm_mb, warm_room < 0 ? 0L : warm_room);
                warm_mb = warm_room;
            }
            if (warm_mb < 0) warm_mb = 0;
        }
        if (warm_mb > 0) {
            char serr[128] = "";
            m->stream = fatcat_stream_open((size_t)warm_mb, 10u << 20,
                                           serr, sizeof serr);
            if (!m->stream)
                fprintf(stderr, "ml4: warm tier disabled (%s)\n", serr);
            /* P4 zero-copy: allocate tier slots as device-mapped pinned
             * memory so the GPU reads streamed experts in place (no GTT
             * copies). Falls back to plain mmap when unavailable. */
            else if (fatcat_gpu_enabled() && fatcat_gpu_unified())
                fatcat_stream_set_allocator(m->stream, gpu_slot_alloc,
                                            gpu_slot_free, NULL);
        }
    }
    return m;
}

void fatcat_ml4_close(fatcat_ml4 *m)
{
    if (!m) return;
    fatcat_stream_close(m->stream);
    free(m->ikv); free(m->idx_scores); free(m->idx_sel);
    free(m->idx_qtmp); free(m->idx_lntmp); free(m->idx_ktmp); free(m->idx_wtmp);
    free(m->att_chk);
    for (size_t i = 0; i < m->n_pins; i++) free(m->pins[i]);
    free(m->pins);
    if (m->scratch) {
        if (m->scratch_dev) fatcat_gpu_host_free(m->scratch);
        else free(m->scratch);
    }
    fatcat_gpu_shutdown();
    free(m->layers);
    free(m->kv);
    free(m->kv_h);
    free(m);
}

void fatcat_ml4_set_token_cb(fatcat_ml4 *m, fatcat_token_cb cb, void *ctx)
{
    m->token_cb = cb;
    m->token_ctx = ctx;
}

void fatcat_ml4_get_health(const fatcat_ml4 *m, uint64_t *tokens, uint64_t *bad)
{
    if (tokens) *tokens = m->health_tokens;
    if (bad) *bad = m->health_bad;
}

void fatcat_ml4_get_prof(const fatcat_ml4 *m, double *attn, double *router,
                         double *shffn, double *routed, double *dense,
                         double *tok_total, uint64_t *tokens,
                         double *attn_core, double *attn_wout,
                         double *absorb, double *scan, double *wv,
                         double *r_slice, double *r_queue, double *r_wait,
                         double *r_act, double *r_pre)
{
    if (attn) *attn = m->p_attn;
    if (router) *router = m->p_router;
    if (shffn) *shffn = m->p_shffn;
    if (routed) *routed = m->p_routed;
    if (dense) *dense = m->p_dense;
    if (tok_total) *tok_total = m->p_tok;
    if (tokens) *tokens = m->p_tokens;
    if (attn_core) *attn_core = m->p_core;
    if (attn_wout) *attn_wout = m->p_wout;
    if (absorb) *absorb = m->p_absorb;
    if (scan) *scan = m->p_scan;
    if (wv) *wv = m->p_wv;
    if (r_slice) *r_slice = m->p_r_slice;
    if (r_queue) *r_queue = m->p_r_queue;
    if (r_wait) *r_wait = m->p_r_wait;
    if (r_act) *r_act = m->p_r_act;
    if (r_pre) *r_pre = m->p_r_pre;
}

fatcat_stream *fatcat_ml4_stream(fatcat_ml4 *m) { return m->stream; }

void fatcat_ml4_get_spec(const fatcat_ml4 *m, uint64_t *total, uint64_t *hit)
{
    if (total) *total = m->spec_total;
    if (hit) *hit = m->spec_hit;
}

fatcat_ml4_info fatcat_ml4_get_info(const fatcat_ml4 *m) { return m->info; }
/* ---------- forward pass ---------- */

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* Resolve expert slice (layer_mat = il*3+t): warm tier first — blocks only
 * on this slice's in-flight prefetch — falling back to the mmap view. On a
 * cold fill the file range is dropped from the page cache (the slot owns the
 * bytes). Sets *key_out to the tier key when referenced (0 = mmap fallback). */
static const char *expert_slice(fatcat_ml4 *m, const wt_t *w, int64_t stride,
                                uint32_t expert, uint64_t layer_mat,
                                uint64_t *key_out)
{
    const char *base = (const char *)w->p + (int64_t)expert * stride;
    *key_out = 0;
    int fd = -1; uint64_t off = 0;
    if (m->stream && fatcat_gguf_resolve(m->g, base, &fd, &off)) {
        if (m->odirect && ((off | (uint64_t)stride) & 511) == 0) {
            int dfd; uint64_t doff;
            if (fatcat_gguf_resolve_direct(m->g, base, &dfd, &doff) &&
                doff == off) fd = dfd;
        }
        uint64_t key = layer_mat * m->info.n_experts + expert + 1;
        int hit = 0;
        const char *p = fatcat_stream_get(m->stream, key, fd, off,
                                          (size_t)stride, &hit);
        if (p) {
            if (!hit)
                fatcat_gguf_drop_range(m->g, base, (size_t)stride);
            *key_out = key;
            return p;
        }
    }
    return base;
}

/* router: softmax -> top-k -> renormalize. Fills ids[] (+ wts[] if given).
 * logits is scratch of size n_experts. Returns topk count. */
static uint32_t router_topk(const fatcat_ml4 *m, const float *rw,
                            const float *xn, float *logits,
                            uint16_t *ids, float *wts)
{
    const uint32_t ne = m->info.n_experts, topk = m->info.n_experts_used;
    const int64_t n_embd = m->info.n_embd;
    for (uint32_t e = 0; e < ne; e++)
        logits[e] = dot_f32(rw + (int64_t)e * n_embd, xn, n_embd);
    float mx = logits[0], sum = 0.0f;
    for (uint32_t e = 1; e < ne; e++) mx = fmaxf(mx, logits[e]);
    for (uint32_t e = 0; e < ne; e++) { logits[e] = expf(logits[e] - mx); sum += logits[e]; }
    for (uint32_t e = 0; e < ne; e++) logits[e] /= sum;
    float wsum = 0.0f;
    for (uint32_t k = 0; k < topk; k++) {
        uint32_t best = 0;
        for (uint32_t e = 1; e < ne; e++)
            if (logits[e] > logits[best]) best = e;
        ids[k] = (uint16_t)best;
        if (wts) { wts[k] = logits[best]; wsum += logits[best]; }
        logits[best] = -1.0f;
    }
    if (wts)
        for (uint32_t k = 0; k < topk; k++) wts[k] /= wsum;  /* weights_norm */
    return topk;
}

/* Queue async reads for expert set ids[] of layer il (all 3 matrices).
 * spec: 0 = needed now (critical queue), 1 = next-layer guess (low queue:
 * drains only after all critical reads, so wrong guesses never delay). */
static void prefetch_experts(fatcat_ml4 *m, uint32_t il, const ml4_layer *L,
                             const uint16_t *ids, uint32_t n, int spec)
{
    const wt_t *w3[3] = { &L->ex_gate, &L->ex_up, &L->ex_down };
    const int64_t st3[3] = { L->ex_gate_stride, L->ex_up_stride,
                             L->ex_down_stride };
    for (uint32_t k = 0; k < n; k++)
        for (int t = 0; t < 3; t++) {
            const char *base = (const char *)w3[t]->p
                             + (int64_t)ids[k] * st3[t];
            int fd = -1; uint64_t off = 0;
            if (fatcat_gguf_resolve(m->g, base, &fd, &off)) {
                if (m->odirect && ((off | (uint64_t)st3[t]) & 511) == 0) {
                    int dfd; uint64_t doff;
                    if (fatcat_gguf_resolve_direct(m->g, base, &dfd, &doff) &&
                        doff == off) fd = dfd;
                }
                fatcat_stream_prefetch(
                    m->stream,
                    ((uint64_t)il * 3 + t) * m->info.n_experts + ids[k] + 1,
                    fd, off, (size_t)st3[t], spec);
            }
        }
}

/* Absorbed MLA for one query token t. Writes KV for position t if owner. */
/* per-forward, per-token DSA selection state carried across layers: shared
 * indexer layers (types[]==0) reuse the previous full layer's selection. */
struct ml4_idx { int64_t n_sel; };  /* 0 = dense (attend to all positions) */

/* DSA lightning indexer for one full layer at position t. Always caches the
 * indexer key; once the sequence exceeds top-k, computes scores and selects
 * the top-k positions into m->idx_sel (ist->n_sel). Semantics mirror
 * llama.cpp's mistral-large4 graph: LayerNorm'd q input, per-head SSMax,
 * RoPE on the first n_rot dims ([rope|nope] head layout), and
 * score(p) = sum_h relu(q_h . k_p) * w_h with w pre-scaled by 1/sqrt(nh*dk).
 * The reference's Hadamard rotation is applied to BOTH q and k, so it is
 * score-invariant (orthonormal transform) and skipped here. */
static void idx_select_layer(const fatcat_ml4 *m, const ml4_layer *L,
                             struct ml4_idx *ist, const float *xn,
                             const float *cs, int64_t t)
{
    const int64_t n_embd = m->info.n_embd;
    const uint32_t nh = m->idx_nh, dk = m->idx_dk;

    /* key side: computed for every token (needed once the cache outgrows top-k) */
    float *ki = m->idx_ktmp;
    gemv(&L->idx_wk, xn, ki, (int64_t)dk, n_embd);
    layernorm(ki, L->idx_kn_w, L->idx_kn_b, m->idx_eps, ki, (int64_t)dk);
    if (L->rope) rope_apply(ki, m->n_rot, m->n_rot, cs);
    float *kc = m->ikv + ((size_t)L->idx_slot * m->ikv_cap + (size_t)t) * dk;
    memcpy(kc, ki, (size_t)dk * sizeof(float));

    ist->n_sel = 0;   /* dense while the whole cache fits in top-k */
    if (t + 1 <= m->idx_topk_eff) return;

    /* query side */
    float *q = m->idx_qtmp;
    layernorm(xn, L->idx_qn_w, L->idx_qn_b, m->idx_eps, m->idx_lntmp, n_embd);
    gemv(&L->idx_wq, m->idx_lntmp, q, (int64_t)nh * dk, n_embd);
    const float lt = logf((float)t + 1.0f);
    for (uint32_t h = 0; h < nh; h++) {
        const float sc = L->idx_ss_s[h] * lt + L->idx_ss_b[h];
        float *qh = q + (int64_t)h * dk;
        for (uint32_t j = 0; j < dk; j++) qh[j] *= sc;
        if (L->rope) rope_apply(qh, m->n_rot, m->n_rot, cs);
    }
    float *w = m->idx_wtmp;
    gemv(&L->idx_proj, xn, w, (int64_t)nh, n_embd);
    const float wsc = 1.0f / sqrtf((float)(nh * dk));
    for (uint32_t h = 0; h < nh; h++) w[h] *= wsc;

    /* scores over all cached positions, then top-k */
    float *sc_arr = m->idx_scores;
    const float *kb = m->ikv + (size_t)L->idx_slot * m->ikv_cap * dk;
    for (int64_t p = 0; p <= t; p++) {
        const float *kp = kb + (size_t)p * dk;
        float s = 0.0f;
        for (uint32_t h = 0; h < nh; h++) {
            const float d = dot_f32(q + (int64_t)h * dk, kp, (int64_t)dk);
            if (d > 0.0f) s += d * w[h];
        }
        sc_arr[p] = s;
    }
    const int64_t kk = m->idx_topk_eff < t + 1 ? m->idx_topk_eff : t + 1;
    fatcat_topk_select(sc_arr, t + 1, kk, m->idx_sel);
    ist->n_sel = kk;
}

/* D4 split: attention_core is the sequential middle of one layer's attention
 * at position t — everything between the (batchable) input projections
 * (wq, wkv_a) and the output projection (wout). q/kv arrive as produced rows
 * (modified in place: rmsnorm of ckv, RoPE, SSMax). o receives the head
 * outputs (nh*v_head). Bit-order identical to the pre-split attention(). */
static void attention_core(fatcat_ml4 *m, const ml4_layer *L, int il,
                           int64_t t, const float *xn, float *q, float *kv,
                           float *q_abs, float *acc, float *o,
                           struct ml4_idx *ist, int sp /* sub-profile on */,
                           int devio /* q/acc/o are device-accessible */);

/* FATCAT_GPU_ATT=2 verification: compare GPU result vs CPU reference
 * bit-wise; report count + max relative difference (absorb must be 0,
 * wv tolerates the known GPU-vs-CPU gemv ULP noise ~1e-4 rel). */
static void att_check(fatcat_ml4 *m, int il, const char *what,
                      const float *got, const float *chk, int64_t n)
{
    int64_t bad = 0;
    int64_t first = -1;
    double maxrel = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &chk[i], 4)) {
            bad++;
            if (first < 0) first = i;
            double d = fabs((double)got[i] - chk[i]);
            double r = d / (fabs((double)chk[i]) + 1e-30);
            if (r > maxrel) maxrel = r;
        }
    }
    if (bad) {
        m->att_chk_bad++;
        if (m->att_chk_logs < 16) {
            m->att_chk_logs++;
            fprintf(stderr, "att-check L%d %s: %lld/%lld differ first[%lld] "
                    "got %.9g ref %.9g maxrel %.3g\n", il, what,
                    (long long)bad, (long long)n, (long long)first,
                    got[first], chk[first], maxrel);
        }
    }
}

static void attention_core(fatcat_ml4 *m, const ml4_layer *L, int il,
                           int64_t t, const float *xn, float *q, float *kv,
                           float *q_abs, float *acc, float *o,
                           struct ml4_idx *ist, int sp /* sub-profile on */,
                           int devio /* q/acc/o are device-accessible */)
{
    const uint32_t nh = m->n_head, kl = m->kv_lora;
    const uint32_t qk = m->qk_nope + m->qk_rope;
    double tp = sp ? prof_now() : 0.0;
    const int64_t slot0 = (int64_t)L->kv_src_idx * m->kv_cap * m->kv_stride;
    const float *kvbase   = m->kv_f16 ? NULL     : m->kv + slot0;
    const uint16_t *kvbase_h = m->kv_f16 ? m->kv_h + slot0 : NULL;

    const bool owns_kv = (L->kv_src == il);
    float *kpe = kv + kl;
    if (owns_kv)
        rmsnorm(kv, L->norm_kv_a, m->eps, kv, (int64_t)kl);   /* ckv in place */

    float cs[64]; /* n_rot cos/sin pairs (n_rot <= 64) */
    if (L->rope) {
        rope_cache_pos(m, t, cs);
        if (owns_kv) rope_apply(kpe, m->qk_rope, m->qk_rope, cs);
        for (uint32_t h = 0; h < nh; h++)
            rope_apply(q + (int64_t)h * qk + m->qk_nope, m->qk_rope, m->qk_rope, cs);
    } else {
        /* NoPE layers keep the rope-slot layout but zero-fill it (pad_rope) */
        if (owns_kv) memset(kpe, 0, (size_t)m->qk_rope * sizeof(float));
        for (uint32_t h = 0; h < nh; h++)
            memset(q + (int64_t)h * qk + m->qk_nope, 0,
                   (size_t)m->qk_rope * sizeof(float));
    }

    /* absorb k into q: q_abs[h] = Wk_b[h] @ q_nope[h]  (512x64 per head) */
    if (sp) tp = prof_now();
    {
        const int64_t qk_nope = m->qk_nope;
        int done = 0;
        /* D5e: one batched GPU kernel; bit-identical to the CPU loop below
         * (kernel mirrors dequant_q8_0 + fatcat_dot_avx512 order exactly). */
        if (devio && m->gpu_att && L->wk_b.dev && (qk_nope % 16) == 0 &&
            fatcat_gpu_absorb_q8_0(L->wk_b.dev, q, q_abs, kl, nh,
                                   qk_nope, qk, (long long)L->wk_b.row_bytes) == 0) {
            fatcat_gpu_wait();
            done = 1;
            if (m->gpu_att == 2 && !m->att_chk)
                m->att_chk = malloc((size_t)nh * kl * sizeof(float));
            if (m->gpu_att == 2 && m->att_chk) {
                for (int64_t hr = 0; hr < (int64_t)nh * kl; hr++) {
                    uint32_t h = (uint32_t)(hr / kl), i = (uint32_t)(hr % kl);
                    float tmp[128];
                    fatcat_dequant(L->wk_b.type,
                                   (const char *)L->wk_b.p + ((int64_t)h * kl + i) * L->wk_b.row_bytes,
                                   tmp, (uint64_t)qk_nope);
                    m->att_chk[hr] = dot_f32(tmp, q + (int64_t)h * qk, qk_nope);
                }
                att_check(m, il, "absorb", q_abs, m->att_chk, (int64_t)nh * kl);
            }
        }
        if (!done) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t hr = 0; hr < (int64_t)nh * kl; hr++) {
                uint32_t h = (uint32_t)(hr / kl), i = (uint32_t)(hr % kl);
                float tmp[128];
                fatcat_dequant(L->wk_b.type,
                               (const char *)L->wk_b.p + ((int64_t)h * kl + i) * L->wk_b.row_bytes,
                               tmp, (uint64_t)qk_nope);
                q_abs[(int64_t)h * kl + i] = dot_f32(tmp, q + (int64_t)h * qk, qk_nope);
            }
        }
    }
    if (sp) { m->p_absorb += prof_now() - tp; tp = prof_now(); }

    /* SSMax: q *= s_h*log(t+1)+b_h (applies to absorbed and pe parts) */
    const float lt = logf((float)t + 1.0f);
    for (uint32_t h = 0; h < nh; h++) {
        float sc = L->ssmax_s[h] * lt + L->ssmax_b[h];
        for (uint32_t i = 0; i < kl; i++) q_abs[(int64_t)h * kl + i] *= sc;
        for (uint32_t i = 0; i < m->qk_rope; i++) q[(int64_t)h * qk + m->qk_nope + i] *= sc;
    }

    if (L->kv_src == il) {  /* owner: publish KV for this position */
        if (m->kv_f16) {
            uint16_t *slot = m->kv_h + slot0 + (size_t)t * m->kv_stride;
            for (uint32_t i = 0; i < kl + m->qk_rope; i++)
                slot[i] = f32_to_f16(kv[i]);
        } else {
            float *slot = m->kv + slot0 + (size_t)t * m->kv_stride;
            memcpy(slot, kv, (size_t)(kl + m->qk_rope) * sizeof(float));
        }
    }

    /* P5 DSA lightning indexer: full layers compute a fresh selection,
     * shared layers reuse the previous one (ist persists across layers).
     * D5d: when the whole KV cache can never exceed top-k (kv_cap <= topk)
     * the selection is dense forever -> skip the indexer entirely; its key
     * publish only matters once the cache can outgrow top-k. When kv_cap >
     * topk the call must run every token (idx_select_layer publishes the
     * indexer key for position t and early-returns while t+1 <= top-k).
     * Known limit: if kv_cap grows past top-k mid-session, indexer keys for
     * earlier positions are absent (zeroed) and rank last in selection. */
    if (m->idx_on && L->idx_on && m->kv_cap > (size_t)m->idx_topk_eff)
        idx_select_layer(m, L, ist, xn, cs, t);

    /* MQA over latent KV: online softmax per head, accumulate acc[h] = sum w*ckv */
    if (sp) tp = prof_now();
    const float kq_scale = 1.0f / sqrtf((float)qk);
    const int64_t np = t + 1;
    const int64_t nsel = (m->idx_on && ist->n_sel > 0 && ist->n_sel < np)
                       ? ist->n_sel : np;
    const bool sparse = nsel < np;
    for (uint32_t h = 0; h < nh; h++) {
        const float *qa = q_abs + (int64_t)h * kl;
        const float *qp = q + (int64_t)h * qk + m->qk_nope;
        float *ac = acc + (int64_t)h * kl;
        memset(ac, 0, (size_t)kl * sizeof(float));
        float m_i = -INFINITY, lsum = 0.0f;
        for (int64_t si = 0; si < nsel; si++) {
            const int64_t p = sparse ? m->idx_sel[si] : si;
            float m_new, corr, w;
            if (m->kv_f16) {   /* f16 KV: fused f16 read kernels */
                const uint16_t *kph = kvbase_h + (size_t)p * m->kv_stride;
                float s = fatcat_cpu_dot_f32_f16(qa, kph, (int64_t)kl) * kq_scale
                        + fatcat_cpu_dot_f32_f16(qp, kph + kl, (int64_t)m->qk_rope) * kq_scale;
                m_new = fmaxf(m_i, s);
                corr = expf(m_i - m_new);
                w = expf(s - m_new);
                lsum = lsum * corr + w;
                fatcat_cpu_kv_accum_f16(ac, corr, w, kph, (int64_t)kl);
            } else {
                const float *kp = kvbase + (size_t)p * m->kv_stride;
                float s = dot_f32(qa, kp, (int64_t)kl) * kq_scale
                        + dot_f32(qp, kp + kl, (int64_t)m->qk_rope) * kq_scale;
                m_new = fmaxf(m_i, s);
                corr = expf(m_i - m_new);
                w = expf(s - m_new);
                lsum = lsum * corr + w;
                for (uint32_t i = 0; i < kl; i++) ac[i] = ac[i] * corr + w * kp[i];
            }
            m_i = m_new;
        }
        for (uint32_t i = 0; i < kl; i++) ac[i] /= lsum;
    }
    if (sp) { m->p_scan += prof_now() - tp; tp = prof_now(); }

    /* absorbed V: o[h] = Wv_b[h] @ acc[h]  (128 rows of 512 per head) */
    if (sp) tp = prof_now();
    {
        int done = 0;
        /* D5e: one batched launch; per-row body identical to the gemv
         * kernel => bit-identical to the 64 CPU gemvs below. */
        if (devio && m->gpu_att && L->wv_b.dev &&
            fatcat_gpu_gemv_q8_0_batched(L->wv_b.dev,
                (long long)m->v_head * (long long)L->wv_b.row_bytes,
                acc, (long long)kl, o, (long long)m->v_head,
                (long long)m->v_head, (long long)kl, (long long)nh) == 0) {
            fatcat_gpu_wait();
            done = 1;
            if (m->gpu_att == 2 && !m->att_chk)
                m->att_chk = malloc((size_t)nh * kl * sizeof(float));
            if (m->gpu_att == 2 && m->att_chk) {
                for (uint32_t h = 0; h < nh; h++) {
                    const wt_t w = { L->wv_b.type,
                                     (const char *)L->wv_b.p + (int64_t)h * m->v_head * L->wv_b.row_bytes,
                                     0, 0, L->wv_b.row_bytes, NULL };
                    gemv(&w, acc + (int64_t)h * kl, m->att_chk + (int64_t)h * m->v_head,
                         (int64_t)m->v_head, (int64_t)kl);
                }
                att_check(m, il, "wv", o, m->att_chk, (int64_t)nh * m->v_head);
            }
        }
        if (!done) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (uint32_t h = 0; h < nh; h++) {
                const wt_t w = { L->wv_b.type,
                                 (const char *)L->wv_b.p + (int64_t)h * m->v_head * L->wv_b.row_bytes,
                                 0, 0, L->wv_b.row_bytes, NULL };
                gemv(&w, acc + (int64_t)h * kl, o + (int64_t)h * m->v_head,
                     (int64_t)m->v_head, (int64_t)kl);
            }
        }
    }
    if (sp) m->p_wv += prof_now() - tp;
}

static void attention(fatcat_ml4 *m, const ml4_layer *L, int il,
                      int64_t t, const float *xn, float *out /* [n_embd] */,
                      float *q, float *q_abs, float *kv, float *acc, float *o,
                      struct ml4_idx *ist)
{
    /* D5b: queue the input projections as ONE stream-ordered batch
     * (x staged once, both kernels launched, single wait) instead of two
     * fully synchronous gemvs. Same kernels, same order => bit-identical.
     * q/kv live in the device-mapped scratch, so kernels may write them. */
    const int64_t qk_rows = (int64_t)m->n_head * (m->qk_nope + m->qk_rope);
    const int64_t kv_rows = (int64_t)(m->kv_lora + m->qk_rope);
    const int need_kv = (L->kv_src == il);
    int queued = 0;
    if (m->scratch_dev && m->gpu_pipe_on && L->wq.dev &&
        (!need_kv || L->wkv_a.dev)) {
        const float *xd = fatcat_gpu_xstage(xn, (size_t)m->info.n_embd * 4);
        if (xd && fatcat_gpu_gemv_dev(L->wq.dev, L->wq.type, xd, q,
                                      qk_rows, (int64_t)m->info.n_embd) == 0 &&
            (!need_kv || fatcat_gpu_gemv_dev(L->wkv_a.dev, L->wkv_a.type, xd, kv,
                                             kv_rows, (int64_t)m->info.n_embd) == 0))
            queued = 1;
        fatcat_gpu_wait();   /* also drains a partial queue before sync retry */
    }
    if (!queued) {
        gemv(&L->wq, xn, q, qk_rows, (int64_t)m->info.n_embd);
        if (need_kv)
            gemv(&L->wkv_a, xn, kv, kv_rows, (int64_t)m->info.n_embd);
    }
    double tc0 = m->prof ? prof_now() : 0.0;
    attention_core(m, L, il, t, xn, q, kv, q_abs, acc, o, ist, m->prof,
                   m->scratch_dev);
    if (m->prof) { double t1 = prof_now(); m->p_core += t1 - tc0; tc0 = t1; }
    gemv(&L->wout, o, out, (int64_t)m->info.n_embd,
         (int64_t)m->n_head * m->v_head);
    if (m->prof) m->p_wout += prof_now() - tc0;
}

/* dense-style FFN: y = down @ (silu(gate@x) * up@x) */
static void ffn_par(const fatcat_ml4 *m, const wt_t *gate, const wt_t *up,
                    const wt_t *down, const float *x, float *y,
                    float *g, float *u, int64_t n_ff)
{
    const int64_t n_embd = m->info.n_embd;
    gemv(gate, x, g, n_ff, n_embd);
    gemv(up, x, u, n_ff, n_embd);
    for (int64_t i = 0; i < n_ff; i++) g[i] = silu(g[i]) * u[i];
    gemv(down, g, y, n_embd, n_ff);
}

static int fatcat_ml4_forward_int(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                                  fatcat_router_cb router_cb, void *cb_ctx,
                                  float *last_logits, float *row_logits);

/* output_norm + lm_head row: identical op order everywhere logits are
 * produced (last row, per-row), so all paths stay bit-identical. */
static void logits_row(fatcat_ml4 *m, const float *x_final, float *row,
                       float *xn)
{
    const int64_t n_embd = m->info.n_embd;
    rmsnorm(x_final, m->output_norm, m->eps, xn, n_embd);
    gemv(&m->output, xn, row, (int64_t)m->info.n_vocab, n_embd);
    /* Gemma-2 style final logit softcap: cap * tanh(x / cap) */
    if (m->logit_softcap > 0.0f) {
        const float cap = m->logit_softcap, inv = 1.0f / cap;
        for (int64_t i = 0; i < (int64_t)m->info.n_vocab; i++)
            row[i] = cap * tanhf(row[i] * inv);
    }
    for (int64_t i = 0; i < (int64_t)m->info.n_vocab; i++)
        if (!isfinite(row[i])) { m->health_bad++; break; }
}

/* D1: batched prefill (layer-outer). Processes the prompt in chunks of
 * C = min(batch_n, n) tokens. Per layer: attention per token, routers for
 * the whole chunk, then ONE expert-slice load per expert serving every
 * token routed to it (grouped via per-expert linked lists). Weight bytes
 * per token drop by the reuse factor (~30x at C=256, 128 experts/layer);
 * the union of experts is prefetched async before the shared-expert FFN
 * so NVMe streams during that compute.
 * Bit-identity: each token's arithmetic keeps the sequential order —
 * shared FFN first, routed contributions accumulated in top-k order —
 * so logits match the token-by-token path exactly.
 * v1 scope: synchronous GEMVs (no GPU async pipe), no router speculation,
 * no hidden-state dump; the caller falls back to the sequential path
 * otherwise. */
static int forward_batched(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                           fatcat_router_cb router_cb, void *cb_ctx,
                           float *last_logits, float *row_logits,
                           float *xn, float *out, float *q, float *q_abs,
                           float *kv, float *acc, float *o, float *fa,
                           float *fb, float *logits)
{
    const int64_t n_embd = m->info.n_embd;
    const uint32_t topk = m->info.n_experts_used, ne = m->info.n_experts;
    const int64_t n_ff_exp = m->n_ff_exp;
    size_t C = n_tokens < m->batch_n ? n_tokens : m->batch_n;
    int rc = -1;
    /* D4 debug: FATCAT_D4_OPS bitmask isolates which batched dense-op
     * group diverges from the sequential path (bit0 attn projections,
     * bit1 shared FFN, bit2 dense FFN). Default 7 = all batched. */
    static int d4_ops = -1;
    if (d4_ops < 0) {
        const char *e = getenv("FATCAT_D4_OPS");
        d4_ops = e ? (int)strtol(e, NULL, 0) : 7;
    }
    const int d4_attn = d4_ops & 1, d4_shffn = d4_ops & 2, d4_dense = d4_ops & 4;

    float *X     = malloc(C * (size_t)n_embd * sizeof(float));   /* residual */
    float *xnbuf = malloc(C * (size_t)n_embd * sizeof(float));   /* norm_ffn out */
    float *shb   = malloc(C * (size_t)n_embd * sizeof(float));   /* shared FFN out */
    float *fcbuf = malloc(C * topk * (size_t)n_embd * sizeof(float)); /* down out */
    const int64_t n_ff_max = n_ff_exp > (int64_t)m->n_ff ? n_ff_exp : (int64_t)m->n_ff;
    float *Xg    = malloc(C * (size_t)n_embd * sizeof(float));   /* gemm in */
    float *Yg    = malloc(C * (size_t)n_ff_max * sizeof(float)); /* gemm gate out */
    float *Yu    = malloc(C * (size_t)n_ff_max * sizeof(float)); /* gemm up out */
    float *Zb    = malloc(C * (size_t)n_ff_max * sizeof(float)); /* silu(g)*u */
    float *Yd    = malloc(C * (size_t)n_embd * sizeof(float));   /* gemm down out */
    /* D4 batched attention projections */
    const int64_t qk_row = (int64_t)m->n_head * (m->qk_nope + m->qk_rope);
    const int64_t kv_row = (int64_t)m->kv_lora + m->qk_rope;
    const int64_t o_row  = (int64_t)m->n_head * m->v_head;
    float *xnab = malloc(C * (size_t)n_embd * sizeof(float));    /* norm_attn out */
    float *qb   = malloc(C * (size_t)qk_row * sizeof(float));    /* wq out */
    float *kvb  = malloc(C * (size_t)kv_row * sizeof(float));    /* wkv_a out */
    float *ob   = malloc(C * (size_t)o_row  * sizeof(float));    /* attn head out */
    float *outb = malloc(C * (size_t)n_embd * sizeof(float));    /* wout out */
    int32_t *slots = malloc(C * topk * sizeof(int32_t));
    uint16_t *ids = malloc(C * topk * sizeof(uint16_t));
    float *wts = malloc(C * topk * sizeof(float));
    int32_t *head = malloc(ne * sizeof(int32_t));
    int32_t *nxt = malloc(C * topk * sizeof(int32_t));
    uint16_t *g_tok = malloc(C * topk * sizeof(uint16_t));
    uint8_t *g_k = malloc(C * topk);
    uint16_t *union_ids = malloc(ne * sizeof(uint16_t));
    struct ml4_idx *ist = calloc(C, sizeof *ist);
    if (!X || !xnbuf || !shb || !fcbuf || !Xg || !Yg || !Yu || !Zb || !Yd ||
        !xnab || !qb || !kvb || !ob || !outb ||
        !slots || !ids || !wts || !head || !nxt ||
        !g_tok || !g_k || !union_ids || !ist) {
        fprintf(stderr, "ml4: oom batched prefill\n");
        goto cleanup;
    }

    for (size_t c0 = 0; c0 < n_tokens; c0 += C) {
        size_t cc = n_tokens - c0 < C ? n_tokens - c0 : C;

        if (m->mem_limit_mb && !m->drop_experts &&
            mem_used_mb() > m->mem_limit_mb) {
            m->drop_experts = 1;
            fprintf(stderr, "mem: system usage > %ld MB - releasing streamed expert pages after each use\n",
                    m->mem_limit_mb);
        }

        /* embeddings -> X */
        for (size_t i = 0; i < cc; i++) {
            uint32_t tok = tokens[c0 + i];
            if (tok >= m->info.n_vocab) { fprintf(stderr, "ml4: token %u OOB\n", tok); goto cleanup; }
            fatcat_dequant(m->tok_embd.type,
                           (const char *)m->tok_embd.p + (int64_t)tok * m->tok_embd.row_bytes,
                           X + i * n_embd, (uint64_t)n_embd);
        }

        for (uint32_t il = 0; il < m->info.n_layers; il++) {
            ml4_layer *L = &m->layers[il];

            /* D4: batched attention input projections (one GEMM per weight
             * for the whole chunk), then the sequential attention middle. */
            if (d4_attn) {
                for (size_t i = 0; i < cc; i++)
                    rmsnorm(X + i * n_embd, L->norm_attn, m->eps,
                            xnab + i * n_embd, n_embd);
                gemm_res(m, &L->wq, xnab, qb, qk_row, n_embd, (int)cc);
                if (L->kv_src == (int)il)
                    gemm_res(m, &L->wkv_a, xnab, kvb, kv_row, n_embd, (int)cc);
                for (size_t i = 0; i < cc; i++) {
                    attention_core(m, L, (int)il, (int64_t)(m->seq_len + c0 + i),
                                   xnab + i * n_embd, qb + i * qk_row,
                                   kvb + i * kv_row, q_abs, acc, ob + i * o_row,
                                   &ist[i], 0, 0);
                }
                gemm_res(m, &L->wout, ob, outb, n_embd, o_row, (int)cc);
                for (size_t i = 0; i < cc; i++)
                    for (int64_t j = 0; j < n_embd; j++)
                        X[i * n_embd + j] += outb[i * n_embd + j];
            } else {
                for (size_t i = 0; i < cc; i++) {
                    rmsnorm(X + i * n_embd, L->norm_attn, m->eps, xn, n_embd);
                    attention(m, L, (int)il, (int64_t)(m->seq_len + c0 + i), xn,
                              out, q, q_abs, kv, acc, o, &ist[i]);
                    for (int64_t j = 0; j < n_embd; j++) X[i * n_embd + j] += out[j];
                }
            }

            if (!L->is_moe) {
                if (d4_dense) {
                    for (size_t i = 0; i < cc; i++)
                        rmsnorm(X + i * n_embd, L->norm_ffn, m->eps,
                                xnbuf + i * n_embd, n_embd);
                    gemm_res(m, &L->ffn_gate, xnbuf, Yg, m->n_ff, n_embd, (int)cc);
                    gemm_res(m, &L->ffn_up, xnbuf, Yu, m->n_ff, n_embd, (int)cc);
                    for (int64_t j = 0; j < (int64_t)cc * m->n_ff; j++)
                        Zb[j] = silu(Yg[j]) * Yu[j];
                    gemm_res(m, &L->ffn_down, Zb, shb, n_embd, m->n_ff, (int)cc);
                    for (size_t i = 0; i < cc; i++)
                        for (int64_t j = 0; j < n_embd; j++)
                            X[i * n_embd + j] += shb[i * n_embd + j];
                } else {
                    for (size_t i = 0; i < cc; i++) {
                        rmsnorm(X + i * n_embd, L->norm_ffn, m->eps, xn, n_embd);
                        ffn_par(m, &L->ffn_gate, &L->ffn_up, &L->ffn_down,
                                xn, out, fa, fb, m->n_ff);
                        for (int64_t j = 0; j < n_embd; j++) X[i * n_embd + j] += out[j];
                    }
                }
                continue;
            }

            /* routers for the whole chunk; keep normalized input for reuse */
            for (size_t i = 0; i < cc; i++) {
                rmsnorm(X + i * n_embd, L->norm_ffn, m->eps,
                        xnbuf + i * n_embd, n_embd);
                router_topk(m, L->router, xnbuf + i * n_embd, logits,
                            ids + i * topk, wts + i * topk);
                if (router_cb)
                    router_cb(cb_ctx, il, (uint32_t)(m->seq_len + c0 + i),
                              ids + i * topk, wts + i * topk, topk);
            }

            /* group token-slots per expert (linked lists) */
            for (uint32_t e = 0; e < ne; e++) head[e] = -1;
            for (size_t i = 0; i < cc; i++)
                for (uint32_t k = 0; k < topk; k++) {
                    uint32_t e = ids[i * topk + k];
                    uint32_t slot = (uint32_t)(i * topk + k);
                    g_tok[slot] = (uint16_t)i; g_k[slot] = (uint8_t)k;
                    nxt[slot] = head[e]; head[e] = (int32_t)slot;
                }

            /* async-prefetch the union of needed experts, then overlap the
             * shared-expert FFNs with the streaming */
            if (m->stream) {
                uint32_t nu = 0;
                for (uint32_t e = 0; e < ne; e++)
                    if (head[e] >= 0) union_ids[nu++] = (uint16_t)e;
                prefetch_experts(m, il, L, union_ids, nu, 0);
            }
            /* D4: shared-expert FFNs batched over the chunk (resident weights) */
            if (d4_shffn) {
                gemm_res(m, &L->sh_gate, xnbuf, Yg, n_ff_exp, n_embd, (int)cc);
                gemm_res(m, &L->sh_up, xnbuf, Yu, n_ff_exp, n_embd, (int)cc);
                for (int64_t j = 0; j < (int64_t)cc * n_ff_exp; j++)
                    Zb[j] = silu(Yg[j]) * Yu[j];
                gemm_res(m, &L->sh_down, Zb, shb, n_embd, n_ff_exp, (int)cc);
            } else {
                for (size_t i = 0; i < cc; i++)
                    ffn_par(m, &L->sh_gate, &L->sh_up, &L->sh_down,
                            xnbuf + i * n_embd, shb + i * n_embd, fa, fb, n_ff_exp);
            }

            /* grouped expert pass: one slice load per expert per chunk */
            for (uint32_t e = 0; e < ne; e++) {
                if (head[e] < 0) continue;
                int n_e = 0;
                for (int32_t it = head[e]; it >= 0; it = nxt[it]) slots[n_e++] = it;
                uint64_t ek3[3];
                const char *gs = expert_slice(m, &L->ex_gate, L->ex_gate_stride,
                                              e, (uint64_t)il * 3 + 0, &ek3[0]);
                const char *us = expert_slice(m, &L->ex_up, L->ex_up_stride,
                                              e, (uint64_t)il * 3 + 1, &ek3[1]);
                const char *ds = expert_slice(m, &L->ex_down, L->ex_down_stride,
                                              e, (uint64_t)il * 3 + 2, &ek3[2]);
                const uint64_t tag = ((uint64_t)il << 34) | e;
                if (n_e == 1) {
                    /* single token: gemv path (identical work, no gather) */
                    const int32_t it = slots[0];
                    const size_t i = g_tok[it];
                    const uint8_t k = g_k[it];
                    gemv_expert(m, &L->ex_gate, gs, L->ex_gate_stride, e,
                                tag | (1ull << 32), xnbuf + i * n_embd, fa,
                                n_ff_exp, n_embd, ek3[0] != 0, 0, NULL);
                    gemv_expert(m, &L->ex_up, us, L->ex_up_stride, e,
                                tag | (2ull << 32), xnbuf + i * n_embd, fb,
                                n_ff_exp, n_embd, ek3[1] != 0, 0, NULL);
                    for (int64_t j = 0; j < n_ff_exp; j++)
                        fa[j] = silu(fa[j]) * fb[j];
                    gemv_expert(m, &L->ex_down, ds, L->ex_down_stride, e,
                                tag | (3ull << 32), fa,
                                fcbuf + (i * topk + k) * (size_t)n_embd,
                                n_embd, n_ff_exp, ek3[2] != 0, 0, NULL);
                } else {
                    /* batched: gather token rows, one decode serves them all */
                    for (int t = 0; t < n_e; t++)
                        memcpy(Xg + (int64_t)t * n_embd,
                               xnbuf + (int64_t)g_tok[slots[t]] * n_embd,
                               (size_t)n_embd * sizeof(float));
                    expert_gemm(m, &L->ex_gate, gs, L->ex_gate_stride, e,
                                tag | (1ull << 32), Xg, Yg, n_ff_exp, n_embd,
                                ek3[0] != 0, n_e);
                    expert_gemm(m, &L->ex_up, us, L->ex_up_stride, e,
                                tag | (2ull << 32), Xg, Yu, n_ff_exp, n_embd,
                                ek3[1] != 0, n_e);
                    for (int64_t j = 0; j < (int64_t)n_e * n_ff_exp; j++)
                        Zb[j] = silu(Yg[j]) * Yu[j];
                    expert_gemm(m, &L->ex_down, ds, L->ex_down_stride, e,
                                tag | (3ull << 32), Zb, Yd, n_embd, n_ff_exp,
                                ek3[2] != 0, n_e);
                    for (int t = 0; t < n_e; t++)
                        memcpy(fcbuf + ((int64_t)g_tok[slots[t]] * topk + g_k[slots[t]]) * n_embd,
                               Yd + (int64_t)t * n_embd,
                               (size_t)n_embd * sizeof(float));
                }
                for (int t = 0; t < 3; t++)
                    if (ek3[t]) fatcat_stream_release(m->stream, ek3[t]);
            }

            /* combine: shared + routed in top-k order (bit-identical to
             * the sequential accumulation order) */
            for (size_t i = 0; i < cc; i++) {
                float *comb = shb + i * n_embd;
                for (uint32_t k = 0; k < topk; k++) {
                    const float *ck = fcbuf + (i * topk + k) * (size_t)n_embd;
                    const float w = wts[i * topk + k];
                    for (int64_t j = 0; j < n_embd; j++) comb[j] += w * ck[j];
                }
                for (int64_t j = 0; j < n_embd; j++)
                    X[i * n_embd + j] += comb[j];
            }
        }

        /* per-token epilogue: health, progress, recency tick */
        for (size_t i = 0; i < cc; i++) {
            m->health_tokens++;
            int bad = 0;
            for (int64_t j = 0; j < n_embd; j++)
                if (!isfinite(X[i * n_embd + j])) { bad = 1; break; }
            if (bad) m->health_bad++;
            if (m->token_cb) m->token_cb(m->token_ctx, c0 + i + 1, n_tokens);
            fatcat_stream_tick(m->stream);
        }

        if (row_logits) {
            /* D3: logits for every position in this chunk (speculative
             * verification). Same per-row op order as the last-row path. */
            for (size_t i = 0; i < cc; i++)
                logits_row(m, X + i * n_embd,
                           row_logits + (c0 + i) * (int64_t)m->info.n_vocab, xn);
        }
        if (last_logits && c0 + cc == n_tokens)
            logits_row(m, X + (cc - 1) * n_embd, last_logits, xn);
    }
    m->seq_len += n_tokens;
    rc = 0;
cleanup:
    free(X); free(xnbuf); free(shb); free(fcbuf); free(Xg); free(Yg); free(Yu);
    free(Zb); free(Yd); free(xnab); free(qb); free(kvb); free(ob); free(outb);
    free(slots); free(ids); free(wts);
    free(head); free(nxt); free(g_tok); free(g_k); free(union_ids); free(ist);
    return rc;
}

int fatcat_ml4_forward(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                       fatcat_router_cb router_cb, void *cb_ctx)
{
    return fatcat_ml4_forward_int(m, tokens, n_tokens, router_cb, cb_ctx, NULL, NULL);
}

int fatcat_ml4_last_logits(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                           float *logits)
{
    if (!m->output.p) { fprintf(stderr, "ml4: output.weight missing\n"); return -1; }
    return fatcat_ml4_forward_int(m, tokens, n_tokens, NULL, NULL, logits, NULL);
}

int fatcat_ml4_forward_rows(fatcat_ml4 *m, const uint32_t *tokens, size_t n,
                            float *rows)
{
    if (!m->output.p) { fprintf(stderr, "ml4: output.weight missing\n"); return -1; }
    return fatcat_ml4_forward_int(m, tokens, n, NULL, NULL, NULL, rows);
}

void fatcat_ml4_reset(fatcat_ml4 *m)
{
    m->seq_len = 0;   /* stale KV beyond seq_len is never read */
}

void fatcat_ml4_rewind(fatcat_ml4 *m, size_t new_len)
{
    /* D3: discard speculative positions. KV is position-indexed and stale
     * entries beyond seq_len are never read, so rewinding the counter is
     * sufficient (mirrors fatcat_ml4_reset). */
    if (new_len <= m->seq_len) m->seq_len = new_len;
}

size_t fatcat_ml4_seq_len(const fatcat_ml4 *m)
{
    return m->seq_len;
}

size_t fatcat_ml4_kv_cap(const fatcat_ml4 *m)
{
    return m->kv_cap;
}

void fatcat_ml4_set_router_cb(fatcat_ml4 *m, fatcat_router_cb cb, void *ctx)
{
    m->router_cb = cb;
    m->router_cb_ctx = ctx;
}

static int fatcat_ml4_forward_int(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                                  fatcat_router_cb router_cb, void *cb_ctx,
                                  float *last_logits, float *row_logits)
{
    const int64_t n_embd = m->info.n_embd;
    const uint32_t nh = m->n_head, qk = m->qk_nope + m->qk_rope, kl = m->kv_lora;
    const uint32_t topk = m->info.n_experts_used;
    int rc = -1;

    /* model-level router callback (set via fatcat_ml4_set_router_cb) acts
     * as the default when the call site passes none: lets tools trace every
     * routing decision (forward, last_logits, engine paths) from one hook. */
    if (!router_cb && m->router_cb) { router_cb = m->router_cb; cb_ctx = m->router_cb_ctx; }

    /* grow KV cache if needed (KV-owning layers only) */
    if (m->seq_len + n_tokens > m->kv_cap) {
        size_t n_own = 0;
        for (uint32_t il = 0; il < m->info.n_layers; il++)
            if (m->layers[il].kv_src == (int)il) n_own++;
        m->kv_stride = kl + m->qk_rope;
        /* cap must cover the TOTAL sequence (seq_len + new tokens), not just
         * this batch.  With FATCAT_KV_TARGET set, allocate exactly that
         * capacity once (no doubling overshoot; calloc is lazy so the cost
         * is virtual until touched).  Otherwise doubling keeps per-token
         * growth amortized O(1). */
        size_t cap = m->seq_len + n_tokens + 64;
        if (m->kv_target) {
            if (m->kv_target > cap) cap = m->kv_target;   /* fixed capacity, no doubling */
        } else if (cap < m->kv_cap * 2) {
            cap = m->kv_cap * 2;
        }
        if (m->mem_budget_mb && m->kv_tok_bytes) {
            long need_mb = (long)((double)cap * m->kv_tok_bytes
                                  / (1024.0 * 1024.0)) + 1;
            if (need_mb + MEM_RESERVE_MB > m->mem_budget_mb) {
                fprintf(stderr, "ml4: KV growth to %zu tokens (%ld MB) exceeds "
                        "memory budget %ld MB - refusing\n",
                        cap, need_mb, m->mem_budget_mb);
                return -1;
            }
            /* make room in the warm tier BEFORE allocating: the grow copy
             * transiently holds old + new KV, so the tier must already be
             * small enough for the new capacity to fit the budget. */
            if (m->stream) {
                long keep = m->mem_budget_mb - MEM_RESERVE_MB - need_mb;
                if (keep < 0) keep = 0;
                size_t freed = fatcat_stream_shrink(m->stream, (size_t)keep << 20);
                if (freed)
                    fprintf(stderr, "mem: KV growing to %.1f GB; warm tier "
                            "shrunk %.2f GB to stay in budget\n",
                            (double)cap * m->kv_tok_bytes / 1e9, freed / 1e9);
            }
        }
        fprintf(stderr, "kv: capacity %zu tokens, %.1f GB (%s)\n", cap,
                (double)(n_own * cap * m->kv_stride * (m->kv_f16 ? 2 : 4)) / 1e9,
                m->kv_f16 ? "f16" : "f32");
        if (m->kv_f16) {
            uint16_t *nk = calloc(n_own * cap * m->kv_stride, sizeof(uint16_t));
            if (!nk) { fprintf(stderr, "ml4: oom growing KV\n"); return -1; }
            size_t old_own = 0;
            for (uint32_t il = 0; il < m->info.n_layers; il++) {
                if (m->layers[il].kv_src != (int)il) continue;
                for (size_t p = 0; p < m->kv_cap; p++)
                    memcpy(nk + (old_own * cap + p) * m->kv_stride,
                           m->kv_h + (old_own * m->kv_cap + p) * m->kv_stride,
                           m->kv_stride * sizeof(uint16_t));
                old_own++;
            }
            free(m->kv_h); m->kv_h = nk;
            if (m->kv_hugepage)
                advise_hugepage(nk, n_own * cap * m->kv_stride * sizeof(uint16_t));
        } else {
            float *nk = calloc(n_own * cap * m->kv_stride, sizeof(float));
            if (!nk) { fprintf(stderr, "ml4: oom growing KV\n"); return -1; }
            /* copy old per owning layer into new layout */
            size_t old_own = 0;
            for (uint32_t il = 0; il < m->info.n_layers; il++) {
                if (m->layers[il].kv_src != (int)il) continue;
                for (size_t p = 0; p < m->kv_cap; p++)
                    memcpy(nk + (old_own * cap + p) * m->kv_stride,
                           m->kv + (old_own * m->kv_cap + p) * m->kv_stride,
                           m->kv_stride * sizeof(float));
                old_own++;
            }
            free(m->kv); m->kv = nk;
            if (m->kv_hugepage)
                advise_hugepage(nk, n_own * cap * m->kv_stride * sizeof(float));
        }
        m->kv_cap = cap;

        /* P5: indexer k cache + selection scratch grow with the same cap */
        if (m->idx_on) {
            float *ni = calloc((size_t)m->n_idx_own * cap * m->idx_dk, sizeof(float));
            float *ns = malloc(cap * sizeof(float));
            int64_t *nsel = malloc(cap * (int64_t)sizeof(int64_t));
            if (!ni || !ns || !nsel) {
                free(ni); free(ns); free(nsel);
                fprintf(stderr, "ml4: oom growing indexer cache\n");
                return -1;
            }
            if (m->ikv)
                for (int sl = 0; sl < m->n_idx_own; sl++)
                    memcpy(ni + (size_t)sl * cap * m->idx_dk,
                           m->ikv + (size_t)sl * m->ikv_cap * m->idx_dk,
                           m->ikv_cap * m->idx_dk * sizeof(float));
            free(m->ikv); m->ikv = ni;
            if (m->kv_hugepage)
                advise_hugepage(ni, (size_t)m->n_idx_own * cap * m->idx_dk * sizeof(float));
            free(m->idx_scores); m->idx_scores = ns;
            free(m->idx_sel); m->idx_sel = nsel;
            m->ikv_cap = cap;
        }
    }

    size_t sz_q = (size_t)nh * qk, sz_abs = (size_t)nh * kl, sz_v = (size_t)nh * m->v_head;
    size_t sz_big = (size_t)(m->n_ff > m->n_ff_exp ? m->n_ff : m->n_ff_exp);
    size_t need = 3 * n_embd + sz_q + 2 * sz_abs + sz_v +
                  ((size_t)3 * ML4_MAX_TOPK + 1) * sz_big +
                  kl + m->qk_rope + ML4_MAX_EXPERTS;
    if (!m->scratch) {   /* allocated once, reused across forwards */
        if (fatcat_gpu_enabled() && fatcat_gpu_unified()) {
            m->scratch = fatcat_gpu_host_alloc(need * sizeof(float));
            m->scratch_dev = m->scratch != NULL;
        }
        if (!m->scratch) m->scratch = malloc(need * sizeof(float));
        m->scratch_n = need;
        if (!m->scratch) { fprintf(stderr, "ml4: oom scratch\n"); return -1; }
    }
    float *buf = m->scratch;
    float *x = buf, *xn = x + n_embd, *out = xn + n_embd;
    float *q = out + n_embd, *q_abs = q + sz_q, *acc = q_abs + sz_abs;
    float *kv = acc + sz_abs, *o = kv + kl + m->qk_rope;
    /* per-expert gate/up/down outputs so async GEMVs can target each
     * expert independently (fa[k], fb[k], fc[k]; fd = shared-expert out) */
    float *fa = o + sz_v, *fb = fa + ML4_MAX_TOPK * sz_big;
    float *fc = fb + ML4_MAX_TOPK * sz_big, *fd = fc + ML4_MAX_TOPK * sz_big;
    float *logits = fd + sz_big;

    /* diagnostics: dump residual stream (embd + after each layer) per token */
    FILE *dump_fp = NULL;
    {
        const char *dp = getenv("FATCAT_DUMP_HID");
        if (dp) { dump_fp = fopen(dp, "wb"); if (!dump_fp) perror("FATCAT_DUMP_HID"); }
    }

    /* D1: batched prefill path (layer-outer, grouped expert loads) */
    if (m->batch_n > 1 && n_tokens > 1 && !dump_fp)
        return forward_batched(m, tokens, n_tokens, router_cb, cb_ctx,
                               last_logits, row_logits, xn, out, q, q_abs, kv, acc, o,
                               fa, fb, logits);

    for (size_t t = 0; t < n_tokens; t++) {
        uint32_t tok = tokens[t];
        double tk0 = m->prof ? prof_now() : 0.0;
        struct ml4_idx ist = { 0 };   /* DSA selection, carried across layers */
        uint16_t prev_spec[ML4_MAX_TOPK], cur_spec[ML4_MAX_TOPK];
        int prev_spec_n = 0, cur_spec_n = 0;
        if (m->mem_limit_mb && !m->drop_experts && mem_used_mb() > m->mem_limit_mb) {
            m->drop_experts = 1;
            fprintf(stderr, "mem: system usage > %ld MB - releasing streamed expert pages after each use\n",
                    m->mem_limit_mb);
            fflush(stderr);
        }
        if (tok >= m->info.n_vocab) { fprintf(stderr, "ml4: token %u OOB\n", tok); goto done; }
        fatcat_dequant(m->tok_embd.type,
                       (const char *)m->tok_embd.p + (int64_t)tok * m->tok_embd.row_bytes,
                       x, (uint64_t)n_embd);
        if (dump_fp) fwrite(x, sizeof(float), (size_t)n_embd, dump_fp);

        for (uint32_t il = 0; il < m->info.n_layers; il++) {
            ml4_layer *L = &m->layers[il];
            double t0 = m->prof ? prof_now() : 0.0;
            rmsnorm(x, L->norm_attn, m->eps, xn, n_embd);
            attention(m, L, (int)il, (int64_t)(m->seq_len + t), xn, out, q, q_abs, kv, acc, o, &ist);
            if (dump_fp) fwrite(out, sizeof(float), (size_t)n_embd, dump_fp); /* attn out */
            for (int64_t i = 0; i < n_embd; i++) x[i] += out[i];
            if (m->prof) { double t1 = prof_now(); m->p_attn += t1 - t0; t0 = t1; }

            /* P3 1A: speculative router for the NEXT MoE layer, run on the
             * current hidden state (this layer's FFN delta is missing).
             * Prefetch is only a hint: wrong guesses cost NVMe bandwidth,
             * never correctness. */
            if (m->prof) t0 = prof_now();
            cur_spec_n = 0;
            if (m->spec_on && m->stream && il + 1 < m->info.n_layers &&
                m->layers[il + 1].is_moe) {
                rmsnorm(x, m->layers[il + 1].norm_ffn, m->eps, xn, n_embd);
                cur_spec_n = (int)router_topk(m, m->layers[il + 1].router,
                                              xn, logits, cur_spec, NULL);
            }
            if (m->prof) { double t1 = prof_now(); m->p_router += t1 - t0; t0 = t1; }

            rmsnorm(x, L->norm_ffn, m->eps, xn, n_embd);
            if (!L->is_moe) {
                ffn_par(m, &L->ffn_gate, &L->ffn_up, &L->ffn_down, xn, out, fa, fb, m->n_ff);
                if (dump_fp) fwrite(out, sizeof(float), (size_t)n_embd, dump_fp); /* ffn out */
                for (int64_t i = 0; i < n_embd; i++) x[i] += out[i];
                if (m->prof) { double t1 = prof_now(); m->p_dense += t1 - t0; t0 = t1; }
            } else {
                uint16_t ids[ML4_MAX_TOPK]; float wts[ML4_MAX_TOPK];
                router_topk(m, L->router, xn, logits, ids, wts);
                if (router_cb) router_cb(cb_ctx, il, (uint32_t)(m->seq_len + t), ids, wts, topk);
                if (m->prof) { double t1 = prof_now(); m->p_router += t1 - t0; t0 = t1; }
                /* speculation accuracy bookkeeping (spec came from layer il-1) */
                if (prev_spec_n > 0) {
                    m->spec_total += topk;
                    for (uint32_t k = 0; k < topk; k++)
                        for (int j = 0; j < prev_spec_n; j++)
                            if (ids[k] == prev_spec[j]) { m->spec_hit++; break; }
                }

                /* shared expert + routed experts */
                /* P3 v2: issue this layer's expert reads NOW (async, via the
                 * stream fill worker) so NVMe streams while the shared-expert
                 * FFN computes below. The speculative next-layer set follows
                 * (lower priority: it must not delay the critical reads). */
                if (m->stream)
                    prefetch_experts(m, il, L, ids, topk, 0);
                if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_pre += t1 - t0; t0 = t1; }
                ffn_par(m, &L->sh_gate, &L->sh_up, &L->sh_down, xn, fd, fa, fb, m->n_ff_exp);
                for (int64_t i = 0; i < n_embd; i++) out[i] = fd[i];
                if (m->prof) { double t1 = prof_now(); m->p_shffn += t1 - t0; t0 = t1; }
                /* 1B: deferred per-slice waits. Each expert_slice() blocks
                 * only on ITS OWN in-flight read, so expert k's GEMVs run
                 * while the other slices of the layer are still streaming.
                 * Falls back to the mmap view when the tier is disabled.
                 * P4 v2 pipelining: with a device-mapped scratch the GEMVs
                 * launch async (zero-copy from the tier slots); one wait
                 * after gate+up, one after down => 2 syncs/layer, and the
                 * GPU computes while later slices stream in. */
                uint64_t ek[ML4_MAX_TOPK][3];   /* tier keys held (0 = none) */
                const int gpu_pipe = m->scratch_dev && m->gpu_pipe_on &&
                                     fatcat_gpu_enabled();
                /* stage xn once: all gate+up GEMVs of the layer share it */
                const float *xn_dev = gpu_pipe
                    ? fatcat_gpu_xstage(xn, (size_t)n_embd * 4) : NULL;
                for (uint32_t k = 0; k < topk; k++) {
                    const uint64_t base_tag = ((uint64_t)il << 34) | ids[k];
                    const char *g = expert_slice(m, &L->ex_gate, L->ex_gate_stride,
                                                 ids[k], (uint64_t)il * 3 + 0,
                                                 &ek[k][0]);
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_slice += t1 - t0; t0 = t1; }
                    gemv_expert(m, &L->ex_gate, g, L->ex_gate_stride, ids[k],
                                base_tag | (1ull << 32), xn, fa + (size_t)k * sz_big,
                                m->n_ff_exp, n_embd, ek[k][0] != 0, gpu_pipe, xn_dev);
                    const char *u = expert_slice(m, &L->ex_up, L->ex_up_stride,
                                                 ids[k], (uint64_t)il * 3 + 1,
                                                 &ek[k][1]);
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_slice += t1 - t0; t0 = t1; }
                    gemv_expert(m, &L->ex_up, u, L->ex_up_stride, ids[k],
                                base_tag | (2ull << 32), xn, fb + (size_t)k * sz_big,
                                m->n_ff_exp, n_embd, ek[k][1] != 0, gpu_pipe, xn_dev);
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_queue += t1 - t0; t0 = t1; }
                }
                if (gpu_pipe) fatcat_gpu_wait();
                if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_wait += t1 - t0; t0 = t1; }
                for (uint32_t k = 0; k < topk; k++) {
                    float *fk = fa + (size_t)k * sz_big;
                    const float *gk = fb + (size_t)k * sz_big;
                    for (int64_t i = 0; i < (int64_t)m->n_ff_exp; i++)
                        fk[i] = silu(fk[i]) * gk[i];
                    const uint64_t base_tag = ((uint64_t)il << 34) | ids[k];
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_act += t1 - t0; t0 = t1; }
                    const char *d = expert_slice(m, &L->ex_down, L->ex_down_stride,
                                                 ids[k], (uint64_t)il * 3 + 2,
                                                 &ek[k][2]);
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_slice += t1 - t0; t0 = t1; }
                    /* stream-ordered copy of this expert's silu output */
                    const float *fk_dev = gpu_pipe
                        ? fatcat_gpu_xstage(fk, (size_t)m->n_ff_exp * 4) : NULL;
                    gemv_expert(m, &L->ex_down, d, L->ex_down_stride, ids[k],
                                base_tag | (3ull << 32), fk, fc + (size_t)k * sz_big,
                                n_embd, m->n_ff_exp, ek[k][2] != 0, gpu_pipe, fk_dev);
                    if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_queue += t1 - t0; t0 = t1; }
                }
                if (gpu_pipe) fatcat_gpu_wait();
                if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; m->p_r_wait += t1 - t0; t0 = t1; }
                for (uint32_t k = 0; k < topk; k++) {
                    const float *ck = fc + (size_t)k * sz_big;
                    for (int64_t i = 0; i < n_embd; i++) out[i] += wts[k] * ck[i];
                }
                for (uint32_t k = 0; k < topk; k++)
                    for (int t = 0; t < 3; t++)
                        if (ek[k][t])
                            fatcat_stream_release(m->stream, ek[k][t]);
                if (dump_fp) fwrite(out, sizeof(float), (size_t)n_embd, dump_fp); /* ffn out */
                for (int64_t i = 0; i < n_embd; i++) x[i] += out[i];
                if (m->prof) { double t1 = prof_now(); m->p_routed += t1 - t0; t0 = t1; }
            }
            /* issue speculative next-layer prefetch AFTER this layer's
             * critical-path reads (lower priority), then rotate the guess
             * so the next layer can score it. Works for dense layers too
             * (covers the dense->MoE boundary). */
            if (m->stream && cur_spec_n > 0)
                prefetch_experts(m, il + 1, &m->layers[il + 1], cur_spec,
                                 (uint32_t)(cur_spec_n < m->specpf
                                             ? cur_spec_n : m->specpf), 1);
            prev_spec_n = cur_spec_n;
            memcpy(prev_spec, cur_spec, sizeof(prev_spec));
            if (dump_fp) fwrite(x, sizeof(float), (size_t)n_embd, dump_fp);
        }
        if (m->prof) { m->p_tok += prof_now() - tk0; m->p_tokens++; }
        m->health_tokens++;
        {   /* health check: hidden state must stay finite */
            int bad = 0;
            for (int64_t i = 0; i < n_embd; i++)
                if (!isfinite(x[i])) { bad = 1; break; }
            if (bad) m->health_bad++;
        }
        if (row_logits)   /* D3 verification rows (sequential fallback) */
            logits_row(m, x, row_logits + t * (int64_t)m->info.n_vocab, xn);
        if (m->token_cb) m->token_cb(m->token_ctx, t + 1, n_tokens);
        fatcat_stream_tick(m->stream);   /* recency clock: one tick per token */
    }
    m->seq_len += n_tokens;
    if (last_logits && n_tokens)
        logits_row(m, x, last_logits, xn);
    rc = 0;
done:
    if (dump_fp) fclose(dump_fp);
    return rc;
}
