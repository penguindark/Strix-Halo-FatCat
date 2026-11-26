/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat — Mistral-Large-4 forward pass (scalar reference, streaming-friendly).
 *
 * Minimal but faithful implementation of the ML4 layer stack, designed for
 * low memory: all weights stay mmap'd in the GGUF (never bulk-copied), only
 * small per-token scratch buffers are allocated. Used by the expert profiler
 * and as the seed of the P2 engine.
 *
 * Supported configuration (validated on the layers 0-2 slice):
 *  - MLA attention (absorbed form, MQA over 512+64 latent keys)
 *  - LeRoPE (learned per-frequency multipliers from rope_freqs.weight)
 *  - SSMax query scaling: q *= s_h * log(pos+1) + b_h
 *  - dense FFN (leading layers) and MoE (top-k routed + shared experts)
 *  - DSA indexer is skipped when seq_len <= indexer top_k (selection = all)
 *  - KV sharing (kv_source_map) supported; layers must read from self or an
 *    earlier layer
 * Not yet: DSA top-k masking for long prompts, IQ1_S experts (P2), batching. */
#ifndef FATCAT_ML4_H
#define FATCAT_ML4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "fatcat/gguf.h"

typedef struct fatcat_ml4 fatcat_ml4;

typedef struct {
    uint32_t n_layers;
    uint32_t n_dense_layers;   /* leading layers with a dense FFN */
    uint32_t n_experts;        /* routed experts per MoE layer */
    uint32_t n_experts_used;   /* top-k */
    uint32_t n_embd;
    uint32_t n_vocab;
} fatcat_ml4_info;

/* Called for every MoE routing decision.
 * layer = global layer id; token = index into the prompt;
 * ids/weights = top-k expert ids and normalized weights. */
typedef void (*fatcat_router_cb)(void *ctx, uint32_t layer, uint32_t token,
                                 const uint16_t *ids, const float *weights,
                                 uint32_t k);

fatcat_ml4 *fatcat_ml4_open(fatcat_gguf *g, char *errbuf, size_t errlen);
void fatcat_ml4_close(fatcat_ml4 *m);
fatcat_ml4_info fatcat_ml4_get_info(const fatcat_ml4 *m);

/* Forward pass over tokens[0..n_tokens) (single sequence, causal).
 * Positions continue from the current sequence length: call repeatedly with
 * one token to decode incrementally. router_cb may be NULL.
 * Returns 0 on success, -1 on error (stderr). */
int fatcat_ml4_forward(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                       fatcat_router_cb router_cb, void *cb_ctx);

/* Rewind the sequence to position 0 (KV entries are overwritten lazily). */
void fatcat_ml4_reset(fatcat_ml4 *m);
size_t fatcat_ml4_seq_len(const fatcat_ml4 *m);
size_t fatcat_ml4_kv_cap(const fatcat_ml4 *m);

/* Default router callback for ALL forward/last_logits calls that pass no
 * cb of their own (tracing, profiling). Pass NULL to clear. */
void fatcat_ml4_set_router_cb(fatcat_ml4 *m, fatcat_router_cb cb, void *ctx);

/* Same forward pass, but also writes final logits for the LAST token into
 * logits[n_vocab] (output_norm + lm_head). Requires output.weight present. */
int fatcat_ml4_last_logits(fatcat_ml4 *m, const uint32_t *tokens, size_t n_tokens,
                           float *logits);

/* Forward pass writing logits for EVERY position into rows[n * n_vocab]
 * (row i = logits after tokens[i], same math as the last-row path).
 * Used by speculative-decode verification. */
int fatcat_ml4_forward_rows(fatcat_ml4 *m, const uint32_t *tokens, size_t n,
                            float *rows);

/* Rewind the sequence to new_len (no-op if new_len > current). Positions
 * beyond new_len are dropped; stale KV is never read. Used to discard
 * rejected speculative tokens. */
void fatcat_ml4_rewind(fatcat_ml4 *m, size_t new_len);

/* Progress hook: called after each processed token with (ctx, done, total). */
typedef void (*fatcat_token_cb)(void *ctx, size_t done, size_t total);
void fatcat_ml4_set_token_cb(fatcat_ml4 *m, fatcat_token_cb cb, void *ctx);

/* DSA indexer top-k: writes the indices of the k largest values of s[0..n)
 * into out[], sorted ascending. Exposed for unit testing. */
void fatcat_topk_select(const float *s, int64_t n, int64_t k, int64_t *out);

/* Health counters: tokens processed and tokens that produced non-finite
 * (NaN/Inf) hidden states or logits. bad == 0 means the math path is healthy. */
void fatcat_ml4_get_health(const fatcat_ml4 *m, uint64_t *tokens, uint64_t *bad);

/* D5a phase profile (FATCAT_PROF=1): accumulated per-token seconds in the
 * sequential decode path, per phase; tok_total covers the same tokens
 * (residual = tok_total - attn - router - shffn - routed - dense).
 * attn_core/attn_wout split the attn bucket (attention_core / wout gemv;
 * the input projections are attn - core - wout). */
void fatcat_ml4_get_prof(const fatcat_ml4 *m, double *attn, double *router,
                         double *shffn, double *routed, double *dense,
                         double *tok_total, uint64_t *tokens,
                         double *attn_core, double *attn_wout,
                         double *absorb, double *scan, double *wv,
                         double *r_slice, double *r_queue, double *r_wait,
                         double *r_act, double *r_pre);

/* The model's warm-tier stream manager (may be NULL if disabled). */
struct fatcat_stream *fatcat_ml4_stream(fatcat_ml4 *m);

/* Router-speculation counters: of the actual top-k expert slots seen,
 * total counts those that had a speculative prefetch pending, hit counts
 * how many of the actual experts were correctly guessed. */
void fatcat_ml4_get_spec(const fatcat_ml4 *m, uint64_t *total, uint64_t *hit);

/* H3 cache-aware tail-skip counters: routed experts dropped (cold miss +
 * weight below FATCAT_SKIP * w_top1) and the sum of their normalized
 * weights scaled by 1000 (quality accounting). */
void fatcat_ml4_get_skip(const fatcat_ml4 *m, uint64_t *n, uint64_t *w);

/* Pista A counters: expert slices served from the mmap view because the
 * warm tier could not serve them, and bytes staged (hipMemcpy'd) into GPU
 * staging slots. Both mark double NVMe reads (ring fill + page re-fault). */
void fatcat_ml4_get_stage(const fatcat_ml4 *m, uint64_t *slice_n,
                          uint64_t *slice_bytes, uint64_t *stage_n,
                          uint64_t *stage_bytes);

#endif /* FATCAT_ML4_H */
