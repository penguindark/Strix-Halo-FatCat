/* fatcat — engine core: context lifecycle and the generation loop.

 * Wraps the GGUF reader + ML4 model into a prefill/decode API with greedy
 * sampling. Text tokenization is out of scope here (use tools/py/tokenize.py
 * to produce token id sequences); the engine consumes ids.
 */
#ifndef FATCAT_ENGINE_H
#define FATCAT_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "fatcat/ml4.h"   /* fatcat_router_cb */

typedef struct fatcat_engine fatcat_engine;

typedef struct {
    const char *model_path;
    int use_gpu;          /* -1 = auto (env FATCAT_GPU / HIP build) */
    long mem_limit_mb;    /* 0 = model default (FATCAT_MEM_LIMIT_MB) */
    long mem_budget_mb;   /* 0 = env/default (FATCAT_MEM_BUDGET_GB): hard
                           * cap on the engine's own memory */
} fatcat_engine_config;

/* Called for every produced token: id = the token just generated,
 * index = 0-based position within the generated continuation,
 * n_new = total requested (index == n_new-1 => last). */
typedef void (*fatcat_gen_cb)(void *ctx, uint32_t id, size_t index, size_t n_new);

fatcat_engine *fatcat_engine_open(const fatcat_engine_config *cfg,
                                  char *errbuf, size_t errlen);
void fatcat_engine_close(fatcat_engine *e);

/* Rewind the conversation (KV positions restart at 0). */
void fatcat_engine_reset(fatcat_engine *e);

/* Current conversation length (tokens in KV) and KV capacity, in tokens.
 * cap == 0 means not yet allocated. */
void fatcat_engine_context(const fatcat_engine *e, size_t *seq, size_t *cap);

/* Install a default router-decision tracer (see fatcat_router_cb in
 * fatcat/ml4.h); fires for prefill and decode steps. NULL clears. */
void fatcat_engine_set_router_cb(fatcat_engine *e, fatcat_router_cb cb, void *ctx);

/* Model dimensions (layers, experts, vocab) for tooling. */
fatcat_ml4_info fatcat_engine_info(const fatcat_engine *e);

/* Warm-tier stream handle for stats tooling (see fatcat/stream.h). */
struct fatcat_stream *fatcat_engine_stream(const fatcat_engine *e);

/* D5a phase profile (FATCAT_PROF=1): per-token seconds per decode phase
 * accumulated over the session (see fatcat_ml4_get_prof). */
void fatcat_engine_prof(const fatcat_engine *e, double *attn, double *router,
                        double *shffn, double *routed, double *dense,
                        double *tok_total, uint64_t *tokens,
                        double *attn_core, double *attn_wout,
                        double *absorb, double *scan, double *wv,
                        double *r_slice, double *r_queue, double *r_wait,
                        double *r_act, double *r_pre);

/* Feed a token sequence without generating (prompt processing). */
int fatcat_engine_prefill(fatcat_engine *e, const uint32_t *tokens, size_t n);

/* Forward one token and return the greedy (argmax) continuation. */
int fatcat_engine_step(fatcat_engine *e, uint32_t token, uint32_t *next_out);

/* Full generation: reset, prefill prompt, sample n_new tokens greedily,
 * invoking cb per token. Returns 0 on success. */
int fatcat_engine_generate(fatcat_engine *e, const uint32_t *prompt, size_t n_prompt,
                           size_t n_new, fatcat_gen_cb cb, void *ctx);

/* Timing stats (seconds; 0 if unused). */
void fatcat_engine_stats(const fatcat_engine *e, double *prefill_s,
                         double *decode_s, size_t *n_prefill, size_t *n_decode);

/* D3 prompt-lookup speculation: enabled + counters (steps, tokens fed into
 * verification passes, tokens emitted). Enabled with FATCAT_SPEC=1. */
void fatcat_engine_spec_stats(const fatcat_engine *e, int *on,
                              uint64_t *steps, uint64_t *fed, uint64_t *emitted);

/* expert-routing speculation accuracy (next-layer prefetch hit rate) */
void fatcat_engine_espec_stats(const fatcat_engine *e, uint64_t *total,
                               uint64_t *hit);

/* decode-window block-device reads (bytes, seconds) — ground-truth NVMe.
 * self_bytes: device reads by this process only (rest is system noise). */
void fatcat_engine_io_stats(const fatcat_engine *e, uint64_t *bytes,
                            double *secs, uint64_t *self_bytes);

/* Discard accepted-but-unemitted speculative tokens (call when a
 * generation run stops early, e.g. on EOS). */
void fatcat_engine_flush(fatcat_engine *e);

#endif /* FATCAT_ENGINE_H */
