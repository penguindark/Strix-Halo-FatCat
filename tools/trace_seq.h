/* trace_seq.h — per-token MoE routing sequence trace writer (FCTS v1).
 *
 * Records every router decision as it happens, for offline warm-tier cache
 * policy simulation (tools/stream-sim). Complements expert-profiler's FCT1
 * aggregate hotness dumps with the actual per-token SEQUENCE, which is what
 * determines cache hit rates (temporal locality, domain shifts).
 *
 * Format (little-endian):
 *   header: "FCTS" u32 version=1 u32 n_layers u32 n_experts u32 topk
 *   records (repeated): u32 token u32 layer u16 ids[topk]
 *   (token = KV position; layer = global layer id; MoE layers only)
 *
 * Single-translation-unit use: include from one .c file per tool.
 */
#ifndef FATCAT_TRACE_SEQ_H
#define FATCAT_TRACE_SEQ_H

#include <stdint.h>
#include <stdio.h>

#include "fatcat/ml4.h"

typedef struct {
    FILE     *f;
    uint32_t  topk;
    uint64_t  records;
} trace_seq;

static void trace_seq_cb(void *ctx, uint32_t layer, uint32_t token,
                         const uint16_t *ids, const float *weights, uint32_t k)
{
    trace_seq *t = (trace_seq *)ctx;
    (void)weights;
    if (k != t->topk) return;   /* format assumes fixed topk */
    fwrite(&token, 4, 1, t->f);
    fwrite(&layer, 4, 1, t->f);
    fwrite(ids, 2, k, t->f);
    t->records++;
}

static int trace_seq_open(trace_seq *t, const char *path, uint32_t n_layers,
                          uint32_t n_experts, uint32_t topk)
{
    t->f = fopen(path, "wb");
    if (!t->f) return -1;
    fwrite("FCTS", 1, 4, t->f);
    uint32_t v = 1;
    fwrite(&v, 4, 1, t->f);
    fwrite(&n_layers, 4, 1, t->f);
    fwrite(&n_experts, 4, 1, t->f);
    fwrite(&topk, 4, 1, t->f);
    t->topk = topk;
    t->records = 0;
    return 0;
}

static void trace_seq_close(trace_seq *t, const char *path)
{
    if (!t->f) return;
    fflush(t->f);
    fprintf(stderr, "trace: %llu records -> %s\n",
            (unsigned long long)t->records, path);
    fclose(t->f);
    t->f = NULL;
}

#endif /* FATCAT_TRACE_SEQ_H */
