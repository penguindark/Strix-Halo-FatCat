/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* logit-dump — per-position fatcat logits for incremental decode testing.
 *
 * Loads the model, feeds tokens one at a time through the incremental
 * decode path (prefill = first token, then step), and dumps the logits
 * produced at every position: n_pos blocks of n_vocab f32.
 *
 * Usage: logit-dump MODEL.gguf --tokens FILE --out OUT.bin
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "fatcat/gguf.h"
#include "fatcat/gpu.h"
#include "fatcat/ml4.h"
#include "fatcat/stream.h"

#include "trace_seq.h"

/* libgomp reads OMP_WAIT_POLICY at load time (before main). The engine's
 * decode path blocks on expert I/O between small GEMV regions; spinning
 * worker threads then burn every core and slow the real work ~3x. Default
 * to passive waiting via a one-time self re-exec unless the user set a
 * policy explicitly. */
static void omp_passive_default(int argc, char **argv)
{
    (void)argc;
    if (getenv("OMP_WAIT_POLICY") || getenv("GOMP_SPINCOUNT") ||
        getenv("FATCAT_OMP_REEXEC"))
        return;
    setenv("OMP_WAIT_POLICY", "passive", 1);
    setenv("FATCAT_OMP_REEXEC", "1", 1);
    execv("/proc/self/exe", argv);
    /* on failure just continue with defaults */
}

int main(int argc, char **argv)
{
    omp_passive_default(argc, argv);
    const char *model = NULL, *tok_path = NULL, *out_path = NULL;
    const char *trace_path = NULL;
    int batch_n = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--trace-seq") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_n = atoi(argv[++i]);
        else if (argv[i][0] != '-') model = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!model || !tok_path || (!out_path && !trace_path)) {
        fprintf(stderr, "usage: %s MODEL --tokens F [--out F] [--trace-seq F] [--batch N]\n", argv[0]);
        return 2;
    }
    if (batch_n > 0) {
        char b[16]; snprintf(b, sizeof b, "%d", batch_n);
        setenv("FATCAT_BATCH", b, 1);   /* read at ml4 open */
    }

    size_t cap = 8192;
    uint32_t *toks = malloc(cap * sizeof *toks);
    size_t n = 0;
    FILE *f = fopen(tok_path, "r");
    if (!f) { perror("tokens"); return 1; }
    long v;
    while (n < cap && fscanf(f, "%ld", &v) == 1) toks[n++] = (uint32_t)v;
    fclose(f);
    if (!n) { fprintf(stderr, "no tokens\n"); return 1; }

    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(model, err, sizeof err);
    if (!g) { fprintf(stderr, "gguf: %s\n", err); return 1; }
    fatcat_ml4 *m = fatcat_ml4_open(g, err, sizeof err);
    if (!m) { fprintf(stderr, "ml4: %s\n", err); return 1; }
    fatcat_ml4_info info = fatcat_ml4_get_info(m);

    float *lg = malloc((size_t)info.n_vocab * sizeof(float));
    FILE *out = out_path ? fopen(out_path, "wb") : NULL;
    if (!lg || (out_path && !out)) { fprintf(stderr, "oom/out\n"); return 1; }

    trace_seq tr = { 0 };
    if (trace_path) {
        if (trace_seq_open(&tr, trace_path, info.n_layers,
                           info.n_experts, info.n_experts_used) != 0) {
            perror("trace-seq"); return 1;
        }
        fatcat_ml4_set_router_cb(m, trace_seq_cb, &tr);
    }

    fatcat_ml4_reset(m);
    size_t rows_written = 0;
    if (batch_n > 0) {
        /* D1: one batched prefill call; writes only the final-position logits */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (fatcat_ml4_last_logits(m, toks, n, lg) != 0) {
            fprintf(stderr, "batched forward failed\n"); return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        if (out) { fwrite(lg, sizeof(float), info.n_vocab, out); rows_written = 1; }
        fprintf(stderr, "batched prefill: %zu tokens in %.1f s = %.2f tok/s (chunk %d)\n",
                n, el, n / (el > 0 ? el : 1), batch_n);
    }
    for (size_t i = 0; batch_n == 0 && i < n; i++) {
        if (fatcat_ml4_last_logits(m, &toks[i], 1, lg) != 0) {
            fprintf(stderr, "forward failed at %zu\n", i); return 1;
        }
        if (out) { fwrite(lg, sizeof(float), info.n_vocab, out); rows_written++; }
        fprintf(stderr, "pos %zu tok %u done\n", i, toks[i]);
        if ((i + 1) % 100 == 0) {   /* periodic so killed runs still report */
            fatcat_stream_stats ps;
            fatcat_stream_get_stats(fatcat_ml4_stream(m), &ps);
            fprintf(stderr, "warm tier @%zu: hits=%llu misses=%llu hit%%=%.1f\n",
                    i + 1, (unsigned long long)ps.hits,
                    (unsigned long long)ps.misses,
                    ps.hits + ps.misses
                        ? 100.0 * ps.hits / (ps.hits + ps.misses) : 0.0);
        }
    }
    if (out) fclose(out);
    if (trace_path) trace_seq_close(&tr, trace_path);
    if (out_path)
        fprintf(stderr, "wrote %zu x %u f32 to %s\n", rows_written, info.n_vocab, out_path);
    {
        fatcat_stream_stats ss;
        fatcat_stream_get_stats(fatcat_ml4_stream(m), &ss);
        if (ss.hits + ss.misses)
            fprintf(stderr, "warm tier [%s]: hits=%llu misses=%llu evictions=%llu "
                    "hit%%=%.1f used=%llu/%llu slots\n",
                    fatcat_stream_policy(fatcat_ml4_stream(m)),
                    (unsigned long long)ss.hits, (unsigned long long)ss.misses,
                    (unsigned long long)ss.evictions,
                    100.0 * ss.hits / (ss.hits + ss.misses),
                    (unsigned long long)ss.used_slots,
                    (unsigned long long)ss.n_slots);
    }
    {
        uint64_t gc = 0, gb = 0, gh = 0;
        fatcat_gpu_stats(&gc, &gb, &gh);
        uint64_t gm_c = 0, gm_f = 0;
        fatcat_gpu_gemm_stats(&gm_c, &gm_f);
        if (gc || gh)
            fprintf(stderr, "gpu: copies=%llu copy_GB=%.2f stage_hits=%llu upload_fails=%llu "
                    "gemm_calls=%llu gemm_fails=%llu\n",
                    (unsigned long long)gc, gb / 1e9, (unsigned long long)gh,
                    (unsigned long long)fatcat_gpu_upload_fails(),
                    (unsigned long long)gm_c, (unsigned long long)gm_f);
    }
    {
        uint64_t st_total = 0, st_hit = 0;
        fatcat_ml4_get_spec(m, &st_total, &st_hit);
        if (st_total)
            fprintf(stderr, "router speculation: %llu/%llu actual experts "
                    "pre-guessed (%.1f%%)\n",
                    (unsigned long long)st_hit, (unsigned long long)st_total,
                    100.0 * st_hit / st_total);
    }
    {   /* peak RSS of this process: keep test footprints auditable */
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            fprintf(stderr, "peak RSS: %ld MB\n", ru.ru_maxrss / 1024);
    }
    fatcat_ml4_close(m);
    fatcat_gguf_close(g);
    free(lg);
    free(toks);
    return 0;
}
