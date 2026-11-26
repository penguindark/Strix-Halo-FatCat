/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat-run — generate tokens with the FatCat engine (P2 entry point).
 *
 * Usage: fatcat-run MODEL.gguf --tokens PROMPT.tokens -n N [--cpu]
 *   --tokens FILE   whitespace-separated token ids (prompt; use
 *                   tools/py/tokenize.py to produce them)
 *   -n N            number of tokens to generate (greedy)
 *   --cpu           disable the GPU backend
 *
 * Prints each generated token id as it appears, then timing stats.
 *
 * Server mode (interactive chat driver, see tools/py/chat.py):
 *   fatcat-run MODEL.gguf --serve
 *   stdin commands (one per line):
 *     P <id>...        prefill tokens (no generation)      -> "OK"
 *     G <max> <eos>    step-generate greedily, stop at eos -> "T <id>" per
 *                      token as produced, then "GEN <count>" (plus " ERR"
 *                      on step failure, " LOOP" if an exact-repetition
 *                      degeneration guard stopped it early)
 *     I                report context usage                -> "INFO <seq> <cap>"
 *     R                rewind conversation (KV restarts)   -> "OK"
 *     Q                quit
 *   The engine stays loaded and the KV cache persists across commands,
 *   so multi-turn conversations pay prefill only for the new turn.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "fatcat/engine.h"
#include "fatcat/stream.h"
#include "fatcat/gpu.h"

#include "trace_seq.h"

static void on_token(void *ctx, uint32_t id, size_t index, size_t n_new)
{
    (void)ctx; (void)n_new;
    printf("%zu %u\n", index, id);
    fflush(stdout);
}

/* Server mode: persistent engine across stdin commands (see header). */
static int serve(fatcat_engine *e)
{
    char line[1 << 16];
    uint32_t last = 0;          /* last token fed or generated (step input) */
    int have_last = 0;
    printf("READY\n"); fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'Q') break;
        if (line[0] == 'R') {
            fatcat_engine_reset(e); have_last = 0;
            printf("OK\n"); fflush(stdout);
            continue;
        }
        if (line[0] == 'I') {
            size_t seq = 0, cap = 0;
            fatcat_engine_context(e, &seq, &cap);
            printf("INFO %zu %zu\n", seq, cap); fflush(stdout);
            continue;
        }
        if (line[0] == 'P') {
            size_t cap = 4096, n = 0;
            uint32_t *toks = malloc(cap * sizeof *toks);
            char *p = line + 1;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
                if (!*p) break;
                char *end;
                long v = strtol(p, &end, 10);
                if (end == p) break;   /* non-numeric: stop parsing */
                if (n == cap) { cap *= 2; toks = realloc(toks, cap * sizeof *toks); }
                toks[n++] = (uint32_t)v;
                p = end;
            }
            int rc = n ? fatcat_engine_prefill(e, toks, n) : 0;
            if (n) { last = toks[n - 1]; have_last = 1; }
            free(toks);
            printf(rc == 0 ? "OK\n" : "ERR prefill\n"); fflush(stdout);
            continue;
        }
        if (line[0] == 'G') {
            long max_new = 128, eos = 2;
            sscanf(line + 1, "%ld %ld", &max_new, &eos);
            size_t cnt = 0, hcap = 0;
            int rc = 0, looped = 0;
            uint32_t *hist = NULL;   /* generation history for loop guard */
            const size_t W = 24;     /* exact-repetition window (tokens) */
            if (!have_last) { printf("ERR no context\n"); fflush(stdout); continue; }
            for (long i = 0; i < max_new && !rc; i++) {
                uint32_t next;
                rc = fatcat_engine_step(e, last, &next);
                if (rc) break;
                printf("T %u\n", next); fflush(stdout);
                last = next; cnt++;
                if ((long)next == eos) { fatcat_engine_flush(e); break; }
                if (cnt + 1 > hcap) {
                    hcap = cnt + 1 < 1024 ? 1024 : (cnt + 1) * 2;
                    hist = realloc(hist, hcap * sizeof *hist);
                }
                hist[cnt - 1] = next;
                /* greedy degeneration guard: the last W tokens exactly
                 * repeat the W before them -> the model is cycling; stop
                 * early instead of burning max_new (minutes of NVMe time) */
                if (cnt >= 2 * W &&
                    memcmp(hist + cnt - W, hist + cnt - 2 * W,
                           W * sizeof *hist) == 0) { looped = 1; break; }
            }
            free(hist);
            printf("GEN %zu%s%s\n", cnt, rc ? " ERR" : "",
                   looped ? " LOOP" : ""); fflush(stdout);
            continue;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *tok_path = NULL;
    size_t n_new = 8;
    int use_gpu = -1;
    int serve_mode = 0;
    const char *trace_path = NULL;
    int oks[8]; int n_oks = 0;   /* --spec-oracle K list */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) use_gpu = 0;
        else if (!strcmp(argv[i], "--serve")) serve_mode = 1;
        else if (!strcmp(argv[i], "--trace-seq") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--spec-oracle") && i + 1 < argc) {
            char *s = argv[++i], *tok;
            for (tok = strtok(s, ","); tok && n_oks < 8; tok = strtok(NULL, ","))
                oks[n_oks++] = atoi(tok);
        }
        else if (argv[i][0] != '-') model = argv[i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!model || (!serve_mode && !tok_path)) {
        fprintf(stderr, "usage: %s MODEL.gguf (--tokens FILE [-n N] | --serve) [--cpu]"
                        " [--spec-oracle K,K,...]\n", argv[0]);
        return 2;
    }

    char err[256];
    fatcat_engine_config cfg = { .model_path = model, .use_gpu = use_gpu };
    fatcat_engine *e = fatcat_engine_open(&cfg, err, sizeof err);
    if (!e) { fprintf(stderr, "engine open failed: %s\n", err); return 1; }

    trace_seq tr = { 0 };
    if (trace_path) {
        fatcat_ml4_info mi = fatcat_engine_info(e);
        if (trace_seq_open(&tr, trace_path, mi.n_layers,
                           mi.n_experts, mi.n_experts_used) != 0) {
            perror("trace-seq"); return 1;
        }
        fatcat_engine_set_router_cb(e, trace_seq_cb, &tr);
    }

    if (serve_mode) {
        int rc = serve(e);
        if (trace_path) trace_seq_close(&tr, trace_path);
        fatcat_engine_close(e);
        return rc;
    }

    FILE *f = fopen(tok_path, "r");
    if (!f) { perror("tokens"); return 1; }
    size_t cap = 256, n = 0;
    uint32_t *prompt = malloc(cap * sizeof *prompt);
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (n == cap) { cap *= 2; prompt = realloc(prompt, cap * sizeof *prompt); }
        prompt[n++] = (uint32_t)v;
    }
    fclose(f);
    if (!n) { fprintf(stderr, "no prompt tokens\n"); return 1; }

    if (fatcat_engine_generate(e, prompt, n, n_new, on_token, NULL) != 0) {
        fprintf(stderr, "generation failed\n");
        fatcat_engine_close(e); free(prompt); return 1;
    }

    if (n_oks) {   /* H2 oracle benchmark instead of the stats report */
        int rc = fatcat_engine_spec_oracle(e, prompt, n, n_new, oks, n_oks);
        free(prompt); fatcat_engine_close(e);
        return rc ? 1 : 0;
    }

    double ps, ds; size_t np, nd;
    fatcat_engine_stats(e, &ps, &ds, &np, &nd);
    {   /* warm-tier counters: the decode-vs-tier story lives in the hit rate */
        fatcat_stream_stats ss;
        fatcat_stream_get_stats(fatcat_engine_stream(e), &ss);
        if (ss.hits + ss.misses)
            fprintf(stderr, "warm tier [%s]: hits=%llu misses=%llu evictions=%llu "
                    "hit%%=%.1f used=%llu/%llu slots\n",
                    fatcat_stream_policy(fatcat_engine_stream(e)),
                    (unsigned long long)ss.hits, (unsigned long long)ss.misses,
                    (unsigned long long)ss.evictions,
                    100.0 * ss.hits / (ss.hits + ss.misses),
                    (unsigned long long)ss.used_slots,
                    (unsigned long long)ss.n_slots);
        if (ss.hit_bytes + ss.miss_bytes)
            fprintf(stderr, "warm bytes: hit %.2f GB  miss %.2f GB "
                    "(late-prefetch %.2f GB, sync-fill %.2f GB)  ring-read %.2f GB "
                    "(spec %.2f GB)\n",
                    ss.hit_bytes / 1e9, ss.miss_bytes / 1e9,
                    ss.wait_bytes / 1e9, ss.cold_bytes / 1e9,
                    ss.ring_bytes / 1e9, ss.ring_spec_bytes / 1e9);
        if (ss.prefill_skips)
            fprintf(stderr, "prefill anti-thrash: %llu misses not promoted "
                    "(FATCAT_NOPROMO)\n", (unsigned long long)ss.prefill_skips);
        {   /* expert-routing speculation: wrong guesses burn NVMe bandwidth
             * on reads that are never consumed */
            uint64_t est, esh;
            fatcat_engine_espec_stats(e, &est, &esh);
            if (est)
                fprintf(stderr, "expert-spec: %llu/%llu routed experts prefetched "
                        "early and used (%.1f%%)\n",
                        (unsigned long long)esh, (unsigned long long)est,
                        100.0 * esh / est);
        }
        {   /* H3 cache-aware tail-skip accounting */
            uint64_t sn, sw;
            fatcat_engine_skip_stats(e, &sn, &sw);
            if (sn)
                fprintf(stderr, "tail-skip: %llu cold experts dropped "
                        "(dropped weight sum %.2f, avg %.4f/expert)\n",
                        (unsigned long long)sn, sw / 1000.0,
                        sw / 1000.0 / sn);
        }
        {   /* ground-truth block-device reads during the decode window */
            uint64_t iob, ios; double iot;
            fatcat_engine_io_stats(e, &iob, &iot, &ios);
            if (iot > 0.0)
                fprintf(stderr, "decode nvme: %.2f GB device reads in %.1fs "
                        "(%.2f GB/s)  engine-only %.2f GB (%.2f GB/s)\n",
                        iob / 1e9, iot, iob / 1e9 / iot,
                        ios / 1e9, ios / 1e9 / iot);
            uint64_t mj = fatcat_engine_majflt_bytes(e);
            if (mj)
                fprintf(stderr, "decode majflt: %.2f GB mmap refaults "
                        "(non-ring reader: %.2f GB)\n",
                        mj / 1e9, (iob > ios ? iob - ios : 0) / 1e9);
        }
        {   /* Pista A: double-read audit (mmap fallback + GPU staging) */
            uint64_t sn, sb, tn, tb;
            fatcat_engine_stage_stats(e, &sn, &sb, &tn, &tb);
            uint64_t gc, gb, gh;
            fatcat_gpu_stats(&gc, &gb, &gh);
            fprintf(stderr, "staging audit: mmap-fallback slices %llu (%.2f GB)  "
                    "gpu-stage %llu (%.2f GB)  gpu-copies %llu (%.2f GB, "
                    "stage-hits %llu)\n",
                    (unsigned long long)sn, sb / 1e9,
                    (unsigned long long)tn, tb / 1e9,
                    (unsigned long long)gc, gb / 1e9,
                    (unsigned long long)gh);
        }
    }
    int spec_on; uint64_t s_steps, s_fed, s_emit;
    fatcat_engine_spec_stats(e, &spec_on, &s_steps, &s_fed, &s_emit);
    fprintf(stderr, "prefill: %zu tok in %.2fs (%.2f tok/s)  decode: %zu tok in %.2fs (%.2f tok/s)\n",
            np, ps, ps > 0 ? np / ps : 0.0, nd, ds, ds > 0 ? nd / ds : 0.0);
    if (spec_on)
        fprintf(stderr, "spec: %zu verify steps, %llu draft tok fed, %llu tok emitted (%.2f tok/step)\n",
                nd, (unsigned long long)s_fed, (unsigned long long)s_emit,
                nd ? (double)s_emit / nd : 0.0);
    {   /* D5a phase profile (FATCAT_PROF=1) */
        double pa, pr, ps, pro, pd, pt, pc, pw, pab, psc, pv;
        double prs, prq, prw, pra, prp; uint64_t ptk;
        fatcat_engine_prof(e, &pa, &pr, &ps, &pro, &pd, &pt, &ptk,
                           &pc, &pw, &pab, &psc, &pv, &prs, &prq, &prw, &pra, &prp);
        if (ptk) {
            fprintf(stderr, "prof/token: attn %.3f (proj %.3f core %.3f [absorb %.3f scan %.3f wv %.3f misc %.3f] wout %.3f) router %.3f shffn %.3f routed %.3f dense %.3f other %.3f = %.3f s (%llu tok)\n",
                    pa / ptk, (pa - pc - pw) / ptk, pc / ptk,
                    pab / ptk, psc / ptk, pv / ptk, (pc - pab - psc - pv) / ptk,
                    pw / ptk, pr / ptk, ps / ptk, pro / ptk, pd / ptk,
                    (pt - pa - pr - ps - pro - pd) / ptk, pt / ptk,
                    (unsigned long long)ptk);
            fprintf(stderr, "prof/token routed-split: slice %.3f queue %.3f wait %.3f act %.3f pre %.3f (rest %.3f)\n",
                    prs / ptk, prq / ptk, prw / ptk, pra / ptk, prp / ptk,
                    (pro - prs - prq - prw - pra - prp) / ptk);
        }
    }
    {   /* peak RSS of this process: keep test footprints auditable */
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            fprintf(stderr, "peak RSS: %ld MB\n", ru.ru_maxrss / 1024);
    }
    if (trace_path) trace_seq_close(&tr, trace_path);
    fatcat_engine_close(e);
    free(prompt);
    return 0;
}
