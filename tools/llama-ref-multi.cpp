/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* llama-ref-multi — per-position reference logits via llama.cpp.
 *
 * Feeds a pre-tokenized sequence in ONE batch decode and dumps the logits
 * at every position: file layout = n_pos blocks of n_vocab f32.
 *
 * Build (ad hoc, against a built llama.cpp tree):
 *   g++ -O2 -std=c++17 tools/llama-ref-multi.cpp \
 *       -I<llama>/include -I<llama>/ggml/include \
 *       -L<llama>/build/src -L<llama>/build/ggml/src -lllama -lggml \
 *       -o build/llama-ref-multi
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tokens.txt out.bin\n", argv[1]); return 2; }

    FILE * f = fopen(argv[2], "r");
    if (!f) { perror("tokens"); return 1; }
    std::vector<llama_token> ids;
    long v;
    while (fscanf(f, "%ld", &v) == 1) ids.push_back((llama_token)v);
    fclose(f);
    fprintf(stderr, "feeding %zu tokens\n", ids.size());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    bool seq = argc > 4 && !strcmp(argv[4], "--seq");
    int n_gen = 0;
    if (argc > 4 && !strcmp(argv[4], "--gen") && argc > 5) n_gen = atoi(argv[5]);
    fprintf(stderr, "mode: %s\n", n_gen ? "generate" : (seq ? "sequential" : "batch"));

    if (n_gen) {
        /* greedy generation from the pre-tokenized prompt; ids printed to stdout */
        llama_batch batch = llama_batch_get_one(ids.data(), (int)ids.size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_token tok;
        {
            const float * lg = llama_get_logits_ith(ctx, (int)ids.size() - 1);
            int best = 0;
            for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
            tok = best;
        }
        for (int i = 0; i < n_gen; i++) {
            printf("%d\n", tok);
            fflush(stdout);
            char piece[256];
            int n = llama_token_to_piece(vocab, tok, piece, sizeof piece, 0, true);
            if (n > 0) { piece[n] = 0; fprintf(stderr, "%s", piece); }
            batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            const float * lg = llama_get_logits_ith(ctx, 0);
            int best = 0;
            for (int j = 1; j < n_vocab; j++) if (lg[j] > lg[best]) best = j;
            tok = best;
        }
        fprintf(stderr, "\n");
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    FILE * out = fopen(argv[3], "wb");
    if (!out) { perror("out"); return 1; }

    if (seq) {
        /* one token per decode — different GEMV/GEMM paths than batch mode */
        for (size_t i = 0; i < ids.size(); i++) {
            llama_batch batch = llama_batch_get_one(&ids[i], 1);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            const float * lg = llama_get_logits_ith(ctx, 0);
            fwrite(lg, 4, n_vocab, out);
        }
    } else {
        llama_batch batch = llama_batch_get_one(ids.data(), (int)ids.size());
        std::vector<int8_t> want_logits(ids.size(), 1);   // logits at every position
        batch.logits = want_logits.data();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (size_t i = 0; i < ids.size(); i++) {
            const float * lg = llama_get_logits_ith(ctx, (int)i);
            fwrite(lg, 4, n_vocab, out);
        }
    }
    fclose(out);
    fprintf(stderr, "wrote %zu x %d f32 to %s\n", ids.size(), n_vocab, argv[3]);

    llama_free(ctx);
    llama_free_model(model);
    return 0;
}
