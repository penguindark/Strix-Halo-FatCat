/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* llama-layer-dump — dump per-layer residual-stream states via the public
 * llama-ext.h per-layer embeddings API (llama_set/get_embeddings_layer_inp).
 *
 * layer_inp[lid] = input of layer lid (lid 0 = token embeddings, lid n = final).
 * Output file: for each lid in 0..n_layer: n_pos x n_embd f32.
 *
 * Build:
 *   g++ -O2 -std=c++17 tools/llama-layer-dump.cpp \
 *       -I<llama>/include -I<llama>/src -I<llama>/ggml/include \
 *       -L<llama>/build/bin -lllama -lggml -o build/llama-layer-dump
 */
#include "llama.h"
#include "llama-ext.h"
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

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd_out(model);
    fprintf(stderr, "n_layer=%d n_embd=%d n_pos=%zu\n", n_layer, n_embd, ids.size());

    for (int lid = 0; lid <= n_layer; lid++) {
        llama_set_embeddings_layer_inp(ctx, (uint32_t)lid, true);
    }

    llama_batch batch = llama_batch_get_one(ids.data(), (int)ids.size());
    std::vector<int8_t> want(ids.size(), 1);
    batch.logits = want.data();
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    FILE * out = fopen(argv[3], "wb");
    if (!out) { perror("out"); return 1; }
    for (int lid = 0; lid <= n_layer; lid++) {
        const float * e = llama_get_embeddings_layer_inp(ctx, (uint32_t)lid);
        if (!e) { fprintf(stderr, "layer %d: no data\n", lid); return 1; }
        fwrite(e, 4, (size_t)ids.size() * n_embd, out);
    }
    fclose(out);
    fprintf(stderr, "wrote %d x %zu x %d f32\n", n_layer + 1, ids.size(), n_embd);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
