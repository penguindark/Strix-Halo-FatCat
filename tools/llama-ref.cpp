/* llama-ref — reference logits via llama.cpp for FatCat cross-checking.
 *
 * Loads a GGUF, feeds a pre-tokenized id sequence (whitespace-separated ints),
 * runs a single forward pass, and writes the top-32 (id, logit) pairs of the
 * last position to a binary file (same format as expert-profiler --check).
 *
 * Build (ad hoc, against a built llama.cpp tree):
 *   g++ -O2 -std=c++17 tools/llama-ref.cpp \
 *       -I<llama>/include -I<llama>/ggml/include \
 *       -L<llama>/build/src -L<llama>/build/ggml/src -lllama -lggml \
 *       -o build/llama-ref
 * Run: LD_LIBRARY_PATH=<llama>/build/src:<llama>/build/ggml/src ./build/llama-ref MODEL.gguf tokens.txt out.bin
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tokens.txt out.bin\n", argv[0]); return 2; }

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
    fprintf(stderr, "vocab size: %d, add_bos: %d\n", llama_vocab_n_tokens(vocab),
            llama_vocab_get_add_bos(vocab));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)ids.size() + 8;
    cp.n_batch = (uint32_t)ids.size();
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    llama_batch batch = llama_batch_get_one(ids.data(), (int)ids.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const float * logits = llama_get_logits_ith(ctx, (int)ids.size() - 1);
    int n_vocab = llama_vocab_n_tokens(vocab);

    /* full logits dump: <out>.full.bin (f32[n_vocab]) */
    {
        std::string fullp = std::string(argv[3]) + ".full.bin";
        FILE * fa = fopen(fullp.c_str(), "wb");
        if (fa) { fwrite(logits, 4, n_vocab, fa); fclose(fa); }
    }
    std::vector<std::pair<float, int>> top;
    for (int i = 0; i < n_vocab; i++) top.push_back({logits[i], i});
    std::partial_sort(top.begin(), top.begin() + 32, top.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    FILE * out = fopen(argv[3], "wb");
    for (int i = 0; i < 32; i++) {
        int id = top[i].second;
        float val = top[i].first;
        fwrite(&id, 4, 1, out);
        fwrite(&val, 4, 1, out);
    }
    fclose(out);

    for (int i = 0; i < 5; i++) {
        char buf[128] = {0};
        llama_token_to_piece(vocab, top[i].second, buf, sizeof(buf), 0, false);
        printf("  %5d  %6.2f  '%s'\n", top[i].second, top[i].first, buf);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
