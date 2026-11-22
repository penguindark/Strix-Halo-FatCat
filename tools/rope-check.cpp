/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* rope-check — golden comparison: ggml_rope_ext (real llama.cpp numerics)
 * vs fatcat's rope_cache_pos transcription, for the Mistral-Large-4 YaRN config.
 *
 * Build:
 *   g++ -O2 -std=c++17 tools/rope-check.cpp -Iinclude \
 *       -I/home/user/app/llama.cpp/ggml/include \
 *       -Lbuild -lfatcat_io \
 *       -L/home/user/app/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu \
 *       -o build/rope-check
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "fatcat/gguf.h"

/* ---- fatcat rope_cache_pos, copied verbatim from src/model/ml4.c ---- */
static float ffreqs[32];
struct fcfg { int n_rot; float freq_base, freq_scale; uint32_t yarn_orig_ctx;
             float yarn_beta_fast, yarn_beta_slow; const float * rope_freqs; };
static void fatcat_rope(const fcfg & m, int64_t pos, float * cs)
{
    const int n_rot = m.n_rot;
    const float theta_scale = powf(m.freq_base, -2.0f / (float)n_rot);
    float corr[2];
    {
        double lb = log((double)m.freq_base);
        float d_fast = (float)(n_rot * log((double)m.yarn_orig_ctx / (2.0 * m.yarn_beta_fast * M_PI)) / (2.0 * lb));
        float d_slow = (float)(n_rot * log((double)m.yarn_orig_ctx / (2.0 * m.yarn_beta_slow * M_PI)) / (2.0 * lb));
        corr[0] = fmaxf(0.0f, floorf(d_fast));
        corr[1] = fminf((float)(n_rot - 1), ceilf(d_slow));
    }
    const float ext_factor = 1.0f;
    const float mscale = 1.0f;
    float theta = (float)pos;
    for (int i0 = 0; i0 < n_rot; i0 += 2) {
        float ff = m.rope_freqs ? m.rope_freqs[i0 / 2] : 1.0f;
        float theta_extrap = theta / ff;
        float t = m.freq_scale * theta_extrap;
        float y = 1.0f - fminf(1.0f, fmaxf(0.0f, ((i0 / 2.0f) - corr[0]) / fmaxf(0.001f, corr[1] - corr[0])));
        float ramp = y * ext_factor;
        if (ramp != 0.0f)
            t = t * (1.0f - ramp) + theta_extrap * ramp;
        cs[i0 + 0] = cosf(t) * mscale;
        cs[i0 + 1] = sinf(t) * mscale;
        theta *= theta_scale;
    }
}

int main(int argc, char ** argv)
{
    const char * model = argc > 1 ? argv[1]
        : "/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf";

    char err[256];
    fatcat_gguf * g = fatcat_gguf_open(model, err, sizeof err);
    if (!g) { fprintf(stderr, "gguf: %s\n", err); return 1; }
    const fatcat_gguf_tensor * t = fatcat_gguf_find_tensor(g, "rope_freqs.weight");
    if (!t) { fprintf(stderr, "no rope_freqs.weight\n"); return 1; }
    memcpy(ffreqs, fatcat_gguf_tensor_ptr(g, t), sizeof ffreqs);

    const int n_rot = 64;
    const float freq_base = 10000.0f, freq_scale = 1.0f / 2048.0f;
    const int n_ctx_orig = 512;
    const float beta_fast = 32.0f, beta_slow = 1.0f, ext_factor = 1.0f;
    const float attn_factor = 1.0f / (1.0f + 0.1f * logf(2048.0f));

    fcfg fc{ n_rot, freq_base, freq_scale, 512, beta_fast, beta_slow, ffreqs };

    ggml_init_params ip = { 1 << 20, nullptr, true };  /* no_alloc: backend owns buffers */
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    fprintf(stderr, "dev=%p be=%p\n", (void*)dev, (void*)be);

    for (int pos = 1; pos <= 7; pos++) {
        ggml_context * ctx = ggml_init(ip);        /* inputs (buffered) */
        ggml_context * cgraph_ctx = ggml_init(ip); /* ops (unbuffered) */
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_rot);
        ggml_tensor * bpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * ff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_rot / 2);
        std::vector<float> av(n_rot), fv(n_rot / 2);
        for (int i = 0; i < n_rot; i++) av[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        memcpy(fv.data(), ffreqs, sizeof ffreqs);
        int32_t pv = pos;

        ggml_tensor * out = ggml_rope_ext(cgraph_ctx, a, bpos, ff, n_rot,
            GGML_ROPE_TYPE_NORMAL, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_rot);
        ggml_cgraph * gr = ggml_new_graph(cgraph_ctx);
        ggml_build_forward_expand(gr, out);
        ggml_build_forward_expand(gr, ggml_cpy(cgraph_ctx, out, res));
        fprintf(stderr, "pre-alloc\n"); ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ctx, be); fprintf(stderr, "post-alloc\n");
        ggml_backend_tensor_set(a, av.data(), 0, n_rot * 4);
        ggml_backend_tensor_set(bpos, &pv, 0, 4);
        ggml_backend_tensor_set(ff, fv.data(), 0, (n_rot / 2) * 4);
        ggml_backend_graph_compute(be, gr);
        fprintf(stderr, "pos %d computed\n", pos);

        std::vector<float> ov(n_rot);
        ggml_backend_tensor_get(res, ov.data(), 0, n_rot * 4);

        float cs[64];
        fatcat_rope(fc, pos, cs);
        double maxd = 0; int wi = 0;
        for (int i = 0; i < n_rot; i++) {
            double d = fabs((double)ov[i] - cs[i]);
            if (d > maxd) { maxd = d; wi = i; }
        }
        printf("pos %d: max|ggml-fatcat| = %.3e (dim %d: ggml=%.6f fatcat=%.6f)\n",
               pos, maxd, wi, ov[wi], cs[wi]);
        ggml_backend_buffer_free(ba);
        ggml_free(ctx);
        ggml_free(cgraph_ctx);
    }
    ggml_backend_free(be);
    fatcat_gguf_close(g);
    return 0;
}
