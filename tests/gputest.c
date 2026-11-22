/* gputest — compare GPU gemv kernels vs CPU scalar path on real tensors. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fatcat/gguf.h"
#include "fatcat/gpu.h"
#include "fatcat/dequant.h"

static float *x, *ga, *gb;

static double cmp(const float *a, const float *b, int64_t n)
{
    /* norm-relative error: ||a-b|| / ||b|| (robust to near-zero elements) */
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        num += d * d; den += (double)b[i] * b[i];
    }
    return den > 0 ? sqrt(num / den) : (num > 0 ? 1e9 : 0);
}

static void cpu_gemv(uint32_t type, const void *w, const float *x, float *y,
                     int64_t n_out, int64_t n_in)
{
    int64_t rb = fatcat_dequant_src_bytes(type, n_in);
    float *tmp = malloc(n_in * 4);
    for (int64_t i = 0; i < n_out; i++) {
        fatcat_dequant(type, (const char *)w + i * rb, tmp, n_in);
        double s = 0;
        for (int64_t j = 0; j < n_in; j++) s += (double)tmp[j] * x[j];
        y[i] = (float)s;
    }
    free(tmp);
}

static void test(fatcat_gguf *g, const char *name, int64_t n_out, int64_t n_in, int expert, int64_t stride)
{
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, name);
    if (!t) { printf("%s: missing\n", name); return; }
    int64_t n_in2 = t->n_dims > 0 ? (int64_t)t->dims[0] : n_in;
    int64_t n_out2 = t->n_dims > 1 ? (int64_t)t->dims[1] : n_out;
    if (n_in2 != n_in || n_out2 != n_out)
        printf("%s: using real dims %lldx%lld (asked %lldx%lld)\n", name,
               (long long)n_out2, (long long)n_in2, (long long)n_out, (long long)n_in);
    n_in = n_in2; n_out = n_out2;
    const char *p = fatcat_gguf_tensor_ptr(g, t) + (int64_t)expert * stride;
    uint32_t type = t->type;
    for (int64_t i = 0; i < n_in; i++) x[i] = (float)((i * 37 % 101) - 50) / 50.0f;
    cpu_gemv(type, p, x, gb, n_out, n_in);
    size_t bytes = (size_t)stride;
    if (expert == 0) bytes = (size_t)fatcat_dequant_src_bytes(type, n_in * n_out);
    void *h = bytes > (9u << 20) ? fatcat_gpu_upload(type, p, bytes)
                                 : fatcat_gpu_stage(type, p, bytes, 1234 + type);
    if (!h) { printf("%s: stage/upload failed\n", name); return; }
    if (!fatcat_gpu_gemv(h, type, x, ga, n_out, n_in)) { printf("%s: gemv failed\n", name); return; }
    fatcat_gpu_release(h);
    double w = cmp(ga, gb, n_out);
    printf("%-36s type=%2u  relerr(norm)=%.3g  %s\n", name, type, w,
           (isnan(ga[0]) || w > 1e-3) ? "FAIL" : "ok");
}

static void test_gemm(fatcat_gguf *g, const char *name, int nt)
{
    /* D4: the batched gemm kernels must be BIT-identical to the gemv
     * kernels they replace (same thread mapping + accumulation order). */
    const fatcat_gguf_tensor *t = fatcat_gguf_find_tensor(g, name);
    if (!t) { printf("%s: missing\n", name); return; }
    int64_t n_in = (int64_t)t->dims[0], n_out = (int64_t)t->dims[1];
    uint32_t type = t->type;
    const char *p = fatcat_gguf_tensor_ptr(g, t);
    size_t bytes = (size_t)fatcat_dequant_src_bytes(type, n_in * n_out);
    void *h = fatcat_gpu_upload(type, p, bytes);
    if (!h) { printf("%s: upload failed\n", name); return; }
    float *X = malloc((size_t)nt * n_in * 4);
    float *yv = malloc((size_t)nt * n_out * 4);
    float *yg = malloc((size_t)nt * n_out * 4);
    for (int64_t i = 0; i < (int64_t)nt * n_in; i++)
        X[i] = (float)(((i + 7) * 37 % 101) - 50) / 50.0f;
    int ok = 1;
    for (int t0 = 0; t0 < nt; t0 += 8) {
        int n = nt - t0 < 8 ? nt - t0 : 8;
        for (int t = 0; t < n; t++)
            if (!fatcat_gpu_gemv(h, type, X + (int64_t)(t0 + t) * n_in,
                                 yv + (int64_t)t * n_out, n_out, n_in)) {
            printf("%s: gemv failed\n", name); ok = 0; break;
        }
        if (fatcat_gpu_gemm(h, type, X + (int64_t)t0 * n_in, yg, n_out, n_in, n)) {
            printf("%s: gemm failed\n", name); ok = 0; break;
        }
        if (memcmp(yv, yg, (size_t)n * n_out * 4) != 0) {
            double e = cmp(yg, yv, (int64_t)n * n_out);
            printf("%-36s nt=%d  GEMM!=GEMV  relerr=%.3g\n", name, n, e);
            ok = 0; break;
        }
    }
    if (ok) printf("%-36s nt=%d  gemm==gemv BIT-IDENTICAL  ok\n", name, nt);
    fatcat_gpu_release(h);
    free(X); free(yv); free(yg);
}

int main(int argc, char **argv)
{
    char err[256];
    if (!fatcat_gpu_init(16, err, sizeof(err))) { printf("gpu: %s\n", err); return 1; }
    fatcat_gguf *g = fatcat_gguf_open(argv[1], err, sizeof(err));
    if (!g) { printf("open: %s\n", err); return 1; }
    x = malloc(24576 * 4); ga = malloc(24576 * 4); gb = malloc(24576 * 4);
    int64_t s_iq1s = fatcat_dequant_src_bytes(19, 6144LL * 6144);
    int64_t s_iq2 = fatcat_dequant_src_bytes(16, 6144LL * 6144);
    int64_t s_q8 = fatcat_dequant_src_bytes(8, 6144LL * 6144);
    int64_t s_q6 = fatcat_dequant_src_bytes(14, 6144LL * 6144);
    test(g, "blk.1.ffn_gate_exps.weight", 6144, 6144, 3, s_iq2);   /* IQ2_XXS */
    test(g, "blk.10.ffn_gate_exps.weight", 6144, 6144, 5, s_iq1s); /* IQ1_S  */
    test(g, "blk.10.ffn_down_exps.weight", 6144, 6144, 5, s_iq2);  /* IQ2_XXS */
    test(g, "blk.1.ffn_down_shexp.weight", 6144, 6144, 0, s_q8);   /* Q8_0   */
    test(g, "blk.1.attn_q.weight", 8192, 6144, 0, s_q6);           /* Q6_K   */
    (void)s_q8; (void)s_q6;
    if (argc > 2) {   /* D4: batched dense gemm bit-identity vs gemv */
        test_gemm(g, "blk.1.attn_q.weight", 9);          /* Q6_K */
        test_gemm(g, "blk.1.ffn_down_shexp.weight", 9);  /* Q8_0 */
    }
    fatcat_gguf_close(g);
    fatcat_gpu_shutdown();
    return 0;
}
