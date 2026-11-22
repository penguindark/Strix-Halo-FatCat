/* fatcat GPU stub: used when the project is built without HIP/ROCm.
 * Mirrors the impl symbols of src/gpu/kernels.hip and always fails, so
 * fatcat_gpu_init() returns false and callers use the CPU path. */
#include <stdio.h>
#include <stdint.h>

int fatcat_gpu_init_impl(uint32_t slots, char *eb, size_t el)
{
    (void)slots;
    snprintf(eb, el, "built without HIP support");
    return -1;
}
void fatcat_gpu_shutdown_impl(void) {}
void fatcat_gpu_stats_impl(uint64_t *c, uint64_t *b, uint64_t *h)
{
    *c = *b = *h = 0;
}
uint64_t fatcat_gpu_upload_fails_impl(void) { return 0; }
const char *fatcat_gpu_devstr_impl(void) { return "none"; }
void *fatcat_gpu_upload_impl(uint32_t type, const void *host, size_t bytes)
{
    (void)type; (void)host; (void)bytes; return NULL;
}
void *fatcat_gpu_stage_impl(uint32_t type, const void *host, size_t bytes, uint64_t tag)
{
    (void)type; (void)host; (void)bytes; (void)tag; return NULL;
}
void fatcat_gpu_free_handle(void *handle) { (void)handle; }
int fatcat_gpu_gemv_impl(void *wh, uint32_t type, const float *x, float *y,
                         long long n_out, long long n_in)
{
    (void)wh; (void)type; (void)x; (void)y; (void)n_out; (void)n_in;
    return -1;
}
int fatcat_gpu_gemm_impl(void *wh, uint32_t type, const float *X, float *Y,
                         long long n_out, long long n_in, int n_tok)
{
    (void)wh; (void)type; (void)X; (void)Y; (void)n_out; (void)n_in; (void)n_tok;
    return -1;
}
void fatcat_gpu_gemm_stats_impl(uint64_t *calls, uint64_t *fails)
{
    *calls = 0; *fails = 0;
}
void *fatcat_gpu_host_alloc_impl(size_t bytes) { (void)bytes; return NULL; }
void fatcat_gpu_host_free_impl(void *p) { (void)p; }
int fatcat_gpu_unified_impl(void) { return 0; }
void *fatcat_gpu_wrap_impl(void *dev_ptr, uint32_t type)
{
    (void)dev_ptr; (void)type; return NULL;
}
int fatcat_gpu_gemv_dev_impl(void *wh, uint32_t type, const float *x, float *y,
                             long long n_out, long long n_in)
{
    (void)wh; (void)type; (void)x; (void)y; (void)n_out; (void)n_in;
    return -1;
}
int fatcat_gpu_wait_impl(void) { return -1; }
int fatcat_gpu_gemv_q8_0_batched_impl(void *wh, long long w_stride,
                                      const float *x, long long x_stride,
                                      float *y, long long y_stride,
                                      long long n_out, long long n_in,
                                      long long n_items)
{
    (void)wh; (void)w_stride; (void)x; (void)x_stride; (void)y;
    (void)y_stride; (void)n_out; (void)n_in; (void)n_items;
    return -1;
}
int fatcat_gpu_absorb_q8_0_impl(void *wh, const float *x, float *y,
                                long long rows_per_item, long long n_items,
                                long long n_in, long long x_stride,
                                long long row_stride)
{
    (void)wh; (void)x; (void)y; (void)rows_per_item; (void)n_items;
    (void)n_in; (void)x_stride; (void)row_stride;
    return -1;
}
void *fatcat_gpu_xstage_impl(const void *x, size_t bytes)
{
    (void)x; (void)bytes; return NULL;
}
double fatcat_gpu_readbw_impl(size_t bytes, int iters)
{
    (void)bytes; (void)iters; return -1.0;
}
