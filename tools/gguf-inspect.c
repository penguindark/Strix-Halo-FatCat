/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* gguf-inspect — dump GGUF header, metadata and tensor directory.
 *
 * Usage: gguf-inspect <file.gguf> [--kv] [--tensors] [--all]
 * Works on single and split GGUFs (pass the -00001-of-NNNNN part). */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/gguf.h"

#define MAX_TYPES 64
typedef struct { uint32_t type; uint64_t count; uint64_t bytes; } type_hist;

static void print_str(FILE *f, fatcat_str s)
{
    fwrite(s.data, 1, s.len, f);
}

static void print_value(FILE *f, const fatcat_gguf_value *v, int indent)
{
    char pad[16];
    snprintf(pad, sizeof pad, "%*s", indent, "");
    switch (v->type) {
    case FATCAT_GGUF_U8:  printf("%" PRIu8, v->u8); break;
    case FATCAT_GGUF_I8:  printf("%" PRId8, v->i8); break;
    case FATCAT_GGUF_U16: printf("%" PRIu16, v->u16); break;
    case FATCAT_GGUF_I16: printf("%" PRId16, v->i16); break;
    case FATCAT_GGUF_U32: printf("%" PRIu32, v->u32); break;
    case FATCAT_GGUF_I32: printf("%" PRId32, v->i32); break;
    case FATCAT_GGUF_F32: printf("%g", (double)v->f32); break;
    case FATCAT_GGUF_BOOL: printf("%s", v->boolean ? "true" : "false"); break;
    case FATCAT_GGUF_U64: printf("%" PRIu64, v->u64); break;
    case FATCAT_GGUF_I64: printf("%" PRId64, v->i64); break;
    case FATCAT_GGUF_F64: printf("%g", v->f64); break;
    case FATCAT_GGUF_STR:
        printf("\""); print_str(f, v->str); printf("\"");
        break;
    case FATCAT_GGUF_ARR:
        printf("[%" PRIu64 " ", v->arr.count);
        if (v->arr.elem_type == FATCAT_GGUF_STR) {
            const fatcat_str *strs = v->arr.elems;
            uint64_t show = v->arr.count < 8 ? v->arr.count : 8;
            printf("str]{");
            for (uint64_t i = 0; i < show; i++)
                printf("%s\"%.*s\"", i ? ", " : "", (int)strs[i].len, strs[i].data);
            if (v->arr.count > show) printf(", ...");
            printf("}");
        } else {
            static const size_t esz[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            size_t sz = esz[v->arr.elem_type];
            uint64_t show = v->arr.count < 128 ? v->arr.count : 128;
            printf("] len>%" PRIu64 " (first:", v->arr.count);
            for (uint64_t i = 0; i < show; i++) {
                const uint8_t *p = (const uint8_t *)v->arr.elems + i * sz;
                printf("%s", i ? ", " : " ");
                switch (v->arr.elem_type) {
                case FATCAT_GGUF_U32: { uint32_t x; memcpy(&x,p,4); printf("%u",x); } break;
                case FATCAT_GGUF_I32: { int32_t x; memcpy(&x,p,4); printf("%d",x); } break;
                case FATCAT_GGUF_F32: { float x; memcpy(&x,p,4); printf("%g",(double)x); } break;
                case FATCAT_GGUF_BOOL: { uint8_t x; memcpy(&x,p,1); printf("%u",x); } break;
                case FATCAT_GGUF_I64: { int64_t x; memcpy(&x,p,8); printf("%" PRId64,x); } break;
                case FATCAT_GGUF_U64: { uint64_t x; memcpy(&x,p,8); printf("%" PRIu64,x); } break;
                default: printf("?"); break;
                }
            }
            if (v->arr.count > show) printf(", ...");
            printf(")");
        }
        break;
    }
    (void)pad;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.gguf> [--kv] [--tensors] [--all]\n", argv[0]);
        return 2;
    }
    bool dump_kv = false, dump_tensors = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--kv") || !strcmp(argv[i], "--all")) dump_kv = true;
        if (!strcmp(argv[i], "--tensors") || !strcmp(argv[i], "--all")) dump_tensors = true;
    }

    char err[256];
    fatcat_gguf *g = fatcat_gguf_open(argv[1], err, sizeof err);
    if (!g) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }

    printf("file:        %s\n", argv[1]);
    printf("version:     %u\n", fatcat_gguf_version(g));
    printf("parts:       %zu\n", fatcat_gguf_num_parts(g));
    printf("alignment:   %u\n", fatcat_gguf_alignment(g));
    printf("data start:  %" PRIu64 "\n", fatcat_gguf_data_start(g));
    printf("kv count:    %zu\n", fatcat_gguf_num_kv(g));
    printf("tensors:     %zu\n", fatcat_gguf_num_tensors(g));

    /* Highlights + type histogram + expert analysis. */
    static const char *hl[] = {
        "general.name", "general.architecture", "general.type",
        "general.quantization_version", "tokenizer.ggml.model",
    };
    for (size_t i = 0; i < sizeof hl / sizeof *hl; i++) {
        const fatcat_gguf_value *v = fatcat_gguf_find_kv(g, hl[i]);
        if (v) { printf("%-22s ", hl[i]); print_value(stdout, v, 0); printf("\n"); }
    }

    type_hist hist[MAX_TYPES] = {0};
    int n_hist = 0;
    uint64_t total_bytes = 0, expert_bytes = 0;
    uint64_t expert_tensors = 0;
    uint64_t max_layer = 0;
    for (size_t i = 0; i < fatcat_gguf_num_tensors(g); i++) {
        const fatcat_gguf_tensor *t = fatcat_gguf_tensor_at(g, i);
        total_bytes += t->n_bytes;
        int h;
        for (h = 0; h < n_hist; h++)
            if (hist[h].type == t->type) break;
        if (h == n_hist) { hist[h].type = t->type; n_hist++; }
        hist[h].count++;
        hist[h].bytes += t->n_bytes;
        /* MoE routed-expert tensors contain ".ffn_" and "_exps." */
        if (t->name.len > 11 && strstr(t->name.data, "_exps.")) {
            expert_bytes += t->n_bytes;
            expert_tensors++;
            /* parse blk.<n>. */
            unsigned layer;
            if (sscanf(t->name.data, "blk.%u.", &layer) == 1 && layer > max_layer)
                max_layer = layer;
        }
    }
    printf("tensor data: %.2f GB\n", (double)total_bytes / 1e9);
    printf("types:\n");
    for (int i = 0; i < n_hist; i++)
        printf("  %-8s %6" PRIu64 " tensors  %8.2f GB\n",
               fatcat_ggml_type_name(hist[i].type) ? fatcat_ggml_type_name(hist[i].type) : "?",
               hist[i].count, (double)hist[i].bytes / 1e9);
    if (expert_tensors) {
        printf("routed experts: %" PRIu64 " tensors, %.2f GB, max blk.%" PRIu64
               " (≈ %.2f GB/layer)\n",
               expert_tensors, (double)expert_bytes / 1e9, max_layer,
               (double)expert_bytes / (double)(max_layer + 1) / 1e9);
    }

    if (dump_kv) {
        printf("\n--- KV metadata ---\n");
        for (size_t i = 0; i < fatcat_gguf_num_kv(g); i++) {
            const fatcat_gguf_kv *kv = fatcat_gguf_kv_at(g, i);
            printf("%.*s = ", (int)kv->key.len, kv->key.data);
            print_value(stdout, &kv->value, 0);
            printf("\n");
        }
    }
    if (dump_tensors) {
        printf("\n--- tensors ---\n");
        printf("%-52s %-8s %-24s %10s %12s\n",
               "name", "type", "dims", "offset", "bytes");
        for (size_t i = 0; i < fatcat_gguf_num_tensors(g); i++) {
            const fatcat_gguf_tensor *t = fatcat_gguf_tensor_at(g, i);
            char dims[64] = {0};
            int off = 0;
            for (uint32_t d = 0; d < t->n_dims; d++)
                off += snprintf(dims + off, sizeof dims - off, "%s%" PRIu64,
                                d ? "x" : "", t->dims[d]);
            printf("%-52.*s %-8s %-24s %12" PRIu64 " %12" PRIu64 "\n",
                   (int)t->name.len, t->name.data,
                   fatcat_ggml_type_name(t->type) ? fatcat_ggml_type_name(t->type) : "?",
                   dims, t->offset, t->n_bytes);
        }
    }

    fatcat_gguf_close(g);
    return 0;
}
