/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_bwd_aux_emit.c -- C-side emitter for the attention
 * backward aux kernels (prep, convert, reduce, page_zero, page_convert).
 * Selects one config by argv[1], fills the rocke_attn_bwd_aux_spec_t exactly
 * like attention_bwd_aux_emit.py, builds the kernel of that stage via
 * rocke_build_attn_bwd_aux_new for the config's arch and prints the .ll
 * (flavor AUTO), so the two outputs can be byte-compared.
 *
 * Config index table (must stay in sync with the Python emitter):
 *    0  prep         gfx950  shipped default
 *    1  prep         gfx942  shipped default
 *    2  prep         gfx950  THD
 *    3  prep         gfx950  tuned gfx950 thd entry: use_worklist
 *    4  prep         gfx942  token-major workspace
 *    5  prep         gfx950  scalar user-tensor access
 *    6  prep         gfx942  narrow_vec 2
 *    7  prep         gfx950  unified family prep (paged kv)
 *    8  prep         gfx942  narrow_vec 4, explicit rows, work list
 *    9  convert      gfx950  shipped default
 *   10  convert      gfx942  shipped default
 *   11  convert      gfx950  tuned gfx950 d64 bf16 entry: convert_rows_per_cta 4x
 *   12  convert      gfx950  tuned gfx950 d64 bf16 thd entry: convert_vec 16
 *   13  convert      gfx942  convert_vec 4, token-major
 *   14  convert      gfx950  narrow_vec 2
 *   15  convert      gfx942  scalar stores
 *   16  reduce       gfx950  dkv_sink slices, default
 *   17  reduce       gfx942  convert_vec 16
 *   18  reduce       gfx950  convert_vec 4, token-major, narrow_vec 4
 *   19  page_zero    gfx950  unified default
 *   20  page_zero    gfx942  unified default
 *   21  page_zero    gfx950  explicit rows
 *   22  page_convert gfx950  unified default
 *   23  page_convert gfx942  scalar stores
 *   24  page_convert gfx942  d32
 *   25  convert      gfx950  rejected: worklist on the convert
 *   26  page_zero    gfx950  rejected: page role without paged kv
 *   27  prep         gfx950  rejected: convert_vec on the prep
 *   28  convert      gfx950  rejected: rows that do not fill a wave
 *   29  prep         gfx942  custom kernel-name stem
 *   30  convert      gfx950  rejected: explicit rows_per_cta 0
 *   31  prep         gfx942  rejected: negative rows_per_cta
 *   32  convert      gfx950  rejected: rows over 1024 threads
 *   33  prep         gfx000  rejected: unknown arch
 *   34  prep         gfx950  batched, token-major workspace
 *   35  prep         gfx942  THD, scalar user-tensor access
 *   36  convert      gfx950  d128, convert_vec 16
 *   37  convert      gfx942  d128, convert_vec 4, THD
 *   38  reduce       gfx950  THD, convert_vec 8
 *   39  reduce       gfx942  explicit rows, d128
 *   40  page_zero    gfx950  rejected: token-major page workspace
 *   41  page_convert gfx942  rejected: token-major page workspace
 *   42  page_convert gfx950  rejected: convert_vec 16
 *   43  page_zero    gfx942  rejected: convert_vec 4
 *   44  page_convert gfx950  narrow_vec 2
 *   45  page_zero    gfx950  scalar, narrow_vec 4
 *   46  convert      gfx1151 wave32 default rows
 *   47  convert      gfx1151 wave32, rows filling one wave
 *   48  prep         gfx1151 wave32, THD work list
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_attention_bwd_aux.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the stage, spec and arch of config `idx`; 0 on success, -1 if unknown. */
static int make_cfg(int idx, const char** stage, rocke_attn_bwd_aux_spec_t* s, const char** arch)
{
    *s = rocke_attn_bwd_aux_spec_default();
    *arch = "gfx950";
    switch(idx)
    {
    case 0: /* prep, gfx950: shipped default */
        *stage = "prep";
        s->head_size = 128;
        return 0;
    case 1: /* prep, gfx942: shipped default */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 128;
        return 0;
    case 2: /* prep, gfx950: THD */
        *stage = "prep";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        return 0;
    case 3: /* prep, gfx950: tuned gfx950 thd entry: use_worklist */
        *stage = "prep";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->worklist = true;
        return 0;
    case 4: /* prep, gfx942: token-major workspace */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 32;
        s->seq_mode = "thd";
        s->ws_layout = "token_major";
        return 0;
    case 5: /* prep, gfx950: scalar user-tensor access */
        *stage = "prep";
        s->head_size = 128;
        s->dtype = "bf16";
        s->stage_vec = 1;
        return 0;
    case 6: /* prep, gfx942: narrow_vec 2 */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 64;
        s->stage_vec = 1;
        s->narrow_vec = 2;
        s->ws_layout = "token_major";
        return 0;
    case 7: /* prep, gfx950: unified family prep (paged kv) */
        *stage = "prep";
        s->head_size = 128;
        s->seq_mode = "thd";
        s->kv_target = "paged";
        return 0;
    case 8: /* prep, gfx942: narrow_vec 4, explicit rows, work list */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->stage_vec = 1;
        s->narrow_vec = 4;
        s->has_rows_per_cta = true;
        s->rows_per_cta = 16;
        s->worklist = true;
        return 0;
    case 9: /* convert, gfx950: shipped default */
        *stage = "convert";
        s->head_size = 128;
        return 0;
    case 10: /* convert, gfx942: shipped default */
        *stage = "convert";
        *arch = "gfx942";
        s->head_size = 128;
        return 0;
    case 11: /* convert, gfx950: tuned gfx950 d64 bf16 entry: convert_rows_per_cta 4x */
        *stage = "convert";
        s->head_size = 64;
        s->dtype = "bf16";
        s->has_rows_per_cta = true;
        s->rows_per_cta = 32;
        return 0;
    case 12: /* convert, gfx950: tuned gfx950 d64 bf16 thd entry: convert_vec 16 */
        *stage = "convert";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->convert_vec = 16;
        return 0;
    case 13: /* convert, gfx942: convert_vec 4, token-major */
        *stage = "convert";
        *arch = "gfx942";
        s->head_size = 32;
        s->seq_mode = "thd";
        s->convert_vec = 4;
        s->ws_layout = "token_major";
        return 0;
    case 14: /* convert, gfx950: narrow_vec 2 */
        *stage = "convert";
        s->head_size = 128;
        s->dtype = "bf16";
        s->stage_vec = 1;
        s->narrow_vec = 2;
        return 0;
    case 15: /* convert, gfx942: scalar stores */
        *stage = "convert";
        *arch = "gfx942";
        s->head_size = 64;
        s->seq_mode = "thd";
        s->stage_vec = 1;
        return 0;
    case 16: /* reduce, gfx950: dkv_sink slices, default */
        *stage = "reduce";
        s->head_size = 128;
        return 0;
    case 17: /* reduce, gfx942: convert_vec 16 */
        *stage = "reduce";
        *arch = "gfx942";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->convert_vec = 16;
        return 0;
    case 18: /* reduce, gfx950: convert_vec 4, token-major, narrow_vec 4 */
        *stage = "reduce";
        s->head_size = 32;
        s->dtype = "bf16";
        s->convert_vec = 4;
        s->ws_layout = "token_major";
        s->stage_vec = 1;
        s->narrow_vec = 4;
        return 0;
    case 19: /* page_zero, gfx950: unified default */
        *stage = "page_zero";
        s->head_size = 128;
        s->seq_mode = "thd";
        s->kv_target = "paged";
        return 0;
    case 20: /* page_zero, gfx942: unified default */
        *stage = "page_zero";
        *arch = "gfx942";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->kv_target = "paged";
        return 0;
    case 21: /* page_zero, gfx950: explicit rows */
        *stage = "page_zero";
        s->head_size = 32;
        s->kv_target = "paged";
        s->has_rows_per_cta = true;
        s->rows_per_cta = 32;
        return 0;
    case 22: /* page_convert, gfx950: unified default */
        *stage = "page_convert";
        s->head_size = 128;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->kv_target = "paged";
        return 0;
    case 23: /* page_convert, gfx942: scalar stores */
        *stage = "page_convert";
        *arch = "gfx942";
        s->head_size = 64;
        s->kv_target = "paged";
        s->stage_vec = 1;
        return 0;
    case 24: /* page_convert, gfx942: d32 */
        *stage = "page_convert";
        *arch = "gfx942";
        s->head_size = 32;
        s->dtype = "bf16";
        s->kv_target = "paged";
        return 0;
    case 25: /* convert, gfx950: rejected: worklist on the convert */
        *stage = "convert";
        s->head_size = 64;
        s->seq_mode = "thd";
        s->worklist = true;
        return 0;
    case 26: /* page_zero, gfx950: rejected: page role without paged kv */
        *stage = "page_zero";
        s->head_size = 64;
        return 0;
    case 27: /* prep, gfx950: rejected: convert_vec on the prep */
        *stage = "prep";
        s->head_size = 64;
        s->convert_vec = 16;
        return 0;
    case 28: /* convert, gfx950: rejected: rows that do not fill a wave */
        *stage = "convert";
        s->head_size = 64;
        s->has_rows_per_cta = true;
        s->rows_per_cta = 3;
        return 0;
    case 29: /* prep, gfx942: custom kernel-name stem */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 128;
        s->name = "custom_bwd";
        return 0;
    case 30: /* convert, gfx950: rejected: explicit rows_per_cta 0 */
        *stage = "convert";
        s->head_size = 64;
        s->has_rows_per_cta = true;
        s->rows_per_cta = 0;
        return 0;
    case 31: /* prep, gfx942: rejected: negative rows_per_cta */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 64;
        s->has_rows_per_cta = true;
        s->rows_per_cta = -1;
        return 0;
    case 32: /* convert, gfx950: rejected: rows over 1024 threads */
        *stage = "convert";
        s->head_size = 64;
        s->has_rows_per_cta = true;
        s->rows_per_cta = 256;
        return 0;
    case 33: /* prep, gfx000: rejected: unknown arch */
        *stage = "prep";
        *arch = "gfx000";
        s->head_size = 64;
        return 0;
    case 34: /* prep, gfx950: batched, token-major workspace */
        *stage = "prep";
        s->head_size = 128;
        s->ws_layout = "token_major";
        return 0;
    case 35: /* prep, gfx942: THD, scalar user-tensor access */
        *stage = "prep";
        *arch = "gfx942";
        s->head_size = 128;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->stage_vec = 1;
        return 0;
    case 36: /* convert, gfx950: d128, convert_vec 16 */
        *stage = "convert";
        s->head_size = 128;
        s->convert_vec = 16;
        return 0;
    case 37: /* convert, gfx942: d128, convert_vec 4, THD */
        *stage = "convert";
        *arch = "gfx942";
        s->head_size = 128;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->convert_vec = 4;
        return 0;
    case 38: /* reduce, gfx950: THD, convert_vec 8 */
        *stage = "reduce";
        s->head_size = 64;
        s->seq_mode = "thd";
        return 0;
    case 39: /* reduce, gfx942: explicit rows, d128 */
        *stage = "reduce";
        *arch = "gfx942";
        s->head_size = 128;
        s->dtype = "bf16";
        s->has_rows_per_cta = true;
        s->rows_per_cta = 16;
        return 0;
    case 40: /* page_zero, gfx950: rejected: token-major page workspace */
        *stage = "page_zero";
        s->head_size = 64;
        s->kv_target = "paged";
        s->ws_layout = "token_major";
        return 0;
    case 41: /* page_convert, gfx942: rejected: token-major page workspace */
        *stage = "page_convert";
        *arch = "gfx942";
        s->head_size = 64;
        s->kv_target = "paged";
        s->ws_layout = "token_major";
        return 0;
    case 42: /* page_convert, gfx950: rejected: convert_vec 16 */
        *stage = "page_convert";
        s->head_size = 64;
        s->kv_target = "paged";
        s->convert_vec = 16;
        return 0;
    case 43: /* page_zero, gfx942: rejected: convert_vec 4 */
        *stage = "page_zero";
        *arch = "gfx942";
        s->head_size = 64;
        s->kv_target = "paged";
        s->convert_vec = 4;
        return 0;
    case 44: /* page_convert, gfx950: narrow_vec 2 */
        *stage = "page_convert";
        s->head_size = 64;
        s->dtype = "bf16";
        s->kv_target = "paged";
        s->stage_vec = 1;
        s->narrow_vec = 2;
        return 0;
    case 45: /* page_zero, gfx950: scalar, narrow_vec 4 */
        *stage = "page_zero";
        s->head_size = 128;
        s->kv_target = "paged";
        s->stage_vec = 1;
        s->narrow_vec = 4;
        return 0;
    case 46: /* convert, gfx1151: wave32 default rows */
        *stage = "convert";
        *arch = "gfx1151";
        s->head_size = 64;
        return 0;
    case 47: /* convert, gfx1151: wave32, rows filling one wave */
        *stage = "convert";
        *arch = "gfx1151";
        s->head_size = 64;
        s->has_rows_per_cta = true;
        s->rows_per_cta = 4;
        return 0;
    case 48: /* prep, gfx1151: wave32, THD work list */
        *stage = "prep";
        *arch = "gfx1151";
        s->head_size = 64;
        s->dtype = "bf16";
        s->seq_mode = "thd";
        s->worklist = true;
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    const char* stage = NULL;
    rocke_attn_bwd_aux_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &stage, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_attn_bwd_aux_new(&b, stage, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* d = rocke_diag_to_string(&diags[i]);
            if(d)
            {
                fputs(d, stdout);
                fputc('\n', stdout);
                free(d);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
