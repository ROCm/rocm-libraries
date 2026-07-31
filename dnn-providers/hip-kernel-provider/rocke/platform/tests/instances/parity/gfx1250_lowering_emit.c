/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1250_lowering_emit.c -- C-side emitter for the gfx1250
 * lowering surface. Builds each kernel identically to gfx1250_lowering_emit.py
 * so run_diff.py can byte-compare the two engines' .ll; see that file for the
 * config-by-config rationale and for why the gfx950 twins are included.
 *
 * arch is per-config (see CONFIGS), flavor = AUTO (matches the Python side).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Every builder call below is its own statement, and the statement order
 * matches the Python emitter exactly. Nesting a builder call inside an argument
 * list would leave the order to the C compiler (argument evaluation order is
 * unspecified), and since each call consumes a value id, a different order
 * emits the same IR under different SSA names -- a byte mismatch with no real
 * defect behind it. */

static rocke_value_t*
    frag_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    o.align = 16;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* Mirrors the Python _frag_operands helper: A and B as <n x elem>, C as
 * <8 x float>, all loaded so the operands stay opaque to the lowerer. */
typedef struct frag_ops
{
    rocke_value_t* tid;
    rocke_value_t* c_ptr;
    rocke_value_t* a;
    rocke_value_t* b;
    rocke_value_t* c;
} frag_ops_t;

static frag_ops_t frag_operands(rocke_ir_builder_t* b, const rocke_type_t* elem, int n)
{
    frag_ops_t f;
    rocke_value_t* a_ptr = frag_param(b, "A", elem, true);
    rocke_value_t* b_ptr = frag_param(b, "B", elem, true);
    f.c_ptr = frag_param(b, "C", rocke_f32(), false);
    f.tid = rocke_b_thread_id_x(b);
    f.a = rocke_b_global_load_vN(b, a_ptr, f.tid, elem, n, /*align=*/0);
    f.b = rocke_b_global_load_vN(b, b_ptr, f.tid, elem, n, /*align=*/0);
    f.c = rocke_b_global_load_vN(b, f.c_ptr, f.tid, rocke_f32(), 8, /*align=*/0);
    return f;
}

/* K=32 f16/bf16 WMMA: the gfx1250 8-operand signature. bf16 is the interesting
 * half -- gfx11/gfx12 bitcast the operands to <16 x i16> before the call, while
 * gfx1250 takes <16 x bfloat> directly. */
static void wmma_k32(rocke_ir_builder_t* b, const rocke_type_t* elem, const char* op_id)
{
    frag_ops_t f = frag_operands(b, elem, 16);
    rocke_value_t* d = rocke_b_mma(b, op_id, f.a, f.b, f.c, NULL, 0);
    rocke_b_global_store(b, f.c_ptr, f.tid, d, /*align=*/1);
    rocke_b_ret(b);
}

static void build_wmma_k32_f16(rocke_ir_builder_t* b)
{
    wmma_k32(b, rocke_f16(), "wmma_gfx1250_f32_16x16x32_f16");
}

static void build_wmma_k32_bf16(rocke_ir_builder_t* b)
{
    wmma_k32(b, rocke_bf16(), "wmma_gfx1250_f32_16x16x32_bf16");
}

/* K=64 fp8/bf8 WMMA: the gfx1250 6-operand signature. The fragments arrive as
 * <8 x i32> (32 packed bytes per lane), so the dtype pair lives only in the
 * op_id and the mangled intrinsic name. */
static void wmma_k64(rocke_ir_builder_t* b, const char* op_id)
{
    frag_ops_t f = frag_operands(b, rocke_i32(), 8);
    rocke_value_t* d = rocke_b_mma(b, op_id, f.a, f.b, f.c, NULL, 0);
    rocke_b_global_store(b, f.c_ptr, f.tid, d, /*align=*/1);
    rocke_b_ret(b);
}

static void build_wmma_k64_fp8_fp8(rocke_ir_builder_t* b)
{
    wmma_k64(b, "wmma_gfx1250_f32_16x16x64_fp8_fp8");
}

static void build_wmma_k64_fp8_bf8(rocke_ir_builder_t* b)
{
    wmma_k64(b, "wmma_gfx1250_f32_16x16x64_fp8_bf8");
}

static void build_wmma_k64_bf8_fp8(rocke_ir_builder_t* b)
{
    wmma_k64(b, "wmma_gfx1250_f32_16x16x64_bf8_fp8");
}

static void build_wmma_k64_bf8_bf8(rocke_ir_builder_t* b)
{
    wmma_k64(b, "wmma_gfx1250_f32_16x16x64_bf8_bf8");
}

/* ds_read_b128_tr_b16. gfx950 has one type-agnostic opcode returning
 * <8 x i16> that the handler reinterprets; gfx1250 has per-element-type
 * opcodes (.v8f16 / .v8bf16) that land in the right type with no reinterpret. */
static void tr16_b128(rocke_ir_builder_t* b, const rocke_type_t* elem)
{
    rocke_param_opts_t o;
    const int shape[] = {64, 8};
    rocke_value_t* out;
    rocke_value_t* smem;
    rocke_value_t* tid;
    rocke_value_t* zero;
    rocke_value_t* v;
    rocke_value_t* idx[2];

    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.align = 16;
    o.align_set = true;
    out = rocke_b_param(b, "out", rocke_ptr_type(b, elem, "global"), &o);
    smem = rocke_b_smem_alloc(b, elem, shape, 2, "tile");
    tid = rocke_b_thread_id_x(b);
    zero = rocke_b_const_i32(b, 0);
    idx[0] = tid;
    idx[1] = zero;
    v = rocke_b_ds_read_tr16_b128(b, smem, idx, 2, elem);
    rocke_b_global_store(b, out, tid, v, /*align=*/1);
    rocke_b_ret(b);
}

static void build_tr16_f16(rocke_ir_builder_t* b)
{
    tr16_b128(b, rocke_f16());
}

static void build_tr16_bf16(rocke_ir_builder_t* b)
{
    tr16_b128(b, rocke_bf16());
}

/* The two LDS workgroup barriers. An s_barrier does not drain outstanding LDS
 * traffic, so each barrier is preceded by a wait: gfx9/10/11 spend one
 * monolithic s_waitcnt, gfx1250 emits split s_wait_loadcnt / s_wait_dscnt (and
 * for the LDS-only barrier, dscnt alone). */
static void build_barrier_drains(rocke_ir_builder_t* b)
{
    rocke_param_opts_t o;
    const int shape[] = {64, 8};
    rocke_value_t* out;
    rocke_value_t* smem;
    rocke_value_t* tid;
    rocke_value_t* zero;
    rocke_value_t* v;
    rocke_value_t* idx[2];

    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.align = 16;
    o.align_set = true;
    out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_f16(), "global"), &o);
    smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "tile");
    tid = rocke_b_thread_id_x(b);
    zero = rocke_b_const_i32(b, 0);
    rocke_b_sync(b);
    idx[0] = tid;
    idx[1] = zero;
    v = rocke_b_smem_load_vN_f16(b, smem, idx, 2, 8);
    rocke_b_sync_lds_only(b);
    rocke_b_global_store(b, out, tid, v, /*align=*/1);
    rocke_b_ret(b);
}

/* The two wait-counter facts: s_wait_asynccnt lowers to nothing without an
 * async-DMA counter and to the intrinsic on gfx1250; s_waitcnt is the mirror
 * image, since llvm.amdgcn.s.waitcnt is not selectable on gfx1250. */
static void build_wait_counters(rocke_ir_builder_t* b)
{
    rocke_b_s_wait_asynccnt(b, 0);
    rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
    rocke_b_s_wait_asynccnt(b, 3);
    rocke_b_s_waitcnt(b, /*vmcnt=*/-1, /*lgkmcnt=*/0, /*expcnt=*/-1);
    rocke_b_ret(b);
}

typedef void (*build_fn_t)(rocke_ir_builder_t*);

typedef struct config
{
    build_fn_t build;
    const char* arch;
} config_t;

/* Each gfx1250 config that tests a *choice* of encoding is followed by its
 * gfx950 twin, so the pair pins both branches. Index-for-index with the Python
 * emitter's CONFIGS list. */
static const config_t CONFIGS[] = {
    {build_wmma_k32_f16, "gfx1250"},
    {build_wmma_k32_bf16, "gfx1250"},
    {build_wmma_k64_fp8_fp8, "gfx1250"},
    {build_wmma_k64_fp8_bf8, "gfx1250"},
    {build_wmma_k64_bf8_fp8, "gfx1250"},
    {build_wmma_k64_bf8_bf8, "gfx1250"},
    {build_tr16_f16, "gfx1250"},
    {build_tr16_f16, "gfx950"},
    {build_tr16_bf16, "gfx1250"},
    {build_tr16_bf16, "gfx950"},
    {build_barrier_drains, "gfx1250"},
    {build_barrier_drains, "gfx950"},
    {build_wait_counters, "gfx1250"},
    {build_wait_counters, "gfx950"},
};

static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_CONFIGS - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "gfx1250_lowering") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    CONFIGS[idx].build(&b);

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, CONFIGS[idx].arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }

    rocke_ir_builder_free(&b);
    return 0;
}
