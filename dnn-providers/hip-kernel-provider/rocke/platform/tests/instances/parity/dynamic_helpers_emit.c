/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/dynamic_helpers_emit.c -- C-side emitter for the runtime-shape
 * (AOT) helpers: rocke_software_pipeline_run_ping_pong_dynamic and the dynamic
 * coordinate transforms (embed_dynamic, unmerge_magic_dynamic, pad_dynamic,
 * the dynamic tensor descriptor). Builds each kernel identically to
 * dynamic_helpers_emit.py so run_diff.py can byte-compare the two engines'
 * .ll; see that file for the config-by-config rationale.
 *
 * arch = gfx950, flavor = AUTO (matches the Python side).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.pipeline.h"
#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/helper_rocke.helpers.transforms.h"
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

#define BLOCK_K 32
#define LDS_ELEMS 256
#define BUFFER_BYTES (1 << 20)
/* Byte offset the buffer resource drops (reads zero) for an invalid coord. */
#define OOB_OFFSET 0x7FFF0000

static rocke_value_t*
    ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, bool readonly)
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

/* ------------------------------------------------------------------------- */
/* run_ping_pong_dynamic                                                     */
/* ------------------------------------------------------------------------- */

typedef struct pp_ctx
{
    rocke_value_t* tid;
    rocke_value_t* rsrc;
    rocke_value_t* soff;
    rocke_value_t* two;
} pp_ctx_t;

/* Python issue(k, buf): stage X[k + tid] into both halves of the pair. */
static void
    pp_issue(rocke_ir_builder_t* b, rocke_value_t* k, const rocke_buffer_pair_t* buf, void* user)
{
    pp_ctx_t* c = (pp_ctx_t*)user;
    rocke_value_t* idx[1] = {c->tid};
    rocke_value_t* elem = rocke_b_add(b, k, c->tid);
    rocke_value_t* voff = rocke_b_mul(b, elem, c->two);
    rocke_value_t* v = rocke_b_buffer_load_f16(b, c->rsrc, voff, c->soff);
    rocke_b_smem_store_f16(b, buf->a, idx, 1, v);
    rocke_b_smem_store_f16(b, buf->b, idx, 1, v);
}

/* Python compute(k, buf, state): acc += f32(a[tid]); cnt += k. */
static void pp_compute(rocke_ir_builder_t* b,
                       rocke_value_t* k,
                       const rocke_buffer_pair_t* buf,
                       rocke_value_t* const* state_in,
                       int num_state,
                       rocke_value_t** state_out,
                       void* user)
{
    pp_ctx_t* c = (pp_ctx_t*)user;
    rocke_value_t* idx[1] = {c->tid};
    (void)num_state;
    rocke_value_t* h = rocke_b_smem_load_vN_f16(b, buf->a, idx, 1, 1);
    rocke_value_t* e = rocke_b_vec_extract(b, h, 0);
    rocke_value_t* f = rocke_b_cast_to_f32(b, e);
    rocke_value_t* acc2 = rocke_b_fadd(b, state_in[0], f);
    rocke_value_t* cnt2 = rocke_b_add(b, state_in[1], k);
    state_out[0] = acc2;
    state_out[1] = cnt2;
}

static void pingpong(rocke_ir_builder_t* b,
                     bool with_lo,
                     bool with_zero_fill,
                     bool wait_vmcnt,
                     bool overlap_vmcnt,
                     const char* schedule_name,
                     bool mask)
{
    const int shape[1] = {LDS_ELEMS};
    rocke_schedule_policy_t sched;
    const rocke_schedule_policy_t* sched_p = NULL;
    pp_ctx_t ctx;

    if(schedule_name != NULL)
    {
        sched = rocke_schedule_policy_for_pipeline(b, schedule_name);
        sched_p = &sched;
    }

    rocke_value_t* x = ptr_param(b, "X", rocke_f16(), true);
    rocke_value_t* y = ptr_param(b, "Y", rocke_f32(), false);
    rocke_value_t* c_out = ptr_param(b, "C", rocke_i32(), false);
    rocke_value_t* k_extent = rocke_b_param(b, "K", rocke_i32(), NULL);
    rocke_value_t* k_lo = with_lo ? rocke_b_param(b, "k_lo", rocke_i32(), NULL) : NULL;
    rocke_value_t* k_zero = with_zero_fill ? rocke_b_param(b, "k_zero", rocke_i32(), NULL) : NULL;
    ctx.tid = rocke_b_thread_id_x(b);
    rocke_value_t* nbytes = rocke_b_const_i32(b, BUFFER_BYTES);
    ctx.rsrc = rocke_b_buffer_rsrc(b, x, nbytes);
    ctx.soff = rocke_b_const_i32(b, 0);
    ctx.two = rocke_b_const_i32(b, 2);
    rocke_buffer_pair_t buffers[2];
    buffers[0].a = rocke_b_smem_alloc(b, rocke_f16(), shape, 1, "a0");
    buffers[0].b = rocke_b_smem_alloc(b, rocke_f16(), shape, 1, "b0");
    buffers[1].a = rocke_b_smem_alloc(b, rocke_f16(), shape, 1, "a1");
    buffers[1].b = rocke_b_smem_alloc(b, rocke_f16(), shape, 1, "b1");

    rocke_value_t* acc0 = rocke_b_const_f32(b, 0.0);
    rocke_value_t* cnt0 = rocke_b_const_i32(b, 0);
    rocke_iter_arg_t iter_args[2] = {{"acc", acc0}, {"cnt", cnt0}};

    /* Python SoftwarePipeline(num_iters=0, wait_vmcnt=..., overlap_vmcnt=...)
     * with sync_after_wait / sync_before_issue left at their True defaults. */
    rocke_software_pipeline_t pipe;
    pipe.wait_vmcnt = wait_vmcnt;
    pipe.sync_after_wait = true;
    pipe.sync_before_issue = true;
    pipe.overlap_vmcnt = overlap_vmcnt;
    pipe.mask_tail_state = mask;

    rocke_value_t* results[2] = {NULL, NULL};
    if(!rocke_software_pipeline_run_ping_pong_dynamic(&pipe,
                                                      b,
                                                      k_extent,
                                                      BLOCK_K,
                                                      k_lo,
                                                      k_zero,
                                                      buffers,
                                                      iter_args,
                                                      2,
                                                      pp_issue,
                                                      pp_compute,
                                                      &ctx,
                                                      sched_p,
                                                      results))
    {
        return;
    }
    rocke_b_global_store(b, y, ctx.tid, results[0], 1);
    rocke_b_global_store(b, c_out, ctx.tid, results[1], 1);
    rocke_b_ret(b);
}

static void build_pingpong_default(rocke_ir_builder_t* b)
{
    pingpong(b, false, false, false, false, NULL, false);
}

static void build_pingpong_split_k(rocke_ir_builder_t* b)
{
    pingpong(b, true, true, true, true, NULL, false);
}

static void build_pingpong_mask(rocke_ir_builder_t* b)
{
    pingpong(b, true, false, true, false, NULL, true);
}

static void build_pingpong_mask_split_k(rocke_ir_builder_t* b)
{
    pingpong(b, true, true, true, true, NULL, true);
}

static void build_pingpong_schedule(rocke_ir_builder_t* b)
{
    pingpong(b, true, false, true, false, "interwave", false);
}

/* ------------------------------------------------------------------------- */
/* Dynamic descriptors                                                       */
/* ------------------------------------------------------------------------- */

/* Python _store_through_valid: load X[off] (zero when !valid), store to Y[tid]. */
static void store_through_valid(rocke_ir_builder_t* b,
                                rocke_value_t* x,
                                rocke_value_t* y,
                                rocke_value_t* tid,
                                rocke_value_t* off,
                                rocke_value_t* valid)
{
    rocke_value_t* nbytes = rocke_b_const_i32(b, BUFFER_BYTES);
    rocke_value_t* rsrc_x = rocke_b_buffer_rsrc(b, x, nbytes);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* off_b = rocke_b_mul(b, off, two);
    rocke_value_t* oob = rocke_b_const_i32(b, OOB_OFFSET);
    rocke_value_t* voff = rocke_b_select(b, valid, off_b, oob);
    rocke_value_t* soff = rocke_b_const_i32(b, 0);
    rocke_value_t* v = rocke_b_buffer_load_f16(b, rsrc_x, voff, soff);
    rocke_value_t* rsrc_y = rocke_b_buffer_rsrc(b, y, nbytes);
    rocke_value_t* tid_b = rocke_b_mul(b, tid, two);
    rocke_b_buffer_store_f16(b, rsrc_y, tid_b, soff, v);
}

static void build_unmerge_embed(rocke_ir_builder_t* b)
{
    rocke_value_t* x = ptr_param(b, "X", rocke_f16(), true);
    rocke_value_t* y = ptr_param(b, "Y", rocke_f16(), false);
    rocke_value_t* o = ptr_param(b, "O", rocke_i32(), false);
    rocke_value_t* p_h = rocke_b_param(b, "p_H", rocke_i32(), NULL);
    rocke_value_t* p_w = rocke_b_param(b, "p_W", rocke_i32(), NULL);
    rocke_value_t* p_sh = rocke_b_param(b, "p_sH", rocke_i32(), NULL);
    rocke_value_t* p_sw = rocke_b_param(b, "p_sW", rocke_i32(), NULL);
    rocke_value_t* p_cs = rocke_b_param(b, "p_conv_stride", rocke_i32(), NULL);
    rocke_value_t* p_mult = rocke_b_param(b, "p_wo_mult", rocke_i32(), NULL);
    rocke_value_t* p_shift = rocke_b_param(b, "p_wo_shift", rocke_i32(), NULL);
    rocke_value_t* p_wo = rocke_b_param(b, "p_Wo", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* dy = rocke_b_block_id_x(b);

    const char* base_names[2] = {"h", "w"};
    rocke_value_t* base_strides[2] = {p_sh, p_sw};
    rocke_dynamic_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive_dynamic(b, "x_hw", base_names, 2, base_strides);
    if(desc == NULL)
        return;

    /* unmerge_magic_dynamic("m", ("ho", "wo"), [(p_mult, p_shift, p_Wo)]) */
    const char* um_into[2] = {"ho", "wo"};
    rocke_magic_triple_t triples[1];
    triples[0].mult = p_mult;
    triples[0].shift = p_shift;
    triples[0].dim = p_wo;
    /* embed_dynamic(("ho", "dy"), "h", strides=[p_cs, 1], offset=-1, lo=0, hi=p_H) */
    const char* h_upper[2] = {"ho", "dy"};
    rocke_value_t* h_strides_v[2] = {p_cs, NULL};
    const int h_strides_c[2] = {0, 1};
    /* embed_dynamic(("wo",), "w", strides=[2], offset=0, lo=0, hi=p_W) */
    const char* w_upper[1] = {"wo"};
    rocke_value_t* w_strides_v[1] = {NULL};
    const int w_strides_c[1] = {2};

    const rocke_transform_t* xforms[3];
    xforms[0] = rocke_unmerge_magic_dynamic(b, "m", um_into, 2, triples);
    xforms[1]
        = rocke_embed_dynamic_mixed(b, h_upper, 2, "h", h_strides_v, h_strides_c, NULL, -1, 0, p_h);
    xforms[2]
        = rocke_embed_dynamic_mixed(b, w_upper, 1, "w", w_strides_v, w_strides_c, NULL, 0, 0, p_w);
    if(xforms[0] == NULL || xforms[1] == NULL || xforms[2] == NULL)
        return;
    rocke_tensor_descriptor_t* chained
        = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 3);
    if(chained == NULL)
        return;

    const char* in_names[2] = {"m", "dy"};
    rocke_value_t* in_values[2] = {tid, dy};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    if(!rocke_transforms_descriptor_offset(b, chained, in_names, in_values, 2, &off, &valid))
        return;
    rocke_b_global_store(b, o, tid, off, 1);
    store_through_valid(b, x, y, tid, off, valid);
    rocke_b_ret(b);
}

/* Python build_unmerge_mixed: int and Value triple members mixed, plus a
 * dim-1 triple (no division). */
static void build_unmerge_mixed(rocke_ir_builder_t* b)
{
    rocke_value_t* x = ptr_param(b, "X", rocke_f16(), true);
    rocke_value_t* y = ptr_param(b, "Y", rocke_f16(), false);
    rocke_value_t* o = ptr_param(b, "O", rocke_i32(), false);
    rocke_value_t* p_sn = rocke_b_param(b, "p_sN", rocke_i32(), NULL);
    rocke_value_t* p_sh = rocke_b_param(b, "p_sH", rocke_i32(), NULL);
    rocke_value_t* p_sw = rocke_b_param(b, "p_sW", rocke_i32(), NULL);
    rocke_value_t* p_mult = rocke_b_param(b, "p_w_mult", rocke_i32(), NULL);
    rocke_value_t* p_w = rocke_b_param(b, "p_W", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);

    const char* base_names[3] = {"n", "h", "w"};
    rocke_value_t* base_strides[3] = {p_sn, p_sh, p_sw};
    rocke_dynamic_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive_dynamic(b, "x_nhw", base_names, 3, base_strides);
    if(desc == NULL)
        return;

    /* unmerge_magic_dynamic("m", ("n","h","w"), [(7, 2, 1), (p_mult, 5, p_W)]) */
    const char* um_into[3] = {"n", "h", "w"};
    rocke_magic_triple_t triples[2];
    memset(triples, 0, sizeof(triples));
    triples[0].mult_c = 7;
    triples[0].shift_c = 2;
    triples[0].dim_c = 1;
    triples[1].mult = p_mult;
    triples[1].shift_c = 5;
    triples[1].dim = p_w;

    const rocke_transform_t* xforms[2];
    xforms[0] = rocke_unmerge_magic_dynamic(b, "m", um_into, 3, triples);
    xforms[1] = rocke_pad_dynamic(b, "w", NULL, p_w);
    if(xforms[0] == NULL || xforms[1] == NULL)
        return;
    rocke_tensor_descriptor_t* chained
        = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 2);
    if(chained == NULL)
        return;

    const char* in_names[1] = {"m"};
    rocke_value_t* in_values[1] = {tid};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    if(!rocke_transforms_descriptor_offset(b, chained, in_names, in_values, 1, &off, &valid))
        return;
    rocke_b_global_store(b, o, tid, off, 1);
    store_through_valid(b, x, y, tid, off, valid);
    rocke_b_ret(b);
}

static void build_pad_dynamic(rocke_ir_builder_t* b)
{
    rocke_value_t* x = ptr_param(b, "X", rocke_f16(), true);
    rocke_value_t* y = ptr_param(b, "Y", rocke_f16(), false);
    rocke_value_t* o = ptr_param(b, "O", rocke_i32(), false);
    rocke_value_t* p_m = rocke_b_param(b, "p_M", rocke_i32(), NULL);
    rocke_value_t* p_n = rocke_b_param(b, "p_N", rocke_i32(), NULL);
    rocke_value_t* p_e = rocke_b_param(b, "p_E", rocke_i32(), NULL);
    rocke_value_t* p_ld = rocke_b_param(b, "p_ld", rocke_i32(), NULL);
    rocke_value_t* p_le = rocke_b_param(b, "p_le", rocke_i32(), NULL);
    rocke_value_t* p_s = rocke_b_param(b, "p_s", rocke_i32(), NULL);
    rocke_value_t* p_off = rocke_b_param(b, "p_off", rocke_i32(), NULL);
    rocke_value_t* p_lo = rocke_b_param(b, "p_lo", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* one = rocke_b_const_i32(b, 1);

    const char* base_names[3] = {"r", "c", "e"};
    rocke_value_t* base_strides[3] = {p_ld, p_le, one};
    rocke_dynamic_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive_dynamic(b, "x_rce", base_names, 3, base_strides);
    if(desc == NULL)
        return;

    /* embed_dynamic(("i",), "r", strides=[p_s], offset=p_off, lo=p_lo, hi=p_M) */
    const char* r_upper[1] = {"i"};
    rocke_value_t* r_strides[1] = {p_s};

    const rocke_transform_t* xforms[3];
    xforms[0] = rocke_embed_dynamic(b, r_upper, 1, "r", r_strides, p_off, p_lo, p_m);
    /* pad_dynamic("c", lo=2, hi=p_N) */
    xforms[1] = rocke_pad_dynamic_lo_const(b, "c", 2, p_n);
    /* pad_dynamic("e", hi=p_E) */
    xforms[2] = rocke_pad_dynamic(b, "e", NULL, p_e);
    if(xforms[0] == NULL || xforms[1] == NULL || xforms[2] == NULL)
        return;
    rocke_tensor_descriptor_t* chained
        = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 3);
    if(chained == NULL)
        return;

    const char* in_names[3] = {"i", "c", "e"};
    rocke_value_t* in_values[3] = {tid, bx, by};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    if(!rocke_transforms_descriptor_offset(b, chained, in_names, in_values, 3, &off, &valid))
        return;
    rocke_b_global_store(b, o, tid, off, 1);
    store_through_valid(b, x, y, tid, off, valid);
    rocke_b_ret(b);
}

typedef void (*build_fn_t)(rocke_ir_builder_t*);

/* Index-for-index with the Python emitter's CONFIGS list. */
static const build_fn_t CONFIGS[] = {
    build_pingpong_default,
    build_pingpong_split_k,
    build_pingpong_schedule,
    build_unmerge_embed,
    build_pad_dynamic,
    build_pingpong_mask,
    build_pingpong_mask_split_k,
    build_unmerge_mixed,
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
    if(rocke_ir_builder_init(&b, "dynamic_helpers") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = LDS_ELEMS */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", LDS_ELEMS);
    CONFIGS[idx](&b);

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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
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
