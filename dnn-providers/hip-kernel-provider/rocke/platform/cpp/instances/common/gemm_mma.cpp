// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_mma.c -- C99 port of the MMA / fragment-load / hotloop-schedule
 * module helpers of build_universal_gemm (rocke/instances/common/gemm_universal.py,
 * lines ~546-736):
 *
 *   _mfma_atom_widths           -> (consumed by rocke_gemm_emit_mfma / _zero_acc)
 *   _emit_mfma                  -> rocke_gemm_emit_mfma
 *   _emit_mma                   -> rocke_gemm_emit_mma
 *   _emit_zero_acc              -> rocke_gemm_emit_zero_acc
 *   _emit_zero_acc_op           -> rocke_gemm_emit_zero_acc_op
 *   _atom_frag_lengths          -> rocke_gemm_atom_frag_lengths
 *   _choose_load_vec            -> rocke_gemm_choose_load_vec
 *   _emit_smem_load             -> rocke_gemm_emit_smem_load
 *   _hotloop_inst_list          -> rocke_gemm_hotloop_inst_list
 *   _hotloop_well_formed        -> rocke_gemm_hotloop_well_formed
 *   _emit_hotloop_schedule      -> rocke_gemm_emit_hotloop_schedule
 *
 * These are the module-level pure helpers (no closure capture); they read only
 * the spec / resolved op and emit IR through the builder in the byte-identical
 * Python builder-call order. Peer phase functions (load/compute/epilogue) live
 * in sibling translation units and are reached via the shared internal header.
 */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_gemm_internal.h"

/* ====================================================================== *
 * _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane)
 *
 * Python:
 *   t = spec.tile; waves = spec.wave_size
 *   a_per = (t.warp_tile_m * t.warp_tile_k) // waves
 *   b_per = (t.warp_tile_k * t.warp_tile_n) // waves
 *   c_per = (t.warp_tile_m * t.warp_tile_n) // waves
 *
 * Module-level Python helper (not in the internal-header prototype set since the
 * contract-driven body uses _atom_frag_lengths). It is needed locally to back
 * rocke_gemm_emit_zero_acc, which is in this file's scope, so it stays file-static.
 * ====================================================================== */
static void rocke__mfma_atom_widths(const rocke_gemm_universal_spec_t* spec,
                                    int* a_per,
                                    int* b_per,
                                    int* c_per)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int waves = spec->wave_size;
    *a_per = (t->warp_tile_m * t->warp_tile_k) / waves;
    *b_per = (t->warp_tile_k * t->warp_tile_n) / waves;
    *c_per = (t->warp_tile_m * t->warp_tile_n) / waves;
}

/* ====================================================================== *
 * _emit_mfma(b, spec, a, bb, c): ISA-named MFMA dispatch (MFMA-only callers).
 *
 * Python routes (storage_dtype, warp_tile key) -> the matching ISA-named builder
 * method. We reproduce the exact branch order. The storage dtype is resolved the
 * same way Python's _storage_dtype does: io_ir_type(spec.data.dtype_a) (the
 * homogeneous-dtype / fp32-acc / RCR validation is performed elsewhere; here we
 * only need the IR storage type to select the f16/bf16 atom family, matching the
 * Python `dtype == F16` / `dtype == BF16` comparisons).
 * ====================================================================== */
rocke_value_t* rocke_gemm_emit_mfma(rocke_ir_builder_t* b,
                                    const rocke_gemm_universal_spec_t* spec,
                                    rocke_value_t* a,
                                    rocke_value_t* bb,
                                    rocke_value_t* c)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int km = t->warp_tile_m;
    int kn = t->warp_tile_n;
    int kk = t->warp_tile_k;
    const rocke_type_t* dtype = rocke_io_ir_type(spec->data.dtype_a);

    if(dtype == rocke_f16())
    {
        if(km == 16 && kn == 16 && kk == 16)
            return rocke_b_mfma_f32_16x16x16_f16(b, a, bb, c);
        if(km == 16 && kn == 16 && kk == 32)
            return rocke_b_mfma_f32_16x16x32_f16(b, a, bb, c);
        if(km == 32 && kn == 32 && kk == 8)
            return rocke_b_mfma_f32_32x32x8_f16(b, a, bb, c);
        if(km == 32 && kn == 32 && kk == 16)
            return rocke_b_mfma_f32_32x32x16_f16(b, a, bb, c);
    }
    if(dtype == rocke_bf16())
    {
        if(km == 16 && kn == 16 && kk == 16)
            return rocke_b_mfma_f32_16x16x16_bf16(b, a, bb, c);
        if(km == 16 && kn == 16 && kk == 32)
            return rocke_b_mfma_f32_16x16x32_bf16(b, a, bb, c);
    }
    /* Python: raise NotImplementedError(f"no MFMA emitter for {dtype} {key}"). */
    if(b && b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_NOTIMPL;
        snprintf(b->err,
                 ROCKE_ERR_MSG_CAP,
                 "no MFMA emitter for %s warp_tile (%d, %d, %d)",
                 dtype ? dtype->name : "?",
                 km,
                 kn,
                 kk);
    }
    return NULL;
}

/* ====================================================================== *
 * _emit_zero_acc(b, spec): zero accumulator sized from spec geometry (MFMA-only).
 *   _, _, c_per = _mfma_atom_widths(spec); return b.zero_vec_f32(c_per)
 * ====================================================================== */
rocke_value_t* rocke_gemm_emit_zero_acc(rocke_ir_builder_t* b,
                                        const rocke_gemm_universal_spec_t* spec)
{
    int a_per, b_per, c_per;
    rocke__mfma_atom_widths(spec, &a_per, &b_per, &c_per);
    (void)a_per;
    (void)b_per;
    return rocke_b_zero_vec_f32(b, c_per);
}

/* ====================================================================== *
 * _atom_frag_lengths(op) -> (a_frag_len, b_frag_len, c_frag_len)
 *
 * Pure read of the resolved MmaOp fragment lengths.
 * ====================================================================== */
void rocke_gemm_atom_frag_lengths(const rocke_mmaop_t* op, int* a_frag, int* b_frag, int* c_frag)
{
    if(a_frag)
        *a_frag = op->a_frag_len;
    if(b_frag)
        *b_frag = op->b_frag_len;
    if(c_frag)
        *c_frag = op->c_frag_len;
}

/* ====================================================================== *
 * _emit_mma(b, op, a, bb, c): target-neutral D = a*bb + c via IRBuilder.mma.
 *
 * Python: return b.mma(op, a, bb, c). The C builder takes op_id + the optional
 * scaled-MX `extra` operand list (NULL/0 here, mirroring the unscaled mma()).
 * ====================================================================== */
rocke_value_t* rocke_gemm_emit_mma(rocke_ir_builder_t* b,
                                   const rocke_mmaop_t* op,
                                   rocke_value_t* a,
                                   rocke_value_t* bb,
                                   rocke_value_t* c)
{
    return rocke_b_mma(b, op->op_id, a, bb, c, NULL, 0);
}

/* ====================================================================== *
 * _emit_zero_acc_op(b, op): zero accumulator sized from op.c_frag_len.
 *   return b.zero_vec_f32(op.c_frag_len)
 * ====================================================================== */
rocke_value_t* rocke_gemm_emit_zero_acc_op(rocke_ir_builder_t* b, const rocke_mmaop_t* op)
{
    return rocke_b_zero_vec_f32(b, op->c_frag_len);
}

/* ====================================================================== *
 * _choose_load_vec(spec) -> width.
 *   t = spec.tile
 *   return choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size)
 *
 * The internal-header prototype returns int. The shared picker returns a status
 * + out-param; on the Python ValueError path (no usable width) we return 0,
 * which the caller never reaches for a spec that passed is_valid_spec.
 * ====================================================================== */
int rocke_gemm_choose_load_vec(const rocke_gemm_universal_spec_t* spec)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int vec = 0;
    rocke_status_t st
        = rocke_choose_load_vec(t->tile_m, t->tile_n, t->tile_k, spec->block_size, &vec);
    if(st != ROCKE_OK)
        return 0; /* Python raises ValueError; unreachable for a valid spec. */
    return vec;
}

/* ====================================================================== *
 * _emit_smem_load(b, smem, row, col, n, dtype):
 *   if dtype == F16 and n == 4: return b.smem_load_v4_f16(smem, row, col)
 *   return b.smem_load_vN(smem, row, col, dtype=dtype, n=n)
 *
 * The C smem_load_vN takes an explicit (row, col) index pair as an indices
 * array of length 2, matching the Python (smem, row, col) call.
 * ====================================================================== */
rocke_value_t* rocke_gemm_emit_smem_load(rocke_ir_builder_t* b,
                                         rocke_value_t* smem,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         int n,
                                         const rocke_type_t* dtype)
{
    if(dtype == rocke_f16() && n == 4)
        return rocke_b_smem_load_v4_f16(b, smem, row, col);
    {
        rocke_value_t* idx[2];
        idx[0] = row;
        idx[1] = col;
        return rocke_b_smem_load_vN(b, smem, idx, 2, dtype, n);
    }
}

/* ====================================================================== *
 * _hotloop_inst_list(spec, load_vec) -> HotLoopInstList
 *
 * Python:
 *   atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
 *   return HotLoopInstList.from_geometry(
 *       atom=atom, block_size=spec.block_size,
 *       m_per_block=t.tile_m, n_per_block=t.tile_n, k_per_block=t.tile_k,
 *       m_repeat=t.mfmas_per_warp_m, n_repeat=t.mfmas_per_warp_n,
 *       a_buffer_load_width=load_vec, b_buffer_load_width=load_vec)
 *
 * All other from_geometry args take their Python defaults (LDS read/write widths
 * = None => atom k-pack; a/b dtype = None => atom.dtype_in; packed_size = 1). The
 * C from_geometry maps None to ROCKE_HLIL_UNSET / NULL / its default-1 args.
 * ====================================================================== */
rocke_hotloop_inst_list_t rocke_gemm_hotloop_inst_list(rocke_ir_builder_t* b,
                                                       const rocke_gemm_universal_spec_t* spec,
                                                       int load_vec)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom(spec->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    int m_repeat = rocke_gemm_tile_mfmas_per_warp_m(t);
    int n_repeat = rocke_gemm_tile_mfmas_per_warp_n(t);
    return rocke_hotloop_inst_list_from_geometry(b,
                                                 atom,
                                                 spec->block_size,
                                                 t->tile_m,
                                                 t->tile_n,
                                                 t->tile_k,
                                                 m_repeat,
                                                 n_repeat,
                                                 load_vec,
                                                 load_vec,
                                                 ROCKE_HLIL_UNSET, /* a_lds_write_width = None */
                                                 ROCKE_HLIL_UNSET, /* b_lds_write_width = None */
                                                 ROCKE_HLIL_UNSET, /* a_lds_read_width  = None */
                                                 ROCKE_HLIL_UNSET, /* b_lds_read_width  = None */
                                                 NULL, /* a_dtype = atom.dtype_in */
                                                 NULL, /* b_dtype = atom.dtype_in */
                                                 1, /* a_packed_size */
                                                 1); /* b_packed_size */
}

/* ====================================================================== *
 * _hotloop_well_formed(il, pipeline) -> bool
 *
 * Faithful port of the v3/v4 non-negative-count guard.
 * ====================================================================== */
bool rocke_gemm_hotloop_well_formed(const rocke_hotloop_inst_list_t* il, const char* pipeline)
{
    int num_buffer_load = il->a_buffer_load_inst_num + il->b_buffer_load_inst_num;
    if(num_buffer_load <= 0)
        return false;
    if(strcmp(pipeline, "compv3") == 0)
    {
        int num_dsread_a = rocke_hlil_num_dsread_a_mfma(il);
        int num_dsread_b = rocke_hlil_num_dsread_b_mfma(il);
        int num_mfma_stage1 = il->c_mfma_inst_num - (num_dsread_a + num_dsread_b);
        if(num_mfma_stage1 < 0)
            return false;
        int num_mfma_per_issue = num_mfma_stage1 / num_buffer_load;
        if(il->a_buffer_load_inst_num <= 0 || il->b_buffer_load_inst_num <= 0)
            return false;
        int dswr_a = il->a_lds_write_inst_num / il->a_buffer_load_inst_num;
        int dswr_b = il->b_lds_write_inst_num / il->b_buffer_load_inst_num;
        return (num_mfma_per_issue - dswr_a) >= 0 && (num_mfma_per_issue - dswr_b) >= 0;
    }
    /* compv4: trailing MFMA group is C_MFMA / num_issue - 3. */
    return (il->c_mfma_inst_num / num_buffer_load - 3) >= 0;
}

/* ====================================================================== *
 * _emit_hotloop_schedule(b, spec, load_vec):
 *
 *   pipeline = spec.trait.pipeline
 *   il = _hotloop_inst_list(spec, load_vec)
 *   policy = SchedulePolicy.for_pipeline(pipeline)
 *   if _hotloop_well_formed(il, pipeline):
 *       if pipeline == "compv3": policy.emit_compv3_hotloop(b, il)
 *       else:                    policy.emit_compv4_hotloop(b, il)
 *       return
 *   b.sched_group_barrier(0x100, 1, 0)
 *   b.sched_group_barrier(0x008, t.mfmas_per_warp_m * t.mfmas_per_warp_n, 0)
 *
 * The C emit_compv3/4_hotloop take a trailing `force` bool (Python default
 * False); we pass false to match the un-forced emission.
 * ====================================================================== */
void rocke_gemm_emit_hotloop_schedule(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      int load_vec)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const char* pipeline = spec->trait.pipeline;
    rocke_hotloop_inst_list_t il = rocke_gemm_hotloop_inst_list(b, spec, load_vec);
    rocke_schedule_policy_t policy = rocke_schedule_policy_for_pipeline(b, pipeline);

    if(rocke_gemm_hotloop_well_formed(&il, pipeline))
    {
        if(strcmp(pipeline, "compv3") == 0)
            rocke_schedule_policy_emit_compv3_hotloop(&policy, b, &il, false);
        else
            rocke_schedule_policy_emit_compv4_hotloop(&policy, b, &il, false);
        return;
    }
    /* Degenerate tile: keep the prior flat hint. */
    rocke_b_sched_group_barrier(b, 0x100, 1, 0); /* one DS read */
    rocke_b_sched_group_barrier(
        b, 0x008, rocke_gemm_tile_mfmas_per_warp_m(t) * rocke_gemm_tile_mfmas_per_warp_n(t), 0);
}
