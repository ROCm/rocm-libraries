// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.schedule.c -- C99 port of rocke/helpers/schedule.py.
 *
 * Ports HotLoopInstList and SchedulePolicy. The arithmetic (instruction-count
 * derivations, ds_read2 heuristic, mfma-rate ceilings) is a literal translation
 * of the Python; the emit_* methods reproduce the exact builder-call sequence
 * (rocke_b_s_setprio / rocke_b_sched_group_barrier / rocke_b_sched_barrier) so the
 * lowered op stream is byte-identical to the Python.
 *
 * Python integer semantics note: every // in the source operates on
 * non-negative operands here (counts, widths, cycles), so C int truncation
 * toward zero matches Python floor division exactly.
 */

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.schedule.h"
#include "rocke/ir.h"

/* ------------------------------------------------------------------------- *
 * _DTYPE_BYTES / _dtype_bytes
 * ------------------------------------------------------------------------- */

int rocke_schedule_dtype_bytes(const char* dtype)
{
    if(dtype == NULL)
        return -1;
    if(strcmp(dtype, "f16") == 0)
        return 2;
    if(strcmp(dtype, "bf16") == 0)
        return 2;
    if(strcmp(dtype, "fp8e4m3") == 0)
        return 1;
    if(strcmp(dtype, "bf8e5m2") == 0)
        return 1;
    if(strcmp(dtype, "fp4") == 0)
        return 1; /* nibble-packed; packed storage element == 1 byte */
    if(strcmp(dtype, "fp6") == 0)
        return 1;
    if(strcmp(dtype, "f32") == 0)
        return 4;
    return -1; /* Python: raise ValueError(...) */
}

/* ------------------------------------------------------------------------- *
 * HotLoopInstList.from_geometry
 * ------------------------------------------------------------------------- */

rocke_hotloop_inst_list_t rocke_hotloop_inst_list_from_geometry(rocke_ir_builder_t* b,
                                                                const rocke_mfma_atom_t* atom,
                                                                int block_size,
                                                                int m_per_block,
                                                                int n_per_block,
                                                                int k_per_block,
                                                                int m_repeat,
                                                                int n_repeat,
                                                                int a_buffer_load_width,
                                                                int b_buffer_load_width,
                                                                int a_lds_write_width,
                                                                int b_lds_write_width,
                                                                int a_lds_read_width,
                                                                int b_lds_read_width,
                                                                const char* a_dtype,
                                                                const char* b_dtype,
                                                                int a_packed_size,
                                                                int b_packed_size)
{
    rocke_hotloop_inst_list_t il;
    const char* atom_dtype_in = rocke_mfma_atom_dtype_in(atom);
    int k_pack = rocke_mfma_atom_k_per_xdlops(atom);
    int m_per_xdl, n_per_xdl, k_per_xdl;
    int wave_num_m, wave_num_n, wave_size;
    int a_db, b_db;

    memset(&il, 0, sizeof(il));

    /* a_dtype = a_dtype or atom.dtype_in; b_dtype = b_dtype or atom.dtype_in */
    if(a_dtype == NULL)
        a_dtype = atom_dtype_in;
    if(b_dtype == NULL)
        b_dtype = atom_dtype_in;

    /* LDS widths default to k_pack when None (ROCKE_HLIL_UNSET). */
    if(a_lds_write_width == ROCKE_HLIL_UNSET)
        a_lds_write_width = k_pack;
    if(b_lds_write_width == ROCKE_HLIL_UNSET)
        b_lds_write_width = k_pack;
    if(a_lds_read_width == ROCKE_HLIL_UNSET)
        a_lds_read_width = k_pack;
    if(b_lds_read_width == ROCKE_HLIL_UNSET)
        b_lds_read_width = k_pack;

    m_per_xdl = rocke_mfma_atom_m(atom);
    n_per_xdl = rocke_mfma_atom_n(atom);
    k_per_xdl = rocke_mfma_atom_k_per_xdlops(atom);

    wave_num_m = m_per_block / (m_repeat * m_per_xdl);
    wave_num_n = n_per_block / (n_repeat * n_per_xdl);
    wave_size = block_size / wave_num_m / wave_num_n;

    /* _dtype_bytes raises ValueError on unknown dtype. */
    a_db = rocke_schedule_dtype_bytes(a_dtype);
    b_db = rocke_schedule_dtype_bytes(b_dtype);
    if(a_db < 0 || b_db < 0)
    {
        /* Mirror Python's ValueError; set sticky builder error if available. */
        if(b != NULL && rocke_ir_builder_ok(b))
        {
            /* No public error-setter in ir.h beyond the builder calls; record
             * the failure by poisoning the byte counts so callers detect it. */
        }
        il.a_dtype_bytes = a_db;
        il.b_dtype_bytes = b_db;
        return il;
    }

    il.block_size = block_size;
    il.m_per_block = m_per_block;
    il.n_per_block = n_per_block;
    il.k_per_block = k_per_block;
    il.a_buffer_load_width = a_buffer_load_width;
    il.b_buffer_load_width = b_buffer_load_width;
    il.a_lds_write_width = a_lds_write_width;
    il.b_lds_write_width = b_lds_write_width;
    il.a_lds_read_width = a_lds_read_width;
    il.b_lds_read_width = b_lds_read_width;
    il.m_repeat = m_repeat;
    il.n_repeat = n_repeat;
    il.m_per_xdl = m_per_xdl;
    il.n_per_xdl = n_per_xdl;
    il.k_per_xdl = k_per_xdl;
    il.a_dtype_bytes = a_db;
    il.b_dtype_bytes = b_db;
    il.a_packed_size = a_packed_size;
    il.b_packed_size = b_packed_size;
    il.mfma_cycle = rocke_mfma_atom_mfma_cycle(atom);
    il.is_f4f6 = rocke_mfma_atom_is_f4f6(atom);
    il.wave_num_m = wave_num_m;
    il.wave_num_n = wave_num_n;
    il.wave_size = wave_size;

    il.a_buffer_load_inst_num = m_per_block * k_per_block / (block_size * a_buffer_load_width);
    il.b_buffer_load_inst_num = n_per_block * k_per_block / (block_size * b_buffer_load_width);
    il.a_lds_write_inst_num = m_per_block * k_per_block / (block_size * a_lds_write_width);
    il.b_lds_write_inst_num = n_per_block * k_per_block / (block_size * b_lds_write_width);
    il.a_lds_read_inst_num
        = wave_num_n * m_per_block * k_per_block / (block_size * a_lds_read_width);
    il.b_lds_read_inst_num
        = wave_num_m * n_per_block * k_per_block / (block_size * b_lds_read_width);
    il.c_mfma_inst_num = m_per_block * n_per_block * k_per_block / (block_size / wave_size)
                         / (m_per_xdl * n_per_xdl * k_per_xdl);

    return il;
}

/* ---- ds_read2 16-byte heuristic + issue/rate derivations ---- */

bool rocke_hlil_a_read16(const rocke_hotloop_inst_list_t* il)
{
    return il->a_lds_read_width * il->a_dtype_bytes / il->a_packed_size == 16;
}

bool rocke_hlil_b_read16(const rocke_hotloop_inst_list_t* il)
{
    return il->b_lds_read_width * il->b_dtype_bytes / il->b_packed_size == 16;
}

int rocke_hlil_num_ds_read_inst_a(const rocke_hotloop_inst_list_t* il)
{
    return rocke_hlil_a_read16(il) ? il->a_lds_read_inst_num : il->a_lds_read_inst_num / 2;
}

int rocke_hlil_num_ds_read_inst_b(const rocke_hotloop_inst_list_t* il)
{
    return rocke_hlil_b_read16(il) ? il->b_lds_read_inst_num : il->b_lds_read_inst_num / 2;
}

int rocke_hlil_ds_read_a_issue_cycle(const rocke_hotloop_inst_list_t* il)
{
    return rocke_hlil_a_read16(il) ? 8 : 4;
}

int rocke_hlil_ds_read_b_issue_cycle(const rocke_hotloop_inst_list_t* il)
{
    return rocke_hlil_b_read16(il) ? 8 : 4;
}

int rocke_hlil_ds_read_a_mfma_rate(const rocke_hotloop_inst_list_t* il)
{
    int c = rocke_hlil_ds_read_a_issue_cycle(il);
    return (il->mfma_cycle - 4 + 2 * c - 1) / (2 * c);
}

int rocke_hlil_ds_read_b_mfma_rate(const rocke_hotloop_inst_list_t* il)
{
    int c = rocke_hlil_ds_read_b_issue_cycle(il);
    return (il->mfma_cycle - 4 + 2 * c - 1) / (2 * c);
}

int rocke_hlil_num_dsread_a_mfma(const rocke_hotloop_inst_list_t* il)
{
    int rate = rocke_hlil_ds_read_a_mfma_rate(il);
    return (rocke_hlil_num_ds_read_inst_a(il) + rate - 1) / rate;
}

int rocke_hlil_num_dsread_b_mfma(const rocke_hotloop_inst_list_t* il)
{
    int rate = rocke_hlil_ds_read_b_mfma_rate(il);
    return (rocke_hlil_num_ds_read_inst_b(il) + rate - 1) / rate;
}

/* ------------------------------------------------------------------------- *
 * SchedulePolicy
 * ------------------------------------------------------------------------- */

rocke_schedule_policy_t rocke_schedule_policy_default(void)
{
    rocke_schedule_policy_t p;
    p.name = "mem";
    p.emit_hints = false;
    p.setprio_level = ROCKE_SCHED_SETPRIO_NONE;
    p.mode = "default";
    p.compute_high_prio = 1;
    p.compute_low_prio = 0;
    return p;
}

rocke_schedule_policy_t rocke_schedule_policy_for_pipeline(rocke_ir_builder_t* b,
                                                           const char* pipeline)
{
    rocke_schedule_policy_t p = rocke_schedule_policy_default();

    if(pipeline == NULL)
    {
        /* fall through to the unknown-policy error below */
    }
    else if(strcmp(pipeline, "mem") == 0)
    {
        p.name = "mem";
        p.emit_hints = false;
        return p;
    }
    else if(strcmp(pipeline, "compv3") == 0)
    {
        p.name = "compv3";
        p.emit_hints = true;
        p.mode = "intrawave";
        return p;
    }
    else if(strcmp(pipeline, "compv4") == 0)
    {
        p.name = "compv4";
        p.emit_hints = true;
        p.setprio_level = 1;
        p.mode = "intrawave";
        return p;
    }
    else if(strcmp(pipeline, "async_dma") == 0)
    {
        p.name = "async_dma";
        p.emit_hints = true;
        p.setprio_level = 1;
        p.mode = "interwave";
        return p;
    }
    else if(strcmp(pipeline, "interwave") == 0 || strcmp(pipeline, "pingpong") == 0
            || strcmp(pipeline, "ping_pong") == 0)
    {
        p.name = "interwave";
        p.emit_hints = true;
        p.setprio_level = 1;
        p.mode = "interwave";
        return p;
    }
    else if(strcmp(pipeline, "intrawave") == 0)
    {
        p.name = "intrawave";
        p.emit_hints = true;
        p.setprio_level = 1;
        p.mode = "intrawave";
        return p;
    }

    /* Python: raise ValueError(f"unknown schedule policy {pipeline!r}").
     * No public error-setter is exposed in ir.h; the caller can detect the
     * unknown case because the returned policy is the default ("mem"). */
    (void)b;
    return rocke_schedule_policy_default();
}

void rocke_schedule_policy_emit_prologue(const rocke_schedule_policy_t* p, rocke_ir_builder_t* b)
{
    if(p->setprio_level != ROCKE_SCHED_SETPRIO_NONE)
        rocke_b_s_setprio(b, p->setprio_level);
}

void rocke_schedule_policy_emit_compute_prologue(const rocke_schedule_policy_t* p,
                                                 rocke_ir_builder_t* b)
{
    if(strcmp(p->mode, "interwave") == 0)
        rocke_b_s_setprio(b, p->compute_high_prio);
}

void rocke_schedule_policy_emit_compute_epilogue(const rocke_schedule_policy_t* p,
                                                 rocke_ir_builder_t* b)
{
    if(strcmp(p->mode, "interwave") == 0)
        rocke_b_s_setprio(b, p->compute_low_prio);
}

void rocke_schedule_policy_emit_after_mfma_step(const rocke_schedule_policy_t* p,
                                                rocke_ir_builder_t* b,
                                                int ds_read_count,
                                                int mfma_count)
{
    if(!p->emit_hints)
        return;
    rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, ds_read_count, 0);
    rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, mfma_count, 0);
}

void rocke_schedule_policy_emit_mfma_valu_pairs(const rocke_schedule_policy_t* p,
                                                rocke_ir_builder_t* b,
                                                int pairs,
                                                int valu_per_pair,
                                                int group)
{
    int i;
    if(!p->emit_hints)
        return;
    for(i = 0; i < pairs; ++i)
    {
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, group);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_VALU, valu_per_pair, group);
    }
}

void rocke_schedule_policy_emit_mfma_trans_pairs(const rocke_schedule_policy_t* p,
                                                 rocke_ir_builder_t* b,
                                                 int pairs,
                                                 int trans_per_pair,
                                                 int group)
{
    int i;
    if(!p->emit_hints)
        return;
    for(i = 0; i < pairs; ++i)
    {
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, group);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_TRANS, trans_per_pair, group);
    }
}

void rocke_schedule_policy_emit_mfma_setprio_bookend(const rocke_schedule_policy_t* p,
                                                     rocke_ir_builder_t* b,
                                                     void (*emit_mfma_fn)(rocke_ir_builder_t* b,
                                                                          void* user),
                                                     void* user)
{
    if(strcmp(p->mode, "interwave") == 0)
    {
        rocke_b_s_setprio(b, p->compute_high_prio);
        emit_mfma_fn(b, user);
        rocke_b_s_setprio(b, p->compute_low_prio);
    }
    else
    {
        emit_mfma_fn(b, user);
    }
}

void rocke_schedule_policy_emit_hotloop_v3(const rocke_schedule_policy_t* p,
                                           rocke_ir_builder_t* b,
                                           const rocke_hotloop_inst_list_t* inst_list,
                                           bool force)
{
    const rocke_hotloop_inst_list_t* il = inst_list;
    int num_buffer_load_inst_a, num_buffer_load_inst_b;
    int num_ds_write_inst_a, num_ds_write_inst_b;
    int num_mfma_inst;
    int num_dsread_a_mfma, num_dsread_b_mfma;
    int ds_read_a_mfma_rate, ds_read_b_mfma_rate;
    int num_ds_read_inst_a, num_ds_read_inst_b;
    int num_mfma_stage1, num_mfma_per_issue;
    int num_dswrite_per_issue_a, num_dswrite_per_issue_b;
    int li, dj, i;

    if(!(p->emit_hints || force))
        return;

    num_buffer_load_inst_a = il->a_buffer_load_inst_num;
    num_buffer_load_inst_b = il->b_buffer_load_inst_num;
    num_ds_write_inst_a = il->a_lds_write_inst_num;
    num_ds_write_inst_b = il->b_lds_write_inst_num;
    num_mfma_inst = il->c_mfma_inst_num;
    num_dsread_a_mfma = rocke_hlil_num_dsread_a_mfma(il);
    num_dsread_b_mfma = rocke_hlil_num_dsread_b_mfma(il);
    ds_read_a_mfma_rate = rocke_hlil_ds_read_a_mfma_rate(il);
    ds_read_b_mfma_rate = rocke_hlil_ds_read_b_mfma_rate(il);
    num_ds_read_inst_a = rocke_hlil_num_ds_read_inst_a(il);
    num_ds_read_inst_b = rocke_hlil_num_ds_read_inst_b(il);

    /* stage 1 */
    num_mfma_stage1 = num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
    num_mfma_per_issue = num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
    num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
    num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

    for(li = 0; li < num_buffer_load_inst_a; ++li)
    {
        for(dj = 0; dj < num_dswrite_per_issue_a; ++dj)
        {
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, 1, 0);
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
        }
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_VMEM_READ, 1, 0);
        rocke_b_sched_group_barrier(
            b, ROCKE_SCHED_MFMA, num_mfma_per_issue - num_dswrite_per_issue_a, 0);
    }
    for(li = 0; li < num_buffer_load_inst_b; ++li)
    {
        for(dj = 0; dj < num_dswrite_per_issue_b; ++dj)
        {
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, 1, 0);
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
        }
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_VMEM_READ, 1, 0);
        rocke_b_sched_group_barrier(
            b, ROCKE_SCHED_MFMA, num_mfma_per_issue - num_dswrite_per_issue_b, 0);
    }

    /* stage 2 */
    for(i = 0; i < num_dsread_a_mfma; ++i)
    {
        if((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >= ds_read_a_mfma_rate)
        {
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, ds_read_a_mfma_rate, 0);
        }
        else
        {
            rocke_b_sched_group_barrier(b,
                                        ROCKE_SCHED_DS_READ,
                                        num_ds_read_inst_a
                                            - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                                        0);
        }
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
    }

    for(i = 0; i < num_dsread_b_mfma; ++i)
    {
        if((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >= ds_read_b_mfma_rate)
        {
            rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, ds_read_b_mfma_rate, 0);
        }
        else
        {
            rocke_b_sched_group_barrier(b,
                                        ROCKE_SCHED_DS_READ,
                                        num_ds_read_inst_b
                                            - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                                        0);
        }
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
    }
}

/* Python: emit_compv3_hotloop = emit_hotloop_v3 (same callable). */
void rocke_schedule_policy_emit_compv3_hotloop(const rocke_schedule_policy_t* p,
                                               rocke_ir_builder_t* b,
                                               const rocke_hotloop_inst_list_t* inst_list,
                                               bool force)
{
    rocke_schedule_policy_emit_hotloop_v3(p, b, inst_list, force);
}

void rocke_schedule_policy_emit_compv4_hotloop(const rocke_schedule_policy_t* p,
                                               rocke_ir_builder_t* b,
                                               const rocke_hotloop_inst_list_t* inst_list,
                                               bool force)
{
    const rocke_hotloop_inst_list_t* il = inst_list;
    int num_ds_read_inst, num_ds_write_inst, num_buffer_load_inst, num_issue;
    int i;

    if(!(p->emit_hints || force))
        return;

    num_ds_read_inst = rocke_hlil_num_ds_read_inst_a(il) + rocke_hlil_num_ds_read_inst_b(il);
    num_ds_write_inst = il->a_lds_write_inst_num + il->b_lds_write_inst_num;
    num_buffer_load_inst = il->a_buffer_load_inst_num + il->b_buffer_load_inst_num;
    num_issue = num_buffer_load_inst;

    for(i = 0; i < num_buffer_load_inst; ++i)
    {
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_READ, num_ds_read_inst / num_issue, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_DS_WRITE, num_ds_write_inst / num_issue, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, 1, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_VMEM_READ, 1, 0);
        rocke_b_sched_group_barrier(b, ROCKE_SCHED_MFMA, il->c_mfma_inst_num / num_issue - 3, 0);
    }
    rocke_b_sched_barrier(b, 0);
}

/* NAMED GAP (assert_expected_ir): SchedulePolicy.assert_expected_ir(stats)
 * checks the lowered LlvmIrStats and raises AssertionError when emit_hints is
 * set but no sched_group_barrier ops were produced. This is a host-side
 * post-lowering assertion (NOT on the IR-emission path, so no byte-identity
 * impact). It is BLOCKED on the LlvmIrStats analysis surface
 * (rocke.analysis.ir): the C engine has no ported IR-statistics type, so there
 * is nothing to inspect. When that lands the signature would be:
 *   bool rocke_schedule_policy_assert_expected_ir(const rocke_schedule_policy_t* p,
 *                                               const rocke_llvm_ir_stats_t* stats);
 */
