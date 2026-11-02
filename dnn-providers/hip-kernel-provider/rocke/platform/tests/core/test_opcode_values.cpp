// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_opcode_values.cpp -- pins the numeric value of every opcode.
 *
 * rocke_opcode_t is part of the C ABI: a caller compiled against an older
 * rocke/ir.h passes and receives opcodes as plain integers. A new opcode must
 * therefore be appended at the end of the enum; inserting one anywhere else
 * renumbers every opcode after it and silently changes what an old caller's
 * integer means (see "when to bump" in rocke/abi.h).
 *
 * The table below lists every opcode with its value, its dotted name and its
 * purity as they stood before any opcode was appended. It must never change: a
 * failure here means an opcode was inserted, removed or reordered. Opcodes
 * appended later are not listed; the second check covers them by requiring
 * that every opcode, old or new, has a name that maps back to it.
 */

#include "rocke/ir.h"
#include <cstdio>
#include <cstring>

namespace
{

struct PinnedOpcode
{
    rocke_opcode_t op;
    int value;
    const char* name;
    bool pure;
};

constexpr PinnedOpcode kPinned[] = {
    {ROCKE_OP_ARITH_CONSTANT, 1, "arith.constant", true},
    {ROCKE_OP_ARITH_CONSTANT_VEC, 2, "arith.constant_vec", true},
    {ROCKE_OP_ARITH_ADD, 3, "arith.add", true},
    {ROCKE_OP_ARITH_SUB, 4, "arith.sub", true},
    {ROCKE_OP_ARITH_MUL, 5, "arith.mul", true},
    {ROCKE_OP_ARITH_DIV, 6, "arith.div", true},
    {ROCKE_OP_ARITH_MOD, 7, "arith.mod", true},
    {ROCKE_OP_ARITH_FADD, 8, "arith.fadd", true},
    {ROCKE_OP_ARITH_FSUB, 9, "arith.fsub", true},
    {ROCKE_OP_ARITH_FMUL, 10, "arith.fmul", true},
    {ROCKE_OP_ARITH_FDIV, 11, "arith.fdiv", true},
    {ROCKE_OP_ARITH_FNEG, 12, "arith.fneg", true},
    {ROCKE_OP_ARITH_FABS, 13, "arith.fabs", false},
    {ROCKE_OP_ARITH_FMA, 14, "arith.fma", false},
    {ROCKE_OP_ARITH_FMAX3, 15, "arith.fmax3", false},
    {ROCKE_OP_ARITH_FMIN3, 16, "arith.fmin3", false},
    {ROCKE_OP_ARITH_CMP, 17, "arith.cmp", true},
    {ROCKE_OP_ARITH_FCMP, 18, "arith.fcmp", true},
    {ROCKE_OP_ARITH_FMAX, 19, "arith.fmax", true},
    {ROCKE_OP_ARITH_FMIN, 20, "arith.fmin", true},
    {ROCKE_OP_ARITH_AND, 21, "arith.and", true},
    {ROCKE_OP_ARITH_OR, 22, "arith.or", true},
    {ROCKE_OP_ARITH_NOT, 23, "arith.not", true},
    {ROCKE_OP_ARITH_SMAX, 24, "arith.smax", true},
    {ROCKE_OP_ARITH_SMIN, 25, "arith.smin", true},
    {ROCKE_OP_ARITH_XOR, 26, "arith.xor", true},
    {ROCKE_OP_ARITH_SHL, 27, "arith.shl", true},
    {ROCKE_OP_ARITH_LSHR, 28, "arith.lshr", true},
    {ROCKE_OP_ARITH_UMUL_HI_I32, 29, "arith.umul_hi_i32", true},
    {ROCKE_OP_ARITH_ZEXT, 30, "arith.zext", true},
    {ROCKE_OP_ARITH_SEXT, 31, "arith.sext", true},
    {ROCKE_OP_ARITH_TRUNC, 32, "arith.trunc", true},
    {ROCKE_OP_ARITH_SELECT, 33, "arith.select", true},
    {ROCKE_OP_ARITH_BITCAST, 34, "arith.bitcast", true},
    {ROCKE_OP_ARITH_TRUNC_F32_TO_F16, 35, "arith.trunc_f32_to_f16", true},
    {ROCKE_OP_ARITH_TRUNC_F32_TO_BF16, 36, "arith.trunc_f32_to_bf16", true},
    {ROCKE_OP_ARITH_RINT_F32, 37, "arith.rint_f32", true},
    {ROCKE_OP_ARITH_CAST_TO_F32, 38, "arith.cast_to_f32", true},
    {ROCKE_OP_ARITH_CAST_F32_TO, 39, "arith.cast_f32_to", true},
    {ROCKE_OP_ARITH_SITOFP_F32, 40, "arith.sitofp_f32", true},
    {ROCKE_OP_ARITH_CVT_FP8_TO_F32, 41, "arith.cvt_fp8_to_f32", true},
    {ROCKE_OP_ARITH_CVT_BF8_TO_F32, 42, "arith.cvt_bf8_to_f32", true},
    {ROCKE_OP_ARITH_CVT_PK_F32_FP8X4, 43, "arith.cvt_pk_f32_fp8x4", true},
    {ROCKE_OP_ARITH_CVT_PK_F32_BF8X4, 44, "arith.cvt_pk_f32_bf8x4", true},
    {ROCKE_OP_ARITH_CVT_PK_FP8_F32X4, 45, "arith.cvt_pk_fp8_f32x4", true},
    {ROCKE_OP_ARITH_CVT_PK_BF8_F32X4, 46, "arith.cvt_pk_bf8_f32x4", true},
    {ROCKE_OP_ARITH_CVT_PK_I8_F32X4, 47, "arith.cvt_pk_i8_f32x4", false},
    {ROCKE_OP_ARITH_CVT_F32_TO_FP8, 48, "arith.cvt_f32_to_fp8", true},
    {ROCKE_OP_ARITH_CVT_F32_TO_BF8, 49, "arith.cvt_f32_to_bf8", true},
    {ROCKE_OP_ARITH_CVT_F32_TO_I8_SAT, 50, "arith.cvt_f32_to_i8_sat", true},
    {ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_FP8, 51, "arith.cvt_scalef32_pk_f32_fp8", true},
    {ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_BF8, 52, "arith.cvt_scalef32_pk_f32_bf8", true},
    {ROCKE_OP_ARITH_CVT_SCALEF32_PK_FP8_F32, 53, "arith.cvt_scalef32_pk_fp8_f32", true},
    {ROCKE_OP_ARITH_CVT_SCALEF32_PK_BF8_F32, 54, "arith.cvt_scalef32_pk_bf8_f32", true},
    {ROCKE_OP_MATH_EXP2, 55, "math.exp2", true},
    {ROCKE_OP_MATH_EXP2_FAST, 56, "math.exp2_fast", true},
    {ROCKE_OP_MATH_LOG2, 57, "math.log2", true},
    {ROCKE_OP_MATH_RCP, 58, "math.rcp", true},
    {ROCKE_OP_MATH_RCP_FAST, 59, "math.rcp_fast", true},
    {ROCKE_OP_MATH_SQRT, 60, "math.sqrt", true},
    {ROCKE_OP_MATH_RSQRT, 61, "math.rsqrt", true},
    {ROCKE_OP_MATH_TANH, 62, "math.tanh", true},
    {ROCKE_OP_GPU_THREAD_ID, 63, "gpu.thread_id", true},
    {ROCKE_OP_GPU_BLOCK_ID, 64, "gpu.block_id", true},
    {ROCKE_OP_GPU_CLUSTER_ID, 65, "gpu.cluster_id", true},
    {ROCKE_OP_GPU_CLUSTER_WORKGROUP_ID, 66, "gpu.cluster_workgroup_id", true},
    {ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_ID, 67, "gpu.cluster_workgroup_max_id", true},
    {ROCKE_OP_GPU_CLUSTER_WORKGROUP_FLAT_ID, 68, "gpu.cluster_workgroup_flat_id", true},
    {ROCKE_OP_GPU_CLUSTER_WORKGROUP_MAX_FLAT_ID, 69, "gpu.cluster_workgroup_max_flat_id", true},
    {ROCKE_OP_MEMREF_GLOBAL_LOAD, 70, "memref.global_load", false},
    {ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED, 71, "memref.global_load_typed", false},
    {ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, 72, "memref.global_load_vN", false},
    {ROCKE_OP_MEMREF_GLOBAL_STORE, 73, "memref.global_store", false},
    {ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED, 74, "memref.global_store_typed", false},
    {ROCKE_OP_MEMREF_GLOBAL_STORE_VN, 75, "memref.global_store_vN", false},
    {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD, 76, "memref.global_atomic_add", false},
    {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, 77, "memref.global_atomic_add_f32", false},
    {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16, 78, "memref.global_atomic_add_pk_bf16", false},
    {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_F16, 79, "memref.global_atomic_add_pk_f16", false},
    {ROCKE_OP_MEMREF_COOPERATIVE_GLOBAL_STORE, 80, "memref.cooperative_global_store", false},
    {ROCKE_OP_VECTOR_ADD, 81, "vector.add", true},
    {ROCKE_OP_VECTOR_SUB, 82, "vector.sub", false},
    {ROCKE_OP_VECTOR_MUL, 83, "vector.mul", true},
    {ROCKE_OP_VECTOR_AND, 84, "vector.and", true},
    {ROCKE_OP_VECTOR_OR, 85, "vector.or", true},
    {ROCKE_OP_VECTOR_SHL, 86, "vector.shl", true},
    {ROCKE_OP_VECTOR_LSHR, 87, "vector.lshr", true},
    {ROCKE_OP_VECTOR_SMAX, 88, "vector.smax", true},
    {ROCKE_OP_VECTOR_SMIN, 89, "vector.smin", true},
    {ROCKE_OP_VECTOR_MAX, 90, "vector.max", true},
    {ROCKE_OP_VECTOR_FMA, 91, "vector.fma", false},
    {ROCKE_OP_VECTOR_SUM, 92, "vector.sum", true},
    {ROCKE_OP_VECTOR_REDUCE_MAX, 93, "vector.reduce_max", true},
    {ROCKE_OP_VECTOR_SPLAT, 94, "vector.splat", true},
    {ROCKE_OP_VECTOR_SELECT, 95, "vector.select", true},
    {ROCKE_OP_VECTOR_CMP, 96, "vector.cmp", true},
    {ROCKE_OP_VECTOR_TRUNC, 97, "vector.trunc", true},
    {ROCKE_OP_VECTOR_SEXT, 98, "vector.sext", true},
    {ROCKE_OP_VECTOR_TRUNC_F32_TO_F16, 99, "vector.trunc_f32_to_f16", true},
    {ROCKE_OP_VECTOR_TRUNC_F32_TO, 100, "vector.trunc_f32_to", true},
    {ROCKE_OP_VECTOR_BITCAST, 101, "vector.bitcast", true},
    {ROCKE_OP_VECTOR_EXTRACT, 102, "vector.extract", true},
    {ROCKE_OP_VECTOR_INSERT, 103, "vector.insert", true},
    {ROCKE_OP_VECTOR_PACK, 104, "vector.pack", true},
    {ROCKE_OP_VECTOR_CONCAT, 105, "vector.concat", true},
    {ROCKE_OP_TILE_SMEM_ALLOC, 106, "tile.smem_alloc", false},
    {ROCKE_OP_TILE_SMEM_STORE, 107, "tile.smem_store", false},
    {ROCKE_OP_TILE_SMEM_STORE_VN, 108, "tile.smem_store_vN", false},
    {ROCKE_OP_TILE_SMEM_STORE_VN_F32, 109, "tile.smem_store_vN_f32", false},
    {ROCKE_OP_TILE_SMEM_STORE_DISTRIBUTED, 110, "tile.smem_store_distributed", false},
    {ROCKE_OP_TILE_SMEM_LOAD_V4, 111, "tile.smem_load_v4", false},
    {ROCKE_OP_TILE_SMEM_LOAD_VN, 112, "tile.smem_load_vN", false},
    {ROCKE_OP_TILE_SMEM_LOAD_VN_F32, 113, "tile.smem_load_vN_f32", false},
    {ROCKE_OP_TILE_SMEM_ADDR_OF, 114, "tile.smem_addr_of", true},
    {ROCKE_OP_TILE_SMEM_PTR_ADD, 115, "tile.smem_ptr_add", true},
    {ROCKE_OP_TILE_LDS_ATOMIC_ADD, 116, "tile.lds_atomic_add", false},
    {ROCKE_OP_TILE_GLOBAL_PTR_ADD, 117, "tile.global_ptr_add", false},
    {ROCKE_OP_TILE_GLOBAL_LOAD_LDS, 118, "tile.global_load_lds", false},
    {ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS, 119, "tile.async_buffer_load_lds", false},
    {ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR, 120, "tile.async_buffer_load_lds_addr", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_LDS_ASYNC, 121, "tile.buffer_load_lds_async", false},
    {ROCKE_OP_TILE_GLOBAL_LOAD_ASYNC_TO_LDS, 122, "tile.global_load_async_to_lds", false},
    {ROCKE_OP_TILE_GLOBAL_STORE_ASYNC_FROM_LDS, 123, "tile.global_store_async_from_lds", false},
    {ROCKE_OP_TILE_GLOBAL_LOAD_TR16_B128, 124, "tile.global_load_tr16_b128", false},
    {ROCKE_OP_TILE_TENSOR_LOAD_TO_LDS, 125, "tile.tensor_load_to_lds", false},
    {ROCKE_OP_TILE_TENSOR_STORE_FROM_LDS, 126, "tile.tensor_store_from_lds", false},
    {ROCKE_OP_TILE_BUFFER_RSRC, 127, "tile.buffer_rsrc", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_F16, 128, "tile.buffer_load_f16", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_VN_F16, 129, "tile.buffer_load_vN_f16", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_VN, 130, "tile.buffer_load_vN", false},
    {ROCKE_OP_TILE_BUFFER_STORE_F16, 131, "tile.buffer_store_f16", false},
    {ROCKE_OP_TILE_BUFFER_STORE_VN_F16, 132, "tile.buffer_store_vN_f16", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_BF16, 133, "tile.buffer_load_bf16", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_VN_BF16, 134, "tile.buffer_load_vN_bf16", false},
    {ROCKE_OP_TILE_BUFFER_STORE_BF16, 135, "tile.buffer_store_bf16", false},
    {ROCKE_OP_TILE_BUFFER_STORE_VN_BF16, 136, "tile.buffer_store_vN_bf16", false},
    {ROCKE_OP_TILE_BUFFER_STORE_F32, 137, "tile.buffer_store_f32", false},
    {ROCKE_OP_TILE_BUFFER_STORE_VN_F32, 138, "tile.buffer_store_vN_f32", false},
    {ROCKE_OP_TILE_MMA, 139, "tile.mma", false},
    {ROCKE_OP_TILE_REGISTER_P_FROM_QK_C, 140, "tile.register_p_from_qk_c", false},
    {ROCKE_OP_TILE_INLINE_ASM, 141, "tile.inline_asm", false},
    {ROCKE_OP_TILE_READFIRSTLANE, 142, "tile.readfirstlane", true},
    {ROCKE_OP_TILE_PIN_SGPR, 143, "tile.pin_sgpr", true},
    {ROCKE_OP_TILE_LANE_ID, 144, "tile.lane_id", true},
    {ROCKE_OP_TILE_WAVE_ALL, 145, "tile.wave_all", true},
    {ROCKE_OP_TILE_WAVE_ANY, 146, "tile.wave_any", true},
    {ROCKE_OP_TILE_WAVE_BALLOT, 147, "tile.wave_ballot", true},
    {ROCKE_OP_TILE_DS_BPERMUTE, 148, "tile.ds_bpermute", true},
    {ROCKE_OP_TILE_DS_BPERMUTE_B64, 149, "tile.ds_bpermute_b64", false},
    {ROCKE_OP_TILE_DS_SWIZZLE_XOR, 150, "tile.ds_swizzle_xor", true},
    {ROCKE_OP_TILE_DS_SWIZZLE, 151, "tile.ds_swizzle", true},
    {ROCKE_OP_TILE_MOV_DPP8, 152, "tile.mov_dpp8", true},
    {ROCKE_OP_TILE_QUAD_PERM, 153, "tile.quad_perm", true},
    {ROCKE_OP_TILE_DPP_XOR, 154, "tile.dpp_xor", true},
    {ROCKE_OP_TILE_WAVE_REDUCE, 155, "tile.wave_reduce", true},
    {ROCKE_OP_TILE_READLANE, 156, "tile.readlane", true},
    {ROCKE_OP_TILE_WRITELANE, 157, "tile.writelane", true},
    {ROCKE_OP_TILE_PERMLANE16, 158, "tile.permlane16", true},
    {ROCKE_OP_TILE_PERMLANE64, 159, "tile.permlane64", true},
    {ROCKE_OP_TILE_ALIGNBYTE, 160, "tile.alignbyte", true},
    {ROCKE_OP_TILE_S_WQM, 161, "tile.s_wqm", true},
    {ROCKE_OP_TILE_AV_LOAD_B128, 162, "tile.av_load_b128", true},
    {ROCKE_OP_TILE_AV_STORE_B128, 163, "tile.av_store_b128", false},
    {ROCKE_OP_TILE_S_ALLOC_VGPR, 164, "tile.s_alloc_vgpr", false},
    {ROCKE_OP_TILE_MOV_DPP, 165, "tile.mov_dpp", false},
    {ROCKE_OP_TILE_PERMLANE32_SWAP, 166, "tile.permlane32_swap", true},
    {ROCKE_OP_TILE_PERM_B32, 167, "tile.perm_b32", true},
    {ROCKE_OP_TILE_PERMLANEX16, 168, "tile.permlanex16", true},
    {ROCKE_OP_TILE_BYTE_PERM, 169, "tile.byte_perm", true},
    {ROCKE_OP_TILE_DS_READ_TR16_B64, 170, "tile.ds_read_tr16_b64", true},
    {ROCKE_OP_TILE_DS_READ_TR16_B128, 171, "tile.ds_read_tr16_b128", false},
    {ROCKE_OP_TILE_DS_READ_TR_B8, 172, "tile.ds_read_tr_b8", true},
    {ROCKE_OP_TILE_SYNC, 173, "tile.sync", false},
    {ROCKE_OP_TILE_SYNC_HALF_BLOCK, 174, "tile.sync_half_block", true},
    {ROCKE_OP_TILE_SYNC_LDS_ONLY, 175, "tile.sync_lds_only", false},
    {ROCKE_OP_TILE_S_BARRIER_BARE, 176, "tile.s_barrier_bare", false},
    {ROCKE_OP_TILE_S_WAITCNT, 177, "tile.s_waitcnt", false},
    {ROCKE_OP_TILE_S_WAIT_ASYNCCNT, 178, "tile.s_wait_asynccnt", false},
    {ROCKE_OP_TILE_S_WAIT_TENSORCNT, 179, "tile.s_wait_tensorcnt", false},
    {ROCKE_OP_TILE_S_BARRIER_SIGNAL, 180, "tile.s_barrier_signal", false},
    {ROCKE_OP_TILE_S_BARRIER_WAIT, 181, "tile.s_barrier_wait", false},
    {ROCKE_OP_TILE_S_BARRIER_INIT, 182, "tile.s_barrier_init", false},
    {ROCKE_OP_TILE_S_BARRIER_SIGNAL_VAR, 183, "tile.s_barrier_signal_var", false},
    {ROCKE_OP_TILE_S_BARRIER_JOIN, 184, "tile.s_barrier_join", false},
    {ROCKE_OP_TILE_S_WAKEUP_BARRIER, 185, "tile.s_wakeup_barrier", false},
    {ROCKE_OP_TILE_S_BARRIER_LEAVE, 186, "tile.s_barrier_leave", false},
    {ROCKE_OP_TILE_ASYNCMARK, 187, "tile.asyncmark", false},
    {ROCKE_OP_TILE_WAIT_ASYNCMARK, 188, "tile.wait_asyncmark", false},
    {ROCKE_OP_TILE_S_WAIT_EVENT, 189, "tile.s_wait_event", false},
    {ROCKE_OP_TILE_S_PREFETCH_INST, 190, "tile.s_prefetch_inst", false},
    {ROCKE_OP_TILE_S_SETREG, 191, "tile.s_setreg", false},
    {ROCKE_OP_TILE_S_PREFETCH_DATA, 192, "tile.s_prefetch_data", false},
    {ROCKE_OP_TILE_S_BUFFER_PREFETCH_DATA, 193, "tile.s_buffer_prefetch_data", false},
    {ROCKE_OP_TILE_GLOBAL_PREFETCH, 194, "tile.global_prefetch", false},
    {ROCKE_OP_TILE_FLAT_PREFETCH, 195, "tile.flat_prefetch", false},
    {ROCKE_OP_TILE_CLUSTER_BARRIER, 196, "tile.cluster_barrier", false},
    {ROCKE_OP_TILE_CLUSTER_LOAD, 197, "tile.cluster_load", false},
    {ROCKE_OP_TILE_CLUSTER_LOAD_ASYNC_TO_LDS, 198, "tile.cluster_load_async_to_lds", false},
    {ROCKE_OP_TILE_GLOBAL_PTR_TO_I64, 199, "tile.global_ptr_to_i64", true},
    {ROCKE_OP_TILE_S_SETPRIO, 200, "tile.s_setprio", false},
    {ROCKE_OP_TILE_IGLP_OPT, 201, "tile.iglp_opt", false},
    {ROCKE_OP_TILE_SCHED_BARRIER, 202, "tile.sched_barrier", false},
    {ROCKE_OP_TILE_SCHED_GROUP_BARRIER, 203, "tile.sched_group_barrier", false},
    {ROCKE_OP_TILE_EXEC_AND_SAVEEXEC, 204, "tile.exec_and_saveexec", false},
    {ROCKE_OP_TILE_EXEC_XOR, 205, "tile.exec_xor", false},
    {ROCKE_OP_TILE_EXEC_OR_SAVEEXEC, 206, "tile.exec_or_saveexec", false},
    {ROCKE_OP_TILE_EXEC_OR, 207, "tile.exec_or", false},
    {ROCKE_OP_SCF_FOR, 208, "scf.for", false},
    {ROCKE_OP_SCF_IF, 209, "scf.if", false},
    {ROCKE_OP_SCF_IF_ELSE, 210, "scf.if_else", false},
    {ROCKE_OP_SCF_YIELD, 211, "scf.yield", false},
    {ROCKE_OP_CF_RETURN, 212, "cf.return", false},
    {ROCKE_OP_MEMREF_FENCE, 213, "memref.fence", false},
    {ROCKE_OP_MEMREF_GLOBAL_FLAG_STORE, 214, "memref.global_flag_store", false},
    {ROCKE_OP_MEMREF_GLOBAL_FLAG_WAIT_EQ, 215, "memref.global_flag_wait_eq", false},
    {ROCKE_OP_VECTOR_UNDEF, 216, "vector.undef", false},
    {ROCKE_OP_TILE_BUFFER_LOAD_F16_D16, 217, "tile.buffer_load_f16_d16", false},
};

/* Each pinned opcode keeps its value, its name and its purity. */
int check_pinned_opcodes()
{
    int failures = 0;
    for(const PinnedOpcode& p : kPinned)
    {
        if(static_cast<int>(p.op) != p.value)
        {
            std::fprintf(stderr,
                         "opcode %s has value %d, but it must stay %d: append new opcodes at "
                         "the end of rocke_opcode_t instead of inserting them\n",
                         p.name,
                         static_cast<int>(p.op),
                         p.value);
            ++failures;
        }
        const char* name = rocke_opcode_name(p.op);
        if(std::strcmp(name, p.name) != 0)
        {
            std::fprintf(stderr,
                         "opcode %d is named \"%s\" but should be \"%s\": the name table in "
                         "core_types.cpp is out of step with the enum\n",
                         p.value,
                         name,
                         p.name);
            ++failures;
        }
        if(rocke_opcode_is_pure(p.op) != p.pure)
        {
            std::fprintf(stderr,
                         "opcode %s should be %s: the purity table in core_types.cpp is out of "
                         "step with the enum\n",
                         p.name,
                         p.pure ? "pure" : "impure");
            ++failures;
        }
    }
    return failures;
}

/* Every opcode, pinned or appended later, has a name that maps back to it. */
int check_every_name_round_trips()
{
    int failures = 0;
    for(int v = ROCKE_OP_INVALID + 1; v < ROCKE_OP__COUNT; ++v)
    {
        const rocke_opcode_t op = static_cast<rocke_opcode_t>(v);
        const char* name = rocke_opcode_name(op);
        if(name == nullptr || name[0] == '\0')
        {
            std::fprintf(stderr, "opcode %d has no name in core_types.cpp\n", v);
            ++failures;
        }
        else if(rocke_opcode_from_name(name) != op)
        {
            std::fprintf(stderr,
                         "opcode %d is named \"%s\", but that name maps to opcode %d\n",
                         v,
                         name,
                         static_cast<int>(rocke_opcode_from_name(name)));
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main()
{
    const int failures = check_pinned_opcodes() + check_every_name_round_trips();
    if(failures != 0)
    {
        std::fprintf(stderr, "%d opcode check(s) failed\n", failures);
        return 1;
    }
    std::printf("opcode values: %zu pinned opcodes and all %d opcode names OK\n",
                sizeof(kPinned) / sizeof(kPinned[0]),
                ROCKE_OP__COUNT - 1);
    return 0;
}
