// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_mx_gemm.c -- C99 port of rocke/instances/common/mx_gemm.py
 * (build_mx_gemm + is_valid_spec + grid + spec helpers).
 *
 * The build entry mirrors the Python build_mx_gemm one op at a time so a
 * reviewer can diff line by line: same param order/attrs, same wave tile, same
 * scaled-MFMA steps and scaled-group loops, same store epilogue. Every builder
 * call is sequenced in the Python's left-to-right evaluation order: each
 * emission consumes one SSA value-counter tick, so the order is load-bearing
 * for byte-identical value numbering.
 *
 * Builder primitives are rocke/ir.h's rocke_b_*; the MX scale helpers, quant
 * type map, MFMA atom, lane decode and the scaled-MFMA matrix layout are the
 * already-ported C helpers.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/instance_mx_gemm.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.mx_scale.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */
#include "rocke/lower_llvm.h"
#include "rocke/mfma_scale_internal.h" /* rocke_mfma_matrix_layout */

/* Python _SCALE_LOADS_FIRST: sched_barrier mask, every instruction class may
 * be scheduled across the barrier except VMEM (0x10) and VMEM read (0x20). */
#define ROCKE_MX_GEMM_SCALE_LOADS_FIRST 0x7CF
/* Python _WAVE_TILE / _WAVE_SIZE: the output tile of one wave and of its
 * MFMAs, and the lanes of a wave. */
#define ROCKE_MX_GEMM_WAVE_TILE 32
#define ROCKE_MX_GEMM_WAVE_SIZE 64
/* Python _SCALED_MFMA_K: K elements per scaled MFMA (two scale groups). */
#define ROCKE_MX_GEMM_SCALED_MFMA_K 64
/* Python _A_SCALE_ROWS_PER_LOAD: A scale codes per load, four consecutive
 * rows of one group (one dword). */
#define ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD 4
/* Python _I32_MAX. */
#define ROCKE_MX_GEMM_I32_MAX 0x7FFFFFFFLL

/* ===================================================================== *
 *  mx_gemm_element_format(mantissa_dtype, arch)
 *
 * Python _MFMA_ELEMENT_FORMATS: the fp8 / bf8 encoding the MFMA of each
 * target reads, the FNUZ encodings on gfx942 and the OCP ones on gfx950.
 * ===================================================================== */
const rocke_mx_element_format_t* rocke_mx_gemm_element_format(const char* mantissa_dtype,
                                                              const char* arch,
                                                              char* err,
                                                              size_t err_cap)
{
    bool fp8;
    bool bf8;

    if(mantissa_dtype == NULL || arch == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "mx_gemm_element_format: NULL argument");
        }
        return NULL;
    }
    fp8 = strcmp(mantissa_dtype, "fp8e4m3") == 0;
    bf8 = strcmp(mantissa_dtype, "bf8e5m2") == 0;
    if(strcmp(arch, "gfx942") == 0 && (fp8 || bf8))
    {
        return fp8 ? &ROCKE_FP8_E4M3_FNUZ : &ROCKE_BF8_E5M2_FNUZ;
    }
    if(strcmp(arch, "gfx950") == 0 && (fp8 || bf8))
    {
        return fp8 ? &ROCKE_FP8_E4M3_OCP : &ROCKE_BF8_E5M2_OCP;
    }
    if(err != NULL && err_cap > 0)
    {
        snprintf(err,
                 err_cap,
                 "mx_gemm: no %s element format known for %s; the MX GEMM reads "
                 "fp8e4m3 / bf8e5m2 on gfx942 and gfx950",
                 mantissa_dtype,
                 arch);
    }
    return NULL;
}

/* ===================================================================== *
 *  rocke_mx_gemm_spec_default -- Python MxGemmSpec dataclass defaults.
 * ===================================================================== */
rocke_mx_gemm_spec_t rocke_mx_gemm_spec_default(void)
{
    rocke_mx_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.block_tile_m = 0; /* caller must set (Python: required field, no default) */
    s.block_tile_n = 0; /* caller must set (Python: required field, no default) */
    s.mantissa_dtype = "fp8e4m3";
    s.group_k = 32;
    s.name = "rocke_mx_gemm";
    return s;
}

/* MxGemmSpec.waves: (block_tile_m // 32) * (block_tile_n // 32). */
long long rocke_mx_gemm_waves(const rocke_mx_gemm_spec_t* spec)
{
    return (long long)(spec->block_tile_m / ROCKE_MX_GEMM_WAVE_TILE)
           * (spec->block_tile_n / ROCKE_MX_GEMM_WAVE_TILE);
}

/* MxGemmSpec.block_size: 64 * waves. */
long long rocke_mx_gemm_block_size(const rocke_mx_gemm_spec_t* spec)
{
    return ROCKE_MX_GEMM_WAVE_SIZE * rocke_mx_gemm_waves(spec);
}

/* MxGemmSpec.group_atom: MfmaAtom.fp8_32x32x16() if mantissa_dtype ==
 * "fp8e4m3" else MfmaAtom.bf8_32x32x16(). */
const struct rocke_mfma_atom* rocke_mx_gemm_group_atom(const rocke_mx_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") == 0)
    {
        return rocke_mfma_atom("fp8e4m3", 32, 32, 16);
    }
    return rocke_mfma_atom("bf8e5m2", 32, 32, 16);
}

/* ===================================================================== *
 *  MxGemmSpec.kernel_name()
 *
 *     kernel_name_join(name, f"M{M}N{N}K{K}", mantissa_dtype,
 *                      f"gk{group_k}", f"t{block_tile_m}x{block_tile_n}")
 * ===================================================================== */
rocke_status_t
    rocke_mx_gemm_kernel_name(const rocke_mx_gemm_spec_t* spec, char* out, size_t out_cap)
{
    char shape[64];
    char gk[32];
    char tile[48];
    const char* parts[4];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(shape, sizeof(shape), "M%dN%dK%d", spec->M, spec->N, spec->K);
    snprintf(gk, sizeof(gk), "gk%d", spec->group_k);
    snprintf(tile, sizeof(tile), "t%dx%d", spec->block_tile_m, spec->block_tile_n);

    parts[0] = shape;
    parts[1] = spec->mantissa_dtype;
    parts[2] = gk;
    parts[3] = tile;

    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* Python _scaled_mfma(spec, arch): the target's E8M0-scaled 32x32x64 MFMA for
 * the spec's mantissas, or NULL when the target has none (gfx942). */
static const rocke_mmaop_t* rocke_mxg_scaled_mfma(const rocke_mx_gemm_spec_t* spec,
                                                  const char* arch)
{
    rocke_mma_scale_filter_t scales;
    scales.a_scale_dtype = "e8m0";
    scales.b_scale_dtype = "e8m0";
    scales.scale_block_k = (rocke_mma_scale_block_k_t)spec->group_k;
    return rocke_archtarget_op_for_shape(rocke_archtarget_from_gfx(arch),
                                         "mfma_scaled",
                                         spec->mantissa_dtype,
                                         spec->mantissa_dtype,
                                         "fp32",
                                         ROCKE_MX_GEMM_WAVE_TILE,
                                         ROCKE_MX_GEMM_WAVE_TILE,
                                         ROCKE_MX_GEMM_SCALED_MFMA_K,
                                         &scales);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason)
 *
 * Mirrors validate_arch_and_block_size (arch resolve + block-size cap) plus the
 * MX-specific checks, in the Python's order and with its texts.
 * ===================================================================== */
static void rocke_mxg_setreason(char* reason, size_t cap, const char* msg)
{
    rocke_spec_set_reason(reason, cap, msg);
}

/* Python _largest_byte_offsets(spec): the largest byte offset the kernel
 * computes into A, B, AScale, BScale and C, in that order. A wave whose tile
 * starts inside C computes the offsets of its whole 32x32 tile, so the rows and
 * columns past the edge count too, up to the next multiple of 32. Called only
 * after M, N and K are known positive and K a multiple of group_k. */
static void rocke_mxg_largest_byte_offsets(const rocke_mx_gemm_spec_t* spec, long long out[5])
{
    const long long M = spec->M;
    const long long N = spec->N;
    const long long K = spec->K;
    const long long groups = K / spec->group_k;
    const long long tile = ROCKE_MX_GEMM_WAVE_TILE;
    const long long padded_rows = (M + tile - 1) / tile * tile;
    const long long padded_cols = (N + tile - 1) / tile * tile;
    out[0] = padded_rows * K - 1;
    out[1] = (K - 1) * N + padded_cols - 1;
    out[2] = (groups - 1) * M + padded_rows - 1;
    out[3] = (groups - 1) * N + padded_cols - 1;
    out[4] = 4 * ((padded_rows - 1) * N + padded_cols) - 1;
}

bool rocke_mx_gemm_is_valid_spec(const rocke_mx_gemm_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    const rocke_archtarget_t* target;
    long long block_size;
    char buf[192];

    if(spec == NULL)
    {
        rocke_mxg_setreason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* for name in ("block_tile_m", "block_tile_n"): a positive multiple of 32 */
    {
        const char* names[2] = {"block_tile_m", "block_tile_n"};
        const int tiles[2] = {spec->block_tile_m, spec->block_tile_n};
        for(int i = 0; i < 2; ++i)
        {
            if(tiles[i] <= 0 || tiles[i] % ROCKE_MX_GEMM_WAVE_TILE)
            {
                snprintf(buf,
                         sizeof(buf),
                         "%s must be a positive multiple of %d, got %d",
                         names[i],
                         ROCKE_MX_GEMM_WAVE_TILE,
                         tiles[i]);
                rocke_mxg_setreason(reason, reason_cap, buf);
                return false;
            }
        }
    }

    /* validate_arch_and_block_size: ArchTarget.from_gfx(arch) + block cap. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s", arch);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    block_size = rocke_mx_gemm_block_size(spec);
    if(block_size > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %lld > %d (hardware cap) on %s",
                 block_size,
                 rocke_archtarget_max_threads_per_block(target),
                 arch);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }

    if(strcmp(spec->mantissa_dtype, "fp8e4m3") != 0 && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unsupported mantissa_dtype '%s'; the MX GEMM reads fp8e4m3 / bf8e5m2 "
                 "(fp4 / fp6 are not supported yet)",
                 spec->mantissa_dtype);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->group_k != 32)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MX spec requires group_k = 32 (shared exponent per 32 mantissa "
                 "elements); got group_k=%d",
                 spec->group_k);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->M <= 0 || spec->N <= 0 || spec->K <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "M, N and K must be positive, got %d, %d, %d",
                 spec->M,
                 spec->N,
                 spec->K);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->K % spec->group_k)
    {
        snprintf(
            buf, sizeof(buf), "K (%d) must be divisible by group_k (%d)", spec->K, spec->group_k);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->M % ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD)
    {
        snprintf(buf,
                 sizeof(buf),
                 "M (%d) must be a multiple of %d: a lane reads the A scale codes of %d "
                 "consecutive rows as one dword",
                 spec->M,
                 ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD,
                 ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    /* for tensor, offset in _largest_byte_offsets(spec).items(): <= _I32_MAX */
    {
        const char* tensors[5] = {"A", "B", "AScale", "BScale", "C"};
        long long offsets[5];
        rocke_mxg_largest_byte_offsets(spec, offsets);
        for(int i = 0; i < 5; ++i)
        {
            if(offsets[i] > ROCKE_MX_GEMM_I32_MAX)
            {
                snprintf(buf,
                         sizeof(buf),
                         "the largest byte offset into %s is %lld, expected at most %lld: the "
                         "kernel computes every buffer offset in i32",
                         tensors[i],
                         offsets[i],
                         ROCKE_MX_GEMM_I32_MAX);
                rocke_mxg_setreason(reason, reason_cap, buf);
                return false;
            }
        }
    }

    rocke_mxg_setreason(reason, reason_cap, "ok");
    return true;
}

/* Python _WaveTile: what every part of the kernel reads to compute one
 * wave's output tile -- its origin (Python _WaveOrigin: the lane and the first
 * row and column of the wave's 32x32 tile), the IR type of the mantissas and
 * the buffer descriptors of the five tensors. */
typedef struct rocke_mxg_tile
{
    rocke_value_t* lane;
    rocke_value_t* row_origin;
    rocke_value_t* col_origin;
    const rocke_type_t* mantissa_type;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* a_scale_rsrc;
    rocke_value_t* b_scale_rsrc;
    rocke_value_t* c_rsrc;
} rocke_mxg_tile_t;

/* Python _wave_origin(b, spec): the wave's 32x32 output tile origin and lane,
 * written to tile->lane, tile->row_origin and tile->col_origin. */
static void rocke_mxg_wave_origin(rocke_ir_builder_t* b,
                                  const rocke_mx_gemm_spec_t* spec,
                                  rocke_mxg_tile_t* tile)
{
    const int waves_n = spec->block_tile_n / ROCKE_MX_GEMM_WAVE_TILE;
    /* tid = b.thread_id_x() */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    /* lane = b.mod(tid, b.const_i32(_WAVE_SIZE)) */
    tile->lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, ROCKE_MX_GEMM_WAVE_SIZE));
    /* wave = b.readfirstlane(b.div(tid, b.const_i32(_WAVE_SIZE))) */
    rocke_value_t* wave = rocke_b_readfirstlane(
        b, rocke_b_div(b, tid, rocke_b_const_i32(b, ROCKE_MX_GEMM_WAVE_SIZE)));
    /* m0 = b.add(b.mul(b.block_id_y(), b.const_i32(spec.block_tile_m)),
     *            b.mul(b.div(wave, b.const_i32(waves_n)), b.const_i32(_WAVE_TILE))) */
    {
        rocke_value_t* by = rocke_b_block_id_y(b);
        rocke_value_t* block_rows = rocke_b_mul(b, by, rocke_b_const_i32(b, spec->block_tile_m));
        rocke_value_t* wave_row = rocke_b_div(b, wave, rocke_b_const_i32(b, waves_n));
        rocke_value_t* wave_rows
            = rocke_b_mul(b, wave_row, rocke_b_const_i32(b, ROCKE_MX_GEMM_WAVE_TILE));
        tile->row_origin = rocke_b_add(b, block_rows, wave_rows);
    }
    /* n0 = b.add(b.mul(b.block_id_x(), b.const_i32(spec.block_tile_n)),
     *            b.mul(b.mod(wave, b.const_i32(waves_n)), b.const_i32(_WAVE_TILE))) */
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* block_cols = rocke_b_mul(b, bx, rocke_b_const_i32(b, spec->block_tile_n));
        rocke_value_t* wave_col = rocke_b_mod(b, wave, rocke_b_const_i32(b, waves_n));
        rocke_value_t* wave_cols
            = rocke_b_mul(b, wave_col, rocke_b_const_i32(b, ROCKE_MX_GEMM_WAVE_TILE));
        tile->col_origin = rocke_b_add(b, block_cols, wave_cols);
    }
}

/* Python _byte(b, rsrc, voffset, soffset, dtype): one 1-byte element through
 * a buffer descriptor. */
static rocke_value_t* rocke_mxg_byte(rocke_ir_builder_t* b,
                                     rocke_value_t* rsrc,
                                     rocke_value_t* voffset,
                                     rocke_value_t* soffset,
                                     const rocke_type_t* dtype)
{
    return rocke_b_vec_extract(b, rocke_b_buffer_load_vN(b, rsrc, voffset, soffset, dtype, 1), 0);
}

/* Python _sum_scaled_groups(b, spec, tile, acc, first_group, end_group, tag):
 * acc plus the scaled dot products of scale groups first_group .. end_group-1,
 * each group added one iteration late. Returns NULL on a builder error. */
static rocke_value_t* rocke_mxg_sum_scaled_groups(rocke_ir_builder_t* b,
                                                  const rocke_mx_gemm_spec_t* spec,
                                                  const rocke_mxg_tile_t* tile,
                                                  rocke_value_t* acc,
                                                  int first_group,
                                                  int end_group,
                                                  const char* tag)
{
    const rocke_mfma_atom_t* atom = rocke_mx_gemm_group_atom(spec);
    const int N = spec->N;
    const int K = spec->K;
    const int c_per_lane = atom->c_per_lane;
    std::vector<rocke_value_t*> b_offsets;
    std::vector<rocke_value_t*> a_scale_offsets;
    std::vector<std::string> names;
    std::vector<rocke_iter_arg_t> iter_args;

    /* lanes = decode_mfma_lanes(b, atom, tile.lane) */
    rocke_lane_decode_t lanes = rocke_decode_mfma_lanes(b, atom, tile->lane);
    /* a_row = b.add(tile.row_origin, lanes.m_in_atom); b_col = b.add(tile.col_origin, lanes.n_in_atom) */
    rocke_value_t* a_row = rocke_b_add(b, tile->row_origin, lanes.m_in_atom);
    rocke_value_t* b_col = rocke_b_add(b, tile->col_origin, lanes.n_in_atom);
    /* k_lane = b.mul(lanes.k_blk, b.const_i32(atom.a_per_lane)) */
    rocke_value_t* k_lane = rocke_b_mul(b, lanes.k_blk, rocke_b_const_i32(b, atom->a_per_lane));
    /* a_offset = b.add(b.mul(a_row, b.const_i32(K)), k_lane) */
    rocke_value_t* a_offset
        = rocke_b_add(b, rocke_b_mul(b, a_row, rocke_b_const_i32(b, K)), k_lane);
    /* b_offsets = [b.add(b.mul(b.add(k_lane, b.const_i32(j)), b.const_i32(N)), b_col)] */
    for(int j = 0; j < atom->b_per_lane; ++j)
    {
        rocke_value_t* k = rocke_b_add(b, k_lane, rocke_b_const_i32(b, j));
        rocke_value_t* row = rocke_b_mul(b, k, rocke_b_const_i32(b, N));
        b_offsets.push_back(rocke_b_add(b, row, b_col));
    }
    /* a_scale_offsets = [b.add(tile.row_origin, atom.lane_to_output(b, tile.lane, q)[0])
     *                    for q in range(0, c_per_lane, _A_SCALE_ROWS_PER_LOAD)]:
     * slots 4q .. 4q + 3 of a lane are four consecutive rows of one column. */
    for(int q = 0; q < c_per_lane; q += ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD)
    {
        rocke_value_t* r = NULL;
        rocke_value_t* c = NULL;
        rocke_mfma_atom_lane_to_output(b, atom, tile->lane, q, &r, &c);
        a_scale_offsets.push_back(rocke_b_add(b, tile->row_origin, r));
    }
    const int atoms_per_group = spec->group_k / atom->k;
    /* zero_code = b.trunc(b.const_i32(0), I8) */
    rocke_value_t* zero_code = rocke_b_trunc(b, rocke_b_const_i32(b, 0), rocke_i8());

    /* loop = b.scf_for_iter(first_group, end_group, 1, [
     *     (f"{tag}_acc", acc), (f"{tag}_prev_dot", atom.zero_acc(b)),
     *     (f"{tag}_prev_a_code{i}", zero_code) x c_per_lane,
     *     (f"{tag}_prev_b_code", zero_code)], iv_name=f"{tag}_group") */
    rocke_for_t loop;
    {
        rocke_value_t* lb = rocke_b_const_i32(b, first_group);
        rocke_value_t* ub = rocke_b_const_i32(b, end_group);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        std::string t(tag);
        names.push_back(t + "_acc");
        names.push_back(t + "_prev_dot");
        for(int i = 0; i < c_per_lane; ++i)
        {
            names.push_back(t + "_prev_a_code" + std::to_string(i));
        }
        names.push_back(t + "_prev_b_code");
        std::string iv = t + "_group";
        iter_args.resize(names.size());
        iter_args[0].name = names[0].c_str();
        iter_args[0].init = acc;
        iter_args[1].name = names[1].c_str();
        iter_args[1].init = rocke_b_zero_vec_f32(b, c_per_lane); /* atom.zero_acc(b) */
        for(size_t k = 2; k < names.size(); ++k)
        {
            iter_args[k].name = names[k].c_str();
            iter_args[k].init = zero_code;
        }
        loop = rocke_b_scf_for_iter(
            b, lb, ub, step, iter_args.data(), (int)iter_args.size(), iv.c_str(), false, true);
    }
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* kg = loop.iv;
        rocke_value_t* acc_in = loop.iter_vars[0];
        rocke_value_t* prev_dot = loop.iter_vars[1];
        std::vector<rocke_value_t*> prev_a(loop.iter_vars + 2, loop.iter_vars + 2 + c_per_lane);
        std::vector<rocke_value_t*> prev_b(c_per_lane, loop.iter_vars[2 + c_per_lane]);
        std::vector<rocke_value_t*> a_codes;
        std::vector<std::pair<rocke_value_t*, rocke_value_t*>> operands;

        /* group_codes = b.mul(kg, b.const_i32(spec.M))
         * for o in a_scale_offsets:
         *     rows = b.buffer_load_vN(tile.a_scale_rsrc, o, group_codes, I8, _A_SCALE_ROWS_PER_LOAD)
         *     a_codes += [b.vec_extract(rows, r) for r in range(_A_SCALE_ROWS_PER_LOAD)] */
        rocke_value_t* group_codes = rocke_b_mul(b, kg, rocke_b_const_i32(b, spec->M));
        for(rocke_value_t* o : a_scale_offsets)
        {
            rocke_value_t* rows = rocke_b_buffer_load_vN(b,
                                                         tile->a_scale_rsrc,
                                                         o,
                                                         group_codes,
                                                         rocke_i8(),
                                                         ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD);
            for(int r = 0; r < ROCKE_MX_GEMM_A_SCALE_ROWS_PER_LOAD; ++r)
            {
                a_codes.push_back(rocke_b_vec_extract(b, rows, r));
            }
        }
        /* b_code = _byte(b, tile.b_scale_rsrc, b_col, b.mul(kg, b.const_i32(N)), I8) */
        rocke_value_t* b_code = rocke_mxg_byte(
            b, tile->b_scale_rsrc, b_col, rocke_b_mul(b, kg, rocke_b_const_i32(b, N)), rocke_i8());
        /* b.sched_barrier(_SCALE_LOADS_FIRST) */
        rocke_b_sched_barrier(b, ROCKE_MX_GEMM_SCALE_LOADS_FIRST);
        for(int t = 0; t < atoms_per_group; ++t)
        {
            /* k_tile = b.add(b.mul(kg, b.const_i32(spec.group_k)),
             *                b.const_i32(t * atom.k)) */
            rocke_value_t* k_group = rocke_b_mul(b, kg, rocke_b_const_i32(b, spec->group_k));
            rocke_value_t* k_tile = rocke_b_add(b, k_group, rocke_b_const_i32(b, t * atom->k));
            /* a = b.buffer_load_vN(tile.a_rsrc, a_offset, k_tile, tile.mantissa_type, atom.a_per_lane) */
            rocke_value_t* a = rocke_b_buffer_load_vN(
                b, tile->a_rsrc, a_offset, k_tile, tile->mantissa_type, atom->a_per_lane);
            /* row_step = b.mul(k_tile, b.const_i32(N)) */
            rocke_value_t* row_step = rocke_b_mul(b, k_tile, rocke_b_const_i32(b, N));
            /* bv = b.vec_pack([_byte(b, tile.b_rsrc, o, row_step, tile.mantissa_type) for o in b_offsets], tile.mantissa_type) */
            std::vector<rocke_value_t*> elems;
            for(rocke_value_t* o : b_offsets)
            {
                elems.push_back(rocke_mxg_byte(b, tile->b_rsrc, o, row_step, tile->mantissa_type));
            }
            rocke_value_t* bv
                = rocke_b_vec_pack(b, elems.data(), (int)elems.size(), tile->mantissa_type);
            operands.emplace_back(a, bv);
        }
        /* acc_out = add_mx_scaled_dot(b, acc_in, prev_dot, prev_codes[:-1],
         *                             [prev_codes[-1]] * atom.c_per_lane) */
        rocke_value_t* acc_out = rocke_add_mx_scaled_dot(
            b, acc_in, prev_dot, prev_a.data(), prev_a.size(), prev_b.data(), prev_b.size());
        /* dot = atom.zero_acc(b); dot = atom.emit(b, a, bv, dot) for each pair */
        rocke_value_t* dot = rocke_b_zero_vec_f32(b, c_per_lane);
        for(const auto& [a, bv] : operands)
        {
            dot = rocke_b_mma(b, atom->name, a, bv, dot, NULL, 0);
        }
        /* b.scf_yield(acc_out, dot, *a_codes, b_code) */
        std::vector<rocke_value_t*> yields;
        yields.push_back(acc_out);
        yields.push_back(dot);
        yields.insert(yields.end(), a_codes.begin(), a_codes.end());
        yields.push_back(b_code);
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);
    if(!rocke_i_live(b) || loop.op == NULL || loop.op->num_results < 3 + c_per_lane)
    {
        return NULL;
    }
    /* acc_last, last_dot, *last_codes = loop.results
     * return add_mx_scaled_dot(b, acc_last, last_dot, last_codes[:-1],
     *                          [last_codes[-1]] * atom.c_per_lane) */
    {
        rocke_value_t** results = loop.op->results;
        std::vector<rocke_value_t*> last_a(results + 2, results + 2 + c_per_lane);
        std::vector<rocke_value_t*> last_b(c_per_lane, results[2 + c_per_lane]);
        return rocke_add_mx_scaled_dot(
            b, results[0], results[1], last_a.data(), last_a.size(), last_b.data(), last_b.size());
    }
}

/* Python _scaled_mfma_steps(b, spec, tile, mma): the accumulator after K // 64
 * scaled MFMAs and the largest scale code each lane passed to them, written
 * to *out_acc / *out_largest_code. Returns false on a builder error. */
static bool rocke_mxg_scaled_mfma_steps(rocke_ir_builder_t* b,
                                        const rocke_mx_gemm_spec_t* spec,
                                        const rocke_mxg_tile_t* tile,
                                        const rocke_mmaop_t* mma,
                                        rocke_value_t** out_acc,
                                        rocke_value_t** out_largest_code)
{
    const int N = spec->N;
    const int K = spec->K;
    /* layout = matrix_layout(spec.mantissa_dtype) */
    const rocke_matrix_fragment_layout_t layout = rocke_mfma_matrix_layout(spec->mantissa_dtype);
    const int chunk = layout.chunk_elements;
    const int chunk_stride = chunk * layout.lane_groups;
    const int chunks_per_lane = rocke_matrix_fragment_chunks_per_lane(&layout);
    std::vector<rocke_value_t*> b_offsets;

    /* row = b.mod(tile.lane, b.const_i32(layout.lanes_per_group)) */
    rocke_value_t* row = rocke_b_mod(b, tile->lane, rocke_b_const_i32(b, layout.lanes_per_group));
    /* group = b.div(tile.lane, b.const_i32(layout.lanes_per_group)) */
    rocke_value_t* group = rocke_b_div(b, tile->lane, rocke_b_const_i32(b, layout.lanes_per_group));
    rocke_value_t* a_row = rocke_b_add(b, tile->row_origin, row);
    rocke_value_t* b_col = rocke_b_add(b, tile->col_origin, row);
    /* k_lane = b.mul(group, b.const_i32(chunk)) */
    rocke_value_t* k_lane = rocke_b_mul(b, group, rocke_b_const_i32(b, chunk));
    /* a_offset = b.add(b.mul(a_row, b.const_i32(K)), k_lane) */
    rocke_value_t* a_offset
        = rocke_b_add(b, rocke_b_mul(b, a_row, rocke_b_const_i32(b, K)), k_lane);
    for(int j = 0; j < chunk; ++j)
    {
        rocke_value_t* k = rocke_b_add(b, k_lane, rocke_b_const_i32(b, j));
        rocke_value_t* row_off = rocke_b_mul(b, k, rocke_b_const_i32(b, N));
        b_offsets.push_back(rocke_b_add(b, row_off, b_col));
    }
    /* a_scale_offset = b.add(b.mul(group, b.const_i32(spec.M)), a_row) */
    rocke_value_t* a_scale_offset
        = rocke_b_add(b, rocke_b_mul(b, group, rocke_b_const_i32(b, spec->M)), a_row);
    /* b_scale_offset = b.add(b.mul(group, b.const_i32(N)), b_col) */
    rocke_value_t* b_scale_offset
        = rocke_b_add(b, rocke_b_mul(b, group, rocke_b_const_i32(b, N)), b_col);
    const int groups_per_step = ROCKE_MX_GEMM_SCALED_MFMA_K / spec->group_k;
    const rocke_type_t* words = rocke_vector_type(b, rocke_i32(), 8);
    /* zero = b.const_i32(0) */
    rocke_value_t* zero = rocke_b_const_i32(b, 0);

    /* loop = b.scf_for_iter(zero, b.const_i32(K // 64), b.const_i32(1),
     *     [("scaled_acc", b.zero_vec_f32(mma.c_frag_len)), ("largest_code", zero)],
     *     iv_name="step") */
    rocke_for_t loop;
    {
        rocke_value_t* ub = rocke_b_const_i32(b, K / ROCKE_MX_GEMM_SCALED_MFMA_K);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        rocke_iter_arg_t iter_args[2];
        iter_args[0].name = "scaled_acc";
        iter_args[0].init = rocke_b_zero_vec_f32(b, mma->srcs[2].frag_len);
        iter_args[1].name = "largest_code";
        iter_args[1].init = zero;
        loop = rocke_b_scf_for_iter(b, zero, ub, step, iter_args, 2, "step", false, true);
    }
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* st = loop.iv;
        rocke_value_t* acc = loop.iter_vars[0];
        rocke_value_t* largest_code = loop.iter_vars[1];
        std::vector<rocke_value_t*> a_chunks;
        std::vector<rocke_value_t*> b_elements;

        /* k_step = b.mul(step, b.const_i32(_SCALED_MFMA_K)) */
        rocke_value_t* k_step
            = rocke_b_mul(b, st, rocke_b_const_i32(b, ROCKE_MX_GEMM_SCALED_MFMA_K));
        /* group_step = b.mul(step, b.const_i32(groups_per_step)) */
        rocke_value_t* group_step = rocke_b_mul(b, st, rocke_b_const_i32(b, groups_per_step));
        /* a_scale = b.zext(_byte(b, tile.a_scale_rsrc, a_scale_offset,
         *                        b.mul(group_step, b.const_i32(spec.M)), I8), I32) */
        rocke_value_t* a_scale_rows = rocke_b_mul(b, group_step, rocke_b_const_i32(b, spec->M));
        rocke_value_t* a_scale = rocke_b_zext(
            b,
            rocke_mxg_byte(b, tile->a_scale_rsrc, a_scale_offset, a_scale_rows, rocke_i8()),
            rocke_i32());
        /* b_scale = b.zext(_byte(b, tile.b_scale_rsrc, b_scale_offset,
         *                        b.mul(group_step, b.const_i32(N)), I8), I32) */
        rocke_value_t* b_scale_rows = rocke_b_mul(b, group_step, rocke_b_const_i32(b, N));
        rocke_value_t* b_scale = rocke_b_zext(
            b,
            rocke_mxg_byte(b, tile->b_scale_rsrc, b_scale_offset, b_scale_rows, rocke_i8()),
            rocke_i32());
        for(int c = 0; c < chunks_per_lane; ++c)
        {
            /* k_chunk = b.add(k_step, b.const_i32(c * chunk_stride)) */
            rocke_value_t* k_chunk = rocke_b_add(b, k_step, rocke_b_const_i32(b, c * chunk_stride));
            /* a_chunks.append(b.buffer_load_vN(tile.a_rsrc, a_offset, k_chunk, tile.mantissa_type, chunk)) */
            a_chunks.push_back(rocke_b_buffer_load_vN(
                b, tile->a_rsrc, a_offset, k_chunk, tile->mantissa_type, chunk));
            /* row_step = b.mul(k_chunk, b.const_i32(N)) */
            rocke_value_t* row_step = rocke_b_mul(b, k_chunk, rocke_b_const_i32(b, N));
            for(rocke_value_t* o : b_offsets)
            {
                b_elements.push_back(
                    rocke_mxg_byte(b, tile->b_rsrc, o, row_step, tile->mantissa_type));
            }
        }
        /* a_operand = b.bitcast(b.vec_concat(*a_chunks), words): matrix_layout
         * gives every fp8 / bf8 operand (the only mantissas is_valid_spec admits)
         * two chunks per lane, the two vectors vec_concat takes. */
        assert(a_chunks.size() == 2);
        rocke_value_t* a
            = rocke_b_bitcast(b, rocke_b_vec_concat(b, a_chunks[0], a_chunks[1]), words);
        /* bv = b.bitcast(b.vec_pack(b_elements, tile.mantissa_type), words) */
        rocke_value_t* bv = rocke_b_bitcast(
            b,
            rocke_b_vec_pack(b, b_elements.data(), (int)b_elements.size(), tile->mantissa_type),
            words);
        /* largest_code = b.smax(largest_code, b.smax(a_scale, b_scale)) */
        largest_code = rocke_b_smax(b, largest_code, rocke_b_smax(b, a_scale, b_scale));
        /* b.scf_yield(b.mma(mma, a, bv, acc, a_scale, b_scale), largest_code) */
        rocke_value_t* scales[2] = {a_scale, b_scale};
        rocke_value_t* yields[2];
        yields[0] = rocke_b_mma(b, mma->op_id, a, bv, acc, scales, 2);
        yields[1] = largest_code;
        rocke_b_scf_yield(b, yields, 2);
    }
    rocke_b_region_leave(b);
    if(!rocke_i_live(b) || loop.op == NULL || loop.op->num_results < 2)
    {
        return false;
    }
    *out_acc = loop.op->results[0];
    *out_largest_code = loop.op->results[1];
    return true;
}

/* Python _store_tile(b, spec, tile, acc): the lane's outputs through a buffer
 * descriptor over C; a row past M lands past the end of C, and when N is not a
 * multiple of 32 a column past N is moved past the end too. */
static void rocke_mxg_store_tile(rocke_ir_builder_t* b,
                                 const rocke_mx_gemm_spec_t* spec,
                                 const rocke_mxg_tile_t* tile,
                                 rocke_value_t* acc)
{
    const rocke_mfma_atom_t* atom = rocke_mx_gemm_group_atom(spec);
    /* past_end = b.const_i32(4 * spec.M * spec.N) if spec.N % _WAVE_TILE else None */
    rocke_value_t* past_end = (spec->N % ROCKE_MX_GEMM_WAVE_TILE)
                                  ? rocke_b_const_i32(b, 4LL * spec->M * spec->N)
                                  : NULL;
    /* zero = b.const_i32(0) */
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    for(int i = 0; i < atom->c_per_lane; ++i)
    {
        rocke_value_t* r = NULL;
        rocke_value_t* c = NULL;
        rocke_mfma_atom_lane_to_output(b, atom, tile->lane, i, &r, &c);
        /* col = b.add(tile.col_origin, c) */
        rocke_value_t* col = rocke_b_add(b, tile->col_origin, c);
        /* element = b.add(b.mul(b.add(tile.row_origin, r), b.const_i32(spec.N)), col) */
        rocke_value_t* out_row = rocke_b_add(b, tile->row_origin, r);
        rocke_value_t* row_start = rocke_b_mul(b, out_row, rocke_b_const_i32(b, spec->N));
        rocke_value_t* element = rocke_b_add(b, row_start, col);
        /* offset = b.mul(element, b.const_i32(4)) */
        rocke_value_t* offset = rocke_b_mul(b, element, rocke_b_const_i32(b, 4));
        if(past_end != NULL)
        {
            /* offset = b.select(b.cmp_lt(col, b.const_i32(spec.N)), offset, past_end) */
            rocke_value_t* inside = rocke_b_cmp_lt(b, col, rocke_b_const_i32(b, spec->N));
            offset = rocke_b_select(b, inside, offset, past_end);
        }
        /* b.buffer_store_vN(tile.c_rsrc, offset, zero,
         *                   b.vec_pack([b.vec_extract(acc, i)], F32)) */
        rocke_value_t* value = rocke_b_vec_extract(b, acc, i);
        rocke_value_t* packed = rocke_b_vec_pack(b, &value, 1, rocke_f32());
        rocke_b_buffer_store_vN(b, tile->c_rsrc, offset, zero, packed);
    }
}

/* Python build_mx_gemm's tile_body(): the buffer descriptors, the scaled or
 * scaled-group path and the stores of one wave tile. */
static void rocke_mxg_tile_body(rocke_ir_builder_t* b,
                                const rocke_mx_gemm_spec_t* spec,
                                const char* arch,
                                const rocke_mmaop_t* mma,
                                rocke_value_t* const* params,
                                rocke_mxg_tile_t* tile)
{
    const int group_count = spec->K / spec->group_k;
    /* tile.a_rsrc = b.buffer_rsrc(A, b.const_i32(spec.M * spec.K)) ... tile.c_rsrc */
    tile->a_rsrc
        = rocke_b_buffer_rsrc(b, params[0], rocke_b_const_i32(b, (int64_t)spec->M * spec->K));
    tile->b_rsrc
        = rocke_b_buffer_rsrc(b, params[2], rocke_b_const_i32(b, (int64_t)spec->K * spec->N));
    tile->a_scale_rsrc
        = rocke_b_buffer_rsrc(b, params[1], rocke_b_const_i32(b, (int64_t)group_count * spec->M));
    tile->b_scale_rsrc
        = rocke_b_buffer_rsrc(b, params[3], rocke_b_const_i32(b, (int64_t)group_count * spec->N));
    tile->c_rsrc = rocke_b_buffer_rsrc(b, params[4], rocke_b_const_i32(b, 4LL * spec->M * spec->N));
    /* zero_acc = spec.group_atom.zero_acc(b) */
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, rocke_mx_gemm_group_atom(spec)->c_per_lane);
    if(mma == NULL)
    {
        rocke_value_t* acc
            = rocke_mxg_sum_scaled_groups(b, spec, tile, zero_acc, 0, group_count, "all");
        if(acc != NULL)
        {
            rocke_mxg_store_tile(b, spec, tile, acc);
        }
        return;
    }
    rocke_value_t* acc = NULL;
    rocke_value_t* largest_code = NULL;
    if(!rocke_mxg_scaled_mfma_steps(b, spec, tile, mma, &acc, &largest_code))
    {
        return;
    }
    /* scaled_groups = (spec.K // _SCALED_MFMA_K) * (_SCALED_MFMA_K // spec.group_k) */
    const int scaled_groups
        = (spec->K / ROCKE_MX_GEMM_SCALED_MFMA_K) * (ROCKE_MX_GEMM_SCALED_MFMA_K / spec->group_k);
    if(scaled_groups < group_count)
    {
        acc = rocke_mxg_sum_scaled_groups(b, spec, tile, acc, scaled_groups, group_count, "tail");
        if(acc == NULL)
        {
            return;
        }
    }
    rocke_mxg_store_tile(b, spec, tile, acc);
    /* element = mx_gemm_element_format(spec.mantissa_dtype, arch)
     * safe_code = largest_safe_mx_scale_code(element, element, _SCALED_MFMA_K) */
    char err[192];
    const rocke_mx_element_format_t* element
        = rocke_mx_gemm_element_format(spec->mantissa_dtype, arch, err, sizeof(err));
    if(element == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err);
        return;
    }
    int safe_code = 0;
    if(rocke_largest_safe_mx_scale_code(
           element, element, ROCKE_MX_GEMM_SCALED_MFMA_K, &safe_code, err, sizeof(err))
       != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", err);
        return;
    }
    /* unsafe = b.wave_any(b.select(b.cmp_gt(largest_code, b.const_i32(safe_code)),
     *                              b.const_i32(1), b.const_i32(0))) */
    rocke_value_t* above = rocke_b_cmp_gt(b, largest_code, rocke_b_const_i32(b, safe_code));
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* zero_flag = rocke_b_const_i32(b, 0);
    rocke_value_t* unsafe = rocke_b_wave_any(b, rocke_b_select(b, above, one, zero_flag));
    /* with b.scf_if(b.cmp_ne(unsafe, b.const_i32(0))):
     *     _store_tile(b, spec, tile, _sum_scaled_groups(b, spec, tile, zero_acc, 0, group_count, "again")) */
    rocke_if_t again = rocke_b_scf_if(b, rocke_b_cmp_ne(b, unsafe, rocke_b_const_i32(b, 0)));
    rocke_b_region_enter(b, again.then_region);
    {
        rocke_value_t* redo
            = rocke_mxg_sum_scaled_groups(b, spec, tile, zero_acc, 0, group_count, "again");
        if(redo != NULL)
        {
            rocke_mxg_store_tile(b, spec, tile, redo);
        }
    }
    rocke_b_region_leave(b);
}

/* ===================================================================== *
 *  build_mx_gemm(spec, arch)
 *
 * One op at a time mirroring the Python. The builder `b` is assumed already
 * initialised by the caller with spec.kernel_name() (public-header contract).
 * ===================================================================== */
rocke_kernel_def_t*
    rocke_build_mx_gemm(rocke_ir_builder_t* b, const rocke_mx_gemm_spec_t* spec, const char* arch)
{
    char reason[192];
    const rocke_type_t* mantissa_ty;
    const rocke_mfma_atom_t* atom;
    const rocke_mmaop_t* mma;
    rocke_value_t* params[5];
    rocke_mxg_tile_t tile;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch=arch); if not ok: raise ValueError */
    if(!rocke_mx_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid mx_gemm spec for %s: %s", arch, reason);
    }
    /* validate_mfma_atom_in_catalog(spec.group_atom, arch, where="mx_gemm") */
    atom = rocke_mx_gemm_group_atom(spec);
    if(atom == NULL || rocke_validate_mfma_atom_in_catalog(b, atom, arch, "mx_gemm") != ROCKE_OK)
    {
        return NULL;
    }
    /* mma = _scaled_mfma(spec, arch) */
    mma = rocke_mxg_scaled_mfma(spec, arch);

    /* mty = quant_ir_type(spec.mantissa_dtype) */
    mantissa_ty = rocke_b_quant_ir_type(b, spec->mantissa_dtype);
    if(mantissa_ty == NULL)
    {
        return NULL;
    }
    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", rocke_mx_gemm_block_size(spec));

    /* ---- kernel params (mirror Python order/attrs) ---- */
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_mant = rocke_ptr_type(b, mantissa_ty, "global");
        const rocke_type_t* ptr_i8 = rocke_ptr_type(b, rocke_i8(), "global");
        const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

        /* A = b.param("A", PtrType(mty, "global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        params[0] = rocke_b_param(b, "A", ptr_mant, &opts);

        /* AScale = b.param("AScale", PtrType(I8, "global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 1;
        opts.align_set = true;
        params[1] = rocke_b_param(b, "AScale", ptr_i8, &opts);

        /* B = b.param("B", PtrType(mty, "global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        params[2] = rocke_b_param(b, "B", ptr_mant, &opts);

        /* BScale = b.param("BScale", PtrType(I8, "global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 1;
        opts.align_set = true;
        params[3] = rocke_b_param(b, "BScale", ptr_i8, &opts);

        /* C = b.param("C", PtrType(F32, "global"), writeonly=True, align=4) */
        memset(&opts, 0, sizeof(opts));
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 4;
        opts.align_set = true;
        params[4] = rocke_b_param(b, "C", ptr_f32, &opts);

        /* b.param("M", I32); b.param("N", I32); b.param("K", I32) */
        (void)rocke_b_param(b, "M", rocke_i32(), NULL);
        (void)rocke_b_param(b, "N", rocke_i32(), NULL);
        (void)rocke_b_param(b, "K", rocke_i32(), NULL);
    }

    /* origin = _wave_origin(b, spec); the tile's other fields as tile_body()
     * builds _WaveTile */
    memset(&tile, 0, sizeof(tile));
    rocke_mxg_wave_origin(b, spec, &tile);
    tile.mantissa_type = mantissa_ty;

    /* A wave whose tile starts past the edge of C does nothing. */
    if(spec->M % spec->block_tile_m || spec->N % spec->block_tile_n)
    {
        /* inside = b.land(b.cmp_lt(tile.row_origin, b.const_i32(spec.M)),
         *                 b.cmp_lt(tile.col_origin, b.const_i32(spec.N))) */
        rocke_value_t* rows_in = rocke_b_cmp_lt(b, tile.row_origin, rocke_b_const_i32(b, spec->M));
        rocke_value_t* cols_in = rocke_b_cmp_lt(b, tile.col_origin, rocke_b_const_i32(b, spec->N));
        rocke_if_t inside = rocke_b_scf_if(b, rocke_b_land(b, rows_in, cols_in));
        rocke_b_region_enter(b, inside.then_region);
        rocke_mxg_tile_body(b, spec, arch, mma, params, &tile);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_mxg_tile_body(b, spec, arch, mma, params, &tile);
    }

    /* b.ret() */
    rocke_b_ret(b);
    return rocke_i_live(b) ? b->kernel : NULL;
}

/* ===================================================================== *
 *  rocke_build_mx_gemm_new -- init the builder with spec.kernel_name(), then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_mx_gemm_new(rocke_ir_builder_t* b,
                                            const rocke_mx_gemm_spec_t* spec,
                                            const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_mx_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_mx_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_mx_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_mx_gemm_lower_to_llvm(const rocke_mx_gemm_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            rocke_mxg_setreason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_mx_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_mx_gemm failed";
            }
            rocke_mxg_setreason(err, err_cap, m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  mx_gemm_grid(spec) -> (n_tiles, m_tiles, 1)
 * ===================================================================== */
void rocke_mx_gemm_grid(const rocke_mx_gemm_spec_t* spec, int* out_gx, int* out_gy, int* out_gz)
{
    int n_tiles;
    int m_tiles;
    if(spec == NULL || spec->block_tile_m <= 0 || spec->block_tile_n <= 0)
    {
        /* No workgroup tile: an empty grid, so a launch fails instead of
         * reading unset dimensions. */
        n_tiles = 0;
        m_tiles = 0;
    }
    else
    {
        n_tiles = (spec->N + spec->block_tile_n - 1) / spec->block_tile_n;
        m_tiles = (spec->M + spec->block_tile_m - 1) / spec->block_tile_m;
    }
    if(out_gx != NULL)
    {
        *out_gx = n_tiles;
    }
    if(out_gy != NULL)
    {
        *out_gy = m_tiles;
    }
    if(out_gz != NULL)
    {
        *out_gz = 1;
    }
}
