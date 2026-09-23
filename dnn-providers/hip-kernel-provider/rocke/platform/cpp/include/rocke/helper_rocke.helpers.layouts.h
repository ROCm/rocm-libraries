/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.layouts.h -- C99 port of selected symbols from
 * rocke/helpers/layouts.py.
 *
 * SCOPE OF THIS PORT (this phase): TransposeLdsReader (and its bound view
 * _BoundTransposeLdsReader).
 *
 * TransposeLdsReader mirrors CK Tile's TransposeLDSLayout<M, K, B> lane-address
 * formulas for the ds_read_b64_tr_b16 path on M=16 MFMA tiles (see
 * composablekernel/include/ck_tile/ops/direct_convolution/utils/
 * transpose_lds_layout.hpp). It is a frozen dataclass of two compile-time ints
 * (K, M=16) plus a couple of pure derived integers, and one bind() that
 * materializes the lane-derived SSA constants once per kernel. The bound view
 * exposes row() (IR-emitting) and a cached col SSA value.
 *
 * The bind()/row() methods DO emit IR (b.div/b.mod/b.mul/b.add/b.const_i32), so
 * the "byte-identical op sequence" requirement applies: the C path issues the
 * rocke_b_* builder calls in the exact same order and with the exact same operands
 * as the Python.
 *
 * Lifetime: the bound view is arena-owned (rocke_ir_builder_t.arena). Nothing is
 * freed individually; the arena bulk-frees the whole graph. This mirrors the
 * Python frozen-dataclass / GC lifetime exactly.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_LAYOUTS_H
#define ROCKE_HELPER_ROCKE_HELPERS_LAYOUTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------- TransposeLdsReader */

/* Unbound, compile-time-only descriptor. Mirrors layouts.py TransposeLdsReader
 * (frozen dataclass: K, M=16). Both fields are plain host ints; the struct emits
 * no IR until bind(). */
typedef struct rocke_transpose_lds_reader
{
    int K;
    int M; /* defaults to 16; use rocke_transpose_lds_reader_make for the default */
} rocke_transpose_lds_reader_t;

/* Construct with the Python default M == 16 (TransposeLdsReader(K)). */
rocke_transpose_lds_reader_t rocke_transpose_lds_reader_make(int K);

/* @property k_lanes -> K // 4. (K_L = K / (64 / M) = K / 4 for M == 16.) */
int rocke_transpose_lds_reader_k_lanes(const rocke_transpose_lds_reader_t* r);

/* reads_per_k_iter(k_step) -> max(1, k_step // K). */
int rocke_transpose_lds_reader_reads_per_k_iter(const rocke_transpose_lds_reader_t* r, int k_step);

/* -------------------------------------------------- _BoundTransposeLdsReader */

/* SSA values produced by TransposeLdsReader.bind(). Mirrors layouts.py
 * _BoundTransposeLdsReader. Arena-owned. */
typedef struct rocke_bound_transpose_lds_reader
{
    rocke_transpose_lds_reader_t reader; /* by value, like the frozen dataclass */
    rocke_value_t* lane;
    rocke_value_t* lane_div_16; /* lane / 16                              */
    rocke_value_t* lane_div_4_mod_4; /* (lane / 4) % 4                         */
    rocke_value_t* col; /* (lane % 4) * 4                         */
} rocke_bound_transpose_lds_reader_t;

/* TransposeLdsReader.bind(b, lane).
 *
 * Materializes the lane-derived constants once per kernel and returns a small
 * arena-owned bound view. Emits, in this exact order:
 *   lane_div_16      = b.div(lane, b.const_i32(16))
 *   lane_div_4_mod_4 = b.mod(b.div(lane, b.const_i32(4)), b.const_i32(4))
 *   col              = b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4))
 *
 * Returns NULL with the builder's sticky error set on allocation/builder
 * failure. */
rocke_bound_transpose_lds_reader_t* rocke_transpose_lds_reader_bind(
    rocke_ir_builder_t* b, const rocke_transpose_lds_reader_t* r, rocke_value_t* lane);

/* _BoundTransposeLdsReader.row(b, k_offset=, read=0).
 *
 * The row LDS index for one ds_read_b64_tr_b16 call. Emits, in this exact order:
 *   b.add(
 *     b.const_i32(k_offset + read * 4),
 *     b.add(
 *       b.mul(self.lane_div_16, b.const_i32(self.reader.k_lanes)),
 *       self.lane_div_4_mod_4))
 *
 * Returns NULL with the builder's sticky error set on builder failure. */
rocke_value_t* rocke_bound_transpose_lds_reader_row(rocke_ir_builder_t* b,
                                                    const rocke_bound_transpose_lds_reader_t* self,
                                                    int k_offset,
                                                    int read);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_LAYOUTS_H */
