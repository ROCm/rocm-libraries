/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.scan.h -- C99 port of selected symbols from
 * rocke/helpers/scan.py.
 *
 * Cooperative block-wide scan + histogram helpers used by CK Tile's MoE-sort
 * and bucket-style kernels. This phase ports two symbols:
 *
 *   - lds_zero_i32             cooperative zero of an (length,) i32 LDS buffer
 *   - block_exclusive_scan_i32 in-place exclusive prefix-sum over an i32 LDS
 *                              buffer (Hillis-Steele tree, log2(N) round-trips)
 *
 * The remaining scan.py symbols (block_histogram_i32, block_two_level_scan_i32)
 * are NOT ported in this phase.
 *
 * Both helpers operate on an LDS allocation the caller owns and assume i32 keys
 * and counts. Each helper reproduces its Python counterpart's rocke_b_* builder-
 * call sequence byte-faithfully (same ops, same order, same operands).
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). These
 * helpers emit IR via the rocke_b_* builder API and bind only to rocke/ir.h's
 * public surface. On a bad argument (length <= 0, or length > block_size for the
 * scan) the helper sets the builder's sticky error and returns without emitting
 * (mirroring the Python ValueError).
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_SCAN_H
#define ROCKE_HELPER_ROCKE_HELPERS_SCAN_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* lds_zero_i32(b, lds_buf, tid=, block_size=, length=) analogue.
 *
 * Cooperatively zero an (length,) i32 LDS allocation. Each lane writes
 * ceil(length / block_size) slots; tail lanes are guarded with an in-bounds
 * predicate so out-of-range slots are skipped. A b.sync() is issued at the end
 * so subsequent atomic adds against the same buffer see the cleared state.
 *
 * Sets the builder sticky error and emits nothing when length <= 0. */
void rocke_lds_zero_i32(
    rocke_ir_builder_t* b, rocke_value_t* lds_buf, rocke_value_t* tid, int block_size, int length);

/* block_exclusive_scan_i32(b, lds_buf, tid=, block_size=, length=) analogue.
 *
 * In-place exclusive prefix-sum over lds_buf[0..length) (i32). Inclusive
 * Hillis-Steele scan (log2(length) LDS round-trips, each barrier-guarded)
 * followed by a one-position right shift to make the scan exclusive.
 *
 * Requires length <= block_size (so every lane handles at most one slot). Sets
 * the builder sticky error and emits nothing when length <= 0 or
 * length > block_size. */
void rocke_block_exclusive_scan_i32(
    rocke_ir_builder_t* b, rocke_value_t* lds_buf, rocke_value_t* tid, int block_size, int length);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_SCAN_H */
