/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.distribution.h -- C99 port of selected symbols from
 * rocke/helpers/distribution.py.
 *
 * SCOPE OF THIS PORT (this phase): make_static_tile_distribution.
 *
 * make_static_tile_distribution() is a *pure host-side* analysis pass: it takes
 * a TileDistributionEncoding (CK Tile's six-template-parameter static encoding)
 * and pre-computes, for every (x_dim, H-level) bucket, the P or Y contributor
 * that feeds that bucket. The result is a TileDistribution that the IR-emitting
 * methods (calculate_x, etc. -- NOT in scope here) consume. Because this
 * function emits NO IR, it makes zero rocke_b_* builder calls; the "byte-identical
 * op sequence" requirement is trivially met (the op sequence is empty) and the
 * fidelity requirement is on the produced contributor table, which is
 * reproduced bucket-for-bucket from the Python.
 *
 * The encoding data structure and its validation (Python __post_init__) are
 * reproduced here too because make_static_tile_distribution reads the encoding
 * and the Python encoding is never constructed without running __post_init__.
 * A rocke_make_tile_distribution_encoding() constructor runs that validation.
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena). Nothing is
 * freed individually; the arena bulk-frees the whole graph. This mirrors the
 * Python frozen-dataclass / GC lifetime exactly.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_DISTRIBUTION_H
#define ROCKE_HELPER_ROCKE_HELPERS_DISTRIBUTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.reduction.h" /* rocke_reduce_combine_t */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ encoding */

/* One hierarchical decomposition row: Hs[i] is a tuple of compile-time ints.
 * Mirrors TileDistributionEncoding.Hs[i]. */
typedef struct rocke_h_row
{
    const int* levels; /* arena-owned array, length == count          */
    int count;
} rocke_h_row_t;

/* One P dim's contribution sequence: a parallel (major, minor) pair list.
 * Mirrors a single entry of Ps2RHs_major / Ps2RHs_minor (both tuples). */
typedef struct rocke_p_seq
{
    const int* major; /* arena-owned, length == count                 */
    const int* minor; /* arena-owned, length == count                 */
    int count;
} rocke_p_seq_t;

/* Static encoding of how an X tile is split across P / Y / R spaces. Fields
 * mirror CK Tile's template parameters one-for-one (see the Python docstring).
 *
 * Major-index convention:
 *   major == 0          -> R (replication) bucket; minor indexes into Rs.
 *   major == 1..num_X   -> X dim (major-1); minor indexes into Hs[major-1].
 */
typedef struct rocke_tile_distribution_encoding
{
    const int* Rs; /* replication lengths (per R dim), arena-owned */
    int num_R;

    const rocke_h_row_t* Hs; /* hierarchical decomposition of each X dim     */
    int num_X;

    const rocke_p_seq_t* Ps; /* Ps2RHs_major/minor, one rocke_p_seq per P dim  */
    int num_P;

    const int* Ys_major; /* Ys2RHs_major, arena-owned, length == num_Y   */
    const int* Ys_minor; /* Ys2RHs_minor, arena-owned, length == num_Y   */
    int num_Y;
} rocke_tile_distribution_encoding_t;

/* Construct + validate an encoding (Python TileDistributionEncoding.__init__
 * including __post_init__). The arrays are copied into the builder arena, so the
 * caller may pass stack/temporary arrays. On a validation failure the builder's
 * sticky error is set (ROCKE_ERR_VALUE) and NULL is returned.
 *
 * Pass num_R==0 / Hs==NULL etc. with the matching count 0 for empty tuples.
 * For each P dim, Ps[i].major/minor are length Ps[i].count. */
rocke_tile_distribution_encoding_t* rocke_make_tile_distribution_encoding(rocke_ir_builder_t* b,
                                                                          const int* Rs,
                                                                          int num_R,
                                                                          const rocke_h_row_t* Hs,
                                                                          int num_X,
                                                                          const rocke_p_seq_t* Ps,
                                                                          int num_P,
                                                                          const int* Ys_major,
                                                                          const int* Ys_minor,
                                                                          int num_Y);

/* --------------------------------------------------------- contributor refs */

/* One mapping from an H bucket to either a P or a Y position.
 * Mirrors distribution.py _HBucketRef. kind is the char 'P' or 'Y'. */
typedef struct rocke_h_bucket_ref
{
    char kind; /* 'P' or 'Y' ('\0' => unmapped / None)        */
    int outer_idx; /* P dim index or Y dim index                  */
    int inner_idx; /* sub-position within Ps[outer_idx] (P only)   */
} rocke_h_bucket_ref_t;

/* --------------------------------------------------------- TileDistribution */

/* Runtime-emission counterpart of an encoding. Carries the encoding plus a
 * precomputed contributor map for each H bucket. Mirrors distribution.py
 * TileDistribution. The IR-emitting methods (calculate_x, ...) are NOT ported
 * in this phase.
 *
 * contributors is a ragged 2D table: contributors[x_dim] has
 * encoding->Hs[x_dim].count entries, one rocke_h_bucket_ref per H level. */
typedef struct rocke_tile_distribution
{
    const rocke_tile_distribution_encoding_t* encoding;
    const rocke_h_bucket_ref_t* const* contributors; /* [num_X][Hs[x].count]   */
} rocke_tile_distribution_t;

/* make_static_tile_distribution(encoding) analogue.
 *
 * Pre-computes the per-(x_dim, level) contributor lookup. R-bucket
 * contributors (major == 0) are skipped (they don't enter X). Ps cover their
 * entries first; Ys fill remaining holes. If, after assignment, any H bucket is
 * unmapped, the builder's sticky error is set (ROCKE_ERR_VALUE) and NULL is
 * returned -- matching the defensive ValueError in the Python.
 *
 * The returned struct and its contributor table are arena-owned. */
rocke_tile_distribution_t*
    rocke_make_static_tile_distribution(rocke_ir_builder_t* b,
                                        const rocke_tile_distribution_encoding_t* encoding);

/* TileDistribution.calculate_x(b, ys, ps) analogue.
 *
 * Returns the X-coord tuple for one (Y, P) position. ``ys`` is a per-Y-dim
 * array of SSA values (length encoding->num_Y). ``ps`` is a per-P-dim array of
 * sub-arrays; ps[i] lists the value(s) feeding P-dim i's H contributions, and
 * ps_counts[i] is the length of ps[i]. ``out_x`` receives one value per X dim
 * (length encoding->num_X). Returns 0 on a rank mismatch (Python ValueError) or
 * builder failure, 1 on success.
 *
 * The math (per X dim, innermost H level first):
 *   x_dim = sum_over_levels(contributor_value * stride_below)
 * emitted with the exact same SSA op order as the Python (b.add(0, c) for the
 * stride-1 level, then b.add(x, b.mul(c, stride)) for outer levels). */
int rocke_tile_distribution_calculate_x(rocke_ir_builder_t* b,
                                        const rocke_tile_distribution_t* dist,
                                        rocke_value_t* const* ys,
                                        int num_ys,
                                        rocke_value_t* const* const* ps,
                                        const int* ps_counts,
                                        int num_ps,
                                        rocke_value_t** out_x,
                                        int out_x_cap);

/* -------------------------------------- reduce distribution + block reduce */

/* make_reduce_tile_distribution_encoding(encoding, reduce_dim_xs).
 *
 * Faithful port of distribution.py::make_reduce_tile_distribution_encoding (the
 * CK Tile detail::make_reduce_tile_distribution_encoding_impl). Collapses the X
 * dims listed in ``reduce_dim_xs`` (length ``num_reduce``): every H level of a
 * reduced X dim not consumed by a Y dim becomes a new R (replication) level; Y
 * dims targeting a reduced X dim are dropped; surviving X dims renumber densely
 * and every P / surviving-Y contribution is re-pointed. Returns a NEW
 * arena-owned encoding (validated via rocke_make_tile_distribution_encoding), or
 * NULL with the builder's sticky error set. */
rocke_tile_distribution_encoding_t*
    rocke_make_reduce_tile_distribution_encoding(rocke_ir_builder_t* b,
                                                 const rocke_tile_distribution_encoding_t* encoding,
                                                 const int* reduce_dim_xs,
                                                 int num_reduce);

/* StaticDistributedTensor analogue: a thread-local f32 (or dtype) register
 * container. Mirrors distribution.py StaticDistributedTensor. ``storage`` holds
 * ``num_storage`` SSA scalars (one per Y-space cell, row-major); the reduce
 * build seeds storage[0] = acc and reads total = storage[0]. */
typedef struct rocke_static_distributed_tensor
{
    const rocke_tile_distribution_t* distribution;
    const rocke_type_t* dtype;
    rocke_value_t** storage;
    int num_storage;
} rocke_static_distributed_tensor_t;

/* make_static_distributed_tensor(distribution, dtype).
 *
 * Returns an uninitialised container with
 * ``num_storage == distribution.num_elements_per_thread`` (product of the
 * encoding's Y_lengths; >=1), every slot NULL. Arena-owned. NULL + sticky error
 * on failure. */
rocke_static_distributed_tensor_t*
    rocke_make_static_distributed_tensor(rocke_ir_builder_t* b,
                                         const rocke_tile_distribution_t* distribution,
                                         const rocke_type_t* dtype);

/* block_tile_reduce_sync(b, reduced, combine, lds_buf, tid, wave_size).
 *
 * Distribution-driven block reduction (CK Tile BlockReduce2d). ``reduced`` is a
 * StaticDistributedTensor over a reduce distribution. Stage 1: warp-local XOR
 * butterfly over the lane-P-owned R levels (in place over every storage slot).
 * Stage 2: cross-warp LDS round-trip over the warp-P-owned R levels (skipped
 * when the reduce is single-warp). ``combine`` must be ROCKE_REDUCE_SUM or
 * ROCKE_REDUCE_MAX (Python raises ValueError otherwise). ``lds_buf`` / ``tid`` are
 * required only when the cross-warp stage runs. Emits the exact same SSA op
 * sequence as the Python. */
void rocke_block_tile_reduce_sync(rocke_ir_builder_t* b,
                                  rocke_static_distributed_tensor_t* reduced,
                                  rocke_reduce_combine_t combine,
                                  rocke_value_t* lds_buf,
                                  rocke_value_t* tid,
                                  int wave_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_DISTRIBUTION_H */
