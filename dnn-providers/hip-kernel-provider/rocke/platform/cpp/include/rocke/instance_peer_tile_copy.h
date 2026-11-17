/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Masked strided global tile copies; HIP, not this builder, owns peer mappings.
 * Python (instances/common/peer_tile_copy.py) -> C ABI (this header)
 * PeerTileCopySpec                          -> rocke_peer_tile_copy_spec_t
 * is_valid_spec                            ->
 * rocke_peer_tile_copy_is_valid_spec build_peer_tile_copy ->
 * rocke_build_peer_tile_copy_new peer_tile_copy_grid                      ->
 * rocke_peer_tile_copy_grid
 *
 * Strings in the spec are borrowed through the build call. The caller owns
 * builders and frees them with rocke_ir_builder_free; lowered text is malloc'd.
 * Source/destination allocations must be disjoint, cover declared view spans,
 * and outlive all local/peer uses. Runtime strides are positive element
 * strides; destination rows must not overlap. Coordinates and byte spans must
 * fit i64.
 */
#ifndef ROCKE_INSTANCE_PEER_TILE_COPY_H
#define ROCKE_INSTANCE_PEER_TILE_COPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct rocke_peer_tile_copy_spec
    {
        int tile_rows;
        int tile_cols;
        int block_size;
        const char* dtype;
        const char* name;
    } rocke_peer_tile_copy_spec_t;

    rocke_peer_tile_copy_spec_t rocke_peer_tile_copy_spec_default(void);
    rocke_status_t rocke_peer_tile_copy_kernel_name(
        const rocke_peer_tile_copy_spec_t* spec, char* out, size_t out_cap);
    bool rocke_peer_tile_copy_is_valid_spec(
        const rocke_peer_tile_copy_spec_t* spec, const char* arch, char* reason, size_t reason_cap);
    rocke_kernel_def_t* rocke_build_peer_tile_copy_new(
        rocke_ir_builder_t* b, const rocke_peer_tile_copy_spec_t* spec, const char* arch);
    rocke_status_t rocke_peer_tile_copy_grid(
        int64_t rows, int64_t cols, const rocke_peer_tile_copy_spec_t* spec, int out_grid[3]);
    rocke_status_t rocke_peer_tile_copy_lower_to_llvm(const rocke_peer_tile_copy_spec_t* spec,
        const char* arch,
        rocke_llvm_flavor_t flavor,
        char** out_ll,
        char* err,
        size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_INSTANCE_PEER_TILE_COPY_H */
