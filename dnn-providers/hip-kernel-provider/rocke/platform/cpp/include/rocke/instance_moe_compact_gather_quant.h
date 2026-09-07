/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Public C99 surface for the C-engine mirror of
 * rocke/instances/common/moe_compact_gather_quant.py.
 *
 * Python (moe_compact_gather_quant.py)  C99 (this header)
 * ------------------------------------  ----------------------------------------
 * MoeCompactGatherQuantSpec             rocke_moe_compact_gather_quant_spec_t
 * is_valid_spec                         rocke_moe_compact_gather_quant_is_valid_spec
 * build_moe_compact_gather_quant        rocke_build_moe_compact_gather_quant
 * moe_compact_gather_quant_grid         rocke_moe_compact_gather_quant_grid
 * moe_compact_gather_quant_signature    rocke_moe_compact_gather_quant_signature
 */
#ifndef ROCKE_INSTANCE_MOE_COMPACT_GATHER_QUANT_H
#define ROCKE_INSTANCE_MOE_COMPACT_GATHER_QUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_MOE_COMPACT_GATHER_QUANT_DEFAULT_ARCH "gfx950"

/* Mirror of the frozen Python dataclass. The three problem dimensions are
 * required in Python and are initialised to zero by the C default constructor;
 * callers set them before validation or build. String fields are borrowed. */
typedef struct rocke_moe_compact_gather_quant_spec
{
    int tokens;
    int hidden;
    int max_blocks;
    int tile_m;
    const char* input_dtype;
    int block_size;
    int vec;
    int wave_size;
    const char* name;
} rocke_moe_compact_gather_quant_spec_t;

rocke_moe_compact_gather_quant_spec_t rocke_moe_compact_gather_quant_spec_default(void);

int rocke_moe_compact_gather_quant_hidden_blocks(const rocke_moe_compact_gather_quant_spec_t* spec);
int rocke_moe_compact_gather_quant_qvecs_per_block(
    const rocke_moe_compact_gather_quant_spec_t* spec);
int rocke_moe_compact_gather_quant_passes_per_thread(
    const rocke_moe_compact_gather_quant_spec_t* spec);
int rocke_moe_compact_gather_quant_output_rows(const rocke_moe_compact_gather_quant_spec_t* spec);

rocke_status_t rocke_moe_compact_gather_quant_kernel_name(
    const rocke_moe_compact_gather_quant_spec_t* spec, char* out, size_t out_cap);

/* Mirrors is_valid_spec(spec, arch). `arch == NULL` selects gfx950. */
bool rocke_moe_compact_gather_quant_is_valid_spec(const rocke_moe_compact_gather_quant_spec_t* spec,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap);

/* `b` must already be initialised with the spec's kernel name. */
rocke_kernel_def_t* rocke_build_moe_compact_gather_quant(
    rocke_ir_builder_t* b, const rocke_moe_compact_gather_quant_spec_t* spec, const char* arch);

/* Initialises `b`, builds the kernel, and leaves builder ownership with the
 * caller. Free it with rocke_ir_builder_free(). */
rocke_kernel_def_t* rocke_build_moe_compact_gather_quant_new(
    rocke_ir_builder_t* b, const rocke_moe_compact_gather_quant_spec_t* spec, const char* arch);

rocke_status_t
    rocke_moe_compact_gather_quant_grid(const rocke_moe_compact_gather_quant_spec_t* spec,
                                        int out[3]);

/* The returned signature array and strings are owned by `arena`. */
rocke_status_t
    rocke_moe_compact_gather_quant_signature(rocke_arena_t* arena,
                                             const rocke_moe_compact_gather_quant_spec_t* spec,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count);

/* On success `*out_ll` is malloc-owned and must be freed by the caller. */
rocke_status_t
    rocke_moe_compact_gather_quant_lower_to_llvm(const rocke_moe_compact_gather_quant_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_COMPACT_GATHER_QUANT_H */
