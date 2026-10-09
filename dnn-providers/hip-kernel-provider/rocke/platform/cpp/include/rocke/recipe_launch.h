/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/recipe_launch.h -- everything a JIT caller needs to actually LAUNCH the
 * kernel a recipe produces: its name, its argument layout, and its geometry.
 *
 * Why this exists
 * ---------------
 * The pure-C path used to stop at LLVM IR text. A caller could take a CBOR
 * bundle to a correct .ll with no Python in the process (rocke/online.h), and
 * from there to a HSACO with libamd_comgr, and then be stuck: it had a compiled
 * kernel and no idea what to launch it with. The grid was not in the bundle at
 * all. It lived in host Python -- expressions like (n + tile_n - 1) / tile_n
 * sitting in a dispatch function -- so the last step of the chain was the one
 * step that could not be taken without an interpreter.
 *
 * Geometry is a function of the shape, and the recipe language already has a
 * way to say that: an intexpr over the spec axes, the same thing the recipe uses
 * to compute every loop bound and offset it emits. So the grid is carried in the
 * recipe and evaluated at replay by the same evaluator, rather than being
 * reimplemented per client:
 *
 *     "launch": {
 *       "grid":  [{"div": [{"add": [{"spec": "N"}, 63]}, 64]}, 1, 1],
 *       "block": [256, 1, 1],
 *       "lds_bytes": 0
 *     }
 *
 * That means geometry cannot drift from the kernel it launches -- it ships in
 * the same artifact, is derived from the same axes, and is covered by the same
 * guard and ABI checks as everything else in the recipe.
 *
 * The argument layout is NOT carried in the recipe, because it does not need to
 * be: the recipe's own `param` instructions already declare it, in order. This
 * reports what the recipe actually declared, so it cannot disagree with the
 * kernel that was built from it.
 *
 * Kernarg offsets
 * ---------------
 * `offset` follows the AMDGPU natural-alignment rule -- each argument sits at an
 * offset aligned to its own size, 8 for pointers and i64, 4 for i32 and f32 --
 * and the caller should copy each value to plan->args[i].offset rather than
 * assuming fields are packed back to back. Those differ the moment a signature
 * mixes widths: (ptr, i32, ptr) is 24 bytes with the trailing pointer at 16, not
 * 20 bytes at 12. Python learned this the hard way (see runtime/packing.py); the
 * offsets are computed here so a C caller does not have to learn it again, and a
 * test pins the two against each other.
 *
 * Typical use, continuing from the guard check in rocke/recipe_guard.h:
 *
 *     rocke_launch_plan_t* plan;
 *     if(rocke_bundle_plan_launch_cbor(blob, blob_len, key, "gfx950",
 *                                      ints, n_ints, strs, n_strs,
 *                                      &plan, err, sizeof err) != ROCKE_OK)
 *         return hard_error(err);
 *
 *     rocke_launch_dims_t grid, block;
 *     unsigned lds;
 *     if(!rocke_launch_plan_geometry(plan, &grid, &block, &lds))
 *         return fallback();          // recipe carries no geometry; see below
 *
 *     std::vector<char> kernarg(rocke_launch_plan_kernarg_size(plan));
 *     for(int i = 0; i < rocke_launch_plan_num_args(plan); i++)
 *     {
 *         const rocke_arg_desc_t* a = rocke_launch_plan_arg(plan, i);
 *         memcpy(kernarg.data() + a->offset, value_for(a->name), a->size);
 *     }
 *     hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z,
 *                           lds, stream, nullptr, extra_with(kernarg));
 *     rocke_launch_plan_free(plan);
 *
 * Cost: building a plan replays the recipe, which is the same work as lowering
 * it (around 1ms for a large kernel). A caller that wants both the .ll and the
 * plan pays that twice. That is deliberate -- keeping the two calls independent
 * is worth more than the millisecond, against a JIT compile that costs orders of
 * magnitude more and is cached afterwards.
 */
#ifndef ROCKE_RECIPE_LAUNCH_H
#define ROCKE_RECIPE_LAUNCH_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/recipe_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_launch_dims
{
    unsigned x, y, z;
} rocke_launch_dims_t;

/* The kinds a kernel argument can have. Pointers are called out separately
 * because they are the ones the caller must substitute a device address for;
 * the scalar kinds are distinguished so a caller knows whether the four bytes
 * it is about to write are an integer or a float. */
typedef enum rocke_arg_kind
{
    ROCKE_ARG_POINTER = 0,
    ROCKE_ARG_I32,
    ROCKE_ARG_I64,
    ROCKE_ARG_F32
} rocke_arg_kind_t;

typedef struct rocke_arg_desc
{
    const char* name; /* as declared by the recipe's `param`          */
    const char* type_name; /* canonical rocke type, e.g. "ptr<f16, global>"*/
    rocke_arg_kind_t kind;
    unsigned size; /* bytes to write                               */
    unsigned offset; /* byte offset into the kernarg buffer          */
} rocke_arg_desc_t;

/* Owns every string and descriptor it hands out; all of them die with it. */
typedef struct rocke_launch_plan rocke_launch_plan_t;

/* Replay a recipe far enough to describe its launch. Returns ROCKE_ERR_VALUE if
 * the recipe's guard refuses this spec -- the same refusal the VM would give,
 * because planning a launch for a shape the kernel will not serve is not a
 * meaningful question. */
rocke_status_t rocke_recipe_plan_launch_cbor(const unsigned char* data,
                                             size_t len,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             rocke_launch_plan_t** out_plan,
                                             char* err,
                                             size_t err_cap);

/* As above, for one recipe out of a bundle. ROCKE_ERR_KEY if no such entry. */
rocke_status_t rocke_bundle_plan_launch_cbor(const unsigned char* data,
                                             size_t len,
                                             const char* key,
                                             const char* arch,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             rocke_launch_plan_t** out_plan,
                                             char* err,
                                             size_t err_cap);

/* The formatted kernel name -- what to pass to hipModuleGetFunction. Never
 * NULL for a plan that was returned successfully. */
const char* rocke_launch_plan_kernel_name(const rocke_launch_plan_t* plan);

/* Geometry, or false if this recipe does not carry a "launch" block.
 *
 * Absence is reported rather than defaulted, for the same reason
 * ROCKE_GUARD_ABSENT is: a recipe recorded before geometry existed, or one whose
 * generator did not supply it, is not the same as a kernel that wants a 1x1x1
 * grid, and quietly returning zeros or ones would turn a missing-metadata bug
 * into a wrong-answer bug at the point where it is hardest to notice.
 *
 * `lds` is DYNAMIC shared memory, the argument hipModuleLaunchKernel takes.
 * Static LDS is already accounted for inside the HSACO; do not add them. Any
 * out-param may be NULL. */
bool rocke_launch_plan_geometry(const rocke_launch_plan_t* plan,
                                rocke_launch_dims_t* out_grid,
                                rocke_launch_dims_t* out_block,
                                unsigned* out_lds_bytes);

/* The kernel's arguments, in declaration order. */
int rocke_launch_plan_num_args(const rocke_launch_plan_t* plan);
const rocke_arg_desc_t* rocke_launch_plan_arg(const rocke_launch_plan_t* plan, int i);

/* Bytes to allocate for the kernarg buffer: the end of the last argument.
 *
 * NOT rounded up to the widest alignment. That is a real convention, and the
 * AMDGPU metadata's kernarg segment size does round up, but the launch path
 * this mirrors does not: runtime/packing.py packs a (ptr,ptr,ptr,i32,i32,i32)
 * GEMM as 36 bytes rather than 40, and that is what has been running. Reporting
 * 40 here would have C callers size their buffer differently from every Python
 * caller for the same kernel, which is the kind of divergence that shows up as
 * an intermittent fault rather than a test failure. If this ever needs to
 * become the padded size, change it in BOTH engines at once. */
unsigned rocke_launch_plan_kernarg_size(const rocke_launch_plan_t* plan);

void rocke_launch_plan_free(rocke_launch_plan_t* plan);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_RECIPE_LAUNCH_H */
