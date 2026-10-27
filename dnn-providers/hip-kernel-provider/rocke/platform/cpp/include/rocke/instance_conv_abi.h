// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_abi.h -- the AOT convolution kernarg ABI, C++ twin of the
 * library's kernels/common/conv_abi.py.
 *
 * An AOT conv kernel takes its problem shape as kernel arguments, and
 * kernargs pack positionally: a builder that declares its params in a
 * different order from the launch signature silently shifts every argument
 * after the divergence. So the order is written down once per engine, as
 * data, and every conv builder emits its params from it:
 *
 *   rocke_conv_arg_names()        <->  conv_arg_names()
 *   rocke_conv_direct_arg_names() <->  conv_direct_arg_names()
 *   rocke_conv_fwd_problem_block()<->  conv_fwd_problem_block()
 *   rocke_conv_emit_param_block() <->  emit_param_block()
 *
 * This is convolution-specific, so it lives next to the conv instances, not
 * in the generic helpers. The two engines' lists are held together by the
 * byte-identity gate: the param names and order are part of the emitted IR
 * of every conv family.
 */
#ifndef ROCKE_INSTANCE_CONV_ABI_H
#define ROCKE_INSTANCE_CONV_ABI_H

#include <stdbool.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors the Python ArgSpec kind strings. */
typedef enum
{
    ROCKE_CONV_ARG_A, /* "a"    -- pointer with the A operand dtype */
    ROCKE_CONV_ARG_B, /* "b"    -- pointer with the B operand dtype */
    ROCKE_CONV_ARG_D, /* "d"    -- pointer with the D operand dtype */
    ROCKE_CONV_ARG_F32_PTR, /* "f32*" */
    ROCKE_CONV_ARG_I32_PTR, /* "i32*" */
    ROCKE_CONV_ARG_I32 /* "i32"  -- scalar */
} rocke_conv_arg_kind_t;

#define ROCKE_CONV_ARG_NAME_CAP 32
#define ROCKE_CONV_MAX_ARGS 96

typedef struct
{
    char name[ROCKE_CONV_ARG_NAME_CAP];
    rocke_conv_arg_kind_t kind;
} rocke_conv_arg_t;

typedef struct
{
    rocke_conv_arg_t items[ROCKE_CONV_MAX_ARGS];
    int count;
} rocke_conv_arg_list_t;

/* The Python kind string ("a", "b", "d", "f32*", "i32*", "i32"). */
const char* rocke_conv_arg_kind_str(rocke_conv_arg_kind_t kind);

/* Implicit-GEMM ABI for one direction ("fwd", "wgrad", "dgrad"). Returns
 * false -- leaving *out empty -- for an unknown direction or for two_stage
 * outside wgrad, the cases conv_arg_names() raises on. */
bool rocke_conv_arg_names(const char* direction,
                          bool is_3d,
                          bool two_stage,
                          rocke_conv_arg_list_t* out);

/* As above, plus wgrad's stream-K variant: `streamk` names the reduction
 * ("atomic", "workspace", "linear" or "tree") of a stream-K kernel, or is NULL
 * for a kernel without stream-K. A stream-K list carries the reduction's
 * buffers after ks_count and then the partition block. Returns false for a
 * non-wgrad direction, an unknown reduction, or stream-K combined with
 * two_stage -- the cases conv_arg_names() raises on. */
bool rocke_conv_arg_names_ex(const char* direction,
                             bool is_3d,
                             bool two_stage,
                             const char* streamk,
                             rocke_conv_arg_list_t* out);

/* Direct grouped conv ABI ("fwd", "dgrad" or "wgrad"); false for any other
 * direction. */
bool rocke_conv_direct_arg_names(const char* direction, rocke_conv_arg_list_t* out);

/* The forward problem block without the leading pointer/byte-size six, for a
 * fused kernel that keeps its own pointer convention (deep_fused_conv_pool). */
void rocke_conv_fwd_problem_block(bool is_3d, rocke_conv_arg_list_t* out);

/* Called for every non-i32 entry; returns the declared param. The caller owns
 * pointer declarations because noalias/readonly/align and the element type are
 * builder-specific. */
typedef rocke_value_t* (*rocke_conv_declare_ptr_fn)(rocke_ir_builder_t* b,
                                                    const char* name,
                                                    rocke_conv_arg_kind_t kind,
                                                    void* user);

/* Where a declared param should be stored, by name. */
typedef struct
{
    const char* name;
    rocke_value_t** slot;
} rocke_conv_param_slot_t;

/* Declare every entry of `args` in order (i32 entries directly, the rest via
 * `declare_ptr`) and store each into the slot of the same name, if any. An
 * entry without a slot is still declared -- the ABI position is what matters
 * -- and a slot the list has no entry for is left untouched, which is how one
 * slot table serves both the 2-D and the 3-D block (the Python side returns a
 * dict and indexes only what the variant has). Returns false, with the
 * builder error set, if a pointer entry has no declare_ptr. */
bool rocke_conv_emit_param_block(rocke_ir_builder_t* b,
                                 const rocke_conv_arg_list_t* args,
                                 rocke_conv_declare_ptr_fn declare_ptr,
                                 void* user,
                                 const rocke_conv_param_slot_t* slots,
                                 int num_slots);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_CONV_ABI_H */
