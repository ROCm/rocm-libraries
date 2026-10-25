// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gett_f64.h -- C++ mirror of instances/common/gett_f64.py: the f64 /
 * complex-f64 GETT (strided GEMM and tensor contraction, every operand read
 * through its own strides). Builds IR byte-identical to the Python builder for
 * the same spec. This API takes a resolved problem (digits) and a resolved
 * spec; contraction planning and the tuned selection live in the Python
 * dispatch layer (gett_f64_plan.py, dispatch/families/gett_f64.py).
 */
#ifndef ROCKE_INSTANCE_GETT_F64_H
#define ROCKE_INSTANCE_GETT_F64_H

#include "rocke/ir.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_GETT_F64_MAX_DIGITS 12
/* Largest grid y (the batch dimension of a launch). */
#define ROCKE_GETT_F64_MAX_GRID_Y 65535
/* Number of size / stride arguments a runtime_sizes kernel can take. */
#define ROCKE_GETT_F64_RUNTIME_ARGS 16
/* Threads per block of the split_k reduce kernel (one output each). */
#define ROCKE_GETT_F64_REDUCE_THREADS 256

/* (extent, stride) pairs, fastest digit first (Python Digits). */
typedef struct rocke_gett_digits
{
    int n;
    int64_t extent[ROCKE_GETT_F64_MAX_DIGITS];
    int64_t stride[ROCKE_GETT_F64_MAX_DIGITS];
} rocke_gett_digits_t;

/* GettF64Problem. Digits are normalized as in Python (extent-1 digits dropped,
 * an empty space is the single digit (1, 0)); use rocke_gett_digits_norm. */
typedef struct rocke_gett_f64_problem
{
    int64_t M, N, K, L;
    rocke_gett_digits_t a_m, a_k, a_l;
    rocke_gett_digits_t b_n, b_k, b_l;
    rocke_gett_digits_t d_m, d_n, d_l;
    bool has_c; /* false: C has D's digits (Python c_* = None) */
    rocke_gett_digits_t c_m, c_n, c_l;
    bool complex_;
    bool conj_a;
    bool conj_b;
} rocke_gett_f64_problem_t;

/* GettF64Spec (lds_pad < 0 means None, waves_per_eu 0 means None). lds_swizzle_mode,
 * lds_stages, prefetch_depth, complex_accs, unroll2, k_stagger, k_block,
 * runtime_sizes and split_k follow the Python knobs of the same names (defaults
 * 0, 2, 1, 2, false, 0, 0, false and 1). With runtime_sizes, `problem` is the
 * layout template. With split_k > 1, rocke_build_gett_f64 builds the partial-sum
 * kernel (A, B, W) and rocke_build_gett_f64_reduce the second kernel. */
typedef struct rocke_gett_f64_spec
{
    rocke_gett_f64_problem_t problem;
    int tile_m;
    int tile_n;
    int tile_k;
    int warps_m;
    int warps_n;
    int lds_pad;
    bool lds_swizzle;
    int group_m;
    int xcds;
    int waves_per_eu;
    bool bilinear;
    bool prefetch;
    bool sched_fence;
    int lds_swizzle_mode;
    int lds_stages;
    int prefetch_depth;
    int complex_accs;
    bool unroll2;
    int k_stagger;
    const char* name;
    int k_block;
    bool runtime_sizes;
    int split_k;
} rocke_gett_f64_spec_t;

/* Normalize raw (extent, stride) pairs in place (Python _norm). */
void rocke_gett_digits_norm(rocke_gett_digits_t* d);

/* Python GettF64Spec defaults around `problem`. */
rocke_gett_f64_spec_t rocke_gett_f64_spec_default(const rocke_gett_f64_problem_t* problem);

/* Python f64_atom(arch) is not None: the catalog has the 16x16x4 f64 atom. */
bool rocke_gett_f64_arch_supported(const char* arch);

/* Python runtime_template: the canonical layout class of a strided batched
 * GEMM. False when an index space has more than one digit. */
bool rocke_gett_f64_runtime_template(const rocke_gett_f64_problem_t* problem,
                                     rocke_gett_f64_problem_t* out);

/* Python runtime_compatible: whether the runtime_sizes kernel of `spec`
 * computes `problem`. */
bool rocke_gett_f64_runtime_compatible(const rocke_gett_f64_spec_t* spec,
                                       const rocke_gett_f64_problem_t* problem,
                                       char* reason,
                                       size_t reason_cap);

/* Python gett_f64_runtime_args: the size / stride argument values of `problem`
 * in kernel-argument order (RUNTIME_SIZE_ARGS, C strides only when bilinear).
 * Writes *count values; returns false (count 0) for an incompatible problem. */
bool rocke_gett_f64_runtime_args(const rocke_gett_f64_spec_t* spec,
                                 const rocke_gett_f64_problem_t* problem,
                                 int64_t out[ROCKE_GETT_F64_RUNTIME_ARGS],
                                 int* count);

/* Python is_valid_spec: false + reason when the spec cannot be built on arch. */
bool rocke_gett_f64_is_valid_spec(const rocke_gett_f64_spec_t* spec,
                                  const char* arch,
                                  char* reason,
                                  size_t reason_cap);

/* Python GettF64Spec.kernel_name. */
rocke_status_t
    rocke_gett_f64_kernel_name(const rocke_gett_f64_spec_t* spec, char* out, size_t out_cap);

/* Python gett_f64_grid(spec): the grid of the spec's own problem (sizes baked in). */
void rocke_gett_f64_grid(const rocke_gett_f64_spec_t* spec, int grid[3]);

/* Python gett_f64_grid(spec, problem): the grid of `problem` (required for a
 * runtime_sizes kernel). False when L exceeds ROCKE_GETT_F64_MAX_GRID_Y. */
bool rocke_gett_f64_grid_for(const rocke_gett_f64_spec_t* spec,
                             const rocke_gett_f64_problem_t* problem,
                             int grid[3]);

/* Initializes b; the returned kernel is owned by b. NULL (with b's error set)
 * for an invalid spec. Mirrors build_gett_f64(spec, arch=arch). */
rocke_kernel_def_t* rocke_build_gett_f64(rocke_ir_builder_t* b,
                                         const rocke_gett_f64_spec_t* spec,
                                         const char* arch);

/* Python gett_f64_workspace_bytes: bytes of the split_k workspace W for
 * `problem` (NULL: the spec's own problem); 0 without split_k. */
int64_t rocke_gett_f64_workspace_bytes(const rocke_gett_f64_spec_t* spec,
                                       const rocke_gett_f64_problem_t* problem);

/* Python gett_f64_reduce_name. */
rocke_status_t
    rocke_gett_f64_reduce_name(const rocke_gett_f64_spec_t* spec, char* out, size_t out_cap);

/* Python gett_f64_reduce_grid (problem NULL: the spec's own problem). */
bool rocke_gett_f64_reduce_grid(const rocke_gett_f64_spec_t* spec,
                                const rocke_gett_f64_problem_t* problem,
                                int grid[3]);

/* Initializes b; mirrors build_gett_f64_reduce(spec, arch=arch): the second
 * kernel of split_k (sums the partials in order, then the epilogue). */
rocke_kernel_def_t* rocke_build_gett_f64_reduce(rocke_ir_builder_t* b,
                                                const rocke_gett_f64_spec_t* spec,
                                                const char* arch);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_GETT_F64_H */
