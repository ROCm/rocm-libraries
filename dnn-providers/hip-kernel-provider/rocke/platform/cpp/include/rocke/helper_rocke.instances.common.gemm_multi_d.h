/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.gemm_multi_d.h -- C99 port of the
 * multiple-D GEMM kernel instance rocke/instances/common/gemm_multi_d.py
 * (CK Tile ``19_gemm_multi_d`` parity).
 *
 * The kernel computes  E = f(A * B, D_0, ..., D_{n-1})  where f is a chain of
 * per-D residual add/mul ops attached as a fused epilogue to the universal
 * GEMM body.
 *
 *   Python (gemm_multi_d.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   MAX_D                                 ROCKE_GEMM_MULTI_D_MAX_D
 *   DOp = (param_name, "add"|"mul")       rocke_gemm_multi_d_op_t
 *   DLoadKind = "stock"|"tiled"|"vector"  rocke_d_load_kind_t
 *   class _MultiDEpilogue(FusedEpilogue)  rocke_multi_d_epilogue_t (+ apply_vec /
 *                                           from_ops)
 *   class GemmMultiDSpec                  rocke_gemm_multi_d_spec_t
 *   is_valid_spec(spec, arch)             rocke_gemm_multi_d_is_valid_spec(...)
 *   _build_fused_epilogue(spec)           rocke_gemm_multi_d_build_fused_epilogue(...)
 *   build_gemm_multi_d(spec, arch)        rocke_build_gemm_multi_d(...)
 *   gemm_multi_d_signature(spec)          rocke_gemm_multi_d_signature(...)
 *   gemm_multi_d_grid(spec, m, n, batch)  rocke_gemm_multi_d_grid(...)
 *
 * PEER DEPENDENCY ON THE FUSE LAYER. The Python module builds its op chain
 * from rocke.helpers.fuse (FusedEpilogue, ResidualAdd, ResidualMul,
 * dtype_to_ir, ir_dtype_global_load). That module is not yet ported; its C
 * counterpart (helper_rocke.helpers.fuse.h) is a peer. To keep this header
 * self-contained and compilable today, the small slice of the fuse interface
 * this port needs is forward-declared below behind the ROCKE_HAVE_FUSE_LAYER
 * guard -- when the real fuse header lands it defines ROCKE_HAVE_FUSE_LAYER and
 * these forward declarations are skipped.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_GEMM_MULTI_D_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_GEMM_MULTI_D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (signature storage) */
#include "rocke/helper_rocke.helpers.fuse.h" /* FusedEpilogue / _MultiDEpilogue port */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t (signature) */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t, build */
#include "rocke/ir.h" /* rocke_status_t, rocke_value_t, rocke_type_t, rocke_ir_builder_t */

/* The fuse layer (rocke/helpers/fuse.py) IS ported -- see
 * helper_rocke.helpers.fuse.h. Declare ROCKE_HAVE_FUSE_LAYER so the legacy
 * forward declarations below (which named a different, never-implemented
 * rocke_fuse_* API) are skipped; this module now links against the real fuse
 * symbols (rocke_residual_add/mul, rocke_fe_*, rocke_mde_*, rocke_fuse_dtype_to_ir_str). */
#define ROCKE_HAVE_FUSE_LAYER 1

#ifdef __cplusplus
extern "C" {
#endif

/* MAX_D = 8 (Python module constant). */
#define ROCKE_GEMM_MULTI_D_MAX_D 8

/* object.__setattr__(base_spec, "_fused_epilogue", ep): attach the fused
 * epilogue to a universal GEMM spec via the side-channel the cshuffle epilogue
 * reads. This is the C analogue of the Python frozen-dataclass back-door write.
 *
 * `ep` points at the FusedEpilogue the multi-D build composed: a plain
 * rocke_fused_epilogue_t for the "stock" load kind, or the base sub-object of a
 * rocke_multi_d_epilogue_t for "tiled"/"vector". `is_mde` records which: when
 * true, `ep` is castable back to rocke_multi_d_epilogue_t* (its first member is
 * the base) so the universal epilogue dispatches apply_vec to the optimised
 * _MultiDEpilogue.apply_vec; when false the base FusedEpilogue.apply_vec runs.
 * The base methods (declare_params / record_runtime / apply_scalar) view `ep`
 * unchanged either way. */
void rocke_gemm_universal_spec_set_fused_epilogue(rocke_gemm_universal_spec_t* spec,
                                                  rocke_fused_epilogue_t* ep,
                                                  bool is_mde);

/* ------------------------------------------------------------------ DLoadKind */
/* DLoadKind = "stock" | "tiled" | "vector". Stored as the canonical lowercase
 * string and compared by strcmp, like the Python Literal. */
typedef enum rocke_d_load_kind
{
    ROCKE_D_LOAD_STOCK = 0,
    ROCKE_D_LOAD_TILED = 1,
    ROCKE_D_LOAD_VECTOR = 2 /* default */
} rocke_d_load_kind_t;

/* --------------------------------------------------------------------- DOp *
 * DOp = (param_name, "add" | "mul"). op_is_mul==false => "add". */
typedef struct rocke_gemm_multi_d_op
{
    const char* param_name; /* non-empty, not in {A,B,C,M,N,K}, unique */
    bool op_is_mul; /* false => "add", true => "mul"           */
} rocke_gemm_multi_d_op_t;

/* ----------------------------------------------------------- GemmMultiDSpec */
typedef struct rocke_gemm_multi_d_spec
{
    rocke_gemm_universal_spec_t base; /* GEMM tile/pipeline/data choices */
    rocke_gemm_multi_d_op_t d_operands[ROCKE_GEMM_MULTI_D_MAX_D];
    size_t num_d_operands; /* len(d_operands)                 */
    const char* d_dtype; /* default "fp16"                  */
    const char* name; /* default "rocke_gemm_multi_d"   */
    rocke_d_load_kind_t d_load_kind; /* default ROCKE_D_LOAD_VECTOR        */
} rocke_gemm_multi_d_spec_t;

/* Default-constructed multi-D spec: base == universal default, no D operands,
 * d_dtype "fp16", name "rocke_gemm_multi_d", d_load_kind vector. The caller
 * fills base (+ finalize it), the D operands, and num_d_operands. */
rocke_gemm_multi_d_spec_t rocke_gemm_multi_d_spec_default(void);

/* GemmMultiDSpec.num_d property. */
static inline size_t rocke_gemm_multi_d_num_d(const rocke_gemm_multi_d_spec_t* spec)
{
    return spec->num_d_operands;
}

/* GemmMultiDSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small / NULL args). */
rocke_status_t rocke_gemm_multi_d_kernel_name(const rocke_gemm_multi_d_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and returns false. On accept returns true and writes "ok". */
bool rocke_gemm_multi_d_is_valid_spec(const rocke_gemm_multi_d_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* _build_fused_epilogue(spec): compose the per-D ResidualAdd/ResidualMul chain
 * and pick the epilogue strategy from spec.d_load_kind ("stock" -> base
 * FusedEpilogue; "tiled"/"vector" -> _MultiDEpilogue.from_ops). All op + chain
 * allocations come from `arena`. Returns the epilogue or NULL on a builder /
 * dtype / allocation error. When non-NULL, *out_is_mde reports whether the
 * returned pointer is the base sub-object of a rocke_multi_d_epilogue_t (true,
 * the "tiled"/"vector" strategies) or a plain rocke_fused_epilogue_t (false, the
 * "stock" strategy) -- the discriminator the spec side-channel needs to
 * dispatch apply_vec. */
rocke_fused_epilogue_t* rocke_gemm_multi_d_build_fused_epilogue(
    rocke_arena_t* arena, const rocke_gemm_multi_d_spec_t* spec, bool* out_is_mde);

/* build_gemm_multi_d(spec, arch): build the IR for one multi-D GEMM instance.
 * Validates, composes the fused epilogue, renames a fresh copy of the base
 * spec to spec.kernel_name(), attaches the epilogue via the side-channel, and
 * delegates to rocke_build_universal_gemm. `b` must be an initialised IRBuilder
 * (created with spec.kernel_name()); `arena` backs the epilogue + op-chain
 * allocations. `arch` NULL => "gfx950". Returns the kernel (b->kernel) on
 * success or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_gemm_multi_d(rocke_ir_builder_t* b,
                                             rocke_arena_t* arena,
                                             const rocke_gemm_multi_d_spec_t* spec,
                                             const char* arch);

/* gemm_multi_d_signature(spec): manifest-style kernarg signature in the exact
 * order the AMDGPU kernarg ABI expects:
 *   A, B, C, M, N, K, [stride_a, stride_b, stride_c?], D0, D1, ...
 * Entries (name + type strings) are arena-owned. On ROCKE_OK *out_items /
 * *out_count hold the array; on failure they are untouched and the status is
 * returned. */
rocke_status_t rocke_gemm_multi_d_signature(rocke_arena_t* arena,
                                            const rocke_gemm_multi_d_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);

/* gemm_multi_d_grid(spec, m, n, batch): same grid as build_universal_gemm,
 * (N_tiles, M_tiles, batch). batch defaults to 1 in Python; pass 1 to match.
 * On success out[0..2] hold (x, y, z); returns ROCKE_ERR_VALUE on a
 * non-positive tile (the Python ValueError) or NULL args. */
rocke_status_t rocke_gemm_multi_d_grid(
    const rocke_gemm_multi_d_spec_t* spec, int m, int n, int batch, int out[3]);

/* The _MultiDEpilogue type + its from_ops / apply_vec live in the fuse port
 * (helper_rocke.helpers.fuse.h: rocke_multi_d_epilogue_t / rocke_mde_from_ops /
 * rocke_mde_apply_vec). This module composes one there in
 * rocke_gemm_multi_d_build_fused_epilogue. */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_GEMM_MULTI_D_H */
