/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_add_rmsnorm2d_rdquant.h -- C99 port of the fused
 * add + RMSNorm + round-to-quant kernel instance builder
 * ck_dsl/instances/common/add_rmsnorm2d_rdquant.py.
 *
 *   Python (add_rmsnorm2d_rdquant.py)        C99 (this header)
 *   --------------------------------------   --------------------------------
 *   class AddRmsnorm2DRdquantSpec            ckc_add_rmsnorm2d_rdquant_spec_t
 *   AddRmsnorm2DRdquantSpec.elems_per_thread ckc_add_rmsnorm2d_rdquant_elems_per_thread
 *   AddRmsnorm2DRdquantSpec.kernel_name()    ckc_add_rmsnorm2d_rdquant_kernel_name
 *   is_valid_spec(spec, arch)                ckc_add_rmsnorm2d_rdquant_is_valid_spec
 *   build_add_rmsnorm2d_rdquant(spec, arch)  ckc_build_add_rmsnorm2d_rdquant
 *   add_rmsnorm2d_rdquant_grid(m, spec)      ckc_add_rmsnorm2d_rdquant_grid
 *   (+ convenience: build -> lower .ll)      ckc_add_rmsnorm2d_rdquant_lower_to_llvm
 *
 * The build mirrors build_add_rmsnorm2d_rdquant() top-to-bottom so a reviewer
 * can diff line by line. It reuses the ported helpers:
 *   - ck_dsl.helpers.reduction : block_lds_reduce_pair, tree_reduce
 *   - ck_dsl.helpers.sweep     : sweep_row_chunks
 *   - ck_dsl.helpers.io        : io_ir_type, store_scalar_from_f32
 *   - ck_dsl.helpers.quant     : quant_ir_type, quant_max_abs,
 *                                quantize_scalar_f32,
 *                                pack_quant_chunk_local_f32,
 *                                store_packed_chunk_local
 *   - ck_dsl.helpers.spec      : IOSpecRule/validate_io, ceil_div_grid,
 *                                kernel_name_join
 *   - ck_dsl.helpers.tensor_view : make_global_view, make_lds_view,
 *                                  make_naive_tensor_view_packed,
 *                                  make_tile_window
 *   - ck_dsl.core.arch         : ArchTarget
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a ckc_add_rmsnorm2d_rdquant_spec_t.
 * ckc_add_rmsnorm2d_rdquant_spec_default() returns a struct with every field
 * set to the Python dataclass default; the caller sets n_per_block (required)
 * and overrides the rest.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_ADD_RMSNORM2D_RDQUANT_H
#define CKC_INSTANCE_ADD_RMSNORM2D_RDQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------- AddRmsnorm2DRdquantSpec *
 *
 * Mirror of Python AddRmsnorm2DRdquantSpec (frozen dataclass):
 *
 *     n_per_block: int
 *     dtype: DType = "f16"          # "f16" | "bf16"
 *     out_dtype: QDType = "i8"      # "i8" | "fp8e4m3" | "bf8e5m2"
 *     block_size: int = 256
 *     vec: int = 4
 *     save_residual: bool = True
 *     save_yscale: bool = True
 *     wave_size: int = 64
 *     name: str = "ck_dsl_add_rmsnorm2d_rdquant"
 */
typedef struct ckc_add_rmsnorm2d_rdquant_spec
{
    int n_per_block;
    const char* dtype;     /* default "f16"  */
    const char* out_dtype; /* default "i8"   */
    int block_size;        /* default 256    */
    int vec;               /* default 4      */
    bool save_residual;    /* default true   */
    bool save_yscale;      /* default true   */
    int wave_size;         /* default 64     */
    const char* name;      /* default "ck_dsl_add_rmsnorm2d_rdquant" */
} ckc_add_rmsnorm2d_rdquant_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set n_per_block. */
ckc_add_rmsnorm2d_rdquant_spec_t ckc_add_rmsnorm2d_rdquant_spec_default(void);

/* AddRmsnorm2DRdquantSpec.elems_per_thread @property: n_per_block / block_size. */
int ckc_add_rmsnorm2d_rdquant_elems_per_thread(const ckc_add_rmsnorm2d_rdquant_spec_t* spec);

/* AddRmsnorm2DRdquantSpec.kernel_name() -> NUL-terminated into out (out_cap):
 *
 *     kernel_name_join(self.name, self.dtype, self.out_dtype,
 *                      f"N{n_per_block}", f"b{block_size}", f"v{vec}",
 *                      flags={"sr": save_residual, "ys": save_yscale})
 *
 * Returns CKC_OK, or CKC_ERR_VALUE (buffer too small / null args). */
ckc_status_t ckc_add_rmsnorm2d_rdquant_kernel_name(const ckc_add_rmsnorm2d_rdquant_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 *
 * Gate (mirrors add_rmsnorm2d_rdquant.is_valid_spec):
 *   - arch resolves via ArchTarget.from_gfx (else reject)
 *   - out_dtype in ("i8", "fp8e4m3", "bf8e5m2")
 *   - fp8/bf8 output requires target.family == "cdna"
 *   - validate_io(IOSpecRule(dtype, block_size, vec, n_per_block,
 *                            max_elems_per_thread=64))
 *   - block_size <= target.max_threads_per_block
 *   - 2 * block_size * 4 LDS bytes fit the target budget
 *
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and false is returned. On accept returns true and writes
 * "". */
bool ckc_add_rmsnorm2d_rdquant_is_valid_spec(const ckc_add_rmsnorm2d_rdquant_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);

/* build_add_rmsnorm2d_rdquant(spec, arch). Builds the IR into the supplied
 * (already ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly
 * as the Python build does, and returns the kernel (b->kernel) on success or
 * NULL with b's sticky error set. `arch` NULL => "gfx950".
 *
 * Kernel signature (both optional outputs enabled):
 *   (A: ptr<dtype>, B: ptr<dtype>, Gamma: ptr<dtype>,
 *    X: ptr<dtype> [if save_residual], QY: ptr<out_dtype>,
 *    YScale: ptr<f32> [if save_yscale],
 *    M: i32, N: i32, eps_rms: f32, eps_q: f32)
 * Grid: (M, 1, 1) -- one CTA per row. Block: block_size threads. */
ckc_kernel_def_t* ckc_build_add_rmsnorm2d_rdquant(ckc_ir_builder_t* b,
                                                  const ckc_add_rmsnorm2d_rdquant_spec_t* spec,
                                                  const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_add_rmsnorm2d_rdquant_new(ckc_ir_builder_t* b,
                                                      const ckc_add_rmsnorm2d_rdquant_spec_t* spec,
                                                      const char* arch);

/* add_rmsnorm2d_rdquant_grid(m, spec) -> ceil_div_grid((m, 1)) == (m, 1, 1).
 * Returns CKC_OK and writes out[0..2]; CKC_ERR_VALUE on a bad argument. */
ckc_status_t
ckc_add_rmsnorm2d_rdquant_grid(int m, const ckc_add_rmsnorm2d_rdquant_spec_t* spec, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err != NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_add_rmsnorm2d_rdquant_lower_to_llvm(const ckc_add_rmsnorm2d_rdquant_spec_t* spec,
                                                     const char* arch,
                                                     ckc_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_ADD_RMSNORM2D_RDQUANT_H */
