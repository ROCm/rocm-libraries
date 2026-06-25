/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_reduce.h -- C99 port of the row-wise reduction kernel instance
 * builder ck_dsl/instances/common/reduce.py (CK Tile ``05_reduce`` parity).
 *
 *   Python (reduce.py)                    C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class Reduce2DSpec                    ckc_reduce2d_spec_t
 *   Reduce2DSpec.elems_per_thread         ckc_reduce2d_elems_per_thread(spec)
 *   Reduce2DSpec.num_warps                ckc_reduce2d_num_warps(spec)
 *   Reduce2DSpec.kernel_name()            ckc_reduce2d_kernel_name(...)
 *   is_valid_spec(spec)                   ckc_reduce2d_is_valid_spec(...)
 *   build_reduce2d(spec)                  ckc_build_reduce2d(b, spec, arch)
 *   reduce2d_grid(m, spec)                ckc_reduce2d_grid(...)
 *   reduce2d_signature(spec)              ckc_reduce2d_signature(...)
 *   (+ convenience: build -> lower .ll)   ckc_reduce2d_lower_to_llvm(...)
 *
 * The build reuses the ported helpers:
 *   ckc/helper_ck_dsl.helpers.io.h          (io_ir_type, store_scalar_from_f32)
 *   ckc/helper_ck_dsl.helpers.reduction.h   (tree_reduce, block_lds_reduce,
 *                                            block_lds_reduce_with_wave_prologue)
 *   ckc/helper_ck_dsl.helpers.sweep.h       (sweep_row_chunks)
 *   ckc/helper_ck_dsl.helpers.spec.h        (validate_io / IOSpecRule,
 *                                            kernel_name_join, ceil_div_grid,
 *                                            SignatureBuilder)
 *   ckc/helper_ck_dsl.helpers.distribution.h (make_static_tile_distribution +
 *                                            the reduce-distribution / static
 *                                            distributed tensor / block tile
 *                                            reduce sync symbols)
 *   ckc/helper_ck_dsl.helpers.tensor_view.h (make_global_view == naive packed,
 *                                            make_tile_window, LDS view)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a ckc_reduce2d_spec_t. ckc_reduce2d_spec_default() returns a
 * struct with every field at the Python dataclass default; the caller then sets
 * n_per_block (required) and overrides the rest as needed.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_REDUCE_H
#define CKC_INSTANCE_REDUCE_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- Reduce2DSpec *
 *
 * Mirror of Python Reduce2DSpec (frozen dataclass):
 *
 *     n_per_block: int
 *     op: ReduceOp = "sum"          # "sum"|"max"|"min"|"mean"|"prod"
 *     block_size: int = 256
 *     vec: int = 4
 *     dtype: DType = "f16"          # "f16"|"bf16"
 *     wave_size: int = 64
 *     name: str = "ck_dsl_reduce2d"
 */
typedef struct ckc_reduce2d_spec
{
    int n_per_block;   /* required */
    const char* op;    /* default "sum" */
    int block_size;    /* default 256 */
    int vec;           /* default 4 */
    const char* dtype; /* default "f16" */
    int wave_size;     /* default 64 */
    const char* name;  /* default "ck_dsl_reduce2d" */
} ckc_reduce2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set n_per_block. */
ckc_reduce2d_spec_t ckc_reduce2d_spec_default(void);

/* Reduce2DSpec.elems_per_thread @property: n_per_block // block_size. */
int ckc_reduce2d_elems_per_thread(const ckc_reduce2d_spec_t* spec);

/* Reduce2DSpec.num_warps @property: block_size // wave_size. */
int ckc_reduce2d_num_warps(const ckc_reduce2d_spec_t* spec);

/* Reduce2DSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, self.op, self.dtype, f"N{n_per_block}",
 *                      f"b{block_size}", f"v{vec}")
 *
 * Returns CKC_OK, or CKC_ERR_VALUE (buffer too small). */
ckc_status_t ckc_reduce2d_kernel_name(const ckc_reduce2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec) -> (ok, reason).
 *
 * Gate (mirrors reduce.is_valid_spec):
 *   - op in ("sum","max","min","mean","prod")
 *   - validate_io(IOSpecRule(dtype, block_size, vec, n_per_block))
 *
 * On reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and false is returned. On accept returns true and writes "ok". */
bool ckc_reduce2d_is_valid_spec(const ckc_reduce2d_spec_t* spec, char* reason, size_t reason_cap);

/* build_reduce2d(spec). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` is accepted for signature parity with the rest of
 * the C port; the Python build_reduce2d takes no arch (it is unused here).
 *
 * Kernel signature: (X: ptr<dtype>, Y: ptr<dtype>, M: i32, N: i32).
 * Grid: (M, 1, 1). Block: block_size threads. */
ckc_kernel_def_t*
ckc_build_reduce2d(ckc_ir_builder_t* b, const ckc_reduce2d_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t*
ckc_build_reduce2d_new(ckc_ir_builder_t* b, const ckc_reduce2d_spec_t* spec, const char* arch);

/* reduce2d_grid(m, spec) -> ceil_div_grid((m, 1)) == (m, 1, 1). Returns CKC_OK
 * and writes out[0..2]. (The Python helper ignores spec; kept for parity.) */
ckc_status_t ckc_reduce2d_grid(int m, const ckc_reduce2d_spec_t* spec, int out[3]);

/* reduce2d_signature(spec): the (X,Y,M,N) manifest. *out_items / *out_count get
 * the arena-owned entry array. `arena` owns the strings; pass a live arena.
 * Returns CKC_OK on success. */
ckc_status_t ckc_reduce2d_signature(ckc_arena_t* arena,
                                    const ckc_reduce2d_spec_t* spec,
                                    const ckc_sig_entry_t** out_items,
                                    size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err != NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_reduce2d_lower_to_llvm(const ckc_reduce2d_spec_t* spec,
                                        const char* arch,
                                        ckc_llvm_flavor_t flavor,
                                        char** out_ll,
                                        char* err,
                                        size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_REDUCE_H */
