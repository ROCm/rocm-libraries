/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_reduce.h -- C99 port of the row-wise reduction kernel instance
 * builder rocke/instances/common/reduce.py (CK Tile ``05_reduce`` parity).
 *
 *   Python (reduce.py)                    C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class Reduce2DSpec                    rocke_reduce2d_spec_t
 *   Reduce2DSpec.elems_per_thread         rocke_reduce2d_elems_per_thread(spec)
 *   Reduce2DSpec.num_warps                rocke_reduce2d_num_warps(spec)
 *   Reduce2DSpec.kernel_name()            rocke_reduce2d_kernel_name(...)
 *   is_valid_spec(spec)                   rocke_reduce2d_is_valid_spec(...)
 *   build_reduce2d(spec)                  rocke_build_reduce2d(b, spec, arch)
 *   reduce2d_grid(m, spec)                rocke_reduce2d_grid(...)
 *   reduce2d_signature(spec)              rocke_reduce2d_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_reduce2d_lower_to_llvm(...)
 *
 * The build reuses the ported helpers:
 *   rocke/helper_rocke.helpers.io.h          (io_ir_type, store_scalar_from_f32)
 *   rocke/helper_rocke.helpers.reduction.h   (tree_reduce, block_lds_reduce,
 *                                            block_lds_reduce_with_wave_prologue)
 *   rocke/helper_rocke.helpers.sweep.h       (sweep_row_chunks)
 *   rocke/helper_rocke.helpers.spec.h        (validate_io / IOSpecRule,
 *                                            kernel_name_join, ceil_div_grid,
 *                                            SignatureBuilder)
 *   rocke/helper_rocke.helpers.distribution.h (make_static_tile_distribution +
 *                                            the reduce-distribution / static
 *                                            distributed tensor / block tile
 *                                            reduce sync symbols)
 *   rocke/helper_rocke.helpers.tensor_view.h (make_global_view == naive packed,
 *                                            make_tile_window, LDS view)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills a rocke_reduce2d_spec_t. rocke_reduce2d_spec_default() returns a
 * struct with every field at the Python dataclass default; the caller then sets
 * n_per_block (required) and overrides the rest as needed.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_REDUCE_H
#define ROCKE_INSTANCE_REDUCE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

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
 *     name: str = "rocke_reduce2d"
 */
typedef struct rocke_reduce2d_spec
{
    int n_per_block; /* required */
    const char* op; /* default "sum" */
    int block_size; /* default 256 */
    int vec; /* default 4 */
    const char* dtype; /* default "f16" */
    int wave_size; /* default 64 */
    const char* name; /* default "rocke_reduce2d" */
} rocke_reduce2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set n_per_block. */
rocke_reduce2d_spec_t rocke_reduce2d_spec_default(void);

/* Reduce2DSpec.elems_per_thread @property: n_per_block // block_size. */
int rocke_reduce2d_elems_per_thread(const rocke_reduce2d_spec_t* spec);

/* Reduce2DSpec.num_warps @property: block_size // wave_size. */
int rocke_reduce2d_num_warps(const rocke_reduce2d_spec_t* spec);

/* Reduce2DSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *     kernel_name_join(self.name, self.op, self.dtype, f"N{n_per_block}",
 *                      f"b{block_size}", f"v{vec}")
 *
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t
    rocke_reduce2d_kernel_name(const rocke_reduce2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec) -> (ok, reason).
 *
 * Gate (mirrors reduce.is_valid_spec):
 *   - op in ("sum","max","min","mean","prod")
 *   - validate_io(IOSpecRule(dtype, block_size, vec, n_per_block))
 *
 * On reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and false is returned. On accept returns true and writes "ok". */
bool rocke_reduce2d_is_valid_spec(const rocke_reduce2d_spec_t* spec,
                                  char* reason,
                                  size_t reason_cap);

/* build_reduce2d(spec). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` is accepted for signature parity with the rest of
 * the C port; the Python build_reduce2d takes no arch (it is unused here).
 *
 * Kernel signature: (X: ptr<dtype>, Y: ptr<dtype>, M: i32, N: i32).
 * Grid: (M, 1, 1). Block: block_size threads. */
rocke_kernel_def_t* rocke_build_reduce2d(rocke_ir_builder_t* b,
                                         const rocke_reduce2d_spec_t* spec,
                                         const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_reduce2d_new(rocke_ir_builder_t* b,
                                             const rocke_reduce2d_spec_t* spec,
                                             const char* arch);

/* reduce2d_grid(m, spec) -> ceil_div_grid((m, 1)) == (m, 1, 1). Returns ROCKE_OK
 * and writes out[0..2]. (The Python helper ignores spec; kept for parity.) */
rocke_status_t rocke_reduce2d_grid(int m, const rocke_reduce2d_spec_t* spec, int out[3]);

/* reduce2d_signature(spec): the (X,Y,M,N) manifest. *out_items / *out_count get
 * the arena-owned entry array. `arena` owns the strings; pass a live arena.
 * Returns ROCKE_OK on success. */
rocke_status_t rocke_reduce2d_signature(rocke_arena_t* arena,
                                        const rocke_reduce2d_spec_t* spec,
                                        const rocke_sig_entry_t** out_items,
                                        size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err != NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_reduce2d_lower_to_llvm(const rocke_reduce2d_spec_t* spec,
                                            const char* arch,
                                            rocke_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_REDUCE_H */
