/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/lower_hip.h -- PUBLIC API for the C99 port of
 * rocke.core.lower_hip (lower_kernel_to_hip).
 *
 * Walks a frozen ckc IR KernelDef and emits a compilable HIP `__global__`
 * kernel source: a prologue of typedefs / AMDGPU vector aliases, an
 * `extern "C" __global__` signature derived from KernelDef.params, and the
 * lowered body (one C++ statement per IR op, plus any `__shared__` decls
 * emitted by tile.smem_alloc).
 *
 * This is the HIP-source mirror of the LLVM-IR path: the same IR in, a
 * human-readable HIP TU out, byte-identical to the Python lowerer's output for
 * the default gfx950 arch. The only entry point is rocke_lower_kernel_to_hip();
 * the rest of the file is shared internal plumbing in lower_hip_internal.h.
 *
 * Binds strictly to the FROZEN IR contract in rocke/ir.h. No new IR types are
 * introduced here -- only the arch seam (rocke_hip_arch_t) and the output struct
 * are HIP-lowerer-local.
 */
#ifndef ROCKE_LOWER_HIP_H
#define ROCKE_LOWER_HIP_H

#include <stdbool.h>

#include "rocke/ir.h"
#include "rocke/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- arch seam */

/* The HIP path has no separate ISA backend class (it emits __builtin_amdgcn_*
 * directly), so only a handful of ArchTarget hardware facts drive arch-keyed
 * decisions. The full Python ArchTarget / MMA catalog is NOT ported; this is
 * the minimal subset the lowerer reads:
 *
 *   - waitcnt encoding family: gfx9/gfx10 split-VMCNT vs gfx11 contiguous,
 *   - whether ds_read_*_tr_* exists (gfx950-class only),
 *   - whether WMMA is available (RDNA/gfx11+; never on CDNA/MFMA targets).
 *
 * Resolve one from a gfx string with rocke_hip_arch_from_gfx(); the default is
 * gfx950, which reproduces the byte-identical CDNA baseline. */
typedef enum rocke_hip_waitcnt_family
{
    ROCKE_HIP_WAITCNT_GFX9_10 = 0, /* split VMCNT [3:0]+[15:14], lgkm [11:8]     */
    ROCKE_HIP_WAITCNT_GFX11 /* contiguous exp[2:0] lgkm[9:4] vm[15:10]    */
} rocke_hip_waitcnt_family_t;

typedef struct rocke_hip_arch
{
    const char* gfx; /* canonical gfx string, e.g. "gfx950" */
    rocke_hip_waitcnt_family_t waitcnt_family; /* s_waitcnt immediate layout        */
    bool has_ds_read_tr; /* ds_read_b{64,128}_tr_b16 present   */
    bool has_wmma; /* RDNA/gfx11+ WMMA instruction        */
    const char* family; /* "cdna"/"rdna" for diagnostics       */
} rocke_hip_arch_t;

/* Resolve an arch seam from a gfx name (NULL => gfx950 default). Always returns
 * a fully-populated struct; an unrecognised gfx falls back to gfx950 facts. */
rocke_hip_arch_t rocke_hip_arch_from_gfx(const char* gfx);

/* ------------------------------------------------------------- options/output */

/* Options for rocke_lower_kernel_to_hip (the Python keyword-only args). A *_set
 * companion flag distinguishes "unset" (use default) from an explicit value, so
 * defaults match Python (launch_bounds=max_workgroup_size, include_prologue=
 * true, arch=gfx950). */
typedef struct rocke_lower_hip_opts
{
    int launch_bounds;
    bool launch_bounds_set; /* default max_wg_size */
    bool include_prologue;
    bool include_prologue_set; /* default true     */
    const char* arch; /* NULL => gfx950                              */
} rocke_lower_hip_opts_t;

/* The compilable-HIP prologue text (typedefs + AMDGPU vector aliases + LLVM
 * intrinsic shims). Pasted at the top of every lowered source. Static storage;
 * never freed. Mirrors Python HIP_PROLOGUE byte-for-byte. */
extern const char* const ROCKE_HIP_PROLOGUE;

/* --------------------------------------------------------------- entry point */

/* Lower `kernel` to a compilable HIP source, appending the full text to `out`
 * (the caller owns/initialises the strbuf). `opts` may be NULL for all
 * defaults. Returns ROCKE_OK on success; on failure returns a non-OK status and
 * leaves a message reachable via the builder error channel embedded in the
 * lowerer (the partial text in `out` is unspecified). An op with no HIP handler
 * yields ROCKE_ERR_NOTIMPL, mirroring Python NotImplementedError.
 *
 * `b` provides the arena for transient allocations (per-op temporaries, smem
 * decl accumulation) and the sticky error channel; it is not mutated as an IR
 * graph. Pass the same builder that owns `kernel`. */
rocke_status_t rocke_lower_kernel_to_hip(rocke_ir_builder_t* b,
                                         const rocke_kernel_def_t* kernel,
                                         const rocke_lower_hip_opts_t* opts,
                                         rocke_strbuf_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_LOWER_HIP_H */
