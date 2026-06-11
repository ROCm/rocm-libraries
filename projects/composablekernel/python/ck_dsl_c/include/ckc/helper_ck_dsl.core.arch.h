/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.core.arch.h -- C99 port of the two ck_dsl.core.arch symbols
 * the helper closure imports: ArchTarget and MmaOp.
 *
 *   Python (ck_dsl.core.arch)         C99 (this header)
 *   ------------------------------    --------------------------------------
 *   class ArchTarget (frozen)         ckc_archtarget_t (alias) + ckc_archtarget_*
 *   class MmaOp (frozen)              ckc_mmaop_t      (alias) + ckc_mmaop_*
 *   ArchTarget.from_gfx(gfx)          ckc_archtarget_from_gfx()
 *   target.mma.op_for_shape(...)      ckc_archtarget_op_for_shape()
 *
 * WHY A SHIM. The canonical, byte-identical port of the WHOLE arch module
 * (LayoutMap, MmaOp, MmaCatalog, ArchTarget, the embedded arch_specs.json SSOT,
 * every lane-coord emitter, the _FragInfo table, normalize_dtype, from_gfx,
 * known_arches, arch_from_isa) already lives in ckc/arch_target.h and its two
 * translation units (arch_target_data.c / arch_target_query.c). It is the single
 * source of truth and is what actually builds the IR-emitting layout maps.
 *
 * This helper file is the binding point the *helper closure* names as
 * `ck_dsl.core.arch` { ArchTarget, MmaOp }: the sibling helpers (atoms, spec,
 * tensor_view, ...) being ported in the same phase import exactly those two
 * symbols. To keep their bindings stable AND avoid forking the SSOT (which would
 * risk divergence from the byte-identical op sequence), this header re-exposes
 * the canonical types/entry points under the helper-module ckc_archtarget_/
 * ckc_mmaop_ namespace via type aliases and thin forwarding wrappers. There is
 * no second copy of the data or the builder-call sequence -- every call lands in
 * the canonical arch_target implementation, so the emitted op sequence is, by
 * construction, identical to Python.
 */
#ifndef CKC_HELPER_CK_DSL_CORE_ARCH_H
#define CKC_HELPER_CK_DSL_CORE_ARCH_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arch_target.h"
#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================== type aliases ============================ */

/* ArchTarget / MmaOp -- the two symbols the helper closure imports. These are
 * the SAME structs as the canonical port; the helper-namespace names are
 * provided so sibling helpers can spell them as the Python module did. */
typedef ckc_arch_target_t ckc_archtarget_t;
typedef ckc_mma_op_t ckc_mmaop_t;

/* Supporting types re-exported for callers that touch them through MmaOp /
 * ArchTarget (LayoutMap from MmaOp.*_layout(), the catalog handle, the role
 * enum). Aliases only -- no new data. */
typedef ckc_layout_map_t ckc_arch_layout_map_t;
typedef ckc_mma_catalog_t ckc_arch_mma_catalog_t;
typedef ckc_mma_role_t ckc_arch_mma_role_t;

/* ============================== MmaOp ================================== */

/* MmaOp.shape -> (m, n, k). Forwards to ckc_mma_op_shape. */
void ckc_mmaop_shape(const ckc_mmaop_t* op, int* m, int* n, int* k);

/* MmaOp.a_layout() / b_layout() / c_layout() / acc_layout(). Each returns the
 * verified lane/slot -> coord map for the role, or NULL when none is registered
 * for the op_id (Python raises NotImplementedError; here, when `b` is non-NULL,
 * the builder's sticky error is set -- pass b=NULL for a pure lookup). Forwards
 * to the canonical ckc_mma_op_*_layout accessors. */
const ckc_arch_layout_map_t* ckc_mmaop_a_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b);
const ckc_arch_layout_map_t* ckc_mmaop_b_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b);
const ckc_arch_layout_map_t* ckc_mmaop_c_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b);
const ckc_arch_layout_map_t* ckc_mmaop_acc_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b);

/* LayoutMap.coord(builder, lane, slot) -> (coord0, coord1). Forwards to
 * ckc_layout_map_coord (validates slot in [0, frag_len); emits arith via b). */
bool ckc_arch_layout_map_coord(const ckc_arch_layout_map_t* m,
                               ckc_ir_builder_t* b,
                               ckc_value_t* lane,
                               int slot,
                               ckc_value_t** out0,
                               ckc_value_t** out1);

/* ============================== ArchTarget ============================= */

/* ArchTarget.from_gfx(gfx). Returns the canonical singleton descriptor for
 * `gfx`, or NULL for an unknown target (Python raises KeyError). Forwards to
 * ckc_arch_target_from_gfx. */
const ckc_archtarget_t* ckc_archtarget_from_gfx(const char* gfx);

/* The MMA catalog of a target (`target.mma`). Returns &t->mma; NULL if t NULL. */
const ckc_arch_mma_catalog_t* ckc_archtarget_mma(const ckc_archtarget_t* t);

/* target.mma.op_for_shape(...): the resolved MmaOp for an exact (m, n, k) atom
 * shape and (normalised) dtype combo, or NULL if absent. `family` NULL => "mma".
 * This is the entry point build_universal_gemm uses to resolve the atom from the
 * target catalog. Forwards to ckc_mma_catalog_op_for_shape on t->mma. */
const ckc_mmaop_t* ckc_archtarget_op_for_shape(const ckc_archtarget_t* t,
                                               const char* family,
                                               const char* a_dtype,
                                               const char* b_dtype,
                                               const char* c_dtype,
                                               int m,
                                               int n,
                                               int k);

/* ArchTarget.isa_triple -> "amdgcn-amd-amdhsa--<gfx>" into `out`. */
const char* ckc_archtarget_isa_triple(const ckc_archtarget_t* t, char* out, size_t out_cap);

/* ArchTarget.fits_lds(bytes_in_use). */
bool ckc_archtarget_fits_lds(const ckc_archtarget_t* t, long bytes_in_use);

/* ArchTarget.supports_dtype_combo(a, b, c, family). family NULL => "mma". */
bool ckc_archtarget_supports_dtype_combo(
    const ckc_archtarget_t* t, const char* a, const char* b, const char* c, const char* family);

/* ArchTarget.max_vector_load_dwords(dtype) (dtype accepted for parity, ignored
 * as in Python: width gated by the buffer-load path). */
int ckc_archtarget_max_vector_load_dwords(const ckc_archtarget_t* t, const char* dtype);

/* ArchTarget.max_threads_per_block property. */
int ckc_archtarget_max_threads_per_block(const ckc_archtarget_t* t);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_CORE_ARCH_H */
