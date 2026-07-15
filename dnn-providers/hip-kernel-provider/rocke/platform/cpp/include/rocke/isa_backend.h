/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/isa_backend.h -- C99 port of rocke.core.isa.backend.
 *
 * The ISA backend owns the per-gfx LLVM details the lowerer needs but cannot
 * keep target-neutral: the module datalayout/triple, the buffer resource
 * descriptor DWORD3, the ``s_waitcnt`` immediate encoding, and the MFMA/WMMA
 * matrix-op emission.
 *
 * Python -> C99 mapping
 * ---------------------
 *   class ISABackend (base, CDNA)        rocke_isa_backend_t + ROCKE_ISA_GFX9_MFMA
 *   class Gfx950Backend                  ROCKE_ISA_GFX950 (byte-identical baseline)
 *   class Gfx9MfmaBackend                ROCKE_ISA_GFX9_MFMA
 *   class Gfx11RdnaBackend               ROCKE_ISA_GFX11_RDNA
 *   class Gfx12RdnaBackend               ROCKE_ISA_GFX12_RDNA
 *   backend_for(arch)                    rocke_backend_for()
 *   backend.triple/.datalayout           rocke_isa_triple()/rocke_isa_datalayout()
 *   backend.module_preamble()            rocke_isa_module_preamble()
 *   backend.buffer_rsrc_word3            rocke_isa_buffer_rsrc_word3()
 *   backend.encode_waitcnt(...)          rocke_isa_encode_waitcnt(...)
 *   backend.emit_wmma(lowerer, op)       rocke_isa_emit_wmma_call(...) (text only)
 *   _RDNA_WMMA / _RDNA_WMMA_INT /        rocke_isa_wmma_lookup() / *_int / *_gfx12
 *     _RDNA_GFX12_WMMA tables
 *
 * Scope note (see the implementation's TODO(port) markers): the Python
 * emit_mma/emit_wmma methods drive a live ``_Lowerer`` (op rebuilding via
 * Op(), _need/_operand/_fresh/_current().emit). That lowerer has no C type yet
 * in this port, so the methods that *dispatch through the lowerer* are not
 * portable here. What IS self-contained -- the byte-identical WMMA call text
 * and its bf16 bitcast prologue, given the SSA operand name strings -- is
 * ported as rocke_isa_emit_wmma_call(), writing into a rocke_strbuf. The MMA op
 * resolution table (op.name -> intrinsic/decl/element-type spec) is fully
 * ported as the lookup functions below.
 */
#ifndef ROCKE_ISA_BACKEND_H
#define ROCKE_ISA_BACKEND_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t */
#include "rocke/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------- backend kind */

/* Selects the concrete ISA backend behaviour. The CDNA kinds share datalayout,
 * triple and the gfx9/10 split s_waitcnt layout (hardware-verified for the base
 * f16 GEMM on gfx942/gfx950); the RDNA kinds diverge on buffer word3, the
 * contiguous s_waitcnt layout, and WMMA emission. */
typedef enum rocke_isa_kind
{
    ROCKE_ISA_GFX9_MFMA = 0, /* gfx908 / gfx90a / gfx942 (CDNA MFMA)             */
    ROCKE_ISA_GFX950, /* gfx950 (CDNA4); historical byte-identical default */
    ROCKE_ISA_GFX11_RDNA, /* gfx11 / gfx1151 / gfx11-generic (RDNA3/3.5 WMMA)  */
    ROCKE_ISA_GFX12_RDNA /* gfx12 / gfx1201 (RDNA4 WMMA, 8-wide fragments)    */
} rocke_isa_kind_t;

/* The s_waitcnt field layout family. Mirrors the two encoder functions in
 * lower_llvm (_encode_waitcnt_gfx9_10 vs _encode_waitcnt_gfx11). */
typedef enum rocke_waitcnt_layout
{
    ROCKE_WAITCNT_GFX9_10 = 0, /* split VMCNT [3:0]+[15:14]                       */
    ROCKE_WAITCNT_GFX11 /* contiguous expcnt[2:0]/lgkmcnt[9:4]/vmcnt[15:10] */
} rocke_waitcnt_layout_t;

/* CDNA (gfx9) buffer resource DWORD3: 32-bit-uint, bounds-checked. Matches CK
 * Tile's hardcoded gfx9 constant. */
#define ROCKE_BUFFER_RSRC_WORD3_CDNA 0x00027000u
/* RDNA (gfx10/11/12) "raw" SRD word3. */
#define ROCKE_BUFFER_RSRC_WORD3_RDNA 0x31014000u

/* The resolved ISA backend. Cheap value type; copyable. `gfx` points at an
 * interned static string owned by the registry (never freed). */
typedef struct rocke_isa_backend
{
    rocke_isa_kind_t kind;
    const char* gfx; /* "gfx942", "gfx1151", ...             */
    int vmcnt_bits; /* arch.vmcnt_bits fact (4 or 6)        */
    uint32_t buffer_rsrc_word3;
    rocke_waitcnt_layout_t waitcnt_layout;
    int wave_size; /* 64 (CDNA) / 32 (RDNA)                */
    bool valid; /* false => resolution failed           */
} rocke_isa_backend_t;

/* ------------------------------------------------------------- registry */

/* Resolve a gfx string to its ISA backend (Python backend_for + BACKEND_REGISTRY).
 * On an unknown gfx, returns a backend with .valid == false and leaves *err (if
 * non-NULL) pointing at a static "no ISA backend registered for ..." message;
 * callers should check .valid. Known gfx: gfx908, gfx90a, gfx942, gfx950,
 * gfx1151, gfx1201, gfx11-generic. */
rocke_isa_backend_t rocke_backend_for(const char* gfx, const char** err);

/* True if `gfx` has a registered backend. */
bool rocke_backend_is_known(const char* gfx);

/* ------------------------------------------------------- module preamble */

/* The shared LLVM target triple ("amdgcn-amd-amdhsa"). Flavor-invariant. */
const char* rocke_isa_triple(const rocke_isa_backend_t* be);
/* The LLVM datalayout string for `flavor` (Python backend.datalayout(flavor)).
 * Only the buffer-fat-pointer address space (p8) drifts between LLVM 20 (ROCm
 * 7.0/7.1) and LLVM 22 (ROCm >= 7.2). */
const char* rocke_isa_datalayout_for_flavor(const rocke_isa_backend_t* be,
                                            rocke_llvm_flavor_t flavor);
/* Flavor-agnostic accessor: the historical shared form (LLVM20). New callers
 * should prefer rocke_isa_datalayout_for_flavor. */
const char* rocke_isa_datalayout(const rocke_isa_backend_t* be);
/* The two leading IR lines for `flavor` (Python backend.module_preamble):
 * `target datalayout = "..."` + newline + `target triple = "..."`. Written
 * into `out` (cleared first). Returns 0 on success, -1 on OOM. */
int rocke_isa_module_preamble_for_flavor(const rocke_isa_backend_t* be,
                                         rocke_llvm_flavor_t flavor,
                                         rocke_strbuf_t* out);
/* Flavor-agnostic preamble (LLVM20 datalayout). */
int rocke_isa_module_preamble(const rocke_isa_backend_t* be, rocke_strbuf_t* out);

/* DWORD3 of the buffer resource descriptor for this target. */
uint32_t rocke_isa_buffer_rsrc_word3(const rocke_isa_backend_t* be);

/* --------------------------------------------------------------- s_waitcnt */

/* Encode an s_waitcnt immediate for this backend's layout. `vmcnt`/`expcnt`/
 * `lgkmcnt` of -1 mean "no wait" (encoded as the field maximum); explicit
 * values clamp to the field maximum (never wrap). Byte-identical to the
 * corresponding lower_llvm encoder for the backend's waitcnt layout. */
int rocke_isa_encode_waitcnt(const rocke_isa_backend_t* be, int vmcnt, int expcnt, int lgkmcnt);

/* Standalone encoders (the two lower_llvm functions), exposed for callers that
 * have a layout but no backend handle. */
int rocke_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt);
int rocke_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt);

/* --------------------------------------------------------------- WMMA tables */

/* One row of the RDNA WMMA spec tables (_RDNA_WMMA / _RDNA_GFX12_WMMA): the
 * float-path mapping op.name -> (decl key, mangled intrinsic, SSA operand
 * element type, call-site operand element type). When call_elt != ssa_elt the
 * emitter bitcasts each operand vector before the call (bf16). `frag_width` is
 * the per-lane A/B operand vector width (16 for RDNA3/3.5, 8 for RDNA4). */
typedef struct rocke_wmma_spec
{
    const char* op_name; /* "tile.wmma_f32_16x16x16_f16"                   */
    const char* decl_key; /* _need() key, e.g. "wmma.f32.16x16x16.f16"      */
    const char* intrinsic; /* fully-mangled @llvm.amdgcn.wmma....            */
    const char* ssa_elt; /* SSA operand element type ("half"/"bfloat")     */
    const char* call_elt; /* call-site operand element type ("half"/"i16")  */
    int frag_width; /* A/B operand vector width (16 or 8)             */
} rocke_wmma_spec_t;

/* One row of the integer WMMA table (_RDNA_WMMA_INT): op.name ->
 * (decl key, mangled intrinsic, A/B i32-vector width, accumulator width). */
typedef struct rocke_wmma_int_spec
{
    const char* op_name;
    const char* decl_key;
    const char* intrinsic;
    int op_vec; /* A/B <op_vec x i32> width (4 for iu8, 2 for iu4) */
    int acc_vec; /* accumulator/result <acc_vec x i32> width (8)    */
} rocke_wmma_int_spec_t;

/* Look up the RDNA3/3.5 float WMMA spec for an op name (incl. leading "tile.").
 * Returns NULL if not in _RDNA_WMMA. */
const rocke_wmma_spec_t* rocke_isa_wmma_lookup(const char* op_name);
/* Look up the RDNA3/3.5 integer WMMA spec (_RDNA_WMMA_INT). NULL if absent. */
const rocke_wmma_int_spec_t* rocke_isa_wmma_int_lookup(const char* op_name);
/* Look up the RDNA4 (gfx12) float WMMA spec (_RDNA_GFX12_WMMA). NULL if absent. */
const rocke_wmma_spec_t* rocke_isa_wmma_gfx12_lookup(const char* op_name);

/* Resolve the float WMMA spec the way `backend.emit_wmma` would for this
 * backend's kind: RDNA4 consults _RDNA_GFX12_WMMA; RDNA3/3.5 consults
 * _RDNA_WMMA. Returns NULL for CDNA kinds or unknown op names. */
const rocke_wmma_spec_t* rocke_isa_resolve_wmma(const rocke_isa_backend_t* be, const char* op_name);

/* ------------------------------------------------------- WMMA call emission */

/* Emit the byte-identical text of a float WMMA call, given the SSA operand
 * NAME strings (e.g. "%v12") rather than live Value/lowerer objects. This is
 * the self-contained core of Gfx11/Gfx12 emit_wmma:
 *
 *   [optional bf16 bitcasts]
 *     %<a_cast> = bitcast <W x ssa_elt> a_name to <W x call_elt>
 *     %<b_cast> = bitcast <W x ssa_elt> b_name to <W x call_elt>
 *   %<result_name> = call <8 x float> @<intrinsic>(
 *     <W x call_elt> a, <W x call_elt> b, <8 x float> c)
 *
 * where W = spec->frag_width. When call_elt == ssa_elt no bitcast is emitted
 * and a_cast_name/b_cast_name are ignored. The caller supplies the fresh
 * bitcast names (Python lowerer._fresh("wmma_a"/"wmma_b")); they may be NULL
 * when no bitcast is needed. Lines are appended to `out` (NOT cleared) with the
 * exact two-space indentation the Python emitter uses. Returns 0 on success,
 * ROCKE_ERR_NOTIMPL if the spec needs a bitcast but a cast name is missing, or
 * -1 on OOM. */
int rocke_isa_emit_wmma_call(rocke_strbuf_t* out,
                             const rocke_wmma_spec_t* spec,
                             const char* result_name,
                             const char* a_name,
                             const char* b_name,
                             const char* c_name,
                             const char* a_cast_name,
                             const char* b_cast_name);

/* Emit the byte-identical text of an integer WMMA (iu8/iu4) call. Mirrors
 * Gfx11Backend._emit_wmma_int: signedA=1, signedB=1, clamp=0, no bitcast.
 *   %<result_name> = call <acc_vec x i32> @<intrinsic>(
 *     i1 1, <op_vec x i32> a, i1 1, <op_vec x i32> b,
 *     <acc_vec x i32> c, i1 0)
 * Appended to `out`. Returns 0 on success, -1 on OOM. */
int rocke_isa_emit_wmma_int_call(rocke_strbuf_t* out,
                                 const rocke_wmma_int_spec_t* spec,
                                 const char* result_name,
                                 const char* a_name,
                                 const char* b_name,
                                 const char* c_name);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ISA_BACKEND_H */
