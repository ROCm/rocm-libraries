/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/lower_hip_internal.h -- PRIVATE shared declarations for the C99 port of
 * rocke.core.lower_hip. NOT a public API: only the lower_hip_*.c translation
 * units include this. The public contract is rocke/lower_hip.h.
 *
 * Everything here is cross-bucket shared state or helpers used by the parallel
 * body files (lower_hip_core.c, lower_hip_cast.c, lower_hip_mem.c,
 * lower_hip_mma.c). The DEFINITIONS of all functions and the prologue constant
 * declared here live in bucket 0 (lower_hip_core.c); the other buckets only
 * call them. Per-op handlers (`rocke_h_op_<dotted_with_underscores>`) are NOT
 * declared here individually -- each bucket defines its own static handlers and
 * registers them through the dispatch table that rocke_h_lower_op() consults
 * (see rocke_h_dispatch below). Cross-cutting handlers reachable from another
 * bucket (e.g. tile.mma re-dispatching to a concrete mfma handler) go through
 * rocke_h_lower_op(), so no per-handler externs are needed.
 *
 * Naming: internal helpers are prefixed rocke_h_ (h = hip lowerer) to keep them
 * out of the public rocke_ / rocke_b_ / rocke_lower_ namespace.
 */
#ifndef ROCKE_LOWER_HIP_INTERNAL_H
#define ROCKE_LOWER_HIP_INTERNAL_H

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/strbuf.h"
#include "rocke/vec.h"

/* The HIP lowerer's private symbols live in the internal C++ namespace ckc; the
 * public entry (rocke_lower_kernel_to_hip in rocke/lower_hip.h) stays at global
 * scope under extern "C". Only the lower_hip_*.c units include this and they
 * open `namespace ckc` around their bodies. */
namespace ckc
{

/* --------------------------------------------------------- lowerer state */

/* The C analog of the Python _Lowerer instance. One per lower_kernel_to_hip
 * call; lives on the stack of bucket 0's driver and is threaded by pointer into
 * every per-op handler. `lines` accumulates the indented body statements and
 * `smem_decls` the hoisted `__shared__` declarations -- both joined with '\n'
 * at the end exactly like Python's self.lines / self.smem_decls lists.
 *
 * `arena` (borrowed from the builder) backs the per-handler temporaries and the
 * two strbuf-vectors. `err`/`status` are the sticky error channel; the first
 * failing handler sets them and every later handler/_emit becomes a no-op, so
 * handlers can be written straight-line and checked once by the driver. */
typedef struct rocke_h_lowerer
{
    const rocke_kernel_def_t* kernel;
    rocke_ir_builder_t* b; /* arena + error channel (borrowed)         */
    rocke_hip_arch_t arch; /* resolved arch seam                       */

    ROCKE_VEC(char*) lines; /* body statements (arena-owned cstrs) */
    ROCKE_VEC(char*) smem_decls; /* hoisted __shared__ decls            */
    int indent; /* current indent depth (starts at 1)  */
    int smem_counter; /* mirrors Python self._smem_counter   */

    rocke_status_t status; /* sticky; ROCKE_OK until first failure   */
    char err[ROCKE_ERR_MSG_CAP];
} rocke_h_lowerer_t;

/* --------------------------------------------------------- error / liveness */

/* Raise the failure as a ckc::Error (mirroring the Python raise
 * NotImplementedError/RuntimeError paths); printf-style. [[noreturn]]: it never
 * returns -- the throw unwinds to the lowerer boundary in core.c, which
 * translates it back into the status code, so the extern "C" ABI is unchanged.
 * The rocke_status_t return type is retained so existing `return rocke_h_fail(...)`
 * call sites stay valid -- the returned value is simply never produced. */
[[noreturn]] rocke_status_t
    rocke_h_fail(rocke_h_lowerer_t* lw, rocke_status_t st, const char* fmt, ...);

/* True iff the lowerer is usable. Internal ops now raise on failure rather than
 * latching a sticky status, so on any reachable path a non-NULL lowerer is OK;
 * this collapses to a NULL guard. */
bool rocke_h_live(const rocke_h_lowerer_t* lw);

/* --------------------------------------------------------- emission / indent */

/* Append one body statement with the current indent prefix (Python _emit:
 * " " * indent + text). `text` is copied into the arena. No-op on a failed
 * lowerer. */
void rocke_h_emit(rocke_h_lowerer_t* lw, const char* text);

/* printf-style rocke_h_emit: format into the arena, then emit with indent. The
 * dominant form, since most handlers build one f-string per statement. */
void rocke_h_emitf(rocke_h_lowerer_t* lw, const char* fmt, ...);

/* Append a raw `__shared__` declaration line to smem_decls (already includes
 * its own 4-space prefix, matching Python). Copied into the arena. */
void rocke_h_emit_smem_decl(rocke_h_lowerer_t* lw, const char* decl);

void rocke_h_push_indent(rocke_h_lowerer_t* lw);
void rocke_h_pop_indent(rocke_h_lowerer_t* lw);

/* --------------------------------------------------------- dispatch / walk */

/* Lower a single op: map opcode -> handler and invoke it. Mirrors Python
 * _Lowerer.lower_op (getattr dispatch). Reachable from every bucket because
 * cross-cutting ops (tile.mma) re-enter the dispatcher to reach a concrete
 * handler in another bucket. Sets ROCKE_ERR_NOTIMPL for an unhandled opcode. */
rocke_status_t rocke_h_lower_op(rocke_h_lowerer_t* lw, const rocke_op_t* op);

/* Lower every op in a region in order (Python lower_region). */
rocke_status_t rocke_h_lower_region(rocke_h_lowerer_t* lw, const rocke_region_t* region);

/* A per-op handler: emits the C++ statement(s) for `op` into `lw`. Returns the
 * lowerer status (ROCKE_OK or the error it just set). */
typedef rocke_status_t (*rocke_h_handler_fn)(rocke_h_lowerer_t* lw, const rocke_op_t* op);

/* Look up the handler for an opcode, or NULL if none. The table is assembled in
 * bucket 0 from each bucket's exported registration array (rocke_h_handlers_*),
 * so adding an op family means adding a registration entry, not editing a
 * central switch. NULL handler => NotImplementedError parity. */
rocke_h_handler_fn rocke_h_dispatch(rocke_opcode_t opcode);

/* Per-bucket handler registration. Each non-core bucket exposes a static array
 * of (opcode, handler) pairs via these accessors; bucket 0 stitches them into
 * the dispatch table at first use. (opcode==ROCKE_OP_INVALID terminates a table.)
 */
typedef struct rocke_h_handler_entry
{
    rocke_opcode_t opcode;
    rocke_h_handler_fn handler;
} rocke_h_handler_entry_t;

/* Bucket 1 (casts/conversions + gpu) handler table. */
const rocke_h_handler_entry_t* rocke_h_handlers_cast(void);
/* Bucket 2 (memory / LDS / buffer / async / atomics) handler table. */
const rocke_h_handler_entry_t* rocke_h_handlers_mem(void);
/* Bucket 3 (mma / cross-lane / barriers / vector / control flow) handler table. */
const rocke_h_handler_entry_t* rocke_h_handlers_mma(void);
/* Bucket 0 (arith scalar/int/float/cmp + math) handler table. */
const rocke_h_handler_entry_t* rocke_h_handlers_arith(void);

/* --------------------------------------------------------- naming / types */

/* Python _name(v): strip a leading '%' from an SSA value name. Returns an
 * arena-owned copy. NULL-safe (returns ""). */
const char* rocke_h_name(rocke_h_lowerer_t* lw, const rocke_value_t* v);

/* Python _type_to_hip(t): canonical HIP C++ type spelling for an IR type.
 *   scalar -> "int"/"float"/"fp16"/"bf16"/"int8_t"/... (the _HIP_TYPE map)
 *   ptr<p,{global,lds}> -> "<pointee>*"
 *   vec<elem x n> -> "f16xN"/"bf16xN"/"f32xN"/"i32xN"/"i16xN"/"i8xN"
 *                    (fp8e4m3/bf8e5m2 -> "i8xN"; i1 -> "boolxN")
 *   smem<elem,...> -> "<elem>*"
 * Returns an arena-owned string. Sets the sticky error and returns "" for an
 * unmappable type (Python KeyError parity). */
const char* rocke_h_type_to_hip(rocke_h_lowerer_t* lw, const rocke_type_t* t);

/* Map a scalar IR type *name* ("i32","f16",...) to its HIP scalar spelling
 * (the raw _HIP_TYPE dict). Returns NULL on an unknown name (caller decides
 * whether that is an error). Used where Python indexes _HIP_TYPE[name] directly
 * (e.g. constants, vector.sum element type). */
const char* rocke_h_hip_scalar(const char* ir_scalar_name);

/* The "f16x"/"bf16x"/"f32x"/"i32x"/"i16x"/"i8x" vector-name prefix for an IR
 * scalar element name, used by the smem/global vN load/store handlers. Returns
 * the Python fallback "f16x" for names not in the small map (matching the
 * .get(elem,"f16x") idiom). Never NULL. `full_map` selects between the small
 * 2-entry {f16,bf16} map (false, used by global_load_vN etc.) and the full
 * 8-entry map including f32/i32/i16/i8/fp8/bf8 (true, used by smem_store_vN). */
const char* rocke_h_vec_prefix(const char* ir_scalar_name, bool full_map);

/* --------------------------------------------------------- literals / encode */

/* Python _f32_literal(val): C++ float-literal text for a double, special-casing
 * nan -> "((float)NAN)", +/-inf -> "((float)+/-INFINITY)", else "<repr>f".
 * NOTE(port): Python emits repr(float) for the finite case; matching that text
 * byte-for-byte is a known port hazard (see notes / unported). Returns an
 * arena-owned string. */
const char* rocke_h_f32_literal(rocke_h_lowerer_t* lw, double val);

/* Python _encode_waitcnt: arch-aware s_waitcnt immediate. Selects the gfx9/10
 * split-VMCNT layout or the gfx11 contiguous layout off lw->arch. -1 on any
 * counter means "no wait" (encoded as that field's architectural max). */
int rocke_h_encode_waitcnt(const rocke_h_lowerer_t* lw, int vmcnt, int expcnt, int lgkmcnt);

/* The two raw encoders behind rocke_h_encode_waitcnt, exposed for any direct
 * callers / tests (Python _encode_waitcnt_gfx9_10 / _encode_waitcnt_gfx11). */
int rocke_h_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt);
int rocke_h_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt);

/* --------------------------------------------------------- arch gates */

/* Python _require_wmma_arch: error out if WMMA is requested on a target with no
 * WMMA instruction (CDNA/MFMA). Returns ROCKE_OK if allowed, else sets the sticky
 * error (ROCKE_ERR_NOTIMPL) and returns it. `op_id` is the atom id for the message.
 * NOTE(port): Python keys off the ArchTarget MMA catalog (mma.by_op_id); that
 * catalog is not ported, so this uses lw->arch.has_wmma instead. */
rocke_status_t rocke_h_require_wmma_arch(rocke_h_lowerer_t* lw, const char* op_id);

/* Python _require_ds_read_tr: error out if a transpose LDS read is requested on
 * a target without ds_read_*_tr_* (keys off lw->arch.has_ds_read_tr). */
rocke_status_t rocke_h_require_ds_read_tr(rocke_h_lowerer_t* lw, const char* op_id);

/* --------------------------------------------------------- smem token helpers */

/* The Python handlers stash the `__shared__` storage symbol name and shape on
 * the producing tile.smem_alloc op (op.attrs["_storage"] / ["_shape"]) so later
 * smem ops can index it. In C, op.attrs is the frozen rocke_attr_map_t which we
 * MUST NOT mutate on a const IR; instead the lowerer maintains a side table
 * keyed by the smem result Value pointer. These two functions are that table.
 *
 * rocke_h_smem_set_storage records "<name>_storage" for the smem alloc result;
 * rocke_h_smem_storage looks it up for a consuming op's smem operand. A lookup
 * miss is the Python "smem op before smem_alloc was lowered" RuntimeError case:
 * the caller should rocke_h_fail. Returns NULL on miss. */
void rocke_h_smem_set_storage(rocke_h_lowerer_t* lw,
                              const rocke_value_t* smem_result,
                              const char* storage_name);
const char* rocke_h_smem_storage(rocke_h_lowerer_t* lw, const rocke_value_t* smem_result);

} /* namespace ckc */

#endif /* ROCKE_LOWER_HIP_INTERNAL_H */
