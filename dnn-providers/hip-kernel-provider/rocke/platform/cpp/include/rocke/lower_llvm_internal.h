/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/lower_llvm_internal.h -- PRIVATE shared declarations for the C99 port of
 * rocke.core.lower_llvm. NOT a public API: only the lower_llvm_*.c translation
 * units include this. The public contract is rocke/lower_llvm.h.
 *
 * The Python lowerer is one stateful object, ``_Lowerer``, with ~140 per-op
 * methods, a block/CFG model, a flavor-keyed intrinsic-declaration table, an
 * smem-global pre-pass, a scf.for/if CFG builder, and an ISA backend. This
 * header is the cross-bucket surface: the ``rocke_lower_t`` state struct, the
 * ``_Block`` model, the ISA backend, the op-dispatch table, and every helper a
 * parallel body file calls.
 *
 * DEFINITIONS of everything declared here live in BUCKET 0
 * (lower_llvm_core.c). The other buckets only call them. The per-op handlers
 * are NOT declared here individually; they are reached through the dispatch
 * table that bucket 0 builds (rocke_ll_dispatch), and each bucket registers its
 * own handlers via a per-bucket rocke_ll_register_<bucket>() hook (see below).
 *
 * Naming: internal helpers are prefixed rocke_ll_ (ll = lower_llvm) to keep them
 * out of the public rocke_ / rocke_b_ namespace.
 *
 * These declarations live in the internal C++ namespace ckc (the engine's
 * private symbols). The public lowerer entry points (rocke_lower_kernel_to_llvm
 * in rocke/lower_llvm.h) stay at global scope under extern "C" -- they are the
 * stable ABI. Everything here is private to the lower_llvm_*.c translation
 * units, which open `namespace ckc` around their bodies.
 */
#ifndef ROCKE_LOWER_LLVM_INTERNAL_H
#define ROCKE_LOWER_LLVM_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/arena.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"
#include "rocke/vec.h"

namespace ckc
{

/* ====================================================================== */
/* Constants (module-level Python data)                                   */
/* ====================================================================== */

/* Verbatim datalayout / triple copied from clang for gfx950 (Python
 * _DATALAYOUT_LLVM20 / _DATALAYOUT_LLVM22 / _TRIPLE). Shared by all CDNA
 * backends today, but the AMDGPU datalayout is FLAVOR-KEYED: only the
 * buffer-fat-pointer address space (p8) drifts between LLVM 20 (ROCm 7.0/7.1)
 * and LLVM 22 (ROCm >= 7.2). ROCKE_LL_DATALAYOUT is a back-compat alias for the
 * LLVM20 form; new code keys on the flavor via rocke_ll_datalayout_for_flavor. */
extern const char* const ROCKE_LL_DATALAYOUT_LLVM20;
extern const char* const ROCKE_LL_DATALAYOUT_LLVM22;
extern const char* const ROCKE_LL_DATALAYOUT; /* == ROCKE_LL_DATALAYOUT_LLVM20 */
extern const char* const ROCKE_LL_TRIPLE;

/* Python _datalayout_for_flavor: LLVM20 => legacy p8 layout, anything else
 * (incl. unexpected values) => the modern LLVM22 layout. */
const char* rocke_ll_datalayout_for_flavor(rocke_llvm_flavor_t flavor);

/* CDNA buffer-resource-descriptor DWORD3 (Python ISABackend.buffer_rsrc_word3
 * == 0x00027000). RDNA word3 differs (0x31014000) -- see backend struct. */
#define ROCKE_LL_BUFFER_RSRC_WORD3_CDNA 0x00027000
/* RDNA (gfx10/11/12) "raw" SRD DWORD3 (Python Gfx11RdnaBackend.buffer_rsrc_word3
 * == 0x31014000). gfx11/gfx12 share this value. */
#define ROCKE_LL_BUFFER_RSRC_WORD3_RDNA 0x31014000

/* ====================================================================== */
/* Intrinsic-declaration table                                            */
/* ====================================================================== */

/* One row of the _INTRINSIC_DECLS dict: a stable key (the _need(...) lookup
 * key, flavor-agnostic) and the LLVM `declare` text. Insertion order in the
 * table array drives finalize() emit order, exactly like the Python dict. */
typedef struct rocke_ll_decl
{
    const char* key;
    const char* decl; /* may be overridden per-flavor; see overrides table */
} rocke_ll_decl_t;

/* The base (LLVM20) declaration table and its length (Python _INTRINSIC_DECLS,
 * insertion-ordered). */
extern const rocke_ll_decl_t ROCKE_LL_INTRINSIC_DECLS[];
extern const int ROCKE_LL_INTRINSIC_DECLS_COUNT;

/* The LLVM22 overrides (Python _INTRINSIC_DECLS_LLVM22_OVERRIDES): same keys,
 * different decl text. */
extern const rocke_ll_decl_t ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[];
extern const int ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES_COUNT;

/* ====================================================================== */
/* ISA backend (the gfx-keyed LLVM details)                               */
/* ====================================================================== */

/* The C analog of rocke.core.isa.backend.ISABackend. For the CDNA targets we
 * port (gfx942 / gfx950) every field is a shared constant or one of two
 * waitcnt encoders, so a plain struct with a function pointer for the waitcnt
 * encoder suffices -- no vtable explosion. RDNA WMMA emission is out of scope
 * (the FROZEN ir.h exposes no WMMA opcodes); emit_mma routes tile.mma to the
 * matching tile.<op_id> CDNA handler.
 *
 * encode_waitcnt: -1 for a counter means "no wait" (architectural max). */
/* RDNA-family discriminator for the lowering path. CDNA targets reject WMMA;
 * RDNA3/3.5 (gfx11) and RDNA4 (gfx12) emit WMMA. The gfx12 op_ids are distinct
 * (``wmma_gfx12_*``), so the kind only needs to separate CDNA from RDNA-any. */
typedef enum rocke_ll_isa_kind
{
    ROCKE_LL_ISA_CDNA = 0, /* gfx908/gfx90a/gfx942/gfx950 (MFMA)         */
    ROCKE_LL_ISA_RDNA /* gfx11 / gfx12 (WMMA)                       */
} rocke_ll_isa_kind_t;

typedef struct rocke_isa_backend
{
    const char* gfx; /* "gfx950", "gfx942", ...                     */
    const char* datalayout; /* ROCKE_LL_DATALAYOUT                           */
    const char* triple; /* ROCKE_LL_TRIPLE                               */
    int buffer_rsrc_word3;
    int (*encode_waitcnt)(int vmcnt, int expcnt, int lgkmcnt);
    rocke_ll_isa_kind_t kind; /* CDNA (reject WMMA) vs RDNA (emit WMMA)      */
} rocke_isa_backend_t;

/* Resolve a gfx string to its backend (Python backend_for). NULL => "gfx950".
 * Returns NULL and sets *st on an unknown arch. */
const rocke_isa_backend_t* rocke_ll_backend_for(const char* arch, rocke_status_t* st);

/* The two CDNA waitcnt encoders (Python _encode_waitcnt_gfx9_10 /
 * _encode_waitcnt_gfx11). gfx11 is registered for completeness; the CDNA
 * lowering path uses gfx9_10. */
int rocke_ll_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt);
int rocke_ll_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt);

/* ====================================================================== */
/* Block / CFG model (Python _Block)                                      */
/* ====================================================================== */

/* A basic block: a label and a growable list of emitted IR lines, plus a
 * `terminated` flag. Lines are arena-owned strdup'd strings (Python list of
 * str). Mirrors _Block; `emit` is rocke_ll_block_emit. */
typedef struct rocke_ll_block
{
    const char* label;
    ROCKE_VEC(char*) lines; /* arena-backed; each line includes no trailing \n */
    bool terminated;
} rocke_ll_block_t;

/* ====================================================================== */
/* Lowerer state (Python _Lowerer)                                        */
/* ====================================================================== */

/* One entry of the smem-global pre-pass: the @global name and its smem type. */
typedef struct rocke_ll_smem_global
{
    const char* gname; /* "@<short>.<kernel>" */
    const rocke_type_t* stype; /* the SmemType (kind == ROCKE_TYPE_SMEM)         */
} rocke_ll_smem_global_t;

/* IR-value-name -> @global-name mapping for smem allocs (Python
 * _smem_storage_name dict). Linear lookup (few allocs per kernel). */
typedef struct rocke_ll_smem_name
{
    const char* value_name; /* the IR Value name (with leading '%')        */
    const char* gname;
} rocke_ll_smem_name_t;

/* A "needed intrinsic" record: the decl key plus the resolved decl text. The
 * decl text is captured at _need() time so dynamically-built decls (Python
 * self._decls[intrin] = ... for vector smax) are preserved in emit order. */
typedef struct rocke_ll_need
{
    const char* key;
    const char* decl; /* the flavor-resolved declaration line        */
} rocke_ll_need_t;

/* The full lowerer state. Allocated on the stack of the entry point; its arena
 * owns every transient string/array. The strbuf `out` (in finalize) is the one
 * heap buffer. */
typedef struct rocke_lower
{
    rocke_arena_t arena; /* owns blocks, lines, fresh names    */
    const rocke_kernel_def_t* kernel;
    const rocke_isa_backend_t* backend;
    rocke_llvm_flavor_t flavor; /* resolved (never AUTO once running) */

    /* block model */
    ROCKE_VEC(rocke_ll_block_t*) blocks; /* blocks[len-1] is _current()       */
    int block_counter;
    int tmp_counter;

    /* needed intrinsics, in first-need order (drives a sorted-by-table emit in
     * finalize; the table order is canonical, this set records membership). */
    ROCKE_VEC(rocke_ll_need_t) needs;
    bool needs_fp_atomic_md; /* _needs_fp_atomic_md      */

    /* dynamically-registered decls (Python self._decls mutation, e.g. vector
     * smax registers "llvm.smax.vNiW"). Keyed; consulted by _need fallback. */
    ROCKE_VEC(rocke_ll_decl_t) dyn_decls;

    /* smem pre-pass */
    ROCKE_VEC(rocke_ll_smem_global_t) smem_globals;
    ROCKE_VEC(rocke_ll_smem_name_t) smem_names;

    /* scf.for yield recording stack (Python _yield_stack: list of list[str]).
     * Each frame is a vector of operand strings. */
    ROCKE_VEC(ROCKE_VEC(const char*) *) yield_stack;

    /* unroll trailing-sync elision marker (Python _unroll_elide_sync_op):
     * points at the specific tile.sync op to skip, or NULL. */
    const rocke_op_t* unroll_elide_sync_op;

    /* sticky error (the lowerer has no builder to carry it). */
    rocke_status_t status;
    char* err; /* arena-owned, ROCKE_ERR_MSG_CAP cap   */
} rocke_lower_t;

/* ====================================================================== */
/* Error model                                                            */
/* ====================================================================== */

/* Raise the lowering failure as a ckc::Error (mirroring the Python `raise`),
 * printf style. [[noreturn]]: it never returns -- the throw unwinds to the
 * lowerer boundary (rocke_lower_kernel_to_llvm), which translates it back into the
 * legacy status code + caller `err` buffer, so the extern "C" ABI is unchanged.
 * Any statement following a rocke_ll_fail() call is therefore unreachable. */
[[noreturn]] void rocke_ll_fail(rocke_lower_t* L, rocke_status_t st, const char* fmt, ...);

/* True if the lowerer is usable (non-null). The lowerer no longer carries a
 * sticky error -- a failure raises instead -- so this is just a null guard. */
bool rocke_ll_live(const rocke_lower_t* L);

/* ====================================================================== */
/* Core plumbing (Python _Lowerer helpers) -- DEFINED IN BUCKET 0         */
/* ====================================================================== */

/* Current (top-of-stack) block (Python _current). Never NULL on a live L. */
rocke_ll_block_t* rocke_ll_current(rocke_lower_t* L);

/* Push a new block "<base>.<++block_counter>" (Python _new_block) and return
 * it; it becomes _current. */
rocke_ll_block_t* rocke_ll_new_block(rocke_lower_t* L, const char* base);

/* Block at index `idx` in L->blocks (helper for the half-block / for-CFG
 * back-patching that the Python code does via self._blocks[i]). */
rocke_ll_block_t* rocke_ll_block_at(rocke_lower_t* L, int idx);
int rocke_ll_block_count(const rocke_lower_t* L);

/* Append a line to block `blk` (Python _Block.emit): strdup's `line` into the
 * arena. Fails (sticky) if the block is already terminated. */
void rocke_ll_block_emit(rocke_lower_t* L, rocke_ll_block_t* blk, const char* line);
/* printf form of the above. */
void rocke_ll_block_emitf(rocke_lower_t* L, rocke_ll_block_t* blk, const char* fmt, ...);

/* Shorthand: emit into the CURRENT block (the dominant `self._current().emit`
 * idiom). printf form. */
void rocke_ll_emit(rocke_lower_t* L, const char* line);
void rocke_ll_emitf(rocke_lower_t* L, const char* fmt, ...);

/* Fresh temp SSA name "%<hint>.<++tmp_counter>" (Python _fresh). Arena-owned;
 * stable for the lowerer's lifetime. */
const char* rocke_ll_fresh(rocke_lower_t* L, const char* hint);

/* Mark an intrinsic as needed by its canonical key (Python _need). Resolves
 * the decl text now (flavor overrides + dyn_decls) so emit order is the table
 * order and dynamic decls survive. No-op if already present. */
void rocke_ll_need(rocke_lower_t* L, const char* key);

/* Register a dynamically-built decl (Python self._decls[key] = decl) then mark
 * it needed. Used by vector.smax. */
void rocke_ll_need_dynamic(rocke_lower_t* L, const char* key, const char* decl);

/* ====================================================================== */
/* Operand / type rendering (Python _operand, _operand_with_type,         */
/* _llvm_type, _smem_storage_type, constant folding) -- BUCKET 0          */
/* ====================================================================== */

/* Textual LLVM operand for a Value: inlines arith.constant literals (i/f32/f16
 * hex), else returns the SSA name. Mirrors _operand. Returned string is
 * arena-owned (constants formatted on demand). */
const char* rocke_ll_operand(rocke_lower_t* L, const rocke_value_t* v);

/* "<type> <operand>" (Python _operand_with_type). */
const char* rocke_ll_operand_with_type(rocke_lower_t* L, const rocke_value_t* v);

/* Map an IR Type to its LLVM textual form (Python _llvm_type). Sets NOTIMPL on
 * an unmapped type and returns "" . */
const char* rocke_ll_llvm_type(rocke_lower_t* L, const rocke_type_t* t);

/* Map an IR type-NAME string (from op.attrs, e.g. iter_args metadata) back to
 * LLVM text (Python _llvm_type_from_name). Handles scalars + "vec<exN>". */
const char* rocke_ll_llvm_type_from_name(rocke_lower_t* L, const char* name);

/* LLVM aggregate storage type for a SmemType: nested arrays of the element
 * (Python _smem_storage_type). Arena-owned. */
const char* rocke_ll_smem_storage_type(rocke_lower_t* L, const rocke_type_t* smem);

/* True if `v` is produced by arith.constant (Python _is_constant). */
bool rocke_ll_is_constant(const rocke_value_t* v);
/* Evaluate a constant Value to int64 (Python _eval_constant). Sets VALUE error
 * + returns 0 if not a constant. */
int64_t rocke_ll_eval_constant(rocke_lower_t* L, const rocke_value_t* v);

/* fp32 / fp16 LLVM hex constant spellings (Python _fp32_hex / _fp16_hex).
 * Arena-owned. */
const char* rocke_ll_fp32_hex(rocke_lower_t* L, double x);
const char* rocke_ll_fp16_hex(rocke_lower_t* L, double x);

/* Escape a string for an LLVM asm/string literal (Python
 * _escape_llvm_asm_string): printable ASCII verbatim, else \XX hex. */
const char* rocke_ll_escape_asm_string(rocke_lower_t* L, const char* s);

/* ====================================================================== */
/* smem-global helpers (Python _collect_smem, _smem_global_name)          */
/* ====================================================================== */

/* Pre-pass: walk `region` recursively recording every tile.smem_alloc as a
 * module-level addrspace(3) global (Python _collect_smem). */
void rocke_ll_collect_smem(rocke_lower_t* L, const rocke_region_t* region);

/* Look up the @global name + SmemType for an smem Value (Python
 * _smem_global_name / _smem_storage_name[...]). Sets KEY error + returns NULL
 * gname if the value was never allocated. `out_stype` may be NULL. */
const char* rocke_ll_smem_global_name(rocke_lower_t* L,
                                      const rocke_value_t* smem,
                                      const rocke_type_t** out_stype);

/* ====================================================================== */
/* yield-stack helpers (Python _yield_stack manipulation)                 */
/* ====================================================================== */

void rocke_ll_yield_push(rocke_lower_t* L); /* append [] frame      */
/* Pop the top frame; returns its operand-string vector via out params. */
void rocke_ll_yield_pop(rocke_lower_t* L, const char* const** out_items, int* out_count);
/* Record yielded operand strings into the top frame (scf.yield). */
void rocke_ll_yield_record(rocke_lower_t* L, const char* operand_str);
int rocke_ll_yield_depth(const rocke_lower_t* L);

/* ====================================================================== */
/* Op dispatch                                                            */
/* ====================================================================== */

/* A per-op handler. Mirrors a Python `_op_<name>(self, op)` method. */
typedef void (*rocke_ll_op_fn)(rocke_lower_t* L, const rocke_op_t* op);

/* Dispatch one op to its handler (Python lower_op): looks up by opcode in the
 * dispatch table built at init. Sets NOTIMPL for an op with no handler. */
void rocke_ll_lower_op(rocke_lower_t* L, const rocke_op_t* op);

/* Lower every op in a region (Python lower_region). */
void rocke_ll_lower_region(rocke_lower_t* L, const rocke_region_t* region);

/* The opcode-indexed handler table. Bucket 0 owns the storage; every bucket's
 * register hook installs its handlers into it. Indexed by rocke_opcode_t. */
extern rocke_ll_op_fn rocke_ll_dispatch[ROCKE_OP__COUNT];

/* Install a handler for an opcode (used by the per-bucket register hooks). */
void rocke_ll_set_handler(rocke_opcode_t opcode, rocke_ll_op_fn fn);

/* Per-bucket registration hooks. Bucket 0's init calls each of these once so
 * every bucket's handlers are present in rocke_ll_dispatch before lowering. Each
 * is DEFINED in its own bucket .c file. */
void rocke_ll_register_arith(void); /* bucket 1 */
void rocke_ll_register_convert(void); /* bucket 2 */
void rocke_ll_register_mem(void); /* bucket 3 */
void rocke_ll_register_mma(void); /* bucket 4 */
void rocke_ll_register_crosslane(void); /* bucket 5 */
void rocke_ll_register_vector(void); /* bucket 6 (also barriers/sched + flow)   */

/* ====================================================================== */
/* Shared multi-bucket op helpers (the Python private _lower_* helpers     */
/* that more than one handler family uses) -- DEFINED IN BUCKET 0          */
/* ====================================================================== */

/* Same-type binary op `%r = <llvm_op> <ty> a, b` (Python _binop). */
void rocke_ll_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op);

/* Vector same-type binary op (Python _vector_binop). */
void rocke_ll_vector_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op);

/* Shared FP8/BF8 MFMA lowering body (Python _lower_mfma_fp8_bf8): bitcasts the
 * <8 x i8> A/B to the flavor-correct packed type (i64 on LLVM22, <2 x i32> on
 * LLVM20) and emits the call. */
void rocke_ll_lower_mfma_fp8_bf8(
    rocke_lower_t* L, const rocke_op_t* op, const char* dtype, int out_vec, const char* intrinsic);

/* Shared horizontal vector reduce (Python _lower_vector_reduce): extract every
 * lane and fold with `llvm_op` starting from `init`. */
void rocke_ll_lower_vector_reduce(rocke_lower_t* L,
                                  const rocke_op_t* op,
                                  const char* llvm_op,
                                  const char* init);

/* Shared ballot emit (Python _emit_wave_ballot): `%r = ballot(pred != 0)`. */
void rocke_ll_emit_wave_ballot(rocke_lower_t* L,
                               const rocke_value_t* pred,
                               const char* result_name);

/* scf.for sub-lowerings (Python _lower_normal_for / _lower_unrolled_for). The
 * dispatcher (_op_scf_for) picks between them; both live in the flow bucket but
 * are declared here because the unrolled path mutates Value names which the
 * shared operand renderer must see -- keep them visible for the flow handlers.
 */
void rocke_ll_lower_normal_for(rocke_lower_t* L, const rocke_op_t* op);
void rocke_ll_lower_unrolled_for(rocke_lower_t* L, const rocke_op_t* op);

/* ====================================================================== */
/* finalize (Python finalize + _param_attrs + _format_agpr_alloc)         */
/* ====================================================================== */

/* Assemble the full module text into `out` (Python finalize). Terminates the
 * current block with `ret void`, then emits datalayout/triple, smem globals,
 * needed declares, the kernel define with its blocks, and the attributes /
 * metadata trailer. */
void rocke_ll_finalize(rocke_lower_t* L, rocke_strbuf_t* out);

/* Render a param's LLVM attribute suffix (Python _param_attrs). Arena-owned;
 * returns "" for non-pointer params. */
const char* rocke_ll_param_attrs(rocke_lower_t* L, const rocke_param_t* p);

/* Format a kernel agpr_alloc attr value "min,max" (Python _format_agpr_alloc).
 * Sets VALUE error on a bad pair; returns "" then. */
const char* rocke_ll_format_agpr_alloc(rocke_lower_t* L, const rocke_attr_value_t* v);

} /* namespace ckc */

#endif /* ROCKE_LOWER_LLVM_INTERNAL_H */
