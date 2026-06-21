/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/ir.h -- THE FROZEN IR CONTRACT for the C99 port of ck_dsl.core.ir.
 *
 * This header is the single source of truth every lowerer (lower_llvm,
 * lower_hip, ir_print, passes, isa/backend, arch) binds to. It is a faithful,
 * explicit translation of the Python SSA IR:
 *
 *   Python                         C99 (this header)
 *   ----------------------------   --------------------------------------------
 *   class Type (frozen)            ckc_type_t  (tagged union, kind discriminant)
 *     VectorType/PtrType/SmemType    -> same struct, kind = VECTOR/PTR/SMEM
 *   class Value (mutable)          ckc_value_t
 *   class Op (mutable)             ckc_op_t
 *   class Region                   ckc_region_t
 *   class Param                    ckc_param_t
 *   class KernelDef                ckc_kernel_def_t
 *   class IRBuilder                ckc_ir_builder_t
 *   op.name : str                  ckc_opcode_t enum (CKC_OP_*) + name table
 *   op.attrs : Dict[str, Any]      ckc_attr_map_t (sorted key -> variant value)
 *   raise ValueError/TypeError     ckc_status_t return code + builder->err msg
 *   **attrs kwargs                 explicit option structs (ckc_param_opts_t...)
 *   @property result/is_pure       ckc_op_result()/ckc_op_is_pure() getters
 *
 * Lifetime: every node returned by the builder is owned by the builder's arena
 * (ckc_ir_builder_t.arena). Nothing is freed individually; ckc_ir_builder_free()
 * (or arena reset) bulk-frees the whole graph -- mirroring Python GC lifetime.
 *
 * Error model: the builder is sticky-failing. The first operation that fails
 * sets builder->status != CKC_OK and records a message in builder->err; every
 * subsequent builder call is a no-op that returns NULL / the error status. This
 * lets kernel authors write straight-line builder code (as in Python) and check
 * ckc_ir_builder_ok() once at the end, instead of checking every call.
 */
#ifndef CKC_IR_H
#define CKC_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ckc/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ status */

typedef enum ckc_status
{
    CKC_OK = 0,
    CKC_ERR_VALUE,  /* maps to Python ValueError                          */
    CKC_ERR_TYPE,   /* maps to Python TypeError                           */
    CKC_ERR_KEY,    /* maps to Python KeyError (unknown op_id / param)    */
    CKC_ERR_OOM,    /* allocation failure                                 */
    CKC_ERR_NOTIMPL /* maps to Python NotImplementedError                 */
} ckc_status_t;

#define CKC_ERR_MSG_CAP 256

/* CKC_ERR_SNPRINTF -- snprintf into a bounded diagnostic/error buffer where
 * truncating an over-long message is INTENTIONAL (the buffer is a fixed
 * CKC_ERR_MSG_CAP-sized field; we never grow it for a long reason string).
 * snprintf is overflow-safe, so the only effect of truncation is a shortened
 * human-readable message -- never memory unsafety and never emitted IR (these
 * are reject/error paths). The localized pragma blesses exactly this idiom while
 * keeping -Werror=format-truncation active everywhere else, so any NEW,
 * unintended truncation (e.g. into a codegen name buffer) is still caught. */
#if defined(__GNUC__)
#define CKC_ERR_SNPRINTF(buf, cap, ...)                                     \
    do                                                                      \
    {                                                                       \
        _Pragma("GCC diagnostic push")                                      \
            _Pragma("GCC diagnostic ignored \"-Wformat-truncation\"")(void) \
                snprintf((buf), (cap), __VA_ARGS__);                        \
        _Pragma("GCC diagnostic pop")                                       \
    } while(0)
#else
#define CKC_ERR_SNPRINTF(buf, cap, ...) (void)snprintf((buf), (cap), __VA_ARGS__)
#endif

/* --------------------------------------------------------------- type kinds */

typedef enum ckc_type_kind
{
    CKC_TYPE_SCALAR = 0, /* i1/i8/i16/i32/i64/bf16/f16/f32/fp8e4m3/bf8e5m2     */
    CKC_TYPE_VECTOR,     /* vec<elem x count>                                  */
    CKC_TYPE_PTR,        /* ptr<pointee, space>                                */
    CKC_TYPE_SMEM        /* smem<elem, [shape...]>                             */
} ckc_type_kind_t;

/* Canonical scalar type tags. The scalar singletons (ckc_i32() etc.) carry one
 * of these so consumers can switch on the element kind without strcmp. */
typedef enum ckc_scalar_kind
{
    CKC_SCALAR_I1 = 0,
    CKC_SCALAR_I8,
    CKC_SCALAR_I16,
    CKC_SCALAR_I32,
    CKC_SCALAR_I64,
    CKC_SCALAR_BF16,
    CKC_SCALAR_F16,
    CKC_SCALAR_F32,
    CKC_SCALAR_FP8E4M3,
    CKC_SCALAR_BF8E5M2,
    CKC_SCALAR__COUNT
} ckc_scalar_kind_t;

/* A Type. `name` is the canonical textual form ("i32", "vec<f16x4>",
 * "ptr<f16,global>", "smem<f16, [64x32]>") -- byte-identical to Python so the
 * printer/lowerers reproduce existing output. Scalar types are interned
 * singletons; composite types are arena-allocated and value-compared by name. */
typedef struct ckc_type
{
    ckc_type_kind_t kind;
    const char* name; /* canonical, arena/static owned, never NULL */

    /* CKC_TYPE_SCALAR */
    ckc_scalar_kind_t scalar; /* valid iff kind == CKC_TYPE_SCALAR         */

    /* CKC_TYPE_VECTOR */
    const struct ckc_type* elem; /* element type (VECTOR and SMEM)            */
    int count;                   /* lane count (VECTOR)                       */

    /* CKC_TYPE_PTR */
    const struct ckc_type* pointee;
    const char* space; /* "global","constant",...                   */

    /* CKC_TYPE_SMEM */
    const int* shape; /* arena-owned array of dim sizes            */
    int rank;         /* number of dims in shape                   */
} ckc_type_t;

/* -------------------------------------------------------------- attr values */

typedef enum ckc_attr_kind
{
    CKC_ATTR_INT = 0, /* int64_t  (value, vec, align, rank, index, num, ...) */
    CKC_ATTR_FLOAT,   /* double   (fp constant value, fill)                  */
    CKC_ATTR_STR,     /* const char* (ity, pred, op_id, elem, elem_type,...) */
    CKC_ATTR_BOOL,    /* bool     (pure, unroll, elide_trailing_barrier,...) */
    CKC_ATTR_LIST,    /* nested attr list (scf.for iter_args metadata)       */
    CKC_ATTR_INT_LIST /* list of bare ints, e.g. agpr_alloc (0,0)            */
} ckc_attr_kind_t;

struct ckc_attr_map; /* forward: a list element is itself a small attr map */

typedef struct ckc_attr_value
{
    ckc_attr_kind_t kind;
    union
    {
        int64_t i;
        double f;
        const char* s; /* arena-owned                         */
        bool b;
        struct
        {
            struct ckc_attr_map** items; /* arena array of maps               */
            int count;
        } list;
        struct
        {
            int64_t* ints; /* arena array of bare ints (l:[ i:.., .. ])      */
            int count;
        } ilist;
    } u;
} ckc_attr_value_t;

typedef struct ckc_attr_entry
{
    const char* key; /* arena-owned                                    */
    ckc_attr_value_t value;
} ckc_attr_entry_t;

/* Op.attrs: an insertion-ordered key->variant map. Small (<=10 entries);
 * lookups are linear by key. ir_print sorts a copy for stable output. */
typedef struct ckc_attr_map
{
    ckc_attr_entry_t* entries; /* arena-owned, grows by reallocation in arena */
    int count;
    int cap;
} ckc_attr_map_t;

/* ----------------------------------------------------------------- opcodes */

/* One enumerator per distinct op name string in ck_dsl.core.ir. Every lowerer
 * dispatches on this enum instead of Python getattr(self, "_op_"+name). The
 * canonical dotted name string is recovered with ckc_opcode_name(). */
typedef enum ckc_opcode
{
    CKC_OP_INVALID = 0,

    /* arith.* */
    CKC_OP_ARITH_CONSTANT,
    CKC_OP_ARITH_CONSTANT_VEC,
    CKC_OP_ARITH_ADD,
    CKC_OP_ARITH_SUB,
    CKC_OP_ARITH_MUL,
    CKC_OP_ARITH_DIV,
    CKC_OP_ARITH_MOD,
    CKC_OP_ARITH_FADD,
    CKC_OP_ARITH_FSUB,
    CKC_OP_ARITH_FMUL,
    CKC_OP_ARITH_FDIV,
    CKC_OP_ARITH_FNEG,
    CKC_OP_ARITH_FABS,
    CKC_OP_ARITH_FMA,
    CKC_OP_ARITH_FMAX3,
    CKC_OP_ARITH_FMIN3,
    CKC_OP_ARITH_CMP,
    CKC_OP_ARITH_FCMP,
    CKC_OP_ARITH_FMAX,
    CKC_OP_ARITH_FMIN,
    CKC_OP_ARITH_AND,
    CKC_OP_ARITH_OR,
    CKC_OP_ARITH_NOT,
    CKC_OP_ARITH_SMAX,
    CKC_OP_ARITH_SMIN,
    CKC_OP_ARITH_XOR,
    CKC_OP_ARITH_SHL,
    CKC_OP_ARITH_LSHR,
    CKC_OP_ARITH_UMUL_HI_I32,
    CKC_OP_ARITH_ZEXT,
    CKC_OP_ARITH_SEXT,
    CKC_OP_ARITH_TRUNC,
    CKC_OP_ARITH_SELECT,
    CKC_OP_ARITH_BITCAST,
    CKC_OP_ARITH_TRUNC_F32_TO_F16,
    CKC_OP_ARITH_RINT_F32,
    CKC_OP_ARITH_CAST_TO_F32,
    CKC_OP_ARITH_CAST_F32_TO,
    CKC_OP_ARITH_SITOFP_F32,
    CKC_OP_ARITH_CVT_FP8_TO_F32,
    CKC_OP_ARITH_CVT_BF8_TO_F32,
    CKC_OP_ARITH_CVT_PK_F32_FP8X4,
    CKC_OP_ARITH_CVT_PK_F32_BF8X4,
    CKC_OP_ARITH_CVT_PK_FP8_F32X4,
    CKC_OP_ARITH_CVT_PK_BF8_F32X4,
    CKC_OP_ARITH_CVT_PK_I8_F32X4,
    CKC_OP_ARITH_CVT_F32_TO_FP8,
    CKC_OP_ARITH_CVT_F32_TO_BF8,
    CKC_OP_ARITH_CVT_F32_TO_I8_SAT,
    CKC_OP_ARITH_CVT_SCALEF32_PK_F32_FP8,
    CKC_OP_ARITH_CVT_SCALEF32_PK_F32_BF8,
    CKC_OP_ARITH_CVT_SCALEF32_PK_FP8_F32,
    CKC_OP_ARITH_CVT_SCALEF32_PK_BF8_F32,

    /* math.* */
    CKC_OP_MATH_EXP2,
    CKC_OP_MATH_LOG2,
    CKC_OP_MATH_RCP,
    CKC_OP_MATH_RCP_FAST,
    CKC_OP_MATH_SQRT,
    CKC_OP_MATH_RSQRT,
    CKC_OP_MATH_TANH,

    /* gpu.* */
    CKC_OP_GPU_THREAD_ID,
    CKC_OP_GPU_BLOCK_ID,

    /* memref.* */
    CKC_OP_MEMREF_GLOBAL_LOAD,
    CKC_OP_MEMREF_GLOBAL_LOAD_TYPED,
    CKC_OP_MEMREF_GLOBAL_LOAD_VN,
    CKC_OP_MEMREF_GLOBAL_STORE,
    CKC_OP_MEMREF_GLOBAL_STORE_TYPED,
    CKC_OP_MEMREF_GLOBAL_STORE_VN,
    CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD,
    CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32,
    CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16,
    CKC_OP_MEMREF_COOPERATIVE_GLOBAL_STORE,

    /* vector.* */
    CKC_OP_VECTOR_ADD,
    CKC_OP_VECTOR_SUB,
    CKC_OP_VECTOR_MUL,
    CKC_OP_VECTOR_AND,
    CKC_OP_VECTOR_OR,
    CKC_OP_VECTOR_SHL,
    CKC_OP_VECTOR_LSHR,
    CKC_OP_VECTOR_SMAX,
    CKC_OP_VECTOR_SMIN,
    CKC_OP_VECTOR_MAX,
    CKC_OP_VECTOR_FMA,
    CKC_OP_VECTOR_SUM,
    CKC_OP_VECTOR_REDUCE_MAX,
    CKC_OP_VECTOR_SPLAT,
    CKC_OP_VECTOR_SELECT,
    CKC_OP_VECTOR_CMP,
    CKC_OP_VECTOR_TRUNC,
    CKC_OP_VECTOR_SEXT,
    CKC_OP_VECTOR_TRUNC_F32_TO_F16,
    CKC_OP_VECTOR_TRUNC_F32_TO,
    CKC_OP_VECTOR_BITCAST,
    CKC_OP_VECTOR_EXTRACT,
    CKC_OP_VECTOR_INSERT,
    CKC_OP_VECTOR_PACK,
    CKC_OP_VECTOR_CONCAT,

    /* tile.* -- memory / lds */
    CKC_OP_TILE_SMEM_ALLOC,
    CKC_OP_TILE_SMEM_STORE,
    CKC_OP_TILE_SMEM_STORE_VN,
    CKC_OP_TILE_SMEM_STORE_VN_F32,
    CKC_OP_TILE_SMEM_STORE_DISTRIBUTED,
    CKC_OP_TILE_SMEM_LOAD_V4,
    CKC_OP_TILE_SMEM_LOAD_VN,
    CKC_OP_TILE_SMEM_LOAD_VN_F32,
    CKC_OP_TILE_SMEM_ADDR_OF,
    CKC_OP_TILE_SMEM_PTR_ADD,
    CKC_OP_TILE_LDS_ATOMIC_ADD,
    CKC_OP_TILE_GLOBAL_PTR_ADD,
    CKC_OP_TILE_GLOBAL_LOAD_LDS,
    CKC_OP_TILE_ASYNC_BUFFER_LOAD_LDS,
    CKC_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR,
    CKC_OP_TILE_BUFFER_RSRC,
    CKC_OP_TILE_BUFFER_LOAD_F16,
    CKC_OP_TILE_BUFFER_LOAD_VN_F16,
    CKC_OP_TILE_BUFFER_STORE_F16,
    CKC_OP_TILE_BUFFER_STORE_VN_F16,

    /* tile.* -- mma */
    CKC_OP_TILE_MMA,
    CKC_OP_TILE_REGISTER_P_FROM_QK_C,

    /* tile.* -- inline asm */
    CKC_OP_TILE_INLINE_ASM,

    /* tile.* -- cross-lane / dpp / permute */
    CKC_OP_TILE_READFIRSTLANE,
    CKC_OP_TILE_PIN_SGPR,
    CKC_OP_TILE_LANE_ID,
    CKC_OP_TILE_WAVE_ALL,
    CKC_OP_TILE_WAVE_ANY,
    CKC_OP_TILE_WAVE_BALLOT,
    CKC_OP_TILE_DS_BPERMUTE,
    CKC_OP_TILE_DS_BPERMUTE_B64,
    CKC_OP_TILE_DS_SWIZZLE_XOR,
    CKC_OP_TILE_MOV_DPP,
    CKC_OP_TILE_PERMLANE32_SWAP,
    CKC_OP_TILE_PERM_B32,
    CKC_OP_TILE_PERMLANEX16,
    CKC_OP_TILE_BYTE_PERM,
    CKC_OP_TILE_DS_READ_TR16_B64,
    CKC_OP_TILE_DS_READ_TR16_B128,
    CKC_OP_TILE_DS_READ_TR_B8,

    /* tile.* -- barriers / scheduling */
    CKC_OP_TILE_SYNC,
    CKC_OP_TILE_SYNC_HALF_BLOCK,
    CKC_OP_TILE_SYNC_LDS_ONLY,
    CKC_OP_TILE_S_BARRIER_BARE,
    CKC_OP_TILE_S_WAITCNT,
    CKC_OP_TILE_S_SETPRIO,
    CKC_OP_TILE_IGLP_OPT,
    CKC_OP_TILE_SCHED_BARRIER,
    CKC_OP_TILE_SCHED_GROUP_BARRIER,

    /* scf.* / cf.* control flow */
    CKC_OP_SCF_FOR,
    CKC_OP_SCF_IF,
    CKC_OP_SCF_YIELD,
    CKC_OP_CF_RETURN,

    CKC_OP__COUNT
} ckc_opcode_t;

/* ---------------------------------------------------------- core IR nodes */

struct ckc_op;
struct ckc_region;

/* SSA value. Mutable: `op` is back-patched after the producing op is built
 * (Python Value.op = op). `name` is "%vN" / "%paramname" / "%k0" form. */
typedef struct ckc_value
{
    const char* name; /* arena-owned, includes leading '%'            */
    const ckc_type_t* type;
    struct ckc_op* op; /* producing op, or NULL for params/iv/iter args */
} ckc_value_t;

/* Operation. `opcode` replaces the Python op.name string; `name` keeps the
 * dotted text for printing. operands/results/regions are arena-backed arrays. */
typedef struct ckc_op
{
    ckc_opcode_t opcode;
    const char* name; /* dotted name, e.g. "arith.add"          */
    ckc_value_t** operands;
    int num_operands;
    ckc_value_t** results;
    int num_results;
    ckc_attr_map_t attrs;
    struct ckc_region** regions;
    int num_regions;
    const char* loc; /* "file:line" or NULL                    */
} ckc_op_t;

/* Region (basic block / control-flow body). */
typedef struct ckc_region
{
    const char* label; /* "entry","body","then",...                       */
    ckc_op_t** ops;
    int num_ops;
    int cap_ops;
} ckc_region_t;

/* Kernel parameter ABI options (the Python **attrs on IRBuilder.param). A field
 * is "unset" via the *_set companion flag so defaults match Python (absent key).
 */
typedef struct ckc_param_opts
{
    bool noalias;
    bool noalias_set;
    bool readonly;
    bool readonly_set;
    bool writeonly;
    bool writeonly_set;
    int align;
    bool align_set;
    const char* addr_space; /* NULL => default "global"                     */
} ckc_param_opts_t;

typedef struct ckc_param
{
    const char* name; /* identifier WITHOUT leading '%'               */
    const ckc_type_t* type;
    ckc_attr_map_t attrs; /* materialised ABI attrs (noalias/align/...)   */
} ckc_param_t;

typedef struct ckc_kernel_def
{
    const char* name;
    ckc_param_t** params;
    int num_params;
    int cap_params;
    ckc_region_t* body;   /* the "entry" region                          */
    ckc_attr_map_t attrs; /* max_workgroup_size, ...                      */
} ckc_kernel_def_t;

/* --------------------------------------------------------------- builder */

#define CKC_REGION_STACK_MAX 64

typedef struct ckc_ir_builder
{
    ckc_arena_t arena; /* owns every node below               */
    int counter;       /* SSA name counter (%vN)              */
    ckc_kernel_def_t* kernel;
    ckc_region_t* region_stack[CKC_REGION_STACK_MAX];
    int region_depth; /* region_stack[depth-1] is current    */

    /* Param lookup: parallel arrays, linear search by name (small N). */
    const char** param_names;
    ckc_value_t** param_values;
    int num_param_lookup;
    int cap_param_lookup;

    /* Sticky error state. */
    ckc_status_t status;
    char err[CKC_ERR_MSG_CAP];
} ckc_ir_builder_t;

/* For-loop handle: the C analog of the _ForBuilder context manager. The caller
 * does:  ckc_for_t f = ckc_b_scf_for(b, lo, hi, step, "k0");
 *        ckc_b_region_enter(b, f.body);   ... body ops using f.iv ...
 *        ckc_b_region_leave(b);
 * iter_vars/iter_inits carry the loop-carried values for scf_for_iter. */
typedef struct ckc_for
{
    ckc_op_t* op;
    ckc_value_t* iv;
    ckc_region_t* body;
    ckc_value_t** iter_vars; /* loop-carried induction values               */
    int num_iter_vars;
} ckc_for_t;

/* If handle: the C analog of _IfBuilder. */
typedef struct ckc_if
{
    ckc_op_t* op;
    ckc_region_t* then_region;
} ckc_if_t;

/* (name, init) pair for scf_for_iter. */
typedef struct ckc_iter_arg
{
    const char* name; /* WITHOUT leading '%'                          */
    ckc_value_t* init;
} ckc_iter_arg_t;

/* Options for inline_asm (Python keyword-only args). */
typedef struct ckc_inline_asm_opts
{
    bool sideeffect; /* default true                                     */
    bool convergent; /* default false                                    */
    bool sideeffect_set;
    bool convergent_set;
} ckc_inline_asm_opts_t;

/* ============================== TYPE SYSTEM ============================== */

/* Interned scalar singletons (Python module-level I1, F32, ...). Always valid;
 * never NULL; never arena-owned (static storage). */
const ckc_type_t* ckc_i1(void);
const ckc_type_t* ckc_i8(void);
const ckc_type_t* ckc_i16(void);
const ckc_type_t* ckc_i32(void);
const ckc_type_t* ckc_i64(void);
const ckc_type_t* ckc_bf16(void);
const ckc_type_t* ckc_f16(void);
const ckc_type_t* ckc_f32(void);
const ckc_type_t* ckc_fp8e4m3(void);
const ckc_type_t* ckc_bf8e5m2(void);

/* Look up a scalar singleton by canonical name ("i32",...); NULL if unknown. */
const ckc_type_t* ckc_scalar_by_name(const char* name);

/* Composite type constructors (arena-allocated, name computed Python-identically).
 * VectorType(elem,count) -> "vec<{elem}x{count}>"
 * PtrType(pointee,space) -> "ptr<{pointee},{space}>"
 * SmemType(elem,shape)   -> "smem<{elem}, [{d0}x{d1}...]>"  */
const ckc_type_t* ckc_vector_type(ckc_ir_builder_t* b, const ckc_type_t* elem, int count);
const ckc_type_t* ckc_ptr_type(ckc_ir_builder_t* b, const ckc_type_t* pointee, const char* space);
const ckc_type_t*
ckc_smem_type(ckc_ir_builder_t* b, const ckc_type_t* elem, const int* shape, int rank);

/* Structural type equality (matches Python frozen-dataclass __eq__: compares by
 * canonical name, which encodes kind + components). */
bool ckc_type_eq(const ckc_type_t* a, const ckc_type_t* b);

/* AMDGPU buffer-load AUX cache-coherency hints (Python module constants). */
typedef enum ckc_cache_policy
{
    CKC_CACHE_ALL    = 0,
    CKC_CACHE_GLOBAL = 1,
    CKC_CACHE_STREAM = 2,
    CKC_NON_TEMPORAL = 3
} ckc_cache_policy_t;

/* ============================== ATTR MAP ================================ */

void ckc_attr_map_init(ckc_attr_map_t* m);
void ckc_attr_set_int(ckc_ir_builder_t* b, ckc_attr_map_t* m, const char* key, int64_t v);
void ckc_attr_set_float(ckc_ir_builder_t* b, ckc_attr_map_t* m, const char* key, double v);
void ckc_attr_set_str(ckc_ir_builder_t* b, ckc_attr_map_t* m, const char* key, const char* v);
void ckc_attr_set_bool(ckc_ir_builder_t* b, ckc_attr_map_t* m, const char* key, bool v);
/* Set a list of bare ints (serialized as l:[ i:v0, i:v1, ... ]). */
void ckc_attr_set_int_list(
    ckc_ir_builder_t* b, ckc_attr_map_t* m, const char* key, const int64_t* vals, int count);
/* Returns the entry for `key`, or NULL if absent. */
const ckc_attr_value_t* ckc_attr_get(const ckc_attr_map_t* m, const char* key);
bool ckc_attr_get_int(const ckc_attr_map_t* m, const char* key, int64_t* out);
bool ckc_attr_get_float(const ckc_attr_map_t* m, const char* key, double* out);
const char* ckc_attr_get_str(const ckc_attr_map_t* m, const char* key); /* NULL if absent */
bool ckc_attr_get_bool(const ckc_attr_map_t* m, const char* key, bool dflt);

/* ============================ OPCODE TABLE ============================== */

/* Canonical dotted name for an opcode ("arith.add"); "" for CKC_OP_INVALID. */
const char* ckc_opcode_name(ckc_opcode_t op);
/* Reverse map: dotted name -> opcode; CKC_OP_INVALID if unknown. */
ckc_opcode_t ckc_opcode_from_name(const char* name);
/* True if the op is side-effect-free (Python PURE_OP_NAMES / is_pure_op_name). */
bool ckc_opcode_is_pure(ckc_opcode_t op);

/* ============================== OP GETTERS ============================== */

/* Python @property Op.result: requires exactly one result, else sets error. */
ckc_value_t* ckc_op_result(ckc_ir_builder_t* b, ckc_op_t* op);
/* Python @property Op.is_pure: attrs["pure"] override, else opcode purity. */
bool ckc_op_is_pure(const ckc_op_t* op);
/* Python @property KernelDef.max_workgroup_size (default 256). */
int ckc_kernel_max_workgroup_size(const ckc_kernel_def_t* k);

/* ============================== BUILDER ================================= */

/* Construct/destruct. ckc_ir_builder_new allocates the builder's own arena and
 * an empty kernel with an "entry" region as the current region. */
ckc_status_t ckc_ir_builder_init(ckc_ir_builder_t* b, const char* kernel_name);
void ckc_ir_builder_free(ckc_ir_builder_t* b);
bool ckc_ir_builder_ok(const ckc_ir_builder_t* b);
ckc_status_t ckc_ir_builder_status(const ckc_ir_builder_t* b);
const char* ckc_ir_builder_error(const ckc_ir_builder_t* b);
ckc_kernel_def_t* ckc_ir_builder_kernel(ckc_ir_builder_t* b);

/* Low-level plumbing (mirrors IRBuilder._fresh/_emit/push_region/pop_region and
 * the generic _op). Lowerers rarely need these; emitters/tests may. */
const char* ckc_b_fresh(ckc_ir_builder_t* b, const char* prefix);
void ckc_b_emit(ckc_ir_builder_t* b, ckc_op_t* op);
void ckc_b_region_enter(ckc_ir_builder_t* b, ckc_region_t* r); /* push */
void ckc_b_region_leave(ckc_ir_builder_t* b);                  /* pop  */
ckc_region_t* ckc_b_current_region(ckc_ir_builder_t* b);

/* Generic op builder. Creates fresh result Values (one per result_types entry,
 * named with result_name_hint), builds the Op, links results back, emits it, and
 * returns it. attrs/regions may be NULL. This is IRBuilder._op. */
ckc_op_t* ckc_b_op(ckc_ir_builder_t* b,
                   ckc_opcode_t opcode,
                   ckc_value_t* const* operands,
                   int num_operands,
                   const ckc_type_t* const* result_types,
                   int num_results,
                   const ckc_attr_map_t* attrs,
                   ckc_region_t* const* regions,
                   int num_regions,
                   const char* result_name_hint,
                   const char* loc);

/* ----- params ----- */
ckc_value_t* ckc_b_param(ckc_ir_builder_t* b,
                         const char* name,
                         const ckc_type_t* t,
                         const ckc_param_opts_t* opts);
ckc_value_t* ckc_b_get_param(ckc_ir_builder_t* b, const char* name);

/* ----- arith constants ----- */
ckc_value_t* ckc_b_const_i32(ckc_ir_builder_t* b, int64_t value);
ckc_value_t* ckc_b_const_i64(ckc_ir_builder_t* b, int64_t value);
ckc_value_t* ckc_b_const_f32(ckc_ir_builder_t* b, double value);
ckc_value_t* ckc_b_fp16_zero(ckc_ir_builder_t* b);
ckc_value_t* ckc_b_zero_vec_f32(ckc_ir_builder_t* b, int n);
ckc_value_t* ckc_b_zero_vec(ckc_ir_builder_t* b, const ckc_type_t* elem, int n);

/* ----- arith integer / logic ----- */
ckc_value_t* ckc_b_add(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_sub(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_mul(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_div(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_mod(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_land(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_lor(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_lnot(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_smax(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_smin(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_xor(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_shl(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_lshr(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_umul_hi_i32(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);

/* ----- arith float ----- */
ckc_value_t* ckc_b_fadd(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fsub(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fmul(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fdiv(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fneg(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_fabs(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_fma(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, ckc_value_t* d);
ckc_value_t* ckc_b_fmax(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fmin(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_fmax3(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, ckc_value_t* d);
ckc_value_t* ckc_b_fmin3(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, ckc_value_t* d);
ckc_value_t* ckc_b_clamp_f32(ckc_ir_builder_t* b, ckc_value_t* v, ckc_value_t* lo, ckc_value_t* hi);

/* ----- comparisons (return i1) ----- */
ckc_value_t* ckc_b_cmp_lt(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_cmp_le(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_cmp_gt(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_cmp_ge(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_cmp_eq(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_cmp_ne(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
/* pred in {olt,ole,ogt,oge,oeq,one,ord,uno} */
ckc_value_t* ckc_b_fcmp(ckc_ir_builder_t* b, const char* pred, ckc_value_t* a, ckc_value_t* c);

/* ----- math ----- */
ckc_value_t* ckc_b_exp2(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_log2(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_rcp(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_rcp_fast(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_sqrt(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_rsqrt(ckc_ir_builder_t* b, ckc_value_t* a);
ckc_value_t* ckc_b_tanh(ckc_ir_builder_t* b, ckc_value_t* a);

/* ----- casts / conversions ----- */
ckc_value_t* ckc_b_zext(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_sext(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_trunc(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_bitcast(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t*
ckc_b_select(ckc_ir_builder_t* b, ckc_value_t* cond, ckc_value_t* lhs, ckc_value_t* rhs);
ckc_value_t*
ckc_b_masked_select(ckc_ir_builder_t* b, ckc_value_t* cond, ckc_value_t* lhs, ckc_value_t* rhs);
ckc_value_t* ckc_b_trunc_f32_to_f16(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_rint_f32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cast_to_f32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cast_f32_to(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_sitofp_f32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_fp8_to_f32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_bf8_to_f32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_pk_f32_fp8x4(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_pk_f32_bf8x4(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t*
ckc_b_cvt_scalef32_pk_f32_fp8x4(ckc_ir_builder_t* b, ckc_value_t* v, ckc_value_t* scale);
ckc_value_t*
ckc_b_cvt_scalef32_pk_f32_bf8x4(ckc_ir_builder_t* b, ckc_value_t* v, ckc_value_t* scale);
ckc_value_t* ckc_b_cvt_f32_to_fp8(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_f32_to_bf8(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_f32_to_i8_sat(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_pk_fp8_f32x4(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_pk_bf8_f32x4(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_cvt_pk_i8_f32x4(ckc_ir_builder_t* b, ckc_value_t* v);

/* ----- atomics ----- */
ckc_value_t* ckc_b_global_atomic_add(ckc_ir_builder_t* b,
                                     ckc_value_t* ptr,
                                     ckc_value_t* idx,
                                     ckc_value_t* value,
                                     const char* ordering /* NULL=>monotonic */);
ckc_value_t* ckc_b_lds_atomic_add(ckc_ir_builder_t* b,
                                  ckc_value_t* smem,
                                  ckc_value_t* const* indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  const char* ordering);
ckc_value_t* ckc_b_global_atomic_add_pk_bf16(ckc_ir_builder_t* b,
                                             ckc_value_t* ptr,
                                             ckc_value_t* idx,
                                             ckc_value_t* value,
                                             const char* ordering);

/* ----- gpu ids ----- */
ckc_value_t* ckc_b_thread_id_x(ckc_ir_builder_t* b);
ckc_value_t* ckc_b_block_id_x(ckc_ir_builder_t* b);
ckc_value_t* ckc_b_block_id_y(ckc_ir_builder_t* b);
ckc_value_t* ckc_b_block_id_z(ckc_ir_builder_t* b);

/* ----- global memory ----- */
ckc_value_t* ckc_b_smem_alloc(
    ckc_ir_builder_t* b, const ckc_type_t* elem, const int* shape, int rank, const char* name_hint);
ckc_value_t* ckc_b_global_load(ckc_ir_builder_t* b,
                               ckc_value_t* ptr,
                               ckc_value_t* idx,
                               const ckc_type_t* dtype,
                               int align /* <=0 => 1 */);
ckc_value_t*
ckc_b_global_load_f16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t*
ckc_b_global_load_f32(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t*
ckc_b_global_load_i32(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t*
ckc_b_global_load_i64(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t*
ckc_b_global_load_bf16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t*
ckc_b_global_load_fp8e4m3(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align);
ckc_value_t* ckc_b_masked_global_load(ckc_ir_builder_t* b,
                                      ckc_value_t* ptr,
                                      ckc_value_t* idx,
                                      ckc_value_t* mask,
                                      ckc_value_t* other,
                                      const ckc_type_t* dtype,
                                      int align);
void ckc_b_global_store(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int align);
ckc_value_t* ckc_b_global_load_vN(ckc_ir_builder_t* b,
                                  ckc_value_t* ptr,
                                  ckc_value_t* idx,
                                  const ckc_type_t* dtype,
                                  int n,
                                  int align /* <=0 => default */);
ckc_value_t*
ckc_b_global_load_vN_f16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int n, int align);

/* ----- vector ops ----- */
ckc_value_t* ckc_b_vector_add(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_sub(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_mul(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_and(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_or(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_shl(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_lshr(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_smax(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_smin(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_max(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_fma(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, ckc_value_t* d);
ckc_value_t* ckc_b_vector_sum(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_vector_reduce_max(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_vector_splat(ckc_ir_builder_t* b, ckc_value_t* scalar, int n);
ckc_value_t*
ckc_b_vector_select(ckc_ir_builder_t* b, ckc_value_t* mask, ckc_value_t* lhs, ckc_value_t* rhs);
ckc_value_t*
ckc_b_vector_cmp(ckc_ir_builder_t* b, const char* pred, ckc_value_t* a, ckc_value_t* c);
ckc_value_t* ckc_b_vector_trunc(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_vector_sext(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);

/* ----- LDS (shared memory) ----- */
void ckc_b_smem_store_f16(ckc_ir_builder_t* b,
                          ckc_value_t* smem,
                          ckc_value_t* const* indices,
                          int num_indices,
                          ckc_value_t* value);
void ckc_b_smem_store_vN(ckc_ir_builder_t* b,
                         ckc_value_t* smem,
                         ckc_value_t* const* indices,
                         int num_indices,
                         ckc_value_t* value,
                         int n);
void ckc_b_smem_store_vN_f16(ckc_ir_builder_t* b,
                             ckc_value_t* smem,
                             ckc_value_t* const* indices,
                             int num_indices,
                             ckc_value_t* value,
                             int n);
ckc_value_t*
ckc_b_smem_load_v4_f16(ckc_ir_builder_t* b, ckc_value_t* smem, ckc_value_t* row, ckc_value_t* col);
ckc_value_t* ckc_b_smem_load_vN(ckc_ir_builder_t* b,
                                ckc_value_t* smem,
                                ckc_value_t* const* indices,
                                int num_indices,
                                const ckc_type_t* dtype,
                                int n);
ckc_value_t* ckc_b_smem_load_vN_f16(
    ckc_ir_builder_t* b, ckc_value_t* smem, ckc_value_t* const* indices, int num_indices, int n);

/* ----- target-neutral MMA ----- */
/* op_id is the atom identifier ("mfma_f32_16x16x16_f16", "wmma_...", ...).
 * extra carries scaled-MX scale operands (a_scale,b_scale) or is NULL/0. */
ckc_value_t* ckc_b_mma(ckc_ir_builder_t* b,
                       const char* op_id,
                       ckc_value_t* a,
                       ckc_value_t* bb,
                       ckc_value_t* c,
                       ckc_value_t* const* extra,
                       int num_extra);

/* ----- inline asm ----- */
/* operands/result_types are explicit arrays; constraints/template are strings.
 * Returns the op (results accessible via op->results) since asm may be 0/1/N
 * results. */
ckc_op_t* ckc_b_inline_asm(ckc_ir_builder_t* b,
                           const char* asm_template,
                           const char* constraints,
                           ckc_value_t* const* operands,
                           int num_operands,
                           const ckc_type_t* const* result_types,
                           int num_results,
                           const ckc_inline_asm_opts_t* opts);

/* ----- cross-lane / vector pack-extract ----- */
ckc_value_t* ckc_b_readfirstlane(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_lane_id(ckc_ir_builder_t* b);
ckc_value_t* ckc_b_vec_extract(ckc_ir_builder_t* b, ckc_value_t* v, int i);
ckc_value_t* ckc_b_vec_insert(ckc_ir_builder_t* b, ckc_value_t* v, ckc_value_t* scalar, int i);
ckc_value_t* ckc_b_vec_pack(ckc_ir_builder_t* b,
                            ckc_value_t* const* components,
                            int num_components,
                            const ckc_type_t* elem);
ckc_value_t* ckc_b_vec_concat(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb);

/* ----- ISA-named MMA wrappers (thin wrappers over ckc_b_mma; kept for parity
 * with the legacy Python helpers so emitters can call them by name). All take
 * (a, b, c) and return <c_frag_len x acc_elem>. The scaled MX atom takes the
 * two extra E8M0 scale operands. */
ckc_value_t*
ckc_b_mfma_f32_16x16x16_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x32_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x16_bf16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x32_bf16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x32_fp8(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x32_bf8(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x8_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x8_bf16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x16_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x16_bf16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x16_fp8(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_32x32x16_bf8(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_4x4x4_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x128_fp4(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_mfma_f32_16x16x96_fp6(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t* ckc_b_mfma_scale_f32_16x16x128_f8f6f4(ckc_ir_builder_t* b,
                                                   ckc_value_t* a,
                                                   ckc_value_t* bb,
                                                   ckc_value_t* c,
                                                   ckc_value_t* a_scale,
                                                   ckc_value_t* b_scale);
ckc_value_t*
ckc_b_wmma_f32_16x16x16_f16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t*
ckc_b_wmma_f32_16x16x16_bf16(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, ckc_value_t* c);
ckc_value_t* ckc_b_wmma_gfx12_f32_16x16x16_f16(ckc_ir_builder_t* b,
                                               ckc_value_t* a,
                                               ckc_value_t* bb,
                                               ckc_value_t* c);
ckc_value_t* ckc_b_wmma_gfx12_f32_16x16x16_bf16(ckc_ir_builder_t* b,
                                                ckc_value_t* a,
                                                ckc_value_t* bb,
                                                ckc_value_t* c);

/* ----- multi-output inline asm (LLVM literal-struct return). Returns the op;
 * its results[] holds the N output Values in declaration order. */
ckc_op_t* ckc_b_inline_asm_multi(ckc_ir_builder_t* b,
                                 const char* asm_template,
                                 const char* constraints,
                                 ckc_value_t* const* operands,
                                 int num_operands,
                                 const ckc_type_t* const* result_types,
                                 int num_results,
                                 const ckc_inline_asm_opts_t* opts);

/* ----- register-fragment reshape (P13) ----- */
ckc_value_t*
ckc_b_register_p_from_qk_c(ckc_ir_builder_t* b, ckc_value_t* qk_c, const ckc_type_t* target_dtype);

/* ----- distributed / cooperative epilogue stores ----- */
void ckc_b_smem_store_distributed(ckc_ir_builder_t* b,
                                  ckc_value_t* smem,
                                  const ckc_attr_map_t* layout_attrs,
                                  ckc_value_t* values);
void ckc_b_cooperative_global_store(ckc_ir_builder_t* b,
                                    ckc_value_t* ptr,
                                    ckc_value_t* addrs,
                                    ckc_value_t* values);

/* ----- uniform / wave-scalar helpers ----- */
ckc_value_t* ckc_b_pin_sgpr(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_to_sgpr_u32(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_wave_all(ckc_ir_builder_t* b, ckc_value_t* predicate);
ckc_value_t* ckc_b_wave_any(ckc_ir_builder_t* b, ckc_value_t* predicate);
ckc_value_t* ckc_b_wave_ballot(ckc_ir_builder_t* b, ckc_value_t* predicate);

/* ----- cross-lane permute / dpp ----- */
ckc_value_t* ckc_b_ds_bpermute(ckc_ir_builder_t* b, ckc_value_t* addr, ckc_value_t* data);
ckc_value_t* ckc_b_ds_bpermute_b64(ckc_ir_builder_t* b, ckc_value_t* addr, ckc_value_t* data);
ckc_value_t* ckc_b_ds_swizzle_xor(ckc_ir_builder_t* b, ckc_value_t* data, int xor_mask);
/* mov_dpp: exactly one of row_shr/row_shl must be >= 0 (the other < 0 = unset). */
ckc_value_t*
ckc_b_mov_dpp(ckc_ir_builder_t* b, ckc_value_t* data, int row_shr, int row_shl, bool bound_ctrl);
/* permlane32_swap returns two values via out params (new_lo, new_hi). */
void ckc_b_permlane32_swap(ckc_ir_builder_t* b,
                           ckc_value_t* lo,
                           ckc_value_t* hi,
                           ckc_value_t** out_lo,
                           ckc_value_t** out_hi);
ckc_value_t*
ckc_b_perm_b32(ckc_ir_builder_t* b, ckc_value_t* src0, ckc_value_t* src1, ckc_value_t* sel);
ckc_value_t* ckc_b_permlanex16(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_byte_perm(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb, int64_t sel);
ckc_value_t* ckc_b_warp_shuffle_xor(ckc_ir_builder_t* b, ckc_value_t* v, int lane_xor);

/* ----- transpose LDS reads ----- */
ckc_value_t* ckc_b_ds_read_tr16_b64(ckc_ir_builder_t* b,
                                    ckc_value_t* smem,
                                    ckc_value_t* const* indices,
                                    int num_indices,
                                    const ckc_type_t* dtype /* NULL=>f16 */);
ckc_value_t* ckc_b_ds_read_tr16_b128(ckc_ir_builder_t* b,
                                     ckc_value_t* smem,
                                     ckc_value_t* const* indices,
                                     int num_indices,
                                     const ckc_type_t* dtype /* NULL=>f16 */);
ckc_value_t* ckc_b_ds_read_tr_b8(ckc_ir_builder_t* b,
                                 ckc_value_t* smem,
                                 ckc_value_t* const* indices,
                                 int num_indices,
                                 const ckc_type_t* dtype /* NULL=>fp8e4m3 */);

/* ----- vector bitcast / packed f32->f16 conversion ----- */
ckc_value_t* ckc_b_vec_bitcast(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);
ckc_value_t* ckc_b_vec_trunc_f32_to_f16(ckc_ir_builder_t* b, ckc_value_t* v);
ckc_value_t* ckc_b_vec_cast_f32_to(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target);

/* ----- LDS pointer arithmetic + async DRAM->LDS ----- */
ckc_value_t* ckc_b_smem_addr_of(ckc_ir_builder_t* b, ckc_value_t* smem);
ckc_value_t* ckc_b_smem_ptr_add(ckc_ir_builder_t* b, ckc_value_t* lds_addr, ckc_value_t* byte_off);
void ckc_b_async_buffer_load_lds_addr(ckc_ir_builder_t* b,
                                      ckc_value_t* rsrc,
                                      ckc_value_t* lds_addr,
                                      ckc_value_t* voffset,
                                      ckc_value_t* soffset,
                                      int dwords,
                                      int coherency);
void ckc_b_async_buffer_load_lds(ckc_ir_builder_t* b,
                                 ckc_value_t* rsrc,
                                 ckc_value_t* lds_ptr,
                                 ckc_value_t* voffset,
                                 ckc_value_t* soffset,
                                 int dwords,
                                 int coherency);
void ckc_b_global_load_lds(ckc_ir_builder_t* b,
                           ckc_value_t* src_ptr,
                           ckc_value_t* byte_off,
                           ckc_value_t* lds_addr,
                           int size_bytes,
                           int coherency);

/* ----- global pointer arithmetic + buffer resource descriptors ----- */
ckc_value_t* ckc_b_global_ptr_add(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* byte_off);
ckc_value_t* ckc_b_buffer_rsrc(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* num_bytes);
ckc_value_t* ckc_b_buffer_load_vN_f16(
    ckc_ir_builder_t* b, ckc_value_t* rsrc, ckc_value_t* voffset, ckc_value_t* soffset, int dwords);
ckc_value_t* ckc_b_buffer_load_f16(ckc_ir_builder_t* b,
                                   ckc_value_t* rsrc,
                                   ckc_value_t* voffset,
                                   ckc_value_t* soffset);
void ckc_b_buffer_store_vN_f16(ckc_ir_builder_t* b,
                               ckc_value_t* rsrc,
                               ckc_value_t* voffset,
                               ckc_value_t* soffset,
                               ckc_value_t* value,
                               int dwords);
void ckc_b_buffer_store_f16(ckc_ir_builder_t* b,
                            ckc_value_t* rsrc,
                            ckc_value_t* voffset,
                            ckc_value_t* soffset,
                            ckc_value_t* value);

/* ----- f32 LDS ops (cshuffle epilogue) ----- */
ckc_value_t*
ckc_b_smem_alloc_f32(ckc_ir_builder_t* b, const int* shape, int rank, const char* name_hint);
void ckc_b_smem_store_vN_f32(ckc_ir_builder_t* b,
                             ckc_value_t* smem,
                             ckc_value_t* const* indices,
                             int num_indices,
                             ckc_value_t* value,
                             int n);
ckc_value_t* ckc_b_smem_load_vN_f32(
    ckc_ir_builder_t* b, ckc_value_t* smem, ckc_value_t* const* indices, int num_indices, int n);

/* ----- vectorised global stores + split-K atomics ----- */
void ckc_b_global_store_vN(ckc_ir_builder_t* b,
                           ckc_value_t* ptr,
                           ckc_value_t* idx,
                           ckc_value_t* value,
                           int n,
                           int align /* <=0 => default */);
void ckc_b_global_store_vN_f16(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int n, int align);
void ckc_b_global_atomic_add_f32(ckc_ir_builder_t* b,
                                 ckc_value_t* ptr,
                                 ckc_value_t* idx,
                                 ckc_value_t* value);
void ckc_b_store_f16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value);
ckc_value_t* ckc_b_zero_vec_f16(ckc_ir_builder_t* b, int n);

/* ----- barriers / scheduling ----- */
void ckc_b_sync(ckc_ir_builder_t* b);
void ckc_b_s_barrier_bare(ckc_ir_builder_t* b);
void ckc_b_sync_half_block(ckc_ir_builder_t* b, ckc_value_t* half_selector);
void ckc_b_sync_lds_only(ckc_ir_builder_t* b);
/* s_waitcnt: pass -1 to leave a counter alone, 0 to fully drain. */
void ckc_b_s_waitcnt(ckc_ir_builder_t* b, int vmcnt, int lgkmcnt, int expcnt);
void ckc_b_s_setprio(ckc_ir_builder_t* b, int level);
void ckc_b_iglp_opt(ckc_ir_builder_t* b, int level);
void ckc_b_sched_barrier(ckc_ir_builder_t* b, int mask);
void ckc_b_sched_group_barrier(ckc_ir_builder_t* b, int mask, int count, int group);

/* ----- compile-time loops (Python static_for / unroll are pure host control
 * flow; in C the caller simply writes a C for-loop calling the body. No IR op is
 * emitted, so no builder entry point is needed. Documented here for parity.) */

/* ----- control flow ----- */
ckc_for_t ckc_b_scf_for(ckc_ir_builder_t* b,
                        ckc_value_t* lo,
                        ckc_value_t* hi,
                        ckc_value_t* step,
                        const char* iv_name /* NULL=>"k0" */);
ckc_for_t ckc_b_scf_for_iter(ckc_ir_builder_t* b,
                             ckc_value_t* lo,
                             ckc_value_t* hi,
                             ckc_value_t* step,
                             const ckc_iter_arg_t* iter_args,
                             int num_iter_args,
                             const char* iv_name /* NULL=>"k0" */,
                             bool unroll,
                             bool elide_trailing_barrier);
void ckc_b_scf_yield(ckc_ir_builder_t* b, ckc_value_t* const* values, int num_values);
ckc_if_t ckc_b_scf_if(ckc_ir_builder_t* b, ckc_value_t* cond);
void ckc_b_ret(ckc_ir_builder_t* b);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_IR_H */
