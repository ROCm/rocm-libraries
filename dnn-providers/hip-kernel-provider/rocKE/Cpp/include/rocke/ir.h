/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/ir.h -- THE FROZEN IR CONTRACT for the C99 port of rocke.core.ir.
 *
 * This header is the single source of truth every lowerer (lower_llvm,
 * lower_hip, ir_print, passes, isa/backend, arch) binds to. It is a faithful,
 * explicit translation of the Python SSA IR:
 *
 *   Python                         C99 (this header)
 *   ----------------------------   --------------------------------------------
 *   class Type (frozen)            rocke_type_t  (tagged union, kind discriminant)
 *     VectorType/PtrType/SmemType    -> same struct, kind = VECTOR/PTR/SMEM
 *   class Value (mutable)          rocke_value_t
 *   class Op (mutable)             rocke_op_t
 *   class Region                   rocke_region_t
 *   class Param                    rocke_param_t
 *   class KernelDef                rocke_kernel_def_t
 *   class IRBuilder                rocke_ir_builder_t
 *   op.name : str                  rocke_opcode_t enum (ROCKE_OP_*) + name table
 *   op.attrs : Dict[str, Any]      rocke_attr_map_t (sorted key -> variant value)
 *   raise ValueError/TypeError     rocke_status_t return code + builder->err msg
 *   **attrs kwargs                 explicit option structs (rocke_param_opts_t...)
 *   @property result/is_pure       rocke_op_result()/rocke_op_is_pure() getters
 *
 * Lifetime: every node returned by the builder is owned by the builder's arena
 * (rocke_ir_builder_t.arena). Nothing is freed individually; rocke_ir_builder_free()
 * (or arena reset) bulk-frees the whole graph -- mirroring Python GC lifetime.
 *
 * Error model: the builder is sticky-failing. The first operation that fails
 * sets builder->status != ROCKE_OK and records a message in builder->err; every
 * subsequent builder call is a no-op that returns NULL / the error status. This
 * lets kernel authors write straight-line builder code (as in Python) and check
 * rocke_ir_builder_ok() once at the end, instead of checking every call.
 */
#ifndef ROCKE_IR_H
#define ROCKE_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ status */

typedef enum rocke_status
{
    ROCKE_OK = 0,
    ROCKE_ERR_VALUE, /* maps to Python ValueError                          */
    ROCKE_ERR_TYPE, /* maps to Python TypeError                           */
    ROCKE_ERR_KEY, /* maps to Python KeyError (unknown op_id / param)    */
    ROCKE_ERR_OOM, /* allocation failure                                 */
    ROCKE_ERR_NOTIMPL /* maps to Python NotImplementedError                 */
} rocke_status_t;

#define ROCKE_ERR_MSG_CAP 256

/* ROCKE_ERR_SNPRINTF -- snprintf into a bounded diagnostic/error buffer where
 * truncating an over-long message is INTENTIONAL (the buffer is a fixed
 * ROCKE_ERR_MSG_CAP-sized field; we never grow it for a long reason string).
 * snprintf is overflow-safe, so the only effect of truncation is a shortened
 * human-readable message -- never memory unsafety and never emitted IR (these
 * are reject/error paths). The localized pragma blesses exactly this idiom while
 * keeping -Werror=format-truncation active everywhere else, so any NEW,
 * unintended truncation (e.g. into a codegen name buffer) is still caught. */
#if defined(__GNUC__)
#define ROCKE_ERR_SNPRINTF(buf, cap, ...)                                   \
    do                                                                      \
    {                                                                       \
        _Pragma("GCC diagnostic push")                                      \
            _Pragma("GCC diagnostic ignored \"-Wformat-truncation\"")(void) \
                snprintf((buf), (cap), __VA_ARGS__);                        \
        _Pragma("GCC diagnostic pop")                                       \
    } while(0)
#else
#define ROCKE_ERR_SNPRINTF(buf, cap, ...) (void)snprintf((buf), (cap), __VA_ARGS__)
#endif

/* --------------------------------------------------------------- type kinds */

typedef enum rocke_type_kind
{
    ROCKE_TYPE_SCALAR = 0, /* i1/i8/i16/i32/i64/bf16/f16/f32/fp8e4m3/bf8e5m2     */
    ROCKE_TYPE_VECTOR, /* vec<elem x count>                                  */
    ROCKE_TYPE_PTR, /* ptr<pointee, space>                                */
    ROCKE_TYPE_SMEM /* smem<elem, [shape...]>                             */
} rocke_type_kind_t;

/* Canonical scalar type tags. The scalar singletons (rocke_i32() etc.) carry one
 * of these so consumers can switch on the element kind without strcmp. */
typedef enum rocke_scalar_kind
{
    ROCKE_SCALAR_I1 = 0,
    ROCKE_SCALAR_I8,
    ROCKE_SCALAR_I16,
    ROCKE_SCALAR_I32,
    ROCKE_SCALAR_I64,
    ROCKE_SCALAR_BF16,
    ROCKE_SCALAR_F16,
    ROCKE_SCALAR_F32,
    ROCKE_SCALAR_FP8E4M3,
    ROCKE_SCALAR_BF8E5M2,
    ROCKE_SCALAR__COUNT
} rocke_scalar_kind_t;

/* A Type. `name` is the canonical textual form ("i32", "vec<f16x4>",
 * "ptr<f16,global>", "smem<f16, [64x32]>") -- byte-identical to Python so the
 * printer/lowerers reproduce existing output. Scalar types are interned
 * singletons; composite types are arena-allocated and value-compared by name. */
typedef struct rocke_type
{
    rocke_type_kind_t kind;
    const char* name; /* canonical, arena/static owned, never NULL */

    /* ROCKE_TYPE_SCALAR */
    rocke_scalar_kind_t scalar; /* valid iff kind == ROCKE_TYPE_SCALAR         */

    /* ROCKE_TYPE_VECTOR */
    const struct rocke_type* elem; /* element type (VECTOR and SMEM)            */
    int count; /* lane count (VECTOR)                       */

    /* ROCKE_TYPE_PTR */
    const struct rocke_type* pointee;
    const char* space; /* "global","constant",...                   */

    /* ROCKE_TYPE_SMEM */
    const int* shape; /* arena-owned array of dim sizes            */
    int rank; /* number of dims in shape                   */
} rocke_type_t;

/* -------------------------------------------------------------- attr values */

typedef enum rocke_attr_kind
{
    ROCKE_ATTR_INT = 0, /* int64_t  (value, vec, align, rank, index, num, ...) */
    ROCKE_ATTR_FLOAT, /* double   (fp constant value, fill)                  */
    ROCKE_ATTR_STR, /* const char* (ity, pred, op_id, elem, elem_type,...) */
    ROCKE_ATTR_BOOL, /* bool     (pure, unroll, elide_trailing_barrier,...) */
    ROCKE_ATTR_LIST, /* nested attr list (scf.for iter_args metadata)       */
    ROCKE_ATTR_INT_LIST /* list of bare ints, e.g. agpr_alloc (0,0)            */
} rocke_attr_kind_t;

struct rocke_attr_map; /* forward: a list element is itself a small attr map */

typedef struct rocke_attr_value
{
    rocke_attr_kind_t kind;
    union
    {
        int64_t i;
        double f;
        const char* s; /* arena-owned                         */
        bool b;
        struct
        {
            struct rocke_attr_map** items; /* arena array of maps               */
            int count;
        } list;
        struct
        {
            int64_t* ints; /* arena array of bare ints (l:[ i:.., .. ])      */
            int count;
        } ilist;
    } u;
} rocke_attr_value_t;

typedef struct rocke_attr_entry
{
    const char* key; /* arena-owned                                    */
    rocke_attr_value_t value;
} rocke_attr_entry_t;

/* Op.attrs: an insertion-ordered key->variant map. Small (<=10 entries);
 * lookups are linear by key. ir_print sorts a copy for stable output. */
typedef struct rocke_attr_map
{
    rocke_attr_entry_t* entries; /* arena-owned, grows by reallocation in arena */
    int count;
    int cap;
} rocke_attr_map_t;

/* ----------------------------------------------------------------- opcodes */

/* One enumerator per distinct op name string in rocke.core.ir. Every lowerer
 * dispatches on this enum instead of Python getattr(self, "_op_"+name). The
 * canonical dotted name string is recovered with rocke_opcode_name(). */
typedef enum rocke_opcode
{
    ROCKE_OP_INVALID = 0,

    /* arith.* */
    ROCKE_OP_ARITH_CONSTANT,
    ROCKE_OP_ARITH_CONSTANT_VEC,
    ROCKE_OP_ARITH_ADD,
    ROCKE_OP_ARITH_SUB,
    ROCKE_OP_ARITH_MUL,
    ROCKE_OP_ARITH_DIV,
    ROCKE_OP_ARITH_MOD,
    ROCKE_OP_ARITH_FADD,
    ROCKE_OP_ARITH_FSUB,
    ROCKE_OP_ARITH_FMUL,
    ROCKE_OP_ARITH_FDIV,
    ROCKE_OP_ARITH_FNEG,
    ROCKE_OP_ARITH_FABS,
    ROCKE_OP_ARITH_FMA,
    ROCKE_OP_ARITH_FMAX3,
    ROCKE_OP_ARITH_FMIN3,
    ROCKE_OP_ARITH_CMP,
    ROCKE_OP_ARITH_FCMP,
    ROCKE_OP_ARITH_FMAX,
    ROCKE_OP_ARITH_FMIN,
    ROCKE_OP_ARITH_AND,
    ROCKE_OP_ARITH_OR,
    ROCKE_OP_ARITH_NOT,
    ROCKE_OP_ARITH_SMAX,
    ROCKE_OP_ARITH_SMIN,
    ROCKE_OP_ARITH_XOR,
    ROCKE_OP_ARITH_SHL,
    ROCKE_OP_ARITH_LSHR,
    ROCKE_OP_ARITH_UMUL_HI_I32,
    ROCKE_OP_ARITH_ZEXT,
    ROCKE_OP_ARITH_SEXT,
    ROCKE_OP_ARITH_TRUNC,
    ROCKE_OP_ARITH_SELECT,
    ROCKE_OP_ARITH_BITCAST,
    ROCKE_OP_ARITH_TRUNC_F32_TO_F16,
    ROCKE_OP_ARITH_RINT_F32,
    ROCKE_OP_ARITH_CAST_TO_F32,
    ROCKE_OP_ARITH_CAST_F32_TO,
    ROCKE_OP_ARITH_SITOFP_F32,
    ROCKE_OP_ARITH_CVT_FP8_TO_F32,
    ROCKE_OP_ARITH_CVT_BF8_TO_F32,
    ROCKE_OP_ARITH_CVT_PK_F32_FP8X4,
    ROCKE_OP_ARITH_CVT_PK_F32_BF8X4,
    ROCKE_OP_ARITH_CVT_PK_FP8_F32X4,
    ROCKE_OP_ARITH_CVT_PK_BF8_F32X4,
    ROCKE_OP_ARITH_CVT_PK_I8_F32X4,
    ROCKE_OP_ARITH_CVT_F32_TO_FP8,
    ROCKE_OP_ARITH_CVT_F32_TO_BF8,
    ROCKE_OP_ARITH_CVT_F32_TO_I8_SAT,
    ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_FP8,
    ROCKE_OP_ARITH_CVT_SCALEF32_PK_F32_BF8,
    ROCKE_OP_ARITH_CVT_SCALEF32_PK_FP8_F32,
    ROCKE_OP_ARITH_CVT_SCALEF32_PK_BF8_F32,

    /* math.* */
    ROCKE_OP_MATH_EXP2,
    ROCKE_OP_MATH_LOG2,
    ROCKE_OP_MATH_RCP,
    ROCKE_OP_MATH_RCP_FAST,
    ROCKE_OP_MATH_SQRT,
    ROCKE_OP_MATH_RSQRT,
    ROCKE_OP_MATH_TANH,

    /* gpu.* */
    ROCKE_OP_GPU_THREAD_ID,
    ROCKE_OP_GPU_BLOCK_ID,

    /* memref.* */
    ROCKE_OP_MEMREF_GLOBAL_LOAD,
    ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
    ROCKE_OP_MEMREF_GLOBAL_LOAD_VN,
    ROCKE_OP_MEMREF_GLOBAL_STORE,
    ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED,
    ROCKE_OP_MEMREF_GLOBAL_STORE_VN,
    ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD,
    ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32,
    ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16,
    ROCKE_OP_MEMREF_COOPERATIVE_GLOBAL_STORE,

    /* vector.* */
    ROCKE_OP_VECTOR_ADD,
    ROCKE_OP_VECTOR_SUB,
    ROCKE_OP_VECTOR_MUL,
    ROCKE_OP_VECTOR_AND,
    ROCKE_OP_VECTOR_OR,
    ROCKE_OP_VECTOR_SHL,
    ROCKE_OP_VECTOR_LSHR,
    ROCKE_OP_VECTOR_SMAX,
    ROCKE_OP_VECTOR_SMIN,
    ROCKE_OP_VECTOR_MAX,
    ROCKE_OP_VECTOR_FMA,
    ROCKE_OP_VECTOR_SUM,
    ROCKE_OP_VECTOR_REDUCE_MAX,
    ROCKE_OP_VECTOR_SPLAT,
    ROCKE_OP_VECTOR_SELECT,
    ROCKE_OP_VECTOR_CMP,
    ROCKE_OP_VECTOR_TRUNC,
    ROCKE_OP_VECTOR_SEXT,
    ROCKE_OP_VECTOR_TRUNC_F32_TO_F16,
    ROCKE_OP_VECTOR_TRUNC_F32_TO,
    ROCKE_OP_VECTOR_BITCAST,
    ROCKE_OP_VECTOR_EXTRACT,
    ROCKE_OP_VECTOR_INSERT,
    ROCKE_OP_VECTOR_PACK,
    ROCKE_OP_VECTOR_CONCAT,

    /* tile.* -- memory / lds */
    ROCKE_OP_TILE_SMEM_ALLOC,
    ROCKE_OP_TILE_SMEM_STORE,
    ROCKE_OP_TILE_SMEM_STORE_VN,
    ROCKE_OP_TILE_SMEM_STORE_VN_F32,
    ROCKE_OP_TILE_SMEM_STORE_DISTRIBUTED,
    ROCKE_OP_TILE_SMEM_LOAD_V4,
    ROCKE_OP_TILE_SMEM_LOAD_VN,
    ROCKE_OP_TILE_SMEM_LOAD_VN_F32,
    ROCKE_OP_TILE_SMEM_ADDR_OF,
    ROCKE_OP_TILE_SMEM_PTR_ADD,
    ROCKE_OP_TILE_LDS_ATOMIC_ADD,
    ROCKE_OP_TILE_GLOBAL_PTR_ADD,
    ROCKE_OP_TILE_GLOBAL_LOAD_LDS,
    ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS,
    ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR,
    ROCKE_OP_TILE_BUFFER_RSRC,
    ROCKE_OP_TILE_BUFFER_LOAD_F16,
    ROCKE_OP_TILE_BUFFER_LOAD_VN_F16,
    ROCKE_OP_TILE_BUFFER_LOAD_VN,
    ROCKE_OP_TILE_BUFFER_STORE_F16,
    ROCKE_OP_TILE_BUFFER_STORE_VN_F16,

    /* tile.* -- mma */
    ROCKE_OP_TILE_MMA,
    ROCKE_OP_TILE_REGISTER_P_FROM_QK_C,

    /* tile.* -- inline asm */
    ROCKE_OP_TILE_INLINE_ASM,

    /* tile.* -- cross-lane / dpp / permute */
    ROCKE_OP_TILE_READFIRSTLANE,
    ROCKE_OP_TILE_PIN_SGPR,
    ROCKE_OP_TILE_LANE_ID,
    ROCKE_OP_TILE_WAVE_ALL,
    ROCKE_OP_TILE_WAVE_ANY,
    ROCKE_OP_TILE_WAVE_BALLOT,
    ROCKE_OP_TILE_DS_BPERMUTE,
    ROCKE_OP_TILE_DS_BPERMUTE_B64,
    ROCKE_OP_TILE_DS_SWIZZLE_XOR,
    ROCKE_OP_TILE_MOV_DPP,
    ROCKE_OP_TILE_PERMLANE32_SWAP,
    ROCKE_OP_TILE_PERM_B32,
    ROCKE_OP_TILE_PERMLANEX16,
    ROCKE_OP_TILE_BYTE_PERM,
    ROCKE_OP_TILE_DS_READ_TR16_B64,
    ROCKE_OP_TILE_DS_READ_TR16_B128,
    ROCKE_OP_TILE_DS_READ_TR_B8,

    /* tile.* -- barriers / scheduling */
    ROCKE_OP_TILE_SYNC,
    ROCKE_OP_TILE_SYNC_HALF_BLOCK,
    ROCKE_OP_TILE_SYNC_LDS_ONLY,
    ROCKE_OP_TILE_S_BARRIER_BARE,
    ROCKE_OP_TILE_S_WAITCNT,
    ROCKE_OP_TILE_S_SETPRIO,
    ROCKE_OP_TILE_IGLP_OPT,
    ROCKE_OP_TILE_SCHED_BARRIER,
    ROCKE_OP_TILE_SCHED_GROUP_BARRIER,

    /* scf.* / cf.* control flow */
    ROCKE_OP_SCF_FOR,
    ROCKE_OP_SCF_IF,
    ROCKE_OP_SCF_YIELD,
    ROCKE_OP_CF_RETURN,

    ROCKE_OP__COUNT
} rocke_opcode_t;

/* ---------------------------------------------------------- core IR nodes */

struct rocke_op;
struct rocke_region;

/* SSA value. Mutable: `op` is back-patched after the producing op is built
 * (Python Value.op = op). `name` is "%vN" / "%paramname" / "%k0" form. */
typedef struct rocke_value
{
    const char* name; /* arena-owned, includes leading '%'            */
    const rocke_type_t* type;
    struct rocke_op* op; /* producing op, or NULL for params/iv/iter args */
} rocke_value_t;

/* Operation. `opcode` replaces the Python op.name string; `name` keeps the
 * dotted text for printing. operands/results/regions are arena-backed arrays. */
typedef struct rocke_op
{
    rocke_opcode_t opcode;
    const char* name; /* dotted name, e.g. "arith.add"          */
    rocke_value_t** operands;
    int num_operands;
    rocke_value_t** results;
    int num_results;
    rocke_attr_map_t attrs;
    struct rocke_region** regions;
    int num_regions;
    const char* loc; /* "file:line" or NULL                    */
} rocke_op_t;

/* Region (basic block / control-flow body). */
typedef struct rocke_region
{
    const char* label; /* "entry","body","then",...                       */
    rocke_op_t** ops;
    int num_ops;
    int cap_ops;
} rocke_region_t;

/* Kernel parameter ABI options (the Python **attrs on IRBuilder.param). A field
 * is "unset" via the *_set companion flag so defaults match Python (absent key).
 */
typedef struct rocke_param_opts
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
} rocke_param_opts_t;

typedef struct rocke_param
{
    const char* name; /* identifier WITHOUT leading '%'               */
    const rocke_type_t* type;
    rocke_attr_map_t attrs; /* materialised ABI attrs (noalias/align/...)   */
} rocke_param_t;

typedef struct rocke_kernel_def
{
    const char* name;
    rocke_param_t** params;
    int num_params;
    int cap_params;
    rocke_region_t* body; /* the "entry" region                          */
    rocke_attr_map_t attrs; /* max_workgroup_size, ...                      */
} rocke_kernel_def_t;

/* --------------------------------------------------------------- builder */

#define ROCKE_REGION_STACK_MAX 64

typedef struct rocke_ir_builder
{
    rocke_arena_t arena; /* owns every node below               */
    int counter; /* SSA name counter (%vN)              */
    rocke_kernel_def_t* kernel;
    rocke_region_t* region_stack[ROCKE_REGION_STACK_MAX];
    int region_depth; /* region_stack[depth-1] is current    */

    /* Param lookup: parallel arrays, linear search by name (small N). */
    const char** param_names;
    rocke_value_t** param_values;
    int num_param_lookup;
    int cap_param_lookup;

    /* Sticky error state. */
    rocke_status_t status;
    char err[ROCKE_ERR_MSG_CAP];
} rocke_ir_builder_t;

/* For-loop handle: the C analog of the _ForBuilder context manager. The caller
 * does:  rocke_for_t f = rocke_b_scf_for(b, lo, hi, step, "k0");
 *        rocke_b_region_enter(b, f.body);   ... body ops using f.iv ...
 *        rocke_b_region_leave(b);
 * iter_vars/iter_inits carry the loop-carried values for scf_for_iter. */
typedef struct rocke_for
{
    rocke_op_t* op;
    rocke_value_t* iv;
    rocke_region_t* body;
    rocke_value_t** iter_vars; /* loop-carried induction values               */
    int num_iter_vars;
} rocke_for_t;

/* If handle: the C analog of _IfBuilder. */
typedef struct rocke_if
{
    rocke_op_t* op;
    rocke_region_t* then_region;
} rocke_if_t;

/* (name, init) pair for scf_for_iter. */
typedef struct rocke_iter_arg
{
    const char* name; /* WITHOUT leading '%'                          */
    rocke_value_t* init;
} rocke_iter_arg_t;

/* Options for inline_asm (Python keyword-only args). */
typedef struct rocke_inline_asm_opts
{
    bool sideeffect; /* default true                                     */
    bool convergent; /* default false                                    */
    bool sideeffect_set;
    bool convergent_set;
} rocke_inline_asm_opts_t;

/* ============================== TYPE SYSTEM ============================== */

/* Interned scalar singletons (Python module-level I1, F32, ...). Always valid;
 * never NULL; never arena-owned (static storage). */
const rocke_type_t* rocke_i1(void);
const rocke_type_t* rocke_i8(void);
const rocke_type_t* rocke_i16(void);
const rocke_type_t* rocke_i32(void);
const rocke_type_t* rocke_i64(void);
const rocke_type_t* rocke_bf16(void);
const rocke_type_t* rocke_f16(void);
const rocke_type_t* rocke_f32(void);
const rocke_type_t* rocke_fp8e4m3(void);
const rocke_type_t* rocke_bf8e5m2(void);

/* Look up a scalar singleton by canonical name ("i32",...); NULL if unknown. */
const rocke_type_t* rocke_scalar_by_name(const char* name);

/* Composite type constructors (arena-allocated, name computed Python-identically).
 * VectorType(elem,count) -> "vec<{elem}x{count}>"
 * PtrType(pointee,space) -> "ptr<{pointee},{space}>"
 * SmemType(elem,shape)   -> "smem<{elem}, [{d0}x{d1}...]>"  */
const rocke_type_t* rocke_vector_type(rocke_ir_builder_t* b, const rocke_type_t* elem, int count);
const rocke_type_t*
    rocke_ptr_type(rocke_ir_builder_t* b, const rocke_type_t* pointee, const char* space);
const rocke_type_t*
    rocke_smem_type(rocke_ir_builder_t* b, const rocke_type_t* elem, const int* shape, int rank);

/* Structural type equality (matches Python frozen-dataclass __eq__: compares by
 * canonical name, which encodes kind + components). */
bool rocke_type_eq(const rocke_type_t* a, const rocke_type_t* b);

/* AMDGPU buffer-load AUX cache-coherency hints (Python module constants). */
typedef enum rocke_cache_policy
{
    ROCKE_CACHE_ALL = 0,
    ROCKE_CACHE_GLOBAL = 1,
    ROCKE_CACHE_STREAM = 2,
    ROCKE_NON_TEMPORAL = 3
} rocke_cache_policy_t;

/* ============================== ATTR MAP ================================ */

void rocke_attr_map_init(rocke_attr_map_t* m);
void rocke_attr_set_int(rocke_ir_builder_t* b, rocke_attr_map_t* m, const char* key, int64_t v);
void rocke_attr_set_float(rocke_ir_builder_t* b, rocke_attr_map_t* m, const char* key, double v);
void rocke_attr_set_str(rocke_ir_builder_t* b, rocke_attr_map_t* m, const char* key, const char* v);
void rocke_attr_set_bool(rocke_ir_builder_t* b, rocke_attr_map_t* m, const char* key, bool v);
/* Set a list of bare ints (serialized as l:[ i:v0, i:v1, ... ]). */
void rocke_attr_set_int_list(
    rocke_ir_builder_t* b, rocke_attr_map_t* m, const char* key, const int64_t* vals, int count);
/* Returns the entry for `key`, or NULL if absent. */
const rocke_attr_value_t* rocke_attr_get(const rocke_attr_map_t* m, const char* key);
bool rocke_attr_get_int(const rocke_attr_map_t* m, const char* key, int64_t* out);
bool rocke_attr_get_float(const rocke_attr_map_t* m, const char* key, double* out);
const char* rocke_attr_get_str(const rocke_attr_map_t* m, const char* key); /* NULL if absent */
bool rocke_attr_get_bool(const rocke_attr_map_t* m, const char* key, bool dflt);

/* ============================ OPCODE TABLE ============================== */

/* Canonical dotted name for an opcode ("arith.add"); "" for ROCKE_OP_INVALID. */
const char* rocke_opcode_name(rocke_opcode_t op);
/* Reverse map: dotted name -> opcode; ROCKE_OP_INVALID if unknown. */
rocke_opcode_t rocke_opcode_from_name(const char* name);
/* True if the op is side-effect-free (Python PURE_OP_NAMES / is_pure_op_name). */
bool rocke_opcode_is_pure(rocke_opcode_t op);

/* ============================== OP GETTERS ============================== */

/* Python @property Op.result: requires exactly one result, else sets error. */
rocke_value_t* rocke_op_result(rocke_ir_builder_t* b, rocke_op_t* op);
/* Python @property Op.is_pure: attrs["pure"] override, else opcode purity. */
bool rocke_op_is_pure(const rocke_op_t* op);
/* Python @property KernelDef.max_workgroup_size (default 256). */
int rocke_kernel_max_workgroup_size(const rocke_kernel_def_t* k);

/* ============================== BUILDER ================================= */

/* Construct/destruct. rocke_ir_builder_new allocates the builder's own arena and
 * an empty kernel with an "entry" region as the current region. */
rocke_status_t rocke_ir_builder_init(rocke_ir_builder_t* b, const char* kernel_name);
void rocke_ir_builder_free(rocke_ir_builder_t* b);
bool rocke_ir_builder_ok(const rocke_ir_builder_t* b);
rocke_status_t rocke_ir_builder_status(const rocke_ir_builder_t* b);
const char* rocke_ir_builder_error(const rocke_ir_builder_t* b);
rocke_kernel_def_t* rocke_ir_builder_kernel(rocke_ir_builder_t* b);

/* Low-level plumbing (mirrors IRBuilder._fresh/_emit/push_region/pop_region and
 * the generic _op). Lowerers rarely need these; emitters/tests may. */
const char* rocke_b_fresh(rocke_ir_builder_t* b, const char* prefix);
void rocke_b_emit(rocke_ir_builder_t* b, rocke_op_t* op);
void rocke_b_region_enter(rocke_ir_builder_t* b, rocke_region_t* r); /* push */
void rocke_b_region_leave(rocke_ir_builder_t* b); /* pop  */
rocke_region_t* rocke_b_current_region(rocke_ir_builder_t* b);

/* Generic op builder. Creates fresh result Values (one per result_types entry,
 * named with result_name_hint), builds the Op, links results back, emits it, and
 * returns it. attrs/regions may be NULL. This is IRBuilder._op. */
rocke_op_t* rocke_b_op(rocke_ir_builder_t* b,
                       rocke_opcode_t opcode,
                       rocke_value_t* const* operands,
                       int num_operands,
                       const rocke_type_t* const* result_types,
                       int num_results,
                       const rocke_attr_map_t* attrs,
                       rocke_region_t* const* regions,
                       int num_regions,
                       const char* result_name_hint,
                       const char* loc);

/* ----- params ----- */
rocke_value_t* rocke_b_param(rocke_ir_builder_t* b,
                             const char* name,
                             const rocke_type_t* t,
                             const rocke_param_opts_t* opts);
rocke_value_t* rocke_b_get_param(rocke_ir_builder_t* b, const char* name);

/* ----- arith constants ----- */
rocke_value_t* rocke_b_const_i32(rocke_ir_builder_t* b, int64_t value);
rocke_value_t* rocke_b_const_i64(rocke_ir_builder_t* b, int64_t value);
rocke_value_t* rocke_b_const_f32(rocke_ir_builder_t* b, double value);
rocke_value_t* rocke_b_fp16_zero(rocke_ir_builder_t* b);
rocke_value_t* rocke_b_zero_vec_f32(rocke_ir_builder_t* b, int n);
rocke_value_t* rocke_b_zero_vec(rocke_ir_builder_t* b, const rocke_type_t* elem, int n);

/* ----- arith integer / logic ----- */
rocke_value_t* rocke_b_add(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_sub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_mul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_div(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_mod(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_land(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_lor(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_lnot(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_smax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_smin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_xor(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_shl(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_lshr(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_umul_hi_i32(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);

/* ----- arith float ----- */
rocke_value_t* rocke_b_fadd(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_fsub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_fmul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_fdiv(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_fneg(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_fabs(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t*
    rocke_b_fma(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d);
rocke_value_t* rocke_b_fmax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_fmin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t*
    rocke_b_fmax3(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d);
rocke_value_t*
    rocke_b_fmin3(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d);
rocke_value_t* rocke_b_clamp_f32(rocke_ir_builder_t* b,
                                 rocke_value_t* v,
                                 rocke_value_t* lo,
                                 rocke_value_t* hi);

/* ----- comparisons (return i1) ----- */
rocke_value_t* rocke_b_cmp_lt(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_cmp_le(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_cmp_gt(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_cmp_ge(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_cmp_eq(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_cmp_ne(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
/* pred in {olt,ole,ogt,oge,oeq,one,ord,uno} */
rocke_value_t*
    rocke_b_fcmp(rocke_ir_builder_t* b, const char* pred, rocke_value_t* a, rocke_value_t* c);

/* ----- math ----- */
rocke_value_t* rocke_b_exp2(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_log2(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_rcp(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_rcp_fast(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_sqrt(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_rsqrt(rocke_ir_builder_t* b, rocke_value_t* a);
rocke_value_t* rocke_b_tanh(rocke_ir_builder_t* b, rocke_value_t* a);

/* ----- casts / conversions ----- */
rocke_value_t* rocke_b_zext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_sext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_trunc(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_bitcast(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_select(rocke_ir_builder_t* b,
                              rocke_value_t* cond,
                              rocke_value_t* lhs,
                              rocke_value_t* rhs);
rocke_value_t* rocke_b_masked_select(rocke_ir_builder_t* b,
                                     rocke_value_t* cond,
                                     rocke_value_t* lhs,
                                     rocke_value_t* rhs);
rocke_value_t* rocke_b_trunc_f32_to_f16(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_rint_f32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cast_to_f32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t*
    rocke_b_cast_f32_to(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_sitofp_f32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_fp8_to_f32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_bf8_to_f32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_pk_f32_fp8x4(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_pk_f32_bf8x4(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_scalef32_pk_f32_fp8x4(rocke_ir_builder_t* b,
                                                 rocke_value_t* v,
                                                 rocke_value_t* scale);
rocke_value_t* rocke_b_cvt_scalef32_pk_f32_bf8x4(rocke_ir_builder_t* b,
                                                 rocke_value_t* v,
                                                 rocke_value_t* scale);
rocke_value_t* rocke_b_cvt_f32_to_fp8(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_f32_to_bf8(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_f32_to_i8_sat(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_pk_fp8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_pk_bf8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_cvt_pk_i8_f32x4(rocke_ir_builder_t* b, rocke_value_t* v);

/* ----- atomics ----- */
rocke_value_t* rocke_b_global_atomic_add(rocke_ir_builder_t* b,
                                         rocke_value_t* ptr,
                                         rocke_value_t* idx,
                                         rocke_value_t* value,
                                         const char* ordering /* NULL=>monotonic */);
rocke_value_t* rocke_b_lds_atomic_add(rocke_ir_builder_t* b,
                                      rocke_value_t* smem,
                                      rocke_value_t* const* indices,
                                      int num_indices,
                                      rocke_value_t* value,
                                      const char* ordering);
rocke_value_t* rocke_b_global_atomic_add_pk_bf16(rocke_ir_builder_t* b,
                                                 rocke_value_t* ptr,
                                                 rocke_value_t* idx,
                                                 rocke_value_t* value,
                                                 const char* ordering);

/* ----- gpu ids ----- */
rocke_value_t* rocke_b_thread_id_x(rocke_ir_builder_t* b);
rocke_value_t* rocke_b_block_id_x(rocke_ir_builder_t* b);
rocke_value_t* rocke_b_block_id_y(rocke_ir_builder_t* b);
rocke_value_t* rocke_b_block_id_z(rocke_ir_builder_t* b);

/* ----- global memory ----- */
rocke_value_t* rocke_b_smem_alloc(rocke_ir_builder_t* b,
                                  const rocke_type_t* elem,
                                  const int* shape,
                                  int rank,
                                  const char* name_hint);
rocke_value_t* rocke_b_global_load(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   const rocke_type_t* dtype,
                                   int align /* <=0 => 1 */);
rocke_value_t* rocke_b_global_load_f16(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align);
rocke_value_t* rocke_b_global_load_f32(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align);
rocke_value_t* rocke_b_global_load_i32(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align);
rocke_value_t* rocke_b_global_load_i64(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align);
rocke_value_t* rocke_b_global_load_bf16(rocke_ir_builder_t* b,
                                        rocke_value_t* ptr,
                                        rocke_value_t* idx,
                                        int align);
rocke_value_t* rocke_b_global_load_fp8e4m3(rocke_ir_builder_t* b,
                                           rocke_value_t* ptr,
                                           rocke_value_t* idx,
                                           int align);
rocke_value_t* rocke_b_masked_global_load(rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          rocke_value_t* idx,
                                          rocke_value_t* mask,
                                          rocke_value_t* other,
                                          const rocke_type_t* dtype,
                                          int align);
void rocke_b_global_store(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, rocke_value_t* value, int align);
rocke_value_t* rocke_b_global_load_vN(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      rocke_value_t* idx,
                                      const rocke_type_t* dtype,
                                      int n,
                                      int align /* <=0 => default */);
rocke_value_t* rocke_b_global_load_vN_f16(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, int n, int align);

/* ----- vector ops ----- */
rocke_value_t* rocke_b_vector_add(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_sub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_mul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_and(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_or(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_shl(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_lshr(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_smax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_smin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t* rocke_b_vector_max(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);
rocke_value_t*
    rocke_b_vector_fma(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d);
rocke_value_t* rocke_b_vector_sum(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_vector_reduce_max(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_vector_splat(rocke_ir_builder_t* b, rocke_value_t* scalar, int n);
rocke_value_t* rocke_b_vector_select(rocke_ir_builder_t* b,
                                     rocke_value_t* mask,
                                     rocke_value_t* lhs,
                                     rocke_value_t* rhs);
rocke_value_t*
    rocke_b_vector_cmp(rocke_ir_builder_t* b, const char* pred, rocke_value_t* a, rocke_value_t* c);
rocke_value_t*
    rocke_b_vector_trunc(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t*
    rocke_b_vector_sext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);

/* ----- LDS (shared memory) ----- */
void rocke_b_smem_store_f16(rocke_ir_builder_t* b,
                            rocke_value_t* smem,
                            rocke_value_t* const* indices,
                            int num_indices,
                            rocke_value_t* value);
void rocke_b_smem_store_vN(rocke_ir_builder_t* b,
                           rocke_value_t* smem,
                           rocke_value_t* const* indices,
                           int num_indices,
                           rocke_value_t* value,
                           int n);
void rocke_b_smem_store_vN_f16(rocke_ir_builder_t* b,
                               rocke_value_t* smem,
                               rocke_value_t* const* indices,
                               int num_indices,
                               rocke_value_t* value,
                               int n);
rocke_value_t* rocke_b_smem_load_v4_f16(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* row,
                                        rocke_value_t* col);
rocke_value_t* rocke_b_smem_load_vN(rocke_ir_builder_t* b,
                                    rocke_value_t* smem,
                                    rocke_value_t* const* indices,
                                    int num_indices,
                                    const rocke_type_t* dtype,
                                    int n);
rocke_value_t* rocke_b_smem_load_vN_f16(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        int n);

/* ----- target-neutral MMA ----- */
/* op_id is the atom identifier ("mfma_f32_16x16x16_f16", "wmma_...", ...).
 * extra carries scaled-MX scale operands (a_scale,b_scale) or is NULL/0. */
rocke_value_t* rocke_b_mma(rocke_ir_builder_t* b,
                           const char* op_id,
                           rocke_value_t* a,
                           rocke_value_t* bb,
                           rocke_value_t* c,
                           rocke_value_t* const* extra,
                           int num_extra);

/* ----- inline asm ----- */
/* operands/result_types are explicit arrays; constraints/template are strings.
 * Returns the op (results accessible via op->results) since asm may be 0/1/N
 * results. */
rocke_op_t* rocke_b_inline_asm(rocke_ir_builder_t* b,
                               const char* asm_template,
                               const char* constraints,
                               rocke_value_t* const* operands,
                               int num_operands,
                               const rocke_type_t* const* result_types,
                               int num_results,
                               const rocke_inline_asm_opts_t* opts);

/* ----- cross-lane / vector pack-extract ----- */
rocke_value_t* rocke_b_readfirstlane(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_lane_id(rocke_ir_builder_t* b);
rocke_value_t* rocke_b_vec_extract(rocke_ir_builder_t* b, rocke_value_t* v, int i);
rocke_value_t*
    rocke_b_vec_insert(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* scalar, int i);
rocke_value_t* rocke_b_vec_pack(rocke_ir_builder_t* b,
                                rocke_value_t* const* components,
                                int num_components,
                                const rocke_type_t* elem);
rocke_value_t* rocke_b_vec_concat(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb);

/* ----- ISA-named MMA wrappers (thin wrappers over rocke_b_mma; kept for parity
 * with the legacy Python helpers so emitters can call them by name). All take
 * (a, b, c) and return <c_frag_len x acc_elem>. The scaled MX atom takes the
 * two extra E8M0 scale operands. */
rocke_value_t* rocke_b_mfma_f32_16x16x16_f16(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x32_f16(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x16_bf16(rocke_ir_builder_t* b,
                                              rocke_value_t* a,
                                              rocke_value_t* bb,
                                              rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x32_bf16(rocke_ir_builder_t* b,
                                              rocke_value_t* a,
                                              rocke_value_t* bb,
                                              rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x32_fp8(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x32_bf8(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x8_f16(rocke_ir_builder_t* b,
                                            rocke_value_t* a,
                                            rocke_value_t* bb,
                                            rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x8_bf16(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x16_f16(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x16_bf16(rocke_ir_builder_t* b,
                                              rocke_value_t* a,
                                              rocke_value_t* bb,
                                              rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x16_fp8(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_32x32x16_bf8(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_4x4x4_f16(rocke_ir_builder_t* b,
                                          rocke_value_t* a,
                                          rocke_value_t* bb,
                                          rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x128_fp4(rocke_ir_builder_t* b,
                                              rocke_value_t* a,
                                              rocke_value_t* bb,
                                              rocke_value_t* c);
rocke_value_t* rocke_b_mfma_f32_16x16x96_fp6(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_mfma_scale_f32_16x16x128_f8f6f4(rocke_ir_builder_t* b,
                                                       rocke_value_t* a,
                                                       rocke_value_t* bb,
                                                       rocke_value_t* c,
                                                       rocke_value_t* a_scale,
                                                       rocke_value_t* b_scale);
rocke_value_t* rocke_b_wmma_f32_16x16x16_f16(rocke_ir_builder_t* b,
                                             rocke_value_t* a,
                                             rocke_value_t* bb,
                                             rocke_value_t* c);
rocke_value_t* rocke_b_wmma_f32_16x16x16_bf16(rocke_ir_builder_t* b,
                                              rocke_value_t* a,
                                              rocke_value_t* bb,
                                              rocke_value_t* c);
rocke_value_t* rocke_b_wmma_gfx12_f32_16x16x16_f16(rocke_ir_builder_t* b,
                                                   rocke_value_t* a,
                                                   rocke_value_t* bb,
                                                   rocke_value_t* c);
rocke_value_t* rocke_b_wmma_gfx12_f32_16x16x16_bf16(rocke_ir_builder_t* b,
                                                    rocke_value_t* a,
                                                    rocke_value_t* bb,
                                                    rocke_value_t* c);

/* ----- multi-output inline asm (LLVM literal-struct return). Returns the op;
 * its results[] holds the N output Values in declaration order. */
rocke_op_t* rocke_b_inline_asm_multi(rocke_ir_builder_t* b,
                                     const char* asm_template,
                                     const char* constraints,
                                     rocke_value_t* const* operands,
                                     int num_operands,
                                     const rocke_type_t* const* result_types,
                                     int num_results,
                                     const rocke_inline_asm_opts_t* opts);

/* ----- register-fragment reshape (P13) ----- */
rocke_value_t* rocke_b_register_p_from_qk_c(rocke_ir_builder_t* b,
                                            rocke_value_t* qk_c,
                                            const rocke_type_t* target_dtype);

/* ----- distributed / cooperative epilogue stores ----- */
void rocke_b_smem_store_distributed(rocke_ir_builder_t* b,
                                    rocke_value_t* smem,
                                    const rocke_attr_map_t* layout_attrs,
                                    rocke_value_t* values);
void rocke_b_cooperative_global_store(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      rocke_value_t* addrs,
                                      rocke_value_t* values);

/* ----- uniform / wave-scalar helpers ----- */
rocke_value_t* rocke_b_pin_sgpr(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_to_sgpr_u32(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t* rocke_b_wave_all(rocke_ir_builder_t* b, rocke_value_t* predicate);
rocke_value_t* rocke_b_wave_any(rocke_ir_builder_t* b, rocke_value_t* predicate);
rocke_value_t* rocke_b_wave_ballot(rocke_ir_builder_t* b, rocke_value_t* predicate);

/* ----- cross-lane permute / dpp ----- */
rocke_value_t* rocke_b_ds_bpermute(rocke_ir_builder_t* b, rocke_value_t* addr, rocke_value_t* data);
rocke_value_t*
    rocke_b_ds_bpermute_b64(rocke_ir_builder_t* b, rocke_value_t* addr, rocke_value_t* data);
rocke_value_t* rocke_b_ds_swizzle_xor(rocke_ir_builder_t* b, rocke_value_t* data, int xor_mask);
/* mov_dpp: exactly one of row_shr/row_shl must be >= 0 (the other < 0 = unset). */
rocke_value_t* rocke_b_mov_dpp(
    rocke_ir_builder_t* b, rocke_value_t* data, int row_shr, int row_shl, bool bound_ctrl);
/* permlane32_swap returns two values via out params (new_lo, new_hi). */
void rocke_b_permlane32_swap(rocke_ir_builder_t* b,
                             rocke_value_t* lo,
                             rocke_value_t* hi,
                             rocke_value_t** out_lo,
                             rocke_value_t** out_hi);
rocke_value_t* rocke_b_perm_b32(rocke_ir_builder_t* b,
                                rocke_value_t* src0,
                                rocke_value_t* src1,
                                rocke_value_t* sel);
rocke_value_t* rocke_b_permlanex16(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t*
    rocke_b_byte_perm(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, int64_t sel);
rocke_value_t* rocke_b_warp_shuffle_xor(rocke_ir_builder_t* b, rocke_value_t* v, int lane_xor);

/* ----- transpose LDS reads ----- */
rocke_value_t* rocke_b_ds_read_tr16_b64(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        const rocke_type_t* dtype /* NULL=>f16 */);
rocke_value_t* rocke_b_ds_read_tr16_b128(rocke_ir_builder_t* b,
                                         rocke_value_t* smem,
                                         rocke_value_t* const* indices,
                                         int num_indices,
                                         const rocke_type_t* dtype /* NULL=>f16 */);
rocke_value_t* rocke_b_ds_read_tr_b8(rocke_ir_builder_t* b,
                                     rocke_value_t* smem,
                                     rocke_value_t* const* indices,
                                     int num_indices,
                                     const rocke_type_t* dtype /* NULL=>fp8e4m3 */);

/* ----- vector bitcast / packed f32->f16 conversion ----- */
rocke_value_t*
    rocke_b_vec_bitcast(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);
rocke_value_t* rocke_b_vec_trunc_f32_to_f16(rocke_ir_builder_t* b, rocke_value_t* v);
rocke_value_t*
    rocke_b_vec_cast_f32_to(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target);

/* ----- LDS pointer arithmetic + async DRAM->LDS ----- */
rocke_value_t* rocke_b_smem_addr_of(rocke_ir_builder_t* b, rocke_value_t* smem);
rocke_value_t*
    rocke_b_smem_ptr_add(rocke_ir_builder_t* b, rocke_value_t* lds_addr, rocke_value_t* byte_off);
void rocke_b_async_buffer_load_lds_addr(rocke_ir_builder_t* b,
                                        rocke_value_t* rsrc,
                                        rocke_value_t* lds_addr,
                                        rocke_value_t* voffset,
                                        rocke_value_t* soffset,
                                        int dwords,
                                        int coherency);
void rocke_b_async_buffer_load_lds(rocke_ir_builder_t* b,
                                   rocke_value_t* rsrc,
                                   rocke_value_t* lds_ptr,
                                   rocke_value_t* voffset,
                                   rocke_value_t* soffset,
                                   int dwords,
                                   int coherency);
void rocke_b_global_load_lds(rocke_ir_builder_t* b,
                             rocke_value_t* src_ptr,
                             rocke_value_t* byte_off,
                             rocke_value_t* lds_addr,
                             int size_bytes,
                             int coherency);

/* ----- global pointer arithmetic + buffer resource descriptors ----- */
rocke_value_t*
    rocke_b_global_ptr_add(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* byte_off);
rocke_value_t*
    rocke_b_buffer_rsrc(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* num_bytes);
rocke_value_t* rocke_b_buffer_load_vN_f16(rocke_ir_builder_t* b,
                                          rocke_value_t* rsrc,
                                          rocke_value_t* voffset,
                                          rocke_value_t* soffset,
                                          int dwords);
rocke_value_t* rocke_b_buffer_load_f16(rocke_ir_builder_t* b,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voffset,
                                       rocke_value_t* soffset);
void rocke_b_buffer_store_vN_f16(rocke_ir_builder_t* b,
                                 rocke_value_t* rsrc,
                                 rocke_value_t* voffset,
                                 rocke_value_t* soffset,
                                 rocke_value_t* value,
                                 int dwords);
void rocke_b_buffer_store_f16(rocke_ir_builder_t* b,
                              rocke_value_t* rsrc,
                              rocke_value_t* voffset,
                              rocke_value_t* soffset,
                              rocke_value_t* value);

/* ----- f32 LDS ops (cshuffle epilogue) ----- */
rocke_value_t* rocke_b_smem_alloc_f32(rocke_ir_builder_t* b,
                                      const int* shape,
                                      int rank,
                                      const char* name_hint);
void rocke_b_smem_store_vN_f32(rocke_ir_builder_t* b,
                               rocke_value_t* smem,
                               rocke_value_t* const* indices,
                               int num_indices,
                               rocke_value_t* value,
                               int n);
rocke_value_t* rocke_b_smem_load_vN_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        int n);

/* ----- vectorised global stores + split-K atomics ----- */
void rocke_b_global_store_vN(rocke_ir_builder_t* b,
                             rocke_value_t* ptr,
                             rocke_value_t* idx,
                             rocke_value_t* value,
                             int n,
                             int align /* <=0 => default */);
void rocke_b_global_store_vN_f16(rocke_ir_builder_t* b,
                                 rocke_value_t* ptr,
                                 rocke_value_t* idx,
                                 rocke_value_t* value,
                                 int n,
                                 int align);
void rocke_b_global_atomic_add_f32(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   rocke_value_t* value);
void rocke_b_store_f16(rocke_ir_builder_t* b,
                       rocke_value_t* ptr,
                       rocke_value_t* idx,
                       rocke_value_t* value);
rocke_value_t* rocke_b_zero_vec_f16(rocke_ir_builder_t* b, int n);

/* ----- barriers / scheduling ----- */
void rocke_b_sync(rocke_ir_builder_t* b);
void rocke_b_s_barrier_bare(rocke_ir_builder_t* b);
void rocke_b_sync_half_block(rocke_ir_builder_t* b, rocke_value_t* half_selector);
void rocke_b_sync_lds_only(rocke_ir_builder_t* b);
/* s_waitcnt: pass -1 to leave a counter alone, 0 to fully drain. */
void rocke_b_s_waitcnt(rocke_ir_builder_t* b, int vmcnt, int lgkmcnt, int expcnt);
void rocke_b_s_setprio(rocke_ir_builder_t* b, int level);
void rocke_b_iglp_opt(rocke_ir_builder_t* b, int level);
void rocke_b_sched_barrier(rocke_ir_builder_t* b, int mask);
void rocke_b_sched_group_barrier(rocke_ir_builder_t* b, int mask, int count, int group);

/* ----- compile-time loops (Python static_for / unroll are pure host control
 * flow; in C the caller simply writes a C for-loop calling the body. No IR op is
 * emitted, so no builder entry point is needed. Documented here for parity.) */

/* ----- control flow ----- */
rocke_for_t rocke_b_scf_for(rocke_ir_builder_t* b,
                            rocke_value_t* lo,
                            rocke_value_t* hi,
                            rocke_value_t* step,
                            const char* iv_name /* NULL=>"k0" */);
rocke_for_t rocke_b_scf_for_iter(rocke_ir_builder_t* b,
                                 rocke_value_t* lo,
                                 rocke_value_t* hi,
                                 rocke_value_t* step,
                                 const rocke_iter_arg_t* iter_args,
                                 int num_iter_args,
                                 const char* iv_name /* NULL=>"k0" */,
                                 bool unroll,
                                 bool elide_trailing_barrier);
void rocke_b_scf_yield(rocke_ir_builder_t* b, rocke_value_t* const* values, int num_values);
rocke_if_t rocke_b_scf_if(rocke_ir_builder_t* b, rocke_value_t* cond);
void rocke_b_ret(rocke_ir_builder_t* b);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_IR_H */
