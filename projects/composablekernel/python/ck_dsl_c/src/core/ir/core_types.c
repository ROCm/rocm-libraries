/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_core_types.c -- the leaf data layer of the C99 IR builder: the Type
 * system (scalar singletons + composite constructors + equality), the
 * insertion-ordered attribute map (variant key->value store), the opcode
 * name/reverse/purity tables, and a handful of Op/KernelDef getters.
 *
 * Faithful port of the corresponding pieces of ck_dsl.core.ir (Type,
 * VectorType, PtrType, SmemType, Op.result/is_pure, KernelDef.max_workgroup_size,
 * PURE_OP_NAMES / is_pure_op_name, and the op.name string vocabulary).
 *
 * All composite types and attr storage live in the builder arena; scalar
 * singletons are static. Nothing here frees individually.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h"

/* ============================== TYPE SYSTEM ============================== */

/* Interned scalar singletons. Static storage: never NULL, never arena-owned,
 * value-compared by canonical name. Mirrors Python module-level I1/F32/... */

#define CKC_SCALAR_SINGLETON(fn, kindenum, cname)            \
    const ckc_type_t *fn(void) {                             \
        static const ckc_type_t t = {                        \
            /* kind    */ CKC_TYPE_SCALAR,                   \
            /* name    */ cname,                             \
            /* scalar  */ kindenum,                          \
            /* elem    */ NULL,                              \
            /* count   */ 0,                                 \
            /* pointee */ NULL,                              \
            /* space   */ NULL,                              \
            /* shape   */ NULL,                              \
            /* rank    */ 0                                  \
        };                                                   \
        return &t;                                           \
    }

CKC_SCALAR_SINGLETON(ckc_i1,      CKC_SCALAR_I1,      "i1")
CKC_SCALAR_SINGLETON(ckc_i8,      CKC_SCALAR_I8,      "i8")
CKC_SCALAR_SINGLETON(ckc_i16,     CKC_SCALAR_I16,     "i16")
CKC_SCALAR_SINGLETON(ckc_i32,     CKC_SCALAR_I32,     "i32")
CKC_SCALAR_SINGLETON(ckc_i64,     CKC_SCALAR_I64,     "i64")
CKC_SCALAR_SINGLETON(ckc_bf16,    CKC_SCALAR_BF16,    "bf16")
CKC_SCALAR_SINGLETON(ckc_f16,     CKC_SCALAR_F16,     "f16")
CKC_SCALAR_SINGLETON(ckc_f32,     CKC_SCALAR_F32,     "f32")
CKC_SCALAR_SINGLETON(ckc_fp8e4m3, CKC_SCALAR_FP8E4M3, "fp8e4m3")
CKC_SCALAR_SINGLETON(ckc_bf8e5m2, CKC_SCALAR_BF8E5M2, "bf8e5m2")

#undef CKC_SCALAR_SINGLETON

const ckc_type_t *ckc_scalar_by_name(const char *name) {
    if (!name)
        return NULL;
    if (strcmp(name, "i1") == 0)      return ckc_i1();
    if (strcmp(name, "i8") == 0)      return ckc_i8();
    if (strcmp(name, "i16") == 0)     return ckc_i16();
    if (strcmp(name, "i32") == 0)     return ckc_i32();
    if (strcmp(name, "i64") == 0)     return ckc_i64();
    if (strcmp(name, "bf16") == 0)    return ckc_bf16();
    if (strcmp(name, "f16") == 0)     return ckc_f16();
    if (strcmp(name, "f32") == 0)     return ckc_f32();
    if (strcmp(name, "fp8e4m3") == 0) return ckc_fp8e4m3();
    if (strcmp(name, "bf8e5m2") == 0) return ckc_bf8e5m2();
    return NULL;
}

/* VectorType(elem, count) -> "vec<{elem}x{count}>" */
const ckc_type_t *ckc_vector_type(ckc_ir_builder_t *b, const ckc_type_t *elem,
                                  int count) {
    ckc_type_t *t;
    if (!ckc_i_live(b))
        return NULL;
    if (!elem)
        return (const ckc_type_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vector_type: NULL element type");
    t = (ckc_type_t *)ckc_arena_calloc(&b->arena, sizeof(*t));
    if (!t)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "vector_type: OOM");
    t->kind = CKC_TYPE_VECTOR;
    t->elem = elem;
    t->count = count;
    t->name = ckc_arena_printf(&b->arena, "vec<%sx%d>", elem->name, count);
    if (!t->name)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "vector_type: OOM");
    return t;
}

/* PtrType(pointee, space) -> "ptr<{pointee},{space}>" */
const ckc_type_t *ckc_ptr_type(ckc_ir_builder_t *b, const ckc_type_t *pointee,
                               const char *space) {
    ckc_type_t *t;
    if (!ckc_i_live(b))
        return NULL;
    if (!pointee)
        return (const ckc_type_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "ptr_type: NULL pointee type");
    if (!space)
        space = "global";
    t = (ckc_type_t *)ckc_arena_calloc(&b->arena, sizeof(*t));
    if (!t)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "ptr_type: OOM");
    t->kind = CKC_TYPE_PTR;
    t->pointee = pointee;
    t->space = ckc_arena_strdup(&b->arena, space);
    if (!t->space)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "ptr_type: OOM");
    t->name = ckc_arena_printf(&b->arena, "ptr<%s,%s>", pointee->name, t->space);
    if (!t->name)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "ptr_type: OOM");
    return t;
}

/* SmemType(elem, shape) -> "smem<{elem}, [{d0}x{d1}...]>" */
const ckc_type_t *ckc_smem_type(ckc_ir_builder_t *b, const ckc_type_t *elem,
                                const int *shape, int rank) {
    ckc_type_t *t;
    int *shape_copy = NULL;
    char *buf;
    size_t cap, off;
    int i;

    if (!ckc_i_live(b))
        return NULL;
    if (!elem)
        return (const ckc_type_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "smem_type: NULL element type");
    if (rank < 0)
        rank = 0;

    t = (ckc_type_t *)ckc_arena_calloc(&b->arena, sizeof(*t));
    if (!t)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "smem_type: OOM");

    if (rank > 0) {
        shape_copy = (int *)ckc_arena_alloc(&b->arena, (size_t)rank * sizeof(int));
        if (!shape_copy)
            return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                     "smem_type: OOM");
        for (i = 0; i < rank; ++i)
            shape_copy[i] = shape ? shape[i] : 0;
    }

    t->kind = CKC_TYPE_SMEM;
    t->elem = elem;
    t->shape = shape_copy;
    t->rank = rank;

    /* Build the "[d0xd1x...]" body, then the full canonical name. The Python
     * form is f"smem<{elem.name}, [{'x'.join(...)}]>". */
    cap = 32 + (size_t)rank * 16;
    buf = (char *)ckc_arena_alloc(&b->arena, cap);
    if (!buf)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "smem_type: OOM");
    off = 0;
    for (i = 0; i < rank; ++i) {
        char tmp[16];
        int n = snprintf(tmp, sizeof(tmp), "%d", shape_copy ? shape_copy[i] : 0);
        if (n < 0)
            n = 0;
        if (i > 0 && off + 1 < cap)
            buf[off++] = 'x';
        if (off + (size_t)n < cap) {
            memcpy(buf + off, tmp, (size_t)n);
            off += (size_t)n;
        }
    }
    buf[off < cap ? off : cap - 1] = '\0';

    t->name = ckc_arena_printf(&b->arena, "smem<%s, [%s]>", elem->name, buf);
    if (!t->name)
        return (const ckc_type_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                 "smem_type: OOM");
    return t;
}

/* Structural equality == Python frozen-dataclass __eq__: canonical name encodes
 * kind + all components, so a name compare is sufficient. NULL-safe. */
bool ckc_type_eq(const ckc_type_t *a, const ckc_type_t *b) {
    if (a == b)
        return true;
    if (!a || !b)
        return false;
    if (!a->name || !b->name)
        return false;
    return strcmp(a->name, b->name) == 0;
}

/* ============================== ATTR MAP ================================ */

void ckc_attr_map_init(ckc_attr_map_t *m) {
    if (!m)
        return;
    m->entries = NULL;
    m->count = 0;
    m->cap = 0;
}

/* Find an existing entry by key (linear; maps are small). */
static ckc_attr_entry_t *ckc_attr_find(ckc_attr_map_t *m, const char *key) {
    int i;
    if (!m || !key)
        return NULL;
    for (i = 0; i < m->count; ++i) {
        if (m->entries[i].key && strcmp(m->entries[i].key, key) == 0)
            return &m->entries[i];
    }
    return NULL;
}

/* Reserve a slot for `key`: return the existing entry (overwrite semantics,
 * matching Python dict assignment) or append a fresh one. Growth reallocates
 * into the arena (old block is leaked into the arena, freed in bulk). Returns
 * NULL on a dead builder or OOM (sticky error set). */
static ckc_attr_entry_t *ckc_attr_reserve(ckc_ir_builder_t *b,
                                          ckc_attr_map_t *m, const char *key) {
    ckc_attr_entry_t *e;
    if (!ckc_i_live(b) || !m || !key)
        return NULL;

    e = ckc_attr_find(m, key);
    if (e)
        return e;

    if (m->count >= m->cap) {
        int new_cap = m->cap ? m->cap * 2 : 4;
        ckc_attr_entry_t *ne = (ckc_attr_entry_t *)ckc_arena_alloc(
            &b->arena, (size_t)new_cap * sizeof(ckc_attr_entry_t));
        if (!ne)
            return (ckc_attr_entry_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                                     "attr: OOM");
        if (m->count > 0 && m->entries)
            memcpy(ne, m->entries, (size_t)m->count * sizeof(ckc_attr_entry_t));
        m->entries = ne;
        m->cap = new_cap;
    }

    e = &m->entries[m->count];
    e->key = ckc_arena_strdup(&b->arena, key);
    if (!e->key)
        return (ckc_attr_entry_t *)ckc_i_set_err(b, CKC_ERR_OOM, "attr: OOM");
    memset(&e->value, 0, sizeof(e->value));
    m->count += 1;
    return e;
}

void ckc_attr_set_int(ckc_ir_builder_t *b, ckc_attr_map_t *m, const char *key,
                      int64_t v) {
    ckc_attr_entry_t *e = ckc_attr_reserve(b, m, key);
    if (!e)
        return;
    e->value.kind = CKC_ATTR_INT;
    e->value.u.i = v;
}

void ckc_attr_set_float(ckc_ir_builder_t *b, ckc_attr_map_t *m, const char *key,
                        double v) {
    ckc_attr_entry_t *e = ckc_attr_reserve(b, m, key);
    if (!e)
        return;
    e->value.kind = CKC_ATTR_FLOAT;
    e->value.u.f = v;
}

void ckc_attr_set_str(ckc_ir_builder_t *b, ckc_attr_map_t *m, const char *key,
                      const char *v) {
    ckc_attr_entry_t *e = ckc_attr_reserve(b, m, key);
    if (!e)
        return;
    e->value.kind = CKC_ATTR_STR;
    if (v) {
        e->value.u.s = ckc_arena_strdup(&b->arena, v);
        if (!e->value.u.s) {
            ckc_i_set_err(b, CKC_ERR_OOM, "attr: OOM");
            return;
        }
    } else {
        e->value.u.s = NULL;
    }
}

void ckc_attr_set_bool(ckc_ir_builder_t *b, ckc_attr_map_t *m, const char *key,
                       bool v) {
    ckc_attr_entry_t *e = ckc_attr_reserve(b, m, key);
    if (!e)
        return;
    e->value.kind = CKC_ATTR_BOOL;
    e->value.u.b = v;
}

const ckc_attr_value_t *ckc_attr_get(const ckc_attr_map_t *m, const char *key) {
    ckc_attr_entry_t *e = ckc_attr_find((ckc_attr_map_t *)m, key);
    return e ? &e->value : NULL;
}

bool ckc_attr_get_int(const ckc_attr_map_t *m, const char *key, int64_t *out) {
    const ckc_attr_value_t *v = ckc_attr_get(m, key);
    if (!v || v->kind != CKC_ATTR_INT)
        return false;
    if (out)
        *out = v->u.i;
    return true;
}

bool ckc_attr_get_float(const ckc_attr_map_t *m, const char *key, double *out) {
    const ckc_attr_value_t *v = ckc_attr_get(m, key);
    if (!v)
        return false;
    /* Accept an int stored where a float is expected (Python's duck typing). */
    if (v->kind == CKC_ATTR_FLOAT) {
        if (out)
            *out = v->u.f;
        return true;
    }
    if (v->kind == CKC_ATTR_INT) {
        if (out)
            *out = (double)v->u.i;
        return true;
    }
    return false;
}

const char *ckc_attr_get_str(const ckc_attr_map_t *m, const char *key) {
    const ckc_attr_value_t *v = ckc_attr_get(m, key);
    if (!v || v->kind != CKC_ATTR_STR)
        return NULL;
    return v->u.s;
}

bool ckc_attr_get_bool(const ckc_attr_map_t *m, const char *key, bool dflt) {
    const ckc_attr_value_t *v = ckc_attr_get(m, key);
    if (!v)
        return dflt;
    switch (v->kind) {
        case CKC_ATTR_BOOL:  return v->u.b;
        case CKC_ATTR_INT:   return v->u.i != 0;
        case CKC_ATTR_FLOAT: return v->u.f != 0.0;
        default:             return dflt;
    }
}

/* ============================ OPCODE TABLE ============================== */

/* Canonical dotted name per opcode, indexed by the enum value. Order MUST match
 * the ckc_opcode enum in ckc/ir.h. CKC_OP_INVALID maps to "". */
static const char *const ckc_opcode_names[CKC_OP__COUNT] = {
    /* CKC_OP_INVALID */ "",

    /* arith.* */
    "arith.constant",
    "arith.constant_vec",
    "arith.add",
    "arith.sub",
    "arith.mul",
    "arith.div",
    "arith.mod",
    "arith.fadd",
    "arith.fsub",
    "arith.fmul",
    "arith.fdiv",
    "arith.fneg",
    "arith.fabs",
    "arith.fma",
    "arith.fmax3",
    "arith.fmin3",
    "arith.cmp",
    "arith.fcmp",
    "arith.fmax",
    "arith.fmin",
    "arith.and",
    "arith.or",
    "arith.not",
    "arith.smax",
    "arith.smin",
    "arith.xor",
    "arith.shl",
    "arith.lshr",
    "arith.umul_hi_i32",
    "arith.zext",
    "arith.sext",
    "arith.trunc",
    "arith.select",
    "arith.bitcast",
    "arith.trunc_f32_to_f16",
    "arith.rint_f32",
    "arith.cast_to_f32",
    "arith.cast_f32_to",
    "arith.sitofp_f32",
    "arith.cvt_fp8_to_f32",
    "arith.cvt_bf8_to_f32",
    "arith.cvt_pk_f32_fp8x4",
    "arith.cvt_pk_f32_bf8x4",
    "arith.cvt_pk_fp8_f32x4",
    "arith.cvt_pk_bf8_f32x4",
    "arith.cvt_pk_i8_f32x4",
    "arith.cvt_f32_to_fp8",
    "arith.cvt_f32_to_bf8",
    "arith.cvt_f32_to_i8_sat",
    "arith.cvt_scalef32_pk_f32_fp8",
    "arith.cvt_scalef32_pk_f32_bf8",
    "arith.cvt_scalef32_pk_fp8_f32",
    "arith.cvt_scalef32_pk_bf8_f32",

    /* math.* */
    "math.exp2",
    "math.log2",
    "math.rcp",
    "math.rcp_fast",
    "math.sqrt",
    "math.rsqrt",
    "math.tanh",

    /* gpu.* */
    "gpu.thread_id",
    "gpu.block_id",

    /* memref.* */
    "memref.global_load",
    "memref.global_load_typed",
    "memref.global_load_vN",
    "memref.global_store",
    "memref.global_store_typed",
    "memref.global_store_vN",
    "memref.global_atomic_add",
    "memref.global_atomic_add_f32",
    "memref.global_atomic_add_pk_bf16",
    "memref.cooperative_global_store",

    /* vector.* */
    "vector.add",
    "vector.sub",
    "vector.mul",
    "vector.and",
    "vector.or",
    "vector.shl",
    "vector.lshr",
    "vector.smax",
    "vector.smin",
    "vector.max",
    "vector.fma",
    "vector.sum",
    "vector.reduce_max",
    "vector.splat",
    "vector.select",
    "vector.cmp",
    "vector.trunc",
    "vector.sext",
    "vector.trunc_f32_to_f16",
    "vector.trunc_f32_to",
    "vector.bitcast",
    "vector.extract",
    "vector.insert",
    "vector.pack",
    "vector.concat",

    /* tile.* memory / lds */
    "tile.smem_alloc",
    "tile.smem_store",
    "tile.smem_store_vN",
    "tile.smem_store_vN_f32",
    "tile.smem_store_distributed",
    "tile.smem_load_v4",
    "tile.smem_load_vN",
    "tile.smem_load_vN_f32",
    "tile.smem_addr_of",
    "tile.smem_ptr_add",
    "tile.lds_atomic_add",
    "tile.global_ptr_add",
    "tile.global_load_lds",
    "tile.async_buffer_load_lds",
    "tile.async_buffer_load_lds_addr",
    "tile.buffer_rsrc",
    "tile.buffer_load_f16",
    "tile.buffer_load_vN_f16",
    "tile.buffer_store_f16",
    "tile.buffer_store_vN_f16",

    /* tile.* mma */
    "tile.mma",
    "tile.register_p_from_qk_c",

    /* tile.* inline asm */
    "tile.inline_asm",

    /* tile.* cross-lane / dpp / permute */
    "tile.readfirstlane",
    "tile.pin_sgpr",
    "tile.lane_id",
    "tile.wave_all",
    "tile.wave_any",
    "tile.wave_ballot",
    "tile.ds_bpermute",
    "tile.ds_bpermute_b64",
    "tile.ds_swizzle_xor",
    "tile.mov_dpp",
    "tile.permlane32_swap",
    "tile.perm_b32",
    "tile.permlanex16",
    "tile.byte_perm",
    "tile.ds_read_tr16_b64",
    "tile.ds_read_tr16_b128",
    "tile.ds_read_tr_b8",

    /* tile.* barriers / scheduling */
    "tile.sync",
    "tile.sync_half_block",
    "tile.sync_lds_only",
    "tile.s_barrier_bare",
    "tile.s_waitcnt",
    "tile.s_setprio",
    "tile.iglp_opt",
    "tile.sched_barrier",
    "tile.sched_group_barrier",

    /* scf.* / cf.* */
    "scf.for",
    "scf.if",
    "scf.yield",
    "cf.return"
};

const char *ckc_opcode_name(ckc_opcode_t op) {
    if (op < 0 || op >= CKC_OP__COUNT)
        return "";
    return ckc_opcode_names[op];
}

ckc_opcode_t ckc_opcode_from_name(const char *name) {
    int i;
    if (!name)
        return CKC_OP_INVALID;
    for (i = 1; i < CKC_OP__COUNT; ++i) {
        if (ckc_opcode_names[i] && strcmp(ckc_opcode_names[i], name) == 0)
            return (ckc_opcode_t)i;
    }
    return CKC_OP_INVALID;
}

/* Per-opcode purity, derived from Python PURE_OP_NAMES / is_pure_op_name. An
 * opcode is pure iff its dotted name appears in that set. Indexed by enum. */
static const bool ckc_opcode_pure[CKC_OP__COUNT] = {
    /* CKC_OP_INVALID */ false,

    /* arith.* */
    /* arith.constant            */ true,
    /* arith.constant_vec        */ true,
    /* arith.add                 */ true,
    /* arith.sub                 */ true,
    /* arith.mul                 */ true,
    /* arith.div                 */ true,
    /* arith.mod                 */ true,
    /* arith.fadd                */ true,
    /* arith.fsub                */ true,
    /* arith.fmul                */ true,
    /* arith.fdiv                */ true,
    /* arith.fneg                */ true,
    /* arith.fabs                */ false,
    /* arith.fma                 */ false,
    /* arith.fmax3               */ false,
    /* arith.fmin3               */ false,
    /* arith.cmp                 */ true,
    /* arith.fcmp                */ true,
    /* arith.fmax                */ true,
    /* arith.fmin                */ true,
    /* arith.and                 */ true,
    /* arith.or                  */ true,
    /* arith.not                 */ true,
    /* arith.smax                */ true,
    /* arith.smin                */ true,
    /* arith.xor                 */ true,
    /* arith.shl                 */ true,
    /* arith.lshr                */ true,
    /* arith.umul_hi_i32         */ true,
    /* arith.zext                */ true,
    /* arith.sext                */ true,
    /* arith.trunc               */ true,
    /* arith.select              */ true,
    /* arith.bitcast             */ true,
    /* arith.trunc_f32_to_f16    */ true,
    /* arith.rint_f32            */ true,
    /* arith.cast_to_f32         */ true,
    /* arith.cast_f32_to         */ true,
    /* arith.sitofp_f32          */ true,
    /* arith.cvt_fp8_to_f32      */ true,
    /* arith.cvt_bf8_to_f32      */ true,
    /* arith.cvt_pk_f32_fp8x4    */ true,
    /* arith.cvt_pk_f32_bf8x4    */ true,
    /* arith.cvt_pk_fp8_f32x4    */ true,
    /* arith.cvt_pk_bf8_f32x4    */ true,
    /* arith.cvt_pk_i8_f32x4     */ false,
    /* arith.cvt_f32_to_fp8      */ true,
    /* arith.cvt_f32_to_bf8      */ true,
    /* arith.cvt_f32_to_i8_sat   */ true,
    /* arith.cvt_scalef32_pk_f32_fp8 */ true,
    /* arith.cvt_scalef32_pk_f32_bf8 */ true,
    /* arith.cvt_scalef32_pk_fp8_f32 */ true,
    /* arith.cvt_scalef32_pk_bf8_f32 */ true,

    /* math.* */
    /* math.exp2     */ true,
    /* math.log2     */ true,
    /* math.rcp      */ true,
    /* math.rcp_fast */ true,
    /* math.sqrt     */ true,
    /* math.rsqrt    */ true,
    /* math.tanh     */ true,

    /* gpu.* */
    /* gpu.thread_id */ true,
    /* gpu.block_id  */ true,

    /* memref.* (all effectful) */
    /* global_load                  */ false,
    /* global_load_typed            */ false,
    /* global_load_vN               */ false,
    /* global_store                 */ false,
    /* global_store_typed           */ false,
    /* global_store_vN              */ false,
    /* global_atomic_add            */ false,
    /* global_atomic_add_f32        */ false,
    /* global_atomic_add_pk_bf16    */ false,
    /* cooperative_global_store     */ false,

    /* vector.* */
    /* vector.add        */ true,
    /* vector.sub        */ false,
    /* vector.mul        */ true,
    /* vector.and        */ true,
    /* vector.or         */ true,
    /* vector.shl        */ true,
    /* vector.lshr       */ true,
    /* vector.smax       */ true,
    /* vector.smin       */ true,
    /* vector.max        */ true,
    /* vector.fma        */ false,
    /* vector.sum        */ true,
    /* vector.reduce_max */ true,
    /* vector.splat      */ true,
    /* vector.select     */ true,
    /* vector.cmp        */ true,
    /* vector.trunc      */ true,
    /* vector.sext       */ true,
    /* vector.trunc_f32_to_f16 */ true,
    /* vector.trunc_f32_to     */ true,
    /* vector.bitcast    */ true,
    /* vector.extract    */ true,
    /* vector.insert     */ true,
    /* vector.pack       */ true,
    /* vector.concat     */ true,

    /* tile.* memory / lds (effectful) */
    /* smem_alloc                 */ false,
    /* smem_store                 */ false,
    /* smem_store_vN              */ false,
    /* smem_store_vN_f32          */ false,
    /* smem_store_distributed     */ false,
    /* smem_load_v4               */ false,
    /* smem_load_vN               */ false,
    /* smem_load_vN_f32           */ false,
    /* smem_addr_of               */ true,
    /* smem_ptr_add               */ true,
    /* lds_atomic_add             */ false,
    /* global_ptr_add             */ false,
    /* global_load_lds            */ false,
    /* async_buffer_load_lds      */ false,
    /* async_buffer_load_lds_addr */ false,
    /* buffer_rsrc                */ false,
    /* buffer_load_f16            */ false,
    /* buffer_load_vN_f16         */ false,
    /* buffer_store_f16           */ false,
    /* buffer_store_vN_f16        */ false,

    /* tile.* mma */
    /* tile.mma                 */ false,
    /* tile.register_p_from_qk_c */ false,

    /* tile.* inline asm */
    /* tile.inline_asm */ false,

    /* tile.* cross-lane / dpp / permute */
    /* readfirstlane     */ true,
    /* pin_sgpr          */ true,
    /* lane_id           */ true,
    /* wave_all          */ true,
    /* wave_any          */ true,
    /* wave_ballot       */ true,
    /* ds_bpermute       */ true,
    /* ds_bpermute_b64   */ false,
    /* ds_swizzle_xor    */ true,
    /* mov_dpp           */ false,
    /* permlane32_swap   */ true,
    /* perm_b32          */ true,
    /* permlanex16       */ true,
    /* byte_perm         */ true,
    /* ds_read_tr16_b64  */ true,
    /* ds_read_tr16_b128 */ false,
    /* ds_read_tr_b8     */ true,

    /* tile.* barriers / scheduling (effectful) */
    /* sync                 */ false,
    /* sync_half_block      */ true,
    /* sync_lds_only        */ false,
    /* s_barrier_bare       */ false,
    /* s_waitcnt            */ false,
    /* s_setprio            */ false,
    /* iglp_opt             */ false,
    /* sched_barrier        */ false,
    /* sched_group_barrier  */ false,

    /* scf.* / cf.* (control flow, effectful) */
    /* scf.for    */ false,
    /* scf.if     */ false,
    /* scf.yield  */ false,
    /* cf.return  */ false
};

bool ckc_opcode_is_pure(ckc_opcode_t op) {
    if (op <= CKC_OP_INVALID || op >= CKC_OP__COUNT)
        return false;
    return ckc_opcode_pure[op];
}

/* ============================== OP GETTERS ============================== */

/* Python @property Op.result: exactly one result required, else ValueError. */
ckc_value_t *ckc_op_result(ckc_ir_builder_t *b, ckc_op_t *op) {
    if (!ckc_i_live(b))
        return NULL;
    if (!op)
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE,
                                            "op.result: NULL op");
    if (op->num_results != 1)
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "op %s has %d results, not 1",
            op->name ? op->name : ckc_opcode_name(op->opcode),
            op->num_results);
    return op->results[0];
}

/* Python @property Op.is_pure: attrs["pure"] override wins, else opcode purity.
 */
bool ckc_op_is_pure(const ckc_op_t *op) {
    const ckc_attr_value_t *v;
    if (!op)
        return false;
    v = ckc_attr_get(&op->attrs, "pure");
    if (v) {
        switch (v->kind) {
            case CKC_ATTR_BOOL:  return v->u.b;
            case CKC_ATTR_INT:   return v->u.i != 0;
            case CKC_ATTR_FLOAT: return v->u.f != 0.0;
            case CKC_ATTR_STR:   return v->u.s != NULL && v->u.s[0] != '\0';
            default:             return false;
        }
    }
    return ckc_opcode_is_pure(op->opcode);
}

/* Python @property KernelDef.max_workgroup_size: attrs["max_workgroup_size"],
 * default 256. */
int ckc_kernel_max_workgroup_size(const ckc_kernel_def_t *k) {
    int64_t v;
    if (!k)
        return 256;
    if (ckc_attr_get_int(&k->attrs, "max_workgroup_size", &v))
        return (int)v;
    return 256;
}
