// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * verify.c -- IR verifier, faithful C99 port of rocke/core/verify.py.
 *
 * Walks a rocke_kernel_def_t and accumulates rocke_diag_t diagnostics. Helper map:
 *
 *   Python                       C99 (this file)
 *   --------------------------   -----------------------------------------------
 *   _Verifier.err/warn           v_err() / v_warn()
 *   _check_type                  check_type()
 *   _check_region                check_region()
 *   _check_op                    check_op()
 *   _check_contract              check_contract()
 *   _check_scf_for / _check_scf_if   check_scf_for() / check_scf_if()
 *   verify(kernel)               rocke_verify()
 *
 * Scoping: Python uses ``dict(scope)`` copies so inner-region defs do not leak
 * to siblings. Here a single growable name->Value table is used; on entering an
 * inner scope we snapshot ``count`` and truncate back to it on exit (defs are
 * append-only within a scope), reproducing the copy semantics without per-op
 * allocation.
 */

#include "rocke/verify.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"

/* ----------------------------------------------------- diagnostic buffer */

typedef struct diag_buf
{
    rocke_diag_t* items;
    size_t count;
    size_t cap;
    int oom;
} diag_buf_t;

static char* vstrdup(const char* s)
{
    if(!s)
    {
        return NULL;
    }
    size_t n = strlen(s);
    char* d = (char*)malloc(n + 1);
    if(d)
    {
        memcpy(d, s, n + 1);
    }
    return d;
}

static void
    diag_push(diag_buf_t* db, rocke_diag_severity_t sev, const rocke_op_t* op, const char* msg)
{
    if(db->count >= db->cap)
    {
        size_t nc = db->cap ? db->cap * 2 : 8;
        rocke_diag_t* ni = (rocke_diag_t*)realloc(db->items, nc * sizeof(rocke_diag_t));
        if(!ni)
        {
            db->oom = 1;
            return;
        }
        db->items = ni;
        db->cap = nc;
    }
    rocke_diag_t* d = &db->items[db->count];
    d->severity = sev;
    d->message = vstrdup(msg);
    d->op = op ? vstrdup(op->name ? op->name : "") : NULL;
    d->loc = (op && op->loc) ? vstrdup(op->loc) : NULL;
    db->count++;
}

/* ----------------------------------------------------- verifier state */

typedef struct vscope_entry
{
    const char* name;
    const rocke_type_t* type;
} vscope_entry_t;

typedef struct verifier
{
    diag_buf_t db;
    vscope_entry_t* scope; /* growable name->type table */
    int scope_count;
    int scope_cap;
} verifier_t;

static void v_errf(verifier_t* v, const rocke_op_t* op, const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    diag_push(&v->db, ROCKE_DIAG_ERROR, op, buf);
}

static void v_warnf(verifier_t* v, const rocke_op_t* op, const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    diag_push(&v->db, ROCKE_DIAG_WARNING, op, buf);
}

static const rocke_type_t* scope_get(verifier_t* v, const char* name)
{
    for(int i = v->scope_count - 1; i >= 0; --i)
    {
        if(strcmp(v->scope[i].name, name) == 0)
        {
            return v->scope[i].type;
        }
    }
    return NULL;
}

static int scope_has(verifier_t* v, const char* name)
{
    return scope_get(v, name) != NULL;
}

static void scope_put(verifier_t* v, const char* name, const rocke_type_t* t)
{
    if(v->scope_count >= v->scope_cap)
    {
        int nc = v->scope_cap ? v->scope_cap * 2 : 32;
        vscope_entry_t* ne = (vscope_entry_t*)realloc(v->scope, (size_t)nc * sizeof(*ne));
        if(!ne)
        {
            v->db.oom = 1;
            return;
        }
        v->scope = ne;
        v->scope_cap = nc;
    }
    v->scope[v->scope_count].name = name;
    v->scope[v->scope_count].type = t;
    v->scope_count++;
}

/* legal pointer address spaces */
static int is_addr_space(const char* s)
{
    return s
           && (strcmp(s, "global") == 0 || strcmp(s, "constant") == 0 || strcmp(s, "shared") == 0
               || strcmp(s, "lds") == 0 || strcmp(s, "private") == 0);
}

static const char* tn(const rocke_type_t* t)
{
    return (t && t->name) ? t->name : "";
}

/* ---- type sanity ---- */

static void
    check_type(verifier_t* v, const rocke_type_t* t, const char* where, const rocke_op_t* op)
{
    if(!t)
    {
        return;
    }
    if(t->kind == ROCKE_TYPE_VECTOR)
    {
        if(t->count <= 0)
        {
            v_errf(v, op, "%s: vector width must be > 0, got %d", where, t->count);
        }
        check_type(v, t->elem, where, op);
    }
    else if(t->kind == ROCKE_TYPE_PTR)
    {
        if(!is_addr_space(t->space))
        {
            v_errf(v,
                   op,
                   "%s: unknown pointer address space '%s' (known: ['constant', "
                   "'global', 'lds', 'private', 'shared'])",
                   where,
                   t->space ? t->space : "");
        }
        check_type(v, t->pointee, where, op);
    }
    else if(t->kind == ROCKE_TYPE_SMEM)
    {
        if(t->rank <= 0)
        {
            v_errf(v, op, "%s: smem type must have a non-empty shape", where);
        }
        for(int i = 0; i < t->rank; ++i)
        {
            if(t->shape[i] <= 0)
            {
                /* Python prints the whole shape tuple; reconstruct "(d0, d1)". */
                char shp[128];
                size_t o = 0;
                o += (size_t)snprintf(shp + o, sizeof(shp) - o, "(");
                for(int j = 0; j < t->rank && o < sizeof(shp); ++j)
                {
                    o += (size_t)snprintf(
                        shp + o, sizeof(shp) - o, "%s%d", j ? ", " : "", t->shape[j]);
                }
                if(t->rank == 1 && o < sizeof(shp))
                {
                    o += (size_t)snprintf(shp + o, sizeof(shp) - o, ",");
                }
                snprintf(shp + o, sizeof(shp) - o, ")");
                v_errf(v, op, "%s: smem shape dims must be > 0, got %s", where, shp);
                break;
            }
        }
        check_type(v, t->elem, where, op);
    }
}

/* ---- per-opcode contract sets ---- */

static int is_binary_same_type(rocke_opcode_t o)
{
    switch(o)
    {
    case ROCKE_OP_ARITH_ADD:
    case ROCKE_OP_ARITH_SUB:
    case ROCKE_OP_ARITH_MUL:
    case ROCKE_OP_ARITH_DIV:
    case ROCKE_OP_ARITH_MOD:
    case ROCKE_OP_ARITH_FADD:
    case ROCKE_OP_ARITH_FSUB:
    case ROCKE_OP_ARITH_FMUL:
    case ROCKE_OP_ARITH_FDIV:
    case ROCKE_OP_ARITH_FMAX:
    case ROCKE_OP_ARITH_FMIN:
    case ROCKE_OP_ARITH_AND:
    case ROCKE_OP_ARITH_OR:
    case ROCKE_OP_ARITH_SMAX:
    case ROCKE_OP_ARITH_SMIN:
    case ROCKE_OP_ARITH_XOR:
    case ROCKE_OP_ARITH_SHL:
    case ROCKE_OP_ARITH_LSHR:
        return 1;
    default:
        return 0;
    }
}

static int is_unary_same_type(rocke_opcode_t o)
{
    switch(o)
    {
    case ROCKE_OP_ARITH_FNEG:
    case ROCKE_OP_ARITH_FABS:
    case ROCKE_OP_ARITH_NOT:
    case ROCKE_OP_MATH_EXP2:
    case ROCKE_OP_MATH_LOG2:
    case ROCKE_OP_MATH_RCP:
    case ROCKE_OP_MATH_RCP_FAST:
    case ROCKE_OP_MATH_SQRT:
    case ROCKE_OP_MATH_RSQRT:
    case ROCKE_OP_MATH_TANH:
        return 1;
    default:
        return 0;
    }
}

static int is_cmp(rocke_opcode_t o)
{
    return o == ROCKE_OP_ARITH_CMP || o == ROCKE_OP_ARITH_FCMP;
}

/* required attr keys per opcode -> returns NULL-terminated static list. */
static const char* const* required_attrs(rocke_opcode_t o, int* n)
{
    static const char* constant_keys[] = {"value", "ity"};
    static const char* pred_keys[] = {"pred"};
    static const char* mma_keys[] = {"op_id"};
    static const char* yield_keys[] = {"num"};
    static const char* asm_keys[] = {"template", "constraints"};
    switch(o)
    {
    case ROCKE_OP_ARITH_CONSTANT:
        *n = 2;
        return constant_keys;
    case ROCKE_OP_ARITH_CMP:
    case ROCKE_OP_ARITH_FCMP:
        *n = 1;
        return pred_keys;
    case ROCKE_OP_TILE_MMA:
        *n = 1;
        return mma_keys;
    case ROCKE_OP_SCF_YIELD:
        *n = 1;
        return yield_keys;
    case ROCKE_OP_TILE_INLINE_ASM:
        *n = 2;
        return asm_keys;
    default:
        *n = 0;
        return NULL;
    }
}

static void check_region(verifier_t* v, const rocke_region_t* region, int is_kernel_body);
static void check_op(verifier_t* v, const rocke_op_t* op);

/* ---- contracts ---- */

static void check_contract(verifier_t* v, const rocke_op_t* op)
{
    rocke_opcode_t o = op->opcode;
    const char* name = op->name;
    if(is_binary_same_type(o))
    {
        if(op->num_operands != 2 || op->num_results != 1)
        {
            v_errf(v,
                   op,
                   "%s: expected 2 operands / 1 result, got %d / %d",
                   name,
                   op->num_operands,
                   op->num_results);
            return;
        }
        const rocke_type_t* a = op->operands[0]->type;
        const rocke_type_t* b = op->operands[1]->type;
        const rocke_type_t* r = op->results[0]->type;
        if(!rocke_type_eq(a, b))
        {
            v_errf(v, op, "%s: operand types differ (%s vs %s)", name, tn(a), tn(b));
        }
        if(!rocke_type_eq(r, a))
        {
            v_errf(v, op, "%s: result type %s != operand type %s", name, tn(r), tn(a));
        }
    }
    else if(is_unary_same_type(o))
    {
        if(op->num_operands != 1 || op->num_results != 1)
        {
            v_errf(v,
                   op,
                   "%s: expected 1 operand / 1 result, got %d / %d",
                   name,
                   op->num_operands,
                   op->num_results);
            return;
        }
        if(!rocke_type_eq(op->operands[0]->type, op->results[0]->type))
        {
            v_errf(v,
                   op,
                   "%s: result type %s != operand type %s",
                   name,
                   tn(op->results[0]->type),
                   tn(op->operands[0]->type));
        }
    }
    else if(is_cmp(o))
    {
        if(op->num_operands != 2 || op->num_results != 1)
        {
            v_errf(v, op, "%s: expected 2 operands / 1 result", name);
            return;
        }
        const rocke_type_t* a = op->operands[0]->type;
        const rocke_type_t* b = op->operands[1]->type;
        if(!rocke_type_eq(a, b))
        {
            v_errf(v, op, "%s: comparison operand types differ (%s vs %s)", name, tn(a), tn(b));
        }
        if(strcmp(tn(op->results[0]->type), "i1") != 0)
        {
            v_errf(v, op, "%s: result must be i1, got %s", name, tn(op->results[0]->type));
        }
    }
    else if(o == ROCKE_OP_ARITH_SELECT)
    {
        if(op->num_operands != 3 || op->num_results != 1)
        {
            v_errf(v, op, "%s: expected 3 operands / 1 result", name);
            return;
        }
        const rocke_type_t* cond = op->operands[0]->type;
        const rocke_type_t* lhs = op->operands[1]->type;
        const rocke_type_t* rhs = op->operands[2]->type;
        if(strcmp(tn(cond), "i1") != 0)
        {
            v_errf(v, op, "%s: condition must be i1, got %s", name, tn(cond));
        }
        if(!rocke_type_eq(lhs, rhs))
        {
            v_errf(v, op, "%s: branch types differ (%s vs %s)", name, tn(lhs), tn(rhs));
        }
    }
    else if(o == ROCKE_OP_VECTOR_EXTRACT)
    {
        if(op->num_operands != 1 || op->num_results != 1)
        {
            v_errf(v, op, "%s: expected 1 operand / 1 result", name);
            return;
        }
        const rocke_type_t* vt = op->operands[0]->type;
        if(!vt || vt->kind != ROCKE_TYPE_VECTOR)
        {
            v_errf(v, op, "%s: operand must be a vector, got %s", name, tn(vt));
        }
        else
        {
            if(!rocke_type_eq(op->results[0]->type, vt->elem))
            {
                v_errf(v,
                       op,
                       "%s: result type %s != vector element %s",
                       name,
                       tn(op->results[0]->type),
                       tn(vt->elem));
            }
            int64_t idx;
            if(rocke_attr_get_int(&op->attrs, "index", &idx))
            {
                if(!(idx >= 0 && idx < vt->count))
                {
                    v_errf(v,
                           op,
                           "%s: extract index %lld out of range [0,%d)",
                           name,
                           (long long)idx,
                           vt->count);
                }
            }
        }
    }
    else if(o == ROCKE_OP_TILE_MMA)
    {
        if(op->num_results != 1)
        {
            v_errf(v, op, "%s: expected exactly 1 result", name);
        }
        if(op->num_operands < 3)
        {
            v_errf(v,
                   op,
                   "%s: expected at least 3 operands (a, b, c), got %d",
                   name,
                   op->num_operands);
        }
    }
    else if(o == ROCKE_OP_CF_RETURN)
    {
        if(op->num_operands || op->num_results)
        {
            v_errf(v, op, "%s: must have no operands and no results", name);
        }
    }
}

/* ---- scf.for ---- */

static void check_scf_for(verifier_t* v, const rocke_op_t* op)
{
    if(op->num_regions <= 0)
    {
        v_errf(v, op, "scf.for missing its body region");
        return;
    }
    if(!rocke_attr_get(&op->attrs, "iv") || !rocke_attr_get(&op->attrs, "iv_type"))
    {
        v_errf(v, op, "scf.for missing 'iv' / 'iv_type' attrs");
    }
    if(op->num_operands < 3)
    {
        v_errf(v, op, "scf.for needs at least lower/upper/step operands");
        return;
    }
    const rocke_type_t* lower = op->operands[0]->type;
    for(int i = 0; i < 3; ++i)
    {
        if(!rocke_type_eq(op->operands[i]->type, lower))
        {
            v_errf(v, op, "scf.for lower/upper/step types must match");
            break;
        }
    }
    int num_iter_inits = op->num_operands - 3;

    const rocke_attr_value_t* iter_meta = rocke_attr_get(&op->attrs, "iter_args");
    int iter_meta_count = 0;
    int have_iter_meta = 0;
    if(iter_meta && iter_meta->kind == ROCKE_ATTR_LIST)
    {
        have_iter_meta = 1;
        iter_meta_count = iter_meta->u.list.count < 0 ? 0 : iter_meta->u.list.count;
    }
    else if(!iter_meta)
    {
        /* attrs.get("iter_args", []) -> [] ; treat as empty list */
        have_iter_meta = 1;
        iter_meta_count = 0;
    }

    if(have_iter_meta)
    {
        if(iter_meta_count != num_iter_inits)
        {
            v_errf(v,
                   op,
                   "scf.for: %d iter_args declared but %d init operands",
                   iter_meta_count,
                   num_iter_inits);
        }
        if(op->num_results != num_iter_inits)
        {
            v_errf(v, op, "scf.for: %d results but %d iter-args", op->num_results, num_iter_inits);
        }
        int lim = op->num_results < num_iter_inits ? op->num_results : num_iter_inits;
        for(int i = 0; i < lim; ++i)
        {
            const rocke_type_t* res = op->results[i]->type;
            const rocke_type_t* init = op->operands[3 + i]->type;
            if(!rocke_type_eq(res, init))
            {
                v_errf(v, op, "scf.for: result type %s != iter init type %s", tn(res), tn(init));
            }
        }
    }

    /* body scope: snapshot + iv + iter-arg block values */
    int saved = v->scope_count;
    const char* iv = rocke_attr_get_str(&op->attrs, "iv");
    if(iv)
    {
        /* iv type comes from the operands' lower type per Python only for the
         * yield check; for scope we register iv with its declared iv_type. The
         * Python body scope uses Type(iv_type); type identity does not feed
         * further checks beyond presence, so use lower's type as a stand-in is
         * NOT done -- register with lower (operands share type). Use iv_type if
         * resolvable to a scalar singleton, else lower. */
        const char* ivt = rocke_attr_get_str(&op->attrs, "iv_type");
        const rocke_type_t* t = ivt ? rocke_scalar_by_name(ivt) : NULL;
        if(!t)
        {
            t = lower; /* compound / unknown: use the loop bound type */
        }
        scope_put(v, iv, t);
    }
    if(have_iter_meta && iter_meta_count > 0)
    {
        int lim = iter_meta_count < num_iter_inits ? iter_meta_count : num_iter_inits;
        for(int i = 0; i < lim; ++i)
        {
            const char* nm = rocke_attr_get_str(iter_meta->u.list.items[i], "name");
            if(nm)
            {
                scope_put(v, nm, op->operands[3 + i]->type);
            }
        }
    }

    const rocke_region_t* body = op->regions[0];
    check_region(v, body, 0);

    /* body should end in scf.yield matching iter-arg types (if any iter-args) */
    if(num_iter_inits > 0)
    {
        if(body->num_ops == 0 || body->ops[body->num_ops - 1]->opcode != ROCKE_OP_SCF_YIELD)
        {
            v_errf(v, op, "scf.for with iter-args: body must end in scf.yield");
        }
        else
        {
            const rocke_op_t* yld = body->ops[body->num_ops - 1];
            if(yld->num_operands != num_iter_inits)
            {
                v_errf(v,
                       yld,
                       "scf.yield yields %d values but loop carries %d iter-args",
                       yld->num_operands,
                       num_iter_inits);
            }
            else
            {
                for(int i = 0; i < num_iter_inits; ++i)
                {
                    if(!rocke_type_eq(yld->operands[i]->type, op->operands[3 + i]->type))
                    {
                        v_errf(v,
                               yld,
                               "scf.yield type %s != iter-arg type %s",
                               tn(yld->operands[i]->type),
                               tn(op->operands[3 + i]->type));
                    }
                }
            }
        }
    }

    v->scope_count = saved; /* drop body scope */
}

/* ---- scf.if ---- */

static void check_scf_if(verifier_t* v, const rocke_op_t* op)
{
    if(op->num_operands != 1)
    {
        v_errf(v, op, "scf.if expects exactly 1 condition operand");
    }
    else if(strcmp(tn(op->operands[0]->type), "i1") != 0)
    {
        v_errf(v, op, "scf.if condition must be i1, got %s", tn(op->operands[0]->type));
    }
    if(op->num_regions <= 0)
    {
        v_errf(v, op, "scf.if missing its 'then' region");
    }
    for(int i = 0; i < op->num_regions; ++i)
    {
        int saved = v->scope_count;
        check_region(v, op->regions[i], 0);
        v->scope_count = saved;
    }
}

/* ---- op ---- */

static void check_op(verifier_t* v, const rocke_op_t* op)
{
    /* 1) operands must dominate (defined + visible). */
    for(int i = 0; i < op->num_operands; ++i)
    {
        const rocke_value_t* o = op->operands[i];
        const rocke_type_t* def = scope_get(v, o->name);
        if(!def)
        {
            v_errf(v, op, "operand '%s' used before definition / out of scope", o->name);
        }
        else if(!rocke_type_eq(def, o->type))
        {
            v_errf(v,
                   op,
                   "operand '%s' type %s does not match its definition type %s",
                   o->name,
                   tn(o->type),
                   tn(def));
        }
    }

    /* 2) required attrs present. */
    int nreq = 0;
    const char* const* req = required_attrs(op->opcode, &nreq);
    for(int i = 0; i < nreq; ++i)
    {
        if(!rocke_attr_get(&op->attrs, req[i]))
        {
            v_errf(v, op, "op '%s' missing required attr '%s'", op->name, req[i]);
        }
    }

    /* 3) per-opcode arity / type contracts. */
    check_contract(v, op);

    /* 4) result types sane; results define new (unique) ids. */
    for(int i = 0; i < op->num_results; ++i)
    {
        const rocke_value_t* r = op->results[i];
        char where[160];
        snprintf(where, sizeof(where), "result '%s'", r->name);
        check_type(v, r->type, where, op);
        if(scope_has(v, r->name))
        {
            v_errf(v, op, "SSA id '%s' redefined", r->name);
        }
        else
        {
            scope_put(v, r->name, r->type);
        }
    }

    /* 5) nested regions, with block-defined values registered first. */
    if(op->num_regions > 0)
    {
        if(op->opcode == ROCKE_OP_SCF_FOR)
        {
            check_scf_for(v, op);
        }
        else if(op->opcode == ROCKE_OP_SCF_IF)
        {
            check_scf_if(v, op);
        }
        else
        {
            for(int i = 0; i < op->num_regions; ++i)
            {
                int saved = v->scope_count;
                check_region(v, op->regions[i], 0);
                v->scope_count = saved;
            }
        }
    }
}

static void check_region(verifier_t* v, const rocke_region_t* region, int is_kernel_body)
{
    if(!region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops; ++i)
    {
        check_op(v, region->ops[i]);
    }
    if(is_kernel_body && region->num_ops == 0)
    {
        v_warnf(v, NULL, "kernel body region is empty");
    }
}

/* ---- entry ---- */

rocke_status_t rocke_verify(const rocke_kernel_def_t* k, rocke_diag_t** out, size_t* n)
{
    if(out)
    {
        *out = NULL;
    }
    if(n)
    {
        *n = 0;
    }
    if(!k || !out || !n)
    {
        return ROCKE_ERR_VALUE;
    }

    verifier_t v;
    memset(&v, 0, sizeof(v));

    /* top-level scope: kernel params */
    for(int i = 0; i < k->num_params; ++i)
    {
        const rocke_param_t* p = k->params[i];
        char where[160];
        snprintf(where, sizeof(where), "param '%s'", p->name ? p->name : "");
        check_type(&v, p->type, where, NULL);
        char vname[160];
        snprintf(vname, sizeof(vname), "%%%s", p->name ? p->name : "");
        if(scope_has(&v, vname))
        {
            v_errf(&v, NULL, "duplicate kernel parameter id '%s'", vname);
        }
        /* param id strings must outlive the scope table -> they live in the
         * kernel arena; but vname here is a stack buffer. Stash a heap copy by
         * registering an arena-independent strdup. Use the param->name + '%'
         * stable storage instead: the Param.name pointer is stable, but we need
         * the leading '%'. Build a small persistent buffer. */
        char* stable = (char*)malloc(strlen(vname) + 1);
        if(!stable)
        {
            v.db.oom = 1;
        }
        else
        {
            strcpy(stable, vname);
            scope_put(&v, stable, p->type);
        }
    }

    check_region(&v, k->body, 1);

    /* The scope holds pointers; param ids were heap-strdup'd above (leak: the
     * scope table is freed below but the strings would leak). Free those before
     * releasing the table. Op result names are arena-owned (kernel), not freed
     * here. We distinguish param strdups by re-walking: they are exactly the
     * first k->num_params entries inserted. Simpler: track and free them. */
    /* Free heap-allocated param-id strings (the only malloc'd scope names). */
    /* (They were inserted first; but inner defs were truncated away already.) */
    /* To avoid double-free / dangling, we conservatively scan for names that
     * begin with '%' and equal "%"+param->name, freeing each once. */
    for(int i = 0; i < k->num_params; ++i)
    {
        char vname[160];
        snprintf(vname, sizeof(vname), "%%%s", k->params[i]->name ? k->params[i]->name : "");
        for(int j = 0; j < v.scope_count; ++j)
        {
            if(v.scope[j].name && strcmp(v.scope[j].name, vname) == 0)
            {
                free((void*)v.scope[j].name);
                v.scope[j].name = NULL;
                break;
            }
        }
    }
    free(v.scope);

    if(v.db.oom)
    {
        rocke_diags_free(v.db.items, v.db.count);
        return ROCKE_ERR_OOM;
    }
    *out = v.db.items;
    *n = v.db.count;
    return ROCKE_OK;
}

void rocke_diags_free(rocke_diag_t* diags, size_t n)
{
    if(!diags)
    {
        return;
    }
    for(size_t i = 0; i < n; ++i)
    {
        free(diags[i].message);
        free(diags[i].op);
        free(diags[i].loc);
    }
    free(diags);
}

char* rocke_diag_to_string(const rocke_diag_t* d)
{
    if(!d)
    {
        return NULL;
    }
    const char* sev = (d->severity == ROCKE_DIAG_ERROR) ? "error" : "warning";
    const char* msg = d->message ? d->message : "";
    size_t cap = strlen(sev) + strlen(msg) + 32;
    if(d->op)
    {
        cap += strlen(d->op) + 8;
    }
    if(d->loc)
    {
        cap += strlen(d->loc) + 8;
    }
    char* s = (char*)malloc(cap);
    if(!s)
    {
        return NULL;
    }
    size_t o = 0;
    o += (size_t)snprintf(s + o, cap - o, "%s: %s", sev, msg);
    if(d->op)
    {
        o += (size_t)snprintf(s + o, cap - o, " [%s]", d->op);
    }
    if(d->loc)
    {
        o += (size_t)snprintf(s + o, cap - o, " @%s", d->loc);
    }
    return s;
}
