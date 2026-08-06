/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * recipe_vm.c -- the "builder recipe" VM (schema "rocke.recipe/v1"). See
 * rocke/recipe_vm.h.
 *
 * The recipe is a small program with three environments:
 *   - spec inputs (ints/strings) supplied at JIT time;
 *   - VM integer registers (loop induction vars + spec-derived integers);
 *   - IR-value registers (rocke_value_t* produced by emit/const/param ops).
 *
 * Compile-time control flow (`static_for`) lets one recipe expand into a
 * shape-specialized kernel at runtime: e.g. a `static_for` whose bound is the
 * spec value `D` unrolls D times in C, exactly as the Python author's
 * Python-time unroll would have, but driven by the runtime spec.
 *
 * Instruction set (each is a JSON object with "op"):
 *   param        {name,type,bind?,attrs?}        -> rocke_b_param
 *   const_i32    {bind,val:<intexpr>}            -> rocke_b_const_i32
 *   const_f32    {bind,fval:<number>}            -> rocke_b_const_f32
 *   thread_id_x  {bind}                          -> rocke_b_thread_id_x
 *   emit         {opcode,in:[<reg>],out?:{bind,type}|outs?,attrs?} -> rocke_b_op
 *   alias        {bind,from}                     -> rebind register
 *   static_for   {var,lo,hi,step?,body:[instr]}  -> compile-time loop
 *   static_if    {pred:<intexpr>,then,else?}     -> compile-time branch
 *   scf_for      {iv,lo,hi,step,iter:[<iterarg>],results:[<reg>],body} -> runtime loop
 *   scf_if       {cond,then}                     -> runtime branch
 *   ret          {}                              -> rocke_b_ret
 *
 * Parametric (rolled) features:
 *   - <intexpr> in const values, attr values (t:i), AND type size fields
 *     (vector count, smem shape).
 *   - Format register names: any <reg> may contain {var}/{spec} tokens that are
 *     substituted with the current loop-index / spec value (e.g. "acc_m{lane}_n0").
 *   - Rolled lists: scf_for `iter`/`results` and emit `in` entries may be a
 *     rolled group {"for":{var,lo,hi,step}, "name":..., "init"?:...} that expands
 *     to a spec-derived NUMBER of entries (variable loop-carry fan / yield).
 *   - Types: "ptr", "vector", scalar, and "smem" {elem, shape:[<intexpr>...]}.
 *     A tile.smem_alloc result is named exactly per its recipe bind so the LDS
 *     global symbol (and thus the HSACO) matches the Python reference.
 *
 * <intexpr> := number | {"spec":NAME} | {"var":NAME} | {"spec_str_eq":[n,lit]}
 *            | {"<OP>":[e,e]}  for OP in add sub mul div mod eq ne lt le gt ge
 */
#include "rocke/recipe_vm.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/cbor_dom.h"
#include "rocke/json_dom.h"

/* ------------------------------------------------------------------ state */

typedef struct
{
    const char* name;
    rocke_value_t* val;
} rv_reg_t;

typedef struct
{
    const char* name;
    long value;
} rv_int_t;

typedef struct
{
    rocke_ir_builder_t* b;
    const rocke_recipe_spec_int_t* ints;
    int n_ints;
    const rocke_recipe_spec_str_t* strs;
    int n_strs;

    rv_reg_t* regs; /* IR-value registers */
    int n_regs, cap_regs;
    rv_int_t* ivars; /* VM integers (loop vars), a scope stack */
    int n_ivars, cap_ivars;

    char** owned; /* interned resolved register names (format-name substitution) */
    int n_owned, cap_owned;

    /* Concrete recipes (recorder-produced, no rolling: every bind is a unique
     * Python SSA name) opt into exact SSA naming: each created value is named
     * "%<bind>" verbatim, so the lowerer (which emits value names verbatim)
     * reproduces the Python .ll byte-for-byte -- not just an equivalent HSACO.
     * Disabled for rolled/parametric recipes, where binds repeat across unrolled
     * iterations and must stay fresh to avoid SSA name collisions. */
    bool exact_names;

    char err[ROCKE_ERR_MSG_CAP];
    bool failed;
} rvm_t;

/* Under exact_names, give `v` the SSA name "%<bind>" (arena-owned), mirroring the
 * portable-IR importer so the lowerer emits it verbatim. No-op otherwise. */
static void rv_name(rvm_t* vm, rocke_value_t* v, const char* bind)
{
    if(!vm->exact_names || !v || !bind)
        return;
    char* nm = rocke_arena_printf(&vm->b->arena, "%%%s", bind);
    if(nm)
        v->name = nm;
}

/* Resolve an opcode name, applying portable-IR aliases. The Python builder names
 * some vectorized fp16 buffer ops without the dtype suffix the engine's opcode
 * registry uses ("tile.buffer_load_vN" vs "tile.buffer_load_vN_f16"); the op is
 * otherwise identical, so normalize on the round-trip. Engine core unchanged. */
/* Portable-IR opcode alias.
 *
 * A few Python IRBuilder ops are dtype-GENERIC -- their spelling carries no
 * dtype and the element type rides in the `elem_type` attr (tile.buffer_load,
 * tile.buffer_load_vN, tile.buffer_store...). The engine's opcode registry
 * instead keys those by a dtype suffix (tile.buffer_load_f16 / _bf16). Resolve
 * the exact name first, then "<name>_<elem_type>", then fall back to the _f16
 * entry -- which is what cpp/helpers/loads.cpp emits natively for this path, so
 * the replayed graph matches a native C++ build. An unrepresentable dtype
 * resolves to nothing and the caller reports it, rather than silently lowering
 * as f16. */
static rocke_opcode_t rv_opcode_from_name(const char* name, const char* elem_type)
{
    if(!name)
        return ROCKE_OP_INVALID;
    rocke_opcode_t op = rocke_opcode_from_name(name);
    if(op != ROCKE_OP_INVALID)
        return op;
    char buf[128];
    if(elem_type && *elem_type)
    {
        snprintf(buf, sizeof buf, "%s_%s", name, elem_type);
        op = rocke_opcode_from_name(buf);
        if(op != ROCKE_OP_INVALID)
            return op;
    }
    snprintf(buf, sizeof buf, "%s_f16", name);
    return rocke_opcode_from_name(buf);
}

/* The `elem_type` string attr of a typed-attr recipe object, or NULL. */
static const char* rv_attr_elem_type(const jd_val_t* attrs)
{
    if(!attrs || attrs->kind != JD_OBJ)
        return NULL;
    const jd_val_t* tv = rocke_jget(attrs, "elem_type");
    return tv ? rocke_jstr(rocke_jget(tv, "v")) : NULL;
}

static void rv_fail(rvm_t* vm, const char* fmt, ...)
{
    if(vm->failed)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(vm->err, sizeof vm->err, fmt, ap);
    va_end(ap);
    vm->failed = true;
}

static bool rv_spec_int(rvm_t* vm, const char* name, long* out)
{
    if(!name || !out)
        return false;
    for(int i = 0; i < vm->n_ints; i++)
        if(strcmp(vm->ints[i].name, name) == 0)
        {
            *out = vm->ints[i].value;
            return true;
        }
    return false;
}

static const char* rv_spec_str(rvm_t* vm, const char* name)
{
    if(!name)
        return NULL;
    for(int i = 0; i < vm->n_strs; i++)
        if(strcmp(vm->strs[i].name, name) == 0)
            return vm->strs[i].value;
    return NULL;
}

static void rv_reg_set(rvm_t* vm, const char* name, rocke_value_t* val)
{
    if(!name || !*name || !val)
    {
        rv_fail(vm, "invalid register binding");
        return;
    }
    for(int i = 0; i < vm->n_regs; i++)
        if(strcmp(vm->regs[i].name, name) == 0)
        {
            vm->regs[i].val = val;
            return;
        }
    if(vm->n_regs == vm->cap_regs)
    {
        int nc = vm->cap_regs ? vm->cap_regs * 2 : 16;
        rv_reg_t* nr = (rv_reg_t*)realloc(vm->regs, (size_t)nc * sizeof(rv_reg_t));
        if(!nr)
        {
            rv_fail(vm, "oom regs");
            return;
        }
        vm->regs = nr;
        vm->cap_regs = nc;
    }
    vm->regs[vm->n_regs].name = name;
    vm->regs[vm->n_regs].val = val;
    vm->n_regs++;
}

static rocke_value_t* rv_reg_get(rvm_t* vm, const char* name)
{
    if(!name)
        return NULL;
    for(int i = vm->n_regs - 1; i >= 0; i--)
        if(strcmp(vm->regs[i].name, name) == 0)
            return vm->regs[i].val;
    return NULL;
}

static bool rv_ivar_get(rvm_t* vm, const char* name, long* out)
{
    if(!name || !out)
        return false;
    for(int i = vm->n_ivars - 1; i >= 0; i--)
        if(strcmp(vm->ivars[i].name, name) == 0)
        {
            *out = vm->ivars[i].value;
            return true;
        }
    return false;
}

static void rv_ivar_push(rvm_t* vm, const char* name, long value)
{
    if(!name || !*name)
    {
        rv_fail(vm, "invalid loop variable");
        return;
    }
    if(vm->n_ivars == vm->cap_ivars)
    {
        int nc = vm->cap_ivars ? vm->cap_ivars * 2 : 16;
        rv_int_t* ni = (rv_int_t*)realloc(vm->ivars, (size_t)nc * sizeof(rv_int_t));
        if(!ni)
        {
            rv_fail(vm, "oom ivars");
            return;
        }
        vm->ivars = ni;
        vm->cap_ivars = nc;
    }
    vm->ivars[vm->n_ivars].name = name;
    vm->ivars[vm->n_ivars].value = value;
    vm->n_ivars++;
}

/* ----------------------------------------------------------- intexpr eval */

static long rv_int(rvm_t* vm, const jd_val_t* e)
{
    if(vm->failed || !e)
        return 0;
    double n;
    if(rocke_jnum(e, &n))
        return (long)n;
    if(e->kind == JD_OBJ)
    {
        const jd_val_t* s = rocke_jget(e, "spec");
        if(s)
        {
            long v = 0; /* left 0 on the failure path; the VM is sticky-failed */
            if(!rv_spec_int(vm, rocke_jstr(s), &v))
                rv_fail(vm, "unknown spec int '%s'", rocke_jstr(s) ? rocke_jstr(s) : "?");
            return v;
        }
        const jd_val_t* var = rocke_jget(e, "var");
        if(var)
        {
            long v = 0; /* left 0 on the failure path; the VM is sticky-failed */
            if(!rv_ivar_get(vm, rocke_jstr(var), &v))
                rv_fail(vm, "unknown loop var '%s'", rocke_jstr(var) ? rocke_jstr(var) : "?");
            return v;
        }
        /* spec_str_eq: ["specname","literal"] -> 1 if the spec string matches. */
        const jd_val_t* sse = rocke_jget(e, "spec_str_eq");
        if(sse && sse->kind == JD_ARR && sse->arr_len == 2)
        {
            const char* sv = rv_spec_str(vm, rocke_jstr(sse->arr[0]));
            const char* lit = rocke_jstr(sse->arr[1]);
            return (sv && lit && strcmp(sv, lit) == 0) ? 1 : 0;
        }
        /* Binary arithmetic + comparisons: {"<op>":[e,e]}. */
        static const char* ops[]
            = {"add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge"};
        for(int k = 0; k < (int)(sizeof ops / sizeof ops[0]); k++)
        {
            const jd_val_t* bin = rocke_jget(e, ops[k]);
            if(!bin)
                continue;
            if(bin->kind != JD_ARR || bin->arr_len != 2)
            {
                rv_fail(vm, "bad intexpr '%s'", ops[k]);
                return 0;
            }
            long a = rv_int(vm, bin->arr[0]);
            long b = rv_int(vm, bin->arr[1]);
            switch(k)
            {
            case 0:
                return a + b;
            case 1:
                return a - b;
            case 2:
                return a * b;
            case 3:
                if(b == 0)
                {
                    rv_fail(vm, "integer division by zero");
                    return 0;
                }
                if(a == LONG_MIN && b == -1)
                {
                    rv_fail(vm, "integer division overflow");
                    return 0;
                }
                return a / b;
            case 4:
                if(b == 0)
                {
                    rv_fail(vm, "integer modulo by zero");
                    return 0;
                }
                if(a == LONG_MIN && b == -1)
                {
                    rv_fail(vm, "integer modulo overflow");
                    return 0;
                }
                return a % b;
            case 5:
                return a == b;
            case 6:
                return a != b;
            case 7:
                return a < b;
            case 8:
                return a <= b;
            case 9:
                return a > b;
            case 10:
                return a >= b;
            }
        }
    }
    rv_fail(vm, "bad intexpr");
    return 0;
}

/* ----------------------------------------------- format names + rolled lists */

/* Intern a string so it outlives the JSON DOM (resolved register names are
 * computed at runtime; the register table keys must stay stable). */
static const char* rv_intern(rvm_t* vm, const char* s)
{
    size_t len = strlen(s) + 1;
    char* dup = (char*)malloc(len);
    if(!dup)
    {
        rv_fail(vm, "oom intern");
        return ""; /* static; vm->failed is set so this is never used */
    }
    memcpy(dup, s, len);
    if(vm->n_owned == vm->cap_owned)
    {
        int nc = vm->cap_owned ? vm->cap_owned * 2 : 16;
        char** no = (char**)realloc(vm->owned, (size_t)nc * sizeof(char*));
        if(!no)
        {
            rv_fail(vm, "oom owned");
            free(dup);
            return "";
        }
        vm->owned = no;
        vm->cap_owned = nc;
    }
    vm->owned[vm->n_owned++] = dup;
    return dup;
}

static bool rv_append(rvm_t* vm,
                      char** out,
                      size_t* len,
                      size_t* cap,
                      const char* src,
                      size_t src_len,
                      const char* what)
{
    if(src_len > SIZE_MAX - *len - 1)
    {
        rv_fail(vm, "%s too long", what);
        return false;
    }
    const size_t need = *len + src_len + 1;
    if(need > *cap)
    {
        size_t nc = *cap;
        while(nc < need)
        {
            if(nc > SIZE_MAX / 2)
            {
                rv_fail(vm, "%s too long", what);
                return false;
            }
            nc *= 2;
        }
        char* grown = (char*)realloc(*out, nc);
        if(!grown)
        {
            rv_fail(vm, "oom %s", what);
            return false;
        }
        *out = grown;
        *cap = nc;
    }
    memcpy(*out + *len, src, src_len);
    *len += src_len;
    (*out)[*len] = '\0';
    return true;
}

/* Resolve a register name that may contain {var} loop-index / spec tokens
 * (e.g. "acc_m{lane}_n0" -> "acc_m2_n0"). Names without '{' pass through with no
 * allocation. */
static const char* rv_resolve_name(rvm_t* vm, const char* raw)
{
    if(!raw || !*raw)
    {
        rv_fail(vm, "missing register name");
        return raw;
    }
    if(!strchr(raw, '{') && !strchr(raw, '}'))
        return raw;
    char buf[256];
    size_t n = 0;
    buf[0] = '\0';
    for(const char* p = raw; *p;)
    {
        if(*p == '{')
        {
            const char* close = strchr(p, '}');
            if(!close)
            {
                rv_fail(vm, "unterminated register name placeholder");
                return raw;
            }
            size_t kl = (size_t)(close - p - 1);
            if(kl == 0 || kl >= 64)
            {
                rv_fail(vm, "invalid register name placeholder");
                return raw;
            }
            char key[64];
            memcpy(key, p + 1, kl);
            key[kl] = '\0';
            long v;
            if(rv_ivar_get(vm, key, &v) || rv_spec_int(vm, key, &v))
            {
                char tmp[32];
                int written = snprintf(tmp, sizeof tmp, "%ld", v);
                if(written < 0 || (size_t)written >= sizeof buf - n)
                {
                    rv_fail(vm, "resolved register name too long");
                    return raw;
                }
                memcpy(buf + n, tmp, (size_t)written);
                n += (size_t)written;
            }
            else
            {
                rv_fail(vm, "unresolved name var '%s'", key);
                return raw;
            }
            p = close + 1;
        }
        else if(*p == '}')
        {
            rv_fail(vm, "unmatched register name placeholder close");
            return raw;
        }
        else
        {
            if(n + 1 >= sizeof buf)
            {
                rv_fail(vm, "resolved register name too long");
                return raw;
            }
            buf[n++] = *p++;
        }
    }
    buf[n] = '\0';
    const char* resolved = rv_intern(vm, buf);
    return resolved;
}

/* A growable list of (interned) register names. */
typedef struct
{
    const char** a;
    int n, cap;
} rv_names_t;

static void rv_names_push(rvm_t* vm, rv_names_t* s, const char* v)
{
    if(s->n == s->cap)
    {
        int nc = s->cap ? s->cap * 2 : 8;
        const char** na = (const char**)realloc(s->a, (size_t)nc * sizeof(char*));
        if(!na)
        {
            rv_fail(vm, "oom namelist");
            return;
        }
        s->a = na;
        s->cap = nc;
    }
    s->a[s->n++] = v;
}

/* Expand a list whose entries are register-name strings OR rolled groups
 * {"for":{var,lo,hi,step}, "name":"r{var}", "init"?:...} into resolved names.
 * Used for emit operands, scf_for results (inits=NULL), and scf_for iter-args
 * (inits collects each entry's "init"). Caller frees names->a / inits->a. */
static void rv_expand_list(rvm_t* vm, const jd_val_t* arr, rv_names_t* names, rv_names_t* inits)
{
    if(!arr || arr->kind != JD_ARR)
    {
        rv_fail(vm, "register list must be an array");
        return;
    }
    for(int i = 0; i < arr->arr_len && !vm->failed; i++)
    {
        const jd_val_t* e = arr->arr[i];
        if(e->kind == JD_STR)
        {
            rv_names_push(vm, names, rv_resolve_name(vm, e->str));
            continue;
        }
        if(e->kind != JD_OBJ)
        {
            rv_fail(vm, "register-list entry must be a string or object");
            return;
        }
        const jd_val_t* fr = rocke_jget(e, "for");
        const char* nm = rocke_jstr(rocke_jget(e, "name"));
        const char* init = inits ? rocke_jstr(rocke_jget(e, "init")) : NULL;
        if(fr)
        {
            if(fr->kind != JD_OBJ)
            {
                rv_fail(vm, "rolled-list for must be an object");
                return;
            }
            const char* var = rocke_jstr(rocke_jget(fr, "var"));
            long lo = rv_int(vm, rocke_jget(fr, "lo"));
            long hi = rv_int(vm, rocke_jget(fr, "hi"));
            const jd_val_t* sn = rocke_jget(fr, "step");
            long step = sn ? rv_int(vm, sn) : 1;
            if(!var || !*var || !nm || !*nm || (inits && (!init || !*init)))
            {
                rv_fail(vm, "rolled-list entry needs var/name%s", inits ? "/init" : "");
                return;
            }
            if(step <= 0)
            {
                rv_fail(vm, "rolled-list step must be positive");
                return;
            }
            for(long iv = lo; iv < hi && !vm->failed; iv += step)
            {
                int mark = vm->n_ivars;
                rv_ivar_push(vm, var, iv);
                rv_names_push(vm, names, rv_resolve_name(vm, nm));
                if(inits)
                    rv_names_push(vm, inits, rv_resolve_name(vm, init));
                vm->n_ivars = mark;
                if(iv > LONG_MAX - step)
                {
                    rv_fail(vm, "rolled-list loop overflow");
                    return;
                }
            }
        }
        else
        {
            if(!nm || !*nm || (inits && (!init || !*init)))
            {
                rv_fail(vm, "register-list entry needs name%s", inits ? "/init" : "");
                return;
            }
            rv_names_push(vm, names, rv_resolve_name(vm, nm));
            if(inits)
                rv_names_push(vm, inits, rv_resolve_name(vm, init));
        }
    }
}

/* -------------------------------------------------------------- type parse */

static const rocke_type_t* rv_type(rvm_t* vm, const jd_val_t* t)
{
    if(!t)
    {
        rv_fail(vm, "missing type");
        return NULL;
    }
    if(t->kind == JD_STR)
    {
        const rocke_type_t* st = rocke_scalar_by_name(t->str);
        if(!st)
            rv_fail(vm, "unknown scalar '%s'", t->str);
        return st;
    }
    if(t->kind == JD_OBJ)
    {
        const char* kind = rocke_jstr(rocke_jget(t, "kind"));
        if(kind && strcmp(kind, "ptr") == 0)
        {
            const rocke_type_t* pointee = rv_type(vm, rocke_jget(t, "pointee"));
            const char* space = rocke_jstr(rocke_jget(t, "space"));
            if(!pointee || !space)
                return vm->failed ? nullptr : (rv_fail(vm, "bad ptr type"), nullptr);
            return rocke_ptr_type(vm->b, pointee, space);
        }
        if(kind && strcmp(kind, "vector") == 0)
        {
            const rocke_type_t* elem = rv_type(vm, rocke_jget(t, "elem"));
            const jd_val_t* cnt = rocke_jget(t, "count");
            if(!elem || !cnt)
                return vm->failed ? nullptr : (rv_fail(vm, "bad vector type"), nullptr);
            int n = (int)rv_int(vm, cnt); /* count may be an intexpr */
            return vm->failed ? NULL : rocke_vector_type(vm->b, elem, n);
        }
        if(kind && strcmp(kind, "smem") == 0)
        {
            const rocke_type_t* elem = rv_type(vm, rocke_jget(t, "elem"));
            const jd_val_t* shape = rocke_jget(t, "shape");
            if(!elem || !shape || shape->kind != JD_ARR)
                return vm->failed ? nullptr : (rv_fail(vm, "bad smem type"), nullptr);
            int rank = shape->arr_len;
            if(rank > 8)
            {
                rv_fail(vm, "smem rank > 8");
                return NULL;
            }
            int dims[8];
            for(int i = 0; i < rank; i++)
                dims[i] = (int)rv_int(vm, shape->arr[i]); /* dims may be intexprs */
            /* exclusive is reconstructed from the op attr in rv_op (the type
             * node, like the canonical type name, omits it). */
            return vm->failed ? NULL : rocke_smem_type(vm->b, elem, dims, rank, 0);
        }
        rv_fail(vm, "unsupported type kind '%s'", kind ? kind : "?");
        return NULL;
    }
    rv_fail(vm, "bad type node");
    return NULL;
}

/* ------------------------------------------------------------- attr build */

static void rv_attrs(rvm_t* vm, const jd_val_t* attrs, rocke_attr_map_t* m)
{
    rocke_attr_map_init(m);
    if(!attrs)
        return;
    if(attrs->kind != JD_OBJ)
    {
        rv_fail(vm, "attrs must be an object");
        return;
    }
    for(int i = 0; i < attrs->obj_len && !vm->failed; i++)
    {
        const char* key = attrs->obj[i].key;
        const jd_val_t* tv = attrs->obj[i].val;
        const char* t = rocke_jstr(rocke_jget(tv, "t"));
        const jd_val_t* v = rocke_jget(tv, "v");
        if(!t || !v)
        {
            rv_fail(vm, "attr '%s' missing t/v", key);
            return;
        }
        if(strcmp(t, "i") == 0)
            rocke_attr_set_int(vm->b, m, key, rv_int(vm, v)); /* v may be intexpr */
        else if(strcmp(t, "f") == 0)
        {
            double d = 0;
            if(!rocke_jnum(v, &d))
            {
                rv_fail(vm, "attr '%s' float value is not numeric", key);
                return;
            }
            rocke_attr_set_float(vm->b, m, key, d);
        }
        else if(strcmp(t, "b") == 0)
        {
            if(v->kind != JD_BOOL)
            {
                rv_fail(vm, "attr '%s' bool value is not boolean", key);
                return;
            }
            rocke_attr_set_bool(vm->b, m, key, v->b);
        }
        else if(strcmp(t, "s") == 0)
        {
            if(v->kind != JD_STR)
            {
                rv_fail(vm, "attr '%s' string value is not a string", key);
                return;
            }
            rocke_attr_set_str(vm->b, m, key, v->str);
        }
        else if(strcmp(t, "l") == 0 && v->kind == JD_ARR)
        {
            int n = v->arr_len;
            int64_t* vals = n ? (int64_t*)malloc((size_t)n * sizeof *vals) : NULL;
            if(n && !vals)
            {
                rv_fail(vm, "oom attr int list");
                return;
            }
            for(int j = 0; j < n && !vm->failed; j++)
            {
                const jd_val_t* wrapped = rocke_jget(v->arr[j], "_");
                const char* item_t = rocke_jstr(rocke_jget(wrapped, "t"));
                const jd_val_t* item_v = rocke_jget(wrapped, "v");
                if(!wrapped || !item_t || strcmp(item_t, "i") != 0 || !item_v)
                {
                    rv_fail(vm, "attr '%s' unsupported non-integer list", key);
                    break;
                }
                vals[j] = (int64_t)rv_int(vm, item_v);
            }
            if(!vm->failed)
                rocke_attr_set_int_list(vm->b, m, key, vals, n);
            free(vals);
        }
        else
            rv_fail(vm, "attr '%s' bad kind '%s'", key, t);
    }
}

/* ---------------------------------------------------------------- execute */

static void rv_exec_list(rvm_t* vm, const jd_val_t* program);

static const char* rv_bind_name(rvm_t* vm, const jd_val_t* instr, const char* fallback)
{
    const char* b = rocke_jstr(rocke_jget(instr, "bind"));
    return b ? rv_resolve_name(vm, b) : fallback;
}

static void rv_exec_instr(rvm_t* vm, const jd_val_t* instr)
{
    if(vm->failed)
        return;
    if(!instr || instr->kind != JD_OBJ)
    {
        rv_fail(vm, "instruction must be an object");
        return;
    }
    const char* op = rocke_jstr(rocke_jget(instr, "op"));
    if(!op)
    {
        rv_fail(vm, "instr missing op");
        return;
    }

    if(strcmp(op, "param") == 0)
    {
        const char* name = rocke_jstr(rocke_jget(instr, "name"));
        const rocke_type_t* type = rv_type(vm, rocke_jget(instr, "type"));
        if(!name || !type)
        {
            rv_fail(vm, "bad param");
            return;
        }
        rocke_param_opts_t opts;
        memset(&opts, 0, sizeof opts);
        const jd_val_t* pa = rocke_jget(instr, "attrs");
        if(pa && pa->kind != JD_OBJ)
        {
            rv_fail(vm, "param attrs must be an object");
            return;
        }
        if(pa)
        {
            for(int k = 0; k < pa->obj_len; k++)
            {
                const char* key = pa->obj[k].key;
                const jd_val_t* v = pa->obj[k].val;
                double d;
                if(strcmp(key, "noalias") == 0)
                {
                    if(v->kind != JD_BOOL)
                    {
                        rv_fail(vm, "param attr 'noalias' must be boolean");
                        return;
                    }
                    opts.noalias = v->b;
                    opts.noalias_set = true;
                }
                else if(strcmp(key, "readonly") == 0)
                {
                    if(v->kind != JD_BOOL)
                    {
                        rv_fail(vm, "param attr 'readonly' must be boolean");
                        return;
                    }
                    opts.readonly = v->b;
                    opts.readonly_set = true;
                }
                else if(strcmp(key, "writeonly") == 0)
                {
                    if(v->kind != JD_BOOL)
                    {
                        rv_fail(vm, "param attr 'writeonly' must be boolean");
                        return;
                    }
                    opts.writeonly = v->b;
                    opts.writeonly_set = true;
                }
                else if(strcmp(key, "align") == 0)
                {
                    if(!rocke_jnum(v, &d) || !(d >= 1.0 && d <= INT_MAX)
                       || d != (double)(int)d || ((int)d & ((int)d - 1)) != 0)
                    {
                        rv_fail(vm,
                                "param attr 'align' must be a positive power-of-two integer "
                                "fitting int");
                        return;
                    }
                    opts.align = (int)d;
                    opts.align_set = true;
                }
                else if(strcmp(key, "addr_space") == 0)
                {
                    if(v->kind != JD_STR)
                    {
                        rv_fail(vm, "param attr 'addr_space' must be a string");
                        return;
                    }
                    opts.addr_space = v->str;
                }
            }
        }
        rocke_value_t* pv = rocke_b_param(vm->b, name, type, &opts);
        if(!pv)
        {
            rv_fail(vm, "param '%s' failed", name);
            return;
        }
        rv_reg_set(vm, rv_bind_name(vm, instr, name), pv);
        return;
    }
    if(strcmp(op, "const_i32") == 0)
    {
        rocke_value_t* v = rocke_b_const_i32(vm->b, rv_int(vm, rocke_jget(instr, "val")));
        const char* b = rv_bind_name(vm, instr, "c");
        rv_name(vm, v, b);
        rv_reg_set(vm, b, v);
        return;
    }
    if(strcmp(op, "const_f32") == 0)
    {
        double d = 0;
        rocke_jnum(rocke_jget(instr, "fval"), &d);
        rocke_value_t* v = rocke_b_const_f32(vm->b, d);
        const char* b = rv_bind_name(vm, instr, "c");
        rv_name(vm, v, b);
        rv_reg_set(vm, b, v);
        return;
    }
    if(strcmp(op, "thread_id_x") == 0)
    {
        rocke_value_t* v = rocke_b_thread_id_x(vm->b);
        const char* b = rv_bind_name(vm, instr, "tid");
        rv_name(vm, v, b);
        rv_reg_set(vm, b, v);
        return;
    }
    if(strcmp(op, "alias") == 0)
    {
        /* Rebind a register to an existing value (e.g. an else-arm result, or a
         * loop-carry / lane-family alias). Both names may be format names. */
        const char* from = rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "from")));
        rocke_value_t* v = from ? rv_reg_get(vm, from) : NULL;
        if(!v)
        {
            rv_fail(vm, "alias unresolved '%s'", from ? from : "?");
            return;
        }
        rv_reg_set(vm, rv_bind_name(vm, instr, "r"), v);
        return;
    }
    if(strcmp(op, "ret") == 0)
    {
        rocke_b_ret(vm->b);
        return;
    }
    if(strcmp(op, "static_for") == 0)
    {
        const char* var = rocke_jstr(rocke_jget(instr, "var"));
        long lo = rv_int(vm, rocke_jget(instr, "lo"));
        long hi = rv_int(vm, rocke_jget(instr, "hi"));
        const jd_val_t* stepn = rocke_jget(instr, "step");
        long step = stepn ? rv_int(vm, stepn) : 1;
        const jd_val_t* body = rocke_jget(instr, "body");
        if(!var || !body || body->kind != JD_ARR || step <= 0)
        {
            rv_fail(vm, "static_for step must be positive");
            return;
        }
        for(long iv = lo; iv < hi && !vm->failed; iv += step)
        {
            int mark = vm->n_ivars;
            rv_ivar_push(vm, var, iv);
            rv_exec_list(vm, body);
            vm->n_ivars = mark; /* pop loop var */
            if(iv > LONG_MAX - step)
            {
                rv_fail(vm, "static_for loop overflow");
                return;
            }
        }
        return;
    }
    if(strcmp(op, "static_if") == 0)
    {
        /* Compile-time branch on a spec predicate (truthy intexpr). */
        long pred = rv_int(vm, rocke_jget(instr, "pred"));
        if(vm->failed)
            return;
        const jd_val_t* arm = rocke_jget(instr, pred ? "then" : "else");
        if(arm)
            rv_exec_list(vm, arm);
        return;
    }
    if(strcmp(op, "scf_for") == 0)
    {
        /* Runtime loop emitted as a real scf.for op (bounds are IR values).
         * iter-args/results may be parametric (rolled groups + format names) ->
         * a spec-derived NUMBER of loop-carries (the variable fan). */
        rocke_value_t* lo
            = rv_reg_get(vm, rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "lo"))));
        rocke_value_t* hi
            = rv_reg_get(vm, rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "hi"))));
        rocke_value_t* step
            = rv_reg_get(vm, rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "step"))));
        const char* iv = rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "iv")));
        if(!lo || !hi || !step || !iv)
        {
            rv_fail(vm, "scf_for needs lo/hi/step/iv");
            return;
        }
        rv_names_t inames = {0}, iinits = {0}, results = {0};
        rv_expand_list(vm, rocke_jget(instr, "iter"), &inames, &iinits);
        rv_expand_list(vm, rocke_jget(instr, "results"), &results, NULL);
        int n_iter = inames.n;
        if(!vm->failed && (iinits.n != n_iter || results.n != n_iter))
            rv_fail(vm, "scf_for iter/init/result counts differ");
        rocke_iter_arg_t* ia
            = n_iter ? (rocke_iter_arg_t*)malloc((size_t)n_iter * sizeof *ia) : NULL;
        for(int i = 0; i < n_iter && !vm->failed; i++)
        {
            ia[i].name = inames.a[i];
            ia[i].init = rv_reg_get(vm, iinits.a[i]);
            if(!ia[i].init)
                rv_fail(vm, "scf_for iter '%s' init unresolved '%s'", inames.a[i], iinits.a[i]);
        }
        if(vm->failed)
        {
            free(ia);
            free(inames.a);
            free(iinits.a);
            free(results.a);
            return;
        }
        const jd_val_t* un = rocke_jget(instr, "unroll");
        const jd_val_t* el = rocke_jget(instr, "elide_trailing_barrier");
        rocke_for_t f = rocke_b_scf_for_iter(
            vm->b, lo, hi, step, ia, n_iter, iv, un ? un->b : false, el ? el->b : true);
        if(!rocke_ir_builder_ok(vm->b))
        {
            rv_fail(vm, "scf_for build: %s", rocke_ir_builder_error(vm->b));
            free(ia);
            free(inames.a);
            free(iinits.a);
            free(results.a);
            return;
        }
        rv_name(vm, f.iv, iv);
        rv_reg_set(vm, iv, f.iv);
        for(int i = 0; i < n_iter; i++)
        {
            rv_name(vm, f.iter_vars[i], inames.a[i]);
            rv_reg_set(vm, inames.a[i], f.iter_vars[i]);
        }
        rocke_b_region_enter(vm->b, f.body);
        rv_exec_list(vm, rocke_jget(instr, "body"));
        rocke_b_region_leave(vm->b);
        if(!vm->failed)
            for(int i = 0; i < results.n && i < f.op->num_results; i++)
            {
                rv_name(vm, f.op->results[i], results.a[i]);
                rv_reg_set(vm, results.a[i], f.op->results[i]);
            }
        free(ia);
        free(inames.a);
        free(iinits.a);
        free(results.a);
        return;
    }
    if(strcmp(op, "scf_if") == 0)
    {
        rocke_value_t* cond
            = rv_reg_get(vm, rv_resolve_name(vm, rocke_jstr(rocke_jget(instr, "cond"))));
        if(!cond)
        {
            rv_fail(vm, "scf_if needs cond");
            return;
        }
        rocke_if_t s = rocke_b_scf_if(vm->b, cond);
        const jd_val_t* then = rocke_jget(instr, "then");
        if(then)
        {
            rocke_b_region_enter(vm->b, s.then_region);
            rv_exec_list(vm, then);
            rocke_b_region_leave(vm->b);
        }
        return;
    }
    if(strcmp(op, "emit") == 0)
    {
        const char* opcode_name = rocke_jstr(rocke_jget(instr, "opcode"));
        rocke_opcode_t opcode
            = rv_opcode_from_name(opcode_name, rv_attr_elem_type(rocke_jget(instr, "attrs")));
        if(opcode == ROCKE_OP_INVALID)
        {
            rv_fail(vm, "unknown opcode '%s'", opcode_name ? opcode_name : "?");
            return;
        }
        /* Operands may be a rolled list (e.g. an scf.yield carrying a fan of
         * loop-carries) and/or format names. */
        rv_names_t innames = {0};
        rv_expand_list(vm, rocke_jget(instr, "in"), &innames, NULL);
        int n_ops = innames.n;
        rocke_value_t** ops = n_ops ? (rocke_value_t**)malloc((size_t)n_ops * sizeof *ops) : NULL;
        for(int i = 0; i < n_ops && !vm->failed; i++)
        {
            ops[i] = rv_reg_get(vm, innames.a[i]);
            if(!ops[i])
                rv_fail(vm, "emit '%s' unresolved operand '%s'", opcode_name, innames.a[i]);
        }
        /* Results: a single "out" {bind,type} or multiple "outs":[{bind,type}]. */
        const jd_val_t* out = rocke_jget(instr, "out");
        const jd_val_t* outs = rocke_jget(instr, "outs");
        const rocke_type_t** rtypes = NULL;
        const char** binds = NULL;
        int n_res = 0;
        if(out && outs)
        {
            rv_fail(vm, "emit '%s' cannot have both out and outs", opcode_name);
        }
        else if(!vm->failed && out && out->kind == JD_OBJ)
        {
            rtypes = (const rocke_type_t**)malloc(sizeof *rtypes);
            binds = (const char**)malloc(sizeof *binds);
            if(!rtypes || !binds)
            {
                rv_fail(vm, "oom emit results");
            }
            else
            {
                rtypes[0] = rv_type(vm, rocke_jget(out, "type"));
                binds[0] = rv_bind_name(vm, out, "r");
                n_res = 1;
            }
        }
        else if(!vm->failed && out)
        {
            rv_fail(vm, "emit '%s' out must be an object", opcode_name);
        }
        else if(!vm->failed && outs && outs->kind == JD_ARR)
        {
            n_res = outs->arr_len;
            if(n_res > 0)
            {
                rtypes = (const rocke_type_t**)malloc((size_t)n_res * sizeof *rtypes);
                binds = (const char**)malloc((size_t)n_res * sizeof *binds);
                if(!rtypes || !binds)
                {
                    rv_fail(vm, "oom emit results");
                }
            }
            for(int i = 0; i < n_res && !vm->failed; i++)
            {
                if(outs->arr[i]->kind != JD_OBJ)
                {
                    rv_fail(vm, "emit '%s' result %d is not an object", opcode_name, i);
                    break;
                }
                rtypes[i] = rv_type(vm, rocke_jget(outs->arr[i], "type"));
                const char* raw_bind = rocke_jstr(rocke_jget(outs->arr[i], "bind"));
                if(!raw_bind || !*raw_bind)
                {
                    rv_fail(vm, "emit '%s' result bind must be a nonempty string", opcode_name);
                    break;
                }
                binds[i] = rv_resolve_name(vm, raw_bind);
                for(int j = 0; j < i && !vm->failed; j++)
                {
                    if(strcmp(binds[i], binds[j]) == 0)
                    {
                        rv_fail(vm, "emit '%s' duplicate result bind '%s'", opcode_name, binds[i]);
                    }
                }
            }
        }
        else if(outs)
        {
            rv_fail(vm, "emit '%s' outs must be an array", opcode_name);
        }
        rocke_attr_map_t m;
        if(!vm->failed)
            rv_attrs(vm, rocke_jget(instr, "attrs"), &m);
        if(vm->failed)
        {
            free(ops);
            free(innames.a);
            free(rtypes);
            free(binds);
            return;
        }
        /* The smem type node deliberately omits the `exclusive` bit; it rides
         * on the smem_alloc op as an attr. Rebuild the result SmemType with
         * exclusive set so the packer's no-alias behavior round-trips
         * (lower_llvm reads it off the SmemType, not the attr). */
        if(opcode == ROCKE_OP_TILE_SMEM_ALLOC && n_res > 0 && rtypes[0]
           && rtypes[0]->kind == ROCKE_TYPE_SMEM && rocke_attr_get_bool(&m, "exclusive", false))
        {
            rtypes[0]
                = rocke_smem_type(vm->b, rtypes[0]->elem, rtypes[0]->shape, rtypes[0]->rank, 1);
            if(!rtypes[0])
            {
                rv_fail(vm, "smem_alloc: exclusive type rebuild failed");
                free(ops);
                free(innames.a);
                free(rtypes);
                free(binds);
                return;
            }
        }
        rocke_op_t* built = rocke_b_op(
            vm->b, opcode, ops, n_ops, n_res ? rtypes : NULL, n_res, &m, NULL, 0, NULL, NULL);
        if(!built || !rocke_ir_builder_ok(vm->b))
        {
            rv_fail(vm,
                    "emit '%s' failed: %s",
                    opcode_name,
                    rocke_ir_builder_ok(vm->b) ? "null" : rocke_ir_builder_error(vm->b));
            free(ops);
            free(innames.a);
            free(rtypes);
            free(binds);
            return;
        }
        if(built->num_results != n_res)
        {
            rv_fail(vm,
                    "emit '%s' produced %d results, expected %d",
                    opcode_name,
                    built->num_results,
                    n_res);
            free(ops);
            free(innames.a);
            free(rtypes);
            free(binds);
            return;
        }
        /* tile.smem_alloc's result value name becomes the LDS global symbol,
         * which DOES affect the compiled object. Name it exactly per the recipe
         * bind (Python's name) so the HSACO is byte-identical -- rocke_b_fresh
         * would otherwise append the VM's own counter. (Other ops' result names
         * are local temps that the backend renumbers, so they stay fresh to
         * avoid collisions in rolled loop bodies.) */
        if(n_res == 1 && built->num_results > 0 && strcmp(opcode_name, "tile.smem_alloc") == 0)
            built->results[0]->name = rocke_arena_printf(&vm->b->arena, "%%%s", binds[0]);
        for(int i = 0; i < n_res && i < built->num_results; i++)
        {
            rv_name(vm, built->results[i], binds[i]);
            rv_reg_set(vm, binds[i], built->results[i]);
        }
        free(ops);
        free(innames.a);
        free(rtypes);
        free(binds);
        return;
    }
    rv_fail(vm, "unknown instr op '%s'", op);
}

static void rv_exec_list(rvm_t* vm, const jd_val_t* program)
{
    if(!program || program->kind != JD_ARR)
    {
        rv_fail(vm, "program/body not an array");
        return;
    }
    for(int i = 0; i < program->arr_len && !vm->failed; i++)
        rv_exec_instr(vm, program->arr[i]);
}

/* ----------------------------------------------------- kernel name format */

/* Expand "{NAME}" tokens in `fmt` using the int/str specs. The returned string
 * is caller-owned. Kernel names are deliberately unbounded here: production
 * spec composition can legitimately exceed a small stack buffer. */
static char* rv_format_name(rvm_t* vm, const char* fmt)
{
    size_t cap = strlen(fmt) + 32;
    if(cap < 64)
        cap = 64;
    char* out = (char*)malloc(cap);
    if(!out)
    {
        rv_fail(vm, "oom kernel name");
        return NULL;
    }
    size_t n = 0;
    out[0] = '\0';
    for(const char* p = fmt; *p;)
    {
        if(*p == '{')
        {
            const char* close = strchr(p, '}');
            if(!close)
            {
                rv_fail(vm, "unterminated kernel name placeholder");
                free(out);
                return NULL;
            }
            char key[64];
            size_t klen = (size_t)(close - p - 1);
            if(klen >= sizeof key)
            {
                rv_fail(vm, "kernel name placeholder too long");
                free(out);
                return NULL;
            }
            memcpy(key, p + 1, klen);
            key[klen] = '\0';
            long iv;
            const char* sv;
            char tmp[32];
            const char* val = NULL;
            if(rv_spec_int(vm, key, &iv))
            {
                snprintf(tmp, sizeof tmp, "%ld", iv);
                val = tmp;
            }
            else if((sv = rv_spec_str(vm, key)) != NULL)
            {
                val = sv;
            }
            if(!val)
            {
                rv_fail(vm, "unresolved kernel name placeholder '%s'", key);
                free(out);
                return NULL;
            }
            if(!rv_append(vm, &out, &n, &cap, val, strlen(val), "kernel name"))
            {
                free(out);
                return NULL;
            }
            p = close + 1;
        }
        else
        {
            if(*p == '}')
            {
                rv_fail(vm, "unmatched kernel name placeholder close");
                free(out);
                return NULL;
            }
            if(!rv_append(vm, &out, &n, &cap, p, 1, "kernel name"))
            {
                free(out);
                return NULL;
            }
            p++;
        }
    }
    return out;
}

static bool rv_validate_specs(rvm_t* vm, const jd_val_t* spec)
{
    if(vm->n_ints < 0 || vm->n_strs < 0 || (vm->n_ints && !vm->ints)
       || (vm->n_strs && !vm->strs))
    {
        rv_fail(vm, "invalid runtime spec arrays");
        return false;
    }
    if(spec && spec->kind != JD_ARR)
    {
        rv_fail(vm, "recipe spec must be an array");
        return false;
    }

    const int n_decl = spec ? spec->arr_len : 0;
    for(int i = 0; i < n_decl; i++)
    {
        const jd_val_t* decl = spec->arr[i];
        const char* name = rocke_jstr(rocke_jget(decl, "name"));
        const char* kind = rocke_jstr(rocke_jget(decl, "kind"));
        if(!decl || decl->kind != JD_OBJ || !name || !*name || !kind
           || (strcmp(kind, "int") != 0 && strcmp(kind, "str") != 0))
        {
            rv_fail(vm, "malformed recipe spec declaration %d", i);
            return false;
        }
        for(int j = 0; j < i; j++)
        {
            const char* prior = rocke_jstr(rocke_jget(spec->arr[j], "name"));
            if(prior && strcmp(prior, name) == 0)
            {
                rv_fail(vm, "duplicate recipe spec '%s'", name);
                return false;
            }
        }

        int int_matches = 0;
        int str_matches = 0;
        for(int j = 0; j < vm->n_ints; j++)
            if(vm->ints[j].name && strcmp(vm->ints[j].name, name) == 0)
                int_matches++;
        for(int j = 0; j < vm->n_strs; j++)
            if(vm->strs[j].name && strcmp(vm->strs[j].name, name) == 0)
                str_matches++;
        if(strcmp(kind, "int") == 0 && (int_matches != 1 || str_matches != 0))
        {
            rv_fail(vm, "runtime spec '%s' must have exactly one int value", name);
            return false;
        }
        if(strcmp(kind, "str") == 0 && (str_matches != 1 || int_matches != 0))
        {
            rv_fail(vm, "runtime spec '%s' must have exactly one string value", name);
            return false;
        }
    }

    for(int i = 0; i < vm->n_ints; i++)
    {
        const char* name = vm->ints[i].name;
        if(!name || !*name)
        {
            rv_fail(vm, "runtime int spec has no name");
            return false;
        }
        bool declared = false;
        for(int j = 0; j < n_decl; j++)
        {
            const char* decl_name = rocke_jstr(rocke_jget(spec->arr[j], "name"));
            const char* kind = rocke_jstr(rocke_jget(spec->arr[j], "kind"));
            if(decl_name && strcmp(decl_name, name) == 0 && kind && strcmp(kind, "int") == 0)
                declared = true;
        }
        if(!declared)
        {
            rv_fail(vm, "undeclared runtime int spec '%s'", name);
            return false;
        }
    }
    for(int i = 0; i < vm->n_strs; i++)
    {
        const char* name = vm->strs[i].name;
        if(!name || !*name || !vm->strs[i].value)
        {
            rv_fail(vm, "runtime string spec has invalid name/value");
            return false;
        }
        bool declared = false;
        for(int j = 0; j < n_decl; j++)
        {
            const char* decl_name = rocke_jstr(rocke_jget(spec->arr[j], "name"));
            const char* kind = rocke_jstr(rocke_jget(spec->arr[j], "kind"));
            if(decl_name && strcmp(decl_name, name) == 0 && kind && strcmp(kind, "str") == 0)
                declared = true;
        }
        if(!declared)
        {
            rv_fail(vm, "undeclared runtime string spec '%s'", name);
            return false;
        }
    }
    return true;
}

/* Execute a recipe whose DOM root has already been parsed (from JSON or CBOR).
 * The DOM must stay alive (its arena owned by the caller) until this returns.
 * Does NOT destroy the caller's arena. */
static rocke_status_t rv_run_root(jd_val_t* root,
                                  const rocke_recipe_spec_int_t* ints,
                                  int n_ints,
                                  const rocke_recipe_spec_str_t* strs,
                                  int n_strs,
                                  rocke_ir_builder_t* out_builder,
                                  rocke_kernel_def_t** out_kernel,
                                  char* err,
                                  size_t err_cap)
{
    if(out_kernel)
        *out_kernel = NULL;
    if(!root || root->kind != JD_OBJ || !out_builder || !out_kernel)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "invalid recipe root or output pointers");
        return ROCKE_ERR_VALUE;
    }
    const char* schema = rocke_jstr(rocke_jget(root, "schema"));
    if(!schema || strcmp(schema, "rocke.recipe/v1") != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "bad/missing schema (want rocke.recipe/v1)");
        return ROCKE_ERR_VALUE;
    }

    rvm_t vm;
    memset(&vm, 0, sizeof vm);
    vm.ints = ints;
    vm.n_ints = n_ints;
    vm.strs = strs;
    vm.n_strs = n_strs;

    const jd_val_t* spec = rocke_jget(root, "spec");
    if(!rv_validate_specs(&vm, spec))
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", vm.err);
        return ROCKE_ERR_VALUE;
    }

    /* Exact SSA naming for CONCRETE recipes only, detected by an empty "spec":
     * with no spec there is no static_for/rolled-list expansion, so every bind is
     * a unique (Python) SSA name and can be applied verbatim -> byte-identical
     * .ll. Parametric recipes (non-empty spec) unroll and reuse binds across
     * iterations, so they must keep fresh names to avoid SSA collisions. */
    vm.exact_names = !spec || spec->kind != JD_ARR || spec->arr_len == 0;

    const char* fmt = rocke_jstr(rocke_jget(root, "kernel_name_fmt"));
    if(!fmt)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "missing kernel_name_fmt");
        return ROCKE_ERR_VALUE;
    }
    char* kname = rv_format_name(&vm, fmt);
    if(!kname)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", vm.err);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st = rocke_ir_builder_init(out_builder, kname);
    free(kname);
    if(st != ROCKE_OK)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "builder init failed (%d)", (int)st);
        return st;
    }
    vm.b = out_builder;

    /* kernel attrs (e.g. max_workgroup_size), typed like portable IR. */
    const jd_val_t* kattrs = rocke_jget(root, "attrs");
    if(kattrs)
    {
        rocke_kernel_def_t* k = rocke_ir_builder_kernel(out_builder);
        rv_attrs(&vm, kattrs, &k->attrs);
        if(vm.failed)
        {
            if(err && err_cap)
                snprintf(err, err_cap, "%s", vm.err);
            rocke_ir_builder_free(out_builder);
            return ROCKE_ERR_VALUE;
        }
    }

    rv_exec_list(&vm, rocke_jget(root, "program"));

    free(vm.regs);
    free(vm.ivars);
    for(int i = 0; i < vm.n_owned; i++)
        free(vm.owned[i]);
    free(vm.owned);

    if(vm.failed)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", vm.err);
        rocke_ir_builder_free(out_builder);
        return ROCKE_ERR_VALUE;
    }
    if(!rocke_ir_builder_ok(out_builder))
    {
        if(err && err_cap)
            snprintf(err, err_cap, "builder error: %s", rocke_ir_builder_error(out_builder));
        rocke_ir_builder_free(out_builder);
        return rocke_ir_builder_status(out_builder);
    }

    *out_kernel = rocke_ir_builder_kernel(out_builder);
    return ROCKE_OK;
}

rocke_status_t rocke_recipe_run_from_json(const char* text,
                                          const rocke_recipe_spec_int_t* ints,
                                          int n_ints,
                                          const rocke_recipe_spec_str_t* strs,
                                          int n_strs,
                                          rocke_ir_builder_t* out_builder,
                                          rocke_kernel_def_t** out_kernel,
                                          char* err,
                                          size_t err_cap)
{
    if(out_kernel)
        *out_kernel = NULL;
    if(!text || !out_builder)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "null text/builder");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = rocke_json_parse(text, &arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st
        = rv_run_root(root, ints, n_ints, strs, n_strs, out_builder, out_kernel, err, err_cap);
    rocke_arena_destroy(&arena);
    return st;
}

rocke_status_t rocke_recipe_run_from_cbor(const unsigned char* data,
                                          size_t len,
                                          const rocke_recipe_spec_int_t* ints,
                                          int n_ints,
                                          const rocke_recipe_spec_str_t* strs,
                                          int n_strs,
                                          rocke_ir_builder_t* out_builder,
                                          rocke_kernel_def_t** out_kernel,
                                          char* err,
                                          size_t err_cap)
{
    if(out_kernel)
        *out_kernel = NULL;
    if(!data || !out_builder)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "null data/builder");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st
        = rv_run_root(root, ints, n_ints, strs, n_strs, out_builder, out_kernel, err, err_cap);
    rocke_arena_destroy(&arena);
    return st;
}

/* Find the recipe DOM for `key` (and `arch`, if non-NULL) inside a parsed
 * bundle (schema "rocke.bundle/v1"). Returns NULL if absent. */
static const jd_val_t* rv_bundle_find(const jd_val_t* bundle, const char* key, const char* arch)
{
    const jd_val_t* entries = rocke_jget(bundle, "entries");
    if(!entries || entries->kind != JD_ARR)
        return NULL;
    for(int i = 0; i < entries->arr_len; i++)
    {
        const jd_val_t* e = entries->arr[i];
        const char* k = rocke_jstr(rocke_jget(e, "key"));
        if(!k || strcmp(k, key) != 0)
            continue;
        if(arch)
        {
            const char* a = rocke_jstr(rocke_jget(e, "arch"));
            if(!a || strcmp(a, arch) != 0)
                continue;
        }
        return rocke_jget(e, "recipe");
    }
    return NULL;
}

rocke_status_t rocke_recipe_run_from_bundle_cbor(const unsigned char* data,
                                                 size_t len,
                                                 const char* key,
                                                 const char* arch,
                                                 const rocke_recipe_spec_int_t* ints,
                                                 int n_ints,
                                                 const rocke_recipe_spec_str_t* strs,
                                                 int n_strs,
                                                 rocke_ir_builder_t* out_builder,
                                                 rocke_kernel_def_t** out_kernel,
                                                 char* err,
                                                 size_t err_cap)
{
    if(out_kernel)
        *out_kernel = NULL;
    if(!data || !key || !out_builder)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "null data/key/builder");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    const char* schema = rocke_jstr(rocke_jget(root, "schema"));
    if(!schema || strcmp(schema, "rocke.bundle/v1") != 0)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "bad/missing schema (want rocke.bundle/v1)");
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    const jd_val_t* recipe = rv_bundle_find(root, key, arch);
    if(!recipe)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "recipe '%s' (arch %s) not in bundle", key, arch ? arch : "*");
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st = rv_run_root(
        (jd_val_t*)recipe, ints, n_ints, strs, n_strs, out_builder, out_kernel, err, err_cap);
    rocke_arena_destroy(&arena);
    return st;
}
