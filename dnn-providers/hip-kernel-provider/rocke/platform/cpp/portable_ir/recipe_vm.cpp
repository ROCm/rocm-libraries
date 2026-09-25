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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/abi.h"
#include "rocke/arena.h"
#include "rocke/cbor_dom.h"
#include "rocke/json_dom.h"
#include "rocke/recipe_guard.h"
#include "rocke/recipe_launch.h"

/* ------------------------------------------------------------------ state */

typedef struct
{
    const char* name;
    rocke_value_t* val;
    int next; /* next reg in this hash bucket, or -1 */
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

    rv_reg_t* regs; /* IR-value registers, name-unique; indexed by reg_buckets */
    int n_regs, cap_regs;
    int* reg_buckets; /* open hash over reg names -> head index, chained via .next */
    int reg_nbuckets;
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
    for(int i = 0; i < vm->n_strs; i++)
        if(strcmp(vm->strs[i].name, name) == 0)
            return vm->strs[i].value;
    return NULL;
}

/* The register table is name-unique (rv_reg_set updates in place), so a hash
 * index over it is a straight swap for the former linear scan. It has to be an
 * index rather than a scan: a recorded production kernel binds thousands of
 * names and looks each operand up again, which is quadratic scanned -- 66ms of
 * strcmp on a recorded flash-attention prefill, against 2ms to lower it. */
static unsigned rv_reg_hash(const char* s)
{
    /* FNV-1a, matching the portable-IR importer's vmap. */
    unsigned h = 2166136261u;
    for(; *s; s++)
    {
        h ^= (unsigned char)*s;
        h *= 16777619u;
    }
    return h;
}

static bool rv_reg_rehash(rvm_t* vm, int want)
{
    int n = vm->reg_nbuckets ? vm->reg_nbuckets : 1024;
    while(n < want)
        n *= 2;
    int* nb = (int*)malloc((size_t)n * sizeof(int));
    if(!nb)
    {
        rv_fail(vm, "oom reg buckets");
        return false;
    }
    for(int i = 0; i < n; i++)
        nb[i] = -1;
    for(int i = 0; i < vm->n_regs; i++)
    {
        unsigned h = rv_reg_hash(vm->regs[i].name) & (unsigned)(n - 1);
        vm->regs[i].next = nb[h];
        nb[h] = i;
    }
    free(vm->reg_buckets);
    vm->reg_buckets = nb;
    vm->reg_nbuckets = n;
    return true;
}

static int rv_reg_find(rvm_t* vm, const char* name)
{
    if(vm->reg_nbuckets == 0)
        return -1;
    unsigned h = rv_reg_hash(name) & (unsigned)(vm->reg_nbuckets - 1);
    for(int i = vm->reg_buckets[h]; i != -1; i = vm->regs[i].next)
        if(strcmp(vm->regs[i].name, name) == 0)
            return i;
    return -1;
}

static void rv_reg_set(rvm_t* vm, const char* name, rocke_value_t* val)
{
    int i = rv_reg_find(vm, name);
    if(i != -1)
    {
        vm->regs[i].val = val;
        return;
    }
    if(vm->reg_nbuckets == 0 && !rv_reg_rehash(vm, 1024))
        return;
    if(vm->n_regs + 1 > (vm->reg_nbuckets * 3) / 4 && !rv_reg_rehash(vm, vm->reg_nbuckets * 2))
        return;
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
    int s = vm->n_regs++;
    vm->regs[s].name = name;
    vm->regs[s].val = val;
    unsigned h = rv_reg_hash(name) & (unsigned)(vm->reg_nbuckets - 1);
    vm->regs[s].next = vm->reg_buckets[h];
    vm->reg_buckets[h] = s;
}

static rocke_value_t* rv_reg_get(rvm_t* vm, const char* name)
{
    int i = rv_reg_find(vm, name);
    return i != -1 ? vm->regs[i].val : NULL;
}

static bool rv_ivar_get(rvm_t* vm, const char* name, long* out)
{
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
        /* Unary functions of ONE operand: {"<fn>": e} (no array wrapper).
           These regenerate constants that a code generator derived from a spec
           value rather than values the kernel computed from it: the operands of
           a strength-reduced unsigned division, emitted as
           (umul_hi(n, M) + n) >> s. The shift is logarithmic in the divisor and
           the multiplier depends on its odd part, so neither is expressible as
           arithmetic on the axis -- the recipe has to regenerate them.
           Mirrors recipe_expand.py::magic_division_constants, and upstream of
           both, helpers/transforms.py::calculate_magic_numbers. */
        const jd_val_t* mm = rocke_jget(e, "magic_multiplier");
        const jd_val_t* ms = rocke_jget(e, "magic_shift");
        if(mm || ms)
        {
            long d = rv_int(vm, mm ? mm : ms);
            if(d < 1 || d > 0x7fffffffL)
            {
                rv_fail(vm, "magic division needs 1 <= divisor < 2^31, got %ld", d);
                return 0;
            }
            int shift = 0;
            while((1LL << shift) < (long long)d)
                shift++;
            if(ms)
                return shift;
            long long mult = ((((1LL << shift) - (long long)d) << 32) / (long long)d) + 1;
            if(mult >= (1LL << 31))
                mult -= (1LL << 32);
            return (long)mult;
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
                return b ? a / b : 0;
            case 4:
                return b ? a % b : 0;
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

/* ------------------------------------------------------------- wire ABI gate */

/* Can this engine read this artifact? Reads the optional
 * "abi": {"min_reader": N, ...} block a generator stamps on a recipe or bundle
 * and refuses when N exceeds what this build understands. See rocke/abi.h.
 *
 * Checked BEFORE the schema string, and the ordering is for the error message
 * rather than for safety. A future artifact will carry both a new schema and a
 * higher min_reader; reporting "bad/missing schema (want rocke.recipe/v1)"
 * sends the reader looking for a corrupt file, while "needs a reader >= 2, this
 * engine is 1" names the actual problem and its fix.
 *
 * An ABSENT block means level 1, so every recipe recorded before this existed
 * still replays. Absence has to mean the floor rather than "unknown": treating
 * it as unreadable would strand existing bundles for no safety gain, since a
 * level-1 artifact is exactly what a level-1 engine was written to read. */
static bool rv_abi_ok(const jd_val_t* node, char* err, size_t err_cap)
{
    const jd_val_t* abi = rocke_jget(node, "abi");
    if(!abi || abi->kind != JD_OBJ)
        return true;
    double need;
    if(!rocke_jnum(rocke_jget(abi, "min_reader"), &need))
        return true;
    if((int)need <= ROCKE_RECIPE_ABI)
        return true;
    if(err && err_cap)
    {
        const char* who = rocke_jstr(rocke_jget(abi, "engine"));
        ROCKE_ERR_SNPRINTF(err,
                           err_cap,
                           "artifact needs a recipe reader >= %d, this engine is %d "
                           "(written by engine %s)",
                           (int)need,
                           ROCKE_RECIPE_ABI,
                           who ? who : "?");
    }
    return false;
}

/* --------------------------------------------------------- guard evaluation */

/* The admission guard on a rolled recipe (schema "rocke.guard/v1"): a short,
 * ORDERED list of intexpr predicates over the free axes, derived at generation
 * time from the family's own Python gate and verified against it out of sample
 * (python/rocke/portable_ir/src/guard.py). See rocke/recipe_guard.h.
 *
 * It reuses rv_int above rather than carrying its own evaluator. That is the
 * point of expressing guards in the recipe language at all: no second grammar to
 * implement, and the CI gate that already pins rv_int against
 * recipe_expand.eval_intexpr covers guards without being extended.
 *
 * Rule ORDER is part of the contract, not presentation, and so is stopping at
 * the first failure. This evaluator and the Python one disagree about `mod` and
 * `div` with a negative left operand (Python floors, C truncates), which has
 * never mattered because spec values are sizes -- and a guard is the first thing
 * to be handed a hostile one. Guards stay clear of it two ways: they only ever
 * test whether a remainder is ZERO, which is the same question in either
 * convention, and the generator puts a bounds rule ahead of the divisibility
 * rule on an axis so a negative is rejected before the `mod` is reached. Do not
 * reorder rules, and do not evaluate them all to collect every reason. */

/* Bound by the caller, as either an int or a string? */
static bool rv_bound(rvm_t* vm, const char* name)
{
    long tmp;
    if(!name)
        return false;
    return rv_spec_int(vm, name, &tmp) || rv_spec_str(vm, name) != NULL;
}

/* Does the caller's binding agree with every field of this recorded point? */
static bool rv_point_matches(rvm_t* vm, const jd_val_t* pt)
{
    if(!pt || pt->kind != JD_OBJ)
        return false;
    for(int i = 0; i < pt->obj_len; i++)
    {
        const char* k = pt->obj[i].key;
        const jd_val_t* v = pt->obj[i].val;
        const char* want = rocke_jstr(v);
        if(want)
        {
            const char* have = rv_spec_str(vm, k);
            if(!have || strcmp(have, want) != 0)
                return false;
            continue;
        }
        double d;
        long have;
        if(!rocke_jnum(v, &d) || !rv_spec_int(vm, k, &have) || have != (long)d)
            return false;
    }
    return true;
}

/* Evaluate `guard` against the spec bound into `vm`. `vm` needs only its
 * ints/strs populated -- no builder, no registers -- which is what lets the
 * public entry points answer without lowering anything. */
static rocke_guard_verdict_t
    rv_guard_eval(rvm_t* vm, const jd_val_t* guard, unsigned flags, char* reason, size_t cap)
{
    if(reason && cap)
        reason[0] = '\0';
    if(!guard || guard->kind != JD_OBJ)
        return ROCKE_GUARD_ABSENT;

    const char* schema = rocke_jstr(rocke_jget(guard, "schema"));
    if(!schema || strcmp(schema, "rocke.guard/v1") != 0)
    {
        /* An unreadable guard is a refusal, never an accept. This engine is
         * older than the bundle it was handed, so it cannot know what the newer
         * guard would have rejected -- and the one thing it must not do is wave
         * through a configuration on the strength of not understanding it. */
        if(reason && cap)
            ROCKE_ERR_SNPRINTF(reason,
                               cap,
                               "unsupported guard schema '%s' (this engine knows rocke.guard/v1)",
                               schema ? schema : "?");
        return ROCKE_GUARD_REFUSED;
    }

    const jd_val_t* freev = rocke_jget(guard, "free");
    if(freev && freev->kind == JD_ARR)
        for(int i = 0; i < freev->arr_len; i++)
        {
            const char* axis = rocke_jstr(freev->arr[i]);
            if(!rv_bound(vm, axis))
            {
                if(reason && cap)
                    ROCKE_ERR_SNPRINTF(reason, cap, "free axis '%s' not bound", axis ? axis : "?");
                return ROCKE_GUARD_REFUSED;
            }
        }

    const jd_val_t* rules = rocke_jget(guard, "rules");
    if(rules && rules->kind == JD_ARR)
        for(int i = 0; i < rules->arr_len; i++)
        {
            const jd_val_t* rule = rules->arr[i];
            const jd_val_t* pred = rocke_jget(rule, "pred");
            const char* why = rocke_jstr(rocke_jget(rule, "reason"));
            bool saved = vm->failed;
            long ok = rv_int(vm, pred);
            if(vm->failed && !saved)
            {
                /* A rule that could not be evaluated decides nothing, so it
                 * cannot be allowed to pass. Clear the sticky failure: the VM
                 * has not emitted anything and the caller gets the verdict. */
                vm->failed = false;
                if(reason && cap)
                    ROCKE_ERR_SNPRINTF(
                        reason, cap, "guard rule %d did not evaluate: %s", i, vm->err);
                return ROCKE_GUARD_REFUSED;
            }
            if(!ok)
            {
                if(reason && cap)
                    ROCKE_ERR_SNPRINTF(reason, cap, "%s", why ? why : "guard rule rejected");
                return ROCKE_GUARD_REFUSED;
            }
        }

    if(flags & ROCKE_GUARD_REQUIRE_VERIFIED)
    {
        const jd_val_t* pts = rocke_jget(guard, "verified");
        if(!pts || pts->kind != JD_ARR || pts->arr_len == 0)
        {
            if(reason && cap)
                ROCKE_ERR_SNPRINTF(
                    reason,
                    cap,
                    "ROCKE_GUARD_REQUIRE_VERIFIED but the guard carries no verified points");
            return ROCKE_GUARD_REFUSED;
        }
        for(int i = 0; i < pts->arr_len; i++)
            if(rv_point_matches(vm, pts->arr[i]))
                return ROCKE_GUARD_ADMITTED;
        if(reason && cap)
            ROCKE_ERR_SNPRINTF(reason,
                               cap,
                               "binding is not one of the %d generator-verified points",
                               pts->arr_len);
        return ROCKE_GUARD_REFUSED;
    }
    return ROCKE_GUARD_ADMITTED;
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

/* Resolve a register name that may contain {var} loop-index / spec tokens
 * (e.g. "acc_m{lane}_n0" -> "acc_m2_n0"). Names without '{' pass through with no
 * allocation. */
static const char* rv_resolve_name(rvm_t* vm, const char* raw)
{
    if(!raw || !strchr(raw, '{'))
        return raw;
    char buf[256];
    size_t n = 0;
    for(const char* p = raw; *p && n + 1 < sizeof buf;)
    {
        if(*p == '{')
        {
            const char* close = strchr(p, '}');
            if(!close)
            {
                buf[n++] = *p++;
                continue;
            }
            char key[64];
            size_t kl = (size_t)(close - p - 1);
            if(kl >= sizeof key)
                kl = sizeof key - 1;
            memcpy(key, p + 1, kl);
            key[kl] = '\0';
            long v;
            if(rv_ivar_get(vm, key, &v) || rv_spec_int(vm, key, &v))
                n += (size_t)snprintf(buf + n, sizeof buf - n, "%ld", v);
            else
            {
                rv_fail(vm, "unresolved name var '%s'", key);
                return raw;
            }
            p = close + 1;
        }
        else
        {
            buf[n++] = *p++;
        }
    }
    buf[n] = '\0';
    return rv_intern(vm, buf);
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
        return;
    for(int i = 0; i < arr->arr_len && !vm->failed; i++)
    {
        const jd_val_t* e = arr->arr[i];
        if(e->kind == JD_STR)
        {
            rv_names_push(vm, names, rv_resolve_name(vm, e->str));
            continue;
        }
        const jd_val_t* fr = rocke_jget(e, "for");
        const char* nm = rocke_jstr(rocke_jget(e, "name"));
        const char* init = inits ? rocke_jstr(rocke_jget(e, "init")) : NULL;
        if(fr)
        {
            const char* var = rocke_jstr(rocke_jget(fr, "var"));
            long lo = rv_int(vm, rocke_jget(fr, "lo"));
            long hi = rv_int(vm, rocke_jget(fr, "hi"));
            const jd_val_t* sn = rocke_jget(fr, "step");
            long step = sn ? rv_int(vm, sn) : 1;
            if(!var || step == 0)
            {
                rv_fail(vm, "bad rolled-list for");
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
            }
        }
        else
        {
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
    if(!attrs || attrs->kind != JD_OBJ)
        return;
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
            rocke_jnum(v, &d);
            rocke_attr_set_float(vm->b, m, key, d);
        }
        else if(strcmp(t, "b") == 0)
            rocke_attr_set_bool(vm->b, m, key, v->b);
        else if(strcmp(t, "s") == 0)
            rocke_attr_set_str(vm->b, m, key, v->str ? v->str : "");
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
        if(pa && pa->kind == JD_OBJ)
        {
            for(int k = 0; k < pa->obj_len; k++)
            {
                const char* key = pa->obj[k].key;
                const jd_val_t* v = pa->obj[k].val;
                double d;
                if(strcmp(key, "noalias") == 0)
                {
                    opts.noalias = v->b;
                    opts.noalias_set = true;
                }
                else if(strcmp(key, "readonly") == 0)
                {
                    opts.readonly = v->b;
                    opts.readonly_set = true;
                }
                else if(strcmp(key, "writeonly") == 0)
                {
                    opts.writeonly = v->b;
                    opts.writeonly_set = true;
                }
                else if(strcmp(key, "align") == 0 && rocke_jnum(v, &d))
                {
                    opts.align = (int)d;
                    opts.align_set = true;
                }
                else if(strcmp(key, "addr_space") == 0)
                {
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
        if(!var || !body || body->kind != JD_ARR || step == 0)
        {
            rv_fail(vm, "bad static_for");
            return;
        }
        for(long iv = lo; iv < hi && !vm->failed; iv += step)
        {
            int mark = vm->n_ivars;
            rv_ivar_push(vm, var, iv);
            rv_exec_list(vm, body);
            vm->n_ivars = mark; /* pop loop var */
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
        rocke_value_t* cond = rv_reg_get(vm, rocke_jstr(rocke_jget(instr, "cond")));
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
        const rocke_type_t* rtypes[16];
        const char* binds[16];
        /* Python's result_name_hint, recorded per instruction. Under exact_names
         * it is redundant (the bind is the finished name); for a rolled recipe it
         * is what makes the VM's fresh names match Python's -- "%mul14" instead
         * of "%v14" -- since both sides append the same builder counter. Absent
         * (older recipe) -> NULL -> the engine's "v", i.e. prior behavior. */
        const char* res_pfx = NULL;
        int n_res = 0;
        if(!vm->failed && out && out->kind == JD_OBJ)
        {
            rtypes[0] = rv_type(vm, rocke_jget(out, "type"));
            binds[0] = rv_bind_name(vm, out, "r");
            res_pfx = rocke_jstr(rocke_jget(out, "pfx"));
            n_res = 1;
        }
        else if(!vm->failed && outs && outs->kind == JD_ARR)
        {
            n_res = outs->arr_len > 16 ? 16 : outs->arr_len;
            for(int i = 0; i < n_res && !vm->failed; i++)
            {
                rtypes[i] = rv_type(vm, rocke_jget(outs->arr[i], "type"));
                binds[i] = rv_bind_name(vm, outs->arr[i], "r");
            }
            /* One hint per op in Python (`_op` names every result from the same
             * hint), so the first entry speaks for all of them. */
            if(n_res > 0)
                res_pfx = rocke_jstr(rocke_jget(outs->arr[0], "pfx"));
        }
        rocke_attr_map_t m;
        if(!vm->failed)
            rv_attrs(vm, rocke_jget(instr, "attrs"), &m);
        if(vm->failed)
        {
            free(ops);
            free(innames.a);
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
                return;
            }
        }
        rocke_op_t* built = rocke_b_op(
            vm->b, opcode, ops, n_ops, n_res ? rtypes : NULL, n_res, &m, NULL, 0, res_pfx, NULL);
        if(!built || !rocke_ir_builder_ok(vm->b))
        {
            rv_fail(vm,
                    "emit '%s' failed: %s",
                    opcode_name,
                    rocke_ir_builder_ok(vm->b) ? "null" : rocke_ir_builder_error(vm->b));
            free(ops);
            free(innames.a);
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

/* Expand "{NAME}" tokens in `fmt` using the int/str specs into out. */
static void rv_format_name(rvm_t* vm, const char* fmt, char* out, size_t cap)
{
    size_t n = 0;
    for(const char* p = fmt; *p && n + 1 < cap;)
    {
        if(*p == '{')
        {
            const char* close = strchr(p, '}');
            if(!close)
            {
                out[n++] = *p++;
                continue;
            }
            char key[64];
            size_t klen = (size_t)(close - p - 1);
            if(klen >= sizeof key)
                klen = sizeof key - 1;
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
            if(val)
                for(const char* q = val; *q && n + 1 < cap;)
                    out[n++] = *q++;
            p = close + 1;
        }
        else
        {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
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
    if(!rv_abi_ok(root, err, err_cap))
        return ROCKE_ERR_VALUE;

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

    /* Exact SSA naming for CONCRETE recipes only, detected by an empty "spec":
     * with no spec there is no static_for/rolled-list expansion, so every bind is
     * a unique (Python) SSA name and can be applied verbatim -> byte-identical
     * .ll. Parametric recipes (non-empty spec) unroll and reuse binds across
     * iterations, so they must keep fresh names to avoid SSA collisions. */
    const jd_val_t* spec = rocke_jget(root, "spec");
    vm.exact_names = !spec || spec->kind != JD_ARR || spec->arr_len == 0;

    /* Admission, before the builder exists and before a single op is emitted.
     * Enforcing here rather than only in the standalone check API means the
     * guard cannot be skipped by a caller who forgot it: every path that
     * replays a recipe -- online, offline, bundle, standalone -- goes through
     * this function. A refusal costs nothing and leaves nothing to free. */
    char gwhy[ROCKE_ERR_MSG_CAP];
    if(rv_guard_eval(&vm, rocke_jget(root, "guard"), 0u, gwhy, sizeof gwhy) == ROCKE_GUARD_REFUSED)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "recipe guard refused this spec: %s", gwhy);
        return ROCKE_ERR_VALUE;
    }

    char kname[256];
    const char* fmt = rocke_jstr(rocke_jget(root, "kernel_name_fmt"));
    if(!fmt)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "missing kernel_name_fmt");
        return ROCKE_ERR_VALUE;
    }
    rv_format_name(&vm, fmt, kname, sizeof kname);

    rocke_status_t st = rocke_ir_builder_init(out_builder, kname);
    if(st != ROCKE_OK)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "builder init failed (%d)", (int)st);
        return st;
    }
    vm.b = out_builder;

    /* kernel attrs (e.g. max_workgroup_size), typed like portable IR. */
    const jd_val_t* kattrs = rocke_jget(root, "attrs");
    if(kattrs && kattrs->kind == JD_OBJ)
    {
        rocke_kernel_def_t* k = rocke_ir_builder_kernel(out_builder);
        for(int i = 0; i < kattrs->obj_len; i++)
        {
            const char* key = kattrs->obj[i].key;
            const jd_val_t* tv = kattrs->obj[i].val;
            const char* t = rocke_jstr(rocke_jget(tv, "t"));
            const jd_val_t* v = rocke_jget(tv, "v");
            double d;
            if(t && v && strcmp(t, "i") == 0 && rocke_jnum(v, &d))
                rocke_attr_set_int(out_builder, &k->attrs, key, (int64_t)d);
        }
    }

    rv_exec_list(&vm, rocke_jget(root, "program"));

    free(vm.regs);
    free(vm.reg_buckets);
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

    /* The bundle carries its own level as well as each recipe's: a bundle can
     * gain container-level structure (a key index, say) independently of what
     * its recipes use, and a reader that cannot navigate the container must not
     * reach the recipes at all. rv_run_root checks the recipe's own block. */
    if(!rv_abi_ok(root, err, err_cap))
    {
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

/* =============================== LAUNCH API ==============================
 * Describe how to launch what a recipe builds: name, argument layout, grid.
 * Contract and rationale: rocke/recipe_launch.h.
 */

struct rocke_launch_plan
{
    char* kernel_name;
    bool has_geometry;
    rocke_launch_dims_t grid;
    rocke_launch_dims_t block;
    unsigned lds_bytes;
    rocke_arg_desc_t* args;
    int n_args;
    unsigned kernarg_size;
};

/* Classify a kernel parameter for kernarg packing.
 *
 * Refuses anything it cannot describe EXACTLY rather than guessing a width.
 * A wrong size here does not fail, it silently shifts every following argument
 * and the kernel reads garbage -- so an unsupported parameter has to stop the
 * plan, not degrade it. This mirrors runtime/packing.py, which raises on the
 * same set. */
static bool rv_arg_classify(const rocke_type_t* t, rocke_arg_kind_t* kind, unsigned* size)
{
    if(!t)
        return false;
    if(t->kind == ROCKE_TYPE_PTR)
    {
        *kind = ROCKE_ARG_POINTER;
        *size = 8;
        return true;
    }
    if(t->kind != ROCKE_TYPE_SCALAR)
        return false;
    switch(t->scalar)
    {
    case ROCKE_SCALAR_I32:
        *kind = ROCKE_ARG_I32, *size = 4;
        return true;
    case ROCKE_SCALAR_I64:
        *kind = ROCKE_ARG_I64, *size = 8;
        return true;
    case ROCKE_SCALAR_F32:
        *kind = ROCKE_ARG_F32, *size = 4;
        return true;
    default:
        return false;
    }
}

static char* rv_strdup(const char* s)
{
    if(!s)
        s = "";
    size_t n = strlen(s) + 1;
    char* p = (char*)malloc(n);
    if(p)
        memcpy(p, s, n);
    return p;
}

/* Read the optional "launch" block. Absent leaves has_geometry false; a block
 * that is present but malformed is an error, because a caller that asked for a
 * grid and silently got none would launch nothing. */
static bool rv_plan_geometry(
    rvm_t* vm, const jd_val_t* root, rocke_launch_plan_t* plan, char* err, size_t cap)
{
    const jd_val_t* L = rocke_jget(root, "launch");
    if(!L)
        return true;
    if(L->kind != JD_OBJ)
    {
        if(err && cap)
            ROCKE_ERR_SNPRINTF(err, cap, "recipe 'launch' is not an object");
        return false;
    }
    struct
    {
        const char* key;
        rocke_launch_dims_t* out;
    } dims[2] = {{"grid", &plan->grid}, {"block", &plan->block}};
    for(int d = 0; d < 2; d++)
    {
        const jd_val_t* a = rocke_jget(L, dims[d].key);
        if(!a || a->kind != JD_ARR || a->arr_len != 3)
        {
            if(err && cap)
                ROCKE_ERR_SNPRINTF(err, cap, "recipe launch.%s must be 3 intexprs", dims[d].key);
            return false;
        }
        long v[3];
        for(int i = 0; i < 3; i++)
        {
            v[i] = rv_int(vm, a->arr[i]);
            if(vm->failed)
            {
                if(err && cap)
                    ROCKE_ERR_SNPRINTF(err, cap, "recipe launch.%s: %s", dims[d].key, vm->err);
                return false;
            }
            /* A non-positive extent means the geometry expression disagrees
             * with the shape it was handed. Launching it would be a no-op or a
             * HIP error far from the cause, so it is named here instead. */
            if(v[i] < 1)
            {
                if(err && cap)
                    ROCKE_ERR_SNPRINTF(err,
                                       cap,
                                       "recipe launch.%s[%d] evaluates to %ld, must be >= 1",
                                       dims[d].key,
                                       i,
                                       v[i]);
                return false;
            }
        }
        dims[d].out->x = (unsigned)v[0];
        dims[d].out->y = (unsigned)v[1];
        dims[d].out->z = (unsigned)v[2];
    }
    const jd_val_t* lds = rocke_jget(L, "lds_bytes");
    if(lds)
    {
        long v = rv_int(vm, lds);
        if(vm->failed || v < 0)
        {
            if(err && cap)
                ROCKE_ERR_SNPRINTF(
                    err, cap, "recipe launch.lds_bytes: %s", vm->failed ? vm->err : "negative");
            return false;
        }
        plan->lds_bytes = (unsigned)v;
    }
    plan->has_geometry = true;
    return true;
}

static rocke_status_t rv_plan_on(jd_val_t* root,
                                 const rocke_recipe_spec_int_t* ints,
                                 int n_ints,
                                 const rocke_recipe_spec_str_t* strs,
                                 int n_strs,
                                 rocke_launch_plan_t** out_plan,
                                 char* err,
                                 size_t err_cap)
{
    *out_plan = NULL;

    /* The signature is whatever the recipe's `param` instructions declared, so
     * it is read off the built kernel rather than re-derived from the DOM.
     * Walking the program for `param` would be cheaper but would have to assume
     * they are all top-level and unconditional; replaying makes no assumption
     * and cannot disagree with the kernel that gets compiled. This also gets
     * guard enforcement for free, since rv_run_root applies it. */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;
    rocke_status_t st = rv_run_root(root, ints, n_ints, strs, n_strs, &b, &kernel, err, err_cap);
    if(st != ROCKE_OK)
        return st;

    rocke_launch_plan_t* plan = (rocke_launch_plan_t*)calloc(1, sizeof *plan);
    if(!plan)
    {
        rocke_ir_builder_free(&b);
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "out of memory");
        return ROCKE_ERR_OOM;
    }

    plan->kernel_name = rv_strdup(kernel ? kernel->name : "");
    int n = kernel ? kernel->num_params : 0;
    if(n > 0)
    {
        plan->args = (rocke_arg_desc_t*)calloc((size_t)n, sizeof *plan->args);
        if(!plan->args)
        {
            rocke_ir_builder_free(&b);
            rocke_launch_plan_free(plan);
            if(err && err_cap)
                ROCKE_ERR_SNPRINTF(err, err_cap, "out of memory");
            return ROCKE_ERR_OOM;
        }
    }
    /* Only now, so the failure path above cannot walk an array that was never
     * allocated while freeing a plan that claims to have n of them. */
    plan->n_args = n;

    /* Natural alignment: each argument at an offset aligned to its own size.
     * See the header -- packing back to back is correct only until a signature
     * mixes widths, and then it is wrong everywhere after the first mix. */
    unsigned off = 0;
    for(int i = 0; i < n; i++)
    {
        const rocke_param_t* p = kernel->params[i];
        rocke_arg_kind_t kind;
        unsigned size;
        if(!rv_arg_classify(p ? p->type : NULL, &kind, &size))
        {
            if(err && err_cap)
                ROCKE_ERR_SNPRINTF(err,
                                   err_cap,
                                   "kernel arg '%s' has type '%s', which has no kernarg "
                                   "representation here",
                                   p && p->name ? p->name : "?",
                                   p && p->type && p->type->name ? p->type->name : "?");
            rocke_ir_builder_free(&b);
            rocke_launch_plan_free(plan);
            return ROCKE_ERR_VALUE;
        }
        off = (off + size - 1) / size * size;
        /* Names are borrowed from the builder's arena, which is freed below. */
        plan->args[i].name = rv_strdup(p->name);
        plan->args[i].type_name = rv_strdup(p->type->name);
        plan->args[i].kind = kind;
        plan->args[i].size = size;
        plan->args[i].offset = off;
        off += size;
    }
    /* End of the last argument, deliberately NOT rounded up -- see the header. */
    plan->kernarg_size = off;

    rvm_t vm;
    memset(&vm, 0, sizeof vm);
    vm.ints = ints;
    vm.n_ints = n_ints;
    vm.strs = strs;
    vm.n_strs = n_strs;
    if(!rv_plan_geometry(&vm, root, plan, err, err_cap))
    {
        rocke_ir_builder_free(&b);
        rocke_launch_plan_free(plan);
        return ROCKE_ERR_VALUE;
    }

    rocke_ir_builder_free(&b);
    *out_plan = plan;
    return ROCKE_OK;
}

rocke_status_t rocke_recipe_plan_launch_cbor(const unsigned char* data,
                                             size_t len,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             rocke_launch_plan_t** out_plan,
                                             char* err,
                                             size_t err_cap)
{
    if(!data || !out_plan)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "null data/out_plan");
        return ROCKE_ERR_VALUE;
    }
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }
    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }
    rocke_status_t st = rv_plan_on(root, ints, n_ints, strs, n_strs, out_plan, err, err_cap);
    rocke_arena_destroy(&arena);
    return st;
}

rocke_status_t rocke_bundle_plan_launch_cbor(const unsigned char* data,
                                             size_t len,
                                             const char* key,
                                             const char* arch,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             rocke_launch_plan_t** out_plan,
                                             char* err,
                                             size_t err_cap)
{
    if(!data || !key || !out_plan)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "null data/key/out_plan");
        return ROCKE_ERR_VALUE;
    }
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }
    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(!root)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "parse: %s", perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }
    if(!rv_abi_ok(root, err, err_cap))
    {
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }
    const char* schema = rocke_jstr(rocke_jget(root, "schema"));
    if(!schema || strcmp(schema, "rocke.bundle/v1") != 0)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(err, err_cap, "bad/missing schema (want rocke.bundle/v1)");
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }
    const jd_val_t* recipe = rv_bundle_find(root, key, arch);
    if(!recipe)
    {
        if(err && err_cap)
            ROCKE_ERR_SNPRINTF(
                err, err_cap, "recipe '%s' (arch %s) not in bundle", key, arch ? arch : "*");
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_KEY;
    }
    rocke_status_t st
        = rv_plan_on((jd_val_t*)recipe, ints, n_ints, strs, n_strs, out_plan, err, err_cap);
    rocke_arena_destroy(&arena);
    return st;
}

const char* rocke_launch_plan_kernel_name(const rocke_launch_plan_t* plan)
{
    return plan ? plan->kernel_name : NULL;
}

bool rocke_launch_plan_geometry(const rocke_launch_plan_t* plan,
                                rocke_launch_dims_t* out_grid,
                                rocke_launch_dims_t* out_block,
                                unsigned* out_lds_bytes)
{
    if(!plan || !plan->has_geometry)
        return false;
    if(out_grid)
        *out_grid = plan->grid;
    if(out_block)
        *out_block = plan->block;
    if(out_lds_bytes)
        *out_lds_bytes = plan->lds_bytes;
    return true;
}

int rocke_launch_plan_num_args(const rocke_launch_plan_t* plan)
{
    return plan ? plan->n_args : 0;
}

const rocke_arg_desc_t* rocke_launch_plan_arg(const rocke_launch_plan_t* plan, int i)
{
    if(!plan || i < 0 || i >= plan->n_args)
        return NULL;
    return &plan->args[i];
}

unsigned rocke_launch_plan_kernarg_size(const rocke_launch_plan_t* plan)
{
    return plan ? plan->kernarg_size : 0u;
}

void rocke_launch_plan_free(rocke_launch_plan_t* plan)
{
    if(!plan)
        return;
    for(int i = 0; plan->args && i < plan->n_args; i++)
    {
        free((void*)plan->args[i].name);
        free((void*)plan->args[i].type_name);
    }
    free(plan->args);
    free(plan->kernel_name);
    free(plan);
}

/* ================================ GUARD API ==============================
 * The enforcement surface for a JIT caller (hipDNN): answer "does this recipe
 * serve this shape" with a CBOR parse and a few integer comparisons, and
 * without building any IR. Contract and usage: rocke/recipe_guard.h.
 */

/* Shared tail of both public checks: bind the spec, evaluate, and report. */
static rocke_status_t rv_check_guard_on(const jd_val_t* recipe,
                                        const rocke_recipe_spec_int_t* ints,
                                        int n_ints,
                                        const rocke_recipe_spec_str_t* strs,
                                        int n_strs,
                                        unsigned flags,
                                        rocke_guard_verdict_t* out_verdict,
                                        char* reason,
                                        size_t reason_cap)
{
    /* An artifact this engine cannot read is an ERROR, not a guard refusal. A
     * caller has to be able to tell "this build is too old for this bundle",
     * fixed by shipping matched artifacts, from "the kernel does not support
     * this shape", fixed by routing elsewhere. Answering REFUSED here would
     * send a deployment problem quietly down the fallback path, where it looks
     * like an unsupported shape and nobody investigates. */
    if(!rv_abi_ok(recipe, reason, reason_cap))
        return ROCKE_ERR_VALUE;

    rvm_t vm;
    memset(&vm, 0, sizeof vm);
    vm.ints = ints;
    vm.n_ints = n_ints;
    vm.strs = strs;
    vm.n_strs = n_strs;

    char why[ROCKE_ERR_MSG_CAP];
    rocke_guard_verdict_t v
        = rv_guard_eval(&vm, rocke_jget(recipe, "guard"), flags, why, sizeof why);
    if(reason && reason_cap)
        ROCKE_ERR_SNPRINTF(reason, reason_cap, "%s", why);
    if(out_verdict)
    {
        *out_verdict = v;
        return ROCKE_OK;
    }
    /* No out-param: the caller wants pass/fail, so a refusal has to arrive as a
     * status or it would read as success. */
    return v == ROCKE_GUARD_REFUSED ? ROCKE_ERR_VALUE : ROCKE_OK;
}

rocke_status_t rocke_recipe_check_guard_cbor(const unsigned char* data,
                                             size_t len,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             unsigned flags,
                                             rocke_guard_verdict_t* out_verdict,
                                             char* reason,
                                             size_t reason_cap)
{
    if(out_verdict)
        *out_verdict = ROCKE_GUARD_REFUSED;
    if(!data)
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(reason, reason_cap, "null recipe data");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(reason, reason_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(!root || !rocke_jstr(rocke_jget(root, "schema")))
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(reason, reason_cap, "parse: %s", root ? "missing schema" : perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    rocke_status_t st = rv_check_guard_on(
        root, ints, n_ints, strs, n_strs, flags, out_verdict, reason, reason_cap);
    rocke_arena_destroy(&arena);
    return st;
}

rocke_status_t rocke_bundle_check_guard_cbor(const unsigned char* data,
                                             size_t len,
                                             const char* key,
                                             const char* arch,
                                             const rocke_recipe_spec_int_t* ints,
                                             int n_ints,
                                             const rocke_recipe_spec_str_t* strs,
                                             int n_strs,
                                             unsigned flags,
                                             rocke_guard_verdict_t* out_verdict,
                                             char* reason,
                                             size_t reason_cap)
{
    if(out_verdict)
        *out_verdict = ROCKE_GUARD_REFUSED;
    if(!data || !key)
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(reason, reason_cap, "null bundle data/key");
        return ROCKE_ERR_VALUE;
    }

    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(reason, reason_cap, "arena init failed");
        return ROCKE_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    if(root && !rv_abi_ok(root, reason, reason_cap))
    {
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    const char* schema = root ? rocke_jstr(rocke_jget(root, "schema")) : NULL;
    if(!root || !schema || strcmp(schema, "rocke.bundle/v1") != 0)
    {
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(
                reason, reason_cap, "parse: %s", root ? "bad/missing bundle schema" : perr);
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_VALUE;
    }

    const jd_val_t* recipe = rv_bundle_find(root, key, arch);
    if(!recipe)
    {
        /* Distinct from a refusal on purpose. In a pruned bundle absence IS the
         * rejection for concrete recipes, so a caller can treat ERR_KEY and
         * REFUSED the same way -- route elsewhere -- while still being able to
         * tell "we never built this" from "we built it and it will not serve
         * this shape". */
        if(reason && reason_cap)
            ROCKE_ERR_SNPRINTF(
                reason, reason_cap, "recipe '%s' (arch %s) not in bundle", key, arch ? arch : "*");
        rocke_arena_destroy(&arena);
        return ROCKE_ERR_KEY;
    }

    rocke_status_t st = rv_check_guard_on(
        recipe, ints, n_ints, strs, n_strs, flags, out_verdict, reason, reason_cap);
    rocke_arena_destroy(&arena);
    return st;
}

bool rocke_bundle_contains(const unsigned char* data, size_t len, const char* key, const char* arch)
{
    if(!data || !key)
        return false;
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
        return false;
    char perr[256];
    jd_val_t* root = rocke_cbor_parse(data, len, &arena, perr, sizeof perr);
    bool found = root && rv_bundle_find(root, key, arch) != NULL;
    rocke_arena_destroy(&arena);
    return found;
}
