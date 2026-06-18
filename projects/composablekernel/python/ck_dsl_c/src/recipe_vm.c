/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * recipe_vm.c -- the "builder recipe" VM (schema "ck.dsl.recipe/v1"). See
 * ckc/recipe_vm.h.
 *
 * The recipe is a small program with three environments:
 *   - spec inputs (ints/strings) supplied at JIT time;
 *   - VM integer registers (loop induction vars + spec-derived integers);
 *   - IR-value registers (ckc_value_t* produced by emit/const/param ops).
 *
 * Compile-time control flow (`static_for`) lets one recipe expand into a
 * shape-specialized kernel at runtime: e.g. a `static_for` whose bound is the
 * spec value `D` unrolls D times in C, exactly as the Python author's
 * Python-time unroll would have, but driven by the runtime spec.
 *
 * Instruction set (each is a JSON object with "op"):
 *   param        {name,type,bind?,attrs?}        -> ckc_b_param
 *   const_i32    {bind,val:<intexpr>}            -> ckc_b_const_i32
 *   const_f32    {bind,fval:<number>}            -> ckc_b_const_f32
 *   thread_id_x  {bind}                          -> ckc_b_thread_id_x
 *   emit         {opcode,in:[reg],out?:{bind,type},attrs?} -> ckc_b_op
 *   static_for   {var,lo,hi,step?,body:[instr]}  -> compile-time loop
 *   ret          {}                              -> ckc_b_ret
 *
 * <intexpr> := number | {"spec":NAME} | {"var":NAME}
 *            | {"add":[e,e]} | {"sub":[e,e]} | {"mul":[e,e]}
 */
#include "ckc/recipe_vm.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/json_dom.h"

/* ------------------------------------------------------------------ state */

typedef struct
{
    const char* name;
    ckc_value_t* val;
} rv_reg_t;

typedef struct
{
    const char* name;
    long value;
} rv_int_t;

typedef struct
{
    ckc_ir_builder_t* b;
    const ckc_recipe_spec_int_t* ints;
    int n_ints;
    const ckc_recipe_spec_str_t* strs;
    int n_strs;

    rv_reg_t* regs; /* IR-value registers */
    int n_regs, cap_regs;
    rv_int_t* ivars; /* VM integers (loop vars), a scope stack */
    int n_ivars, cap_ivars;

    char err[CKC_ERR_MSG_CAP];
    bool failed;
} rvm_t;

static void rv_fail(rvm_t* vm, const char* fmt, ...)
{
    if (vm->failed)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(vm->err, sizeof vm->err, fmt, ap);
    va_end(ap);
    vm->failed = true;
}

static bool rv_spec_int(rvm_t* vm, const char* name, long* out)
{
    for (int i = 0; i < vm->n_ints; i++)
        if (strcmp(vm->ints[i].name, name) == 0) {
            *out = vm->ints[i].value;
            return true;
        }
    return false;
}

static const char* rv_spec_str(rvm_t* vm, const char* name)
{
    for (int i = 0; i < vm->n_strs; i++)
        if (strcmp(vm->strs[i].name, name) == 0)
            return vm->strs[i].value;
    return NULL;
}

static void rv_reg_set(rvm_t* vm, const char* name, ckc_value_t* val)
{
    for (int i = 0; i < vm->n_regs; i++)
        if (strcmp(vm->regs[i].name, name) == 0) {
            vm->regs[i].val = val;
            return;
        }
    if (vm->n_regs == vm->cap_regs) {
        int nc = vm->cap_regs ? vm->cap_regs * 2 : 16;
        rv_reg_t* nr = (rv_reg_t*)realloc(vm->regs, (size_t)nc * sizeof(rv_reg_t));
        if (!nr) {
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

static ckc_value_t* rv_reg_get(rvm_t* vm, const char* name)
{
    for (int i = vm->n_regs - 1; i >= 0; i--)
        if (strcmp(vm->regs[i].name, name) == 0)
            return vm->regs[i].val;
    return NULL;
}

static bool rv_ivar_get(rvm_t* vm, const char* name, long* out)
{
    for (int i = vm->n_ivars - 1; i >= 0; i--)
        if (strcmp(vm->ivars[i].name, name) == 0) {
            *out = vm->ivars[i].value;
            return true;
        }
    return false;
}

static void rv_ivar_push(rvm_t* vm, const char* name, long value)
{
    if (vm->n_ivars == vm->cap_ivars) {
        int nc = vm->cap_ivars ? vm->cap_ivars * 2 : 16;
        rv_int_t* ni = (rv_int_t*)realloc(vm->ivars, (size_t)nc * sizeof(rv_int_t));
        if (!ni) {
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
    if (vm->failed || !e)
        return 0;
    double n;
    if (ckc_jnum(e, &n))
        return (long)n;
    if (e->kind == JD_OBJ) {
        const jd_val_t* s = ckc_jget(e, "spec");
        if (s) {
            long v;
            if (!rv_spec_int(vm, ckc_jstr(s), &v))
                rv_fail(vm, "unknown spec int '%s'", ckc_jstr(s) ? ckc_jstr(s) : "?");
            return v;
        }
        const jd_val_t* var = ckc_jget(e, "var");
        if (var) {
            long v;
            if (!rv_ivar_get(vm, ckc_jstr(var), &v))
                rv_fail(vm, "unknown loop var '%s'", ckc_jstr(var) ? ckc_jstr(var) : "?");
            return v;
        }
        /* spec_str_eq: ["specname","literal"] -> 1 if the spec string matches. */
        const jd_val_t* sse = ckc_jget(e, "spec_str_eq");
        if (sse && sse->kind == JD_ARR && sse->arr_len == 2) {
            const char* sv = rv_spec_str(vm, ckc_jstr(sse->arr[0]));
            const char* lit = ckc_jstr(sse->arr[1]);
            return (sv && lit && strcmp(sv, lit) == 0) ? 1 : 0;
        }
        /* Binary arithmetic + comparisons: {"<op>":[e,e]}. */
        static const char* ops[] = {"add", "sub", "mul", "div", "mod",
                                     "eq",  "ne",  "lt",  "le",  "gt", "ge"};
        for (int k = 0; k < (int)(sizeof ops / sizeof ops[0]); k++) {
            const jd_val_t* bin = ckc_jget(e, ops[k]);
            if (!bin)
                continue;
            if (bin->kind != JD_ARR || bin->arr_len != 2) {
                rv_fail(vm, "bad intexpr '%s'", ops[k]);
                return 0;
            }
            long a = rv_int(vm, bin->arr[0]);
            long b = rv_int(vm, bin->arr[1]);
            switch (k) {
                case 0: return a + b;
                case 1: return a - b;
                case 2: return a * b;
                case 3: return b ? a / b : 0;
                case 4: return b ? a % b : 0;
                case 5: return a == b;
                case 6: return a != b;
                case 7: return a < b;
                case 8: return a <= b;
                case 9: return a > b;
                case 10: return a >= b;
            }
        }
    }
    rv_fail(vm, "bad intexpr");
    return 0;
}

/* -------------------------------------------------------------- type parse */

static const ckc_type_t* rv_type(rvm_t* vm, const jd_val_t* t)
{
    if (!t) {
        rv_fail(vm, "missing type");
        return NULL;
    }
    if (t->kind == JD_STR) {
        const ckc_type_t* st = ckc_scalar_by_name(t->str);
        if (!st)
            rv_fail(vm, "unknown scalar '%s'", t->str);
        return st;
    }
    if (t->kind == JD_OBJ) {
        const char* kind = ckc_jstr(ckc_jget(t, "kind"));
        if (kind && strcmp(kind, "ptr") == 0) {
            const ckc_type_t* pointee = rv_type(vm, ckc_jget(t, "pointee"));
            const char* space = ckc_jstr(ckc_jget(t, "space"));
            if (!pointee || !space)
                return vm->failed ? NULL : (rv_fail(vm, "bad ptr type"), NULL);
            return ckc_ptr_type(vm->b, pointee, space);
        }
        if (kind && strcmp(kind, "vector") == 0) {
            const ckc_type_t* elem = rv_type(vm, ckc_jget(t, "elem"));
            double cnt;
            if (!elem || !ckc_jnum(ckc_jget(t, "count"), &cnt))
                return vm->failed ? NULL : (rv_fail(vm, "bad vector type"), NULL);
            return ckc_vector_type(vm->b, elem, (int)cnt);
        }
        rv_fail(vm, "unsupported type kind '%s'", kind ? kind : "?");
        return NULL;
    }
    rv_fail(vm, "bad type node");
    return NULL;
}

/* ------------------------------------------------------------- attr build */

static void rv_attrs(rvm_t* vm, const jd_val_t* attrs, ckc_attr_map_t* m)
{
    ckc_attr_map_init(m);
    if (!attrs || attrs->kind != JD_OBJ)
        return;
    for (int i = 0; i < attrs->obj_len && !vm->failed; i++) {
        const char* key = attrs->obj[i].key;
        const jd_val_t* tv = attrs->obj[i].val;
        const char* t = ckc_jstr(ckc_jget(tv, "t"));
        const jd_val_t* v = ckc_jget(tv, "v");
        if (!t || !v) {
            rv_fail(vm, "attr '%s' missing t/v", key);
            return;
        }
        if (strcmp(t, "i") == 0)
            ckc_attr_set_int(vm->b, m, key, rv_int(vm, v)); /* v may be intexpr */
        else if (strcmp(t, "f") == 0) {
            double d = 0;
            ckc_jnum(v, &d);
            ckc_attr_set_float(vm->b, m, key, d);
        } else if (strcmp(t, "b") == 0)
            ckc_attr_set_bool(vm->b, m, key, v->b);
        else if (strcmp(t, "s") == 0)
            ckc_attr_set_str(vm->b, m, key, v->str ? v->str : "");
        else
            rv_fail(vm, "attr '%s' bad kind '%s'", key, t);
    }
}

/* ---------------------------------------------------------------- execute */

static void rv_exec_list(rvm_t* vm, const jd_val_t* program);

static const char* rv_bind_name(const jd_val_t* instr, const char* fallback)
{
    const char* b = ckc_jstr(ckc_jget(instr, "bind"));
    return b ? b : fallback;
}

static void rv_exec_instr(rvm_t* vm, const jd_val_t* instr)
{
    if (vm->failed)
        return;
    const char* op = ckc_jstr(ckc_jget(instr, "op"));
    if (!op) {
        rv_fail(vm, "instr missing op");
        return;
    }

    if (strcmp(op, "param") == 0) {
        const char* name = ckc_jstr(ckc_jget(instr, "name"));
        const ckc_type_t* type = rv_type(vm, ckc_jget(instr, "type"));
        if (!name || !type) {
            rv_fail(vm, "bad param");
            return;
        }
        ckc_param_opts_t opts;
        memset(&opts, 0, sizeof opts);
        const jd_val_t* pa = ckc_jget(instr, "attrs");
        if (pa && pa->kind == JD_OBJ) {
            for (int k = 0; k < pa->obj_len; k++) {
                const char* key = pa->obj[k].key;
                const jd_val_t* v = pa->obj[k].val;
                double d;
                if (strcmp(key, "noalias") == 0) {
                    opts.noalias = v->b;
                    opts.noalias_set = true;
                } else if (strcmp(key, "readonly") == 0) {
                    opts.readonly = v->b;
                    opts.readonly_set = true;
                } else if (strcmp(key, "writeonly") == 0) {
                    opts.writeonly = v->b;
                    opts.writeonly_set = true;
                } else if (strcmp(key, "align") == 0 && ckc_jnum(v, &d)) {
                    opts.align = (int)d;
                    opts.align_set = true;
                } else if (strcmp(key, "addr_space") == 0) {
                    opts.addr_space = v->str;
                }
            }
        }
        ckc_value_t* pv = ckc_b_param(vm->b, name, type, &opts);
        if (!pv) {
            rv_fail(vm, "param '%s' failed", name);
            return;
        }
        rv_reg_set(vm, rv_bind_name(instr, name), pv);
        return;
    }
    if (strcmp(op, "const_i32") == 0) {
        ckc_value_t* v = ckc_b_const_i32(vm->b, rv_int(vm, ckc_jget(instr, "val")));
        rv_reg_set(vm, rv_bind_name(instr, "c"), v);
        return;
    }
    if (strcmp(op, "const_f32") == 0) {
        double d = 0;
        ckc_jnum(ckc_jget(instr, "fval"), &d);
        rv_reg_set(vm, rv_bind_name(instr, "c"), ckc_b_const_f32(vm->b, d));
        return;
    }
    if (strcmp(op, "thread_id_x") == 0) {
        rv_reg_set(vm, rv_bind_name(instr, "tid"), ckc_b_thread_id_x(vm->b));
        return;
    }
    if (strcmp(op, "alias") == 0) {
        /* Rebind a register to an existing value (e.g. an else-arm result). */
        const char* from = ckc_jstr(ckc_jget(instr, "from"));
        ckc_value_t* v = from ? rv_reg_get(vm, from) : NULL;
        if (!v) {
            rv_fail(vm, "alias unresolved '%s'", from ? from : "?");
            return;
        }
        rv_reg_set(vm, rv_bind_name(instr, "r"), v);
        return;
    }
    if (strcmp(op, "ret") == 0) {
        ckc_b_ret(vm->b);
        return;
    }
    if (strcmp(op, "static_for") == 0) {
        const char* var = ckc_jstr(ckc_jget(instr, "var"));
        long lo = rv_int(vm, ckc_jget(instr, "lo"));
        long hi = rv_int(vm, ckc_jget(instr, "hi"));
        const jd_val_t* stepn = ckc_jget(instr, "step");
        long step = stepn ? rv_int(vm, stepn) : 1;
        const jd_val_t* body = ckc_jget(instr, "body");
        if (!var || !body || body->kind != JD_ARR || step == 0) {
            rv_fail(vm, "bad static_for");
            return;
        }
        for (long iv = lo; iv < hi && !vm->failed; iv += step) {
            int mark = vm->n_ivars;
            rv_ivar_push(vm, var, iv);
            rv_exec_list(vm, body);
            vm->n_ivars = mark; /* pop loop var */
        }
        return;
    }
    if (strcmp(op, "static_if") == 0) {
        /* Compile-time branch on a spec predicate (truthy intexpr). */
        long pred = rv_int(vm, ckc_jget(instr, "pred"));
        if (vm->failed)
            return;
        const jd_val_t* arm = ckc_jget(instr, pred ? "then" : "else");
        if (arm)
            rv_exec_list(vm, arm);
        return;
    }
    if (strcmp(op, "scf_for") == 0) {
        /* Runtime loop emitted as a real scf.for op (bounds are IR values). */
        ckc_value_t* lo = rv_reg_get(vm, ckc_jstr(ckc_jget(instr, "lo")));
        ckc_value_t* hi = rv_reg_get(vm, ckc_jstr(ckc_jget(instr, "hi")));
        ckc_value_t* step = rv_reg_get(vm, ckc_jstr(ckc_jget(instr, "step")));
        const char* iv = ckc_jstr(ckc_jget(instr, "iv"));
        if (!lo || !hi || !step || !iv) {
            rv_fail(vm, "scf_for needs lo/hi/step/iv");
            return;
        }
        const jd_val_t* iter = ckc_jget(instr, "iter");
        int n_iter = (iter && iter->kind == JD_ARR) ? iter->arr_len : 0;
        ckc_iter_arg_t ia[16];
        const char* iter_regs[16];
        if (n_iter > 16) {
            rv_fail(vm, "too many iter_args");
            return;
        }
        for (int i = 0; i < n_iter; i++) {
            const char* nm = ckc_jstr(ckc_jget(iter->arr[i], "name"));
            ia[i].name = nm;
            ia[i].init = rv_reg_get(vm, ckc_jstr(ckc_jget(iter->arr[i], "init")));
            iter_regs[i] = nm;
            if (!nm || !ia[i].init) {
                rv_fail(vm, "bad scf_for iter arg");
                return;
            }
        }
        const jd_val_t* un = ckc_jget(instr, "unroll");
        const jd_val_t* el = ckc_jget(instr, "elide_trailing_barrier");
        ckc_for_t f = ckc_b_scf_for_iter(vm->b, lo, hi, step, ia, n_iter, iv,
                                         un ? un->b : false, el ? el->b : true);
        if (!ckc_ir_builder_ok(vm->b)) {
            rv_fail(vm, "scf_for build: %s", ckc_ir_builder_error(vm->b));
            return;
        }
        rv_reg_set(vm, iv, f.iv);
        for (int i = 0; i < n_iter; i++)
            rv_reg_set(vm, iter_regs[i], f.iter_vars[i]);
        ckc_b_region_enter(vm->b, f.body);
        rv_exec_list(vm, ckc_jget(instr, "body"));
        ckc_b_region_leave(vm->b);
        if (vm->failed)
            return;
        const jd_val_t* results = ckc_jget(instr, "results");
        if (results && results->kind == JD_ARR)
            for (int i = 0; i < results->arr_len && i < f.op->num_results; i++)
                rv_reg_set(vm, ckc_jstr(results->arr[i]), f.op->results[i]);
        return;
    }
    if (strcmp(op, "scf_if") == 0) {
        ckc_value_t* cond = rv_reg_get(vm, ckc_jstr(ckc_jget(instr, "cond")));
        if (!cond) {
            rv_fail(vm, "scf_if needs cond");
            return;
        }
        ckc_if_t s = ckc_b_scf_if(vm->b, cond);
        const jd_val_t* then = ckc_jget(instr, "then");
        if (then) {
            ckc_b_region_enter(vm->b, s.then_region);
            rv_exec_list(vm, then);
            ckc_b_region_leave(vm->b);
        }
        return;
    }
    if (strcmp(op, "emit") == 0) {
        const char* opcode_name = ckc_jstr(ckc_jget(instr, "opcode"));
        ckc_opcode_t opcode = opcode_name ? ckc_opcode_from_name(opcode_name) : CKC_OP_INVALID;
        if (opcode == CKC_OP_INVALID) {
            rv_fail(vm, "unknown opcode '%s'", opcode_name ? opcode_name : "?");
            return;
        }
        const jd_val_t* in = ckc_jget(instr, "in");
        int n_ops = (in && in->kind == JD_ARR) ? in->arr_len : 0;
        ckc_value_t* ops[16];
        if (n_ops > 16) {
            rv_fail(vm, "too many operands");
            return;
        }
        for (int i = 0; i < n_ops; i++) {
            const char* rn = ckc_jstr(in->arr[i]);
            ops[i] = rn ? rv_reg_get(vm, rn) : NULL;
            if (!ops[i]) {
                rv_fail(vm, "emit '%s' unresolved operand '%s'", opcode_name, rn ? rn : "?");
                return;
            }
        }
        const jd_val_t* out = ckc_jget(instr, "out");
        const ckc_type_t* rtype = NULL;
        if (out && out->kind == JD_OBJ)
            rtype = rv_type(vm, ckc_jget(out, "type"));
        if (vm->failed)
            return;
        ckc_attr_map_t m;
        rv_attrs(vm, ckc_jget(instr, "attrs"), &m);
        if (vm->failed)
            return;
        const ckc_type_t* rtypes[1];
        int n_res = 0;
        if (rtype) {
            rtypes[0] = rtype;
            n_res = 1;
        }
        ckc_op_t* built =
            ckc_b_op(vm->b, opcode, ops, n_ops, n_res ? rtypes : NULL, n_res, &m, NULL, 0, NULL, NULL);
        if (!built || !ckc_ir_builder_ok(vm->b)) {
            rv_fail(vm, "emit '%s' failed: %s", opcode_name,
                    ckc_ir_builder_ok(vm->b) ? "null" : ckc_ir_builder_error(vm->b));
            return;
        }
        if (out && out->kind == JD_OBJ && built->num_results > 0) {
            const char* bind = ckc_jstr(ckc_jget(out, "bind"));
            rv_reg_set(vm, bind ? bind : "r", built->results[0]);
        }
        return;
    }
    rv_fail(vm, "unknown instr op '%s'", op);
}

static void rv_exec_list(rvm_t* vm, const jd_val_t* program)
{
    if (!program || program->kind != JD_ARR) {
        rv_fail(vm, "program/body not an array");
        return;
    }
    for (int i = 0; i < program->arr_len && !vm->failed; i++)
        rv_exec_instr(vm, program->arr[i]);
}

/* ----------------------------------------------------- kernel name format */

/* Expand "{NAME}" tokens in `fmt` using the int/str specs into out. */
static void rv_format_name(rvm_t* vm, const char* fmt, char* out, size_t cap)
{
    size_t n = 0;
    for (const char* p = fmt; *p && n + 1 < cap;) {
        if (*p == '{') {
            const char* close = strchr(p, '}');
            if (!close) {
                out[n++] = *p++;
                continue;
            }
            char key[64];
            size_t klen = (size_t)(close - p - 1);
            if (klen >= sizeof key)
                klen = sizeof key - 1;
            memcpy(key, p + 1, klen);
            key[klen] = '\0';
            long iv;
            const char* sv;
            char tmp[32];
            const char* val = NULL;
            if (rv_spec_int(vm, key, &iv)) {
                snprintf(tmp, sizeof tmp, "%ld", iv);
                val = tmp;
            } else if ((sv = rv_spec_str(vm, key)) != NULL) {
                val = sv;
            }
            if (val)
                for (const char* q = val; *q && n + 1 < cap;)
                    out[n++] = *q++;
            p = close + 1;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
}

ckc_status_t ckc_recipe_run_from_json(const char* text,
                                      const ckc_recipe_spec_int_t* ints,
                                      int n_ints,
                                      const ckc_recipe_spec_str_t* strs,
                                      int n_strs,
                                      ckc_ir_builder_t* out_builder,
                                      ckc_kernel_def_t** out_kernel,
                                      char* err,
                                      size_t err_cap)
{
    if (out_kernel)
        *out_kernel = NULL;
    if (!text || !out_builder) {
        if (err && err_cap)
            snprintf(err, err_cap, "null text/builder");
        return CKC_ERR_VALUE;
    }

    ckc_arena_t arena;
    if (ckc_arena_init(&arena, 0) != 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "arena init failed");
        return CKC_ERR_OOM;
    }

    char perr[256];
    jd_val_t* root = ckc_json_parse(text, &arena, perr, sizeof perr);
    if (!root) {
        if (err && err_cap)
            snprintf(err, err_cap, "parse: %s", perr);
        ckc_arena_destroy(&arena);
        return CKC_ERR_VALUE;
    }

    const char* schema = ckc_jstr(ckc_jget(root, "schema"));
    if (!schema || strcmp(schema, "ck.dsl.recipe/v1") != 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "bad/missing schema (want ck.dsl.recipe/v1)");
        ckc_arena_destroy(&arena);
        return CKC_ERR_VALUE;
    }

    rvm_t vm;
    memset(&vm, 0, sizeof vm);
    vm.ints = ints;
    vm.n_ints = n_ints;
    vm.strs = strs;
    vm.n_strs = n_strs;

    char kname[256];
    const char* fmt = ckc_jstr(ckc_jget(root, "kernel_name_fmt"));
    if (!fmt) {
        ckc_arena_destroy(&arena);
        if (err && err_cap)
            snprintf(err, err_cap, "missing kernel_name_fmt");
        return CKC_ERR_VALUE;
    }
    rv_format_name(&vm, fmt, kname, sizeof kname);

    ckc_status_t st = ckc_ir_builder_init(out_builder, kname);
    if (st != CKC_OK) {
        ckc_arena_destroy(&arena);
        if (err && err_cap)
            snprintf(err, err_cap, "builder init failed (%d)", (int)st);
        return st;
    }
    vm.b = out_builder;

    /* kernel attrs (e.g. max_workgroup_size), typed like portable IR. */
    const jd_val_t* kattrs = ckc_jget(root, "attrs");
    if (kattrs && kattrs->kind == JD_OBJ) {
        ckc_kernel_def_t* k = ckc_ir_builder_kernel(out_builder);
        for (int i = 0; i < kattrs->obj_len; i++) {
            const char* key = kattrs->obj[i].key;
            const jd_val_t* tv = kattrs->obj[i].val;
            const char* t = ckc_jstr(ckc_jget(tv, "t"));
            const jd_val_t* v = ckc_jget(tv, "v");
            double d;
            if (t && v && strcmp(t, "i") == 0 && ckc_jnum(v, &d))
                ckc_attr_set_int(out_builder, &k->attrs, key, (int64_t)d);
        }
    }

    rv_exec_list(&vm, ckc_jget(root, "program"));

    free(vm.regs);
    free(vm.ivars);
    ckc_arena_destroy(&arena);

    if (vm.failed) {
        if (err && err_cap)
            snprintf(err, err_cap, "%s", vm.err);
        ckc_ir_builder_free(out_builder);
        return CKC_ERR_VALUE;
    }
    if (!ckc_ir_builder_ok(out_builder)) {
        if (err && err_cap)
            snprintf(err, err_cap, "builder error: %s", ckc_ir_builder_error(out_builder));
        ckc_ir_builder_free(out_builder);
        return ckc_ir_builder_status(out_builder);
    }

    *out_kernel = ckc_ir_builder_kernel(out_builder);
    return CKC_OK;
}
