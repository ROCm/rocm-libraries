// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/portable_ir/recipe_vm_replay.cpp -- hermetic test for the recipe VM
 * (cpp/portable_ir/recipe_vm.cpp).
 *
 * A recipe is the compact artifact the runtime actually ships: it encodes the
 * *builder algorithm* once -- including its compile-time control flow -- and
 * the VM replays it against concrete spec values at JIT time. So one recipe
 * covers a whole kernel family, and the specialization happens in C with no
 * CPython anywhere in the process.
 *
 * The recipe below is the C++ twin of
 * python/rocke/portable_ir/examples/recipe_toy.py: a D-unrolled multiply-
 * accumulate whose `static_for` bound is the spec value D. Embedding it as a
 * string literal keeps this test hermetic -- it needs no Python, no artifact
 * file, and no network, so it can run as a plain ctest anywhere the engine
 * builds.
 *
 * What is actually pinned here:
 *   - the VM runs a recipe end to end and produces a lowerable kernel;
 *   - the spec drives structure, not just naming: D=4 and D=8 expand the
 *     static_for to 4 and 8 multiply-accumulates respectively;
 *   - spec strings reach the kernel name via kernel_name_fmt;
 *   - replay is deterministic (same spec -> byte-identical .ll), which is the
 *     precondition for the cross-engine byte-identity gate in
 *     python/rocke/portable_ir/drivers/parity_matrix.py;
 *   - scf_if_else rebuilds a converging if/else (both arms, one join block),
 *     including one nested in an arm, and a missing condition is an error.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include "rocke/recipe_vm.h"

namespace
{

int g_fail = 0;

void check(bool cond, const char* msg)
{
    if(!cond)
    {
        std::printf("  FAIL: %s\n", msg);
        ++g_fail;
    }
}

// The toy recipe: acc = sum over d in [0, D) of A[tid+d] * B[tid+d], stored to
// C[tid]. `hi` of the static_for is {"spec": "D"}, so D is resolved at replay
// time rather than baked in.
const char* kToyRecipe = R"json({
  "schema": "rocke.recipe/v1",
  "kernel_name_fmt": "rocke_recipe_toy_d{D}_{dtype}",
  "spec": [{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
  "attrs": {"max_workgroup_size": {"t": "i", "v": 64}},
  "program": [
    {"op": "param", "name": "A", "bind": "A",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"},
     "attrs": {"noalias": true, "readonly": true, "align": 16}},
    {"op": "param", "name": "B", "bind": "B",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"},
     "attrs": {"noalias": true, "readonly": true, "align": 16}},
    {"op": "param", "name": "C", "bind": "C",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"},
     "attrs": {"noalias": true, "writeonly": true, "align": 16}},
    {"op": "thread_id_x", "bind": "tid"},
    {"op": "const_f32", "bind": "acc", "fval": 0.0},
    {"op": "static_for", "var": "d", "lo": 0, "hi": {"spec": "D"}, "step": 1,
     "body": [
       {"op": "const_i32", "bind": "off", "val": {"var": "d"}},
       {"op": "emit", "opcode": "arith.add", "in": ["tid", "off"],
        "out": {"bind": "i", "type": "i32"}},
       {"op": "emit", "opcode": "memref.global_load_typed", "in": ["A", "i"],
        "out": {"bind": "a", "type": "f32"},
        "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}},
       {"op": "emit", "opcode": "memref.global_load_typed", "in": ["B", "i"],
        "out": {"bind": "bb", "type": "f32"},
        "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}},
       {"op": "emit", "opcode": "arith.fmul", "in": ["a", "bb"],
        "out": {"bind": "p", "type": "f32"}},
       {"op": "emit", "opcode": "arith.fadd", "in": ["acc", "p"],
        "out": {"bind": "acc", "type": "f32"}}
     ]},
    {"op": "emit", "opcode": "memref.global_store_typed", "in": ["C", "tid", "acc"],
     "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}},
    {"op": "ret"}
  ]
})json";

// Replay the recipe at (D, dtype) and lower it. Returns false (with a printed
// diagnostic) on any failure; on success fills out_ll and out_name.
bool replay(long D, const char* dtype, std::string* out_ll, std::string* out_name)
{
    const rocke_recipe_spec_int_t ints[] = {{"D", D}};
    const rocke_recipe_spec_str_t strs[] = {{"dtype", dtype}};

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st
        = rocke_recipe_run_from_json(kToyRecipe, ints, 1, strs, 1, &b, &kernel, err, sizeof(err));
    if(st != ROCKE_OK || !kernel)
    {
        std::printf("  FAIL: recipe run at D=%ld failed (status %d): %s\n", D, (int)st, err);
        ++g_fail;
        return false;
    }

    out_name->assign(kernel->name ? kernel->name : "");

    char* ll = nullptr;
    char lerr[ROCKE_ERR_MSG_CAP];
    lerr[0] = '\0';
    // Pin the flavor rather than taking AUTO: this test asserts on structure,
    // and a flavor swing from the environment would change the datalayout and
    // the buffer intrinsic spellings underneath it.
    st = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &ll, lerr, sizeof(lerr));
    if(st != ROCKE_OK || !ll)
    {
        std::printf("  FAIL: lower at D=%ld failed (status %d): %s\n", D, (int)st, lerr);
        ++g_fail;
        rocke_ir_builder_free(&b);
        return false;
    }
    out_ll->assign(ll);
    std::free(ll);
    rocke_ir_builder_free(&b);
    return true;
}

// Count non-overlapping occurrences of `needle` in `hay`.
int count(const std::string& hay, const char* needle)
{
    int n = 0;
    const size_t step = std::strlen(needle);
    for(size_t pos = hay.find(needle); pos != std::string::npos; pos = hay.find(needle, pos + step))
    {
        ++n;
    }
    return n;
}

// A recipe with no program but `ret`, carrying one kernel attr. Just enough to
// reach the recipe VM's kernel-attr loop.
std::string attr_recipe(const std::string& key, const std::string& attr_json)
{
    return std::string(R"json({"schema": "rocke.recipe/v1", "kernel_name_fmt": "attr_probe",
  "spec": [], "attrs": {")json")
           + key + "\": " + attr_json + R"json(}, "program": [{"op": "ret"}]})json";
}

// A bare-int list as ir_export writes it: {"t": "l", "v": [{"_": {"t": "i", "v": N}}, ...]}.
std::string int_list(const std::vector<std::string>& items)
{
    std::string out = R"json({"t": "l", "v": [)json";
    for(size_t i = 0; i < items.size(); ++i)
    {
        out += std::string(i ? ", " : "") + R"json({"_": {"t": "i", "v": )json" + items[i] + "}}";
    }
    return out + "]}";
}

// Replay `recipe`; on success lower it for gfx1250 into *out_ll. Returns the
// status, and the VM's message in *out_err.
rocke_status_t run_attr_recipe(const std::string& recipe, std::string* out_ll, std::string* out_err)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_recipe_run_from_json(
        recipe.c_str(), nullptr, 0, nullptr, 0, &b, &kernel, err, sizeof(err));
    out_err->assign(err);
    if(st != ROCKE_OK || !kernel)
    {
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }
    char* ll = nullptr;
    char lerr[ROCKE_ERR_MSG_CAP];
    lerr[0] = '\0';
    st = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_LLVM23, "gfx1250", &ll, lerr, sizeof(lerr));
    if(st == ROCKE_OK && ll)
    {
        out_ll->assign(ll);
    }
    std::free(ll);
    rocke_ir_builder_free(&b);
    return st;
}

void check_invalid_integer_values()
{
    std::string ll, err;
    for(const char* spelling : {"NaN", "Infinity", "-Infinity", "9007199254740993"})
    {
        const std::string label = std::string("kernel integer attr ") + spelling;
        check(run_attr_recipe(
                  attr_recipe("bad", std::string(R"json({"t":"i","v":)json") + spelling + "}"),
                  &ll,
                  &err)
                  != ROCKE_OK,
              (label + " is rejected").c_str());
        check(err.find("not an integer") != std::string::npos,
              (label + " has a diagnostic (got: " + err + ")").c_str());

        const std::string recipe
            = std::string(
                  R"json({"schema":"rocke.recipe/v1","kernel_name_fmt":"bad_intexpr","spec":[],"attrs":{},"program":[{"op":"const_i32","bind":"bad","val":)json")
              + spelling + R"json(},{"op":"ret"}]})json";
        ll.clear();
        err.clear();
        const std::string expr_label = std::string("integer expression ") + spelling;
        check(run_attr_recipe(recipe, &ll, &err) != ROCKE_OK,
              (expr_label + " is rejected").c_str());
        check(err.find("not an integer") != std::string::npos,
              (expr_label + " has a diagnostic (got: " + err + ")").c_str());
    }
}

// A list kernel attr is either carried exactly or the replay fails -- never
// dropped, because a kernel compiled without its cluster_dims is a wrong launch.
void check_kernel_list_attrs()
{
    std::string ll, err;

    check(run_attr_recipe(attr_recipe("cluster_dims", int_list({"2", "2", "1"})), &ll, &err)
              == ROCKE_OK,
          "well-formed cluster_dims replays");
    check(ll.find("\"amdgpu-cluster-dims\"=\"2,2,1\"") != std::string::npos,
          "cluster_dims reaches the lowered kernel");

    struct bad_t
    {
        const char* what;
        std::string attr;
        const char* message;
    };
    std::vector<std::string> sixteen_plus_one(17, "1");
    const bad_t bad[] = {
        {"non-array value", R"json({"t": "l", "v": 3})json", "must be an array"},
        {"missing value", R"json({"t": "l"})json", "must be an array"},
        {"non-int item",
         R"json({"t": "l", "v": [{"_": {"t": "s", "v": "x"}}]})json",
         "item 0 is not a bare int"},
        {"item without a payload", R"json({"t": "l", "v": [{}]})json", "item 0 is not a bare int"},
        {"non-integral item", int_list({"2", "1.5"}), "item 1 is not an integer"},
        {"too many items", int_list(sixteen_plus_one), "at most 16"},
        {"wrong tag s with array value",
         R"json({"t": "s", "v": [{"_": {"t": "i", "v": 2}}, {"_": {"t": "i", "v": 1}}, {"_": {"t": "i", "v": 1}}]})json",
         "array value requires tag"},
        {"wrong tag b with array value",
         R"json({"t": "b", "v": [{"_": {"t": "i", "v": 2}}, {"_": {"t": "i", "v": 1}}, {"_": {"t": "i", "v": 1}}]})json",
         "array value requires tag"},
        {"missing tag with array value",
         R"json({"v": [{"_": {"t": "i", "v": 2}}, {"_": {"t": "i", "v": 1}}, {"_": {"t": "i", "v": 1}}]})json",
         "array value requires tag"},
    };
    for(const bad_t& c : bad)
    {
        // Same shape under both keys that take a list, so no key is exempt.
        for(const char* key : {"cluster_dims", "agpr_alloc"})
        {
            ll.clear();
            err.clear();
            const std::string what = std::string(c.what) + " under " + key;
            check(run_attr_recipe(attr_recipe(key, c.attr), &ll, &err) != ROCKE_OK,
                  (what + " is rejected").c_str());
            check(err.find(c.message) != std::string::npos,
                  (what + " names the defect (got: " + err + ")").c_str());
            check(err.find(key) != std::string::npos, (what + " names the attr").c_str());
        }
    }

    // agpr_alloc may also be a "min,max" string, so only cluster_dims insists on "l".
    const bad_t bad_cluster[] = {
        {"int tag", R"json({"t": "i", "v": 2})json", "requires tag \"l\""},
        {"string tag", R"json({"t": "s", "v": "2,2,1"})json", "requires tag \"l\""},
        {"missing tag", R"json({"v": 2})json", "requires tag \"l\""},
    };
    for(const bad_t& c : bad_cluster)
    {
        ll.clear();
        err.clear();
        const std::string what = std::string("cluster_dims with ") + c.what;
        check(run_attr_recipe(attr_recipe("cluster_dims", c.attr), &ll, &err) != ROCKE_OK,
              (what + " is rejected").c_str());
        check(err.find(c.message) != std::string::npos,
              (what + " names the defect (got: " + err + ")").c_str());
    }
}

// scf_if_else: C[tid] = tid < 32 ? A[tid] : (tid < 48 ? B[tid] : 0). The inner
// if/else sits in the outer else arm, so arm routing and nesting both show up
// in the lowered block structure. Values defined in an arm (a, bb) are used in
// that arm only, the way the Python builder scopes them.
const char* kIfElseRecipe = R"json({
  "schema": "rocke.recipe/v1",
  "kernel_name_fmt": "rocke_recipe_if_else",
  "spec": [],
  "attrs": {"max_workgroup_size": {"t": "i", "v": 64}},
  "program": [
    {"op": "param", "name": "A", "bind": "A",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"}, "attrs": {}},
    {"op": "param", "name": "B", "bind": "B",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"}, "attrs": {}},
    {"op": "param", "name": "C", "bind": "C",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"}, "attrs": {}},
    {"op": "thread_id_x", "bind": "tid"},
    {"op": "const_i32", "bind": "c32", "val": 32},
    {"op": "const_i32", "bind": "c48", "val": 48},
    {"op": "emit", "opcode": "arith.cmp", "in": ["tid", "c32"],
     "out": {"bind": "lo", "type": "i1"}, "attrs": {"pred": {"t": "s", "v": "lt"}}},
    {"op": "scf_if_else", "cond": "lo",
     "then": [
       {"op": "emit", "opcode": "memref.global_load_typed", "in": ["A", "tid"],
        "out": {"bind": "a", "type": "f32"},
        "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}},
       {"op": "emit", "opcode": "memref.global_store_typed", "in": ["C", "tid", "a"],
        "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}}
     ],
     "else": [
       {"op": "emit", "opcode": "arith.cmp", "in": ["tid", "c48"],
        "out": {"bind": "mid", "type": "i1"}, "attrs": {"pred": {"t": "s", "v": "lt"}}},
       {"op": "scf_if_else", "cond": "mid",
        "then": [
          {"op": "emit", "opcode": "memref.global_load_typed", "in": ["B", "tid"],
           "out": {"bind": "bb", "type": "f32"},
           "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}},
          {"op": "emit", "opcode": "memref.global_store_typed", "in": ["C", "tid", "bb"],
           "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}}
        ],
        "else": [
          {"op": "const_f32", "bind": "z", "fval": 0.0},
          {"op": "emit", "opcode": "memref.global_store_typed", "in": ["C", "tid", "z"],
           "attrs": {"align": {"t": "i", "v": 4}, "elem_type": {"t": "s", "v": "f32"}}}
        ]}
     ]},
    {"op": "ret"}
  ]
})json";

// Run a spec-free recipe and lower it (gfx950, llvm20). Returns the status;
// *out_err gets the VM or lowering diagnostic.
rocke_status_t run_plain_recipe(const char* recipe, std::string* out_ll, std::string* out_err)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st
        = rocke_recipe_run_from_json(recipe, nullptr, 0, nullptr, 0, &b, &kernel, err, sizeof(err));
    out_err->assign(err);
    if(st != ROCKE_OK || !kernel)
    {
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }
    char* ll = nullptr;
    char lerr[ROCKE_ERR_MSG_CAP];
    lerr[0] = '\0';
    st = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &ll, lerr, sizeof(lerr));
    if(st == ROCKE_OK && ll)
    {
        out_ll->assign(ll);
    }
    else
    {
        out_err->assign(lerr);
    }
    std::free(ll);
    rocke_ir_builder_free(&b);
    return st;
}

void check_scf_if_else()
{
    std::string ll, err;
    check(run_plain_recipe(kIfElseRecipe, &ll, &err) == ROCKE_OK,
          ("scf_if_else recipe replays and lowers (got: " + err + ")").c_str());
    // Two if/else ops -> two then/else/join block triples, every arm's store
    // survives, and each arm ends in a branch to its own join block.
    check(count(ll, "\nif.then") == 2, "two then blocks");
    check(count(ll, "\nif.else") == 2, "two else blocks");
    check(count(ll, "\nif.end") == 2, "two join blocks");
    check(count(ll, "store float") == 3, "all three arm stores are emitted");
    check(count(ll, "br i1") == 2, "one conditional branch per if/else");

    std::string again, err2;
    check(run_plain_recipe(kIfElseRecipe, &again, &err2) == ROCKE_OK && again == ll,
          "scf_if_else replay is deterministic");

    // A dangling condition is reported, not silently dropped.
    std::string bad = kIfElseRecipe;
    const std::string from = R"("cond": "lo")";
    bad.replace(bad.find(from), from.size(), R"("cond": "nope")");
    ll.clear();
    err.clear();
    check(run_plain_recipe(bad.c_str(), &ll, &err) != ROCKE_OK,
          "scf_if_else with an unresolved cond is rejected");
    check(err.find("scf_if_else") != std::string::npos,
          ("unresolved cond names the op (got: " + err + ")").c_str());
}

// A recipe whose one instruction is a pointer param carrying `attrs_json`.
std::string param_recipe(const std::string& attrs_json)
{
    return std::string(R"json({"schema": "rocke.recipe/v1", "kernel_name_fmt": "param_probe",
  "spec": [], "attrs": {}, "program": [
    {"op": "param", "name": "P", "bind": "P",
     "type": {"kind": "ptr", "pointee": "f32", "space": "global"}, "attrs": )json")
           + attrs_json + R"json(}, {"op": "ret"}]})json";
}

// Every pointer attribute the lowerer emits reaches the replayed kernel, and a
// param attr the VM cannot read fails the replay: dropping it would lower a
// different kernel than the one recorded.
void check_param_attrs()
{
    std::string ll, err;
    check(
        run_attr_recipe(
            param_recipe(R"json({"nonnull": true, "nocapture": false, "dereferenceable": 64})json"),
            &ll,
            &err)
            == ROCKE_OK,
        ("nonnull / nocapture / dereferenceable replay (got: " + err + ")").c_str());
    check(
        ll.find("ptr addrspace(1) nonnull dereferenceable(64) %P") != std::string::npos,
        "nonnull and dereferenceable reach the lowered param, and nocapture=false drops nocapture");

    struct bad_t
    {
        const char* what;
        const char* attrs;
        const char* message;
    };
    const bad_t bad[] = {
        {"unknown key",
         R"json({"el": 1048576})json",
         "param 'P' has unknown attr 'el' -- a param attr is one of " ROCKE_PARAM_ATTR_KEYS},
        {"non-bool flag", R"json({"nonnull": 1})json", "param 'P' attr nonnull is not a bool"},
        {"non-integral dereferenceable",
         R"json({"dereferenceable": 1.5})json",
         "param 'P' dereferenceable is not an integer"},
        {"non-integral align", R"json({"align": "16"})json", "param 'P' align is not an integer"},
        {"non-string addr_space",
         R"json({"addr_space": 3})json",
         "param 'P' attr addr_space is not a string"},
        {"non-object attrs", R"json([])json", "param 'P' attrs is not an object"},
    };
    for(const bad_t& c : bad)
    {
        ll.clear();
        err.clear();
        const std::string what = std::string("param attr with ") + c.what;
        check(run_attr_recipe(param_recipe(c.attrs), &ll, &err) != ROCKE_OK,
              (what + " is rejected").c_str());
        check(err.find(c.message) != std::string::npos,
              (what + " names the defect (got: " + err + ")").c_str());
    }
}

} // namespace

// The VM reads an integer parameter's facts -- ints, and a relation as the
// canonical intexpr JSON string the recording builder records -- and they lower
// to no byte; a declaration the builder refuses fails the replay with its text.
void check_param_facts()
{
    const auto recipe = [](const std::string& ldc_attrs) {
        return std::string(R"json({"schema": "rocke.recipe/v1", "kernel_name_fmt": "fact_probe",
  "spec": [], "attrs": {}, "program": [
    {"op": "param", "name": "M", "bind": "M", "type": "i32",
     "attrs": {"ge": 1, "multiple_of": 8}},
    {"op": "param", "name": "ldc", "bind": "ldc", "type": "i32", "attrs": )json")
               + ldc_attrs + R"json(}, {"op": "ret"}]})json";
    };
    std::string ll, err;
    check(run_attr_recipe(recipe(R"json({"ge": "{\"param\":\"M\"}", "le": 4096})json"), &ll, &err)
              == ROCKE_OK,
          ("param facts replay (got: " + err + ")").c_str());
    check(ll.find("(i32 %M, i32 %ldc)") != std::string::npos,
          "param facts lower to plain i32 arguments");
    const char* const bad[][2] = {
        {R"json({"ge": "{\"param\":\"K\"}"})json",
         "ldc: the relation names K, which is not a parameter of this kernel"},
        {R"json({"ge": true})json", "param 'ldc' ge is not an integer or a relation"},
    };
    for(const auto& c : bad)
    {
        ll.clear();
        err.clear();
        check(run_attr_recipe(recipe(c[0]), &ll, &err) != ROCKE_OK, c[0]);
        check(err.find(c[1]) != std::string::npos, (std::string(c[0]) + " got: " + err).c_str());
    }
}

int main()
{
    std::printf("portable_ir recipe_vm_replay:\n");

    check_param_attrs();
    check_param_facts();
    check_kernel_list_attrs();
    check_invalid_integer_values();
    check_scf_if_else();

    std::string ll4, name4, ll8, name8;
    if(!replay(4, "f32", &ll4, &name4) || !replay(8, "f32", &ll8, &name8))
    {
        std::printf("FAIL: recipe replay did not produce IR.\n");
        return 1;
    }

    // kernel_name_fmt interpolates both the int and the string spec value.
    check(name4 == "rocke_recipe_toy_d4_f32", "kernel name at D=4");
    check(name8 == "rocke_recipe_toy_d8_f32", "kernel name at D=8");
    check(ll4.find("rocke_recipe_toy_d4_f32") != std::string::npos,
          "kernel name appears in the lowered IR");

    // The spec drives structure: the static_for body is emitted D times, so the
    // multiply-accumulate count tracks D exactly. This is the whole point of a
    // recipe over a concrete IR graph -- one artifact, many shapes.
    check(count(ll4, "fmul float") == 4, "D=4 expands to 4 multiplies");
    check(count(ll8, "fmul float") == 8, "D=8 expands to 8 multiplies");
    check(count(ll4, "fadd float") == 4, "D=4 expands to 4 accumulates");
    check(count(ll8, "fadd float") == 8, "D=8 expands to 8 accumulates");
    check(ll4 != ll8, "different D produces different IR");

    // Determinism: replay is a pure function of (recipe, spec). The byte-identity
    // gate against the Python lowerer is only meaningful if this holds.
    std::string ll4_again, name4_again;
    if(replay(4, "f32", &ll4_again, &name4_again))
    {
        check(ll4 == ll4_again, "replay at the same spec is byte-identical");
    }

    if(g_fail == 0)
    {
        std::printf("PASS: recipe VM replays and specializes correctly.\n");
        return 0;
    }
    std::printf("FAIL: %d check(s) failed.\n", g_fail);
    return 1;
}
