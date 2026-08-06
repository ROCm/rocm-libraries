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
 *     python/rocke/portable_ir/drivers/parity_matrix.py.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
  "attrs": {
    "max_workgroup_size": {"t": "i", "v": 64},
    "agpr_alloc": {"t": "l", "v": [
      {"_": {"t": "i", "v": 0}}, {"_": {"t": "i", "v": 0}}
    ]}
  },
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

void check_rejected(const char* recipe,
                    const char* want,
                    const char* label,
                    const rocke_recipe_spec_int_t* ints = nullptr,
                    int n_ints = 0,
                    const rocke_recipe_spec_str_t* strs = nullptr,
                    int n_strs = 0)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st
        = rocke_recipe_run_from_json(recipe, ints, n_ints, strs, n_strs, &b, &kernel, err, sizeof(err));
    check(st == ROCKE_ERR_VALUE, label);
    check(kernel == nullptr, "rejected recipe has no kernel");
    check(std::string(err).find(want) != std::string::npos, want);
}

void check_replayed(const char* recipe, const char* label)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_recipe_run_from_json(
        recipe, nullptr, 0, nullptr, 0, &b, &kernel, err, sizeof(err));
    check(st == ROCKE_OK && kernel != nullptr, label);
    if(st == ROCKE_OK)
        rocke_ir_builder_free(&b);
}

void check_long_name(const char* recipe, size_t minimum_length)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    const rocke_status_t st = rocke_recipe_run_from_json(
        recipe, nullptr, 0, nullptr, 0, &b, &kernel, err, sizeof(err));
    check(st == ROCKE_OK && kernel != nullptr, "long kernel name replays");
    if(st == ROCKE_OK)
    {
        check(std::strlen(kernel->name) >= minimum_length, "long kernel name is not truncated");
        rocke_ir_builder_free(&b);
    }
}

const char* kNegativeStaticFor = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"static_for","var":"i","lo":0,"hi":1,"step":-1,"body":[]}
  ]
})json";

const char* kZeroDiv = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"const_i32","bind":"x","val":{"div":[1,0]}}
  ]
})json";

const char* kManyResults = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"emit","opcode":"tile.inline_asm","in":[],"outs":[
      {"bind":"r0","type":"i32"},{"bind":"r1","type":"i32"},
      {"bind":"r2","type":"i32"},{"bind":"r3","type":"i32"},
      {"bind":"r4","type":"i32"},{"bind":"r5","type":"i32"},
      {"bind":"r6","type":"i32"},{"bind":"r7","type":"i32"},
      {"bind":"r8","type":"i32"},{"bind":"r9","type":"i32"},
      {"bind":"r10","type":"i32"},{"bind":"r11","type":"i32"},
      {"bind":"r12","type":"i32"},{"bind":"r13","type":"i32"},
      {"bind":"r14","type":"i32"},{"bind":"r15","type":"i32"},
      {"bind":"r16","type":"i32"}
    ],"attrs":{
      "template":{"t":"s","v":""},
      "constraints":{"t":"s","v":"=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v"},
      "sideeffect":{"t":"b","v":true},
      "convergent":{"t":"b","v":false}
    }}
  ]
})json";

const char* kLongKernelName = R"json({
  "schema":"rocke.recipe/v1",
  "kernel_name_fmt":"rocke_long_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "spec":[], "program":[{"op":"ret"}]
})json";

const char* kSpecRecipe = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad",
  "spec":[{"name":"D","kind":"int"}], "program":[{"op":"ret"}]
})json";

const char* kBadOperands = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"emit","opcode":"tile.inline_asm","in":{}}
  ]
})json";

const char* kMissingIterInit = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"const_i32","bind":"lo","val":0}, {"op":"const_i32","bind":"hi","val":1},
    {"op":"const_i32","bind":"step","val":1},
    {"op":"scf_for","iv":"iv","lo":"lo","hi":"hi","step":"step",
     "iter":[{"name":"carry"}],"results":["result"],"body":[]}
  ]
})json";

const char* kBadAttr = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[],
  "attrs":{"bad":{"t":"f","v":"not-a-number"}}, "program":[]
})json";

const char* kBadRegisterPlaceholder = R"json({
  "schema":"rocke.recipe/v1", "kernel_name_fmt":"bad", "spec":[], "program":[
    {"op":"const_i32","bind":"value{","val":0}
  ]
})json";

} // namespace

int main()
{
    std::printf("portable_ir recipe_vm_replay:\n");

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
    check(ll4.find("\"amdgpu-agpr-alloc\"=\"0,0\"") != std::string::npos,
          "integer-list kernel attribute survives replay");

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

    // Invalid parametric constructs must be rejected, never silently changed
    // (division) or allowed to spin forever (a non-progressing static_for).
    check_rejected(kNegativeStaticFor, "static_for step must be positive", "negative static_for rejected");
    check_rejected(kZeroDiv, "integer division by zero", "zero integer division rejected");

    // The VM now carries an arbitrary result list to rocke_b_op rather than
    // silently truncating it at sixteen. The generic builder accepts this
    // synthetic op, proving all 17 result declarations reached it intact.
    check_replayed(kManyResults, "large result list is not truncated");
    check_long_name(kLongKernelName, 300);

    // Runtime values must match the recipe's declared name and kind exactly.
    const rocke_recipe_spec_str_t wrong_kind[] = {{"D", "4"}};
    const rocke_recipe_spec_int_t duplicate_specs[] = {{"D", 4}, {"D", 8}};
    const rocke_recipe_spec_int_t extra_specs[] = {{"D", 4}, {"E", 8}};
    check_rejected(kSpecRecipe, "exactly one int", "missing runtime spec rejected");
    check_rejected(kSpecRecipe, "exactly one int", "wrong runtime spec kind rejected", nullptr, 0,
                   wrong_kind, 1);
    check_rejected(kSpecRecipe, "exactly one int", "duplicate runtime spec rejected", duplicate_specs, 2);
    check_rejected(kSpecRecipe, "undeclared runtime int", "extra runtime spec rejected", extra_specs, 2);

    // Malformed collections, attributes, and format names fail at the schema
    // boundary instead of becoming an empty list/default value/truncated name.
    check_rejected(kBadOperands, "register list must be an array", "bad operand list rejected");
    check_rejected(kMissingIterInit, "needs name/init", "missing loop init rejected");
    check_rejected(kBadAttr, "float value is not numeric", "bad scalar attr rejected");
    check_rejected(kBadRegisterPlaceholder, "unterminated register", "bad register format rejected");

    if(g_fail == 0)
    {
        std::printf("PASS: recipe VM replays and specializes correctly.\n");
        return 0;
    }
    std::printf("FAIL: %d check(s) failed.\n", g_fail);
    return 1;
}
