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
