// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_workgroup_size.cpp -- the exact workgroup_size kernel attr in the C++
 * engine (mirror of tests/core/test_workgroup_size.py).
 *
 * rocke_kernel_max_workgroup_size is the one accessor both lowerers read: it
 * returns workgroup_size when set, else max_workgroup_size, else 256. So a
 * kernel that states only workgroup_size=64 lowers to the same LLVM function
 * attributes and HIP launch bounds as one that states max_workgroup_size=64.
 * The verifier refuses a kernel that sets both attrs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"
#include "rocke/verify.h"

static const char* const BOTH_MESSAGE
    = "kernel sets both workgroup_size and max_workgroup_size -- set one: "
      "workgroup_size is the exact launch block size, max_workgroup_size an upper "
      "bound on it";

static int failures = 0;

static void check(bool ok, const char* what)
{
    if(!ok)
    {
        fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

/* A kernel that stores its thread id to Y[tid]; each named attr is set to its value. */
static rocke_kernel_def_t*
    build(rocke_ir_builder_t* b, const char* attr_a, int value_a, const char* attr_b, int value_b)
{
    if(rocke_ir_builder_init(b, "workgroup_size_probe") != ROCKE_OK)
        return NULL;
    if(attr_a)
        rocke_attr_set_int(b, &b->kernel->attrs, attr_a, value_a);
    if(attr_b)
        rocke_attr_set_int(b, &b->kernel->attrs, attr_b, value_b);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_b_global_store(b, y, tid, tid, 1);
    rocke_b_ret(b);
    return rocke_ir_builder_ok(b) ? rocke_ir_builder_kernel(b) : NULL;
}

static void check_accessor(void)
{
    struct
    {
        const char* attr_a;
        int value_a;
        const char* attr_b;
        int value_b;
        int expected;
    } cases[] = {
        {NULL, 0, NULL, 0, 256},
        {"max_workgroup_size", 128, NULL, 0, 128},
        {"workgroup_size", 64, NULL, 0, 64},
        {"max_workgroup_size", 256, "workgroup_size", 64, 64},
    };
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        rocke_kernel_def_t* k = build(&b, c.attr_a, c.value_a, c.attr_b, c.value_b);
        check(k != NULL, "build the probe kernel");
        if(k)
            check(rocke_kernel_max_workgroup_size(k) == c.expected,
                  "rocke_kernel_max_workgroup_size reads workgroup_size first");
        rocke_ir_builder_free(&b);
    }
}

static char* lower_llvm(const char* attr, const char* arch)
{
    rocke_ir_builder_t b;
    char* text = NULL;
    rocke_kernel_def_t* k = build(&b, attr, 64, NULL, 0);
    if(k && rocke_lower_kernel_to_llvm(k, ROCKE_LLVM_FLAVOR_LLVM22, arch, &text) != ROCKE_OK)
        text = NULL;
    rocke_ir_builder_free(&b);
    return text;
}

static void check_llvm(void)
{
    const struct
    {
        const char* arch;
        const char* attrs;
    } cases[] = {
        {"gfx942", "\"amdgpu-flat-work-group-size\"=\"64,64\""},
        {"gfx1201", "\"amdgpu-flat-work-group-size\"=\"32,64\""},
    };
    for(const auto& c : cases)
    {
        char* exact = lower_llvm("workgroup_size", c.arch);
        char* bound = lower_llvm("max_workgroup_size", c.arch);
        check(exact != NULL && bound != NULL, "lower both probe kernels");
        if(exact && bound)
        {
            check(strstr(exact, c.attrs) != NULL,
                  "workgroup_size=64 sets the flat work-group size");
            check(strcmp(exact, bound) == 0,
                  "workgroup_size and max_workgroup_size lower to the same bytes");
        }
        free(exact);
        free(bound);
    }
}

static void check_hip(void)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = build(&b, "workgroup_size", 64, NULL, 0);
    rocke_strbuf_t text;
    check(k != NULL && rocke_strbuf_init(&text, 256) == 0, "build the probe kernel");
    if(k)
    {
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx942";
        check(rocke_lower_kernel_to_hip(&b, k, &opts, &text) == ROCKE_OK, "lower to HIP");
        check(strstr(rocke_strbuf_cstr(&text), "__launch_bounds__(64)") != NULL,
              "workgroup_size=64 sets __launch_bounds__(64)");
        rocke_strbuf_free(&text);
    }
    rocke_ir_builder_free(&b);
}

static bool verifier_reports(rocke_kernel_def_t* k, const char* message)
{
    rocke_diag_t* diags = NULL;
    size_t n = 0;
    bool found = false;
    rocke_verify(k, &diags, &n);
    for(size_t i = 0; i < n; ++i)
        found |= diags[i].message != NULL && strcmp(diags[i].message, message) == 0;
    rocke_diags_free(diags, n);
    return found;
}

static void check_verifier(void)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = build(&b, "workgroup_size", 64, "max_workgroup_size", 64);
    check(k != NULL && verifier_reports(k, BOTH_MESSAGE),
          "the verifier refuses a kernel that sets both block-size attrs");
    rocke_ir_builder_free(&b);

    k = build(&b, "workgroup_size", 64, NULL, 0);
    check(k != NULL && !verifier_reports(k, BOTH_MESSAGE),
          "the verifier accepts a kernel that sets only workgroup_size");
    rocke_ir_builder_free(&b);
}

int main(void)
{
    check_accessor();
    check_llvm();
    check_hip();
    check_verifier();
    if(failures)
    {
        fprintf(stderr, "test_workgroup_size: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_workgroup_size: OK\n");
    return 0;
}
