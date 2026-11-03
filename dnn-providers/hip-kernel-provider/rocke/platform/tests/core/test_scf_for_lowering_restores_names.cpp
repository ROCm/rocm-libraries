// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_scf_for_lowering_restores_names.cpp -- a failed LLVM
 * lowering gives the values of an scf.for body their names back.
 *
 * While a loop body lowers, the C++ engine renames the kernel's values: the
 * values a rolled loop's header defines, when an earlier loop already defined
 * one of its names, and the induction variable, carried values and body results
 * of each trip of an unrolled loop. The new names live in the lowerer's arena,
 * which is freed when the lowering returns. Each kernel below fails to lower
 * inside such a body (a tile.trap with an operand, written past the builder),
 * and every renamed value must still hold the very string it held before, not
 * one in the freed arena (the Python engine restores them in a finally block).
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

namespace
{

int g_failures = 0;

void check(bool ok, const std::string& what)
{
    if(!ok)
    {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

struct Kernel
{
    rocke_ir_builder_t b;

    explicit Kernel(const char* name)
    {
        if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", name);
            std::exit(1);
        }
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }

    rocke_value_t* c(int v)
    {
        return rocke_b_const_i32(&b, v);
    }

    /* A loop over [0, 2) named `iv_name` carrying one i32 named `carried`. */
    rocke_for_t loop(const char* iv_name, const char* carried, bool unroll)
    {
        rocke_iter_arg_t arg = {carried, c(0)};
        return rocke_b_scf_for_iter(&b, c(0), c(2), c(1), &arg, 1, iv_name, unroll, true);
    }
};

/* The values whose names must survive the failed lowering, with the name
 * string each holds before it. */
struct Watched
{
    std::vector<rocke_value_t*> values;
    std::vector<const char*> names;

    void add(rocke_value_t* value)
    {
        values.push_back(value);
        names.push_back(value->name);
    }
};

/* Emit, in `loop`'s body, a sum of its induction variable and carried value,
 * then a tile.trap with an operand (which the lowering refuses), then the
 * yield. Watches the loop's values and the sum. */
void build_failing_body(Kernel& k, const rocke_for_t& loop, Watched& watched)
{
    rocke_b_region_enter(&k.b, loop.body);
    rocke_value_t* sum = rocke_b_add(&k.b, loop.iter_vars[0], loop.iv);
    rocke_b_op(&k.b, ROCKE_OP_TILE_TRAP, &sum, 1, NULL, 0, NULL, NULL, 0, NULL, NULL);
    rocke_b_scf_yield(&k.b, &sum, 1);
    rocke_b_region_leave(&k.b);
    watched.add(loop.iv);
    watched.add(loop.iter_vars[0]);
    watched.add(sum);
}

void expect_names_restored(Kernel& k, const Watched& watched, const std::string& what)
{
    rocke_b_ret(&k.b);
    check(rocke_ir_builder_ok(&k.b), what + ": build failed: " + rocke_ir_builder_error(&k.b));
    char* text = NULL;
    const rocke_status_t status = rocke_lower_kernel_to_llvm(
        rocke_ir_builder_kernel(&k.b), ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &text);
    check(status != ROCKE_OK, what + ": the malformed trap lowered");
    free(text);
    for(size_t i = 0; i < watched.values.size(); ++i)
    {
        check(watched.values[i]->name == watched.names[i],
              what + ": value " + std::to_string(i) + " was not given its name back");
    }
}

void rolled_loop_renamed_after_a_loop_of_the_same_names()
{
    Kernel k("restore_rolled");
    rocke_for_t first = k.loop("k0", "acc", false);
    rocke_b_region_enter(&k.b, first.body);
    rocke_b_scf_yield(&k.b, first.iter_vars, 1);
    rocke_b_region_leave(&k.b);
    Watched watched;
    build_failing_body(k, k.loop("k0", "acc", false), watched);
    expect_names_restored(k, watched, "rolled loop");
}

void unrolled_loop_trip()
{
    Kernel k("restore_unrolled");
    Watched watched;
    build_failing_body(k, k.loop("k0", "acc", true), watched);
    expect_names_restored(k, watched, "unrolled loop");
}

} // namespace

int main()
{
    rolled_loop_renamed_after_a_loop_of_the_same_names();
    unrolled_loop_trip();
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("scf.for lowering restores names: all checks passed\n");
    return 0;
}
