// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_unrolled_for_nested_regions.cpp -- the C++ engine lowers an
 * unrolled scf.for whose body holds nested regions.
 *
 * The body is emitted once per trip, and every value it defines must get a name
 * of its own in each trip, while each trip reads that trip's induction variable
 * and carried values -- including the uses and definitions inside regions
 * nested in the body: an scf.if body, an scf.if_else branch and the scf.yield
 * that ends it, and an inner scf.for, unrolled or not. A shared-memory
 * allocation in the body names one buffer, which every trip shares.
 *
 * Each kernel below puts one of those shapes in a loop of TRIPS unrolled trips
 * and lowers it with the C++ engine. In the LLVM IR, every local name must be
 * defined once, and every local and global name the kernel reads must be
 * defined. One more kernel holds two rolled loops side by side whose names
 * would collide when the second renames its values.
 * tests/core/test_unrolled_for_nested_regions.py builds the same kernels in
 * Python, checks that both engines emit the same text, and runs them on a GPU.
 */
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

namespace
{

constexpr int k_lanes = 64;
constexpr int k_trips = 4;
/* Lanes below k_split take the then side of a branch. */
constexpr int k_split = 24;

int g_failures = 0;

void check(bool ok, const std::string& what)
{
    if(!ok)
    {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

/* `kernel` lowered by the C++ engine for gfx942; "" (and a failure, labelled
 * `name`) when it does not lower. */
std::string lower(const rocke_kernel_def_t* kernel, const std::string& name)
{
    char* text = NULL;
    char err[ROCKE_ERR_MSG_CAP] = {0};
    const rocke_status_t status = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &text, err, sizeof err);
    check(status == ROCKE_OK, name + ": lowering failed: " + err);
    std::string lowered = text ? text : "";
    free(text);
    return lowered;
}

/* A kernel with params Y and Z (both i32 pointers) and the lane id. */
struct Kernel
{
    rocke_ir_builder_t b;
    std::string name;
    rocke_value_t* y = nullptr;
    rocke_value_t* z = nullptr;
    rocke_value_t* lane = nullptr;

    explicit Kernel(const char* kernel_name)
        : name(kernel_name)
    {
        if(rocke_ir_builder_init(&b, kernel_name) != ROCKE_OK)
        {
            fprintf(stderr, "FAIL: builder init for %s\n", kernel_name);
            std::exit(1);
        }
        const rocke_type_t* i32_ptr = rocke_ptr_type(&b, rocke_i32(), "global");
        y = rocke_b_param(&b, "Y", i32_ptr, NULL);
        z = rocke_b_param(&b, "Z", i32_ptr, NULL);
        lane = rocke_b_thread_id_x(&b);
    }
    ~Kernel()
    {
        rocke_ir_builder_free(&b);
    }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    rocke_value_t* c(int value)
    {
        return rocke_b_const_i32(&b, value);
    }

    /* A loop over [0, trip_count) with induction variable `iv_name` carrying
     * one i32 named `carried` from `init`. */
    rocke_for_t loop(rocke_value_t* init,
                     rocke_value_t* trip_count,
                     const char* iv_name,
                     const char* carried,
                     bool unroll)
    {
        rocke_iter_arg_t arg = {carried, init};
        return rocke_b_scf_for_iter(&b, c(0), trip_count, c(1), &arg, 1, iv_name, unroll, true);
    }

    /* The unrolled loop of k_trips trips the kernels below test: "k" over
     * "acc". */
    rocke_for_t unrolled(rocke_value_t* init)
    {
        return loop(init, c(k_trips), "k", "acc", true);
    }

    /* Run `body(iv, carried)` in `handle`'s body; it returns the value the
     * body yields. */
    void body(const rocke_for_t& handle,
              const std::function<rocke_value_t*(rocke_value_t*, rocke_value_t*)>& emit)
    {
        rocke_b_region_enter(&b, handle.body);
        rocke_value_t* yielded = emit(handle.iv, handle.iter_vars[0]);
        rocke_b_scf_yield(&b, &yielded, 1);
        rocke_b_region_leave(&b);
    }

    /* Run `emit` in an scf.if on `cond`. */
    void if_then(rocke_value_t* cond, const std::function<void()>& emit)
    {
        rocke_if_t branch = rocke_b_scf_if(&b, cond);
        rocke_b_region_enter(&b, branch.then_region);
        emit();
        rocke_b_region_leave(&b);
    }

    /* Store `result` to Z[lane] and end the kernel; false (and a failure)
     * when the build failed. */
    bool finish(rocke_value_t* result)
    {
        rocke_b_global_store(&b, z, lane, result, 1);
        rocke_b_ret(&b);
        if(!rocke_ir_builder_ok(&b))
        {
            check(false, name + ": build failed: " + rocke_ir_builder_error(&b));
            return false;
        }
        return true;
    }

    /* Store `result` to Z[lane], end the kernel and lower it. */
    std::string finish_and_lower(rocke_value_t* result)
    {
        return finish(result) ? lower(rocke_ir_builder_kernel(&b), name) : "";
    }

    /* Store `result` to Z[lane], end the kernel and serialize it. */
    std::string finish_and_serialize(rocke_value_t* result)
    {
        if(!finish(result))
        {
            return "";
        }
        char* text = NULL;
        check(rocke_ir_serialize(rocke_ir_builder_kernel(&b), &text) == ROCKE_OK,
              name + ": serialization failed");
        std::string serialized = text ? text : "";
        free(text);
        return serialized;
    }
};

/* Trip k stores 10 * k to Y[lane] when lane == k: the induction variable is
 * read only inside the nested scf.if. */
std::string iv_used_only_in_nested_if()
{
    Kernel k("unroll_iv_in_nested_if");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        k.if_then(rocke_b_cmp_eq(&k.b, k.lane, iv), [&] {
            rocke_b_global_store(&k.b, k.y, k.lane, rocke_b_mul(&k.b, iv, k.c(10)), 1);
        });
        return rocke_b_add(&k.b, acc, k.lane);
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* Each trip defines 100 * k + lane inside an scf.if. */
std::string result_defined_in_nested_if()
{
    Kernel k("unroll_result_in_nested_if");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        k.if_then(rocke_b_cmp_lt(&k.b, k.lane, k.c(k_split)), [&] {
            rocke_value_t* value = rocke_b_add(&k.b, rocke_b_mul(&k.b, iv, k.c(100)), k.lane);
            rocke_b_global_store(&k.b, k.y, k.lane, value, 1);
        });
        return rocke_b_add(&k.b, acc, iv);
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* An if/else gives the carried value on its then branch and the lane on its
 * else branch; the carried value is read only by the then branch's scf.yield. */
std::string carried_value_used_only_in_branch_yield()
{
    Kernel k("unroll_carried_in_branch_yield");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        const rocke_type_t* result_types[1] = {rocke_i32()};
        rocke_if_else_t branch = rocke_b_scf_if_else_results(
            &k.b, rocke_b_cmp_lt(&k.b, k.lane, k.c(k_split)), result_types, 1);
        rocke_b_region_enter(&k.b, branch.then_region);
        rocke_b_scf_yield(&k.b, &acc, 1);
        rocke_b_region_leave(&k.b);
        rocke_b_region_enter(&k.b, branch.else_region);
        rocke_b_scf_yield(&k.b, &k.lane, 1);
        rocke_b_region_leave(&k.b);
        return rocke_b_add(&k.b, branch.op->results[0], iv);
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* An scf.if inside an scf.if inside the body; the innermost one reads the
 * induction variable and the carried value and defines a value. */
std::string nested_if_in_if()
{
    Kernel k("unroll_if_in_if");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        k.if_then(rocke_b_cmp_lt(&k.b, k.lane, k.c(k_split)), [&] {
            k.if_then(rocke_b_cmp_eq(&k.b, k.lane, iv), [&] {
                rocke_b_global_store(&k.b, k.y, k.lane, rocke_b_add(&k.b, acc, k.c(1000)), 1);
            });
        });
        return rocke_b_add(&k.b, acc, k.c(1));
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* A rolled scf.for inside the body: trip k sums j over 0 <= j <= k on top of
 * the lane. The inner loop's induction variable and carried value are defined
 * once per trip. */
std::string rolled_loop_in_unrolled_loop()
{
    Kernel k("unroll_rolled_inner_loop");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        rocke_for_t inner
            = k.loop(k.lane, rocke_b_add(&k.b, iv, k.c(1)), "j", "s", /*unroll=*/false);
        k.body(inner, [&](rocke_value_t* j, rocke_value_t* s) { return rocke_b_add(&k.b, s, j); });
        return rocke_b_add(&k.b, acc, inner.op->results[0]);
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* Two rolled scf.for loops one after the other in the body, both naming their
 * values j and s, so the kernel holds 2 * k_trips loops of the same names. */
std::string rolled_loops_side_by_side_in_unrolled_loop()
{
    Kernel k("unroll_rolled_loops_side_by_side");
    rocke_for_t loop = k.unrolled(k.lane);
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        rocke_value_t* carried = acc;
        for(int i = 0; i < 2; ++i)
        {
            rocke_for_t inner
                = k.loop(carried, rocke_b_add(&k.b, iv, k.c(1)), "j", "s", /*unroll=*/false);
            k.body(inner,
                   [&](rocke_value_t* j, rocke_value_t* s) { return rocke_b_add(&k.b, s, j); });
            carried = inner.op->results[0];
        }
        return carried;
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* Inside an scf.if in each trip, lanes below k_split allocate shared memory,
 * store 100 * k + lane to it and copy it back to Y[lane]. */
std::string shared_memory_in_nested_if()
{
    Kernel k("unroll_shared_memory_in_nested_if");
    rocke_for_t loop = k.unrolled(k.c(0));
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        k.if_then(rocke_b_cmp_lt(&k.b, k.lane, k.c(k_split)), [&] {
            const int shape[1] = {k_lanes};
            rocke_value_t* lds = rocke_b_smem_alloc(&k.b, rocke_i32(), shape, 1, "lds");
            rocke_value_t* value = rocke_b_add(&k.b, rocke_b_mul(&k.b, iv, k.c(100)), k.lane);
            rocke_b_smem_store_vN(&k.b, lds, &k.lane, 1, value, 1);
            rocke_value_t* stored = rocke_b_smem_load_vN(&k.b, lds, &k.lane, 1, rocke_i32(), 1);
            rocke_b_global_store(&k.b, k.y, k.lane, rocke_b_vec_extract(&k.b, stored, 0), 1);
        });
        return rocke_b_add(&k.b, acc, iv);
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* An unrolled scf.for of three trips inside the body; its body reads the outer
 * induction variable and defines a value. */
std::string unrolled_loop_in_unrolled_loop()
{
    Kernel k("unroll_unrolled_inner_loop");
    rocke_for_t loop = k.unrolled(k.lane);
    k.body(loop, [&](rocke_value_t* iv, rocke_value_t* acc) {
        rocke_for_t inner = k.loop(acc, k.c(3), "j", "s", /*unroll=*/true);
        k.body(inner, [&](rocke_value_t* j, rocke_value_t* s) {
            return rocke_b_add(&k.b, s, rocke_b_mul(&k.b, iv, j));
        });
        return inner.op->results[0];
    });
    return k.finish_and_lower(loop.op->results[0]);
}

/* Two rolled scf.for loops one after the other in `k`, both with induction
 * variable j; the first carries a value named `first_carried`, the second one
 * named s. Returns the second loop's result. */
rocke_value_t* rolled_loops_side_by_side(Kernel& k, const char* first_carried)
{
    rocke_value_t* carried = k.lane;
    for(const char* name : {first_carried, "s"})
    {
        rocke_for_t loop = k.loop(carried, k.c(2), "j", name, /*unroll=*/false);
        k.body(loop, [&](rocke_value_t* j, rocke_value_t* s) { return rocke_b_add(&k.b, s, j); });
        carried = loop.op->results[0];
    }
    return carried;
}

/* When two loops share their names, the second renames its values and carries
 * %s.for.header.<n>. In this kernel the first loop gives its own carried value
 * that name, so the second loop's renamed name is taken too. The builder
 * refuses a value name that is not an identifier, so the first loop is built
 * with a placeholder name and the renamed name is put in its serialized IR:
 * serialized IR is the way such a name reaches the lowering. */
std::string loop_named_like_a_renamed_loop_value()
{
    static const std::regex renamed_carried(R"(%(s\.for\.header\.[0-9]+) = phi)");
    static const std::regex placeholder(R"(%first\b)");
    std::string plain;
    std::string serialized;
    {
        Kernel k("rolled_loops_side_by_side_plain");
        plain = k.finish_and_lower(rolled_loops_side_by_side(k, "s"));
    }
    {
        Kernel k("rolled_loops_side_by_side_placeholder");
        serialized = k.finish_and_serialize(rolled_loops_side_by_side(k, "first"));
    }
    std::smatch renamed;
    if(!std::regex_search(plain, renamed, renamed_carried))
    {
        check(false,
              "loop_named_like_a_renamed_loop_value: the second loop did not rename its "
              "carried value");
        return "";
    }
    const std::string edited = std::regex_replace(serialized, placeholder, "%" + renamed[1].str());
    check(edited != serialized,
          "loop_named_like_a_renamed_loop_value: the serialized IR has no %first to rename");
    rocke_ir_builder_t parsed;
    if(rocke_ir_builder_init(&parsed, "parsed") != ROCKE_OK)
    {
        check(false, "loop_named_like_a_renamed_loop_value: builder init failed");
        return "";
    }
    rocke_kernel_def_t* kernel = NULL;
    std::string lowered;
    if(rocke_ir_parse(edited.c_str(), &parsed, &kernel) == ROCKE_OK && kernel)
    {
        lowered = lower(kernel, "loop_named_like_a_renamed_loop_value");
    }
    else
    {
        check(false,
              std::string("loop_named_like_a_renamed_loop_value: parse failed: ")
                  + rocke_ir_builder_error(&parsed));
    }
    rocke_ir_builder_free(&parsed);
    return lowered;
}

/* The names `text` defines and reads, by sigil ('%' local, '@' global). A name
 * is defined by an assignment ("<name> = ..."), as a parameter or function in a
 * define / declare line, or (locals) as a block label ("<label>:"). */
struct Names
{
    std::map<std::string, int> local_definitions;
    std::set<std::string> globals_defined;
    std::set<std::string> locals_read;
    std::set<std::string> globals_read;
};

/* Count `name` as defined in `names`. */
void define(Names& names, const std::string& name)
{
    if(name[0] == '%')
    {
        ++names.local_definitions[name];
    }
    else
    {
        names.globals_defined.insert(name);
    }
}

Names names_of(const std::string& text)
{
    static const std::regex name_pattern(R"([%@][-a-zA-Z$._0-9]+)");
    static const std::regex assignment(R"(^\s*[%@][-a-zA-Z$._0-9]+ = )");
    static const std::regex label(R"(^([-a-zA-Z$._0-9]+):)");
    Names names;
    std::istringstream lines(text);
    std::string line;
    while(std::getline(lines, line))
    {
        std::sregex_iterator name_on_line(line.begin(), line.end(), name_pattern);
        const std::sregex_iterator end;
        std::smatch match;
        if(line.rfind("define ", 0) == 0 || line.rfind("declare ", 0) == 0)
        {
            /* The function and its parameters. */
            for(; name_on_line != end; ++name_on_line)
            {
                define(names, name_on_line->str());
            }
            continue;
        }
        if(std::regex_search(line, assignment))
        {
            define(names, name_on_line->str()); /* the first name is the one assigned */
            ++name_on_line;
        }
        else if(std::regex_search(line, match, label))
        {
            define(names, "%" + match[1].str());
        }
        for(; name_on_line != end; ++name_on_line)
        {
            const std::string name = name_on_line->str();
            (name[0] == '%' ? names.locals_read : names.globals_read).insert(name);
        }
    }
    return names;
}

void expect_well_named(const std::string& kernel, const std::string& text)
{
    if(text.empty())
    {
        return; /* the lowering failure is already reported */
    }
    const Names names = names_of(text);
    for(const auto& [name, count] : names.local_definitions)
    {
        check(count == 1, kernel + ": " + name + " is defined " + std::to_string(count) + " times");
    }
    for(const std::string& name : names.locals_read)
    {
        check(names.local_definitions.count(name) == 1,
              kernel + ": " + name + " is read, never defined");
    }
    for(const std::string& name : names.globals_read)
    {
        check(names.globals_defined.count(name) == 1,
              kernel + ": " + name + " is read, never defined");
    }
}

} // namespace

int main()
{
    const std::pair<const char*, std::function<std::string()>> cases[] = {
        {"iv_used_only_in_nested_if", iv_used_only_in_nested_if},
        {"result_defined_in_nested_if", result_defined_in_nested_if},
        {"carried_value_used_only_in_branch_yield", carried_value_used_only_in_branch_yield},
        {"nested_if_in_if", nested_if_in_if},
        {"rolled_loop_in_unrolled_loop", rolled_loop_in_unrolled_loop},
        {"rolled_loops_side_by_side_in_unrolled_loop", rolled_loops_side_by_side_in_unrolled_loop},
        {"shared_memory_in_nested_if", shared_memory_in_nested_if},
        {"unrolled_loop_in_unrolled_loop", unrolled_loop_in_unrolled_loop},
        {"loop_named_like_a_renamed_loop_value", loop_named_like_a_renamed_loop_value},
    };
    for(const auto& [name, lower] : cases)
    {
        expect_well_named(name, lower());
    }
    if(g_failures)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("unrolled scf.for with nested regions: all checks passed\n");
    return 0;
}
