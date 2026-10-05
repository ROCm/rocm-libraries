// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvRegistry and GroupedConvDispatcher using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

using namespace ck_tile::dispatcher;

namespace {
GroupedConvKernelKey phase0_key()
{
    GroupedConvKernelKey key;
    key.dtype_in = key.dtype_wei = key.dtype_out = "fp16";
    key.layout = "nhwgc_gkyxc_nhwgk";
    key.ndim_spatial = 2;
    key.op = GroupedConvOp::Forward;
    key.arch = "gfx1100";
    key.name = "grouped_conv_fwd_fp16_nhwgc_2d_compv3";
    return key;
}

GroupedConvKernelKey named_key(const std::string& name)
{
    auto key = phase0_key();
    key.name = name;
    return key;
}

GroupedConvProblem phase0_problem()
{
    GroupedConvProblem problem;
    problem.dtype_in = problem.dtype_wei = problem.dtype_out = "fp16";
    problem.layout = "nhwgc_gkyxc_nhwgk";
    problem.ndim_spatial = 2;
    problem.op = GroupedConvOp::Forward;
    problem.arch = "gfx1100";
    return problem;
}

GroupedConvKernelInstancePtr executable(const GroupedConvKernelKey& key, bool supported = true)
{
    return std::make_shared<GroupedConvKernelInstance>(
        key, [](const GroupedConvProblem&, void*) { return 0.0f; },
        [supported](const GroupedConvProblem&) { return supported; });
}
} // namespace

void test_grouped_conv_key_and_id()
{
    std::cout << "  test_grouped_conv_key_and_id... ";
    const auto base = phase0_key();
    const GroupedConvKernelKeyHash hash;
    assert(base.kernel_id().find("cktile_gconv_v1") == 0);

    auto check_changed = [&](auto mutate) {
        auto changed = base;
        mutate(changed);
        assert(!(changed == base));
        assert(changed.kernel_id() != base.kernel_id());
        assert(hash(changed) != hash(base));
    };
    check_changed([](auto& k) { k.dtype_wei = "bf16"; });
    check_changed([](auto& k) { k.dtype_out = "fp32"; });
    check_changed([](auto& k) { k.layout = "ndhwgc_gkzyxc_ndhwgk"; });
    check_changed([](auto& k) { k.ndim_spatial = 3; });
    check_changed([](auto& k) { k.op = GroupedConvOp::BackwardData; });
    check_changed([](auto& k) { k.arch = "gfx1101"; });
    check_changed([](auto& k) { k.name += "_dsb"; });

    // Descriptive fields are reported, not identifying: they must not split the key.
    auto described = base;
    described.tile_m += 1;
    described.pipeline = "wavelet";
    assert(described == base);
    assert(described.kernel_id() == base.kernel_id());

    // Length framing keeps neighbouring fields from running together.
    auto split_a = base;
    auto split_b = base;
    split_a.arch = "a|1:b";
    split_a.name = "";
    split_b.arch = "a";
    split_b.name = "1:b|0:";
    assert(split_a.kernel_id() != split_b.kernel_id());
    std::cout << "PASSED\n";
}

void test_grouped_conv_executable_registration_validation()
{
    std::cout << "  test_grouped_conv_executable_registration_validation... ";
    GroupedConvRegistry reg;
    auto key = phase0_key();
    assert(!reg.register_kernel(key, nullptr));
    assert(!reg.register_kernel(key, std::make_shared<GroupedConvKernelInstance>(
                                        key, [](const auto&, void*) { return 0.0f; })));
    assert(!reg.register_kernel(key, std::make_shared<GroupedConvKernelInstance>(
                                        key, GroupedConvKernelInstance::RunFn{},
                                        [](const auto&) { return true; })));
    auto unnamed = key;
    unnamed.name.clear();
    assert(!reg.register_kernel(unnamed, executable(unnamed)));
    assert(reg.register_kernel(key, executable(key)));
    std::cout << "PASSED\n";
}

void test_grouped_conv_supported_lookup()
{
    std::cout << "  test_grouped_conv_supported_lookup... ";
    GroupedConvRegistry reg;
    auto problem = phase0_problem();
    auto a = named_key("kernel_a");
    auto b = named_key("kernel_b");
    auto unsupported = named_key("kernel_c_unsupported");
    assert(reg.register_kernel(b, executable(b), Priority::High));
    assert(reg.register_kernel(a, executable(a), Priority::High));
    assert(reg.register_kernel(unsupported, executable(unsupported, false), Priority::High));
    // Equal priority, so the lowest kernel ID wins.
    assert(reg.find_best_supported(problem)->key().name == a.name);
    auto all = reg.find_all_supported(problem);
    assert(all.size() == 2);
    assert(all[0]->kernel_id() < all[1]->kernel_id());
    assert(reg.find_by_id(problem, b.kernel_id()) != nullptr);
    assert(reg.find_by_id(problem, "unknown") == nullptr);

    auto mismatch = problem;
    mismatch.dtype_out = "fp32";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.layout = "other";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.ndim_spatial = 3;
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.arch = "gfx1101";
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);
    mismatch = problem; mismatch.op = GroupedConvOp::BackwardData;
    assert(reg.find_by_id(mismatch, a.kernel_id()) == nullptr);

    auto invalid = problem;
    invalid.G = 0;
    assert(reg.find_all_supported(invalid).empty());
    assert(reg.find_best_supported(invalid) == nullptr);
    assert(reg.find_by_id(invalid, a.kernel_id()) == nullptr);

    // Padding policy lives in the support callback, not the registry.
    auto asymmetric = problem;
    asymmetric.padding_right[2]++;
    assert(asymmetric.compute_output_size());
    assert(reg.find_all_supported(asymmetric).size() == 2);

    // Selection must not depend on registration order.
    GroupedConvRegistry reversed;
    assert(reversed.register_kernel(a, executable(a), Priority::High));
    assert(reversed.register_kernel(b, executable(b), Priority::High));
    assert(reversed.find_best_supported(problem)->kernel_id() == a.kernel_id());
    auto reversed_all = reversed.find_all_supported(problem);
    assert(reversed_all.size() == 2 && reversed_all[0]->kernel_id() == all[0]->kernel_id() &&
           reversed_all[1]->kernel_id() == all[1]->kernel_id());
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_basic()
{
    std::cout << "  test_grouped_conv_registry_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    reg.set_name("test_registry");
    assert(reg.get_name() == "test_registry");

    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_distinct_names_do_not_collide()
{
    std::cout << "  test_grouped_conv_distinct_names_do_not_collide... ";
    GroupedConvRegistry reg;
    auto wavelet = named_key("grouped_conv_fwd_fp16_nhwgc_3d_wavelet_128x128x64");
    auto compv3  = named_key("grouped_conv_fwd_fp16_nhwgc_3d_compv3_128x128x64");
    assert(reg.register_kernel(wavelet, executable(wavelet)));
    assert(reg.register_kernel(compv3, executable(compv3)));
    assert(reg.size() == 2);
    assert(reg.find_by_id(phase0_problem(), wavelet.kernel_id())->key().name == wavelet.name);
    assert(reg.find_by_id(phase0_problem(), compv3.kernel_id())->key().name == compv3.name);
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_all_kernels()
{
    std::cout << "  test_grouped_conv_registry_all_kernels... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    auto all = reg.all_kernels();
    assert(all.size() == 1);
    assert(all[0]->name().find("grouped_conv_") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_clear()
{
    std::cout << "  test_grouped_conv_registry_clear... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));
    assert(reg.size() == 1);

    reg.clear();
    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_thread_safe()
{
    std::cout << "  test_grouped_conv_registry_thread_safe... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    const int num_threads     = 4;
    const int sets_per_thread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for(int t = 0; t < num_threads; t++)
    {
        threads.emplace_back([t, &reg, &success_count]() {
            for(int k = 0; k < sets_per_thread; k++)
            {
                const auto key =
                    named_key("kernel_t" + std::to_string(t) + "_k" + std::to_string(k));
                if(reg.register_kernel(key, executable(key)))
                {
                    success_count++;
                }
            }
        });
    }

    for(auto& th : threads)
        th.join();

    assert(reg.size() == num_threads * sets_per_thread);
    assert(success_count.load() == num_threads * sets_per_thread);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_export_json()
{
    std::cout << "  test_grouped_conv_registry_export_json... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    std::string json = reg.export_json(false);
    assert(!json.empty());
    assert(json.find("\"kernels\"") != std::string::npos);
    assert(json.find("\"metadata\"") != std::string::npos);
    assert(json.find("grouped_conv_") != std::string::npos);

    std::string json_stats = reg.export_json(true);
    assert(json_stats.find("\"statistics\"") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_filter()
{
    std::cout << "  test_grouped_conv_registry_filter... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto small = named_key("kernel_small");
    auto large = named_key("kernel_large");
    large.tile_m = 256;
    auto other_dtype = named_key("kernel_bf16");
    other_dtype.dtype_in = "bf16";
    reg.register_kernel(small, executable(small));
    reg.register_kernel(large, executable(large));
    reg.register_kernel(other_dtype, executable(other_dtype));

    auto fp16_only =
        reg.filter([](const GroupedConvKernelInstance& k) { return k.key().dtype_in == "fp16"; });
    assert(fp16_only.size() == 2);

    auto large_tile = reg.filter([](const GroupedConvKernelInstance& k) {
        return k.key().tile_m >= 256 || k.key().tile_n >= 256;
    });
    assert(large_tile.size() == 1);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_basic()
{
    std::cout << "  test_grouped_conv_dispatcher_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = phase0_problem();

    float time = dispatcher.run(problem, nullptr);
    assert(time >= 0.0f);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_select()
{
    std::cout << "  test_grouped_conv_dispatcher_select... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    auto key = phase0_key();
    reg.register_kernel(key, executable(key));

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = phase0_problem();

    const auto* selected = dispatcher.select(problem);
    assert(selected != nullptr);
    assert(selected->name().find("grouped_conv_") != std::string::npos);
    assert(selected->matches(problem));

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate()
{
    std::cout << "  test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate... ";
    GroupedConvRegistry reg;

    auto unsupported_key = named_key("grouped_conv_fwd_unsupported");
    assert(reg.register_kernel(unsupported_key, executable(unsupported_key, false)));

    auto supported_key = named_key("grouped_conv_fwd_supported");
    assert(reg.register_kernel(supported_key, executable(supported_key)));

    GroupedConvDispatcher dispatcher(&reg);
    dispatcher.set_strategy(GroupedConvDispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic([](const auto&) {
        return std::vector<std::string>{"grouped_conv_fwd_unsupported",
                                       "grouped_conv_fwd_supported"};
    });

    // The first ranked name is registered but unsupported, so the gate must skip it.
    const auto* selected = dispatcher.select(phase0_problem());
    assert(selected != nullptr);
    assert(selected->key().name == supported_key.name);
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Registry ===\n\n";
    test_grouped_conv_registry_basic();
    test_grouped_conv_key_and_id();
    test_grouped_conv_executable_registration_validation();
    test_grouped_conv_supported_lookup();
    test_grouped_conv_distinct_names_do_not_collide();
    test_grouped_conv_registry_all_kernels();
    test_grouped_conv_registry_clear();
    test_grouped_conv_registry_thread_safe();
    test_grouped_conv_registry_export_json();
    test_grouped_conv_registry_filter();
    test_grouped_conv_dispatcher_basic();
    test_grouped_conv_dispatcher_select();
    test_grouped_conv_dispatcher_heuristic_uses_executable_candidate_gate();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
