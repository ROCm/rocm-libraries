// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/backends/generated_conv_backend.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

namespace {
struct FakeLauncher
{
    inline static int launch_count = 0;
    inline static hipStream_t stream{};
    inline static bool throw_on_launch = false;
    inline static const void* input_ptr{};
    inline static const void* weight_ptr{};
    inline static const void* output_ptr{};
    inline static int k_batch{};
    inline static int supported_k_batch{};

    template <typename Args>
    static float launch(const Args& args, const ck_tile::stream_config& config)
    {
        ++launch_count;
        stream     = config.stream_id_;
        input_ptr  = args.in_ptr;
        weight_ptr = args.wei_ptr;
        output_ptr = args.out_ptr;
        k_batch    = args.k_batch;
        assert(!config.time_kernel_);
        assert(config.cold_niters_ == 0);
        assert(config.nrepeat_ == 1);
        assert(!config.is_gpu_timer_);
        if(throw_on_launch)
            throw std::runtime_error("fake launch failed");
        return 7.0f;
    }

    static bool is_supported(const ck_tile::conv::ConvParam&, int k_batch)
    {
        supported_k_batch = k_batch;
        return k_batch == 1;
    }
};

GroupedConvProblem make_problem()
{
    GroupedConvProblem problem;
    problem.arch           = "gfx1100";
    problem.ndim_spatial   = 2;
    problem.G              = 1;
    problem.N              = 1;
    problem.C              = 16;
    problem.K              = 16;
    problem.input_spatial  = {1, 8, 8};
    problem.filter_spatial = {1, 1, 1};
    problem.stride         = {1, 1, 1};
    problem.dilation       = {1, 1, 1};
    problem.padding_left   = {0, 0, 0};
    problem.padding_right  = {0, 0, 0};
    const bool output_valid = problem.compute_output_size();
    assert(output_valid);
    (void)output_valid;
    return problem;
}

ConvInvocationContext make_context(int ordinal = 3)
{
    return {reinterpret_cast<const void*>(0x10),
            reinterpret_cast<const void*>(0x20),
            reinterpret_cast<void*>(0x30),
            ordinal,
            1,
            [](int device) {
                assert(device == 3);
                return std::string{"gfx1100:sramecc+:xnack-"};
            }};
}

void test_default_context()
{
    const ConvInvocationContext context;
    assert(context.input_ptr == nullptr);
    assert(context.weight_ptr == nullptr);
    assert(context.output_ptr == nullptr);
    assert(context.device_ordinal == -1);
    assert(context.k_batch == 1);
    assert(!context.arch_provider);
    assert(!context.benchmarking);
    assert(context.warmup == 0);
    assert(context.repeat == 1);
}

void test_single_launch_and_stream_forwarding()
{
    FakeLauncher::launch_count    = 0;
    FakeLauncher::throw_on_launch = false;
    const auto stream = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1234));
    ScopedConvInvocationContext invocation(make_context());
    const auto result = make_conv_fwd_run_fn<FakeLauncher, 2>()(make_problem(), stream);
    assert(result == 7.0f);
    assert(FakeLauncher::launch_count == 1);
    assert(FakeLauncher::stream == reinterpret_cast<hipStream_t>(stream));
}

void test_context_restoration()
{
    ConvInvocationContext outer = make_context();
    outer.input_ptr              = reinterpret_cast<const void*>(0x40);
    ScopedConvInvocationContext outer_scope(outer);
    {
        ScopedConvInvocationContext inner_scope(make_context());
        assert(conv_invocation_context().input_ptr == reinterpret_cast<const void*>(0x10));
    }
    assert(conv_invocation_context().input_ptr == reinterpret_cast<const void*>(0x40));

    FakeLauncher::throw_on_launch = true;
    try
    {
        ScopedConvInvocationContext inner_scope(make_context());
        make_conv_fwd_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::runtime_error&)
    {
    }
    FakeLauncher::throw_on_launch = false;
    assert(conv_invocation_context().input_ptr == reinterpret_cast<const void*>(0x40));
}

void test_backward_context()
{
    FakeLauncher::launch_count = 0;
    {
        auto context    = make_context();
        context.k_batch = 1;
        ScopedConvInvocationContext invocation(std::move(context));
        make_conv_bwd_data_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(FakeLauncher::launch_count == 1);
        assert(FakeLauncher::input_ptr == reinterpret_cast<const void*>(0x30));
        assert(FakeLauncher::weight_ptr == reinterpret_cast<const void*>(0x20));
        assert(FakeLauncher::output_ptr == reinterpret_cast<const void*>(0x10));
        assert(FakeLauncher::k_batch == 1);
    }

    {
        auto context    = make_context();
        context.k_batch = 4;
        ScopedConvInvocationContext invocation(std::move(context));
        make_conv_bwd_weight_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(FakeLauncher::launch_count == 2);
        assert(FakeLauncher::input_ptr == reinterpret_cast<const void*>(0x10));
        assert(FakeLauncher::weight_ptr == reinterpret_cast<const void*>(0x30));
        assert(FakeLauncher::output_ptr == reinterpret_cast<const void*>(0x20));
        assert(FakeLauncher::k_batch == 4);

        assert((!make_conv_bwd_weight_is_supported_fn<FakeLauncher, 2>()(make_problem())));
        assert(FakeLauncher::supported_k_batch == 4);
    }
}

void test_device_ordinal_and_arch_validation()
{
    int observed_ordinal  = -1;
    auto context          = make_context(7);
    context.arch_provider = [&observed_ordinal](int ordinal) {
        observed_ordinal = ordinal;
        return std::string{"gfx1100:xnack-"};
    };
    ScopedConvInvocationContext invocation(std::move(context));
    validate_conv_invocation(make_problem(), conv_invocation_context(), ConvKBatchRule::MustBeOne);
    assert(observed_ordinal == 7);
}

void test_dispatcher_establishes_production_context()
{
    GroupedConvRegistry registry;
    GroupedConvKernelKey key;
    key.dtype_in = key.dtype_wei = key.dtype_out = "fp16";
    key.layout       = "nhwgc_gkyxc_nhwgk";
    key.ndim_spatial = 2;
    key.op           = GroupedConvOp::Forward;
    key.arch         = "gfx1100";
    key.name         = "production_context";

    const auto stream = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x5678));
    auto run = [stream](const GroupedConvProblem&, void* observed_stream) {
        const auto& context = conv_invocation_context();
        assert(context.input_ptr == reinterpret_cast<const void*>(0x10));
        assert(context.weight_ptr == reinterpret_cast<const void*>(0x20));
        assert(context.output_ptr == reinterpret_cast<void*>(0x30));
        assert(context.device_ordinal == 9);
        assert(context.k_batch == 1);
        assert(observed_stream == stream);
        return 3.0f;
    };
    auto supported = [](const GroupedConvProblem&) { return true; };
    auto instance  = std::make_shared<GroupedConvKernelInstance>(
        key, std::move(run), std::move(supported));
    if(!registry.register_kernel(key, std::move(instance)))
        throw std::runtime_error("failed to register production context test kernel");

    auto problem       = make_problem();
    problem.dtype_in   = problem.dtype_wei = problem.dtype_out = "fp16";
    problem.layout     = "nhwgc_gkyxc_nhwgk";
    problem.op         = GroupedConvOp::Forward;
    GroupedConvDispatcher dispatcher(&registry);
    const auto result = dispatcher.run(reinterpret_cast<const void*>(0x10),
                                       reinterpret_cast<const void*>(0x20),
                                       reinterpret_cast<void*>(0x30),
                                       problem,
                                       9,
                                       stream,
                                       [](int ordinal) {
                                           assert(ordinal == 9);
                                           return std::string{"gfx1100"};
                                       });
    assert(result == 3.0f);
    assert(conv_invocation_context().input_ptr == nullptr);
}

void test_negative_device_ordinal_is_rejected()
{
    auto context          = make_context();
    context.device_ordinal = -1;
    ScopedConvInvocationContext invocation(std::move(context));
    try
    {
        validate_conv_invocation(make_problem(), conv_invocation_context(), ConvKBatchRule::MustBeOne);
        assert(false);
    }
    catch(const std::invalid_argument&)
    {
    }
}

void test_direction_specific_support_rejects_asymmetric_padding()
{
    auto problem = make_problem();
    problem.padding_right[2] = 1;
    assert((!make_conv_fwd_is_supported_fn<FakeLauncher, 2>()(problem)));
    assert((!make_conv_bwd_data_is_supported_fn<FakeLauncher, 2>()(problem)));
    assert((!make_conv_bwd_weight_is_supported_fn<FakeLauncher, 2>()(problem)));

    problem.padding_left[2] = 1;
    assert((make_conv_fwd_is_supported_fn<FakeLauncher, 2>()(problem)));
    assert((make_conv_bwd_data_is_supported_fn<FakeLauncher, 2>()(problem)));
    assert((make_conv_bwd_weight_is_supported_fn<FakeLauncher, 2>()(problem)));
}

void test_arch_mismatch_and_query_failure()
{
    auto mismatch          = make_context();
    mismatch.arch_provider = [](int) { return std::string{"gfx942"}; };
    try
    {
        ScopedConvInvocationContext invocation(std::move(mismatch));
        validate_conv_invocation(make_problem(), conv_invocation_context(), ConvKBatchRule::MustBeOne);
        assert(false);
    }
    catch(const std::runtime_error&)
    {
    }

    auto failure          = make_context();
    failure.arch_provider = [](int) -> std::string { throw std::runtime_error("query failed"); };
    try
    {
        ScopedConvInvocationContext invocation(std::move(failure));
        validate_conv_invocation(make_problem(), conv_invocation_context(), ConvKBatchRule::MustBeOne);
        assert(false);
    }
    catch(const std::runtime_error& error)
    {
        assert(std::string(error.what()) == "query failed");
    }
}

void test_bwd_data_rejects_k_batch_not_one()
{
    auto context    = make_context();
    context.k_batch = 4;
    ScopedConvInvocationContext invocation(std::move(context));
    try
    {
        make_conv_bwd_data_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::invalid_argument&)
    {
    }
}

void test_bwd_paths_reject_negative_device_ordinal()
{
    auto context           = make_context();
    context.device_ordinal = -1;
    ScopedConvInvocationContext invocation(std::move(context));
    try
    {
        make_conv_bwd_data_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::invalid_argument&)
    {
    }
    try
    {
        make_conv_bwd_weight_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::invalid_argument&)
    {
    }
}

void test_bwd_paths_reject_arch_mismatch()
{
    auto mismatch          = make_context();
    mismatch.arch_provider = [](int) { return std::string{"gfx942"}; };
    ScopedConvInvocationContext invocation(std::move(mismatch));
    try
    {
        make_conv_bwd_data_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::runtime_error&)
    {
    }
    try
    {
        make_conv_bwd_weight_run_fn<FakeLauncher, 2>()(make_problem(), nullptr);
        assert(false);
    }
    catch(const std::runtime_error&)
    {
    }
}
} // namespace

int main()
{
    test_default_context();
    test_single_launch_and_stream_forwarding();
    test_context_restoration();
    test_backward_context();
    test_device_ordinal_and_arch_validation();
    test_negative_device_ordinal_is_rejected();
    test_dispatcher_establishes_production_context();
    test_direction_specific_support_rejects_asymmetric_padding();
    test_arch_mismatch_and_query_failure();
    test_bwd_data_rejects_k_batch_not_one();
    test_bwd_paths_reject_negative_device_ordinal();
    test_bwd_paths_reject_arch_mismatch();
    std::cout << "All grouped convolution backend tests passed\n";
}
