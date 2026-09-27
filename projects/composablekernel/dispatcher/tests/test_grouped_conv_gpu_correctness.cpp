// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/register_all_grouped_conv_kernels.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Set by CMake to the architecture the linked catalog was generated for.
#ifndef CK_DISPATCHER_TEST_ARCH
#define CK_DISPATCHER_TEST_ARCH "gfx1100"
#endif

namespace {
constexpr int skip_code    = 77;
constexpr const char* target_arch = CK_DISPATCHER_TEST_ARCH;

std::string normalize_arch(std::string arch)
{
    const auto separator = arch.find(':');
    if(separator != std::string::npos) arch.resize(separator);
    return arch;
}

// fp16 and bf16 buffers are both 16 bit, so the harness stays dtype-agnostic and
// only the conversion differs. Every value used here is exactly representable.
std::uint16_t encode(float value, bool bf16)
{
    if(bf16)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<std::uint16_t>(bits >> 16);
    }
    const __half half_value = __float2half(value);
    std::uint16_t bits      = 0;
    std::memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

float decode(std::uint16_t raw, bool bf16)
{
    if(bf16)
    {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
        float value              = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    __half half_value{};
    std::memcpy(&half_value, &raw, sizeof(half_value));
    return __half2float(half_value);
}

bool check_catalog(const ck_tile::dispatcher::GroupedConvRegistry& registry)
{
    bool found_2d = false, found_3d = false;
    std::set<std::string> ids;
    for(const auto* kernel : registry.all_kernels())
    {
        if(kernel == nullptr || !kernel->executable()) return false;
        const auto& key = kernel->key();
        if(key.arch != target_arch || key.dtype_wei != key.dtype_in ||
           key.dtype_out != key.dtype_in ||
           key.op != ck_tile::dispatcher::GroupedConvOp::Forward || key.pipeline != "compv3" ||
           key.warp_m != 16 || key.warp_n != 16 || key.warp_k != 16)
            return false;
        // Every generated instance must carry its own identity, or lookup by ID
        // cannot tell two catalog entries apart.
        if(key.name.empty() || !ids.insert(kernel->kernel_id()).second) return false;
        found_2d = found_2d || key.ndim_spatial == 2;
        found_3d = found_3d || key.ndim_spatial == 3;
    }
    return found_2d && found_3d && ids.size() == registry.size();
}

struct Case
{
    const char* name;
    const char* dtype;
    int ndim;
    std::int64_t groups;
    std::int64_t channels_per_group;
    std::array<std::int64_t, 3> input;    // {D, H, W}, D == 1 for 2D
    std::array<std::int64_t, 3> filter;   // {Z, Y, X}
    std::array<std::int64_t, 3> stride;
    std::array<std::int64_t, 3> dilation;
    std::array<std::int64_t, 3> padding;  // symmetric
};

ck_tile::dispatcher::GroupedConvProblem make_problem(const Case& c)
{
    using ck_tile::dispatcher::GroupedConvProblemBuilder;
    const auto channels = c.channels_per_group * c.groups;
    auto builder        = GroupedConvProblemBuilder{}
                       .batch(1)
                       .channels(channels, channels)
                       .groups(c.groups)
                       .data_types(c.dtype, c.dtype, c.dtype)
                       .arch(target_arch);
    if(c.ndim == 2)
        return builder.ndim(2)
            .layout("nhwgc_gkyxc_nhwgk")
            .input_size(c.input[1], c.input[2])
            .filter_size(c.filter[1], c.filter[2])
            .stride(c.stride[1], c.stride[2])
            .dilation(c.dilation[1], c.dilation[2])
            .padding(c.padding[1], c.padding[2])
            .build();
    return builder.ndim(3)
        .layout("ndhwgc_gkzyxc_ndhwgk")
        .input_size(c.input[0], c.input[1], c.input[2])
        .filter_size(c.filter[0], c.filter[1], c.filter[2])
        .stride(c.stride[0], c.stride[1], c.stride[2])
        .dilation(c.dilation[0], c.dilation[1], c.dilation[2])
        .padding(c.padding[0], c.padding[1], c.padding[2])
        .build();
}

// With all-ones input and weights every output element equals the number of
// filter taps that land inside the input, times the per-group channel count.
// That is enough to catch stride/dilation/padding indexing and total-vs-per-group
// channel confusion without a full convolution reference.
std::vector<float> expected_per_position(const ck_tile::dispatcher::GroupedConvProblem& p)
{
    std::vector<float> expected;
    for(std::int64_t od = 0; od < p.output_spatial[0]; ++od)
        for(std::int64_t oh = 0; oh < p.output_spatial[1]; ++oh)
            for(std::int64_t ow = 0; ow < p.output_spatial[2]; ++ow)
            {
                const std::array<std::int64_t, 3> out{od, oh, ow};
                std::int64_t taps = 0;
                for(std::int64_t z = 0; z < p.filter_spatial[0]; ++z)
                    for(std::int64_t y = 0; y < p.filter_spatial[1]; ++y)
                        for(std::int64_t x = 0; x < p.filter_spatial[2]; ++x)
                        {
                            const std::array<std::int64_t, 3> tap{z, y, x};
                            bool inside = true;
                            for(int d = 0; d < 3; ++d)
                            {
                                const std::int64_t idx = out[d] * p.stride[d] -
                                                         p.padding_left[d] +
                                                         tap[d] * p.dilation[d];
                                inside = inside && idx >= 0 && idx < p.input_spatial[d];
                            }
                            taps += inside ? 1 : 0;
                        }
                expected.push_back(static_cast<float>(taps * (p.C / p.G)));
            }
    return expected;
}

std::size_t volume(const std::array<std::int64_t, 3>& extents)
{
    return static_cast<std::size_t>(extents[0] * extents[1] * extents[2]);
}

bool run_case(ck_tile::dispatcher::GroupedConvRegistry& registry,
              int device,
              hipStream_t stream,
              const Case& c)
{
    const auto problem   = make_problem(c);
    const bool bf16      = std::string(c.dtype) == "bf16";
    const auto expected  = expected_per_position(problem);
    const auto per_pixel = static_cast<std::size_t>(problem.K);

    std::vector<std::uint16_t> input(volume(problem.input_spatial) *
                                         static_cast<std::size_t>(problem.C),
                                     encode(1.0F, bf16));
    std::vector<std::uint16_t> weight(static_cast<std::size_t>(problem.K) *
                                          volume(problem.filter_spatial) *
                                          static_cast<std::size_t>(problem.C / problem.G),
                                      encode(1.0F, bf16));
    std::vector<std::uint16_t> output(expected.size() * per_pixel, encode(0.0F, bf16));

    void* input_device  = nullptr;
    void* weight_device = nullptr;
    void* output_device = nullptr;
    const auto bytes    = [](const std::vector<std::uint16_t>& v) {
        return v.size() * sizeof(std::uint16_t);
    };
    if(hipMalloc(&input_device, bytes(input)) != hipSuccess ||
       hipMalloc(&weight_device, bytes(weight)) != hipSuccess ||
       hipMalloc(&output_device, bytes(output)) != hipSuccess)
        return false;
    const auto cleanup = [&] {
        (void)hipFree(input_device);
        (void)hipFree(weight_device);
        (void)hipFree(output_device);
    };
    if(hipMemcpyAsync(input_device, input.data(), bytes(input), hipMemcpyHostToDevice,
                      stream) != hipSuccess ||
       hipMemcpyAsync(weight_device, weight.data(), bytes(weight), hipMemcpyHostToDevice,
                      stream) != hipSuccess ||
       hipMemcpyAsync(output_device, output.data(), bytes(output), hipMemcpyHostToDevice,
                      stream) != hipSuccess)
    {
        cleanup();
        return false;
    }

    ck_tile::dispatcher::GroupedConvDispatcher dispatcher(&registry);
    if(dispatcher.select_kernel(problem) == nullptr)
    {
        std::cerr << c.name << ": no kernel selected\n";
        cleanup();
        return false;
    }
    try
    {
        dispatcher.run(input_device, weight_device, output_device, problem, device,
                       reinterpret_cast<void*>(stream));
    }
    catch(const std::exception& error)
    {
        std::cerr << c.name << ": execution failed: " << error.what() << "\n";
        cleanup();
        return false;
    }

    hipEvent_t complete = nullptr;
    if(hipEventCreate(&complete) != hipSuccess || hipEventRecord(complete, stream) != hipSuccess ||
       hipEventSynchronize(complete) != hipSuccess ||
       hipMemcpy(output.data(), output_device, bytes(output), hipMemcpyDeviceToHost) !=
           hipSuccess)
    {
        if(complete != nullptr) (void)hipEventDestroy(complete);
        cleanup();
        return false;
    }
    (void)hipEventDestroy(complete);
    cleanup();

    for(std::size_t position = 0; position < expected.size(); ++position)
        for(std::size_t channel = 0; channel < per_pixel; ++channel)
        {
            const float value = decode(output[position * per_pixel + channel], bf16);
            // bf16 keeps 8 mantissa bits, so the tolerance has to scale with the sum.
            const float tolerance =
                bf16 ? std::max(1.0F, expected[position] / 128.0F) : 0.5F;
            if(std::abs(value - expected[position]) > tolerance)
            {
                std::cerr << c.name << ": position " << position << " channel " << channel
                          << " expected " << expected[position] << " got " << value << "\n";
                return false;
            }
        }
    return true;
}

// Datatype coverage is gated here rather than in check_catalog: a datatype the
// catalog failed to generate shows up as "no kernel selected" for its cases.
const Case cases[] = {
    {"fp16 2d g1 1x1", "fp16", 2, 1, 64, {1, 4, 4}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
    {"fp16 3d g2 1x1x1", "fp16", 3, 2, 64, {2, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
    {"fp16 2d g2 3x3 s2 d2 p2", "fp16", 2, 2, 64, {1, 7, 7}, {1, 3, 3}, {1, 2, 2}, {1, 2, 2},
     {0, 2, 2}},
    {"bf16 2d g1 1x1", "bf16", 2, 1, 64, {1, 4, 4}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
    {"bf16 3d g2 2x3x3 s122 p011", "bf16", 3, 2, 64, {3, 5, 5}, {2, 3, 3}, {1, 2, 2}, {1, 1, 1},
     {0, 1, 1}},
};

// Vector loads need the per-group channel count to divide evenly, so an odd
// per-group channel count must be reported as unsupported rather than run.
const Case unsupported_case{
    "fp16 2d g1 c3 (vector divisibility)", "fp16", 2, 1, 3, {1, 4, 4}, {1, 1, 1},
    {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};

} // namespace

int main()
{
    int device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) return skip_code;
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess) return skip_code;
    hipDeviceProp_t properties{};
    if(hipGetDeviceProperties(&properties, device) != hipSuccess ||
       normalize_arch(properties.gcnArchName) != target_arch)
        return skip_code;

    ck_tile::dispatcher::GroupedConvRegistry registry;
    ck_tile::dispatcher::register_all_grouped_conv_fwd_kernels(registry, target_arch);
    if(!check_catalog(registry))
    {
        std::cerr << "generated " << target_arch << " catalog is empty or has invalid metadata\n";
        return 1;
    }

    if(!registry.find_all_supported(make_problem(unsupported_case)).empty())
    {
        std::cerr << unsupported_case.name << ": expected no supported kernel\n";
        return 1;
    }

    hipStream_t stream = nullptr;
    if(hipStreamCreate(&stream) != hipSuccess) return 1;
    const auto guard = [&](bool ok) {
        if(!ok) (void)hipStreamDestroy(stream);
        return ok;
    };

    // A launch must be refused when the device does not match problem.arch.
    ck_tile::dispatcher::GroupedConvDispatcher dispatcher(&registry);
    bool rejected_mismatch = false;
    try
    {
        dispatcher.run(nullptr, nullptr, nullptr, make_problem(cases[0]), device,
                       reinterpret_cast<void*>(stream),
                       [](int) { return std::string{"gfx942"}; });
    }
    catch(const std::runtime_error& error)
    {
        // Match the message so an unrelated failure cannot pass for a rejection.
        rejected_mismatch =
            std::string(error.what()).find("device architecture mismatch") != std::string::npos;
    }
    if(!guard(rejected_mismatch))
    {
        std::cerr << "arch mismatch was not rejected\n";
        return 1;
    }

    for(const auto& c : cases)
        if(!guard(run_case(registry, device, stream, c)))
        {
            std::cerr << target_arch << " grouped convolution gate failed: " << c.name << "\n";
            return 1;
        }

    (void)hipStreamDestroy(stream);
    std::cout << target_arch << ": all " << (sizeof(cases) / sizeof(cases[0])) << " cases passed\n";
    return 0;
}
