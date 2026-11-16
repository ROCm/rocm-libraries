// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "kernel_includes.hpp"

// The epilogue activations of ConvEpilogue.h on the device, in both forms, against a double
// reference: the few-ulp form (an fp32 output) within its stated ulp bounds, the fast form (a
// 16-bit output) within 0.1 ulp of fp16 and bf16, and both with PyTorch's NaN and infinity
// results.

namespace
{

constexpr int FUNCTIONS = 9;
constexpr std::array<const char*, FUNCTIONS> NAMES{"sigmoid",
                                                   "silu",
                                                   "swish(1.702)",
                                                   "gelu",
                                                   "gelu_tanh",
                                                   "tanh",
                                                   "elu",
                                                   "softplus(1)",
                                                   "softplus(2)"};

constexpr const char* SOURCE = R"(
#define CONV_EPI 1
#define CONV_EPI_CODE return v;
#include "ConvEpilogue.h"
template <int P>
__device__ float activation(int f, float v)
{
    switch(f)
    {
    case 0: return convEpiSigmoid<P>(v);
    case 1: return convEpiSwish<P>(v, 1.0f);
    case 2: return convEpiSwish<P>(v, 1.702f);
    case 3: return convEpiGelu<P>(v);
    case 4: return convEpiGeluTanh<P>(v);
    case 5: return convEpiTanh<P>(v);
    case 6: return convEpiElu<P>(v, 1.0f);
    case 7: return convEpiSoftplus<P>(v, 1.0f);
    default: return convEpiSoftplus<P>(v, 2.0f);
    }
}
extern "C" __global__ void activations(const float* x, float* y, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= n)
    {
        return;
    }
    for(int f = 0; f < 9; ++f)
    {
        y[(2 * f) * n + i] = activation<0>(f, x[i]);
        y[(2 * f + 1) * n + i] = activation<1>(f, x[i]);
    }
}
)";

double sigmoidRef(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double gated(double x, double gate)
{
    return gate == 0.0 ? 0.0 : x * gate;
}

double reference(int f, double x)
{
    switch(f)
    {
    case 0:
        return sigmoidRef(x);
    case 1:
        return gated(x, sigmoidRef(x));
    case 2:
        return gated(x, sigmoidRef(static_cast<double>(1.702F) * x));
    case 3:
        return gated(x, 0.5 * std::erfc(-x * 0.70710678118654752440));
    case 4:
    {
        const double u = 0.79788456080286535588 * (x + 0.044715 * x * x * x);
        return gated(x, sigmoidRef(2.0 * u));
    }
    case 5:
        return std::tanh(x);
    case 6:
        // + 0.0: elu(-0) is +0, as PyTorch computes it (alpha (e^x - 1)).
        return x > 0.0 ? x : std::expm1(x) + 0.0;
    case 7:
        return x > 20.0 ? x : std::log1p(std::exp(x));
    default:
        return 2.0 * x > 20.0 ? x : std::log1p(std::exp(2.0 * x)) / 2.0;
    }
}

/// The fp32 ulp of @p r.
double ulp32(double r)
{
    int e = 0;
    std::frexp(std::fabs(r), &e);
    return std::ldexp(1.0, std::max(e - 24, -149));
}

/// The ulp of @p r in a format with @p mantissa stored bits and minimum normal exponent @p minExp.
double ulpOf(double r, int mantissa, int minExp)
{
    int e = 0;
    std::frexp(std::fabs(r), &e);
    return std::ldexp(1.0, std::max(e - 1, minExp) - mantissa);
}

/// Whether swish's gate underflows at @p x (silu at beta 1, swish at 1.702).
bool gateUnderflows(int f, float x)
{
    return (f == 1 && x < -87.0F) || (f == 2 && 1.702F * x < -87.0F);
}

/// The few-ulp form's bound for function @p f at input @p x (0: not checked there).
double preciseBound(int f, float x)
{
    switch(f)
    {
    case 0:
        return 4.0;
    case 1:
        return x > -80.0F ? 4.5 : 0.0;
    case 2:
        return std::fabs(x) <= 8.0F ? 14.0 : 0.0;
    case 3:
        return 10.0;
    case 4:
        if(std::fabs(x) <= 8.0F)
        {
            return 100.0;
        }
        return x > -10.0F ? 200.0 : 0.0;
    case 5:
    case 6:
        return 2.0;
    default:
        return 5.5;
    }
}

std::vector<float> inputs()
{
    std::vector<float> x;
    // Every binade from 2^-30 to 2^7, both signs, 256 mantissas each.
    uint32_t state = 12345U;
    for(int e = -30; e <= 7; ++e)
    {
        for(int i = 0; i < 256; ++i)
        {
            state = state * 1664525U + 1013904223U;
            const float m = 1.0F + static_cast<float>(state >> 8) * 0x1p-24F;
            x.push_back(std::ldexp(m, e));
            x.push_back(-std::ldexp(m, e));
        }
    }
    // A dense grid over [-16, 16].
    for(int i = -16 * 512; i <= 16 * 512; ++i)
    {
        x.push_back(static_cast<float>(i) / 512.0F);
    }
    for(const float s : {INFINITY,
                         -INFINITY,
                         NAN,
                         0.0F,
                         -0.0F,
                         3.4028235e38F,
                         -3.4028235e38F,
                         1e-45F,
                         -1e-45F,
                         -88.5F,
                         -104.0F,
                         20.0F,
                         20.000002F})
    {
        x.push_back(s);
    }
    return x;
}

class TestGpuConvEpilogueActivations : public ::testing::Test
{
protected:
    /// Runs both forms of every activation over @p x: [function][form][input].
    static std::vector<float> run(const std::vector<float>& x)
    {
        hipDeviceProp_t prop{};
        EXPECT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
        std::string arch(prop.gcnArchName);
        arch = arch.substr(0, arch.find(':'));
        const hip_kernel_provider::compilation::KernelCompileOptions options(arch);
        const std::vector<std::string> optionStrings = options;
        std::vector<const char*> optionPointers;
        optionPointers.reserve(optionStrings.size());
        for(const auto& option : optionStrings)
        {
            optionPointers.push_back(option.c_str());
        }
        std::vector<std::string_view> includeTexts;
        std::vector<const char*> includeNames;
        hip_plugin::getKernelIncList(includeTexts, includeNames);
        const std::vector<std::string> includeStrings(includeTexts.begin(), includeTexts.end());
        std::vector<const char*> includePointers;
        includePointers.reserve(includeStrings.size());
        for(const auto& text : includeStrings)
        {
            includePointers.push_back(text.c_str());
        }
        hiprtcProgram program = nullptr;
        EXPECT_EQ(hiprtcCreateProgram(&program,
                                      SOURCE,
                                      "ConvEpilogueActivations.cpp",
                                      static_cast<int>(includePointers.size()),
                                      includePointers.data(),
                                      includeNames.data()),
                  HIPRTC_SUCCESS);
        const auto compiled = hiprtcCompileProgram(
            program, static_cast<int>(optionPointers.size()), optionPointers.data());
        size_t logSize = 0;
        std::string log;
        if(hiprtcGetProgramLogSize(program, &logSize) == HIPRTC_SUCCESS && logSize > 1)
        {
            log.resize(logSize);
            static_cast<void>(hiprtcGetProgramLog(program, log.data()));
        }
        EXPECT_EQ(compiled, HIPRTC_SUCCESS) << log;
        size_t codeSize = 0;
        EXPECT_EQ(hiprtcGetCodeSize(program, &codeSize), HIPRTC_SUCCESS);
        std::vector<char> code(codeSize);
        EXPECT_EQ(hiprtcGetCode(program, code.data()), HIPRTC_SUCCESS);
        hiprtcDestroyProgram(&program);

        hipModule_t module = nullptr;
        hipFunction_t function = nullptr;
        EXPECT_EQ(hipModuleLoadData(&module, code.data()), hipSuccess);
        EXPECT_EQ(hipModuleGetFunction(&function, module, "activations"), hipSuccess);
        const int n = static_cast<int>(x.size());
        float* dx = nullptr;
        float* dy = nullptr;
        const size_t outCount = static_cast<size_t>(2 * FUNCTIONS) * x.size();
        EXPECT_EQ(hipMalloc(&dx, x.size() * sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMalloc(&dy, outCount * sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        int count = n;
        std::array<void*, 3> args{
            static_cast<void*>(&dx), static_cast<void*>(&dy), static_cast<void*>(&count)};
        const auto blocks = static_cast<unsigned>((n + 255) / 256);
        EXPECT_EQ(hipModuleLaunchKernel(
                      function, blocks, 1, 1, 256, 1, 1, 0, nullptr, args.data(), nullptr),
                  hipSuccess);
        std::vector<float> y(outCount);
        EXPECT_EQ(hipMemcpy(y.data(), dy, outCount * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        static_cast<void>(hipFree(dx));
        static_cast<void>(hipFree(dy));
        static_cast<void>(hipModuleUnload(module));
        return y;
    }
};

TEST_F(TestGpuConvEpilogueActivations, BothFormsWithinTheirBounds)
{
    SKIP_IF_NO_DEVICES();
    const auto x = inputs();
    const auto y = run(x);
    const size_t n = x.size();
    for(int f = 0; f < FUNCTIONS; ++f)
    {
        const char* name = NAMES[static_cast<size_t>(f)];
        double worstPrecise = 0.0;
        double worstHalf = 0.0;
        double worstBf16 = 0.0;
        int failures = 0;
        for(size_t i = 0; i < n; ++i)
        {
            const double r = reference(f, static_cast<double>(x[i]));
            for(int form = 0; form < 2; ++form)
            {
                const float got = y[(static_cast<size_t>(2 * f + form) * n) + i];
                if(std::isnan(r) || std::isinf(r) || std::isnan(got) || std::isinf(got))
                {
                    // NaN gives NaN, infinities their limits.
                    const bool same = (std::isnan(r) && std::isnan(got))
                                      || (!std::isnan(r) && static_cast<double>(got) == r);
                    if(!same && failures++ < 10)
                    {
                        ADD_FAILURE() << name << " form " << form << " x " << x[i] << " got " << got
                                      << " want " << r;
                    }
                    continue;
                }
                if(r == 0.0 && got == 0.0F
                   && std::signbit(static_cast<double>(got)) != std::signbit(r))
                {
                    if(failures++ < 10)
                    {
                        ADD_FAILURE()
                            << name << " form " << form << " x " << x[i] << ": sign of zero";
                    }
                }
                const double err = std::fabs(static_cast<double>(got) - r);
                // Below 2^-122, or where swish's gate e^(beta x) underflows (beta x < -87), the
                // result may flush to zero: a magnitude under 1e-36 either way.
                if(std::fabs(r) < 0x1p-122 || gateUnderflows(f, x[i]))
                {
                    EXPECT_LT(err, 2e-36) << name << " form " << form << " x " << x[i];
                    continue;
                }
                if(form == 1)
                {
                    const double bound = preciseBound(f, x[i]);
                    if(bound > 0.0)
                    {
                        const double ulps = err / ulp32(r);
                        worstPrecise = std::max(worstPrecise, ulps);
                        if(ulps > bound && failures++ < 10)
                        {
                            ADD_FAILURE() << name << " few-ulp form x " << x[i] << ": " << ulps
                                          << " ulp, bound " << bound;
                        }
                    }
                    continue;
                }
                // The fast form: under 0.1 ulp of a 16-bit output.
                const double halfUlps = std::fabs(r) <= 65504.0 ? err / ulpOf(r, 10, -14) : 0.0;
                const double bf16Ulps = err / ulpOf(r, 7, -126);
                worstHalf = std::max(worstHalf, halfUlps);
                worstBf16 = std::max(worstBf16, bf16Ulps);
                if((halfUlps > 0.1 || bf16Ulps > 0.1) && failures++ < 10)
                {
                    ADD_FAILURE() << name << " fast form x " << x[i] << ": " << halfUlps
                                  << " fp16 ulp, " << bf16Ulps << " bf16 ulp";
                }
            }
        }
        std::cout << name << ": few-ulp form " << worstPrecise << " fp32 ulp; fast form "
                  << worstHalf << " fp16 ulp, " << worstBf16 << " bf16 ulp\n";
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
