// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    constexpr int32_t maxRotatingBufferNum = 32;
    constexpr int32_t rotatingBufferMiB    = 1;

    struct ByteRange
    {
        uintptr_t begin;
        uintptr_t end;
    };

    class RotatingBufferSizing : public testing::TestWithParam<std::tuple<int32_t, bool>>
    {
    };
} // namespace

template <typename T>
static void setOption(po::variables_map& args, std::string const& name, T value)
{
    args[name].value() = std::move(value);
}

// The first problem alone exceeds the rotating buffer, the second needs several copies.
static po::variables_map makeRotatingArgs(int32_t mode, bool useBeta)
{
    po::variables_map args;
    setOption(args, "problem-identifier", std::string("Contraction_l_Ailk_Bljk_Cijk_Dijk"));
    setOption(args,
              "problem-size",
              std::vector<std::vector<size_t>>{{1024, 512, 1, 512}, {128, 64, 1, 128}});
    setOption(args, "strided-batched", true);
    setOption(args, "batch-mode", 0);
    setOption(args, "grouped-gemm", false);
    setOption(args, "sparse", 0);
    setOption(args, "high-precision-accumulate", false);
    setOption(args, "kernel-language", KernelLanguage::Any);
    setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
    setOption(args, "deterministic-mode", false);
    setOption(args, "c-equal-d", false);
    setOption(args, "uniform-summation-order", 0);
    setOption(args, "type", rocisa::DataType::BFloat16);
    setOption(args, "a-type", rocisa::DataType::BFloat16);
    setOption(args, "b-type", rocisa::DataType::BFloat16);
    setOption(args, "c-type", rocisa::DataType::Float);
    setOption(args, "d-type", rocisa::DataType::Float);
    setOption(args, "alpha-type", rocisa::DataType::Float);
    setOption(args, "beta-type", rocisa::DataType::Float);
    setOption(args, "compute-input-type-A", rocisa::DataType::BFloat16);
    setOption(args, "compute-input-type-B", rocisa::DataType::BFloat16);
    setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
    setOption(args, "activation-compute-type", rocisa::DataType::Float);
    setOption(args, "mx-a-block", 0);
    setOption(args, "mx-b-block", 0);
    setOption(args, "mx-a-free-block", 1);
    setOption(args, "mx-b-free-block", 1);
    setOption(args, "mx-a-type", rocisa::DataType::E8);
    setOption(args, "mx-b-type", rocisa::DataType::E8);
    setOption(args, "mx-scale-format", 0);
    setOption(args, "fused-gemm-a2a", false);
    setOption(args, "metadata-layout", 0);
    setOption(args, "a-ops", TensorOps{});
    setOption(args, "b-ops", TensorOps{});
    setOption(args, "c-ops", TensorOps{});
    setOption(args, "d-ops", TensorOps{});
    setOption(args, "use-gradient", false);
    setOption(args, "output-amaxD", false);
    setOption(args, "use-scaleAB", std::string());
    setOption(args, "use-scaleCD", false);
    setOption(args, "use-bias", 0);
    setOption(args, "bias-source", static_cast<int32_t>(ContractionProblemGemm::TENSOR::D));
    setOption(args, "bias-type-args", std::vector<rocisa::DataType>{rocisa::DataType::Float});
    setOption(args, "use-scaleAlphaVec", 0);
    setOption(args, "device-idx", 0);
    setOption(args, "num-elements-to-validate", 0);
    setOption(args, "pristine-on-gpu", true);
    setOption(args, "prune-mode", PruneSparseMode::PruneRandom);
    setOption(args, "bounds-check", BoundsCheckMode::Disable);
    setOption(args, "init-a", InitMode::Random);
    setOption(args, "init-b", InitMode::Random);
    setOption(args, "init-c", InitMode::Random);
    setOption(args, "init-beta", useBeta ? InitMode::Two : InitMode::Zero);
    setOption(args, "rotating-buffer-size", rotatingBufferMiB);
    setOption(args, "rotating-buffer-mode", mode);
    return args;
}

// Counts the same tensors as the copy made for the rotation: C only matters when beta is set.
static int64_t getCopyBytes(ContractionProblemGemm const& problem)
{
    int64_t bytes = 0;
    for(ContractionProblemGemm::TENSOR tensor : {ContractionProblemGemm::TENSOR::A,
                       ContractionProblemGemm::TENSOR::B,
                       ContractionProblemGemm::TENSOR::C,
                       ContractionProblemGemm::TENSOR::D})
    {
        if(tensor == ContractionProblemGemm::TENSOR::C && problem.beta() == 0.0)
            continue;
        bytes += static_cast<int64_t>(problem.tensors()[tensor].totalAllocatedBytes());
    }
    return bytes;
}

static int32_t getExpectedCopies(int64_t copyBytes)
{
    int64_t const rotatingBytes = static_cast<int64_t>(rotatingBufferMiB) * 1024 * 1024;
    int64_t const num           = (rotatingBytes + copyBytes - 1) / copyBytes;
    return static_cast<int32_t>(std::max<int64_t>(0, std::min<int64_t>(maxRotatingBufferNum, num) - 1));
}

static void appendRanges(std::vector<ByteRange>&       ranges,
                         ContractionProblemGemm const& problem,
                         ContractionInputs const&      inputs)
{
    std::function<void(void const*, int32_t)> add = [&](void const* ptr, int32_t tensor) {
        size_t const bytes = problem.tensors()[tensor].totalAllocatedBytes();
        EXPECT_NE(ptr, nullptr) << "tensor " << tensor;
        auto const begin = reinterpret_cast<uintptr_t>(ptr);
        ranges.push_back({begin, begin + bytes});
    };
    add(inputs.a, ContractionProblemGemm::TENSOR::A);
    add(inputs.b, ContractionProblemGemm::TENSOR::B);
    if(problem.beta() != 0.0)
        add(inputs.c, ContractionProblemGemm::TENSOR::C);
    add(inputs.d, ContractionProblemGemm::TENSOR::D);
}

static void expectDisjoint(std::vector<ByteRange> ranges)
{
    std::sort(ranges.begin(), ranges.end(), [](ByteRange const& lhs, ByteRange const& rhs) {
        return lhs.begin < rhs.begin;
    });
    for(size_t i = 1; i < ranges.size(); i++)
        EXPECT_LE(ranges[i - 1].end, ranges[i].begin) << "overlapping tensors " << i;
}

static std::vector<uint8_t> readDevice(void const* ptr, size_t bytes)
{
    std::vector<uint8_t> host(bytes);
    EXPECT_EQ(hipMemcpy(host.data(), ptr, bytes, hipMemcpyDeviceToHost), hipSuccess);
    return host;
}


// Every problem of a multi-size run must get its own number of rotating copies, regardless of
// the size of the largest problem that the pristine inputs are allocated for.
TEST_P(RotatingBufferSizing, EachProblemRotatesAccordingToItsOwnSize)
{
    int32_t deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "no HIP device available";

    int32_t const mode    = std::get<0>(GetParam());
    bool const    useBeta = std::get<1>(GetParam());
    po::variables_map     args = makeRotatingArgs(mode, useBeta);
    ClientProblemFactory  factory(args);
    DataInitialization    dataInit(args, factory);
    std::vector<std::shared_ptr<ContractionProblem>> const& problems = factory.problems();
    ASSERT_EQ(problems.size(), 2u);

    std::vector<int32_t> expectedCopies;
    for(std::shared_ptr<ContractionProblem> const& problem : problems)
        expectedCopies.push_back(
            getExpectedCopies(getCopyBytes(static_cast<ContractionProblemGemm const&>(*problem))));
    ASSERT_EQ(expectedCopies[0], 0) << "the large problem must not rotate";
    ASSERT_GT(expectedCopies[1], 1) << "the small problem must rotate several times";

    for(size_t p = 0; p < problems.size(); p++)
    {
        auto const& problem = static_cast<ContractionProblemGemm const&>(*problems[p]);
        std::shared_ptr<ProblemInputs>              inputs = dataInit.prepareGPUInputs(problems[p].get());
        std::vector<std::shared_ptr<ProblemInputs>> inputArr;
        ASSERT_NO_THROW(inputArr = dataInit.prepareRotatingGPUOutput(
                            maxRotatingBufferNum, problems[p].get(), inputs, nullptr))
            << "problem " << p;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(static_cast<int32_t>(inputArr.size()), 1 + expectedCopies[p]) << "problem " << p;

        std::vector<ByteRange> ranges;
        for(std::shared_ptr<ProblemInputs> const& input : inputArr)
            appendRanges(ranges, problem, *std::static_pointer_cast<ContractionInputs>(input));
        expectDisjoint(ranges);

        // Mode 1 only reserves the copies, it does not fill them.
        if(mode != 0)
            continue;
        size_t const aBytes = problem.tensors()[ContractionProblemGemm::TENSOR::A].totalAllocatedBytes();
        std::vector<uint8_t> const source = readDevice(std::static_pointer_cast<ContractionInputs>(inputArr[0])->a, aBytes);
        for(size_t i = 1; i < inputArr.size(); i++)
        {
            std::vector<uint8_t> const copy = readDevice(std::static_pointer_cast<ContractionInputs>(inputArr[i])->a, aBytes);
            EXPECT_EQ(std::memcmp(source.data(), copy.data(), aBytes), 0)
                << "copy " << i << " of problem " << p;
        }
    }
}

// Each branch once: C is copied only with beta, and mode 1 does not fill the copies.
INSTANTIATE_TEST_SUITE_P(Modes,
                         RotatingBufferSizing,
                         testing::Values(std::make_tuple(0, true), std::make_tuple(1, false)),
                         [](testing::TestParamInfo<std::tuple<int32_t, bool>> const& info) {
                             return std::string("Mode") + std::to_string(std::get<0>(info.param))
                                    + (std::get<1>(info.param) ? "Beta" : "NoBeta");
                         });
