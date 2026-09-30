// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/reducetensor.hpp>
#include <miopen/tensor.hpp>

#include <gtest/gtest.h>

#include "get_handle.hpp"
#include "workspace.hpp"

#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

// Regression tests for the reduce shape limits: shapes the legacy kernels cannot handle must be
// rejected with an error instead of being silently truncated to int32. The rejected shapes are
// only described, never allocated.
namespace {

enum class ReduceCheck
{
    SizingRejects,       // GetWorkspaceSize() and GetIndicesSize() throw
    ReduceTensorRejects, // ReduceTensor() throws
    SizingAccepts,       // GetWorkspaceSize() and GetIndicesSize() do not throw
    ReduceTensorAccepts, // ReduceTensor() runs and gives the correct result
};

struct ReduceShapeLimitsCase
{
    std::string name;
    ReduceCheck check;
    std::vector<std::size_t> inLengths;
    std::vector<std::size_t> inStrides;
    std::vector<std::size_t> outLengths;
    std::vector<std::size_t> outStrides;
    std::string message;

    friend std::ostream& operator<<(std::ostream& os, const ReduceShapeLimitsCase& c)
    {
        return os << c.name;
    }
};

std::vector<ReduceShapeLimitsCase> GetReduceShapeLimitsCases()
{
    constexpr std::size_t intMax = std::numeric_limits<int>::max();
    const std::vector<std::size_t> rank7(7, 2);
    const std::vector<std::size_t> rank7Strides{64, 32, 16, 8, 4, 2, 1};
    const std::vector<std::size_t> rank7Out(7, 1);
    const std::string rank7Message = "at most number of dimensions of 6";
    const std::string tooLarge     = "input tensor has a length or stride exceeding INT32_MAX";

    using enum ReduceCheck;
    // clang-format off
    return {
        {"Rank7", SizingRejects, rank7, rank7Strides, rank7Out, rank7Out, rank7Message},
        {"Rank7ReduceTensor", ReduceTensorRejects, rank7, rank7Strides, rank7Out, rank7Out, rank7Message},
        {"LengthAboveIntMax", ReduceTensorRejects, {intMax + 1, 1}, {1, 1}, {1, 1}, {1, 1}, tooLarge},
        {"StrideAboveIntMax", ReduceTensorRejects, {2, 4}, {intMax + 1, 1}, {1, 4}, {4, 1}, tooLarge},
        {"OutputStrideAboveIntMax", ReduceTensorRejects, {2, 4}, {4, 1}, {2, 1}, {intMax + 1, 1},
         "output tensor has a length or stride exceeding INT32_MAX"},
        // every length and stride fits, but the fp32 byte span does not
        {"ByteSpanAboveIntMax", ReduceTensorRejects, {2, (1ULL << 28) + 1}, {(1ULL << 28) + 1, 1}, {2, 1}, {1, 1},
         "input tensor spans more than INT32_MAX bytes"},
        // sizing stays permissive: the RNN sizes reductions it may never run (ReductionWorkspaceSize)
        {"StrideAboveIntMaxSizing", SizingAccepts, {2, 4}, {intMax + 1, 1}, {1, 4}, {4, 1}, ""},
        {"RnnLikeSizing", SizingAccepts, {1, 8, 16}, {1ULL << 32, 16, 1}, {1, 1, 16}, {16, 16, 1}, ""},
        // the stride of a length-1 dimension is never used, so it may exceed INT32_MAX
        {"RnnLikeReduceTensor", ReduceTensorAccepts, {1, 8, 16}, {1ULL << 32, 16, 1}, {1, 1, 16}, {16, 16, 1}, ""},
    };
    // clang-format on
}

} // anonymous namespace

class GPU_ReduceShapeLimits_FP32 : public ::testing::TestWithParam<ReduceShapeLimitsCase>
{
};

TEST_P(GPU_ReduceShapeLimits_FP32, EnforcesLimits)
{
    const auto& c = GetParam();
    auto&& handle = get_handle();

    // MAX with flattened indices, so that GetIndicesSize() has something to size
    const miopen::ReduceTensorDescriptor reduceDesc{MIOPEN_REDUCE_TENSOR_MAX,
                                                    miopenFloat,
                                                    MIOPEN_NOT_PROPAGATE_NAN,
                                                    MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES,
                                                    MIOPEN_32BIT_INDICES};
    const miopen::TensorDescriptor inDesc{miopenFloat, c.inLengths, c.inStrides};
    const miopen::TensorDescriptor outDesc{miopenFloat, c.outLengths, c.outStrides};

    const auto expectBadParm = [&](auto&& call) {
        try
        {
            call();
            ADD_FAILURE() << "no exception thrown";
        }
        catch(const miopen::Exception& e)
        {
            EXPECT_EQ(e.status, miopenStatusBadParm);
            EXPECT_NE(e.message.find(c.message), std::string::npos) << e.message;
        }
    };

    switch(c.check)
    {
    case ReduceCheck::SizingRejects:
        expectBadParm([&] { std::ignore = reduceDesc.GetWorkspaceSize(handle, inDesc, outDesc); });
        expectBadParm([&] { std::ignore = reduceDesc.GetIndicesSize(inDesc, outDesc); });
        break;
    case ReduceCheck::ReduceTensorRejects: {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        // oversized buffer sizes, so the "not enough" checks cannot fire first
        expectBadParm([&] {
            reduceDesc.ReduceTensor(handle,
                                    nullptr,
                                    std::numeric_limits<std::size_t>::max(),
                                    nullptr,
                                    std::numeric_limits<std::size_t>::max(),
                                    &alpha,
                                    inDesc,
                                    nullptr,
                                    &beta,
                                    outDesc,
                                    nullptr);
        });
        break;
    }
    case ReduceCheck::SizingAccepts:
        EXPECT_NO_THROW(std::ignore = reduceDesc.GetWorkspaceSize(handle, inDesc, outDesc));
        EXPECT_NO_THROW(std::ignore = reduceDesc.GetIndicesSize(inDesc, outDesc));
        break;
    case ReduceCheck::ReduceTensorAccepts: {
        // the input is 3-D with a leading length-1 dimension; only the elements reachable through
        // the other two dimensions are allocated
        const std::size_t rows = c.inLengths[1];
        const std::size_t cols = c.inLengths[2];
        std::vector<float> input(rows * cols);
        std::iota(input.begin(), input.end(), 0.0f);

        const miopen::ReduceTensorDescriptor addDesc{MIOPEN_REDUCE_TENSOR_ADD,
                                                     miopenFloat,
                                                     MIOPEN_NOT_PROPAGATE_NAN,
                                                     MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                                     MIOPEN_32BIT_INDICES};
        auto input_dev  = handle.Write(input);
        auto output_dev = handle.Write(std::vector<float>(cols, 0.0f));
        Workspace wspace{addDesc.GetWorkspaceSize(handle, inDesc, outDesc)};

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        addDesc.ReduceTensor(handle,
                             nullptr,
                             0,
                             wspace.ptr(),
                             wspace.size(),
                             &alpha,
                             inDesc,
                             input_dev.get(),
                             &beta,
                             outDesc,
                             output_dev.get());

        const auto output = handle.Read<float>(output_dev, cols);
        for(std::size_t k = 0; k < cols; k++)
        {
            float expected = 0.0f;
            for(std::size_t j = 0; j < rows; j++)
                expected += input[j * cols + k];
            EXPECT_EQ(output[k], expected) << "column " << k;
        }
        break;
    }
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ReduceShapeLimits_FP32,
                         ::testing::ValuesIn(GetReduceShapeLimitsCases()),
                         [](const auto& info) { return info.param.name; });
