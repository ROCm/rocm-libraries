// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/conv/RepackProgram.hpp"

/**
 * @file TestGpuRepackKernels.cpp
 * @brief The strided copy on the device against a host reference: random shapes of rank 1 to
 *        6, random permutations, subsampling source strides, odd extents, both element sizes;
 *        and the entry point each kind of step selects. im2col and col2im are checked end to
 *        end, through the convolutions in TestGpuConvGemm.cpp.
 */
namespace
{

using namespace hipblaslt_plugin;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

struct CopyCase
{
    std::vector<int64_t> dims;
    std::vector<int64_t> srcStrides;
    std::vector<int64_t> dstStrides;
    int64_t srcSpan = 0;
    int64_t dstSpan = 0;
};

/// A random shape, read through a permuted and subsampled source and written packed in
/// another random order.
CopyCase randomCase(std::mt19937& random, int rank)
{
    CopyCase result;
    // Small enough at rank 6 that the subsampled source stays a few hundred thousand elements.
    std::uniform_int_distribution<int64_t> extent(1, rank <= 3 ? 7 : 4);
    std::uniform_int_distribution<int64_t> step(1, rank <= 3 ? 3 : 2);
    for(int axis = 0; axis < rank; ++axis)
    {
        result.dims.push_back(extent(random));
    }

    const auto stridesFor = [&](bool subsample, int64_t& span) {
        std::vector<size_t> order(static_cast<size_t>(rank));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<int64_t> strides(static_cast<size_t>(rank), 0);
        int64_t running = 1;
        for(const auto axis : order)
        {
            strides[axis] = running * (subsample ? step(random) : 1);
            running = strides[axis] * result.dims[axis];
        }
        span = running;
        return strides;
    };
    result.srcStrides = stridesFor(true, result.srcSpan);
    result.dstStrides = stridesFor(false, result.dstSpan);
    return result;
}

template <typename Bits>
void runCopy(const CopyCase& copyCase, const RepackProgram& program)
{
    std::vector<Bits> src(static_cast<size_t>(copyCase.srcSpan));
    std::iota(src.begin(), src.end(), Bits(1));
    std::vector<Bits> want(static_cast<size_t>(copyCase.dstSpan), Bits(0));

    StridedCopyArgs args{};
    args.rank = static_cast<int>(copyCase.dims.size());
    args.total = 1;
    for(size_t axis = 0; axis < copyCase.dims.size(); ++axis)
    {
        args.dims[axis] = copyCase.dims[axis];
        args.srcStrides[axis] = copyCase.srcStrides[axis];
        args.dstStrides[axis] = copyCase.dstStrides[axis];
        args.total *= copyCase.dims[axis];
    }
    for(int64_t index = 0; index < args.total; ++index)
    {
        int64_t rest = index;
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for(size_t axis = copyCase.dims.size(); axis-- > 0;)
        {
            const int64_t coordinate = rest % copyCase.dims[axis];
            rest /= copyCase.dims[axis];
            srcOffset += coordinate * copyCase.srcStrides[axis];
            dstOffset += coordinate * copyCase.dstStrides[axis];
        }
        want[static_cast<size_t>(dstOffset)] = src[static_cast<size_t>(srcOffset)];
    }

    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    ASSERT_EQ(hipMalloc(&deviceSrc, src.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceDst, want.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceSrc, src.data(), src.size() * sizeof(Bits), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(deviceDst, 0, want.size() * sizeof(Bits)), hipSuccess);

    const RepackStep step{args,
                          sizeof(Bits) == 4 ? data_objects::DataType::FLOAT
                                            : data_objects::DataType::HALF,
                          0,
                          0};
    program.launch(step, deviceSrc, deviceDst, nullptr);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<Bits> got(want.size());
    ASSERT_EQ(hipMemcpy(got.data(), deviceDst, got.size() * sizeof(Bits), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipFree(deviceSrc), hipSuccess);
    ASSERT_EQ(hipFree(deviceDst), hipSuccess);
    EXPECT_EQ(got, want);
}

TEST(TestGpuRepackKernels, StridedCopyMatchesTheHostOverRandomPermutations)
{
    SKIP_IF_NO_DEVICES();
    const RepackProgram program;
    std::mt19937 random(2026);
    for(int rank = 1; rank <= CONV_REPACK_MAX_RANK; ++rank)
    {
        for(int trial = 0; trial < 4; ++trial)
        {
            const auto copyCase = randomCase(random, rank);
            SCOPED_TRACE(::testing::Message() << "rank " << rank << " trial " << trial);
            runCopy<uint16_t>(copyCase, program);
            runCopy<uint32_t>(copyCase, program);
        }
    }
}

TEST(TestRepackProgram, EntryNamesFollowTheStepKind)
{
    EXPECT_EQ(RepackProgram::entryName({StridedCopyArgs{}, data_objects::DataType::HALF, 0, 0}),
              "StridedCopy2");
    EXPECT_EQ(RepackProgram::entryName({StridedCopyArgs{}, data_objects::DataType::FLOAT, 0, 0}),
              "StridedCopy4");
    EXPECT_EQ(RepackProgram::entryName({Im2ColArgs{}, data_objects::DataType::BFLOAT16, 0, 0}),
              "Im2Col2");
    EXPECT_EQ(RepackProgram::entryName({Im2ColArgs{}, data_objects::DataType::FLOAT, 0, 0}),
              "Im2Col4");
    EXPECT_EQ(RepackProgram::entryName({Col2ImArgs{}, data_objects::DataType::FLOAT, 0, 0}),
              "Col2ImFloat");
    EXPECT_EQ(RepackProgram::entryName({Col2ImArgs{}, data_objects::DataType::HALF, 0, 0}),
              "Col2ImHalf");
    EXPECT_EQ(RepackProgram::entryName({Col2ImArgs{}, data_objects::DataType::BFLOAT16, 0, 0}),
              "Col2ImBfloat16");
    EXPECT_EQ(RepackProgram::entryName({StridedCastArgs{}, data_objects::DataType::HALF, 0, 0}),
              "StridedCastFloatToHalf");
    EXPECT_EQ(RepackProgram::entryName({StridedCastArgs{}, data_objects::DataType::BFLOAT16, 0, 0}),
              "StridedCastFloatToBfloat16");
    // An fp32 cast still adds the batched backward weights' partial sums.
    EXPECT_EQ(RepackProgram::entryName({StridedCastArgs{}, data_objects::DataType::FLOAT, 0, 0}),
              "StridedCastFloatToFloat");
}

} // namespace
