// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/conv/RepackProgram.hpp"
#include "kernels/ConvRepackLaunch.hpp"

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

// The run kernels are checked at every run width (no thread floor: these problems are small).

/// A random disjoint-filter col2im (stride at least the dilated extent on every axis): 2-D or
/// 3-D, groups, padding, flip, dx channels-first, channels-last or strided, both column orders,
/// widths and channel counts that do and do not fill 16-byte runs.
struct DisjointCase
{
    ConvColumnGeometry g{};
    int64_t xSpan = 0;
    int64_t columnElements = 0;
};

DisjointCase randomDisjointCase(std::mt19937& random, int trial)
{
    const auto pick
        = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(random); };
    DisjointCase result;
    auto& g = result.g;
    g.n = pick(1, 3);
    g.groups = pick(1, 2);
    g.cPerGroup = trial % 6 == 0 ? 8 * pick(1, 2) : pick(1, 6);
    const bool threeD = trial % 4 == 3;
    for(int axis = 0; axis < 3; ++axis)
    {
        const bool used = axis > 0 || threeD;
        g.filter[axis] = used ? pick(1, 3) : 1;
        g.dilation[axis] = used && g.filter[axis] > 1 && pick(0, 3) == 0 ? 2 : 1;
        const long long extent = (g.filter[axis] - 1) * g.dilation[axis] + 1;
        g.stride[axis] = used ? extent + pick(0, 2) : 1;
        g.prePadding[axis] = used ? pick(0, 2) : 0;
        const long long post = used ? pick(0, 2) : 0;
        g.input[axis] = used ? (axis == 2 && trial % 2 == 0 ? 8 * pick(2, 5) : pick(3, 37)) : 1;
        g.output[axis] = (g.input[axis] + g.prePadding[axis] + post - extent) / g.stride[axis] + 1;
        if(g.output[axis] < 1)
        {
            g.input[axis] += extent;
            g.output[axis]
                = (g.input[axis] + g.prePadding[axis] + post - extent) / g.stride[axis] + 1;
        }
    }
    g.flipFilter = pick(0, 1);
    g.channelsInner = pick(0, 1);
    g.columnsFirst = pick(0, 1);
    const long long c = g.groups * g.cPerGroup;
    const long long d = g.input[0];
    const long long h = g.input[1];
    const long long w = g.input[2];
    switch(trial % 3)
    {
    case 0: // channels-last
        g.xStrides[1] = 1;
        g.xStrides[4] = c;
        g.xStrides[3] = c * w;
        g.xStrides[2] = c * w * h;
        g.xStrides[0] = c * w * h * d;
        break;
    case 1: // channels-first
        g.xStrides[4] = 1;
        g.xStrides[3] = w;
        g.xStrides[2] = w * h;
        g.xStrides[1] = w * h * d;
        g.xStrides[0] = w * h * d * c;
        break;
    default: // channels-first rows padded to an odd pitch, or every other width element
        g.xStrides[4] = pick(0, 1) == 0 ? 1 : 2;
        g.xStrides[3] = g.xStrides[4] * w + 3;
        g.xStrides[2] = g.xStrides[3] * h;
        g.xStrides[1] = g.xStrides[2] * d;
        g.xStrides[0] = g.xStrides[1] * c;
        break;
    }
    result.xSpan = g.xStrides[0] * g.n;
    result.columnElements = g.n * g.output[0] * g.output[1] * g.output[2] * c * g.filter[0]
                            * g.filter[1] * g.filter[2];
    return result;
}

/// The generic disjoint col2im (ConvRepack.cpp col2imDisjoint) on the host.
template <typename Bits>
void disjointReference(const DisjointCase& caseData,
                       const std::vector<Bits>& col,
                       std::vector<Bits>& x)
{
    const auto& g = caseData.g;
    const long long channels = g.groups * g.cPerGroup;
    const long long taps = g.filter[0] * g.filter[1] * g.filter[2];
    const long long columns = channels * taps;
    const long long outputs = g.output[0] * g.output[1] * g.output[2];
    const long long total = g.n * channels * g.input[0] * g.input[1] * g.input[2];
    for(long long index = 0; index < total; ++index)
    {
        long long rest = index;
        long long position[3];
        for(int axis = 2; axis >= 0; --axis)
        {
            position[axis] = rest % g.input[axis];
            rest /= g.input[axis];
        }
        const long long c = rest % channels;
        const long long n = rest / channels;
        long long o = 0;
        long long element = 0;
        bool read = true;
        for(int axis = 0; axis < 3; ++axis)
        {
            const long long s = position[axis] + g.prePadding[axis];
            const long long coordinate = s >= 0 ? s / g.stride[axis] : 0;
            const long long offset = s - coordinate * g.stride[axis];
            const long long tap = offset / g.dilation[axis];
            read = read && s >= 0 && coordinate < g.output[axis] && offset % g.dilation[axis] == 0
                   && tap < g.filter[axis];
            o = o * g.output[axis] + coordinate;
            element
                = element * g.filter[axis] + (g.flipFilter != 0 ? g.filter[axis] - 1 - tap : tap);
        }
        long long target = n * g.xStrides[0] + c * g.xStrides[1];
        for(int axis = 0; axis < 3; ++axis)
        {
            target += position[axis] * g.xStrides[2 + axis];
        }
        const long long group = c / g.cPerGroup;
        const long long column = g.channelsInner != 0 ? (group * taps + element) * g.cPerGroup
                                                            + (c - group * g.cPerGroup)
                                                      : c * taps + element;
        x[static_cast<size_t>(target)]
            = read ? col[static_cast<size_t>(g.columnsFirst != 0
                                                 ? (n * columns + column) * outputs + o
                                                 : (n * outputs + o) * columns + column)]
                   : Bits(0);
    }
}

template <typename Bits>
void runDisjoint(const DisjointCase& caseData, const RepackProgram& program)
{
    std::vector<Bits> col(static_cast<size_t>(caseData.columnElements));
    std::iota(col.begin(), col.end(), Bits(1));
    constexpr Bits SENTINEL = Bits(0xabab);
    std::vector<Bits> want(static_cast<size_t>(caseData.xSpan), SENTINEL);
    disjointReference(caseData, col, want);

    void* deviceCol = nullptr;
    void* deviceX = nullptr;
    ASSERT_EQ(hipMalloc(&deviceCol, col.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceX, want.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceCol, col.data(), col.size() * sizeof(Bits), hipMemcpyHostToDevice),
              hipSuccess);
    const std::vector<Bits> sentinel(want.size(), SENTINEL);
    ASSERT_EQ(
        hipMemcpy(deviceX, sentinel.data(), sentinel.size() * sizeof(Bits), hipMemcpyHostToDevice),
        hipSuccess);

    Col2ImArgs args{};
    args.geometry = caseData.g;
    args.total = caseData.g.n * caseData.g.groups * caseData.g.cPerGroup * caseData.g.input[0]
                 * caseData.g.input[1] * caseData.g.input[2];
    const RepackStep step{args,
                          sizeof(Bits) == 4 ? data_objects::DataType::FLOAT
                                            : data_objects::DataType::HALF,
                          0,
                          0};
    program.launch(step, deviceCol, deviceX, nullptr, 0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<Bits> got(want.size());
    ASSERT_EQ(hipMemcpy(got.data(), deviceX, got.size() * sizeof(Bits), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipFree(deviceCol), hipSuccess);
    ASSERT_EQ(hipFree(deviceX), hipSuccess);
    EXPECT_EQ(got, want);
}

TEST(TestGpuRepackKernels, DisjointCol2imMatchesTheGenericKernelBitForBit)
{
    SKIP_IF_NO_DEVICES();
    const RepackProgram program;
    std::mt19937 random(2027);
    std::set<std::string> entries;
    for(int trial = 0; trial < 96; ++trial)
    {
        const auto caseData = randomDisjointCase(random, trial);
        SCOPED_TRACE(::testing::Message() << "trial " << trial);
        for(const int bytes : {2, 4})
        {
            Col2ImArgs probe{};
            probe.geometry = caseData.g;
            probe.x = nullptr; // 16-byte aligned, as hipMalloc's are
            entries.insert(col2imLaunch(probe, bytes == 2 ? "Half" : "Float", bytes, 0).entry);
        }
        runDisjoint<uint16_t>(caseData, program);
        runDisjoint<uint32_t>(caseData, program);
    }
    // Every disjoint kernel ran: both run axes, scalar and 16-byte runs, and the generic one.
    for(const char* entry : {"Col2ImDisjointWidth2V1",
                             "Col2ImDisjointWidth2V8",
                             "Col2ImDisjointWidth4V4",
                             "Col2ImDisjointChannels2V8",
                             "Col2ImDisjointChannels4V4",
                             "Col2ImDisjoint2"})
    {
        EXPECT_EQ(entries.count(entry), 1u) << entry;
    }
}

/// The generic im2col (ConvRepack.cpp im2col) on the host.
template <typename Bits>
void im2colReference(const ConvColumnGeometry& g,
                     const std::vector<Bits>& x,
                     std::vector<Bits>& col)
{
    const long long taps = g.filter[0] * g.filter[1] * g.filter[2];
    const long long columns = g.groups * g.cPerGroup * taps;
    const long long outputs = g.output[0] * g.output[1] * g.output[2];
    const long long total = g.n * outputs * columns;
    for(long long index = 0; index < total; ++index)
    {
        const long long column
            = g.columnsFirst != 0 ? (index / outputs) % columns : index % columns;
        const long long row = g.columnsFirst != 0
                                  ? (index / (outputs * columns)) * outputs + index % outputs
                                  : index / columns;
        const long long group = column / (g.cPerGroup * taps);
        const long long inGroup = column % (g.cPerGroup * taps);
        const long long channel = g.channelsInner != 0 ? inGroup % g.cPerGroup : inGroup / taps;
        const long long element = g.channelsInner != 0 ? inGroup / g.cPerGroup : inGroup % taps;
        const long long n = row / outputs;
        long long position = row % outputs;
        long long rest = element;
        long long offset = n * g.xStrides[0] + (group * g.cPerGroup + channel) * g.xStrides[1];
        bool inside = true;
        for(int axis = 2; axis >= 0; --axis)
        {
            const long long o = position % g.output[axis];
            position /= g.output[axis];
            const long long e = rest % g.filter[axis];
            rest /= g.filter[axis];
            const long long tap = g.flipFilter != 0 ? g.filter[axis] - 1 - e : e;
            const long long i = o * g.stride[axis] - g.prePadding[axis] + tap * g.dilation[axis];
            inside = inside && i >= 0 && i < g.input[axis];
            offset += i * g.xStrides[2 + axis];
        }
        col[static_cast<size_t>(index)] = inside ? x[static_cast<size_t>(offset)] : Bits(0);
    }
}

template <typename Bits>
void runDisjointIm2col(const DisjointCase& caseData, const RepackProgram& program)
{
    std::vector<Bits> x(static_cast<size_t>(caseData.xSpan));
    std::iota(x.begin(), x.end(), Bits(1));
    std::vector<Bits> want(static_cast<size_t>(caseData.columnElements));
    im2colReference(caseData.g, x, want);

    void* deviceX = nullptr;
    void* deviceCol = nullptr;
    ASSERT_EQ(hipMalloc(&deviceX, x.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceCol, want.size() * sizeof(Bits)), hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceX, x.data(), x.size() * sizeof(Bits), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(deviceCol, 0xab, want.size() * sizeof(Bits)), hipSuccess);

    Im2ColArgs args{};
    args.geometry = caseData.g;
    args.total = caseData.columnElements;
    const RepackStep step{args,
                          sizeof(Bits) == 4 ? data_objects::DataType::FLOAT
                                            : data_objects::DataType::HALF,
                          0,
                          0};
    program.launch(step, deviceX, deviceCol, nullptr, 0);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<Bits> got(want.size());
    ASSERT_EQ(hipMemcpy(got.data(), deviceCol, got.size() * sizeof(Bits), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipFree(deviceX), hipSuccess);
    ASSERT_EQ(hipFree(deviceCol), hipSuccess);
    EXPECT_EQ(got, want);
}

TEST(TestGpuRepackKernels, DisjointIm2colMatchesTheGenericKernelBitForBit)
{
    SKIP_IF_NO_DEVICES();
    const RepackProgram program;
    std::mt19937 random(2028);
    std::set<std::string> entries;
    for(int trial = 0; trial < 96; ++trial)
    {
        auto caseData = randomDisjointCase(random, trial);
        caseData.g.columnsFirst = 1; // the run kernel's columns; channels-last ones keep theirs
        SCOPED_TRACE(::testing::Message() << "trial " << trial);
        for(const int bytes : {2, 4})
        {
            Im2ColArgs probe{};
            probe.geometry = caseData.g;
            entries.insert(im2colLaunch(probe, bytes, 0).entry);
        }
        runDisjointIm2col<uint16_t>(caseData, program);
        runDisjointIm2col<uint32_t>(caseData, program);
    }
    for(const char* entry :
        {"Im2ColDisjointOutputs2V1", "Im2ColDisjointOutputs2V8", "Im2ColDisjointOutputs4V4"})
    {
        EXPECT_EQ(entries.count(entry), 1u) << entry;
    }
}

TEST(TestGpuRepackKernels, NarrowTransposeMatchesTheHostAcrossTiles)
{
    // Channels-first to channels-last over several 64 x 64 tiles with ragged edges: odd channel
    // and pixel counts (16-bit pairs straddling the edges and misaligned rows), an even pair, and
    // a subsampled source.
    SKIP_IF_NO_DEVICES();
    const RepackProgram program;
    for(const auto& [n, c, pixels, step] : std::vector<std::array<int64_t, 4>>{
            {2, 70, 131, 1}, {1, 128, 4225, 1}, {3, 33, 64, 1}, {2, 66, 96, 2}})
    {
        CopyCase copyCase;
        copyCase.dims = {n, c, pixels};
        copyCase.srcStrides = {c * pixels * step, pixels * step, step};
        copyCase.dstStrides = {c * pixels, 1, c};
        copyCase.srcSpan = n * c * pixels * step;
        copyCase.dstSpan = n * c * pixels;
        SCOPED_TRACE(::testing::Message()
                     << n << " x " << c << " x " << pixels << " step " << step);
        runCopy<uint16_t>(copyCase, program);
        runCopy<uint32_t>(copyCase, program);
    }
}

TEST(TestGpuRepackKernels, WideCastSumsManyPartialsInAFixedTree)
{
    // 300 outputs (a 12 x 25 view of a padded destination), each the sum of 200 fp32 partials:
    // the wide cast (64 lanes, every 64th partial in order, then a halving tree). The host model
    // follows the same order, so fp32 output matches it exactly.
    SKIP_IF_NO_DEVICES();
    const RepackProgram program;
    constexpr int64_t ROWS = 12, COLS = 25, BATCHES = 200, PITCH = 32;
    std::mt19937 random(7);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> src(static_cast<size_t>(ROWS * COLS * BATCHES));
    for(auto& v : src)
    {
        v = value(random);
    }
    std::vector<float> want(static_cast<size_t>(ROWS * PITCH), -7.0f);
    for(int64_t r = 0; r < ROWS; ++r)
    {
        for(int64_t c = 0; c < COLS; ++c)
        {
            std::array<float, 64> lane{};
            for(int64_t l = 0; l < 64; ++l)
            {
                float sum = 0.0f;
                for(int64_t b = l; b < BATCHES; b += 64)
                {
                    sum += src[static_cast<size_t>(b * ROWS * COLS + r * COLS + c)];
                }
                lane[static_cast<size_t>(l)] = sum;
            }
            for(int64_t width = 32; width > 0; width /= 2)
            {
                for(int64_t l = 0; l < width; ++l)
                {
                    lane[static_cast<size_t>(l)] += lane[static_cast<size_t>(l + width)];
                }
            }
            want[static_cast<size_t>(r * PITCH + c)] = lane[0];
        }
    }
    void* deviceSrc = nullptr;
    void* deviceDst = nullptr;
    ASSERT_EQ(hipMalloc(&deviceSrc, src.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceDst, want.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceSrc, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    const std::vector<float> fill(want.size(), -7.0f);
    ASSERT_EQ(hipMemcpy(deviceDst, fill.data(), fill.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    StridedCastArgs args{};
    args.rank = 2;
    args.dims[0] = ROWS;
    args.dims[1] = COLS;
    args.srcStrides[0] = COLS;
    args.srcStrides[1] = 1;
    args.dstStrides[0] = PITCH;
    args.dstStrides[1] = 1;
    args.total = ROWS * COLS;
    args.batches = static_cast<int>(BATCHES);
    args.batchStride = ROWS * COLS;
    EXPECT_EQ(castLaunch(args, "StridedCastFloatToFloat").entry, "StridedCastWideFloatToFloat");
    const RepackStep step{args, data_objects::DataType::FLOAT, 0, 0};
    program.launch(step, deviceSrc, deviceDst, nullptr);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<float> got(want.size());
    ASSERT_EQ(hipMemcpy(got.data(), deviceDst, got.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    ASSERT_EQ(hipFree(deviceSrc), hipSuccess);
    ASSERT_EQ(hipFree(deviceDst), hipSuccess);
    EXPECT_EQ(got, want);
    // Few partials keep the per-element loop.
    args.batches = 63;
    EXPECT_TRUE(castLaunch(args, "StridedCastFloatToFloat").entry.empty());
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
