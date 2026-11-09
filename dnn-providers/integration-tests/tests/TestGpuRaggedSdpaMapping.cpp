// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The GPU reference kernel addresses ragged tensors through raggedBatchRows (the same header the
// kernel is compiled with). On the host it must agree with the Data SDK's container functions
// (RFC 0014 §4.5) on every layout of RFC 0014 §4.5.1: same origins, lengths and addressed range.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn-gpu-ref/detail/RaggedBatchMapping.hpp>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLayoutTestSupport.hpp>

using hipdnn_data_sdk::utilities::detail::raggedBatches;
using hipdnn_data_sdk::utilities::detail::raggedFootprint;
using hipdnn_gpu_ref::raggedBatchRows;
using hipdnn_gpu_ref::raggedTokenElement;
using hipdnn_test_sdk::utilities::ragged_layout_test::RaggedLayoutCase;
using hipdnn_test_sdk::utilities::ragged_layout_test::raggedLayoutCase;
using hipdnn_test_sdk::utilities::ragged_layout_test::raggedLayoutCases;
using hipdnn_test_sdk::utilities::ragged_layout_test::stridedAuxBuffer;

namespace
{

constexpr int SEQ_AXIS = 2;

// The rows of RFC 0014 §4.5.1 the container accepts.
std::vector<RaggedLayoutCase> acceptedLayouts()
{
    std::vector<RaggedLayoutCase> accepted;
    for(const auto& layout : raggedLayoutCases())
    {
        if(!layout.rejected)
        {
            accepted.push_back(layout);
        }
    }
    return accepted;
}

std::vector<int64_t> elementOffsetsOf(const RaggedLayoutCase& layout)
{
    std::vector<int64_t> offsets;
    offsets.reserve(layout.offsets.size());
    for(const auto offset : layout.offsets)
    {
        offsets.push_back(static_cast<int64_t>(offset) * layout.multiplier);
    }
    return offsets;
}

std::vector<int64_t> seqLensOf(const RaggedLayoutCase& layout)
{
    return {layout.seqLen.begin(), layout.seqLen.end()};
}

} // namespace

// Per batch, the kernel-side mapping gives the container's origin and length (RFC 0014 §4.5), and
// its live rows are the container's rows.
TEST(TestGpuRaggedSdpaMapping, BatchRowsMatchModel)
{
    for(const auto& layout : acceptedLayouts())
    {
        SCOPED_TRACE(layout.id);
        const auto seqLens = layout.seqLen.empty() ? std::vector<int64_t>{} : seqLensOf(layout);
        const auto model = raggedBatches(elementOffsetsOf(layout),
                                         layout.seqLen.empty() ? nullptr : &seqLens,
                                         layout.strides[SEQ_AXIS],
                                         layout.dims[SEQ_AXIS]);
        ASSERT_EQ(model.lengths, layout.lengths);
        const int* seqLen = layout.seqLen.empty() ? nullptr : layout.seqLen.data();
        for(int64_t b = 0; b < layout.dims[0]; ++b)
        {
            const auto rows = raggedBatchRows(layout.offsets.data(),
                                              1,
                                              seqLen,
                                              1,
                                              b,
                                              layout.multiplier,
                                              layout.strides[SEQ_AXIS],
                                              layout.dims[SEQ_AXIS]);
            EXPECT_EQ(rows.origin, model.origins[static_cast<size_t>(b)]) << "batch " << b;
            EXPECT_EQ(rows.length, model.lengths[static_cast<size_t>(b)]) << "batch " << b;
            for(int64_t s = 0; s < layout.dims[SEQ_AXIS]; ++s)
            {
                const bool live = s < rows.length;
                EXPECT_EQ(raggedTokenElement(rows, s, layout.strides[SEQ_AXIS]),
                          live
                              ? model.origins[static_cast<size_t>(b)] + s * layout.strides[SEQ_AXIS]
                              : -1)
                    << "batch " << b << " row " << s;
            }
        }
    }
}

// Every live (b, s, h, d) element the mapping addresses lies inside RFC 0014 §4.5's elementSpace,
// the last one is its end, and the distinct addresses are the live elements (zero-stride axes
// collapsed).
TEST(TestGpuRaggedSdpaMapping, AddressedRangeEqualsFootprint)
{
    for(const auto& layout : acceptedLayouts())
    {
        SCOPED_TRACE(layout.id);
        const auto seqLens = layout.seqLen.empty() ? std::vector<int64_t>{} : seqLensOf(layout);
        const auto model = raggedBatches(elementOffsetsOf(layout),
                                         layout.seqLen.empty() ? nullptr : &seqLens,
                                         layout.strides[SEQ_AXIS],
                                         layout.dims[SEQ_AXIS]);
        const auto footprint = raggedFootprint(layout.dims, layout.strides, SEQ_AXIS, model);
        EXPECT_EQ(footprint.elementCount, layout.count);
        EXPECT_EQ(footprint.elementSpace, layout.space);

        const int* seqLen = layout.seqLen.empty() ? nullptr : layout.seqLen.data();
        std::set<int64_t> addresses;
        for(int64_t b = 0; b < layout.dims[0]; ++b)
        {
            const auto rows = raggedBatchRows(layout.offsets.data(),
                                              1,
                                              seqLen,
                                              1,
                                              b,
                                              layout.multiplier,
                                              layout.strides[SEQ_AXIS],
                                              layout.dims[SEQ_AXIS]);
            for(int64_t s = 0; s < rows.length; ++s)
            {
                for(int64_t h = 0; h < layout.dims[1]; ++h)
                {
                    for(int64_t d = 0; d < layout.dims[3]; ++d)
                    {
                        addresses.insert(raggedTokenElement(rows, s, layout.strides[SEQ_AXIS])
                                         + h * layout.strides[1] + d * layout.strides[3]);
                    }
                }
            }
        }
        if(layout.space == 0)
        {
            EXPECT_TRUE(addresses.empty());
            continue;
        }
        EXPECT_EQ(static_cast<size_t>(*addresses.rbegin()) + 1, layout.space);
        const size_t broadcast = layout.strides[1] == 0 ? static_cast<size_t>(layout.dims[1]) : 1;
        EXPECT_EQ(addresses.size(), layout.count / broadcast);
    }
}

// The layouts RFC 0014 §4.5 rejects have no valid mapping: rows of different batches overlap (L10)
// or the dense view's sequence span exceeds the head pitch (L11).
TEST(TestGpuRaggedSdpaMapping, ModelRejectsAliasingLayouts)
{
    for(const char* id : {"L10", "L11"})
    {
        SCOPED_TRACE(id);
        const auto& layout = raggedLayoutCase(id);
        ASSERT_TRUE(layout.rejected);
        const auto seqLens = layout.seqLen.empty() ? std::vector<int64_t>{} : seqLensOf(layout);
        const auto model = raggedBatches(elementOffsetsOf(layout),
                                         layout.seqLen.empty() ? nullptr : &seqLens,
                                         layout.strides[SEQ_AXIS],
                                         layout.dims[SEQ_AXIS]);
        EXPECT_THROW(raggedFootprint(layout.dims, layout.strides, SEQ_AXIS, model),
                     std::invalid_argument);
    }
}

// Length and offset tensors with batch stride 2 give the rows of the packed ones; the slots
// between their values are never read.
TEST(TestGpuRaggedSdpaMapping, StridedAuxMatchesPacked)
{
    const auto& layout = raggedLayoutCase("L5s");
    const auto stridedLens = stridedAuxBuffer(layout.seqLen, 2);
    const auto stridedOffsets = stridedAuxBuffer(layout.offsets, 2);
    for(int64_t b = 0; b < layout.dims[0]; ++b)
    {
        const auto packed = raggedBatchRows(layout.offsets.data(),
                                            1,
                                            layout.seqLen.data(),
                                            1,
                                            b,
                                            layout.multiplier,
                                            layout.strides[SEQ_AXIS],
                                            layout.dims[SEQ_AXIS]);
        const auto rows = raggedBatchRows(stridedOffsets.data(),
                                          2,
                                          stridedLens.data(),
                                          2,
                                          b,
                                          layout.multiplier,
                                          layout.strides[SEQ_AXIS],
                                          layout.dims[SEQ_AXIS]);
        EXPECT_EQ(rows.origin, packed.origin) << "batch " << b;
        EXPECT_EQ(rows.length, packed.length) << "batch " << b;

        // Without lengths the block width comes from offsets b and b + 1.
        const auto packedBlock = raggedBatchRows(layout.offsets.data(),
                                                 1,
                                                 nullptr,
                                                 0,
                                                 b,
                                                 layout.multiplier,
                                                 layout.strides[SEQ_AXIS],
                                                 layout.dims[SEQ_AXIS]);
        const auto stridedBlock = raggedBatchRows(stridedOffsets.data(),
                                                  2,
                                                  nullptr,
                                                  0,
                                                  b,
                                                  layout.multiplier,
                                                  layout.strides[SEQ_AXIS],
                                                  layout.dims[SEQ_AXIS]);
        EXPECT_EQ(stridedBlock.origin, packedBlock.origin) << "batch " << b;
        EXPECT_EQ(stridedBlock.length, packedBlock.length) << "batch " << b;
    }
}
