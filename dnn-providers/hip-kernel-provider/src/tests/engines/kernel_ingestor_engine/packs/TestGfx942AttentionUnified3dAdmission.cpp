// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

#include <gtest/gtest.h>

#include "Gfx942AttentionUnified3dTestGraphs.hpp"

/**
 * @file TestGfx942AttentionUnified3dAdmission.cpp
 * @brief The engine's graph match against rocKE's own strided decode admission.
 *
 * Gfx942AttentionUnified3dAdmissionTable.inc holds, per request, what
 * `dispatch.attention.strided_decode.make_candidate().admits(req)` answers at each
 * `kv_block_size` (16, 32, 64); gen_gfx942_attention_unified_3d_admission.py writes it and
 * documents the graph -> request mapping. The drift test
 * descriptor-packaging/tests/rocke/test_gfx942_admission_table_current.py keeps it current.
 *
 * For each row this builds the hipDNN graph and asserts:
 *
 *     graph match accepts  <=>  (rocKE admits at SOME kv_block_size) AND not windowed
 *
 * THE ONE DELIBERATE EXCEPTION is the window: rocKE admits a bottom-right sliding window,
 * this engine refuses every window at graph match in v1 (no shipped variant carries a
 * sliding_window, and the scorer's split rule holds only without one). Those rows are
 * asserted refused, and the test also asserts rocKE does admit some of them, so the
 * exception stays visible rather than vacuous.
 *
 * Then, for every row the graph match accepts, kernel match against a synthetic record
 * of the graph's cell at each block size accepts exactly when that block size's row was
 * admitted.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{
namespace
{

enum class AdmissionMask
{
    NO_MASK,
    TOP_LEFT_CAUSAL,
    BOTTOM_RIGHT_CAUSAL,
    WINDOW_128
};

struct AdmissionRow
{
    int64_t batch;
    int64_t hq;
    int64_t hkv;
    int64_t seqLenQ;
    int64_t seqLenKv;
    int64_t headSize;
    std::string_view dtype;
    AdmissionMask mask;
    bool admitted16;
    bool admitted32;
    bool admitted64;
};

constexpr AdmissionRow ADMISSION_TABLE[] = {
#include "Gfx942AttentionUnified3dAdmissionTable.inc"
};

Mask maskFor(AdmissionMask mask)
{
    switch(mask)
    {
    case AdmissionMask::NO_MASK:
        return Mask::NO_MASK;
    case AdmissionMask::TOP_LEFT_CAUSAL:
        return Mask::TOP_LEFT_CAUSAL;
    case AdmissionMask::BOTTOM_RIGHT_CAUSAL:
        return Mask::BOTTOM_RIGHT_CAUSAL;
    case AdmissionMask::WINDOW_128:
    default:
        return Mask::WINDOW_128;
    }
}

GraphSpec graphFor(const AdmissionRow& row)
{
    GraphSpec spec;
    spec.batch = row.batch;
    spec.numQueryHeads = row.hq;
    spec.numKvHeads = row.hkv;
    spec.seqLenQ = row.seqLenQ;
    spec.seqLenKv = row.seqLenKv;
    spec.headSize = row.headSize;
    spec.dataType
        = row.dtype == "bf16" ? data_objects::DataType::BFLOAT16 : data_objects::DataType::HALF;
    spec.mask = maskFor(row.mask);
    // Contiguous Q/O for the row's own Sq (the Sq = 2 rows are refused on Sq, not layout).
    spec.qStrides = std::vector<int64_t>{
        row.seqLenQ * row.hq * row.headSize, row.headSize, row.hq * row.headSize, 1};
    spec.oStrides = spec.qStrides;
    return spec;
}

std::string describe(const AdmissionRow& row)
{
    return "B" + std::to_string(row.batch) + " Hq" + std::to_string(row.hq) + " Hkv"
           + std::to_string(row.hkv) + " Sq" + std::to_string(row.seqLenQ) + " Skv"
           + std::to_string(row.seqLenKv) + " D" + std::to_string(row.headSize) + " "
           + std::string(row.dtype) + " mask " + std::to_string(static_cast<int>(row.mask));
}

TEST(TestGfx942AttentionUnified3dAdmission, TableCoversThePlannedGrid)
{
    // 3 batches x 5 head pairs x 2 Sq x 5 Skv x 4 D x 2 dtypes x 4 masks.
    EXPECT_EQ(std::size(ADMISSION_TABLE), 3U * 5U * 2U * 5U * 4U * 2U * 4U);
}

TEST(TestGfx942AttentionUnified3dAdmission, GraphMatchAgreesWithRocKeExceptForWindows)
{
    std::size_t accepted = 0;
    std::size_t windowedAdmittedByRocKe = 0;
    for(const auto& row : ADMISSION_TABLE)
    {
        const bool rocKeAdmits = row.admitted16 || row.admitted32 || row.admitted64;
        const bool windowed = row.mask == AdmissionMask::WINDOW_128;
        if(windowed && rocKeAdmits)
        {
            ++windowedAdmittedByRocKe;
        }
        // v1 exception: every window is refused, whatever rocKE says.
        const bool expected = rocKeAdmits && !windowed;
        const bool actual = graphMatches(graphFor(row));
        EXPECT_EQ(actual, expected) << describe(row) << " (rocKE admits: " << rocKeAdmits << ")";
        accepted += actual ? 1 : 0;
    }
    EXPECT_GT(accepted, 0U) << "an engine that refuses every row passes nothing";
    EXPECT_GT(windowedAdmittedByRocKe, 0U)
        << "rocKE no longer admits any windowed row: the documented window exception is "
           "vacuous and this test should be revisited";
}

TEST(TestGfx942AttentionUnified3dAdmission, KernelMatchAgreesPerBlockSize)
{
    for(const auto& row : ADMISSION_TABLE)
    {
        const auto spec = graphFor(row);
        if(!graphMatches(spec))
        {
            continue;
        }
        const bool admitted[] = {row.admitted16, row.admitted32, row.admitted64};
        const int64_t blocks[] = {16, 32, 64};
        for(std::size_t i = 0; i < std::size(blocks); ++i)
        {
            KernelSpec kernel;
            kernel.dtype = row.dtype == "bf16" ? "BF16" : "FP16";
            kernel.headSize = row.headSize;
            kernel.numQueryHeads = row.hq;
            kernel.numKvHeads = row.hkv;
            kernel.blockSize = blocks[i];
            EXPECT_EQ(kernelMatchesGraph(spec, kernel), admitted[i])
                << describe(row) << " block " << blocks[i];
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
