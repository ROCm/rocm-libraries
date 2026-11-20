// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "Gfx942AttentionUnified3dTestGraphs.hpp"

/**
 * @file TestGfx942AttentionUnified3dAdmission.cpp
 * @brief The engine's graph match against rocKE's own strided decode admission.
 *
 * Gfx942AttentionUnified3dAdmissionTable.inc holds, per request, what
 * `dispatch.attention.strided_decode.make_candidate().admits(req)` answers at each
 * `kv_block_size` (16, 32, 64); gen_gfx942_attention_unified_3d_admission.py writes it and
 * documents the graph -> request mapping. The build generates it from the rocKE source tree
 * (kernel_ingestor_engine/CMakeLists.txt), so a change to rocKE's admission fails here.
 *
 * For each row this builds the hipDNN graph and asserts:
 *
 *     graph match accepts  <=>  (rocKE admits at SOME kv_block_size) AND not windowed
 *                               AND the v1 catalog ships the shape
 *
 * THE DELIBERATE EXCEPTIONS are v1 scope, refused at graph match whatever rocKE says:
 *
 *  - the window: rocKE admits a bottom-right sliding window; no shipped variant carries a
 *    sliding_window, and the scorer's split rule holds only without one;
 *  - head sizes other than 64, 128 and 256: rocKE also admits 192;
 *  - GQA ratios outside {1, 2, 4, 8, 16}: rocKE admits every ratio from 1 to 16.
 *
 * Those rows are asserted refused, and the test also asserts rocKE does admit some of each,
 * so every exception stays visible rather than vacuous.
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

const std::vector<AdmissionRow> ADMISSION_TABLE = {
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
    // 3 batches x 7 head pairs x 2 Sq x 5 Skv x 5 D x 2 dtypes x 4 masks.
    EXPECT_EQ(std::size(ADMISSION_TABLE), 3U * 7U * 2U * 5U * 5U * 2U * 4U);
}

/// The head sizes and GQA ratios the v1 catalog ships.
bool shippedByV1(const AdmissionRow& row)
{
    const bool headSize = row.headSize == 64 || row.headSize == 128 || row.headSize == 256;
    const int64_t ratio = row.hq % row.hkv == 0 ? row.hq / row.hkv : 0;
    const bool gqa = ratio == 1 || ratio == 2 || ratio == 4 || ratio == 8 || ratio == 16;
    return headSize && gqa;
}

TEST(TestGfx942AttentionUnified3dAdmission, GraphMatchAgreesWithRocKeExceptForV1Scope)
{
    std::size_t accepted = 0;
    std::size_t windowedAdmittedByRocKe = 0;
    std::size_t unshippedAdmittedByRocKe = 0;
    for(const auto& row : ADMISSION_TABLE)
    {
        const bool rocKeAdmits = row.admitted16 || row.admitted32 || row.admitted64;
        const bool windowed = row.mask == AdmissionMask::WINDOW_128;
        const bool shipped = shippedByV1(row);
        if(windowed && rocKeAdmits)
        {
            ++windowedAdmittedByRocKe;
        }
        if(!windowed && !shipped && rocKeAdmits)
        {
            ++unshippedAdmittedByRocKe;
        }
        // v1 exceptions: every window and every unshipped shape is refused.
        const bool expected = rocKeAdmits && !windowed && shipped;
        const bool actual = graphMatches(graphFor(row));
        EXPECT_EQ(actual, expected) << describe(row) << " (rocKE admits: " << rocKeAdmits << ")";
        accepted += actual ? 1 : 0;
    }
    EXPECT_GT(accepted, 0U) << "an engine that refuses every row passes nothing";
    EXPECT_GT(windowedAdmittedByRocKe, 0U)
        << "rocKE no longer admits any windowed row: the documented window exception is "
           "vacuous and this test should be revisited";
    EXPECT_GT(unshippedAdmittedByRocKe, 0U)
        << "rocKE no longer admits any head size or GQA ratio outside the v1 catalog: the "
           "documented scope exception is vacuous and this test should be revisited";
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
        const std::array<bool, 3> admitted = {row.admitted16, row.admitted32, row.admitted64};
        const std::array<int64_t, 3> blocks = {16, 32, 64};
        for(std::size_t i = 0; i < blocks.size(); ++i)
        {
            KernelSpec kernel;
            kernel.dtype = row.dtype == "bf16" ? "BF16" : "FP16";
            kernel.headSize = row.headSize;
            // graphMatches passed, so hkv divides hq with a supported ratio.
            kernel.numQueriesPerKv = row.hq / row.hkv;
            kernel.blockSize = blocks[i];
            EXPECT_EQ(kernelMatchesGraph(spec, kernel), admitted[i])
                << describe(row) << " block " << blocks[i];
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
