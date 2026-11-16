// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Types.hpp>

#include "../kernel_ingestor_engine/FlydslSdpaFrontendGraphs.hpp"

/**
 * @file IntegrationGpuSdpaForwardRagged.cpp
 * @brief The ASM SDPA forward's group-mode (ragged, packed THD) kernels against hipDNN's CPU
 *        reference graph executor, which serves compact THD. IntegrationGpuSdpaForward only checks
 *        that group-mode graphs are supported: its harness does not fill the offset tables. Here
 *        the frontend-built lane (FlydslSdpaFrontendGraphs.hpp) fills them, pins the ASM engine
 *        and compares O, sequences of unequal query and key lengths included, so a bottom-right
 *        causal mask must sit on each sequence's own diagonal.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

struct AsmRaggedCase
{
    const char* name;
    FrontendSdpaSpec spec;
};

} // namespace

/// Each case pins the ASM engine, which needs no FlyDSL objects; the test skips by the ASM
/// kernels' own architectures.
template <>
struct LaneNeedsFlydsl<AsmRaggedCase> : std::false_type
{
};

namespace
{

using IntegrationGpuAsmSdpaForwardRagged = IntegrationGpuFlydslFrontendLane<AsmRaggedCase>;

std::string caseName(const ::testing::TestParamInfo<AsmRaggedCase>& info)
{
    return info.param.name;
}

/// bf16 THD with Q/O through one offset table and K/V through another, each tensor's token
/// stride as its multiplier (the AITER group-mode contract), no LSE; head dims 128 and 192/128,
/// bottom-right causal with @p causal, else no mask. The second sequence has more queries than
/// keys, so its first rows see no key under the bottom-right mask.
std::vector<AsmRaggedCase> asmRaggedCases(bool causal)
{
    std::vector<AsmRaggedCase> cases;
    for(const bool mla : {false, true})
    {
        FrontendSdpaSpec spec;
        spec.dataType = hipdnn_frontend::DataType::BFLOAT16;
        spec.batch = 3;
        spec.queryHeads = 8;
        spec.kvHeads = 8;
        spec.seqQ = 160;
        spec.seqKv = 200;
        spec.headSize = mla ? 192 : 128;
        if(mla)
        {
            spec.valueHeadSize = 128;
        }
        spec.packedLensQ = {67, 160, 33};
        spec.packedLensKv = {143, 40, 200};
        spec.withStats = false;
        spec.leftBound.reset();
        if(causal)
        {
            spec.rightBound = 0;
            spec.alignment = hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT;
        }
        else
        {
            spec.rightBound.reset();
        }
        spec.engine = std::string(hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME);
        const std::string name = std::string("asm_ragged_") + (mla ? "d192_v128" : "d128")
                                 + (causal ? "_causal_bottom_right" : "_no_mask");
        spec.name = name;
        const char* caseLabel = mla ? "D192V128NoMask" : "D128NoMask";
        if(causal)
        {
            caseLabel = mla ? "D192V128CausalBottomRight" : "D128CausalBottomRight";
        }
        cases.push_back({caseLabel, spec});
    }
    return cases;
}

} // namespace

TEST_P(IntegrationGpuAsmSdpaForwardRagged, MatchesTheCpuReference)
{
    if(_arch != "gfx942" && _arch != "gfx950")
    {
        GTEST_SKIP() << "the ASM group-mode kernels ship for gfx942 and gfx950 only";
    }
    auto spec = GetParam().spec;
    // The backend must offer the ASM engine here; the lane then pins it.
    spec.engineArch = _arch;
    auto sdpa = buildFrontendSdpaGraph(spec);
    validateServeAndVerify(sdpa, spec.dataType);
}

// Bottom-right causal (each sequence on its own diagonal) in every tier; no mask in the
// comprehensive tier.
INSTANTIATE_TEST_SUITE_P(Thd,
                         IntegrationGpuAsmSdpaForwardRagged,
                         ::testing::ValuesIn(asmRaggedCases(true)),
                         caseName);

INSTANTIATE_TEST_SUITE_P(ComprehensiveThd,
                         IntegrationGpuAsmSdpaForwardRagged,
                         ::testing::ValuesIn(asmRaggedCases(false)),
                         caseName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
