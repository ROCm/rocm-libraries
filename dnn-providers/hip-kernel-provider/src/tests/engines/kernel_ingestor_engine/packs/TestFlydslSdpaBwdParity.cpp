// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdParityCases.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdParity.cpp
 * @brief The cases of FlydslSdpaBwdParityCases.hpp, launched on a gfx11 device.
 *
 * A served case is launched on the device and every gradient is compared with the harness's
 * double-precision reference. It covers bottom-right causal cross-attention in both length
 * orders, windows narrower than a tile, left-only, to the right of the diagonal and wider than
 * the keys, V narrower and wider than Q/K, rows no key reaches (dQ is zero, nothing is NaN),
 * biases that mask whole rows at the lowest finite value or at -inf and leading keys at -inf,
 * head dim 256 and the generic head dims across the gfx11 rung boundaries, per-head and
 * query-dtype biases, grouped and multi-query attention, head-major, sequence-major and packed
 * QKV layouts with V, dO and the log-sum-exp through layouts of their own, and the scale as a
 * runtime tensor or as the default 1. A declined case is checked to be absent from the
 * manager's applicable engines and refused when the engine is selected by id.
 *
 * Forward features the backward contract does not have -- decode, sinks, the LSE output, MLA
 * absorption -- have no case here.
 *
 * The host half, which needs no device, is TestFlydslSdpaBwdParityGraphs.cpp: a census suite
 * over the installed gfx11 shard. This suite skips on any device that is not a gfx11 one, the
 * architecture the verdict table describes.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_parity;

// ---------------------------------------------------------------------------
// device: the cases, launched
// ---------------------------------------------------------------------------

/// The split engine on a gfx11 device: the architecture the verdict table describes.
class TestGpuFlydslSdpaBwdSplitParity : public FlydslSdpaBwdDevice,
                                        public ::testing::WithParamInterface<size_t>
{
protected:
    TestGpuFlydslSdpaBwdSplitParity()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }

    void SetUp() override
    {
        FlydslSdpaBwdDevice::SetUp();
        if(IsSkipped())
        {
            return;
        }
        const auto arch = packedArchDeviceProperties().gcnArchName;
        if(arch.rfind("gfx11", 0) != 0)
        {
            GTEST_SKIP() << "the parity cases follow the gfx11 verdict table; " << arch
                         << " is not a gfx11 device";
        }
    }
};

TEST_P(TestGpuFlydslSdpaBwdSplitParity, MatchesTheReferenceOrIsDeclined)
{
    const auto& parity = parityCases()[GetParam()];
    if(parity.run.has_value())
    {
        expectCase(SPLIT_ENGINE, *parity.run);
        return;
    }
    expectDeclined(SPLIT_ENGINE, parity.spec);
}

INSTANTIATE_TEST_SUITE_P(, TestGpuFlydslSdpaBwdSplitParity, parityIndices(), parityName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
