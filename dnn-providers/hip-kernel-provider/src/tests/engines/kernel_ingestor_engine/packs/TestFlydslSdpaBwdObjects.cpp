// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"

/**
 * @file TestFlydslSdpaBwdObjects.cpp
 * @brief The shipped hipkernel:flydsl_sdpa_bwd_merged and hipkernel:flydsl_sdpa_bwd_split
 *        objects, run end to end on the device and checked against a host reference.
 *
 * Each case builds one SDPA backward graph and runs it through the harness of
 * FlydslSdpaBwdDeviceHarness.hpp: the engine manager is asked whether the engine under test
 * claims it, that engine's plan is built (the manager takes the engine by id, so no other
 * engine can serve the graph) and executed on deterministic inputs, and dQ, dK, dV and, on a
 * bias graph, dBias are compared with a double-precision backward pass over the same inputs. A
 * graph the engine must decline is checked to be absent from the manager's applicable engines
 * and refused when the engine is selected by id.
 *
 * The cases run on every device whose architecture has objects in the install: gfx950, and a
 * gfx11 device served by the gfx11-generic content. They skip on any other device and on an
 * install that staged no shard for the device (flydslShipsForThisDevice). On a device the
 * install ships them for, a missing object fails. The dropout reference draws from the
 * 64-bit Philox stream on both architectures. The forward-parity cases are
 * TestFlydslSdpaBwdParity.cpp; the per-object suite is TestFlydslSdpaBwdObjectsPerObject.cpp.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

/// Skips the running test unless the device is a gfx950; a test already skipped stays as it is.
void skipUnlessGfx950()
{
    if(::testing::Test::IsSkipped())
    {
        return;
    }
    const auto arch = packedArchDeviceProperties().gcnArchName;
    if(arch.rfind("gfx950", 0) != 0)
    {
        GTEST_SKIP() << "the lowest-finite bias cases run on a gfx950 device; " << arch
                     << " is covered by the gfx11 parity suite or ships no such engine";
    }
}

class TestGpuFlydslSdpaBwdMergedDispatch : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedDispatch()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdSplitDispatch : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitDispatch()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdSplitDispatchFp32 : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitDispatchFp32()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

/// A bias that masks whole rows at the lowest finite value of its dtype. The cases run on a
/// gfx950 device only: the gfx11 split kernels' lowest-finite bias cases are in the gfx11
/// parity suite (TestFlydslSdpaBwdParity.cpp), and no other architecture ships these engines.
class TestGpuFlydslSdpaBwdMergedLowestFiniteBias : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedLowestFiniteBias()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }

    void SetUp() override
    {
        FlydslSdpaBwdDevice::SetUp();
        skipUnlessGfx950();
    }
};

class TestGpuFlydslSdpaBwdSplitLowestFiniteBias : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitLowestFiniteBias()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }

    void SetUp() override
    {
        FlydslSdpaBwdDevice::SetUp();
        skipUnlessGfx950();
    }
};

// One suite per dtype: the datatype token ends the suite name.
using TestGpuFlydslSdpaBwdMergedLowestFiniteBiasBfp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedLowestFiniteBiasFp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasBfp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp32 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

BwdCase plainCase(DataType dtype)
{
    BwdCase c;
    c.dtype = dtype;
    return c;
}

BwdCase causalCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.headDim = 128;
    c.mask = BwdMask::TOP_LEFT;
    return c;
}

BwdCase biasCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.bias = true;
    return c;
}

BwdCase dropoutCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.dropout = 0.2F;
    return c;
}

BwdCase groupedCase(DataType dtype)
{
    BwdCase c = causalCase(dtype);
    c.heads = 8;
    c.kvHeads = 2;
    return c;
}

BwdCase paddedCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.lengthsQ = {96, 70};
    c.lengthsKv = {96, 50};
    return c;
}

/// Dropout over a query length that differs from the key length, the key length not a multiple
/// of 4: a Philox row holds a partial 4-wide draw.
BwdCase raggedDropoutCase(DataType dtype, bool causal)
{
    BwdCase c = dropoutCase(dtype);
    c.seqQ = 70;
    c.seqKv = 45;
    c.mask = causal ? BwdMask::TOP_LEFT : BwdMask::NONE;
    return c;
}

/// A bias, and its gradient unless @p dbias is false, broadcast over the named axes.
BwdCase broadcastBiasCase(DataType dtype, bool overBatch, bool overHead, bool dbias = true)
{
    BwdCase c = biasCase(dtype);
    c.biasOverBatch = overBatch;
    c.biasOverHead = overHead;
    c.dbias = dbias;
    return c;
}

/// A head dim that is not a power of two; the rows are a whole vector apart.
BwdCase headDimCase(DataType dtype, int64_t headDim)
{
    BwdCase c = plainCase(dtype);
    c.headDim = headDim;
    return c;
}

/// A head dim below the tile with a non-finite value in the padding of O's rows (the
/// elements between the head dim and the row pitch). The merged dK/dV role reads O for Delta
/// in vectors that reach into that padding and must not let it through: a Delta that read it
/// would turn dK NaN, which the comparison counts as a miss.
BwdCase nonFiniteOPaddingCase(DataType dtype, int64_t headDim, BwdOPadFill fill)
{
    BwdCase c = headDimCase(dtype, headDim);
    c.oPadFill = fill;
    return c;
}

/// Every key of every third row of a bias at the dtype's lowest finite value, on the shape of
/// the gfx11 parity case of the same name.
BwdCase lowestFiniteBiasCase(DataType dtype)
{
    BwdCase c = biasCase(dtype);
    c.seqQ = 200;
    c.seqKv = 300;
    c.biasFill = BwdBiasFill::LOWEST_ROWS;
    return c;
}

/// Bias beside a padding mask is outside both engines' contract.
SdpaBwdSpec biasWithPaddingSpec()
{
    return SdpaBwdSpec{}.withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addSeqLens(g);
    });
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, PlainMatchesTheReference)
{
    expectCase(MERGED_ENGINE, plainCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, CausalMatchesTheReference)
{
    expectCase(MERGED_ENGINE, causalCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, BiasAndDbiasMatchTheReference)
{
    expectCase(MERGED_ENGINE, biasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, DropoutMatchesTheReference)
{
    expectCase(MERGED_ENGINE, dropoutCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, GroupedQueryMatchesTheReference)
{
    expectCase(MERGED_ENGINE, groupedCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, PaddedVarlenMatchesTheReference)
{
    expectCase(MERGED_ENGINE, paddedCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, HeadDimOf23IgnoresNaNInOsPadding)
{
    expectCase(MERGED_ENGINE,
               nonFiniteOPaddingCase(DataType::BFLOAT16, 23, BwdOPadFill::NAN_VALUE));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, HeadDimOf7IgnoresInfinityInOsPadding)
{
    expectCase(MERGED_ENGINE, nonFiniteOPaddingCase(DataType::HALF, 7, BwdOPadFill::POS_INF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, DeclinesAFullPrecisionGraph)
{
    expectDeclined(MERGED_ENGINE, fp32Spec());
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PlainMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, plainCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, CausalMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, causalCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, biasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DropoutMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, dropoutCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, GroupedQueryMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, groupedCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PaddedVarlenMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, paddedCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, PlainMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, plainCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, CausalMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, causalCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, biasCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::HALF, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, CausalDropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::BFLOAT16, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverBatchMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::BFLOAT16, true, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::HALF, false, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverBatchAndHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::BFLOAT16, true, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::HALF, false, false, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, HeadDimOf72MatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::BFLOAT16, 72));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, HeadDimOf77AtAPaddedRowStrideMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::HALF, 77));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, DropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::FLOAT, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasBroadcastOverBatchAndHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::FLOAT, true, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::FLOAT, false, false, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, HeadDimOf37AtAPaddedRowStrideMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::FLOAT, 37));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DeclinesBiasBesideAPaddingMask)
{
    expectDeclined(SPLIT_ENGINE, biasWithPaddingSpec());
}

TEST_F(TestGpuFlydslSdpaBwdMergedLowestFiniteBiasBfp16, MatchesTheReference)
{
    expectCase(MERGED_ENGINE, lowestFiniteBiasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedLowestFiniteBiasFp16, MatchesTheReference)
{
    expectCase(MERGED_ENGINE, lowestFiniteBiasCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitLowestFiniteBiasBfp16, MatchesTheReference)
{
    expectCase(SPLIT_ENGINE, lowestFiniteBiasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp16, MatchesTheReference)
{
    expectCase(SPLIT_ENGINE, lowestFiniteBiasCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp32, MatchesTheReference)
{
    expectCase(SPLIT_ENGINE, lowestFiniteBiasCase(DataType::FLOAT));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
