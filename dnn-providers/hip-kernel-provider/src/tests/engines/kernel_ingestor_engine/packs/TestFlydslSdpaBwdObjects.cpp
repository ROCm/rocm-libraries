// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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
        GTEST_SKIP() << "the bias edge-value cases run on a gfx950 device; " << arch
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

/// Bottom-right causal graphs with more queries than keys, so the leading query rows see no key.
/// gfx950 only, as the lowest-finite bias cases: the gfx11 split kernels' keyless-row cases are in the
/// parity suite (TestFlydslSdpaBwdParity.cpp).
class TestGpuFlydslSdpaBwdMergedKeylessRows : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedKeylessRows()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }

    void SetUp() override
    {
        FlydslSdpaBwdDevice::SetUp();
        skipUnlessGfx950();
    }
};

class TestGpuFlydslSdpaBwdSplitKeylessRows : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitKeylessRows()
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
using TestGpuFlydslSdpaBwdMergedKeylessRowsBfp16 = TestGpuFlydslSdpaBwdMergedKeylessRows;
using TestGpuFlydslSdpaBwdSplitKeylessRowsBfp16 = TestGpuFlydslSdpaBwdSplitKeylessRows;
using TestGpuFlydslSdpaBwdSplitKeylessRowsFp16 = TestGpuFlydslSdpaBwdSplitKeylessRows;
using TestGpuFlydslSdpaBwdSplitKeylessRowsFp32 = TestGpuFlydslSdpaBwdSplitKeylessRows;
using TestGpuFlydslSdpaBwdMergedLowestFiniteBiasBfp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedLowestFiniteBiasFp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasBfp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLowestFiniteBiasFp32 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedNanBiasBfp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedNanBiasFp16 = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitNanBiasBfp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitNanBiasFp16 = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedLargeRowBroadcastBiasBfp16
    = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdMergedLargeRowBroadcastBiasFp16
    = TestGpuFlydslSdpaBwdMergedLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasBfp16
    = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasFp16
    = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasF32Bfp16
    = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;
using TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasF32Fp16
    = TestGpuFlydslSdpaBwdSplitLowestFiniteBias;

/// A FLOAT bias beside a 16-bit Q on the split engine, on every device that ships it.
class TestGpuFlydslSdpaBwdSplitF32Bias : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitF32Bias()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

/// A bias whose rows are not a whole 16-byte vector apart, or that has one row for every query.
class TestGpuFlydslSdpaBwdSplitBiasPitch : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitBiasPitch()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

using TestGpuFlydslSdpaBwdSplitF32BiasBfp16 = TestGpuFlydslSdpaBwdSplitF32Bias;
using TestGpuFlydslSdpaBwdSplitF32BiasFp16 = TestGpuFlydslSdpaBwdSplitF32Bias;
using TestGpuFlydslSdpaBwdSplitBiasPitchBfp16 = TestGpuFlydslSdpaBwdSplitBiasPitch;
using TestGpuFlydslSdpaBwdSplitBiasPitchFp16 = TestGpuFlydslSdpaBwdSplitBiasPitch;

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

/// A NaN in the bias at query row 5, key 7 of every batch and head, over finite scores everywhere
/// else, with a log-sum-exp and output that come from the clean bias. The NaN must reach P: a
/// dK/dV exponent clamp that turns a NaN exponent into 0 gives P = 1 there and finite dK and dV
/// at key 7, where the reference is NaN. The graph is biasCase's (a dense bias with its gradient).
BwdCase nanBiasCase(DataType dtype)
{
    BwdCase c = biasCase(dtype);
    c.nanBiasAt = std::pair<int64_t, int64_t>{5, 7};
    return c;
}

/// A bottom-right causal graph with Sq > Skv: query row r sees keys up to r + Skv - Sq, so the first
/// Sq - Skv = 160 rows (two and a half 64-row tiles) see no key and the next ones see 1..96. The head dim and
/// class are those of causalCase: the object selection does not depend on the diagonal, so the
/// same shipped rows serve both.
BwdCase keylessRowsCase(DataType dtype)
{
    BwdCase c = causalCase(dtype);
    c.mask = BwdMask::BOTTOM_RIGHT;
    c.seqQ = 256;
    c.seqKv = 96;
    c.keylessDqExactlyZero = true;
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

/// A FLOAT bias beside a 16-bit Q, and its FLOAT gradient unless @p dbias is false. The edge
/// fills mask every third row, on the shape of lowestFiniteBiasCase; the lowest-finite fill makes
/// P = 1 in those rows, so their gradients are far larger than the ordinary rows' and each row is
/// held to its own bound.
BwdCase f32BiasCase(DataType dtype, bool dbias, BwdBiasFill fill = BwdBiasFill::VALUES)
{
    BwdCase c = biasCase(dtype);
    c.biasType = DataType::FLOAT;
    c.dbias = dbias;
    c.biasFill = fill;
    c.perRowTolerance = fill == BwdBiasFill::LOWEST_ROWS;
    if(fill != BwdBiasFill::VALUES)
    {
        c.seqQ = 200;
        c.seqKv = 300;
    }
    return c;
}

/// A bias, without a gradient unless @p dbias, whose rows are @p pitch elements apart (a whole
/// 16-byte vector when 0) over @p seqKv keys.
BwdCase pitchedBiasCase(
    DataType dtype, int64_t headDim, int64_t seqKv, int64_t pitch, bool dbias = false)
{
    BwdCase c = biasCase(dtype);
    c.batch = 1;
    c.heads = 2;
    c.kvHeads = 2;
    c.seqQ = 64;
    c.seqKv = seqKv;
    c.headDim = headDim;
    c.biasPitch = pitch;
    c.dbias = dbias;
    return c;
}

/// A bias of one row for every query row, packed with no padding behind that row, over a key
/// length that is not a multiple of 8: a staged load of its last 16-byte chunk would leave the
/// allocation.
BwdCase rowBroadcastBiasCase(DataType dtype, int64_t seqKv)
{
    BwdCase c = biasCase(dtype);
    c.seqKv = seqKv;
    c.biasOverRows = true;
    c.biasPitch = seqKv;
    c.dbias = false;
    return c;
}

/// A row-broadcast bias (one row for every query row) of one large constant @p value, over
/// @p seqQ query rows that are no multiple of any kernel tile. The padded rows of the last tile
/// read the same bias row beside a log-sum-exp of zero, so a kernel that forms P = exp(bias) there
/// overflows the 16-bit P (f16 from a bias of about 11, bf16 from about 89) and turns dV NaN.
/// @p biasType FLOAT makes the bias an f32 one beside the 16-bit Q/K/V, which the dK/dV kernel
/// loads through its own f32-bias path.
BwdCase largeRowBroadcastBiasCase(DataType dtype,
                                  double value,
                                  int64_t seqQ,
                                  std::optional<DataType> biasType = std::nullopt)
{
    BwdCase c = rowBroadcastBiasCase(dtype, 64);
    c.biasType = biasType;
    c.seqQ = seqQ;
    c.biasFill = BwdBiasFill::CONSTANT;
    c.biasValue = value;
    return c;
}

/// The same bias on Q = K = 0 and V = dO = 1, head dim 16, one query row: with @p seqKv 1 the
/// graph of AOTriton's bias symmetry test, whose 1 x 1 bias is a row broadcast.
BwdCase zeroScoresBroadcastBiasCase(DataType dtype, double value, int64_t seqKv)
{
    BwdCase c = rowBroadcastBiasCase(dtype, seqKv);
    c.batch = 1;
    c.heads = 1;
    c.kvHeads = 1;
    c.seqQ = 1;
    c.headDim = 16;
    c.data = BwdData::ZERO_SCORES;
    c.biasFill = BwdBiasFill::CONSTANT;
    c.biasValue = value;
    return c;
}

/// Every cell of the large-bias family for @p engine: the merged engine declines a key length of 1.
void expectLargeRowBroadcastBias(std::string_view engine,
                                 DataType dtype,
                                 double value,
                                 std::optional<DataType> biasType = std::nullopt)
{
    for(const int64_t seqQ : {100, 1})
    {
        SCOPED_TRACE("Sq " + std::to_string(seqQ));
        expectCase(engine, largeRowBroadcastBiasCase(dtype, value, seqQ, biasType));
    }
    if(biasType.has_value())
    {
        // The zero-scores graph has head dim 16, which no f32-bias object serves.
        return;
    }
    SCOPED_TRACE("zero scores");
    expectCase(engine, zeroScoresBroadcastBiasCase(dtype, value, engine == SPLIT_ENGINE ? 1 : 64));
}

/// Hq/Hk/Hv query, key and value heads: V's heads differ from K's, so the split engine runs
/// its value-heads objects.
struct ValueHeadsParam
{
    DataType dtype;
    int64_t heads;
    int64_t kvHeads;
    int64_t valueHeads;
    bool causal;
    float dropout;
    int64_t headDim = 64;
};

BwdCase valueHeadsCase(const ValueHeadsParam& p)
{
    BwdCase c = plainCase(p.dtype);
    c.heads = p.heads;
    c.kvHeads = p.kvHeads;
    c.valueHeads = p.valueHeads;
    c.headDim = p.headDim;
    c.mask = p.causal ? BwdMask::TOP_LEFT : BwdMask::NONE;
    c.dropout = p.dropout;
    return c;
}

/// A padded batch over V's own heads: every element of dQ, dK and dV past a batch's lengths is
/// zero.
BwdCase paddedValueHeadsCase(DataType dtype)
{
    BwdCase c = paddedCase(dtype);
    c.heads = 8;
    c.kvHeads = 4;
    c.valueHeads = 2;
    c.tailRowsZero = true;
    return c;
}

/// Dropout over V's own heads whose Philox counters cross 2^31: the stream's offset starts
/// 70,000 below it and the 64 planes of 2 x 32 heads draw 2,304 counters each, so the plane
/// that holds 2^31 is the 31st and every later one lies past it, in both launches of the
/// dK-only/dV-only pair.
BwdCase philoxPastTwoToTheThirtyFirstCase()
{
    BwdCase c = valueHeadsCase({DataType::BFLOAT16, 32, 8, 4, true, 0.2F});
    c.offset = (int64_t{1} << 31) - 70000;
    return c;
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

TEST_F(TestGpuFlydslSdpaBwdMergedNanBiasBfp16, PropagatesToDkAndDv)
{
    expectCase(MERGED_ENGINE, nanBiasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedNanBiasFp16, PropagatesToDkAndDv)
{
    expectCase(MERGED_ENGINE, nanBiasCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitNanBiasBfp16, PropagatesToDkAndDv)
{
    expectCase(SPLIT_ENGINE, nanBiasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitNanBiasFp16, PropagatesToDkAndDv)
{
    expectCase(SPLIT_ENGINE, nanBiasCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedLargeRowBroadcastBiasFp16, GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(MERGED_ENGINE, DataType::HALF, 16.0);
    expectLargeRowBroadcastBias(MERGED_ENGINE, DataType::HALF, 64.0);
}

TEST_F(TestGpuFlydslSdpaBwdMergedLargeRowBroadcastBiasBfp16, GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(MERGED_ENGINE, DataType::BFLOAT16, 90.0);
}

TEST_F(TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasFp16, GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::HALF, 16.0);
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::HALF, 64.0);
}

TEST_F(TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasBfp16, GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::BFLOAT16, 90.0);
}

TEST_F(TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasF32Fp16,
       GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::HALF, 16.0, DataType::FLOAT);
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::HALF, 64.0, DataType::FLOAT);
}

TEST_F(TestGpuFlydslSdpaBwdSplitLargeRowBroadcastBiasF32Bfp16,
       GradientsAreFiniteAndMatchTheReference)
{
    expectLargeRowBroadcastBias(SPLIT_ENGINE, DataType::BFLOAT16, 90.0, DataType::FLOAT);
}

TEST_F(TestGpuFlydslSdpaBwdMergedKeylessRowsBfp16, DqIsZeroAndGradientsMatchTheReference)
{
    expectCase(MERGED_ENGINE, keylessRowsCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitKeylessRowsBfp16, DqIsZeroAndGradientsMatchTheReference)
{
    expectCase(SPLIT_ENGINE, keylessRowsCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitKeylessRowsFp16, DqIsZeroAndGradientsMatchTheReference)
{
    expectCase(SPLIT_ENGINE, keylessRowsCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitKeylessRowsFp32, DqIsZeroAndGradientsMatchTheReference)
{
    expectCase(SPLIT_ENGINE, keylessRowsCase(DataType::FLOAT));
}

// ---------------------------------------------------------------------------
// FLOAT bias beside a 16-bit Q
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16,
       BiasAndDbiasBroadcastOverBatchAndHeadsMatchTheReference)
{
    auto c = f32BiasCase(DataType::BFLOAT16, true);
    c.biasOverBatch = true;
    c.biasOverHead = true;
    expectCase(SPLIT_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, LowestFiniteRowsWithDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, true, BwdBiasFill::LOWEST_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, LowestFiniteRowsWithoutDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, false, BwdBiasFill::LOWEST_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, NegativeInfinityRowsWithDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, true, BwdBiasFill::NEG_INF_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasBfp16, NegativeInfinityRowsWithoutDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::BFLOAT16, false, BwdBiasFill::NEG_INF_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16,
       BiasAndDbiasBroadcastOverBatchAndHeadsMatchTheReference)
{
    auto c = f32BiasCase(DataType::HALF, true);
    c.biasOverBatch = true;
    c.biasOverHead = true;
    expectCase(SPLIT_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, LowestFiniteRowsWithDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, true, BwdBiasFill::LOWEST_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, LowestFiniteRowsWithoutDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, false, BwdBiasFill::LOWEST_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, NegativeInfinityRowsWithDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, true, BwdBiasFill::NEG_INF_ROWS));
}

TEST_F(TestGpuFlydslSdpaBwdSplitF32BiasFp16, NegativeInfinityRowsWithoutDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, f32BiasCase(DataType::HALF, false, BwdBiasFill::NEG_INF_ROWS));
}

// ---------------------------------------------------------------------------
// Bias row pitches: odd 2-byte aligned (157 elements), 20 bytes (10 elements) and
// 12,808 bytes (6,404 elements); no gradient, as a dBias whose rows are not a whole vector
// apart is declined (BI-9). The aligned control keeps its dBias.
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchFp16, OddPitchMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::HALF, 128, 157, 157));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchFp16, PitchOf20BytesMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::HALF, 128, 10, 10));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchFp16, PitchOf12808BytesMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::HALF, 128, 6400, 6404));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchFp16, AlignedPitchWithRaggedKeysMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::HALF, 128, 157, 0, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchBfp16, OddPitchMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::BFLOAT16, 64, 157, 157));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchBfp16, PitchOf20BytesMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::BFLOAT16, 64, 10, 10));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchBfp16, PitchOf12808BytesMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::BFLOAT16, 64, 6400, 6404));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchBfp16, AlignedPitchWithRaggedKeysMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, pitchedBiasCase(DataType::BFLOAT16, 64, 157, 0, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchFp16, RowBroadcastBiasOverRaggedKeysMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, rowBroadcastBiasCase(DataType::HALF, 100));
}

TEST_F(TestGpuFlydslSdpaBwdSplitBiasPitchBfp16, RowBroadcastBiasOverRaggedKeysMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, rowBroadcastBiasCase(DataType::BFLOAT16, 157));
}

// ---------------------------------------------------------------------------
// Gradients for a V whose heads differ from K's
// ---------------------------------------------------------------------------

class TestGpuFlydslSdpaBwdSplitValueHeads : public FlydslSdpaBwdDevice,
                                            public ::testing::WithParamInterface<ValueHeadsParam>
{
protected:
    TestGpuFlydslSdpaBwdSplitValueHeads()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

TEST_P(TestGpuFlydslSdpaBwdSplitValueHeads, ComputesGradientsForDistinctKeyAndValueHeads)
{
    expectCase(SPLIT_ENGINE, valueHeadsCase(GetParam()));
}

std::string valueHeadsName(const ::testing::TestParamInfo<ValueHeadsParam>& info)
{
    const auto& p = info.param;
    return std::string(p.dtype == DataType::HALF ? "Fp16" : "Bfp16") + "Hq"
           + std::to_string(p.heads) + "Hk" + std::to_string(p.kvHeads) + "Hv"
           + std::to_string(p.valueHeads) + "D" + std::to_string(p.headDim)
           + (p.causal ? "Causal" : "Plain") + (p.dropout > 0.0F ? "Dropout" : "");
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuFlydslSdpaBwdSplitValueHeads,
    ::testing::Values(ValueHeadsParam{DataType::BFLOAT16, 32, 8, 4, false, 0.0F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 8, 4, true, 0.0F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 4, 8, false, 0.0F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 4, 8, true, 0.0F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 8, 4, false, 0.2F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 8, 4, true, 0.2F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 4, 8, false, 0.2F},
                      ValueHeadsParam{DataType::BFLOAT16, 32, 4, 8, true, 0.2F},
                      ValueHeadsParam{DataType::HALF, 32, 8, 4, true, 0.2F},
                      ValueHeadsParam{DataType::HALF, 32, 4, 8, false, 0.2F},
                      ValueHeadsParam{DataType::HALF, 32, 8, 4, true, 0.2F, 128},
                      ValueHeadsParam{DataType::BFLOAT16, 24, 8, 6, false, 0.2F},
                      ValueHeadsParam{DataType::HALF, 24, 8, 6, true, 0.0F}),
    valueHeadsName);

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PaddedBatchWithDistinctKeyAndValueHeadsZeroesItsTails)
{
    expectCase(SPLIT_ENGINE, paddedValueHeadsCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PaddedBatchWithDistinctKeyAndValueHeadsZeroesItsTailsBf16)
{
    expectCase(SPLIT_ENGINE, paddedValueHeadsCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DropoutPlanePastTwoToTheThirtyFirstMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, philoxPastTwoToTheThirtyFirstCase());
}

// ---------------------------------------------------------------------------
// host: the harness binds a FLOAT bias beside a 16-bit Q
// ---------------------------------------------------------------------------

/// A FLOAT bias beside a HALF Q is a FLOAT tensor in the graph, is staged in the buffer at its own
/// width and keeps its lowest finite value through the reference: a bias rounded to Q's dtype
/// would turn that value into -inf, and the rows it masks into rows no key reaches.
TEST(TestFlydslSdpaBwdHarness, BindsAFloatBiasBesideAHalfQueryAtItsOwnDtype)
{
    auto c = f32BiasCase(DataType::HALF, true, BwdBiasFill::LOWEST_ROWS);
    c.batch = 1;
    c.heads = 2;
    c.kvHeads = 2;
    c.seqQ = 6;
    c.seqKv = 30;

    const auto spec = specOf(c);
    auto graph = makeSdpaBwdGraph(spec);
    for(const auto& edit : spec.edits)
    {
        edit(graph);
    }
    EXPECT_EQ(graph.tensor("q").dtype, DataType::HALF);
    EXPECT_EQ(graph.tensor("bias").dtype, DataType::FLOAT);
    EXPECT_EQ(graph.tensor("dbias").dtype, DataType::FLOAT);

    constexpr auto LOWEST = std::numeric_limits<float>::lowest();
    const auto dense = biasInputsOf<float>(c);
    ASSERT_EQ(dense.size(), c.biasElements());
    // Rows 0 and 3 of every head are masked; row 1 is noise.
    EXPECT_EQ(dense[at(0, 0, 0, 5, 2, 6, 30)], static_cast<double>(LOWEST));
    EXPECT_EQ(dense[at(0, 1, 3, 29, 2, 6, 30)], static_cast<double>(LOWEST));
    EXPECT_GT(dense[at(0, 0, 1, 5, 2, 6, 30)], static_cast<double>(LOWEST));

    // Rows of 30 keys sit a whole 16-byte vector of FLOAT (32 elements) apart.
    const auto staged = stagedBias<float>(c, dense);
    ASSERT_EQ(staged.size(), biasViewOf(c).extent());
    EXPECT_EQ(staged[0], LOWEST);
    EXPECT_EQ(staged[30], 0.0F);
    EXPECT_EQ(staged[31], 0.0F);
    EXPECT_EQ(staged[32 + 5], dense[at(0, 0, 1, 5, 2, 6, 30)]);

    // The reference sees a finite score in the masked rows, so their statistics are finite.
    Inputs in;
    in.q = widened(roundedTo<hipdnn_data_sdk::types::half>(fillValues(size_t{2} * 6 * 64, 1U)));
    in.k = widened(roundedTo<hipdnn_data_sdk::types::half>(fillValues(size_t{2} * 30 * 64, 2U)));
    in.v = widened(roundedTo<hipdnn_data_sdk::types::half>(fillValues(size_t{2} * 30 * 64, 3U)));
    in.dO = widened(roundedTo<hipdnn_data_sdk::types::half>(fillValues(size_t{2} * 6 * 64, 4U)));
    in.bias = dense;
    const auto forward = forwardOf(c, in);
    EXPECT_EQ(forward.lse[0], LOWEST);
    EXPECT_TRUE(std::isfinite(forward.lse[0]));
    EXPECT_TRUE(std::isfinite(forward.lse[1]));
    const auto grads = backwardOf(c, in, forward.o, forward.lse);
    EXPECT_EQ(grads.dbias.size(), dense.size());
    // The same bias at -inf makes row 0 a row no key reaches.
    c.biasFill = BwdBiasFill::NEG_INF_ROWS;
    in.bias = biasInputsOf<float>(c);
    EXPECT_EQ(forwardOf(c, in).lse[0], -std::numeric_limits<float>::infinity());
}

// ---------------------------------------------------------------------------
// host: the gradient bound and the repeat-launch comparison
// ---------------------------------------------------------------------------

/// Four rows of four elements: row 0 holds values near 1000, as a row with P = 1 over many keys
/// does, and rows 1 to 3 values near 1.
std::vector<double> rowsOfTwoScales()
{
    return {1000, 900, 800, 700, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7};
}

/// A 10% error in one element of an ordinary row is far inside the normwise bound, which the
/// row of 1000 sets, and outside that row's own bound.
TEST(TestFlydslSdpaBwdHarness, PerRowBoundRejectsAnErrorInAnOrdinaryRowTheNormwiseBoundAccepts)
{
    constexpr double TOLERANCE = 4e-3;
    const auto expected = rowsOfTwoScales();
    const std::vector<bool> all(expected.size(), true);
    auto got = expected;
    got[5] *= 1.1;

    EXPECT_FALSE(worstGradientMiss(got, expected, all, TOLERANCE).has_value());
    const auto miss = worstGradientMiss(got, expected, all, TOLERANCE, 4);
    ASSERT_TRUE(miss.has_value());
    EXPECT_EQ(miss->index, 5U);
    EXPECT_DOUBLE_EQ(miss->scale, 1.0);
    EXPECT_DOUBLE_EQ(miss->bound, TOLERANCE * 1.0);

    // The bound is not tightened below the tolerance: an error within it, in either kind of row,
    // passes both ways.
    auto within = expected;
    within[1] *= 1.0 + TOLERANCE / 2;
    within[5] *= 1.0 + TOLERANCE / 2;
    EXPECT_FALSE(worstGradientMiss(within, expected, all, TOLERANCE).has_value());
    EXPECT_FALSE(worstGradientMiss(within, expected, all, TOLERANCE, 4).has_value());
}

TEST(TestFlydslSdpaBwdHarness, PerRowBoundNamesTheWorstRowAndIgnoresElementsOutOfScope)
{
    constexpr double TOLERANCE = 2e-2;
    const auto expected = rowsOfTwoScales();
    std::vector<bool> scope(expected.size(), true);
    auto got = expected;
    got[5] += 0.5;
    got[9] += 0.25;
    const auto worst = worstGradientMiss(got, expected, scope, TOLERANCE, 4);
    ASSERT_TRUE(worst.has_value());
    EXPECT_EQ(worst->index, 5U);

    // An element past a valid length is not compared, whatever it holds.
    scope[5] = false;
    scope[9] = false;
    got[13] = std::numeric_limits<double>::quiet_NaN();
    scope[13] = false;
    EXPECT_FALSE(worstGradientMiss(got, expected, scope, TOLERANCE, 4).has_value());

    // An unwritten element is a miss in the row it is in, and a row whose reference is all
    // zero holds nothing but zeros.
    scope[13] = true;
    EXPECT_EQ(worstGradientMiss(got, expected, scope, TOLERANCE, 4)->index, 13U);
    std::vector<double> zeros(8, 0.0);
    auto notZero = zeros;
    notZero[6] = 1e-30;
    EXPECT_EQ(worstGradientMiss(notZero, zeros, std::vector<bool>(8, true), TOLERANCE, 4)->index,
              6U);
}

TEST(TestFlydslSdpaBwdHarness, RepeatLaunchComparisonFlagsAOneBitDifference)
{
    const std::vector<float> first{1.0F, -2.5F, 0.0F, 3.25F};
    EXPECT_FALSE(firstBitwiseMismatch(first, first).has_value());

    auto oneBit = first;
    uint32_t bits = 0;
    std::memcpy(&bits, &oneBit[3], sizeof(bits));
    bits ^= 1U;
    std::memcpy(&oneBit[3], &bits, sizeof(bits));
    ASSERT_NE(oneBit[3], first[3]);
    EXPECT_EQ(firstBitwiseMismatch(first, oneBit), std::optional<size_t>(3));

    // Equal as numbers, different as bits.
    auto negativeZero = first;
    negativeZero[2] = -0.0F;
    ASSERT_EQ(negativeZero[2], first[2]);
    EXPECT_EQ(firstBitwiseMismatch(first, negativeZero), std::optional<size_t>(2));

    // A shorter launch differs where it ends.
    EXPECT_EQ(firstBitwiseMismatch(first, std::vector<float>(first.begin(), first.begin() + 2)),
              std::optional<size_t>(2));
}

TEST(TestFlydslSdpaBwdHarness, RepeatLaunchComparisonFlagsAnUnwrittenOutput)
{
    using Half = hipdnn_data_sdk::types::half;
    const std::vector<Half> first{Half(1.0F), Half(-0.5F), Half(0.25F)};

    // Poison is all-ones bytes: a quiet NaN in f16. An element the second launch does not write
    // keeps it, and differs from what the first launch wrote there.
    auto unwritten = first;
    std::memset(&unwritten[1], 0xFF, sizeof(Half));
    EXPECT_TRUE(std::isnan(static_cast<float>(unwritten[1])));
    EXPECT_EQ(firstBitwiseMismatch(first, unwritten), std::optional<size_t>(1));

    // The poison compares equal to itself, so a buffer neither launch wrote is left to the
    // reference comparison, which counts a NaN as a miss.
    EXPECT_FALSE(firstBitwiseMismatch(unwritten, unwritten).has_value());
    const std::vector<double> expected{1.0, -0.5, 0.25};
    const std::vector<double> got{1.0, std::numeric_limits<double>::quiet_NaN(), 0.25};
    EXPECT_EQ(worstGradientMiss(got, expected, std::vector<bool>(3, true), 4e-2)->index, 1U);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
