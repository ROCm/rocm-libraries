// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDumpTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdPhiloxDevice.cpp
 * @brief The SDPA backward's dropout mask against the forward's: the forward engine of the
 *        device writes the keep mask of a dropping graph (`rng_dump`), and the backward engines
 *        run the same problem, from the same seed and offset, against a reference that takes
 *        that dump as its explicit mask.
 *
 * Padded and packed (THD) forms of one GQA problem, f16, top-left causal, sequences of unequal
 * lengths and a key count that is no multiple of 4, a seed past 32 bits and an offset that
 * carries into the counter's high word. The dump is first checked against the host Philox
 * oracle, so a failing gradient says whether the forward's mask or the backward's counters
 * are off. The forward is the CDNA engine on gfx950 and the RDNA engine on gfx11; the backward
 * is the merged engine (gfx950) and the split engine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
namespace bwd_device = testing::bwd_device;

constexpr float P = 0.25F;
constexpr uint64_t SEED = 0x0123456789ABCDEFULL;
constexpr uint64_t OFFSET = (uint64_t{1} << 32) - 7;
constexpr int64_t BATCH = 3;
constexpr int64_t HEADS = 4;
constexpr int64_t KV_HEADS = 2;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t SEQ_Q = 40;
constexpr int64_t SEQ_KV = 70;

/// The sequences' query and key lengths, in tokens: a full batch, one short in both, one short
/// in queries and a few keys short.
const std::vector<int32_t> LENGTHS_Q{40, 17, 29};
const std::vector<int32_t> LENGTHS_KV{70, 33, 64};

/// The forward engine of this device and the ops it launches for a dump, or nothing on a device
/// with no FlyDSL SDPA forward.
struct ForwardEngine
{
    std::string_view engine;
    std::string_view op;
};

std::optional<ForwardEngine> forwardEngineOfThisDevice()
{
    const auto arch = testing::packedArchDeviceProperties().gcnArchName;
    if(arch.rfind("gfx950", 0) == 0)
    {
        return ForwardEngine{"hipkernel:flydsl_sdpa_cdna", "sdpa_cdna"};
    }
    if(arch.rfind("gfx11", 0) == 0)
    {
        return ForwardEngine{"hipkernel:flydsl_sdpa_rdna", "sdpa_rdna"};
    }
    return std::nullopt;
}

/// The forward graph of the problem, padded or packed, dropping with a dump.
testing::SdpaGraphSpec forwardSpec(bool packed)
{
    testing::SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.batch = BATCH;
    spec.heads = HEADS;
    spec.kvHeads = KV_HEADS;
    spec.headDim = HEAD_DIM;
    spec.seqLenQ = SEQ_Q;
    spec.seqLenKv = SEQ_KV;
    spec.mask = testing::SdpaMask::TOP_LEFT_CAUSAL;
    spec.stats = testing::SdpaStats::LSE;
    testing::setDropoutWithRngDump(spec, P, SEED, OFFSET);
    if(packed)
    {
        testing::setPacked(spec, LENGTHS_Q, LENGTHS_KV);
    }
    else
    {
        testing::setPadding(spec, LENGTHS_Q, LENGTHS_KV);
    }
    return spec;
}

/// The backward case of the same problem.
bwd_device::BwdCase backwardCase(bool packed)
{
    bwd_device::BwdCase c;
    c.dtype = DataType::HALF;
    c.batch = BATCH;
    c.heads = HEADS;
    c.kvHeads = KV_HEADS;
    c.seqQ = SEQ_Q;
    c.seqKv = SEQ_KV;
    c.headDim = HEAD_DIM;
    c.mask = bwd_device::BwdMask::TOP_LEFT;
    c.dropout = P;
    c.seed = static_cast<int64_t>(SEED);
    c.offset = static_cast<int64_t>(OFFSET);
    c.lengthsQ = LENGTHS_Q;
    c.lengthsKv = LENGTHS_KV;
    c.packed = packed;
    c.layout = packed ? bwd_device::BwdLayout::SEQ_MAJOR : bwd_device::BwdLayout::HEAD_MAJOR;
    return c;
}

/// Zeroes a device buffer.
void zero(const bwd_device::DeviceBuffer& buffer, size_t bytes)
{
    ASSERT_EQ(hipSuccess, hipMemset(buffer.get(), 0, bytes));
}

/// Runs the forward engine on @p spec and returns the RNG dump it wrote: FLOAT [B, Hq, Sq, Skv].
/// Q, K and V are zero: the mask does not depend on the data.
void runForwardForTheDump(const testing::SdpaGraphSpec& spec, std::vector<float>& dump)
{
    const auto forward = forwardEngineOfThisDevice();
    ASSERT_TRUE(forward.has_value());
    const auto builder = testing::buildFlydslSdpaRdnaGraph(spec);
    const auto graph = bwd_device::wrap(builder);

    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = bwd_device::idOf(forward->engine);
    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    ASSERT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << forward->engine << " does not claim the forward graph";
    hipdnn_test_sdk::utilities::MockEngineConfig config;
    bwd_device::stubEngineId(config, engineId);
    Context context;
    ASSERT_NO_THROW(manager.initializeExecutionContext(handle, graph, config, context));
    ASSERT_TRUE(context.hasValidPlan());

    constexpr size_t HALF_BYTES = 2;
    const size_t qBytes = static_cast<size_t>(BATCH * SEQ_Q * HEADS * HEAD_DIM) * HALF_BYTES;
    const size_t kvBytes = static_cast<size_t>(BATCH * SEQ_KV * KV_HEADS * HEAD_DIM) * HALF_BYTES;
    const size_t statsBytes = static_cast<size_t>(BATCH * HEADS * SEQ_Q) * sizeof(float);
    const bwd_device::DeviceBuffer q(qBytes);
    const bwd_device::DeviceBuffer k(kvBytes);
    const bwd_device::DeviceBuffer v(kvBytes);
    const bwd_device::DeviceBuffer o(qBytes);
    const bwd_device::DeviceBuffer stats(statsBytes);
    zero(q, qBytes);
    zero(k, kvBytes);
    zero(v, kvBytes);
    zero(o, qBytes);
    zero(stats, statsBytes);
    bwd_device::DeviceBuffer seed(sizeof(int64_t));
    bwd_device::DeviceBuffer offset(sizeof(int64_t));
    seed.upload(std::vector<int64_t>{static_cast<int64_t>(SEED)});
    offset.upload(std::vector<int64_t>{static_cast<int64_t>(OFFSET)});
    const testing::RngDumpExtents extents{BATCH, HEADS, SEQ_Q, SEQ_KV};
    testing::DeviceRngDump rngDump(extents, true);

    std::vector<hipdnnPluginDeviceBuffer_t> buffers{
        {testing::FLYDSL_SDPA_RDNA_Q_UID, q.get()},
        {testing::FLYDSL_SDPA_RDNA_K_UID, k.get()},
        {testing::FLYDSL_SDPA_RDNA_V_UID, v.get()},
        {testing::FLYDSL_SDPA_RDNA_O_UID, o.get()},
        {testing::FLYDSL_SDPA_RDNA_STATS_UID, stats.get()},
        {testing::FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID, seed.get()},
        {testing::FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID, offset.get()},
        {testing::FLYDSL_SDPA_RDNA_RNG_DUMP_UID, rngDump.device()}};
    bwd_device::DeviceBuffer lengthsQ(spec.seqLensQ.size() * sizeof(int32_t));
    bwd_device::DeviceBuffer lengthsKv(spec.seqLensKv.size() * sizeof(int32_t));
    if(!spec.seqLensQ.empty())
    {
        lengthsQ.upload(spec.seqLensQ);
        lengthsKv.upload(spec.seqLensKv);
        buffers.push_back({testing::FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, lengthsQ.get()});
        buffers.push_back({testing::FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, lengthsKv.get()});
    }
    const auto tableQ = testing::sdpaPackedTable({}, spec.packedSeqLensQ);
    const auto tableKv = testing::sdpaPackedTable({}, spec.packedSeqLensKv);
    bwd_device::DeviceBuffer tablesQ(spec.packed() ? tableQ.size() * sizeof(int32_t) : 0);
    bwd_device::DeviceBuffer tablesKv(spec.packed() ? tableKv.size() * sizeof(int32_t) : 0);
    if(spec.packed())
    {
        tablesQ.upload(tableQ);
        tablesKv.upload(tableKv);
        buffers.push_back({testing::FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, tablesQ.get()});
        buffers.push_back({testing::FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, tablesKv.get()});
    }

    const auto& plan = context.plan();
    const testing::GuardedWorkspace workspace(plan.getWorkspaceSize(handle));
    ASSERT_NO_THROW(plan.execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    dump = rngDump.readBack();
}

/// The forward's dump for the padded or packed form of the problem is the oracle's mask, and the
/// gradients of @p engine for the same graph match the reference that takes the dump as its mask.
void expectGradientsMatchTheForwardsDump(std::string_view engine, bool packed)
{
    const auto forward = forwardEngineOfThisDevice();
    if(!forward.has_value())
    {
        GTEST_SKIP() << "no FlyDSL SDPA forward engine serves "
                     << testing::packedArchDeviceProperties().gcnArchName;
    }
    for(const auto op : {forward->op, std::string_view{"sdpa_dropout_mask"}})
    {
        if(!testing::flydslShipsForThisDevice(op))
        {
            testing::reportUnstaged(op,
                                    "the forward engine that writes the rng_dump is not staged");
            return;
        }
    }
    std::vector<float> dump;
    ASSERT_NO_FATAL_FAILURE(runForwardForTheDump(forwardSpec(packed), dump));
    ASSERT_NO_FATAL_FAILURE(testing::expectRngDumpIsTheOracleMask(
        dump, testing::RngDumpExtents{BATCH, HEADS, SEQ_Q, SEQ_KV}, P, SEED, OFFSET));

    auto c = backwardCase(packed);
    c.keepMask = std::move(dump);
    bwd_device::expectCase(engine, c);
}

class TestGpuFlydslSdpaBwdMergedRngDumpChain : public bwd_device::FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedRngDumpChain()
        : FlydslSdpaBwdDevice(bwd_device::MERGED_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdSplitRngDumpChain : public bwd_device::FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitRngDumpChain()
        : FlydslSdpaBwdDevice(bwd_device::SPLIT_OPS)
    {
    }
};

// The datatype token ends the suite name.
using TestGpuFlydslSdpaBwdMergedRngDumpChainFp16 = TestGpuFlydslSdpaBwdMergedRngDumpChain;
using TestGpuFlydslSdpaBwdSplitRngDumpChainFp16 = TestGpuFlydslSdpaBwdSplitRngDumpChain;

TEST_F(TestGpuFlydslSdpaBwdMergedRngDumpChainFp16, PaddedGradientsMatchTheForwardsDump)
{
    expectGradientsMatchTheForwardsDump(bwd_device::MERGED_ENGINE, false);
}

TEST_F(TestGpuFlydslSdpaBwdMergedRngDumpChainFp16, PackedGradientsMatchTheForwardsDump)
{
    expectGradientsMatchTheForwardsDump(bwd_device::MERGED_ENGINE, true);
}

TEST_F(TestGpuFlydslSdpaBwdSplitRngDumpChainFp16, PaddedGradientsMatchTheForwardsDump)
{
    expectGradientsMatchTheForwardsDump(bwd_device::SPLIT_ENGINE, false);
}

TEST_F(TestGpuFlydslSdpaBwdSplitRngDumpChainFp16, PackedGradientsMatchTheForwardsDump)
{
    expectGradientsMatchTheForwardsDump(bwd_device::SPLIT_ENGINE, true);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
