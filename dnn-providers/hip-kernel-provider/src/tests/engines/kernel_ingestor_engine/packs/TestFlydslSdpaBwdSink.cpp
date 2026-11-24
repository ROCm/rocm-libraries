// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"

/**
 * @file TestFlydslSdpaBwdSink.cpp
 * @brief The sink, dSink and deterministic-algorithm fields of the FlyDSL SDPA backward graph,
 *        on a gfx950 device, and the split engine's sink handling where no sink twin ships.
 *
 * The split engine serves a sink graph. A graph with a dSink runs the prep sink twin and the
 * dsink_reduce kernel (the prep stage's kernel carries a `dsink_reduce` source); a graph with a
 * sink and no dSink runs the plain prep rows, because the sink is already in the sink-inclusive
 * LSE the caller supplies. The merged engine reads that same LSE, so it serves a sink graph
 * without a dSink (a FLOAT one too, since no kernel reads it) and declines one with a dSink; the
 * split engine declines a FLOAT sink with a dSink and an fp32 graph with a dSink.
 *
 * The reference is CpuFpReferenceSdpa: its forward pass, with the sink, gives the O and the
 * sink-inclusive LSE the device is handed, and its backward pass (SdpaBwdOptions::setSink) gives
 * dQ, dK, dV and dSink. The sink and dSink are in the graph's I/O dtype. dQ, dK, dV and dSink are
 * held to the harness's bar (a fraction of each gradient's largest element), and dSink also to
 * cuDNN's (atol 4e-2, rtol 2e-2). Every served case asserts the engine claims the graph, builds
 * the plan through the engine's id, and, for the split engine, which prep rows the plan launches.
 *
 * Determinism: the engines' kernels have one writer per output element, so a graph that sets
 * is_deterministic_algorithm produces bit-identical gradients on a second run.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

using Tensorf = hipdnn_data_sdk::utilities::Tensor<float>;
using Tensori = hipdnn_data_sdk::utilities::Tensor<int32_t>;

/// The ops of both engines: every case here runs on a device that ships all of them.
inline const std::vector<std::string_view> SINK_OPS{
    "sdpa_bwd_merged", "sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"};

class TestGpuFlydslSdpaBwdSink : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSink()
        : FlydslSdpaBwdDevice(SINK_OPS)
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
        if(arch.rfind("gfx950", 0) != 0)
        {
            GTEST_SKIP() << "the sink twin and dsink_reduce objects ship for gfx950; " << arch
                         << " is not one";
        }
    }
};

// One suite per dtype: the datatype token ends the suite name.
using TestGpuFlydslSdpaBwdSinkBfp16 = TestGpuFlydslSdpaBwdSink;
using TestGpuFlydslSdpaBwdSinkFp16 = TestGpuFlydslSdpaBwdSink;

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// A BwdCase and what the graph adds to it.
struct SinkCase
{
    BwdCase shape;
    bool withSink = true;
    bool withDsink = true;
    /// The sink's and dSink's dtype; unset is the graph's I/O dtype.
    std::optional<DataType> sinkType;
    bool deterministic = false;
};

SdpaBwdSpec specOfSink(const SinkCase& c)
{
    auto spec = specOf(c.shape);
    const auto sinkType = c.sinkType.value_or(c.shape.dtype);
    if(c.withSink)
    {
        spec = spec.withEdit([=](SdpaBwdGraph& g) { addSink(g, sinkType); });
    }
    if(c.withDsink)
    {
        spec = spec.withEdit([=](SdpaBwdGraph& g) { addDsink(g, sinkType); });
    }
    if(c.deterministic)
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { g.attributes.isDeterministicAlgorithm = true; });
    }
    return spec;
}

/// MHA at head dim 64, or GQA (8 query heads over 2) at head dim 128; dense, or padded in both
/// lengths; non-causal or top-left causal.
SinkCase sinkCase(DataType dtype, bool gqa, bool padded, bool causal)
{
    SinkCase c;
    c.shape.dtype = dtype;
    if(gqa)
    {
        c.shape.heads = 8;
        c.shape.kvHeads = 2;
        c.shape.headDim = 128;
    }
    if(padded)
    {
        c.shape.lengthsQ = {96, 70};
        c.shape.lengthsKv = {96, 50};
    }
    c.shape.mask = causal ? BwdMask::TOP_LEFT : BwdMask::NONE;
    return c;
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

Tensorf denseTensor(const std::vector<int64_t>& dims, const std::vector<double>& values = {})
{
    Tensorf tensor(dims, denseStrides(dims));
    auto* host = tensor.memory().hostData();
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        host[i] = values.empty() ? 0.0F : static_cast<float>(values[i]);
    }
    tensor.memory().markHostModified();
    return tensor;
}

std::vector<double> valuesOf(Tensorf& tensor)
{
    const auto* host = tensor.memory().hostData();
    return std::vector<double>(host, host + tensor.elementCount());
}

Tensori lengthsTensor(int64_t batch, const std::vector<int32_t>& lengths)
{
    Tensori tensor({batch, 1, 1, 1}, denseStrides({batch, 1, 1, 1}));
    auto* host = tensor.memory().hostData();
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        host[i] = lengths[i];
    }
    tensor.memory().markHostModified();
    return tensor;
}

/// What the device is given and what the reference says it must produce, dense logical (B, H,
/// S, D), every device input rounded to T.
template <typename T>
struct Staged
{
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> dO;
    /// Forward output rounded to T, and the sink-inclusive log-sum-exp (B, Hq, Sq).
    std::vector<double> o;
    std::vector<float> lse;
    /// One value per query head, in the graph's I/O dtype.
    std::vector<double> sink;
    std::vector<double> expectedDq;
    std::vector<double> expectedDk;
    std::vector<double> expectedDv;
    std::vector<double> expectedDsink;
};

template <typename T>
Staged<T> stageOf(const SinkCase& c)
{
    const auto& s = c.shape;
    const int64_t dqk = s.headDim;
    const int64_t dv = s.valueDim();
    Staged<T> st;
    st.q = widened(
        roundedTo<T>(fillValues(static_cast<size_t>(s.batch * s.heads * s.seqQ * dqk), 1U)));
    st.k = widened(
        roundedTo<T>(fillValues(static_cast<size_t>(s.batch * s.kvHeads * s.seqKv * dqk), 2U)));
    st.v = widened(roundedTo<T>(
        fillValues(static_cast<size_t>(s.batch * s.valueHeadCount() * s.seqKv * dv), 3U)));
    st.dO = widened(
        roundedTo<T>(fillValues(static_cast<size_t>(s.batch * s.heads * s.seqQ * dv), 4U)));
    if(s.padded())
    {
        // The gradient's input is zero past each batch's length, as a packed batch's padding is.
        const auto scope = scopeOf(s, true, s.heads, dv);
        for(size_t i = 0; i < st.dO.size(); ++i)
        {
            st.dO[i] = scope[i] ? st.dO[i] : 0.0;
        }
    }
    if(c.withSink)
    {
        // [0, 4] in steps of 1/16: exact in f16 and bf16, and large enough that the sink holds a
        // real share of every softmax row, so dSink is not near zero.
        auto raw = fillValues(static_cast<size_t>(s.heads), 6U);
        for(auto& value : raw)
        {
            value = 2.0 + 2.0 * value;
        }
        st.sink = widened(roundedTo<T>(raw));
    }

    const std::vector<int64_t> qDims{s.batch, s.heads, s.seqQ, dqk};
    const std::vector<int64_t> kDims{s.batch, s.kvHeads, s.seqKv, dqk};
    const std::vector<int64_t> vDims{s.batch, s.valueHeadCount(), s.seqKv, dv};
    const std::vector<int64_t> oDims{s.batch, s.heads, s.seqQ, dv};
    const std::vector<int64_t> statsDims{s.batch, s.heads, s.seqQ, 1};
    const std::vector<int64_t> sinkDims{1, s.heads, 1, 1};

    const auto q = denseTensor(qDims, st.q);
    const auto k = denseTensor(kDims, st.k);
    const auto v = denseTensor(vDims, st.v);
    const auto dO = denseTensor(oDims, st.dO);
    auto sink = denseTensor(sinkDims, st.sink);
    Tensori lengthsQ = lengthsTensor(s.batch, s.lengthsQ);
    Tensori lengthsKv = lengthsTensor(s.batch, s.lengthsKv);

    const auto scale = static_cast<float>(s.scale());
    const bool causal = s.mask != BwdMask::NONE;
    const bool topLeft = s.mask != BwdMask::BOTTOM_RIGHT;

    auto o = denseTensor(oDims);
    auto lse = denseTensor(statsDims);
    {
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float> options;
        options.setScale(scale).setLse(&lse);
        if(causal)
        {
            options.setBand(-1, 0, topLeft);
        }
        if(s.padded())
        {
            options.setSeqLens(&lengthsQ, &lengthsKv);
        }
        if(c.withSink)
        {
            options.setSink(&sink);
        }
        hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(q, k, v, o, options);
    }
    st.o = widened(roundedTo<T>(valuesOf(o)));
    const auto lseValues = valuesOf(lse);
    st.lse.assign(lseValues.begin(), lseValues.end());

    // The backward pass is handed the O the device will read, rounded to T.
    const auto oRounded = denseTensor(oDims, st.o);
    auto dQ = denseTensor(qDims);
    auto dK = denseTensor(kDims);
    auto dV = denseTensor(vDims);
    auto dSink = denseTensor(sinkDims);
    {
        hipdnn_test_sdk::utilities::SdpaBwdOptions<float> options;
        options.setScale(scale).setLse(&lse);
        if(causal)
        {
            options.setBand(-1, 0, topLeft);
        }
        if(s.padded())
        {
            options.setSeqLens(&lengthsQ, &lengthsKv);
        }
        if(c.withSink && c.withDsink)
        {
            options.setSink(&sink, &dSink);
        }
        else if(c.withSink)
        {
            options.setSink(&sink, nullptr);
        }
        hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::backward(
            q, k, v, oRounded, dO, dQ, dK, dV, options);
    }
    st.expectedDq = valuesOf(dQ);
    st.expectedDk = valuesOf(dK);
    st.expectedDv = valuesOf(dV);
    if(c.withDsink)
    {
        st.expectedDsink = valuesOf(dSink);
    }
    return st;
}

// ---------------------------------------------------------------------------
// Device run
// ---------------------------------------------------------------------------

/// The device's outputs as the buffers hold them, bit for bit.
template <typename T>
struct Outputs
{
    std::vector<T> dq;
    std::vector<T> dk;
    std::vector<T> dv;
    std::vector<T> dsink;
};

/// The split engine's prep stage launches dsink_reduce after the sink twin exactly when the
/// graph has a dSink; any other graph runs a single-launch prep row.
void expectPrepLaunches(const Context& context, std::string_view engine, bool withDsink)
{
    if(engine != SPLIT_ENGINE)
    {
        return;
    }
    const auto* program = dynamic_cast<const hipdnn_plugin_sdk::ingestor::SlotProgramPlan<Handle>*>(
        &context.plan());
    ASSERT_NE(program, nullptr) << "the split engine's plan is not a slot program";
    const auto stage = std::find_if(program->stages().begin(),
                                    program->stages().end(),
                                    [](const auto& candidate) { return candidate.slot == "prep"; });
    ASSERT_NE(stage, program->stages().end()) << "the plan has no prep stage";
    const auto& launched = stage->plan->kernel();
    const bool launchesReduce = launched.sources.find("dsink_reduce") != launched.sources.end();
    EXPECT_EQ(launchesReduce, withDsink)
        << launched.name << (withDsink ? " launches no dsink_reduce" : " launches dsink_reduce");
    if(withDsink)
    {
        EXPECT_NE(launched.sources.find("prep"), launched.sources.end())
            << launched.name << " launches no prep sink twin";
    }
}

/// Builds @p engine's plan for @p c's graph through the engine manager by the engine's id, runs
/// it once on @p staged's inputs, and downloads the gradients into @p out.
template <typename T>
void execute(std::string_view engine, const SinkCase& c, const Staged<T>& staged, Outputs<T>& out)
{
    const auto& s = c.shape;
    const auto builder = buildSdpaBwdGraph(specOfSink(c));
    const auto graph = wrap(builder);

    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = idOf(engine);

    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    ASSERT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << engine << " does not claim the graph";

    hipdnn_test_sdk::utilities::MockEngineConfig config;
    stubEngineId(config, engineId);
    Context context;
    ASSERT_NO_THROW(manager.initializeExecutionContext(handle, graph, config, context));
    ASSERT_TRUE(context.hasValidPlan());
    expectPrepLaunches(context, engine, c.withDsink);
    if(::testing::Test::HasFatalFailure())
    {
        return;
    }

    const auto views = viewsOf(s);
    auto elements = views.groupElements();
    std::array<std::vector<T>, GROUP_COUNT> hostData;
    for(size_t group = 0; group < GROUP_COUNT; ++group)
    {
        hostData[group].assign(elements[group], T(0.0F));
    }
    scatter(hostData[views.q.group], staged.q, views.q);
    scatter(hostData[views.k.group], staged.k, views.k);
    scatter(hostData[views.v.group], staged.v, views.v);
    scatter(hostData[views.o.group], staged.o, views.o);
    scatter(hostData[views.dO.group], staged.dO, views.dO);
    std::vector<float> statsHost(elements[views.stats.group], 0.0F);
    {
        const auto& view = views.stats;
        size_t next = 0;
        for(int64_t b = 0; b < view.dims[0]; ++b)
        {
            for(int64_t h = 0; h < view.dims[1]; ++h)
            {
                for(int64_t row = 0; row < view.dims[2]; ++row)
                {
                    statsHost[static_cast<size_t>(b * view.strides[0] + h * view.strides[1]
                                                  + row * view.strides[2])]
                        = staged.lse[next++];
                }
            }
        }
    }

    std::vector<std::unique_ptr<DeviceBuffer>> device(GROUP_COUNT);
    for(size_t group = 0; group < GROUP_COUNT; ++group)
    {
        const auto bytes
            = elements[group] * (group == views.stats.group ? sizeof(float) : sizeof(T));
        if(bytes > 0)
        {
            device[group] = std::make_unique<DeviceBuffer>(bytes);
        }
    }
    device[views.q.group]->upload(hostData[views.q.group]);
    device[views.k.group]->upload(hostData[views.k.group]);
    device[views.v.group]->upload(hostData[views.v.group]);
    device[views.o.group]->upload(hostData[views.o.group]);
    device[views.dO.group]->upload(hostData[views.dO.group]);
    device[views.stats.group]->upload(statsHost);
    for(const auto* view : {&views.dq, &views.dk, &views.dv})
    {
        device[view->group]->poison();
    }
    const auto pointerOf = [&](const TensorView& view) {
        return static_cast<char*>(device[view.group]->get())
               + static_cast<size_t>(view.offset) * sizeof(T);
    };
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{uid::Q, pointerOf(views.q)},
                                                    {uid::K, pointerOf(views.k)},
                                                    {uid::V, pointerOf(views.v)},
                                                    {uid::O, pointerOf(views.o)},
                                                    {uid::DO, pointerOf(views.dO)},
                                                    {uid::STATS, device[views.stats.group]->get()},
                                                    {uid::DQ, pointerOf(views.dq)},
                                                    {uid::DK, pointerOf(views.dk)},
                                                    {uid::DV, pointerOf(views.dv)}};

    // The sink and dSink are 16-byte aligned (hipMalloc) and one vector long at least.
    const auto sinkElements = static_cast<size_t>((s.heads + 7) / 8 * 8);
    std::unique_ptr<DeviceBuffer> sink;
    std::unique_ptr<DeviceBuffer> dsink;
    if(c.withSink && c.sinkType == DataType::FLOAT)
    {
        // A FLOAT sink: no kernel reads it without a dSink, but the buffer is the graph's.
        std::vector<float> sinkHost(sinkElements, 0.0F);
        for(size_t h = 0; h < staged.sink.size(); ++h)
        {
            sinkHost[h] = static_cast<float>(staged.sink[h]);
        }
        sink = std::make_unique<DeviceBuffer>(sinkElements * sizeof(float));
        sink->upload(sinkHost);
        buffers.push_back({uid::SINK, sink->get()});
    }
    else if(c.withSink)
    {
        std::vector<T> sinkHost(sinkElements, T(0.0F));
        for(size_t h = 0; h < staged.sink.size(); ++h)
        {
            sinkHost[h] = T(static_cast<float>(staged.sink[h]));
        }
        sink = std::make_unique<DeviceBuffer>(sinkElements * sizeof(T));
        sink->upload(sinkHost);
        buffers.push_back({uid::SINK, sink->get()});
    }
    if(c.withDsink)
    {
        dsink = std::make_unique<DeviceBuffer>(sinkElements * sizeof(T));
        dsink->poison();
        buffers.push_back({uid::DSINK, dsink->get()});
    }
    DeviceBuffer lengthsQ(s.lengthsQ.size() * sizeof(int32_t));
    DeviceBuffer lengthsKv(s.lengthsKv.size() * sizeof(int32_t));
    if(s.padded())
    {
        lengthsQ.upload(s.lengthsQ);
        lengthsKv.upload(s.lengthsKv);
        buffers.push_back({uid::SEQ_LEN_Q, lengthsQ.get()});
        buffers.push_back({uid::SEQ_LEN_KV, lengthsKv.get()});
    }

    // Poisoned, so a kernel that reads workspace it did not write fails the case.
    const auto workspaceBytes = context.plan().getWorkspaceSize(handle);
    const GuardedWorkspace workspace(workspaceBytes);
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMemset(workspace.get(), 0xFF, workspaceBytes));
    }
    ASSERT_NO_THROW(context.plan().execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    EXPECT_EQ(workspace.overwrittenGuardBytes(), 0U)
        << "the launch wrote past the " << workspaceBytes << "-byte workspace it reported";

    out.dq = device[views.dq.group]->template download<T>();
    out.dk = device[views.dk.group]->template download<T>();
    out.dv = device[views.dv.group]->template download<T>();
    if(c.withDsink)
    {
        out.dsink = dsink->template download<T>();
    }
}

/// dQ, dK, dV and dSink against the reference.
template <typename T>
void expectMatchesReference(const SinkCase& c, const Staged<T>& staged, const Outputs<T>& out)
{
    const auto& s = c.shape;
    const auto views = viewsOf(s);
    const double tolerance = toleranceOf(s.dtype);
    expectGradient("dQ",
                   gather(out.dq, views.dq),
                   staged.expectedDq,
                   scopeOf(s, true, s.heads, s.headDim),
                   tolerance);
    expectGradient("dK",
                   gather(out.dk, views.dk),
                   staged.expectedDk,
                   scopeOf(s, false, s.kvHeads, s.headDim),
                   tolerance);
    expectGradient("dV",
                   gather(out.dv, views.dv),
                   staged.expectedDv,
                   scopeOf(s, false, s.valueHeadCount(), s.valueDim()),
                   tolerance);
    if(!c.withDsink)
    {
        return;
    }
    const auto heads = static_cast<size_t>(s.heads);
    ASSERT_GE(out.dsink.size(), heads);
    const auto got = widened(
        std::vector<T>(out.dsink.begin(), out.dsink.begin() + static_cast<std::ptrdiff_t>(heads)));
    expectGradient("dSink", got, staged.expectedDsink, std::vector<bool>(heads, true), tolerance);
    // cuDNN's bar for dSink (fp16.py: atol 4e-2, rtol 2e-2).
    for(size_t h = 0; h < heads; ++h)
    {
        EXPECT_LE(std::abs(got[h] - staged.expectedDsink[h]),
                  4e-2 + 2e-2 * std::abs(staged.expectedDsink[h]))
            << "dSink of head " << h << ": got " << got[h] << ", expected "
            << staged.expectedDsink[h];
    }
}

template <typename T>
void expectServed(std::string_view engine, const SinkCase& c)
{
    const auto staged = stageOf<T>(c);
    Outputs<T> out;
    execute<T>(engine, c, staged, out);
    if(::testing::Test::HasFatalFailure())
    {
        return;
    }
    expectMatchesReference(c, staged, out);
}

/// Two runs of one graph on the same inputs: every output buffer is bit-identical, and the
/// first run matches the reference (so two buffers of poison cannot pass).
template <typename T>
void expectRepeatable(std::string_view engine, const SinkCase& c)
{
    const auto staged = stageOf<T>(c);
    Outputs<T> first;
    Outputs<T> second;
    execute<T>(engine, c, staged, first);
    if(::testing::Test::HasFatalFailure())
    {
        return;
    }
    execute<T>(engine, c, staged, second);
    if(::testing::Test::HasFatalFailure())
    {
        return;
    }
    expectMatchesReference(c, staged, first);
    const auto identical = [](const char* name, const std::vector<T>& a, const std::vector<T>& b) {
        ASSERT_EQ(a.size(), b.size()) << name;
        EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
            << name << " differs between two runs of the same graph";
    };
    identical("dQ", first.dq, second.dq);
    identical("dK", first.dk, second.dk);
    identical("dV", first.dv, second.dv);
    if(c.withDsink)
    {
        identical("dSink", first.dsink, second.dsink);
    }
}

void expectServedAs(std::string_view engine, const SinkCase& c)
{
    if(c.shape.dtype == DataType::HALF)
    {
        expectServed<hipdnn_data_sdk::types::half>(engine, c);
    }
    else
    {
        ASSERT_EQ(DataType::BFLOAT16, c.shape.dtype);
        expectServed<hipdnn_data_sdk::types::bfloat16>(engine, c);
    }
}

void expectRepeatableAs(std::string_view engine, const SinkCase& c)
{
    if(c.shape.dtype == DataType::HALF)
    {
        expectRepeatable<hipdnn_data_sdk::types::half>(engine, c);
    }
    else
    {
        ASSERT_EQ(DataType::BFLOAT16, c.shape.dtype);
        expectRepeatable<hipdnn_data_sdk::types::bfloat16>(engine, c);
    }
}

/// Dense and padded, non-causal and causal.
void expectDsinkGraphsServed(DataType dtype, bool gqa)
{
    for(const bool padded : {false, true})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE(std::string(gqa ? "gqa" : "mha") + (padded ? " padded" : " dense")
                         + (causal ? " causal" : " non-causal"));
            expectServedAs(SPLIT_ENGINE, sinkCase(dtype, gqa, padded, causal));
            if(::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Served: a dSink graph on the split engine
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, SplitServesADsinkGraphOfMhaThroughTheSinkTwin)
{
    expectDsinkGraphsServed(DataType::BFLOAT16, false);
}

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, SplitServesADsinkGraphOfGqaThroughTheSinkTwin)
{
    expectDsinkGraphsServed(DataType::BFLOAT16, true);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, SplitServesADsinkGraphOfMhaThroughTheSinkTwin)
{
    expectDsinkGraphsServed(DataType::HALF, false);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, SplitServesADsinkGraphOfGqaThroughTheSinkTwin)
{
    expectDsinkGraphsServed(DataType::HALF, true);
}

// ---------------------------------------------------------------------------
// Served: a sink graph with no dSink
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, SplitServesASinkOnlyGraphWithThePlainPrepRows)
{
    auto c = sinkCase(DataType::BFLOAT16, true, false, true);
    c.withDsink = false;
    expectServedAs(SPLIT_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, SplitServesASinkOnlyGraphWithThePlainPrepRows)
{
    auto c = sinkCase(DataType::HALF, false, true, false);
    c.withDsink = false;
    expectServedAs(SPLIT_ENGINE, c);
}

/// A sink without dSink on the merged engine: dense and padded, non-causal and causal. dQ, dK
/// and dV depend on the sink only through the sink-inclusive Stats, which the harness hands the
/// device; the merged plan is built by the engine's id, so the merged engine served the graph.
void expectMergedServesASinkOnlyGraph(DataType dtype, bool gqa)
{
    for(const bool padded : {false, true})
    {
        for(const bool causal : {false, true})
        {
            SCOPED_TRACE(std::string(gqa ? "gqa" : "mha") + (padded ? " padded" : " dense")
                         + (causal ? " causal" : " non-causal"));
            auto c = sinkCase(dtype, gqa, padded, causal);
            c.withDsink = false;
            expectServedAs(MERGED_ENGINE, c);
            if(::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
}

// A FLOAT sink without a dSink: no kernel reads it, the sink-inclusive Stats carry it.
TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, MergedServesAFloatSinkWithoutDsink)
{
    auto c = sinkCase(DataType::BFLOAT16, true, true, true);
    c.withDsink = false;
    c.sinkType = DataType::FLOAT;
    expectServedAs(MERGED_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, SplitServesAFloatSinkWithoutDsink)
{
    auto c = sinkCase(DataType::BFLOAT16, false, false, true);
    c.withDsink = false;
    c.sinkType = DataType::FLOAT;
    expectServedAs(SPLIT_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, MergedServesASinkOnlyGraphOfMhaThroughTheCallersStats)
{
    expectMergedServesASinkOnlyGraph(DataType::BFLOAT16, false);
}

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, MergedServesASinkOnlyGraphOfGqaThroughTheCallersStats)
{
    expectMergedServesASinkOnlyGraph(DataType::BFLOAT16, true);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, MergedServesASinkOnlyGraphOfMhaThroughTheCallersStats)
{
    expectMergedServesASinkOnlyGraph(DataType::HALF, false);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, MergedServesASinkOnlyGraphOfGqaThroughTheCallersStats)
{
    expectMergedServesASinkOnlyGraph(DataType::HALF, true);
}

// ---------------------------------------------------------------------------
// Declined
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSink, SplitDeclinesAFloatSinkWithADsink)
{
    auto c = sinkCase(DataType::BFLOAT16, false, false, false);
    c.sinkType = DataType::FLOAT;
    c.withDsink = true;
    expectDeclined(SPLIT_ENGINE, specOfSink(c));
}

TEST_F(TestGpuFlydslSdpaBwdSink, SplitDeclinesAnFp32GraphWithASink)
{
    SinkCase c;
    c.shape.dtype = DataType::FLOAT;
    expectDeclined(SPLIT_ENGINE, specOfSink(c));
}

TEST_F(TestGpuFlydslSdpaBwdSink, MergedDeclinesADsinkGraph)
{
    auto c = sinkCase(DataType::BFLOAT16, false, false, false);
    c.withDsink = true;
    expectDeclined(MERGED_ENGINE, specOfSink(c));
}

// ---------------------------------------------------------------------------
// The split engine where no sink twin ships (gfx942, gfx90a, gfx11-generic)
// ---------------------------------------------------------------------------

/// The split engine opts into dSink on every arch, but only the gfx950 content ships the prep
/// sink twin. Elsewhere no prep row carries `sink=1`, so a dSink graph finds no kernel and the
/// engine declines it, while a sink-only graph runs the plain prep rows as on gfx950.
class TestGpuFlydslSdpaBwdSinkNoTwin : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSinkNoTwin()
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
        if(arch.rfind("gfx950", 0) == 0)
        {
            GTEST_SKIP() << "the gfx950 content ships the sink twin";
        }
    }
};

using TestGpuFlydslSdpaBwdSinkNoTwinBfp16 = TestGpuFlydslSdpaBwdSinkNoTwin;

TEST_F(TestGpuFlydslSdpaBwdSinkNoTwinBfp16, SplitDeclinesADsinkGraph)
{
    auto c = sinkCase(DataType::BFLOAT16, false, false, false);
    c.withDsink = true;
    expectDeclined(SPLIT_ENGINE, specOfSink(c));
}

TEST_F(TestGpuFlydslSdpaBwdSinkNoTwinBfp16, SplitServesASinkOnlyGraphWithThePlainPrepRows)
{
    auto c = sinkCase(DataType::BFLOAT16, false, false, true);
    c.withDsink = false;
    expectServedAs(SPLIT_ENGINE, c);
}

// ---------------------------------------------------------------------------
// Determinism (is_deterministic_algorithm)
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, SplitIsBitwiseRepeatableWhenTheGraphAsksForDeterminism)
{
    auto c = sinkCase(DataType::BFLOAT16, true, true, true);
    c.withSink = false;
    c.withDsink = false;
    c.deterministic = true;
    expectRepeatableAs(SPLIT_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSinkBfp16, MergedIsBitwiseRepeatableWhenTheGraphAsksForDeterminism)
{
    auto c = sinkCase(DataType::BFLOAT16, true, true, true);
    c.withSink = false;
    c.withDsink = false;
    c.deterministic = true;
    expectRepeatableAs(MERGED_ENGINE, c);
}

TEST_F(TestGpuFlydslSdpaBwdSinkFp16, SplitDsinkIsBitwiseRepeatableWhenTheGraphAsksForDeterminism)
{
    auto c = sinkCase(DataType::HALF, true, true, true);
    c.deterministic = true;
    expectRepeatableAs(SPLIT_ENGINE, c);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
