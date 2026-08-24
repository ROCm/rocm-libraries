// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Two gates, both required: the ingestor gate is this engine's, and HIPDNN_ENABLE_SDPA is
// the frontend's -- Graph::sdpa() does not exist without it.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuRockeAttentionDense.cpp
 * @brief A rocKE-produced kernel executed end to end, through the public frontend API, on
 *        a device.
 *
 * The path under test spans both halves of the pipeline:
 *
 *   build time -- an authored descriptor names a rocKE builder; the packager imports it,
 *     lowers it through comgr, writes the code object into the per-arch kpack, and
 *     rewrites the UKD into `kind: "kpack"` form;
 *   run time  -- the loader walks the staged tree, resolves the archive relative to the
 *     descriptor that declared it, loads the module and resolves the symbol.
 *
 * The runtime does not branch on which tool produced the code object: a rocKE one is bytes
 * in an archive exactly like a hipcc one. This suite holds that property in place.
 *
 * Like its kpack sibling, the suite sets no HIPDNN_DESCRIPTOR_DIR and its CTest entry
 * carries no ENVIRONMENT override: the module-relative walk from the loaded plugin is part
 * of what is under test, and pointing the binary somewhere else would test the override.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

/// The engine the rocKE descriptor set declares. Registered by
/// packs/AttentionDenseNative.cpp and named by attention_dense.ued.json; the two spellings
/// must agree or the loader drops the set during its symbol pre-flight.
constexpr const char* ROCKE_ENGINE_NAME = "hipkernel:AttentionDense";

/// The one arch the shipped descriptor is built for, matching its `arch` field.
constexpr const char* ROCKE_DESCRIPTOR_ARCH = "gfx942";

/// The problem the shipped descriptor's spec is compiled for. A rocKE kernel bakes its
/// shape in -- these are constants in the code object and appear in its symbol name -- so
/// the graph must match exactly or the engine's kernel matcher declines it. Restated here
/// rather than read from the descriptor so that editing the descriptor's spec without
/// editing this graph fails rather than silently retargets the test.
constexpr int64_t BATCH = 1;
constexpr int64_t NUM_QUERY_HEADS = 8;
constexpr int64_t NUM_KV_HEADS = 8;
constexpr int64_t SEQLEN_Q = 256;
constexpr int64_t SEQLEN_KV = 256;
constexpr int64_t HEAD_SIZE = 128;

/// 1/sqrt(head_size), the scale the kernel's `scale` argument expects. Written as an
/// expression rather than a literal so it tracks HEAD_SIZE.
const float ATTENTION_SCALE = 1.0F / std::sqrt(static_cast<float>(HEAD_SIZE));

/// bf16 attention accumulates in fp32 over SEQLEN_KV keys and rounds its output to bf16,
/// which carries 8 mantissa bits. The reference computes the same maths in fp32
/// throughout, so the two differ by the output rounding plus accumulated
/// order-of-addition error: a few parts in 1e-2 at this sequence length, not the 1e-5 an
/// fp32-vs-fp32 comparison would justify.
constexpr float BF16_TOLERANCE = 2.0e-2F;

/// Dims [B, H, S, D]; strides describe the BSHD (token-major) buffer the kernel reads.
///
/// The rocKE dense kernel's ABI is BSHD: its docstring declares "q/out are [B, S, Hq, D]
/// and k/v are [B, Skv, Hkv, D], dense contiguous", and its emitted addressing agrees
/// (stride_q_tok = Hq * D).
std::shared_ptr<TensorAttributes>
    makeAttentionTensor(int64_t uid, const std::string& name, int64_t heads, int64_t sequence)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({BATCH, heads, sequence, HEAD_SIZE})
        .set_stride({sequence * heads * HEAD_SIZE, HEAD_SIZE, heads * HEAD_SIZE, 1})
        .set_data_type(DataType::BFLOAT16);
    return tensor;
}

/// The one graph shape the rocKE descriptor claims: a single causal SDPA node, no mask
/// tensor, no dropout, no paging, no stats.
///
/// The output tensor is returned alongside the graph because the caller registers a
/// tolerance against it. executeAndVerify() cannot be used: it hardcodes uid 3 as the
/// output and derives its tolerance as reductionLength * FLT_EPSILON, an fp32 budget,
/// while this output is uid 4 and bf16.
struct DenseSdpaGraph
{
    std::shared_ptr<Graph> graph;
    std::shared_ptr<TensorAttributes> output;
};

DenseSdpaGraph buildDenseSdpaGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("rocke_attention_dense")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = makeAttentionTensor(1, "Q", NUM_QUERY_HEADS, SEQLEN_Q);
    auto k = makeAttentionTensor(2, "K", NUM_KV_HEADS, SEQLEN_KV);
    auto v = makeAttentionTensor(3, "V", NUM_KV_HEADS, SEQLEN_KV);

    SdpaAttributes attributes;
    attributes.set_name("rocke_attention_dense")
        .set_causal_mask(true)
        .set_attn_scale(ATTENTION_SCALE);

    auto [o, stats] = graph->sdpa(q, k, v, attributes);
    EXPECT_EQ(stats, nullptr) << "generate_stats was not requested, so no stats tensor "
                                 "should have been produced";
    // O declares the same BSHD layout as Q/K/V: the kernel writes its output token-major
    // with heads interleaved, exactly as it reads its inputs. Inferred dims and strides
    // would give O a different layout from the one the kernel writes.
    o->set_uid(4)
        .set_name("O")
        .set_output(true)
        .set_dim({BATCH, NUM_QUERY_HEADS, SEQLEN_Q, HEAD_SIZE})
        .set_stride(
            {SEQLEN_Q * NUM_QUERY_HEADS * HEAD_SIZE, HEAD_SIZE, NUM_QUERY_HEADS * HEAD_SIZE, 1})
        .set_data_type(DataType::BFLOAT16);

    return {graph, o};
}

/// The directory the loader walks, derived the way the loader derives it: from the plugin
/// module, not from a configure-time path. Same helper shape as the kpack suite's.
std::filesystem::path packagedDescriptorRoot()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    return std::filesystem::weakly_canonical(getCurrentExecutableDirectory()
                                             / pluginTarget.parent_path()
                                             / HIPDNN_PACKAGED_FIXTURE_SUBDIR);
}

/// The packed rocKE descriptor, or an empty path when production packaging did not run.
///
/// Located by walking for the file rather than by joining a fixed relative path: the
/// authored subpath is preserved into the shard, so a fixed path would couple this test to
/// a folder name authors are free to change.
std::filesystem::path findRockeDenseDescriptor()
{
    const auto root = packagedDescriptorRoot();

    std::error_code ec;
    if(!std::filesystem::is_directory(root, ec))
    {
        return {};
    }

    std::vector<std::filesystem::path> found;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(
            root, std::filesystem::directory_options::skip_permission_denied, ec))
    {
        if(entry.path().filename() == "attention_dense.kdp.json")
        {
            found.push_back(entry.path());
        }
    }

    std::sort(found.begin(), found.end());
    return found.empty() ? std::filesystem::path{} : found.front();
}

} // namespace

class IntegrationGpuRockeAttentionDense
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float, int>
{
protected:
    void SetUp() override
    {
        IntegrationGraphVerificationHarness<float, int>::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }

        // The descriptor declares `arch: ["gfx942"]`, and the ingestor drops a pack whose
        // arch the device does not satisfy (archSupports, KernelIngestorStateManager).
        // On any other device the engine is therefore never a candidate and every case
        // below asserts about an engine that cannot appear.
        const auto arch = hip_kernel_provider_common::getDeviceString(_stream);
        if(arch != ROCKE_DESCRIPTOR_ARCH)
        {
            GTEST_SKIP() << "the rocKE attention_dense descriptor is built for "
                         << ROCKE_DESCRIPTOR_ARCH << "; this device is " << arch;
        }

        // Skipping is right only for "production packaging did not run" -- no hipcc, no
        // production source root, ingestor off upstream. Once the descriptor is on disk,
        // everything below is an assertion.
        _descriptor = findRockeDenseDescriptor();
        if(_descriptor.empty())
        {
            GTEST_SKIP() << "no rocKE attention_dense descriptor under " << packagedDescriptorRoot()
                         << " -- production packaging did not run. Configure with "
                            "-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON, "
                            "-DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON and a "
                            "HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT.";
        }
    }

    static int64_t rockeEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ROCKE_ENGINE_NAME);
    }

    std::filesystem::path _descriptor;
};

// ---------------------------------------------------------------------------
// The descriptor set reaches the runtime
// ---------------------------------------------------------------------------

/// The rocKE engine is one the runtime actually built.
///
/// Separate from the dispatch case below: an engine that fails to load and one that loads
/// but declines the graph are different defects with the same symptom at `build_plans`.
/// This case also fails if the descriptor's `hipkernel.attention_dense.*` symbols and the
/// native pack's registrations drift apart, since the loader drops a set naming an
/// unregistered symbol.
TEST_F(IntegrationGpuRockeAttentionDense, RockeEngineLoadsFromThePackagedTree)
{
    auto [graph, output] = buildDenseSdpaGraph();
    static_cast<void>(output);

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), rockeEngineId()),
              rankedEngineIds.end())
        << "the rocKE engine '" << ROCKE_ENGINE_NAME << "' (id " << rockeEngineId()
        << ") did not offer itself for a graph its descriptor claims.\n"
        << "  descriptor : " << _descriptor << "\n"
        << "Either the descriptor set was dropped at load -- every rejection is logged at "
           "ERROR naming the file and the reason -- or its graph_match declined this graph.";
}

/// A graph the kernel cannot serve is declined at match time rather than executed with the
/// wrong addressing.
///
/// The dense kernel bakes BSHD strides into its code object, so it indexes a BHSD graph as
/// BSHD and returns wrong values without faulting and without raising a status. Declining
/// also lets another engine take the graph.
TEST_F(IntegrationGpuRockeAttentionDense, DeclinesAGraphWhoseLayoutItCannotServe)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Identical to the served graph except in memory order: BHSD packs each head
    // contiguously (head stride S*D, token stride D) where the kernel expects heads
    // interleaved within a token.
    const auto bhsd = [&](int64_t uid, const std::string& name, int64_t heads, int64_t sequence) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid)
            .set_name(name)
            .set_dim({BATCH, heads, sequence, HEAD_SIZE})
            .set_stride({heads * sequence * HEAD_SIZE, sequence * HEAD_SIZE, HEAD_SIZE, 1})
            .set_data_type(DataType::BFLOAT16);
        return tensor;
    };

    auto q = bhsd(1, "Q", NUM_QUERY_HEADS, SEQLEN_Q);
    auto k = bhsd(2, "K", NUM_KV_HEADS, SEQLEN_KV);
    auto v = bhsd(3, "V", NUM_KV_HEADS, SEQLEN_KV);

    SdpaAttributes attributes;
    attributes.set_name("rocke_attention_dense_bhsd")
        .set_causal_mask(true)
        .set_attn_scale(ATTENTION_SCALE);

    auto [o, stats] = graph->sdpa(q, k, v, attributes);
    ASSERT_EQ(stats, nullptr);
    o->set_uid(4).set_name("O").set_output(true).set_data_type(DataType::BFLOAT16);

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), rockeEngineId()),
              rankedEngineIds.end())
        << "the rocKE engine offered itself for a BHSD graph. Its kernel bakes BSHD "
           "addressing, so it would run and return wrong numbers rather than fail.";
}

// ---------------------------------------------------------------------------
// ...and the kernel it names runs, correctly
// ---------------------------------------------------------------------------

/// The whole loop: authored rocKE descriptor -> comgr -> kpack -> staged tree -> loaded,
/// dispatched, and numerically correct against the CPU reference on a device.
TEST_F(IntegrationGpuRockeAttentionDense, ExecutesARockeKernelOnDevice)
{
    auto [graph, output] = buildDenseSdpaGraph();

    // Pinned: other engines can serve SDPA, so an unpinned green result would prove only
    // that something computed attention.
    graph->set_preferred_engine_id_ext(rockeEngineId());

    // verifyGraph() drives build() itself. The plan builder catches each kernel's failure,
    // logs it at WARN with the real message, then throws a summary naming only a count
    // ("could not build a plan for any of its 1 applicable kernel(s)"). Record the WARN so
    // a failure here reports the cause rather than the tally.
    hipdnnSeverity_t savedLogLevel = HIPDNN_SEV_OFF;
    ASSERT_EQ(getGlobalLogLevel(savedLogLevel).code, ErrorCode::OK);
    auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    ASSERT_EQ(setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                 HIPDNN_SEV_WARN,
                                 LogCallbackMode::SYNC,
                                 this)
                  .code,
              ErrorCode::OK);
    ASSERT_EQ(setGlobalLogLevel(HIPDNN_SEV_WARN).code, ErrorCode::OK);

    registerValidator(output, BF16_TOLERANCE);
    verifyGraph(*graph, /*seed=*/0);
    const bool verified = !HasFatalFailure() && !HasNonfatalFailure();

    setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                       HIPDNN_SEV_OFF,
                       LogCallbackMode::SYNC,
                       this);
    setGlobalLogLevel(savedLogLevel);

    ASSERT_TRUE(verified) << "the rocKE kernel did not execute. What the engine logged "
                             "while trying:\n"
                          << recorder.getRecordedLogsAsString();

    int64_t servingEngineId = 0;
    ASSERT_EQ(graph->get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
    EXPECT_EQ(servingEngineId, rockeEngineId())
        << "engine id " << servingEngineId << " served the graph, not the pinned rocKE engine "
        << rockeEngineId() << ", so the numeric agreement above is not evidence about rocKE.\n"
        << "  descriptor : " << _descriptor;
}

/// Which layout the kernel reads, determined without a reference.
///
/// Softmax rows sum to 1, so with V constant across tokens within a head the output is
/// that constant regardless of Q, K, the causal mask, the scale, or any numerical detail:
///
///     out[b,h,s,:] = sum_j P[s,j] * V[j,:] = c_h * sum_j P[s,j] = c_h
///
/// A distinct c_h = h + 1 per head turns the output into a read-out of which head's V the
/// kernel picked up: a correct kernel writes h+1 throughout head h, and a layout
/// disagreement yields another head's integer rather than noise. The comparison against
/// the CPU reference cannot make that distinction, since both present as differing values.
///
/// Writes and reads both go through getIndex(), so the declared strides decide where each
/// element lands and where it is read back from.
TEST_F(IntegrationGpuRockeAttentionDense, ReadsOperandsInTheLayoutItDeclares)
{
    using hipdnn_data_sdk::types::bfloat16;

    auto [graph, output] = buildDenseSdpaGraph();
    graph->set_preferred_engine_id_ext(rockeEngineId());
    ASSERT_EQ(graph->build(_handle).code, ErrorCode::OK);

    hipdnn_test_sdk::utilities::GraphTensorBundle bundle;
    hipdnn_test_sdk::utilities::GraphTensorBundle unused;
    std::vector<int64_t> outputIds;
    generateBundles(*graph, unused, bundle, outputIds);

    // Q and K may be anything: the identity above holds for any softmax row.
    bundle.randomizeTensor(1, -1.0F, 1.0F, /*seed=*/1);
    bundle.randomizeTensor(2, -1.0F, 1.0F, /*seed=*/2);

    // V constant per head, addressed by coordinate through the tensor's own index map.
    auto& v = bundle.getTensor(3);

    // The strides must survive into the bundle, which generateBundles builds from the
    // graph's tensor attributes. Normalised to packed row-major, the probe would write
    // heads S*D apart while the kernel reads them D apart, landing every head inside head
    // 0's first token row -- the same symptom as a kernel layout fault.
    ASSERT_EQ(v.strides().size(), 4U);
    EXPECT_EQ(v.strides()[0], SEQLEN_KV * NUM_KV_HEADS * HEAD_SIZE) << "batch stride";
    EXPECT_EQ(v.strides()[1], HEAD_SIZE) << "HEAD stride: BSHD requires D, not S*D";
    EXPECT_EQ(v.strides()[2], NUM_KV_HEADS * HEAD_SIZE) << "token stride: BSHD requires H*D";
    EXPECT_EQ(v.strides()[3], 1) << "element stride";
    auto* vData = static_cast<bfloat16*>(v.rawHostData());
    ASSERT_NE(vData, nullptr);
    for(int64_t h = 0; h < NUM_QUERY_HEADS; ++h)
    {
        const auto value = static_cast<bfloat16>(static_cast<float>(h + 1));
        for(int64_t sq = 0; sq < SEQLEN_KV; ++sq)
        {
            for(int64_t d = 0; d < HEAD_SIZE; ++d)
            {
                vData[v.getIndex(std::vector<int64_t>{0, h, sq, d})] = value;
            }
        }
    }
    v.markHostModified();

    auto pack = bundle.toDeviceVariantPack();
    ASSERT_EQ(graph->execute(_handle, pack, nullptr).code, ErrorCode::OK);
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

    // Check the probe's own input transport before drawing any conclusion from the output:
    // per-head constants that did not survive the host-to-device round trip would make
    // every head read head 0's value for reasons unrelated to the kernel. Marking V
    // device-modified forces the readback to come from the device rather than re-reading
    // the host copy just written.
    v.markDeviceModified();
    const auto* vBack = static_cast<const bfloat16*>(v.rawHostData());
    ASSERT_NE(vBack, nullptr);
    for(int64_t h = 0; h < NUM_QUERY_HEADS; ++h)
    {
        const auto back = static_cast<float>(
            vBack[v.getIndex(std::vector<int64_t>{0, h, SEQLEN_KV / 2, HEAD_SIZE / 2})]);
        ASSERT_FLOAT_EQ(back, static_cast<float>(h + 1))
            << "V head " << h
            << " did not survive the round trip, so the probe's own input is wrong and its "
               "output says nothing about the kernel";
    }

    auto& o = bundle.getTensor(4);
    o.markDeviceModified();
    const auto* oData = static_cast<const bfloat16*>(o.rawHostData());
    ASSERT_NE(oData, nullptr);

    // The conclusion below reads O by coordinate, so O's head stride decides whether the
    // eight reads address eight distinct places at all. Packed row-major instead of BSHD,
    // every read would land inside head 0's first token row and return head 0's value
    // whatever the kernel wrote. Assert the strides, then that the offsets are distinct.
    ASSERT_EQ(o.strides().size(), 4U);
    EXPECT_EQ(o.strides()[1], HEAD_SIZE) << "O head stride: BSHD requires D, not S*D";
    EXPECT_EQ(o.strides()[2], NUM_QUERY_HEADS * HEAD_SIZE) << "O token stride: BSHD requires H*D";

    std::set<int64_t> probedOffsets;
    for(int64_t h = 0; h < NUM_QUERY_HEADS; ++h)
    {
        probedOffsets.insert(o.getIndex(std::vector<int64_t>{0, h, SEQLEN_Q / 2, HEAD_SIZE / 2}));
    }
    ASSERT_EQ(probedOffsets.size(), static_cast<size_t>(NUM_QUERY_HEADS))
        << "the per-head probe reads collapsed onto fewer addresses than heads, so this "
           "test cannot distinguish the heads it claims to compare";

    for(int64_t h = 0; h < NUM_QUERY_HEADS; ++h)
    {
        const auto got = static_cast<float>(
            oData[o.getIndex(std::vector<int64_t>{0, h, SEQLEN_Q / 2, HEAD_SIZE / 2})]);
        EXPECT_NEAR(got, static_cast<float>(h + 1), 0.1F)
            << "head " << h << " returned " << got << ", i.e. it read head "
            << (static_cast<int>(got) - 1)
            << "'s V. The kernel does not read operands in the layout this test declares.";
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
