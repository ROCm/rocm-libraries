// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Two gates, both required. The ingestor gate is this engine's; HIPDNN_ENABLE_SDPA is
// the frontend's, and Graph::sdpa() does not exist without it -- a build with SDPA off
// looks clean for the wrong reason, silently compiling none of this.
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
 * @brief The first rocKE-produced kernel executed end to end, through the public frontend
 *        API, on a device.
 *
 * This closes a loop that no test previously crossed. The two halves each had their own
 * coverage and nothing joined them:
 *
 *   build time -- an authored descriptor names a rocKE builder; the packager imports it,
 *     lowers it through comgr, writes the code object into the per-arch kpack, and
 *     rewrites the UKD into `kind: "kpack"` form;
 *   run time  -- the loader walks the staged tree, resolves the archive relative to the
 *     descriptor that declared it, loads the module and resolves the symbol.
 *
 * What makes this the rocKE proof rather than another kpack proof is only WHERE the bytes
 * came from. Nothing in the runtime knows or cares: a rocKE code object is bytes in an
 * archive exactly like a hipcc one, which is the property this suite exists to keep true.
 * If a change ever makes the runtime branch on producer, this is the test that notices.
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

/// The problem the shipped descriptor's spec is compiled for. A rocKE kernel bakes its
/// shape in -- these are constants in the code object and appear in its symbol name -- so
/// the graph must match exactly or the engine's kernel matcher declines it. Restated here
/// rather than read from the descriptor on purpose: if someone edits the descriptor's spec
/// without editing this graph, the mismatch should FAIL, not silently retarget the test.
constexpr int64_t BATCH = 1;
constexpr int64_t NUM_QUERY_HEADS = 8;
constexpr int64_t NUM_KV_HEADS = 8;
constexpr int64_t SEQLEN_Q = 256;
constexpr int64_t SEQLEN_KV = 256;
constexpr int64_t HEAD_SIZE = 128;

/// 1/sqrt(head_size), the scale the kernel's `scale` argument expects. Written as a
/// literal expression rather than a constant so it tracks HEAD_SIZE.
const float ATTENTION_SCALE = 1.0F / std::sqrt(static_cast<float>(HEAD_SIZE));

/// bf16 attention accumulates in fp32 over SEQLEN_KV keys and rounds its output to bf16,
/// which carries 8 mantissa bits. The reference computes the same maths in fp32
/// throughout, so the two differ by the output rounding plus accumulated order-of-addition
/// error -- a few parts in 1e-2 at this sequence length, not the 1e-5 an fp32-vs-fp32
/// comparison would justify.
constexpr float BF16_TOLERANCE = 2.0e-2F;

/// Dims [B, H, S, D]; strides describe the BSHD (token-major) buffer the kernel reads.
///
/// The rocKE dense kernel consumes BSHD -- its docstring is explicit ("q/out are
/// [B, S, Hq, D] and k/v are [B, Skv, Hkv, D], dense contiguous",
/// attention_dense.py:1815) and its emitted addressing agrees (stride_q_tok = Hq * D).
/// Declaring those strides is therefore a description of the kernel's real ABI, not a
/// reinterpretation of it.
///
/// This only works because the tensor iterator now honours index order. It previously
/// selected its linear fast path on isPacked() alone, which any stride PERMUTATION
/// satisfies, so the random fill wrote in BHSD order while stride-aware reads fetched
/// in BSHD order -- reads and writes disagreeing, surfacing only as wrong numbers.
/// See visitsInIndexOrder in data_sdk Tensor.hpp.
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
/// tensor, no dropout, no paging, no stats -- every one of which the engine's graph
/// matcher refuses, because the dense kernel implements none of them.
/// The graph, plus its output tensor: the caller needs the tensor to register a
/// tolerance for it. executeAndVerify() cannot be used here -- it hardcodes uid 3 as the
/// output and derives its tolerance as reductionLength * FLT_EPSILON, which is an fp32
/// budget. This output is uid 4 and bf16.
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
    // O must declare the SAME BSHD layout as Q/K/V. The rocKE dense kernel writes its
    // output token-major with heads interleaved, exactly as it reads its inputs; leaving
    // the dims and strides to be inferred here gave O a different layout from the one the
    // kernel writes, so every per-head read landed inside head 0's region.
    o->set_uid(4)
        .set_name("O")
        .set_output(true)
        .set_dim({BATCH, NUM_QUERY_HEADS, SEQLEN_Q, HEAD_SIZE})
        .set_stride(
            {SEQLEN_Q * NUM_QUERY_HEADS * HEAD_SIZE, HEAD_SIZE, NUM_QUERY_HEADS * HEAD_SIZE, 1})
        .set_data_type(DataType::BFLOAT16);

    return {graph, o};
}

/// The directory the loader walks, derived the way the LOADER derives it -- from the
/// plugin module, not from a configure-time path. Same helper shape as the kpack suite's.
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
/// authored subpath is preserved into the shard, so hardcoding it here would couple this
/// test to a folder name that authors are free to change.
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

        // Skipping is right ONLY for "production packaging did not run" -- no hipcc, no
        // production source root, ingestor off upstream. Once the descriptor is on disk
        // everything below is an assertion, because then a failure is the regression this
        // suite exists to catch.
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
/// Weaker than the dispatch case below and deliberately separate: an engine that fails to
/// load and one that loads but declines the graph are different defects with the same
/// symptom at `build_plans`, and this separates them. It is also the case that fails if
/// the descriptor's `hipkernel.attention_dense.*` symbols and the native pack's
/// registrations ever drift apart, since the loader drops a set naming an unregistered
/// symbol.
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
        << "Either the descriptor set was dropped at load (every rejection is logged at "
           "ERROR naming the file and the reason -- an unregistered native symbol, a "
           "schema violation, or an unresolved cross-reference), or its graph_match "
           "declined this graph.";
}

/// The inverse, and the one that actually guards correctness: a graph the kernel CANNOT
/// serve must be declined at match time, not executed with the wrong addressing.
///
/// The dense kernel bakes BSHD strides into its code object, so a BHSD graph is not a
/// slower case or a degraded one -- it is silently wrong output, because the kernel keeps
/// indexing as BSHD whatever the tensor declares. Nothing faults and no status is raised,
/// so without this the failure only shows up as bad numbers in a caller far away.
///
/// Declining also has a positive effect: it lets another engine take the graph.
TEST_F(IntegrationGpuRockeAttentionDense, DeclinesAGraphWhoseLayoutItCannotServe)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Identical to the served graph in every respect except memory order: BHSD packs each
    // head contiguously (head stride S*D, token stride D) where the kernel expects heads
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
/// dispatched, and numerically correct against the CPU reference on a real device.
///
/// The engine is PINNED. Other engines can serve SDPA, so without a pin a green result
/// here would prove only that something computed attention -- which is exactly the way an
/// end-to-end test quietly stops testing the thing it was written for.
TEST_F(IntegrationGpuRockeAttentionDense, ExecutesARockeKernelOnDevice)
{
    auto [graph, output] = buildDenseSdpaGraph();

    // PINNED. Other engines can serve SDPA, so without a pin a green result here would
    // prove only that something computed attention -- exactly how an end-to-end test
    // quietly stops testing the thing it was written for.
    graph->set_preferred_engine_id_ext(rockeEngineId());

    // verifyGraph() drives build() itself, so the plan is built there. Everything the
    // build has to get right on the way is asserted after it: that the rocKE engine is
    // the one serving, and that its answer is right.
    // The plan builder catches each kernel's failure, logs it at WARN with the real
    // message, and then throws a summary naming only a COUNT. Without capturing the WARN
    // the summary is all a failure here would show -- "could not build a plan for any of
    // its 1 applicable kernel(s)" -- which says nothing about why. Record it so a failure
    // reports the cause rather than the tally.
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
        << rockeEngineId()
        << " -- so the numeric agreement above is not evidence about rocKE at all.\n"
        << "  descriptor : " << _descriptor;
}

/// Which layout does the kernel ACTUALLY read? Answered without a reference.
///
/// Softmax rows sum to 1, so if V is CONSTANT across tokens within a head, the output
/// is that constant regardless of Q, K, the causal mask, the scale, or any numerical
/// detail:
///
///     out[b,h,s,:] = sum_j P[s,j] * V[j,:] = c_h * sum_j P[s,j] = c_h
///
/// Giving each head a distinct c_h = h + 1 turns the output into a direct read-out of
/// which head's V the kernel picked up. A correct kernel writes h+1 throughout head h;
/// a layout disagreement yields a clean OTHER integer -- naming the head it actually
/// read -- rather than noise. That is what separates a layout bug from a numeric one,
/// a distinction the comparison against the CPU reference cannot make, since both
/// present as "the values differ".
///
/// Writes and reads both go through getIndex(), so the DECLARED strides decide where
/// each element lands and where it is read back from -- which is precisely the property
/// under test.
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

    // Assert the strides SURVIVED into the bundle. generateBundles builds these tensors
    // from the graph's tensor attributes, and everything below depends on them being the
    // BSHD strides this test declared -- if the frontend normalised them to packed
    // row-major, the probe would write heads S*D apart while the kernel reads them D
    // apart, and every head would land inside head 0's first token row. That is
    // indistinguishable from a kernel bug unless it is checked here.
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

    // Verify the probe's OWN input transport before drawing any conclusion from the
    // output. If the per-head constants did not survive the host->device round trip,
    // every head would read head 0's value for reasons that have nothing to do with the
    // kernel. Marking V device-modified forces the readback to come FROM the device
    // rather than re-reading the host copy just written, so this checks what the GPU
    // actually received. Whatever the output says afterwards is then a statement about
    // the kernel, not about the harness.
    v.markDeviceModified();
    const auto* vBack = static_cast<const bfloat16*>(v.rawHostData());
    ASSERT_NE(vBack, nullptr);
    for(int64_t h = 0; h < NUM_QUERY_HEADS; ++h)
    {
        const auto back = static_cast<float>(
            vBack[v.getIndex(std::vector<int64_t>{0, h, SEQLEN_KV / 2, HEAD_SIZE / 2})]);
        ASSERT_FLOAT_EQ(back, static_cast<float>(h + 1))
            << "V head " << h
            << " did not survive the round trip; the probe's own input "
               "is wrong and its output tells us nothing about the kernel";
    }

    auto& o = bundle.getTensor(4);
    o.markDeviceModified();
    const auto* oData = static_cast<const bfloat16*>(o.rawHostData());
    ASSERT_NE(oData, nullptr);

    // The conclusion below reads O by coordinate, so O's head stride decides whether the
    // eight reads address eight different places at all. If O were packed row-major
    // instead of BSHD, every read would land inside head 0's first token row and return
    // head 0's value no matter what the kernel wrote -- producing this test's exact
    // symptom from a harness fault. Assert the strides, then assert the offsets really
    // are distinct, so a passing conclusion cannot rest on eight reads of one address.
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
