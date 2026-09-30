// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

#include <cmath>
#include <fstream>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <stdexcept>

// Finding 2 fix: include Utils.hpp so HIP_KERNEL_RETURN_FALSE_IF is defined
#include "core/Utils.hpp"

#include "../asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "Flash2Dispatch.hpp"
#include "HipFlash2FwdPlan.hpp"
#include "HipFlash2FwdPlanBuilder_v2.hpp"
#include "HipFlash2KernelUtils.hpp"

namespace hip_flash2_engine
{

using namespace hip_kernel_provider_common;
using namespace hipdnn_flatbuffers_sdk;

// ---------------------------------------------------------------------------
// isApplicable
// ---------------------------------------------------------------------------
bool HipFlash2FwdPlanBuilder::isApplicable(const Handle& handle,
                                           const flatbuffer_utilities::IGraph& opGraph) const
{
    // Finding 2 fix: renamed to HIP_KERNEL_LOG_PREFIX (required by the macro)
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[HipFlash2FwdPlanBuilder::isApplicable] ";

    // Device check: gfx942 only (gfx950 not yet supported)
    std::string archId;
    try
    {
        archId = getDeviceString(handle.getStream());
        HIP_KERNEL_RETURN_FALSE_IF(archId != "gfx942",
                                   "Device not supported (actual: " + archId
                                       + "); only gfx942 is supported");
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR(HIP_KERNEL_LOG_PREFIX << "getDeviceString failed: " << e.what());
        return false;
    }

    // Single SDPA node
    auto& nodeWrappers = opGraph.nodeWrappers();
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.size() != 1, "Graph must have exactly one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodeWrappers.front()->attributesType()
                                   != data_objects::NodeAttributes::SdpaAttributes,
                               "Node must be SdpaAttributes");

    const auto& attrs = nodeWrappers.front()->attributesAs<data_objects::SdpaAttributes>();

    // Unsupported optional features
    HIP_KERNEL_RETURN_FALSE_IF(attrs.dropout_probability().has_value()
                                   && attrs.dropout_probability().value() != 0.f,
                               "dropout not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.alibi_mask(), "alibi_mask not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.padding_mask(), "padding_mask not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.attn_mask_tensor_uid(), "attn_mask tensor not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_k_tensor_uid(), "page_table_k not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.page_table_v_tensor_uid(), "page_table_v not supported");
    HIP_KERNEL_RETURN_FALSE_IF(attrs.generate_stats().value_or(false),
                               "LSE stats output not supported");
    // K2: reject mask types the kernel does not implement
    {
        const auto maskType = asm_sdpa_engine::plan_utils::getMaskType(attrs);
        HIP_KERNEL_RETURN_FALSE_IF(
            maskType != asm_sdpa_engine::plan_utils::MaskType::NO_MASK
                && maskType != asm_sdpa_engine::plan_utils::MaskType::TOP_LEFT_CAUSAL,
            "Only NO_MASK and TOP_LEFT_CAUSAL are supported");
    }
    HIP_KERNEL_RETURN_FALSE_IF(attrs.seq_len_q_tensor_uid().has_value()
                                   || attrs.seq_len_kv_tensor_uid().has_value(),
                               "variable-length (group) batch mode not supported");

    // Tensor shapes
    const auto& tensorMap = opGraph.getTensorMap();
    auto* qTensor = tensorMap.at(attrs.q_tensor_uid());
    auto* kTensor = tensorMap.at(attrs.k_tensor_uid());
    auto* vTensor = tensorMap.at(attrs.v_tensor_uid());
    auto* oTensor = tensorMap.at(attrs.o_tensor_uid());

    HIP_KERNEL_RETURN_FALSE_IF(qTensor->dims()->size() != 4, "Q must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(kTensor->dims()->size() != 4, "K must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(vTensor->dims()->size() != 4, "V must be rank-4");
    HIP_KERNEL_RETURN_FALSE_IF(oTensor->dims()->size() != 4, "O must be rank-4");

    // Data type: FP16 only
    const auto qType = qTensor->data_type();
    const auto kType = kTensor->data_type();
    const auto vType = vTensor->data_type();
    const auto oType = oTensor->data_type();
    const bool fp16
        = (qType == data_objects::DataType::HALF) && (kType == data_objects::DataType::HALF)
          && (vType == data_objects::DataType::HALF) && (oType == data_objects::DataType::HALF);
    HIP_KERNEL_RETURN_FALSE_IF(!fp16, "only FP16 Q/K/V/O is supported");

    // head_dim: {64, 128}
    const int headDim = static_cast<int>(qTensor->dims()->Get(3));
    HIP_KERNEL_RETURN_FALSE_IF(headDim != 64 && headDim != 128,
                               "head_dim must be 64 or 128 (actual: " + std::to_string(headDim)
                                   + ")");

    const int headDimV = static_cast<int>(vTensor->dims()->Get(3));
    HIP_KERNEL_RETURN_FALSE_IF(headDimV != headDim, "head_dim_v must equal head_dim_qk");

    // GQA divisibility (I1)
    const int numHeadsQ = static_cast<int>(qTensor->dims()->Get(1));
    const int numHeadsKv = static_cast<int>(kTensor->dims()->Get(1));
    HIP_KERNEL_RETURN_FALSE_IF(numHeadsKv <= 0 || numHeadsQ % numHeadsKv != 0,
                               "num_heads_q must be divisible by num_heads_kv for GQA (q="
                                   + std::to_string(numHeadsQ) + " kv=" + std::to_string(numHeadsKv)
                                   + ")");

    // Flash2 shape variables (declare before guards that reference them)
    const int seqLenQ = static_cast<int>(qTensor->dims()->Get(2));
    const int seqLenKv = static_cast<int>(kTensor->dims()->Get(2));

    // K3: reject partial query tiles (divergent __syncthreads under my_valid)
    HIP_KERNEL_RETURN_FALSE_IF(
        seqLenQ % 64 != 0,
        "seq_len_q must be a multiple of 64 -- partial tile causes divergent __syncthreads");

    // Flash2 crossover heuristic
    HIP_KERNEL_RETURN_FALSE_IF(!useFlash2ForShape(seqLenQ, seqLenKv),
                               "shape below Flash2 crossover threshold (seq_q="
                                   + std::to_string(seqLenQ) + " seq_kv=" + std::to_string(seqLenKv)
                                   + ")");

    // Stride contiguity: head_dim stride must be 1 for Q, K, V, O
    // The kernel assumes contiguous innermost dimension for all four tensors.
    HIP_KERNEL_RETURN_FALSE_IF(qTensor->strides()->Get(3) != 1,
                               "Q head_dim stride must be 1 (non-contiguous not supported)");
    HIP_KERNEL_RETURN_FALSE_IF(kTensor->strides()->Get(3) != 1,
                               "K head_dim stride must be 1 (non-contiguous not supported)");
    HIP_KERNEL_RETURN_FALSE_IF(vTensor->strides()->Get(3) != 1,
                               "V head_dim stride must be 1 (non-contiguous not supported)");
    HIP_KERNEL_RETURN_FALSE_IF(oTensor->strides()->Get(3) != 1,
                               "O head_dim stride must be 1 (non-contiguous not supported)");

    return true;
}

// ---------------------------------------------------------------------------
// getMaxWorkspaceSize
// ---------------------------------------------------------------------------
size_t HipFlash2FwdPlanBuilder::getMaxWorkspaceSize(const Handle& /*handle*/,
                                                    const flatbuffer_utilities::IGraph& /*opGraph*/,
                                                    const Settings& /*executionSettings*/) const
{
    return 0;
}

// ---------------------------------------------------------------------------
// initializeExecutionSettings
// ---------------------------------------------------------------------------
void HipFlash2FwdPlanBuilder::initializeExecutionSettings(
    const Handle& /*handle*/,
    const flatbuffer_utilities::IGraph& /*opGraph*/,
    const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    Settings& /*executionSettings*/) const
{
    HIPDNN_PLUGIN_LOG_INFO("HipFlash2FwdPlanBuilder::initializeExecutionSettings -- no-op");
}

// ---------------------------------------------------------------------------
// buildPlan
// ---------------------------------------------------------------------------
void HipFlash2FwdPlanBuilder::buildPlan(const Handle& handle,
                                        const flatbuffer_utilities::IGraph& opGraph,
                                        const flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
                                        Context& executionContext) const
{
    std::string archId;
    try
    {
        archId = getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        const std::string msg
            = std::string("HipFlash2FwdPlanBuilder::buildPlan -- getDeviceString: ") + e.what();
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    Flash2FwdParams params = extractParams(handle, opGraph);
    params.archString = archId;

    // ---- Select the kernel variant for this shape --------------------------
    // Falls back to the legacy single-kernel object when a per-variant .co is
    // not installed, so an engine built from an older kernels/ directory keeps
    // exactly its previous behaviour. This makes the dispatcher inert until a
    // matching variant set is shipped -- it cannot regress the current engine.
    int cuCount = 304; // gfx942 default; overridden from the device below
    {
        hipDeviceProp_t prop{};
        int dev = 0;
        if(hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess
           && prop.multiProcessorCount > 0)
        {
            cuCount = prop.multiProcessorCount;
        }
    }

    const Flash2Selection sel = selectFlash2Config(
        params.batch, params.numHeadsQ, params.seqLenQ, params.headDim, params.causal, cuCount);

    // Split-K execution is not yet plumbed through execute() (it needs a second
    // merge launch plus a workspace pointer). Record the decision, run single
    // pass for now.
    params.splitK = 1;
    params.workspaceBytes = 0;

    std::string coPath = flash2CoPath(archId, sel.variant.tag);
    {
        const std::ifstream probe(coPath, std::ios::binary);
        if(probe.good())
        {
            params.variantTag = sel.variant.tag;
            params.blockDim = sel.variant.blockDim;
            params.qPerCta = sel.variant.qPerCta;
        }
        else
        {
            HIPDNN_PLUGIN_LOG_INFO("HipFlash2FwdPlanBuilder -- variant '"
                                   << sel.variant.tag
                                   << "' not installed, using legacy kernel object");
            coPath = flash2CoPath(archId);
            params.variantTag = K_FLASH2_LEGACY.tag;
            params.blockDim = K_FLASH2_LEGACY.blockDim;
            params.qPerCta = K_FLASH2_LEGACY.qPerCta;
        }
    }
    const char* funcName = flash2KernelName(params.headDim);
    if(funcName == nullptr)
    {
        const std::string msg = "HipFlash2FwdPlanBuilder::buildPlan -- unsupported head_dim="
                                + std::to_string(params.headDim);
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    HIPDNN_PLUGIN_LOG_INFO("HipFlash2FwdPlanBuilder::buildPlan -- loading " << coPath
                                                                            << " fn=" << funcName);

    auto kernelOpt = loadKernelModule(coPath, funcName);
    if(!kernelOpt)
    {
        const std::string msg
            = "HipFlash2FwdPlanBuilder::buildPlan -- failed to load kernel from: " + coPath;
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    // Verify the object was actually built with the geometry the variant table
    // claims. The file probe above only proves a file exists at that path --
    // it says nothing about how the object was compiled. A mismatch is not
    // benign in EITHER direction: an object built for fewer threads than the
    // table claims fails every launch with hipError 719, and one built for
    // more threads launches fine and computes silently wrong results.
    //
    // The comparison is therefore exact, not an upper bound. An earlier
    // revision used `<`, which caught only the loud direction; S. Reeder
    // demonstrated the gap by copying the 512-thread w8q2k4 object over the
    // w4q1k4 filename (table: 256 threads) and running the integration suite:
    // `<` threw 0 times and produced 4 wrong results, `!=` threw 4 times and
    // produced 0. Exact comparison cannot false-positive here because every
    // variant carries __launch_bounds__(F2_BLOCK, 1), so
    // max_flat_workgroup_size equals the compiled block size exactly on all
    // five (measured 256/512/512/512/512 against a table of the same).
    if(!params.variantTag.empty())
    {
        int maxThreads = 0;
        const hipError_t attrErr = hipFuncGetAttribute(
            &maxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernelOpt->function());
        if(attrErr != hipSuccess)
        {
            const std::string msg
                = "HipFlash2FwdPlanBuilder::buildPlan -- hipFuncGetAttribute failed for " + coPath
                  + ": " + std::string(hipGetErrorString(attrErr));
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }
        if(maxThreads != static_cast<int>(params.blockDim))
        {
            const std::string msg
                = "HipFlash2FwdPlanBuilder::buildPlan -- variant '" + params.variantTag
                  + "' geometry mismatch: " + coPath + " was built for "
                  + std::to_string(maxThreads) + " threads/block but the variant table claims "
                  + std::to_string(params.blockDim)
                  + ". The installed kernel object does not match the variant it is named for.";
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }
    }

    executionContext.setPlan(
        std::make_unique<HipFlash2FwdPlan>(std::move(*kernelOpt), std::move(params)));
}

// ---------------------------------------------------------------------------
// getCustomKnobs
// ---------------------------------------------------------------------------
std::vector<data_objects::KnobT>
    HipFlash2FwdPlanBuilder::getCustomKnobs(const Handle& /*handle*/,
                                            const flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return {};
}

// ---------------------------------------------------------------------------
// extractParams (private helper) -- Finding 1 fix: restored missing body
// ---------------------------------------------------------------------------
Flash2FwdParams HipFlash2FwdPlanBuilder::extractParams(const Handle& /*handle*/,
                                                       const flatbuffer_utilities::IGraph& opGraph)
{
    Flash2FwdParams p{};

    auto& sdpaNode = opGraph.getNodeWrapper(0);
    auto& attrs = sdpaNode.attributesAs<data_objects::SdpaAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    p.qUid = attrs.q_tensor_uid();
    p.kUid = attrs.k_tensor_uid();
    p.vUid = attrs.v_tensor_uid();
    p.oUid = attrs.o_tensor_uid();

    auto* q = tensorMap.at(p.qUid);
    auto* k = tensorMap.at(p.kUid);
    auto* v = tensorMap.at(p.vUid);
    auto* o = tensorMap.at(p.oUid);

    p.batch = static_cast<int>(q->dims()->Get(0));
    p.numHeadsQ = static_cast<int>(q->dims()->Get(1));
    p.seqLenQ = static_cast<int>(q->dims()->Get(2));
    p.headDim = static_cast<int>(q->dims()->Get(3));

    p.numHeadsK = static_cast<int>(k->dims()->Get(1));
    p.seqLenKv = static_cast<int>(k->dims()->Get(2));

    p.attnScale = 0.0f;
    if(attrs.attn_scale_value().has_value())
    {
        p.attnScale = attrs.attn_scale_value().value();
    }

    p.causal = attrs.causal_mask();

    p.qStrideBatch = q->strides()->Get(0);
    p.qStrideHead = q->strides()->Get(1);
    p.qStrideSeq = q->strides()->Get(2);
    p.kStrideBatch = k->strides()->Get(0);
    p.kStrideHead = k->strides()->Get(1);
    p.kStrideSeq = k->strides()->Get(2);
    p.vStrideBatch = v->strides()->Get(0);
    p.vStrideHead = v->strides()->Get(1);
    p.vStrideSeq = v->strides()->Get(2);
    p.oStrideBatch = o->strides()->Get(0);
    p.oStrideHead = o->strides()->Get(1);
    p.oStrideSeq = o->strides()->Get(2);

    return p;
}

} // namespace hip_flash2_engine
