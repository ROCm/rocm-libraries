// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>
#include <set>
#include <string>

#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenBinaryPointwiseChecks.hpp"
#include "engines/plans/MiopenPointwiseTensorChecks.hpp"

namespace miopen_plugin::binary_pointwise_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

namespace
{

// The message the outer catch prepends. Kept out of every throw below so a helper's message
// and this prefix never double up (see
// pointwise_applicability::validatePointwiseIoTensors' message-prefix contract).
constexpr auto LOG_PREFIX = "Binary pointwise plan builder: ";

// Bundles a tensor's identity with its rank, computed once and reused across every check below
// instead of calling attr->dims()->size() repeatedly.
struct TensorInfo
{
    const TensorAttributes* attr;
    int64_t uid;
    uint32_t rank;
};

// Packed, channels-first strides. For every axis with dims[i] > 1, strides[i] must equal the
// product of the dims to its right; axes with dims[i] == 1 are unconstrained (never
// dereferenced). This single pass also covers the "last axis" special case in the spec: the
// running product starts at 1, so an all-ones tail already forces strides[r-1] == 1 there.
void checkPacked(const TensorInfo& tensor, const std::string& label)
{
    const auto* dims = tensor.attr->dims();
    const auto* strides = tensor.attr->strides();

    int64_t expectedStride = 1;
    for(int i = static_cast<int>(tensor.rank) - 1; i >= 0; --i)
    {
        const auto idx = static_cast<flatbuffers::uoffset_t>(i);
        if((*dims)[idx] > 1 && (*strides)[idx] != expectedStride)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                label + " tensor (uid " + std::to_string(tensor.uid)
                    + ") is not packed / channels-first at axis " + std::to_string(i));
        }
        expectedStride *= (*dims)[idx];
    }
}

void validateBinaryPointwise(const IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "applicable only for single-node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::PointwiseAttributes}))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "graph contains unsupported node attributes");
    }

    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "only supports nodes with an fp32 compute_data_type");
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    switch(attrs.operation())
    {
    case PointwiseMode::ADD:
    case PointwiseMode::SUB:
    case PointwiseMode::MUL:
    case PointwiseMode::MAX_OP:
    case PointwiseMode::MIN_OP:
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "unsupported pointwise mode: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(
                    attrs.operation())));
    }

    const auto in1Uid = attrs.in_1_tensor_uid();
    if(!in1Uid.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "requires in_1_tensor_uid to be present");
    }

    if(attrs.in_2_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "does not support ternary nodes; in_2_tensor_uid must not be present");
    }

    const auto in0Uid = attrs.in_0_tensor_uid();
    const auto in1UidValue = *in1Uid;
    const auto outUid = attrs.out_0_tensor_uid();

    const auto& tensorMap = opGraph.getTensorMap();
    auto findAttr = [&tensorMap](int64_t uid) -> const TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* in0Attr = findAttr(in0Uid);
    const auto* in1Attr = findAttr(in1UidValue);
    const auto* outAttr = findAttr(outUid);

    if(in0Attr == nullptr || in1Attr == nullptr || outAttr == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "one or more of in_0/in_1/out_0 tensor uids do not resolve in the graph's tensor map");
    }

    pointwise_applicability::validatePointwiseIoTensors({in0Attr, in1Attr, outAttr},
                                                        "Binary pointwise");

    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(in0Attr)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(in1Attr)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(outAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "does not support pass-by-value tensors");
    }

    if(in0Attr->ragged_offset_tensor_uid().has_value()
       || in1Attr->ragged_offset_tensor_uid().has_value()
       || outAttr->ragged_offset_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "does not support ragged tensors");
    }

    if(outUid == in0Uid || outUid == in1UidValue)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "does not support in-place execution; out_0_tensor_uid must differ from in_0 and in_1");
    }

    const TensorInfo in0{in0Attr, in0Uid, in0Attr->dims()->size()};
    const TensorInfo in1{in1Attr, in1UidValue, in1Attr->dims()->size()};
    const TensorInfo out{outAttr, outUid, outAttr->dims()->size()};

    // rank(out) in [3, 5]. The lower bound is load-bearing: without it a rank-0 tensor would
    // index strides[r-1] out of bounds in checkPacked.
    if(out.rank < 3 || out.rank > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "output tensor rank must be between 3 and 5, got " + std::to_string(out.rank));
    }

    // Strict rank equality, no padding: hipDNN's broadcast rules don't specify a pad direction,
    // and MIOpen's solver dispatch keys on A's rank without comparing it to C's, so a padded
    // rank could silently dispatch wrong.
    if(in0.rank != out.rank || in1.rank != out.rank)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "all operands must have the same rank as the output; hipDNN's implicit-broadcast "
            "rules are ambiguous for rank-mismatched pointwise inputs, so this provider declines "
            "them");
    }

    for(const auto& tensor : {in0, in1, out})
    {
        const auto* dims = tensor.attr->dims();
        const auto* strides = tensor.attr->strides();
        for(flatbuffers::uoffset_t i = 0; i < tensor.rank; ++i)
        {
            if((*dims)[i] <= 0 || (*strides)[i] <= 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "tensor (uid " + std::to_string(tensor.uid)
                        + ") has a non-positive dim or stride at axis " + std::to_string(i));
            }
        }
    }

    // numel(out) <= INT32_MAX -- MIOpen narrows work_per_wg to int.
    const auto* outDims = out.attr->dims();
    int64_t numel = 1;
    for(flatbuffers::uoffset_t i = 0; i < out.rank; ++i)
    {
        numel *= (*outDims)[i];
    }
    if(numel > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "output element count " + std::to_string(numel) + " exceeds INT32_MAX");
    }

    checkPacked(in0, "in_0");
    checkPacked(in1, "in_1");
    checkPacked(out, "out_0");

    // A = in_0, no operand swap. Requiring dims equality (not merely numel) is deliberate -- a
    // same-count/different-shape A is a silent-wrong-answer path.
    const auto* in0Dims = in0.attr->dims();
    for(flatbuffers::uoffset_t i = 0; i < out.rank; ++i)
    {
        if((*in0Dims)[i] != (*outDims)[i])
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "the first input must have dimensions exactly equal to the output; MIOpen's "
                "tensorOp cannot broadcast its first operand, and this provider does not swap "
                "operands");
        }
    }

    const auto* in1Dims = in1.attr->dims();
    for(flatbuffers::uoffset_t i = 0; i < out.rank; ++i)
    {
        if((*in1Dims)[i] != 1 && (*in1Dims)[i] != (*outDims)[i])
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "the second input is not broadcastable to the output at axis " + std::to_string(i));
        }
    }
}

} // namespace

bool isSupported(const IGraph& opGraph)
{
    try
    {
        validateBinaryPointwise(opGraph);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(LOG_PREFIX << e.what());
        return false;
    }
}

} // namespace miopen_plugin::binary_pointwise_applicability
