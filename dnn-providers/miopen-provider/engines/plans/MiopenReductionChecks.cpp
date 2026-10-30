// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenReductionChecks.hpp"

namespace miopen_plugin::reduction_applicability
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

// MIOpen's legacy miopenReduceTensor predates the solver framework. It validates only some of
// its limits (fewer before #12874), and none of bf16, aliasing or overlapping output strides, so
// a graph outside them returns a wrong answer (or fails to compile a kernel at execute time)
// instead of an error. Every rule below names the MIOpen behaviour it guards against; function
// names refer to projects/miopen/src/reducetensor.cpp.
//
// alpha/beta need no rule: hipDNN's ReductionAttributes has no scaling fields and the plan
// always passes alpha = 1, beta = 0. That keeps the C = alpha * op(A) + beta * C blend, which
// MIOpen accepts for every op but tests only for ADD, MUL and AVG, out of reach.

namespace
{

// The message the outer catch prepends; kept out of every throw below.
constexpr auto LOG_PREFIX = "Reduction plan builder: ";

// miopenReduceTensor takes at most 6 dimensions. Before #12874 only ReduceTensor enforced this;
// miopenGetReductionWorkspaceSize did not, so check before the plan queries the workspace.
constexpr uint32_t MAX_RANK = 6;

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

// default_tunable_generic_reduction (src/include/miopen/reduce_tunables.hpp): 256 threads per
// block, each block copying two elements per thread at a time.
constexpr int64_t LEGACY_BLOCK_SIZE = 256;
constexpr int64_t LEGACY_BLOCK_SLICE = 2 * LEGACY_BLOCK_SIZE;

[[noreturn]] void decline(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, message);
}

// Element count of dims already checked to be positive and at most INT32_MAX, saturating at
// INT32_MAX + 1 so that the caller's limit check cannot be fooled by a wrapped product.
int64_t elementCount(const flatbuffers::Vector<int64_t>& dims)
{
    int64_t count = 1;
    for(const auto dim : dims)
    {
        if(count > INT32_LIMIT / dim)
        {
            return INT32_LIMIT + 1;
        }
        count *= dim;
    }
    return count;
}

void checkTensorSupported(const TensorAttributes& tensor, const std::string& label)
{
    // A single-node graph has no producer or consumer for a virtual tensor, and MIOpen reads
    // and writes real device buffers.
    if(tensor.virtual_())
    {
        decline(label + " tensor must be non-virtual");
    }

    // A pass-by-value scalar lives on the host and has no device buffer to reduce from or into.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        decline(label + " tensor must not be pass-by-value");
    }

    // Legacy reduce addresses dense strided tensors only.
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        decline(label + " tensor must not be ragged");
    }

    const auto dataType = tensor.data_type();
    // BFLOAT16 is declined because the legacy CK reduce kernels have no bf16 entry in
    // ck::get_datatype_from_enum (legacy_composable_kernel/.../data_type_enum_helper.hpp).
    // mapDataTypeId accepts miopenBFloat16, so the descriptors build and the workspace query
    // succeeds, but hipRTC fails to compile the kernel inside miopenReduceTensor.
    if(dataType != DataType::FLOAT && dataType != DataType::HALF)
    {
        decline(label + " tensor data type "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType))
                + " is not supported; only FLOAT and HALF are");
    }

    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr)
    {
        decline(label + " tensor dims or strides are null");
    }
    if(dims->size() != strides->size())
    {
        decline(label + " tensor dims and strides size mismatch");
    }

    // ValidateReduceDescriptors (rank) and the tensor descriptor (non-positive extents).
    if(dims->empty() || dims->size() > MAX_RANK)
    {
        decline(label + " tensor rank must be between 1 and " + std::to_string(MAX_RANK) + ", got "
                + std::to_string(dims->size()));
    }

    // ReduceTensor narrows every length and stride to int, and the kernels address global
    // memory through 32-bit byte offsets. MIOpen releases before #12874 (which added
    // ValidateReduceTensorFitsIntoInt) do neither check, so anything larger is silently
    // truncated. MIOpen's tensor descriptor also rejects zero lengths and strides, and a
    // negative one has no MIOpen representation.
    for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
    {
        const auto dim = (*dims)[i];
        const auto stride = (*strides)[i];
        if(dim <= 0 || stride <= 0)
        {
            decline(label + " tensor has a non-positive dim or stride at axis "
                    + std::to_string(i));
        }
        if(dim > INT32_LIMIT || stride > INT32_LIMIT)
        {
            decline(label + " tensor has a dim or stride exceeding INT32_MAX at axis "
                    + std::to_string(i));
        }
    }

    if(tensorSpanBytes(tensor) > INT32_LIMIT)
    {
        decline(label + " tensor spans more than INT32_MAX bytes");
    }
}

// Distinct output elements must not share an address: the kernel writes each one exactly once,
// so aliased elements race. Neither hipDNN nor MIOpen checks this. Taking the axes of length
// above 1 by increasing stride, each stride must step past everything the smaller axes reach.
void checkOutputDoesNotOverlap(const TensorAttributes& output)
{
    const auto* dims = output.dims();
    const auto* strides = output.strides();

    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, dim)
    for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
    {
        if((*dims)[i] > 1)
        {
            axes.emplace_back((*strides)[i], (*dims)[i]);
        }
    }
    std::sort(axes.begin(), axes.end());

    // The checked spans bound every term below by INT32_MAX.
    int64_t lastOffset = 0;
    for(const auto& [stride, dim] : axes)
    {
        if(stride <= lastOffset)
        {
            decline("output tensor strides make distinct elements overlap");
        }
        lastOffset += (dim - 1) * stride;
    }
}

} // namespace

void checkSupported(const IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        decline("applicable only for single-node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    if(!opGraph.hasOnlySupportedAttributes(
           std::set<NodeAttributes>{NodeAttributes::ReductionAttributes}))
    {
        decline("graph contains unsupported node attributes");
    }

    // Execute-time override shapes can diverge from the dims baked into the MIOpen descriptors
    // the plan builds (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        decline("does not support override shapes");
    }

    // The plan always passes miopenFloat as the reduce compute type.
    if(opGraph.getNode(0).compute_data_type() != DataType::FLOAT)
    {
        decline("only supports nodes with an fp32 compute_data_type");
    }

    const auto& attrs = opGraph.getNodeWrapper(0).attributesAs<ReductionAttributes>();

    // MIOPEN_REDUCE_TENSOR_MUL_NO_ZEROS is commented out in miopen.h (miopenReduceTensorOp_t),
    // and NOT_SET names no operation.
    if(!toMiopenReduceTensorOp(attrs.mode()).has_value())
    {
        decline("unsupported reduction mode: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode(attrs.mode())));
    }

    // MIOpen documents no determinism guarantee for miopenReduceTensor, and its legacy kernels
    // pick the reduction method, and with it the summation order, from the device's wavefront
    // width. MIOPEN_ENGINE_DETERMINISTIC does not register this builder.
    if(attrs.is_deterministic())
    {
        decline("deterministic reduction is not supported");
    }

    const auto xUid = attrs.in_tensor_uid();
    const auto yUid = attrs.out_tensor_uid();

    // In-place is unsupported and unchecked: the kernels take A and C as __restrict__ pointers.
    if(xUid == yUid)
    {
        decline("does not support in-place execution; out_tensor_uid must differ from "
                "in_tensor_uid");
    }

    const auto& tensorMap = opGraph.getTensorMap();
    const auto xIt = tensorMap.find(xUid);
    const auto yIt = tensorMap.find(yUid);
    if(xIt == tensorMap.end() || yIt == tensorMap.end() || xIt->second == nullptr
       || yIt->second == nullptr)
    {
        decline("in/out tensor uids do not resolve in the graph's tensor map");
    }

    const auto& x = *xIt->second;
    const auto& y = *yIt->second;
    checkTensorSupported(x, "input");
    checkTensorSupported(y, "output");

    // A and C are separate kernel parameters, but MIOpen's own reduce tests and driver only run
    // matching types, so a mixed pair is unvalidated.
    if(x.data_type() != y.data_type())
    {
        decline("input and output tensors must have the same data type");
    }

    const auto& xDims = *x.dims();
    const auto& yDims = *y.dims();

    // ValidateReduceDescriptors: A and C must have the same rank.
    if(xDims.size() != yDims.size())
    {
        decline("input and output tensors must have the same rank");
    }

    // A dim of C that is 1 is reduced; any other dim must match A (ValidateReduceDescriptors).
    // This is also hipDNN's convention (ReductionNode::pre_validate_node), so no translation is
    // needed.
    bool hasReducedAxis = false;
    for(flatbuffers::uoffset_t i = 0; i < xDims.size(); ++i)
    {
        if(yDims[i] != 1 && yDims[i] != xDims[i])
        {
            decline("output dim at axis " + std::to_string(i)
                    + " must be 1 or equal to the input dim");
        }
        hasReducedAxis = hasReducedAxis || (yDims[i] == 1 && xDims[i] > 1);
    }

    // With no output dim of 1, ReduceTensor throws "at least one dimension of the input tensor
    // should be reduced". With output dims of 1 only where the input is 1 too, it "reduces"
    // over a single element, which trips assert(toReduceLength > 1) in debug builds.
    if(!hasReducedAxis)
    {
        decline("no axis is reduced; at least one output dim must be 1 where the input dim is "
                "greater than 1");
    }

    // ReduceTensor narrows the output and reduced element counts, and the grid size derived
    // from them, to int. The byte span does not bound them when input strides overlap (a
    // stride-1 axis next to another stride-1 axis reads the same elements repeatedly), so check
    // the input's element count, which bounds all three.
    const auto inputElements = elementCount(xDims);
    if(inputElements > INT32_LIMIT)
    {
        decline("input tensor has more than INT32_MAX elements");
    }

    // With a single output, a reduction over more than four blocks of input runs
    // Reduce_MultiBlock with one block per LEGACY_BLOCK_SIZE inputs, uncapped, and
    // get_padding_need pads each block to LEGACY_BLOCK_SLICE elements. The kernel takes that
    // padded length as int.
    if(elementCount(yDims) == 1)
    {
        const auto blocks = (inputElements + LEGACY_BLOCK_SIZE - 1) / LEGACY_BLOCK_SIZE;
        if(blocks * LEGACY_BLOCK_SLICE > INT32_LIMIT)
        {
            decline("a single-output reduction over this many elements exceeds MIOpen's int32 "
                    "padded reduce length");
        }
    }

    checkOutputDoesNotOverlap(y);
}

std::optional<miopenReduceTensorOp_t> toMiopenReduceTensorOp(ReductionMode mode)
{
    switch(mode)
    {
    case ReductionMode::ADD:
        return MIOPEN_REDUCE_TENSOR_ADD;
    case ReductionMode::MUL:
        return MIOPEN_REDUCE_TENSOR_MUL;
    case ReductionMode::MIN_OP:
        return MIOPEN_REDUCE_TENSOR_MIN;
    case ReductionMode::MAX_OP:
        return MIOPEN_REDUCE_TENSOR_MAX;
    case ReductionMode::AMAX:
        return MIOPEN_REDUCE_TENSOR_AMAX;
    case ReductionMode::AVG:
        return MIOPEN_REDUCE_TENSOR_AVG;
    case ReductionMode::NORM1:
        return MIOPEN_REDUCE_TENSOR_NORM1;
    case ReductionMode::NORM2:
        return MIOPEN_REDUCE_TENSOR_NORM2;
    case ReductionMode::NOT_SET:
    case ReductionMode::MUL_NO_ZEROS:
    default:
        return std::nullopt;
    }
}

int64_t tensorSpanBytes(const TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    // Capping each term keeps the sum of at most six terms from wrapping once the span is
    // already out of range.
    int64_t lastOffset = 0;
    for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
    {
        const auto dim = std::min((*dims)[i], INT32_LIMIT);
        const auto stride = std::min((*strides)[i], INT32_LIMIT);
        lastOffset += std::min((dim - 1) * stride, INT32_LIMIT);
    }

    const int64_t elementBytes = tensor.data_type() == DataType::FLOAT ? 4 : 2;
    return (lastOffset + 1) * elementBytes;
}

bool isSupported(const IGraph& opGraph)
{
    try
    {
        checkSupported(opGraph);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(LOG_PREFIX << e.what());
        return false;
    }
}

} // namespace miopen_plugin::reduction_applicability
