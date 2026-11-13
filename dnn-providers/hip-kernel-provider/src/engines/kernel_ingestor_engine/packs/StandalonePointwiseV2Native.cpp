// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/descriptors/hip/standalone_pointwise/kernels/StandalonePointwiseContractV2.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace contract = hkp_standalone_pointwise_v2;
constexpr auto DISPATCH = "hipkernel.standalone_pointwise.dispatch.v2";

struct Launch
{
    contract::Metadata metadata;
    std::array<int64_t, 4> uids{};
    std::array<int32_t, 3> kinds{}; // 0: device buffer, 1: baked scalar, 2: runtime host scalar.
    int32_t arity = 0;
};

std::optional<Launch> decline(const std::string& reason)
{
    HIPDNN_PLUGIN_LOG_INFO("StandalonePointwise declined: " << reason);
    return std::nullopt;
}

int32_t arityFor(data::PointwiseMode mode)
{
    switch(mode)
    {
    case data::PointwiseMode::ABS:
    case data::PointwiseMode::NEG:
    case data::PointwiseMode::IDENTITY:
    case data::PointwiseMode::RELU_FWD:
    case data::PointwiseMode::SIGMOID_FWD:
    case data::PointwiseMode::TANH_FWD:
    case data::PointwiseMode::GELU_FWD:
    case data::PointwiseMode::GELU_APPROX_TANH_FWD:
    case data::PointwiseMode::SWISH_FWD:
    case data::PointwiseMode::ELU_FWD:
    case data::PointwiseMode::SOFTPLUS_FWD:
    case data::PointwiseMode::GEN_INDEX:
        return 1;
    case data::PointwiseMode::ADD:
    case data::PointwiseMode::SUB:
    case data::PointwiseMode::MUL:
    case data::PointwiseMode::MIN_OP:
    case data::PointwiseMode::MAX_OP:
    case data::PointwiseMode::CMP_GT:
    case data::PointwiseMode::RELU_BWD:
    case data::PointwiseMode::SIGMOID_BWD:
    case data::PointwiseMode::TANH_BWD:
    case data::PointwiseMode::ELU_BWD:
    case data::PointwiseMode::SOFTPLUS_BWD:
    case data::PointwiseMode::SWISH_BWD:
        return 2;
    case data::PointwiseMode::BINARY_SELECT:
        return 3;
    default:
        return 0;
    }
}

int32_t storage(data::DataType type)
{
    switch(type)
    {
    case data::DataType::FLOAT:
        return contract::FLOAT;
    case data::DataType::HALF:
        return contract::HALF;
    case data::DataType::BFLOAT16:
        return contract::BFLOAT16;
    case data::DataType::BOOLEAN:
        return contract::BOOLEAN;
    case data::DataType::INT32:
        return contract::INT32;
    default:
        return -1;
    }
}

float readHostScalar(const void* pointer, int32_t type)
{
    if(pointer == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "null runtime pointwise scalar");
    }
    switch(type)
    {
    case contract::FLOAT:
    {
        float value;
        std::memcpy(&value, pointer, sizeof(value));
        return value;
    }
    case contract::HALF:
    {
        hipdnn_data_sdk::types::half value;
        std::memcpy(&value, pointer, sizeof(value));
        return static_cast<float>(value);
    }
    case contract::BFLOAT16:
    {
        hipdnn_data_sdk::types::bfloat16 value;
        std::memcpy(&value, pointer, sizeof(value));
        return static_cast<float>(value);
    }
    case contract::BOOLEAN:
    {
        bool value;
        std::memcpy(&value, pointer, sizeof(value));
        return value ? 1.0f : 0.0f;
    }
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "invalid pointwise scalar storage");
    }
}

std::optional<float> bakedScalar(const data::TensorAttributes& tensor)
{
    switch(tensor.data_type())
    {
    case data::DataType::FLOAT:
        if(const auto* value = tensor.value_as_Float32Value())
        {
            return value->value();
        }
        break;
    case data::DataType::HALF:
        if(const auto* value = tensor.value_as_Float16Value())
        {
            return static_cast<float>(hipdnn_data_sdk::types::half(value->value()));
        }
        break;
    case data::DataType::BFLOAT16:
        if(const auto* value = tensor.value_as_BFloat16Value())
        {
            return static_cast<float>(hipdnn_data_sdk::types::bfloat16(value->value()));
        }
        break;
    case data::DataType::BOOLEAN:
        if(const auto* value = tensor.value_as_BoolValue())
        {
            return value->value() ? 1.0f : 0.0f;
        }
        break;
    default:
        break;
    }
    return std::nullopt;
}

bool dtypeCombination(const Launch& launch)
{
    const auto& types = launch.metadata.types;
    const auto a = types[0];
    const auto b = types[1];
    const auto out = types[3];
    if(a < contract::FLOAT || a > contract::BFLOAT16)
    {
        return false;
    }
    if(launch.metadata.operation == static_cast<int32_t>(data::PointwiseMode::GEN_INDEX))
    {
        return out == contract::FLOAT || out == contract::INT32;
    }
    if(launch.metadata.operation == static_cast<int32_t>(data::PointwiseMode::CMP_GT))
    {
        return a == b && out == contract::BOOLEAN;
    }
    if(out < contract::FLOAT || out > contract::BFLOAT16)
    {
        return false;
    }
    if(launch.arity == 1)
    {
        return a == out || a == contract::FLOAT;
    }
    if(launch.arity == 3)
    {
        return a == b && out == a && types[2] == contract::BOOLEAN;
    }
    if(b < contract::FLOAT || b > contract::BFLOAT16)
    {
        return false;
    }
    // CPU graph executor FLOAT-compute signatures: equal inputs may retain their
    // storage or upcast; FLOAT with either low-precision input produces FLOAT.
    return (a == b && (out == a || out == contract::FLOAT))
           || (out == contract::FLOAT && (a == contract::FLOAT || b == contract::FLOAT));
}

std::optional<Launch> binding(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }
    if(node.attributes() == nullptr)
    {
        return decline("missing pointwise attributes");
    }
    const auto& attributes = node.attributesAs<data::PointwiseAttributes>();
    Launch launch;
    launch.arity = arityFor(attributes.operation());
    if(launch.arity == 0)
    {
        return decline(std::string("unsupported standalone pointwise mode: ")
                       + data::EnumNamePointwiseMode(attributes.operation()));
    }
    if(node.computeDataType() != data::DataType::FLOAT)
    {
        return decline("compute_data_type must be FLOAT");
    }
    if((launch.arity >= 2) != attributes.in_1_tensor_uid().has_value()
       || (launch.arity == 3) != attributes.in_2_tensor_uid().has_value())
    {
        return decline("operand count does not match mode");
    }
    const bool relu = attributes.operation() == data::PointwiseMode::RELU_FWD
                      || attributes.operation() == data::PointwiseMode::RELU_BWD;
    const bool swish = attributes.operation() == data::PointwiseMode::SWISH_FWD
                       || attributes.operation() == data::PointwiseMode::SWISH_BWD;
    const bool elu = attributes.operation() == data::PointwiseMode::ELU_FWD
                     || attributes.operation() == data::PointwiseMode::ELU_BWD;
    const bool softplus = attributes.operation() == data::PointwiseMode::SOFTPLUS_FWD
                          || attributes.operation() == data::PointwiseMode::SOFTPLUS_BWD;
    const bool genIndex = attributes.operation() == data::PointwiseMode::GEN_INDEX;
    if((!relu
        && (attributes.relu_lower_clip() || attributes.relu_upper_clip()
            || attributes.relu_lower_clip_slope()))
       || (attributes.swish_beta() && !swish) || (attributes.elu_alpha() && !elu)
       || (attributes.softplus_beta() && !softplus) || (!genIndex && attributes.axis_tensor_uid()))
    {
        return decline("irrelevant/unsupported relu, swish, elu_alpha, softplus_beta or GEN_INDEX "
                       "axis parameter");
    }
    auto& meta = launch.metadata;
    meta.operation = static_cast<int32_t>(attributes.operation());
    meta.lower = attributes.relu_lower_clip().value_or(0.0f);
    const bool parameterizedRelu = attributes.relu_lower_clip() || attributes.relu_upper_clip()
                                   || attributes.relu_lower_clip_slope();
    meta.reluParameterized = parameterizedRelu ? 1 : 0;
    meta.upper = attributes.relu_upper_clip().value_or(std::numeric_limits<float>::max());
    meta.slope = attributes.relu_lower_clip_slope().value_or(0.0f);
    meta.beta = softplus ? attributes.softplus_beta().value_or(1.0f)
                         : attributes.swish_beta().value_or(1.0f);
    meta.alpha = attributes.elu_alpha().value_or(1.0f);
    if(!std::isfinite(meta.lower) || std::isnan(meta.upper)
       || meta.upper == -std::numeric_limits<float>::infinity() || !std::isfinite(meta.slope)
       || !std::isfinite(meta.beta) || !std::isfinite(meta.alpha) || (softplus && meta.beta <= 0.0f)
       || meta.lower > meta.upper)
    {
        return decline("invalid clipping/slope/beta parameter");
    }
    launch.uids = {attributes.in_0_tensor_uid(),
                   attributes.in_1_tensor_uid().value_or(0),
                   attributes.in_2_tensor_uid().value_or(0),
                   attributes.out_0_tensor_uid()};
    const auto& tensors = context.graph.getTensorMap();
    std::array<const data::TensorAttributes*, 4> operands{};
    for(size_t operand = 0; operand < operands.size(); ++operand)
    {
        if(operand >= static_cast<size_t>(launch.arity) && operand != 3)
        {
            continue;
        }
        const auto found = tensors.find(launch.uids[operand]);
        if(found == tensors.end() || found->second == nullptr)
        {
            return decline("missing operand tensor");
        }
        const auto& tensor = *found->second;
        operands[operand] = &tensor;
        if(tensor.virtual_() || tensor.ragged_offset_tensor_uid())
        {
            return decline("virtual/ragged operands are unsupported");
        }
        const auto* dims = tensor.dims();
        const auto* strides = tensor.strides();
        if(dims == nullptr || strides == nullptr || dims->size() != strides->size() || dims->empty()
           || dims->size() > 5)
        {
            return decline("operand rank/strides must describe rank 1-5");
        }
        int64_t count = 1;
        int64_t span = 0;
        for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
        {
            const auto dim = dims->Get(axis);
            const auto stride = strides->Get(axis);
            constexpr auto LIMIT = std::numeric_limits<int64_t>::max() / 4;
            if(dim <= 0 || stride <= 0 || count > LIMIT / dim || dim - 1 > (LIMIT - span) / stride)
            {
                return decline("non-positive or overflowing extent/stride");
            }
            count *= dim;
            span += (dim - 1) * stride;
        }
        meta.types[operand] = storage(tensor.data_type());
        if(meta.types[operand] < 0)
        {
            return decline("unsupported storage dtype");
        }
        if(operand == 3)
        {
            if(tensor.is_runtime_pass_by_value() || tensor.value_type() != data::TensorValue::NONE)
            {
                return decline("output cannot be a scalar value operand");
            }
            meta.elements = count;
            std::fill(std::begin(meta.dims), std::end(meta.dims), 1);
            std::vector<std::pair<int64_t, int64_t>> axes;
            for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
            {
                meta.dims[5 - dims->size() + axis] = dims->Get(axis);
                if(dims->Get(axis) > 1)
                {
                    axes.emplace_back(strides->Get(axis), dims->Get(axis));
                }
            }
            std::sort(axes.begin(), axes.end());
            int64_t required = 1;
            for(const auto& [stride, dim] : axes)
            {
                if(stride < required)
                {
                    return decline("overlapping output strides");
                }
                required += (dim - 1) * stride;
            }
        }
        else if(tensor.is_runtime_pass_by_value() || tensor.value_type() != data::TensorValue::NONE)
        {
            if(count != 1)
            {
                return decline("value operands must be single-element tensors");
            }
            if(tensor.is_runtime_pass_by_value() && tensor.value_type() != data::TensorValue::NONE)
            {
                return decline("operand cannot be both runtime PBV and baked");
            }
            launch.kinds[operand] = tensor.is_runtime_pass_by_value() ? 2 : 1;
            meta.scalarMask |= 1 << operand;
            if(launch.kinds[operand] == 1)
            {
                const auto value = bakedScalar(tensor);
                if(!value)
                {
                    return decline("baked scalar union does not match dtype");
                }
                meta.scalars[operand] = *value;
            }
        }
    }
    if(!dtypeCombination(launch))
    {
        return decline("unsupported input/output dtype combination");
    }
    if(genIndex)
    {
        const auto axis = attributes.axis_tensor_uid();
        const auto rank = operands[3]->dims()->size();
        if(!axis || *axis < 0 || *axis >= rank)
        {
            return decline("GEN_INDEX requires an axis in [0, output rank)");
        }
        meta.axis = static_cast<int32_t>(5 - static_cast<int64_t>(rank) + *axis);
        if(meta.dims[static_cast<size_t>(meta.axis)] > std::numeric_limits<int32_t>::max())
        {
            return decline("GEN_INDEX axis extent exceeds INT32 range");
        }
    }
    for(size_t operand = 0; operand < operands.size(); ++operand)
    {
        if(operands[operand] == nullptr)
        {
            continue;
        }
        const auto* dims = operands[operand]->dims();
        for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
        {
            const auto aligned = 5 - dims->size() + axis;
            const auto dim = dims->Get(axis);
            if(dim != 1 && dim != meta.dims[aligned])
            {
                return decline("operand does not right-align broadcast to output");
            }
            meta.strides[operand][aligned] = dim == 1 ? 0 : operands[operand]->strides()->Get(axis);
        }
        if(operand != 3 && launch.uids[operand] == launch.uids[3])
        {
            return decline("output uid aliases an input");
        }
    }
    return launch;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto launch = binding(context);
    if(!launch)
    {
        return std::nullopt;
    }
    return BoundTokens{{"standalone.elements", launch->metadata.elements}};
}

bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return kernel.getIntMetadata("contract_version") == 2;
}

double score(const MatchContext& /*context*/,
             const BoundTokens& /*bound*/,
             const KernelDefinition& /*kernel*/)
{
    return 0;
}

const std::vector<KernelArgument>& signature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const std::vector<KernelArgument> s_result{
        s_pointer,
        s_pointer,
        s_pointer,
        s_pointer,
        {"by_value", static_cast<uint32_t>(sizeof(contract::Metadata)), 0, ""}};
    return s_result;
}

MetadataValues savedValues(const Launch& launch)
{
    const auto& meta = launch.metadata;
    MetadataValues values{{"elements", meta.elements},
                          {"arity", static_cast<int64_t>(launch.arity)},
                          {"operation", static_cast<int64_t>(meta.operation)},
                          {"scalar_mask", static_cast<int64_t>(meta.scalarMask)},
                          {"relu_parameterized", static_cast<int64_t>(meta.reluParameterized)},
                          {"lower", static_cast<double>(meta.lower)},
                          {"upper", static_cast<double>(meta.upper)},
                          {"slope", static_cast<double>(meta.slope)},
                          {"beta", static_cast<double>(meta.beta)}};
    values.emplace("alpha", static_cast<double>(meta.alpha));
    values.emplace("axis", static_cast<int64_t>(meta.axis));
    for(size_t axis = 0; axis < 5; ++axis)
    {
        values.emplace("dim." + std::to_string(axis), meta.dims[axis]);
    }
    for(size_t operand = 0; operand < 4; ++operand)
    {
        const auto root = std::to_string(operand);
        values.emplace("uid." + root, launch.uids[operand]);
        values.emplace("type." + root, static_cast<int64_t>(meta.types[operand]));
        if(operand < 3)
        {
            values.emplace("kind." + root, static_cast<int64_t>(launch.kinds[operand]));
            values.emplace("scalar." + root, static_cast<double>(meta.scalars[operand]));
        }
        for(size_t axis = 0; axis < 5; ++axis)
        {
            values.emplace("stride." + root + "." + std::to_string(axis),
                           meta.strides[operand][axis]);
        }
    }
    return values;
}

Launch readSaved(const SavedLaunchInputs& inputs)
{
    requireLaunchContract(inputs, DISPATCH);
    Launch launch;
    auto& meta = launch.metadata;
    launch.arity = static_cast<int32_t>(requireLaunchInt(inputs, "arity", 1, 3));
    meta.operation = static_cast<int32_t>(requireLaunchInt(inputs, "operation", 1, 47));
    meta.scalarMask = static_cast<int32_t>(requireLaunchInt(inputs, "scalar_mask", 0, 7));
    meta.reluParameterized
        = static_cast<int32_t>(requireLaunchInt(inputs, "relu_parameterized", 0, 1));
    meta.elements
        = requireLaunchInt(inputs, "elements", 1, std::numeric_limits<int64_t>::max() / 4);
    meta.lower = static_cast<float>(requireLaunchDouble(inputs, "lower"));
    meta.upper = static_cast<float>(requireLaunchDouble(inputs, "upper"));
    meta.slope = static_cast<float>(requireLaunchDouble(inputs, "slope"));
    meta.beta = static_cast<float>(requireLaunchDouble(inputs, "beta"));
    meta.alpha = static_cast<float>(requireLaunchDouble(inputs, "alpha"));
    meta.axis = static_cast<int32_t>(requireLaunchInt(inputs, "axis", 0, 4));
    int64_t count = 1;
    bool valid = arityFor(static_cast<data::PointwiseMode>(meta.operation)) == launch.arity
                 && std::isfinite(meta.lower) && !std::isnan(meta.upper) && meta.lower <= meta.upper
                 && std::isfinite(meta.slope) && std::isfinite(meta.beta)
                 && std::isfinite(meta.alpha)
                 && ((meta.operation != 39 && meta.operation != 40) || meta.beta > 0);
    for(size_t axis = 0; axis < 5; ++axis)
    {
        meta.dims[axis] = requireLaunchInt(
            inputs, "dim." + std::to_string(axis), 1, std::numeric_limits<int64_t>::max() / 4);
        if(count > std::numeric_limits<int64_t>::max() / 4 / meta.dims[axis])
        {
            valid = false;
        }
        else
        {
            count *= meta.dims[axis];
        }
    }
    valid = valid && count == meta.elements;
    if(meta.operation == static_cast<int32_t>(data::PointwiseMode::GEN_INDEX))
    {
        valid = valid
                && meta.dims[static_cast<size_t>(meta.axis)] <= std::numeric_limits<int32_t>::max();
    }
    for(size_t operand = 0; operand < 4; ++operand)
    {
        const auto root = std::to_string(operand);
        launch.uids[operand] = requireLaunchInt(inputs, "uid." + root);
        meta.types[operand] = static_cast<int32_t>(requireLaunchInt(inputs, "type." + root, 0, 4));
        if(operand < 3)
        {
            launch.kinds[operand]
                = static_cast<int32_t>(requireLaunchInt(inputs, "kind." + root, 0, 1));
            meta.scalars[operand]
                = static_cast<float>(requireLaunchDouble(inputs, "scalar." + root));
            valid = valid
                    && ((meta.scalarMask & (1 << operand)) != 0) == (launch.kinds[operand] == 1);
            if(operand >= static_cast<size_t>(launch.arity))
            {
                valid = valid && launch.kinds[operand] == 0 && launch.uids[operand] == 0
                        && meta.types[operand] == contract::FLOAT && meta.scalars[operand] == 0.0f;
            }
        }
        int64_t span = 0;
        std::vector<std::pair<int64_t, int64_t>> axes;
        for(size_t axis = 0; axis < 5; ++axis)
        {
            const auto stride = requireLaunchInt(inputs,
                                                 "stride." + root + "." + std::to_string(axis),
                                                 0,
                                                 std::numeric_limits<int64_t>::max() / 4);
            meta.strides[operand][axis] = stride;
            if((operand < 3 && operand >= static_cast<size_t>(launch.arity))
               || meta.dims[axis] == 1)
            {
                valid = valid && stride == 0;
            }
            constexpr auto LIMIT = std::numeric_limits<int64_t>::max() / 4;
            if(stride != 0 && meta.dims[axis] - 1 > (LIMIT - span) / stride)
            {
                valid = false;
            }
            else
            {
                span += (meta.dims[axis] - 1) * stride;
            }
            if(operand == 3 && meta.dims[axis] > 1)
            {
                axes.emplace_back(stride, meta.dims[axis]);
            }
            if(operand < 3 && launch.kinds[operand] == 1 && stride != 0)
            {
                valid = false;
            }
        }
        std::sort(axes.begin(), axes.end());
        int64_t required = 1;
        for(const auto& [stride, dim] : axes)
        {
            if(stride < required
               || (dim - 1 > (std::numeric_limits<int64_t>::max() / 4 - required)
                                 / std::max<int64_t>(stride, 1)))
            {
                valid = false;
            }
            else
            {
                required += (dim - 1) * stride;
            }
        }
    }
    for(size_t operand = 0; operand < static_cast<size_t>(launch.arity); ++operand)
    {
        valid = valid && launch.uids[operand] != launch.uids[3];
    }
    if(!valid || !dtypeCombination(launch) || inputs.values.size() != savedValues(launch).size())
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "invalid standalone pointwise launch metadata");
    }
    return launch;
}

class PreparedStandalone : public IngestorPreparedDispatch
{
public:
    PreparedStandalone(IngestorKernelCode code, Launch launch)
        : _code(std::move(code))
        , _launch(launch)
    {
        _code.setBlockSize(256);
        _code.setGridSize(static_cast<unsigned int>(
            std::min<int64_t>((launch.metadata.elements + 255) / 256, 65535)));
    }
    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }
    const Launch& launchInputs() const
    {
        return _launch;
    }

private:
    IngestorKernelCode _code;
    Launch _launch;
};

compilation::KpackModuleCache& moduleCache()
{
    static compilation::KpackModuleCache s_cache;
    return s_cache;
}

class StandaloneDispatch : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }
    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto launch = binding(context);
        if(!launch || tryGetBoundInt(bound, "standalone.elements") != launch->metadata.elements
           || !kernelMatches(context, bound, kernel))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "invalid standalone pointwise match");
        }
        static const HipMlopsKernelCompiler s_compiler;
        static const compilation::KpackKernelLoader s_loader(moduleCache());
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code
            = buildIngestorKernelCode(s_compiler, s_loader, context, kernel, options, signature());
        return std::make_unique<PreparedStandalone>(std::move(code), *launch);
    }
    bool supportsSaving() const override
    {
        return true;
    }
    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& launch = dynamic_cast<const PreparedStandalone&>(prepared).launchInputs();
        if(std::find(launch.kinds.begin(), launch.kinds.end(), 2) != launch.kinds.end())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "StandalonePointwise plan save declined: runtime pass-by-value operand");
            return std::nullopt;
        }
        return SavedLaunchInputs{DISPATCH, savedValues(launch)};
    }
    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        auto restored = restoreIngestorLaunch<Launch>(
            inputs, std::move(code), signature(), &readSaved, deviceOrdinal);
        return std::make_unique<PreparedStandalone>(std::move(restored.code), restored.launch);
    }
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* buffers,
                uint32_t count,
                void* /*workspace*/) const override
    {
        const auto& plan = dynamic_cast<const PreparedStandalone&>(prepared);
        auto launch = plan.launchInputs();
        std::array<void*, 4> pointers{};
        for(size_t operand = 0; operand < pointers.size(); ++operand)
        {
            if(operand >= static_cast<size_t>(launch.arity) && operand != 3)
            {
                continue;
            }
            if(operand != 3 && launch.kinds[operand] == 1)
            {
                continue;
            }
            const auto buffer
                = hipdnn_plugin_sdk::findDeviceBuffer(launch.uids[operand], buffers, count);
            if(operand != 3 && launch.kinds[operand] == 2)
            {
                launch.metadata.scalars[operand]
                    = readHostScalar(buffer.ptr, launch.metadata.types[operand]);
            }
            else
            {
                if(buffer.ptr == nullptr)
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                                   "null pointwise device operand");
                }
                pointers[operand] = buffer.ptr;
            }
        }
        plan.kernelCode()
            .kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    pointers[0],
                    pointers[1],
                    pointers[2],
                    pointers[3],
                    launch.metadata);
    }
};

const StandaloneDispatch& dispatch()
{
    static const StandaloneDispatch s_handler;
    return s_handler;
}
} // namespace

void registerStandalonePointwiseV2Symbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add("hipkernel.standalone_pointwise.graph_match.v2", &graphMatches);
    scope.add("hipkernel.standalone_pointwise.kernel_match.v2", &kernelMatches);
    scope.add("hipkernel.standalone_pointwise.score.v2", &score);
    scope.add(DISPATCH, &dispatch());
}

void resetStandalonePointwiseV2ModuleCache()
{
    moduleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
#endif
