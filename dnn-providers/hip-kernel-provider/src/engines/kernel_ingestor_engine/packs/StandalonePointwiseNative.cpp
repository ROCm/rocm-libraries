// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

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
#include "engines/kernel_ingestor_engine/descriptors/hip/standalone_pointwise/kernels/StandalonePointwiseContract.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace contract = hkp_standalone_pointwise;
constexpr auto DISPATCH = "hipkernel.standalone_pointwise.dispatch.v1";
constexpr auto LIMIT = std::numeric_limits<int64_t>::max() / 4;

struct Launch
{
    contract::Metadata metadata;
    std::array<int64_t, 3> uids{};
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
        return 2;
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
    default:
        return -1;
    }
}

bool dtypeCombination(const Launch& launch)
{
    const auto& types = launch.metadata.types;
    const auto a = types[0];
    const auto b = types[1];
    const auto out = types[2];
    if(a < contract::FLOAT || a > contract::BFLOAT16)
    {
        return false;
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
    if(b < contract::FLOAT || b > contract::BFLOAT16)
    {
        return false;
    }
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
        return decline(std::string("unsupported unary/binary mode: ")
                       + data::EnumNamePointwiseMode(attributes.operation()));
    }
    if(node.computeDataType() != data::DataType::FLOAT)
    {
        return decline("compute_data_type must be FLOAT");
    }
    if((launch.arity == 2) != attributes.in_1_tensor_uid().has_value()
       || attributes.in_2_tensor_uid())
    {
        return decline("operand count does not match mode");
    }
    if(attributes.relu_lower_clip() || attributes.relu_upper_clip()
       || attributes.relu_lower_clip_slope() || attributes.swish_beta() || attributes.elu_alpha()
       || attributes.softplus_beta() || attributes.axis_tensor_uid())
    {
        return decline("activation parameters and GEN_INDEX axes are not supported");
    }
    auto& meta = launch.metadata;
    meta.operation = static_cast<int32_t>(attributes.operation());
    launch.uids = {attributes.in_0_tensor_uid(),
                   attributes.in_1_tensor_uid().value_or(0),
                   attributes.out_0_tensor_uid()};
    const auto& tensors = context.graph.getTensorMap();
    std::array<const data::TensorAttributes*, 3> operands{};
    for(size_t operand = 0; operand < operands.size(); ++operand)
    {
        if(operand == 1 && launch.arity == 1)
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
        if(tensor.is_runtime_pass_by_value() || tensor.value_type() != data::TensorValue::NONE)
        {
            return decline("baked and runtime pass-by-value operands are not supported");
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
        if(operand == 2)
        {
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
    }
    if(!dtypeCombination(launch))
    {
        return decline("unsupported input/output dtype combination");
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
        if(operand != 2 && launch.uids[operand] == launch.uids[2])
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
    return kernel.getIntMetadata("contract_version") == 1;
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
        {"by_value", static_cast<uint32_t>(sizeof(contract::Metadata)), 0, ""}};
    return s_result;
}

MetadataValues savedValues(const Launch& launch)
{
    const auto& meta = launch.metadata;
    MetadataValues values{{"elements", meta.elements},
                          {"arity", static_cast<int64_t>(launch.arity)},
                          {"operation", static_cast<int64_t>(meta.operation)}};
    for(size_t axis = 0; axis < 5; ++axis)
    {
        values.emplace("dim." + std::to_string(axis), meta.dims[axis]);
    }
    for(size_t operand = 0; operand < 3; ++operand)
    {
        const auto root = std::to_string(operand);
        values.emplace("uid." + root, launch.uids[operand]);
        values.emplace("type." + root, static_cast<int64_t>(meta.types[operand]));
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
    launch.arity = static_cast<int32_t>(requireLaunchInt(inputs, "arity", 1, 2));
    meta.operation = static_cast<int32_t>(
        requireLaunchInt(inputs, "operation", 1, static_cast<int64_t>(data::PointwiseMode::MAX)));
    meta.elements = requireLaunchInt(inputs, "elements", 1, LIMIT);
    int64_t count = 1;
    bool valid = arityFor(static_cast<data::PointwiseMode>(meta.operation)) == launch.arity;
    for(size_t axis = 0; axis < 5; ++axis)
    {
        meta.dims[axis] = requireLaunchInt(inputs, "dim." + std::to_string(axis), 1, LIMIT);
        if(count > LIMIT / meta.dims[axis])
        {
            valid = false;
        }
        else
        {
            count *= meta.dims[axis];
        }
    }
    valid = valid && count == meta.elements;
    for(size_t operand = 0; operand < 3; ++operand)
    {
        const auto root = std::to_string(operand);
        launch.uids[operand] = requireLaunchInt(inputs, "uid." + root);
        meta.types[operand] = static_cast<int32_t>(
            requireLaunchInt(inputs, "type." + root, contract::FLOAT, contract::BOOLEAN));
        const bool unused = operand == 1 && launch.arity == 1;
        if(unused)
        {
            valid = valid && launch.uids[operand] == 0 && meta.types[operand] == contract::FLOAT;
        }
        int64_t span = 0;
        std::vector<std::pair<int64_t, int64_t>> axes;
        for(size_t axis = 0; axis < 5; ++axis)
        {
            const auto stride
                = requireLaunchInt(inputs, "stride." + root + "." + std::to_string(axis), 0, LIMIT);
            meta.strides[operand][axis] = stride;
            if(unused || meta.dims[axis] == 1)
            {
                valid = valid && stride == 0;
            }
            if(stride != 0 && meta.dims[axis] - 1 > (LIMIT - span) / stride)
            {
                valid = false;
            }
            else
            {
                span += (meta.dims[axis] - 1) * stride;
            }
            if(operand == 2 && meta.dims[axis] > 1)
            {
                axes.emplace_back(stride, meta.dims[axis]);
            }
        }
        std::sort(axes.begin(), axes.end());
        int64_t required = 1;
        for(const auto& [stride, dim] : axes)
        {
            if(stride < required || dim - 1 > (LIMIT - required) / std::max<int64_t>(stride, 1))
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
        valid = valid && launch.uids[operand] != launch.uids[2];
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
        const auto& launch = plan.launchInputs();
        std::array<void*, 3> pointers{};
        for(size_t operand = 0; operand < pointers.size(); ++operand)
        {
            if(operand == 1 && launch.arity == 1)
            {
                continue;
            }
            const auto buffer
                = hipdnn_plugin_sdk::findDeviceBuffer(launch.uids[operand], buffers, count);
            if(buffer.ptr == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                               "null pointwise device operand");
            }
            pointers[operand] = buffer.ptr;
        }
        plan.kernelCode()
            .kernelForStream(handle.getStream())
            .launch(handle.getStream(), pointers[0], pointers[1], pointers[2], launch.metadata);
    }
};

const StandaloneDispatch& dispatch()
{
    static const StandaloneDispatch s_handler;
    return s_handler;
}
} // namespace

void registerStandalonePointwiseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add("hipkernel.standalone_pointwise.graph_match", &graphMatches);
    scope.add("hipkernel.standalone_pointwise.kernel_match", &kernelMatches);
    scope.add("hipkernel.standalone_pointwise.score", &score);
    scope.add(DISPATCH, &dispatch());
    registerStandalonePointwiseV2Symbols(scope);
}

void resetStandalonePointwiseModuleCache()
{
    moduleCache().clear();
    resetStandalonePointwiseV2ModuleCache();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
#endif
