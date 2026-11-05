// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file Gfx1250ConvFwdAotNative.cpp
 * @brief hipkernel:Gfx1250ConvFwdAot: rocKE's gfx1250 forward-convolution AOT kernels.
 *
 * The catalog is rocKE's gfx1250 forward instance list
 * (rocke/library/dispatch/grouped_convolution_instances/gfx1250_fwd.json), one kernel per
 * instance (tools/make_gfx1250_conv_fwd_aot_config.py). Every kernel is shape-generic --
 * the problem is its arguments -- so which kernels serve a graph, and how each launches,
 * is decided here per graph, by the port of rocKE's dispatcher in
 * Gfx1250ConvFwdGeometry.hpp over each kernel's KMD fields.
 *
 * The graph: on a gfx1250 device, one 2-D forward cross-correlation, x/w/y fp16 or bf16
 * (all three alike), packed channels-last (x NHWC, w KYXC, y NHWK), symmetric padding,
 * groups inferred from the filter's channels. workspaceBytes() and prepare() share one plan function, so a
 * kernel that refuses a graph refuses it in both.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace geometry = gfx1250_conv_fwd;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx1250_conv_fwd_aot.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx1250_conv_fwd_aot.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx1250_conv_fwd_aot.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx1250_conv_fwd_aot.dispatch";

// Operand uids, bound by the graph match for dispatch.
constexpr std::string_view X_TOKEN = "gfx1250_conv_fwd_aot.x.uid";
constexpr std::string_view W_TOKEN = "gfx1250_conv_fwd_aot.w.uid";
constexpr std::string_view Y_TOKEN = "gfx1250_conv_fwd_aot.y.uid";
/// The whole matched problem, so a kernel matcher reads it from what graph_match bound
/// instead of walking the graph again for every kernel: problemTokenValues() order.
/// Private (`_`): the graph features already describe the problem, so this is no feature.
constexpr std::string_view PROBLEM_TOKEN = "_gfx1250_conv_fwd_aot.problem";
constexpr size_t PROBLEM_TOKEN_SIZE = 18;

constexpr size_t SPATIAL_RANK = 2;
constexpr uint32_t TENSOR_RANK = 4;
/// Alignment of each workspace buffer within the hipDNN workspace.
constexpr int64_t WORKSPACE_ALIGNMENT = 256;

// ---------------------------------------------------------------------------
// Graph matching
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

std::optional<geometry::DataType> dataTypeOf(const data_objects::TensorAttributes& tensor)
{
    switch(tensor.data_type())
    {
    case data_objects::DataType::HALF:
        return geometry::DataType::FP16;
    case data_objects::DataType::BFLOAT16:
        return geometry::DataType::BF16;
    default:
        return std::nullopt;
    }
}

/// The tensor's dims when it is a rank-4 device tensor of @p order layout: @p order lists
/// the logical axes (dims index) from slowest to fastest, and the strides must be packed
/// in that order.
std::optional<std::array<int64_t, 4>> packedDims(const data_objects::TensorAttributes& tensor,
                                                 const std::array<flatbuffers::uoffset_t, 4>& order)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != TENSOR_RANK
       || strides->size() != TENSOR_RANK || tensor.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return std::nullopt;
    }
    std::array<int64_t, 4> out{};
    for(flatbuffers::uoffset_t i = 0; i < TENSOR_RANK; ++i)
    {
        out[i] = dims->Get(i);
        if(out[i] < 1)
        {
            return std::nullopt;
        }
    }
    int64_t expected = 1;
    for(size_t i = TENSOR_RANK; i-- > 0;)
    {
        const auto axis = order[i];
        // An extent-one axis is indexed at zero only, so its stride is free.
        if(out[axis] != 1 && strides->Get(axis) != expected)
        {
            return std::nullopt;
        }
        expected *= out[axis];
    }
    return out;
}

/// Every geometry value the kernels take is an i32 argument.
using geometry::I32_MAX;

bool readPair(const flatbuffers::Vector<int64_t>* values, int64_t minimum, int64_t& a, int64_t& b)
{
    if(values == nullptr || values->size() != SPATIAL_RANK)
    {
        return false;
    }
    a = values->Get(0);
    b = values->Get(1);
    return a >= minimum && b >= minimum && a <= I32_MAX && b <= I32_MAX;
}

/// The bytes of an fp16/bf16 tensor of @p dims elements, when the element count does not
/// overflow and every kernel's i32 byte-size and offset arguments can hold its bytes.
bool fitsI32Bytes(std::initializer_list<int64_t> dims)
{
    const auto elements = pack_features::checkedProduct(dims);
    return elements.has_value() && *elements <= I32_MAX / 2;
}

bool dimsFitI32(const std::array<int64_t, 4>& dims)
{
    return std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d <= I32_MAX; });
}

/// The graph's operands and the problem they make.
struct MatchedProblem
{
    std::array<int64_t, 3> uids; ///< x, w, y
    geometry::Problem problem;
};

/// The problem of a graph this engine serves: on a gfx1250 device, one 2-D forward
/// cross-correlation over distinct packed channels-last fp16/bf16 tensors of one dtype,
/// with symmetric padding, whose every tensor fits the kernels' i32 arguments.
std::optional<MatchedProblem> matchProblem(const MatchContext& context)
{
    if(!context.graph.isValid() || context.graph.getGraph().is_override_shape_enabled()
       || !hipdnn_plugin_sdk::archMatches(context.deviceProperties.gcnArchName,
                                          "gfx1250",
                                          hipdnn_plugin_sdk::ArchMatchMode::PREFIX)
       || context.deviceProperties.warpSize != geometry::GFX1250_WAVE_SIZE
       || context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || node.attributes() == nullptr || node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    const auto* conv = &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
    if(conv->conv_mode() != data_objects::ConvMode::CROSS_CORRELATION)
    {
        return std::nullopt;
    }
    geometry::Problem p;
    int64_t postH = 0;
    int64_t postW = 0;
    if(!readPair(conv->stride(), 1, p.strideH, p.strideW)
       || !readPair(conv->dilation(), 1, p.dilationH, p.dilationW)
       || !readPair(conv->pre_padding(), 0, p.padH, p.padW)
       || !readPair(conv->post_padding(), 0, postH, postW) || postH != p.padH || postW != p.padW)
    {
        return std::nullopt;
    }
    const std::array<int64_t, 3> uids
        = {conv->x_tensor_uid(), conv->w_tensor_uid(), conv->y_tensor_uid()};
    if(uids[0] == uids[1] || uids[0] == uids[2] || uids[1] == uids[2])
    {
        return std::nullopt;
    }
    const auto* x = findTensor(context, uids[0]);
    const auto* w = findTensor(context, uids[1]);
    const auto* y = findTensor(context, uids[2]);
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        return std::nullopt;
    }
    const auto dtype = dataTypeOf(*x);
    if(!dtype.has_value() || dataTypeOf(*w) != dtype || dataTypeOf(*y) != dtype)
    {
        return std::nullopt;
    }
    // Logical dims are (N, C, H, W) / (K, C/G, Y, X); memory is channels-last.
    const std::array<flatbuffers::uoffset_t, 4> channelsLast = {0, 2, 3, 1};
    const auto xDims = packedDims(*x, channelsLast);
    const auto wDims = packedDims(*w, channelsLast);
    const auto yDims = packedDims(*y, channelsLast);
    // Bounding every extent to i32 also keeps the output-extent arithmetic within int64.
    if(!xDims || !wDims || !yDims || !dimsFitI32(*xDims) || !dimsFitI32(*wDims)
       || !dimsFitI32(*yDims))
    {
        return std::nullopt;
    }
    p.dtype = *dtype;
    p.n = (*xDims)[0];
    p.c = (*xDims)[1];
    p.h = (*xDims)[2];
    p.w = (*xDims)[3];
    p.k = (*wDims)[0];
    p.y = (*wDims)[2];
    p.x = (*wDims)[3];
    // hipDNN carries no groups attribute: a filter holding C / groups channels encodes it.
    const auto cpg = (*wDims)[1];
    if(p.c % cpg != 0)
    {
        return std::nullopt;
    }
    p.groups = p.c / cpg;
    if(p.k % p.groups != 0 || p.ho() < 1 || p.wo() < 1)
    {
        return std::nullopt;
    }
    if((*yDims)[0] != p.n || (*yDims)[1] != p.k || (*yDims)[2] != p.ho() || (*yDims)[3] != p.wo())
    {
        return std::nullopt;
    }
    if(!fitsI32Bytes({p.n, p.h, p.w, p.c}) || !fitsI32Bytes({p.k, p.y, p.x, cpg})
       || !fitsI32Bytes({p.n, p.ho(), p.wo(), p.k}))
    {
        return std::nullopt;
    }
    return MatchedProblem{uids, p};
}

/// PROBLEM_TOKEN's value: the operand uids, the problem, then the dtype.
std::vector<int64_t> problemTokenValues(const MatchedProblem& matched)
{
    const auto& p = matched.problem;
    return {matched.uids[0],
            matched.uids[1],
            matched.uids[2],
            p.n,
            p.c,
            p.k,
            p.h,
            p.w,
            p.y,
            p.x,
            p.groups,
            p.strideH,
            p.strideW,
            p.padH,
            p.padW,
            p.dilationH,
            p.dilationW,
            static_cast<int64_t>(p.dtype)};
}

/// The problem graphMatches resolved for @p context, read back from @p bound: the SDK
/// passes every kernel matcher what graph_match bound for that same context. Falls back
/// to matching the graph when the token is absent, as for a caller with no graph match.
std::optional<MatchedProblem> matchedFromBound(const MatchContext& context,
                                               const BoundTokens& bound)
{
    static const std::string s_problemToken(PROBLEM_TOKEN);
    const auto it = bound.find(s_problemToken);
    const auto* values
        = it == bound.end() ? nullptr : std::get_if<std::vector<int64_t>>(&it->second);
    if(values == nullptr || values->size() != PROBLEM_TOKEN_SIZE)
    {
        return matchProblem(context);
    }
    const auto& v = *values;
    geometry::Problem p;
    p.n = v[3];
    p.c = v[4];
    p.k = v[5];
    p.h = v[6];
    p.w = v[7];
    p.y = v[8];
    p.x = v[9];
    p.groups = v[10];
    p.strideH = v[11];
    p.strideW = v[12];
    p.padH = v[13];
    p.padW = v[14];
    p.dilationH = v[15];
    p.dilationW = v[16];
    p.dtype = static_cast<geometry::DataType>(v[17]);
    return MatchedProblem{{v[0], v[1], v[2]}, p};
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto matched = matchProblem(context);
    if(!matched)
    {
        return std::nullopt;
    }
    return BoundTokens{{std::string(X_TOKEN), matched->uids[0]},
                       {std::string(W_TOKEN), matched->uids[1]},
                       {std::string(Y_TOKEN), matched->uids[2]},
                       {std::string(PROBLEM_TOKEN), problemTokenValues(*matched)}};
}

// ---------------------------------------------------------------------------
// The catalog kernels, prepared once
// ---------------------------------------------------------------------------

/// A kernel as the matchers read it: its dtype, and its instance when the metadata is a
/// complete instance this port runs.
struct Prepared
{
    std::string dtype;
    std::optional<geometry::CatalogKernel> instance;
};

/// The instance @p kernel's KMD fields describe, or nullopt when it is no kpack kernel or
/// the fields are not an instance this port runs.
Prepared decode(const KernelDefinition& kernel)
{
    geometry::InstanceGeometry::Ints ints;
    geometry::InstanceGeometry::Strings strings;
    for(const auto& [name, value] : kernel.metadata)
    {
        if(const auto* i = std::get_if<int64_t>(&value))
        {
            ints.emplace(name, *i);
        }
        else if(const auto* text = std::get_if<std::string>(&value))
        {
            strings.emplace(name, *text);
        }
    }
    const auto dtype = strings.find("dtype");
    Prepared out{dtype == strings.end() ? std::string() : dtype->second, std::nullopt};
    if(kernel.source.kind != KernelSourceKind::KPACK)
    {
        return out;
    }
    try
    {
        out.instance = geometry::CatalogKernel::parse(
            geometry::InstanceGeometry(std::move(ints), std::move(strings)));
    }
    catch(const std::exception&)
    {
        out.instance.reset();
    }
    return out;
}

/// The KernelPrepareFn registered under the kernel matcher's symbol: the SDK runs it once
/// per kernel when the engine loads, so matching a graph reads Prepared instead of parsing
/// the kernel's fields every time. Must not throw.
std::shared_ptr<const void> prepareKernel(const KernelDefinition& kernel)
{
    try
    {
        return std::make_shared<const Prepared>(decode(kernel));
    }
    catch(...)
    {
        return nullptr;
    }
}

/// @p kernel's Prepared: the one prepareKernel stored, or one decoded into @p scratch for a
/// definition that carries none (one built outside the engine's shared state).
const Prepared& preparedOf(const KernelDefinition& kernel, Prepared& scratch)
{
    if(kernel.preparedBy == &prepareKernel && kernel.prepared != nullptr)
    {
        return *static_cast<const Prepared*>(kernel.prepared.get());
    }
    scratch = decode(kernel);
    return scratch;
}

/// The instance of @p kernel when it serves @p matched, else nullptr (a kernel of another
/// dtype, one this port does not run, or one whose admission refuses the problem).
const geometry::CatalogKernel* servingInstance(const MatchedProblem& matched,
                                               const KernelDefinition& kernel,
                                               Prepared& scratch)
{
    const auto& prepared = preparedOf(kernel, scratch);
    if(prepared.dtype != geometry::toString(matched.problem.dtype) || !prepared.instance
       || prepared.instance->refuse(matched.problem).has_value())
    {
        return nullptr;
    }
    return &*prepared.instance;
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    Prepared scratch;
    return matched.has_value() && servingInstance(*matched, kernel, scratch) != nullptr;
}

/// The fallback order without a trained model: the catalog's priority, which the config
/// tool sets highest for the instance list's first entry (the SDK's declared order).
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Plans, workspace and launch
// ---------------------------------------------------------------------------

/// The plan of @p kernel on the matched graph, refusing a kernel the kernel matcher refuses, and
/// where each of its workspaces sits within the hipDNN workspace.
struct MatchedPlan
{
    MatchedProblem problem;
    geometry::Plan plan;
    std::map<std::string, int64_t> workspaceOffsets;
    int64_t workspaceBytes = 0;
};

MatchedPlan matchedPlan(const MatchContext& context,
                        const BoundTokens& bound,
                        const KernelDefinition& kernel)
{
    const auto matched = matchedFromBound(context, bound);
    Prepared scratch;
    const auto* instance
        = matched.has_value() ? servingInstance(*matched, kernel, scratch) : nullptr;
    if(instance == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            describeDescriptor("kernel", kernel.name, kernel.kernelId)
                + " cannot run this convolution");
    }
    MatchedPlan out;
    out.problem = *matched;
    out.plan = instance->plan(matched->problem);
    for(const auto& [name, bytes] : out.plan.workspaces)
    {
        out.workspaceOffsets[name] = out.workspaceBytes;
        out.workspaceBytes += geometry::alignUp(bytes, WORKSPACE_ALIGNMENT);
    }
    return out;
}

std::vector<KernelArgument> signatureOf(const geometry::Step& step)
{
    std::vector<KernelArgument> out;
    uint32_t offset = 0;
    for(const auto& arg : step.args)
    {
        const bool pointer = arg.kind == geometry::Arg::Kind::POINTER;
        const uint32_t size = pointer ? 8 : 4;
        offset = (offset + size - 1) / size * size;
        out.push_back(
            KernelArgument{pointer ? "global_buffer" : "by_value", size, offset, arg.name});
        offset += size;
    }
    return out;
}

unsigned int launchDim(int64_t value, const char* what)
{
    if(value < 1 || value > I32_MAX)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       std::string("gfx1250 conv fwd: ") + what
                                                           + " out of range");
    }
    return static_cast<unsigned int>(value);
}

/// One launch ready to run: its code, sized, and its arguments.
struct PreparedStep
{
    std::shared_ptr<const IngestorKernelCode> code;
    std::vector<geometry::Arg> args;
};

class PreparedGfx1250ConvFwd : public PreparedDispatch
{
public:
    PreparedGfx1250ConvFwd(std::vector<PreparedStep> steps,
                           std::map<std::string, int64_t> workspaceOffsets,
                           geometry::Problem problem,
                           std::array<int64_t, 3> uids)
        : _steps(std::move(steps))
        , _workspaceOffsets(std::move(workspaceOffsets))
        , _problem(problem)
        , _uids(uids)
    {
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const
    {
        const auto cpg = _problem.cpg();
        const std::map<std::string, std::pair<char*, int64_t>> operands = {
            {"x",
             {bufferOf(_uids[0], deviceBuffers, numDeviceBuffers),
              2 * _problem.n * _problem.h * _problem.w * _problem.c}},
            {"w",
             {bufferOf(_uids[1], deviceBuffers, numDeviceBuffers),
              2 * _problem.k * _problem.y * _problem.x * cpg}},
            {"y",
             {bufferOf(_uids[2], deviceBuffers, numDeviceBuffers),
              2 * _problem.n * _problem.ho() * _problem.wo() * _problem.k}},
        };
        const auto base = [&](const std::string& buffer) -> char* {
            const auto operand = operands.find(buffer);
            if(operand != operands.end())
            {
                return operand->second.first;
            }
            return static_cast<char*>(workspace) + _workspaceOffsets.at(buffer);
        };
        for(const auto& step : _steps)
        {
            // One 8-byte slot per argument, in ABI order; HIP reads each argument's own size
            // (a pointer, or an i32 in the slot's low bytes).
            std::vector<uint64_t> slots(step.args.size());
            std::vector<void*> params(step.args.size());
            for(size_t i = 0; i < step.args.size(); ++i)
            {
                const auto& arg = step.args[i];
                if(arg.kind == geometry::Arg::Kind::POINTER)
                {
                    const void* address = base(arg.buffer) + arg.value;
                    std::memcpy(&slots[i], static_cast<const void*>(&address), sizeof(address));
                }
                else
                {
                    const auto value = static_cast<int32_t>(arg.kind == geometry::Arg::Kind::BYTES
                                                                ? operands.at(arg.buffer).second
                                                                : arg.value);
                    std::memcpy(&slots[i], &value, sizeof(value));
                }
                params[i] = &slots[i];
            }
            step.code->kernelForStream(stream).launchWithParams(stream, params.data());
        }
    }

private:
    static char* bufferOf(int64_t uid, const hipdnnPluginDeviceBuffer_t* buffers, uint32_t count)
    {
        return static_cast<char*>(hipdnn_plugin_sdk::findDeviceBuffer(uid, buffers, count).ptr);
    }

    std::vector<PreparedStep> _steps;
    std::map<std::string, int64_t> _workspaceOffsets;
    geometry::Problem _problem;
    std::array<int64_t, 3> _uids;
};

/// The engine's one dispatch: an instance launches its `kernel_source` once per plan step.
class Gfx1250ConvFwdDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit Gfx1250ConvFwdDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(matchedPlan(context, bound, kernel).workspaceBytes);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto matched = matchedPlan(context, bound, kernel);
        if(matched.plan.steps.empty())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                describeDescriptor("kernel", kernel.name, kernel.kernelId)
                    + ": an empty launch plan");
        }
        // The code is built once and shared by every step (one per batch chunk); those
        // steps share a grid too.
        const auto& first = matched.plan.steps.front();
        auto code = buildIngestorKernelCode(_kpackLoader, context, kernel, signatureOf(first));
        code.setBlockSize(launchDim(first.block[0], "block"), 1, 1);
        code.setGridSize(launchDim(first.grid[0], "grid x"),
                         launchDim(first.grid[1], "grid y"),
                         launchDim(first.grid[2], "grid z"));
        const auto shared = std::make_shared<const IngestorKernelCode>(std::move(code));
        std::vector<PreparedStep> steps;
        for(const auto& step : matched.plan.steps)
        {
            if(step.grid != first.grid || step.block != first.block)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    describeDescriptor("kernel", kernel.name, kernel.kernelId)
                        + ": two steps need different launch geometry");
            }
            steps.push_back(PreparedStep{shared, step.args});
        }
        return std::make_unique<PreparedGfx1250ConvFwd>(std::move(steps),
                                                        matched.workspaceOffsets,
                                                        matched.problem.problem,
                                                        matched.problem.uids);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        dynamic_cast<const PreparedGfx1250ConvFwd&>(prepared).launch(
            handle.getStream(), deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

Gfx1250ConvFwdDispatchHandler& dispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx1250ConvFwdAotKpackModuleCache());
    static Gfx1250ConvFwdDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

compilation::KpackModuleCache& gfx1250ConvFwdAotKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx1250ConvFwdAotModuleCache()
{
    gfx1250ConvFwdAotKpackModuleCache().clear();
}

void registerGfx1250ConvFwdAotSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &prepareKernel);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &dispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
