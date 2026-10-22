#include <miopen/conv/solvers.hpp>

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_ops.hpp>

#include <hipconv/hipconv.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_HIPCONV)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

// The maximum number of kernel configurations to include.
//
// Hipconv returns the estimated top-k configurations for the given layer parameters.
// Ensure that every call site requests the same number of configs, so that the config
// index is consistent across calls.
constexpr std::size_t MAX_CONFIGS = hipconv::ALL_RANKED_CONFIGS;

// The device the context's stream runs on, or nullopt for an architecture hipconv lacks.
static std::optional<hipconv::Device> ResolveDevice(const ExecutionContext& ctx)
{
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return std::nullopt;
    return hipconv::Device{*arch, static_cast<int>(ctx.GetStream().GetMaxComputeUnits())};
}

// Translate a MIOpen problem into hipconv's parameter struct.
static hipconv::ConvParams ToHipconvParams(const ProblemDescription& problem)
{
    hipconv::ConvParams par{};

    if(problem.IsDirectionForward())
        par.direction = hipconv::Direction::Fprop;
    else if(problem.IsDirectionBackwardData())
        par.direction = hipconv::Direction::Dgrad;
    else
        par.direction = hipconv::Direction::Wgrad;

    par.n  = ProblemInterpreter::GetBatchN(problem);
    par.c  = ProblemInterpreter::GetInputChannelC(problem);
    par.h  = ProblemInterpreter::GetInputHeightHi(problem);
    par.w  = ProblemInterpreter::GetInputWidthWi(problem);
    par.k  = ProblemInterpreter::GetOutputChannelK(problem);
    par.kh = ProblemInterpreter::GetFilterHeightY(problem);
    par.kw = ProblemInterpreter::GetFilterWidthX(problem);

    par.pad_h      = ProblemInterpreter::GetInputLeftPadH(problem);
    par.pad_w      = ProblemInterpreter::GetInputLeftPadW(problem);
    par.stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    par.stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    par.dilation_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    par.dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    par.groups     = ProblemInterpreter::GetGroupCountG(problem);

    par.p = ProblemInterpreter::GetOutputHeightHo(problem);
    par.q = ProblemInterpreter::GetOutputWidthWo(problem);
    // Every output extent must be set: ConvParams leaves them at -1 for
    // "unspecified", and ConvSize multiplies them into a size_t.
    par.e = 1;

    if(problem.Is3d())
    {
        par.dims       = 3;
        par.d          = ProblemInterpreter::GetInputDepthDi(problem);
        par.kd         = ProblemInterpreter::GetFilterDepthZ(problem);
        par.pad_d      = ProblemInterpreter::GetInputLeftPadD(problem);
        par.stride_d   = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
        par.dilation_d = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
        par.e          = ProblemInterpreter::GetOutputDepthDo(problem);
    }

    if(problem.IsFp16())
    {
        par.input_type  = hipconv::DataType::fp16;
        par.weight_type = hipconv::DataType::fp16;
        par.output_type = hipconv::DataType::fp16;
    }
    else if(problem.IsBfp16())
    {
        par.input_type  = hipconv::DataType::bf16;
        par.weight_type = hipconv::DataType::bf16;
        par.output_type = hipconv::DataType::bf16;
    }
    else if(problem.IsFp32() && problem.UseTF32())
    {
        // tf32 has fp32 operands and, storing fp32, an fp32 output.
        par.input_type  = hipconv::DataType::tf32;
        par.weight_type = hipconv::DataType::tf32;
        par.output_type = hipconv::DataType::fp32;
    }
    else
    {
        MIOPEN_THROW("ConvHipConv: unsupported data type.");
    }

    // Always NHWC, whatever the problem's layout.
    //
    // hipconv implements NHWC kernels only: every kernel family rejects
    // `par.order != TensorOrder::NHWC`. An NCHW problem is served by transposing its
    // tensors into packed NHWC scratch, launching there, and transposing the produced
    // tensor back, so by the time hipconv sees the buffers they are NHWC. Asking for
    // TensorOrder::NCHW would instead match no kernel at all.
    par.order = hipconv::TensorOrder::NHWC;

    // Fold to 2D once here, so every call site sees the same params.
    //
    // unfolded() does not read par.order, and folding depth into the batch is a
    // reshape only when channels are last.
    return par.order == hipconv::TensorOrder::NHWC ? par.unfolded() : par;
}

// ===================== NCHW staging =====================

// hipconv's three launch arguments, named by the forward-convention operand each
// carries. `internal::ConvOperandTag` is the CK solvers' vocabulary for this, reused
// here so the two NHWC-only backends describe their operands the same way.
//
//              arg 0 (in)     arg 1 (wei)   arg 2 (out, transposed back)
//   Fprop      Input   = x    Weights = w   Output  = y
//   Dgrad      Output  = dy   Weights = w   Input   = dx
//   Wgrad      Input   = x    Output  = dy  Weights = dw
static std::array<internal::ConvOperandTag, 3> GetSlotOperands(const ProblemDescription& problem)
{
    using internal::ConvOperandTag;
    if(problem.IsDirectionForward())
        return {ConvOperandTag::Input, ConvOperandTag::Weights, ConvOperandTag::Output};
    if(problem.IsDirectionBackwardData())
        return {ConvOperandTag::Output, ConvOperandTag::Weights, ConvOperandTag::Input};
    return {ConvOperandTag::Input, ConvOperandTag::Output, ConvOperandTag::Weights};
}

// The descriptor of the tensor an operand tag names.
//
// Sizes and element types MUST come from the descriptors, not from the
// forward-convention ProblemInterpreter accessors: MIOpen swaps x and y for the
// backward passes, so the accessors describe a different tensor than the one being
// moved. ProblemDescription's own `in`/`out` carry that swap (its ctor documents `in`
// as x for Forward and y for Backward*), which is exactly the mapping ConvTensors
// applies to the invoke params, so a tag resolves to the same tensor on both sides.
static const TensorDescriptor& GetOperandDescriptor(const ProblemDescription& problem,
                                                    internal::ConvOperandTag tag)
{
    if(tag == internal::ConvOperandTag::Weights)
        return problem.GetWeights();
    const bool wants_x = tag == internal::ConvOperandTag::Input;
    return wants_x == problem.IsDirectionForward() ? problem.GetIn() : problem.GetOut();
}

// The NCHW <-> NHWC transposes a problem needs, per hipconv launch argument.
//
// A slot is empty when the problem is already NHWC, or when its transpose is a layout
// no-op - BatchedTransposeSolution::IsSkippable(), i.e. a unit channel or spatial
// extent, which covers depthwise weights and 1x1 filters - in which case the caller's
// pointer goes to hipconv unchanged. (The CK path has no equivalent and always pays
// for all three.)
//
// The base class holds either direction: TransposeSolutionDefault2Nhwc and
// TransposeSolutionNhwc2Default only permute constructor arguments and add no state,
// so storing them sliced loses nothing.
struct HipConvTransposePlan
{
    std::array<std::optional<BatchedTransposeSolution>, 3> slot;
    std::array<internal::ConvOperandTag, 3> tag{};

    size_t Bytes(size_t i) const
    {
        const auto& s = slot[i];
        return s.has_value() ? s->GetOutputTensorSize() : 0;
    }

    // The operand an argument carries, for logging.
    const char* GetOperandName(size_t i) const
    {
        if(tag[i] == internal::ConvOperandTag::Input)
            return "input";
        if(tag[i] == internal::ConvOperandTag::Weights)
            return "weights";
        return "output";
    }
};

static HipConvTransposePlan MakeTransposePlan(const ExecutionContext& ctx,
                                              const ProblemDescription& problem)
{
    HipConvTransposePlan plan;
    plan.tag = GetSlotOperands(problem);
    if(!problem.IsLayoutDefault())
        return plan;

    for(size_t i = 0; i < plan.slot.size(); ++i)
    {
        const auto& desc = GetOperandDescriptor(problem, plan.tag[i]);
        const auto& lens = desc.GetLengths();

        // 4D and 32-bit-safe: IsSupportedProblem() gates on
        // BatchedTransposeSolution::IsApplicable(), which rejects anything else before
        // these narrowing casts (and the ctors' overflow checks) are reached.
        const auto n = static_cast<uint32_t>(lens[0]);
        const auto c = static_cast<uint32_t>(lens[1]);
        const auto h = static_cast<uint32_t>(lens[2]);
        const auto w = static_cast<uint32_t>(lens[3]);

        // Arguments 0 and 1 feed hipconv, argument 2 receives from it.
        if(i < 2)
        {
            const TransposeSolutionDefault2Nhwc sol(ctx, desc.GetType(), n, c, h, w);
            if(!sol.IsSkippable())
                plan.slot[i] = sol;
        }
        else
        {
            const TransposeSolutionNhwc2Default sol(ctx, desc.GetType(), n, c, h, w);
            if(!sol.IsSkippable())
                plan.slot[i] = sol;
        }
    }

    return plan;
}

// Whether `problem` is a wgrad whose dw is narrower than fp32.
static bool HasNarrowDw(const ProblemDescription& problem)
{
    return problem.IsDirectionBackwardWrW() && !problem.IsFp32();
}

// Bytes of fp32 staging a kernel launched with `par` needs before its dw can be written.
//
// A kernel that writes dw in the weight type needs none; for one that writes fp32 for an
// fp16/bf16 problem, the solver stages that output and casts it down.
static size_t GetWgradCastSize(const ProblemDescription& problem, const hipconv::ConvParams& par)
{
    if(!HasNarrowDw(problem) || par.weight_grad_type != hipconv::DataType::fp32)
        return 0;
    return problem.GetWeights().GetElementSize() * GetTypeSize(miopenFloat);
}

// Workspace layout shared by GetWorkspaceSize() and GetSolution(), so the two can
// never disagree about where a sub-buffer lives.
//
// Slots 0-2 are the transpose staging buffers (0 when the slot is unused), slot 3 the
// wgrad fp32 cast buffer, slot 4 hipconv's own per-kernel workspace. Slots 3 and 4
// depend on the config: GetWorkspaceSize() reports the largest total over all configs,
// and GetSolution() sizes the picked one. Keeping them last leaves the transpose offsets
// identical between the two.
static MultiBufferWorkspaceTraits
GetWorkspaceLayout(const HipConvTransposePlan& plan, size_t cast_sz, size_t hipconv_sz)
{
    return MultiBufferWorkspaceTraits(
        {plan.Bytes(0), plan.Bytes(1), plan.Bytes(2), cast_sz, hipconv_sz});
}

// Everything this solver requires of a problem that needs neither the GPU nor the
// hipconv registry to decide: rank, data type, and layout.
//
// GetWorkspaceSize() shares the gate with IsApplicable() because ToHipconvParams()
// (which throws on an unsupported type) and MakeTransposePlan() (4D, packed,
// transposable) are only defined on problems that pass it.
static bool IsSupportedProblem(const ProblemDescription& problem)
{
    if(!problem.Is2d() && !problem.Is3d())
        return false;
    // fp16, bf16, and tf32 (fp32 data with tf32 compute enabled).
    if(!problem.IsFp16() && !problem.IsBfp16() && !(problem.IsFp32() && problem.UseTF32()))
        return false;
    // A non-packed tensor has no flat buffer, which neither hipconv nor the transposes
    // can address.
    if(problem.HasNonPackedTensors())
        return false;

    // NDHWC reaches hipconv directly, so 3D needs nothing of the transposes.
    if(problem.IsLayoutNHWC())
        return true;
    if(!problem.IsLayoutDefault())
        return false;

    // NCHW goes through packed NHWC scratch, and MakeTransposePlan() reads four extents
    // positionally, so 3D would silently lose one. TransposeSolutionDefault2Ndhwc is
    // what a 3D channels-first path would be built on.
    if(!problem.Is2d())
        return false;
    return BatchedTransposeSolution::IsApplicable(problem.GetInDataType(),
                                                  problem.GetIn().GetLengths()) &&
           BatchedTransposeSolution::IsApplicable(problem.GetWeightsDataType(),
                                                  problem.GetWeights().GetLengths()) &&
           BatchedTransposeSolution::IsApplicable(problem.GetOutDataType(),
                                                  problem.GetOut().GetLengths());
}

// A config's perf-db record, "<config_version>:family[full description]".
//
// A full description selects one config only within a hipconv minor version, so a record
// from another version, or from a build with no version, must match nothing.
static std::string HipConvConfigRecord(hipconv::ConvKernelHandle kernel)
{
    return std::string{hipconv::config_version()} + ":" + hipconv::describe_kernel(kernel);
}

// Whether `record` carries this build's config version; "unknown" carries none.
static bool HasCurrentConfigVersion(std::string_view record)
{
    const auto version = hipconv::config_version();
    return version != "unknown" && record.starts_with(version) && record.size() > version.size() &&
           record[version.size()] == ':';
}

// The kernel a perf-db record names, or nullptr if the record has another config version or
// names no kernel of this architecture.
static hipconv::ConvKernelHandle FindKernel(const hipconv::Device& device, std::string_view record)
{
    if(!HasCurrentConfigVersion(record))
        return nullptr;
    record.remove_prefix(hipconv::config_version().size() + 1);
    return hipconv::find_kernel(device.arch, record).value_or(nullptr);
}

// `par` with dw in the weight type.
static hipconv::ConvParams WithNarrowDw(hipconv::ConvParams par)
{
    par.weight_grad_type = par.weight_type;
    return par;
}

// The params the solver launches `kernel` with: dw in the weight type if `kernel` accepts that,
// else fp32.
static hipconv::ConvParams GetKernelParams(hipconv::ConvKernelHandle kernel,
                                           const hipconv::Device& device,
                                           const ProblemDescription& problem)
{
    auto par = ToHipconvParams(problem);
    if(HasNarrowDw(problem) && hipconv::is_applicable(kernel, device, WithNarrowDw(par)))
        par.weight_grad_type = par.weight_type;
    return par;
}

// Whether the solver may launch `kernel` on `problem`, the condition under which GetConfigs()
// lists it.
static bool IsSelectable(hipconv::ConvKernelHandle kernel,
                         const hipconv::Device& device,
                         const ProblemDescription& problem)
{
    const auto par = GetKernelParams(kernel, device, problem);
    return hipconv::is_applicable(kernel, device, par) &&
           (!problem.GetConv().attribute.deterministic ||
            hipconv::is_deterministic(kernel, device, par));
}

// The configs the solver chooses among, best first.
//
// For an fp16/bf16 wgrad, GetConfigs() enumerates twice: with dw in the weight type, which
// only some kernels write, and with fp32 dw, which the solver casts down for the others. Each
// kernel is listed once, with the narrower dw where it writes it, and the union is ranked by
// each entry's WTI. Under the deterministic attribute, the kernels that add into their output
// atomically are dropped.
static std::vector<hipconv::ConvKernelHandle> GetConfigs(const hipconv::Device& device,
                                                         const ProblemDescription& problem)
{
    const bool deterministic = problem.GetConv().attribute.deterministic;
    std::vector<std::pair<hipconv::ConvKernelHandle, float>> ranked;
    std::unordered_set<hipconv::ConvKernelHandle> listed;
    const auto add = [&](const hipconv::ConvParams& par) {
        for(auto* kernel : hipconv::get_valid_configs(device, par, MAX_CONFIGS))
        {
            if(!listed.insert(kernel).second)
                continue;
            if(deterministic && !hipconv::is_deterministic(kernel, device, par))
                continue;
            ranked.emplace_back(kernel,
                                hipconv::get_weighted_throughput_index(kernel, device, par));
        }
    };

    const auto par = ToHipconvParams(problem);
    if(HasNarrowDw(problem))
        add(WithNarrowDw(par));
    add(par);

    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::vector<hipconv::ConvKernelHandle> cfgs;
    cfgs.reserve(ranked.size());
    for(const auto& entry : ranked)
        cfgs.push_back(entry.first);
    return cfgs;
}

// The kernel a perf config selects, or nullptr.
//
// IsValid() caches it, so a config already validated for this problem resolves without
// enumerating.
static hipconv::ConvKernelHandle ResolveKernel(const ExecutionContext& ctx,
                                               const hipconv::Device& device,
                                               const ProblemDescription& problem,
                                               const PerformanceConfigConvHipConv& config)
{
    auto* const cached = static_cast<hipconv::ConvKernelHandle>(config.kernel);
    if(cached != nullptr && IsSelectable(cached, device, problem))
        return cached;
    if(!config.IsValid(ctx, problem))
        return nullptr;
    return static_cast<hipconv::ConvKernelHandle>(config.kernel);
}

// The kernel's identity as MIOPEN_PERFORMANCE_LOGS should report it.
//
// A perf-log kernel record carries a name and nothing else, so the name has to hold both
// halves of a hipconv kernel's identity: the family (hipconv::name) and the config it was
// compiled with (hipconv::describe_config), joined as family[field=value,...]. The
// bracketed half is verbatim what hipconv::matches_descriptor() accepts, so a consumer
// that splits on '[' can hand it back to hipconv to re-select this exact kernel. A family
// that publishes no descriptor fields reports the bare family name.
static std::string HipConvKernelLabel(hipconv::ConvKernelHandle kernel)
{
    const auto family = hipconv::name(kernel);
    const auto config = hipconv::describe_config(kernel);
    if(config.empty())
        return std::string{family};
    return std::string{family} + "[" + config + "]";
}

// Time one hipconv launch and file it under `label` in the performance log.
//
// hipconv launches through the HIP runtime instead of MIOpen's HIPOCKernel, so the
// AddKernelToJsonAccumulator call in HIPOCKernelInvoke::run never fires for it. Without
// this the JSON record carries an empty kernels[] and falls back to naming the config
// after the solver, so the log says ConvHipConv ran but not which hipconv kernel or
// tuning config produced the time, which is exactly what a heuristic keys on.
//
// Engaged only when performance logging is on and the handle is profiling, matching
// HIPOCKernelInvoke::run, which logs MIOpen's own kernels from the event pair it records
// under the same two conditions. A production run pays one env-var read and a branch,
// and a non-profiling run keeps its asynchronous launches free of an added sync.
//
// Scoped tightly around the hipconv launch so that the NCHW transposes, which MIOpen
// logs as their own kernels, are not folded into the hipconv kernel's time.
class ScopedHipConvKernelLog
{
public:
    // `label` must outlive the scope; callers pass an invoker-captured string.
    ScopedHipConvKernelLog(const Handle& handle, const std::string& label)
        : stream(handle.GetStream()),
          name(label),
          engaged(IsLoggingKernel() && handle.IsProfilingEnabled() && !label.empty())
    {
        if(!engaged)
            return;
        start = make_hip_event();
        stop  = make_hip_event();
        (void)hipEventRecord(start.get(), stream);
    }

    ~ScopedHipConvKernelLog()
    {
        if(!engaged)
            return;
        (void)hipEventRecord(stop.get(), stream);
        (void)hipEventSynchronize(stop.get());
        float elapsed_ms = 0.0f;
        (void)hipEventElapsedTime(&elapsed_ms, start.get(), stop.get());
        AddKernelToJsonAccumulator(name, elapsed_ms, false);
    }

    ScopedHipConvKernelLog(const ScopedHipConvKernelLog&)            = delete;
    ScopedHipConvKernelLog& operator=(const ScopedHipConvKernelLog&) = delete;

private:
    hipStream_t stream;
    const std::string& name;
    HipEventPtr start;
    HipEventPtr stop;
    bool engaged;
};

// ===================== PerformanceConfigConvHipConv =====================

void PerformanceConfigConvHipConv::InitFromDevice(const void* device,
                                                  const ProblemDescription& problem)
{
    const auto cfgs = GetConfigs(*static_cast<const hipconv::Device*>(device), problem);
    config_count    = static_cast<int>(cfgs.size());
    index           = cfgs.empty() ? -1 : 0;
    kernel          = cfgs.empty() ? nullptr : cfgs[0];
    descriptor      = cfgs.empty() ? std::string{} : HipConvConfigRecord(cfgs[0]);
}

void PerformanceConfigConvHipConv::HeuristicInit(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    const auto device = ResolveDevice(ctx);
    if(!device.has_value())
        return;
    InitFromDevice(&*device, problem);
}

bool PerformanceConfigConvHipConv::SetNextValue(const ProblemDescription&)
{
    if(index + 1 >= config_count)
        return false;
    ++index;
    return true;
}

bool PerformanceConfigConvHipConv::IsValidValue() const { return index >= 0; }

bool PerformanceConfigConvHipConv::IsValid(const ExecutionContext& ctx,
                                           const ProblemDescription& problem) const
{
    const auto device = ResolveDevice(ctx);
    if(!device.has_value())
        return false;

    // Search path: `index` selects from GetConfigs(), and `descriptor` is set from that kernel
    // so the perf db stores a benchmarked pick by name.
    if(index >= 0)
    {
        const auto cfgs = GetConfigs(*device, problem);
        // Also sizes the config list on behalf of SetNextValue, which has no
        // ExecutionContext to resolve the device with. ComputedIterator
        // (generic_search.hpp) calls IsValid before every SetNextValue.
        config_count = static_cast<int>(cfgs.size());
        if(index >= config_count)
            return false;
        kernel     = cfgs[index];
        descriptor = HipConvConfigRecord(cfgs[index]);
        return true;
    }

    // Perf-db and perf-config-picker path: look the record up without enumerating.
    auto* const found = FindKernel(*device, descriptor);
    if(found == nullptr || !IsSelectable(found, *device, problem))
        return false;
    kernel = found;
    return true;
}

bool PerformanceConfigConvHipConv::operator==(const PerformanceConfigConvHipConv& other) const
{
    return descriptor == other.descriptor;
}

// ===================== ConvHipConv =====================

bool ConvHipConv::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_HIPCONV))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!IsSupportedProblem(problem))
        return false;

    const auto device = ResolveDevice(ctx);
    if(!device.has_value())
        return false;

    // Every kernel that writes 16-bit dw also writes fp32 dw, so find_config() on the fp32-dw
    // params decides applicability; GetConfigs() is needed only under the deterministic
    // attribute.
    if(!problem.GetConv().attribute.deterministic)
        return hipconv::find_config(*device, ToHipconvParams(problem)).has_value();
    return !GetConfigs(*device, problem).empty();
}

size_t ConvHipConv::GetWorkspaceSize(const ExecutionContext& ctx,
                                     const ProblemDescription& problem) const
{
    if(!IsSupportedProblem(problem))
        return 0;

    // Max over configs: Find sizes one buffer here, before a config is picked, and both the
    // cast staging and the per-kernel workspace (for example, a kernel's reformatted weights)
    // vary by config.
    const auto plan = MakeTransposePlan(ctx, problem);
    size_t size     = GetWorkspaceLayout(plan, 0, 0).GetSize();
    if(const auto device = ResolveDevice(ctx); device.has_value())
    {
        for(auto* kernel : GetConfigs(*device, problem))
        {
            const auto par = GetKernelParams(kernel, *device, problem);
            const auto wt  = GetWorkspaceLayout(plan,
                                               GetWgradCastSize(problem, par),
                                               hipconv::get_workspace_size(kernel, *device, par));
            size           = std::max(size, wt.GetSize());
        }
    }
    return size;
}

// Estimated quality, consulted only on the immediate-mode fallback (no Find).
//
// The selected kernel reports its own quality: hipconv scores grouped and
// large-channel direct configs at full utilization and small-channel direct (a
// coverage fallback) below that, so a faster MIOpen solver can outrank it. Find
// is unaffected: it benchmarks and ignores this.
float ConvHipConv::GetWti(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    const auto device = ResolveDevice(ctx);
    if(!device.has_value())
        return wti_approximate_worst;
    const auto cfgs = GetConfigs(*device, problem);
    if(cfgs.empty())
        return wti_approximate_worst;
    return hipconv::get_weighted_throughput_index(
        cfgs[0], *device, GetKernelParams(cfgs[0], *device, problem));
}

PerformanceConfigConvHipConv
ConvHipConv::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                         const ProblemDescription& problem) const
{
    PerformanceConfigConvHipConv config;
    config.HeuristicInit(ctx, problem);
    return config;
}

bool ConvHipConv::IsValidPerformanceConfig(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const PerformanceConfigConvHipConv& config) const
{
    return config.IsValid(ctx, problem);
}

PerformanceConfigConvHipConv ConvHipConv::Search(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem,
                                                 const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvHipConv::GetSolution(const ExecutionContext& ctx,
                                      const ProblemDescription& problem,
                                      const PerformanceConfigConvHipConv& config) const
{
    ConvSolution result;

    const auto resolved = ResolveDevice(ctx);
    if(!resolved.has_value())
        MIOPEN_THROW("ConvHipConv: unsupported architecture.");
    const hipconv::Device device = *resolved;

    auto* const kernel = ResolveKernel(ctx, device, problem, config);
    if(kernel == nullptr)
        MIOPEN_THROW("ConvHipConv: performance config does not resolve to a kernel.");
    const auto par = GetKernelParams(kernel, device, problem);

    MIOPEN_LOG_I(hipconv::name(kernel) << ": " << hipconv::describe_config(kernel));

    // Name the kernel for MIOPEN_PERFORMANCE_LOGS. Empty unless logging is on, which
    // leaves ScopedHipConvKernelLog inert; each invoker below captures it by copy.
    const std::string kernel_label =
        IsPerformanceLoggingEnabled() ? HipConvKernelLabel(kernel) : std::string{};

    // NCHW staging. Empty for an NHWC problem, in which case no transpose kernel is
    // built, every staging size below is 0, and the invokers hand hipconv the caller's
    // tensors directly.
    const auto plan          = MakeTransposePlan(ctx, problem);
    const auto cast_sz       = GetWgradCastSize(problem, par);
    const auto hipconv_ws_sz = hipconv::get_workspace_size(kernel, device, par);
    const auto wt            = GetWorkspaceLayout(plan, cast_sz, hipconv_ws_sz);

    result.workspace_sz = wt.GetSize();

    // One internal::TransposeInstance per staged argument, the same object the CK
    // solvers drive their layout transforms with: it owns the staging buffer's size and
    // offset, the kernel's index into kernels[] (which mirrors construction_params), and
    // the patched kernel args. An empty optional is an argument that needs no transpose.
    //
    // Data movement goes through TransposeInstance's explicit-pointer ConvertFrom /
    // ConvertTo rather than TransposeInstanceTagged's ConvTensors overloads: this solver
    // names hipconv's three arguments directly, so the tag-driven pointer pick (and the
    // backward-pass x/y unswap it forces) would only add a second, desynchronizable
    // mapping. The tags stay in the plan, where they document the wiring.
    std::array<std::optional<internal::TransposeInstance>, 3> trans;

    for(size_t i = 0; i < plan.slot.size(); ++i)
    {
        const auto& slot = plan.slot[i];
        if(!slot.has_value())
            continue;
        const auto kernel_idx = result.construction_params.size();
        result.construction_params.push_back(slot->GetKernelInfo());
        trans[i].emplace(*slot, kernel_idx, wt, i);
        MIOPEN_LOG_I2("ConvHipConv: operand " << plan.GetOperandName(i) << " transpose "
                                              << slot->GetKernelName());
    }

    const auto cast_off       = wt.GetOffset(3);
    const auto hipconv_ws_off = wt.GetOffset(4);
    const auto workspace_sz   = result.workspace_sz;

    if(problem.IsDirectionBackwardWrW())
    {
        const bool need_cast  = cast_sz != 0;
        const auto lowp_quant = problem.GetConv().lowp_quant;

        // fp32 view of the wgrad output. The cast is elementwise over packed buffers, so
        // only the element count matters: this describes the fp32 staging correctly
        // whether it holds NCHW or (under NCHW staging) NHWC-ordered gradients.
        const TensorDescriptor cast_desc(
            miopenFloat, problem.GetWeights().GetLengths(), problem.GetWeights().GetStrides());

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) mutable {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
                decltype(auto) wrw_ctx =
                    primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& tensors   = wrw_ctx.tensors;
                const auto& workSpace = wrw_ctx.workSpace;

                if(workspace_sz > 0 &&
                   (workSpace == nullptr || wrw_ctx.workSpaceSize < workspace_sz))
                    MIOPEN_THROW("ConvHipConv: not enough workspace for wgrad.");

                // Whole-scope event timing, so the transposes and the cast are billed to
                // this solver too. It overrides the running total TransposeInstance keeps
                // (HipEventProfiler's destructor resets before accumulating), so the two
                // do not double count.
                const HipEventProfiler profiler(handle);

                for(auto& t : trans)
                    if(t.has_value())
                        t->AssignBuffer(handle, workSpace);

                if(trans[0].has_value())
                    trans[0]->ConvertFrom(handle, kernels, tensors.x);
                if(trans[1].has_value())
                    trans[1]->ConvertFrom(handle, kernels, tensors.dy);

                const auto cast_buf = cast_sz != 0
                                          ? handle.CreateSubBuffer(workSpace, cast_off, cast_sz)
                                          : shared<Data_t>{};
                const auto hipconv_ws =
                    hipconv_ws_sz != 0
                        ? handle.CreateSubBuffer(workSpace, hipconv_ws_off, hipconv_ws_sz)
                        : shared<Data_t>{};

                // Where hipconv writes the gradient: the cast staging when the kernel
                // writes fp32 for a narrower dw, else the NHWC staging, else dw itself.
                void* const dw_staged =
                    trans[2].has_value() ? trans[2]->GetBufferPtr() : tensors.dw;
                void* const hipconv_dst = need_cast ? cast_buf.get() : dw_staged;

                {
                    const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                    if(const auto status = hipconv::launch(
                           kernel,
                           device,
                           par,
                           trans[0].has_value() ? trans[0]->GetBufferPtr() : tensors.x,
                           trans[1].has_value() ? trans[1]->GetBufferPtr() : tensors.dy,
                           hipconv_dst,
                           hipconv_ws.get(),
                           handle.GetStream());
                       status != hipSuccess)
                        MIOPEN_THROW_HIP_STATUS(status, "ConvHipConv: wgrad launch failed.");
                }

                // fp32 -> weight type. Elementwise over packed buffers, so it runs on
                // whichever layout dw_staged holds.
                if(need_cast)
                    CastTensor(handle,
                               &lowp_quant,
                               false,
                               cast_desc,
                               cast_buf.get(),
                               tensors.dwDesc,
                               dw_staged,
                               0,
                               0);

                // hipconv wrote dw as NHWC; hand MIOpen back the NCHW it asked for.
                if(trans[2].has_value())
                    trans[2]->ConvertTo(handle, kernels, tensors.dw);
            };
        };
    }
    else
    {
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) mutable {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
                decltype(auto) data_ctx =
                    primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                const auto& tensors   = data_ctx.tensors;
                const auto& workSpace = data_ctx.workSpace;

                if(workspace_sz > 0 &&
                   (workSpace == nullptr || data_ctx.workSpaceSize < workspace_sz))
                    MIOPEN_THROW("ConvHipConv: not enough workspace for direct kernel.");

                // See the wgrad invoker: this subsumes TransposeInstance's own timing.
                const HipEventProfiler profiler(handle);

                for(auto& t : trans)
                    if(t.has_value())
                        t->AssignBuffer(handle, workSpace);

                if(trans[0].has_value())
                    trans[0]->ConvertFrom(handle, kernels, tensors.in);
                if(trans[1].has_value())
                    trans[1]->ConvertFrom(handle, kernels, tensors.w);

                const auto hipconv_ws =
                    hipconv_ws_sz != 0
                        ? handle.CreateSubBuffer(workSpace, hipconv_ws_off, hipconv_ws_sz)
                        : shared<Data_t>{};

                {
                    const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                    if(const auto status = hipconv::launch(
                           kernel,
                           device,
                           par,
                           trans[0].has_value() ? trans[0]->GetBufferPtr() : tensors.in,
                           trans[1].has_value() ? trans[1]->GetBufferPtr() : tensors.w,
                           trans[2].has_value() ? trans[2]->GetBufferPtr() : tensors.out,
                           hipconv_ws.get(),
                           handle.GetStream());
                       status != hipSuccess)
                        MIOPEN_THROW_HIP_STATUS(status, "ConvHipConv: direct launch failed.");
                }

                if(trans[2].has_value())
                    trans[2]->ConvertTo(handle, kernels, tensors.out);
            };
        };
    }

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen

#else // MIOPEN_USE_HIPCONV

// hipconv is not built into this configuration.
//
// The solver is still registered so its solver id stays stable across build
// configs, but every method is an inert stub and IsApplicable returns false.

#include <miopen/generic_search.hpp>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

void PerformanceConfigConvHipConv::HeuristicInit(const ExecutionContext&, const ProblemDescription&)
{
}
bool PerformanceConfigConvHipConv::IsValidValue() const { return false; }
bool PerformanceConfigConvHipConv::SetNextValue(const ProblemDescription&) { return false; }
bool PerformanceConfigConvHipConv::IsValid(const ExecutionContext&, const ProblemDescription&) const
{
    return false;
}
bool PerformanceConfigConvHipConv::operator==(const PerformanceConfigConvHipConv&) const
{
    return true;
}
void PerformanceConfigConvHipConv::InitFromDevice(const void*, const ProblemDescription&) {}

bool ConvHipConv::IsApplicable(const ExecutionContext&, const ProblemDescription&) const
{
    return false;
}
size_t ConvHipConv::GetWorkspaceSize(const ExecutionContext&, const ProblemDescription&) const
{
    return 0;
}
float ConvHipConv::GetWti(const ExecutionContext&, const ProblemDescription&) const
{
    return wti_approximate_worst;
}
PerformanceConfigConvHipConv
ConvHipConv::GetDefaultPerformanceConfig(const ExecutionContext&, const ProblemDescription&) const
{
    return {};
}
bool ConvHipConv::IsValidPerformanceConfig(const ExecutionContext&,
                                           const ProblemDescription&,
                                           const PerformanceConfigConvHipConv&) const
{
    return false;
}
PerformanceConfigConvHipConv ConvHipConv::Search(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem,
                                                 const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}
ConvSolution ConvHipConv::GetSolution(const ExecutionContext&,
                                      const ProblemDescription&,
                                      const PerformanceConfigConvHipConv&) const
{
    MIOPEN_THROW("ConvHipConv: built without MIOPEN_USE_HIPCONV.");
}

} // namespace conv
} // namespace solver
} // namespace miopen

#endif // MIOPEN_USE_HIPCONV
