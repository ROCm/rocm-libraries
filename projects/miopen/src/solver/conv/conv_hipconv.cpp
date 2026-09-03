#include <miopen/conv/solvers.hpp>

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_tuning_mode.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_ops.hpp>

#include <hipconv/hipconv.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <string>

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

// Translate a MIOpen problem into hipconv's parameter struct.
static hipconv::Conv2dParams ToHipconvParams(const ProblemDescription& problem)
{
    hipconv::Conv2dParams par{};

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

    par.order = problem.IsLayoutNHWC() ? hipconv::TensorOrder::NHWC : hipconv::TensorOrder::NCHW;

    return par;
}

// Resolve the kernel handle a perf-config selected.
static hipconv::ConvKernelHandle ResolveKernel(hipconv::ArchHandle arch,
                                               const hipconv::Conv2dParams& par,
                                               const PerformanceConfigConvHipConv& config)
{
    if(config.index < 0)
        return nullptr;
    const auto cfgs = hipconv::get_valid_configs(arch, par, MAX_CONFIGS);
    if(config.index >= static_cast<int>(cfgs.size()))
        return nullptr;
    return cfgs[config.index];
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

void PerformanceConfigConvHipConv::InitFromArch(const void* arch, const ProblemDescription& problem)
{
    const auto par = ToHipconvParams(problem);
    const auto cfgs =
        hipconv::get_valid_configs(static_cast<hipconv::ArchHandle>(arch), par, MAX_CONFIGS);
    config_count = static_cast<int>(cfgs.size());
    index        = cfgs.empty() ? -1 : 0;
}

void PerformanceConfigConvHipConv::HeuristicInit(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return;
    InitFromArch(*arch, problem);
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
    // Size the config list here, on behalf of SetNextValue.
    //
    // ComputedIterator (generic_search.hpp) constructs a config, calls IsValid, and only
    // then calls SetNextValue, which has no ExecutionContext to resolve the arch with.
    if(config_count < 0)
    {
        const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
        if(!arch.has_value())
            return false;
        config_count = static_cast<int>(
            hipconv::get_valid_configs(*arch, ToHipconvParams(problem), MAX_CONFIGS).size());
    }
    return IsValidValue() && index < config_count;
}

bool PerformanceConfigConvHipConv::operator==(const PerformanceConfigConvHipConv& other) const
{
    return index == other.index;
}

// ===================== ConvHipConv =====================

bool ConvHipConv::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_HIPCONV))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    // fp16, bf16, and tf32 (fp32 data with tf32 compute enabled).
    if(!problem.IsFp16() && !problem.IsBfp16() && !(problem.IsFp32() && problem.UseTF32()))
        return false;
    // The wgrad kernel uses atomicAdd and is non-deterministic.
    if(problem.IsDirectionBackwardWrW() && problem.GetConv().attribute.deterministic)
        return false;

    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return false;

    const auto par = ToHipconvParams(problem);
    return hipconv::find_config(*arch, par).has_value();
}

size_t ConvHipConv::GetWorkspaceSize(const ExecutionContext& ctx,
                                     const ProblemDescription& problem) const
{
    if(problem.IsDirectionBackwardWrW())
    {
        // fp32 wgrad kernels do not use a workspace.
        if(problem.IsFp32())
            return 0;

        // fp16/bf16 wgrad needs an fp32 scratch.
        //
        // The kernel returns the gradient as fp32 but MIOpen wants dw in the
        // weight type, so stage the fp32 output in a workspace before converting.
        const auto k           = ProblemInterpreter::GetOutputChannelK(problem);
        const auto c           = ProblemInterpreter::GetInputChannelC(problem);
        const auto y           = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x           = ProblemInterpreter::GetFilterWidthX(problem);
        const auto group       = ProblemInterpreter::GetGroupCountG(problem);
        const auto c_per_group = c / group;
        return static_cast<size_t>(k) * y * x * c_per_group * sizeof(float);
    }

    // Max over configs: Find sizes one buffer here before picking a config, and
    // the direct_l1 formatted-weights size varies by config (block_k padding).
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return 0;
    const auto par  = ToHipconvParams(problem);
    const auto cfgs = hipconv::get_valid_configs(*arch, par, MAX_CONFIGS);
    size_t max_ws   = 0;
    for(auto* kernel : cfgs)
        max_ws = std::max(max_ws, hipconv::get_workspace_size(kernel, par));
    return max_ws;
}

// Estimated quality, consulted only on the immediate-mode fallback (no Find).
//
// The selected kernel reports its own quality: hipconv scores grouped and
// large-channel direct configs at full utilization and small-channel direct (a
// coverage fallback) below that, so a faster MIOpen solver can outrank it. Find
// is unaffected: it benchmarks and ignores this.
float ConvHipConv::GetWti(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return wti_approximate_worst;
    const auto par    = ToHipconvParams(problem);
    const auto kernel = hipconv::find_config(*arch, par);
    if(!kernel.has_value())
        return wti_approximate_worst;
    return hipconv::get_weighted_throughput_index(*kernel, par);
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

    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        MIOPEN_THROW("ConvHipConv: unsupported architecture.");

    const auto par     = ToHipconvParams(problem);
    auto* const kernel = ResolveKernel(*arch, par, config);
    if(kernel == nullptr)
        MIOPEN_THROW("ConvHipConv: performance config does not resolve to a kernel.");

    MIOPEN_LOG_I(hipconv::name(kernel) << ": " << hipconv::describe_config(kernel));

    // Name the kernel for MIOPEN_PERFORMANCE_LOGS. Empty unless logging is on, which
    // leaves ScopedHipConvKernelLog inert; each invoker below captures it by copy.
    const std::string kernel_label =
        IsPerformanceLoggingEnabled() ? HipConvKernelLabel(kernel) : std::string{};

    if(problem.IsDirectionBackwardWrW())
    {
        const auto workspace_size = GetWorkspaceSize(ctx, problem);
        result.workspace_sz       = workspace_size;

        // fp32 dw takes the kernel's fp32 output directly; no workspace.
        //
        // fp16/bf16 dw is narrower, so that path stages the fp32 output through
        // a workspace and casts it down. Today fp32 reaches here only via tf32.
        if(problem.IsFp32())
        {
            result.invoker_factory = [=](const std::vector<Kernel>&) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                    decltype(auto) wrw_ctx =
                        primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                    const auto& tensors = wrw_ctx.tensors;

                    const HipEventProfiler profiler(handle);
                    const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                    if(const auto status = hipconv::launch(kernel,
                                                           par,
                                                           tensors.x,
                                                           tensors.dy,
                                                           tensors.dw,
                                                           nullptr,
                                                           handle.GetStream());
                       status != hipSuccess)
                        MIOPEN_THROW_HIP_STATUS(status, "ConvHipConv: wgrad launch failed.");
                };
            };
            return result;
        }

        const auto lowp_quant = problem.GetConv().lowp_quant;
        // fp32 intermediate buffer, same shape as the weights.
        const TensorDescriptor cast_desc(
            miopenFloat, problem.GetWeights().GetLengths(), problem.GetWeights().GetStrides());

        result.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) wrw_ctx =
                    primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& tensors       = wrw_ctx.tensors;
                const auto& workSpace     = wrw_ctx.workSpace;
                const auto& workSpaceSize = wrw_ctx.workSpaceSize;

                if(workSpace == nullptr || workSpaceSize < workspace_size)
                    MIOPEN_THROW("ConvHipConv: not enough workspace for wgrad.");

                const HipEventProfiler profiler(handle);

                // wgrad kernel writes fp32 into the workspace...
                //
                // Scoped so the perf log times the hipconv kernel alone; the cast below
                // is logged separately by its own MIOpen kernel.
                {
                    const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                    if(const auto status = hipconv::launch(kernel,
                                                           par,
                                                           tensors.x,
                                                           tensors.dy,
                                                           workSpace,
                                                           nullptr,
                                                           handle.GetStream());
                       status != hipSuccess)
                        MIOPEN_THROW_HIP_STATUS(status, "ConvHipConv: wgrad launch failed.");
                }

                // ...then cast fp32 workspace -> fp16 dw.
                CastTensor(handle,
                           &lowp_quant,
                           false,
                           cast_desc,
                           workSpace,
                           tensors.dwDesc,
                           tensors.dw,
                           0,
                           0);
            };
        };
    }
    else
    {
        // direct_l1 (groups=1 fprop/dgrad) formats its weights into this
        // workspace before the conv; a null pointer faults at a low address.
        const auto workspace_size = hipconv::get_workspace_size(kernel, par);
        result.workspace_sz       = workspace_size;

        result.invoker_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx =
                    primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                const auto& tensors = data_ctx.tensors;

                if(workspace_size > 0 &&
                   (data_ctx.workSpace == nullptr || data_ctx.workSpaceSize < workspace_size))
                    MIOPEN_THROW("ConvHipConv: not enough workspace for direct kernel.");

                const HipEventProfiler profiler(handle);
                const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                if(const auto status = hipconv::launch(kernel,
                                                       par,
                                                       tensors.in,
                                                       tensors.w,
                                                       tensors.out,
                                                       data_ctx.workSpace,
                                                       handle.GetStream());
                   status != hipSuccess)
                    MIOPEN_THROW_HIP_STATUS(status, "ConvHipConv: direct launch failed.");
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
void PerformanceConfigConvHipConv::InitFromArch(const void*, const ProblemDescription&) {}

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
