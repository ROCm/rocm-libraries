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

    // Always NHWC, whatever the problem's layout.
    //
    // hipconv implements NHWC kernels only: every kernel family rejects
    // `par.order != TensorOrder::NHWC`. An NCHW problem is served by transposing its
    // tensors into packed NHWC scratch, launching there, and transposing the produced
    // tensor back, so by the time hipconv sees the buffers they are NHWC. Asking for
    // TensorOrder::NCHW would instead match no kernel at all.
    par.order = hipconv::TensorOrder::NHWC;

    return par;
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
        return slot[i].has_value() ? slot[i]->GetOutputTensorSize() : 0;
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

// Bytes of fp32 staging the wgrad output needs before it can be written to dw.
//
// The hipconv wgrad kernels emit fp32. An fp32 (tf32) problem takes that output as
// is; fp16/bf16 stages it and casts it down to the weight type.
static size_t GetWgradCastSize(const ProblemDescription& problem)
{
    if(!problem.IsDirectionBackwardWrW() || problem.IsFp32())
        return 0;
    return problem.GetWeights().GetElementSize() * GetTypeSize(miopenFloat);
}

// Workspace layout shared by GetWorkspaceSize() and GetSolution(), so the two can
// never disagree about where a sub-buffer lives.
//
// Slots 0-2 are the transpose staging buffers (0 when the slot is unused), slot 3 the
// wgrad fp32 cast buffer. hipconv's own per-kernel workspace goes last because it is
// the only slot whose size depends on the config: GetWorkspaceSize() has to report
// the maximum over all configs while GetSolution() sizes the one that was picked, and
// keeping it last leaves every other offset identical between the two.
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
    if(!problem.Is2d())
        return false;
    // fp16, bf16, and tf32 (fp32 data with tf32 compute enabled).
    if(!problem.IsFp16() && !problem.IsBfp16() && !(problem.IsFp32() && problem.UseTF32()))
        return false;

    if(problem.IsLayoutNHWC())
        return true;
    if(!problem.IsLayoutDefault())
        return false;

    // NCHW goes through packed NHWC scratch: a non-packed tensor has no flat buffer to
    // transpose, and the batched-transpose kernels cover only a fixed set of element
    // types and 32-bit extents.
    if(problem.HasNonPackedTensors())
        return false;
    return BatchedTransposeSolution::IsApplicable(problem.GetInDataType(),
                                                  problem.GetIn().GetLengths()) &&
           BatchedTransposeSolution::IsApplicable(problem.GetWeightsDataType(),
                                                  problem.GetWeights().GetLengths()) &&
           BatchedTransposeSolution::IsApplicable(problem.GetOutDataType(),
                                                  problem.GetOut().GetLengths());
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
    if(!IsSupportedProblem(problem))
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
    if(!IsSupportedProblem(problem))
        return 0;

    // Max over configs: Find sizes one buffer here, before a config is picked, and the
    // per-kernel workspace (the direct_l1 formatted weights, say) varies by config.
    size_t hipconv_ws = 0;
    if(const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName()); arch.has_value())
    {
        const auto par = ToHipconvParams(problem);
        for(auto* kernel : hipconv::get_valid_configs(*arch, par, MAX_CONFIGS))
            hipconv_ws = std::max(hipconv_ws, hipconv::get_workspace_size(kernel, par));
    }

    return GetWorkspaceLayout(
               MakeTransposePlan(ctx, problem), GetWgradCastSize(problem), hipconv_ws)
        .GetSize();
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

    // NCHW staging. Empty for an NHWC problem, in which case no transpose kernel is
    // built, every staging size below is 0, and the invokers hand hipconv the caller's
    // tensors directly.
    const auto plan          = MakeTransposePlan(ctx, problem);
    const auto cast_sz       = GetWgradCastSize(problem);
    const auto hipconv_ws_sz = hipconv::get_workspace_size(kernel, par);
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
        if(!plan.slot[i].has_value())
            continue;
        const auto kernel_idx = result.construction_params.size();
        result.construction_params.push_back(plan.slot[i]->GetKernelInfo());
        trans[i].emplace(*plan.slot[i], kernel_idx, wt, i);
        MIOPEN_LOG_I2("ConvHipConv: operand " << plan.GetOperandName(i) << " transpose "
                                              << plan.slot[i]->GetKernelName());
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

                // Where hipconv's fp32 gradient lands: the cast staging when dw is
                // narrower, else the NHWC staging, else dw itself.
                void* const dw_staged =
                    trans[2].has_value() ? trans[2]->GetBufferPtr() : tensors.dw;
                void* const hipconv_dst = need_cast ? cast_buf.get() : dw_staged;

                {
                    const ScopedHipConvKernelLog kernel_log(handle, kernel_label);
                    if(const auto status = hipconv::launch(
                           kernel,
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
