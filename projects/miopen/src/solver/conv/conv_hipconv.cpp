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
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_ops.hpp>

#include <hipconv/hipconv.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>

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

    // hipconv only implements NHWC kernels (every kernel family rejects
    // `par.order != TensorOrder::NHWC`), so always request NHWC regardless of the
    // problem's actual layout. GetSolution() transposes NCHW ("Default" layout)
    // tensors to/from NHWC scratch buffers around the hipconv launch.
    par.order = hipconv::TensorOrder::NHWC;

    return par;
}

// Resolve the kernel handle a perf-config selected.
//
// The config-list index is cross-checked against the recorded kernel name, so a
// stale/misindexed entry is rejected rather than launching the wrong kernel.
static hipconv::ConvKernelHandle ResolveKernel(hipconv::ArchHandle arch,
                                               const hipconv::Conv2dParams& par,
                                               const PerformanceConfigConvHipConv& config)
{
    if(config.index < 0)
        return nullptr;
    const auto cfgs = hipconv::get_valid_configs(arch, par, MAX_CONFIGS);
    if(config.index >= static_cast<int>(cfgs.size()))
        return nullptr;
    auto* kernel = cfgs[config.index];
    if(hipconv::name(kernel) != config.kernel_name)
        return nullptr;
    return kernel;
}

static size_t GetPackedTensorBytes(const TensorDescriptor& td)
{
    return td.GetElementSize() * GetTypeSize(td.GetType());
}

// Workspace layout shared by GetWorkspaceSize() and GetSolution(), so the two
// never disagree about where a given sub-buffer lives.
//
// Slots 0-2 are NCHW<->NHWC transpose scratch: zero-sized (and unused) unless
// the problem layout is "Default" (NCHW).
//   Fprop/Dgrad: 0 = in, 1 = weights, 2 = out
//   Wgrad:       0 = x (forward input), 1 = dy (output gradient), 2 unused
// Slot 3 is direction-specific:
//   Fprop/Dgrad: hipconv's own per-kernel workspace (e.g. direct_l1 formats its
//                weights into this before the conv)
//   Wgrad:       fp32 cast buffer for fp16/bf16 dw; unused (0) for fp32/tf32,
//                whose dw output needs no cast
static MultiBufferWorkspaceTraits GetWorkspaceLayout(const ProblemDescription& problem,
                                                     size_t hipconv_or_cast_sz)
{
    size_t buf0 = 0, buf1 = 0, buf2 = 0;
    if(problem.IsLayoutDefault())
    {
        if(problem.IsDirectionBackwardWrW())
        {
            buf0 = GetPackedTensorBytes(problem.GetOut()); // x
            buf1 = GetPackedTensorBytes(problem.GetIn());  // dy
        }
        else
        {
            buf0 = GetPackedTensorBytes(problem.GetIn());
            buf1 = GetPackedTensorBytes(problem.GetWeights());
            buf2 = GetPackedTensorBytes(problem.GetOut());
        }
    }
    return MultiBufferWorkspaceTraits({buf0, buf1, buf2, hipconv_or_cast_sz});
}

// ===================== PerformanceConfigConvHipConv =====================

void PerformanceConfigConvHipConv::InitFromArch(const void* arch, const ProblemDescription& problem)
{
    const auto par = ToHipconvParams(problem);
    const auto cfgs =
        hipconv::get_valid_configs(static_cast<hipconv::ArchHandle>(arch), par, MAX_CONFIGS);
    if(cfgs.empty())
    {
        index       = -1;
        kernel_name = "";
        return;
    }
    index       = 0;
    kernel_name = std::string(hipconv::name(cfgs[0]));
}

void PerformanceConfigConvHipConv::HeuristicInit(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return;
    InitFromArch(*arch, problem);
}

bool PerformanceConfigConvHipConv::SetNextValue(const ProblemDescription& problem)
{
    const auto arch = hipconv::resolve_arch(GetCurrentDeviceName());
    if(!arch.has_value())
        return false;
    const auto par  = ToHipconvParams(problem);
    const auto cfgs = hipconv::get_valid_configs(*arch, par, MAX_CONFIGS);

    if(index + 1 >= static_cast<int>(cfgs.size()))
        return false;
    ++index;
    kernel_name = std::string(hipconv::name(cfgs[index]));
    return true;
}

bool PerformanceConfigConvHipConv::IsValidValue() const { return index >= 0; }

bool PerformanceConfigConvHipConv::IsValid(const ExecutionContext& ctx,
                                           const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return false;
    const auto par = ToHipconvParams(problem);
    return ResolveKernel(*arch, par, *this) != nullptr;
}

bool PerformanceConfigConvHipConv::operator==(const PerformanceConfigConvHipConv& other) const
{
    return index == other.index && kernel_name == other.kernel_name;
}

std::string PerformanceConfigConvHipConv::GetCurrentDeviceName()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return {};
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return {};
    return props.gcnArchName;
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

    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;

    // The NCHW path transposes through a packed NHWC scratch buffer; a
    // non-packed tensor can't be handled by a flat-buffer transpose.
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
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
        // fp32 (tf32) wgrad kernels write their output directly into the
        // weight-gradient tensor; fp16/bf16 wgrad needs an fp32 scratch to hold
        // the kernel's fp32 output before it's cast down to the weight type.
        const auto wgrad_cast_sz =
            problem.IsFp32() ? size_t{0} : problem.GetWeights().GetElementSize() * sizeof(float);
        return GetWorkspaceLayout(problem, wgrad_cast_sz).GetSize();
    }

    // Max over configs: hipconv's own per-kernel workspace (e.g. the direct_l1
    // formatted-weights size) varies by config, and no config is chosen yet here.
    const auto arch    = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    size_t hipconv_ws  = 0;
    if(arch.has_value())
    {
        const auto par  = ToHipconvParams(problem);
        const auto cfgs = hipconv::get_valid_configs(*arch, par, MAX_CONFIGS);
        for(auto* kernel : cfgs)
            hipconv_ws = std::max(hipconv_ws, hipconv::get_workspace_size(kernel, par));
    }
    return GetWorkspaceLayout(problem, hipconv_ws).GetSize();
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

    const bool need_transpose = problem.IsLayoutDefault();

    // Transpose dimensions MUST come from the actual tensor descriptors, not
    // from forward-convention ToHipconvParams()/ProblemInterpreter values.
    // MIOpen swaps the in/out tensors for backward passes, so tensors.in /
    // tensors.out carry different channel/spatial extents than the forward
    // input/output.
    //
    //   Fprop: in = x,  out = y
    //   Dgrad: in = dy, out = dx   (swapped)
    //   Wgrad: in = dy, out = x    (swapped); weights = dw
    //
    // problem.GetIn()/GetOut()/GetWeights() are the exact descriptors of
    // tensors.in / tensors.out / tensors.w for every direction, so their
    // lengths and element types always match the tensor being transposed.
    const auto& in_lens  = problem.GetIn().GetLengths();      // tensors.in
    const auto& out_lens = problem.GetOut().GetLengths();     // tensors.out
    const auto& wei_lens = problem.GetWeights().GetLengths(); // weights

    const auto in_n = static_cast<int>(in_lens[0]);
    const auto in_c = static_cast<int>(in_lens[1]);
    const auto in_h = static_cast<int>(in_lens[2]);
    const auto in_w = static_cast<int>(in_lens[3]);

    const auto out_n = static_cast<int>(out_lens[0]);
    const auto out_c = static_cast<int>(out_lens[1]);
    const auto out_h = static_cast<int>(out_lens[2]);
    const auto out_w = static_cast<int>(out_lens[3]);

    const auto wei_k = static_cast<int>(wei_lens[0]);
    const auto wei_c = static_cast<int>(wei_lens[1]);
    const auto wei_y = static_cast<int>(wei_lens[2]);
    const auto wei_x = static_cast<int>(wei_lens[3]);

    // Build transpose kernels for NCHW <-> NHWC if needed. These go into
    // construction_params so MIOpen compiles them; at runtime they are invoked
    // via handle.Run(kernels[idx]).
    //
    // Fprop/Dgrad kernel indices:
    //   0 = trans_in   (NCHW -> NHWC, tensors.in)
    //   1 = trans_wei  (NCHW -> NHWC, weights)
    //   2 = trans_out  (NHWC -> NCHW, tensors.out)
    //
    // Wgrad kernel indices:
    //   0 = trans_x    (NCHW -> NHWC, forward input x  = problem.GetOut())
    //   1 = trans_dy   (NCHW -> NHWC, output grad dy   = problem.GetIn())
    //
    // Wgrad's dw output needs no transpose back: hipconv writes it directly in
    // the layout problem.GetWeights() describes, the same assumption the
    // existing fp16/bf16 cast path below already relies on.
    std::vector<std::vector<OpKernelArg>> trans_kernel_args;

    if(need_transpose)
    {
        if(problem.IsDirectionBackwardWrW())
        {
            // x (forward input) == tensors.x == problem.GetOut()
            TransposeSolutionDefault2Nhwc trans_x(
                ctx, problem.GetOut().GetType(), out_n, out_c, out_h, out_w);
            result.construction_params.push_back(trans_x.GetKernelInfo());
            trans_kernel_args.push_back(trans_x.GetKernelArg());

            // dy (output gradient) == tensors.dy == problem.GetIn()
            TransposeSolutionDefault2Nhwc trans_dy(
                ctx, problem.GetIn().GetType(), in_n, in_c, in_h, in_w);
            result.construction_params.push_back(trans_dy.GetKernelInfo());
            trans_kernel_args.push_back(trans_dy.GetKernelArg());
        }
        else
        {
            // Input (tensors.in): NCHW -> NHWC
            TransposeSolutionDefault2Nhwc trans_in(
                ctx, problem.GetIn().GetType(), in_n, in_c, in_h, in_w);
            result.construction_params.push_back(trans_in.GetKernelInfo());
            trans_kernel_args.push_back(trans_in.GetKernelArg());

            // Weights: NCHW -> NHWC
            TransposeSolutionDefault2Nhwc trans_wei(
                ctx, problem.GetWeights().GetType(), wei_k, wei_c, wei_y, wei_x);
            result.construction_params.push_back(trans_wei.GetKernelInfo());
            trans_kernel_args.push_back(trans_wei.GetKernelArg());

            // Output (tensors.out): NHWC -> NCHW
            TransposeSolutionNhwc2Default trans_out(
                ctx, problem.GetOut().GetType(), out_n, out_c, out_h, out_w);
            result.construction_params.push_back(trans_out.GetKernelInfo());
            trans_kernel_args.push_back(trans_out.GetKernelArg());
        }
    }

    if(problem.IsDirectionBackwardWrW())
    {
        // fp32 dw (tf32) takes the kernel's fp32 output directly; no cast.
        //
        // fp16/bf16 dw is narrower, so that path stages the fp32 output through
        // a workspace buffer and casts it down. Today fp32 reaches here only
        // via tf32.
        const bool need_cast = !problem.IsFp32();
        const auto cast_sz =
            need_cast ? problem.GetWeights().GetElementSize() * sizeof(float) : size_t{0};
        const auto wt        = GetWorkspaceLayout(problem, cast_sz);
        result.workspace_sz  = wt.GetSize();

        const auto lowp_quant = problem.GetConv().lowp_quant;
        // fp32 intermediate buffer, same shape as the weights.
        const TensorDescriptor cast_desc(
            miopenFloat, problem.GetWeights().GetLengths(), problem.GetWeights().GetStrides());

        const size_t x_sz  = need_transpose ? GetPackedTensorBytes(problem.GetOut()) : size_t{0};
        const size_t dy_sz = need_transpose ? GetPackedTensorBytes(problem.GetIn()) : size_t{0};

        result.invoker_factory =
            [=, trans_args = std::move(trans_kernel_args)](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                    decltype(auto) wrw_ctx =
                        primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                    const auto& tensors       = wrw_ctx.tensors;
                    const auto& workSpace     = wrw_ctx.workSpace;
                    const auto& workSpaceSize = wrw_ctx.workSpaceSize;

                    if(result.workspace_sz > 0 &&
                       (workSpace == nullptr || workSpaceSize < result.workspace_sz))
                        MIOPEN_THROW("ConvHipConv: not enough workspace for wgrad.");

                    const HipEventProfiler profiler(handle);

                    const void* launch_x  = tensors.x;
                    const void* launch_dy = tensors.dy;

                    if(need_transpose)
                    {
                        auto ws_x  = handle.CreateSubBuffer(workSpace, wt.GetOffset(0), x_sz);
                        auto ws_dy = handle.CreateSubBuffer(workSpace, wt.GetOffset(1), dy_sz);

                        // Transpose input (x): NCHW -> NHWC
                        auto args_x = trans_args[0];
                        args_x[0]   = OpKernelArg(ws_x.get());
                        args_x[1]   = OpKernelArg(tensors.x);
                        handle.Run(kernels[0])(args_x);

                        // Transpose output-grad (dy): NCHW -> NHWC
                        auto args_dy = trans_args[1];
                        args_dy[0]   = OpKernelArg(ws_dy.get());
                        args_dy[1]   = OpKernelArg(tensors.dy);
                        handle.Run(kernels[1])(args_dy);

                        launch_x  = ws_x.get();
                        launch_dy = ws_dy.get();
                    }

                    void* dst = need_cast ? static_cast<char*>(workSpace) + wt.GetOffset(3)
                                          : tensors.dw;

                    auto hip_status = hipconv::launch(
                        kernel, par, launch_x, launch_dy, dst, nullptr, handle.GetStream());
                    if(hip_status != hipSuccess)
                        MIOPEN_THROW_HIP_STATUS(hip_status, "ConvHipConv: wgrad launch failed.");

                    if(need_cast)
                    {
                        // ...cast the fp32 workspace -> the weight type.
                        CastTensor(handle,
                                  &lowp_quant,
                                  false,
                                  cast_desc,
                                  static_cast<char*>(workSpace) + wt.GetOffset(3),
                                  tensors.dwDesc,
                                  tensors.dw,
                                  0,
                                  0);
                    }
                };
            };
    }
    else
    {
        // direct_l1 (groups=1 fprop/dgrad) formats its weights into this
        // workspace before the conv; a null pointer faults at a low address.
        const auto hipconv_ws_sz = hipconv::get_workspace_size(kernel, par);
        const auto wt            = GetWorkspaceLayout(problem, hipconv_ws_sz);
        result.workspace_sz      = wt.GetSize();

        const size_t in_sz  = need_transpose ? GetPackedTensorBytes(problem.GetIn()) : size_t{0};
        const size_t wei_sz = need_transpose ? GetPackedTensorBytes(problem.GetWeights()) : size_t{0};
        const size_t out_sz = need_transpose ? GetPackedTensorBytes(problem.GetOut()) : size_t{0};

        result.invoker_factory =
            [=, trans_args = std::move(trans_kernel_args)](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                    decltype(auto) data_ctx =
                        primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                    const auto& tensors = data_ctx.tensors;

                    if(result.workspace_sz > 0 && (data_ctx.workSpace == nullptr ||
                                                    data_ctx.workSpaceSize < result.workspace_sz))
                        MIOPEN_THROW("ConvHipConv: not enough workspace for direct kernel.");

                    const HipEventProfiler profiler(handle);

                    void* hipconv_ws = hipconv_ws_sz > 0
                                           ? static_cast<char*>(data_ctx.workSpace) + wt.GetOffset(3)
                                           : nullptr;

                    if(need_transpose)
                    {
                        auto ws_in  = handle.CreateSubBuffer(data_ctx.workSpace, wt.GetOffset(0), in_sz);
                        auto ws_wei = handle.CreateSubBuffer(data_ctx.workSpace, wt.GetOffset(1), wei_sz);
                        auto ws_out = handle.CreateSubBuffer(data_ctx.workSpace, wt.GetOffset(2), out_sz);

                        // Transpose input: NCHW -> NHWC
                        auto args_in = trans_args[0];
                        args_in[0]   = OpKernelArg(ws_in.get());
                        args_in[1]   = OpKernelArg(tensors.in);
                        handle.Run(kernels[0])(args_in);

                        // Transpose weights: NCHW -> NHWC
                        auto args_wei = trans_args[1];
                        args_wei[0]   = OpKernelArg(ws_wei.get());
                        args_wei[1]   = OpKernelArg(tensors.w);
                        handle.Run(kernels[1])(args_wei);

                        auto hip_status = hipconv::launch(kernel,
                                                          par,
                                                          ws_in.get(),
                                                          ws_wei.get(),
                                                          ws_out.get(),
                                                          hipconv_ws,
                                                          handle.GetStream());
                        if(hip_status != hipSuccess)
                            MIOPEN_THROW_HIP_STATUS(hip_status, "ConvHipConv: direct launch failed.");

                        // Transpose output: NHWC -> NCHW
                        auto args_out = trans_args[2];
                        args_out[0]   = OpKernelArg(tensors.out);
                        args_out[1]   = OpKernelArg(ws_out.get());
                        handle.Run(kernels[2])(args_out);
                    }
                    else
                    {
                        auto hip_status = hipconv::launch(kernel,
                                                          par,
                                                          tensors.in,
                                                          tensors.w,
                                                          tensors.out,
                                                          hipconv_ws,
                                                          handle.GetStream());
                        if(hip_status != hipSuccess)
                            MIOPEN_THROW_HIP_STATUS(hip_status, "ConvHipConv: direct launch failed.");
                    }
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
std::string PerformanceConfigConvHipConv::GetCurrentDeviceName() { return {}; }

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
