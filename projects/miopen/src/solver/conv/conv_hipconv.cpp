#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_ops.hpp>

#include <hipconv/hipconv.hpp>

#include <hip/hip_runtime.h>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_HIPCONV)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

// Always set order=NHWC for hipconv regardless of the problem layout.
// When the problem is NCHW, the caller transposes tensors before/after launch.
//
// hipconv expects parameters in forward convention: n/c/h/w describe the
// forward input, k the forward output channels, p/q the forward output
// spatial dims.  The `direction` field tells hipconv which gradient to
// compute.  MIOpen's ProblemDescription swaps in/out tensors for backward
// passes, so we must use the raw convolution descriptor + un-swapped
// tensor dimensions.
static hipconv::Conv2dParams ToHipconvParams(const ProblemDescription& problem)
{
    hipconv::Conv2dParams par{};

    if(problem.IsDirectionForward())
        par.direction = hipconv::Direction::Fprop;
    else if(problem.IsDirectionBackwardData())
        par.direction = hipconv::Direction::Dgrad;
    else
        par.direction = hipconv::Direction::Wgrad;

    // Forward-convention dimensions regardless of direction.
    // MIOpen swaps in/out tensors for all backward passes:
    //   Fwd: in = x, out = y
    //   Bwd: in = dy, out = dx    (swapped)
    //   WrW: in = dy, out = x     (swapped)
    // hipconv expects forward-convention params with direction telling it what to compute.
    const bool is_swapped = !problem.IsDirectionForward();

    par.n  = problem.GetBatchSize();
    par.c  = is_swapped ? problem.GetOutChannels() : problem.GetInChannels();
    par.h  = is_swapped ? problem.GetOutHeight()   : problem.GetInHeight();
    par.w  = is_swapped ? problem.GetOutWidth()    : problem.GetInWidth();
    par.k  = is_swapped ? problem.GetInChannels()  : problem.GetOutChannels();
    par.kh = problem.GetWeightsHeight();
    par.kw = problem.GetWeightsWidth();

    par.pad_h      = problem.GetPadH();
    par.pad_w      = problem.GetPadW();
    par.stride_h   = problem.GetKernelStrideH();
    par.stride_w   = problem.GetKernelStrideW();
    par.dilation_h = problem.GetDilationH();
    par.dilation_w = problem.GetDilationW();
    par.groups     = problem.GetGroupCount();

    par.p = is_swapped ? problem.GetInHeight()  : problem.GetOutHeight();
    par.q = is_swapped ? problem.GetInWidth()   : problem.GetOutWidth();

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

    if(par.direction == hipconv::Direction::Wgrad)
        par.weight_grad_type = hipconv::DataType::fp32;

    par.order = hipconv::TensorOrder::NHWC;

    return par;
}

static std::optional<hipconv::ArchHandle> GetCurrentArch()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return std::nullopt;
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::nullopt;
    return hipconv::resolve_arch(props.gcnArchName);
}

static std::string MakeKernelId(hipconv::ConvKernelHandle k)
{
    auto desc = hipconv::describe_config(k);
    return std::string(hipconv::name(k)) + "[" + desc + "]";
}

static hipconv::ConvKernelHandle
FindKernelHandle(hipconv::ArchHandle arch,
                 const hipconv::Conv2dParams& par,
                 const std::string& kernel_id)
{
    const auto cfgs = hipconv::get_valid_configs(arch, par);
    for(auto* k : cfgs)
    {
        if(MakeKernelId(k) == kernel_id)
            return k;
    }
    return nullptr;
}

static size_t GetPackedTensorBytes(const TensorDescriptor& td)
{
    return td.GetElementSize() * GetTypeSize(td.GetType());
}

// ===================== PerformanceConfigConvHipConv =====================

static void InitValidKernels(PerformanceConfigConvHipConv& self,
                             hipconv::ArchHandle arch,
                             const ProblemDescription& problem)
{
    const auto par  = ToHipconvParams(problem);
    const auto cfgs = hipconv::get_valid_configs(arch, par);

    self.valid_kernels.clear();
    self.valid_kernels.reserve(cfgs.size());
    for(auto* k : cfgs)
        self.valid_kernels.push_back(MakeKernelId(k));

    if(!self.valid_kernels.empty())
    {
        self.index     = 0;
        self.kernel_id = self.valid_kernels[0];
    }
}

void PerformanceConfigConvHipConv::HeuristicInit(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    const auto arch_opt = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch_opt.has_value())
        return;
    InitValidKernels(*this, *arch_opt, problem);
}

bool PerformanceConfigConvHipConv::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        const auto arch_opt = GetCurrentArch();
        if(!arch_opt.has_value())
            return false;
        InitValidKernels(*this, *arch_opt, problem);
        return true;
    }
    if(index + 1 < static_cast<int>(valid_kernels.size()))
    {
        ++index;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvHipConv::IsValidValue() const { return !kernel_id.empty(); }

bool PerformanceConfigConvHipConv::IsValid(const ExecutionContext& ctx,
                                           const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;

    const auto arch_opt = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch_opt.has_value())
        return false;

    const auto par = ToHipconvParams(problem);
    return FindKernelHandle(*arch_opt, par, kernel_id) != nullptr;
}

bool PerformanceConfigConvHipConv::operator==(const PerformanceConfigConvHipConv& other) const
{
    return kernel_id == other.kernel_id;
}

// ===================== ConvHipConv =====================

bool ConvHipConv::HasArchBackend(std::string_view arch_name)
{
    return hipconv::resolve_arch(arch_name).has_value();
}

bool ConvHipConv::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_HIPCONV))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    const auto arch = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch.has_value())
        return false;

    if(problem.IsDirectionBackwardWrW() && problem.GetConv().attribute.deterministic)
        return false;

    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;

    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;

    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;

    const auto par = ToHipconvParams(problem);
    const auto cfg = hipconv::find_config(*arch, par);
    return cfg.has_value();
}

size_t ConvHipConv::GetWorkspaceSize(const ExecutionContext&,
                                     const ProblemDescription& problem) const
{
    if(problem.IsDirectionBackwardWrW())
    {
        // fp32 cast buffer has the same element count as the weight tensor.
        const auto wgrad_cast_sz =
            problem.GetWeights().GetElementSize() * sizeof(float);

        if(problem.IsLayoutDefault())
        {
            // Wgrad + NCHW buffers, matching the invoker's order:
            //   trans_0 = x  (forward input)  = problem.GetOut()
            //   trans_1 = dy (output gradient) = problem.GetIn()
            //   fp32 dw cast buffer
            MultiBufferWorkspaceTraits wt({GetPackedTensorBytes(problem.GetOut()),
                                           GetPackedTensorBytes(problem.GetIn()),
                                           wgrad_cast_sz});
            return wt.GetSize();
        }
        // Wgrad + NHWC: [fp32_dw]
        return wgrad_cast_sz;
    }

    if(problem.IsLayoutDefault())
    {
        // Fprop/Dgrad + NCHW: [nhwc_in | nhwc_wei | nhwc_out]
        MultiBufferWorkspaceTraits wt({GetPackedTensorBytes(problem.GetIn()),
                                       GetPackedTensorBytes(problem.GetWeights()),
                                       GetPackedTensorBytes(problem.GetOut())});
        return wt.GetSize();
    }

    return 0;
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

    const auto arch_opt = hipconv::resolve_arch(ctx.GetStream().GetDeviceName());
    if(!arch_opt.has_value())
        MIOPEN_THROW("ConvHipConv: unsupported architecture.");

    const auto arch = arch_opt.value();
    const auto par  = ToHipconvParams(problem);

    auto* kernel_handle = FindKernelHandle(arch, par, config.kernel_id);
    if(kernel_handle == nullptr)
        MIOPEN_THROW("ConvHipConv: no matching kernel for the stored config.");

    const bool need_transpose = problem.IsLayoutDefault();
    result.workspace_sz       = GetWorkspaceSize(ctx, problem);

    // Transpose dimensions MUST come from the actual tensor descriptors, not
    // from forward-convention ProblemInterpreter values.  MIOpen swaps the
    // in/out tensors for backward passes, so tensors.in / tensors.out carry
    // different channel/spatial extents than the forward input/output.  Using
    // forward-convention dims here would size a transpose (and its destination
    // buffer) for the wrong tensor and fault.
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

    // Build transpose kernels for NCHW <-> NHWC if needed.  These go into
    // construction_params so MIOpen compiles them; at runtime they are
    // invoked via handle.Run(kernels[idx]).
    //
    // Fprop/Dgrad kernel indices:
    //   0 = trans_in   (NCHW -> NHWC, tensors.in)
    //   1 = trans_wei  (NCHW -> NHWC, weights)
    //   2 = trans_out  (NHWC -> NCHW, tensors.out)
    //
    // Wgrad kernel indices:
    //   0 = trans_x    (NCHW -> NHWC, forward input x  = problem.GetOut())
    //   1 = trans_dy   (NCHW -> NHWC, output grad dy   = problem.GetIn())

    std::vector<std::vector<OpKernelArg>> trans_kernel_args;
    size_t trans_0_size = 0;
    size_t trans_1_size = 0;
    size_t trans_2_size = 0;

    if(need_transpose)
    {
        if(problem.IsDirectionBackwardWrW())
        {
            // x (forward input) == tensors.x == problem.GetOut()
            TransposeSolutionDefault2Nhwc trans_x(
                ctx, problem.GetOut().GetType(), out_n, out_c, out_h, out_w);
            result.construction_params.push_back(trans_x.GetKernelInfo());
            trans_kernel_args.push_back(trans_x.GetKernelArg());
            trans_0_size = trans_x.GetOutputTensorSize();

            // dy (output gradient) == tensors.dy == problem.GetIn()
            TransposeSolutionDefault2Nhwc trans_dy(
                ctx, problem.GetIn().GetType(), in_n, in_c, in_h, in_w);
            result.construction_params.push_back(trans_dy.GetKernelInfo());
            trans_kernel_args.push_back(trans_dy.GetKernelArg());
            trans_1_size = trans_dy.GetOutputTensorSize();
        }
        else
        {
            // Input (tensors.in): NCHW -> NHWC
            TransposeSolutionDefault2Nhwc trans_in(
                ctx, problem.GetIn().GetType(), in_n, in_c, in_h, in_w);
            result.construction_params.push_back(trans_in.GetKernelInfo());
            trans_kernel_args.push_back(trans_in.GetKernelArg());
            trans_0_size = trans_in.GetOutputTensorSize();

            // Weights: NCHW -> NHWC
            TransposeSolutionDefault2Nhwc trans_wei(
                ctx, problem.GetWeights().GetType(), wei_k, wei_c, wei_y, wei_x);
            result.construction_params.push_back(trans_wei.GetKernelInfo());
            trans_kernel_args.push_back(trans_wei.GetKernelArg());
            trans_1_size = trans_wei.GetOutputTensorSize();

            // Output (tensors.out): NHWC -> NCHW
            TransposeSolutionNhwc2Default trans_out(
                ctx, problem.GetOut().GetType(), out_n, out_c, out_h, out_w);
            result.construction_params.push_back(trans_out.GetKernelInfo());
            trans_kernel_args.push_back(trans_out.GetKernelArg());
            trans_2_size = trans_out.GetOutputTensorSize();
        }
    }

    if(problem.IsDirectionBackwardWrW())
    {
        const auto lowp_quant = problem.GetConv().lowp_quant;
        const auto wei_desc   = problem.GetWeights();

        const TensorDescriptor cast_desc(
            miopenFloat, wei_desc.GetLengths(), wei_desc.GetStrides());

        result.invoker_factory =
            [=, trans_args = std::move(trans_kernel_args)](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                    decltype(auto) wrw_ctx =
                        primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                    const auto& tensors       = wrw_ctx.tensors;
                    const auto& workSpace     = wrw_ctx.workSpace;
                    const auto& workSpaceSize = wrw_ctx.workSpaceSize;

                    if(workSpace == nullptr || workSpaceSize < result.workspace_sz)
                        MIOPEN_THROW("Not enough workspace for ConvHipConv wgrad.");

                    const HipEventProfiler profiler(handle);

                    const void* launch_x  = tensors.x;
                    const void* launch_dy = tensors.dy;
                    void* cast_buf        = workSpace;

                    if(need_transpose)
                    {
                        const auto wgrad_cast_sz =
                            wei_desc.GetElementSize() * sizeof(float);
                        MultiBufferWorkspaceTraits wt(
                            {trans_0_size, trans_1_size, wgrad_cast_sz});
                        auto ws_x  = handle.CreateSubBuffer(workSpace, wt.GetOffset(0), trans_0_size);
                        auto ws_dy = handle.CreateSubBuffer(workSpace, wt.GetOffset(1), trans_1_size);
                        cast_buf   = static_cast<char*>(workSpace) + wt.GetOffset(2);

                        // Transpose input (x): NCHW -> NHWC
                        auto args_x     = trans_args[0];
                        args_x[0]       = OpKernelArg(ws_x.get());
                        args_x[1]       = OpKernelArg(tensors.x);
                        handle.Run(kernels[0])(args_x);

                        // Transpose output-grad (dy): NCHW -> NHWC
                        auto args_dy    = trans_args[1];
                        args_dy[0]      = OpKernelArg(ws_dy.get());
                        args_dy[1]      = OpKernelArg(tensors.dy);
                        handle.Run(kernels[1])(args_dy);

                        launch_x  = ws_x.get();
                        launch_dy = ws_dy.get();
                    }

                    auto hip_status = hipconv::launch(kernel_handle,
                                                      par,
                                                      launch_x,
                                                      launch_dy,
                                                      cast_buf,
                                                      nullptr,
                                                      handle.GetStream());
                    if(hip_status != hipSuccess)
                        MIOPEN_THROW("ConvHipConv wgrad launch failed.");

                    CastTensor(handle,
                               &lowp_quant,
                               false,
                               cast_desc,
                               cast_buf,
                               tensors.dwDesc,
                               tensors.dw,
                               0,
                               0);
                };
            };
    }
    else
    {
        result.invoker_factory =
            [=, trans_args = std::move(trans_kernel_args)](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                    decltype(auto) data_ctx =
                        primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                    const auto& tensors = data_ctx.tensors;

                    const HipEventProfiler profiler(handle);

                    if(need_transpose)
                    {
                        const auto& workSpace     = data_ctx.workSpace;
                        const auto& workSpaceSize = data_ctx.workSpaceSize;

                        if(workSpace == nullptr || workSpaceSize < result.workspace_sz)
                            MIOPEN_THROW("Not enough workspace for ConvHipConv transpose.");

                        MultiBufferWorkspaceTraits wt(
                            {trans_0_size, trans_1_size, trans_2_size});

                        auto ws_in  = handle.CreateSubBuffer(workSpace, wt.GetOffset(0), trans_0_size);
                        auto ws_wei = handle.CreateSubBuffer(workSpace, wt.GetOffset(1), trans_1_size);
                        auto ws_out = handle.CreateSubBuffer(workSpace, wt.GetOffset(2), trans_2_size);

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

                        auto hip_status = hipconv::launch(kernel_handle,
                                                          par,
                                                          ws_in.get(),
                                                          ws_wei.get(),
                                                          ws_out.get(),
                                                          nullptr,
                                                          handle.GetStream());
                        if(hip_status != hipSuccess)
                            MIOPEN_THROW("ConvHipConv launch failed.");

                        // Transpose output: NHWC -> NCHW
                        auto args_out = trans_args[2];
                        args_out[0]   = OpKernelArg(tensors.out);
                        args_out[1]   = OpKernelArg(ws_out.get());
                        handle.Run(kernels[2])(args_out);
                    }
                    else
                    {
                        auto hip_status = hipconv::launch(kernel_handle,
                                                          par,
                                                          tensors.in,
                                                          tensors.w,
                                                          tensors.out,
                                                          nullptr,
                                                          handle.GetStream());
                        if(hip_status != hipSuccess)
                            MIOPEN_THROW("ConvHipConv launch failed.");
                    }
                };
            };
    }

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
