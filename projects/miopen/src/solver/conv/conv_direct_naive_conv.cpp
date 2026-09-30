/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "miopen/env.hpp"
#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/datatype.hpp>
#include <miopen/solver/zero_tensor.hpp>
#include <ostream>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_USE_PACKED_KERNELS);

// Absolute runtime override (in MACs) for the naive-conv work-size gate. 0 (unset)
// => derive the limit from the device's capabilities; set to a very large value to
// effectively disable the gate. Wins over the time budget below.
// See ConvDirectNaiveConvWorkLimit.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK);

// Wall-time budget (in milliseconds) that one naive launch may occupy. This is the
// quantity the gate actually cares about -- it is converted to a MAC budget using
// the live device's throughput. 0 (unset) => NAIVE_CONV_DEFAULT_BUDGET_MS. Lower it
// on a short-watchdog OS, raise it when the watchdog is disabled or TdrDelay was
// extended. Ignored when MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK is set.
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS);

namespace miopen {

namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

// Block size for naive conv kernels. Passed to kernels via -DNAIVE_CONV_BLOCK_SIZE
// so shared memory arrays match the launch configuration.
constexpr size_t NAIVE_CONV_BLOCK_SIZE = 256;

// Minimum spatial work (n * ho * wo, or n * do * ho * wo for 3D) before WRW
// enables cross-block tiling with atomicAdd. Below this threshold, a single
// block per filter position is sufficient and avoids the cost of zeroing the
// weight buffer and atomic contention. 262144 = 1024 * 256, roughly the point
// where one block can no longer cover the spatial dimension in its thread-loop.
constexpr size_t WRW_SPATIAL_TILING_THRESHOLD = 262144;

// The fp8 naive conv path lives in a separate source file. The BWD-data block-size
// heuristic special-cases it (see NaiveConv2DBWDBlockSize), and the invoker uses it
// to select the fp8 kernel's distinct argument list.
constexpr const char* FP8_NAIVE_CONV_KERNEL_FILE = "fp8_naive_conv.cpp";

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext& ctx,
                                         const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return (device_name == "gfx906" || device_name == "gfx908") && ctx.rmv.IsV3() &&
           problem.IsLayoutDefault() && (problem.IsFp16() || problem.IsFp32() || problem.IsBfp16());
}

// Check tensor data type respectively
bool IsInputFp32(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetOutDataType() == miopenFloat &&
            problem.GetWeightsDataType() == miopenFloat) ||
           (problem.GetInDataType() == miopenFloat && problem.GetOutDataType() == miopenFloat);
}

bool IsInputFp16(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetOutDataType() == miopenHalf && problem.GetWeightsDataType() == miopenHalf) ||
           (problem.GetInDataType() == miopenHalf && problem.GetOutDataType() == miopenHalf);
}

bool IsInputBfp16(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetOutDataType() == miopenBFloat16 &&
            problem.GetWeightsDataType() == miopenBFloat16) ||
           (problem.GetInDataType() == miopenBFloat16 &&
            problem.GetOutDataType() == miopenBFloat16);
}

bool IsInputInt8(const ProblemDescription& problem)
{
    return (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetOutDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8) ||
           (problem.GetInDataType() == miopenInt8 && problem.GetOutDataType() == miopenInt8);
}

bool IsAccInt32(const ProblemDescription& problem) { return IsInputInt8(problem); }

bool IsOutputFp32(const ProblemDescription& problem)
{
    return problem.IsFp32() ||
           (problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
            problem.GetOutDataType() == miopenFloat);
}

bool IsOutputFp16(const ProblemDescription& problem) { return problem.IsFp16(); }

bool IsOutputBfp16(const ProblemDescription& problem) { return problem.IsBfp16(); }

bool IsFp8Kernel(const ProblemDescription& problem)
{
    return problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8();
}

bool IsOutputInt8(const ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt8;
}

bool IsOutputInt32(const ProblemDescription& problem)
{
    return problem.GetInDataType() == miopenInt8 && problem.GetWeightsDataType() == miopenInt8 &&
           problem.GetOutDataType() == miopenInt32;
}

std::string ConvDirectNaiveConvKernelName(const ProblemDescription& problem)
{
    std::ostringstream kernel_name;

    /// \todo remove packed reference convolution kernels --amberhassaan
#ifndef NDEBUG // enable in debug mode only
    if(env::enabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_USE_PACKED_KERNELS))
    {
        kernel_name << "naive_conv_ab_packed_";
    }
    else
#endif
    {
        kernel_name << "naive_conv_ab_nonpacked_";
    }

    // NOLINTBEGIN(*-braces-around-statements)
    if(problem.IsDirectionForward())
        kernel_name << "fwd_";
    else if(problem.IsDirectionBackwardData())
        kernel_name << "bwd_";
    else if(problem.IsDirectionBackwardWrW())
        kernel_name << "wrw_";
    else
        MIOPEN_THROW("unsupported convolution direction");
    // NOLINTEND(*-braces-around-statements)

    if(problem.IsLayoutDefault())
    {
        if(problem.Is2d())
            kernel_name << "nchw_";
        else
            kernel_name << "ncdhw_";
    }
    else if(problem.IsLayoutNHWC())
    {
        if(problem.Is2d())
            kernel_name << "nhwc_";
        else
            kernel_name << "ndhwc_";
    }
    else
    {
        MIOPEN_THROW("unsupported tensor layout");
    }

    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
    {
        kernel_name << miopen::GetDataType(ProblemInterpreter::GetInputDataType(problem));
        kernel_name << "_" << miopen::GetDataType(problem.GetWeightsDataType());
        kernel_name << "_" << miopen::GetDataType(ProblemInterpreter::GetOutputDataType(problem));
        return kernel_name.str();
    }
    else if(IsInputFp32(problem))
    {
        kernel_name << "float_";
    }
    else if(IsInputFp16(problem))
    {
        kernel_name << "half_";
    }
    else if(IsInputBfp16(problem))
    {
        kernel_name << "hip_bfloat16_";
    }
    else if(IsInputInt8(problem))
    {
        kernel_name << "int8_t_";
    }
    else
    {
        MIOPEN_THROW("unsupported data type:");
    }

    if(IsAccInt32(problem))
        kernel_name << "int32_t_";
    else if(IsInputFp32(problem) || IsInputFp16(problem) || IsInputBfp16(problem))
        kernel_name << "float_";
    else
        MIOPEN_THROW("unsupported data type:");

    // NOLINTBEGIN(*-braces-around-statements)
    if(IsOutputFp32(problem))
        kernel_name << "float";
    else if(IsOutputFp16(problem))
        kernel_name << "half";
    else if(IsOutputBfp16(problem))
        kernel_name << "hip_bfloat16";
    else if(IsOutputInt8(problem))
        kernel_name << "int8_t";
    else if(IsOutputInt32(problem))
        kernel_name << "int32_t";
    else
        MIOPEN_THROW("unsupported data type:");
    // NOLINTEND(*-braces-around-statements)

    // only float support tf32
    bool use_tf32 = (IsInputFp32(problem) && IsOutputFp32(problem) && problem.UseTF32());
    kernel_name << "_" << static_cast<int>(use_tf32);

    return kernel_name.str();
}

std::string ConvDirectNaiveConvKernelFile(const ExecutionContext& ctx,
                                          const ProblemDescription& problem)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    // The above function, ConvDirectNaiveConvKernelName is not in sync for the asm kernel,
    // resulting in empty code objects. This happens for systems with COv3 as the default type.
    // if(device_name == "gfx906" || device_name == "gfx908")
    // {
    //     if(ctx.rmv.IsV3() && problem.IsLayoutDefault() && !problem.IsFp8() &&
    //        !problem.IsTensorsCasted() && !problem.IsBfp8())
    //         return "naive_conv_gcn.s";
    // }
    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
        return FP8_NAIVE_CONV_KERNEL_FILE;
    if(problem.IsDirectionForward())
        return "naive_conv_fwd.cpp";
    if(problem.IsDirectionBackwardData())
        return "naive_conv_bwd.cpp";
    return "naive_conv_wrw.cpp";
}

std::string ConvDirectNaiveConvCompileOption(const ExecutionContext& ctx,
                                             const ProblemDescription& problem)
{
    fs::path filename = ConvDirectNaiveConvKernelFile(ctx, problem);
    if(filename.extension() == ".s")
    {
        std::ostringstream options;
        GenerateClangDefsym(options, "ROCM_METADATA_VERSION", 5);
        return options.str();
    }
    std::ostringstream ss;
    ss << ctx.general_compile_options;
    if(problem.IsFp8() || problem.IsTensorsCasted() || problem.IsBfp8())
    {
        ss << " -DINPUT_TYPE="
           << miopen::GetDataType(ProblemInterpreter::GetInputDataType(problem));
        ss << " -DWEIGHTS_TYPE=" << miopen::GetDataType(problem.GetWeightsDataType());
        ss << " -DOUTPUT_TYPE="
           << miopen::GetDataType(ProblemInterpreter::GetOutputDataType(problem));
        const auto in_cast_type = ProblemInterpreter::GetInputCastType(problem);
        if(in_cast_type)
            ss << " -DINPUT_CAST_TYPE=" << miopen::GetDataType(*in_cast_type);
        const auto wei_cast_type = problem.GetWeightsCastType();
        if(wei_cast_type)
            ss << " -DWEIGHTS_CAST_TYPE=" << miopen::GetDataType(*wei_cast_type);
        const auto out_cast_type = ProblemInterpreter::GetOutputCastType(problem);
        if(out_cast_type)
            ss << " -DOUTPUT_CAST_TYPE=" << miopen::GetDataType(*out_cast_type);
        ss << " -DMIOPEN_FP8_CLIPPING=" << MIOPEN_FP8_CLIPPING;
        ss << " -DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" << MIOPEN_FP8_IEEE_EXPONENT_BIAS;
        //     Let the kernel choose its accumulator (double for naive kernels )
    }

    ss << " -DNAIVE_CONV_BLOCK_SIZE=" << NAIVE_CONV_BLOCK_SIZE;

    return ss.str();
}

bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem)
{
    if(ConvDirectNaiveConvIsAssemblyKernel(ctx, problem))
    {
        if(!ctx.use_asm_kernels)
            return false;
    }
    else
    {
        if(!ctx.use_hip_kernels)
            return false;
    }
    return true;
}

// The naive kernel is un-tiled and saturates the GPU, so its wall-time is
// proportional to the total MAC count: N * K * C_per_group * output_spatial_volume *
// filter_volume. On very large problems (e.g. VAE-decode / 3D shapes) a single naive
// launch runs for multiple seconds; when MIOpen *benchmarks* candidate solvers during
// Find, or immediate mode ranks one without measuring it, executing naive trips the
// OS GPU watchdog (TDR) -- even when a fast solver also applies and would ultimately
// win. Naive stays *applicable* at any size so it can serve as the universal
// fallback; the selection layers (EvaluateInvokers and GetSolutionsFallback) consume
// this threshold to drop naive when it is over the limit AND a non-naive alternative
// also applies. When naive is the sole applicable solver it still runs, so a shape
// only naive can serve is served (and, if huge, may TDR on a short-watchdog OS -- an
// honest "extend coverage here" signal, not masked).
size_t ConvDirectNaiveConvWork(const ProblemDescription& problem)
{
    const size_t n           = problem.GetInBatchSize();
    const size_t k           = problem.GetOutChannels();
    const auto group         = static_cast<size_t>(problem.GetGroupCount());
    const size_t c_per_group = (group == 0) ? problem.GetInChannels() //
                                            : problem.GetInChannels() / group;

    const size_t out_vol = problem.GetOutWidth() * problem.GetOutHeight() *
                           (problem.Is2d() ? size_t{1} : problem.GetOutDepth());
    const size_t fil_vol = problem.GetWeightsWidth() * problem.GetWeightsHeight() *
                           (problem.Is2d() ? size_t{1} : problem.GetWeightsDepth());

    return n * k * c_per_group * out_vol * fil_vol;
}

// Sustained MACs the *naive* kernel retires per lane per 1000 clocks. The naive inner
// loop is a single FMA fed by two un-cached global loads, with no LDS staging and no
// register blocking, so it is memory-latency bound and runs far below the
// one-FMA-per-lane-per-clock issue peak (which in these units would be 1000).
//
// Calibrated against two measured sweeps totalling 109 timed shape/direction points:
// gfx1151 (RDNA3.5, 80 hw CUs, wave32, 2.9 GHz) and gfx950 (CDNA4, 256 hw CUs, wave64,
// 2.4 GHz). Over the 39 shape/direction pairs measured on both, gfx950 ran a median
// 5.04x faster on shapes >= 5 GMAC against a 5.30x lane-cycle ratio -- so the linear
// device scaling this model assumes holds to within ~5% across a 5x spread of part size
// and across both wavefront widths. That scaling, not the constant, is the load-bearing
// claim; it is what lets one number cover CDNA and RDNA.
//
// 3 is the conservative end of the measured spread rather than its centre: implied rates
// centre on a median of 11 on gfx1151 and 7 on gfx950, so the derived limit fires well
// before naive reaches the budget on a typical shape. It also keeps the derived limits
// in the same magnitude as the fixed 16 GMAC this model replaces, so the change
// contributes device *scaling* without simultaneously moving the absolute calibration.
// Too high risks a TDR; too low over-gates naive on shapes it would have won (the class
// of regression #9513 was filed against).
//
// Known weakness, deliberately not addressed here: MAC count is not occupancy-aware.
// Naive's grid is group * n * k_per_group blocks wide, so a shape whose (n, k_per_group)
// product is below the part's CU count under-fills it and runs far under the modeled
// throughput. Every measured point that fell short of the model is of this form -- 3D
// convolutions at n=1, k=64 on a 256-CU part -- and one of them (32x128x128, bwd,
// 58 GMAC) reached 1256 ms against the 1000 ms budget. Such shapes are *under*-gated:
// safe for coverage, unsafe for TDR. Note that measurement predates the intra-grid
// spatial tiling added in #7529, which widens exactly these grids.
//
// To re-measure: force naive with MIOPEN_DEBUG_FIND_ONLY_SOLVER, pin
// MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK high so the gate does not skip what you are
// timing, and run MIOpenDriver -t 1 over a range of shapes. Each shape then implies
// work_macs * 1000 / (hw_cus * wavefront * clock_khz * measured_ms), which is this
// constant. Note hw_cus is the doubled value from Handle::GetMaxHardwareComputeUnits()
// on gfx1*. The spread of that quantity across shapes also indicates whether MAC count
// is predictive at all, which is the assumption the whole gate rests on.
constexpr size_t NAIVE_MAC_PER_1K_LANE_CYCLES = 3;

// Wall-time a single naive launch may occupy, default and ceiling. Windows' default
// TdrDelay is 2 s, so 1 s leaves 2x margin. The ceiling exists only to keep an
// outlandish env value from overflowing the multiply below.
constexpr size_t NAIVE_CONV_DEFAULT_BUDGET_MS = 1000;
constexpr size_t NAIVE_CONV_MAX_BUDGET_MS     = 60 * 1000;

// Limit used when the device cannot report the capabilities the model needs. This is
// the fixed pre-device-aware default; it is conservative for every current part.
constexpr size_t NAIVE_CONV_FALLBACK_MAX_WORK =
    static_cast<size_t>(16) * 1000 * 1000 * 1000; // ~16 GMAC

// Convert a wall-time budget into a MAC budget using the live device's throughput.
// What we need to bound is the *duration* of one naive launch -- on a short-watchdog
// OS, a launch that outlives the watchdog is a GPU reset, and by the time it fires it
// is far too late to recover. We cannot measure the launch without running it, so we
// bound the work instead:
//
//   lane_cycles/ms = hw_cus * wavefront_width * clock_khz
//   naive_macs/ms  = lane_cycles/ms * NAIVE_MAC_PER_1K_LANE_CYCLES / 1000
//   limit (MACs)   = naive_macs/ms * budget_ms
//
// Clock arrives in kHz, which is already "cycles per ms", so the whole computation
// stays in integer math with no unit juggling. Every term but the throughput constant
// is read from the device, so the limit tracks the part actually running the kernel --
// a 304-CU MI300X earns ~123 GMAC at the default budget while a 32-WGP iGPU earns
// ~8.9 GMAC, with no per-arch table to hand-maintain.
//
// Takes capabilities as plain integers rather than a Handle so the model is unit
// testable on the CPU against synthetic devices.
size_t ConvDirectNaiveConvWorkLimit(size_t hw_cus, size_t wavefront_width, size_t clock_khz)
{
    const size_t absolute_override = env::value(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_WORK);
    if(absolute_override != 0)
        return absolute_override;

    // A device that cannot report its own capabilities gives the model nothing to
    // scale from. Fall back to the fixed limit rather than deriving a nonsense
    // (possibly zero) budget that would gate every shape.
    if(hw_cus == 0 || wavefront_width == 0 || clock_khz == 0)
    {
        MIOPEN_LOG_I2("ConvDirectNaiveConv work limit: incomplete device capabilities (cus "
                      << hw_cus << ", wavefront " << wavefront_width << ", clock " << clock_khz
                      << " kHz); using fixed fallback " << NAIVE_CONV_FALLBACK_MAX_WORK << " MACs");
        return NAIVE_CONV_FALLBACK_MAX_WORK;
    }

    size_t budget_ms = env::value(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_MAX_TIME_MS);
    if(budget_ms == 0)
        budget_ms = NAIVE_CONV_DEFAULT_BUDGET_MS;
    budget_ms = std::min(budget_ms, NAIVE_CONV_MAX_BUDGET_MS);

    // clock_khz is cycles/ms, so this product is lane-cycles/ms directly.
    const size_t macs_per_ms =
        hw_cus * wavefront_width * clock_khz * NAIVE_MAC_PER_1K_LANE_CYCLES / 1000;
    return macs_per_ms * budget_ms;
}

bool ConvDirectNaiveConvExceedsWorkLimit(const ExecutionContext& ctx,
                                         const ProblemDescription& problem)
{
    const auto& handle = ctx.GetStream();
    const size_t work  = ConvDirectNaiveConvWork(problem);
    const size_t limit = ConvDirectNaiveConvWorkLimit(
        handle.GetMaxHardwareComputeUnits(), handle.GetWavefrontWidth(), handle.GetClockRateKhz());

    if(work > limit)
    {
        MIOPEN_LOG_I2("ConvDirectNaiveConv over work limit: work "
                      << work << " MACs exceeds limit " << limit << " for "
                      << handle.GetDeviceName()
                      << " -- defer from selection when an alternative applies (avoids "
                         "TDR-prone naive launch)");
        return true;
    }
    return false;
}

/// Figure out the index of C (channel) stride so we can expand it into
/// (G, C_per_group). Return value G_stride_idx is the position of G stride
/// in the stride vector, such that the (G_stride_idx - 1) is the index that
/// contains C's stride as a multiplying factor
int conv_internal::GetGroupStrideIndex(const ProblemDescription& problem)
{
    int G_stride_idx = -1;
    if(problem.IsLayoutDefault())
    {
        G_stride_idx = 1;
    }
    else
    {
        assert(problem.IsLayoutNHWC());
        assert(problem.Is2d() || problem.Is3d());
        //
        // G_stride_idx = problem.Is2d() ? 3 : 4;
        // For NHWC, MIOpen stores strides in NCHW order, so we are interested in 1 + W's
        // stride as that will be the value of G_stride_idx;
        G_stride_idx = problem.Is2d() ? 4 : 5;
    }
    assert(G_stride_idx != -1);
    return G_stride_idx;
}

void conv_internal::DebugPrintTensorStrides(const TensorDescriptor& inDesc,
                                            const TensorDescriptor& wDesc,
                                            const TensorDescriptor& outDesc)
{

    auto printOneStrideVec = [](const char* name, const auto& vec) {
        MIOPEN_LOG_I(name << " = [");
        for(const size_t v : vec)
        {
            MIOPEN_LOG_I(v << ",");
        }
        MIOPEN_LOG_I("]\n");
    };

    printOneStrideVec("inDesc = ", inDesc.GetStrides());
    printOneStrideVec("wDesc = ", wDesc.GetStrides());
    printOneStrideVec("outDesc = ", outDesc.GetStrides());
}

// Maximum grid size to avoid uint32_t overflow in hipExtModuleLaunchKernel.
// The HIP API uses uint32_t for grid dimensions, so we must ensure:
// grid_size * block_size < 2^32
// max_grid_size = 2^32 / block_size = 4,294,967,296 / 256 = 16,777,216
constexpr size_t MAX_GRID_SIZE = static_cast<size_t>(16) * 1024 * 1024; // 16M work groups max

// Chooses the workgroup size for the 2D naive conv BWD-data launch.
inline size_t
NaiveConv2DBWDBlockSize(const Handle& handle, const std::string& kernel_file, size_t grid_size)
{
    constexpr size_t default_block = 256;

    // The 1024 path is only valid when the launch derives its spatial tiling from this block size.
    // The per-direction naive_conv_bwd.cpp does: num_spatial_tiles = ceil(thread_length/block_size)
    // and covers the domain with an `if(tid < thread_length)` guard, so any block size stays
    // correct. fp8_naive_conv.cpp instead hardcodes `tid += 256` in its own loop, so it must stay
    // at the default regardless of grid size.
    if(kernel_file == FP8_NAIVE_CONV_KERNEL_FILE)
        return default_block;

    // 1024-thread work groups while the grid is too small to fill the GPU. The launch fixes the
    // work-group count (in the grid.x dimension) at n*hi (NHWC) / n*c (NCHW); a wider group never
    // reaches more CUs -- it only puts more resident waves on the ones it does reach, which is what
    // hides this memory-bound kernel's latency. Worth 1.3x-2.1x geomean, measured on gfx90a, gfx942
    // and gfx950 only.
    //
    // NOTE: the 5*CU crossover below was tuned upstream against a gridDim.y=1 grid-stride launch.
    // This branch's BWD-d kernel instead spreads the spatial domain across gridDim.y tiles, so the
    // effective occupancy is grid_size * num_spatial_tiles. The heuristic is preserved as-is as a
    // known-good no-regression starting point, but the crossover should be re-validated on CDNA CI
    // against this launch geometry before being treated as optimal.
    constexpr size_t large_block = 1024;
    const auto device_name       = handle.GetDeviceName();
    if(!(StartsWith(device_name, "gfx90a") || StartsWith(device_name, "gfx942") ||
         StartsWith(device_name, "gfx950")))
        return default_block;

    // Measured crossover: at or above this many work groups per CU, 256 is the faster of the two.
    constexpr size_t large_block_max_blocks_per_cu = 5;
    if(grid_size >= large_block_max_blocks_per_cu * handle.GetMaxComputeUnits())
        return default_block;

    MIOPEN_LOG_I2("naive conv bwd block size: " << large_block << " (grid_size=" << grid_size
                                                << ")");
    return large_block;
}

// Helper function to calculate batch chunk size to prevent grid size overflow.
// Keeps the original if/else structure for layout handling.
// Parameters are passed by reference for speed.
inline void
CalculateBatchChunkSize(size_t grid_size_per_batch, int n, int& batch_chunk_size, size_t& grid_size)
{
    batch_chunk_size = static_cast<int>(MAX_GRID_SIZE / grid_size_per_batch);
    if(batch_chunk_size < 1)
        batch_chunk_size = 1;
    if(batch_chunk_size > n)
        batch_chunk_size = n;

    grid_size = grid_size_per_batch * batch_chunk_size;
}

// Helper function to prepare batch iteration for chunked kernel execution.
// Calculates offsets, pointers, grid size and updates kernel configuration.
// Parameters are passed by reference for speed.
inline void PrepareBatchedKernelRun(int batch_start,
                                    int batch_chunk_size,
                                    int n,
                                    size_t in_batch_stride,
                                    size_t out_batch_stride,
                                    size_t in_type_size,
                                    size_t out_type_size,
                                    size_t grid_size_per_batch,
                                    size_t block_size,
                                    const void* in_base,
                                    void* out_base,
                                    Kernel& kern_copy,
                                    const void*& in_ptr,
                                    void*& out_ptr,
                                    int& current_batch_size)
{
    current_batch_size = std::min(batch_chunk_size, n - batch_start);

    // Calculate byte offsets for input and output tensors
    size_t in_offset_bytes  = static_cast<size_t>(batch_start) * in_batch_stride * in_type_size;
    size_t out_offset_bytes = static_cast<size_t>(batch_start) * out_batch_stride * out_type_size;

    // Cast to char* for byte-level pointer arithmetic
    in_ptr  = static_cast<const char*>(in_base) + in_offset_bytes;
    out_ptr = static_cast<char*>(out_base) + out_offset_bytes;

    // Calculate and set grid size for this chunk
    size_t current_grid_size = grid_size_per_batch * current_batch_size;
    kern_copy.gdims[0]       = current_grid_size * block_size;
}

namespace conv_internal {
::miopen::solver::ConvSolution
GetConv2DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size          = NAIVE_CONV_BLOCK_SIZE;
    int batch_chunk_size       = 1;
    size_t grid_size_per_batch = 1;
    size_t grid_size           = 1;
    size_t thread_length       = 1;
    bool is_layout_default     = problem.IsLayoutDefault();

    if(is_layout_default)
    {
        grid_size_per_batch = static_cast<size_t>(k);
        thread_length       = static_cast<size_t>(ho) * wo;
        CalculateBatchChunkSize(grid_size_per_batch, n, batch_chunk_size, grid_size);
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size_per_batch = static_cast<size_t>(ho);
        thread_length       = static_cast<size_t>(wo) * k;
        CalculateBatchChunkSize(grid_size_per_batch, n, batch_chunk_size, grid_size);
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }
    // Spatial tiling: only tile when grid_size is too small to keep the GPU busy.
    // When grid_size is already large enough (e.g., NHWC with ho blocks), tiling
    // adds overhead without benefit. Not implemented for fp8 kernels.
    size_t num_spatial_tiles = 1;
    if(!IsFp8Kernel(problem) && grid_size < 32)
        num_spatial_tiles = (thread_length + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == FP8_NAIVE_CONV_KERNEL_FILE);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    // Capture tensor element sizes for offset calculation
    const auto in_type_size  = GetTypeSize(problem.GetInDataType());
    const auto out_type_size = GetTypeSize(problem.GetOutDataType());

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx =
                primitive_parameters.CastTo<::miopen::conv::DataInvokeParams>();
            const auto& tensors = data_ctx.tensors;
            float elapsed       = 0;
            auto in_strides     = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));

            // Get batch strides for offset calculation
            const auto& orig_in_strides  = tensors.inDesc.GetStrides();
            const auto& orig_out_strides = tensors.outDesc.GetStrides();
            size_t in_batch_stride       = orig_in_strides[0];
            size_t out_batch_stride      = orig_out_strides[0];

            if(is_f8)
            {
                // FP8 path: process batches in chunks
                for(int batch_start = 0; batch_start < n; batch_start += batch_chunk_size)
                {
                    const void* in_ptr = nullptr;
                    void* out_ptr      = nullptr;
                    int current_batch_size;
                    auto kern_copy = kern;

                    PrepareBatchedKernelRun(batch_start,
                                            batch_chunk_size,
                                            n,
                                            in_batch_stride,
                                            out_batch_stride,
                                            in_type_size,
                                            out_type_size,
                                            grid_size_per_batch,
                                            block_size,
                                            tensors.in,
                                            tensors.out,
                                            kern_copy,
                                            in_ptr,
                                            out_ptr,
                                            current_batch_size);

                    handle.Run(kern_copy)(in_ptr,
                                          tensors.w,
                                          out_ptr,
                                          in_strides,
                                          wei_strides,
                                          out_strides,
                                          hi,
                                          wi,
                                          current_batch_size,
                                          k_per_group,
                                          c_per_group,
                                          ho,
                                          wo,
                                          sy,
                                          sx,
                                          dy,
                                          dx,
                                          py,
                                          px,
                                          fy,
                                          fx,
                                          group,
                                          problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                              miopenF8RoundingModeStochastic,
                                          problem.GetConv().attribute.fp8rounding_mode.GetSeed());

                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
            }
            else
            {
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();

                // Process batches in chunks to avoid exceeding GPU grid limits
                for(int batch_start = 0; batch_start < n; batch_start += batch_chunk_size)
                {
                    const void* in_ptr = nullptr;
                    void* out_ptr      = nullptr;
                    int current_batch_size;
                    auto kern_copy = kern;

                    PrepareBatchedKernelRun(batch_start,
                                            batch_chunk_size,
                                            n,
                                            in_batch_stride,
                                            out_batch_stride,
                                            in_type_size,
                                            out_type_size,
                                            grid_size_per_batch,
                                            block_size,
                                            tensors.in,
                                            tensors.out,
                                            kern_copy,
                                            in_ptr,
                                            out_ptr,
                                            current_batch_size);

                    handle.Run(kern_copy)(in_ptr,
                                          tensors.w,
                                          alpha_val,
                                          beta_val,
                                          out_ptr,
                                          in_strides,
                                          wei_strides,
                                          out_strides,
                                          hi,
                                          wi,
                                          current_batch_size,
                                          k_per_group,
                                          c_per_group,
                                          ho,
                                          wo,
                                          sy,
                                          sx,
                                          dy,
                                          dx,
                                          py,
                                          px,
                                          fy,
                                          fx,
                                          group);

                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

::miopen::solver::ConvSolution
GetConv3DFWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size          = NAIVE_CONV_BLOCK_SIZE;
    int batch_chunk_size       = 1;
    size_t grid_size_per_batch = 1;
    size_t grid_size           = 1;
    size_t thread_length       = 1;
    bool is_layout_default     = problem.IsLayoutDefault();

    if(is_layout_default)
    {
        grid_size_per_batch = static_cast<size_t>(k);
        thread_length       = static_cast<size_t>(do_) * ho * wo;
        CalculateBatchChunkSize(grid_size_per_batch, n, batch_chunk_size, grid_size);
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size_per_batch = static_cast<size_t>(group) * do_;
        thread_length       = static_cast<size_t>(ho) * wo * k_per_group;
        CalculateBatchChunkSize(grid_size_per_batch, n, batch_chunk_size, grid_size);
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }
    size_t num_spatial_tiles = 1;
    if(!IsFp8Kernel(problem) && grid_size < 32)
        num_spatial_tiles = (thread_length + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    // Capture tensor element sizes for offset calculation
    const auto in_type_size  = GetTypeSize(problem.GetInDataType());
    const auto out_type_size = GetTypeSize(problem.GetOutDataType());

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx =
                primitive_parameters.CastTo<::miopen::conv::DataInvokeParams>();
            const auto& tensors = data_ctx.tensors;
            float elapsed       = 0;
            auto in_strides     = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));

            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();

            // Get batch strides for offset calculation
            const auto& orig_in_strides  = tensors.inDesc.GetStrides();
            const auto& orig_out_strides = tensors.outDesc.GetStrides();
            size_t in_batch_stride       = orig_in_strides[0];
            size_t out_batch_stride      = orig_out_strides[0];

            // Process batches in chunks to avoid exceeding GPU grid limits
            for(int batch_start = 0; batch_start < n; batch_start += batch_chunk_size)
            {
                const void* in_ptr = nullptr;
                void* out_ptr      = nullptr;
                int current_batch_size;
                auto kern_copy = kern;

                PrepareBatchedKernelRun(batch_start,
                                        batch_chunk_size,
                                        n,
                                        in_batch_stride,
                                        out_batch_stride,
                                        in_type_size,
                                        out_type_size,
                                        grid_size_per_batch,
                                        block_size,
                                        tensors.in,
                                        tensors.out,
                                        kern_copy,
                                        in_ptr,
                                        out_ptr,
                                        current_batch_size);

                handle.Run(kern_copy)(in_ptr,
                                      tensors.w,
                                      alpha_val,
                                      beta_val,
                                      out_ptr,
                                      in_strides,
                                      wei_strides,
                                      out_strides,
                                      di,
                                      hi,
                                      wi,
                                      current_batch_size,
                                      k_per_group,
                                      c_per_group,
                                      do_,
                                      ho,
                                      wo,
                                      sz,
                                      sy,
                                      sx,
                                      dz,
                                      dy,
                                      dx,
                                      pz,
                                      py,
                                      px,
                                      fz,
                                      fy,
                                      fx,
                                      group);

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

::miopen::solver::ConvSolution
GetConv2DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size = NAIVE_CONV_BLOCK_SIZE;
    size_t grid_size  = static_cast<size_t>(k);

    // Cross-block WRW tiling accumulates partial sums into the weight buffer via atomicAdd
    // across gridDim.y tile-blocks. Only fp32 is safe: naive_atomic_add<float> (naive_conv.hpp)
    // forwards to hardware atomicAdd, exact, no rounding. Everything else is excluded:
    //  - int32 (int8-input) and fp8: unimplemented — no naive_atomic_add<int32_t> overload
    //    (would race), and fp8_naive_conv.cpp never reads gridDim.y.
    //  - fp16/bf16: implemented but unsafe — CAS-based atomicAdd rounds to 16-bit per block,
    //    causing catastrophic cancellation once the running sum is large.
    // IsOutputFp32 also matches int8-in/int8-weights/float-out, so !IsAccInt32 must stay too.
    size_t spatial           = static_cast<size_t>(n) * ho * wo;
    size_t num_spatial_tiles = 1;
    if(!IsAccInt32(problem) && IsOutputFp32(problem) && spatial > WRW_SPATIAL_TILING_THRESHOLD)
        num_spatial_tiles = (spatial + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == FP8_NAIVE_CONV_KERNEL_FILE);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.xDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.dwDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.dyDesc.GetStrides(), G_stride_idx));
            if(is_f8)
            {
                handle.Run(kern)(tensors.x,
                                 tensors.dw,
                                 tensors.dy,
                                 in_strides,
                                 wei_strides,
                                 out_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group,
                                 problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                     miopenF8RoundingModeStochastic,
                                 problem.GetConv().attribute.fp8rounding_mode.GetSeed());
            }
            else
            {
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();

                auto kern_copy = kern;
                if(num_spatial_tiles > 1)
                {
                    if(alpha_val == 1.0 && beta_val == 0.0)
                    {
                        // Zero weight buffer before atomicAdd accumulation
                        ZeroTensor(handle, tensors.dwDesc, tensors.dw);
                        if(handle.IsProfilingEnabled())
                            elapsed += handle.GetKernelTime();
                    }
                    else
                    {
                        // atomicAdd bypasses alpha/beta; fall back to serial
                        kern_copy.gdims[1] = 1;
                    }
                }

                handle.Run(kern_copy)(tensors.x,
                                      tensors.dw,
                                      alpha_val,
                                      beta_val,
                                      tensors.dy,
                                      in_strides,
                                      wei_strides,
                                      out_strides,
                                      hi,
                                      wi,
                                      n,
                                      k_per_group,
                                      c_per_group,
                                      ho,
                                      wo,
                                      sy,
                                      sx,
                                      dy,
                                      dx,
                                      py,
                                      px,
                                      fy,
                                      fx,
                                      group);
            }
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

::miopen::solver::ConvSolution
GetConv3DWRWSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size = NAIVE_CONV_BLOCK_SIZE;
    size_t grid_size  = static_cast<size_t>(k);

    // Cross-block spatial tiling for WRW — see 2D WRW comment for details.
    size_t spatial           = static_cast<size_t>(n) * do_ * ho * wo;
    size_t num_spatial_tiles = 1;
    if(!IsAccInt32(problem) && IsOutputFp32(problem) && spatial > WRW_SPATIAL_TILING_THRESHOLD)
        num_spatial_tiles = (spatial + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.xDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.dwDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.dyDesc.GetStrides(), G_stride_idx));

            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();

            auto kern_copy = kern;
            if(num_spatial_tiles > 1)
            {
                if(alpha_val == 1.0 && beta_val == 0.0)
                {
                    // Zero weight buffer before atomicAdd accumulation
                    ZeroTensor(handle, tensors.dwDesc, tensors.dw);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else
                {
                    // atomicAdd bypasses alpha/beta; fall back to serial
                    kern_copy.gdims[1] = 1;
                }
            }

            handle.Run(kern_copy)(tensors.x,
                                  tensors.dw,
                                  alpha_val,
                                  beta_val,
                                  tensors.dy,
                                  in_strides,
                                  wei_strides,
                                  out_strides,
                                  di,
                                  hi,
                                  wi,
                                  n,
                                  k_per_group,
                                  c_per_group,
                                  do_,
                                  ho,
                                  wo,
                                  sz,
                                  sy,
                                  sx,
                                  dz,
                                  dy,
                                  dx,
                                  pz,
                                  py,
                                  px,
                                  fz,
                                  fy,
                                  fx,
                                  group);

            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

::miopen::solver::ConvSolution
GetConv2DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t grid_size     = 1;
    size_t thread_length = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size     = static_cast<size_t>(n) * c;
        thread_length = static_cast<size_t>(hi) * wi;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size     = static_cast<size_t>(n) * hi;
        thread_length = static_cast<size_t>(wi) * c;
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }

    const auto kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    // On CDNA (gfx90a/gfx942/gfx950) a small grid uses 1024-thread groups; 256 otherwise. Must be
    // computed before num_spatial_tiles below, which derives the tile count from block_size.
    size_t block_size = NaiveConv2DBWDBlockSize(ctx.GetStream(), kernel_file, grid_size);

    // BWD-d kernel uses a single if(tid < thread_length) guard with no loop,
    // so tiling is required for correctness — unlike FWD which has a grid-stride
    // loop and only tiles for occupancy. Not implemented for fp8 kernels.
    size_t num_spatial_tiles =
        IsFp8Kernel(problem) ? 1 : (thread_length + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = kernel_file;
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    const auto is_f8 = (kernel.kernel_file == FP8_NAIVE_CONV_KERNEL_FILE);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<5>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));
            /// \ref backward_tensors_reversed_why
            if(is_f8)
            {
                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 tensors.in,
                                 out_strides,
                                 wei_strides,
                                 in_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group,
                                 problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                     miopenF8RoundingModeStochastic,
                                 problem.GetConv().attribute.fp8rounding_mode.GetSeed());
            }
            else
            {
                double alpha_val = data_ctx.alpha.GetAsDouble();
                double beta_val  = data_ctx.beta.GetAsDouble();
                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 alpha_val,
                                 beta_val,
                                 tensors.in,
                                 out_strides,
                                 wei_strides,
                                 in_strides,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group);
            }
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    result.construction_params.push_back(kernel);
    return result;
}

::miopen::solver::ConvSolution
GetConv3DBWDSolution(const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    ::miopen::solver::ConvSolution result;

    int di          = ProblemInterpreter::GetInputDepthDi(problem);
    int hi          = ProblemInterpreter::GetInputHeightHi(problem);
    int wi          = ProblemInterpreter::GetInputWidthWi(problem);
    int n           = ProblemInterpreter::GetBatchN(problem);
    int k           = ProblemInterpreter::GetOutputChannelK(problem);
    int c           = ProblemInterpreter::GetInputChannelC(problem);
    int do_         = ProblemInterpreter::GetOutputDepthDo(problem);
    int ho          = ProblemInterpreter::GetOutputHeightHo(problem);
    int wo          = ProblemInterpreter::GetOutputWidthWo(problem);
    int sz          = ProblemInterpreter::GetAdjustedConvolutionStrideD(problem);
    int sy          = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
    int sx          = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
    int dz          = ProblemInterpreter::GetAdjustedConvolutionDilationD(problem);
    int dy          = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
    int dx          = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
    int pz          = ProblemInterpreter::GetInputLeftPadD(problem);
    int py          = ProblemInterpreter::GetInputLeftPadH(problem);
    int px          = ProblemInterpreter::GetInputLeftPadW(problem);
    int fz          = ProblemInterpreter::GetFilterDepthZ(problem);
    int fy          = ProblemInterpreter::GetFilterHeightY(problem);
    int fx          = ProblemInterpreter::GetFilterWidthX(problem);
    int group       = ProblemInterpreter::GetGroupCountG(problem);
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size    = NAIVE_CONV_BLOCK_SIZE;
    size_t grid_size     = 1;
    size_t thread_length = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size     = static_cast<size_t>(n) * c;
        thread_length = static_cast<size_t>(di) * hi * wi;
    }
    else if(problem.IsLayoutNHWC())
    {
        grid_size     = static_cast<size_t>(group) * n * di;
        thread_length = static_cast<size_t>(hi) * wi * c_per_group;
    }
    else
    {
        MIOPEN_THROW("Unsupported layout");
    }
    // BWD-d kernel uses a single if(tid < thread_length) guard with no loop,
    // so tiling is required for correctness — unlike FWD which has a grid-stride
    // loop and only tiles for occupancy. Not implemented for fp8 kernels.
    size_t num_spatial_tiles =
        IsFp8Kernel(problem) ? 1 : (thread_length + block_size - 1) / block_size;

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(num_spatial_tiles);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    int G_stride_idx = GetGroupStrideIndex(problem);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            float elapsed           = 0;
            auto in_strides         = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.inDesc.GetStrides(), G_stride_idx));
            // For weights, we split K to (G, K_per_group), which is always index 0
            auto wei_strides =
                MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, tensors.wDesc.GetStrides()));
            auto out_strides = MakeStrideArray<6>(
                SplitStrideCtoGC(group, tensors.outDesc.GetStrides(), G_stride_idx));
            /// \anchor backward_tensors_reversed_why
            /// \todo Someone made the silly decision of swapping in and
            /// out pointers in ConvTensors for backward pass, so now I have to
            /// pass out in place of in, out_strides in place of in_strides and
            /// vice-versa --amberhassaan
            double alpha_val = data_ctx.alpha.GetAsDouble();
            double beta_val  = data_ctx.beta.GetAsDouble();
            handle.Run(kern)(tensors.out,
                             tensors.w,
                             alpha_val,
                             beta_val,
                             tensors.in,
                             out_strides,
                             wei_strides,
                             in_strides,
                             di,
                             hi,
                             wi,
                             n,
                             k_per_group,
                             c_per_group,
                             do_,
                             ho,
                             wo,
                             sz,
                             sy,
                             sx,
                             dz,
                             dy,
                             dx,
                             pz,
                             py,
                             px,
                             fz,
                             fy,
                             fx,
                             group);

            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace conv_internal
} // namespace conv
} // namespace solver
} // namespace miopen
