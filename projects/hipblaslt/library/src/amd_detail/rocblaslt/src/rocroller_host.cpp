/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*********************************************************
 * The implementation of the rocblaslt<->rocRoller interface layer. *
 *********************************************************/

#include "rocroller_host.hpp"
#include "rocroller_host_internal.hpp"
#include "Debug.hpp"
#include "handle.h"
#include "utility.hpp"

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include <Tensile/analytical/Utils.hpp>

using namespace rocRoller;

const int DEFAULT_WGM                  = 2;
const int MAX_BITS_WORKGROUPTILE_M     = 8;
const int MAX_BITS_WORKGROUPTILE_N     = 8;
const int MAX_BITS_WORKGROUPTILE_K     = 7;
const int MAX_BITS_PREFETCH_IN_FLIGHT  = 4;
const int REQUIRED_MULTIPLE_M_N        = 16;
const int REQUIRED_MULTIPLE_K          = 32;
const int USE_WORKGROUP_MAPPING_K_SIZE = 4096;

/**
 * @brief SolutionIndex Parameters
 *
 * All of the parameters that are used to generated a unique solution index.
 * There can be multiple kernels of the same KernelType that have different
 * SolutionIndexParameters.
 *
 */
struct SolutionIndexParameters
{
    WorkGroupTileSize workgroupTile;
    int               prefetchInFlight;
    bool              workgroupMapping;
};

/**
 * @brief Solution Parameters
 *
 * Everything needed to generate a kernel
 *
 */
struct SolutionParameters
{
    // Datatype of inputs and outputs
    KernelType kernelType;

    // Workgroup Tile size
    WorkGroupTileSize workgroupTile;

    // Machine Instruction
    MachineInstructionSize machineInstruction;

    // Number of wave tiles to execute per workgroup
    uint wavefrontSize  = 64;
    int  workgroupSizeX = 2 * wavefrontSize;
    int  workgroupSizeY = 2;

    // Other options
    bool loadLDSA    = true;
    bool loadLDSB    = true;
    bool storeLDSD   = false;
    bool direct2LDSA = true;
    bool direct2LDSB = true;

    bool prefetch          = true;
    int  prefetchInFlight  = 2;
    int  prefetchLDSFactor = 1;
    bool prefetchMixMemOps = true;
    bool betaInFma         = true;

    // Unroll Options
    unsigned int unrollX = 0;
    unsigned int unrollY = 0;

    std::string scheduler;

    bool streamK        = false;
    bool streamKTwoTile = false;

    // Scale options
    bool loadLDSScaleA = false;
    bool loadLDSScaleB = false;
    bool swizzleScale  = true;
    bool prefetchScale = true;

    // Workgroup Mapping
    int workgroupMappingDim = 0;
    bool workgroupRemapXCC = true;

    std::string toString() const;
};

/**
 * @brief GemmKernel
 *
 * Everything needed to launch a kernel
 *
 */
struct GemmKernel
{
    CommandPtr                          command;
    CommandKernelPtr                    commandKernel;
    std::shared_ptr<SolutionParameters> params;

    Operations::OperationTag tagTensorA;
    Operations::OperationTag tagTensorB;
    Operations::OperationTag tagTensorC;
    Operations::OperationTag tagTensorD;

    Operations::OperationTag tagScalarAlpha;
    Operations::OperationTag tagScalarBeta;

    Operations::OperationTag tagTensorScaleA;
    Operations::OperationTag tagTensorScaleB;

    Operations::OperationTag tagWGM;
};

/**
 * @brief RocRollerHandle
 *
 * State that is needed for executing rocRoller kernels
 *
 */
struct RocRollerHandle
{
    // Map of kernels that have already been generated.
    // The first level of the map is indexed with a KernelType.
    // The second level of the map is indexed with a hash value of a
    // SolutionIndexParameters type.
    // The value is a GemmKernel.
    std::map<KernelType, std::map<int, std::shared_ptr<GemmKernel>>> generatedKernels;
};

/**
 * @brief Create a new rocRoller handle.
 *
 * This should be done whenever a hipBLASLt handle is created.
 *
 * @param handle
 */
void rocroller_create_handle(void** handle)
{
    *handle = new RocRollerHandle();
}

/**
 * @brief Destroy a rocRoller handle
 *
 * This should be done whenever a hipBLASLt handle is destroyed.
 *
 * @param handle
 */
void rocroller_destroy_handle(void* handle)
{
    delete static_cast<RocRollerHandle*>(handle);
}

// Hash function for a SolutionIndexParameters
// A hash function is used because we can only store a 64bit value in a
// rocblaslt_matmul_algo data field for a solution index.
namespace std
{
    template <>
    struct hash<SolutionIndexParameters>
    {
        std::size_t operator()(const SolutionIndexParameters& params) const
        {
            size_t       result = params.workgroupTile.k / REQUIRED_MULTIPLE_K;
            unsigned int pos    = MAX_BITS_WORKGROUPTILE_K;
            result |= ((params.workgroupTile.n / REQUIRED_MULTIPLE_M_N) << pos);
            pos += MAX_BITS_WORKGROUPTILE_N;
            result |= ((params.workgroupTile.m / REQUIRED_MULTIPLE_M_N) << pos);
            pos += MAX_BITS_WORKGROUPTILE_M;
            result |= (params.prefetchInFlight << pos);
            pos += MAX_BITS_PREFETCH_IN_FLIGHT;
            result |= ((params.workgroupMapping ? 1 : 0) << pos);

            AssertFatal(result < INT_MAX, "Solution Index is too large");
            // Set top bit indicating it is a rocRoller index
            result |= (1 << 31);
            return result;
        }
    };
}

inline unsigned int mask(unsigned int numBits)
{
    return (1 << numBits) - 1;
}

/**
 * Convert a solution index back into SolutionIndexParameters
 */
SolutionIndexParameters indexToParameters(int index)
{
    SolutionIndexParameters result;
    unsigned int            pos = 0;

    result.workgroupTile.k
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_K)) * REQUIRED_MULTIPLE_K;
    pos += MAX_BITS_WORKGROUPTILE_K;
    result.workgroupTile.n
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_N)) * REQUIRED_MULTIPLE_M_N;
    pos += MAX_BITS_WORKGROUPTILE_N;
    result.workgroupTile.m
        = ((index >> pos) & mask(MAX_BITS_WORKGROUPTILE_M)) * REQUIRED_MULTIPLE_M_N;
    pos += MAX_BITS_WORKGROUPTILE_M;
    result.prefetchInFlight = (index >> pos) & mask(MAX_BITS_PREFETCH_IN_FLIGHT);
    pos += MAX_BITS_PREFETCH_IN_FLIGHT;
    result.workgroupMapping = (index >> pos) & 1;

    return result;
}

inline std::string scaleModeOption(RocblasltContractionProblem::ScalingFormat scale)
{
    switch(scale)
    {
    case RocblasltContractionProblem::ScalingFormat::Scalar:
        return "1";
    case RocblasltContractionProblem::ScalingFormat::Vector:
        return "2";
    case RocblasltContractionProblem::ScalingFormat::Block:
        return "3";
    default:
        return "";
    }
}

inline void logBench(const RocblasltContractionProblem& prob,
                     const int&                         solutionIndex,
                     bool                               flush,
                     const int32_t&                     rotatingBufferSize,
                     const int32_t&                     coldIterations,
                     const int32_t&                     hotIterations)
{
    auto s = log_str(__func__,
                     "--api_method",
                     "c",
                     "-m",
                     prob.m,
                     "-n",
                     prob.n,
                     "-k",
                     prob.k,
                     "--lda",
                     prob.col_stride_a,
                     "--ldb",
                     prob.col_stride_b,
                     "--ldc",
                     prob.col_stride_c,
                     "--ldd",
                     prob.col_stride_d,
                     "--stride_a",
                     prob.batch_stride_a,
                     "--stride_b",
                     prob.batch_stride_b,
                     "--stride_c",
                     prob.batch_stride_c,
                     "--stride_d",
                     prob.batch_stride_d,
                     "--alpha",
                     *((float*)prob.alpha),
                     "--beta",
                     *((float*)prob.beta),
                     "--transA",
                     prob.trans_a == HIPBLAS_OP_T ? "T" : "N",
                     "--transB",
                     prob.trans_b == HIPBLAS_OP_T ? "T" : "N",
                     "--batch_count",
                     prob.batch_count,
                     "--scaleA",
                     scaleModeOption(prob.scaleAType),
                     "--scaleB",
                     scaleModeOption(prob.scaleBType),
                     "--a_type",
                     hipDataType_to_bench_string(prob.a_type),
                     "--b_type",
                     hipDataType_to_bench_string(prob.b_type),
                     "--c_type",
                     hipDataType_to_bench_string(prob.c_type),
                     "--d_type",
                     hipDataType_to_bench_string(prob.d_type),
                     "--compute_type",
                     "f32_r",
                     "--algo_method",
                     "index",
                     "--solution_index",
                     solutionIndex,
                     flush ? "--flush" : "",
                     "--rotating",
                     rotatingBufferSize,
                     "--cold_iters",
                     coldIterations,
                     "--iters",
                     hotIterations);

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
        log_bench_from_str(s);
    if(rocblaslt::Debug::Instance().printLogAsMarker())
    {
        rocblaslt::Debug::Instance().logMarkerStart(s.c_str());
        rocblaslt::Debug::Instance().logMarkerStop();
    }
}

inline void logProfile(const RocblasltContractionProblem& prob,
                       bool                               flush,
                       const int32_t&                     rotatingBufferSize,
                       const int32_t&                     coldIterations,
                       const int32_t&                     hotIterations)
{
    log_profile("matmul",
                "M",
                prob.m,
                "N",
                prob.n,
                "K",
                prob.k,
                "lda",
                prob.col_stride_a,
                "ldb",
                prob.col_stride_b,
                "ldc",
                prob.col_stride_c,
                "ldd",
                prob.col_stride_d,
                "stride_a",
                prob.batch_stride_a,
                "stride_b",
                prob.batch_stride_b,
                "stride_c",
                prob.batch_stride_c,
                "stride_d",
                prob.batch_stride_e,
                "alpha",
                *((float*)prob.alpha),
                "beta",
                *((float*)prob.beta),
                "transA",
                prob.trans_a == HIPBLAS_OP_T ? "T" : "N",
                "transB",
                prob.trans_b == HIPBLAS_OP_T ? "T" : "N",
                "batch_count",
                prob.batch_count,
                "scaleA",
                scaleModeOption(prob.scaleAType),
                "scaleB",
                scaleModeOption(prob.scaleBType),
                "a_type",
                hipDataType_to_bench_string(prob.a_type),
                "b_type",
                hipDataType_to_bench_string(prob.b_type),
                "c_type",
                hipDataType_to_bench_string(prob.c_type),
                "d_type",
                hipDataType_to_bench_string(prob.d_type),
                "compute_type",
                "f32_r",
                "flush",
                flush ? "true" : "false",
                "rotating",
                rotatingBufferSize,
                "cold_iters",
                coldIterations,
                "iters",
                hotIterations);
}

inline void logExtendedProfile(const RocblasltContractionProblem& prob,
                               const int&                         solutionIndex,
                               const std::string&                 kernelName,
                               bool                               flush,
                               const int32_t&                     rotatingBufferSize,
                               const int32_t&                     coldIterations,
                               const int32_t&                     hotIterations)
{
    log_profile("matmul",
                "M",
                prob.m,
                "N",
                prob.n,
                "K",
                prob.k,
                "lda",
                prob.col_stride_a,
                "ldb",
                prob.col_stride_b,
                "ldc",
                prob.col_stride_c,
                "ldd",
                prob.col_stride_d,
                "stride_a",
                prob.batch_stride_a,
                "stride_b",
                prob.batch_stride_b,
                "stride_c",
                prob.batch_stride_c,
                "stride_d",
                prob.batch_stride_e,
                "alpha",
                *((float*)prob.alpha),
                "beta",
                *((float*)prob.beta),
                "transA",
                prob.trans_a == HIPBLAS_OP_T ? "T" : "N",
                "transB",
                prob.trans_b == HIPBLAS_OP_T ? "T" : "N",
                "batch_count",
                prob.batch_count,
                "scaleA",
                scaleModeOption(prob.scaleAType),
                "scaleB",
                scaleModeOption(prob.scaleBType),
                "a_type",
                hipDataType_to_bench_string(prob.a_type),
                "b_type",
                hipDataType_to_bench_string(prob.b_type),
                "c_type",
                hipDataType_to_bench_string(prob.c_type),
                "d_type",
                hipDataType_to_bench_string(prob.d_type),
                "compute_type",
                "f32_r",
                "flush",
                flush ? "true" : "false",
                "rotating",
                rotatingBufferSize,
                "cold_iters",
                coldIterations,
                "iters",
                hotIterations,
                "solution_index",
                solutionIndex,
                "kernel_name",
                kernelName);
}

std::string SolutionParameters::toString() const
{
    std::stringstream result;

    result << "WorkGroupTile:" << workgroupTile.m << "x" << workgroupTile.n << "x"
           << workgroupTile.k << std::endl;
    result << "MachineInstruction:" << machineInstruction.m << "x" << machineInstruction.n << "x"
           << machineInstruction.k << std::endl;
    result << "WorkgroupSize:" << workgroupSizeX << "x" << workgroupSizeY << std::endl;
    result << "LDS Usage";
    result << " A:" << (direct2LDSA ? "DirectToLDS" : (loadLDSA ? "On" : "Off"));
    result << " B:" << (direct2LDSB ? "DirectToLDS" : (loadLDSB ? "On" : "Off"));
    result << " D:" << (storeLDSD ? "On" : "Off") << std::endl;
    result << "Workgroup Mapping: Dim:" << workgroupMappingDim << " RemapXCC:" << workgroupRemapXCC << std::endl;
    result << "Prefetch:" << prefetch << " InFlight:" << prefetchInFlight
           << " LDSFactor:" << prefetchLDSFactor << " MixMemOps:" << prefetchMixMemOps << std::endl;
    result << "Block Scale Options:" << " Swizzle Scale:" << swizzleScale
           << " Prefetch Scale:" << prefetchScale << " loadLDS A:" << loadLDSScaleA
           << " loadLDS B:" << loadLDSScaleB << std::endl;

    return result.str();
}

/**
 * @brief Convert hipDataType to a rocRoller::Datatype
 *
 * @param type
 * @return rocRoller::DataType
 */
rocRoller::DataType hipDataType_to_rocRoller_type(hipDataType type)
{
    // Older versions of ROCm do not have these types defined,
    // so they need to be handled specially.
    if(static_cast<int>(type) == HIP_R_6F_E2M3_EXT)
    {
        return rocRoller::DataType::FP6;
    }
    if(static_cast<int>(type) == HIP_R_6F_E3M2_EXT)
    {
        return rocRoller::DataType::BF6;
    }
    if(static_cast<int>(type) == HIP_R_4F_E2M1_EXT)
    {
        return rocRoller::DataType::FP4;
    }

    switch(type)
    {
    case HIP_R_16F:
        return rocRoller::DataType::Half;
    case HIP_R_32F:
        return rocRoller::DataType::Float;
    case HIP_R_16BF:
        return rocRoller::DataType::BFloat16;
    case HIP_R_8F_E4M3_FNUZ:
        return rocRoller::DataType::FP8;
    case HIP_R_8F_E5M2_FNUZ:
        return rocRoller::DataType::BF8;
    case HIP_R_8F_E4M3:
        return rocRoller::DataType::FP8;
    case HIP_R_8F_E5M2:
        return rocRoller::DataType::BF8;
    default:
        return rocRoller::DataType::None;
    }
}

/**
 * @brief Convert a rocblaslt_compute_type to a rocRoller::DataType
 *
 * @param type
 * @return rocRoller::DataType
 */
rocRoller::DataType rocblaslt_compute_type_to_rocRoller_type(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16:
        return rocRoller::DataType::Half;
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_xf32:
    case rocblaslt_compute_f32_fast_f16:
    case rocblaslt_compute_f32_fast_bf16:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
    case rocblaslt_compute_f32_fast_f8:
    case rocblaslt_compute_f32_fast_bf8:
    case rocblaslt_compute_f32_fast_f8bf8:
    case rocblaslt_compute_f32_fast_bf8f8:
        return rocRoller::DataType::Float;
    case rocblaslt_compute_f64:
        return rocRoller::DataType::Double;
    case rocblaslt_compute_i32:
        return rocRoller::DataType::Int32;
    default:
        return rocRoller::DataType::None;
    }
}

/**
 * @brief Generate a KernelType from a RocblasltContractionProblem
 *
 *
 * @param prob
 * @return kernelType
 */
KernelType genKernelType(const RocblasltContractionProblem& prob)
{
    KernelType kernelType;

    kernelType.typeA      = hipDataType_to_rocRoller_type(prob.a_type);
    kernelType.typeB      = hipDataType_to_rocRoller_type(prob.b_type);
    kernelType.typeC      = hipDataType_to_rocRoller_type(prob.c_type);
    kernelType.typeD      = hipDataType_to_rocRoller_type(prob.d_type);
    kernelType.typeAcc    = rocblaslt_compute_type_to_rocRoller_type(prob.compute_type);
    kernelType.transA     = prob.trans_a;
    kernelType.transB     = prob.trans_b;
    kernelType.scaleAMode = prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block
                                ? rocRoller::Operations::ScaleMode::Separate
                                : rocRoller::Operations::ScaleMode::None;
    kernelType.scaleBMode = prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block
                                ? rocRoller::Operations::ScaleMode::Separate
                                : rocRoller::Operations::ScaleMode::None;
    kernelType.scaleABlockRowSize = prob.scaleABlockRowSize;
    kernelType.scaleABlockColSize = prob.scaleABlockColSize;
    kernelType.scaleBBlockRowSize = prob.scaleBBlockRowSize;
    kernelType.scaleBBlockColSize = prob.scaleBBlockColSize;

    return kernelType;
}


/**
 * @brief Choose the SolutionIndexParameters to use for a given problem
 *
 * Examine the KernelType and problem size to determine the kernel to use
 * to compute the problem.
 *
 * Return a list of SolutionIndexParameters, in sorted order, based on how many kernels are requested.
 *
 * @param kernelType
 * @param prob
 * @return std::vector<SolutionIndexParameters>
 */
std::vector<SolutionIndexParameters> chooseSolutionIndexParameters(
    const KernelType& kernelType, const RocblasltContractionProblem& prob, int requestedAlgoCount)
{
    std::vector<SolutionIndexParameters> params;

    std::vector<TensileLite::analytical::TileTuple> tile_list = getTileListForKernelType(kernelType);

    size_t elementSizeA_bits = rocRoller::DataTypeInfo::Get(kernelType.typeA).elementBits; 
    size_t elementSizeB_bits = rocRoller::DataTypeInfo::Get(kernelType.typeB).elementBits;
    size_t elementSizeC_bits = rocRoller::DataTypeInfo::Get(kernelType.typeC).elementBits; 

    size_t maxAB_bits = std::max(elementSizeA_bits, elementSizeB_bits);
    TensileLite::analytical::DataType dataType = TensileLite::analytical::DataType::Float8;
    if(maxAB_bits == 6)
        dataType = TensileLite::analytical::DataType::Float6;
    else if(maxAB_bits == 4)
        dataType = TensileLite::analytical::DataType::Float4;

    const TensileLite::analytical::Hardware analaytical_hardware = TensileLite::analytical::Hardware::getHardwareForDevice(0);

    int WGM = std::sqrt(std::floor(analaytical_hardware.N_CU / analaytical_hardware.NUM_XCD));

    auto selected_tiles = TensileLite::analytical::select_best_macro_tile_size(
        prob.m,
        prob.n,
        prob.k,
        prob.batch_count,
        prob.trans_a == hipblasOperation_t::HIPBLAS_OP_T,
        prob.trans_b == hipblasOperation_t::HIPBLAS_OP_T,
        analaytical_hardware,
        tile_list,
        elementSizeA_bits,
        elementSizeB_bits,
        elementSizeC_bits,
        dataType,
        kernelType.scaleABlockRowSize * kernelType.scaleABlockColSize, //Handle A vs B block size.
        0.8,
        false,
        false,
        WGM);

    for(auto const& selected_tile : selected_tiles)
    {
        WorkGroupTileSize wgt{(int)std::get<1>(selected_tile), (int)std::get<2>(selected_tile), (int)std::get<3>(selected_tile)};

        if((requestedAlgoCount == -1)
           || (prob.m % wgt.m == 0 && prob.n % wgt.n == 0 && prob.k % wgt.k == 0))
        {
            // FP8 kernels run out of registers with larger tile sizes
            if((kernelType.typeA == rocRoller::DataType::FP8
                || kernelType.typeA == rocRoller::DataType::BF8
                || kernelType.typeB == rocRoller::DataType::FP8
                || kernelType.typeB == rocRoller::DataType::BF8)
               && wgt.m + wgt.n > 256)
                continue;

            params.push_back({wgt, 1, true});

            // Other datatypes run out of registers when prefetchInFlight is too
            // large.
            // There is an error with smaller tile sizes and larger prefetchInFlight.
            if(kernelType.typeA == rocRoller::DataType::FP4
               && kernelType.typeB == rocRoller::DataType::FP4 && wgt.m > 32 && wgt.n > 32
               && (prob.k % (wgt.k * 4) == 0))
            {
                params.back().prefetchInFlight = 4;
            }
            else if(prob.k % (wgt.k * 2) == 0)
            {
                params.back().prefetchInFlight = 2;
            }
            else
            {
                params.back().prefetchInFlight = 1;
            }

            if (prob.k < USE_WORKGROUP_MAPPING_K_SIZE)
            {
                params.back().workgroupMapping = false;
            }
        }
    }

    return params;
}

/**
 * @brief Generate all of the solution parameters needed to create a kernel.
 *
 * This should only take into account the KernelType and SolutionIndexParameters
 * when deciding on the rest of the parameters to use for the kernel.
 *
 * @param kernelType
 * @param solutionIndexParameters
 * @return std::shared_ptr<SolutionParameters>
 */
std::shared_ptr<SolutionParameters>
    genSolutionParameters(const KernelType&              kernelType,
                          const SolutionIndexParameters& solutionIndexParameters)
{
    auto gemm = std::make_shared<SolutionParameters>();

    gemm->kernelType = kernelType;

    gemm->workgroupTile = solutionIndexParameters.workgroupTile;

    gemm->machineInstruction = pickMI(gemm->kernelType.typeA, gemm->kernelType.typeB, gemm->workgroupTile);

    if(gemm->workgroupTile.m / gemm->machineInstruction.m == 1)
        gemm->workgroupSizeX = gemm->wavefrontSize;
    if(gemm->workgroupTile.n / gemm->machineInstruction.n == 1)
        gemm->workgroupSizeY = 1;

    if(solutionIndexParameters.prefetchInFlight == 1)
    {
        gemm->prefetch = false;
    }
    else
    {
        gemm->prefetchInFlight = solutionIndexParameters.prefetchInFlight;
    }

    // Direct To LDS only supported in certain situations
    if(kernelType.typeA == rocRoller::DataType::FP6 || kernelType.typeA == rocRoller::DataType::BF6)
        gemm->direct2LDSA = false;
    if(kernelType.typeB == rocRoller::DataType::FP6 || kernelType.typeB == rocRoller::DataType::BF6)
        gemm->direct2LDSB = false;
    if((kernelType.typeA == rocRoller::DataType::FP4
        || kernelType.typeB == rocRoller::DataType::FP4)
       && (solutionIndexParameters.workgroupTile.m <= 64
           || solutionIndexParameters.workgroupTile.n <= 64))
    {
        gemm->direct2LDSA = false;
        gemm->direct2LDSB = false;
    }

    if(gemm->direct2LDSA == false || gemm->direct2LDSB == false)
    {
        gemm->prefetchLDSFactor = 2;
    }

    // Swizzle Scale only support in certain situations
    // Swizzle Scale also runs out of registers with FP8
    if(solutionIndexParameters.workgroupTile.m >= 128
       && solutionIndexParameters.workgroupTile.n >= 128)
    {
        gemm->swizzleScale  = true;
        gemm->loadLDSScaleA = false;
        gemm->loadLDSScaleB = false;
    }
    else
    {
        gemm->swizzleScale  = false;
        gemm->prefetchScale = false;
        gemm->loadLDSScaleA = true;
        gemm->loadLDSScaleB = true;
    }

    // LDS can only be used for scaling data with certain workgroup tile sizes
    auto workgroupSize = gemm->workgroupSizeX * gemm->workgroupSizeY;
    auto numScaleElementsA = 0;
    if(gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize != 0)
    {
        numScaleElementsA = gemm->workgroupTile.m
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize));
    }
    auto numScaleElementsB = 0;
    if(gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize != 0)
    {
        numScaleElementsB = gemm->workgroupTile.n
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize));
    }
    if(numScaleElementsA % workgroupSize != 0)
    {
        gemm->loadLDSScaleA     = false;
        gemm->prefetchMixMemOps = false;
    }
    if(numScaleElementsB % workgroupSize != 0)
    {
        gemm->loadLDSScaleB     = false;
        gemm->prefetchMixMemOps = false;
    }

    if(!solutionIndexParameters.workgroupMapping)
    {
        gemm->workgroupMappingDim = -1;
        gemm->workgroupRemapXCC = false;
    }
    else
    {
        gemm->workgroupMappingDim = 0;
        gemm->workgroupRemapXCC = true;
    }

    return gemm;
}

/**
 * @brief Set the required conditions in order to run a provided kernel
 *
 * @param gemmKernel
 */
void setPredicates(std::shared_ptr<GemmKernel> gemmKernel)
{

    auto command        = gemmKernel->command;
    auto commandKernel  = gemmKernel->commandKernel;
    auto solutionParams = gemmKernel->params;

    using namespace rocRoller::Expression;
    auto params = commandKernel->getCommandParameters();

    // predicate building blocks
    // A sizes
    auto aSizes = std::get<Operations::Tensor>(*(command->findTag(gemmKernel->tagTensorA))).sizes();
    std::vector<ExpressionPtr> aSizeExps(aSizes.size());
    std::transform(aSizes.begin(), aSizes.end(), aSizeExps.begin(), [](auto arg) {
        return arg->expression();
    });
    // B sizes
    auto bSizes = std::get<Operations::Tensor>(*(command->findTag(gemmKernel->tagTensorB))).sizes();
    std::vector<ExpressionPtr> bSizeExps(bSizes.size());
    std::transform(bSizes.begin(), bSizes.end(), bSizeExps.begin(), [](auto arg) {
        return arg->expression();
    });

    // parameters
    auto unrollKExp        = literal(params->unrollK);
    auto workgroupTileMExp = literal(solutionParams->workgroupTile.m);
    auto workgroupTileNExp = literal(solutionParams->workgroupTile.n);
    auto workgroupTileKExp = literal(solutionParams->workgroupTile.k);

    // constants
    auto zero = literal(0u);
    auto one  = literal(1u);

    // sanitize parameters
    auto sanUnrollKExp
        = convert(DataType::UInt32, conditional(unrollKExp <= zero, one, unrollKExp));

    // predicates
    std::stringstream ss;
    auto              unrollXPredicate = (aSizeExps[0] % workgroupTileMExp == zero);
    ss << "M must be a multiple of workgroupTile.m=" << solutionParams->workgroupTile.m;
    setComment(unrollXPredicate, ss.str());
    commandKernel->addPredicate(unrollXPredicate);
    ss.str("");

    auto unrollYPredicate = (bSizeExps[1] % workgroupTileNExp == zero);
    ss << "N must be a multiple of workgroupTile.n=" << solutionParams->workgroupTile.n;
    setComment(unrollYPredicate, ss.str());
    commandKernel->addPredicate(unrollYPredicate);
    ss.str("");

    auto unrollKPredicate = (aSizeExps[1] % (workgroupTileKExp * sanUnrollKExp) == zero);
    ss << "K must be a multiple of workgroupTile.k=" << solutionParams->workgroupTile.k
       << " * unrollK=" << rocRoller::Expression::evaluate(sanUnrollKExp);
    setComment(unrollKPredicate, ss.str());
    commandKernel->addPredicate(unrollKPredicate);
    ss.str("");
}

std::string genScaleModeString(Operations::ScaleMode mode)
{
    if(mode == Operations::ScaleMode::Separate)
        return "B";
    if(mode == Operations::ScaleMode::SingleScale)
        return "S";
    return "";
}

std::string genKernelName(std::shared_ptr<SolutionParameters> gemm)
{
    std::ostringstream rv;
    rv << "RR_GEMM_" << (gemm->kernelType.transA == HIPBLAS_OP_N ? "N" : "T")
       << (gemm->kernelType.transB == HIPBLAS_OP_N ? "N" : "T");

    rv << "_";
    for(auto const& t : {gemm->kernelType.typeA,
                         gemm->kernelType.typeB,
                         gemm->kernelType.typeC,
                         gemm->kernelType.typeD,
                         gemm->kernelType.typeAcc})
        rv << toString(t) << "_";

    if(gemm->kernelType.scaleAMode != Operations::ScaleMode::None)
        rv << "SA_" << genScaleModeString(gemm->kernelType.scaleAMode) << "_";
    if(gemm->kernelType.scaleBMode != Operations::ScaleMode::None)
        rv << "SB_" << genScaleModeString(gemm->kernelType.scaleBMode) << "_";

    rv << "WGT_";
    rocRoller::streamJoin(
        rv, std::vector{gemm->workgroupTile.m, gemm->workgroupTile.n, gemm->workgroupTile.k}, "x");

    rv << "_UR_" << gemm->prefetchInFlight;

    if (gemm->workgroupMappingDim != -1)
    {
        rv <<"_WGM_";
    }

    return rv.str();
}

/**
 * @brief Generate a GEMM Kernel
 *
 * This involves creating the Command describing the KernelType
 * and setting all of the parameters.
 *
 * @param gemm
 * @return std::shared_ptr<GemmKernel>
 */
std::shared_ptr<GemmKernel> genGemmKernel(std::shared_ptr<SolutionParameters> gemm)
{
    // -------------------------------------------------------------
    // Create Command object describing problem

    auto dataTypeA = gemm->kernelType.typeA;
    auto dataTypeB = gemm->kernelType.typeB;
    auto dataTypeC = gemm->kernelType.typeC;
    auto dataTypeD = gemm->kernelType.typeD;

    auto command = std::make_shared<Command>();

    std::vector<size_t> oneStridesN = std::vector<size_t>({(size_t)1});
    std::vector<size_t> oneStridesT = std::vector<size_t>({(size_t)0, (size_t)1});

    auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
        2, dataTypeA, gemm->kernelType.transA == HIPBLAS_OP_N ? oneStridesN : oneStridesT)); // A
    auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

    auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
        2, dataTypeB, gemm->kernelType.transB == HIPBLAS_OP_N ? oneStridesN : oneStridesT)); // B
    auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

    auto mulInputA = tagLoadA;
    auto mulInputB = tagLoadB;

    AssertFatal(gemm->kernelType.scaleAMode == Operations::ScaleMode::None
                    || gemm->kernelType.scaleAMode == Operations::ScaleMode::SingleScale
                    || gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate,
                "Scale mode not supported!",
                ShowValue(gemm->kernelType.scaleAMode));
    AssertFatal(gemm->kernelType.scaleBMode == Operations::ScaleMode::None
                    || gemm->kernelType.scaleBMode == Operations::ScaleMode::SingleScale
                    || gemm->kernelType.scaleBMode == Operations::ScaleMode::Separate,
                "Scale mode not supported!",
                ShowValue(gemm->kernelType.scaleBMode));

    std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
        tagTensorScaleB, tagLoadScaleB, tagBlockScaleB, tagWGM;

    if(gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
            2,
            gemm->kernelType.scaleTypeA,
            gemm->kernelType.transA == HIPBLAS_OP_N ? oneStridesN : oneStridesT));
        tagLoadScaleA
            = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleA));

        tagBlockScaleA = mulInputA = command->addOperation(rocRoller::Operations::BlockScale(
            tagLoadA,
            2,
            tagLoadScaleA,
            {gemm->kernelType.scaleABlockColSize, gemm->kernelType.scaleABlockRowSize}));
    }

    if(gemm->kernelType.scaleBMode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleB = command->addOperation(rocRoller::Operations::Tensor(
            2,
            gemm->kernelType.scaleTypeA,
            gemm->kernelType.transB == HIPBLAS_OP_N ? oneStridesN : oneStridesT));
        tagLoadScaleB
            = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleB));

        tagBlockScaleB = mulInputB = command->addOperation(rocRoller::Operations::BlockScale(
            tagLoadB,
            2,
            tagLoadScaleB,
            {gemm->kernelType.scaleBBlockColSize, gemm->kernelType.scaleBBlockRowSize}));
    }

    auto tagTensorC
        = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeC, oneStridesN)); // C
    auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

    auto tagScalarAlpha
        = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
    auto tagLoadAlpha = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

    auto tagScalarBeta
        = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
    auto tagLoadBeta = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

    auto tagAB = command->addOperation(rocRoller::Operations::T_Mul(mulInputA, mulInputB)); // A * B

    rocRoller::Operations::T_Execute execute(command->getNextTag());
    auto tagBetaC = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, tagLoadC)); // beta * C

    auto tagAlphaAB
        = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)

    rocRoller::Operations::OperationTag tagStoreD;
    if(gemm->betaInFma)
    {
        tagStoreD = execute.addXOp(
            rocRoller::Operations::E_Add(tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
    }
    else
    {
        tagStoreD = execute.addXOp(
            rocRoller::Operations::E_Add(tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
    }

    command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

    auto tagTensorD
        = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD, oneStridesN)); // D
    Operations::OperationTag tagScalarSeed;
    if(gemm->kernelType.typeAcc == gemm->kernelType.typeD)
    {
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));
    }
    else
    {
        // If Matrix C and D are of different types, an explicit type conversion is required

        auto cvtOp = rocRoller::Operations::T_Execute(command->getNextTag());
        // (SR)Convert( alpha * (A * B) + beta * C )
        auto tagCvt = cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagStoreD, dataTypeD));
        command->addOperation(std::move(cvtOp));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));
    }

    auto tagScratch = command->allocateTag();
    command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                              tagScratch,
                              ArgumentType::Value,
                              DataDirection::ReadWrite,
                              rocRoller::SCRATCH);

    if(gemm->workgroupMappingDim != -1)
    {
        tagWGM = command->allocateTag();
        command->allocateArgument(DataType::Int32,
                                *tagWGM,
                                ArgumentType::Value,
                                DataDirection::ReadOnly,
                                rocRoller::WGM);
    }

    // -------------------------------------------------------------
    // Set the parameters

    auto params = std::make_shared<CommandParameters>();
    params->setManualKernelDimension(2);

    AssertFatal(gemm->workgroupTile.m * gemm->workgroupTile.k
                    >= gemm->machineInstruction.m * gemm->machineInstruction.k,
                "Not enough elements (A).");
    AssertFatal(gemm->workgroupTile.n * gemm->workgroupTile.k
                    >= gemm->machineInstruction.n * gemm->machineInstruction.k,
                "Not enough elements (B).");

    uint wavetilePerWavefrontM = gemm->wavefrontSize * gemm->workgroupTile.m
                                 / gemm->machineInstruction.m / gemm->workgroupSizeX;
    uint wavetilePerWavefrontN
        = gemm->workgroupTile.n / gemm->machineInstruction.n / gemm->workgroupSizeY;

    AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch.");
    AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch.");

    AssertFatal(gemm->workgroupTile.m % (gemm->machineInstruction.m * wavetilePerWavefrontM) == 0,
                "WaveTile size mismatch (M)",
                ShowValue(gemm->workgroupTile.m),
                ShowValue(gemm->machineInstruction.m),
                ShowValue(wavetilePerWavefrontM));
    AssertFatal(gemm->workgroupTile.n % (gemm->machineInstruction.n * wavetilePerWavefrontN) == 0,
                "WaveTile size mismatch (N)",
                ShowValue(gemm->workgroupTile.n),
                ShowValue(gemm->machineInstruction.n),
                ShowValue(wavetilePerWavefrontN));

    // TODO: Calculate these values internally based on workgroup sizes.
    params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

    {
        auto memoryTypeA = MemoryType::WAVE;
        if(gemm->direct2LDSA)
            memoryTypeA = MemoryType::WAVE_Direct2LDS;
        else if(gemm->loadLDSA)
            memoryTypeA = MemoryType::LDS;

        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.k},
            LayoutType::MATRIX_A,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            memoryTypeA);
        params->setDimensionInfo(tagLoadA, macTileA);
    }

    if(gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        // TODO: verify the division of scale block size is correct
        auto const scaleBlockSize
            = gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize;
        auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.k / (int)scaleBlockSize},
            LayoutType::MATRIX_A,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k / (int)scaleBlockSize,
             gemm->machineInstruction.b},
            gemm->loadLDSScaleA ? MemoryType::LDS : MemoryType::WAVE);
        params->setDimensionInfo(*tagLoadScaleA, macTileAScale);
    }

    {
        auto memoryTypeB = MemoryType::WAVE;
        if(gemm->direct2LDSB)
            memoryTypeB = MemoryType::WAVE_Direct2LDS;
        else if(gemm->loadLDSB)
            memoryTypeB = MemoryType::LDS;

        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.k, gemm->workgroupTile.n},
            LayoutType::MATRIX_B,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            memoryTypeB);
        params->setDimensionInfo(tagLoadB, macTileB);
    }

    if(gemm->kernelType.scaleBMode == Operations::ScaleMode::Separate)
    {
        // TODO: verify the division of scale block size is correct
        auto const scaleBlockSize
            = gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize;
        auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.k / (int)scaleBlockSize, gemm->workgroupTile.n},
            LayoutType::MATRIX_B,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k / (int)scaleBlockSize,
             gemm->machineInstruction.b},
            gemm->loadLDSScaleB ? MemoryType::LDS : MemoryType::WAVE);
        params->setDimensionInfo(*tagLoadScaleB, macTileBScale);
    }

    {
        auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.n},
            LayoutType::MATRIX_ACCUMULATOR,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b});
        params->setDimensionInfo(tagLoadC, macTileC);
    }

    {
        auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.n},
            LayoutType::MATRIX_ACCUMULATOR,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            gemm->storeLDSD ? MemoryType::WAVE_LDS : MemoryType::WAVE);
        params->setDimensionInfo(tagStoreD, macTileD);
    }

    params->unrollX       = gemm->unrollX;
    params->unrollY       = gemm->unrollY;
    params->swizzleScale  = gemm->swizzleScale;
    params->prefetchScale = gemm->prefetchScale;

    if(gemm->prefetch)
    {
        params->prefetch          = true;
        params->unrollK           = gemm->prefetchInFlight;
        params->prefetchInFlight  = gemm->prefetchInFlight;
        params->prefetchLDSFactor = gemm->prefetchLDSFactor;
        params->prefetchMixMemOps = gemm->prefetchMixMemOps;
    }
    else
    {
        params->prefetch = false;
    }

    params->transposeMemoryAccess.set(LayoutType::MATRIX_A, gemm->kernelType.transA == HIPBLAS_OP_T);
    params->transposeMemoryAccess.set(LayoutType::MATRIX_B, gemm->kernelType.transB == HIPBLAS_OP_T);

    uint workgroupSizeX = gemm->workgroupSizeX * gemm->workgroupSizeY;
    uint workgroupSizeY = 1;

    // Workgroup Mapping
    if(gemm->workgroupMappingDim != -1)
    {
        auto dim  = gemm->workgroupMappingDim;

        AssertFatal(
            dim == 0 || dim == 1,
            "Only 0 (M) or 1 (N) are supported dimensions for workgroup mapping.",
            ShowValue(dim));

        params->workgroupMapping = {dim, nullptr};
    }

    if(gemm->workgroupRemapXCC)
    {
        params->workgroupRemapXCC = 8;
    }

    params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
    params->setManualWavefrontCount(
        {static_cast<uint>(gemm->workgroupTile.m / gemm->machineInstruction.m
                           / wavetilePerWavefrontM),
         static_cast<uint>(gemm->workgroupTile.n / gemm->machineInstruction.n
                           / wavetilePerWavefrontN)});

    // -------------------------------------------------------------
    // Create CommandKernel

    std::string kernelName    = genKernelName(gemm);
    auto        context       = Context::ForDefaultHipDevice(kernelName);
    auto        commandKernel = std::make_shared<CommandKernel>(command, kernelName);
    commandKernel->setContext(context);
    commandKernel->setCommandParameters(params);
    commandKernel->generateKernel();

    // -------------------------------------------------------------
    // Create GemmKernel

    auto gemmKernel           = std::make_shared<GemmKernel>();
    gemmKernel->command       = command;
    gemmKernel->commandKernel = commandKernel;
    gemmKernel->params        = gemm;

    gemmKernel->tagTensorA = tagTensorA;
    gemmKernel->tagTensorB = tagTensorB;
    gemmKernel->tagTensorC = tagTensorC;
    gemmKernel->tagTensorD = tagTensorD;

    gemmKernel->tagScalarAlpha = tagScalarAlpha;
    gemmKernel->tagScalarBeta  = tagScalarBeta;

    if(tagTensorScaleA)
        gemmKernel->tagTensorScaleA = *tagTensorScaleA;

    if(tagTensorScaleB)
        gemmKernel->tagTensorScaleB = *tagTensorScaleB;

    if(tagWGM)
        gemmKernel->tagWGM = *tagWGM;

    setPredicates(gemmKernel);

    return gemmKernel;
}

/**
 * Generate a kernel from a given SolutionIndexParameters value.
 */
rocblaslt_status
    genKernelFromSolutionIndexParameters(RocRollerHandle*             rocroller_handle,
                                         KernelType                   kernelType,
                                         SolutionIndexParameters      solutionIndexParameter,
                                         int                          solutionIndex,
                                         std::shared_ptr<GemmKernel>& kernel)
{
    auto params = genSolutionParameters(kernelType, solutionIndexParameter);
    try
    {
        kernel                                                        = genGemmKernel(params);
        rocroller_handle->generatedKernels[kernelType][solutionIndex] = kernel;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return rocblaslt_status_not_implemented;
    }

    return rocblaslt_status_success;
}

/**
 * @brief Find the best rocRoller kernels for a given problem
 *
 * This mimics the functionality of getBestSolutions in tensile_host.cpp
 *
 * For a given kernel type and problem, determines the SolutionIndexParameters
 * that should be used.
 *
 * Checks to see if a kernel has already been generated for the chosen SolutionIndexParameters.
 *
 * If it hasn't, a new kernel will be generated and stored in generatedKernels.
 *
 * At the moment, only returns a single solution.
 *
 * @param handle
 * @param prob
 * @param requestedAlgoCount
 * @param heuristicResultsArray
 * @param returnAlgoCount
 * @return rocblaslt_status
 */
rocblaslt_status
    getRocRollerBestSolutions(rocblaslt_handle                   handle,
                              const RocblasltContractionProblem& prob,
                              int                                requestedAlgoCount,
                              rocblaslt_matmul_heuristic_result  heuristicResultsArray[],
                              int*                               returnAlgoCount)
{
    RocRollerHandle* rocroller_handle = static_cast<RocRollerHandle*>(handle->rocroller_handle);
    auto             kernelType       = genKernelType(prob);
    int              index;

    if(prob.bias != nullptr)
    {
        std::cerr << "rocRoller does not support bias" << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(prob.batch_count != 1)
    {
        std::cerr << "rocRoller only supports 1 batch_count not " << prob.batch_count << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(auto scale_type = hipDataType_to_rocRoller_type(prob.scale_type);
       scale_type != rocRoller::DataType::None && scale_type != rocRoller::DataType::Float)
    {
        std::cerr << "rocRoller only supports F32 as scale type not " << scale_type << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(kernelType.typeAcc != rocRoller::DataType::Float)
    {
        std::cerr << "rocRoller only supports F32 accumulation, not " << kernelType.typeAcc
                  << std::endl;
        return rocblaslt_status_invalid_value;
    }

    auto existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    if(existingKernelType == rocroller_handle->generatedKernels.end())
    {
        rocroller_handle->generatedKernels[kernelType] = {};
    }

    auto solutionIndexParameters
        = chooseSolutionIndexParameters(kernelType, prob, requestedAlgoCount);

    int i = 0;
    for(auto const& solutionIndexParameter : solutionIndexParameters)
    {
        if(requestedAlgoCount != -1 && i >= requestedAlgoCount)
            break;

        index = static_cast<int>(std::hash<SolutionIndexParameters>{}(solutionIndexParameter));
        auto existingSolutionIndex = rocroller_handle->generatedKernels[kernelType].find(index);
        // If kernel doesn't already exist, generate it
        if(existingSolutionIndex == rocroller_handle->generatedKernels[kernelType].end())
        {
            std::shared_ptr<GemmKernel> kernel;
            auto                        status = genKernelFromSolutionIndexParameters(
                rocroller_handle, kernelType, solutionIndexParameter, index, kernel);
            if(status != rocblaslt_status_success)
                continue;
        }

        // Fill out heuristicResultsArray
        // The most important thing to do is set the solutionIndex
        memset(heuristicResultsArray[i].algo.data, 0, sizeof(heuristicResultsArray[i].algo.data));
        int* solutionIndex = (int*)(heuristicResultsArray[i].algo.data);
        *solutionIndex     = index;
        heuristicResultsArray[i].algo.max_workspace_bytes = 0;
        heuristicResultsArray[i].algo.fallback            = false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize            = 0;
        i++;
    }

    *returnAlgoCount = i;
    for(; i < requestedAlgoCount; i++)
    {
        heuristicResultsArray[i].state = rocblaslt_status_invalid_value;
    }

    return rocblaslt_status_success;
}

/**
 * Return all of the possible solutions for a KernelType
 */
rocblaslt_status
    getAllSolutionsRocRoller(RocblasltContractionProblem&                    prob,
                             rocblaslt_handle                                handle,
                             std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                             size_t                                          maxWorkSpaceBytes)
{
    heuristicResults.resize(possibleTileSizes.size());
    int  returnAlgoCount;
    auto result
        = getRocRollerBestSolutions(handle, prob, -1, heuristicResults.data(), &returnAlgoCount);
    heuristicResults.resize(returnAlgoCount);
    return result;
}

/**
 * Return a list of heuristicResults for a given list of solution indices
 */
void getRocRollerSolutionsFromIndex(
    rocblaslt_handle                                handle,
    int                                             solutionIndex,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
    size_t                                          maxWorkSpaceBytes)
{
    rocblaslt_matmul_heuristic_result result;
    memset(&result, 0, sizeof(rocblaslt_matmul_heuristic_result));
    memset(result.algo.data, 0, sizeof(result.algo.data));
    int* index                      = (int*)(result.algo.data);
    *index                          = solutionIndex;
    result.algo.max_workspace_bytes = maxWorkSpaceBytes;
    result.algo.fallback            = false;
    result.state                    = rocblaslt_status_success;
    result.workspaceSize            = 0;
    heuristicResults.push_back(result);
}

/**
 * @brief Set the arguments to call a rocRoller kernel
 *
 * @param gemm
 * @param prob
 * @return CommandArguments
 */
CommandArguments createCommandArguments(std::shared_ptr<GemmKernel>        gemm,
                                        const RocblasltContractionProblem& prob,
                                        int wgm)
{
    CommandArguments commandArgs = gemm->command->createArguments();

    size_t M = prob.m;
    size_t N = prob.n;
    size_t K = prob.k;

    TensorDescriptor descA(gemm->params->kernelType.typeA,
                           {M, K},
                           gemm->params->kernelType.transA == HIPBLAS_OP_T ? "T" : "N");
    TensorDescriptor descB(gemm->params->kernelType.typeB,
                           {K, N},
                           gemm->params->kernelType.transB == HIPBLAS_OP_T ? "T" : "N");

    // TODO: Have to typecast void* pointer to something that CommandArgumentValue accepts
    setCommandTensorArg(commandArgs, gemm->tagTensorA, descA, (float*)nullptr);
    setCommandTensorArg(commandArgs, gemm->tagTensorB, descB, (float*)nullptr);

    if(gemm->params->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = prob.scaleABlockRowSize * prob.scaleABlockColSize;
        // TODO: Datatype should be F8 E8M0
        TensorDescriptor descAScale(gemm->params->kernelType.typeA,
                                    {size_t(M), size_t(K / scaleBlockSize)},
                                    gemm->params->kernelType.transA == HIPBLAS_OP_T ? "T" : "N");
        setCommandTensorArg(commandArgs, gemm->tagTensorScaleA, descAScale, (float*)nullptr);
    }
    if(gemm->params->kernelType.scaleBMode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = prob.scaleBBlockRowSize * prob.scaleBBlockColSize;
        // TODO: Datatype should be F8 E8M0
        TensorDescriptor descBScale(gemm->params->kernelType.typeB,
                                    {size_t(K / scaleBlockSize), size_t(N)},
                                    gemm->params->kernelType.transB == HIPBLAS_OP_T ? "T" : "N");
        setCommandTensorArg(commandArgs, gemm->tagTensorScaleB, descBScale, (float*)nullptr);
    }

    TensorDescriptor descC(gemm->params->kernelType.typeC, {M, N}, "N");
    setCommandTensorArg(commandArgs, gemm->tagTensorC, descC, (float*)nullptr);

    commandArgs.setArgument(gemm->tagScalarAlpha, ArgumentType::Value, *((float*)prob.alpha));
    commandArgs.setArgument(gemm->tagScalarBeta, ArgumentType::Value, *((float*)prob.beta));

    TensorDescriptor descD(gemm->params->kernelType.typeD, {M, N}, "N");
    setCommandTensorArg(commandArgs, gemm->tagTensorD, descD, (float*)nullptr);

    commandArgs.setArgument(gemm->tagTensorA, ArgumentType::Value, (float*)prob.A);
    commandArgs.setArgument(gemm->tagTensorB, ArgumentType::Value, (float*)prob.B);
    commandArgs.setArgument(gemm->tagTensorC, ArgumentType::Value, (float*)prob.C);
    commandArgs.setArgument(gemm->tagTensorD, ArgumentType::Value, (float*)prob.D);

    if(gemm->params->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        commandArgs.setArgument(gemm->tagTensorScaleA, ArgumentType::Value, (uint8_t*)prob.scaleA);
    }

    if(gemm->params->kernelType.scaleBMode == Operations::ScaleMode::Separate)
    {
        commandArgs.setArgument(gemm->tagTensorScaleB, ArgumentType::Value, (uint8_t*)prob.scaleB);
    }

    if(gemm->params->workgroupMappingDim != -1)
    {
        AssertFatal(wgm > 0,
                    "Workgroup mapping size must be a positive non-zero integer.",
                    ShowValue(wgm));

        commandArgs.setArgument(gemm->tagWGM, ArgumentType::Value, wgm);
    }

    return commandArgs;
}

/**
 * @brief Get a kernel based on the provided problem and algo.
 *
 * @param handle
 * @param prob
 * @param algo
 * @param kernel
 * @return rocblaslt_status
 */
rocblaslt_status getKernelFromAlgo(rocblaslt_handle                   handle,
                                   const RocblasltContractionProblem& prob,
                                   const rocblaslt_matmul_algo*       algo,
                                   std::shared_ptr<GemmKernel>&       kernel)
{
    int* solutionIndex = (int*)algo->data;

    if(solutionIndex == 0)
        return rocblaslt_status_not_implemented;

    RocRollerHandle* rocroller_handle = static_cast<RocRollerHandle*>(handle->rocroller_handle);
    auto             kernelType       = genKernelType(prob);

    auto existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    // If KernelType doesn't exist yet, add an empty container for it to map.
    if(existingKernelType == rocroller_handle->generatedKernels.end())
    {
        rocroller_handle->generatedKernels[kernelType] = {};
        existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    }

    auto existingKernel = existingKernelType->second.find(*solutionIndex);
    if(existingKernel != existingKernelType->second.end())
    {
        kernel = existingKernel->second;
        return rocblaslt_status_success;
    }
    else
    {
        auto solutionIndexParameter = indexToParameters(*solutionIndex);

        auto status = genKernelFromSolutionIndexParameters(
            rocroller_handle, kernelType, solutionIndexParameter, *solutionIndex, kernel);
        return status;
    }
}

rocblaslt_status isRocRollerSolutionSupported(rocblaslt_handle             handle,
                                              RocblasltContractionProblem& prob,
                                              rocblaslt_matmul_algo*       algo,
                                              size_t*                      workspaceSizeInBytes)
{
    std::shared_ptr<GemmKernel> kernel;
    auto                        status = getKernelFromAlgo(handle, prob, algo, kernel);
    if(status != rocblaslt_status_success)
        return status;

    auto commandArgs = createCommandArguments(kernel, prob, DEFAULT_WGM);

    auto runtimeArgs = commandArgs.runtimeArguments();
    if(!kernel->commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error))
    {
        return rocblaslt_status_invalid_value;
    }

    return rocblaslt_status_success;
}

/**
 * @brief Execute a GEMM operation.
 *
 * @param gemm
 * @param prob
 * @return rocblaslt_status
 */
rocblaslt_status runGemmKernel(std::shared_ptr<GemmKernel>        gemm,
                               const RocblasltContractionProblem& prob)
{
    auto commandArgs = createCommandArguments(gemm, prob, DEFAULT_WGM);

    auto runtimeArgs = commandArgs.runtimeArguments();

    if(!gemm->commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error))
    {
        return rocblaslt_status_invalid_value;
    }

    // TODO: Add scratch space when needed

    gemm->commandKernel->launchKernel(runtimeArgs, prob.stream);
    return rocblaslt_status_success;
}

/**
 * @brief Execute a contraction problem.
 *
 * This mimics the behavior of runContractionProblem in tensile_host.cpp
 *
 * If an algo has not been provided, call getRocRollerBestSolutions to find one.
 *
 * Find the kernel to run in generatedKernels and execute it.
 *
 * @param handle
 * @param algo
 * @param prob
 * @return rocblaslt_status
 */
rocblaslt_status runRocRollerContractionProblem(rocblaslt_handle                   handle,
                                                const rocblaslt_matmul_algo*       algo,
                                                const RocblasltContractionProblem& prob)
{
    rocblaslt_matmul_heuristic_result heuristicResult;
    if(algo == nullptr)
    {
        int  returnAlgoCount;
        auto status
            = getRocRollerBestSolutions(handle, prob, 1, &heuristicResult, &returnAlgoCount);
        if(status != rocblaslt_status_success)
            return status;
        if(returnAlgoCount == 0)
        {
            return rocblaslt_status_not_implemented;
        }
        algo = &heuristicResult.algo;
    }

    // Get the values of static member variables flush and rotating size from UserClientArguments
    UserClientArguments ClientArguments;
    bool                flush              = ClientArguments.GetFlushValue();
    int32_t             rotatingBufferSize = ClientArguments.GetRotatingBufferSizeValue();
    int32_t             hotIterations      = ClientArguments.GetHotIterationsValue();
    int32_t             coldIterations     = ClientArguments.GetColdIterationsValue();

    int* solutionIndex = (int*)algo->data;

    if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
       || rocblaslt::Debug::Instance().printLogAsMarker())
    {
        logBench(prob, *solutionIndex, flush, rotatingBufferSize, coldIterations, hotIterations);
    }

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_profile)
    {
        logProfile(prob, flush, rotatingBufferSize, coldIterations, hotIterations);
    }

    std::shared_ptr<GemmKernel> kernel;
    auto                        status = getKernelFromAlgo(handle, prob, algo, kernel);
    if(status != rocblaslt_status_success)
        return status;

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_extended_profile)
    {
        auto kernelName = genKernelName(kernel->params);
        logExtendedProfile(prob,
                           *solutionIndex,
                           kernelName,
                           flush,
                           rotatingBufferSize,
                           coldIterations,
                           hotIterations);
    }

    return runGemmKernel(kernel, prob);
}
