#include "rocroller_host.hpp"
#include "handle.h"
#include "utility.hpp"

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

using namespace rocRoller;

const int MAX_BITS_WORKGROUPTILE_M = 8;
const int MAX_BITS_WORKGROUPTILE_N = 8;
const int MAX_BITS_WORKGROUPTILE_K = 7;
const int REQUIRED_MULTIPLE_M_N    = 16;
const int REQUIRED_MULTIPLE_K      = 32;

/**
 * @brief KernelType
 *
 * All of the values required for different types of kernels.
 * This should not include any optimization flags.
 *
 */
struct KernelType
{
    rocRoller::DataType typeA;
    rocRoller::DataType typeB;
    rocRoller::DataType typeC;
    rocRoller::DataType typeD;
    rocRoller::DataType typeAcc = rocRoller::DataType::Float;

    hipblasOperation_t transA;
    hipblasOperation_t transB;

    rocRoller::Operations::ScaleMode scaleAMode;
    rocRoller::Operations::ScaleMode scaleBMode;

    size_t scaleABlockRowSize = 32u;
    size_t scaleABlockColSize = 1u;
    size_t scaleBBlockRowSize = 1u;
    size_t scaleBBlockColSize = 32u;

    auto operator<=>(const KernelType& other) const = default;
};

/**
 * @brief WorkGroupTileSize
 *
 * The size of a tile that will be executed by a work group.
 *
 */
struct WorkGroupTileSize
{
    int m;
    int n;
    int k;
};

/**
 * @brief MachineInstructionSize
 *
 * The machine instruction that will be used for matrix multiplication operations
 *
 */
struct MachineInstructionSize
{
    int m = -1;
    int n = -1;
    int k = -1;
    int b = -1;
};

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
    bool loadLDSA  = true;
    bool loadLDSB  = true;
    bool storeLDSD = true;

    bool prefetch          = true;
    int  prefetchInFlight  = 2;
    int  prefetchLDSFactor = 2;
    bool betaInFma         = true;

    // Unroll Options
    unsigned int unrollX = 0;
    unsigned int unrollY = 0;

    std::string scheduler;

    bool streamK        = false;
    bool streamKTwoTile = false;

    bool loadLDSScaleA = false;
    bool loadLDSScaleB = false;
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

    return result;
}

inline std::string scaleModeOption(std::string                                arg,
                                   RocblasltContractionProblem::ScalingFormat scale)
{
    switch(scale)
    {
    case RocblasltContractionProblem::ScalingFormat::Scalar:
        return arg + " 1";
    case RocblasltContractionProblem::ScalingFormat::Vector:
        return arg + " 2";
    case RocblasltContractionProblem::ScalingFormat::Block:
        return arg + " 3";
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
    log_bench(__func__,
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
              scaleModeOption("--scaleA", prob.scaleAType),
              scaleModeOption("--scaleB", prob.scaleBType),
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

const std::vector<WorkGroupTileSize> possibleTileSizes = {
    {256, 256, 64}, {256, 128, 64}, {128, 256, 64}, {256, 64, 64}, {64, 256, 64},  {128, 128, 64},
    {256, 32, 64},  {32, 256, 64},  {128, 64, 64},  {64, 128, 64}, {256, 16, 128}, {16, 256, 128},
    {128, 32, 64},  {32, 128, 64},  {64, 64, 64},   {64, 32, 64},  {32, 64, 64},   {64, 16, 128},
    {16, 64, 128},  {32, 32, 64},   {32, 16, 128},  {16, 32, 128}, {16, 16, 128}};

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

    for(auto const& wgt : possibleTileSizes)
    {
        if((requestedAlgoCount == -1)
           || (prob.m % wgt.m == 0 && prob.n % wgt.n == 0 && prob.k % (wgt.k * 2) == 0))
        {
            params.emplace_back(wgt);

            if(kernelType.typeA == rocRoller::DataType::Half
               || kernelType.typeA == rocRoller::DataType::BFloat16
               || kernelType.typeA == rocRoller::DataType::Float)
            {
                params.back().workgroupTile.k = 32;
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

    // Choose the Machine Instruction to use
    if(gemm->kernelType.typeA == rocRoller::DataType::Half
       || gemm->kernelType.typeA == rocRoller::DataType::BFloat16)
    {
        gemm->machineInstruction = {32, 32, 8, 1};
    }
    else if(gemm->kernelType.typeA == rocRoller::DataType::Float)
    {
        gemm->machineInstruction = {32, 32, 2, 1};
    }
    else
    {
        if(gemm->workgroupTile.m % 32 == 0 && gemm->workgroupTile.n % 32 == 0)
            gemm->machineInstruction = {32, 32, 64, 1};
        else
            gemm->machineInstruction = {16, 16, 128, 1};
    }

    if(gemm->workgroupTile.m / gemm->machineInstruction.m == 1)
        gemm->workgroupSizeX = gemm->wavefrontSize;
    if(gemm->workgroupTile.n / gemm->machineInstruction.n == 1)
        gemm->workgroupSizeY = 1;

    // LDS can only be used for scaling data with certain workgroup tile sizes
    auto workgroupSize = gemm->workgroupSizeX * gemm->workgroupSizeY;
    auto numScaleElementsA
        = gemm->workgroupTile.m
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleABlockRowSize * gemm->kernelType.scaleABlockColSize));
    auto numScaleElementsB
        = gemm->workgroupTile.n
          * (gemm->workgroupTile.k
             / (gemm->kernelType.scaleBBlockRowSize * gemm->kernelType.scaleBBlockColSize));
    if(numScaleElementsA % workgroupSize != 0)
        gemm->loadLDSScaleA = false;
    if(numScaleElementsB % workgroupSize != 0)
        gemm->loadLDSScaleB = false;

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

    AssertFatal(gemm->kernelType.scaleAMode == gemm->kernelType.scaleBMode,
                "Scale modes must match",
                ShowValue(gemm->kernelType.scaleAMode),
                ShowValue(gemm->kernelType.scaleBMode));
    AssertFatal(gemm->kernelType.scaleAMode == Operations::ScaleMode::None
                    || gemm->kernelType.scaleAMode == Operations::ScaleMode::SingleScale
                    || gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate,
                "Scale mode not supported!",
                ShowValue(gemm->kernelType.scaleAMode));

    std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
        tagTensorScaleB, tagLoadScaleB, tagBlockScaleB;

    if(gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
            2,
            DataType::UInt8,
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
            DataType::UInt8,
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
        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.k},
            LayoutType::MATRIX_A,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            gemm->loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
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
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.k, gemm->workgroupTile.n},
            LayoutType::MATRIX_B,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            gemm->loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
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

    params->unrollX = gemm->unrollX;
    params->unrollY = gemm->unrollY;

    if(gemm->prefetch)
    {
        params->prefetch          = true;
        params->unrollK           = gemm->prefetchInFlight;
        params->prefetchInFlight  = gemm->prefetchInFlight;
        params->prefetchLDSFactor = gemm->prefetchLDSFactor;

        params->prefetchMixMemOps = false;

        if(gemm->prefetchLDSFactor != 0)
        {
            params->prefetchMixMemOps = true;
        }

        if((gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate && !gemm->loadLDSScaleA)
           || (gemm->kernelType.scaleBMode == Operations::ScaleMode::Separate
               && !gemm->loadLDSScaleB))
        {
            params->prefetchMixMemOps = false;
        }
    }
    else
    {
        params->prefetch = false;
    }

    params->transposeMemoryAccess[LayoutType::MATRIX_A] = gemm->kernelType.transA == HIPBLAS_OP_T;
    params->transposeMemoryAccess[LayoutType::MATRIX_B] = gemm->kernelType.transB == HIPBLAS_OP_T;

    uint workgroupSizeX = gemm->workgroupSizeX * gemm->workgroupSizeY;
    uint workgroupSizeY = 1;

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
    int returnAlgoCount;
    return getRocRollerBestSolutions(handle, prob, -1, heuristicResults.data(), &returnAlgoCount);
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
                                        const RocblasltContractionProblem& prob)
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

    auto commandArgs = createCommandArguments(kernel, prob);

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
    auto commandArgs = createCommandArguments(gemm, prob);

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

    int* solutionIndex = (int*)algo->data;

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
    {
        // TODO: Fill in other parameters after merge with
        //       mainline.
        logBench(prob, *solutionIndex, false, 0, 0, 0);
    }

    std::shared_ptr<GemmKernel> kernel;
    auto                        status = getKernelFromAlgo(handle, prob, algo, kernel);
    if(status != rocblaslt_status_success)
        return status;

    return runGemmKernel(kernel, prob);
}
