// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm.hpp"
#include "runtime_args_selection.hpp"

#include "utility.hpp"

#include <rocRoller/KernelOptions_detail.hpp>

using namespace rocRoller;

namespace
{
    const int SHUFFLE_M = 16;
    const int SHUFFLE_N = 16;
    const int SHUFFLE_K = 32;
} // namespace

void RocRollerGemmKernel::setPredicates()
{
    using namespace rocRoller::Expression;

    // predicate building blocks
    // A sizes
    auto aSizes = std::get<Operations::Tensor>(*(command->findTag(tagTensorA))).sizes();
    std::vector<ExpressionPtr> aSizeExps(aSizes.size());
    std::transform(aSizes.begin(), aSizes.end(), aSizeExps.begin(), [](auto arg) {
        return arg->expression();
    });
    // B sizes
    auto bSizes = std::get<Operations::Tensor>(*(command->findTag(tagTensorB))).sizes();
    std::vector<ExpressionPtr> bSizeExps(bSizes.size());
    std::transform(bSizes.begin(), bSizes.end(), bSizeExps.begin(), [](auto arg) {
        return arg->expression();
    });

    // parameters
    auto workgroupTileMExp = literal(params->workgroupTile.m);
    auto workgroupTileNExp = literal(params->workgroupTile.n);
    auto workgroupTileKExp = literal(params->workgroupTile.k);

    // constants
    auto zero = literal(0u);

    // predicates
    std::stringstream ss;
    auto              unrollXPredicate = (aSizeExps[0] % workgroupTileMExp == zero);
    ss << "M must be a multiple of workgroupTile.m=" << params->workgroupTile.m;
    rocRoller::Expression::setComment(unrollXPredicate, ss.str());
    commandKernel->addPredicate(unrollXPredicate);
    ss.str("");

    auto unrollYPredicate = (bSizeExps[1] % workgroupTileNExp == zero);
    ss << "N must be a multiple of workgroupTile.n=" << params->workgroupTile.n;
    rocRoller::Expression::setComment(unrollYPredicate, ss.str());
    commandKernel->addPredicate(unrollYPredicate);
    ss.str("");

    auto unrollKPredicate = (aSizeExps[1] % workgroupTileKExp == zero);
    ss << "K must be a multiple of workgroupTile.k=" << params->workgroupTile.k;
    rocRoller::Expression::setComment(unrollKPredicate, ss.str());
    commandKernel->addPredicate(unrollKPredicate);
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
    rv << "RR_GEMM_" << (gemm->kernelType.transA ? "T" : "N")
       << (gemm->kernelType.transB ? "T" : "N");

    rv << "_";
    for(auto const& t : {gemm->kernelType.typeA,
                         gemm->kernelType.typeB,
                         gemm->kernelType.typeC,
                         gemm->kernelType.typeD,
                         gemm->kernelType.typeAcc})
        rv << toString(t) << "_";

    if(gemm->kernelType.scaleTypeA.mode != Operations::ScaleMode::None)
    {
        rv << "SA_" << genScaleModeString(gemm->kernelType.scaleTypeA.mode);
        rv << toString(gemm->kernelType.scaleTypeA.type) << "_";
        if(gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate)
        {
            rv << gemm->kernelType.scaleTypeA.blockRowSize;
            if(gemm->kernelType.scaleTypeA.blockColSize != 1)
                rv << "x" << gemm->kernelType.scaleTypeA.blockColSize;
            rv << "_";
            if(gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 3)
            {
                rv << "PSTA_" << gemm->kernelType.scaleTypeA.preSwizzleTile[0];
                rv << "x" << gemm->kernelType.scaleTypeA.preSwizzleTile[1];
                rv << "x" << gemm->kernelType.scaleTypeA.preSwizzleTile[2];
                rv << "_";
            }
            if(gemm->kernelType.scaleTypeA.preTile.size() == 2)
            {
                rv << "PTA_" << gemm->kernelType.scaleTypeA.preTile[0];
                rv << "x" << gemm->kernelType.scaleTypeA.preTile[1];
                rv << "_";
            }
        }
    }

    if(gemm->kernelType.scaleTypeB.mode != Operations::ScaleMode::None)
    {
        rv << "SB_" << genScaleModeString(gemm->kernelType.scaleTypeB.mode);
        rv << toString(gemm->kernelType.scaleTypeB.type) << "_";
        if(gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate)
        {
            rv << gemm->kernelType.scaleTypeB.blockColSize;
            if(gemm->kernelType.scaleTypeB.blockRowSize != 1)
                rv << "x" << gemm->kernelType.scaleTypeB.blockRowSize;
            rv << "_";

            if(gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 3)
            {
                rv << "PSTB_" << gemm->kernelType.scaleTypeB.preSwizzleTile[0];
                rv << "x" << gemm->kernelType.scaleTypeB.preSwizzleTile[1];
                rv << "x" << gemm->kernelType.scaleTypeB.preSwizzleTile[2];
                rv << "_";
            }
            if(gemm->kernelType.scaleTypeB.preTile.size() == 2)
            {
                rv << "PTB_" << gemm->kernelType.scaleTypeB.preTile[0];
                rv << "x" << gemm->kernelType.scaleTypeB.preTile[1];
                rv << "_";
            }
        }
    }

    if(gemm->streamK != StreamKMode::None)
    {
        rv << "SK_";
    }
    rv << "WGT_";
    rocRoller::streamJoin(
        rv, std::vector{gemm->workgroupTile.m, gemm->workgroupTile.n, gemm->workgroupTile.k}, "x");

    rv << "_UR_" << gemm->prefetchInFlight;

    if(gemm->workgroupMappingDim != -1)
    {
        rv << "_WGM_";
    }

    return rv.str();
}

std::shared_ptr<RocRollerGemmKernel> RocRollerGemmKernel::generate(std::shared_ptr<SolutionParameters> gemm)
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
        2, dataTypeA, {}, gemm->kernelType.transA ? oneStridesT : oneStridesN));

    auto loadInputA = tagTensorA;

    if(gemm->kernelType.swizzleA)
    {
        loadInputA = command->addOperation(Operations::SubTileTranspose(
            loadInputA, {SHUFFLE_M, SHUFFLE_K}, gemm->kernelType.transA));
    }

    auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadInputA));

    auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
        2, dataTypeB, {}, gemm->kernelType.transB ? oneStridesT : oneStridesN));

    auto loadInputB = tagTensorB;

    if(gemm->kernelType.swizzleB)
    {
        loadInputB = command->addOperation(Operations::SubTileTranspose(
            loadInputB, {SHUFFLE_K, SHUFFLE_N}, gemm->kernelType.transB));
    }

    auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadInputB));

    auto mulInputA = tagLoadA;
    auto mulInputB = tagLoadB;

    AssertFatal(gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::None
                    || gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::SingleScale
                    || gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate,
                "Scale mode not supported!",
                ShowValue(gemm->kernelType.scaleTypeA.mode));
    AssertFatal(gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::None
                    || gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::SingleScale
                    || gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate,
                "Scale mode not supported!",
                ShowValue(gemm->kernelType.scaleTypeB.mode));

    std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
        tagTensorScaleB, tagLoadScaleB, tagBlockScaleB, tagSKGrid, tagWGM;
    std::map<Operations::ScratchPolicy, Operations::OperationTag> tagScratch;

    if(gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleA = command->addOperation(
            rocRoller::Operations::Tensor(2,
                                          gemm->kernelType.scaleTypeA.type,
                                          {},
                                          gemm->kernelType.transA ? oneStridesT : oneStridesN));
        Operations::OperationTag loadScaleInputA = *tagTensorScaleA;
        if(gemm->kernelType.scaleTypeA.preTile.size() == 2)
        {
            AssertFatal(gemm->kernelType.transA, "Can only pre-tile A if it is transposed");
            loadScaleInputA = command->addOperation(rocRoller::Operations::SubTileTranspose(
                loadScaleInputA,
                gemm->kernelType.scaleTypeA.preTile,
                gemm->kernelType.transA));
        }
        tagLoadScaleA
            = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadScaleInputA));

        auto scaleInputA = tagLoadScaleA;

        auto validPreSwizzleTileA = gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 3
                                    || gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 0;
        AssertFatal(validPreSwizzleTileA,
                    "Invalid preSwizzleTile",
                    ShowValue(gemm->kernelType.scaleTypeA.preSwizzleTile));

        // auto validSwizzleTileA = gemm->swizzleTileSize.m == gemm->swizzleTileSize.n && gemm->swizzleTileSize.k == gemm->swizzleTileSize.l && gemm->swizzleTile.

        if(gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 3)
        {
            AssertFatal(gemm->kernelType.scaleTypeA.preSwizzleTile[0] == 32
                            && gemm->kernelType.scaleTypeA.preSwizzleTile[1] == 8
                            && gemm->kernelType.scaleTypeA.preSwizzleTile[2] == 4,
                        "Pre-swizzled scale tile is not compatible with swizzle tile A",
                        ShowValue(gemm->kernelType.scaleTypeA.preSwizzleTile));
            auto validSwizzleTileA = gemm->swizzleTileSize.m == 32 && gemm->swizzleTileSize.k == 8;
            AssertFatal(validSwizzleTileA,
                        "Pre-swizzled scale tile is not compatible with swizzle tile A",
                        ShowValue(gemm->swizzleTileSize.m),
                        ShowValue(gemm->swizzleTileSize.k));

            scaleInputA = command->addOperation(rocRoller::Operations::SubTileTranspose(
                *tagLoadScaleA, gemm->kernelType.scaleTypeA.preSwizzleTile));
        }

        tagBlockScaleA = mulInputA = command->addOperation(rocRoller::Operations::BlockScale(
            tagLoadA,
            2,
            scaleInputA,
            {gemm->kernelType.scaleTypeA.blockColSize, gemm->kernelType.scaleTypeA.blockRowSize}));
    }

    if(gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleB = command->addOperation(
            rocRoller::Operations::Tensor(2,
                                          gemm->kernelType.scaleTypeB.type,
                                          {},
                                          gemm->kernelType.transB ? oneStridesT : oneStridesN));
        Operations::OperationTag loadScaleInputB = *tagTensorScaleB;
        if(gemm->kernelType.scaleTypeB.preTile.size() == 2)
        {
            AssertFatal(!gemm->kernelType.transB, "Can only pre-tile B if it is not transposed");
            loadScaleInputB = command->addOperation(rocRoller::Operations::SubTileTranspose(
                loadScaleInputB,
                gemm->kernelType.scaleTypeB.preTile,
                gemm->kernelType.transB));
        }
        tagLoadScaleB
            = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadScaleInputB));

        auto scaleInputB = tagLoadScaleB;

        auto validPreSwizzleTileB = gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 3
                                    || gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 0;
        AssertFatal(validPreSwizzleTileB,
                    "Invalid preSwizzleTile",
                    ShowValue(gemm->kernelType.scaleTypeB.preSwizzleTile));

        if(gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 3)
        {
            AssertFatal(gemm->kernelType.scaleTypeB.preSwizzleTile[0] == 32
                            && gemm->kernelType.scaleTypeB.preSwizzleTile[1] == 8
                            && gemm->kernelType.scaleTypeB.preSwizzleTile[2] == 4,
                        "Pre-swizzled scale tile is not compatible with swizzle tile B",
                        ShowValue(gemm->kernelType.scaleTypeB.preSwizzleTile));
            auto validSwizzleTileB = gemm->swizzleTileSize.n == 32 && gemm->swizzleTileSize.l == 8;
            AssertFatal(validSwizzleTileB,
                        "Pre-swizzled scale tile is not compatible with swizzle tile B",
                        ShowValue(gemm->swizzleTileSize.n),
                        ShowValue(gemm->swizzleTileSize.l));
            scaleInputB = command->addOperation(rocRoller::Operations::SubTileTranspose(
                *tagLoadScaleB, gemm->kernelType.scaleTypeB.preSwizzleTile));
        }

        tagBlockScaleB = mulInputB = command->addOperation(rocRoller::Operations::BlockScale(
            tagLoadB,
            2,
            scaleInputB,
            {gemm->kernelType.scaleTypeB.blockColSize, gemm->kernelType.scaleTypeB.blockRowSize}));
    }

    auto tagTensorC
        = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeC, {}, oneStridesN)); // C
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
        = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD, {}, oneStridesN)); // D
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

    if(gemm->streamK != StreamKMode::None)
    {
        tagSKGrid = command->allocateTag();
        command->allocateArgument(DataType::UInt32,
                                  *tagSKGrid,
                                  ArgumentType::Value,
                                  DataDirection::ReadOnly,
                                  rocRoller::NUMWGS);

        // Create Scratch operations for each ScratchPolicy
        for(int i = 0; i < static_cast<int>(Operations::ScratchPolicy::Count); ++i)
        {
            auto policy = static_cast<Operations::ScratchPolicy>(i);
            auto tag    = command->allocateTag();
            command->addOperation(Operations::Scratch(tag, policy));
            command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                      tag,
                                      ArgumentType::Value,
                                      DataDirection::ReadWrite,
                                      rocRoller::getScratchName(policy));
            tagScratch[policy] = tag;
        }
    }

    if(gemm->workgroupMappingDim != -1)
    {
        tagWGM = command->allocateTag();
        command->allocateArgument(
            DataType::Int32, *tagWGM, ArgumentType::Value, DataDirection::ReadOnly, rocRoller::WGM);
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
        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.k},
            LayoutType::MATRIX_A,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k,
             gemm->machineInstruction.b},
            GetMemoryType(gemm->loadPathA));
        params->setDimensionInfo(tagLoadA, macTileA);
    }

    if(gemm->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate)
    {
        // TODO: verify the division of scale block size is correct
        auto const scaleBlockSize
            = gemm->kernelType.scaleTypeA.blockRowSize * gemm->kernelType.scaleTypeA.blockColSize;

        auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.m, gemm->workgroupTile.k / (int)scaleBlockSize},
            LayoutType::MATRIX_A,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k / (int)scaleBlockSize,
             gemm->machineInstruction.b},
            GetMemoryType(gemm->loadPathAScale),
            {}, // miTileSizes - use default (same as subTileSizes)
            {gemm->swizzleTileSize.m, gemm->swizzleTileSize.n, gemm->swizzleTileSize.k, 1});
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
            GetMemoryType(gemm->loadPathB));
        params->setDimensionInfo(tagLoadB, macTileB);
    }

    if(gemm->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate)
    {
        // TODO: verify the division of scale block size is correct
        auto const scaleBlockSize
            = gemm->kernelType.scaleTypeB.blockRowSize * gemm->kernelType.scaleTypeB.blockColSize;

        auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
            {gemm->workgroupTile.k / (int)scaleBlockSize, gemm->workgroupTile.n},
            LayoutType::MATRIX_B,
            {gemm->machineInstruction.m,
             gemm->machineInstruction.n,
             gemm->machineInstruction.k / (int)scaleBlockSize,
             gemm->machineInstruction.b},
            GetMemoryType(gemm->loadPathBScale),
            {}, // miTileSizes - use default (same as subTileSizes)
            {gemm->swizzleTileSize.m, gemm->swizzleTileSize.n, gemm->swizzleTileSize.l, 1});
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
            Parameters::Solution::IsLDSStore(gemm->storePath) ? MemoryType::WAVE_LDS
                                                              : MemoryType::WAVE);
        params->setDimensionInfo(tagStoreD, macTileD);
    }

    params->swizzleScale  = gemm->swizzleScale;
    params->prefetchScale = gemm->prefetchScale;
    params->tailLoops     = gemm->tailLoops;

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

    params->transposeMemoryAccess.set(LayoutType::MATRIX_A, gemm->kernelType.transA);
    params->transposeMemoryAccess.set(LayoutType::MATRIX_B, gemm->kernelType.transB);

    uint workgroupSizeX = gemm->workgroupSizeX * gemm->workgroupSizeY;
    uint workgroupSizeY = 1;

    // Workgroup Mapping
    if(gemm->workgroupMappingDim != -1)
    {
        auto dim = gemm->workgroupMappingDim;

        AssertFatal(dim == 0 || dim == 1,
                    "Only 0 (M) or 1 (N) are supported dimensions for workgroup mapping.",
                    ShowValue(dim));

        params->workgroupMappingDim = dim;
    }

    if(gemm->workgroupRemapXCC)
    {
        params->workgroupRemapXCC = 8;
    }

    if(gemm->streamK != StreamKMode::None)
    {
        params->streamK = gemm->streamK;

        params->loopOverOutputTilesDimensions = {0, 1};
    }

    params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
    params->setManualWavefrontCount(
        {static_cast<uint>(gemm->workgroupTile.m / gemm->machineInstruction.m
                           / wavetilePerWavefrontM),
         static_cast<uint>(gemm->workgroupTile.n / gemm->machineInstruction.n
                           / wavetilePerWavefrontN)});

    AssertFatal(gemm->kernelType.scaleTypeA.preSwizzleTile.size()
                    == gemm->kernelType.scaleTypeB.preSwizzleTile.size(),
                "A and B must have the same shuffle parameter");

    // -------------------------------------------------------------
    // Create CommandKernel

    std::string kernelName = genKernelName(gemm);
    auto context = Context::ForDefaultHipDevice(
        kernelName,
        {{.scaleSkipPermlane = (gemm->kernelType.scaleTypeA.preSwizzleTile.size() == 3
                                && gemm->kernelType.scaleTypeB.preSwizzleTile.size() == 3)
                                   ? ScaleSkipPermlaneMode::PreSwizzleScaleGFX950
                                   : ScaleSkipPermlaneMode::None}});
    auto commandKernel = std::make_shared<CommandKernel>(command, kernelName);
    commandKernel->setContext(context);
    commandKernel->setCommandParameters(params);
    commandKernel->generateKernel();
    commandKernel->loadKernel();

    // -------------------------------------------------------------
    // Create GemmKernel

    auto gemmKernel           = std::make_shared<RocRollerGemmKernel>();
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

    gemmKernel->tagScratch = tagScratch;

    if(tagSKGrid)
        gemmKernel->tagSKGrid = *tagSKGrid;

    if(tagWGM)
        gemmKernel->tagWGM = *tagWGM;

    gemmKernel->setPredicates();

    auto flatWorkgroupSize = workgroupSizeX;
    int  occupancy;
    AssertFatal(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occupancy, commandKernel->getHipFunction(), flatWorkgroupSize, 0)
                == (hipError_t)HIP_SUCCESS);

    gemmKernel->occupancy = occupancy;

    return gemmKernel;
}

size_t RocRollerGemmKernel::workspaceRequired(const RocblasltContractionProblem& prob)
{
    CommandArguments commandArgs = command->createArguments();

    if(params->streamK != StreamKMode::None)
    {
        commandArgs.setArgument(
            tagSKGrid, ArgumentType::Value, chooseStreamKGridSize(shared_from_this(), prob));
    }

    auto runtimeArgs = commandArgs.runtimeArguments();

    // Only return scratch space for ScratchPolicy::None (uses prob.workspace)
    return commandKernel->scratchSpaceRequired(Operations::ScratchPolicy::None, runtimeArgs);
}

bool RocRollerGemmKernel::exceedsBufferAddressingLimit(
    const RocblasltContractionProblem& prob) const
{
    // rocRoller addresses each tensor through one AMD buffer descriptor whose num_records
    // (byte-count) field is 32 bits: it is set to ToBytes(tensor size, elementBits) and packed
    // into the low 32 bits by BufferDescriptor::SetSize (see rocRoller's AssignIndexExpressions
    // and CodeGen/Buffer). A byte extent >= 4 GiB therefore wraps num_records, after which
    // out-of-bounds accesses are silently dropped (stores) or read back as 0 (loads) -- the
    // kernel runs but produces an all-zero/wrong result. Detect that here so such problems are
    // reported unsupported ("no solution found") instead of returning silently-wrong output.
    //
    // The byte extent below mirrors num_records exactly because both use elementBits (note:
    // not sizeof(T), which differs for sub-byte types such as FP4/FP6). Extents are packed:
    // createCommandArguments builds {M,N}/{M,K}/{K,N} descriptors with no leading-dimension
    // padding, and batch_count is always 1 on this path (enforced in getRocRollerBestSolutions).
    constexpr size_t kMaxBufferBytes = size_t(1) << 32; // 4 GiB
    auto             bytesOf = [](size_t elems, rocRoller::DataType t) -> size_t {
        return (elems * rocRoller::DataTypeInfo::Get(t).elementBits + 7u) / 8u; // ceil for sub-byte types
    };
    const size_t M = prob.m, N = prob.n, K = prob.k;
    auto         over = [&](const char* name, size_t bytes) -> bool {
        if(bytes < kMaxBufferBytes)
            return false;
        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            msg << "rocRoller cannot address operand " << name << " (" << bytes
                << " bytes): a single buffer descriptor is limited to < 4 GiB (32-bit "
                   "num_records). Reporting problem as unsupported.";
            log_info("exceedsBufferAddressingLimit", msg.str());
        }
        return true;
    };
    return over("A", bytesOf(M * K, params->kernelType.typeA))
           || over("B", bytesOf(K * N, params->kernelType.typeB))
           || over("C", bytesOf(M * N, params->kernelType.typeC))
           || over("D", bytesOf(M * N, params->kernelType.typeD));
}

bool RocRollerGemmKernel::isSupportedProblem(const RocblasltContractionProblem& prob)
{
    if(exceedsBufferAddressingLimit(prob))
        return false;

    auto workSpaceRequired = this->workspaceRequired(prob);

    if(workSpaceRequired > prob.workspaceSize)
        return false;

    auto commandArgs = createCommandArguments(prob, DEFAULT_WGM);
    auto runtimeArgs = commandArgs.runtimeArguments();

    return commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error);
}

CommandArguments RocRollerGemmKernel::createCommandArguments(const RocblasltContractionProblem& prob,
                                                    int                                wgm)
{
    CommandArguments commandArgs = command->createArguments();

    size_t M = prob.m;
    size_t N = prob.n;
    size_t K = prob.k;

    TensorDescriptor descA(
        params->kernelType.typeA, {M, K}, params->kernelType.transA ? "T" : "N");
    TensorDescriptor descB(
        params->kernelType.typeB, {K, N}, params->kernelType.transB ? "T" : "N");

    // TODO: Have to typecast void* pointer to something that CommandArgumentValue accepts
    setCommandTensorArg(commandArgs, tagTensorA, descA, (float*)nullptr);
    setCommandTensorArg(commandArgs, tagTensorB, descB, (float*)nullptr);

    if(params->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = params->kernelType.scaleTypeA.blockRowSize
                                    * params->kernelType.scaleTypeA.blockColSize;
        TensorDescriptor descAScale(
            params->kernelType.scaleTypeA.type,
            {size_t(M), size_t(K / scaleBlockSize)},
            params->kernelType.transA ? "T" : "N");
        setCommandTensorArg(commandArgs, tagTensorScaleA, descAScale, (float*)nullptr);
    }
    if(params->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = params->kernelType.scaleTypeB.blockRowSize
                                    * params->kernelType.scaleTypeB.blockColSize;
        TensorDescriptor descBScale(
            params->kernelType.scaleTypeB.type,
            {size_t(K / scaleBlockSize), size_t(N)},
            params->kernelType.transB ? "T" : "N");
        setCommandTensorArg(commandArgs, tagTensorScaleB, descBScale, (float*)nullptr);
    }

    TensorDescriptor descC(params->kernelType.typeC, {M, N}, "N");
    setCommandTensorArg(commandArgs, tagTensorC, descC, (float*)nullptr);

    commandArgs.setArgument(tagScalarAlpha, ArgumentType::Value, *((float*)prob.alpha));
    commandArgs.setArgument(tagScalarBeta, ArgumentType::Value, *((float*)prob.beta));

    TensorDescriptor descD(params->kernelType.typeD, {M, N}, "N");
    setCommandTensorArg(commandArgs, tagTensorD, descD, (float*)nullptr);

    commandArgs.setArgument(tagTensorA, ArgumentType::Value, (float*)prob.A);
    commandArgs.setArgument(tagTensorB, ArgumentType::Value, (float*)prob.B);
    commandArgs.setArgument(tagTensorC, ArgumentType::Value, (float*)prob.C);
    commandArgs.setArgument(tagTensorD, ArgumentType::Value, (float*)prob.D);

    if(params->kernelType.scaleTypeA.mode == Operations::ScaleMode::Separate)
    {
        commandArgs.setArgument(tagTensorScaleA, ArgumentType::Value, (uint8_t*)prob.scaleA);
    }

    if(params->kernelType.scaleTypeB.mode == Operations::ScaleMode::Separate)
    {
        commandArgs.setArgument(tagTensorScaleB, ArgumentType::Value, (uint8_t*)prob.scaleB);
    }

    if(params->workgroupMappingDim != -1)
    {
        AssertFatal(
            wgm > 0, "Workgroup mapping size must be a positive non-zero integer.", ShowValue(wgm));

        commandArgs.setArgument(tagWGM, ArgumentType::Value, wgm);
    }

    if(params->streamK != StreamKMode::None)
    {
        commandArgs.setArgument(
            tagSKGrid, ArgumentType::Value, chooseStreamKGridSize(shared_from_this(), prob));
    }

    return commandArgs;
}

rocblaslt_status RocRollerGemmKernel::run(const RocblasltContractionProblem& prob)
{
    // Also guard the execution path: an algo selected for a smaller problem (or resolved by
    // index) can be replayed on a larger one, bypassing isSupportedProblem.
    if(exceedsBufferAddressingLimit(prob))
        return rocblaslt_status_invalid_value;

    auto workSpaceRequired = this->workspaceRequired(prob);

    if(workSpaceRequired > prob.workspaceSize)
    {
        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            msg << "Input workspace size " << prob.workspaceSize
                << " is less than the required workspace size ";
            msg << workSpaceRequired << std::endl;
            log_info(__func__, msg.str());
        }
        return rocblaslt_status_invalid_value;
    }
    auto commandArgs = createCommandArguments(prob, DEFAULT_WGM);

    if(params->streamK != StreamKMode::None)
    {
        auto runtimeArgs = commandArgs.runtimeArguments();

        // Use prob.workspace for ScratchPolicy::None
        auto noneScratchSize = commandKernel->scratchSpaceRequired(
            Operations::ScratchPolicy::None, runtimeArgs);
        if(noneScratchSize > 0 && prob.workspace != nullptr)
        {
            commandArgs.setArgument(tagScratch.at(Operations::ScratchPolicy::None),
                                    ArgumentType::Value,
                                    static_cast<unsigned char*>(prob.workspace));
        }

        // Use prob.Synchronizer for ScratchPolicy::ZeroedBeforeAndAfter
        auto zeroedScratchSize = commandKernel->scratchSpaceRequired(
            Operations::ScratchPolicy::ZeroedBeforeAndAfter, runtimeArgs);
        if(zeroedScratchSize > 0 && prob.Synchronizer != nullptr)
        {
            commandArgs.setArgument(
                tagScratch.at(Operations::ScratchPolicy::ZeroedBeforeAndAfter),
                ArgumentType::Value,
                static_cast<unsigned char*>(prob.Synchronizer));
        }
    }

    auto runtimeArgs = commandArgs.runtimeArguments();

    if(!commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error))
    {
        return rocblaslt_status_invalid_value;
    }

    commandKernel->launchKernel(runtimeArgs, prob.stream);

    return rocblaslt_status_success;
}
