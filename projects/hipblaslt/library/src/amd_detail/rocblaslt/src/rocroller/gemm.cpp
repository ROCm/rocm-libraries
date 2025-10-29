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

#include "gemm.hpp"
#include "runtime_args_selection.hpp"

#include "utility.hpp"

using namespace rocRoller;

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
    rv << "RR_GEMM_" << (gemm->kernelType.transA ? "T" : "N")
       << (gemm->kernelType.transB ? "T" : "N");

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
        2, dataTypeA, gemm->kernelType.transA ? oneStridesT : oneStridesN)); // A
    auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

    auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
        2, dataTypeB, gemm->kernelType.transB ? oneStridesT : oneStridesN)); // B
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
        tagTensorScaleB, tagLoadScaleB, tagBlockScaleB, tagScratch, tagSKGrid, tagWGM;

    if(gemm->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
            2,
            gemm->kernelType.scaleTypeA,
            gemm->kernelType.transA ? oneStridesT : oneStridesN));
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
            gemm->kernelType.transB ? oneStridesT : oneStridesN));
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

    if (gemm->streamK)
    {
        tagSKGrid = command->allocateTag();
        command->allocateArgument(DataType::UInt32,
                                *tagSKGrid,
                                ArgumentType::Value,
                                DataDirection::ReadOnly,
                                rocRoller::NUMWGS);

        tagScratch = command->allocateTag();
        command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                *tagScratch,
                                ArgumentType::Value,
                                DataDirection::ReadWrite,
                                rocRoller::SCRATCH);
    }

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
            GetMemoryType(gemm->loadPathB));
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

    params->transposeMemoryAccess.set(LayoutType::MATRIX_A, gemm->kernelType.transA);
    params->transposeMemoryAccess.set(LayoutType::MATRIX_B, gemm->kernelType.transB);

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

        params->workgroupMappingDim = dim;
    }

    if(gemm->workgroupRemapXCC)
    {
        params->workgroupRemapXCC = 8;
    }

    if(gemm->streamK)
    {
        StreamKMode streamKMode = StreamKMode::Standard;
        if(gemm->streamKTwoTile)
            streamKMode = StreamKMode::TwoTile;
        params->streamK = streamKMode;

        params->loopOverOutputTilesDimensions = {0, 1};
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
    commandKernel->loadKernel();

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

    if(tagScratch)
        gemmKernel->tagScratch = *tagScratch;

    if(tagSKGrid)
        gemmKernel->tagSKGrid = *tagSKGrid;

    if(tagWGM)
        gemmKernel->tagWGM = *tagWGM;

    setPredicates(gemmKernel);

    auto flatWorkgroupSize = workgroupSizeX;
    int occupancy;
    AssertFatal(
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, commandKernel->getHipFunction(), flatWorkgroupSize, 0)
        == (hipError_t)HIP_SUCCESS);

    gemmKernel->occupancy = occupancy;

    return gemmKernel;
}

size_t workspaceRequired(std::shared_ptr<GemmKernel> gemm, const RocblasltContractionProblem& prob)
{
    CommandArguments commandArgs = gemm->command->createArguments();

    if(gemm->params->streamK)
    {
        commandArgs.setArgument(gemm->tagSKGrid, ArgumentType::Value, chooseStreamKGridSize(gemm, prob));
    }

    auto runtimeArgs = commandArgs.runtimeArguments();

    return gemm->commandKernel->scratchSpaceRequired(runtimeArgs);
}

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
                           gemm->params->kernelType.transA ? "T" : "N");
    TensorDescriptor descB(gemm->params->kernelType.typeB,
                           {K, N},
                           gemm->params->kernelType.transB ? "T" : "N");

    // TODO: Have to typecast void* pointer to something that CommandArgumentValue accepts
    setCommandTensorArg(commandArgs, gemm->tagTensorA, descA, (float*)nullptr);
    setCommandTensorArg(commandArgs, gemm->tagTensorB, descB, (float*)nullptr);

    if(gemm->params->kernelType.scaleAMode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = prob.scaleABlockRowSize * prob.scaleABlockColSize;
        TensorDescriptor descAScale(gemm->params->kernelType.typeA,
                                    {size_t(M), size_t(K / scaleBlockSize)},
                                    gemm->params->kernelType.transA ? "T" : "N");
        setCommandTensorArg(commandArgs, gemm->tagTensorScaleA, descAScale, (float*)nullptr);
    }
    if(gemm->params->kernelType.scaleBMode == Operations::ScaleMode::Separate)
    {
        auto const scaleBlockSize = prob.scaleBBlockRowSize * prob.scaleBBlockColSize;
        TensorDescriptor descBScale(gemm->params->kernelType.typeB,
                                    {size_t(K / scaleBlockSize), size_t(N)},
                                    gemm->params->kernelType.transB ? "T" : "N");
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

    if(gemm->params->streamK)
    {
        commandArgs.setArgument(gemm->tagSKGrid, ArgumentType::Value, chooseStreamKGridSize(gemm, prob));
    }

    return commandArgs;
}

rocblaslt_status runGemmKernel(std::shared_ptr<GemmKernel>        gemm,
                               const RocblasltContractionProblem& prob)
{
    auto workSpaceRequired = workspaceRequired(gemm, prob);

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

    auto commandArgs = createCommandArguments(gemm, prob, DEFAULT_WGM);

    // Add scratch space
    if(workSpaceRequired > 0)
    {
        commandArgs.setArgument(
            gemm->tagScratch, ArgumentType::Value, static_cast<unsigned char*>(prob.workspace));
    }

    auto runtimeArgs = commandArgs.runtimeArguments();

    if(!gemm->commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error))
    {
        return rocblaslt_status_invalid_value;
    }

    gemm->commandKernel->launchKernel(runtimeArgs, prob.stream);
    return rocblaslt_status_success;
}
