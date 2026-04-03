// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include "client/GEMMParameters.hpp"
#include "client/GEMMSolution.hpp"
#include "client/visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            class DataParallelGEMMSolution : public GEMMSolution
            {
                Operations::OperationTag                m_tagA, m_tagB, m_tagC, m_tagD;
                std::optional<Operations::OperationTag> m_tagCvt;
                Operations::OperationTag m_tagTensorA, m_tagTensorB, m_tagTensorC, m_tagScalarAlpha,
                    m_tagScalarBeta, m_tagTensorD;

                std::optional<Operations::OperationTag> m_tagTensorScaleA, m_tagLoadScaleA,
                    m_tagBlockScaleA, m_tagTensorScaleB, m_tagLoadScaleB, m_tagBlockScaleB;

                Operations::OperationTag m_tagWGM;

            public:
                using GEMMSolution::GEMMSolution;

                ABCDTags getABCDTags() const override
                {
                    return {m_tagTensorA, m_tagTensorB, m_tagTensorC, m_tagTensorD};
                }

                ABScaleTags getABScaleTags() const override
                {
                    return {m_tagTensorScaleA, m_tagTensorScaleB};
                }

            protected:
                CommandPtr makeCommand(SolutionParameters const& solutionParams) override
                {
                    auto command = std::make_shared<Command>();

                    auto typeA   = fromString<DataType>(solutionParams.types.typeA);
                    auto typeB   = fromString<DataType>(solutionParams.types.typeB);
                    auto typeC   = fromString<DataType>(solutionParams.types.typeC);
                    auto typeD   = fromString<DataType>(solutionParams.types.typeD);
                    auto typeAcc = fromString<DataType>(solutionParams.types.typeAcc);

                    auto unitStrides = [](TransposeType t) -> std::vector<size_t> {
                        switch(t)
                        {
                        case TransposeType::T:
                            return {(size_t)0, (size_t)1};
                        case TransposeType::N:
                            return {(size_t)1};
                        default:
                            Throw<FatalError>("Bad transpose option");
                        }
                    };

                    m_tagTensorA = command->addOperation(
                        Operations::Tensor(2, typeA, {}, unitStrides(solutionParams.types.transA)));

                    auto loadInputA = m_tagTensorA;

                    auto pretileA = not solutionParams.types.pretileA.empty();
                    if(pretileA)
                    {
                        AssertFatal(solutionParams.types.transA
                                        == Client::GEMMClient::TransposeType::T,
                                    "Pretiling A is only supported when A is TransposeType::T.",
                                    ShowValue(solutionParams.types.transA));

                        AssertFatal(solutionParams.types.pretileA.size() == 2,
                                    "pretileA must have size 2 (MxK tile dimensions).",
                                    ShowValue(solutionParams.types.pretileA.size()));

                        loadInputA = command->addOperation(Operations::SubTileTranspose(
                            loadInputA, solutionParams.types.pretileA, true));
                    }

                    m_tagA = command->addOperation(Operations::T_Load_Tiled(loadInputA));

                    m_tagTensorB = command->addOperation(
                        Operations::Tensor(2, typeB, {}, unitStrides(solutionParams.types.transB)));

                    auto loadInputB = m_tagTensorB;

                    auto pretileB = not solutionParams.types.pretileB.empty();
                    if(pretileB)
                    {
                        AssertFatal(solutionParams.types.transB
                                        == Client::GEMMClient::TransposeType::N,
                                    "Pretiling B is only supported when B is TransposeType::N.",
                                    ShowValue(solutionParams.types.transB));

                        AssertFatal(solutionParams.types.pretileB.size() == 2,
                                    "pretileB must have size 2 (KxN tile dimensions).",
                                    ShowValue(solutionParams.types.pretileB.size()));

                        loadInputB = command->addOperation(Operations::SubTileTranspose(
                            loadInputB, solutionParams.types.pretileB));
                    }

                    m_tagB = command->addOperation(Operations::T_Load_Tiled(loadInputB));

                    auto mulInputA = m_tagA;
                    auto mulInputB = m_tagB;

                    AssertFatal(
                        solutionParams.types.scaleA == Operations::ScaleMode::None
                            || solutionParams.types.scaleA == Operations::ScaleMode::Separate
                            || solutionParams.types.scaleA == Operations::ScaleMode::SingleScale,
                        "Scale mode not supported!",
                        ShowValue(solutionParams.types.scaleA));
                    AssertFatal(
                        solutionParams.types.scaleB == Operations::ScaleMode::None
                            || solutionParams.types.scaleB == Operations::ScaleMode::Separate
                            || solutionParams.types.scaleB == Operations::ScaleMode::SingleScale,
                        "Scale mode not supported!",
                        ShowValue(solutionParams.types.scaleB));

                    AssertFatal(solutionParams.types.scaleA == Operations::ScaleMode::None
                                    || solutionParams.types.scaleTypeA != DataType::None,
                                "Scale mode is set but scale type was not provided!",
                                ShowValue(solutionParams.types.scaleA),
                                ShowValue(solutionParams.types.scaleTypeA));

                    AssertFatal(solutionParams.types.scaleB == Operations::ScaleMode::None
                                    || solutionParams.types.scaleTypeB != DataType::None,
                                "Scale mode is set but scale type was not provided!",
                                ShowValue(solutionParams.types.scaleB),
                                ShowValue(solutionParams.types.scaleTypeB));

                    if(solutionParams.types.scaleA == Operations::ScaleMode::Separate)
                    {
                        m_tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
                            2,
                            solutionParams.types.scaleTypeA,
                            {},
                            unitStrides(solutionParams.types.transA)));

                        auto loadScaleInputA = m_tagTensorScaleA;

                        auto isPreTiled = not solutionParams.types.scalePretileA.empty();
                        if(isPreTiled)
                        {
                            AssertFatal(solutionParams.types.transA == TransposeType::T);
                            AssertFatal(solutionParams.types.scalePretileA.size() == 2,
                                        ShowValue(solutionParams.types.scalePretileA));
                            loadScaleInputA
                                = command->addOperation(rocRoller::Operations::SubTileTranspose(
                                    loadScaleInputA.value(),
                                    solutionParams.types.scalePretileA,
                                    solutionParams.types.transA == TransposeType::T));
                        }

                        m_tagLoadScaleA = command->addOperation(
                            rocRoller::Operations::T_Load_Tiled(loadScaleInputA.value()));

                        auto scaleInputA = m_tagLoadScaleA;

                        if(solutionParams.types.scaleSkipPermlane
                           == rocRoller::ScaleSkipPermlaneMode::PreSwizzleScaleGFX950)
                        {
                            AssertFatal(
                                not solutionParams.types.scalePretileA.empty()
                                    && not solutionParams.types.scalePretileB.empty(),
                                "PreSwizzleScaleGFX950 requires pretile scale (scalePretileA and "
                                "scalePretileB non-empty).");
                        }

                        if(solutionParams.types.scaleSkipPermlane
                           != rocRoller::ScaleSkipPermlaneMode::None)
                        {
                            AssertFatal(solutionParams.types.scaleShuffleTileA.size() == 3,
                                        ShowValue(solutionParams.types.scaleShuffleTileA));

                            scaleInputA
                                = command->addOperation(rocRoller::Operations::SubTileTranspose(
                                    scaleInputA.value(), solutionParams.types.scaleShuffleTileA));
                        }
                        m_tagBlockScaleA = mulInputA
                            = command->addOperation(rocRoller::Operations::BlockScale(
                                m_tagA,
                                2,
                                scaleInputA,
                                {1,
                                 static_cast<unsigned long>(solutionParams.types.scaleBlockSize)}));
                    }
                    else if(solutionParams.types.scaleA == Operations::ScaleMode::SingleScale)
                    {
                        m_tagTensorScaleA = command->addOperation(
                            rocRoller::Operations::Scalar(solutionParams.types.scaleTypeA));
                        m_tagLoadScaleA = command->addOperation(
                            rocRoller::Operations::T_Load_Scalar(m_tagTensorScaleA.value()));
                        m_tagBlockScaleA = mulInputA = command->addOperation(
                            rocRoller::Operations::BlockScale(m_tagA, 0, m_tagLoadScaleA));
                    }

                    if(solutionParams.types.scaleB == Operations::ScaleMode::Separate)
                    {
                        m_tagTensorScaleB = command->addOperation(rocRoller::Operations::Tensor(
                            2,
                            solutionParams.types.scaleTypeB,
                            {},
                            unitStrides(solutionParams.types.transB)));

                        auto loadScaleInputB = m_tagTensorScaleB;

                        auto isPreTiled = not solutionParams.types.scalePretileB.empty();
                        if(isPreTiled)
                        {
                            AssertFatal(solutionParams.types.transB == TransposeType::N);
                            AssertFatal(solutionParams.types.scalePretileB.size() == 2,
                                        ShowValue(solutionParams.types.scalePretileB));
                            loadScaleInputB
                                = command->addOperation(rocRoller::Operations::SubTileTranspose(
                                    loadScaleInputB.value(),
                                    solutionParams.types.scalePretileB,
                                    solutionParams.types.transB == TransposeType::T));
                        }

                        m_tagLoadScaleB = command->addOperation(
                            rocRoller::Operations::T_Load_Tiled(loadScaleInputB.value()));

                        auto scaleInputB = m_tagLoadScaleB;

                        if(solutionParams.types.scaleSkipPermlane
                           != rocRoller::ScaleSkipPermlaneMode::None)
                        {
                            AssertFatal(solutionParams.types.scaleShuffleTileB.size() == 3);

                            scaleInputB
                                = command->addOperation(rocRoller::Operations::SubTileTranspose(
                                    *scaleInputB, solutionParams.types.scaleShuffleTileB));
                        }

                        m_tagBlockScaleB = mulInputB
                            = command->addOperation(rocRoller::Operations::BlockScale(
                                m_tagB,
                                2,
                                scaleInputB,
                                {static_cast<unsigned long>(solutionParams.types.scaleBlockSize),
                                 1}));
                    }
                    else if(solutionParams.types.scaleB == Operations::ScaleMode::SingleScale)
                    {
                        m_tagTensorScaleB = command->addOperation(
                            rocRoller::Operations::Scalar(solutionParams.types.scaleTypeB));
                        m_tagLoadScaleB = command->addOperation(
                            rocRoller::Operations::T_Load_Scalar(m_tagTensorScaleB.value()));
                        m_tagBlockScaleB = mulInputB = command->addOperation(
                            rocRoller::Operations::BlockScale(m_tagB, 0, m_tagLoadScaleB));
                    }

                    m_tagTensorC
                        = command->addOperation(Operations::Tensor(2, typeC, {}, {(size_t)1})); // C
                    m_tagC = command->addOperation(Operations::T_Load_Tiled(m_tagTensorC));

                    m_tagScalarAlpha
                        = command->addOperation(Operations::Scalar(DataType::Float)); // alpha
                    auto tagLoadAlpha
                        = command->addOperation(Operations::T_Load_Scalar(m_tagScalarAlpha));

                    m_tagScalarBeta
                        = command->addOperation(Operations::Scalar(DataType::Float)); // beta
                    auto tagLoadBeta
                        = command->addOperation(Operations::T_Load_Scalar(m_tagScalarBeta));

                    auto tagAB = command->addOperation(
                        Operations::T_Mul(mulInputA, mulInputB, typeAcc)); // A * B

                    Operations::T_Execute execute(command->getNextTag());
                    auto                  tagBetaC
                        = execute.addXOp(Operations::E_Mul(tagLoadBeta, m_tagC)); // beta * C
                    auto tagAlphaAB
                        = execute.addXOp(Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)
                    if(solutionParams.betaInFma)
                    {
                        m_tagD = execute.addXOp(
                            Operations::E_Add(tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
                    }
                    else
                    {
                        m_tagD = execute.addXOp(
                            Operations::E_Add(tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
                    }
                    command->addOperation(std::move(execute));

                    m_tagTensorD
                        = command->addOperation(Operations::Tensor(2, typeD, {}, {(size_t)1})); // D
                    // command->addOperation(Operations::T_Store_Tiled(m_tagD, m_tagTensorD));
                    if(solutionParams.types.typeAcc == solutionParams.types.typeD)
                    {
                        command->addOperation(Operations::T_Store_Tiled(m_tagD, m_tagTensorD));
                    }
                    else
                    {
                        // If Matrix C and D are of different types, an explicit type conversion is required

                        auto cvtOp = Operations::T_Execute(command->getNextTag());
                        // (SR)Convert( alpha * (A * B) + beta * C )
                        auto tagCvt = cvtOp.addXOp(Operations::E_Cvt(m_tagD, typeD));
                        command->addOperation(std::move(cvtOp));
                        command->addOperation(Operations::T_Store_Tiled(tagCvt, m_tagTensorD));
                    }

                    if(solutionParams.workgroupMappingDim != -1)
                    {
                        m_tagWGM = command->allocateTag();
                        command->allocateArgument(DataType::Int32,
                                                  m_tagWGM,
                                                  ArgumentType::Value,
                                                  DataDirection::ReadOnly,
                                                  rocRoller::WGM);
                    }

                    return command;
                }

                CommandParametersPtr
                    makeCommandParameters(CommandPtr                command,
                                          SolutionParameters const& solutionParams) override
                {
                    auto params = std::make_shared<CommandParameters>();

                    params->tailLoops = solutionParams.tailLoops;

                    int wave_m = 0, wave_n = 0, wave_k = 0, wave_b = 0;

                    auto typeA = fromString<DataType>(solutionParams.types.typeA);
                    auto typeB = fromString<DataType>(solutionParams.types.typeB);
                    auto typeC = fromString<DataType>(solutionParams.types.typeC);
                    auto typeD = fromString<DataType>(solutionParams.types.typeD);

                    if(typeA == DataType::Float && typeB == DataType::Float)
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 2;
                        wave_b = 1;
                    }
                    else if((typeA == DataType::Half && typeB == DataType::Half)
                            || (typeA == DataType::BFloat16 && typeB == DataType::BFloat16))
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 8;
                        wave_b = 1;
                    }
                    else if((typeA == DataType::FP8 && typeB == DataType::FP8)
                            || (typeA == DataType::BF8 && typeB == DataType::BF8))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 32;
                        wave_b = 1;
                    }
                    else if((typeA == DataType::FP4 && typeB == DataType::FP4)
                            || (typeA == DataType::FP6 && typeB == DataType::FP6)
                            || (typeA == DataType::BF6 && typeB == DataType::BF6))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 128;
                        wave_b = 1;
                    }
                    else if(typeA != typeB && isUnpackedF8F6F4(typeA) && isUnpackedF8F6F4(typeB))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 128;
                        wave_b = 1;
                    }
                    else
                    {
                        Throw<FatalError>("Unsupported datatype combination in client");
                    }

                    if(solutionParams.waveM > 0)
                        wave_m = solutionParams.waveM;
                    if(solutionParams.waveN > 0)
                        wave_n = solutionParams.waveN;
                    if(solutionParams.waveK > 0)
                        wave_k = solutionParams.waveK;
                    if(solutionParams.waveB > 0)
                        wave_b = solutionParams.waveB;

                    AssertFatal(solutionParams.macM * solutionParams.macK
                                        * DataTypeInfo::Get(typeA).elementBytes
                                    > wave_m * wave_k,
                                "Not enough elements (A).");
                    AssertFatal(solutionParams.macN * solutionParams.macK
                                        * DataTypeInfo::Get(typeA).elementBytes
                                    > wave_n * wave_k,
                                "Not enough elements (B).");

                    auto const arch = GPUArchitectureLibrary::getInstance()->GetArch(
                        solutionParams.architecture);
                    uint wavefrontSize = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                    uint wavetilePerWavefrontM = wavefrontSize * solutionParams.macM / wave_m
                                                 / solutionParams.workgroupSizeX;
                    uint wavetilePerWavefrontN
                        = solutionParams.macN / wave_n / solutionParams.workgroupSizeY;

                    AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch.");
                    AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch.");

                    AssertFatal(solutionParams.macM % (wave_m * wavetilePerWavefrontM) == 0,
                                "WaveTile size mismatch (M)",
                                ShowValue(solutionParams.macM),
                                ShowValue(wave_m),
                                ShowValue(wavetilePerWavefrontM));
                    AssertFatal(solutionParams.macN % (wave_n * wavetilePerWavefrontN) == 0,
                                "WaveTile size mismatch (N)",
                                ShowValue(solutionParams.macN),
                                ShowValue(wave_n),
                                ShowValue(wavetilePerWavefrontN));

                    if(solutionParams.types.scaleA == Operations::ScaleMode::Separate
                       || solutionParams.types.scaleB == Operations::ScaleMode::Separate)
                    {
                        AssertFatal(
                            arch.isSupportedScaleBlockSize(solutionParams.types.scaleBlockSize),
                            fmt::format(
                                "Architecture {} does not support block scaling (size: {}).",
                                solutionParams.architecture.toString(),
                                solutionParams.types.scaleBlockSize));
                        AssertFatal(
                            solutionParams.waveK % solutionParams.types.scaleBlockSize == 0,
                            fmt::format("waveK: {} must be a multiple of the scale block size: {}",
                                        solutionParams.waveK,
                                        solutionParams.types.scaleBlockSize));
                    }

                    AssertFatal(solutionParams.types.scaleA == Operations::ScaleMode::None
                                    || arch.isSupportedScaleType(solutionParams.types.scaleTypeA),
                                fmt::format("Scale mode for A set but architecture {} does not "
                                            "support scale type {}.",
                                            solutionParams.architecture.toString(),
                                            toString(solutionParams.types.scaleTypeA)));

                    AssertFatal(solutionParams.types.scaleB == Operations::ScaleMode::None
                                    || arch.isSupportedScaleType(solutionParams.types.scaleTypeB),
                                fmt::format("Scale mode for B set but architecture {} does not "
                                            "support scale type {}.",
                                            solutionParams.architecture.toString(),
                                            toString(solutionParams.types.scaleTypeB)));

                    if(solutionParams.swizzleScale)
                    {
                        if(solutionParams.types.scaleA == Operations::ScaleMode::Separate)
                        {
                            AssertFatal(solutionParams.swizzleTileSize.m > 0
                                            && solutionParams.swizzleTileSize.k > 0,
                                        "Invalid SwizzleTileSize for A.",
                                        ShowValue(solutionParams.swizzleTileSize.m),
                                        ShowValue(solutionParams.swizzleTileSize.k));
                        }

                        if(solutionParams.types.scaleB == Operations::ScaleMode::Separate)
                        {
                            AssertFatal(solutionParams.swizzleTileSize.n > 0
                                            && solutionParams.swizzleTileSize.l > 0,
                                        "Invalid SwizzleTileSize for B.",
                                        ShowValue(solutionParams.swizzleTileSize.n),
                                        ShowValue(solutionParams.swizzleTileSize.l));
                        }
                    }

                    params->setManualKernelDimension(2);
                    params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

                    auto memoryTypeA = GetMemoryType(solutionParams.loadPathA);
                    auto memoryTypeB = GetMemoryType(solutionParams.loadPathB);

                    AssertFatal(solutionParams.padLDSA.first >= -1
                                    && solutionParams.padLDSA.second >= -1,
                                "Invalid LDS padding (A)",
                                ShowValue(solutionParams.padLDSA.first),
                                ShowValue(solutionParams.padLDSA.second));
                    AssertFatal(solutionParams.padLDSB.first >= -1
                                    && solutionParams.padLDSB.second >= -1,
                                "Invalid LDS padding (B)",
                                ShowValue(solutionParams.padLDSB.first),
                                ShowValue(solutionParams.padLDSB.second));

                    params->ldsPadding[LayoutType::MATRIX_A] = solutionParams.padLDSA;
                    params->ldsPadding[LayoutType::MATRIX_B] = solutionParams.padLDSB;

                    auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macK},
                        LayoutType::MATRIX_A,
                        {wave_m, wave_n, wave_k, wave_b},
                        memoryTypeA);
                    auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macK, solutionParams.macN},
                        LayoutType::MATRIX_B,
                        {wave_m, wave_n, wave_k, wave_b},
                        memoryTypeB);
                    auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b});

                    // Determine if type conversion is needed
                    bool needsConversion
                        = (solutionParams.types.typeAcc != solutionParams.types.typeD);

                    // m_tagD: If conversion is needed, use WAVE (not LDS) since it won't be stored directly
                    auto memoryTypeD = GetMemoryType(solutionParams.storePath);
                    auto macTileD    = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b},
                        (IsLDSStore(solutionParams.storePath) && !needsConversion)
                               ? memoryTypeD
                               : MemoryType::WAVE);

                    params->setDimensionInfo(m_tagA, macTileA);
                    params->setDimensionInfo(m_tagB, macTileB);
                    params->setDimensionInfo(m_tagC, macTileC);
                    params->setDimensionInfo(m_tagD, macTileD);

                    if(m_tagCvt.has_value())
                    {
                        // For type conversion, this is what gets stored - use LDS store path if specified
                        auto macTileCvt = KernelGraph::CoordinateGraph::MacroTile(
                            {solutionParams.macM, solutionParams.macN},
                            LayoutType::MATRIX_ACCUMULATOR,
                            {wave_m, wave_n, wave_k, wave_b},
                            IsLDSStore(solutionParams.storePath) ? memoryTypeD : MemoryType::WAVE);
                        params->setDimensionInfo(*m_tagCvt, macTileCvt);
                    }

                    if(solutionParams.types.scaleA == Operations::ScaleMode::Separate)
                    {
                        AssertFatal(
                            solutionParams.loadPathAScale
                                    != Parameters::Solution::LoadPath::BufferToLDS
                                || solutionParams.swizzleScale,
                            "If loadPathAScale is BufferToLDS, swizzleScale must be enabled");
                        auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
                            {solutionParams.macM,
                             solutionParams.macK / solutionParams.types.scaleBlockSize},
                            LayoutType::MATRIX_A,
                            {solutionParams.waveM,
                             solutionParams.waveN,
                             solutionParams.waveK / solutionParams.types.scaleBlockSize,
                             solutionParams.waveB},
                            GetMemoryType(solutionParams.loadPathAScale),
                            {},
                            {solutionParams.swizzleTileSize.m,
                             solutionParams.swizzleTileSize.n,
                             solutionParams.swizzleTileSize.k,
                             1});
                        params->setDimensionInfo(*m_tagLoadScaleA, macTileAScale);
                    }
                    if(solutionParams.types.scaleB == Operations::ScaleMode::Separate)
                    {
                        AssertFatal(
                            solutionParams.loadPathBScale
                                    != Parameters::Solution::LoadPath::BufferToLDS
                                || solutionParams.swizzleScale,
                            "If loadPathBScale is BufferToLDS, swizzleScale must be enabled");
                        auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
                            {solutionParams.macK / solutionParams.types.scaleBlockSize,
                             solutionParams.macN},
                            LayoutType::MATRIX_B,
                            {solutionParams.waveM,
                             solutionParams.waveN,
                             solutionParams.waveK / solutionParams.types.scaleBlockSize,
                             solutionParams.waveB},
                            GetMemoryType(solutionParams.loadPathBScale),
                            {},
                            {solutionParams.swizzleTileSize.m,
                             solutionParams.swizzleTileSize.n,
                             solutionParams.swizzleTileSize.l,
                             1});
                        params->setDimensionInfo(*m_tagLoadScaleB, macTileBScale);
                    }

                    params->swizzleScale  = solutionParams.swizzleScale;
                    params->prefetchScale = solutionParams.prefetchScale;

                    if(solutionParams.prefetch)
                    {
                        params->prefetch          = true;
                        params->unrollK           = std::max(2, solutionParams.prefetchInFlight);
                        params->prefetchInFlight  = solutionParams.prefetchInFlight;
                        params->prefetchLDSFactor = solutionParams.prefetchLDSFactor;
                        params->prefetchMixMemOps = solutionParams.prefetchMixMemOps;
                    }
                    else
                    {
                        params->prefetch = false;
                    }

                    params->transposeMemoryAccess.set(
                        LayoutType::MATRIX_A, solutionParams.types.transA == TransposeType::T);
                    params->transposeMemoryAccess.set(
                        LayoutType::MATRIX_B, solutionParams.types.transB == TransposeType::T);

                    uint workgroup_size_x
                        = solutionParams.workgroupSizeX * solutionParams.workgroupSizeY;
                    uint workgroup_size_y = 1;

                    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

                    if(solutionParams.workgroupMappingDim != -1)
                    {
                        auto dim = solutionParams.workgroupMappingDim;

                        AssertFatal(
                            dim == 0 || dim == 1,
                            "Only 0 (M) or 1 (N) are supported dimensions for workgroup mapping.",
                            ShowValue(dim));

                        // CommandSolution::generateKernelGraph creates the size Expression
                        // and initializes the workgroupMappingValue
                        params->workgroupMappingDim = dim;
                    }

                    if(solutionParams.workgroupRemapXCC)
                    {
                        AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
                                    "XCC-aware workgroup remapping not available on: ",
                                    arch.target().toString());
                        if(solutionParams.workgroupRemapXCCValue != -1)
                        {
                            params->workgroupRemapXCC = solutionParams.workgroupRemapXCCValue;
                        }
                        else
                        {
                            params->workgroupRemapXCC
                                = arch.GetCapability(GPUCapability::DefaultRemapXCCValue);
                        }
                    }

                    params->setManualWavefrontCount(
                        {static_cast<uint>(solutionParams.macM / wave_m / wavetilePerWavefrontM),
                         static_cast<uint>(solutionParams.macN / wave_n / wavetilePerWavefrontN)});

                    return params;
                }

                CommandArguments commandArguments(CommandPtr               command,
                                                  ProblemParameters const& problemParams,
                                                  RunParameters const&     runParams) const override
                {
                    CommandArguments commandArgs = command->createArguments();

                    size_t M = problemParams.m;
                    size_t N = problemParams.n;
                    size_t K = problemParams.k;

                    TensorDescriptor descA(fromString<DataType>(problemParams.types.typeA),
                                           {M, K},
                                           problemParams.types.transA == TransposeType::T ? "T"
                                                                                          : "N");
                    TensorDescriptor descB(fromString<DataType>(problemParams.types.typeB),
                                           {K, N},
                                           problemParams.types.transB == TransposeType::T ? "T"
                                                                                          : "N");

                    setCommandTensorArg(commandArgs, m_tagTensorA, descA, (float*)nullptr);
                    setCommandTensorArg(commandArgs, m_tagTensorB, descB, (float*)nullptr);

                    TensorDescriptor descC(
                        fromString<DataType>(problemParams.types.typeC), {M, N}, "N");
                    setCommandTensorArg(commandArgs, m_tagTensorC, descC, (float*)nullptr);

                    commandArgs.setArgument(
                        m_tagScalarAlpha, ArgumentType::Value, problemParams.alpha);
                    commandArgs.setArgument(
                        m_tagScalarBeta, ArgumentType::Value, problemParams.beta);

                    TensorDescriptor descD(
                        fromString<DataType>(problemParams.types.typeD), {M, N}, "N");
                    setCommandTensorArg(commandArgs, m_tagTensorD, descD, (float*)nullptr);

                    if(problemParams.workgroupMappingDim != -1)
                    {
                        auto const workgroupMappingDim   = problemParams.workgroupMappingDim;
                        auto const workgroupMappingValue = runParams.workgroupMappingValue;

                        AssertFatal(workgroupMappingDim == 0 || workgroupMappingDim == 1,
                                    "Only 0 (M) or 1 (N) are supported dimensions for workgroup "
                                    "mapping dim.",
                                    ShowValue(workgroupMappingDim));

                        AssertFatal(workgroupMappingValue > 0,
                                    "Workgroup mapping value must be a positive integer "
                                    "when work group dimension is specified.",
                                    ShowValue(workgroupMappingValue));

                        commandArgs.setArgument(
                            m_tagWGM, ArgumentType::Value, workgroupMappingValue);
                    }

                    return commandArgs;
                }

                void setPredicates(CommandPtr                command,
                                   CommandKernelPtr          commandKernel,
                                   SolutionParameters const& solutionParams) override
                {
                    using namespace rocRoller::Expression;
                    auto params = commandKernel->getCommandParameters();

                    // predicate building blocks
                    // A sizes
                    auto aSizes
                        = std::get<Operations::Tensor>(*(command->findTag(m_tagTensorA))).sizes();
                    std::vector<ExpressionPtr> aSizeExps(aSizes.size());
                    std::transform(aSizes.begin(), aSizes.end(), aSizeExps.begin(), [](auto arg) {
                        return arg->expression();
                    });

                    // parameters
                    auto unrollKExp = literal(params->unrollK);
                    auto macKExp    = literal(solutionParams.macK);

                    // constants
                    auto zero = literal(0u);
                    auto one  = literal(1u);

                    // sanitize parameters
                    auto sanUnrollKExp = convert(DataType::UInt32,
                                                 conditional(unrollKExp == zero, one, unrollKExp));

                    // predicates
                    // unrollK size match predicates

                    if(params->tailLoops and not params->streamK)
                    {
                        auto unrollKPredicate = (aSizeExps[1] % macKExp == zero);
                        setComment(unrollKPredicate, "K must be a multiple of macK.");
                        commandKernel->addPredicate(unrollKPredicate);
                    }
                    else
                    {
                        auto unrollKPredicate = (aSizeExps[1] % (macKExp * sanUnrollKExp) == zero);
                        setComment(unrollKPredicate,
                                   "K must be a multiple of macK * unrollK (unrollK may be "
                                   "set by prefetchInFlight)");
                        commandKernel->addPredicate(unrollKPredicate);
                    }
                }
            };
        }
    }
}
