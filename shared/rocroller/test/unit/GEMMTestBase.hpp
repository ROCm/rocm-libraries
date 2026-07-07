// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/WorkgroupClusters_detail.hpp>

#include "GPUContextFixture.hpp"

#include <common/GEMMProblem.hpp>
#include <common/mxDataGen.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>

namespace GEMMTests
{
    std::set<int> NonZeroDSReadOffsets(std::string const& instruction, std::string const& s);
    std::set<int> Direct2LDSWriteStrides(std::string const& s);

    template <typename T>
    concept isF8 = std::is_same_v<T, rocRoller::FP8> || std::is_same_v<T, rocRoller::BF8>;

    template <typename T>
    concept isF6F4 = std::is_same_v<T, rocRoller::FP6> || std::is_same_v<T, rocRoller::BF6> || std::
        is_same_v<T, rocRoller::FP4>;

    template <typename... Ts>
    class BaseGEMMContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<rocRoller::GPUArchitectureTarget, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            auto device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

        int m_scaleValueIndex = 0;

    public:
        uint8_t rotatingSingleScaleValue(rocRoller::DataType scaleType)
        {
            using namespace rocRoller;
            AssertFatal(isScaleType(scaleType));
            const std::vector<float> scaleValues{1.0, 2.0, 4.0, 8.0};
            m_scaleValueIndex = (++m_scaleValueIndex) % scaleValues.size();
            return floatToScale(scaleType, scaleValues[m_scaleValueIndex]);
        }

        template <typename TA,
                  typename TB  = TA,
                  typename TC  = TA,
                  typename TD  = TC,
                  typename ACC = float>
        void basicGEMM(const GEMMProblem&      gemm,
                       bool                    debuggable  = false,
                       bool                    setIdentity = false,
                       int                     numIters    = 1,
                       bool                    notSetC     = false,
                       std::optional<uint32_t> srCvtSeed   = std::nullopt)
        {
            using namespace rocRoller;
            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_fp8,
                                        GPUCapability::HasWMMA_f32_16x16x16_f8,
                                        GPUCapability::HasWMMA_f32_16x16x64_f8,
                                        GPUCapability::HasWMMA_f32_16x16x128_f8);
            }

            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4,
                                        GPUCapability::HasWMMA_f8f6f4);
            }

            if((isF8<TA> || isF8<TB>)&&(gemm.waveK >= 64))
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4,
                                        GPUCapability::HasWMMA_f8f6f4,
                                        GPUCapability::HasWMMA_32x16x128_f4);
            }

            if(gemm.scaleAMode != Operations::ScaleMode::None
               || gemm.scaleBMode != Operations::ScaleMode::None)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4,
                                        GPUCapability::HasWMMA_scale_f8f6f4,
                                        GPUCapability::HasWMMA_scale_32x16x128_f4,
                                        GPUCapability::HasWMMA_scale16_32x16x128_f4);
                const auto  scaleType = gemm.scaleAMode != Operations::ScaleMode::None
                                            ? gemm.scaleTypeA
                                            : gemm.scaleTypeB;
                const auto& arch      = m_context->targetArchitecture();
                AssertFatal(gemm.scaleAMode == Operations::ScaleMode::None
                                || arch.isSupportedScaleType(gemm.scaleTypeA),
                            fmt::format("Scale mode for A set but architecture {} does not "
                                        "support scale type {}.",
                                        arch.target().toString(),
                                        toString(gemm.scaleTypeA)));
                AssertFatal(gemm.scaleBMode == Operations::ScaleMode::None
                                || arch.isSupportedScaleType(gemm.scaleTypeB),
                            fmt::format("Scale mode for B set but architecture {} does not "
                                        "support scale type {}.",
                                        arch.target().toString(),
                                        toString(gemm.scaleTypeB)));
            }

            AssertFatal(gemm.scaleAMode == Operations::ScaleMode::None
                            || gemm.scaleAMode == Operations::ScaleMode::SingleScale
                            || gemm.scaleAMode == Operations::ScaleMode::Separate,
                        "Scale mode not supported!",
                        ShowValue(gemm.scaleAMode));
            AssertFatal(gemm.scaleBMode == Operations::ScaleMode::None
                            || gemm.scaleBMode == Operations::ScaleMode::SingleScale
                            || gemm.scaleBMode == Operations::ScaleMode::Separate,
                        "Scale mode not supported!",
                        ShowValue(gemm.scaleBMode));

            auto dataTypeA   = TypeInfo<TA>::Var.dataType;
            auto dataTypeB   = TypeInfo<TB>::Var.dataType;
            auto dataTypeC   = TypeInfo<TC>::Var.dataType;
            auto dataTypeD   = TypeInfo<TD>::Var.dataType;
            auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

            // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
            int   M     = gemm.m;
            int   N     = gemm.n;
            int   K     = gemm.k;
            float alpha = gemm.alpha;
            float beta  = gemm.beta;

            AssertFatal(M % gemm.macM == 0,
                        "MacroTile size mismatch (M)",
                        ShowValue(M),
                        ShowValue(gemm.macM));
            AssertFatal(N % gemm.macN == 0,
                        "MacroTile size mismatch (N)",
                        ShowValue(N),
                        ShowValue(gemm.macN));
            AssertFatal(K >= gemm.macK, "K must be >= macK", ShowValue(K), ShowValue(gemm.macK));
            AssertFatal(K % gemm.macK == 0 || gemm.tailLoops,
                        "K must be a multiple of macK (or enable tailLoops)",
                        ShowValue(K),
                        ShowValue(gemm.macK));
            AssertFatal(gemm.macK >= gemm.waveK,
                        "macK must be >= waveK",
                        ShowValue(gemm.macK),
                        ShowValue(gemm.waveK));
            AssertFatal(gemm.macK % gemm.waveK == 0,
                        "macK must be a multiple of waveK",
                        ShowValue(gemm.macK),
                        ShowValue(gemm.waveK));

            if(gemm.scaleAMode == Operations::ScaleMode::Separate
               || gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(
                    m_context->targetArchitecture().isSupportedScaleBlockSize(gemm.scaleBlockSize),
                    fmt::format("Architecture {} does not support block scaling (size: {}).",
                                m_context->targetArchitecture().target().toString(),
                                gemm.scaleBlockSize));
            }

            if(gemm.unrollK > 0 && !gemm.tailLoops)
            {
                AssertFatal(K % (gemm.macK * gemm.unrollK) == 0,
                            "MacroTile size mismatch (K unroll)");
            }

            auto bpeA = DataTypeInfo::Get(dataTypeA).elementBytes;
            auto bpeB = DataTypeInfo::Get(dataTypeB).elementBytes;
            AssertFatal(gemm.macM * gemm.macK * bpeA >= gemm.waveM * gemm.waveK,
                        "Not enough elements (A).");
            AssertFatal(gemm.macN * gemm.macK * bpeB >= gemm.waveN * gemm.waveK,
                        "Not enough elements (B).");

            AssertFatal(gemm.workgroupSizeX % gemm.wavefrontSize == 0,
                        "Workgroup Size X must be multiply of wave front size");

            uint wavetilePerWavefrontM
                = gemm.wavefrontSize * gemm.macM / gemm.waveM / gemm.workgroupSizeX;
            uint wavetilePerWavefrontN = gemm.macN / gemm.waveN / gemm.workgroupSizeY;

            AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch (M).");
            AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch (N).");

            AssertFatal(gemm.macM % (gemm.waveM * wavetilePerWavefrontM) == 0,
                        "WaveTile size mismatch (M)");
            AssertFatal(gemm.macN % (gemm.waveN * wavetilePerWavefrontN) == 0,
                        "WaveTile size mismatch (N)");

            Log::debug("GEMMTest jamming: {}x{}", wavetilePerWavefrontM, wavetilePerWavefrontN);

            uint workgroupSizeX = gemm.workgroupSizeX * gemm.workgroupSizeY;
            uint workgroupSizeY = 1;

            uint numWorkgroupX;
            uint numWorkgroupY;

            if(gemm.loopOverTiles > 0)
            {
                // multiple output macro tiles per workgroup
                numWorkgroupX = M * N / gemm.macM / gemm.macN / 2;
                numWorkgroupY = 1;
            }
            else if(gemm.streamK)
            {
                numWorkgroupX = gemm.numWGs;
                numWorkgroupY = 1;
            }
            else
            {
                // one output macro tile per workgroup
                numWorkgroupX = M / gemm.macM;
                numWorkgroupY = N / gemm.macN;
            }

            m_kernelOptions->scaleSkipPermlane = gemm.scaleSkipPermlane;
            m_kernelOptions->ldsSwizzleMode    = gemm.ldsSwizzleMode;

            // Host data
            using PackedTypeA = typename PackedTypeOf<TA>::type;
            using PackedTypeB = typename PackedTypeOf<TB>::type;
            std::vector<PackedTypeA> hostA;
            std::vector<PackedTypeB> hostB;
            std::vector<TC>          hostC;

            std::vector<uint8_t> hostScaleA, hostScaleB;

            TensorDescriptor descA(dataTypeA, {size_t(M), size_t(K)}, gemm.transA);
            TensorDescriptor descB(dataTypeB, {size_t(K), size_t(N)}, gemm.transB);
            TensorDescriptor descC(dataTypeD, {size_t(M), size_t(N)}, "N");
            TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

            auto seed = 31415u;
            if(gemm.scaleAMode == Operations::ScaleMode::Separate
               || gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                auto const& arch = m_context->targetArchitecture();

                auto scaleBlockSize = gemm.scaleBlockSize;
                AssertFatal(scaleBlockSize > 0, "scaleBlockSize must be set to scale A or B.");
                AssertFatal(
                    arch.isSupportedScaleBlockSize(scaleBlockSize),
                    fmt::format("Architecture {} does not support block scaling (size: {}).",
                                arch.target().toString(),
                                scaleBlockSize));
                AssertFatal(gemm.k % scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        gemm.k,
                                        scaleBlockSize));
                DGenInput(seed,
                          hostA,
                          descA,
                          hostB,
                          descB,
                          hostC,
                          descC,
                          hostScaleA,
                          hostScaleB,
                          gemm.scaleTypeA,
                          gemm.scaleTypeB,
                          -1.f,
                          1.f,
                          static_cast<uint>(scaleBlockSize));
            }
            else
            {
                DGenInput(seed, hostA, descA, hostB, descB, hostC, descC);
            }

            if(setIdentity)
            {
                SetIdentityMatrix(hostA, K, M);
                SetIdentityMatrix(hostB, N, K);

                std::fill(hostC.begin(), hostC.end(), static_cast<TD>(0.0));
            }

            // Pre-tile A on the host when pretileA is set (kernel expects pre-tiled layout).
            // pretileA is only supported for transA == "T"; A is stored in memory as KxM, so we
            // pass (K, M) and (tileK, tileM) to preSwizzle to match the client (gemm.cpp).
            std::vector<PackedTypeA> hostAForKernel(hostA);
            if(!gemm.pretileA.empty() && gemm.pretileA.size() == 2)
            {
                AssertFatal(gemm.transA == "T", "Pre-tiling A only supported for TransposeType::T");
                auto const packing = TypeInfo<PackedTypeA>::ElementBits / TypeInfo<TA>::ElementBits;
                std::vector<size_t> sizes       = descA.sizes();
                std::vector<size_t> preTileSize = gemm.pretileA;
                AssertFatal(M % preTileSize[0] == 0,
                            "A matrix dimension M must be divisible by pretileA tile size in M.",
                            ShowValue(M),
                            ShowValue(preTileSize[0]));
                AssertFatal(K % preTileSize[1] == 0,
                            "A matrix dimension K must be divisible by pretileA tile size in K.",
                            ShowValue(K),
                            ShowValue(preTileSize[1]));
                if(packing > 1)
                {
                    AssertFatal(sizes[1] % packing == 0,
                                "pretileA: K dimension must be a multiple of packing factor (",
                                packing,
                                ") for packed type A.");
                    AssertFatal(preTileSize[1] % packing == 0,
                                "pretileA: tile K must be a multiple of packing factor (",
                                packing,
                                ") for packed type A.");
                    sizes[1] /= packing;
                    preTileSize[1] /= packing;
                }

                // The preSwizzle helper assumes column-major; so we swap sizes here.
                std::vector<size_t> swappedSizes       = {sizes[1], sizes[0]};
                std::vector<size_t> swappedPreTileSize = {preTileSize[1], preTileSize[0]};
                hostAForKernel = DGen::preSwizzle(hostA, swappedSizes, {}, swappedPreTileSize);
            }

            // Pre-tile B on the host when pretileB is set (kernel expects pre-tiled layout)
            std::vector<PackedTypeB> hostBForKernel(hostB);
            if(!gemm.pretileB.empty() && gemm.pretileB.size() == 2)
            {
                auto const packing = TypeInfo<PackedTypeB>::ElementBits / TypeInfo<TB>::ElementBits;
                std::vector<size_t> sizes       = descB.sizes();
                std::vector<size_t> preTileSize = gemm.pretileB;
                AssertFatal(K % preTileSize[0] == 0,
                            "B matrix dimension K must be divisible by pretileB tile size in K.",
                            ShowValue(K),
                            ShowValue(preTileSize[0]));
                AssertFatal(N % preTileSize[1] == 0,
                            "B matrix dimension N must be divisible by pretileB tile size in N.",
                            ShowValue(N),
                            ShowValue(preTileSize[1]));
                if(packing > 1)
                {
                    AssertFatal(sizes[0] % packing == 0,
                                "pretileB: K dimension must be a multiple of packing factor (",
                                packing,
                                ") for packed type B.");
                    AssertFatal(preTileSize[0] % packing == 0,
                                "pretileB: tile K must be a multiple of packing factor (",
                                packing,
                                ") for packed type B.");
                    sizes[0] /= packing;
                    preTileSize[0] /= packing;
                }
                hostBForKernel = DGen::preSwizzle(hostB, sizes, {}, preTileSize);
            }

            auto deviceA = make_shared_device<TA>(hostAForKernel);
            auto deviceB = make_shared_device<TB>(hostBForKernel);

            std::shared_ptr<TC> deviceC = (notSetC) ? nullptr : make_shared_device(hostC);
            std::shared_ptr<TD> deviceD = make_shared_device<TD>(M * N, TD{});

            std::shared_ptr<uint8_t> deviceScaleA, deviceScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                if((gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                   || (!gemm.scalePretileA.empty()))
                {
                    auto descScaleA = descA.withNormalizedDimensions();
                    {
                        auto scaleSizes = descScaleA.sizes();
                        scaleSizes[0] /= gemm.scaleBlockSize;
                        descScaleA = TensorDescriptor(descScaleA.dataType(), std::move(scaleSizes));
                    }
                    std::vector<size_t> preSwizzleSize;
                    if(gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                    {
                        AssertFatal(gemm.scaleShuffleTileA.size() == 3);
                        preSwizzleSize = gemm.scaleShuffleTileA;
                    }
                    std::vector<size_t> preTileSize;
                    if(!gemm.scalePretileA.empty())
                    {
                        AssertFatal(gemm.transA == "T",
                                    "Can only pre-tile scale A if A is TransposeType::T");
                        preTileSize = {gemm.scalePretileA[1], gemm.scalePretileA[0]};
                    }
                    auto tmpScaleA = DGen::preSwizzle(
                        hostScaleA, descScaleA.sizes(), preSwizzleSize, preTileSize);
                    deviceScaleA = make_shared_device(tmpScaleA);
                }
                else
                {
                    deviceScaleA = make_shared_device(hostScaleA);
                }
            }
            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                if((gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                   || (!gemm.scalePretileB.empty()))
                {
                    auto descScaleB = descB.withNormalizedDimensions();
                    {
                        auto scaleSizes = descScaleB.sizes();
                        scaleSizes[0] /= gemm.scaleBlockSize;
                        descScaleB = TensorDescriptor(descScaleB.dataType(), std::move(scaleSizes));
                    }
                    std::vector<size_t> preSwizzleSize;
                    if(gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                    {
                        AssertFatal(gemm.scaleShuffleTileB.size() == 3);
                        preSwizzleSize = gemm.scaleShuffleTileB;
                    }
                    std::vector<size_t> preTileSize;
                    if(!gemm.scalePretileB.empty())
                    {
                        AssertFatal(gemm.transB == "N",
                                    "Can only pre-tile scale B if B is TransposeType::N");
                        preTileSize = {gemm.scalePretileB[0], gemm.scalePretileB[1]};
                    }
                    auto tmpScaleB = DGen::preSwizzle(
                        hostScaleB, descScaleB.sizes(), preSwizzleSize, preTileSize);
                    deviceScaleB = make_shared_device(tmpScaleB);
                }
                else
                {
                    deviceScaleB = make_shared_device(hostScaleB);
                }
            }

            // In SingleScale mode, don't need to copy to device
            if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
                hostScaleA = std::vector<uint8_t>{rotatingSingleScaleValue(gemm.scaleTypeA)};

            if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
                hostScaleB = std::vector<uint8_t>{rotatingSingleScaleValue(gemm.scaleTypeB)};

            auto command = std::make_shared<Command>();

            std::vector<size_t> oneStridesN
                = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

            std::vector<size_t> oneStridesT = gemm.literalStrides
                                                  ? std::vector<size_t>({(size_t)0, (size_t)1})
                                                  : std::vector<size_t>({});

            auto stridesA   = gemm.transA == "N" ? oneStridesN : oneStridesT;
            auto tagTensorA = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeA, {}, stridesA)); // A
            auto loadInputA = tagTensorA;
            if(not gemm.pretileA.empty())
            {
                AssertFatal(gemm.pretileA.size() == 2,
                            "pretileA must have size 2 (MxK tile dimensions).",
                            ShowValue(gemm.pretileA.size()));

                loadInputA = command->addOperation(
                    rocRoller::Operations::SubTileTranspose(loadInputA, gemm.pretileA, true));
            }
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadInputA));

            auto stridesB   = gemm.transB == "N" ? oneStridesN : oneStridesT;
            auto tagTensorB = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeB, {}, stridesB)); // B
            auto loadInputB = tagTensorB;
            if(not gemm.pretileB.empty())
            {
                AssertFatal(gemm.pretileB.size() == 2,
                            "pretileB must have size 2 (KxN tile dimensions).",
                            ShowValue(gemm.pretileB.size()));

                loadInputB = command->addOperation(
                    rocRoller::Operations::SubTileTranspose(loadInputB, gemm.pretileB));
            }
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(loadInputB));

            auto mulInputA = tagLoadA;
            auto mulInputB = tagLoadB;

            std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
                tagTensorScaleB, tagLoadScaleB, tagBlockScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                bool scalePreTiledA  = !gemm.scalePretileA.empty();
                tagTensorScaleA      = command->addOperation(rocRoller::Operations::Tensor(
                    2, gemm.scaleTypeA, {}, gemm.transA == "N" ? oneStridesN : oneStridesT));
                auto scaleLoadInputA = *tagTensorScaleA;
                if(scalePreTiledA)
                {
                    AssertFatal(gemm.transA == "T");
                    AssertFatal(gemm.scalePretileA.size() == 2);
                    scaleLoadInputA = command->addOperation(rocRoller::Operations::SubTileTranspose(
                        scaleLoadInputA, gemm.scalePretileA, true));
                }
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(scaleLoadInputA));
                auto scaleInputA = *tagLoadScaleA;
                if(gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                {
                    AssertFatal(gemm.scaleShuffleTileA.size() == 3);
                    scaleInputA = command->addOperation(rocRoller::Operations::SubTileTranspose(
                        scaleInputA, gemm.scaleShuffleTileA));
                }
                tagBlockScaleA = mulInputA
                    = command->addOperation(rocRoller::Operations::BlockScale(
                        tagLoadA,
                        2,
                        scaleInputA,
                        {1, static_cast<unsigned int>(gemm.scaleBlockSize)}));
            }
            else if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
            {
                tagTensorScaleA
                    = command->addOperation(rocRoller::Operations::Scalar(gemm.scaleTypeA));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Scalar(*tagTensorScaleA));
                tagBlockScaleA = mulInputA = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadA, 0, tagLoadScaleA));
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                bool scalePreTiledB  = !gemm.scalePretileB.empty();
                tagTensorScaleB      = command->addOperation(rocRoller::Operations::Tensor(
                    2, gemm.scaleTypeB, {}, gemm.transB == "N" ? oneStridesN : oneStridesT));
                auto scaleLoadInputB = *tagTensorScaleB;
                if(scalePreTiledB)
                {
                    AssertFatal(gemm.transB == "N");
                    AssertFatal(gemm.scalePretileB.size() == 2);
                    scaleLoadInputB = command->addOperation(rocRoller::Operations::SubTileTranspose(
                        scaleLoadInputB, gemm.scalePretileB, false));
                }
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(scaleLoadInputB));
                auto scaleInputB = *tagLoadScaleB;
                if(gemm.scaleSkipPermlane == ScaleSkipPermlaneMode::PreSwizzleScale)
                {
                    AssertFatal(gemm.scaleShuffleTileB.size() == 3);
                    scaleInputB = command->addOperation(rocRoller::Operations::SubTileTranspose(
                        scaleInputB, gemm.scaleShuffleTileB));
                }
                tagBlockScaleB = mulInputB
                    = command->addOperation(rocRoller::Operations::BlockScale(
                        tagLoadB,
                        2,
                        scaleInputB,
                        {static_cast<unsigned int>(gemm.scaleBlockSize), 1}));
            }
            else if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
            {
                tagTensorScaleB
                    = command->addOperation(rocRoller::Operations::Scalar(gemm.scaleTypeB));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Scalar(*tagTensorScaleB));
                tagBlockScaleB = mulInputB = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadB, 0, tagLoadScaleB));
            }

            auto tagTensorC = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeC, {}, oneStridesN)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagScalarAlpha
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
            auto tagLoadAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

            auto tagScalarBeta
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
            auto tagLoadBeta
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

            auto tagAB = command->addOperation(
                rocRoller::Operations::T_Mul(mulInputA, mulInputB, dataTypeAcc)); // A * B

            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto                             tagBetaC
                = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, tagLoadC)); // beta * C

            auto tagAlphaAB = execute.addXOp(
                rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)

            rocRoller::Operations::OperationTag tagStoreD;
            if(gemm.betaInFma)
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
            }
            else
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
            }

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            auto tagTensorD = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeD, {}, oneStridesN)); // D
            Operations::OperationTag tagScalarSeed;
            if constexpr(std::is_same_v<TC, TD>)
            {
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));
            }
            else
            {
                Operations::OperationTag tagLoadSeed;
                // If Matrix C and D are of different types, an explicit type conversion is required
                if(srCvtSeed.has_value())
                {
                    tagScalarSeed = command->addOperation(
                        rocRoller::Operations::Scalar(DataType::UInt32)); // alpha
                    tagLoadSeed = command->addOperation(
                        rocRoller::Operations::T_Load_Scalar(tagScalarSeed));
                }

                auto cvtOp = rocRoller::Operations::T_Execute(command->getNextTag());
                // (SR)Convert( alpha * (A * B) + beta * C )
                auto tagCvt
                    = srCvtSeed.has_value()
                          ? cvtOp.addXOp(rocRoller::Operations::E_StochasticRoundingCvt(
                              tagStoreD, tagLoadSeed, dataTypeD))
                          : cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagStoreD, dataTypeD));
                tagStoreD = command->addOperation(std::move(cvtOp));
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));
            }

            std::map<Operations::ScratchPolicy, Operations::OperationTag> scratchTags;
            Operations::OperationTag                                      tagNumWGs;
            if(gemm.streamK)
            {
                tagNumWGs      = command->allocateTag();
                auto numWGsArg = command->allocateArgument(DataType::UInt32,
                                                           tagNumWGs,
                                                           ArgumentType::Value,
                                                           DataDirection::ReadOnly,
                                                           rocRoller::NUMWGS);

                scratchTags[Operations::ScratchPolicy::None] = command->allocateTag();
                command->addOperation(
                    rocRoller::Operations::Scratch(scratchTags.at(Operations::ScratchPolicy::None),
                                                   Operations::ScratchPolicy::None));
                command->allocateArgument(
                    VariableType(DataType::UInt32, PointerType::PointerGlobal),
                    scratchTags.at(Operations::ScratchPolicy::None),
                    ArgumentType::Value,
                    DataDirection::ReadWrite,
                    getScratchName(Operations::ScratchPolicy::None));

                scratchTags[Operations::ScratchPolicy::ZeroedBeforeAndAfter]
                    = command->allocateTag();
                command->addOperation(rocRoller::Operations::Scratch(
                    scratchTags.at(Operations::ScratchPolicy::ZeroedBeforeAndAfter),
                    Operations::ScratchPolicy::ZeroedBeforeAndAfter));
                command->allocateArgument(
                    VariableType(DataType::UInt32, PointerType::PointerGlobal),
                    scratchTags.at(Operations::ScratchPolicy::ZeroedBeforeAndAfter),
                    ArgumentType::Value,
                    DataDirection::ReadWrite,
                    getScratchName(Operations::ScratchPolicy::ZeroedBeforeAndAfter));
            }

            Operations::OperationTag tagWGM;
            if(gemm.workgroupMappingDim != -1)
            {
                tagWGM      = command->allocateTag();
                auto wgmArg = command->allocateArgument(DataType::Int32,
                                                        tagWGM,
                                                        ArgumentType::Value,
                                                        DataDirection::ReadOnly,
                                                        rocRoller::WGM);
            }

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});

            if(gemm.workgroupClusterSizeX > 0 || gemm.workgroupClusterSizeY > 0
               || gemm.workgroupClusterSizeZ > 0)
            {
#ifndef ROCROLLER_HAS_HIP_WORKGROUP_CLUSTERS
                GTEST_SKIP() << "Workgroup cluster feature is disabled: the installed ROCm/HIP "
                                "version does not support hipLaunchAttributeClusterDimension.";
#endif
                REQUIRE_ARCH_CAP(GPUCapability::HasWorkgroupClusters);

                auto const workgroupClusterSizeX
                    = gemm.workgroupClusterSizeX > 0 ? gemm.workgroupClusterSizeX : 1;
                auto const workgroupClusterSizeY
                    = gemm.workgroupClusterSizeY > 0 ? gemm.workgroupClusterSizeY : 1;
                auto const workgroupClusterSizeZ
                    = gemm.workgroupClusterSizeZ > 0 ? gemm.workgroupClusterSizeZ : 1;

                AssertFatal(workgroupClusterSizeX * workgroupClusterSizeY * workgroupClusterSizeZ
                                <= WorkgroupClustersDetail::MaxWorkgroupsPerCluster,
                            fmt::format("Requested ClusterSize {}x{}x{} exceeds maximum allowed "
                                        "number of workgroups per cluster ({})",
                                        ShowValue(workgroupClusterSizeX),
                                        ShowValue(workgroupClusterSizeY),
                                        ShowValue(workgroupClusterSizeZ),
                                        WorkgroupClustersDetail::MaxWorkgroupsPerCluster));

                params->setManualWorkgroupClusterSize(
                    {workgroupClusterSizeX, workgroupClusterSizeY, workgroupClusterSizeZ});
            }
            // TODO: Calculate these values internally based on workgroup sizes.
            params->setManualWavefrontCount(
                {static_cast<uint>(gemm.macM / gemm.waveM / wavetilePerWavefrontM),
                 static_cast<uint>(gemm.macN / gemm.waveN / wavetilePerWavefrontN)});
            params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);
            params->setSplitStoreTileIntoWaveBlocks(gemm.splitStoreTileIntoWaveBlocks);

            // Set LDS padding for MATRIX_A and MATRIX_B
            params->ldsPadding[LayoutType::MATRIX_A] = gemm.padA;
            params->ldsPadding[LayoutType::MATRIX_B] = gemm.padB;

            params->swizzleScale                  = gemm.swizzleScale;
            params->prefetchScale                 = gemm.prefetchScale;
            params->fuseLoops                     = gemm.fuseLoops;
            params->tailLoops                     = gemm.tailLoops;
            params->allowAmbiguousMemoryNodes     = gemm.allowAmbiguousMemoryNodes;
            params->unrollK                       = gemm.unrollK;
            params->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
            params->prefetch                      = gemm.prefetch;
            params->prefetchInFlight              = gemm.prefetchInFlight;
            params->prefetchLDSFactor             = gemm.prefetchLDSFactor;
            params->prefetchMixMemOps             = gemm.prefetchMixMemOps;
            params->transposeMemoryAccess.set(LayoutType::MATRIX_A, gemm.transA == "T");
            params->transposeMemoryAccess.set(LayoutType::MATRIX_B, gemm.transB == "T");

            if(gemm.workgroupMappingDim != -1)
            {
                params->workgroupMappingDim = gemm.workgroupMappingDim;
            }

            if(gemm.workgroupRemapXCC)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
                params->workgroupRemapXCC = m_context->targetArchitecture().GetCapability(
                    GPUCapability::DefaultRemapXCCValue);
            }

            if(gemm.loopOverTiles > 0)
            {
                params->loopOverOutputTilesDimensions = {0, 1};
                params->loopOverOutputTilesCoordSizes
                    = {static_cast<uint>(M / gemm.macM), static_cast<uint>(N / gemm.macN)};
                params->loopOverOutputTilesIteratedTiles = 2;
            }

            if(gemm.streamK)
            {
                AssertFatal(
                    numWorkgroupY == 1,
                    "Current scratch space implementation assumes that the kernel is launched "
                    "with numWorkgroupY == 1");

                params->loopOverOutputTilesDimensions = {0, 1};
                params->streamK                       = gemm.streamK;
            }

            auto memoryTypeA = GetMemoryType(gemm.loadPathA);
            auto memoryTypeB = GetMemoryType(gemm.loadPathB);

            {
                auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    memoryTypeA);
                params->setDimensionInfo(tagLoadA, macTileA);
            }

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.waveK % gemm.scaleBlockSize == 0,
                            fmt::format("waveK: {} must be a multiple of the scale block size: {}",
                                        gemm.waveK,
                                        gemm.scaleBlockSize));
                AssertFatal(gemm.loadScalePathA != SolutionParams::LoadPath::BufferToLDS
                                || gemm.swizzleScale,
                            "If loadScalePathA is BufferToLDS, swizzleScale must be enabled");
                auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK / gemm.scaleBlockSize},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    GetMemoryType(gemm.loadScalePathA),
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    {gemm.swizzleM, gemm.swizzleN, gemm.swizzleK, gemm.swizzleB});
                params->setDimensionInfo(*tagLoadScaleA, macTileAScale);
            }

            {
                auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    memoryTypeB);
                params->setDimensionInfo(tagLoadB, macTileB);
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.waveK % gemm.scaleBlockSize == 0,
                            fmt::format("waveK: {} must be a multiple of the scale block size: {}",
                                        gemm.waveK,
                                        gemm.scaleBlockSize));
                AssertFatal(gemm.loadScalePathB != SolutionParams::LoadPath::BufferToLDS
                                || gemm.swizzleScale,
                            "If loadScalePathB is BufferToLDS, swizzleScale must be enabled");
                auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK / gemm.scaleBlockSize, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    GetMemoryType(gemm.loadScalePathB),
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    {gemm.swizzleM, gemm.swizzleN, gemm.swizzleK, gemm.swizzleB});
                params->setDimensionInfo(*tagLoadScaleB, macTileBScale);
            }

            {
                auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB});
                params->setDimensionInfo(tagLoadC, macTileC);
            }

            {
                auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    gemm.storePath == SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer
                        ? MemoryType::WAVE_LDS
                        : MemoryType::WAVE);
                params->setDimensionInfo(tagStoreD, macTileD);
            }

            CommandKernel commandKernel(command, testKernelName());

            // TODO Some test have loops, we need to reset the context.
            m_context = createContext();

            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            setCommandTensorArg(commandArgs, tagTensorA, descA, deviceA.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, deviceB.get());
            setCommandTensorArg(commandArgs, tagTensorC, descC, deviceC.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, deviceD.get());

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(K % gemm.scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        K,
                                        gemm.scaleBlockSize));
                TensorDescriptor descAScale
                    = TensorDescriptor(gemm.scaleTypeA,
                                       {static_cast<size_t>(M), size_t(K / gemm.scaleBlockSize)},
                                       gemm.transA);
                setCommandTensorArg(
                    commandArgs, tagTensorScaleA.value(), descAScale, deviceScaleA.get());
            }
            else if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
            {
                commandArgs.setArgument(*tagTensorScaleA, ArgumentType::Value, hostScaleA[0]);
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(K % gemm.scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        K,
                                        gemm.scaleBlockSize));
                TensorDescriptor descBScale
                    = TensorDescriptor(gemm.scaleTypeB,
                                       {size_t(K / gemm.scaleBlockSize), static_cast<size_t>(N)},
                                       gemm.transB);
                setCommandTensorArg(
                    commandArgs, tagTensorScaleB.value(), descBScale, deviceScaleB.get());
            }
            else if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
            {
                commandArgs.setArgument(*tagTensorScaleB, ArgumentType::Value, hostScaleB[0]);
            }

            commandArgs.setArgument(tagScalarAlpha, ArgumentType::Value, alpha);
            commandArgs.setArgument(tagScalarBeta, ArgumentType::Value, beta);
            if(srCvtSeed.has_value())
                commandArgs.setArgument(tagScalarSeed, ArgumentType::Value, srCvtSeed.value());

            // Create scratch space
            size_t scratchSpaceRequired[static_cast<int>(Operations::ScratchPolicy::Count)];
            std::shared_ptr<uint8_t>
                deviceScratch[static_cast<int>(Operations::ScratchPolicy::Count)];
            std::fill(std::begin(scratchSpaceRequired), std::end(scratchSpaceRequired), 0);
            std::fill(std::begin(deviceScratch), std::end(deviceScratch), nullptr);
            if(gemm.streamK)
            {
                commandArgs.setArgument(tagNumWGs, ArgumentType::Value, gemm.numWGs);
                for(int i = 0; i < static_cast<int>(Operations::ScratchPolicy::Count); ++i)
                {
                    auto policy             = static_cast<Operations::ScratchPolicy>(i);
                    scratchSpaceRequired[i] = commandKernel.scratchSpaceRequired(
                        policy, commandArgs.runtimeArguments());
                    if(scratchSpaceRequired[i] > 0)
                    {
                        deviceScratch[i] = make_shared_device<uint8_t>(scratchSpaceRequired[i], 0);
                        commandArgs.setArgument(
                            scratchTags.at(policy), ArgumentType::Value, deviceScratch[i].get());
                    }
                }
            }

            if(gemm.workgroupMappingDim != -1)
            {
                commandArgs.setArgument(tagWGM, ArgumentType::Value, gemm.workgroupMappingValue);
            }

            // Host result
            std::vector<TD> h_result(M * N, TD{});
            if(gemm.scaleAMode != Operations::ScaleMode::None
               || gemm.scaleBMode != Operations::ScaleMode::None)
            {
                rocRoller::ScaledCPUMM(h_result,
                                       hostC,
                                       hostA,
                                       hostB,
                                       hostScaleA,
                                       hostScaleB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       beta,
                                       gemm.transA == "T",
                                       gemm.transB == "T",
                                       gemm.scaleBlockSize,
                                       gemm.scaleTypeA,
                                       gemm.scaleTypeB);
            }
            else if constexpr(std::is_same_v<TC, TD>)
            {
                rocRoller::CPUMM(h_result,
                                 hostC,
                                 hostA,
                                 hostB,
                                 M,
                                 N,
                                 K,
                                 alpha,
                                 beta,
                                 gemm.transA == "T",
                                 gemm.transB == "T");
            }
            else
            {
                std::vector<TC> hostD(M * N, TC{});
                rocRoller::CPUMM(hostD,
                                 hostC,
                                 hostA,
                                 hostB,
                                 M,
                                 N,
                                 K,
                                 alpha,
                                 beta,
                                 gemm.transA == "T",
                                 gemm.transB == "T");
                ASSERT_EQ(hostD.size(), h_result.size());
                bool const isSRConversion = srCvtSeed.has_value();
                for(size_t i = 0; i < hostD.size(); i++)
                {
                    if(isSRConversion)
                    {
                        // SR conversion currently only supports F32 to FP8/BF8
                        AssertFatal((std::is_same_v<TC, float>),
                                    "Source type of SR conversion only accepts float");
                        AssertFatal((std::is_same_v<TD, FP8>) || (std::is_same_v<TD, BF8>),
                                    "Destionation type of SR conversion can only be FP8/BF8");

                        int constexpr exp_width      = std::is_same_v<TD, FP8> ? 4 : 5;
                        int constexpr mantissa_width = 7 - exp_width;
                        bool constexpr is_bf8        = std::is_same_v<TD, BF8>;

                        auto const f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);

                        if(f8Mode == rocRoller::F8Mode::NaNoo)
                        {
                            h_result[i].data = DataTypes::cast_to_f8<mantissa_width,
                                                                     exp_width,
                                                                     float,
                                                                     false /* is_ocp */,
                                                                     is_bf8,
                                                                     true /*negative_zero_nan*/,
                                                                     true /*clip*/>(
                                hostD[i],
                                true /* is stochastic rounding? */,
                                srCvtSeed.value() /* seed for stochastic rounding */);
                        }
                        else
                        {
                            h_result[i].data = DataTypes::cast_to_f8<mantissa_width,
                                                                     exp_width,
                                                                     float,
                                                                     true /* is_ocp */,
                                                                     is_bf8,
                                                                     true /*negative_zero_nan*/,
                                                                     true /*clip*/>(
                                hostD[i],
                                true /* is stochastic rounding? */,
                                srCvtSeed.value() /* seed for stochastic rounding */);
                        }
                    }
                    else
                        h_result[i] = TD(hostD[i]);
                }
            }

            // Device result
            std::vector<TD> d_result(M * N);

            for(int iteration = 0; iteration < numIters; ++iteration)
            {
                ASSERT_THAT(hipMemset(deviceD.get(), 0, M * N * sizeof(TD)), HasHipSuccess(0));
                if(iteration == 0)
                {
                    for(int i = 0; i < static_cast<int>(Operations::ScratchPolicy::Count); ++i)
                    {
                        if(scratchSpaceRequired[i] > 0)
                        {
                            ASSERT_THAT(
                                hipMemset(deviceScratch[i].get(), 0, scratchSpaceRequired[i]),
                                HasHipSuccess(0));
                        }
                    }
                }

                commandKernel.launchKernel(commandArgs.runtimeArguments());

                ASSERT_THAT(
                    hipMemcpy(
                        d_result.data(), deviceD.get(), M * N * sizeof(TD), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

                auto tol = gemmAcceptableError<TA, TB, TD>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(d_result, h_result, tol);
                Log::info("RNorm is {} (acceptable {}, iteration {})",
                          res.relativeNormL2,
                          res.acceptableError.relativeL2Tolerance,
                          iteration);

                // Verify ZeroedBeforeAndAfter scratch is all zeros after kernel execution
                auto zeroedIdx
                    = static_cast<size_t>(Operations::ScratchPolicy::ZeroedBeforeAndAfter);
                if(scratchSpaceRequired[zeroedIdx] > 0)
                {
                    std::vector<uint8_t> zeroedResult(scratchSpaceRequired[zeroedIdx]);
                    ASSERT_THAT(hipMemcpy(zeroedResult.data(),
                                          deviceScratch[zeroedIdx].get(),
                                          scratchSpaceRequired[zeroedIdx],
                                          hipMemcpyDeviceToHost),
                                HasHipSuccess(0));

                    bool allZeros = true;
                    for(size_t i = 0; i < zeroedResult.size(); ++i)
                    {
                        if(zeroedResult[i] != 0)
                        {
                            allZeros = false;
                            // Print as uint32 since flags are UInt32
                            size_t flagIndex = i / sizeof(uint32_t);
                            std::cerr << "Non-zero at byte " << i << " (flag index " << flagIndex
                                      << "): " << static_cast<int>(zeroedResult[i]) << std::endl;
                        }
                    }
                    EXPECT_TRUE(allZeros)
                        << "ZeroedBeforeAndAfter scratch should be all zeros after kernel "
                           "execution (size="
                        << scratchSpaceRequired[zeroedIdx] << " bytes)";
                }

                if(debuggable && !res.ok)
                {
                    for(size_t i = 0; i < M; i++)
                    {
                        for(size_t j = 0; j < N; j++)
                        {
                            auto a = d_result[i * N + j];
                            auto b = h_result[i * N + j];
                            if((a - b) * (a - b) / (b * b)
                               > res.acceptableError.relativeL2Tolerance)
                            {
                                std::cout << std::setw(8) << i << std::setw(8) << j //
                                          << std::setw(16) << std::scientific << a //
                                          << std::setw(16) << std::scientific << b //
                                          << std::setw(16) << std::scientific << a - b //
                                          << std::endl;
                            }
                        }
                    }
                }
                EXPECT_TRUE(res.ok) << res.message();
            }
        }

        template <typename TA>
        void basicGEMMMixed(rocRoller::DataType typeB, GEMMProblem const& problem)
        {
            using namespace rocRoller;
            if(typeB == rocRoller::DataType::FP8)
                basicGEMM<TA, FP8, float>(problem);
            else if(typeB == rocRoller::DataType::BF8)
                basicGEMM<TA, BF8, float>(problem);
            else if(typeB == rocRoller::DataType::FP6)
                basicGEMM<TA, FP6, float>(problem);
            else if(typeB == rocRoller::DataType::BF6)
                basicGEMM<TA, BF6, float>(problem);
            else if(typeB == rocRoller::DataType::FP4)
                basicGEMM<TA, FP4, float>(problem);
            else
                Throw<FatalError>("Invalid type.");
        }

        void basicGEMMMixed(rocRoller::DataType typeA,
                            rocRoller::DataType typeB,
                            GEMMProblem const&  problem)
        {
            using namespace rocRoller;
            if(typeA == rocRoller::DataType::FP8)
                basicGEMMMixed<FP8>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF8)
                basicGEMMMixed<BF8>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP6)
                basicGEMMMixed<FP6>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF6)
                basicGEMMMixed<BF6>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP4)
                basicGEMMMixed<FP4>(typeB, problem);
            else
                Throw<FatalError>("Invalid type.");
        }
    };
}
