/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include <common/mxDataGen.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;

namespace MatrixMultiplyTest
{

    template <typename T>
    concept isF8 = std::is_same_v<T, FP8> || std::is_same_v<T, BF8>;

    template <typename T>
    concept isF6F4 = std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP4>;

    template <typename T>
    concept isF16 = std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>;

    template <typename T>
    concept isF32 = std::is_same_v<T, float>;

    /**
     * @brief Return a reasonable random value range for datatype T.
     *
     * The return value is usually passed to the random generator to
     * obtain values in (-range, range), and these will be used to
     * populate matrices for (small) GEMM problems.
     *
     * The value returned *may or may not* correspond to the maximum
     * representable value of T.
     */
    template <typename T>
    float range()
    {
        // Not the maximum range.
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, Half>)
            return 10.f;
        // Maximum range
        if constexpr(std::is_same_v<T, FP8>)
            return 448.f;
        // Maximum range; kinda extreme
        if constexpr(std::is_same_v<T, BF8>)
            return 57344.f;
        // Maximum range
        if constexpr(std::is_same_v<T, FP6>)
            return 7.5f;
        // Maximum range
        if constexpr(std::is_same_v<T, BF6>)
            return 28.f;
        // FP4, maximum range
        return 6.f;
    }

    struct ScaleParams
    {
        DataType scaleTypeA     = DataType::None;
        DataType scaleTypeB     = DataType::None;
        uint     scaleBlockSize = 1;
    };

    template <typename... Ts>
    class BaseMatrixMultiplyContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<rocRoller::GPUArchitectureTarget, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            GPUArchitectureTarget device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

    public:
        CommandKernelPtr commandKernel;

        template <typename TA, typename TB, typename TD, typename ACC = float>
        void matrixMultiplyMacroTile(int               wave_m,
                                     int               wave_n,
                                     int               wave_k,
                                     int               wave_b,
                                     bool              useLDSB     = true,
                                     std::string       transA      = "N",
                                     std::string       transB      = "N",
                                     const ScaleParams scaleParams = {})
        {
            commandKernel = nullptr;

            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_fp8,
                                        GPUCapability::HasWMMA_f32_16x16x16_f8);
            }
            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            if((isF8<TA> || isF8<TB>)&&(wave_k >= 64))
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            const bool scaleA         = scaleParams.scaleTypeA != DataType::None;
            const bool scaleB         = scaleParams.scaleTypeB != DataType::None;
            const auto scaleTypeA     = scaleParams.scaleTypeA;
            const auto scaleTypeB     = scaleParams.scaleTypeB;
            const auto scaleBlockSize = scaleParams.scaleBlockSize;

            if(scaleA || scaleB)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
                const auto& arch = m_context->targetArchitecture();
                AssertFatal(!scaleA || arch.isSupportedScaleType(scaleTypeA),
                            fmt::format("Scale A set but target {} does not support scale type {}.",
                                        arch.target().toString(),
                                        toString(scaleTypeA)));
                AssertFatal(!scaleB || arch.isSupportedScaleType(scaleTypeB),
                            fmt::format("Scale B set but target {} does not support scale type {}.",
                                        arch.target().toString(),
                                        toString(scaleTypeB)));
            }

            auto dataTypeA   = TypeInfo<TA>::Var.dataType;
            auto dataTypeB   = TypeInfo<TB>::Var.dataType;
            auto dataTypeD   = TypeInfo<TD>::Var.dataType;
            auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

            // matrix size: A is MxK; B is KxN; D is MxN
            int mac_m = wave_m;
            int mac_n = wave_n;
            int mac_k = 32;

            unsigned M = mac_m;
            unsigned N = mac_n;
            unsigned K = 32;

            if constexpr(isF8<TA> && isF8<TB>)
            {
                mac_k = 2 * wave_k;
                K     = 2 * mac_k;
            }
            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                mac_k = 2 * wave_k;
                K     = 4 * mac_k;
            }

            if constexpr(isF16<TA> || isF16<TB>)
            {
                mac_k = 4 * wave_k;
                K     = 8 * mac_k;
            }

            if constexpr(isF32<TA> || isF32<TB>)
            {
                mac_k = 4 * wave_k;
                K     = 8 * mac_k;
            }

            Log::debug("MatrixMultiplyMacroTile: Matrix {}x{}x{}", M, N, K);
            Log::debug("MatrixMultiplyMacroTile: WGTile {}x{}x{}", mac_m, mac_n, mac_k);
            Log::debug("MatrixMultiplyMacroTile: MI   {}x{}x{}x{}", wave_m, wave_n, wave_k, wave_b);

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");
            AssertFatal(K % mac_k == 0, "MacroTile size mismatch (K)");

            AssertFatal(mac_m == wave_m, "Single output MacroTile.");
            AssertFatal(mac_n == wave_n, "Single output MacroTile.");

            auto const& arch             = m_context->targetArchitecture();
            uint        workgroup_size_x = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
            uint        workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeA).elementBits, 8u);
            AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

            auto NX = std::make_shared<Expression::Expression>(workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            std::vector<size_t> unitStridesN = {1, 0};
            std::vector<size_t> unitStridesT = {0, 1};

            auto command    = std::make_shared<Command>();
            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeA, transA == "N" ? unitStridesN : unitStridesT));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeB, transB == "N" ? unitStridesN : unitStridesT));
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            std::optional<rocRoller::Operations::OperationTag> tagTensorScaleA, tagLoadScaleA,
                tagTensorScaleB, tagLoadScaleB;

            if(scaleA)
            {
                tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
                    2, scaleTypeA, transA == "N" ? unitStridesN : unitStridesT));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleA));
            }

            if(scaleB)
            {
                tagTensorScaleB = command->addOperation(rocRoller::Operations::Tensor(
                    2, scaleTypeB, transB == "N" ? unitStridesN : unitStridesT));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleB));
            }

            rocRoller::Operations::OperationTag tagStoreD;

            if(!scaleA)
            {
                ASSERT_FALSE(scaleB);

                tagStoreD = command->addOperation(
                    rocRoller::Operations::T_Mul(tagLoadA, tagLoadB, dataTypeAcc)); // D = A * B
            }
            else
            {
                ASSERT_TRUE(scaleB);

                AssertFatal(
                    arch.isSupportedScaleBlockSize(scaleBlockSize),
                    fmt::format("Architecture {} does not support block scaling (size: {}).",
                                arch.target().toString(),
                                scaleBlockSize));

                auto scaledA = command->addOperation(rocRoller::Operations::BlockScale(
                    tagLoadA, 2, tagLoadScaleA, {1, scaleBlockSize}));
                auto scaledB = command->addOperation(rocRoller::Operations::BlockScale(
                    tagLoadB, 2, tagLoadScaleB, {scaleBlockSize, 1}));

                tagStoreD = command->addOperation(
                    rocRoller::Operations::T_Mul(scaledA, scaledB, dataTypeAcc)); // D = A * B
            }

            auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD));
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            params->packMultipleElementsInto1VGPR = true;
            params->enableLongDwordInstructions   = true;

            params->transposeMemoryAccess.set(LayoutType::MATRIX_A, transA == "T");
            params->transposeMemoryAccess.set(LayoutType::MATRIX_B, transB == "T");

            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA
                = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                          LayoutType::MATRIX_A,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          MemoryType::WAVE_LDS);
            params->setDimensionInfo(tagLoadA, macTileA);

            if(scaleA)
            {
                AssertFatal(wave_k % scaleBlockSize == 0,
                            fmt::format("wave_k: {} must be a multiple of the scale block size: {}",
                                        wave_k,
                                        scaleBlockSize));
                auto macTileScaleA = KernelGraph::CoordinateGraph::MacroTile(
                    {mac_m, static_cast<int>(mac_k / scaleBlockSize)},
                    LayoutType::MATRIX_A,
                    {wave_m, wave_n, static_cast<int>(wave_k / scaleBlockSize), wave_b},
                    MemoryType::WAVE);
                params->setDimensionInfo(tagLoadScaleA.value(), macTileScaleA);
            }

            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n},
                LayoutType::MATRIX_B,
                {wave_m, wave_n, wave_k, wave_b},
                useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(tagLoadB, macTileB);

            if(scaleB)
            {
                AssertFatal(wave_k % scaleBlockSize == 0,
                            fmt::format("wave_k: {} must be a multiple of the scale block size: {}",
                                        wave_k,
                                        scaleBlockSize));
                auto macTileScaleB = KernelGraph::CoordinateGraph::MacroTile(
                    {static_cast<int>(mac_k / scaleBlockSize), mac_n},
                    LayoutType::MATRIX_B,
                    {wave_m, wave_n, static_cast<int>(wave_k / scaleBlockSize), wave_b},
                    MemoryType::WAVE);
                params->setDimensionInfo(tagLoadScaleB.value(), macTileScaleB);
            }

            params->setManualWavefrontCount({1u, 1u});

            commandKernel = std::make_shared<CommandKernel>(command, "MatrixMultiplyMacroTile");
            commandKernel->setContext(m_context);
            commandKernel->setCommandParameters(params);
            commandKernel->generateKernel();

            if(isLocalDevice())
            {
                TensorDescriptor descA(dataTypeA, {M, K}, transA);
                TensorDescriptor descB(dataTypeB, {K, N}, transB);
                TensorDescriptor descD(dataTypeD, {M, N}, {1u, M});

                float rangeA = range<TA>();
                float rangeB = range<TB>();

                uint32_t seed = 9861u;

                auto       blockScalingA = (scaleA) ? scaleBlockSize : 1;
                auto       blockScalingB = (scaleB) ? scaleBlockSize : 1;
                const auto dgenA
                    = getDataGenerator<TA>(descA, -rangeA, rangeA, seed, blockScalingA);
                const auto dgenB
                    = getDataGenerator<TB>(descB, -rangeB, rangeB, seed, blockScalingB);

                auto A = getRandomVector<TA>(dgenA, scaleA);
                auto B = getRandomVector<TB>(dgenB, scaleB);

                std::vector<uint8_t> hostScaleA, hostScaleB;

                auto d_A = make_shared_device(A);
                auto d_B = make_shared_device(B);
                auto d_D = make_shared_device<ACC>(descD.totalAllocatedElements());

                std::shared_ptr<uint8_t> d_scaleA, d_scaleB;

                if(scaleA)
                {
                    hostScaleA = dgenA.getScaleBytes();
                    d_scaleA   = make_shared_device(hostScaleA);
                }
                if(scaleB)
                {
                    hostScaleB = dgenB.getScaleBytes();
                    d_scaleB   = make_shared_device(hostScaleB);
                }

                CommandArguments commandArgs = command->createArguments();

                setCommandTensorArg(commandArgs, tagTensorA, descA, (TA*)d_A.get());
                setCommandTensorArg(commandArgs, tagTensorB, descB, (TB*)d_B.get());
                setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

                if(scaleA)
                {
                    AssertFatal(K % scaleBlockSize == 0,
                                fmt::format("K: {} must be a multiple of the scale block size: {}",
                                            K,
                                            scaleBlockSize));
                    TensorDescriptor descScaleA(dataTypeA, {M, K / scaleBlockSize}, transA);
                    setCommandTensorArg(
                        commandArgs, tagTensorScaleA.value(), descScaleA, d_scaleA.get());
                }
                if(scaleB)
                {
                    AssertFatal(K % scaleBlockSize == 0,
                                fmt::format("K: {} must be a multiple of the scale block size: {}",
                                            K,
                                            scaleBlockSize));
                    TensorDescriptor descScaleB(dataTypeB, {K / scaleBlockSize, N}, transB);
                    setCommandTensorArg(
                        commandArgs, tagTensorScaleB.value(), descScaleB, d_scaleB.get());
                }

                commandKernel->launchKernel(commandArgs.runtimeArguments());

                std::vector<TD> D(descD.totalAllocatedElements());
                ASSERT_THAT(hipMemcpy(D.data(),
                                      d_D.get(),
                                      descD.totalAllocatedElements() * sizeof(TD),
                                      hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<TD> c_D(descD.totalAllocatedElements(), TD{});
                std::vector<TD> c_C(descD.totalAllocatedElements(), TD{});

                float alpha = 1.0f;

                if(scaleA)
                {
                    ASSERT_TRUE(scaleB);

                    rocRoller::ScaledCPUMM(c_D,
                                           c_C,
                                           A,
                                           B,
                                           hostScaleA,
                                           hostScaleB,
                                           M,
                                           N,
                                           K,
                                           alpha,
                                           0.0,
                                           transA == "T",
                                           transB == "T",
                                           scaleBlockSize,
                                           scaleTypeA,
                                           scaleTypeB);
                }
                else
                {
                    ASSERT_FALSE(scaleB);
                    CPUMM(c_D, c_C, A, B, M, N, K, alpha, 0.0, transA == "T", transB == "T");
                }

                auto tol = gemmAcceptableError<TA, TB, TD>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }

        template <typename TA>
        void matrixMultiplyMacroTileMixed(rocRoller::DataType typeB,
                                          int                 m,
                                          int                 n,
                                          int                 k,
                                          int                 b,
                                          bool                useLDSB     = true,
                                          std::string         transA      = "N",
                                          std::string         transB      = "N",
                                          const ScaleParams   scaleParams = {})
        {
            if(typeB == rocRoller::DataType::FP8)
                matrixMultiplyMacroTile<TA, FP8, float>(
                    m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeB == rocRoller::DataType::BF8)
                matrixMultiplyMacroTile<TA, BF8, float>(
                    m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeB == rocRoller::DataType::FP6)
                matrixMultiplyMacroTile<TA, FP6, float>(
                    m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeB == rocRoller::DataType::BF6)
                matrixMultiplyMacroTile<TA, BF6, float>(
                    m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeB == rocRoller::DataType::FP4)
                matrixMultiplyMacroTile<TA, FP4, float>(
                    m, n, k, b, useLDSB, transA, transB, scaleParams);
            else
                Throw<FatalError>("Invalid type.");
        }

        void matrixMultiplyMacroTileMixed(rocRoller::DataType typeA,
                                          rocRoller::DataType typeB,
                                          int                 m,
                                          int                 n,
                                          int                 k,
                                          int                 b,
                                          bool                useLDSB     = true,
                                          std::string         transA      = "N",
                                          std::string         transB      = "N",
                                          const ScaleParams   scaleParams = {})
        {
            if(typeA == rocRoller::DataType::FP8)
                matrixMultiplyMacroTileMixed<FP8>(
                    typeB, m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeA == rocRoller::DataType::BF8)
                matrixMultiplyMacroTileMixed<BF8>(
                    typeB, m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeA == rocRoller::DataType::FP6)
                matrixMultiplyMacroTileMixed<FP6>(
                    typeB, m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeA == rocRoller::DataType::BF6)
                matrixMultiplyMacroTileMixed<BF6>(
                    typeB, m, n, k, b, useLDSB, transA, transB, scaleParams);
            else if(typeA == rocRoller::DataType::FP4)
                matrixMultiplyMacroTileMixed<FP4>(
                    typeB, m, n, k, b, useLDSB, transA, transB, scaleParams);
            else
                Throw<FatalError>("Invalid type.");
        }

        template <typename TA, typename TB, typename TD, typename ACC = float>
        void matrixMultiplyAB(int  wave_m,
                              int  wave_n,
                              int  wave_k,
                              int  wave_b,
                              bool useLDS = false,
                              bool transA = false,
                              bool transB = false)
        {
            // matrix size: A is MxK; B is KxN; D is MxN
            int const M = 1024;
            int const N = 1024;
            int const K = 512;

            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_fp8,
                                        GPUCapability::HasWMMA_f32_16x16x16_f8);
            }
            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            auto dataTypeA   = TypeInfo<TA>::Var.dataType;
            auto dataTypeB   = TypeInfo<TB>::Var.dataType;
            auto dataTypeD   = TypeInfo<TD>::Var.dataType;
            auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

            const auto wavefrontCountX = 2;
            const auto wavefrontCountY = 2;

            // output macro tile size; we will launch 2x2 waves
            int mac_m = wavefrontCountX * wave_m;
            int mac_n = wavefrontCountY * wave_n;
            int mac_k = 2 * wave_k;

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

            auto       arch = m_context->targetArchitecture();
            const auto wfs  = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

            uint workgroup_size_x = wavefrontCountX * wavefrontCountY * wfs;
            uint workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeA).elementBits, 8u);
            AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

            uint num_workgroup_x = M / mac_m;
            uint num_workgroup_y = N / mac_n;

            auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            TensorDescriptor descA(dataTypeA, {M, K}, transA ? "T" : "N");
            TensorDescriptor descB(dataTypeB, {K, N}, transB ? "T" : "N");
            TensorDescriptor descD(dataTypeD, {M, N}, {1u, M});

            auto seed = 9861u;
            auto A    = DGenVector<TA>(descA, -1.f, 1.f, seed + 1);
            auto B    = DGenVector<TB>(descB, -1.f, 1.f, seed + 2);

            auto d_A = make_shared_device(A);
            auto d_B = make_shared_device(B);
            auto d_D = make_shared_device<ACC>(M * N);

            auto command = std::make_shared<Command>();

            std::vector<size_t> unitStridesN = {1, 0};
            std::vector<size_t> unitStridesT = {0, 1};

            auto tagTensorA = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeA, transA ? unitStridesT : unitStridesN));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeB, transB ? unitStridesT : unitStridesN)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagStoreD = command->addOperation(
                rocRoller::Operations::T_Mul(tagLoadA, tagLoadB, dataTypeAcc)); // D = A * B

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            setCommandTensorArg(commandArgs, tagTensorA, descA, (TA*)d_A.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, (TB*)d_B.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_k},
                LayoutType::MATRIX_A,
                {wave_m, wave_n, wave_k, wave_b},
                useLDS ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n},
                LayoutType::MATRIX_B,
                {wave_m, wave_n, wave_k, wave_b},
                useLDS ? MemoryType::WAVE_LDS : MemoryType::WAVE);

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);
            params->setManualWavefrontCount({wavefrontCountX, wavefrontCountY});
            params->transposeMemoryAccess.set(LayoutType::MATRIX_A, transA);
            params->transposeMemoryAccess.set(LayoutType::MATRIX_B, transB);

            CommandKernel commandKernel(command, "MatrixMultiplyAB");
            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<TD> D(M * N);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(TD), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<TD> c_D(M * N, TD{});
                std::vector<TD> c_C(M * N, TD{});

                CPUMM(c_D, c_C, A, B, M, N, K, 1.0f, 0.0, transA, transB);

                auto tol = gemmAcceptableError<TA, TB, TD>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }

        template <typename T, typename ACC = float>
        void matrixMultiplyABC(int wave_m, int wave_n, int wave_k, int wave_b)
        {
            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);

            // matrix size: A is MxK; B is KxN; D is MxN
            unsigned M = 1024;
            unsigned N = 1024;
            unsigned K = 512;

            const auto wavefrontCountX = 2;
            const auto wavefrontCountY = 2;

            // output macro tile size
            int mac_m = wavefrontCountX * wave_m;
            int mac_n = wavefrontCountY * wave_n;
            int mac_k = 2 * wave_k;

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

            auto       arch = m_context->targetArchitecture();
            const auto wfs  = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

            uint workgroup_size_x = wavefrontCountX * wavefrontCountY * wfs;
            uint workgroup_size_y = 1;

            uint num_workgroup_x = M / mac_m;
            uint num_workgroup_y = N / mac_n;

            auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
            auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            auto dataType    = TypeInfo<T>::Var.dataType;
            auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

            TensorDescriptor descA(dataType, {M, K}, {1u, M});
            TensorDescriptor descB(dataType, {K, N}, {1u, K});
            TensorDescriptor descC(dataType, {M, N}, {1u, M});
            TensorDescriptor descD(dataType, {M, N}, {1u, M});

            auto seed = 9861u;

            auto A = DGenVector<T>(descA, -1.f, 1.f, seed + 1);
            auto B = DGenVector<T>(descB, -1.f, 1.f, seed + 2);
            auto C = DGenVector<T>(descC, -1.f, 1.f, seed + 3);

            auto d_A = make_shared_device(A);
            auto d_B = make_shared_device(B);
            auto d_C = make_shared_device(C);
            auto d_D = make_shared_device<T>(M * N);

            auto command = std::make_shared<Command>();

            auto tagTensorA
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagTensorC
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagAB = command->addOperation(
                rocRoller::Operations::T_Mul(tagLoadA, tagLoadB, dataTypeAcc)); // A * B

            auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
            auto tagStoreD
                = execute.addXOp(rocRoller::Operations::E_Add(tagAB, tagLoadC)); // D = A * B + C
            command->addOperation(std::move(execute));

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            setCommandTensorArg(commandArgs, tagTensorA, descA, (T*)d_A.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, (T*)d_B.get());
            setCommandTensorArg(commandArgs, tagTensorC, descC, (T*)d_C.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);

            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
            auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);
            params->setDimensionInfo(tagLoadC, macTileC);
            params->setManualWavefrontCount({wavefrontCountX, wavefrontCountY});
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            CommandKernel commandKernel(command, "ABC");
            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<T> D(M * N);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<T> c_D(M * N, T{});
                CPUMM(c_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

                auto tol = gemmAcceptableError<T, T, T>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(D, c_D, tol);

                Log::info("RNorm is {}", res.relativeNormL2);
                ASSERT_TRUE(res.ok) << res.message();
            }
        }
    };
}
