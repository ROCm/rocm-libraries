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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Utilities.hpp"

#include <common/GEMMProblem.hpp>

using namespace rocRoller;

namespace rocRollerTest
{
    struct ConversionSettings
    {
        ConversionSettings() = delete;

        ConversionSettings(size_t nx, size_t ny, int m, int n, int t_m, int t_n)
            : nx(nx)
            , ny(ny)
            , m(m)
            , n(n)
            , threadTileM(t_m)
            , threadTileN(t_n)
        {
            AssertFatal(m > 0 && n > 0 && threadTileM > 0 && threadTileN > 0,
                        "Invalid Test Dimensions");
            unsigned int workgroup_size_x = m / threadTileM;
            unsigned int workgroup_size_y = n / threadTileN;

            AssertFatal((size_t)m * n
                            == threadTileM * threadTileN * workgroup_size_x * workgroup_size_y,
                        "MacroTile size mismatch");

            // TODO: Handle when thread tiles include out of range indices
            AssertFatal(nx % threadTileM == 0, "Thread tile size must divide tensor size");
            AssertFatal(ny % threadTileN == 0, "Thread tile size must divide tensor size");
        }

        template <typename SrcType>
        auto generateData(unsigned const seed = 129674u) const
        {
            RandomGenerator random(seed);
            return random.vector<SrcType>(nx * ny, -100.0, 100.0);
        }

        size_t nx; //> tensor size x
        size_t ny; //> tensor size y
        int    m; //> macro tile size x
        int    n; //> macro tile size y
        int    threadTileM; //> thread tile size x
        int    threadTileN; //> thread tile size y
    };

    class ConversionTest : public CurrentGPUContextFixture
    {
    public:
        /*
         *  Testing: D = Convert(A * B + C)
        */
        template <typename TypeAB, typename TypeC, typename TypeD>
        void matrixMultiplyABC(
            int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError);

        /*
         *  Testing: D = Convert(A * B)
        */
        template <typename TypeAB, typename TypeD>
        void matrixMultiply(int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError);

        /*
         *  Testing: C = Convert(A) + Convert(B)
        */
        template <typename DestType, typename SrcType>
        void convertAdd(std::vector<SrcType>& a,
                        std::vector<SrcType>& b,
                        ConversionSettings&   cs,
                        bool const            loadLDS_A);

        /*
         *  Testing: C = Convert(A)
        */
        template <typename DestType, typename SrcType>
        void convertTo(std::vector<SrcType>&         srcData,
                       ConversionSettings const&     cs,
                       bool const                    loadLDS_A,
                       std::optional<uint32_t> const seed = std::nullopt);
    };

    template <typename TypeAB, typename TypeC, typename TypeD>
    void ConversionTest::matrixMultiplyABC(
        int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError)
    {
        // Need to use mfma instructions to calculate (A * B) + C and then convert the
        // result into D
        // e.g., D (Half) = A (Float) * B (Float) + C (Float)
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 512;
        int N = 512;
        int K = 256;

        // output macro tile size
        int mac_m = 2 * wave_m;
        int mac_n = 2 * wave_n;
        int mac_k = 2 * wave_k;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<TypeAB>(M * K, -1.f, 1.f);
        auto B = random.vector<TypeAB>(K * N, -1.f, 1.f);
        auto C = random.vector<TypeC>(M * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_C = make_shared_device(C);
        auto d_D = make_shared_device<TypeD>(M * N);

        auto       command    = std::make_shared<Command>();
        auto const dataTypeAB = TypeInfo<TypeAB>::Var.dataType;
        auto const dataTypeC  = TypeInfo<TypeC>::Var.dataType;
        auto const dataTypeD  = TypeInfo<TypeD>::Var.dataType;

        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // A
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeC)); // C
        auto tagLoadC   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

        auto tagAB
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto tagAdd  = execute.addXOp(rocRoller::Operations::E_Add(tagAB, tagLoadC)); //  A * B + C
        command->addOperation(std::move(execute));

        auto cvtOp  = rocRoller::Operations::T_Execute(command->getNextTag()); // Convert(A * B + C)
        auto tagCvt = cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagAdd, dataTypeD));
        command->addOperation(std::move(cvtOp));

        auto tagTensorD = command->addOperation(
            rocRoller::Operations::Tensor(2, dataTypeD)); // D = Convert(A * B + C)
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        TensorDescriptor descA(dataTypeAB, {size_t(M), size_t(K)}, "N");
        TensorDescriptor descB(dataTypeAB, {size_t(K), size_t(N)}, "N");
        TensorDescriptor descC(dataTypeC, {size_t(M), size_t(N)}, "N");
        TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

        setCommandTensorArg(commandArgs, tagTensorA, descA, d_A.get());
        setCommandTensorArg(commandArgs, tagTensorB, descB, d_B.get());
        setCommandTensorArg(commandArgs, tagTensorC, descC, d_C.get());
        setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWavefrontCount({2u, 2u});
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

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

        CommandKernel commandKernel(command, "MatrixMultiplyABC");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<TypeD> gpu_D(M * N);
        ASSERT_THAT(hipMemcpy(gpu_D.data(), d_D.get(), M * N * sizeof(TypeD), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<float> tmp_D(M * N, 0.f);
        CPUMM(tmp_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

        std::vector<TypeD> cpu_D;
        cpu_D.reserve(M * N);
        for(size_t i = 0; i < M * N; i++)
            cpu_D.emplace_back(TypeD(tmp_D[i]));

        auto tol = gemmAcceptableError<TypeAB, TypeAB, TypeD>(
            M, N, K, m_context->targetArchitecture().target());
        auto res = compare(gpu_D, cpu_D, tol);

        Log::info("MatrixMultiplyABC and Conversion RNorm is {}", res.relativeNormL2);
        ASSERT_TRUE(res.ok) << res.message();
    }

    template <typename TypeAB, typename TypeD>
    void ConversionTest::matrixMultiply(
        int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError)
    {
        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 512;
        int N = 512;
        int K = 256;

        // output macro tile size
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<TypeAB>(M * K, -1.f, 1.f);
        auto B = random.vector<TypeAB>(K * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<TypeD>(M * N);

        auto       command    = std::make_shared<Command>();
        auto const dataTypeAB = TypeInfo<TypeAB>::Var.dataType;
        auto const dataTypeD  = TypeInfo<TypeD>::Var.dataType;

        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // A
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagAB
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

        auto cvtOp  = rocRoller::Operations::T_Execute(command->getNextTag()); // Convert(A * B)
        auto tagCvt = cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagAB, dataTypeD));
        command->addOperation(std::move(cvtOp));

        auto tagTensorD = command->addOperation(
            rocRoller::Operations::Tensor(2, dataTypeD)); // D = Convert(A * B)
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        TensorDescriptor descA(dataTypeAB, {size_t(M), size_t(K)}, "N");
        TensorDescriptor descB(dataTypeAB, {size_t(K), size_t(N)}, "N");
        TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

        setCommandTensorArg(commandArgs, tagTensorA, descA, d_A.get());
        setCommandTensorArg(commandArgs, tagTensorB, descB, d_B.get());
        setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWavefrontCount({2u, 2u});
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        CommandKernel commandKernel(command, "MatrixMultiply");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<TypeD> gpu_D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(gpu_D.data(), d_D.get(), M * N * sizeof(TypeD), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<TypeAB> tmp_D(M * N, 0.f);
        CPUMM(tmp_D, tmp_D, A, B, M, N, K, 1.0, 1.0, false, false);

        std::vector<TypeD> cpu_D;
        cpu_D.reserve(M * N);
        for(size_t i = 0; i < M * N; i++)
            cpu_D.emplace_back(TypeD(tmp_D[i]));

        auto tol = gemmAcceptableError<TypeAB, TypeAB, TypeD>(
            M, N, K, m_context->targetArchitecture().target());
        auto res = compare(gpu_D, cpu_D, tol);

        Log::info("D = Convert(A * B) RNorm is {}", res.relativeNormL2);
        ASSERT_TRUE(res.ok) << res.message();
    }

    template <typename DestType, typename SrcType>
    void ConversionTest::convertAdd(std::vector<SrcType>& a,
                                    std::vector<SrcType>& b,
                                    ConversionSettings&   cs,
                                    bool const            loadLDS_A)
    {
        static_assert(!std::is_same_v<DestType, SrcType>,
                      "Source and destination types for conversion must be different");

        auto command = std::make_shared<Command>();

        auto const srcDataType = TypeInfo<SrcType>::Var.dataType;
        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, srcDataType));
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));
        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, srcDataType));
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto const destDataType = TypeInfo<DestType>::Var.dataType;
        auto       execute      = rocRoller::Operations::T_Execute(command->getNextTag());
        auto       tagCvtA
            = execute.addXOp(rocRoller::Operations::E_Cvt(tagLoadA, destDataType)); // Convert A
        auto tagCvtB
            = execute.addXOp(rocRoller::Operations::E_Cvt(tagLoadB, destDataType)); // Convert B
        auto tagC = execute.addXOp(
            rocRoller::Operations::E_Add(tagCvtA, tagCvtB)); // C = converted(A) + converted(B)
        command->addOperation(std::move(execute));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, destDataType));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagC, tagTensorC));

        CommandArguments commandArgs = command->createArguments();

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<DestType>(a.size());

        TensorDescriptor descA(srcDataType, {size_t(cs.nx), size_t(cs.ny)}, "T");
        TensorDescriptor descB(srcDataType, {size_t(cs.nx), size_t(cs.ny)}, "T");
        TensorDescriptor descC(destDataType, {size_t(cs.nx), size_t(cs.ny)}, "T");

        setCommandTensorArg(commandArgs, tagTensorA, descA, d_a.get());
        setCommandTensorArg(commandArgs, tagTensorB, descB, d_b.get());
        setCommandTensorArg(commandArgs, tagTensorC, descC, d_c.get());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTileVGPR = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::VGPR, {cs.threadTileM, cs.threadTileN});
        auto macTileLDS = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::LDS, {cs.threadTileM, cs.threadTileN});

        unsigned int workgroup_size_x = cs.m / cs.threadTileM;
        unsigned int workgroup_size_y = cs.n / cs.threadTileN;
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        params->setDimensionInfo(tagLoadA, loadLDS_A ? macTileLDS : macTileVGPR);
        params->setDimensionInfo(tagLoadB, macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(tagC, macTileVGPR);

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(
            cs.nx / cs.threadTileM); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(
            cs.ny / cs.threadTileN); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        CommandKernel commandKernel(command, "convertAndAdd");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<DestType> gpuResult(a.size());
        ASSERT_THAT(
            hipMemcpy(
                gpuResult.data(), d_c.get(), gpuResult.size() * sizeof(DestType), hipMemcpyDefault),
            HasHipSuccess(0));

        // Reference result generated on CPU
        std::vector<DestType> cpuResult;
        cpuResult.reserve(gpuResult.size());
        for(size_t i = 0; i < a.size(); i++)
            cpuResult.emplace_back(DestType(a[i]) + DestType(b[i]));

        auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
        auto res = compare(gpuResult, cpuResult, tol);
        EXPECT_TRUE(res.ok) << res.message();
        Log::info("C = Convert(A) + Convert(B) RNorm is {}", res.relativeNormL2);
    }

    template <typename DestType, typename SrcType>
    void ConversionTest::convertTo(std::vector<SrcType>&         srcData,
                                   ConversionSettings const&     cs,
                                   bool const                    loadLDS_A,
                                   std::optional<uint32_t> const seed)
    {
        static_assert(!std::is_same_v<DestType, SrcType>,
                      "Source and destination types for conversion must be different");

        unsigned int workgroup_size_x = cs.m / cs.threadTileM;
        unsigned int workgroup_size_y = cs.n / cs.threadTileN;

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(
            cs.nx / cs.threadTileM); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(
            cs.ny / cs.threadTileN); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        auto command = std::make_shared<Command>();

        auto const srcDataType = TypeInfo<SrcType>::Var.dataType;
        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, srcDataType));
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto const               destDataType = TypeInfo<DestType>::Var.dataType;
        Operations::OperationTag tagSeed, tagLoadSeed;
        if(seed.has_value())
        {
            tagSeed     = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
            tagLoadSeed = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));
        }

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        // Convert A to destination type
        auto tagCvtA = seed.has_value()
                           ? execute.addXOp(rocRoller::Operations::E_StochasticRoundingCvt(
                               tagLoadA, tagLoadSeed, destDataType))
                           : execute.addXOp(rocRoller::Operations::E_Cvt(tagLoadA, destDataType));
        command->addOperation(std::move(execute));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, destDataType));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvtA, tagTensorC));

        CommandArguments commandArgs = command->createArguments();

        auto             d_a = make_shared_device(srcData);
        TensorDescriptor descA(srcDataType, {size_t(cs.nx), size_t(cs.ny)}, "T");
        setCommandTensorArg(commandArgs, tagTensorA, descA, d_a.get());

        auto             d_c = make_shared_device<DestType>(srcData.size());
        TensorDescriptor descC(destDataType, {size_t(cs.nx), size_t(cs.ny)}, "T");
        setCommandTensorArg(commandArgs, tagTensorC, descC, d_c.get());

        if(seed.has_value())
            commandArgs.setArgument(tagSeed, ArgumentType::Value, seed.value());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        auto macTileVGPR = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::VGPR, {cs.threadTileM, cs.threadTileN});
        auto macTileLDS = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::LDS, {cs.threadTileM, cs.threadTileN});

        params->setDimensionInfo(tagLoadA, loadLDS_A ? macTileLDS : macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(tagCvtA, macTileVGPR);

        CommandKernel commandKernel(command, "DirectConvert");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<DestType> gpuResult(srcData.size());
        ASSERT_THAT(
            hipMemcpy(
                gpuResult.data(), d_c.get(), gpuResult.size() * sizeof(DestType), hipMemcpyDefault),
            HasHipSuccess(0));

        // Reference result generated on CPU
        std::vector<DestType> cpuResult;
        cpuResult.reserve(gpuResult.size());
        for(auto v : srcData)
        {
            if(not seed.has_value())
            {
                cpuResult.emplace_back(DestType(v));
            }
            else
            {
                if constexpr(std::is_same_v<DestType, FP8> || std::is_same_v<DestType, BF8>)
                {
                    int constexpr exp_width      = std::is_same_v<DestType, FP8> ? 4 : 5;
                    int constexpr mantissa_width = 7 - exp_width;

                    cpuResult.emplace_back(0);

                    bool constexpr is_bf8 = std::is_same_v<DestType, BF8>;
                    auto const f8Mode     = Settings::getInstance()->get(Settings::F8ModeOption);

                    if(f8Mode == rocRoller::F8Mode::NaNoo)
                    {
                        cpuResult.back().data = DataTypes::cast_to_f8<mantissa_width,
                                                                      exp_width,
                                                                      float,
                                                                      false /* is_ocp */,
                                                                      is_bf8,
                                                                      true /*negative_zero_nan*/,
                                                                      true /*clip*/>(
                            v /* value to be converted   */,
                            true /* is stochastic rounding? */,
                            seed.value() /* seed for stochastic rounding */);
                    }
                    else
                    {
                        cpuResult.back().data = DataTypes::cast_to_f8<mantissa_width,
                                                                      exp_width,
                                                                      float,
                                                                      true /* is_ocp */,
                                                                      is_bf8,
                                                                      true /*negative_zero_nan*/,
                                                                      true /*clip*/>(
                            v /* value to be converted   */,
                            true /* is stochastic rounding? */,
                            seed.value() /* seed for stochastic rounding */);
                    }
                }
                else
                {
                    AssertFatal(true, "Destionation tyoe of SR conversion can only be FP8/BF8");
                }
            }
        }

        auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
        auto res = compare(gpuResult, cpuResult, tol);
        EXPECT_TRUE(res.ok) << res.message();
        Log::info("C = Convert(A) RNorm is {}", res.relativeNormL2);
    }

    TEST_F(ConversionTest, GPU_FloatToFP8_VGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::FP8>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToFP8_LDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::FP8>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToBF8_VGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::BF8>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToBF8_LDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::BF8>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToHalf_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::Half>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToHalf_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::Half>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_HalfToFloat_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<Half>();
        convertTo<float>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_HalfToFloat_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<Half>();
        convertTo<float>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_BF16ToFloat_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<BFloat16>();
        convertTo<float>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_FloatToBF16_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<BFloat16>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_AddFloatToHalf_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<float>(12345u);
        auto               b = cs.generateData<float>(56789u);
        // C (Half) = Convert( A(Float) ) + Convert( B(Float) )
        convertAdd<rocRoller::Half>(a, b, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_AddFloatToHalf_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<float>(12345u);
        auto               b = cs.generateData<float>(56789u);
        // C (Half) = Convert( A(Float) ) + Convert( B(Float) )
        convertAdd<rocRoller::Half>(a, b, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_AddHalfToFloat_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<Half>(12345u);
        auto               b = cs.generateData<Half>(56789u);
        // C (Float) = Convert( A(Half) ) + Convert( B(Half) )
        convertAdd<float>(a, b, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_AddHalfToFloat_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<Half>(12345u);
        auto               b = cs.generateData<Half>(56789u);
        // C (Float) = Convert( A(Half) ) + Convert( B(Half) )
        convertAdd<float>(a, b, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, GPU_MatrixMultiplyABC_F32_Half)
    {
        // D (Half) = Convert( A (F32) * B (F32) + C (F32) )
        matrixMultiplyABC<float, float, Half>(32, 32, 2, 1, 2.e-6);
    }

    TEST_F(ConversionTest, GPU_MatrixMultiplyABC_F32_FP8)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        // D (FP8) = Convert( A (FP8) * B (FP8) + C (F32) )
        matrixMultiplyABC<FP8, float, FP8>(16, 16, 32, 1, 2.e-6);
    }

    TEST_F(ConversionTest, GPU_MatrixMultiplyABC_F32_BF8)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        // D (BF8) = Convert( A (BF8) * B (BF8) + C (F32) )
        matrixMultiplyABC<BF8, float, BF8>(16, 16, 32, 1, 2.e-6);
    }

    TEST_F(ConversionTest, GPU_MatrixMultiply)
    {
        // D (Half) = Convert( A (F32) * B (F32) )
        matrixMultiply<float, Half>(32, 32, 2, 1, 2.e-6);
    }

    TEST_F(ConversionTest, GPU_SR_Float2FP8_VGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::FP8>(
            srcData, cs, false /* load A in LDS */, 12345u /* seed for SR conversion */);
    }

    TEST_F(ConversionTest, GPU_SR_Float2BF8_VGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::BF8>(
            srcData, cs, false /* load A in LDS */, 12345u /* seed for SR conversion */);
    }

    TEST_F(ConversionTest, GPU_SR_Float2FP8_LDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::FP8>(
            srcData, cs, true /* load A in LDS */, 12345u /* seed for SR conversion */);
    }

    TEST_F(ConversionTest, GPU_SR_Float2BF8_LDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::BF8>(
            srcData, cs, true /* load A in LDS */, 12345u /* seed for SR conversion */);
    }
}
