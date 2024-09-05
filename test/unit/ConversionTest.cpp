#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Utilities.hpp"

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
        template <typename TypeABC, typename TypeD>
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
        void convertTo(std::vector<SrcType>&     srcData,
                       ConversionSettings const& cs,
                       bool const                loadLDS_A);
    };

    template <typename TypeABC, typename TypeD>
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

        auto A = random.vector<TypeABC>(M * K, -1.f, 1.f);
        auto B = random.vector<TypeABC>(K * N, -1.f, 1.f);
        auto C = random.vector<TypeABC>(M * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_C = make_shared_device(C);
        auto d_D = make_shared_device<TypeD>(M * N);

        auto       command     = std::make_shared<Command>();
        auto const dataTypeABC = TypeInfo<TypeABC>::Var.dataType;
        auto const dataTypeD   = TypeInfo<TypeD>::Var.dataType;

        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeABC)); // A
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeABC)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeABC)); // C
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

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);
        // tiled?
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_B.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);

        commandArgs.setArgument(tagTensorC, ArgumentType::Value, d_C.get());
        commandArgs.setArgument(tagTensorC, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 1, (size_t)M);

        commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
        commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

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

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setManualWavefrontCount({2u, 2u});

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "MatrixMultiplyABC", params, postParams);
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<TypeD> gpu_D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(gpu_D.data(), d_D.get(), M * N * sizeof(TypeD), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<TypeABC> tmp_D(M * N, 0.f);
        CPUMM(tmp_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

        std::vector<TypeD> cpu_D;
        cpu_D.reserve(M * N);
        for(size_t i = 0; i < M * N; i++)
            cpu_D.emplace_back(TypeD(tmp_D[i]));

        auto tol = gemmAcceptableError<TypeABC, TypeABC, TypeD>(
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

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);
        // tiled?
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_B.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);

        commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
        commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setManualWavefrontCount({2u, 2u});

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "MatrixMultiply", params, postParams);
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
        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(tagTensorC, ArgumentType::Value, d_c.get());

        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, a.size());
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)cs.nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)1);

        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, b.size());
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)cs.nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)1);

        commandArgs.setArgument(tagTensorC, ArgumentType::Limit, a.size());
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 0, (size_t)cs.nx);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 1, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 0, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 1, (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTileVGPR = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::VGPR, {cs.threadTileM, cs.threadTileN});
        auto macTileLDS = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::LDS, {cs.threadTileM, cs.threadTileN});

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(
            cs.nx / cs.threadTileM); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(
            cs.ny / cs.threadTileN); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z
        params->setManualWorkitemCount({NX, NY, NZ});

        unsigned int workgroup_size_x = cs.m / cs.threadTileM;
        unsigned int workgroup_size_y = cs.n / cs.threadTileN;
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        params->setDimensionInfo(tagLoadA, loadLDS_A ? macTileLDS : macTileVGPR);
        params->setDimensionInfo(tagLoadB, macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(tagC, macTileVGPR);

        CommandKernel commandKernel(command, "convertAndAdd", params);
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
    void ConversionTest::convertTo(std::vector<SrcType>&     srcData,
                                   ConversionSettings const& cs,
                                   bool const                loadLDS_A)
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

        auto const destDataType = TypeInfo<DestType>::Var.dataType;
        auto       execute      = rocRoller::Operations::T_Execute(command->getNextTag());
        auto       tagCvtA      = execute.addXOp(
            rocRoller::Operations::E_Cvt(tagLoadA, destDataType)); // Convert A to destination type
        command->addOperation(std::move(execute));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, destDataType));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvtA, tagTensorC));

        CommandArguments commandArgs = command->createArguments();

        auto d_a = make_shared_device(srcData);
        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)cs.nx * cs.ny);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)cs.nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)1);

        auto d_c = make_shared_device<DestType>(srcData.size());
        commandArgs.setArgument(tagTensorC, ArgumentType::Value, d_c.get());
        commandArgs.setArgument(tagTensorC, ArgumentType::Limit, (size_t)cs.nx * cs.ny);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 0, (size_t)cs.nx);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 1, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 0, (size_t)cs.ny);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 1, (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTileVGPR = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::VGPR, {cs.threadTileM, cs.threadTileN});
        auto macTileLDS = KernelGraph::CoordinateGraph::MacroTile(
            {cs.m, cs.n}, MemoryType::LDS, {cs.threadTileM, cs.threadTileN});

        params->setDimensionInfo(tagLoadA, loadLDS_A ? macTileLDS : macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(tagCvtA, macTileVGPR);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "DirectConvert", params);
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
            cpuResult.emplace_back(DestType(v));

        auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
        auto res = compare(gpuResult, cpuResult, tol);
        EXPECT_TRUE(res.ok) << res.message();
        Log::info("C = Convert(A) RNorm is {}", res.relativeNormL2);
    }

    TEST_F(ConversionTest, Float2Half_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::Half>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, Float2Half_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<float>();
        convertTo<rocRoller::Half>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, Half2Float_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<Half>();
        convertTo<float>(srcData, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, Half2Float_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               srcData = cs.generateData<Half>();
        convertTo<float>(srcData, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, AddFloat2Half_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<float>(12345u);
        auto               b = cs.generateData<float>(56789u);
        convertAdd<rocRoller::Half>(a, b, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, AddFloat2Half_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<float>(12345u);
        auto               b = cs.generateData<float>(56789u);
        convertAdd<rocRoller::Half>(a, b, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, AddHalf2Float_LDS)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<Half>(12345u);
        auto               b = cs.generateData<Half>(56789u);
        convertAdd<float>(a, b, cs, true /* load A in LDS */);
    }

    TEST_F(ConversionTest, AddHalf2Float_VGPR)
    {
        ConversionSettings cs(256, 512, 16, 8, 4, 4);
        auto               a = cs.generateData<Half>(12345u);
        auto               b = cs.generateData<Half>(56789u);
        convertAdd<float>(a, b, cs, false /* load A in LDS */);
    }

    TEST_F(ConversionTest, MatrixMultiplyABC)
    {
        // Matrix A, B and C are float and matrix D is Half.
        // Result of (A * B) + C is float (mfma) and then we
        // convert it to Half
        matrixMultiplyABC<float, Half>(32, 32, 2, 1, 2.e-6);
    }

    TEST_F(ConversionTest, MatrixMultiply)
    {
        // Matrix A and B are float and matrix D is Half.
        // Result of (A * B) is float (mfma) and then we
        // convert it to Half
        matrixMultiply<float, Half>(32, 32, 2, 1, 2.e-6);
    }

}
