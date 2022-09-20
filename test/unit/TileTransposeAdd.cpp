
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Float.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "DataTypes/DataTypes.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Scheduling/Observers/FileWritingObserver.hpp"
#include "SourceMatcher.hpp"
#include "Utilities/Error.hpp"

using namespace rocRoller;

namespace TileTransposeAddTest
{
    struct Transpose
    {
        bool a;
        bool b;
        bool c;
    };

    class TileTransposeAddTestGPU
        : public CurrentGPUContextFixture,
          public ::testing::WithParamInterface<
              std::tuple<bool, bool, bool, size_t, size_t, int, int, int, int>>
    {
    };

    void TileTransposeAdd(Transpose transpose,
                          size_t    nx, // tensor size x
                          size_t    ny, // tensor size y
                          int       m, // macro tile size x
                          int       n, // macro tile size y
                          int       t_m, // thread tile size x
                          int       t_n) // thread tile size y
    {
        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0, "Invalid Test Dimensions");

        unsigned int workgroup_size_x = m / t_m;
        unsigned int workgroup_size_y = n / t_n;

        AssertFatal(nx == ny || (transpose.a == transpose.b && transpose.a == transpose.c),
                    "Invalid Test Dimensions");

        AssertFatal((size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // TODO: Handle when thread tiles include out of range indices
        AssertFatal(nx % t_m == 0, "Thread tile size must divide tensor size");
        AssertFatal(ny % t_n == 0, "Thread tile size must divide tensor size");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator random(129674u + nx + ny + m + n + t_m
                               + t_n); //Use different seeds for the different sizes.
        auto            a = random.vector<int>(nx * ny, -100, 100);
        auto            b = random.vector<int>(nx * ny, -100, 100);
        auto            r = random.vector<int>(nx * ny, -100, 100);
        auto            x = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 1))); // b

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(2, 0, 0))); // a + a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 1, 1))); // b + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // 2a + 2b

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Int32, 2, 4))); // c

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)((ny * !transpose.a) + transpose.a));
        runtimeArgs.append("d_a_stride_1", (size_t)((nx * transpose.a) + !transpose.a));

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_size_0", (size_t)nx);
        runtimeArgs.append("d_b_size_1", (size_t)ny);
        runtimeArgs.append("d_b_stride_0", (size_t)((ny * !transpose.b) + transpose.b));
        runtimeArgs.append("d_b_stride_1", (size_t)((nx * transpose.b) + !transpose.b));

        runtimeArgs.append("user2", d_c.get());
        runtimeArgs.append("d_c_limit", (size_t)nx * ny);
        runtimeArgs.append("d_c_stride_0", (size_t)((ny * !transpose.c) + transpose.c));
        runtimeArgs.append("d_c_stride_1", (size_t)((nx * transpose.c) + !transpose.c));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile_0
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_1
            = KernelGraph::CoordinateTransform::MacroTile(1, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_3
            = KernelGraph::CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_4 = KernelGraph::CoordinateTransform::MacroTile(
            4, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);
        params->setDimensionInfo(mac_tile_3);
        params->setDimensionInfo(mac_tile_4);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileAdd", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            for(size_t j = 0; j < ny; ++j)
            {
                auto idx = [i, j, nx, ny](bool t) { return t ? (j * nx + i) : (i * ny) + j; };
                x[idx(transpose.c)] = a[idx(transpose.a)] + a[idx(transpose.a)]
                                      + b[idx(transpose.b)] + b[idx(transpose.b)];
            }
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_P(TileTransposeAddTestGPU, TileTransposeAddTest_GPU)
    {
        Transpose transpose
            = {std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam())};

        auto nx  = std::get<3>(GetParam());
        auto ny  = std::get<4>(GetParam());
        auto m   = std::get<5>(GetParam());
        auto n   = std::get<6>(GetParam());
        auto t_m = std::get<7>(GetParam());
        auto t_n = std::get<8>(GetParam());

        TileTransposeAdd(transpose, nx, ny, m, n, t_m, t_n);
    }

    std::vector<TileTransposeAddTestGPU::ParamType> testableParams(
        ::testing::internal::ParamGenerator<TileTransposeAddTestGPU::ParamType> inputParamGenerator)
    {
        std::vector<TileTransposeAddTestGPU::ParamType> retval;
        for(auto const& param : inputParamGenerator)
        {
            Transpose transpose = {std::get<0>(param), std::get<1>(param), std::get<2>(param)};

            auto nx = std::get<3>(param);
            auto ny = std::get<4>(param);

            if(nx == ny || (transpose.a == transpose.b && transpose.a == transpose.c))
                retval.push_back(param);
        }
        return retval;
    }

    INSTANTIATE_TEST_SUITE_P(
        TileTransposeAddTestGPU,
        TileTransposeAddTestGPU,
        testing::ValuesIn(testableParams(testing::Combine(testing::Bool(),
                                                          testing::Bool(),
                                                          testing::Bool(),
                                                          testing::Values(256, 260, 512),
                                                          testing::Values(256, 1000),
                                                          testing::Values(16, 8),
                                                          testing::Values(8),
                                                          testing::Values(4),
                                                          testing::Values(4, 2)))));
}
