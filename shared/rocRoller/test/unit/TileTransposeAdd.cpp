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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace TileTransposeAddTest
{
    struct Transpose
    {
        bool a;
        bool b;
        bool c;
    };

    class GPU_TileTransposeAddTest
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

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Int32)); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Int32)); // B
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto tag2A   = execute.addXOp(rocRoller::Operations::E_Add(tagLoadA, tagLoadA)); // A + A
        auto tag2B   = execute.addXOp(rocRoller::Operations::E_Add(tagLoadB, tagLoadB)); // B + B
        auto tagC    = execute.addXOp(rocRoller::Operations::E_Add(tag2A, tag2B)); // C = 2A + 2B
        command->addOperation(std::move(execute));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Int32));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagC, tagTensorC));

        CommandArguments commandArgs = command->createArguments();

        TensorDescriptor descA(DataType::Int32,
                               {size_t(transpose.a ? ny : nx), size_t(transpose.a ? nx : ny)},
                               {(size_t)((ny * !transpose.a) + transpose.a),
                                (size_t)((nx * transpose.a) + !transpose.a)});
        TensorDescriptor descB(DataType::Int32,
                               {size_t(transpose.b ? ny : nx), size_t(transpose.b ? nx : ny)},
                               {(size_t)((ny * !transpose.b) + transpose.b),
                                (size_t)((nx * transpose.b) + !transpose.b)});
        TensorDescriptor descC(DataType::Int32,
                               {size_t(transpose.c ? ny : nx), size_t(transpose.c ? nx : ny)},
                               {(size_t)((ny * !transpose.c) + transpose.c),
                                (size_t)((nx * transpose.c) + !transpose.c)});

        setCommandTensorArg(commandArgs, tagTensorA, descA, d_a.get());
        setCommandTensorArg(commandArgs, tagTensorB, descB, d_b.get());
        setCommandTensorArg(commandArgs, tagTensorC, descC, d_c.get());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTile
            = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        params->setDimensionInfo(tagLoadA, macTile);
        params->setDimensionInfo(tagLoadB, macTile);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        auto launch = std::make_shared<CommandLaunchParameters>();
        launch->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileAdd");
        commandKernel.setContext(Context::ForDefaultHipDevice("TensorTileAdd"));
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.setLaunchParameters(launch);
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            for(size_t j = 0; j < ny; ++j)
            {
                auto idx = [i, j, nx, ny](bool t) { return t ? (j * nx + i) : (i * ny + j); };
                x[idx(transpose.c)] = a[idx(transpose.a)] + a[idx(transpose.a)]
                                      + b[idx(transpose.b)] + b[idx(transpose.b)];
            }
        }

        auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
        auto res = compare(r, x, tol);

        Log::info("RNorm is {}", res.relativeNormL2);
        ASSERT_TRUE(res.ok) << res.message();
    }

    TEST_P(GPU_TileTransposeAddTest, GPU_TileTransposeAddTest)
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

    std::vector<GPU_TileTransposeAddTest::ParamType>
        testableParams(::testing::internal::ParamGenerator<GPU_TileTransposeAddTest::ParamType>
                           inputParamGenerator)
    {
        std::vector<GPU_TileTransposeAddTest::ParamType> retval;
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
        GPU_TileTransposeAddTest,
        GPU_TileTransposeAddTest,
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
