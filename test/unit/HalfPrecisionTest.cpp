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
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    struct HalfPrecisionProblem
    {
        std::shared_ptr<Command>            command;
        rocRoller::Operations::OperationTag resultTag, aTag, bTag;
    };

    class HalfPrecisionTest : public CurrentGPUContextFixture
    {
    };

    void genHalfPrecisionMultiplyAdd(rocRoller::ContextPtr m_context,
                                     HalfPrecisionProblem& prob,
                                     int                   N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        prob.command = std::make_shared<Command>();

        prob.resultTag  = prob.command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.resultTag, ArgumentType::Value));
        prob.aTag       = prob.command->allocateTag();
        auto a_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.aTag, ArgumentType::Value));
        prob.bTag       = prob.command->allocateTag();
        auto b_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.bTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto b_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield a_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            for(int i = 0; i < N / 2; i++)
            {
                co_yield m_context->mem()->loadGlobal(v_a, a_ptr, i * 4, 4);
                co_yield m_context->mem()->loadGlobal(v_b, b_ptr, i * 4, 4);

                co_yield generateOp<Expression::Multiply>(v_b, v_a, v_b);
                co_yield generateOp<Expression::Add>(v_a, v_a, v_b);

                co_yield m_context->mem()->storeGlobal(v_result, v_a, i * 4, 4);
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void genHalfPrecisionMultiplyAddConvert(rocRoller::ContextPtr m_context,
                                            HalfPrecisionProblem& prob,
                                            int                   N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        prob.command = std::make_shared<Command>();

        prob.resultTag  = prob.command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.resultTag, ArgumentType::Value));
        prob.aTag       = prob.command->allocateTag();
        auto a_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.aTag, ArgumentType::Value));
        prob.bTag       = prob.command->allocateTag();
        auto b_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.bTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto b_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto a    = v_a->expression();
            auto b    = v_b->expression();
            auto expr = Expression::convert<DataType::Half>(
                Expression::convert<DataType::Float>(a)
                + (Expression::convert<DataType::Float>(a)
                   * Expression::convert<DataType::Float>(b)));

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield a_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            for(int i = 0; i < N; i++)
            {
                co_yield m_context->mem()->loadGlobal(v_a, a_ptr, i * 2, 2);
                co_yield m_context->mem()->loadGlobal(v_b, b_ptr, i * 2, 2);

                co_yield Expression::generate(v_a, expr, m_context);

                co_yield m_context->mem()->storeGlobal(v_result, v_a, i * 2, 2);
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void
        executeHalfPrecisionMultiplyAdd(rocRoller::ContextPtr m_context, int N, bool convertToFloat)
    {
        HalfPrecisionProblem prob;
        if(convertToFloat)
            genHalfPrecisionMultiplyAddConvert(m_context, prob, N);
        else
            genHalfPrecisionMultiplyAdd(m_context, prob, N);

        RandomGenerator   random(314273u);
        auto              a = random.vector<Half>(N, -1.0, 1.0);
        auto              b = random.vector<Half>(N, -1.0, 1.0);
        std::vector<Half> result(N);

        auto d_a      = make_shared_device(a);
        auto d_b      = make_shared_device(b);
        auto d_result = make_shared_device<Half>(N);

        CommandArguments commandArgs = prob.command->createArguments();

        commandArgs.setArgument(prob.resultTag, ArgumentType::Value, d_result.get());
        commandArgs.setArgument(prob.aTag, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(prob.bTag, ArgumentType::Value, d_b.get());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(Half) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
        {
            ASSERT_NEAR(result[i], a[i] + a[i] * b[i], 0.001) << ShowValue(a[i]) << ShowValue(b[i]);
        }
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionMultiplyAdd)
    {
        executeHalfPrecisionMultiplyAdd(m_context, 8, false);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionMultiplyAddConvert)
    {
        executeHalfPrecisionMultiplyAdd(m_context, 8, true);
    }

    void genHalfPrecisionPack(rocRoller::ContextPtr m_context, HalfPrecisionProblem& prob, int N)
    {
        AssertFatal(N % 2 == 0, "HalfPrecisionPack tests should only operate on even sizes");

        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        prob.command = std::make_shared<Command>();

        prob.resultTag  = prob.command->allocateTag();
        auto result_exp = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.resultTag, ArgumentType::Value));
        prob.aTag       = prob.command->allocateTag();
        auto a_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.aTag, ArgumentType::Value));
        prob.bTag       = prob.command->allocateTag();
        auto b_exp      = std::make_shared<Expression::Expression>(prob.command->allocateArgument(
            {DataType::Half, PointerType::PointerGlobal}, prob.bTag, ArgumentType::Value));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});
        k->addArgument(
            {"b", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, b_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto a_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto b_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Halfx2, 1);

            auto v_hi = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto v_lo = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Half, 1);

            auto lds_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto mask = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto lds = Register::Value::AllocateLDS(m_context, DataType::Half, N * 2);

            co_yield v_a->allocate();
            co_yield a_ptr->allocate();
            co_yield b_ptr->allocate();
            co_yield v_result->allocate();
            co_yield v_hi->allocate();
            co_yield v_lo->allocate();
            co_yield lds_offset->allocate();
            co_yield mask->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(a_ptr, s_a, "Move pointer.");
            co_yield m_context->copier()->copy(b_ptr, s_b, "Move pointer.");

            co_yield m_context->copier()->copy(
                lds_offset, Register::Value::Literal(lds->getLDSAllocation()->offset()));
            co_yield m_context->copier()->copy(
                mask, Register::Value::Literal(0xFFFF), "Create register with mask for lower bits");

            for(int i = 0; i < N; i++)
            {
                // Load and pack from flat into registers
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::MemoryKind::Global,
                                                       v_a,
                                                       a_ptr,
                                                       Register::Value::Literal(i * 2),
                                                       b_ptr,
                                                       Register::Value::Literal(i * 2));

                // Perform addition. This will be a+a and b+b
                co_yield generateOp<Expression::Add>(v_a, v_a, v_a);

                // Store and Pack into LDS
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::LogicalShiftR>(
                    v_hi, v_a, Register::Value::Literal(16));
                co_yield m_context->mem()->packAndStore(MemoryInstructions::MemoryKind::Local,
                                                        lds_offset,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal(i * 4));
            }

            co_yield m_context->mem()->barrier();

            // Load all values of a+a from LDS, then store in flat
            for(int i = 0; i < N / 2; i++)
            {
                // Load and pack from LDS
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::MemoryKind::Local,
                                                       v_a,
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8),
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 4));

                // Store and pack into flat
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::LogicalShiftR>(
                    v_hi, v_a, Register::Value::Literal(16));

                co_yield m_context->mem()->packAndStore(MemoryInstructions::MemoryKind::Global,
                                                        v_result,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal(i * 4));
            }

            // Load all values of b+b from LDS, then store in flat
            for(int i = 0; i < N / 2; i++)
            {
                // Load and pack from LDS
                co_yield m_context->mem()->loadAndPack(MemoryInstructions::MemoryKind::Local,
                                                       v_a,
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 2),
                                                       lds_offset,
                                                       Register::Value::Literal(i * 8 + 6));

                // Store and pack into flat
                co_yield generateOp<Expression::BitwiseAnd>(v_lo, v_a, mask);
                co_yield generateOp<Expression::LogicalShiftR>(
                    v_hi, v_a, Register::Value::Literal(16));

                co_yield m_context->mem()->packAndStore(MemoryInstructions::MemoryKind::Global,
                                                        v_result,
                                                        v_lo,
                                                        v_hi,
                                                        Register::Value::Literal((N / 2 + i) * 4));
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeHalfPrecisionPack(rocRoller::ContextPtr m_context, int N)
    {
        HalfPrecisionProblem prob;
        genHalfPrecisionPack(m_context, prob, N);

        RandomGenerator   random(316473u);
        auto              a = random.vector<Half>(N, -1.0, 1.0);
        auto              b = random.vector<Half>(N, -1.0, 1.0);
        std::vector<Half> result(N * 2);

        auto d_a      = make_shared_device(a);
        auto d_b      = make_shared_device(b);
        auto d_result = make_shared_device<Half>(N * 2);

        CommandArguments commandArgs = prob.command->createArguments();

        commandArgs.setArgument(prob.resultTag, ArgumentType::Value, d_result.get());
        commandArgs.setArgument(prob.aTag, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(prob.bTag, ArgumentType::Value, d_b.get());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(result.data(), d_result.get(), sizeof(Half) * N * 2, hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_NEAR(result[i], a[i] + a[i], 0.001);

        for(int i = 0; i < N; i++)
            EXPECT_NEAR(result[i + N], b[i] + b[i], 0.001);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionPack)
    {
        executeHalfPrecisionPack(m_context, 8);
    }

    void AddTiles(size_t nx, // tensor size x
                  size_t ny, // tensor size y
                  int    m, // macro tile size x
                  int    n, // macro tile size y
                  int    t_m, // thread tile size x
                  int    t_n) // thread tile size y
    {
        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0, "Invalid Test Dimensions");

        unsigned int workgroup_size_x = m / t_m;
        unsigned int workgroup_size_y = n / t_n;

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
        auto            a = random.vector<Half>(nx * ny, -100.0, 100.0);
        auto            b = random.vector<Half>(nx * ny, -100.0, 100.0);
        auto            r = random.vector<Half>(nx * ny, -100.0, 100.0);
        auto            x = random.vector<Half>(nx * ny, -100.0, 100.0);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<Half>(nx * ny);

        auto command  = std::make_shared<Command>();
        auto dataType = DataType::Half;

        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto tag2A   = execute.addXOp(rocRoller::Operations::E_Add(tagLoadA, tagLoadA)); // A + A
        auto tag2B   = execute.addXOp(rocRoller::Operations::E_Add(tagLoadB, tagLoadB)); // B + B
        auto tagC    = execute.addXOp(rocRoller::Operations::E_Add(tag2A, tag2B)); // C = 2A + 2B
        command->addOperation(std::move(execute));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, dataType));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagC, tagTensorC));

        CommandArguments commandArgs = command->createArguments();

        TensorDescriptor descA(dataType, {size_t(nx), size_t(ny)}, "T");
        TensorDescriptor descB(dataType, {size_t(nx), size_t(ny)}, "T");
        TensorDescriptor descC(dataType, {size_t(nx), size_t(ny)}, "T");

        setCommandTensorArg(commandArgs, tagTensorA, descA, d_a.get());
        setCommandTensorArg(commandArgs, tagTensorB, descB, d_b.get());
        setCommandTensorArg(commandArgs, tagTensorC, descC, d_c.get());

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto macTileVGPR
            = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto macTileLDS
            = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});

        params->setDimensionInfo(tagLoadA, macTileLDS);
        params->setDimensionInfo(tagLoadB, macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(tagC, macTileVGPR);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        CommandKernel commandKernel(command, "HalfPrecisionAdd");
        commandKernel.setContext(Context::ForDefaultHipDevice("HalfPrecisionAdd"));
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(Half), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            for(size_t j = 0; j < ny; ++j)
            {
                x[(i * ny) + j]
                    = a[(i * ny) + j] + a[(i * ny) + j] + b[(i * ny) + j] + b[(i * ny) + j];
            }
        }

        auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
        auto res = compare(r, x, tol);
        EXPECT_TRUE(res.ok) << res.message();
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionAdd)
    {
        AddTiles(256, 512, 16, 8, 4, 4);
    }

    TEST_F(HalfPrecisionTest, GPU_HalfPrecisionAsserts)
    {

        auto vf32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 1);

        auto vh16_1
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 1);
        auto vh16_2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Half, 1);
        auto vh16x2
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Halfx2, 1);

        auto addr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int64, 1);

        auto buff_desc = std::make_shared<BufferDescriptor>(m_context);
        auto buff_opts = BufferInstructionOptions();

        // copy
        EXPECT_THROW(m_context->schedule(m_context->copier()->packHalf(vf32, vh16_1, vh16_2)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->copier()->packHalf(vh16x2, vf32, vf32)),
                     FatalError);

        // memory instructions
        EXPECT_THROW(m_context->schedule(m_context->mem()->loadGlobal(vh16x2, addr, 0, 4, true)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->mem()->loadLocal(vh16x2, addr, 0, 4, "", true)),
                     FatalError);

        EXPECT_THROW(m_context->schedule(m_context->mem()->loadAndPack(
                         MemoryInstructions::MemoryKind::Global, vf32, addr, addr, addr, addr, "")),
                     FatalError);

        EXPECT_THROW(m_context->schedule(
                         m_context->mem()->loadAndPack(MemoryInstructions::MemoryKind::Buffer,
                                                       vf32,
                                                       addr,
                                                       addr,
                                                       addr,
                                                       addr,
                                                       "",
                                                       buff_desc)),
                     FatalError);
    }

}
