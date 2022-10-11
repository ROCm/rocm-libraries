#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class DependencyTest : public CurrentGPUContextFixture,
                           public ::testing::WithParamInterface<Scheduling::SchedulerProcedure>
    {
    };

    /**
     * double_and_check will double the value and check if its below a certain threshold.
     * First call is to double up to 1000 starting with 6. The second call is up to 100, so
     * thos double_and_check should return immediately since we are already above 100.
     * Interleaving these two will have the double_and_check(100) vcc be set to true when
     * double_and_check(1000) checks for looping and will quit early. Also interleave comments
     * to see how they affect scheduling.
     **/
    TEST_P(DependencyTest, ForLoopsWithVCC)
    {
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(floatPtr);
        auto val_arg  = command->allocateArgument(floatVal);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_arg2  = command->allocateArgument(floatPtr);
        auto val_arg2  = command->allocateArgument(floatVal);
        auto size_arg2 = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto ptr_exp2  = std::make_shared<Expression::Expression>(ptr_arg2);
        auto val_exp2  = std::make_shared<Expression::Expression>(val_arg2);
        auto size_exp2 = std::make_shared<Expression::Expression>(size_arg2);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr1",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val1", {DataType::Float}, DataDirection::ReadOnly, val_exp});

        k->addArgument({"ptr2",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp2});
        k->addArgument({"val2", {DataType::Float}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        std::vector<Generator<Instruction>> sequences;

        auto double_and_check = [&](float check_val, int n) -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr" + std::to_string(n), s_ptr);
            co_yield m_context->argLoader()->getValue("val" + std::to_string(n), s_value);

            auto v_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);
            auto v_value = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto v_target = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_ptr->allocate();
            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");
            co_yield v_value->allocate();
            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            // v_target is val we check against
            co_yield m_context->copier()->copy(
                v_target, Register::Value::Literal(check_val), "Move value");

            auto loop_start = Register::Value::Label("main_loop_" + std::to_string(n));
            co_yield Instruction::Label(loop_start);

            Register::ValuePtr s_condition;
            s_condition = m_context->getVCC();

            // Double the input value.
            co_yield Expression::generate(
                v_value, v_value->expression() + v_value->expression(), m_context);
            // Compare against the stop value.
            co_yield Expression::generate(
                s_condition, v_value->expression() < v_target->expression(), m_context);

            co_yield m_context->brancher()->branchIfNonZero(
                loop_start, s_condition, "// Conditionally branching to the label register.");

            co_yield m_context->mem()->storeFlat(v_ptr, v_value, "", 4);
        };

        auto comment_generator = [&](int max_comments) -> Generator<Instruction> {
            int i = 0;
            while(i < max_comments)
            {
                co_yield Instruction::Comment("Comment " + std::to_string(i++));
            }
        };

        sequences.push_back(double_and_check(1000.f, 2));
        sequences.push_back(double_and_check(100.f, 1));
        sequences.push_back(comment_generator(5));

        std::shared_ptr<Scheduling::Scheduler> scheduler
            = Component::Get<Scheduling::Scheduler>(GetParam(), m_context);

        m_context->schedule((*scheduler)(sequences));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        auto         ptr  = make_shared_device<float>();
        float        val  = 6.0f;
        unsigned int size = 1;

        auto         ptr2  = make_shared_device<float>();
        float        val2  = 3.0f;
        unsigned int size2 = 1;

        ASSERT_THAT(hipMemset(ptr.get(), 0, sizeof(float)), HasHipSuccess(0));
        ASSERT_THAT(hipMemset(ptr2.get(), 0, sizeof(float)), HasHipSuccess(0));

        KernelArguments runtimeArgs;

        runtimeArgs.append("ptr1", ptr.get());
        runtimeArgs.append("val1", val);
        runtimeArgs.append("size1", size);

        runtimeArgs.append("ptr2", ptr2.get());
        runtimeArgs.append("val2", val2);
        runtimeArgs.append("size2", size2);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        float resultValue1 = 0.0f;
        float resultValue2 = 0.0f;
        ASSERT_THAT(hipMemcpy(&resultValue1, ptr.get(), sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(&resultValue2, ptr2.get(), sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        if(GetParam() == Scheduling::SchedulerProcedure::Sequential)
        {
            EXPECT_GT(resultValue1, 100.0f);
            EXPECT_GT(resultValue2, 1000.0f);
        }
        else
        {
            EXPECT_GT(resultValue1, 100.0f);
            EXPECT_LT(resultValue2, 1000.0f);
        }
    }

    /**
     * scalar_overflow lambda will set scc to true with a scalar addition overflow.
     * scalar_compare will set scc to false with a comparison of two values.
     * scalar_overflow will check if we overflow and set the result accordingly.
     * However, scalar_compare will overwrite scc before the branch checks scc.
     **/
    TEST_P(DependencyTest, SCC)
    {
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType intPtr{DataType::Int32, PointerType::PointerGlobal};
        VariableType intVal{DataType::Int32, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(intPtr);
        auto val_arg  = command->allocateArgument(intVal);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Int32}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        std::vector<Generator<Instruction>> sequences;

        auto r_value
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        auto r_ptr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Raw32, 2);

        auto scalar_overflow = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            co_yield r_ptr->allocate();
            co_yield m_context->copier()->copy(r_ptr, s_ptr, "Move pointer");
            co_yield r_value->allocate();
            co_yield m_context->copier()->copy(r_value, s_value, "Move value");

            auto s_res = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);
            auto s_lhs = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);
            auto s_rhs = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);

            co_yield m_context->copier()->fill(s_lhs, Register::Value::Literal(0x7FEFFFFF));
            co_yield m_context->copier()->fill(s_rhs, Register::Value::Literal(0x6DCBFFFF));
            co_yield generateOp<Expression::Add>(s_res, s_lhs, s_rhs);

            auto scc_zero = Register::Value::Label("scc_zero");
            auto end      = Register::Value::Label("end");

            co_yield m_context->brancher()->branchIfNonZero(scc_zero, m_context->getSCC());
            co_yield m_context->copier()->copy(r_value, Register::Value::Literal(1729));
            co_yield m_context->brancher()->branch(end);
            co_yield_(Instruction::Label(scc_zero));
            co_yield m_context->copier()->copy(r_value, Register::Value::Literal(3014));
            co_yield_(Instruction::Label(end));
            co_yield m_context->mem()->storeFlat(r_ptr, r_value, "", 4);
        };

        auto scalar_compare = [&]() -> Generator<Instruction> {
            auto s_small = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);
            auto s_big = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);
            auto s_condition = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Int32, 1);
            co_yield m_context->copier()->fill(s_small, Register::Value::Literal(2));
            co_yield m_context->copier()->fill(s_big, Register::Value::Literal(5));
            co_yield generateOp<Expression::Add>(s_small, s_small, Register::Value::Literal(1));
            co_yield generateOp<Expression::Add>(s_big, s_big, Register::Value::Literal(2));
            co_yield generateOp<Expression::Multiply>(s_small, s_small, s_small);
            co_yield generateOp<Expression::Multiply>(s_big, s_big, s_big);
            co_yield Expression::generate(
                s_condition, s_small->expression() > s_big->expression(), m_context);
        };

        sequences.push_back(scalar_overflow());
        sequences.push_back(scalar_compare());

        std::shared_ptr<Scheduling::Scheduler> scheduler
            = Component::Get<Scheduling::Scheduler>(GetParam(), m_context);

        m_context->schedule((*scheduler)(sequences));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        auto         h_ptr  = make_shared_device<int>();
        int          h_val  = 11;
        unsigned int h_size = 1;

        ASSERT_THAT(hipMemset(h_ptr.get(), 0, sizeof(int)), HasHipSuccess(0));

        KernelArguments runtimeArgs;
        runtimeArgs.append("ptr", h_ptr.get());
        runtimeArgs.append("val", h_val);
        runtimeArgs.append("size", h_size);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        int resultValue = 0;
        ASSERT_THAT(hipMemcpy(&resultValue, h_ptr.get(), sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        if(GetParam() == Scheduling::SchedulerProcedure::Sequential)
        {
            EXPECT_EQ(resultValue, 3014);
        }
        else
        {
            EXPECT_EQ(resultValue, 1729);
        }
    }

    /**
     * VCC test that interleaves two streams that both set vcc to different values.
     * Output of the kernel is dependent on vcc value.
     **/
    TEST_P(DependencyTest, VCC)
    {
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType intPtr{DataType::Int32, PointerType::PointerGlobal};
        VariableType intVal{DataType::Int32, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(intPtr);
        auto val_arg  = command->allocateArgument(intVal);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Int32}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        std::vector<Generator<Instruction>> sequences;

        auto v_value
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_ptr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Raw32, 2);

        auto set_vcc = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_lhs = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            auto v_rhs = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            auto v_res = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_ptr->allocate();
            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");
            co_yield v_value->allocate();
            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            co_yield m_context->copier()->fill(v_lhs, Register::Value::Literal(2));
            co_yield m_context->copier()->fill(v_rhs, Register::Value::Literal(4));

            Register::ValuePtr vcc;
            vcc = m_context->getVCC();

            co_yield Expression::generate(
                vcc, v_rhs->expression() > v_lhs->expression(), m_context);

            co_yield Expression::generate(
                v_res, v_lhs->expression() - v_rhs->expression(), m_context);
            co_yield Expression::generate(
                v_res, v_rhs->expression() * v_rhs->expression(), m_context);
            co_yield Expression::generate(
                v_res, v_res->expression() + v_lhs->expression(), m_context);
            co_yield Expression::generate(
                v_res, v_res->expression() * v_res->expression(), m_context);

            auto label = Register::Value::Label("label");
            auto end   = Register::Value::Label("end");

            co_yield m_context->brancher()->branchIfNonZero(label, vcc);
            co_yield m_context->copier()->copy(v_value, Register::Value::Literal(20));
            co_yield m_context->brancher()->branch(end);
            co_yield Instruction::Label(label);
            co_yield m_context->copier()->copy(v_value, Register::Value::Literal(10));
            co_yield Instruction::Label(end);

            co_yield m_context->mem()->storeFlat(v_ptr, v_value, "", 4);
        };

        auto set_vcc2 = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_lhs2 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            auto v_rhs2 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            auto v_res2 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield m_context->copier()->fill(v_lhs2, Register::Value::Literal(2));
            co_yield m_context->copier()->fill(v_rhs2, Register::Value::Literal(8));

            Register::ValuePtr vcc;
            vcc = m_context->getVCC();

            co_yield Expression::generate(
                v_res2, v_rhs2->expression() + v_lhs2->expression(), m_context);
            co_yield Expression::generate(
                v_res2, v_res2->expression() * v_lhs2->expression(), m_context);
            co_yield Expression::generate(
                v_res2, v_res2->expression() * v_res2->expression(), m_context);

            co_yield Expression::generate(
                vcc, v_lhs2->expression() == v_rhs2->expression(), m_context);
        };

        sequences.push_back(set_vcc());
        sequences.push_back(set_vcc2());

        std::shared_ptr<Scheduling::Scheduler> scheduler
            = Component::Get<Scheduling::Scheduler>(GetParam(), m_context);

        m_context->schedule((*scheduler)(sequences));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        auto         h_ptr  = make_shared_device<int>();
        int          h_val  = 11;
        unsigned int h_size = 1;

        ASSERT_THAT(hipMemset(h_ptr.get(), 0, sizeof(int)), HasHipSuccess(0));

        KernelArguments runtimeArgs;
        runtimeArgs.append("ptr", h_ptr.get());
        runtimeArgs.append("val", h_val);
        runtimeArgs.append("size", h_size);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        int resultValue = 0;
        ASSERT_THAT(hipMemcpy(&resultValue, h_ptr.get(), sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        if(GetParam() == Scheduling::SchedulerProcedure::Sequential)
        {
            EXPECT_EQ(resultValue, 10);
        }
        else
        {
            EXPECT_EQ(resultValue, 20);
        }
    }

    /**
     * loop computes input^(power-1). double_branch contains an if_loop than
     * can be evaluated to be false always since we are checking against a
     * const value. This is used to see how we treat conditonal branches that 
     * can be treated as branches or always fall through. The double_branch
     * alters a value within the loop while the iterations have not ended if
     * interleaved together. math is used to interleave instructions that have
     * no dependencies.
     **/
    TEST_P(DependencyTest, Branching)
    {
        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(floatPtr);
        auto val_arg  = command->allocateArgument(floatVal);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto onef = std::make_shared<Expression::Expression>(1.f);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Float}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        std::vector<Generator<Instruction>> sequences;

        auto v_value
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 1);
        auto v_ptr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Raw32, 2);

        auto loop = [&](float power) -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_target = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_ptr->allocate();
            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");
            co_yield v_value->allocate();
            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            // v_target is val we check against
            co_yield m_context->copier()->copy(
                v_target, Register::Value::Literal(power), "Move value");

            auto loop_start = Register::Value::Label("loop");
            co_yield Instruction::Label(loop_start);

            Register::ValuePtr s_condition;
            s_condition = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::Raw32, 2);

            // Double the input value.
            co_yield Expression::generate(
                v_value, v_value->expression() * v_value->expression(), m_context);

            co_yield generateOp<Expression::Subtract>(
                v_target, v_target, Register::Value::Literal(1.f));

            // Compare against the stop value.
            co_yield Expression::generate(s_condition, v_target->expression() == onef, m_context);

            co_yield m_context->brancher()->branchIfZero(
                loop_start, s_condition, "//Conditionally branching to the label register.");
        };

        auto double_branch = [&]() -> Generator<Instruction> {
            auto wavefront_size = k->wavefront_size();

            auto s0 = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::UInt32, wavefront_size / 32);
            co_yield s0->allocate();

            co_yield m_context->copier()->copy(s0, Register::Value::Literal(0xFFFF), "Move value");

            auto if_label = Register::Value::Label("if_label");

            co_yield m_context->brancher()->branchIfZero(if_label, s0);

            auto v_temp = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            co_yield m_context->copier()->copy(v_temp, Register::Value::Literal(32.f), "");

            co_yield generateOp<Expression::Subtract>(v_value, v_value, v_temp);

            co_yield_(Instruction::Label(if_label));

            co_yield Expression::generate(
                v_value, v_value->expression() + v_value->expression(), m_context);

            co_yield m_context->copier()->copy(s0, Register::Value::Literal(0), "Zero out value");

            co_yield m_context->mem()->storeFlat(v_ptr, v_value, "", 4);
        };

        auto math = [&]() -> Generator<Instruction> {
            auto v_lhs = std::make_shared<Register::Value>(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            co_yield m_context->copier()->fill(v_lhs, Register::Value::Literal(1));
            auto v_rhs = std::make_shared<Register::Value>(
                m_context, Register::Type::Vector, DataType::Int32, 1);
            co_yield m_context->copier()->fill(v_rhs, Register::Value::Literal(2));

            co_yield Expression::generate(
                v_lhs, v_lhs->expression() + v_rhs->expression(), m_context);
            co_yield Expression::generate(
                v_lhs, v_lhs->expression() * v_rhs->expression(), m_context);
        };

        sequences.push_back(loop(3.f)); // res = (4 * 4) * (4 * 4) = 256
        sequences.push_back(math()); // res untouched in this block
        sequences.push_back(double_branch()); // res = 256 - 32; res = res + res  = 448

        std::shared_ptr<Scheduling::Scheduler> scheduler
            = Component::Get<Scheduling::Scheduler>(GetParam(), m_context);

        m_context->schedule((*scheduler)(sequences));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        auto         ptr  = make_shared_device<float>();
        float        val  = 4.0f;
        unsigned int size = 1;

        ASSERT_THAT(hipMemset(ptr.get(), 0, sizeof(float)), HasHipSuccess(0));

        KernelArguments runtimeArgs;
        runtimeArgs.append("ptr", ptr.get());
        runtimeArgs.append("val", val);
        runtimeArgs.append("size", size);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        float resultValue = 0.0f;
        ASSERT_THAT(hipMemcpy(&resultValue, ptr.get(), sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        if(GetParam() == Scheduling::SchedulerProcedure::Sequential)
        {
            EXPECT_EQ(resultValue, 448.0f);
        }
        else
        {
            EXPECT_NE(resultValue, 448.0f);
        }
    }

    INSTANTIATE_TEST_SUITE_P(DependencyTest,
                             DependencyTest,
                             ::testing::ValuesIn({Scheduling::SchedulerProcedure::Sequential,
                                                  Scheduling::SchedulerProcedure::RoundRobin}));
}