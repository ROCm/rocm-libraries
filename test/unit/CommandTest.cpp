

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "SourceMatcher.hpp"

#include <rocRoller/Operations/Command.hpp>

using namespace rocRoller;

TEST(CommandTest, Basic)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    auto load_linear = std::make_shared<Operations::Operation>(Operations::T_Load_Linear());

    Operations::T_Store_Linear tsl;
    tsl.setTag(0);

    auto store_linear = std::make_shared<Operations::Operation>(std::move(tsl));
    auto execute      = std::make_shared<Operations::Operation>(Operations::T_Execute());

    command->addOperation(load_linear);
    command->addOperation(store_linear);
    command->addOperation(execute);

    EXPECT_EQ(load_linear, command->findTag(0));

    EXPECT_EQ(3, command->operations().size());
}

TEST(CommandTest, ToString)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    command->addOperation(std::make_shared<Operations::Operation>(Operations::T_Load_Linear()));
    command->addOperation(std::make_shared<Operations::Operation>(Operations::T_Execute()));

    EXPECT_EQ(2, command->operations().size());

    std::ostringstream msg;
    msg << *command;

    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_LOAD_LINEAR"));
    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_EXECUTE"));

    EXPECT_EQ(msg.str(), command->toString());
}

TEST(CommandTest, VectorAdd)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto load_A = command->allocateTag();
    auto load_B = command->allocateTag();
    auto res    = command->allocateTag();
    command->addOperation(Operations::T_Load_Linear(DataType::Float, 1, load_A));
    command->addOperation(Operations::T_Load_Linear(DataType::Float, 1, load_B));

    Operations::T_Execute execute;
    execute.addXOp(Operations::E_Add(res, load_A, load_B));

    command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

    command->addOperation(Operations::T_Store_Linear(1, res));

    std::string result = R"(
        T_LOAD_LINEAR.Float.d1 0, (base=&0, lim=&8, sizes={&16 }, strides={&24 })
        T_LOAD_LINEAR.Float.d1 1, (base=&32, lim=&40, sizes={&48 }, strides={&56 })
        T_EXECUTE 0 1
          E_Add 2, 0, 1
        T_STORE_LINEAR.d1 2, (base=&64, lim=&72, sizes={}, strides={&80 }))";
    EXPECT_EQ(NormalizedSource(command->toString()), NormalizedSource(result));

    {
        std::string        expected = R"([
            Load_Linear_0_pointer: PointerGlobal: Float(offset: 0, size: 8, read_only),
            Load_Linear_0_extent: Value: Int64(offset: 8, size: 8, read_only),
            Load_Linear_0_size_0: Value: Int64(offset: 16, size: 8, read_only),
            Load_Linear_0_stride_0: Value: Int64(offset: 24, size: 8, read_only),
            Load_Linear_1_pointer: PointerGlobal: Float(offset: 32, size: 8, read_only),
            Load_Linear_1_extent: Value: Int64(offset: 40, size: 8, read_only),
            Load_Linear_1_size_0: Value: Int64(offset: 48, size: 8, read_only),
            Load_Linear_1_stride_0: Value: Int64(offset: 56, size: 8, read_only),
            Store_Linear_2_pointer: PointerGlobal: Int32(offset: 64, size: 8, write_only),
            Store_Linear_2_extent: Value: Int64(offset: 72, size: 8, read_only),
            Store_Linear_2_stride_0: Value: Int64(offset: 80, size: 8, read_only)
        ])";
        std::ostringstream msg;
        msg << command->getArguments();
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(msg.str()));
    }
}

TEST(CommandTest, DuplicateOp)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto execute = std::make_shared<Operations::Operation>(Operations::T_Execute());

    command->addOperation(execute);
#ifdef NDEBUG
    GTEST_SKIP() << "Skipping assertion check in release mode.";
#endif

    EXPECT_THROW({ command->addOperation(execute); }, FatalError);
}

TEST(CommandTest, XopInputOutputs)
{
    Operations::T_Execute execute;

    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Sub(2, 0, 1)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(3, 2, 1)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<int>({0, 1}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<int>({2, 3}));

    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Abs(4, 3)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(5, 6)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<int>({0, 1, 6}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<int>({2, 3, 4, 5}));
}
