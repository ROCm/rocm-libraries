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

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

using namespace rocRoller;

class CommandTest : public SimpleFixture
{
};

TEST_F(CommandTest, Basic)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    auto tagTensor = command->addOperation(Operations::Tensor(1, DataType::Int32));

    auto load_linear
        = std::make_shared<Operations::Operation>(Operations::T_Load_Linear(tagTensor));
    auto tagLoad = command->addOperation(load_linear);

    Operations::T_Store_Linear tsl(tagLoad, tagTensor);
    // tag value is assigned to an operation when it's added to the Command Graph
    EXPECT_EQ(-1, static_cast<int32_t>(tsl.getTag()));
    auto store_linear = std::make_shared<Operations::Operation>(std::move(tsl));
    auto execute
        = std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag()));

    command->addOperation(store_linear);
    command->addOperation(execute);

    EXPECT_EQ(load_linear, command->findTag(Operations::OperationTag(1)));

    EXPECT_EQ(4, command->operations().size());
}

TEST_F(CommandTest, ToString)
{
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    command->addOperation(std::make_shared<Operations::Operation>(
        Operations::T_Load_Linear(Operations::OperationTag())));
    command->addOperation(
        std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag())));

    EXPECT_EQ(2, command->operations().size());

    std::ostringstream msg;
    msg << *command;

    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_LOAD_LINEAR"));
    EXPECT_THAT(msg.str(), ::testing::HasSubstr("T_EXECUTE"));

    EXPECT_EQ(msg.str(), command->toString());
}

TEST_F(CommandTest, ConvertOp)
{
    // A command to convert a Float tensor into Half type
    auto command = std::make_shared<rocRoller::Command>();

    EXPECT_EQ(0, command->operations().size());

    auto tagTensor = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto load_linear
        = std::make_shared<Operations::Operation>(Operations::T_Load_Linear(tagTensor));
    auto tagLoad = command->addOperation(load_linear);

    auto cvtOp = std::make_shared<Operations::XOp>(Operations::E_Cvt(tagLoad, DataType::Half));
    Operations::T_Execute execute(command->getNextTag());
    auto                  tagConvert = execute.addXOp(cvtOp);
    command->addOperation(execute);

    EXPECT_EQ(execute.getInputs(), std::unordered_set<Operations::OperationTag>({tagLoad}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<Operations::OperationTag>({tagConvert}));

    constexpr auto result = R"(
        Tensor.Float.d1 0, (base=&0, lim=&8, sizes={&16 }, strides={&24 })
        T_LOAD_LINEAR 1 Source 0
        T_EXECUTE 1
        E_Cvt 2, 1
        )";

    EXPECT_EQ(NormalizedSource(command->toString()), NormalizedSource(result));
}

TEST_F(CommandTest, VectorAdd)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto tagTensorA = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto tagLoadA   = command->addOperation(Operations::T_Load_Linear(tagTensorA));

    auto tagTensorB = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto tagLoadB   = command->addOperation(Operations::T_Load_Linear(tagTensorB));

    Operations::T_Execute execute(command->getNextTag());
    auto                  tagResult = execute.addXOp(Operations::E_Add(tagLoadA, tagLoadB));
    command->addOperation(std::move(execute));

    auto tagTensorResult = command->addOperation(Operations::Tensor(1, DataType::Float));
    command->addOperation(Operations::T_Store_Linear(tagResult, tagTensorResult));

    std::string result = R"(
        Tensor.Float.d1 0, (base=&0, lim=&8, sizes={&16 }, strides={&24 })
        T_LOAD_LINEAR 1 Source 0
        Tensor.Float.d1 2, (base=&32, lim=&40, sizes={&48 }, strides={&56 })
        T_LOAD_LINEAR 3 Source 2
        T_EXECUTE 1 3
        E_Add 4, 1, 3
        Tensor.Float.d1 6, (base=&64, lim=&72, sizes={&80 }, strides={&88 })
        T_STORE_LINEAR 7 Source 4 Dest 6
        )";
    EXPECT_EQ(NormalizedSource(command->toString()), NormalizedSource(result));

    {
        std::string expected = R"([
            Tensor_0_pointer: PointerGlobal: Float(offset: 0, size: 8, read_write),
            Tensor_0_extent: Value: Int64(offset: 8, size: 8, read_only),
            Tensor_0_size_0: Value: Int64(offset: 16, size: 8, read_only),
            Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only),
            Tensor_2_pointer: PointerGlobal: Float(offset: 32, size: 8, read_write),
            Tensor_2_extent: Value: Int64(offset: 40, size: 8, read_only),
            Tensor_2_size_0: Value: Int64(offset: 48, size: 8, read_only),
            Tensor_2_stride_0: Value: Int64(offset: 56, size: 8, read_only),
            Tensor_6_pointer: PointerGlobal: Float(offset: 64, size: 8, read_write),
            Tensor_6_extent: Value: Int64(offset: 72, size: 8, read_only),
            Tensor_6_size_0: Value: Int64(offset: 80, size: 8, read_only),
            Tensor_6_stride_0: Value: Int64(offset: 88, size: 8, read_only)
        ])";

        std::ostringstream msg;
        msg << command->getArguments();
        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(msg.str()));
    }
}

TEST_F(CommandTest, DuplicateOp)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto execute
        = std::make_shared<Operations::Operation>(Operations::T_Execute(command->getNextTag()));

    command->addOperation(execute);
#ifdef NDEBUG
    GTEST_SKIP() << "Skipping assertion check in release mode.";
#endif

    EXPECT_THROW({ command->addOperation(execute); }, FatalError);
}

TEST_F(CommandTest, XopInputOutputs)
{
    auto command = std::make_shared<rocRoller::Command>();
    command->allocateTag();
    command->allocateTag();
    Operations::OperationTag tagA(0);
    Operations::OperationTag tagB(1);
    Operations::OperationTag tagC(2);
    Operations::OperationTag tagD(3);
    Operations::OperationTag tagE(4);
    Operations::OperationTag tagF(5);

    Operations::T_Execute execute(command->getNextTag());
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Sub(tagA, tagB)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(tagC, tagB)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<Operations::OperationTag>({tagA, tagB}));
    EXPECT_EQ(execute.getOutputs(), std::unordered_set<Operations::OperationTag>({tagC, tagD}));

    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Abs(tagD)));
    execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(tagE)));

    EXPECT_EQ(execute.getInputs(), std::unordered_set<Operations::OperationTag>({tagA, tagB}));
    EXPECT_EQ(execute.getOutputs(),
              std::unordered_set<Operations::OperationTag>({tagC, tagD, tagE, tagF}));
}

TEST_F(CommandTest, BlockScaleInline)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto dataTensor = command->addOperation(Operations::Tensor(1, DataType::Int32));

    auto block_scale = Operations::BlockScale(dataTensor, 3);
    auto block_scale_tag
        = command->addOperation(std::make_shared<Operations::Operation>(block_scale));

    EXPECT_EQ(block_scale.scaleMode(), Operations::ScaleMode::Inline);
    EXPECT_EQ(block_scale.strides(), std::vector<size_t>({32, 1, 1}));
    EXPECT_EQ(command->getOperation<Operations::BlockScale>(block_scale_tag).getInputs(),
              std::unordered_set<Operations::OperationTag>({dataTensor}));
}

TEST_F(CommandTest, BlockScaleSeparate)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto dataTensor  = command->addOperation(Operations::Tensor(1, DataType::Int32));
    auto scaleTensor = command->addOperation(Operations::Tensor(1, DataType::Int32));

    auto block_scale = Operations::BlockScale(dataTensor, 3, scaleTensor, {4, 32});
    auto block_scale_tag
        = command->addOperation(std::make_shared<Operations::Operation>(block_scale));

    EXPECT_EQ(block_scale.scaleMode(), Operations::ScaleMode::Separate);
    EXPECT_EQ(block_scale.strides(), std::vector<size_t>({4, 32, 1}));
    EXPECT_EQ(command->getOperation<Operations::BlockScale>(block_scale_tag).getInputs(),
              std::unordered_set<Operations::OperationTag>({dataTensor, scaleTensor}));
}

TEST_F(CommandTest, SetCommandArguments)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto tagTensorA = command->addOperation(Operations::Tensor(1, DataType::Float));
    auto tagLoadA   = command->addOperation(Operations::T_Load_Linear(tagTensorA));

    auto tagScalarB = command->addOperation(Operations::Scalar(DataType::Float));
    auto tagLoadB   = command->addOperation(Operations::T_Load_Scalar(tagScalarB));

    Operations::T_Execute execute(command->getNextTag());
    auto                  tagResult = execute.addXOp(Operations::E_Mul(tagLoadA, tagLoadB));
    command->addOperation(std::move(execute));

    auto tagTensorResult = command->addOperation(Operations::Tensor(1, DataType::Float));
    command->addOperation(Operations::T_Store_Linear(tagResult, tagTensorResult));

    CommandArguments commandArgs = command->createArguments();

    commandArgs.setArgument(tagTensorA, ArgumentType::Limit, 10);
    EXPECT_THROW({ commandArgs.setArgument(tagTensorA, ArgumentType::Size, 10); }, FatalError);
    commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, 10);
    EXPECT_THROW({ commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1); }, FatalError);
    commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, 1);

    commandArgs.setArgument(tagScalarB, ArgumentType::Value, 2);
    EXPECT_THROW({ commandArgs.setArgument(tagScalarB, ArgumentType::Limit, 10); }, FatalError);
}

TEST_F(CommandTest, GPU_TensorDescriptor)
{
    auto command = std::make_shared<rocRoller::Command>();

    auto tagTensor = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float));

    TensorDescriptor desc(DataType::Float, {512u, 1024u}, "T");

    CommandArguments commandArgs = command->createArguments();
    auto             device      = make_shared_device<float>(512u * 1024u);
    setCommandTensorArg(commandArgs, tagTensor, desc, device.get());

    // Below should NOT error out as arguments have been set in setCommandTensorArg
    EXPECT_NO_THROW({ commandArgs.setArgument(tagTensor, ArgumentType::Value, device.get()); });
    EXPECT_NO_THROW({ commandArgs.setArgument(tagTensor, ArgumentType::Size, 0, 512u); });
    EXPECT_NO_THROW({ commandArgs.setArgument(tagTensor, ArgumentType::Size, 1, 1024u); });
    EXPECT_NO_THROW({ commandArgs.setArgument(tagTensor, ArgumentType::Stride, 0, 1024u); });
    EXPECT_NO_THROW({ commandArgs.setArgument(tagTensor, ArgumentType::Stride, 1, 1u); });
}

TEST_F(CommandTest, GetRuntimeArguments)
{
    std::unordered_map<DataType, size_t> typeSizes
        = {{DataType::Float, 4},        {DataType::Double, 8},
           {DataType::ComplexFloat, 8}, {DataType::ComplexDouble, 16},
           {DataType::Half, 2},         {DataType::Halfx2, 4},
           {DataType::FP8, 1},          {DataType::FP8x4, 4},
           {DataType::BF8, 1},          {DataType::BF8x4, 4},
           {DataType::Int8x4, 4},       {DataType::Int8, 1},
           {DataType::Int16, 2},        {DataType::Int32, 4},
           {DataType::Int64, 8},        {DataType::BFloat16, 2},
           {DataType::Raw32, 4},        {DataType::UInt8x4, 4},
           {DataType::UInt8, 1},        {DataType::UInt16, 2},
           {DataType::UInt32, 4},       {DataType::UInt64, 8},
           {DataType::E8M0, 1},         {DataType::E8M0x4, 4}};

    // Check scalar
    for(auto const& [type, bytes] : typeSizes)
    {
        auto                  command   = std::make_shared<rocRoller::Command>();
        [[maybe_unused]] auto tagScalar = command->addOperation(Operations::Scalar(type));

        CommandArguments commandArgs      = command->createArguments();
        auto             runtimeArguments = commandArgs.runtimeArguments();

        EXPECT_EQ(runtimeArguments.size(), bytes);
        EXPECT_EQ(runtimeArguments.size_bytes(), bytes);
    }

    // Check tensor
    for(auto const& iter : typeSizes)
    {
        auto                  command   = std::make_shared<rocRoller::Command>();
        [[maybe_unused]] auto tagTensor = command->addOperation(Operations::Tensor(1, iter.first));

        CommandArguments commandArgs      = command->createArguments();
        auto             runtimeArguments = commandArgs.runtimeArguments();

        // The total number of bytes required by pointer, extent, size and stride
        EXPECT_EQ(runtimeArguments.size(), 32);
        EXPECT_EQ(runtimeArguments.size_bytes(), 32);
    }
}

TEST_F(CommandTest, CommandKernelPredicates)
{
    auto command = std::make_shared<Command>();

    VariableType intVal{DataType::Int32, PointerType::Value};

    auto valTag  = command->allocateTag();
    auto val_arg = command->allocateArgument(intVal, valTag, ArgumentType::Value);

    auto val_exp = std::make_shared<Expression::Expression>(val_arg);

    CommandKernel commandKernel(command, "PredicateTestKernel");

    auto p1 = val_exp % Expression::literal(2) == Expression::literal(0);
    Expression::setComment(p1, "val must be even");
    auto p2 = val_exp % Expression::literal(5) == Expression::literal(0);
    Expression::setComment(p2, "val must be divisible by 5");

    commandKernel.addPredicate(p1);
    commandKernel.addPredicate(p2);

    CommandArguments commandArgs = command->createArguments();
    for(int i = 1; i < 100; i++)
    {
        commandArgs.setArgument(valTag, ArgumentType::Value, i);
        auto runtimeArguments = commandArgs.runtimeArguments();
        EXPECT_EQ(commandKernel.matchesPredicates(runtimeArguments), i % 2 == 0 && i % 5 == 0);
        if(!(i % 2 == 0 && i % 5 == 0))
        {
            EXPECT_THROW(commandKernel.launchKernel(runtimeArguments), FatalError);
        }
    }
}
