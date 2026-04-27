// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"
#include "rocRoller/ExecutableKernel.hpp"
#include "rocRoller/Utilities/Logging.hpp"

#include <common/TestValues.hpp>
#include <common/Utilities.hpp>

#include <functional>
#include <random>
#include <utility>
#include <vector>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace rocRoller;

namespace ArgumentLoaderGPUTest
{
    using ExprFunc = std::function<Expression::ExpressionPtr(Expression::ExpressionPtr,
                                                             Expression::ExpressionPtr)>;

    struct ArgumentLoaderExprKernel : public AssemblyTestKernel
    {

        struct ScalarArgument
        {
            std::string              name;
            DataType                 type;
            Operations::OperationTag valueTag;
            CommandArgumentPtr       valueArg;
            Operations::OperationTag pointerTag;
            CommandArgumentPtr       pointerArg;
        };

        /**
         * Function that takes the workgroup index registers and returns an expression.
         */
        using ResultExpressionFunc
            = std::function<Expression::ExpressionPtr(std::array<Register::ValuePtr, 3> const&)>;

        ArgumentLoaderExprKernel(ContextPtr context, CommandPtr command);

        void setArgumentOptions(int numPreloaded, std::optional<size_t> shuffleSeed = std::nullopt);

        void setKernelDimensions(int dims);
        void setNumWorkgroups(std::array<unsigned int, 3> numWorkgroups);
        std::array<unsigned int, 3> const& numWorkgroups() const;

        void setVectorResultArgument(std::string          name,
                                     DataType             type,
                                     ResultExpressionFunc expression);

        void                               addScalarArgument(std::string name, DataType type);
        std::vector<ScalarArgument> const& scalarArgs() const;

        struct ExpectedKernelStatistics
        {
            std::map<int, int> scalarLoadsByDWordCount;
            int                numWaits = 0;

            void combine(ExpectedKernelStatistics const& other)
            {
                for(auto const& [width, count] : other.scalarLoadsByDWordCount)
                    scalarLoadsByDWordCount[width] += count;
                numWaits += other.numWaits;
            }
        };

        ExpectedKernelStatistics kernelStatistics() const;

        void operator()(std::vector<std::pair<CommandArgumentValue, std::shared_ptr<void>>> const&
                                                  scalarArgsValues,
                        std::shared_ptr<uint32_t> resultArray = nullptr);

        void generate() override;

    protected:
        CommandPtr m_command;

        std::vector<ScalarArgument> m_scalarArgs;

        std::string              m_vectorResultArgName;
        DataType                 m_vectorResultArgType;
        Operations::OperationTag m_vectorResultTag;
        CommandArgumentPtr       m_vectorResultArg;

        std::optional<size_t> m_shuffleSeed;
        int                   m_numPreloaded = 0;

        int                         m_kernelDimensions = 1;
        std::array<unsigned int, 3> m_numWorkgroups    = {1, 1, 1};

        ResultExpressionFunc m_resultExpressionFunc = nullptr;
    };

    /**
     * Tests the ArgumentLoader using a number of GPU kernels with different numbers of scalar
     * arguments, preloaded, eagerly loaded, and lazily loaded.
     *
     * The kernels optionally:
     *
     * - Shuffle the arguments to test the ArgumentLoader's ability to handle alignment padding.
     * - Preload a subset of the arguments to test that feature.
     * - Load the arguments lazily to test that feature.
     * - Generate a result expression that is a function of the workgroup index to ensure that
     *   the workgroup index register is located correctly regardless of how arguments are
     *   loaded. This is done with 1, 2, and 3 dimensional kernels.
     */
    TEST_CASE("ArgumentLoader GPU kernel", "[argument-loader][gpu]")
    {
        auto numScalarArgs    = GENERATE(1, 4, 14, 42);
        auto numPreloadedArgs = GENERATE(0, 2, -1);

        auto numWorkgroupRegs = GENERATE(0, 1, 2, 3);

        auto                  doShuffle = GENERATE(true, false);
        std::optional<size_t> shuffleSeed;
        if(doShuffle)
        {
            shuffleSeed = GENERATE(Catch::Generators::take(
                1, Catch::Generators::random(0ul, std::numeric_limits<size_t>::max())));
        }

        std::vector<DataType> dataTypes = {DataType::Int32, DataType::Double};

        auto lazyLoad = GENERATE(true, false);

        CAPTURE(numScalarArgs, numPreloadedArgs, doShuffle, shuffleSeed.value_or(0ul), lazyLoad);

        DYNAMIC_SECTION("numScalarArgs="
                        << numScalarArgs << ", numPreloadedArgs=" << numPreloadedArgs
                        << ", numWorkgroupRegs=" << numWorkgroupRegs << ", doShuffle=" << doShuffle
                        << ", lazyLoad=" << lazyLoad)
        {

            auto context = TestContext::ForTestDevice({{.lazyLoadKernelArguments = lazyLoad}},
                                                      numScalarArgs,
                                                      numPreloadedArgs,
                                                      numWorkgroupRegs,
                                                      doShuffle,
                                                      lazyLoad);

            auto maxPreloaded
                = context->targetArchitecture().GetCapability(GPUCapability::MaxPreloadedKernargs);
            if(numPreloadedArgs < 0)
                numPreloadedArgs = maxPreloaded;

            auto command = std::make_shared<Command>();

            ArgumentLoaderExprKernel kernel(context.get(), command);
            kernel.setKernelDimensions(std::max(1, numWorkgroupRegs));
            if(numWorkgroupRegs > 0)
            {
                std::array<unsigned int, 3> workgroupSize = {7, 68, 75};
                if(numWorkgroupRegs < 3)
                    workgroupSize[2] = 1;
                if(numWorkgroupRegs < 2)
                    workgroupSize[1] = 1;
                kernel.setNumWorkgroups(workgroupSize);
                kernel.setVectorResultArgument(
                    "result",
                    DataType::UInt32,
                    [&](std::array<Register::ValuePtr, 3> const& workgroupIndex) {
                        auto result = workgroupIndex[0]->expression() * Expression::literal(2);

                        if(numWorkgroupRegs > 1)
                            result = result * workgroupIndex[1]->expression();
                        if(numWorkgroupRegs > 2)
                            result = result + workgroupIndex[2]->expression();

                        return result;
                    });
            }

            int totalRegs = 0;
            for(int i = 0; i < numScalarArgs; i++)
            {
                auto dataType = dataTypes.at(GENERATE_COPY(Catch::Generators::take(
                    1, Catch::Generators::random(0ul, dataTypes.size() - 1))));
                totalRegs += DataTypeInfo::Get(dataType).registerCount + 3;
                if((totalRegs + 10) > context->kernelOptions()->maxSGPRs)
                    break;

                kernel.addScalarArgument(fmt::format("arg_{}", i), dataType);
            }
            CAPTURE(totalRegs, context->kernelOptions()->maxSGPRs);

            REQUIRE_NOTHROW(kernel.getExecutableKernel());

            auto output        = context.output();
            auto expectedStats = kernel.kernelStatistics();

            auto countSubstr = [](std::string const& str, std::string const& substr) -> int {
                int    count = 0;
                size_t pos   = 0;
                while((pos = str.find(substr, pos)) != std::string::npos)
                {
                    count++;
                    pos += substr.length();
                }
                return count;
            };

            if(!doShuffle)
            {
                bool splitCounters = context->targetArchitecture().HasCapability(
                    GPUCapability::HasSplitWaitCounters);

                std::string inst = splitCounters ? "s_wait_kmcnt" : "s_waitcnt";
                CAPTURE(inst);
                // Shuffling makes the number of waitcnt instructions very hard to predict.
                CHECK(countSubstr(output, inst) == expectedStats.numWaits);
            }

            for(auto const& [width, count] : expectedStats.scalarLoadsByDWordCount)
            {
                std::string inst = "s_load_dword";
                if(width > 1)
                    inst += fmt::format("x{}", width);
                inst += " ";
                CAPTURE(width, inst);
                CHECK(countSubstr(output, inst) == count);
            }

            std::map<DataType, std::vector<CommandArgumentValue>> values;
            auto getRandomValue = [&](DataType dataType) -> CommandArgumentValue {
                auto iter = values.find(dataType);
                if(iter == values.end())
                {
                    iter = values.emplace(dataType, TestValues::byType(dataType)).first;
                }
                auto idx = GENERATE_COPY(Catch::Generators::take(
                    1, Catch::Generators::random(0ul, iter->second.size() - 1)));
                return iter->second.at(idx);
            };

            std::vector<std::pair<CommandArgumentValue, std::shared_ptr<void>>> scalarArgsValues;
            for(auto const& scalarArg : kernel.scalarArgs())
            {
                auto value1 = getRandomValue(scalarArg.type);
                auto value2 = getRandomValue(scalarArg.type);
                scalarArgsValues.push_back({value1, make_shared_device(value2)});
            }

            std::shared_ptr<uint32_t> resultArray;

            if(numWorkgroupRegs > 0)
            {
                resultArray = make_shared_device<uint32_t>(kernel.numWorkgroups()[0]
                                                           * kernel.numWorkgroups()[1]
                                                           * kernel.numWorkgroups()[2]);
            }

            kernel(scalarArgsValues, resultArray);

            for(auto const& [value, valuePtr] : scalarArgsValues)
            {
                auto const& valuePtrRef = valuePtr;
                auto        visitor     = [&](auto value) {
                    using T = typename std::decay_t<decltype(value)>;
                    if constexpr(CCommandArgumentValue<T*>)
                    {
                        auto ptr = std::reinterpret_pointer_cast<T>(valuePtrRef);
                        CHECK_THAT(ptr, HasDeviceScalarEqualTo(value));
                    }
                };
                std::visit(visitor, value);
            }

            if(numWorkgroupRegs > 0)
            {
                std::vector<uint32_t> expectedResult(kernel.numWorkgroups()[0]
                                                     * kernel.numWorkgroups()[1]
                                                     * kernel.numWorkgroups()[2]);
                int                   strideJ = kernel.numWorkgroups()[0];
                int strideK = kernel.numWorkgroups()[0] * kernel.numWorkgroups()[1];

                for(int i = 0; i < kernel.numWorkgroups()[0]; i++)
                {
                    for(int j = 0; j < kernel.numWorkgroups()[1]; j++)
                    {
                        for(int k = 0; k < kernel.numWorkgroups()[2]; k++)
                        {
                            uint32_t expected = i * 2;
                            if(numWorkgroupRegs > 1)
                                expected = expected * j;
                            if(numWorkgroupRegs > 2)
                                expected = expected + k;
                            expectedResult.at(i + strideJ * j + k * strideK) = expected;
                        }
                    }
                }

                CHECK_THAT(resultArray, HasDeviceVectorEqualTo(expectedResult));
            }
        }
    }

    ArgumentLoaderExprKernel::ArgumentLoaderExprKernel(ContextPtr context, CommandPtr command)
        : AssemblyTestKernel(context)
        , m_command(command)
    {
    }

    void ArgumentLoaderExprKernel::setArgumentOptions(int                   numPreloaded,
                                                      std::optional<size_t> shuffleSeed)
    {
        m_shuffleSeed  = shuffleSeed;
        m_numPreloaded = numPreloaded;
    }

    void ArgumentLoaderExprKernel::setKernelDimensions(int dims)
    {
        m_kernelDimensions = dims;
    }
    void ArgumentLoaderExprKernel::setNumWorkgroups(std::array<unsigned int, 3> numWorkgroups)
    {
        m_numWorkgroups = numWorkgroups;
    }

    std::array<unsigned int, 3> const& ArgumentLoaderExprKernel::numWorkgroups() const
    {
        return m_numWorkgroups;
    }
    std::vector<ArgumentLoaderExprKernel::ScalarArgument> const&
        ArgumentLoaderExprKernel::scalarArgs() const
    {
        return m_scalarArgs;
    }

    void ArgumentLoaderExprKernel::setVectorResultArgument(std::string          name,
                                                           DataType             type,
                                                           ResultExpressionFunc expression)
    {
        m_vectorResultArgName = std::move(name);
        m_vectorResultArgType = type;
        m_vectorResultTag     = m_command->allocateTag();
        m_vectorResultArg
            = m_command->allocateArgument(VariableType{type, PointerType::PointerGlobal},
                                          m_vectorResultTag,
                                          ArgumentType::Value,
                                          DataDirection::WriteOnly,
                                          m_vectorResultArgName);
        m_resultExpressionFunc = std::move(expression);
    }

    void ArgumentLoaderExprKernel::addScalarArgument(std::string name, DataType type)
    {
        auto pointerTag = m_command->allocateTag();
        auto pointerArg
            = m_command->allocateArgument(VariableType{type, PointerType::PointerGlobal},
                                          pointerTag,
                                          ArgumentType::Value,
                                          DataDirection::WriteOnly,
                                          name + "_pointer");

        auto valueTag = m_command->allocateTag();
        auto valueArg = m_command->allocateArgument(VariableType{type, PointerType::Value},
                                                    valueTag,
                                                    ArgumentType::Value,
                                                    DataDirection::ReadOnly,
                                                    name + "_value");

        m_scalarArgs.push_back({std::move(name), type, valueTag, valueArg, pointerTag, pointerArg});
    }

    void ArgumentLoaderExprKernel::operator()(
        std::vector<std::pair<CommandArgumentValue, std::shared_ptr<void>>> const& scalarArgsValues,
        std::shared_ptr<uint32_t>                                                  resultArray)
    {
        auto commandArgs = m_command->createArguments();

        KernelInvocation invocation;
        invocation.workitemCount = m_numWorkgroups;

        REQUIRE(scalarArgsValues.size() == m_scalarArgs.size());

        for(int i = 0; i < scalarArgsValues.size(); i++)
        {
            auto const& [argValue, argValuePtr] = scalarArgsValues[i];
            auto const& argValuePtrRef          = argValuePtr;
            auto const& scalarArg               = m_scalarArgs[i];
            commandArgs.setArgument(scalarArg.valueTag, ArgumentType::Value, argValue);
            auto visitor = [&](auto value) {
                using T = typename std::decay_t<decltype(value)>;
                if constexpr(CCommandArgumentValue<T*>)
                {
                    auto ptr = std::reinterpret_pointer_cast<T>(argValuePtrRef);
                    commandArgs.setArgument(scalarArg.pointerTag, ArgumentType::Value, ptr.get());
                }
                else
                {
                    throw std::runtime_error("Unsupported command argument value type");
                }
            };

            std::visit(visitor, argValue);
        }

        if(m_resultExpressionFunc)
        {
            commandArgs.setArgument(m_vectorResultTag, ArgumentType::Value, resultArray.get());
        }

        KernelArguments kernelArgs(true);

        for(auto karg : m_context->kernel()->arguments())
        {
            auto value = Expression::evaluate(karg.getExpression(), commandArgs.runtimeArguments());
            kernelArgs.append(karg.getName(), value);
        }

        AssemblyTestKernel::operator()(invocation, kernelArgs);
    }

    void ArgumentLoaderExprKernel::generate()
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(m_kernelDimensions);

        for(auto const& arg : m_scalarArgs)
        {
            k->addArgument({arg.name + "_pointer",
                            {arg.type, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            std::make_shared<Expression::Expression>(arg.pointerArg)});
            k->addArgument({arg.name + "_value",
                            arg.type,
                            DataDirection::ReadOnly,
                            std::make_shared<Expression::Expression>(arg.valueArg)});
        }
        if(m_vectorResultArg)
        {
            k->addArgument({m_vectorResultArgName,
                            {m_vectorResultArgType, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            std::make_shared<Expression::Expression>(m_vectorResultArg)});
        }

        if(m_shuffleSeed)
        {
            auto         kargs = k->resetArguments();
            std::mt19937 rng(*m_shuffleSeed);
            std::shuffle(kargs.begin(), kargs.end(), rng);

            for(auto& arg : kargs)
            {
                arg.setOffset(-1);
                k->addArgument(std::move(arg));
            }
        }

        if(m_numPreloaded > 0)
        {
            auto kargs = k->resetArguments();

            int count = 0;

            for(auto& arg : kargs)
            {
                arg.setPreloaded(arg.getOffset() + arg.getSize() <= m_numPreloaded);

                arg.setOffset(-1);
                k->addArgument(std::move(arg));
                count++;
            }
        }

        auto kb = [&]() -> Generator<Instruction> {
            for(auto const& arg : m_scalarArgs)
            {
                auto               typeInfo = DataTypeInfo::Get(arg.type);
                Register::ValuePtr argValue, argPointer;
                co_yield m_context->argLoader()->getValue(arg.name + "_value", argValue);
                co_yield m_context->argLoader()->getValue(arg.name + "_pointer", argPointer);

                co_yield m_context->copier()->ensureType(
                    argPointer, argPointer, Register::Type::Vector);
                co_yield m_context->copier()->ensureType(
                    argValue, argValue, Register::Type::Vector);

                co_yield m_context->mem()->storeGlobal(
                    argPointer, argValue, 0, typeInfo.elementBytes);
            }

            if(m_resultExpressionFunc)
            {
                co_yield Instruction::Comment("Result expression");
                auto               resultExpression = m_resultExpressionFunc(k->workgroupIndex());
                Register::ValuePtr resultPointer,
                    resultValue = Register::Value::Placeholder(
                        m_context, Register::Type::Scalar, m_vectorResultArgType, 1);
                auto typeInfo = DataTypeInfo::Get(m_vectorResultArgType);

                co_yield Expression::generate(resultValue, resultExpression, m_context);

                co_yield m_context->argLoader()->getValue(m_vectorResultArgName, resultPointer);

                auto resultIndexExpr = k->workgroupIndex()[0]->expression();
                int  stride          = 1;
                for(int i = 1; i < m_kernelDimensions; i++)
                {
                    stride *= m_numWorkgroups[i - 1];
                    resultIndexExpr
                        = resultIndexExpr
                          + k->workgroupIndex()[i]->expression() * Expression::literal(stride);
                }
                resultIndexExpr = resultIndexExpr * Expression::literal(typeInfo.elementBytes);

                Register::ValuePtr storePointer;

                co_yield Expression::generate(
                    storePointer, resultPointer->expression() + resultIndexExpr, m_context);

                co_yield m_context->copier()->ensureType(
                    storePointer, storePointer, Register::Type::Vector);
                co_yield m_context->copier()->ensureType(
                    resultValue, resultValue, Register::Type::Vector);

                co_yield m_context->mem()->storeGlobal(
                    storePointer, resultValue, 0, typeInfo.elementBytes);
            }
            else
            {
                co_yield Instruction::Comment("No result expression");
            }
        };

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());
        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    ArgumentLoaderExprKernel::ExpectedKernelStatistics
        ArgumentLoaderExprKernel::kernelStatistics() const
    {
        ExpectedKernelStatistics stats;

        const auto widths = {16, 8, 4, 2, 1};

        for(auto width : widths)
            stats.scalarLoadsByDWordCount[width] = 0;

        auto lazyLoad = m_context->kernelOptions()->lazyLoadKernelArguments;
        Log::debug("LazyLoad: {}", lazyLoad);

        int eagerBegin = 0, eagerEnd = 0;

        for(auto const& karg : m_context->kernel()->arguments())
        {
            if(karg.getPreloaded())
                eagerBegin = karg.getOffset() + karg.getSize();

            if(!karg.getPreloaded() && !lazyLoad)
                eagerEnd = karg.getOffset() + karg.getSize();

            if(lazyLoad && !karg.getPreloaded())
            {
                stats.numWaits++;
                if(karg.getSize() == 4)
                {
                    stats.scalarLoadsByDWordCount[1]++;
                }
                else if(karg.getSize() == 8)
                {
                    stats.scalarLoadsByDWordCount[2]++;
                }
                else
                {
                    FAIL("Unsupported argument size: " + std::to_string(karg.getSize()));
                }
            }
        }

        if(lazyLoad && stats.numWaits > 0)
            stats.numWaits = CeilDivide(stats.numWaits, 2);

        if(!lazyLoad)
        {
            ExpectedKernelStatistics eagerStats;

            int numEagerBytes = eagerEnd - eagerBegin;

            Log::debug("numEagerBytes: {}", numEagerBytes);
            if(numEagerBytes > 0)
                eagerStats.numWaits++;

            int numLoadedBytes = 0;

            for(auto widthIter = widths.begin();
                numEagerBytes > numLoadedBytes && widthIter != widths.end();
                ++widthIter)
            {
                auto widthBytes      = *widthIter * 4;
                auto numInstructions = (numEagerBytes - numLoadedBytes) / widthBytes;

                eagerStats.scalarLoadsByDWordCount[*widthIter] += numInstructions;

                numLoadedBytes += numInstructions * widthBytes;
            }

            if(numEagerBytes > numLoadedBytes)
            {
                eagerStats.scalarLoadsByDWordCount[1]++;
            }

            Log::debug("numEagerBytes={}, numLoadedBytes={}, eagerBegin={}, eagerEnd={}",
                       numEagerBytes,
                       numLoadedBytes,
                       eagerBegin,
                       eagerEnd);
            stats.combine(eagerStats);
        }

        return stats;
    }

}
