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

#include <cmath>
#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace RandomNumberGenerationTest
{
    TEST_CASE("Single random number generator", "[gpu][prng]")
    {
        auto seedMode = GENERATE(rocRoller::Operations::RandomNumberGenerator::SeedMode::Default,
                                 rocRoller::Operations::RandomNumberGenerator::SeedMode::PerThread);

        DYNAMIC_SECTION(seedMode)
        {
            auto nx = 32u; // Number of elements in Tensor A

            // Command = Store_Linear( E_Add( A, E_RandomNumber( RandomNumberGenerator(Seed) ) ))
            auto command = std::make_shared<Command>();

            // Seed
            auto tagSeed = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
            auto tagLoadSeed = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));

            // RandomNumberGenerator(Seed)
            auto prng = command->addOperation(
                rocRoller::Operations::RandomNumberGenerator(tagLoadSeed, seedMode));

            // A
            auto tagA = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Linear(tagA));

            // A + E_RandomNumber(RNG(Seed))
            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto randomNumber = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
            auto addOp = execute.addXOp(rocRoller::Operations::E_Add(randomNumber, tagLoadA));

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            // Store_Linear
            auto outputTag
                = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
            command->addOperation(rocRoller::Operations::T_Store_Linear(addOp, outputTag));

            CommandKernel commandKernel(command, "SingleRandomNumberGenerator");
            auto          context = TestContext::ForTestDevice();
            commandKernel.setContext(context.get());
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            uint32_t seed = 123u;
            commandArgs.setArgument(tagSeed, ArgumentType::Value, seed);

            std::vector<uint32_t> A(nx, 0u);
            auto                  d_A = make_shared_device(A);
            commandArgs.setArgument(tagA, ArgumentType::Value, d_A.get());
            commandArgs.setArgument(tagA, ArgumentType::Limit, nx);
            commandArgs.setArgument(tagA, ArgumentType::Size, 0, nx);
            commandArgs.setArgument(tagA, ArgumentType::Stride, 0, (size_t)1);

            std::vector<uint32_t> output(nx, 0u);
            auto                  d_output = make_shared_device(output);
            commandArgs.setArgument(outputTag, ArgumentType::Value, d_output.get());
            commandArgs.setArgument(outputTag, ArgumentType::Limit, nx);
            commandArgs.setArgument(outputTag, ArgumentType::Size, 0, nx);
            commandArgs.setArgument(outputTag, ArgumentType::Stride, 0, (size_t)1);

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            std::vector<uint32_t> result(nx);
            CHECK_THAT(
                hipMemcpy(result.data(), d_output.get(), sizeof(uint32_t) * nx, hipMemcpyDefault),
                HasHipSuccess(0));

            for(size_t i = 0; i < result.size(); i++)
            {
                if(seedMode == rocRoller::Operations::RandomNumberGenerator::SeedMode::Default)
                    CHECK(result[i] == LFSRRandomNumberGenerator(seed));
                else
                    CHECK(result[i] == LFSRRandomNumberGenerator(seed + i));
            }
        }
    }

    TEST_CASE("Two random number generators", "[gpu][prng]")
    {
        auto seedMode = GENERATE(rocRoller::Operations::RandomNumberGenerator::SeedMode::Default,
                                 rocRoller::Operations::RandomNumberGenerator::SeedMode::PerThread);

        DYNAMIC_SECTION(seedMode)
        {
            auto nx = 32u; // Number of elements in Tensor A

            std::vector<rocRoller::Operations::OperationTag> seedTags;
            std::vector<rocRoller::Operations::OperationTag> inputTags;
            std::vector<rocRoller::Operations::OperationTag> outputTags;

            auto const numPRNGs = 4;
            auto       command  = std::make_shared<Command>();

            for(size_t i = 0; i < numPRNGs; i++)
            {
                // Seed
                auto tagSeed
                    = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
                auto tagLoadSeed
                    = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));
                seedTags.push_back(tagSeed);

                // RandomNumberGenerator(Seed)
                auto prng = command->addOperation(
                    rocRoller::Operations::RandomNumberGenerator(tagLoadSeed, seedMode));

                // A
                auto tagA
                    = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
                auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Linear(tagA));
                inputTags.push_back(tagA);

                // A + E_RandomNumber(RNG(SeedA))
                rocRoller::Operations::T_Execute execute(command->getNextTag());
                auto randomNumber = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
                auto addOp = execute.addXOp(rocRoller::Operations::E_Add(randomNumber, tagLoadA));

                command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

                // Store_Linear
                auto outputTag
                    = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
                command->addOperation(rocRoller::Operations::T_Store_Linear(addOp, outputTag));
                outputTags.push_back(outputTag);
            }

            CommandKernel commandKernel(command, "TwoRandomNumberGenerators");
            auto          context = TestContext::ForTestDevice();
            commandKernel.setContext(context.get());
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            std::vector<uint32_t> seeds(numPRNGs);
            std::iota(seeds.begin(), seeds.end(), 1);
            for(size_t i = 0; i < numPRNGs; i++)
                commandArgs.setArgument(seedTags[i], ArgumentType::Value, seeds[i]);

            // All PRNGs share A
            std::vector<uint32_t> A(nx, 0u);
            auto                  d_A = make_shared_device(A);
            for(auto tag : inputTags)
            {
                commandArgs.setArgument(tag, ArgumentType::Value, d_A.get());
                commandArgs.setArgument(tag, ArgumentType::Limit, nx);
                commandArgs.setArgument(tag, ArgumentType::Size, 0, nx);
                commandArgs.setArgument(tag, ArgumentType::Stride, 0, (size_t)1);
            }

            std::vector<std::shared_ptr<uint32_t>> d_outputs;
            for(size_t i = 0; i < numPRNGs; i++)
            {
                auto d_output = make_shared_device<uint32_t>(nx, 0);
                commandArgs.setArgument(outputTags[i], ArgumentType::Value, d_output.get());
                commandArgs.setArgument(outputTags[i], ArgumentType::Limit, nx);
                commandArgs.setArgument(outputTags[i], ArgumentType::Size, 0, nx);
                commandArgs.setArgument(outputTags[i], ArgumentType::Stride, 0, (size_t)1);
                d_outputs.push_back(d_output);
            }

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            std::vector<std::vector<uint32_t>> results(numPRNGs);
            for(size_t i = 0; i < numPRNGs; i++)
            {
                results[i].resize(nx);
                CHECK_THAT(hipMemcpy(results[i].data(),
                                     d_outputs[i].get(),
                                     sizeof(uint32_t) * nx,
                                     hipMemcpyDefault),
                           HasHipSuccess(0));
            }

            for(auto const& result : results)
                for(size_t i = 0; i < numPRNGs; i++)
                {
                    for(size_t j = 0; j < result.size(); j++)
                    {
                        if(seedMode
                           == rocRoller::Operations::RandomNumberGenerator::SeedMode::Default)
                            CHECK(results[i][j] == LFSRRandomNumberGenerator(seeds[i]));
                        else
                            CHECK(results[i][j] == LFSRRandomNumberGenerator(seeds[i] + j));
                    }
                }
        }
    }

    TEST_CASE("Generate two random values", "[gpu][prng]")
    {
        auto seedMode = GENERATE(rocRoller::Operations::RandomNumberGenerator::SeedMode::Default,
                                 rocRoller::Operations::RandomNumberGenerator::SeedMode::PerThread);

        DYNAMIC_SECTION(seedMode)
        {
            auto nx = 32u; // Number of elements in Tensor A

            auto command = std::make_shared<Command>();

            // Seed
            auto tagSeed = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
            auto tagLoadSeed = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));

            // RandomNumberGenerator(Seed)
            auto prng = command->addOperation(
                rocRoller::Operations::RandomNumberGenerator(tagLoadSeed, seedMode));

            // A
            auto tagA = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Linear(tagA));

            // A + E_RandomNumber(RNG(Seed))
            rocRoller::Operations::OperationTag addOp1;
            {
                rocRoller::Operations::T_Execute execute(command->getNextTag());
                auto randomNumber1 = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
                addOp1 = execute.addXOp(rocRoller::Operations::E_Add(randomNumber1, tagLoadA));
                command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
            }

            // addOp1 + E_RandomNumber(RNG(Seed))
            rocRoller::Operations::OperationTag addOp2;
            {
                rocRoller::Operations::T_Execute execute(command->getNextTag());
                auto randomNumber2 = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
                addOp2 = execute.addXOp(rocRoller::Operations::E_Add(randomNumber2, addOp1));
                command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
            }

            // Store 1st addition result
            auto outputTag1
                = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
            command->addOperation(rocRoller::Operations::T_Store_Linear(addOp1, outputTag1));

            // Store 2nd addition result
            auto outputTag2
                = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
            command->addOperation(rocRoller::Operations::T_Store_Linear(addOp2, outputTag2));

            auto context = TestContext::ForTestDevice({}, seedMode);

            CommandKernel commandKernel(command, context->kernel()->kernelName());
            commandKernel.setContext(context.get());
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            uint32_t seed = 123u;
            commandArgs.setArgument(tagSeed, ArgumentType::Value, seed);

            std::vector<uint32_t> A(nx, 0u);
            auto                  d_A = make_shared_device(A);
            commandArgs.setArgument(tagA, ArgumentType::Value, d_A.get());
            commandArgs.setArgument(tagA, ArgumentType::Limit, nx);
            commandArgs.setArgument(tagA, ArgumentType::Size, 0, nx);
            commandArgs.setArgument(tagA, ArgumentType::Stride, 0, (size_t)1);

            std::vector<uint32_t> output1(nx, 0u);
            auto                  d_output1 = make_shared_device(output1);
            commandArgs.setArgument(outputTag1, ArgumentType::Value, d_output1.get());
            commandArgs.setArgument(outputTag1, ArgumentType::Limit, nx);
            commandArgs.setArgument(outputTag1, ArgumentType::Size, 0, nx);
            commandArgs.setArgument(outputTag1, ArgumentType::Stride, 0, (size_t)1);

            std::vector<uint32_t> output2(nx, 0u);
            auto                  d_output2 = make_shared_device(output2);
            commandArgs.setArgument(outputTag2, ArgumentType::Value, d_output2.get());
            commandArgs.setArgument(outputTag2, ArgumentType::Limit, nx);
            commandArgs.setArgument(outputTag2, ArgumentType::Size, 0, nx);
            commandArgs.setArgument(outputTag2, ArgumentType::Stride, 0, (size_t)1);

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            std::vector<uint32_t> result1(nx);
            CHECK_THAT(
                hipMemcpy(result1.data(), d_output1.get(), sizeof(uint32_t) * nx, hipMemcpyDefault),
                HasHipSuccess(0));

            std::vector<uint32_t> result2(nx);
            CHECK_THAT(
                hipMemcpy(result2.data(), d_output2.get(), sizeof(uint32_t) * nx, hipMemcpyDefault),
                HasHipSuccess(0));

            for(size_t i = 0; i < nx; i++)
            {
                uint32_t randomNumber1
                    = (seedMode == rocRoller::Operations::RandomNumberGenerator::SeedMode::Default)
                          ? LFSRRandomNumberGenerator(seed)
                          : LFSRRandomNumberGenerator(seed + i);
                auto randomNumber2 = LFSRRandomNumberGenerator(randomNumber1);
                auto finalOutput   = randomNumber1 + randomNumber2;
                CHECK(finalOutput == result2[i]);

                // These two random numbers have no dependency, so they might be generated in
                // any order.
                CHECK(
                    ((result1[i] == randomNumber1) or (result2[i] - result1[i] == randomNumber1)));
                CHECK(
                    ((result1[i] == randomNumber2) or (result2[i] - result1[i] == randomNumber2)));
            }
        }
    }

    TEST_CASE("PRNG works for a two dimensional kernel", "[gpu][prng]")
    {
        auto seedMode = GENERATE(rocRoller::Operations::RandomNumberGenerator::SeedMode::Default,
                                 rocRoller::Operations::RandomNumberGenerator::SeedMode::PerThread);

        DYNAMIC_SECTION(seedMode)
        {
            auto nx = 32u; // Length of X dim in Tensor A
            auto ny = 16u; // Length of Y dim in Tensor A

            // Thread tile size. A thread tile is the elements processed by a thread (workitem)
            auto t_m = 4;
            auto t_n = 2;

            // Macro tile size. A macro tile is the elements processed by a workgroup
            auto m = 8;
            auto n = 4;

            // Number of workitems (in each dimension) in a workgroup
            unsigned int workgroup_size_x = m / t_m;
            unsigned int workgroup_size_y = n / t_n;

            // Number of workgroups (in each dimension) of a kernel
            auto NX = std::make_shared<Expression::Expression>(nx / t_m);
            auto NY = std::make_shared<Expression::Expression>(ny / t_n);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            auto command = std::make_shared<Command>();

            // Seed
            auto tagSeed = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
            auto tagLoadSeed = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));

            // RandomNumberGenerator(Seed)
            auto prng = command->addOperation(
                rocRoller::Operations::RandomNumberGenerator(tagLoadSeed, seedMode));

            // A
            auto tagA = command->addOperation(rocRoller::Operations::Tensor(2, DataType::UInt32));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagA));

            // A + E_RandomNumber(RNG(Seed))
            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto prn   = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
            auto addOp = execute.addXOp(rocRoller::Operations::E_Add(prn, tagLoadA));

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            auto outputTag
                = command->addOperation(rocRoller::Operations::Tensor(2, DataType::UInt32));
            command->addOperation(rocRoller::Operations::T_Store_Tiled(addOp, outputTag));

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            auto macTile
                = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
            params->setDimensionInfo(tagLoadA, macTile);
            params->setDimensionInfo(outputTag, macTile);

            CommandKernel commandKernel(command, "PRNG_2D");

            auto context = TestContext::ForTestDevice();
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            auto launch = std::make_shared<CommandLaunchParameters>();
            launch->setManualWorkitemCount({NX, NY, NZ});
            commandKernel.setLaunchParameters(launch);

            CommandArguments commandArgs = command->createArguments();

            uint32_t seed = 345u;
            commandArgs.setArgument(tagSeed, ArgumentType::Value, seed);

            std::vector<uint32_t> A(nx * ny, 0u);
            auto                  d_A = make_shared_device(A);
            TensorDescriptor      descA(DataType::UInt32, {nx, ny}, {ny, 1});
            setCommandTensorArg(commandArgs, tagA, descA, d_A.get());

            std::vector<uint32_t> output(nx * ny, 0u);
            auto                  d_output = make_shared_device(output);
            TensorDescriptor      descOutput(DataType::UInt32, {nx, ny}, {ny, 1});
            setCommandTensorArg(commandArgs, outputTag, descOutput, d_output.get());

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            std::vector<uint32_t> result(nx * ny, 0u);
            CHECK_THAT(
                hipMemcpy(
                    result.data(), d_output.get(), sizeof(uint32_t) * nx * ny, hipMemcpyDefault),
                HasHipSuccess(0));

            if(seedMode == rocRoller::Operations::RandomNumberGenerator::SeedMode::Default)
            {
                // In Default mode, all threads (workitems) get the same random value
                // as their seeds are identical.
                CHECK(std::all_of(
                    result.begin(), result.end(), [&](int value) { return value == result[0]; }));
            }
            else
            {
                // In PerThread mode, threads (workitems) might get different random values
                // as their seeds have incorporated thread IDs
                CHECK_FALSE(std::all_of(
                    result.begin(), result.end(), [&](int value) { return value == result[0]; }));

                // In this test, the thread tile dimension is 4x2, so values in each 4x2
                // submatrix should be identical as they are generated by the same thread
                //
                // For example (look at each 4x2 submatrix where all values in it
                // are identical):
                //
                //    690  690  692  692  694  694  696  696  ...
                //    690  690  692  692  694  694  696  696
                //    690  690  692  692  694  694  696  696
                //    690  690  692  692  694  694  696  696
                //    692  692  694  694  696  696  698  698
                //    692  692  694  694  696  696  698  698
                //    692  692  694  694  696  696  698  698
                //    692  692  694  694  696  696  698  698
                //    ...
                for(size_t i = 0; i < nx; i += t_m)
                {
                    for(size_t j = 0; j < ny; j += t_n)
                    {
                        // Random values in each thread tile should be identical
                        for(size_t k = 0; k < t_m; k++)
                        {
                            for(size_t l = 0; l < t_n; l++)
                            {
                                CHECK(result[i * ny + j] == result[(i + k) * ny + (j + l)]);
                            }
                        }
                    }
                }
            }
        }
    }

    TEST_CASE("Check random numbers are generated in topological order",
              "[prng][kernel-graph][transformation][graph-transforms]")
    {
        auto seedMode = rocRoller::Operations::RandomNumberGenerator::SeedMode::Default;

        auto nx = 32u; // Number of elements in Tensor A

        auto command = std::make_shared<Command>();

        // Seed
        auto tagSeed     = command->addOperation(rocRoller::Operations::Scalar(DataType::UInt32));
        auto tagLoadSeed = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagSeed));

        // RandomNumberGenerator(Seed)
        auto prng = command->addOperation(
            rocRoller::Operations::RandomNumberGenerator(tagLoadSeed, seedMode));
        // A
        auto tagA     = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Linear(tagA));

        rocRoller::Operations::T_Execute execute(command->getNextTag());

        // Create three random numbers
        auto rn1 = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
        auto rn2 = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));
        auto rn3 = execute.addXOp(rocRoller::Operations::E_RandomNumber(prng));

        // A + RN3
        auto addOp = execute.addXOp(rocRoller::Operations::E_Add(tagLoadA, rn3));
        // A + RN3 - RN2
        auto subOp = execute.addXOp(rocRoller::Operations::E_Sub(addOp, rn2));
        // (A + RN3 - RN2) * RN1
        auto mulOp = execute.addXOp(rocRoller::Operations::E_Mul(subOp, rn1));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        // Store_Linear
        auto outputTag = command->addOperation(rocRoller::Operations::Tensor(1, DataType::UInt32));
        command->addOperation(rocRoller::Operations::T_Store_Linear(mulOp, outputTag));

        using namespace rocRoller::KernelGraph;
        using GD = rocRoller::Graph::Direction;

        auto  kgraph0  = translate(command);
        auto& control  = kgraph0.control;
        auto  seedPRNG = control.getNodes<ControlGraph::SeedPRNG>().to<std::vector>();
        CHECK(seedPRNG.size() == 1u); // Should only have one PRNG node
        auto randomNumbers
            = control.getOutputNodeIndices<ControlGraph::Sequence>(seedPRNG[0]).to<std::vector>();
        CHECK(randomNumbers.size() == 3u); // Should have three random number nodes

        // Initial control graph looks like this (see control.toDOT()):
        //
        //                             SeedPRNG
        //            ____________________|___________________
        //           |                    |                   |
        //           |                    |                   |
        //           v                    v                   v
        //          RN1                  RN2                 RN3
        //           |                    |                   |
        //           |                    |                   |
        //           |                    |                   v
        //           |                    |                  Add
        //           |                    v                   |
        //           |                   Sub <-----------------
        //           v                    |
        //          Mul <------------------
        {
            auto childNodes = control.childNodes(seedPRNG[0]).to<std::vector>();
            CHECK(childNodes.size() == 3u);
        }
        for(auto node : randomNumbers)
        {
            auto parentNodes = control.parentNodes(node).to<std::vector>();
            CHECK(parentNodes.size() == 1);
            CHECK(parentNodes[0] == seedPRNG[0]);
        }

        for(auto n0 : {0, 1, 2})
        {
            for(auto n1 : {0, 1, 2})
            {
                if(n0 != n1)
                {
                    auto path = control
                                    .path<GD::Downstream>(std::vector<int>{randomNumbers[n0]},
                                                          std::vector<int>{randomNumbers[n1]})
                                    .to<std::vector>();
                    CHECK(path.empty());
                }
            }
        }

        // Apply transformation
        auto context          = TestContext::ForDefaultTarget();
        auto addPRNGTransform = std::make_shared<rocRoller::KernelGraph::AddPRNG>(context.get());
        auto kgraph1          = kgraph0.transform(addPRNGTransform);

        // After transformation, each RN will be replaced by a subgraph rooted at Block like
        // this (RN1 and RN2 are omitted):
        //
        //                             SeedPRNG
        //            ____________________|___________________
        //           |                    |                   |
        //           |                    |                   |
        //           v                    v                   v       [Body]
        //         Block                Block               Block -------------
        //           |                    |                   |               |
        //           |                    |                   |               |
        //           |                    |                   v               v
        //           |                    |                  Add           Assign (RandomNumber)
        //           |                    |                   |               |
        //           v                    v                   v               |
        //           .                    .                   .               v
        //           .                    .                   .              RN3
        //           .                    .                   .
        {
            for(auto block :
                kgraph1.control.getNodes<rocRoller::KernelGraph::ControlGraph::Block>())
            {
                // From SeedPRNG to a Block, the path should have 3 elements (nodes and edges)
                auto path = kgraph1.control
                                .path<GD::Downstream>(std::vector<int>{seedPRNG[0]},
                                                      std::vector<int>{block})
                                .to<std::vector>();
                CHECK(path.size() == 3u);
            }
            for(auto randomNumber : randomNumbers)
            {
                // From SeedPRNG to a RandomNumber, the path should have 7 elements (nodes and edges)
                auto path = kgraph1.control
                                .path<GD::Downstream>(std::vector<int>{seedPRNG[0]},
                                                      std::vector<int>{randomNumber})
                                .to<std::vector>();
                CHECK(path.size() == 7u);
            }
        }
    }

}
