// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>

#include <array>
#include <cstring>
#include <string>
#include <vector>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm makeProblem(rocisa::DataType f8, bool halfA)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            halfA ? rocisa::DataType::Half : f8,
                                                            halfA ? f8 : rocisa::DataType::Half,
                                                            f8,
                                                            f8,
                                                            16,
                                                            16,
                                                            256,
                                                            1,
                                                            16,
                                                            4096,
                                                            256,
                                                            4096,
                                                            16,
                                                            256,
                                                            16,
                                                            256,
                                                            0.0);
        problem.setComputeInputTypeA(f8);
        problem.setComputeInputTypeB(f8);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setUseScaleAB("Scalar");
        problem.setScaleA(rocisa::DataType::Float, 1);
        problem.setScaleB(rocisa::DataType::Float, 1);
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    void configureSolution(ContractionSolution&          solution,
                           ContractionProblemGemm const& problem,
                           int                           gsu)
    {
        solution.kernelName                        = "float8_scaling_test";
        solution.kernelArgsLog                     = true;
        solution.problemType.aType                 = problem.a().dataType();
        solution.problemType.bType                 = problem.b().dataType();
        solution.problemType.cType                 = problem.c().dataType();
        solution.problemType.dType                 = problem.d().dataType();
        solution.problemType.computeInputTypeA     = problem.computeInputTypeA();
        solution.problemType.computeInputTypeB     = problem.computeInputTypeB();
        solution.problemType.useScaleAB            = problem.useScaleAB();
        solution.sizeMapping.macroTile             = TensileLite::dim3(16, 16, 1);
        solution.sizeMapping.threadTile            = TensileLite::dim3(1, 1, 1);
        solution.sizeMapping.workGroupSize         = TensileLite::dim3(64, 1, 1);
        solution.sizeMapping.depthU                = 32;
        solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
        solution.sizeMapping.globalSplitU          = gsu;
        solution.sizeMapping.globalAccumulation    = 2; // MultipleBuffer
        solution.sizeMapping.workspaceSizePerElemC = sizeof(float);
        solution.sizeMapping.workGroupMapping      = 1;
        solution.sizeMapping.workGroupMappingXCC   = 1;
        solution.sizeMapping.CUOccupancy           = 1;
    }

    void expectPointer(KernelArguments const& args, std::string const& name, void const* expected)
    {
        auto it = KernelArguments::const_iterator(args, name);
        ASSERT_NE(it, args.end()) << name;
        auto [bytes, size] = *it;
        ASSERT_EQ(size, sizeof(void const*));
        void const* actual = nullptr;
        std::memcpy(&actual, bytes, size);
        EXPECT_EQ(actual, expected) << name;
    }
}

TEST(Float8ConversionArguments, ScalarNarrowingScaleIsConsumedOnlyByMainKernel)
{
    const AMDGPU               hardware(AMDGPU::Processor::gfx942, 304, "test-gfx942");
    std::array<uint8_t, 16384> workspace{};
    float                      a = 0.9f, b = 1.0f, c = 0.0f, d = 0.0f, scaleA = 3.0f, scaleB = 0.5f;
    ContractionInputs          inputs(&a, &b, &c, &d, 1.0f, 0.0f);
    inputs.ws     = workspace.data();
    inputs.scaleA = &scaleA;
    inputs.scaleB = &scaleB;
    for(auto f8 : {rocisa::DataType::Float8,
                   rocisa::DataType::BFloat8,
                   rocisa::DataType::Float8_fnuz,
                   rocisa::DataType::BFloat8_fnuz})
        for(bool halfA : {true, false})
            for(int gsu : {2, 4})
            {
                SCOPED_TRACE(::testing::Message()
                             << int(f8) << " HalfA=" << halfA << " GSU=" << gsu);
                auto                problem = makeProblem(f8, halfA);
                ContractionSolution solution;
                configureSolution(solution, problem, gsu);
                const auto calls = solution.solve(problem, inputs, hardware);
                ASSERT_EQ(calls.size(), 2);
                expectPointer(calls.front().args, "scaleA", &scaleA);
                expectPointer(calls.front().args, "scaleB", &scaleB);
                // Null means unit scale in the conversion helper. The Half
                // scale was consumed before narrowing; the other scale remains.
                expectPointer(calls.back().args, "scaleA", halfA ? nullptr : &scaleA);
                expectPointer(calls.back().args, "scaleB", halfA ? &scaleB : nullptr);
                EXPECT_EQ(inputs.scaleA, &scaleA);
                EXPECT_EQ(inputs.scaleB, &scaleB);
            }
}

TEST(Float8ConversionArguments, PreservesVectorAndNonNarrowingScales)
{
    const AMDGPU               hardware(AMDGPU::Processor::gfx942, 304, "test-gfx942");
    std::array<uint8_t, 16384> workspace{};
    float                      a = 0.9f, b = 1.0f, c = 0.0f, d = 0.0f;
    std::array<float, 16>      scaleA{}, scaleB{};
    ContractionInputs          inputs(&a, &b, &c, &d, 1.0f, 0.0f);
    inputs.ws     = workspace.data();
    inputs.scaleA = scaleA.data();
    inputs.scaleB = scaleB.data();
    for(auto f8 : {rocisa::DataType::Float8,
                   rocisa::DataType::BFloat8,
                   rocisa::DataType::Float8_fnuz,
                   rocisa::DataType::BFloat8_fnuz})
        for(bool halfA : {true, false})
            for(auto mode : {"Scalar", "Vector"})
                for(bool narrow : {false, true})
                    for(int accumulation : {2, 3}) // MultipleBuffer / SingleKernel
                        for(int gsu : {1, 2, 4})
                        {
                            SCOPED_TRACE(::testing::Message()
                                         << int(f8) << " HalfA=" << halfA << " " << mode
                                         << " narrow=" << narrow << " accumulation=" << accumulation
                                         << " GSU=" << gsu);
                            auto problem = makeProblem(f8, halfA);
                            problem.setUseScaleAB(mode);
                            if(!narrow)
                            {
                                problem.setComputeInputTypeA(rocisa::DataType::Half);
                                problem.setComputeInputTypeB(rocisa::DataType::Half);
                            }
                            ContractionSolution solution;
                            configureSolution(solution, problem, gsu);
                            solution.sizeMapping.globalAccumulation = accumulation;
                            const auto calls      = solution.solve(problem, inputs, hardware);
                            const bool conversion = accumulation == 2 && gsu > 1;
                            ASSERT_EQ(calls.size(), conversion ? 2 : 1);
                            expectPointer(calls.front().args, "scaleA", scaleA.data());
                            expectPointer(calls.front().args, "scaleB", scaleB.data());
                            if(conversion)
                            {
                                const bool consumed = narrow && std::string(mode) == "Scalar";
                                expectPointer(calls.back().args,
                                              "scaleA",
                                              consumed && halfA ? nullptr : scaleA.data());
                                expectPointer(calls.back().args,
                                              "scaleB",
                                              consumed && !halfA ? nullptr : scaleB.data());
                            }
                        }
}

TEST(Float8ConversionArguments, GroupedConversionConsumesEachNarrowedScalarOnce)
{
    const AMDGPU               hardware(AMDGPU::Processor::gfx942, 304, "test-gfx942");
    std::array<uint8_t, 32768> workspace{};
    float                      a = 0.9f, b = 1.0f, c = 0.0f, d = 0.0f;
    std::array<float, 2>       scaleA = {3.0f, 1.5f}, scaleB = {0.5f, 2.0f};
    for(auto f8 : {rocisa::DataType::Float8,
                   rocisa::DataType::BFloat8,
                   rocisa::DataType::Float8_fnuz,
                   rocisa::DataType::BFloat8_fnuz})
        for(bool halfA : {true, false})
            for(auto mode : {"Scalar", "Vector"})
                for(int gsu : {2, 4})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << int(f8) << " HalfA=" << halfA << " " << mode << " GSU=" << gsu);
                    auto problem = makeProblem(f8, halfA);
                    problem.setUseScaleAB(mode);
                    problem.setGroupedGemm(true);
                    problem.setGroupedGemmCount(2);
                    std::vector<ContractionProblemGemm> problems(2, problem);
                    ContractionSolution::GroupedInputs  inputs;
                    inputs.ws = workspace.data();
                    for(size_t i = 0; i < problems.size(); ++i)
                    {
                        inputs.grouped.emplace_back(&a, &b, &c, &d, 1.0f, 0.0f);
                        inputs.grouped.back().ws     = workspace.data();
                        inputs.grouped.back().scaleA = &scaleA[i];
                        inputs.grouped.back().scaleB = &scaleB[i];
                    }
                    ContractionSolution solution;
                    configureSolution(solution, problem, gsu);
                    solution.problemType.groupedGemm = true;
                    KernelArguments mainArgs(true), conversionArgs(true);
                    solution.generateSingleCallGroupedGemm<true>(
                        problems, inputs, hardware, mainArgs);
                    solution.generateOutputConversionCallGroupedGemm<true>(
                        problems, inputs, hardware, conversionArgs);
                    for(size_t i = 0; i < problems.size(); ++i)
                    {
                        const std::string suffix = i == 0 ? "" : "_1";
                        expectPointer(mainArgs, "scaleA" + suffix, &scaleA[i]);
                        expectPointer(mainArgs, "scaleB" + suffix, &scaleB[i]);
                        const bool scalar = std::string(mode) == "Scalar";
                        expectPointer(conversionArgs,
                                      "scaleA" + suffix,
                                      scalar && halfA ? nullptr : &scaleA[i]);
                        expectPointer(conversionArgs,
                                      "scaleB" + suffix,
                                      scalar && !halfA ? nullptr : &scaleB[i]);
                    }
                }
}

TEST(Float8ConversionArguments, OtherNarrowingConversionsRetainBothScales)
{
    using DT = rocisa::DataType;
    const AMDGPU               hardware(AMDGPU::Processor::gfx942, 304, "test-gfx942");
    std::array<uint8_t, 32768> workspace{};
    float                      a = 0.9f, b = 1.0f, c = 0.0f, d = 0.0f, scaleA = 3.0f, scaleB = 0.5f;
    for(auto compute : {DT::Half, DT::BFloat16})
        for(int floatOperands : {1, 2, 3}) // A, B, or both stored in Float
            for(bool grouped : {false, true})
                for(int gsu : {2, 4})
                {
                    SCOPED_TRACE(::testing::Message()
                                 << int(compute) << " floatOperands=" << floatOperands
                                 << " grouped=" << grouped << " GSU=" << gsu);
                    auto problem = ContractionProblemGemm::GEMM_Strides(
                        false,
                        false,
                        floatOperands & 1 ? DT::Float : compute,
                        floatOperands & 2 ? DT::Float : compute,
                        compute,
                        compute,
                        16,
                        16,
                        256,
                        1,
                        16,
                        4096,
                        256,
                        4096,
                        16,
                        256,
                        16,
                        256,
                        0.0);
                    problem.setComputeInputTypeA(compute);
                    problem.setComputeInputTypeB(compute);
                    problem.setAlphaType(DT::Float);
                    problem.setBetaType(DT::Float);
                    problem.setHighPrecisionAccumulate(true);
                    problem.setUseScaleAB("Scalar");
                    problem.setScaleA(DT::Float, 1);
                    problem.setScaleB(DT::Float, 1);
                    problem.setParams().setUniformSummationOrder(0);
                    ContractionSolution solution;
                    configureSolution(solution, problem, gsu);
                    ContractionInputs inputs(&a, &b, &c, &d, 1.0f, 0.0f);
                    inputs.ws     = workspace.data();
                    inputs.scaleA = &scaleA;
                    inputs.scaleB = &scaleB;
                    if(!grouped)
                    {
                        const auto calls = solution.solve(problem, inputs, hardware);
                        ASSERT_EQ(calls.size(), 2);
                        for(auto const& call : calls)
                        {
                            expectPointer(call.args, "scaleA", &scaleA);
                            expectPointer(call.args, "scaleB", &scaleB);
                        }
                    }
                    else
                    {
                        problem.setGroupedGemm(true);
                        problem.setGroupedGemmCount(2);
                        solution.problemType.groupedGemm = true;
                        std::vector<ContractionProblemGemm> problems(2, problem);
                        ContractionSolution::GroupedInputs  groupedInputs;
                        groupedInputs.ws      = workspace.data();
                        groupedInputs.grouped = {inputs, inputs};
                        KernelArguments mainArgs(true), conversionArgs(true);
                        solution.generateSingleCallGroupedGemm<true>(
                            problems, groupedInputs, hardware, mainArgs);
                        solution.generateOutputConversionCallGroupedGemm<true>(
                            problems, groupedInputs, hardware, conversionArgs);
                        for(auto const* args : {&mainArgs, &conversionArgs})
                            for(auto const* suffix : {"", "_1"})
                            {
                                expectPointer(*args, "scaleA" + std::string(suffix), &scaleA);
                                expectPointer(*args, "scaleB" + std::string(suffix), &scaleB);
                            }
                    }
                }
}
