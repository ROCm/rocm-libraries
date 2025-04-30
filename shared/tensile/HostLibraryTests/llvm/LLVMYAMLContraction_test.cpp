/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Serialization.hpp>
#include <Tensile/llvm/YAML.hpp>

using namespace Tensile;

TEST(LLVMYAMLContractionTest, Simple)
{
    std::string mydoc = "name: foo\n"
                        "sizeMapping:\n"
                        "  globalAccumulation: 0\n"
                        "  workspaceSizePerElemC: 0\n"
                        "  workGroup: [1,2,3]\n"
                        "  macroTile: [2,4,6]\n"
                        "  threadTile: [2,2,2]\n"
                        "  depthU: 8\n"
                        "  globalSplitU: 1\n"
                        "  staggerStrideShift: 3\n"
                        "  staggerU: 32\n"
                        "  workGroupMapping: 8\n"
                        "  sourceKernel: false\n"
                        "  persistentKernel: 0\n"
                        "  persistentKernelAlongBatch: false\n"
                        "index: 0\n"
                        "hardwarePredicate: { type: TruePred }\n"
                        "problemPredicate:  { type: TruePred }\n"
                        "taskPredicate:  { type: TruePred }\n"
                        "debugKernel: false\n"
                        "problemType:\n"
                        "  operationIdentifier: foo\n"
                        "  highPrecisionAccumulate: false\n"
                        "  useBeta: true\n"
                        "  aType: Float\n"
                        "  bType: Float\n"
                        "  cType: Float\n"
                        "  dType: Float\n";
    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    ContractionSolution s;

    yin >> s;
    ASSERT_FALSE(yin.error());

    EXPECT_EQ(s.name(), "foo");
    EXPECT_EQ(s.sizeMapping.workGroupSize, Tensile::dim3(1, 2, 3));
    EXPECT_EQ(s.sizeMapping.macroTile, Tensile::dim3(2, 4, 6));
}

TEST(LLVMYAMLContractionTest, Predicate)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: Not, value: {type: FalsePred}},\n"
                        "        {type: Free0SizeMultiple, index: 0, value: 2}]";
    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());

    ContractionProblem prob = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    EXPECT_NE(p, nullptr);

    EXPECT_EQ((*p)(prob), true);

    llvm::yaml::Output yout(llvm::outs());
    yout << p;
}

TEST(LLVMYAMLContractionTest, ContractionLibrary)
{
    std::string mydoc = "solutions:\n"
                        "  - name: foo\n"
                        "    sizeMapping:\n"
                        "      globalAccumulation: 0\n"
                        "      workspaceSizePerElemC: 0\n"
                        "      workGroup: [1,2,3]\n"
                        "      macroTile: [1,2,3]\n"
                        "      threadTile: [1,2,3]\n"
                        "      depthU: 8\n"
                        "      globalSplitU: 1\n"
                        "      staggerStrideShift: 3\n"
                        "      staggerU: 32\n"
                        "      workGroupMapping: 8\n"
                        "      sourceKernel: false\n"
                        "      persistentKernel: 0\n"
                        "      persistentKernelAlongBatch: false\n"
                        "    index: 0\n"
                        "    hardwarePredicate: { type: TruePred }\n"
                        "    problemPredicate:  { type: TruePred }\n"
                        "    taskPredicate:  { type: TruePred }\n"
                        "    debugKernel: false\n"
                        "    problemType:\n"
                        "      operationIdentifier: foo\n"
                        "      highPrecisionAccumulate: false\n"
                        "      useBeta: true\n"
                        "      aType: Float\n"
                        "      bType: Float\n"
                        "      cType: Float\n"
                        "      dType: Float\n"
                        "library:\n"
                        "  type: Hardware\n"
                        "  rows:\n"
                        "      - predicate: { type: AMDGPU, value: { type: "
                        "Processor, value: gfx900 } }\n"
                        "        library:\n"
                        "          type: Problem\n"
                        "          rows:\n"
                        "              - predicate: { type: Free0SizeMultiple, "
                        "index: 0, value: 2 }\n"
                        "                library: { type: Single, index: 0 }\n"
                        "";

    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    MasterContractionLibrary l;

    yin >> l;

    ASSERT_FALSE(yin.error());

    llvm::yaml::Output yout(llvm::outs());
    yout << l;
}
