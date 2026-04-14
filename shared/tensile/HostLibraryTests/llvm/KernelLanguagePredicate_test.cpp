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

TEST(KernelLanguagePredicateTest, Any)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: KernelLanguageCompatible, value: Any}]";

    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    ContractionProblem prob = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    prob.setKernelLanguage(KernelLanguage::Any);
    EXPECT_EQ((*p)(prob), true);

    prob.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_EQ((*p)(prob), false);

    prob.setKernelLanguage(KernelLanguage::Source);
    EXPECT_EQ((*p)(prob), false);
}

TEST(KernelLanguagePredicateTest, Asm)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: KernelLanguageCompatible, value: Assembly}]";

    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    ContractionProblem prob = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    prob.setKernelLanguage(KernelLanguage::Any);
    EXPECT_EQ((*p)(prob), true);

    prob.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_EQ((*p)(prob), true);

    prob.setKernelLanguage(KernelLanguage::Source);
    EXPECT_EQ((*p)(prob), false);
}

TEST(KernelLanguagePredicateTest, Source)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: KernelLanguageCompatible, value: Source}]";

    LibraryIOContext<ContractionSolution> context{std::string(""), {}, nullptr};
    llvm::yaml::Input                     yin(mydoc, &context);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    ContractionProblem prob = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    prob.setKernelLanguage(KernelLanguage::Any);
    EXPECT_EQ((*p)(prob), true);

    prob.setKernelLanguage(KernelLanguage::Assembly);
    EXPECT_EQ((*p)(prob), false);

    prob.setKernelLanguage(KernelLanguage::Source);
    EXPECT_EQ((*p)(prob), true);
}
