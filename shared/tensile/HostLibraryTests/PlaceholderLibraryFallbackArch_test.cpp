/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>

#include <string>

using namespace Tensile;

namespace
{
    using Placeholder = PlaceholderLibrary<ContractionProblem>;

    const std::string fallbackPrefix
        = "TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_fallback";

    ContractionSolution makeSolution(bool sourceKernel)
    {
        ContractionSolution solution;
        solution.sizeMapping.sourceKernel = sourceKernel;
        return solution;
    }

    std::string codeObject(std::string const& prefix, bool sourceKernel, AMDGPU const& hardware)
    {
        Placeholder lib;
        lib.filePrefix = prefix;
        return lib.getCodeObjectFileName(hardware, makeSolution(sourceKernel));
    }
}

// A generic-family library (e.g. gfx9-generic) used on a member GPU (e.g. gfx90c):
// the producer already suffixed the fallback prefix with the generic target, which
// differs from the hardware arch. The hardware arch must not be appended a second time.
TEST(PlaceholderLibraryFallbackArch, GenericSuffixIsNotExtended)
{
    AMDGPU gfx90c(AMDGPU::Processor::gfx90c, 7, 1, "test");

    EXPECT_EQ(codeObject(fallbackPrefix + "_gfx9-generic", true, gfx90c),
              fallbackPrefix + "_gfx9-generic.hsaco");
    EXPECT_EQ(codeObject(fallbackPrefix + "_gfx10-3-generic", true, gfx90c),
              fallbackPrefix + "_gfx10-3-generic.hsaco");
}

// Unchanged behaviour: unsuffixed fallback prefixes get the hardware arch, prefixes
// already suffixed with the hardware arch are left alone, assembly kernels use .co.
TEST(PlaceholderLibraryFallbackArch, ExistingBehaviourUnchanged)
{
    AMDGPU gfx90c(AMDGPU::Processor::gfx90c, 7, 1, "test");
    AMDGPU gfx942(AMDGPU::Processor::gfx942, 304, 0, "test");

    EXPECT_EQ(codeObject(fallbackPrefix, true, gfx90c), fallbackPrefix + "_gfx90c.hsaco");
    EXPECT_EQ(codeObject(fallbackPrefix, true, gfx942), fallbackPrefix + "_gfx942.hsaco");
    EXPECT_EQ(codeObject(fallbackPrefix + "_gfx90c", true, gfx90c),
              fallbackPrefix + "_gfx90c.hsaco");

    const std::string asmPrefix = "TensileLibrary_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx90c";
    EXPECT_EQ(codeObject(asmPrefix, false, gfx90c), asmPrefix + ".co");
}
