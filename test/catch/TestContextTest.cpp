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

#include "TestContext.hpp"

#include <rocRoller/AssemblyKernel.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("TestContext::EscapeKernelName escapes bad characters", "[meta-test][infrastructure]")
{
    CHECK(TestContext::EscapeKernelName("goodKernelName") == "goodKernelName");
    CHECK(TestContext::EscapeKernelName("good__kernel__name") == "good__kernel__name");

    CHECK(TestContext::EscapeKernelName("my bad kernel name") == "my_bad_kernel_name");

    CHECK(TestContext::EscapeKernelName("My Bad Kernel Name") == "My_Bad_Kernel_Name");

    CHECK(TestContext::EscapeKernelName("my bad kernel name: and more 'bad', (characters)")
          == "my_bad_kernel_name_and_more_bad_characters");

    CHECK(TestContext::EscapeKernelName("my bad kernel name: xnack+")
          == "my_bad_kernel_name_xnackp");
    CHECK(TestContext::EscapeKernelName("my bad kernel name: xnack-")
          == "my_bad_kernel_name_xnackm");

    CHECK(TestContext::EscapeKernelName("my _bad_ kernel name_") == "my__bad__kernel_name_");

    CHECK(TestContext::EscapeKernelName("") == "_");
}

TEST_CASE("TestContext::KernelName includes test name", "[meta-test][infrastructure]")
{
    rocRoller::GPUArchitectureTarget target
        = {rocRoller::GPUArchitectureGFX::GFX90A, {.xnack = true}};
    CHECK(TestContext::KernelName("my kernel", 5, target)
          == "TestContext_KernelName_includes_test_name_my_kernel_5_gfx90a_xnackp");
}

TEST_CASE("TestContext::ForTestDevice gives a usable GPU.", "[meta-test][infrastructure][gpu]")
{
    using namespace Catch::Matchers;
    auto context = TestContext::ForTestDevice();

    REQUIRE(context.get() != nullptr);
    auto kernelName = context->kernel()->kernelName();
    CHECK_THAT(kernelName, StartsWith("TestContext"));
    CHECK_THAT(kernelName, ContainsSubstring("usable"));

    CHECK(context->hipDeviceIndex() >= 0);
}

TEST_CASE("TestContext::ForTarget does not give a usable GPU.", "[meta-test][infrastructure]")
{
    using namespace Catch::Matchers;
    auto context = TestContext::ForTarget({rocRoller::GPUArchitectureGFX::GFX942});

    REQUIRE(context.get() != nullptr);
    auto kernelName = context->kernel()->kernelName();
    CHECK_THAT(kernelName, StartsWith("TestContext"));
    CHECK_THAT(kernelName, ContainsSubstring("does_not"));
    CHECK_THAT(kernelName, ContainsSubstring("gfx942"));

    CHECK(context->hipDeviceIndex() == -1);
}
