

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
    rocRoller::GPUArchitectureTarget target("gfx90a:xnack+");
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
    auto context = TestContext::ForTarget("gfx942");

    REQUIRE(context.get() != nullptr);
    auto kernelName = context->kernel()->kernelName();
    CHECK_THAT(kernelName, StartsWith("TestContext"));
    CHECK_THAT(kernelName, ContainsSubstring("does_not"));
    CHECK_THAT(kernelName, ContainsSubstring("gfx942"));

    CHECK(context->hipDeviceIndex() == -1);
}
