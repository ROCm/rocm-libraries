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

#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Utilities/Error.hpp>

TEST_CASE("HipSuccessMatcher works", "[meta-test][gpu]")
{
    std::vector<int> cpuValue = {5};

    auto gpuValue = make_shared_device(cpuValue);

    int copiedValue;

    REQUIRE_THAT(hipMemcpy(&copiedValue, gpuValue.get(), sizeof(int), hipMemcpyDefault),
                 HasHipSuccess());

    auto matcher = HasHipSuccess();

    CHECK(matcher.match(hipSuccess));
    CHECK_FALSE(matcher.match(hipErrorInvalidValue));

    CHECK_THAT(matcher.describe(), Catch::Matchers::ContainsSubstring("Hip error"));
}

TEST_CASE("HipSuccessMatcher REQUIRE", "[meta-test][!shouldfail]")
{
    hipError_t failure = hipErrorInvalidValue;
    REQUIRE_THAT(failure, HasHipSuccess());
}

TEST_CASE("HipSuccessMatcher CHECK", "[meta-test][!shouldfail]")
{
    hipError_t failure = hipErrorInvalidValue;
    CHECK_THAT(failure, HasHipSuccess());
}

TEST_CASE("DeviceScalarMatcher works", "[meta-test][gpu]")
{
    SECTION("Int")
    {
        std::vector<int> cpuValue = {5};

        auto gpuValue = make_shared_device(cpuValue);

        CHECK_THAT(gpuValue, HasDeviceScalarEqualTo(5));

        auto differentValueChecker = HasDeviceScalarEqualTo(6);

        SECTION("Failure")
        {
            CHECK_FALSE(differentValueChecker.match(gpuValue));

            CHECK_THAT(differentValueChecker.describe(),
                       Catch::Matchers::ContainsSubstring("Device value: 5")
                           && Catch::Matchers::ContainsSubstring("Expected Value: 6"));
        }

        SECTION("Success")
        {
            cpuValue[0] = 6;

            REQUIRE_THAT(hipMemcpy(gpuValue.get(), cpuValue.data(), sizeof(int), hipMemcpyDefault),
                         HasHipSuccess());

            CHECK(differentValueChecker.match(gpuValue));
        }
    }

    SECTION("Float")
    {
        std::vector<float> cpuValue = {5.25f};

        auto gpuValue = make_shared_device(cpuValue);

        CHECK_THAT(gpuValue, HasDeviceScalarEqualTo(5.25f));
    }
}

TEST_CASE("CustomDeviceScalarMatcher works", "[meta-test][gpu]")
{
    std::vector<float> cpuValue = {5.0f};

    auto gpuValue = make_shared_device(cpuValue);

    auto expectedValue = cpuValue[0] + std::numeric_limits<float>::epsilon() * 10;

    CHECK_THAT(gpuValue, HasDeviceScalar(Catch::Matchers::WithinULP(expectedValue, 10)));

    auto differentValueChecker = HasDeviceScalar(Catch::Matchers::WithinULP(6.0f, 0));

    SECTION("Failure")
    {
        CHECK_FALSE(differentValueChecker.match(gpuValue));

        CHECK_THAT(differentValueChecker.describe(),
                   Catch::Matchers::ContainsSubstring("Device value: 5")
                       && Catch::Matchers::ContainsSubstring("ULPs of"));
    }

    SECTION("Success")
    {
        cpuValue[0] = 6;

        REQUIRE_THAT(hipMemcpy(gpuValue.get(), cpuValue.data(), sizeof(int), hipMemcpyDefault),
                     HasHipSuccess());

        CHECK(differentValueChecker.match(gpuValue));
    }
}
