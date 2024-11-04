#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Utilities/Error.hpp>

TEST_CASE("CommandArgumentValue toString", "[command][toString]")
{
    using cav = rocRoller::CommandArgumentValue;

    SECTION("Values")
    {
        CHECK(toString(cav(5)) == "5");
        CHECK(toString(cav(-5ll)) == "-5");

        CHECK(toString(cav(3643653242342ull)) == "3643653242342");

        CHECK(toString(cav(4.5f)) == "4.50000");
        CHECK(toString(cav(4.5)) == "4.50000");

        CHECK(toString(cav(rocRoller::FP8(4.5))) == "4.50000");
    }

    SECTION("Pointers")
    {
        float x      = 5.f;
        void* ptr    = &x;
        auto  ptrStr = rocRoller::concatenate(ptr);

        CHECK(toString(cav(reinterpret_cast<int32_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<int64_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint8_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint32_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<uint64_t*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<float*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<double*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::Half*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::BFloat16*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::FP8*>(ptr))) == ptrStr);
        CHECK(toString(cav(reinterpret_cast<rocRoller::BF8*>(ptr))) == ptrStr);
    }
}
