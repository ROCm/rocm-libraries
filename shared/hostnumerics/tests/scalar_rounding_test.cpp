// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cfenv>
#include <cstdint>
#include <roc/hostnumerics/scalar.hpp>

namespace {
using namespace roc::hostnumerics;

class RoundingModeRestore {
   public:
    RoundingModeRestore() : mode(std::fegetround()) {}
    ~RoundingModeRestore() {
        std::fesetround(mode);
    }

   private:
    int mode;
};

TEST(ScalarRounding, NearestEvenAndTowardZeroIgnoreHostRoundingMode) {
    const RoundingModeRestore restore;
    struct Case {
        double value;
        int32_t nearest;
        int32_t truncated;
    };
    const std::array<Case, 12> cases{{
        {0.5, 0, 0},
        {1.5, 2, 1},
        {2.5, 2, 2},
        {3.5, 4, 3},
        {-0.5, 0, 0},
        {-1.5, -2, -1},
        {-2.5, -2, -2},
        {-3.5, -4, -3},
        {2.25, 2, 2},
        {2.75, 3, 2},
        {-2.25, -2, -2},
        {-2.75, -3, -2},
    }};
    const ScalarConversionOptions nearest{IntegerRounding::NearestEven, IntegerOverflow::Reject};
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        for (const auto& test : cases) {
            // A runtime load prevents constant folding under the default rounding mode.
            volatile double value = test.value;
            EXPECT_EQ(convertScalar<int32_t>(value, nearest), test.nearest);
            EXPECT_EQ(convertScalar<int32_t>(value, {}), test.truncated);
        }
        volatile double positiveTie = 127.5;
        volatile double negativeTie = -128.5;
        EXPECT_FALSE(getConvertScalar<int8_t>(positiveTie, nearest));
        EXPECT_EQ(convertScalar<int8_t>(negativeTie, nearest), -128);
        EXPECT_EQ(convertScalar<int8_t>(positiveTie,
                                        {IntegerRounding::NearestEven, IntegerOverflow::Saturate}),
                  127);
        EXPECT_EQ(convertScalar<int8_t>(
                      positiveTie, {IntegerRounding::NearestEven, IntegerOverflow::ModuloWrap}),
                  -128);
    }
}
}  // namespace
