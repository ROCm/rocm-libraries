// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testNearestEvenIgnoresHostRoundingMode() {
    const RoundingModeRestore restore;
    const ScalarConversionOptions nearestEvenReject{
        IntegerRounding::NearestEven,
        IntegerOverflow::Reject,
    };
    const ScalarConversionOptions nearestEvenSaturate{
        IntegerRounding::NearestEven,
        IntegerOverflow::Saturate,
    };
    const ScalarConversionOptions nearestEvenWrap{
        IntegerRounding::NearestEven,
        IntegerOverflow::ModuloWrap,
    };

    struct Case {
        double value;
        int32_t expected;
    };
    const std::array<Case, 8> cases{{
        {0.5, 0},
        {1.5, 2},
        {2.5, 2},
        {3.5, 4},
        {-0.5, 0},
        {-1.5, -2},
        {-2.5, -2},
        {-3.5, -4},
    }};
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        require(std::fesetround(mode) == 0, "Host did not accept a standard rounding mode.");
        for (const auto& test : cases) {
            volatile double value = test.value;
            require(convertScalar<int32_t>(value, nearestEvenReject) == test.expected,
                    "Nearest-even conversion depended on the host rounding mode.");
        }

        volatile double positiveBoundaryTie = 127.5;
        volatile double negativeBoundaryTie = -128.5;
        require(convertScalar<int8_t>(positiveBoundaryTie, nearestEvenSaturate) == 127 &&
                    convertScalar<int8_t>(positiveBoundaryTie, nearestEvenWrap) == -128 &&
                    convertScalar<int8_t>(negativeBoundaryTie, nearestEvenReject) == -128,
                "Boundary tie conversion depended on the host rounding mode.");
    }
}

}  // namespace scalar_codec_test
