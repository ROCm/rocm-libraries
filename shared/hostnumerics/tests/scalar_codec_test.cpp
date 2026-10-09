// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

int main() {
    using namespace scalar_codec_test;
    testScalarTypeInfoContract();
    testIntegerConversionPrimitives();
    testNearestEvenIgnoresHostRoundingMode();
    testIntegerCodecPolicies();
    testScalarEncodings();
    return 0;
}
