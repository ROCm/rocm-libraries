// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <limits>
#include <roc/hostnumerics/scalar_type.hpp>
#include <roc/hostnumerics/version.hpp>

int main(int argc, char**) {
    // Intentional CI demonstration: signed overflow must fail under UBSan.
    volatile int count = std::numeric_limits<int>::max();
    volatile int nextCount = count + argc;
    (void)nextCount;

    using namespace roc::hostnumerics;
    return version() == "0.1.0" && scalarTypeInfo(ScalarType::Float16).storageBits == 16 ? 0 : 1;
}
