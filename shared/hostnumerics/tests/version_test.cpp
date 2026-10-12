// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/hostnumerics/scalar_type.hpp>
#include <roc/hostnumerics/version.hpp>

int main(int argc, char**) {
    // Intentional CI demonstration: the runtime tests do not reach this branch.
    if (argc > 1) {
        const int* value = nullptr;
        return *value;
    }

    using namespace roc::hostnumerics;
    return version() == "0.1.0" && scalarTypeInfo(ScalarType::Float16).storageBits == 16 ? 0 : 1;
}
