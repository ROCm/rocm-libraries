// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/hostnumerics/version.hpp>

int main() {
    return roc::hostnumerics::version() == "0.1.0" ? 0 : 1;
}
