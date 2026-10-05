// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/version.hpp>

int main() {
    return roc::host_numerics::version() == "0.1.0" ? 0 : 1;
}
