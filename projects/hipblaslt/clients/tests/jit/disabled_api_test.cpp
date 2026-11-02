// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "test_helpers.hpp"
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>

// Every installed header lives in the public include directory, so a JIT header
// found there could be installed.
#if __has_include(<hipblaslt/hipblaslt-jit.hpp>)
#error "JIT headers must not be part of the public hipBLASLt include tree"
#endif

TEST_CASE("public headers exclude JIT and the extension API links without it", "[jit-cpu]")
{
    hipblaslt_ext::GemmPreference preference;
    preference.setMaxWorkspaceBytes(4096);
    hipblasLtMatmulAlgo_t algo{};
    {
        INFO(("The extension API did not link without JIT"));
        REQUIRE((preference.getMaxWorkspaceBytes() == 4096
            && hipblaslt_ext::getIndexFromAlgo(algo) == 0));
    }
    std::cout << "PASS public headers exclude JIT and the extension API links without it\n";
}
