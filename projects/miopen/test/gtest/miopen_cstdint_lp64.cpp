// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

// Regression test for TheRock#6258 / LP64 typedef conflict in miopen_cstdint.hpp.
//
// On LP64 platforms (64-bit Linux, macOS), <stdint.h> defines uint64_t as
// 'unsigned long', but __hip_internal::uint64_t is 'unsigned long long'.
// These are distinct C++ types despite being the same width, so a typedef
// redefinition is a hard error when both are visible during JIT compilation.
//
// This test exercises the real comgr JIT path via Handle::AddKernel with an
// inline kernel that includes miopen_cstdint.hpp alongside system headers.

#include <gtest/gtest.h>

#include <miopen/handle.hpp>
#include <miopen/manage_ptr.hpp>

#include "get_handle.hpp"

static std::string CstdintKernelSource()
{
    // Deliberately include <stdint.h> before miopen_cstdint.hpp to reproduce
    // the original conflict: system uint64_t (unsigned long) vs the MIOpen
    // typedef (previously unsigned long long via __hip_internal).
    return "#ifndef MIOPEN_HIP_RUNTIME_COMPILE\n"
           "#include <hip/hip_runtime.h>\n"
           "#endif\n"
           "#include <stdint.h>\n"
           "#include \"miopen_cstdint.hpp\"\n"
           "extern \"C\" {\n"
           "__global__ void cstdint_write(uint64_t* data) {\n"
           "    if(threadIdx.x == 0 && blockIdx.x == 0)\n"
           "        data[0] = (uint64_t)42;\n"
           "}\n"
           "}\n";
}

TEST(CPU_MiopenCstdintLP64_NONE, JitCompileWithMiopenCstdint)
{
    auto&& h = get_handle();

    std::vector<uint64_t> data_in(1, 0);
    auto data_dev = h.Write(data_in);

    h.AddKernel("NoAlgo",
                "",
                "cstdint_test.cpp",
                "cstdint_write",
                {1, 1, 1},
                {1, 1, 1},
                "",
                0,
                CstdintKernelSource())(data_dev.get());

    auto data_out = h.Read<uint64_t>(data_dev, 1);
    EXPECT_EQ(data_out[0], 42u);
}
