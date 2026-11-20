// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <optional>

namespace miopen::test {

// Entry point for the child processes of a multi-process test. A test that starts copies of
// miopen_gtest as workers registers one at static-initialization time; main() calls it before
// InitGoogleTest(). It returns the exit code when argv is a worker's command line, otherwise
// nullopt.
//
// Workers must not go through gtest: InitGoogleTest() evaluates every test-case generator, and
// several of them initialize the GPU to pick cases. Many workers doing that at once hangs or runs
// out of memory on small APUs.
using ChildProcessMain = std::optional<int> (*)(int argc, char** argv);

inline ChildProcessMain& child_process_main()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static ChildProcessMain entry = nullptr;
    return entry;
}

} // namespace miopen::test
