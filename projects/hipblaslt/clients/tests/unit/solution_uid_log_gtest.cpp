// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the SolutionUID text that hipblaslt-bench prints
// after each result: format_solution_uid and log_solution_identity in
// argument_model.hpp. See README.md in this directory for the dual-target
// build and the HostUnit/smoke_ naming.

#include "argument_model.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace
{
    // Expected strings match tensilelite.SolutionIdGen.encode_solution_uid.
    TEST(HostUnitSolutionUidLog, smoke_FormatsCanonicalBase62)
    {
        EXPECT_EQ(ArgumentLogging::format_solution_uid(0), "0u0");
        EXPECT_EQ(ArgumentLogging::format_solution_uid(61), "0uz");
        EXPECT_EQ(ArgumentLogging::format_solution_uid(62), "0u10");
        EXPECT_EQ(ArgumentLogging::format_solution_uid(200), "0u3E");
        EXPECT_EQ(ArgumentLogging::format_solution_uid(std::numeric_limits<uint64_t>::max()),
                  "0uLygHa16AHYF");
    }

    TEST(HostUnitSolutionUidLog, smoke_SolutionLinesIncludeUid)
    {
        std::ostringstream out;
        ArgumentLogging::log_solution_identity(out, 5, 200, "solution", "kernel");
        EXPECT_EQ(out.str(),
                  "    --Solution index: 5\n"
                  "    --Solution UID:   0u3E\n"
                  "    --Solution name:  solution\n"
                  "    --kernel name:    kernel\n");
    }

    TEST(HostUnitSolutionUidLog, smoke_SolutionLinesOmitZeroUid)
    {
        std::ostringstream out;
        ArgumentLogging::log_solution_identity(out, 5, 0, "solution", "kernel");
        EXPECT_EQ(out.str(),
                  "    --Solution index: 5\n"
                  "    --Solution name:  solution\n"
                  "    --kernel name:    kernel\n");
    }
} // namespace
