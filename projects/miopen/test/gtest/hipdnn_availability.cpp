// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the availability decision behind
// src/private/hipdnn_forward.hpp's IsAvailable(). The seam only exists when
// MIOPEN_ENABLE_HIPDNN_WRAPPER is ON, so this file compiles to zero tests
// otherwise. No hipDNN install and no GPU are needed.
//
// What cannot be tested in-process is the live probe: it loads the backend once
// and caches the answer for the life of the process, so a test cannot make the
// backend disappear and watch MIOpen react.

#include <gtest/gtest.h>

#ifdef MIOPEN_ENABLE_HIPDNN_WRAPPER

#include "../../src/private/hipdnn_forward.hpp"

#include <ostream>
#include <string>

namespace miopen {
namespace wrapper {
namespace hipdnn {
// Readable failure diagnostics; GoogleTest finds this via argument-dependent
// lookup.
static void PrintTo(BackendState state, std::ostream* os)
{
    *os << "BackendState::" << DescribeBackendState(state);
}
} // namespace hipdnn
} // namespace wrapper
} // namespace miopen

namespace {

using miopen::wrapper::hipdnn::BackendState;
using miopen::wrapper::hipdnn::ClassifyBackend;
using miopen::wrapper::hipdnn::DescribeBackendState;

// Stands in for HIPDNN_FRONTEND_VERSION_MAJOR. The real value is deliberately
// not used: the decision is about how two majors compare, and pulling in a
// hipDNN header here would put one on every test executable's include path.
constexpr int kBuildMajor = 7;

TEST(CPU_HipdnnAvailability_NONE, MissingBackendIsUnavailable)
{
    // One case, not two: a backend the frontend refused reports a negative major
    // exactly the way one that never loaded does.
    EXPECT_EQ(ClassifyBackend(-1, kBuildMajor), BackendState::Missing);
}

TEST(CPU_HipdnnAvailability_NONE, UnexpectedMajorVersionIsUnavailable)
{
    EXPECT_EQ(ClassifyBackend(kBuildMajor + 1, kBuildMajor), BackendState::MajorVersionMismatch);
    EXPECT_EQ(ClassifyBackend(kBuildMajor - 1, kBuildMajor), BackendState::MajorVersionMismatch);
    EXPECT_EQ(ClassifyBackend(0, kBuildMajor), BackendState::MajorVersionMismatch);
}

TEST(CPU_HipdnnAvailability_NONE, MatchingMajorIsUsable)
{
    EXPECT_EQ(ClassifyBackend(kBuildMajor, kBuildMajor), BackendState::Usable);
    EXPECT_EQ(ClassifyBackend(0, 0), BackendState::Usable);
}

TEST(CPU_HipdnnAvailability_NONE, EveryStateHasItsOwnExplanation)
{
    const BackendState states[] = {BackendState::Usable,
                                   BackendState::Missing,
                                   BackendState::MajorVersionMismatch,
                                   BackendState::HandleCreationFailed};

    for(const BackendState state : states)
    {
        const std::string text = DescribeBackendState(state);
        EXPECT_FALSE(text.empty());
        for(const BackendState other : states)
        {
            if(other != state)
                EXPECT_NE(text, DescribeBackendState(other));
        }
    }
}

} // namespace

#endif // MIOPEN_ENABLE_HIPDNN_WRAPPER
