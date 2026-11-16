// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../../library/src/amd_detail/rocblaslt/src/include/candidate_preparation.hpp"

#include <gtest/gtest.h>

// The launch catch maps RocRollerStreamKDataParallelError through
// statusForRocRollerStreamKDataParallel. rocblaslt_status_invalid_value is the
// code RocBlasLtStatusToHIPStatus turns into HIPBLAS_STATUS_INVALID_VALUE.
// That converter is not exported from libhipblaslt, so this host binary checks
// the rocblaslt status the catch assigns.
TEST(HostUnitRocRollerStreamKRefusal, smoke_StatusIsInvalidValue)
{
    TensileLite::RocRollerStreamKDataParallelError const refusal(
        "rocRoller Stream-K kernel probe cannot run data-parallel tiles");
    EXPECT_EQ(rocblaslt::statusForRocRollerStreamKDataParallel(refusal),
              rocblaslt_status_invalid_value);
}
