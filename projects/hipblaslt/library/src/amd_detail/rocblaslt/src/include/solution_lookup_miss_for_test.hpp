// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test hooks exported from libhipblaslt. Not part of any supported interface.
// The library and the tests both include this file, so a signature change
// fails to compile instead of failing to link.

#pragma once

#include <hipblaslt/hipblaslt-export.h>

extern "C" {

// Which Tensile solution lookups report no solutions, so a test can reach the
// no-solution paths on any architecture.
enum rocblaslt_testing_solution_lookup_miss
{
    rocblaslt_testing_solution_lookup_miss_none = 0,
    // getBestSolutions and getAllSolutions both return no solutions.
    rocblaslt_testing_solution_lookup_miss_all = 1,
};

// Sets the lookups that miss, for every handle, until set back to _none.
HIPBLASLT_EXPORT void
    rocblaslt_testing_set_solution_lookup_miss_for_test(rocblaslt_testing_solution_lookup_miss miss);
}
