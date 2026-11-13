// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test hooks exported from libhipblaslt. Not part of any supported interface.
// The library and the tests both include this file, so a signature change
// fails to compile instead of failing to link.

#pragma once

#include <hipblaslt/hipblaslt-export.h>

extern "C" {

// Clears the search type recorded by the last getAllSolutions call.
HIPBLASLT_EXPORT void rocblaslt_testing_reset_last_get_all_search_type_for_test();

// The SolutionLibrarySearchType the last getAllSolutions call received, as an
// int, or -1 if none was made since the last reset.
HIPBLASLT_EXPORT int rocblaslt_testing_last_get_all_search_type_for_test();
}
