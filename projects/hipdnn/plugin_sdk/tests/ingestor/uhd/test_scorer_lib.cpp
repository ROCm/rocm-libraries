// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file test_scorer_lib.cpp
 * @brief Test scorer library for CustomLibraryAdapter tests.
 *
 * Compiled into a .so/.dll that CustomLibraryAdapter loads via dlopen.
 * Provides several C ABI scorer functions with different behaviors.
 */

#include <cstddef>

// Explicit symbol visibility for dlopen/dlsym
#if defined(_WIN32) || defined(__CYGWIN__)
#define SCORER_EXPORT __declspec(dllexport)
#else
#define SCORER_EXPORT __attribute__((visibility("default")))
#endif

extern "C"
{

/// Simple linear scorer: sum all features.
SCORER_EXPORT double test_linear_scorer(const double* features, size_t num_features)
{
    double sum = 0.0;
    for(size_t i = 0; i < num_features; ++i)
    {
        sum += features[i];
    }
    return sum;
}

/// Constant scorer: always returns 42.0.
SCORER_EXPORT double test_constant_scorer(const double* /*features*/, size_t /*num_features*/)
{
    return 42.0;
}

/// Feature product scorer: multiply first two features.
SCORER_EXPORT double test_product_scorer(const double* features, size_t num_features)
{
    if(num_features < 2)
    {
        return 0.0;
    }
    return features[0] * features[1];
}

} // extern "C"
