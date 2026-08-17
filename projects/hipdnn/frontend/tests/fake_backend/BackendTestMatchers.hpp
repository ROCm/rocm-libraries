// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gmock/gmock.h>

#include <cstring>
#include <string>
#include <vector>

namespace hipdnn_frontend::test
{

// Matcher: checks that a void* points to a value equal to `expected`
template <typename T>
auto pointsToScalar(T expected)
{
    return ::testing::Truly([expected](const void* ptr) {
        T actual;
        std::memcpy(&actual, ptr, sizeof(T));
        return actual == expected;
    });
}

// Matcher: checks that a void* points to a contiguous array matching `expected`
template <typename T>
auto pointsToVector(const std::vector<T>& expected)
{
    return ::testing::Truly([expected](const void* ptr) {
        std::vector<T> actual(expected.size());
        std::memcpy(actual.data(), ptr, expected.size() * sizeof(T));
        return actual == expected;
    });
}

// Matcher: checks that a void* points to a char array matching `expected`
inline auto pointsToString(const std::string& expected)
{
    return ::testing::Truly([expected](const void* ptr) {
        auto actual = std::string(static_cast<const char*>(ptr), expected.size());
        return actual == expected;
    });
}

} // namespace hipdnn_frontend::test
