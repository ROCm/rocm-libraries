// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

struct MatmulResult
{
    bool                completed = false;
    std::optional<bool> norm_passed;

    void record_norm(bool passed)
    {
        norm_passed = norm_passed.value_or(true) && passed;
    }

    const char* numerical_status() const
    {
        return norm_passed ? (*norm_passed ? "PASSED" : "FAILED") : "NOT_CHECKED";
    }

    const char* status() const
    {
        return completed ? numerical_status() : "INCOMPLETE";
    }

    bool failed() const
    {
        return !completed || (norm_passed && !*norm_passed);
    }
};
