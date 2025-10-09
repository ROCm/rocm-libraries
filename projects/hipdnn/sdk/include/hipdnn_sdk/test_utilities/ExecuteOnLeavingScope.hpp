// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace hipdnn_sdk::test_utilities
{
template <class F>
class ExecuteOnLeavingScope
{
    F _func;

public:
    ExecuteOnLeavingScope(F func)
        : _func(std::move(func))
    {
    }
    ExecuteOnLeavingScope(const ExecuteOnLeavingScope&) = delete;
    ExecuteOnLeavingScope(ExecuteOnLeavingScope&&) = delete;
    ExecuteOnLeavingScope& operator=(const ExecuteOnLeavingScope&) = delete;
    ExecuteOnLeavingScope& operator=(ExecuteOnLeavingScope&&) = delete;

    ~ExecuteOnLeavingScope()
    {
        _func();
    }
};
}
