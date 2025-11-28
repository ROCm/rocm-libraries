// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace hipdnn_sdk::test_utilities
{
template <class F>
class ScopedExecute
{
    F _func;

public:
    ScopedExecute(F func)
        : _func(std::move(func))
    {
    }
    ScopedExecute(const ScopedExecute&) = delete;
    ScopedExecute(ScopedExecute&&) = delete;
    ScopedExecute& operator=(const ScopedExecute&) = delete;
    ScopedExecute& operator=(ScopedExecute&&) = delete;

    ~ScopedExecute()
    {
        _func();
    }
};
}
