// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/hstu_problem.hpp"
#include "ck_tile/dispatcher/hstu_registry.hpp"

#include <string>

namespace ck_tile {
namespace dispatcher {

class HstuDispatcher
{
    public:
    explicit HstuDispatcher(HstuRegistry* registry = nullptr);

    void set_benchmarking(bool enabled) { benchmarking_ = enabled; }
    void set_timing(int cold_niters, int nrepeat)
    {
        cold_niters_ = cold_niters;
        nrepeat_     = nrepeat;
    }

    [[nodiscard]] std::vector<HstuKernelEntry> list_kernels() const;

    private:
    HstuRegistry* registry_ = nullptr;
    bool benchmarking_      = false;
    int cold_niters_        = 1;
    int nrepeat_            = 3;
};

} // namespace dispatcher
} // namespace ck_tile
