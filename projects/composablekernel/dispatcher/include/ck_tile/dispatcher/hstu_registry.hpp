// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/hstu_kernel_key.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {

struct HstuKernelEntry
{
    HstuKernelKey key;
    std::function<float(void* stream)> run;
};

class HstuRegistry : public BaseRegistry<HstuRegistry, std::string, HstuKernelEntry>
{
    using Base = BaseRegistry<HstuRegistry, std::string, HstuKernelEntry>;

    public:
    bool register_kernel(const HstuKernelKey& key, std::function<float(void* stream)> run_fn);

    [[nodiscard]] std::vector<HstuKernelEntry> get_all() const;
    static HstuRegistry& instance();
};

} // namespace dispatcher
} // namespace ck_tile
