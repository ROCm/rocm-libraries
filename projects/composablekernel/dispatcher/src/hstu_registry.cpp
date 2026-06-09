// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/hstu_registry.hpp"

namespace ck_tile {
namespace dispatcher {

bool HstuRegistry::register_kernel(const HstuKernelKey& key,
                                   std::function<float(void* stream)> run_fn)
{
    auto entry = std::make_shared<HstuKernelEntry>(HstuKernelEntry{key, std::move(run_fn)});
    return Base::register_kernel(key.encode_identifier(), std::move(entry));
}

std::vector<HstuKernelEntry> HstuRegistry::get_all() const
{
    std::vector<HstuKernelEntry> out;
    for(const auto& ptr : get_all_instances())
    {
        if(ptr)
            out.push_back(*ptr);
    }
    return out;
}

HstuRegistry& HstuRegistry::instance()
{
    static HstuRegistry reg;
    return reg;
}

} // namespace dispatcher
} // namespace ck_tile
