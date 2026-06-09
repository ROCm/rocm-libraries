// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/hstu_dispatcher.hpp"

namespace ck_tile {
namespace dispatcher {

HstuDispatcher::HstuDispatcher(HstuRegistry* registry)
    : registry_(registry ? registry : &HstuRegistry::instance())
{
}

std::vector<HstuKernelEntry> HstuDispatcher::list_kernels() const
{
    return registry_->get_all();
}

} // namespace dispatcher
} // namespace ck_tile
