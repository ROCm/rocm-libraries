// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile::direct_conv 
{

template <typename... Args>
CK_TILE_HOST void LogInfo(Args&&... args) noexcept
{
    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
    {
        CK_TILE_INFO(std::forward<Args>(args)...);
    }
}

}
