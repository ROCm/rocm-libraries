// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * @brief Address of the Debug singleton as seen from inside tensilelite-host.
     *
     * Defined in src/Debug.cpp and deliberately kept out of the installed
     * headers: it exists only so a separate link unit can compare this address
     * against its own Debug::Instance().
     */
    TENSILELITEHOST_EXPORT const void* debugInstanceAddress();
} // namespace TensileLite
