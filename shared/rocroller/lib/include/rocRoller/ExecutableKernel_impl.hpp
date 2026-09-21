// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/ExecutableKernel.hpp>

namespace rocRoller
{
    inline bool ExecutableKernel::kernelLoaded() const
    {
        return m_kernelLoaded;
    }
}
