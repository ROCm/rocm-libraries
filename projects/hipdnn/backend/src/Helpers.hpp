// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "HipdnnException.hpp"
#include "HipdnnStatus.h"
#include "LastErrorManager.hpp"

#include <iostream>

namespace hipdnn_backend
{

template <class F>
hipdnnStatus_t tryCatch(F f, std::string const& prefix = std::string{})
{
    try
    {
        f();
    }
    catch(const HipdnnException& ex)
    {
        return LastErrorManager::setLastError(ex.getStatus(), (prefix + ex.what()).c_str());
    }
    catch(const std::exception& ex)
    {
        return LastErrorManager::setLastError(HIPDNN_STATUS_INTERNAL_ERROR,
                                              (prefix + ex.what()).c_str());
    }
    catch(...)
    {
        return LastErrorManager::setLastError(HIPDNN_STATUS_INTERNAL_ERROR,
                                              (prefix + "Unknown exception occured").c_str());
    }
    return HIPDNN_STATUS_SUCCESS;
}
} // namespace hipdnn_backend
