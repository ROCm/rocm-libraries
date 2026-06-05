// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "HipdnnException.hpp"
#include "HipdnnStatus.h"
#include "LastErrorManager.hpp"

#include <spdlog/fmt/fmt.h>

namespace hipdnn_backend
{

template <typename T>
std::string logPtr(T* ptr)
{
    // This function is not invoked if the macro is a no-op.
    if(ptr == nullptr)
    {
        return "nullptr";
    }

    try
    {
        return ptr->toString();
    }
    catch(const std::exception& e)
    {
        return fmt::format("InvalidPtr[{:p}] ({})", static_cast<void*>(ptr), e.what());
    }
    catch(...)
    {
        return fmt::format("InvalidPtr[{:p}] (Unknown exception)", static_cast<void*>(ptr));
    }
}

template <class F>
hipdnnStatus_t tryCatch(F f, const std::string& prefix = std::string{})
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
