// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "HipdnnException.hpp"
#include "HipdnnStatus.h"
#include "LastErrorManager.hpp"

// TODO: remove
#include <iostream>
#include <spdlog/fmt/fmt.h>
namespace hipdnn_backend
{

template <typename T>
std::string logPtr(T* ptr)
{
    std::cout << "Logging ptr\n";
    // This function is not invoked if the macro is a no-op.
    if(ptr != nullptr)
    {
        return ptr->toString();
    }
    return fmt::format("{:p}", static_cast<void*>(ptr));
}

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
