// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>

#include <stdexcept>

namespace hipdnn_integration_tests
{

// Single shared hipdnnHandle_t for the entire test binary.
//
// Created lazily on first use, destroyed in main() after RUN_ALL_TESTS().
inline hipdnnHandle_t getSharedHandle()
{
    static hipdnnHandle_t s_handle = [] {
        hipdnnHandle_t h;
        if(hipdnnCreate(&h) != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("getSharedHandle: hipdnnCreate failed");
        }
        return h;
    }();
    return s_handle;
}

} // namespace hipdnn_integration_tests
