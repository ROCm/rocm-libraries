// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <hip/hip_runtime.h>

namespace hipdnn_integration_tests
{

// Throws Error(what + " failed: " + hipGetErrorString(status)) when status is not
// hipSuccess. Shared by every call site that turns a device-call failure into one of
// this project's own exception types, so the message stays worded the same way
// wherever it is reported.
template <class Error>
void checkDeviceCall(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw Error(std::string(what) + " failed: " + hipGetErrorString(status));
    }
}

} // namespace hipdnn_integration_tests
