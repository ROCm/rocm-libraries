// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <miopen/miopen.h>

#include <hip/hip_runtime_api.h>

// RAII wrapper for miopenHandle_t. Cannot use the DescGuard template because
// miopenCreateWithStream takes an extra hipStream_t argument. The handle is
// default-constructed empty; call create(stream) to allocate it lazily.
class HandleGuard
{
public:
    HandleGuard() = default;

    explicit HandleGuard(hipStream_t stream) { status = miopenCreateWithStream(&handle, stream); }

    ~HandleGuard()
    {
        if(handle != nullptr)
            miopenDestroy(handle);
    }

    void create(hipStream_t stream)
    {
        if(handle != nullptr)
            miopenDestroy(handle);
        status = miopenCreateWithStream(&handle, stream);
    }

    operator miopenHandle_t() { return handle; }
    miopenHandle_t get() { return handle; }
    miopenStatus_t getStatus() const { return status; }

    HandleGuard(const HandleGuard&)            = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
    HandleGuard(HandleGuard&&)                 = delete;
    HandleGuard& operator=(HandleGuard&&)      = delete;

private:
    miopenHandle_t handle = nullptr;
    miopenStatus_t status = miopenStatusSuccess;
};
