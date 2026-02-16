// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle/HandleFactory.hpp"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"
#include "utilities/PointerToString.hpp"

namespace hipdnn_backend
{

void HandleFactory::createHandle(hipdnnHandle_t* handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    *handle = new hipdnnHandle();

    HIPDNN_BACKEND_LOG_INFO("Created handle: " << ptrToString(*handle));
}

void HandleFactory::destroyHandle(hipdnnHandle_t handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    delete handle;

    HIPDNN_BACKEND_LOG_INFO("Destroyed handle: " << ptrToString(handle));
}

} // namespace hipdnn_backend
