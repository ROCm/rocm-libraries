// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "handle/HandleFactory.hpp"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"

namespace hipdnn_backend
{

void HandleFactory::createHandle(hipdnnHandle_t* handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    *handle = new hipdnnHandle();

    HIPDNN_LOG_INFO("Created handle: {:p}", static_cast<void*>(*handle));
}

void HandleFactory::destroyHandle(hipdnnHandle_t handle)
{
    THROW_IF_NULL(handle, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "handle is null.");

    delete handle;

    HIPDNN_LOG_INFO("Destroyed handle: {:p}", static_cast<void*>(handle));
}

} // namespace hipdnn_backend
