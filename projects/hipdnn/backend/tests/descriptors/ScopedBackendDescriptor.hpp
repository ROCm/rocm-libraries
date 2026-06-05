// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>

#include <memory>

namespace hipdnn_backend::test_utilities
{

struct BackendDescriptorDeleter
{
    void operator()(hipdnnBackendDescriptor_t desc) const
    {
        hipdnnBackendDestroyDescriptor(desc);
    }
};

/// RAII wrapper for hipdnnBackendDescriptor_t in backend tests.
/// Calls hipdnnBackendDestroyDescriptor on destruction if non-null.
using ScopedBackendDescriptor
    = std::unique_ptr<std::remove_pointer_t<hipdnnBackendDescriptor_t>, BackendDescriptorDeleter>;

} // namespace hipdnn_backend::test_utilities
