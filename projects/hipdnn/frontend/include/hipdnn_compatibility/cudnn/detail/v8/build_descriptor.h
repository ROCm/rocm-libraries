// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_backend_base.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Logging.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// The build() of the convolution, matmul, pointwise, reduction, resample, RNG and tensor
// descriptor builders. The pointer is created before makeRecord validates, so a failed
// descriptor still has a (null) raw descriptor. BackendDescriptor befriends this to install the
// built pointer.
template <typename Descriptor, typename MakeRecord>
Descriptor&& buildDescriptor(Descriptor& descriptor,
                             cudnnBackendDescriptorType_t type,
                             MakeRecord&& makeRecord)
{
    const Outcome outcome = guarded([&] {
        descriptor.initialize_managed_backend_pointer(type);
        descriptor.pointer = makeManaged(type, std::forward<MakeRecord>(makeRecord)());
    });
    if(!outcome)
    {
        set_error_and_throw_exception(&descriptor, outcome.status, outcome.message.c_str());
        return std::move(descriptor);
    }

    // A retry after a failed build() must not keep that failure's status.
    descriptor.set_status(CUDNN_STATUS_SUCCESS);
    descriptor.set_error("");
    CUDNN_FE_LOG_LABEL_ENDL(descriptor);
    return std::move(descriptor);
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
