// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_backend_base.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_backend_base.h
 * @brief Owning wrapper and common base of the cuDNN v0.x descriptor classes.
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_shim.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_backend_base.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-pass-by-value,performance-unnecessary-value-param,modernize-use-equals-default):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class OpaqueBackendPointer
{
    cudnnBackendDescriptor_t m_desc = nullptr;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

public:
    OpaqueBackendPointer(const OpaqueBackendPointer&) = delete;
    OpaqueBackendPointer& operator=(const OpaqueBackendPointer&) = delete;
    OpaqueBackendPointer(OpaqueBackendPointer&&) = default;

    OpaqueBackendPointer(cudnnBackendDescriptorType_t type)
    {
        status = detail::create_descriptor(type, &m_desc);
    }
    ~OpaqueBackendPointer()
    {
        detail::destroy_descriptor(m_desc);
    };
    const cudnnBackendDescriptor_t& get_backend_descriptor() const
    {
        return m_desc;
    }
    cudnnStatus_t get_status() const
    {
        return status;
    }
    bool is_good() const
    {
        return status == CUDNN_STATUS_SUCCESS;
    }
};

using ManagedOpaqueDescriptor = std::shared_ptr<OpaqueBackendPointer>;

inline ManagedOpaqueDescriptor make_shared_backend_pointer(cudnnBackendDescriptorType_t type)
{
    return std::make_shared<OpaqueBackendPointer>(type);
}

class BackendDescriptor
{
public:
    virtual std::string describe() const = 0;

    cudnnBackendDescriptor_t get_raw_desc() const
    {
        return pointer->get_backend_descriptor();
    }

    cudnnStatus_t get_status() const
    {
        return status;
    }

    void set_status(const cudnnStatus_t status_) const
    {
        status = status_;
    }

    void set_error(const char* message) const
    {
        err_msg = message;
    }

    const char* get_error() const
    {
        return err_msg.c_str();
    }

    ManagedOpaqueDescriptor get_desc() const
    {
        return pointer;
    }

    cudnnStatus_t initialize_managed_backend_pointer(cudnnBackendDescriptorType_t type)
    {
        pointer = make_shared_backend_pointer(type);
        return pointer->get_status();
    }

protected:
    BackendDescriptor(ManagedOpaqueDescriptor pointer_, cudnnStatus_t status_, std::string err_msg_)
        : pointer(pointer_)
        , status(status_)
        , err_msg(err_msg_)
    {
    }
    BackendDescriptor() = default;

    virtual ~BackendDescriptor() {};

    ManagedOpaqueDescriptor pointer;

    mutable cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    mutable std::string err_msg;
};

// NOLINTEND(readability-identifier-naming,modernize-pass-by-value,performance-unnecessary-value-param,modernize-use-equals-default)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
