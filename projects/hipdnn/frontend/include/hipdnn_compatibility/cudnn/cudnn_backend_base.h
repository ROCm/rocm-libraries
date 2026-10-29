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
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_shim.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_backend_base.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-pass-by-value,performance-unnecessary-value-param,modernize-use-equals-default):
// upstream's spelling and shapes are the source-compatible surface.

namespace detail::v8
{
struct RecordAccess;
template <typename Descriptor, typename MakeRecord>
Descriptor&& buildDescriptor(Descriptor& descriptor,
                             cudnnBackendDescriptorType_t type,
                             MakeRecord&& makeRecord);
} // namespace detail::v8

// Upstream wraps one cuDNN descriptor. Here it pairs the shim record behind a builder object with a
// non-owning view of the hipDNN descriptor (if any) that the record owns; get_backend_descriptor()
// is what get_raw_desc() and cudnnBackendExecute see. The public members are upstream's; the record
// and type are reached only through detail::v8::RecordAccess.
class OpaqueBackendPointer
{
    friend struct detail::v8::RecordAccess;

    // Only RecordAccess can create a key, so only the shim can call the record constructor, which
    // stays public for std::make_shared. The explicit default constructor also keeps the key from
    // being an aggregate that `{}` could build.
    class RecordKey
    {
        friend struct detail::v8::RecordAccess;
        explicit RecordKey() = default;
    };

    cudnnBackendDescriptor_t m_desc = nullptr;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cudnnBackendDescriptorType_t m_type{};
    std::shared_ptr<const detail::v8::Record> m_record;

public:
    OpaqueBackendPointer(const OpaqueBackendPointer&) = delete;
    OpaqueBackendPointer& operator=(const OpaqueBackendPointer&) = delete;
    OpaqueBackendPointer(OpaqueBackendPointer&&) = default;

    OpaqueBackendPointer(cudnnBackendDescriptorType_t type)
        : m_type(type)
    {
    }
    OpaqueBackendPointer(RecordKey /*key*/,
                         cudnnBackendDescriptorType_t type,
                         std::shared_ptr<const detail::v8::Record> record,
                         cudnnBackendDescriptor_t view)
        : m_desc(view)
        , m_type(type)
        , m_record(std::move(record))
    {
    }
    ~OpaqueBackendPointer() = default;

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

namespace detail::v8
{

struct RecordAccess
{
    static OpaqueBackendPointer::RecordKey key()
    {
        return OpaqueBackendPointer::RecordKey{};
    }
    static cudnnBackendDescriptorType_t type(const OpaqueBackendPointer& pointer)
    {
        return pointer.m_type;
    }
    static const std::shared_ptr<const Record>& record(const OpaqueBackendPointer& pointer)
    {
        return pointer.m_record;
    }
};

inline ManagedOpaqueDescriptor makeManaged(cudnnBackendDescriptorType_t type,
                                           std::shared_ptr<const Record> record,
                                           cudnnBackendDescriptor_t view = nullptr)
{
    return std::make_shared<OpaqueBackendPointer>(
        RecordAccess::key(), type, std::move(record), view);
}

// The record of the expected kind, or null when the pointer is empty or holds another kind.
template <typename T>
std::shared_ptr<const T> recordOf(const ManagedOpaqueDescriptor& pointer)
{
    if(!pointer)
    {
        return nullptr;
    }
    const std::shared_ptr<const Record>& record = RecordAccess::record(*pointer);
    if(!record || !T::isKind(record->kind()))
    {
        return nullptr;
    }
    return recordAs<T>(record);
}

} // namespace detail::v8

class BackendDescriptor
{
    // The descriptor builders' shared build() installs the built pointer.
    template <typename Descriptor, typename MakeRecord>
    friend Descriptor&& detail::v8::buildDescriptor(Descriptor& descriptor,
                                                    cudnnBackendDescriptorType_t type,
                                                    MakeRecord&& makeRecord);

public:
    virtual std::string describe() const = 0;

    cudnnBackendDescriptor_t get_raw_desc() const
    {
        // Upstream dereferences unchecked; a build() that failed before creating the pointer
        // (NV_CUDNN_DISABLE_EXCEPTION) leaves it null.
        return pointer ? pointer->get_backend_descriptor() : nullptr;
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
