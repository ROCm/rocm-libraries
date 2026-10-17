// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_VariantPack.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_VariantPack.h
 * @brief cuDNN FE v0.x `VariantPack_v8` and its builder, on the shim's backend C API.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_VariantPack.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class VariantPack_v8 : public BackendDescriptor
{
public:
    friend class VariantPackBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR :" << " has " << num_ptrs << " data pointers";
        return ss.str();
    }

    VariantPack_v8(VariantPack_v8&& from) = default;
    VariantPack_v8& operator=(VariantPack_v8&& from) = default;

    ~VariantPack_v8() = default;

private:
    VariantPack_v8() = default;
    VariantPack_v8(const VariantPack_v8&) = delete;
    VariantPack_v8& operator=(const VariantPack_v8&) = delete;

    void* workspace = nullptr;
    std::vector<void*> data_pointers;
    std::vector<int64_t> uid;
    int64_t num_ptrs = -1;
};

class VariantPackBuilder_v8
{
public:
    auto setDataPointers(int64_t num_ptr, void** ptrs) -> VariantPackBuilder_v8&
    {
        m_variant_pack.data_pointers.reserve(static_cast<size_t>(num_ptr));
        std::copy(ptrs, ptrs + num_ptr, std::back_inserter(m_variant_pack.data_pointers));
        m_variant_pack.num_ptrs = num_ptr;
        return *this;
    }
    auto setUids(int64_t num_uids, const int64_t* uid) -> VariantPackBuilder_v8&
    {
        return setUids(num_uids, const_cast<int64_t*>(uid));
    }

    auto setUids(int64_t num_uids, int64_t* uid) -> VariantPackBuilder_v8&
    {
        m_variant_pack.uid.reserve(static_cast<size_t>(num_uids));
        std::copy(uid, uid + num_uids, std::back_inserter(m_variant_pack.uid));
        return *this;
    }
    auto setDataPointers(const std::set<std::pair<uint64_t, void*>>& data_pointers)
        -> VariantPackBuilder_v8&
    {
        m_variant_pack.num_ptrs = static_cast<int64_t>(data_pointers.size());
        m_variant_pack.uid.reserve(static_cast<size_t>(m_variant_pack.num_ptrs));
        m_variant_pack.data_pointers.reserve(static_cast<size_t>(m_variant_pack.num_ptrs));
        for(auto& data_pointer : data_pointers)
        {
            m_variant_pack.uid.push_back(static_cast<int64_t>(data_pointer.first));
            m_variant_pack.data_pointers.push_back(data_pointer.second);
        }
        return *this;
    }
    auto setWorkspacePointer(void* ws) -> VariantPackBuilder_v8&
    {
        m_variant_pack.workspace = ws;
        return *this;
    }

    VariantPack_v8&& build()
    {
        auto status = m_variant_pack.initialize_managed_backend_pointer(
            CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       CUDNN_TYPE_VOID_PTR,
                                       m_variant_pack.num_ptrs,
                                       m_variant_pack.data_pointers.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_variant_pack,
                                          status,
                                          "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       CUDNN_TYPE_INT64,
                                       static_cast<int64_t>(m_variant_pack.uid.size()),
                                       m_variant_pack.uid.data());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_variant_pack,
                                          status,
                                          "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       &m_variant_pack.workspace);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_variant_pack,
                                          status,
                                          "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_VARIANT_PACK_WORKSPACE Failed");
            return std::move(m_variant_pack);
        }

        status = detail::finalize(m_variant_pack.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_variant_pack);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_variant_pack);
        return std::move(m_variant_pack);
    }

    explicit VariantPackBuilder_v8() = default;
    ~VariantPackBuilder_v8() = default;
    VariantPackBuilder_v8(VariantPackBuilder_v8&&) = delete;
    VariantPackBuilder_v8(const VariantPackBuilder_v8&) = delete;
    VariantPackBuilder_v8& operator=(const VariantPackBuilder_v8&) = delete;

private:
    VariantPack_v8 m_variant_pack;
};

using VariantPack = VariantPack_v8;
using VariantPackBuilder = VariantPackBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
