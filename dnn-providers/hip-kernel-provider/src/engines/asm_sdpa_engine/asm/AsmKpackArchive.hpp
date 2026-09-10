// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Process-level singleton managing per-arch kpack archive handles.
//
// Lazily opens a .kpack archive on first request for a given arch, caches the
// handle for the process lifetime, and closes all handles in the destructor.
// Thread-safe: the mutex guards archive open; kpack_get_kernel is documented
// as thread-safe on the same handle.

#pragma once

#include "PluginModuleDir.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <rocm_kpack/kpack.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace asm_sdpa_engine::asm_kernels
{

class AsmKpackArchive
{
public:
    static AsmKpackArchive& instance()
    {
        static AsmKpackArchive s_singleton;
        return s_singleton;
    }

    /// RAII wrapper for kernel bytes extracted from a kpack archive.
    /// Automatically calls kpack_free_kernel on destruction.
    struct KernelData
    {
        void* data = nullptr;
        size_t size = 0;

        ~KernelData()
        {
            if(data != nullptr)
            {
                kpack_free_kernel(data);
            }
        }

        KernelData() = default;
        KernelData(void* d, size_t s)
            : data(d)
            , size(s)
        {
        }

        KernelData(KernelData&& o) noexcept
            : data(std::exchange(o.data, nullptr))
            , size(o.size)
        {
        }

        KernelData& operator=(KernelData&& o) noexcept
        {
            std::swap(data, o.data);
            size = o.size;
            return *this;
        }

        KernelData(const KernelData&) = delete;
        KernelData& operator=(const KernelData&) = delete;
    };

    /// Extract kernel bytes from the archive for a given TOC key and arch.
    KernelData getKernel(const std::string& tocKey, const std::string& arch)
    {
        kpack_archive_t archive = getOrOpenArchive(arch);

        void* data = nullptr;
        size_t size = 0;
        const kpack_error_t err
            = kpack_get_kernel(archive, tocKey.c_str(), arch.c_str(), &data, &size);
        if(err != KPACK_SUCCESS)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "kpack_get_kernel failed for tocKey='" + tocKey + "' arch='" + arch
                    + "' (error=" + std::to_string(static_cast<int>(err)) + ")");
        }

        return KernelData{data, size};
    }

    AsmKpackArchive(const AsmKpackArchive&) = delete;
    AsmKpackArchive& operator=(const AsmKpackArchive&) = delete;
    AsmKpackArchive(AsmKpackArchive&&) = delete;
    AsmKpackArchive& operator=(AsmKpackArchive&&) = delete;

private:
    AsmKpackArchive() = default;

    ~AsmKpackArchive()
    {
        for(auto& [arch, archive] : _archives)
        {
            kpack_close(archive);
        }
    }

    kpack_archive_t getOrOpenArchive(const std::string& arch)
    {
        const std::lock_guard<std::mutex> lock(_mutex);

        auto it = _archives.find(arch);
        if(it != _archives.end())
        {
            return it->second;
        }

        auto kpackPath = currentPluginDirectory() / "arch_content" / "asm_sdpa" / arch
                         / ("hip_kernel_provider_sdpa_" + arch + ".kpack");

        kpack_archive_t archive = nullptr;
        const kpack_error_t err = kpack_open(kpackPath.string().c_str(), &archive);
        if(err != KPACK_SUCCESS)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "kpack_open failed for '" + kpackPath.string()
                    + "' (error=" + std::to_string(static_cast<int>(err)) + ")");
        }

        HIPDNN_PLUGIN_LOG_INFO("Opened kpack archive: " << kpackPath.string());
        _archives[arch] = archive;
        return archive;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, kpack_archive_t> _archives;
};

} // namespace asm_sdpa_engine::asm_kernels
