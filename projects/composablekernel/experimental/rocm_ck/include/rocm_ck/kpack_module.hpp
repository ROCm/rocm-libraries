// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — kpack archive/module RAII wrappers. Requires HIP runtime.
//
// RAII wrappers for kpack archive and kernel loading. Handles archive
// open/close, GPU detection, HIP module creation, and cleanup.

#pragma once

#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cstdio>
#include <string>

namespace rocm_ck {

/// RAII wrapper for a kpack archive. Opens the archive, prints available
/// architectures, and detects the current GPU. Non-copyable.
class KpackArchive
{
    public:
    KpackArchive() = default;
    ~KpackArchive()
    {
        if(archive_ != nullptr)
            kpack_close(archive_);
    }

    KpackArchive(const KpackArchive&)            = delete;
    KpackArchive& operator=(const KpackArchive&) = delete;

    KpackArchive(KpackArchive&& other) noexcept
        : archive_(other.archive_), gpu_arch_(std::move(other.gpu_arch_))
    {
        other.archive_ = nullptr;
    }

    KpackArchive& operator=(KpackArchive&& other) noexcept
    {
        if(this != &other)
        {
            if(archive_ != nullptr)
                kpack_close(archive_);
            archive_       = other.archive_;
            gpu_arch_      = std::move(other.gpu_arch_);
            other.archive_ = nullptr;
        }
        return *this;
    }

    /// Open a kpack archive, print its architecture list, and detect the GPU.
    /// Returns false on failure (errors printed to stderr).
    bool open(const char* path)
    {
        kpack_error_t kerr = kpack_open(path, &archive_);
        if(kerr != KPACK_SUCCESS)
        {
            std::fprintf(stderr, "Failed to open archive '%s' (error %d)\n", path, kerr);
            return false;
        }

        // Print architectures
        size_t arch_count = 0;
        kpack_get_architecture_count(archive_, &arch_count);
        std::printf("Opened %s — architectures:", path);
        for(size_t i = 0; i < arch_count; ++i)
        {
            const char* arch_name = nullptr;
            kpack_get_architecture(archive_, i, &arch_name);
            std::printf("%s %s", (i > 0 ? "," : ""), arch_name);
        }
        std::printf("\n");

        // Detect GPU
        gpu_arch_ = get_gpu_arch();
        if(gpu_arch_.empty())
        {
            std::fprintf(stderr, "Failed to detect GPU architecture\n");
            return false;
        }
        std::printf("Detected GPU: %s\n", gpu_arch_.c_str());

        return true;
    }

    const char* gpu_arch() const { return gpu_arch_.c_str(); }
    kpack_archive_t handle() const { return archive_; }

    private:
    kpack_archive_t archive_ = nullptr;
    std::string gpu_arch_;
};

/// RAII wrapper for a single kernel loaded from a kpack archive.
/// Manages the code object memory, HIP module, and function handle.
/// Non-copyable.
class KpackKernel
{
    public:
    KpackKernel() = default;
    ~KpackKernel()
    {
        // Destructor can't propagate errors — best-effort cleanup
        if(module_ != nullptr)
            (void)hipModuleUnload(module_);
        if(code_object_ != nullptr)
            kpack_free_kernel(code_object_);
    }

    KpackKernel(const KpackKernel&)            = delete;
    KpackKernel& operator=(const KpackKernel&) = delete;

    KpackKernel(KpackKernel&& other) noexcept
        : code_object_(other.code_object_), module_(other.module_), function_(other.function_)
    {
        other.code_object_ = nullptr;
        other.module_      = nullptr;
        other.function_    = nullptr;
    }

    KpackKernel& operator=(KpackKernel&& other) noexcept
    {
        if(this != &other)
        {
            if(module_ != nullptr)
                (void)hipModuleUnload(module_);
            if(code_object_ != nullptr)
                kpack_free_kernel(code_object_);
            code_object_       = other.code_object_;
            module_            = other.module_;
            function_          = other.function_;
            other.code_object_ = nullptr;
            other.module_      = nullptr;
            other.function_    = nullptr;
        }
        return *this;
    }

    /// Load a kernel by name from the archive for the archive's detected GPU.
    /// Returns false if the kernel is not found for this architecture.
    bool load(const KpackArchive& archive, const char* kernel_name)
    {
        size_t code_object_size = 0;
        kpack_error_t kerr      = kpack_get_kernel(
            archive.handle(), kernel_name, archive.gpu_arch(), &code_object_, &code_object_size);
        if(kerr != KPACK_SUCCESS)
        {
            std::fprintf(stderr,
                         "%s: no kernel for '%s' (error %d), skipping\n",
                         kernel_name,
                         archive.gpu_arch(),
                         kerr);
            return false;
        }

        HIP_CHECK(hipModuleLoadData(&module_, code_object_));
        HIP_CHECK(hipModuleGetFunction(&function_, module_, kernel_name));
        return true;
    }

    hipFunction_t function() const { return function_; }

    private:
    void* code_object_      = nullptr;
    hipModule_t module_     = nullptr;
    hipFunction_t function_ = nullptr;
};

} // namespace rocm_ck
