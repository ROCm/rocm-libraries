// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_RUNTIME_MODULE_H_
#define ROCM_INTERFACES_RUNTIME_MODULE_H_

#include <filesystem>
#include <memory>
#include <string>

namespace rocm::interfaces {

class Module {
   public:
    static std::shared_ptr<Module> open(const std::filesystem::path& path);

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    ~Module();

    void* symbol(const char* name) const;
    const std::filesystem::path& path() const noexcept {
        return path_;
    }

   private:
    Module(std::filesystem::path path, void* native_handle)
        : path_(std::move(path)), native_handle_(native_handle) {}

    std::filesystem::path path_;
    void* native_handle_ = nullptr;
};

}  // namespace rocm::interfaces
#endif
