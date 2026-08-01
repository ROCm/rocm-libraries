// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocm/interfaces/runtime/module.h"

#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rocm::interfaces {

std::shared_ptr<Module> Module::open(const std::filesystem::path& path) {
    if (path.empty()) throw std::invalid_argument("provider module path is empty");
#if defined(_WIN32)
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    if (!handle) throw std::runtime_error("LoadLibraryW failed for " + path.string());
    return std::shared_ptr<Module>(new Module(path, handle));
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* message = dlerror();
        throw std::runtime_error("dlopen failed for " + path.string() + ": " +
                                 (message ? message : "unknown error"));
    }
    return std::shared_ptr<Module>(new Module(path, handle));
#endif
}

Module::~Module() {
    if (!native_handle_) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(native_handle_));
#else
    dlclose(native_handle_);
#endif
}

void* Module::symbol(const char* name) const {
    if (!name || !*name) throw std::invalid_argument("provider symbol name is empty");
#if defined(_WIN32)
    auto* result =
        reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(native_handle_), name));
#else
    dlerror();
    void* result = dlsym(native_handle_, name);
#endif
    if (!result) throw std::runtime_error("provider symbol not found: " + std::string(name));
    return result;
}

}  // namespace rocm::interfaces
