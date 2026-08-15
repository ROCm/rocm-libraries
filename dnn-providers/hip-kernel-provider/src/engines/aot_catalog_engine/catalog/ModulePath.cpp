// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/ModulePath.hpp"

#include <filesystem>
#include <system_error>

#if defined(_WIN32)
#include <array>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace aot_catalog_engine::catalog
{

namespace fs = std::filesystem;

std::string thisModuleDir()
{
    fs::path modulePath;

#if defined(_WIN32)
    // Resolve the module that contains &thisModuleDir -> this plugin's .dll.
    HMODULE handle = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&thisModuleDir),
                          &handle)
       != 0)
    {
        std::array<char, MAX_PATH> buf{};
        const DWORD len = GetModuleFileNameA(handle, buf.data(), static_cast<DWORD>(buf.size()));
        if(len > 0 && len < buf.size())
        {
            modulePath = fs::path(std::string(buf.data(), len)).parent_path();
        }
    }
#else
    // dladdr on &thisModuleDir yields the .so that this code is linked into.
    Dl_info info;
    if(dladdr(reinterpret_cast<const void*>(&thisModuleDir), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        modulePath = fs::path(info.dli_fname).parent_path();
    }
#endif

    if(modulePath.empty())
    {
        return {};
    }

    std::error_code ec;
    const fs::path canonical = fs::weakly_canonical(fs::absolute(modulePath, ec), ec);
    if(ec)
    {
        return modulePath.string(); // best effort if canonicalization fails
    }
    return canonical.string();
}

} // namespace aot_catalog_engine::catalog
