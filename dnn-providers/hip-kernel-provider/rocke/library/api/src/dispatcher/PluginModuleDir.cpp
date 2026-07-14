// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/PluginModuleDir.hpp"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rocke_client::dispatcher
{

// NOTE: This function must be defined in a translation unit compiled into
// rocke_client_impl so that the dladdr / GetModuleHandleExA call resolves
// the rocke-client SHARED plugin, not the test executable or any other DSO.
std::filesystem::path currentPluginDirectory()
{
#ifdef _WIN32
    HMODULE handle = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&currentPluginDirectory),
                          &handle)
       == TRUE)
    {
        char buf[MAX_PATH];
        const DWORD len = GetModuleFileNameA(handle, buf, MAX_PATH);
        if(len > 0 && len < MAX_PATH)
        {
            return std::filesystem::weakly_canonical(std::filesystem::absolute(
                std::filesystem::path(std::string(buf, len)).parent_path()));
        }
    }
    throw std::runtime_error("GetModuleHandleExA/GetModuleFileNameA failed for "
                             "currentPluginDirectory");
#else
    Dl_info info{};
    if(dladdr(reinterpret_cast<const void*>(&currentPluginDirectory), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        return std::filesystem::weakly_canonical(
            std::filesystem::absolute(std::filesystem::path(info.dli_fname).parent_path()));
    }
    throw std::runtime_error("dladdr failed to resolve module path for "
                             "currentPluginDirectory");
#endif
}

} // namespace rocke_client::dispatcher
