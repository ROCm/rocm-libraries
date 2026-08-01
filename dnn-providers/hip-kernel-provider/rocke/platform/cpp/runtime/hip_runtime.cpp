// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hip_runtime.cpp -- optional HIP device-target discovery for the C99 ABI.
 *
 * The engine remains usable for CPU-only and cross-compile callers: an
 * explicit target never loads HIP. Runtime discovery dynamically resolves the
 * small HIP query surface and lets HIP own device visibility/remapping.
 */
#include "rocke/runtime_hip.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "rocke/runtime_hip_internal.hpp"

namespace
{

constexpr std::size_t rocke_hip_props_cap = 4096;

void clear_buffer(char* out, std::size_t cap)
{
    if(out != nullptr && cap > 0)
    {
        out[0] = '\0';
    }
}

void write_error(char* err, std::size_t err_cap, const char* fmt, ...)
{
    if(err == nullptr || err_cap == 0)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(err, err_cap, fmt, ap);
    va_end(ap);
    err[err_cap - 1] = '\0';
}

const char* hip_error_text(const ckc::HipApi& api, int status)
{
    if(api.get_error_string == nullptr)
    {
        return nullptr;
    }
    return api.get_error_string(status);
}

void write_hip_error(
    char* err, std::size_t err_cap, const ckc::HipApi& api, const char* operation, int status)
{
    const char* detail = hip_error_text(api, status);
    if(detail != nullptr && detail[0] != '\0')
    {
        write_error(err, err_cap, "%s failed with HIP status %d (%s)", operation, status, detail);
    }
    else
    {
        write_error(err, err_cap, "%s failed with HIP status %d", operation, status);
    }
}

bool is_gfx_char(unsigned char c)
{
    return (c >= static_cast<unsigned char>('0') && c <= static_cast<unsigned char>('9'))
           || (c >= static_cast<unsigned char>('a') && c <= static_cast<unsigned char>('z'));
}

rocke_status_t extract_gfx(const unsigned char* props,
                           std::size_t props_size,
                           char* out_gfx,
                           std::size_t out_gfx_cap,
                           char* err,
                           std::size_t err_cap)
{
    for(std::size_t i = 0; i + 3 < props_size; ++i)
    {
        if(props[i] != static_cast<unsigned char>('g')
           || props[i + 1] != static_cast<unsigned char>('f')
           || props[i + 2] != static_cast<unsigned char>('x') || !is_gfx_char(props[i + 3]))
        {
            continue;
        }
        std::size_t end = i + 3;
        while(end < props_size && is_gfx_char(props[end]))
        {
            ++end;
        }
        std::size_t len = end - i;
        if(len + 1 > out_gfx_cap)
        {
            write_error(err,
                        err_cap,
                        "HIP device architecture requires %zu bytes; output capacity is %zu",
                        len + 1,
                        out_gfx_cap);
            return ROCKE_ERR_VALUE;
        }
        std::memcpy(out_gfx, props + i, len);
        out_gfx[len] = '\0';
        return ROCKE_OK;
    }
    write_error(err, err_cap, "HIP device properties contain no gfx architecture token");
    return ROCKE_ERR_HIP_RUNTIME;
}

struct LoadedHipApi
{
    ckc::HipApi api{};
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
};

LoadedHipApi loaded_hip;
bool loaded_hip_ready = false;
std::mutex loaded_hip_mutex;

std::vector<unsigned long> version_key(const std::string& value)
{
    std::vector<unsigned long> key;
    unsigned long number = 0;
    bool in_number = false;
    for(unsigned char c : value)
    {
        if(c >= static_cast<unsigned char>('0') && c <= static_cast<unsigned char>('9'))
        {
            unsigned long digit = static_cast<unsigned long>(c - static_cast<unsigned char>('0'));
            constexpr unsigned long max = std::numeric_limits<unsigned long>::max();
            number = number > (max - digit) / 10 ? max : number * 10 + digit;
            in_number = true;
        }
        else if(in_number)
        {
            key.push_back(number);
            number = 0;
            in_number = false;
        }
    }
    if(in_number)
    {
        key.push_back(number);
    }
    return key;
}

bool path_version_newer(const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    std::vector<unsigned long> lhs_key = version_key(lhs.string());
    std::vector<unsigned long> rhs_key = version_key(rhs.string());
    if(lhs_key != rhs_key)
    {
        return lhs_key > rhs_key;
    }
    return lhs.string() > rhs.string();
}

void append_unique(std::vector<std::string>& candidates, const std::filesystem::path& path)
{
    std::string value = path.string();
    if(std::find(candidates.begin(), candidates.end(), value) == candidates.end())
    {
        candidates.push_back(std::move(value));
    }
}

#if defined(_WIN32)

template <typename T>
T lookup_symbol(HMODULE handle, const char* name)
{
    FARPROC symbol = GetProcAddress(handle, name);
    return reinterpret_cast<T>(symbol);
}

bool populate_api(HMODULE handle, ckc::HipApi* api)
{
    api->get_device = lookup_symbol<ckc::hip_get_device_fn>(handle, "hipGetDevice");
    api->get_device_properties
        = lookup_symbol<ckc::hip_get_device_properties_fn>(handle, "hipGetDevicePropertiesR0600");
    if(api->get_device_properties == nullptr)
    {
        api->get_device_properties
            = lookup_symbol<ckc::hip_get_device_properties_fn>(handle, "hipGetDeviceProperties");
    }
    api->get_error_string
        = lookup_symbol<ckc::hip_get_error_string_fn>(handle, "hipGetErrorString");
    return api->get_device != nullptr && api->get_device_properties != nullptr;
}

bool use_owned_handle(HMODULE handle, LoadedHipApi* out)
{
    ckc::HipApi api{};
    if(handle == nullptr || !populate_api(handle, &api))
    {
        return false;
    }
    out->api = api;
    out->handle = handle;
    return true;
}

bool open_hip_library(
    const char* path, LoadedHipApi* out, char* err, std::size_t err_cap, const char* source)
{
    HMODULE handle = LoadLibraryExA(
        path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(handle == nullptr)
    {
        handle = LoadLibraryA(path);
    }
    if(handle == nullptr)
    {
        if(source != nullptr)
        {
            write_error(err,
                        err_cap,
                        "cannot load HIP runtime from %s (Windows error %lu)",
                        source,
                        static_cast<unsigned long>(GetLastError()));
        }
        return false;
    }
    if(!use_owned_handle(handle, out))
    {
        if(source != nullptr)
        {
            write_error(
                err, err_cap, "HIP runtime from %s is missing device query symbols", source);
        }
        FreeLibrary(handle);
        return false;
    }
    return true;
}

bool pin_loaded_hip(LoadedHipApi* out)
{
    constexpr const char* names[] = {"amdhip64_7.dll", "amdhip64.dll"};
    for(const char* name : names)
    {
        HMODULE handle = nullptr;
        if(GetModuleHandleExA(0, name, &handle) == 0)
        {
            continue;
        }
        if(use_owned_handle(handle, out))
        {
            return true;
        }
        FreeLibrary(handle);
    }
    return false;
}

void append_windows_root_candidates(std::vector<std::string>& candidates, const char* root)
{
    if(root == nullptr || root[0] == '\0')
    {
        return;
    }
    std::filesystem::path bin = std::filesystem::path(root) / "bin";
    std::error_code ec;
    if(!std::filesystem::is_directory(bin, ec))
    {
        return;
    }
    append_unique(candidates, bin / "amdhip64.dll");
    std::vector<std::filesystem::path> versioned;
    for(std::filesystem::directory_iterator it(bin, ec), end; !ec && it != end; it.increment(ec))
    {
        std::string name = it->path().filename().string();
        if(name.rfind("amdhip64", 0) == 0 && it->path().extension() == ".dll")
        {
            versioned.push_back(it->path());
        }
    }
    std::sort(versioned.begin(), versioned.end(), path_version_newer);
    for(const auto& path : versioned)
    {
        append_unique(candidates, path);
    }
}

std::vector<std::string> hip_library_candidates()
{
    std::vector<std::string> candidates;
    constexpr const char* roots[] = {"HIP_PATH", "ROCM_PATH", "ROCM_HOME"};
    for(const char* root : roots)
    {
        append_windows_root_candidates(candidates, std::getenv(root));
    }
    append_unique(candidates, "amdhip64_7.dll");
    append_unique(candidates, "amdhip64.dll");
    return candidates;
}

rocke_status_t load_hip_api(LoadedHipApi* out, char* err, std::size_t err_cap)
{
    const char* override_path = std::getenv("ROCKE_HIP_LIB");
    if(override_path != nullptr && override_path[0] != '\0')
    {
        return open_hip_library(override_path, out, err, err_cap, "ROCKE_HIP_LIB")
                   ? ROCKE_OK
                   : ROCKE_ERR_HIP_RUNTIME;
    }
    if(pin_loaded_hip(out))
    {
        return ROCKE_OK;
    }
    for(const std::string& candidate : hip_library_candidates())
    {
        if(open_hip_library(candidate.c_str(), out, nullptr, 0, nullptr))
        {
            return ROCKE_OK;
        }
    }
    write_error(
        err, err_cap, "cannot load a HIP runtime with device query symbols; set ROCKE_HIP_LIB");
    return ROCKE_ERR_HIP_RUNTIME;
}

#else

template <typename T>
T lookup_symbol(void* handle, const char* name)
{
    return reinterpret_cast<T>(dlsym(handle, name));
}

bool populate_api(void* handle, ckc::HipApi* api)
{
    api->get_device = lookup_symbol<ckc::hip_get_device_fn>(handle, "hipGetDevice");
    api->get_device_properties
        = lookup_symbol<ckc::hip_get_device_properties_fn>(handle, "hipGetDevicePropertiesR0600");
    if(api->get_device_properties == nullptr)
    {
        api->get_device_properties
            = lookup_symbol<ckc::hip_get_device_properties_fn>(handle, "hipGetDeviceProperties");
    }
    api->get_error_string
        = lookup_symbol<ckc::hip_get_error_string_fn>(handle, "hipGetErrorString");
    return api->get_device != nullptr && api->get_device_properties != nullptr;
}

bool use_owned_handle(void* handle, LoadedHipApi* out)
{
    ckc::HipApi api{};
    if(handle == nullptr || !populate_api(handle, &api))
    {
        return false;
    }
    out->api = api;
    out->handle = handle;
    return true;
}

bool open_hip_library(const char* path,
                      LoadedHipApi* out,
                      char* err,
                      std::size_t err_cap,
                      const char* source,
                      int flags = RTLD_NOW | RTLD_LOCAL)
{
    dlerror();
    void* handle = dlopen(path, flags);
    if(handle == nullptr)
    {
        if(source != nullptr)
        {
            const char* detail = dlerror();
            write_error(err,
                        err_cap,
                        "cannot load HIP runtime from %s: %s",
                        source,
                        detail != nullptr ? detail : "unknown loader error");
        }
        return false;
    }
    if(!use_owned_handle(handle, out))
    {
        if(source != nullptr)
        {
            write_error(
                err, err_cap, "HIP runtime from %s is missing device query symbols", source);
        }
        dlclose(handle);
        return false;
    }
    return true;
}

bool pin_global_hip(LoadedHipApi* out)
{
    void* symbol = dlsym(RTLD_DEFAULT, "hipGetDevice");
    if(symbol == nullptr)
    {
        return false;
    }
    Dl_info info{};
    if(dladdr(symbol, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0')
    {
        return false;
    }
    return open_hip_library(info.dli_fname, out, nullptr, 0, nullptr);
}

bool pin_loaded_hip(LoadedHipApi* out)
{
    if(pin_global_hip(out))
    {
        return true;
    }
#if defined(RTLD_NOLOAD)
    constexpr const char* names[] = {"libamdhip64.so.7", "libamdhip64.so"};
    for(const char* name : names)
    {
        if(open_hip_library(name, out, nullptr, 0, nullptr, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD))
        {
            return true;
        }
    }
#endif
    return false;
}

void append_posix_libdir(std::vector<std::string>& candidates, const std::filesystem::path& libdir)
{
    std::error_code ec;
    if(!std::filesystem::is_directory(libdir, ec))
    {
        return;
    }
    append_unique(candidates, libdir / "libamdhip64.so");
    append_unique(candidates, libdir / "libamdhip64.so.7");
}

std::vector<std::filesystem::path> list_versioned_directories(const std::filesystem::path& parent,
                                                              const char* prefix)
{
    std::vector<std::filesystem::path> directories;
    std::error_code ec;
    for(std::filesystem::directory_iterator it(parent, ec), end; !ec && it != end; it.increment(ec))
    {
        std::string name = it->path().filename().string();
        std::error_code type_error;
        if(name.rfind(prefix, 0) == 0 && it->is_directory(type_error) && !type_error)
        {
            directories.push_back(it->path());
        }
    }
    std::sort(directories.begin(), directories.end(), path_version_newer);
    return directories;
}

std::vector<std::string> hip_library_candidates()
{
    std::vector<std::string> candidates;
    constexpr const char* roots[] = {"ROCM_PATH", "ROCM_HOME"};
    for(const char* root_name : roots)
    {
        const char* root = std::getenv(root_name);
        if(root != nullptr && root[0] != '\0')
        {
            append_posix_libdir(candidates, std::filesystem::path(root) / "lib");
        }
    }

    std::vector<std::filesystem::path> opt_roots = list_versioned_directories("/opt", "rocm");
    for(const auto& root : opt_roots)
    {
        for(const auto& core : list_versioned_directories(root, "core-"))
        {
            append_posix_libdir(candidates, core / "lib");
        }
    }
    for(const auto& root : opt_roots)
    {
        append_posix_libdir(candidates, root / "lib");
    }
    append_unique(candidates, "libamdhip64.so");
    append_unique(candidates, "libamdhip64.so.7");
    return candidates;
}

rocke_status_t load_hip_api(LoadedHipApi* out, char* err, std::size_t err_cap)
{
    const char* override_path = std::getenv("ROCKE_HIP_LIB");
    if(override_path != nullptr && override_path[0] != '\0')
    {
        return open_hip_library(override_path, out, err, err_cap, "ROCKE_HIP_LIB")
                   ? ROCKE_OK
                   : ROCKE_ERR_HIP_RUNTIME;
    }
    if(pin_loaded_hip(out))
    {
        return ROCKE_OK;
    }
    for(const std::string& candidate : hip_library_candidates())
    {
        if(open_hip_library(candidate.c_str(), out, nullptr, 0, nullptr))
        {
            return ROCKE_OK;
        }
    }
    write_error(
        err, err_cap, "cannot load a HIP runtime with device query symbols; set ROCKE_HIP_LIB");
    return ROCKE_ERR_HIP_RUNTIME;
}

#endif

rocke_status_t get_loaded_hip_api(const ckc::HipApi** out, char* err, std::size_t err_cap)
{
    std::lock_guard<std::mutex> lock(loaded_hip_mutex);
    if(!loaded_hip_ready)
    {
        LoadedHipApi candidate{};
        rocke_status_t status = load_hip_api(&candidate, err, err_cap);
        if(status != ROCKE_OK)
        {
            return status;
        }
        loaded_hip = candidate;
        loaded_hip_ready = true;
    }
    *out = &loaded_hip.api;
    return ROCKE_OK;
}

rocke_status_t
    report_exception(const char* operation, const char* detail, char* err, std::size_t err_cap)
{
    write_error(err,
                err_cap,
                "%s failed: %s",
                operation,
                detail != nullptr ? detail : "unknown C++ exception");
    return ROCKE_ERR_HIP_RUNTIME;
}

template <typename Fn>
rocke_status_t
    guard_hip_status(const char* operation, char* err, std::size_t err_cap, Fn&& fn) noexcept
{
    try
    {
        return fn();
    }
    catch(const std::exception& e)
    {
        return report_exception(operation, e.what(), err, err_cap);
    }
    catch(...)
    {
        return report_exception(operation, nullptr, err, err_cap);
    }
}

} /* namespace */

namespace ckc
{

rocke_status_t hip_get_device_arch_with_api(const HipApi& api,
                                            int device,
                                            char* out_gfx,
                                            std::size_t out_gfx_cap,
                                            char* err,
                                            std::size_t err_cap) noexcept
{
    return guard_hip_status("HIP device architecture query", err, err_cap, [&]() {
        clear_buffer(out_gfx, out_gfx_cap);
        clear_buffer(err, err_cap);
        if(out_gfx == nullptr || out_gfx_cap == 0)
        {
            write_error(err, err_cap, "HIP architecture output buffer is NULL or empty");
            return ROCKE_ERR_VALUE;
        }
        if(device < 0)
        {
            write_error(err, err_cap, "HIP device ordinal must be non-negative, got %d", device);
            return ROCKE_ERR_VALUE;
        }
        if(api.get_device_properties == nullptr)
        {
            write_error(err, err_cap, "HIP runtime is missing hipGetDeviceProperties");
            return ROCKE_ERR_HIP_RUNTIME;
        }

        unsigned char props[rocke_hip_props_cap]{};
        int status = api.get_device_properties(props, device);
        if(status != 0)
        {
            write_hip_error(err, err_cap, api, "hipGetDeviceProperties", status);
            return ROCKE_ERR_HIP_RUNTIME;
        }
        return extract_gfx(props, sizeof(props), out_gfx, out_gfx_cap, err, err_cap);
    });
}

rocke_status_t hip_get_current_device_arch_with_api(const HipApi& api,
                                                    int* out_device,
                                                    char* out_gfx,
                                                    std::size_t out_gfx_cap,
                                                    char* err,
                                                    std::size_t err_cap) noexcept
{
    return guard_hip_status("HIP current-device architecture query", err, err_cap, [&]() {
        clear_buffer(out_gfx, out_gfx_cap);
        clear_buffer(err, err_cap);
        if(out_device == nullptr)
        {
            write_error(err, err_cap, "HIP current-device output is NULL");
            return ROCKE_ERR_VALUE;
        }
        if(api.get_device == nullptr)
        {
            write_error(err, err_cap, "HIP runtime is missing hipGetDevice");
            return ROCKE_ERR_HIP_RUNTIME;
        }

        int device = -1;
        int status = api.get_device(&device);
        if(status != 0)
        {
            write_hip_error(err, err_cap, api, "hipGetDevice", status);
            return ROCKE_ERR_HIP_RUNTIME;
        }
        rocke_status_t result
            = hip_get_device_arch_with_api(api, device, out_gfx, out_gfx_cap, err, err_cap);
        if(result == ROCKE_OK)
        {
            *out_device = device;
        }
        return result;
    });
}

rocke_status_t resolve_compile_target_with_api(const HipApi* api,
                                               const char* requested_gfx,
                                               rocke_resolved_target_t* out,
                                               char* err,
                                               std::size_t err_cap) noexcept
{
    return guard_hip_status("compile target resolution", err, err_cap, [&]() {
        clear_buffer(err, err_cap);
        if(out == nullptr)
        {
            write_error(err, err_cap, "resolved-target output is NULL");
            return ROCKE_ERR_VALUE;
        }

        rocke_resolved_target_t resolved{};
        if(requested_gfx != nullptr)
        {
            if(requested_gfx[0] == '\0')
            {
                write_error(err, err_cap, "explicit compile target must not be empty");
                return ROCKE_ERR_VALUE;
            }
            resolved.target = rocke_arch_target_from_gfx(requested_gfx);
            if(resolved.target == nullptr)
            {
                write_error(err, err_cap, "unknown explicit compile target '%s'", requested_gfx);
                return ROCKE_ERR_KEY;
            }
            resolved.device = -1;
            resolved.from_runtime = false;
            *out = resolved;
            return ROCKE_OK;
        }
        if(api == nullptr)
        {
            write_error(
                err, err_cap, "HIP runtime API is unavailable for automatic target resolution");
            return ROCKE_ERR_HIP_RUNTIME;
        }

        char gfx[64];
        int device = -1;
        rocke_status_t status
            = hip_get_current_device_arch_with_api(*api, &device, gfx, sizeof(gfx), err, err_cap);
        if(status != ROCKE_OK)
        {
            return status;
        }
        resolved.target = rocke_arch_target_from_gfx(gfx);
        if(resolved.target == nullptr)
        {
            write_error(
                err, err_cap, "HIP device %d reports unsupported architecture '%s'", device, gfx);
            return ROCKE_ERR_NOTIMPL;
        }
        resolved.device = device;
        resolved.from_runtime = true;
        *out = resolved;
        return ROCKE_OK;
    });
}

} /* namespace ckc */

rocke_status_t rocke_hip_get_device_arch(
    int device, char* out_gfx, size_t out_gfx_cap, char* err, size_t err_cap)
{
    try
    {
        clear_buffer(out_gfx, out_gfx_cap);
        clear_buffer(err, err_cap);
        const ckc::HipApi* api = nullptr;
        rocke_status_t status = get_loaded_hip_api(&api, err, err_cap);
        if(status != ROCKE_OK)
        {
            return status;
        }
        return ckc::hip_get_device_arch_with_api(*api, device, out_gfx, out_gfx_cap, err, err_cap);
    }
    catch(const std::exception& e)
    {
        return report_exception("HIP device architecture query", e.what(), err, err_cap);
    }
    catch(...)
    {
        return report_exception("HIP device architecture query", nullptr, err, err_cap);
    }
}

rocke_status_t rocke_hip_get_current_device_arch(
    int* out_device, char* out_gfx, size_t out_gfx_cap, char* err, size_t err_cap)
{
    try
    {
        clear_buffer(out_gfx, out_gfx_cap);
        clear_buffer(err, err_cap);
        const ckc::HipApi* api = nullptr;
        rocke_status_t status = get_loaded_hip_api(&api, err, err_cap);
        if(status != ROCKE_OK)
        {
            return status;
        }
        return ckc::hip_get_current_device_arch_with_api(
            *api, out_device, out_gfx, out_gfx_cap, err, err_cap);
    }
    catch(const std::exception& e)
    {
        return report_exception("HIP current-device architecture query", e.what(), err, err_cap);
    }
    catch(...)
    {
        return report_exception("HIP current-device architecture query", nullptr, err, err_cap);
    }
}

rocke_status_t rocke_resolve_compile_target(const char* requested_gfx,
                                            rocke_resolved_target_t* out,
                                            char* err,
                                            size_t err_cap)
{
    return guard_hip_status("compile target resolution", err, err_cap, [&]() {
        if(requested_gfx != nullptr)
        {
            return ckc::resolve_compile_target_with_api(nullptr, requested_gfx, out, err, err_cap);
        }

        const ckc::HipApi* api = nullptr;
        rocke_status_t status = get_loaded_hip_api(&api, err, err_cap);
        if(status != ROCKE_OK)
        {
            return status;
        }
        return ckc::resolve_compile_target_with_api(api, nullptr, out, err, err_cap);
    });
}
