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

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>

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
    rocke_status_t status = ROCKE_ERR_HIP_RUNTIME;
    char error[ROCKE_ERR_MSG_CAP]{};
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
};

LoadedHipApi loaded_hip;
std::once_flag loaded_hip_once;

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

void load_hip_api()
{
    constexpr const char* candidates[] = {"amdhip64_7.dll", "amdhip64.dll"};
    for(const char* candidate : candidates)
    {
        HMODULE handle = GetModuleHandleA(candidate);
        if(handle != nullptr && populate_api(handle, &loaded_hip.api))
        {
            loaded_hip.handle = handle;
            loaded_hip.status = ROCKE_OK;
            return;
        }
    }

    const char* override_path = std::getenv("ROCKE_HIP_LIB");
    if(override_path != nullptr && override_path[0] != '\0')
    {
        HMODULE handle = LoadLibraryA(override_path);
        if(handle == nullptr)
        {
            write_error(loaded_hip.error,
                        sizeof(loaded_hip.error),
                        "cannot load HIP runtime from ROCKE_HIP_LIB (Windows error %lu)",
                        static_cast<unsigned long>(GetLastError()));
            return;
        }
        if(!populate_api(handle, &loaded_hip.api))
        {
            write_error(loaded_hip.error,
                        sizeof(loaded_hip.error),
                        "HIP runtime from ROCKE_HIP_LIB is missing device query symbols");
            FreeLibrary(handle);
            return;
        }
        loaded_hip.handle = handle;
        loaded_hip.status = ROCKE_OK;
        return;
    }

    for(const char* candidate : candidates)
    {
        HMODULE handle = LoadLibraryA(candidate);
        if(handle == nullptr)
        {
            continue;
        }
        if(populate_api(handle, &loaded_hip.api))
        {
            loaded_hip.handle = handle;
            loaded_hip.status = ROCKE_OK;
            return;
        }
        FreeLibrary(handle);
    }
    write_error(loaded_hip.error,
                sizeof(loaded_hip.error),
                "cannot load a HIP runtime with device query symbols; set ROCKE_HIP_LIB");
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

void load_hip_api()
{
    ckc::HipApi process_api{};
    if(populate_api(RTLD_DEFAULT, &process_api))
    {
        loaded_hip.api = process_api;
        loaded_hip.status = ROCKE_OK;
        return;
    }

    const char* override_path = std::getenv("ROCKE_HIP_LIB");
    if(override_path != nullptr && override_path[0] != '\0')
    {
        void* handle = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
        if(handle == nullptr)
        {
            const char* detail = dlerror();
            write_error(loaded_hip.error,
                        sizeof(loaded_hip.error),
                        "cannot load HIP runtime from ROCKE_HIP_LIB: %s",
                        detail != nullptr ? detail : "unknown loader error");
            return;
        }
        if(!populate_api(handle, &loaded_hip.api))
        {
            write_error(loaded_hip.error,
                        sizeof(loaded_hip.error),
                        "HIP runtime from ROCKE_HIP_LIB is missing device query symbols");
            dlclose(handle);
            return;
        }
        loaded_hip.handle = handle;
        loaded_hip.status = ROCKE_OK;
        return;
    }

    constexpr const char* candidates[] = {"libamdhip64.so.7", "libamdhip64.so"};
    for(const char* candidate : candidates)
    {
        void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if(handle == nullptr)
        {
            continue;
        }
        if(populate_api(handle, &loaded_hip.api))
        {
            loaded_hip.handle = handle;
            loaded_hip.status = ROCKE_OK;
            return;
        }
        dlclose(handle);
    }
    write_error(loaded_hip.error,
                sizeof(loaded_hip.error),
                "cannot load a HIP runtime with device query symbols; set ROCKE_HIP_LIB");
}

#endif

rocke_status_t get_loaded_hip_api(const ckc::HipApi** out, char* err, std::size_t err_cap)
{
    std::call_once(loaded_hip_once, load_hip_api);
    if(loaded_hip.status != ROCKE_OK)
    {
        write_error(err, err_cap, "%s", loaded_hip.error);
        return loaded_hip.status;
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
