// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hip_loader_integration_test.cpp -- process-isolated tests for HIP library
 * discovery, precedence, retry, and module ownership.
 */
#include "rocke/runtime_hip.h"

#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{

bool set_env(const char* name, const char* value)
{
#if defined(_WIN32)
    return _putenv_s(name, value != nullptr ? value : "") == 0;
#else
    return value != nullptr ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
}

void clear_discovery_env()
{
    (void)set_env("ROCKE_HIP_LIB", nullptr);
    (void)set_env("HIP_PATH", nullptr);
    (void)set_env("ROCM_PATH", nullptr);
    (void)set_env("ROCM_HOME", nullptr);
}

bool expect_arch(const char* expected)
{
    char gfx[32];
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t status = rocke_hip_get_device_arch(0, gfx, sizeof(gfx), err, sizeof(err));
    if(status != ROCKE_OK)
    {
        std::fprintf(stderr, "HIP query failed with status %d: %s\n", status, err);
        return false;
    }
    if(std::strcmp(gfx, expected) != 0)
    {
        std::fprintf(stderr, "expected architecture %s, got %s\n", expected, gfx);
        return false;
    }
    return true;
}

#if defined(_WIN32)
using module_handle_t = HMODULE;

module_handle_t load_global(const char* path)
{
    return LoadLibraryA(path);
}

void unload(module_handle_t handle)
{
    if(handle != nullptr)
    {
        FreeLibrary(handle);
    }
}
#else
using module_handle_t = void*;

module_handle_t load_global(const char* path)
{
    return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
}

void unload(module_handle_t handle)
{
    if(handle != nullptr)
    {
        dlclose(handle);
    }
}
#endif

int test_override_precedence(const char* preloaded_path, const char* override_path)
{
    clear_discovery_env();
    module_handle_t preloaded = load_global(preloaded_path);
    if(preloaded == nullptr)
    {
        std::fprintf(stderr, "cannot preload fake HIP runtime\n");
        return 1;
    }
    if(!set_env("ROCKE_HIP_LIB", override_path))
    {
        std::fprintf(stderr, "cannot set ROCKE_HIP_LIB\n");
        unload(preloaded);
        return 1;
    }
    bool ok = expect_arch("gfx950");
    unload(preloaded);
    return ok ? 0 : 1;
}

int test_retry(const char* valid_path)
{
    clear_discovery_env();
    std::string missing = std::string(valid_path) + ".missing";
    if(!set_env("ROCKE_HIP_LIB", missing.c_str()))
    {
        return 1;
    }

    rocke_resolved_target_t explicit_target{};
    char explicit_err[ROCKE_ERR_MSG_CAP];
    rocke_status_t explicit_status = rocke_resolve_compile_target(
        "gfx942", &explicit_target, explicit_err, sizeof(explicit_err));
    if(explicit_status != ROCKE_OK || explicit_target.target == nullptr
       || std::strcmp(explicit_target.target->gfx, "gfx942") != 0 || explicit_target.device != -1
       || explicit_target.from_runtime)
    {
        std::fprintf(stderr,
                     "explicit target unexpectedly used HIP: %d %s\n",
                     explicit_status,
                     explicit_err);
        return 1;
    }

    char gfx[32];
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t first = rocke_hip_get_device_arch(0, gfx, sizeof(gfx), err, sizeof(err));
    if(first != ROCKE_ERR_HIP_RUNTIME || std::strstr(err, "ROCKE_HIP_LIB") == nullptr)
    {
        std::fprintf(stderr, "first load did not fail through ROCKE_HIP_LIB: %d %s\n", first, err);
        return 1;
    }
    if(!set_env("ROCKE_HIP_LIB", valid_path))
    {
        return 1;
    }
    return expect_arch("gfx950") ? 0 : 1;
}

int test_root_discovery(const char* root)
{
    clear_discovery_env();
#if defined(_WIN32)
    if(!set_env("HIP_PATH", root))
#else
    if(!set_env("ROCM_PATH", root))
#endif
    {
        return 1;
    }
    return expect_arch("gfx942") ? 0 : 1;
}

int test_loaded_runtime_pin(const char* preloaded_path)
{
    clear_discovery_env();
    module_handle_t preloaded = load_global(preloaded_path);
    if(preloaded == nullptr)
    {
        std::fprintf(stderr, "cannot preload fake HIP runtime\n");
        return 1;
    }
    if(!expect_arch("gfx942"))
    {
        unload(preloaded);
        return 1;
    }
    unload(preloaded);
    return expect_arch("gfx942") ? 0 : 1;
}

} /* namespace */

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::fprintf(stderr, "usage: hip_loader_integration_test SCENARIO PATH [PATH]\n");
        return 2;
    }
    if(std::strcmp(argv[1], "override") == 0 && argc == 4)
    {
        return test_override_precedence(argv[2], argv[3]);
    }
    if(std::strcmp(argv[1], "retry") == 0 && argc == 3)
    {
        return test_retry(argv[2]);
    }
    if(std::strcmp(argv[1], "root") == 0 && argc == 3)
    {
        return test_root_discovery(argv[2]);
    }
    if(std::strcmp(argv[1], "pin") == 0 && argc == 3)
    {
        return test_loaded_runtime_pin(argv[2]);
    }
    std::fprintf(stderr, "invalid HIP loader test arguments\n");
    return 2;
}
