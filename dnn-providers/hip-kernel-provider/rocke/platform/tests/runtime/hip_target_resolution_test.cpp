// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hip_target_resolution_test.cpp -- deterministic host tests for optional HIP
 * target discovery and explicit compile-target resolution.
 */
#include "rocke/runtime_hip.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "rocke/runtime_hip_internal.hpp"

namespace
{

std::atomic<int> get_device_calls{0};
std::atomic<int> get_properties_calls{0};
std::atomic<int> failures{0};
std::atomic<int> get_device_status{0};
std::atomic<int> get_properties_status{0};
thread_local int current_device = 0;
const char* device_arches[] = {"gfx942:sramecc+:xnack-", "gfx950", "", "gfx950"};

void expect(bool condition, const char* message)
{
    if(!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

int fake_get_device(int* device)
{
    ++get_device_calls;
    int status = get_device_status.load();
    if(status == 0)
    {
        *device = current_device;
    }
    return status;
}

int fake_get_properties(void* raw, int device)
{
    ++get_properties_calls;
    int status = get_properties_status.load();
    if(status != 0)
    {
        return status;
    }
    if(device < 0 || device >= static_cast<int>(sizeof(device_arches) / sizeof(device_arches[0])))
    {
        return 101;
    }
    const char* arch = device_arches[device];
    std::memcpy(static_cast<unsigned char*>(raw) + 512, arch, std::strlen(arch) + 1);
    return 0;
}

const char* fake_error_string(int status)
{
    return status == 77 ? "fake HIP failure" : "fake HIP status";
}

const ckc::HipApi fake_api{fake_get_device, fake_get_properties, fake_error_string};

void reset_fake()
{
    get_device_calls = 0;
    get_properties_calls = 0;
    get_device_status = 0;
    get_properties_status = 0;
    current_device = 0;
}

void test_explicit_target_does_not_touch_hip()
{
    reset_fake();
    rocke_resolved_target_t out{};
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t status
        = ckc::resolve_compile_target_with_api(&fake_api, "gfx942", &out, err, sizeof(err));
    expect(status == ROCKE_OK, "explicit gfx942 target resolves");
    expect(out.target != nullptr && std::strcmp(out.target->gfx, "gfx942") == 0,
           "explicit target returns catalog object");
    expect(out.device == -1 && !out.from_runtime, "explicit target records its provenance");
    expect(get_device_calls == 0 && get_properties_calls == 0, "explicit target does not call HIP");

    status = ckc::resolve_compile_target_with_api(&fake_api, "", &out, err, sizeof(err));
    expect(status == ROCKE_ERR_VALUE, "empty explicit target is rejected");
    status = ckc::resolve_compile_target_with_api(
        &fake_api, "gfx_not_in_catalog", &out, err, sizeof(err));
    expect(status == ROCKE_ERR_KEY, "unknown explicit target is rejected");
}

void test_current_visible_device_and_suffix()
{
    reset_fake();
    current_device = 3;
    int device = -1;
    char gfx[32];
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t status = ckc::hip_get_current_device_arch_with_api(
        fake_api, &device, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_OK, "current HIP-visible device resolves");
    expect(device == 3, "current HIP-visible ordinal is preserved");
    expect(std::strcmp(gfx, "gfx950") == 0, "current device architecture is returned");

    current_device = 0;
    status = ckc::hip_get_current_device_arch_with_api(
        fake_api, &device, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_OK, "suffixed architecture resolves");
    expect(std::strcmp(gfx, "gfx942") == 0, "HIP feature suffix is stripped");

    rocke_resolved_target_t target{};
    status = ckc::resolve_compile_target_with_api(&fake_api, nullptr, &target, err, sizeof(err));
    expect(status == ROCKE_OK, "automatic target resolves through current device");
    expect(target.device == 0 && target.from_runtime, "runtime target records visible ordinal");
    expect(target.target != nullptr && std::strcmp(target.target->gfx, "gfx942") == 0,
           "runtime target is catalog validated");
}

void test_runtime_failures()
{
    reset_fake();
    int device = -1;
    char gfx[32];
    char err[ROCKE_ERR_MSG_CAP];

    get_device_status = 77;
    rocke_status_t status = ckc::hip_get_current_device_arch_with_api(
        fake_api, &device, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME, "hipGetDevice failure is a HIP runtime error");
    expect(std::strstr(err, "fake HIP failure") != nullptr,
           "hipGetDevice failure includes HIP detail");

    reset_fake();
    get_properties_status = 77;
    status = ckc::hip_get_device_arch_with_api(fake_api, 0, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME, "property failure is a HIP runtime error");
    expect(std::strstr(err, "hipGetDeviceProperties") != nullptr,
           "property failure names the operation");

    reset_fake();
    status = ckc::hip_get_device_arch_with_api(fake_api, 2, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME, "missing gfx token is a HIP runtime error");

    reset_fake();
    char tiny[4];
    status = ckc::hip_get_device_arch_with_api(fake_api, 0, tiny, sizeof(tiny), err, sizeof(err));
    expect(status == ROCKE_ERR_VALUE, "too-small architecture output is rejected");
}

int throwing_get_device(int*)
{
    throw std::runtime_error("injected hipGetDevice exception");
}

int throwing_get_properties(void*, int)
{
    throw 7;
}

int failing_get_properties(void*, int)
{
    return 77;
}

const char* throwing_error_string(int)
{
    throw std::runtime_error("injected hipGetErrorString exception");
}

void test_callback_exceptions_are_translated()
{
    int device = -1;
    char gfx[32];
    char err[ROCKE_ERR_MSG_CAP];

    const ckc::HipApi get_device_api{throwing_get_device, fake_get_properties, fake_error_string};
    rocke_status_t status = ckc::hip_get_current_device_arch_with_api(
        get_device_api, &device, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME,
           "throwing hipGetDevice callback becomes a HIP runtime error");
    expect(std::strstr(err, "injected hipGetDevice exception") != nullptr,
           "hipGetDevice callback exception is diagnosed");

    const ckc::HipApi get_properties_api{
        fake_get_device, throwing_get_properties, fake_error_string};
    status = ckc::hip_get_device_arch_with_api(
        get_properties_api, 0, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME,
           "non-standard property callback exception becomes a HIP runtime error");
    expect(std::strstr(err, "unknown C++ exception") != nullptr,
           "non-standard property callback exception is diagnosed");

    const ckc::HipApi error_string_api{
        fake_get_device, failing_get_properties, throwing_error_string};
    status = ckc::hip_get_device_arch_with_api(
        error_string_api, 0, gfx, sizeof(gfx), err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME,
           "throwing hipGetErrorString callback becomes a HIP runtime error");
    expect(std::strstr(err, "injected hipGetErrorString exception") != nullptr,
           "hipGetErrorString callback exception is diagnosed");

    rocke_resolved_target_t target{};
    status
        = ckc::resolve_compile_target_with_api(&get_device_api, nullptr, &target, err, sizeof(err));
    expect(status == ROCKE_ERR_HIP_RUNTIME,
           "compile-target resolution propagates a callback exception as status");
}

int unsupported_get_properties(void* raw, int device)
{
    (void)device;
    constexpr const char* arch = "gfx9999";
    std::memcpy(static_cast<unsigned char*>(raw) + 512, arch, std::strlen(arch) + 1);
    return 0;
}

void test_unsupported_runtime_target()
{
    reset_fake();
    const ckc::HipApi unsupported_api{
        fake_get_device, unsupported_get_properties, fake_error_string};
    rocke_resolved_target_t target{};
    char err[ROCKE_ERR_MSG_CAP];
    rocke_status_t status = ckc::resolve_compile_target_with_api(
        &unsupported_api, nullptr, &target, err, sizeof(err));
    expect(status == ROCKE_ERR_NOTIMPL, "unsupported runtime gfx is not implemented");
    expect(std::strstr(err, "gfx9999") != nullptr, "unsupported runtime gfx is diagnosed");
}

void concurrent_query(int device, const char* expected)
{
    current_device = device;
    for(int i = 0; i < 1000; ++i)
    {
        int resolved_device = -1;
        char gfx[32];
        char err[ROCKE_ERR_MSG_CAP];
        rocke_status_t status = ckc::hip_get_current_device_arch_with_api(
            fake_api, &resolved_device, gfx, sizeof(gfx), err, sizeof(err));
        if(status != ROCKE_OK || resolved_device != device || std::strcmp(gfx, expected) != 0)
        {
            ++failures;
            return;
        }
    }
}

void test_concurrent_current_devices()
{
    reset_fake();
    std::thread first(concurrent_query, 0, "gfx942");
    std::thread second(concurrent_query, 1, "gfx950");
    first.join();
    second.join();
    expect(get_device_calls == 2000 && get_properties_calls == 2000,
           "current device is queried for every concurrent resolution");
}

} /* namespace */

int main()
{
    test_explicit_target_does_not_touch_hip();
    test_current_visible_device_and_suffix();
    test_runtime_failures();
    test_callback_exceptions_are_translated();
    test_unsupported_runtime_target();
    test_concurrent_current_devices();
    if(failures != 0)
    {
        std::fprintf(stderr, "%d HIP target-resolution test(s) failed\n", failures.load());
        return 1;
    }
    std::puts("HIP target-resolution host tests passed");
    return 0;
}
