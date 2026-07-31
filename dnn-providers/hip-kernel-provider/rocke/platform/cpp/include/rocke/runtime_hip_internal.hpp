// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/runtime_hip_internal.hpp -- injectable HIP function table used by the
 * optional runtime loader and deterministic host tests.
 */
#ifndef ROCKE_RUNTIME_HIP_INTERNAL_HPP
#define ROCKE_RUNTIME_HIP_INTERNAL_HPP

#include <cstddef>

#include "rocke/runtime_hip.h"

namespace ckc
{

using hip_get_device_fn = int (*)(int*);
using hip_get_device_properties_fn = int (*)(void*, int);
using hip_get_error_string_fn = const char* (*)(int);

struct HipApi
{
    hip_get_device_fn get_device;
    hip_get_device_properties_fn get_device_properties;
    hip_get_error_string_fn get_error_string;
};

rocke_status_t hip_get_device_arch_with_api(const HipApi& api,
                                            int device,
                                            char* out_gfx,
                                            std::size_t out_gfx_cap,
                                            char* err,
                                            std::size_t err_cap) noexcept;

rocke_status_t hip_get_current_device_arch_with_api(const HipApi& api,
                                                    int* out_device,
                                                    char* out_gfx,
                                                    std::size_t out_gfx_cap,
                                                    char* err,
                                                    std::size_t err_cap) noexcept;

rocke_status_t resolve_compile_target_with_api(const HipApi* api,
                                               const char* requested_gfx,
                                               rocke_resolved_target_t* out,
                                               char* err,
                                               std::size_t err_cap) noexcept;

} /* namespace ckc */

#endif /* ROCKE_RUNTIME_HIP_INTERNAL_HPP */
