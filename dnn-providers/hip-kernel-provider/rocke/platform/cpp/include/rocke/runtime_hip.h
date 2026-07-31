/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/runtime_hip.h -- optional HIP device-target discovery for the C ABI.
 *
 * Python (runtime/hip_module.py)       C99 (this file)
 * ----------------------------------   ------------------------------------
 * get_device_arch(device)          ->  rocke_hip_get_device_arch()
 * hipGetDevice + get_device_arch   ->  rocke_hip_get_current_device_arch()
 * explicit arch or runtime query   ->  rocke_resolve_compile_target()
 */
#ifndef ROCKE_RUNTIME_HIP_H
#define ROCKE_RUNTIME_HIP_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arch_target.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A validated compile target. `target` borrows program-lifetime storage from
 * the architecture catalog. `device` is a HIP-visible ordinal for runtime
 * resolution and -1 for an explicit compile target. */
typedef struct rocke_resolved_target
{
    const rocke_arch_target_t* target;
    int device;
    bool from_runtime;
} rocke_resolved_target_t;

/* Query one HIP-visible device ordinal and write its suffix-free gfx token.
 * The caller owns `out_gfx`; on failure it is left empty and a bounded
 * diagnostic is written to `err` when provided. */
rocke_status_t rocke_hip_get_device_arch(
    int device, char* out_gfx, size_t out_gfx_cap, char* err, size_t err_cap);

/* Query the calling thread's current HIP-visible device and its gfx token.
 * HIP owns visibility/remapping (including HIP_VISIBLE_DEVICES). */
rocke_status_t rocke_hip_get_current_device_arch(
    int* out_device, char* out_gfx, size_t out_gfx_cap, char* err, size_t err_cap);

/* Resolve a compile target. A non-NULL `requested_gfx` is authoritative,
 * validated without loading HIP, and recorded with device=-1. NULL queries the
 * current HIP device. On failure `out` is untouched. */
rocke_status_t rocke_resolve_compile_target(const char* requested_gfx,
                                            rocke_resolved_target_t* out,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_RUNTIME_HIP_H */
