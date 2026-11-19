// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Profiling hooks that the private library exports for the hipDNN forwarding
// code in the wrapper. A forwarded call runs its kernels through hipDNN, not
// through the caller's MIOpen handle, so the wrapper times the call itself and
// stores the result on the handle, where miopenGetKernelTime reads it.
//
// They are not in miopen.h and have no _impl form, so the wrapper does not
// re-export them (check_public_abi.py check-wrapper). They are defined in
// handle_api.cpp only when MIOPEN_BUILDING_PRIVATE is set, so a build without
// MIOPEN_ENABLE_HIPDNN_WRAPPER exports nothing new. This header is not
// installed; it is the one declaration both libraries compile against.
#pragma once

#include <miopen/miopen.h>

#ifdef MIOPEN_BUILDING_PRIVATE
#define MIOPEN_WRAPPER_HOOK_EXPORT MIOPEN_EXPORT
#else
#define MIOPEN_WRAPPER_HOOK_EXPORT
#endif

// Reports whether profiling is enabled on `handle` (miopenEnableProfiling).
extern "C" MIOPEN_WRAPPER_HOOK_EXPORT miopenStatus_t
miopenWrapperIsProfilingEnabled(miopenHandle_t handle, bool* enabled);

// Sets the time miopenGetKernelTime next reports for `handle`, replacing the
// previous value as a natively run call does.
extern "C" MIOPEN_WRAPPER_HOOK_EXPORT miopenStatus_t
miopenWrapperSetKernelTime(miopenHandle_t handle, float time);
