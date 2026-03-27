// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Vector add host-only API for rocm_ck elementwise example.
//
// HOST ONLY: arg assembly, launch helpers, host-side validation.
// This header must NOT be included from .hip files (device compilation).
// Device code includes rocm_vector_add_dev.hpp; shared types and make_kernel()
// live in rocm_vector_add_spec.hpp.
//
// Compilation boundary:
//   _spec.hpp — schema types + consteval factory (both passes)
//   _api.hpp (this) — host-only helpers (host pass only, #error on device)
//   _dev.hpp     — CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#error "rocm_vector_add_api.hpp is host-only. Device code should include rocm_vector_add_dev.hpp."
#endif

#include <rocm_ck/elementwise_spec.hpp>

namespace rocm_ck {

// Host-only runtime code goes here — anything that needs the standard library
// or HIP host runtime and cannot survive --cuda-device-only compilation:
//
//   - make_args():  named builder for the generic Args struct
//                   (e.g., make_args(kernel).tensor("A", ptr, {N}).scalar("alpha", 1.0f))
//   - launch():     compute grid size from kernel descriptor + problem size,
//                   call hipModuleLaunchKernel
//   - validate():   runtime checks on problem dimensions, alignment, dtype match
//   - Error types:  rich error reporting with std::string, std::optional
//
// Metaprogramming (consteval factories, structural types, static_asserts)
// lives in rocm_vector_add_spec.hpp, which is shared with device code.

} // namespace rocm_ck
