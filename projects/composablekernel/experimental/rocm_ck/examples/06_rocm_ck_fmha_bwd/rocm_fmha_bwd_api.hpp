// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unified API header for all FMHA BWD kernel families.
//
// HOST ONLY: includes the per-kernel _api.hpp headers which have
// #error on __HIP_DEVICE_COMPILE__. Device code should include
// the per-kernel _dev.hpp headers directly.
//
// This header has NO CK Tile dependency.

#pragma once

// Spec headers (shared — consteval, device-safe)
#include "rocm_fmha_bwd_ograd_dot_o_spec.hpp"
#include "rocm_fmha_bwd_dqdkdv_spec.hpp"
#include "rocm_fmha_bwd_convert_dq_spec.hpp"
// API headers (host-only — grid_size, future launch helpers)
#include "rocm_fmha_bwd_ograd_dot_o_api.hpp"
#include "rocm_fmha_bwd_dqdkdv_api.hpp"
#include "rocm_fmha_bwd_convert_dq_api.hpp"
