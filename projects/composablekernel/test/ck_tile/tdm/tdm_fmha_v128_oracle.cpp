// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Keep device copies and CPU validation out of the performance executable.
#define CK_TILE_FMHA_TDM_V128_TEST_ORACLE 1
#include "tdm_fmha_v128_fwd.cpp"
