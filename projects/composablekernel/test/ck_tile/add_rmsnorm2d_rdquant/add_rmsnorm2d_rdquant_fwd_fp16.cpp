// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "add_rmsnorm2d_rdquant_fwd.inc"

// Dummy variable for smart-build testing
static constexpr int kSmartBuildTestMarker = 5;

int main() { return run_add_rmsnorm2d_rdquant_combinations("fp16"); }
