// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pooling2d.hpp"

int main(int argc, const char* argv[]) { test_drive<pooling2d_driver<bfloat16>>(argc, argv); }
