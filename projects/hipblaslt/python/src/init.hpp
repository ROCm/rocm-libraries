// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <nanobind/nanobind.h>

void init_enums(nanobind::module_& m);
void init_device_array(nanobind::module_& m);   // Phase 2
void init_descriptors(nanobind::module_& m);    // Phase 3
void init_matmul(nanobind::module_& m);         // Phase 3
void init_convert(nanobind::module_& m);        // Task 15: fp8 pack/unpack
