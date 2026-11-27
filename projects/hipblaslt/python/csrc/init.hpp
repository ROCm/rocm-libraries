// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <nanobind/nanobind.h>

#include <string>

void init_enums(nanobind::module_& m);
void init_descriptors(nanobind::module_& m);
void init_device_array(nanobind::module_& m);

// True if `value` is a bound value of the C enum `c_enum` (as named in enums.cpp).
bool enum_has_value(const std::string& c_enum, long long value);
