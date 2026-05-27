// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <nanobind/nanobind.h>

void graphBindings(nanobind::module_& m);
void tensorBindings(nanobind::module_& m);
void attributesBindings(nanobind::module_& m);
void typesBindings(nanobind::module_& m);
void handleBindings(nanobind::module_& m);
void memoryBindings(nanobind::module_& m);
