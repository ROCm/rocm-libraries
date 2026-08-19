// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

namespace roc::host_validation::python_bindings {
void registerGenerationBindings(nanobind::module_& module);
void registerMxBindings(nanobind::module_& module);
}  // namespace roc::host_validation::python_bindings
