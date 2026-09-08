// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics::python_bindings {
Tensor scalarFromPython(nanobind::handle value);

void registerComparisonBindings(nanobind::module_& module);
void registerGenerationBindings(nanobind::module_& module);
void registerGemmBindings(nanobind::module_& module);
void registerMxBindings(nanobind::module_& module);
void registerOperationBindings(nanobind::module_& module);
}  // namespace roc::host_numerics::python_bindings
