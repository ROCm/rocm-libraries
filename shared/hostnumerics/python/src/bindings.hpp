// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

#include <roc/hostnumerics/tensor.hpp>

namespace roc::hostnumerics::python_bindings {
Tensor scalarFromPython(nanobind::handle value);
Tensor tensorOperand(nanobind::handle value, ScalarType scalarType);

void registerCoreBindings(nanobind::module_& module);
void registerSelectionBindings(nanobind::module_& module);
void registerComparisonBindings(nanobind::module_& module);
void registerGenerationBindings(nanobind::module_& module);
void registerGemmBindings(nanobind::module_& module);
void registerMxBindings(nanobind::module_& module);
void registerOperationBindings(nanobind::module_& module);
void registerTensorBindings(nanobind::module_& module);
}  // namespace roc::hostnumerics::python_bindings
