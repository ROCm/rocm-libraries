// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "bindings.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_roc_host_numerics, module) {
    roc::host_numerics::python_bindings::registerCoreBindings(module);
    roc::host_numerics::python_bindings::registerGenerationBindings(module);
    roc::host_numerics::python_bindings::registerMxBindings(module);
    roc::host_numerics::python_bindings::registerSelectionBindings(module);
    roc::host_numerics::python_bindings::registerTensorBindings(module);
    roc::host_numerics::python_bindings::registerGemmBindings(module);
    roc::host_numerics::python_bindings::registerComparisonBindings(module);
    roc::host_numerics::python_bindings::registerOperationBindings(module);
}
