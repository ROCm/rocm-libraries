// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "bindings.hpp"

#include <nanobind/nanobind.h>

#include <roc/hostnumerics/version.hpp>

namespace nb = nanobind;

NB_MODULE(_hostnumerics, module) {
    module.attr("__version__") = roc::hostnumerics::version().data();
    roc::hostnumerics::python_bindings::registerCoreBindings(module);
    roc::hostnumerics::python_bindings::registerGenerationBindings(module);
    roc::hostnumerics::python_bindings::registerMxBindings(module);
    roc::hostnumerics::python_bindings::registerSelectionBindings(module);
    roc::hostnumerics::python_bindings::registerTensorBindings(module);
    roc::hostnumerics::python_bindings::registerGemmBindings(module);
    roc::hostnumerics::python_bindings::registerComparisonBindings(module);
    roc::hostnumerics::python_bindings::registerOperationBindings(module);
}
