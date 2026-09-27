// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <roc/host_numerics/version.hpp>

namespace nb = nanobind;

NB_MODULE(_roc_host_numerics, module) {
    module.attr("__version__") = roc::host_numerics::version().data();
}
