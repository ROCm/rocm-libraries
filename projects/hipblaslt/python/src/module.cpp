// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>
#include "status.hpp"
#include "init.hpp"

namespace nb = nanobind;
using hipblaslt_py::HipblasLtError;

static bool hip_available()
{
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

NB_MODULE(_core, m)
{
    m.doc() = "Low-level hipBLASLt bindings.";
    m.attr("__version__") = "0.1.0";
    m.def("hip_available", &hip_available,
          "Return True if at least one HIP device is visible.");

    nb::exception<HipblasLtError>(m, "HipblasLtError");

    m.def("_raise_test_status", [](int code) {
        hipblaslt_py::check_status(static_cast<hipblasStatus_t>(code), "_raise_test_status");
    }, "Debug hook: raise HipblasLtError for a nonzero status code.");

    init_enums(m);
    init_device_array(m);
    init_descriptors(m);
}
