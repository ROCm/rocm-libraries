// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <hip/hip_runtime.h>

namespace nb = nanobind;

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
}
