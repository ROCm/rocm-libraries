// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "init.hpp"
#include "status.hpp"

#include <hipblaslt/hipblaslt-version.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using hipblaslt_py::HipblasLtError;

#define HIPBLASLT_PY_STR2(x) #x
#define HIPBLASLT_PY_STR(x) HIPBLASLT_PY_STR2(x)

namespace
{
    bool hip_available()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    // hipblasLtGetVersion's encoding: major * 100000 + minor * 100 + patch.
    constexpr int built_version_int = HIPBLASLT_VERSION_MAJOR * 100000
                                      + HIPBLASLT_VERSION_MINOR * 100 + HIPBLASLT_VERSION_PATCH;

    std::pair<int, std::string> built_library_version()
    {
        return {built_version_int, HIPBLASLT_PY_STR(HIPBLASLT_VERSION_TWEAK)};
    }

    // hipblasLtGetVersion and hipblasLtGetGitRevision need a handle, and hipblasLtCreate exits
    // the process when there's no device, so this returns None without one.
    std::optional<std::pair<int, std::string>> loaded_library_version()
    {
        if(!hip_available())
            return std::nullopt;
        hipblasLtHandle_t handle = nullptr;
        HIPBLASLT_CHECK(hipblasLtCreate(&handle));
        int             version  = 0;
        char            rev[128] = {};
        hipblasStatus_t s1       = hipblasLtGetVersion(handle, &version);
        hipblasStatus_t s2       = hipblasLtGetGitRevision(handle, rev);
        hipblasStatus_t s3       = hipblasLtDestroy(handle);
        hipblaslt_py::check_status(s1, "hipblasLtGetVersion");
        hipblaslt_py::check_status(s2, "hipblasLtGetGitRevision");
        hipblaslt_py::check_status(s3, "hipblasLtDestroy");
        return std::make_pair(version, std::string(rev));
    }
} // namespace

NB_MODULE(_core, m)
{
    m.doc()               = "Low-level hipBLASLt bindings.";
    m.attr("__version__") = HIPBLASLT_PY_STR(HIPBLASLT_VERSION_MAJOR) "." HIPBLASLT_PY_STR(
        HIPBLASLT_VERSION_MINOR) "." HIPBLASLT_PY_STR(HIPBLASLT_VERSION_PATCH);

    nb::exception<HipblasLtError>(m, "HipblasLtError");

    m.def("hip_available", &hip_available, "Return True if at least one HIP device is visible.");
    m.def("_built_library_version",
          &built_library_version,
          "(version, git revision) of the hipBLASLt these bindings were compiled against.");
    m.def("_loaded_library_version",
          &loaded_library_version,
          "(version, git revision) of the hipBLASLt loaded in this process, or None without a "
          "device. Creates and destroys a handle, so import never calls it.");
    m.def(
        "_raise_test_status",
        [](int code) { hipblaslt_py::check_status_code(code, "_raise_test_status"); },
        "Test hook: raise HipblasLtError for a nonzero status value, which needn't be one the "
        "installed header has.");

    init_enums(m);
    init_descriptors(m);
    init_device_array(m);
}
