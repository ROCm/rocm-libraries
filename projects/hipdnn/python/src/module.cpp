// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <HipdnnBackendPluginLoadingMode.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(hipdnn_frontend_python, m) // NOLINT(modernize-avoid-c-arrays)
{
    m.doc() = "Python bindings for the hipDNN frontend library";

    // Initialize bindings from other files
    typesBindings(m); // Types and enums first
    handleBindings(m); // Handle management
    memoryBindings(m); // Memory management
    tensorBindings(m); // Then tensor attributes
    attributesBindings(m); // Then node attributes
    graphBindings(m); // Finally graph which uses all of the above

    // Module-level function to set engine plugin paths
    m.def(
        "set_engine_plugin_paths",
        [](const std::vector<std::string>& paths,
           hipdnnPluginLoadingMode_ext_t mode = HIPDNN_DEFAULT_PLUGIN_LOADING_MODE) {
            // Convert vector<string> to const char* const*
            std::vector<const char*> cPaths;
            cPaths.reserve(paths.size());
            for(const auto& path : paths)
            {
                cPaths.push_back(path.c_str());
            }

            const auto status = hipdnnSetEnginePluginPaths_ext(cPaths.size(), cPaths.data(), mode);

            if(status != HIPDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to set engine plugin paths");
            }
        },
        nb::arg("paths"),
        nb::arg("mode") = HIPDNN_DEFAULT_PLUGIN_LOADING_MODE,
        "Set custom engine plugin paths. Must be called before creating any handles.");

    m.def(
        "engine_id_to_name",
        [](int64_t id) -> std::string {
            try
            {
                return std::string(hipdnn_data_sdk::utilities::getEngineNameFromId(id));
            }
            catch(const std::out_of_range&)
            {
                return std::string{};
            }
        },
        nb::arg("engine_id"),
        "Look up the registered engine name for a given engine ID. "
        "Returns an empty string if the ID is not registered.");
}
