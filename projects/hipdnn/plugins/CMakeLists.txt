# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Extra flag to be used for the plugins when building them as part of the main hipdnn build
option(BUILD_PLUGIN_AS_DEPENDENCY "Flag to build the plugins as part of the main hipdnn build" ON)

add_subdirectory(miopen_legacy_plugin)
