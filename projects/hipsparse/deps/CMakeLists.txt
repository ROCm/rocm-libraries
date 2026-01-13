# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(hipsparse_deps_dummy LANGUAGES NONE)

# No-op configure/build to satisfy transitional build system
message(STATUS "hipSPARSE deps: transitional no-op build. Nothing will be built.")
add_custom_target(dummy_all ALL
    COMMENT "No-op: nothing to build for hipSPARSE external dependencies"
)

# Provide a successful no-op install so `make install` returns zero
install(CODE [[
    message(STATUS "hipSPARSE deps: no-op install. Nothing to install.")
]])
