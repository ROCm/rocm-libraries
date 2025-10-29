# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(hipsparselt_deps_dummy LANGUAGES NONE)

# No-op configure/build to satisfy transitional build system
message(STATUS "hipSPARSELt deps: transitional no-op build. Nothing will be built.")
add_custom_target(dummy_all ALL
  COMMENT "No-op: nothing to build for hipSPARSELt external dependencies")

# Provide a successful no-op install so `make install` returns zero
install(CODE [[
  message(STATUS "hipSPARSELt deps: no-op install. Nothing to install.")
]])