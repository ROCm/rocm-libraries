# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Throwaway harness so the install-tree assertion exercises the REAL install()
# rule. It enables the production pack+install against a fixture source root
# (HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT passed in by StageInstall.cmake) and
# is never part of the outer provider build. HIPKERNELPROVIDER_ENABLE_TESTS is
# forced OFF so the packaging module wires only production, not its own tests
# (no recursion).

cmake_minimum_required(VERSION 3.25.2)
project(hkp_install_harness NONE)

set(HIPKERNELPROVIDER_ENABLE_TESTS OFF)

# The packaging module lives one directory up from this harness.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")
include(HkpPackaging)

hkp_add_packaging()
