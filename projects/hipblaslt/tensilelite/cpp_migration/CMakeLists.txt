# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build for the TensileLite "tensile_writer" C++ migration scaffold.
#
# This extension is intentionally dependency-light: the subtile geometry layer
# is pure integer/float math, so it needs only nanobind + Python (no HIP, no
# rocisa, no origami). That keeps it fast to build inside hipblaslt-dev:latest
# and independent of the heavy hipblaslt build.

cmake_minimum_required(VERSION 3.25)

project(tensile_writer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# scikit-build-core pre-installs nanobind (from build-system.requires) and
# injects its CMake path, so find_package works without FetchContent.
# nanobind requires Python to be found first.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tensile_writer NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/main.cpp")

target_include_directories(_tensile_writer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")

# Install the compiled extension into the tensile_writer package directory in
# the wheel (the pure-Python package files are picked up by wheel.packages).
install(TARGETS _tensile_writer LIBRARY DESTINATION tensile_writer)

# ---------------------------------------------------------------------------
# C++ unit tests (cpp/tests) — intentionally NOT wired in here.
#
# The gtest target under cpp/tests/ (cpp/tests/CMakeLists.txt) is a deliberately
# standalone build: the subtile geometry / tile_info / emit_leaves headers are
# pure integer/float math with no nanobind, Python, HIP, or rocisa dependency,
# so the tests only need a C++20 compiler and GTest. We do NOT add it via
# add_subdirectory() here on purpose:
#
#   * This CMakeLists is the scikit-build-core wheel-build entry point (driven by
#     pip / pyproject.toml), so it requires Python + nanobind. Pulling the gtest
#     target into it would (a) drag GTest into the wheel/pip build surface and
#     (b) couple the dependency-light test build to the nanobind extension build
#     — exactly what the standalone test layout was designed to avoid.
#   * The wheel build never enables a test flag, so an add_subdirectory guard
#     would not actually surface the tests to CI anyway.
#
# The tests are therefore an explicit, opt-in build. Run them directly (verified
# in hipblaslt-dev:latest, no GPU required):
#
#   cd projects/hipblaslt/tensilelite/cpp_migration
#   cmake -S cpp/tests -B build_tests -G Ninja
#   cmake --build build_tests
#   ctest --test-dir build_tests --output-on-failure
#
# CI / new C++ test wiring: add new gtest sources to cpp/tests/CMakeLists.txt and
# invoke the standalone build above; do not expect them to be picked up through
# this wheel-build CMakeLists.
# ---------------------------------------------------------------------------
