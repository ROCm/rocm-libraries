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
