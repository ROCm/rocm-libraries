# ************************************************************************
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ************************************************************************

cmake_minimum_required(VERSION 3.16)

message(STATUS "Configuring StinkyTofu Python Module")

# ============================================================================
# Find Python and nanobind
# ============================================================================

# nanobind requires find_package(Python ...) not Python3
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find nanobind via pip/site-packages
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_RESULT
)

if(nanobind_RESULT EQUAL 0)
    message(STATUS "Found nanobind at: ${nanobind_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
    find_package(nanobind CONFIG REQUIRED)
else()
    message(WARNING "nanobind not found. Python bindings will not be built.")
    message(WARNING "Install nanobind with: pip install nanobind")
    return()
endif()

# ============================================================================
# Create the StinkyTofu C++ Bridge Library
# ============================================================================

# This library provides the C++ API that will be exposed to Python
add_library(stinkytofu_bridge STATIC
    src/StinkyTofu.cpp
    src/StinkyIR.cpp
)

target_include_directories(stinkytofu_bridge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/shared/stinkytofu/include
        ${CMAKE_SOURCE_DIR}/shared/stinkytofu/hardware/include
        # For generated files like gfxIsa.inc
        ${TABLEGEN_OUTPUT_DIR}
)

target_link_libraries(stinkytofu_bridge
    PUBLIC
        stinkytofu
        gfxisa
)

# Ensure generated files are created before building the bridge
add_dependencies(stinkytofu_bridge tablegen_generated)

# Enable RTTI, exceptions, and PIC for the bridge library
# (required for nanobind, std::runtime_error, and linking into shared module)
target_compile_options(stinkytofu_bridge PRIVATE
    -frtti
    -fexceptions
    -fPIC
)

# Remove -fno-rtti and -fno-exceptions if they were inherited
get_target_property(BRIDGE_COMPILE_OPTIONS stinkytofu_bridge COMPILE_OPTIONS)
if(BRIDGE_COMPILE_OPTIONS)
    list(REMOVE_ITEM BRIDGE_COMPILE_OPTIONS "-fno-rtti" "-fno-exceptions")
    set_target_properties(stinkytofu_bridge PROPERTIES COMPILE_OPTIONS "${BRIDGE_COMPILE_OPTIONS}")
endif()

# ============================================================================
# Create the Python Module
# ============================================================================

# Temporarily remove global compile options to avoid inheriting -fno-rtti -fno-exceptions
get_directory_property(SAVED_COMPILE_OPTIONS COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")

nanobind_add_module(stinkytofu_python
    NB_STATIC  # Build as static module
    src/python_bindings.cpp
)

# Restore global compile options
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${SAVED_COMPILE_OPTIONS}")

target_link_libraries(stinkytofu_python PRIVATE
    stinkytofu_bridge
)

# Set output name to just "stinkytofu" (without the _python suffix)
set_target_properties(stinkytofu_python PROPERTIES
    OUTPUT_NAME stinkytofu
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_module
)

# Enable RTTI and exceptions for the Python bindings
target_compile_options(stinkytofu_python PRIVATE
    -frtti
    -fexceptions
)

# Also ensure nanobind-static target has RTTI and exceptions enabled
if(TARGET nanobind-static)
    get_target_property(NB_COMPILE_OPTIONS nanobind-static COMPILE_OPTIONS)
    if(NB_COMPILE_OPTIONS)
        list(REMOVE_ITEM NB_COMPILE_OPTIONS "-fno-rtti" "-fno-exceptions")
        set_target_properties(nanobind-static PROPERTIES COMPILE_OPTIONS "${NB_COMPILE_OPTIONS}")
    endif()
    target_compile_options(nanobind-static PRIVATE -frtti -fexceptions)
endif()

# ============================================================================
# Installation (Optional)
# ============================================================================

install(TARGETS stinkytofu_python
    LIBRARY DESTINATION ${Python_SITEARCH}
    COMPONENT python_bindings
)

message(STATUS "StinkyTofu Python module configured successfully")
message(STATUS "  - Python module will be built as: ${CMAKE_BINARY_DIR}/python_module/stinkytofu${Python_SOABI}.so")
message(STATUS "  - Bridge library: stinkytofu_bridge (with RTTI and exceptions)")
