# ************************************************************************
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
)
FetchContent_MakeAvailable(nanobind)

# ============================================================================
# Note: StinkyTofu and StinkyIR have been moved to the core stinkytofu library
# The Python module now links directly against the core library
# ============================================================================

# ============================================================================
# Create the Python Module
# ============================================================================

# Temporarily remove global compile options to avoid inheriting -fno-rtti -fno-exceptions
get_directory_property(SAVED_COMPILE_OPTIONS COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")

nanobind_add_module(stinkytofu_python
    NB_STATIC  # Build as static module
    src/python_bindings.cpp
    src/logical_count.cpp
)

# Restore global compile options
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${SAVED_COMPILE_OPTIONS}")

target_include_directories(stinkytofu_python PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../hardware/include"
    # For generated files like gfxIsa.inc
    ${TABLEGEN_OUTPUT_DIR}
)

target_link_libraries(stinkytofu_python PRIVATE
    stinkytofu
    gfxisa
)

# Set output name to _stinkytofu (private module, loaded by __init__.py)
# Place in stinkytofu/ subdirectory
set_target_properties(stinkytofu_python PROPERTIES
    OUTPUT_NAME _stinkytofu
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/stinkytofu
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
)
# On Windows there is no RPATH; copy stinkytofu.dll next to _stinkytofu.pyd
# so the dynamic linker can find it at runtime.
if(WIN32)
    add_custom_command(TARGET stinkytofu_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:stinkytofu>
            $<TARGET_FILE_DIR:stinkytofu_python>
        COMMENT "Copying stinkytofu.dll next to _stinkytofu.pyd")
endif()

# Copy __init__.py and generate _build_info.py at configure time (not post-build)
# so the package is importable immediately from the build tree after cmake configure.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/stinkytofu/__init__.py"
    "${CMAKE_BINARY_DIR}/lib/stinkytofu/__init__.py"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/stinkytofu/_build_info.py.in"
    "${CMAKE_BINARY_DIR}/lib/stinkytofu/_build_info.py"
    @ONLY
)
# _build_info.py is intentionally NOT in install() rules so it is absent
# from wheels and pre-built packages, acting as the source-build sentinel.

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

# Use a relative path so the install respects CMAKE_INSTALL_PREFIX.
# Python_SITEARCH is absolute (e.g. /usr/local/lib/python3.12/dist-packages)
# which would bypass CMAKE_INSTALL_PREFIX and require root.
set(STINKYTOFU_PYTHON_SITE_DIR
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages"
    CACHE PATH "Python module install directory (relative to CMAKE_INSTALL_PREFIX)")

# Install C++ module (_stinkytofu.so)
install(TARGETS stinkytofu_python
    LIBRARY DESTINATION ${STINKYTOFU_PYTHON_SITE_DIR}/stinkytofu
    COMPONENT python_bindings
)

# Co-install libstinkytofu.so / stinkytofu.dll next to _stinkytofu so the
# dynamic linker can find it at runtime.
install(FILES $<TARGET_FILE:stinkytofu>
    DESTINATION ${STINKYTOFU_PYTHON_SITE_DIR}/stinkytofu
    COMPONENT python_bindings
)

# Install Python wrapper (__init__.py)
install(FILES stinkytofu/__init__.py
    DESTINATION ${STINKYTOFU_PYTHON_SITE_DIR}/stinkytofu
    COMPONENT python_bindings
)

# scikit-build-core (pip install) install rules.
# SKBUILD is set by scikit-build-core when building a wheel.
# _build_info.py is intentionally excluded so wheels behave as pre-built packages.
if(DEFINED SKBUILD)
    install(TARGETS stinkytofu_python DESTINATION stinkytofu)
    install(FILES $<TARGET_FILE:stinkytofu> DESTINATION stinkytofu)
    install(FILES stinkytofu/__init__.py DESTINATION stinkytofu)
endif()

message(STATUS "StinkyTofu Python module configured successfully")
message(STATUS "  - Python module will be built as: ${CMAKE_BINARY_DIR}/python_module/stinkytofu${Python_SOABI}.so")
message(STATUS "  - Linking directly to core stinkytofu library (StinkyTofu and StinkyIR now in core)")
