# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Only header target, does not include dependencies

# Configure a header file to pass the rocThrust version
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocthrust_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocthrust_version.hpp"
  @ONLY
)

# Only header target, does not include dependencies
add_library(rocthrust INTERFACE)
target_include_directories(rocthrust
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:rocthrust/include>
)

target_link_libraries(rocthrust
  INTERFACE
    roc::rocprim_hip
    hip::device
)

# TODO: Remove after the Rocprim HC API removal
# We will use rocPRIM HIP API
target_compile_definitions(rocthrust
  INTERFACE
    ROCPRIM_HIP_API=1
)

# Installation

include(GNUInstallDirs)

# We need to install headers manually as rocm_install_targets
# does not support header-only libraries (INTERFACE targets)
rocm_install_targets(
  TARGETS rocthrust
  EXPORT rocthrust-targets
#   INCLUDE
#   ${CMAKE_SOURCE_DIR}/thrust
#   ${CMAKE_BINARY_DIR}/thrust
  PREFIX rocthrust
)
install(
  DIRECTORY
    "./"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/rocthrust/include/thrust/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "*.hpp"
  PATTERN "*.in"
  PATTERN "*.inl"
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Export targets
rocm_export_targets(
  TARGETS rocthrust-targets
  NAME rocthrust
  PREFIX rocthrust
)

# Create symlinks
rocm_install_symlink_subdir(rocthrust)
