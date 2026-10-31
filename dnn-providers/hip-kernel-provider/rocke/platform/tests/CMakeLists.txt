# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# By-layer C++ test registration for the rocKE engine. Included via
# add_subdirectory(tests) from platform/CMakeLists.txt. Registration remains in
# this directory scope so CTest properties work with the project's CMake 3.16
# minimum; concern-specific definitions live in cmake/ to reduce PR contention.

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CoreTests.cmake")

if(BUILD_TESTING)
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/PortableIrTests.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/InstanceTests.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/NativeGoldenTests.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/TestCategoryLabels.cmake")
endif()
