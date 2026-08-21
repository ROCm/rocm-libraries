# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(WIN32)
    include("${CMAKE_CURRENT_LIST_DIR}/windows-amdclang.cmake")
else()
    include("${CMAKE_CURRENT_LIST_DIR}/linux-amdclang.cmake")
endif()
