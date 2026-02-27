# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(ADD_KERNELS_SOURCE include_inliner.cpp addkernels.cpp)

add_executable(addkernels EXCLUDE_FROM_ALL ${ADD_KERNELS_SOURCE})
target_include_directories(addkernels PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
if(HAS_LIB_STD_FILESYSTEM)
    target_link_libraries(addkernels PRIVATE stdc++fs)
endif()

clang_tidy_check(addkernels)
