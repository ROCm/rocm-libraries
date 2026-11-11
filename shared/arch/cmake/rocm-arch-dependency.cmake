# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Including this module resolves the package and makes the catalog queries available.
if(NOT TARGET rocm::arch)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/rocm-arch-config.cmake")
        # Installed module: use the package beside it, independent of prefix lookup.
        include("${CMAKE_CURRENT_LIST_DIR}/rocm-arch-config.cmake")
    else()
        find_package(rocm-arch 0.4 QUIET CONFIG)
        if(NOT rocm-arch_FOUND)
            get_filename_component(_rocm_arch_source_directory
                                   "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
            add_subdirectory("${_rocm_arch_source_directory}"
                             "${CMAKE_CURRENT_BINARY_DIR}/rocm-arch")
            unset(_rocm_arch_source_directory)
        endif()
    endif()
endif()
