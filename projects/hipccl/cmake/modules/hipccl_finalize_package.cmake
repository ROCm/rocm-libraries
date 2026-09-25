# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Shared by hipccl2/CMakeLists.txt and hipccl3/CMakeLists.txt: produces the
# single unified "hipccl" CPack package and the find_package(hipccl) CMake
# package config. This only consolidates the mechanical, identical-between-
# layouts CMake calls - see each root CMakeLists.txt's own comments for the
# full rationale (why packaging is unconditional, why individual rocprim/
# hipcub/rocthrust standalone packaging is disabled, etc.), since that
# context is specific to decisions made in each layout's own copies.
#
# Usage:
#   hipccl_finalize_package([DESCRIPTION <text>])
# DESCRIPTION defaults to a generic string if not given.
macro(hipccl_finalize_package)
    cmake_parse_arguments(HIPCCL_PKG "" "DESCRIPTION" "" ${ARGN})
    if(NOT HIPCCL_PKG_DESCRIPTION)
        set(HIPCCL_PKG_DESCRIPTION "hipCCL: unified rocPRIM, hipCUB, and rocThrust package")
    endif()

    set(HIP_RUNTIME_MINIMUM 4.5.0)
    rocm_package_add_dependencies(SHARED_DEPENDS "hip-runtime-amd >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT and BSD and ASL 2.0")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

    rocm_create_package(
        NAME hipccl
        DESCRIPTION "${HIPCCL_PKG_DESCRIPTION}"
        MAINTAINER "hipccl-maintainer@amd.com"
        HEADER_ONLY
    )

    include(CMakePackageConfigHelpers)

    set(HIPCCL_PACKAGE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/hipccl")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/package/hipccl-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config.cmake"
        INSTALL_DESTINATION "${HIPCCL_PACKAGE_CONFIG_INSTALL_DIR}"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config-version.cmake"
        VERSION "${HIPCCL_VERSION}"
        COMPATIBILITY AnyNewerVersion
        ARCH_INDEPENDENT
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/hipccl-config-version.cmake"
        DESTINATION "${HIPCCL_PACKAGE_CONFIG_INSTALL_DIR}"
        COMPONENT devel
    )
endmacro()
