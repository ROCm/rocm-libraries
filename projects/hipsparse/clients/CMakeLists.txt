# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT HIPSPARSE_ENABLE_HOST)
    find_package(hipsparse REQUIRED)
endif()

add_subdirectory(common)

if(HIPSPARSE_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(HIPSPARSE_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
    
    rocm_install(
        FILES "${HIPSPARSE_COMMON}" "${HIPSPARSE_TEMPLATE}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test"
    )
    rocm_install(
        PROGRAMS "${HIPSPARSE_GENTEST}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/hipsparse/test"
    )
    rocm_install(
        PROGRAMS "${HIPSPARSE_CONVERT}"
        COMPONENT clients-common
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    rocm_install(
        FILES "${HIPSPARSE_CLIENTMATRICES}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test"
    )
endif()
