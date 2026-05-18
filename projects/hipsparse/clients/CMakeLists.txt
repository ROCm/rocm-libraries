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
    add_subdirectory(tests)

    set(HIPSPARSE_COMMON "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_common.yaml")
    set(HIPSPARSE_TEMPLATE "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_template.yaml")
    set(HIPSPARSE_GENTEST "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_gentest.py")
    set(HIPSPARSE_CONVERT "${PROJECT_BINARY_DIR}/clients/hipsparse_mtx2csr")
    set(HIPSPARSE_CLIENTMATRICES "${PROJECT_SOURCE_DIR}/cmake/hipsparse_clientmatrices.cmake")
    
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
