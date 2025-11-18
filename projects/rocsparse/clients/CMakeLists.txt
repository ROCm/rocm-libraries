# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Set directory-level runtime output for all client executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staging")

add_subdirectory(tools)

if(ROCSPARSE_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(ROCSPARSE_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ROCSPARSE_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(testings)
add_subdirectory(common)

if(ROCSPARSE_BUILD_TESTING OR BUILD_TESTING)
    # Build custom matrices if they don't exist
    if(NOT EXISTS "${CMAKE_MATRICES_DIR}")
        set(CMAKE_MATRICES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/matrices CACHE STRING "Matrices directory")
        file(MAKE_DIRECTORY ${CMAKE_MATRICES_DIR})

        set(CONVERT_SOURCE ${PROJECT_SOURCE_DIR}/deps/convert.cpp CACHE STRING "Convert tool mtx2csr")
        include(${PROJECT_SOURCE_DIR}/cmake/ClientMatrices.cmake)
    else()
        if(NOT CMAKE_MATRICES_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/matrices")
            message(STATUS "Copying matrix files from ${CMAKE_MATRICES_DIR} to ${CMAKE_CURRENT_SOURCE_DIR}/matrices")
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_MATRICES_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/matrices
                RESULT_VARIABLE STATUS
            )
            if(STATUS AND NOT STATUS EQUAL 0)
                message(FATAL_ERROR "Failed to copy matrix .csr files")
            endif()
        endif()
    endif()

    # Build mtx2csr converter executable
    if(NOT CONVERT_SOURCE)
        set(CONVERT_SOURCE "${PROJECT_SOURCE_DIR}/deps/convert.cpp")
    endif()

    # Compile converter using CXX compiler directly
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -Wl,--build-id=sha1
                -o ${PROJECT_BINARY_DIR}/mtx2csr.exe
        RESULT_VARIABLE STATUS
    )
    if(STATUS AND NOT STATUS EQUAL 0)
        message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.")
    endif()

    # Create custom targets and commands for test data files
    set(ROCSPARSE_CLIENTMATRICES "${PROJECT_SOURCE_DIR}/cmake/rocsparse_clientmatrices.cmake")
    set(ROCSPARSE_CONVERT "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_mtx2csr")

    add_custom_command(
        OUTPUT "${ROCSPARSE_CONVERT}"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${PROJECT_BINARY_DIR}/mtx2csr.exe" "${ROCSPARSE_CONVERT}"
        DEPENDS "${PROJECT_BINARY_DIR}/mtx2csr.exe"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set(ROCSPARSE_COMMON "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_common.yaml")
    add_custom_command(
        OUTPUT "${ROCSPARSE_COMMON}"
        COMMAND ${CMAKE_COMMAND} -E copy
                include/rocsparse_common.yaml "${ROCSPARSE_COMMON}"
        DEPENDS include/rocsparse_common.yaml
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set(ROCSPARSE_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_template.yaml")
    add_custom_command(
        OUTPUT "${ROCSPARSE_TEMPLATE}"
        COMMAND ${CMAKE_COMMAND} -E copy
                include/rocsparse_template.yaml "${ROCSPARSE_TEMPLATE}"
        DEPENDS include/rocsparse_template.yaml
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set(ROCSPARSE_SMOKE "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_smoke.yaml")
    set(ROCSPARSE_REGRESSION "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_regression.yaml")
    set(ROCSPARSE_EXTENDED "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_extended.yaml")

    add_custom_command(
        OUTPUT "${ROCSPARSE_SMOKE}" "${ROCSPARSE_REGRESSION}" "${ROCSPARSE_EXTENDED}"
        COMMAND ${CMAKE_COMMAND} -E copy
                include/rocsparse_smoke.yaml "${ROCSPARSE_SMOKE}"
        COMMAND ${CMAKE_COMMAND} -E copy
                include/rocsparse_regression.yaml "${ROCSPARSE_REGRESSION}"
        COMMAND ${CMAKE_COMMAND} -E copy
                include/rocsparse_extended.yaml "${ROCSPARSE_EXTENDED}"
        DEPENDS
            include/rocsparse_smoke.yaml
            include/rocsparse_regression.yaml
            include/rocsparse_extended.yaml
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set(ROCSPARSE_GENTEST "${CMAKE_CURRENT_BINARY_DIR}/staging/rocsparse_gentest.py")
    add_custom_command(
        OUTPUT "${ROCSPARSE_GENTEST}"
        COMMAND ${CMAKE_COMMAND} -E copy
                common/rocsparse_gentest.py "${ROCSPARSE_GENTEST}"
        DEPENDS common/rocsparse_gentest.py
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_custom_target(
        rocsparse-common
        DEPENDS
            "${ROCSPARSE_COMMON}"
            "${ROCSPARSE_TEMPLATE}"
            "${ROCSPARSE_SMOKE}"
            "${ROCSPARSE_GENTEST}"
            "${ROCSPARSE_CONVERT}"
    )

    # Install test support files
    rocm_install(
        FILES "${ROCSPARSE_COMMON}" "${ROCSPARSE_TEMPLATE}" "${ROCSPARSE_SMOKE}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsparse/test"
    )

    rocm_install(
        PROGRAMS "${ROCSPARSE_GENTEST}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/rocsparse/test"
    )

    rocm_install(
        PROGRAMS "${ROCSPARSE_CONVERT}"
        COMPONENT clients-common
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    rocm_install(
        FILES "${ROCSPARSE_CLIENTMATRICES}"
        COMPONENT clients-common
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsparse/test"
    )
endif()
