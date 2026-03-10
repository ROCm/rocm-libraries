# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# --- Client dependencies ---

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
    find_package(LAPACK 3.7 REQUIRED CONFIG)
else()
    find_package(LAPACK 3.7 REQUIRED)
endif()

if(NOT LAPACK_LIBRARIES)
    set(LAPACK_LIBRARIES
        ${LAPACK_blas_LIBRARIES}
        ${LAPACK_lapack_LIBRARIES}
    )
endif()

# --- Fortran client bindings ---

if(HIPSOLVER_ENABLE_FORTRAN)
    if(HIPSOLVER_BUILD_TESTING OR HIPSOLVER_ENABLE_BENCHMARKS)
        add_library(hipsolver_fortran_client STATIC)
        target_sources(hipsolver_fortran_client PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hipsolver_fortran.f90"
        )
        target_link_libraries(hipsolver_fortran_client PUBLIC hipsolver_fortran)
        target_include_directories(hipsolver_fortran_client PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/hipsolver>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/hipsolver/internal>
        )
        target_compile_definitions(hipsolver_fortran_client INTERFACE HAVE_HIPSOLVER_FORTRAN_CLIENT)
    endif()
endif()

# --- Common client library ---

if(HIPSOLVER_BUILD_TESTING OR HIPSOLVER_ENABLE_BENCHMARKS)
    add_library(hipsolver-clients-common OBJECT)

    target_compile_features(hipsolver-clients-common PRIVATE cxx_std_17)

    target_sources(hipsolver-clients-common PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common/clients_utility.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/hipsolver_datatype2string.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/lapack_host_reference.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/utility.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocsolvercommon/rocsolver_test.cpp"
    )

    target_include_directories(hipsolver-clients-common
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    target_link_libraries(hipsolver-clients-common
        PUBLIC roc::hipsolver
        PRIVATE Threads::Threads
    )

    target_link_libraries(hipsolver-clients-common PUBLIC
        $<$<PLATFORM_ID:Linux>:stdc++fs>
    )

    # Ensure PIC for shared library builds
    set_target_properties(hipsolver-clients-common PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # Copy and point to sparse test data
    file(COPY
        "${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/"
        DESTINATION "${PROJECT_BINARY_DIR}/staging/sparsedata/"
    )
    install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsolver/test"
        COMPONENT tests
    )

    if(HIPSOLVER_BUILD_TESTING)
        add_subdirectory(gtest)
    endif()

    if(HIPSOLVER_ENABLE_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()
endif()

if(HIPSOLVER_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
