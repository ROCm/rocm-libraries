# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

if(HIPSOLVER_ENABLE_HIPBLAS_TESTS)
    find_package(hipblas REQUIRED CONFIG)
endif()

if(HIPSOLVER_ENABLE_HIPSPARSE_TESTS)
    find_package(hipsparse REQUIRED CONFIG)
endif()

# --- Test executable ---

add_executable(hipsolver-test)

target_compile_features(hipsolver-test PRIVATE cxx_std_17)

target_sources(hipsolver-test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsolver_gtest_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/determinism_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/params_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/getrs_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/getrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gebrd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gels_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/geqrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gesv_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gesvd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gesvda_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gesvdj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/potrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/potri_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/potrs_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/syevd_heevd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/syevj_heevj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sygvd_hegvd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sygvj_hegvj_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/syevdx_heevdx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sygvdx_hegvdx_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sytrd_hetrd_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sytrf_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orgbr_ungbr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orgqr_ungqr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orgtr_ungtr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ormqr_unmqr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ormtr_unmtr_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrrf_refactlu_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrrf_solve_gtest.cpp"
)

# --- Coverage ---

if(HIPSOLVER_BUILD_COVERAGE)
    target_compile_options(hipsolver-test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipsolver-test PRIVATE -fprofile-instr-generate)
endif()

# --- Includes ---

target_include_directories(hipsolver-test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_include_directories(hipsolver-test SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# --- Armor ---

add_armor_flags(hipsolver-test "${HIPSOLVER_ARMOR_LEVEL}")

# --- Definitions ---

target_compile_definitions(hipsolver-test PRIVATE GOOGLE_TEST)

# --- ASAN flags ---

if(HIPSOLVER_ENABLE_ASAN)
    target_compile_options(hipsolver-test PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsolver-test PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

# --- Linking ---

target_link_libraries(hipsolver-test PRIVATE
    ${LAPACK_LIBRARIES}
    hipsolver-clients-common
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    Threads::Threads
    roc::hipsolver
)

if(HIPSOLVER_ENABLE_HIPBLAS_TESTS)
    target_link_libraries(hipsolver-test PRIVATE roc::hipblas)
    target_sources(hipsolver-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_include1_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblas_include2_gtest.cpp"
    )
endif()

if(HIPSOLVER_ENABLE_HIPSPARSE_TESTS)
    target_link_libraries(hipsolver-test PRIVATE roc::hipsparse)
    target_sources(hipsolver-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrlsvchol_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrlsvqr_gtest.cpp"
    )
endif()

if(HIPSOLVER_ENABLE_FORTRAN)
    target_link_libraries(hipsolver-test PRIVATE hipsolver_fortran_client)
endif()

# --- Platform-specific linking ---

if(NOT HIPSOLVER_ENABLE_CUDA)
    target_link_libraries(hipsolver-test PRIVATE hip::host)

    if(CUSTOM_TARGET)
        target_link_libraries(hipsolver-test PRIVATE hip::${CUSTOM_TARGET})
    endif()

    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
        target_link_libraries(hipsolver-test PRIVATE -lpthread -lm)
    endif()
else()
    target_compile_definitions(hipsolver-test PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipsolver-test PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipsolver-test PRIVATE ${CUDA_LIBRARIES} Threads::Threads)
endif()

# --- Output and install ---

set_target_properties(hipsolver-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

rocm_install(TARGETS hipsolver-test COMPONENT tests)

# --- Windows DLL copy ---

if(WIN32)
    file(GLOB third_party_dlls
        LIST_DIRECTORIES OFF
        CONFIGURE_DEPENDS
        ${cblas_DIR}/bin/*.dll
        ${GTest_DIR}/bin/*.dll
        $ENV{rocblas_DIR}/bin/*.dll
        $ENV{rocsolver_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/hipinfo.exe
        ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach(file_i ${third_party_dlls})
        add_custom_command(TARGET hipsolver-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/
        )
    endforeach()
    if(DEFINED ENV{rocblas_DIR})
        add_custom_command(TARGET hipsolver-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                $ENV{rocblas_DIR}/bin/rocblas/library
                ${PROJECT_BINARY_DIR}/staging/library
        )
    else()
        message(WARNING "ENV{rocblas_DIR} not set, tests may be missing .dlls")
    endif()
endif()
