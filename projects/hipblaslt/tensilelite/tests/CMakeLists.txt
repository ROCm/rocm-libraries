# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(tensilelite-tests)
add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AlmostEqual_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestUtils.hpp"
)

target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../client/include")

if(HIPBLASLT_ENABLE_LLVM OR HIPBLASLT_ENABLE_MSGPACK)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/RangeLibrary_test.cpp"
    )
endif()


target_link_libraries(tensilelite-tests
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
    PRIVATE
        hip::device
        ${CMAKE_DL_LIBS}
)

gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
endif()


enable_testing()

#################################################
#        Tests for cpu-gemm-driver              #
#################################################

function(add_cpu_gemm_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "ARGS" ${ARGN})

    add_test(
        NAME "CPUGemm.${TEST_NAME}"
        COMMAND cpu-gemm-driver ${ARG_ARGS}
    )
endfunction()

# 2. Basic Sanity Check (F32)
add_cpu_gemm_test(F32_Sanity
    ARGS --M 31 --N 41 --K 51 --type f32 --validate 1
)

# 3. Transpose Combinations (T/N, N/T, T/T)
add_cpu_gemm_test(F32_TransA
    ARGS --M 31 --N 41 --K 51 --transA 1 --transB 0 --type f32 --validate 1
)
add_cpu_gemm_test(F32_TransB
    ARGS --M 31 --N 41 --K 51 --transA 0 --transB 1 --type f32 --validate 1
)
add_cpu_gemm_test(F32_TransAB
    ARGS --M 31 --N 41 --K 51 --transA 1 --transB 1 --type f32 --validate 1
)

# 4. Small sizes
add_cpu_gemm_test(F32_Odd_Sizes
    ARGS --M 2 --N 3 --K 5 --type f32 --validate 1
)

# 5. BFloat16 Tests
# These verify your "cast to f32 -> compute -> cast back" logic
add_cpu_gemm_test(BF16_Standard
    ARGS --M 128 --N 128 --K 128 --type bf16 --validate 1
)

# 6. FP16 Tests
add_cpu_gemm_test(F16_Standard
    ARGS --M 128 --N 128 --K 128 --type f16 --validate 1
)

# 7. Verify fast flag on / off works
add_cpu_gemm_test(FastPath_On
    ARGS --M 256 --N 256 --K 256 --tryFastPath 1 --validate 1 --type f32
)
add_cpu_gemm_test(FastPath_Off
    ARGS --M 256 --N 256 --K 256 --tryFastPath 0 --validate 1 --type f32
)
