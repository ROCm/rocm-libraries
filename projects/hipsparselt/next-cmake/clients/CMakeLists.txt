# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Client Configuration
# ============================================================================

find_package(Threads REQUIRED)
find_package(OpenMP)

set(CLIENT_COMMON_LIBRARIES)

if(TARGET OpenMP::OpenMP_CXX)
    list(APPEND CLIENT_COMMON_LIBRARIES OpenMP::OpenMP_CXX)
endif()
list(APPEND CLIENT_COMMON_LIBRARIES Threads::Threads)

# ============================================================================
# Client build helpers
# ============================================================================

# Function to configure common client target properties
function(configure_client_target target_name)
    # Set C++ standard at target level
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    
    # Set target properties
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Common include directories
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/include"
    )
    
    # Platform-specific configuration
    if(HIPSPARSELT_ENABLE_CUDA)
        target_compile_definitions(${target_name} PRIVATE 
            __HIP_PLATFORM_NVIDIA__
        )
        target_include_directories(${target_name} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${target_name} PRIVATE 
            CUDA::cudart
            CUDA::cuda_driver
        )
    else()
        target_compile_definitions(${target_name} PRIVATE 
            __HIP_PLATFORM_AMD__
            ROCM_USE_FLOAT16
        )
        target_link_libraries(${target_name} PRIVATE 
            hip::host 
            hip::device
        )
    endif()
    
    # Compiler-specific options
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target_name} PRIVATE -mf16c)
    endif()
    
    # Internal API access
    target_compile_definitions(${target_name} PRIVATE HIPSPARSELT_INTERNAL_API)
    
    # Link with hipSPARSELt and common libraries
    target_link_libraries(${target_name} PRIVATE 
        roc::hipsparselt
        ${CLIENT_COMMON_LIBRARIES}
    )
endfunction()

# ============================================================================
# Sample executables
# ============================================================================

if(HIPSPARSELT_ENABLE_SAMPLES)
    # Sample executables
    set(SAMPLE_EXECUTABLES
        example_spmm_strided_batched
        example_prune_strip
        example_compress
    )
    
    foreach(sample_name ${SAMPLE_EXECUTABLES})
        add_executable(${sample_name} 
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/samples/${sample_name}.cpp"
        )
        configure_client_target(${sample_name})
        
        # Install sample
        install(TARGETS ${sample_name}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT samples
        )
    endforeach()
endif()

# ============================================================================
# Benchmark executable
# ============================================================================

if(HIPSPARSELT_ENABLE_BENCHMARKS)
    # Find required dependencies for benchmarks
    find_package(cblas REQUIRED CONFIG)
    
    # Set up BLAS library (prefer BLIS if available)
    if(HIPSPARSELT_ENABLE_BLIS)
        find_package(BLIS QUIET)
        if(BLIS_FOUND)
            set(BLAS_LIBRARY BLIS::blis)
        else()
            set(BLAS_LIBRARY cblas)
        endif()
    else()
        set(BLAS_LIBRARY cblas)
    endif()
    
    # Common source files for benchmarks and tests
    set(BENCHMARK_COMMON_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/singletons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/utility.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/cblas_interface.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/argument_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_parse_data.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_arguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_random.cpp"
    )
    
    # Benchmark executable
    add_executable(hipsparselt-bench
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/benchmarks/client.cpp"
        ${BENCHMARK_COMMON_SOURCES}
    )
    
    configure_client_target(hipsparselt-bench)
    
    # Benchmark-specific configuration
    target_compile_definitions(hipsparselt-bench PRIVATE 
        HIPSPARSELT_BENCH
    )
    
    # Link with BLAS
    target_link_libraries(hipsparselt-bench PRIVATE ${BLAS_LIBRARY})
    
    # Platform-specific linking
    if(HIPSPARSELT_ENABLE_CUDA)
        target_link_libraries(hipsparselt-bench PRIVATE 
            -llapack 
            -lcblas
        )
    else()
        target_link_libraries(hipsparselt-bench PRIVATE 
            lapack 
            cblas
        )
    endif()
    
    # Additional libraries for benchmarks
    if(NOT WIN32)
        target_link_libraries(hipsparselt-bench PRIVATE 
            -lm 
            -lstdc++fs
        )
        
        # Fortran runtime
        if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
            target_link_libraries(hipsparselt-bench PRIVATE -lgfortran)
        else()
            target_link_libraries(hipsparselt-bench PRIVATE -lflang_rt.runtime)
        endif()
    endif()
    
    # Install benchmark
    install(TARGETS hipsparselt-bench
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT benchmarks
    )
endif()

# ============================================================================
# Test executable
# ============================================================================

if(HIPSPARSELT_ENABLE_TESTS)
    # Find GTest
    find_package(GTest REQUIRED)
    
    # Test source files
    set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/hipsparselt_gtest_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/hipsparselt_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/prune_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/compress_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/spmm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/auxiliary_gtest.cpp"
    )
    
    # Test executable
    add_executable(hipsparselt-test
        ${TEST_SOURCES}
        ${BENCHMARK_COMMON_SOURCES}
    )
    
    configure_client_target(hipsparselt-test)
    
    # Test-specific configuration
    target_compile_definitions(hipsparselt-test PRIVATE 
        GOOGLE_TEST
    )
    
    # Link with GTest and BLAS
    target_link_libraries(hipsparselt-test PRIVATE 
        GTest::gtest 
        GTest::gtest_main
        ${BLAS_LIBRARY}
    )
    
    # Platform-specific linking for tests
    if(HIPSPARSELT_ENABLE_CUDA)
        target_link_libraries(hipsparselt-test PRIVATE 
            -llapack 
            -lcblas
        )
    else()
        target_link_libraries(hipsparselt-test PRIVATE 
            lapack 
            cblas
        )
    endif()
    
    # Additional libraries for tests
    if(NOT WIN32)
        target_link_libraries(hipsparselt-test PRIVATE 
            -lm 
            -lstdc++fs
        )
        
        # Fortran runtime
        if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
            target_link_libraries(hipsparselt-test PRIVATE -lgfortran)
        else()
            target_link_libraries(hipsparselt-test PRIVATE -lflang_rt.runtime)
        endif()
    endif()
    
    # Generate test data
    set(TEST_DATA_FILE "${PROJECT_BINARY_DIR}/staging/hipsparselt_gtest.data")
    add_custom_command(
        OUTPUT "${TEST_DATA_FILE}"
        COMMAND ${Python3_EXECUTABLE} 
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_gentest.py"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../clients/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/hipsparselt_gtest.yaml"
            -o "${TEST_DATA_FILE}"
        DEPENDS 
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_gentest.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/include/hipsparselt_common.yaml"
            "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest/hipsparselt_gtest.yaml"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../clients/gtest"
        COMMENT "Generating test data for hipsparselt-test"
    )
    
    add_custom_target(hipsparselt-test-data DEPENDS "${TEST_DATA_FILE}")
    add_dependencies(hipsparselt-test hipsparselt-test-data)
    
    # Install test executable and data
    install(TARGETS hipsparselt-test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tests
    )
    
    install(FILES "${TEST_DATA_FILE}"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tests
    )
endif()

# ============================================================================
# Common files installation
# ============================================================================

# Install common YAML files and scripts
set(COMMON_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../clients/include/hipsparselt_common.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../clients/include/hipsparselt_template.yaml"
)

set(COMMON_SCRIPTS
    "${CMAKE_CURRENT_SOURCE_DIR}/../clients/common/hipsparselt_gentest.py"
)

install(FILES ${COMMON_FILES}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT clients-common
)

install(PROGRAMS ${COMMON_SCRIPTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT clients-common
)

# ============================================================================
# Package component setup
# ============================================================================

# Set up package components
if(HIPSPARSELT_ENABLE_SAMPLES)
    rocm_package_setup_component(samples)
endif()

if(HIPSPARSELT_ENABLE_BENCHMARKS)
    rocm_package_setup_component(benchmarks)
endif()

if(HIPSPARSELT_ENABLE_TESTS)
    rocm_package_setup_component(tests)
endif()

rocm_package_setup_client_component(clients-common) 