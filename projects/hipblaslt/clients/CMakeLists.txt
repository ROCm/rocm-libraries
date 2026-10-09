# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipblaslt-bench)
add_library(hipblaslt-clients-common STATIC)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite"
)

if(HIPBLASLT_ENABLE_COVERAGE)
    target_include_directories(hipblaslt-clients-common
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include
            ${PROJECT_SOURCE_DIR}/tensilelite/rocisa
            ${PROJECT_SOURCE_DIR}/../../shared/origami/include
    )
endif()

add_library(hipblaslt::hipblaslt-clients-common ALIAS hipblaslt-clients-common)

target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        roc::hostnumerics
        roc::hostnumerics-amd-gpu-layout
        OpenMP::OpenMP_CXX
    PRIVATE
        hip::device
        roc::hostnumerics-blas
        roc::tensilelite-host
)

if(HIPBLASLT_ENABLE_ROCROLLER)
    target_compile_definitions(hipblaslt-clients-common PRIVATE HIPBLASLT_USE_ROCROLLER)
endif()


if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(hipblaslt-clients-common PUBLIC)
endif()

target_link_libraries(hipblaslt-bench
    PRIVATE
        hip::device
        hipblaslt::hipblaslt-clients-common
        OpenMP::OpenMP_CXX
)

target_include_directories(hipblaslt-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench/include")

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hipblaslt-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(common/include)
add_subdirectory(common/src)
add_subdirectory(bench)

add_executable(hipblaslt-api-overhead "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_api_overhead.cpp")
target_link_libraries(hipblaslt-api-overhead PRIVATE hipblaslt-clients-common hip::device)
target_include_directories(hipblaslt-api-overhead PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/samples/common")

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk
    PRIVATE
        hipblaslt-clients-common
        hip::device
        roc::hostnumerics
)

add_executable(hipblaslt-bench-extop-layernorm "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)

add_executable(hipblaslt-bench-extop-matrixtransform "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)

add_executable(hipblaslt-bench-extop-softmax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)

add_executable(hipblaslt-bench-extop-amax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)

if(HIPBLASLT_ENABLE_GEMM_A2A_FUSION AND NOT WIN32)
    find_package(NUMA REQUIRED)
    find_package(hsa-runtime64 REQUIRED)
    find_package(hsakmt REQUIRED)

    include(HsakmtLinkInterface)
    hipblaslt_sanitize_hsakmt_link_interface()

    # samples/30_hipblaslt_a2a_sdma reuses these targets and this variable from
    # the enclosing scope.
    set(HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/client/include")

    add_executable(hipblaslt-bench-a2a "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_a2a.cpp")
    target_link_libraries(hipblaslt-bench-a2a
        PRIVATE hipblaslt-clients-common hip::device
                hsakmt::hsakmt hsa-runtime64::hsa-runtime64)
    target_include_directories(hipblaslt-bench-a2a
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench/include"
                "${HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR}")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/a2a/hipblaslt-bench-a2a-launch" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-bench-a2a-launch" COPYONLY)
endif()

# CU-contention cotenant helper (clients/scripts/cotenant). A standalone HIP
# binary, compiled as CXX so it picks up GPU_TARGETS via hip::device like the
# other clients (the project does not enable the HIP language). POSIX-only
# (unistd.h, pause(), usleep(), getpid()), so it is not built on Windows.
if(NOT WIN32)
    add_executable(hipblaslt-cotenant-kernel "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant-kernel.hip")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant-kernel.hip"
        PROPERTIES LANGUAGE CXX)
    target_link_libraries(hipblaslt-cotenant-kernel PRIVATE hip::device)
    target_compile_features(hipblaslt-cotenant-kernel PRIVATE cxx_std_17)
endif()

if(HIPBLASLT_ENABLE_YAML)
    find_package(LLVM REQUIRED)
    add_executable(hipblaslt-sequence "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_sequence.cpp")
    target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
    target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML)
    target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})
endif()
if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_executable(hipblaslt-test)
    if(amd_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE amd_smi)
    endif()
    target_link_libraries(hipblaslt-test
        PRIVATE
            hip::device
            hipblaslt::hipblaslt-clients-common
            roc::hostnumerics
            roc::tensilelite-host
            GTest::gtest
            OpenMP::OpenMP_CXX
            ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST)
    if(HIPBLASLT_ENABLE_GEMM_A2A_FUSION AND NOT WIN32)
        target_include_directories(hipblaslt-test
            PRIVATE "${HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR}")
        target_link_libraries(hipblaslt-test
            PRIVATE hsakmt::hsakmt hsa-runtime64::hsa-runtime64)
    endif()
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_USE_ROCROLLER)
    endif()
    if(HIPBLASLT_ENABLE_COVERAGE)
        target_compile_options(
            hipblaslt-test
            PRIVATE
                -g
                -O0
                -fprofile-instr-generate
                -fcoverage-mapping
        )
        target_link_options(hipblaslt-test PRIVATE -fprofile-instr-generate)
        target_compile_definitions(hipblaslt-test PRIVATE CODE_COVERAGE)
        # For coverage testing, link against hipblaslt's object files to access internal symbols
        # This ensures we test the real library code, not stubs
        target_link_libraries(hipblaslt-test PRIVATE
            $<TARGET_OBJECTS:hipblaslt>
            hipblaslt-tuning-store
            roc::tensilelite-host
            ${CMAKE_DL_LIBS}
        )
        # Add roctracer if HIPBLASLT_ENABLE_MARKER is enabled
        if(HIPBLASLT_ENABLE_MARKER)
            target_link_libraries(hipblaslt-test PRIVATE ${rocTracer})
        endif()
        # Add msgpack if not using YAML
        if(NOT HIPBLASLT_ENABLE_YAML)
            target_link_libraries(hipblaslt-test PRIVATE ${hipblaslt_msgpack_target})
        endif()
        # Add rocroller dependencies if enabled
        if(HIPBLASLT_ENABLE_ROCROLLER)
            target_link_libraries(hipblaslt-test PRIVATE
                $<TARGET_OBJECTS:hipblaslt-rocroller>
                rocroller
            )
        endif()
    endif()

    # The tuning file's row format and store, tested on their own: no device,
    # no solution library and no test hooks, so they run wherever the tests
    # build.
    add_executable(hipblaslt-test-tuning-store
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/src/tuning_store_test.cpp")
    target_link_libraries(hipblaslt-test-tuning-store
        PRIVATE
            hipblaslt-tuning-store
            GTest::gtest
            GTest::gtest_main
    )

    add_subdirectory(tests)
endif()
if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/hipblaslt_gentest.py" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_common.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_template.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bench/sequence.yaml" "${CMAKE_CURRENT_BINARY_DIR}/sequence.yaml" COPYONLY)
if(NOT WIN32)
    # Place the launcher next to the built kernel so it auto-finds the binary in
    # the build tree (matching the installed bin/ layout) without --binary.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-cotenant" COPYONLY)
endif()
