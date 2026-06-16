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

add_library(hipblaslt::hipblaslt-clients-common ALIAS hipblaslt-clients-common)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(hipblaslt-clients-common PUBLIC BLIS::BLIS)
endif()
target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        OpenMP::OpenMP_CXX
        ${LAPACK_LIBRARIES}
    PRIVATE
        hip::device
        tensilelite::tensilelite-host
)

if(HIPBLASLT_ENABLE_MXDATAGENERATOR)
    # PUBLIC because public headers such as testing_matmul.hpp guard
    # `<mxDataGen.hpp>` on `HIPBLASLT_ENABLE_MXDATAGENERATOR`, so downstream
    # targets (hipblaslt-test, hipblaslt-bench, ...) need to see the macro,
    # the include path, and the C++20 floor all the way through.
    target_link_libraries(hipblaslt-clients-common PUBLIC hipblaslt::mxdatagen)
endif()

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

# `clients/common/CMakeLists.txt` itself only declares the mxdatagen helper
# (which the project root already added when needed); the `include` and
# `src` subdirectories populate hipblaslt-clients-common's sources, so they
# are added directly here, avoiding a second `add_subdirectory(common)` that
# would clash with the root-level one.
add_subdirectory(common/include)
add_subdirectory(common/src)
add_subdirectory(bench)

add_executable(hipblaslt-api-overhead "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_api_overhead.cpp")
target_link_libraries(hipblaslt-api-overhead PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-api-overhead PRIVATE ROCBLASLT_INTERNAL_API)
target_include_directories(hipblaslt-api-overhead PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/samples/common")

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-layernorm "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-layernorm PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-matrixtransform "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-matrixtransform PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-softmax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-softmax PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-amax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-amax PRIVATE ROCBLASLT_INTERNAL_API)

if(HIPBLASLT_ENABLE_YAML)
    find_package(LLVM REQUIRED)
    add_executable(hipblaslt-sequence "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_sequence.cpp")
    target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
    target_compile_definitions(hipblaslt-sequence PRIVATE ROCBLASLT_INTERNAL_API)
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
            tensilelite::tensilelite-host
            GTest::gtest
            OpenMP::OpenMP_CXX
            ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST)
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_USE_ROCROLLER)
    endif()

    add_subdirectory(tests)
endif()
if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/hipblaslt_gentest.py" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_common.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_template.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bench/sequence.yaml" "${CMAKE_CURRENT_BINARY_DIR}/sequence.yaml" COPYONLY)
