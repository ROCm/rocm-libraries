# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipblaslt-bench)
add_library(hipblaslt-clients-common STATIC)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite" # not sure about this
)

add_library(hipblaslt::hipblaslt-clients-common ALIAS hipblaslt-clients-common)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(hipblaslt-clients-common PRIVATE BLIS::BLIS)
endif()

target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        ${LAPACK_LIBRARIES} 
        ${BLAS_LIBRARIES}
        ${CBLAS_LIBRARIES}
    PRIVATE
        hip::device
)

# grep doesn't return any hits
# target_compile_definitions(hipblaslt-clients-common PRIVATE CLIENTS_NO_FORTRAN) # necessary?

if(HIPBLASLT_ENABLE_ROCROLLER)
    if(NOT ROCM_LIBRARIES_SUPERBUILD)
        find_package(mxDataGenerator REQUIRED)
    endif()
    target_compile_definitions(hipblaslt-clients-common PRIVATE HIPBLASLT_USE_ROCROLLER)
    target_link_libraries(hipblaslt-clients-common PRIVATE roc::rocroller roc::mxDataGenerator)
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

add_subdirectory(common)
add_subdirectory(bench)

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-layernorm "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-layernorm PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-matrixtransform "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-matrixtransform PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-softmax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-softmax PRIVATE ROCBLASLT_INTERNAL_API )

add_executable(hipblaslt-bench-extop-amax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-amax PRIVATE ROCBLASLT_INTERNAL_API )

find_package(LLVM REQUIRED)
add_executable(hipblaslt-sequence "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_sequence.cpp")
target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-sequence PRIVATE ROCBLASLT_INTERNAL_API)
target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML)
target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})

if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(hipblaslt-test)
    if(rocm_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE rocm_smi64)
    endif()
    target_link_libraries(hipblaslt-test 
        PRIVATE
            hip::device
            hipblaslt::hipblaslt-clients-common
            GTest::gtest
            OpenMP::OpenMP_CXX
    )
    if(HIPBLASLT_ENABLE_BLIS)
        target_link_libraries(hipblaslt-test PRIVATE BLIS::BLIS) # necessary?
    endif()
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST)
    # not getting any hits on HIPBLASLT_TEST
    #target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_TEST)
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_USE_ROCROLLER) # should this be public on hipblaslt?
        target_link_libraries(hipblaslt-test PRIVATE roc::rocroller)  # same
    endif()
    add_subdirectory(tests)
endif()

if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
