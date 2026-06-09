cmake_minimum_required(VERSION 3.25)
project(verify_conv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HIPDNN_SOURCE_DIR "/home/AMD/ysoliman/rocm-libraries-heuristic-prototype/projects/hipdnn")
set(HIPDNN_BUILD_DIR "${HIPDNN_SOURCE_DIR}/build")
set(ROCM_ROOT "/cluster/apps/ubuntu-24/rocm/rocm-7.1.1.70101")

add_executable(verify_conv verify_conv.cpp)
target_compile_definitions(
    verify_conv PRIVATE HIPDNN_ENABLE_SDPA FMT_HEADER_ONLY __HIP_PLATFORM_AMD__
                        __HIP_PLATFORM_AMD__=1)

target_include_directories(
    verify_conv
    PRIVATE
        "${HIPDNN_SOURCE_DIR}/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/backend/src/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/plugin_sdk/include"
        "${HIPDNN_SOURCE_DIR}/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/_deps/flatbuffers-src/include"
        "${ROCM_ROOT}/include")

target_link_libraries(
    verify_conv
    PRIVATE
        "${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so"
        "${ROCM_ROOT}/lib/libamdhip64.so"
        pthread
        dl)

add_executable(verify_conv_realistic verify_conv_realistic.cpp)
target_compile_definitions(
    verify_conv_realistic PRIVATE HIPDNN_ENABLE_SDPA FMT_HEADER_ONLY __HIP_PLATFORM_AMD__
                                  __HIP_PLATFORM_AMD__=1)
target_include_directories(verify_conv_realistic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(
    verify_conv_realistic
    PRIVATE
        "${HIPDNN_SOURCE_DIR}/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/backend/src/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/plugin_sdk/include"
        "${HIPDNN_SOURCE_DIR}/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/_deps/flatbuffers-src/include"
        "${ROCM_ROOT}/include")
target_link_libraries(
    verify_conv_realistic
    PRIVATE
        "${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so"
        "${ROCM_ROOT}/lib/libamdhip64.so"
        pthread
        dl)

add_executable(verify_conv_miopen verify_conv_miopen.cpp)
target_compile_definitions(
    verify_conv_miopen PRIVATE HIPDNN_ENABLE_SDPA FMT_HEADER_ONLY __HIP_PLATFORM_AMD__
                               __HIP_PLATFORM_AMD__=1)
target_include_directories(
    verify_conv_miopen
    PRIVATE
        "${HIPDNN_SOURCE_DIR}/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/backend/src/backend/include"
        "${HIPDNN_SOURCE_DIR}/build/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/plugin_sdk/include"
        "${HIPDNN_SOURCE_DIR}/test_sdk/include"
        "${HIPDNN_SOURCE_DIR}/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/_deps/flatbuffers-src/include"
        "${ROCM_ROOT}/include")
target_link_libraries(
    verify_conv_miopen
    PRIVATE
        "${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so"
        "${ROCM_ROOT}/lib/libamdhip64.so"
        pthread
        dl)
