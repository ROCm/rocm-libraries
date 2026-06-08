cmake_minimum_required(VERSION 3.25)
project(sdpa_heuristic CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(HIPDNN_SOURCE_DIR
    "/home/AMD/ysoliman/rocm-libraries-heuristic-prototype/projects/hipdnn"
    CACHE PATH "hipDNN source directory")

add_library(sdpa_heuristic SHARED sdpa_heuristic.cpp)
target_include_directories(
    sdpa_heuristic
    PRIVATE
        "${HIPDNN_SOURCE_DIR}/plugin_sdk/include"
        "${HIPDNN_SOURCE_DIR}/data_sdk/include"
        "${HIPDNN_SOURCE_DIR}/flatbuffers_sdk/include"
        "${HIPDNN_SOURCE_DIR}/build/_deps/flatbuffers-src/include")

set_target_properties(sdpa_heuristic PROPERTIES OUTPUT_NAME sdpa_heuristic)
