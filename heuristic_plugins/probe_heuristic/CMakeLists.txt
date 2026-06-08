cmake_minimum_required(VERSION 3.25)
project(probe_heuristic CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(HIPDNN_SOURCE_DIR
    "/home/AMD/ysoliman/rocm-libraries-heuristic-prototype/projects/hipdnn"
    CACHE PATH "hipDNN source directory")

add_library(probe_heuristic SHARED probe_heuristic.cpp)
target_include_directories(
    probe_heuristic
    PRIVATE
        "${HIPDNN_SOURCE_DIR}/plugin_sdk/include"
        "${HIPDNN_SOURCE_DIR}/data_sdk/include")

set_target_properties(probe_heuristic PROPERTIES OUTPUT_NAME probe_heuristic)
