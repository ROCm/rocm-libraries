cmake_minimum_required(VERSION 3.21)
project(mxfp6_gemm LANGUAGES CXX HIP)

# Default to the MI350X (gfx950) target; override with -DCMAKE_HIP_ARCHITECTURES=...
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx950)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- library: libmxfp6gemm ----------------------------------------------------------------
# The GEMM launch (device). The public API in include/mxfp6/gemm.hpp is device-free, so a
# host translation unit can use the library without a HIP compiler. Host preprocess helpers
# (include/mxfp6/{preprocess,types}.hpp) are header-only. (The CPU reference used by the test
# lives in tests/reference.hpp — it is not part of the library's public surface.)
add_library(mxfp6gemm STATIC src/gemm.cpp)
set_source_files_properties(src/gemm.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(mxfp6gemm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(mxfp6gemm PUBLIC cxx_std_17)

# ---- test: links the library, drives it through the public API -----------------------------
enable_testing()
add_executable(test_gemm tests/test_gemm.cpp)
set_source_files_properties(tests/test_gemm.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(test_gemm PRIVATE mxfp6gemm)
add_test(NAME gemm COMMAND test_gemm)
