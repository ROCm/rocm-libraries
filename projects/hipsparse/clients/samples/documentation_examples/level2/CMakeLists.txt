# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_bsrmv_cpp)
target_sources(example_hipsparse_bsrmv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmv_cpp.cpp")
target_link_libraries(example_hipsparse_bsrmv_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrsv2_cpp)
target_sources(example_hipsparse_bsrsv2_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrsv2_cpp.cpp")
target_link_libraries(example_hipsparse_bsrsv2_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrmv_cpp)
target_sources(example_hipsparse_csrmv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmv_cpp.cpp")
target_link_libraries(example_hipsparse_csrmv_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrsv2_cpp)
target_sources(example_hipsparse_csrsv2_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsv2_cpp.cpp")
target_link_libraries(example_hipsparse_csrsv2_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gemvi_cpp)
target_sources(example_hipsparse_gemvi_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gemvi_cpp.cpp")
target_link_libraries(example_hipsparse_gemvi_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_hybmv_cpp)
target_sources(example_hipsparse_hybmv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_hybmv_cpp.cpp")
target_link_libraries(example_hipsparse_hybmv_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_bsrmv_c)
target_sources(example_hipsparse_bsrmv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmv_c.c")
target_link_libraries(example_hipsparse_bsrmv_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrsv2_c)
target_sources(example_hipsparse_bsrsv2_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrsv2_c.c")
target_link_libraries(example_hipsparse_bsrsv2_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrmv_c)
target_sources(example_hipsparse_csrmv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmv_c.c")
target_link_libraries(example_hipsparse_csrmv_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrsv2_c)
target_sources(example_hipsparse_csrsv2_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsv2_c.c")
target_link_libraries(example_hipsparse_csrsv2_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gemvi_c)
target_sources(example_hipsparse_gemvi_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gemvi_c.c")
target_link_libraries(example_hipsparse_gemvi_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_hybmv_c)
target_sources(example_hipsparse_hybmv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_hybmv_c.c")
target_link_libraries(example_hipsparse_hybmv_c PRIVATE hipsparse::samples-common)

# Fortran examples (only if Fortran support enabled and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_executable(example_hipsparse_bsrmv_fortran)
    target_sources(example_hipsparse_bsrmv_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmv_fortran.f90")
    add_dependencies(example_hipsparse_bsrmv_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_bsrmv_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrmv_fortran)
    target_sources(example_hipsparse_csrmv_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmv_fortran.f90")
    add_dependencies(example_hipsparse_csrmv_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrmv_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrsv2_fortran)
    target_sources(example_hipsparse_csrsv2_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsv2_fortran.f90")
    add_dependencies(example_hipsparse_csrsv2_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrsv2_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_hybmv_fortran)
    target_sources(example_hipsparse_hybmv_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_hybmv_fortran.f90")
    add_dependencies(example_hipsparse_hybmv_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_hybmv_fortran PRIVATE hipsparse::fortran-samples-common)
    
endif()
