# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_bsrmm_cpp)
target_sources(example_hipsparse_bsrmm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmm_cpp.cpp")
target_link_libraries(example_hipsparse_bsrmm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrsm_cpp)
target_sources(example_hipsparse_bsrsm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrsm_cpp.cpp")
target_link_libraries(example_hipsparse_bsrsm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrmm_cpp)
target_sources(example_hipsparse_csrmm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmm_cpp.cpp")
target_link_libraries(example_hipsparse_csrmm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrsm2_cpp)
target_sources(example_hipsparse_csrsm2_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsm2_cpp.cpp")
target_link_libraries(example_hipsparse_csrsm2_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gemmi_cpp)
target_sources(example_hipsparse_gemmi_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gemmi_cpp.cpp")
target_link_libraries(example_hipsparse_gemmi_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_bsrmm_c)
target_sources(example_hipsparse_bsrmm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmm_c.c")
target_link_libraries(example_hipsparse_bsrmm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrsm_c)
target_sources(example_hipsparse_bsrsm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrsm_c.c")
target_link_libraries(example_hipsparse_bsrsm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrmm_c)
target_sources(example_hipsparse_csrmm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmm_c.c")
target_link_libraries(example_hipsparse_csrmm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrsm2_c)
target_sources(example_hipsparse_csrsm2_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsm2_c.c")
target_link_libraries(example_hipsparse_csrsm2_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gemmi_c)
target_sources(example_hipsparse_gemmi_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gemmi_c.c")
target_link_libraries(example_hipsparse_gemmi_c PRIVATE hipsparse::samples-common)

# Fortran examples (only if Fortran support enabled and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_executable(example_hipsparse_bsrmm_fortran)
    target_sources(example_hipsparse_bsrmm_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrmm_fortran.f90")
    add_dependencies(example_hipsparse_bsrmm_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_bsrmm_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrmm_fortran)
    target_sources(example_hipsparse_csrmm_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrmm_fortran.f90")
    add_dependencies(example_hipsparse_csrmm_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrmm_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrsm2_fortran)
    target_sources(example_hipsparse_csrsm2_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrsm2_fortran.f90")
    add_dependencies(example_hipsparse_csrsm2_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrsm2_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_gemmi_fortran)
    target_sources(example_hipsparse_gemmi_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gemmi_fortran.f90")
    add_dependencies(example_hipsparse_gemmi_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_gemmi_fortran PRIVATE hipsparse::fortran-samples-common)
    
endif()
