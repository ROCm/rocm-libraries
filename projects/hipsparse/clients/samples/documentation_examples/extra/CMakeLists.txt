# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_csrgeam_cpp)
target_sources(example_hipsparse_csrgeam_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam_cpp.cpp")
target_link_libraries(example_hipsparse_csrgeam_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgeam2_cpp)
target_sources(example_hipsparse_csrgeam2_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam2_cpp.cpp")
target_link_libraries(example_hipsparse_csrgeam2_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgemm_cpp)
target_sources(example_hipsparse_csrgemm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm_cpp.cpp")
target_link_libraries(example_hipsparse_csrgemm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgemm2_cpp)
target_sources(example_hipsparse_csrgemm2_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm2_cpp.cpp")
target_link_libraries(example_hipsparse_csrgemm2_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_csrgeam_c)
target_sources(example_hipsparse_csrgeam_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam_c.c")
target_link_libraries(example_hipsparse_csrgeam_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgeam2_c)
target_sources(example_hipsparse_csrgeam2_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam2_c.c")
target_link_libraries(example_hipsparse_csrgeam2_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgemm_c)
target_sources(example_hipsparse_csrgemm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm_c.c")
target_link_libraries(example_hipsparse_csrgemm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrgemm2_c)
target_sources(example_hipsparse_csrgemm2_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm2_c.c")
target_link_libraries(example_hipsparse_csrgemm2_c PRIVATE hipsparse::samples-common)

# Fortran examples (only if Fortran support enabled and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_executable(example_hipsparse_csrgeam_fortran)
    target_sources(example_hipsparse_csrgeam_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam_fortran.f90")
    add_dependencies(example_hipsparse_csrgeam_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrgeam_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrgeam2_fortran)
    target_sources(example_hipsparse_csrgeam2_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgeam2_fortran.f90")
    add_dependencies(example_hipsparse_csrgeam2_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrgeam2_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrgemm_fortran)
    target_sources(example_hipsparse_csrgemm_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm_fortran.f90")
    add_dependencies(example_hipsparse_csrgemm_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrgemm_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrgemm2_fortran)
    target_sources(example_hipsparse_csrgemm2_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrgemm2_fortran.f90")
    add_dependencies(example_hipsparse_csrgemm2_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrgemm2_fortran PRIVATE hipsparse::fortran-samples-common)
    
endif()
