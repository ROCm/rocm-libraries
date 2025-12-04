# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_bsric02_cpp)
target_sources(example_hipsparse_bsric02_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsric02_cpp.cpp")
target_link_libraries(example_hipsparse_bsric02_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrilu02_cpp)
target_sources(example_hipsparse_bsrilu02_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrilu02_cpp.cpp")
target_link_libraries(example_hipsparse_bsrilu02_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csric02_cpp)
target_sources(example_hipsparse_csric02_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csric02_cpp.cpp")
target_link_libraries(example_hipsparse_csric02_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrilu02_cpp)
target_sources(example_hipsparse_csrilu02_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrilu02_cpp.cpp")
target_link_libraries(example_hipsparse_csrilu02_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_cpp)
target_sources(example_hipsparse_gtsv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_cpp.cpp")
target_link_libraries(example_hipsparse_gtsv_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_interleaved_batch_cpp)
target_sources(example_hipsparse_gtsv_interleaved_batch_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_interleaved_batch_cpp.cpp")
target_link_libraries(example_hipsparse_gtsv_interleaved_batch_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_no_pivot_cpp)
target_sources(example_hipsparse_gtsv_no_pivot_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_no_pivot_cpp.cpp")
target_link_libraries(example_hipsparse_gtsv_no_pivot_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_bsric02_c)
target_sources(example_hipsparse_bsric02_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsric02_c.c")
target_link_libraries(example_hipsparse_bsric02_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_bsrilu02_c)
target_sources(example_hipsparse_bsrilu02_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrilu02_c.c")
target_link_libraries(example_hipsparse_bsrilu02_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csric02_c)
target_sources(example_hipsparse_csric02_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csric02_c.c")
target_link_libraries(example_hipsparse_csric02_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_csrilu02_c)
target_sources(example_hipsparse_csrilu02_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrilu02_c.c")
target_link_libraries(example_hipsparse_csrilu02_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_c)
target_sources(example_hipsparse_gtsv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_c.c")
target_link_libraries(example_hipsparse_gtsv_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_interleaved_batch_c)
target_sources(example_hipsparse_gtsv_interleaved_batch_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_interleaved_batch_c.c")
target_link_libraries(example_hipsparse_gtsv_interleaved_batch_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gtsv_no_pivot_c)
target_sources(example_hipsparse_gtsv_no_pivot_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gtsv_no_pivot_c.c")
target_link_libraries(example_hipsparse_gtsv_no_pivot_c PRIVATE hipsparse::samples-common)

# Fortran examples (only if Fortran support enabled and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_executable(example_hipsparse_bsric02_fortran)
    target_sources(example_hipsparse_bsric02_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsric02_fortran.f90")
    add_dependencies(example_hipsparse_bsric02_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_bsric02_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_bsrilu02_fortran)
    target_sources(example_hipsparse_bsrilu02_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_bsrilu02_fortran.f90")
    add_dependencies(example_hipsparse_bsrilu02_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_bsrilu02_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csric02_fortran)
    target_sources(example_hipsparse_csric02_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csric02_fortran.f90")
    add_dependencies(example_hipsparse_csric02_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csric02_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_csrilu02_fortran)
    target_sources(example_hipsparse_csrilu02_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_csrilu02_fortran.f90")
    add_dependencies(example_hipsparse_csrilu02_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_csrilu02_fortran PRIVATE hipsparse::fortran-samples-common)
    
endif()
