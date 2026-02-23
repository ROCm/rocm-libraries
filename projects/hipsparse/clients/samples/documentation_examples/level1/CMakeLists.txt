# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Base examples (only doti.cpp exists without suffix)
add_executable(example_hipsparse_doti)
target_sources(example_hipsparse_doti PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_doti.cpp")
target_link_libraries(example_hipsparse_doti PRIVATE hipsparse::samples-common)

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_axpyi_cpp)
target_sources(example_hipsparse_axpyi_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_axpyi_cpp.cpp")
target_link_libraries(example_hipsparse_axpyi_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_doti_cpp)
target_sources(example_hipsparse_doti_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_doti_cpp.cpp")
target_link_libraries(example_hipsparse_doti_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gthr_cpp)
target_sources(example_hipsparse_gthr_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gthr_cpp.cpp")
target_link_libraries(example_hipsparse_gthr_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_roti_cpp)
target_sources(example_hipsparse_roti_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_roti_cpp.cpp")
target_link_libraries(example_hipsparse_roti_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sctr_cpp)
target_sources(example_hipsparse_sctr_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sctr_cpp.cpp")
target_link_libraries(example_hipsparse_sctr_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_axpyi_c)
target_sources(example_hipsparse_axpyi_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_axpyi_c.c")
target_link_libraries(example_hipsparse_axpyi_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_doti_c)
target_sources(example_hipsparse_doti_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_doti_c.c")
target_link_libraries(example_hipsparse_doti_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gthr_c)
target_sources(example_hipsparse_gthr_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gthr_c.c")
target_link_libraries(example_hipsparse_gthr_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_roti_c)
target_sources(example_hipsparse_roti_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_roti_c.c")
target_link_libraries(example_hipsparse_roti_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sctr_c)
target_sources(example_hipsparse_sctr_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sctr_c.c")
target_link_libraries(example_hipsparse_sctr_c PRIVATE hipsparse::samples-common)

# Fortran examples (only if Fortran support enabled and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_executable(example_hipsparse_axpyi_fortran)
    target_sources(example_hipsparse_axpyi_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_axpyi_fortran.f90")
    add_dependencies(example_hipsparse_axpyi_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_axpyi_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_doti_fortran)
    target_sources(example_hipsparse_doti_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_doti_fortran.f90")
    add_dependencies(example_hipsparse_doti_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_doti_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_gthr_fortran)
    target_sources(example_hipsparse_gthr_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gthr_fortran.f90")
    add_dependencies(example_hipsparse_gthr_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_gthr_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_roti_fortran)
    target_sources(example_hipsparse_roti_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_roti_fortran.f90")
    add_dependencies(example_hipsparse_roti_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_roti_fortran PRIVATE hipsparse::fortran-samples-common)
    
    add_executable(example_hipsparse_sctr_fortran)
    target_sources(example_hipsparse_sctr_fortran PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sctr_fortran.f90")
    add_dependencies(example_hipsparse_sctr_fortran hipsparse_fortran)
    target_link_libraries(example_hipsparse_sctr_fortran PRIVATE hipsparse::fortran-samples-common)
endif()
