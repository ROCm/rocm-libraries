# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(add_rocsparse_example EXAMPLE_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

    target_compile_options(${EXAMPLE_TARGET} PRIVATE -Wno-unused-command-line-argument)
    target_compile_features(${EXAMPLE_TARGET} PRIVATE cxx_std_17)
    
    target_include_directories(${EXAMPLE_TARGET} PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
    )
    
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE
        roc::rocsparse
        hip::host
        hip::device
    )
endfunction()

# Empty C program for checking rocsparse headers are C compatible
if(NOT WIN32)
    add_executable(example_c_headers example_c_headers.c)
    target_link_libraries(example_c_headers PRIVATE roc::rocsparse hip::host hip::device)
endif()

# Auxiliary
add_rocsparse_example(example_handle.cpp)

# Level 1
add_rocsparse_example(example_axpyi.cpp)
add_rocsparse_example(example_doti.cpp)
add_rocsparse_example(example_dotci.cpp)
add_rocsparse_example(example_gthr.cpp)
add_rocsparse_example(example_gthrz.cpp)
add_rocsparse_example(example_roti.cpp)
add_rocsparse_example(example_sctr.cpp)

# Level 2
add_rocsparse_example(example_bsrmv.cpp)
add_rocsparse_example(example_bsrsv.cpp)
add_rocsparse_example(example_coomv.cpp)
add_rocsparse_example(example_csrmv.cpp)
add_rocsparse_example(example_ellmv.cpp)
add_rocsparse_example(example_hybmv.cpp)
add_rocsparse_example(example_csrsv.cpp)
add_rocsparse_example(example_csritsv.cpp)
add_rocsparse_example(example_gebsrmv.cpp)
add_rocsparse_example(example_gemvi.cpp)
add_rocsparse_example(example_v2_spmv_csr.cpp)
add_rocsparse_example(example_sptrsv.cpp)

# Deprecated examples with warning suppression
add_rocsparse_example(example_spmv_coo.cpp)
target_compile_options(example_spmv_coo PRIVATE -Wno-deprecated-declarations)

add_rocsparse_example(example_spmv_csr.cpp)
target_compile_options(example_spmv_csr PRIVATE -Wno-deprecated-declarations)

add_rocsparse_example(example_spmv_ell.cpp)
target_compile_options(example_spmv_ell PRIVATE -Wno-deprecated-declarations)

# Level 3
add_rocsparse_example(example_bsrmm.cpp)
add_rocsparse_example(example_csrmm.cpp)
add_rocsparse_example(example_spmm_csr.cpp)
add_rocsparse_example(example_spmm_coo.cpp)
add_rocsparse_example(example_spmm_batched_csr.cpp)
add_rocsparse_example(example_spmm_batched_coo.cpp)
add_rocsparse_example(example_bsrsm.cpp)
add_rocsparse_example(example_csrsm.cpp)
add_rocsparse_example(example_gemmi.cpp)
add_rocsparse_example(example_gtsv.cpp)

# Precond
add_rocsparse_example(example_gpsv.cpp)

# Extra
add_rocsparse_example(example_csrgeam.cpp)
add_rocsparse_example(example_csrgemm.cpp)
add_rocsparse_example(example_spgemm.cpp)
add_rocsparse_example(example_spgeam.cpp)
add_rocsparse_example(example_spgeam_2.cpp)
add_rocsparse_example(example_spgeam_3.cpp)

# Managed
add_rocsparse_example(example_csrmv_managed.cpp)
add_rocsparse_example(example_csrsv_managed.cpp)

# Fortran examples
if(ROCSPARSE_ENABLE_FORTRAN AND NOT WIN32)
    function(add_rocsparse_fortran_example EXAMPLE_SOURCE)
        get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})
        
        target_compile_options(${EXAMPLE_TARGET} PRIVATE -Wno-unused-command-line-argument)
        
        target_include_directories(${EXAMPLE_TARGET} PRIVATE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
        )
        
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE
            roc::rocsparse
            rocsparse-fortran-module
            hip::host
            # Fortran calling C++ requires explicit C++ runtime deps
            Threads::Threads
            $<$<PLATFORM_ID:Linux>:m>
            $<$<PLATFORM_ID:Linux>:stdc++>
            $<$<PLATFORM_ID:Linux>:${CMAKE_DL_LIBS}>
        )
    endfunction()

    add_rocsparse_fortran_example(example_fortran_gtsv.f90)
    add_rocsparse_fortran_example(example_fortran_csric0.f90)
    add_rocsparse_fortran_example(example_fortran_csrilu0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrmm.f90)
    add_rocsparse_fortran_example(example_fortran_bsric0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrilu0.f90)
    add_rocsparse_fortran_example(example_fortran_bsrsv.f90)
    add_rocsparse_fortran_example(example_fortran_csrsv.f90)
    add_rocsparse_fortran_example(example_fortran_spmv.f90)
    add_rocsparse_fortran_example(example_fortran_csrsm.f90)
    add_rocsparse_fortran_example(example_fortran_gemmi.f90)
    add_rocsparse_fortran_example(example_fortran_auxiliary.f90)
    add_rocsparse_fortran_example(example_fortran_dotci.f90)
    add_rocsparse_fortran_example(example_fortran_roti.f90)
endif()

# Documentation examples
file(GLOB_RECURSE DOCUMENTATION_EXAMPLES documentation_examples/*.cpp)
foreach(file ${DOCUMENTATION_EXAMPLES})
    add_rocsparse_example(${file})
endforeach()
