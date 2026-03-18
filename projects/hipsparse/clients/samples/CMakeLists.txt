# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C samples (for header compatibility check)
if(NOT WIN32 AND HIPSPARSE_ENABLE_HIP)
    add_executable(example_c_headers)
    target_sources(example_c_headers PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_c_headers.c"
    )
    target_link_libraries(example_c_headers PRIVATE hipsparse::samples-common)
    set_target_properties(example_c_headers PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    if(HIPSPARSE_ENABLE_HIP)
        add_executable(example_axpyi)
        target_sources(example_axpyi PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/example_axpyi.c"
        )
        target_link_libraries(example_axpyi PRIVATE hipsparse::samples-common)
        set_target_properties(example_axpyi PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
        )
    endif()
endif()


add_executable(example_handle)
target_sources(example_handle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_handle.cpp"
)
target_link_libraries(example_handle PRIVATE hipsparse::samples-common)
set_target_properties(example_handle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

# C++ samples that require older CUDA versions
if(NOT HIPSPARSE_ENABLE_CUDA OR (HIPSPARSE_ENABLE_CUDA AND CUDAToolkit_VERSION VERSION_LESS 11))
    add_executable(example_csrmv)
    target_sources(example_csrmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_csrmv.cpp"
    )
    target_link_libraries(example_csrmv PRIVATE hipsparse::samples-common)
    set_target_properties(example_csrmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    add_executable(example_hybmv)
    target_sources(example_hybmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_hybmv.cpp"
    )
    target_link_libraries(example_hybmv PRIVATE hipsparse::samples-common)
    set_target_properties(example_hybmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
endif()

# Fortran samples (only if Fortran support enabled, using HIP, and building with library)
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HIP AND HIPSPARSE_ENABLE_HOST)
    # Fortran example: example_fortran_auxiliary
    add_executable(example_fortran_auxiliary)
    target_sources(example_fortran_auxiliary PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_auxiliary.f90"
    )
    add_dependencies(example_fortran_auxiliary hipsparse_fortran)
    target_link_libraries(example_fortran_auxiliary PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_auxiliary PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_csrsv2
    add_executable(example_fortran_csrsv2)
    target_sources(example_fortran_csrsv2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrsv2.f90"
    )
    add_dependencies(example_fortran_csrsv2 hipsparse_fortran)
    target_link_libraries(example_fortran_csrsv2 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_csrsv2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_bsric02
    add_executable(example_fortran_bsric02)
    target_sources(example_fortran_bsric02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsric02.f90"
    )
    add_dependencies(example_fortran_bsric02 hipsparse_fortran)
    target_link_libraries(example_fortran_bsric02 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_bsric02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_csric02
    add_executable(example_fortran_csric02)
    target_sources(example_fortran_csric02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csric02.f90"
    )
    add_dependencies(example_fortran_csric02 hipsparse_fortran)
    target_link_libraries(example_fortran_csric02 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_csric02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_bsrilu02
    add_executable(example_fortran_bsrilu02)
    target_sources(example_fortran_bsrilu02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsrilu02.f90"
    )
    add_dependencies(example_fortran_bsrilu02 hipsparse_fortran)
    target_link_libraries(example_fortran_bsrilu02 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_bsrilu02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_csrilu02
    add_executable(example_fortran_csrilu02)
    target_sources(example_fortran_csrilu02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrilu02.f90"
    )
    add_dependencies(example_fortran_csrilu02 hipsparse_fortran)
    target_link_libraries(example_fortran_csrilu02 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_csrilu02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_csrsm2
    add_executable(example_fortran_csrsm2)
    target_sources(example_fortran_csrsm2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrsm2.f90"
    )
    add_dependencies(example_fortran_csrsm2 hipsparse_fortran)
    target_link_libraries(example_fortran_csrsm2 PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_csrsm2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_dotci
    add_executable(example_fortran_dotci)
    target_sources(example_fortran_dotci PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_dotci.f90"
    )
    add_dependencies(example_fortran_dotci hipsparse_fortran)
    target_link_libraries(example_fortran_dotci PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_dotci PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_roti
    add_executable(example_fortran_roti)
    target_sources(example_fortran_roti PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_roti.f90"
    )
    add_dependencies(example_fortran_roti hipsparse_fortran)
    target_link_libraries(example_fortran_roti PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_roti PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_spmv
    add_executable(example_fortran_spmv)
    target_sources(example_fortran_spmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_spmv.f90"
    )
    add_dependencies(example_fortran_spmv hipsparse_fortran)
    target_link_libraries(example_fortran_spmv PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_spmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
    
    # Fortran example: example_fortran_bsrmm
    add_executable(example_fortran_bsrmm)
    target_sources(example_fortran_bsrmm PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsrmm.f90"
    )
    add_dependencies(example_fortran_bsrmm hipsparse_fortran)
    target_link_libraries(example_fortran_bsrmm PRIVATE hipsparse::fortran-samples-common)
    set_target_properties(example_fortran_bsrmm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
    )
endif()

# Documentation examples (HIP only)
if(HIPSPARSE_ENABLE_HIP)
    add_subdirectory(documentation_examples)
endif()
