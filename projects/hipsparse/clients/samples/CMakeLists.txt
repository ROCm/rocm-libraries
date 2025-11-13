# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Common utility source for samples
set(HIPSPARSE_SAMPLES_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../common/utility.cpp")

# C samples (for header compatibility check)
if(NOT WIN32)
    add_executable(example_c_headers)
    target_sources(example_c_headers PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_c_headers.c"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_c_headers PRIVATE cxx_std_17)
    target_include_directories(example_c_headers PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    target_link_libraries(example_c_headers PRIVATE roc::hipsparse)
    if(HIPSPARSE_ENABLE_CUDA)
        target_compile_definitions(example_c_headers PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(example_c_headers PRIVATE ${HIP_INCLUDE_DIRS})
        target_link_libraries(example_c_headers PRIVATE ${CUDA_LIBRARIES})
    elseif(HIPSPARSE_ENABLE_HIP)
        target_link_libraries(example_c_headers PRIVATE hip::host hip::device)
    endif()
    set_target_properties(example_c_headers PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    if(HIPSPARSE_ENABLE_HIP)
        add_executable(example_axpyi)
        target_sources(example_axpyi PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/example_axpyi.c"
            ${HIPSPARSE_SAMPLES_COMMON}
        )
        target_compile_features(example_axpyi PRIVATE cxx_std_17)
        target_include_directories(example_axpyi PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        )
        target_link_libraries(example_axpyi PRIVATE roc::hipsparse hip::host hip::device)
        set_target_properties(example_axpyi PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
        )
    endif()
endif()

# C++ sample: example_handle
add_executable(example_handle)
target_sources(example_handle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_handle.cpp"
    ${HIPSPARSE_SAMPLES_COMMON}
)
target_compile_features(example_handle PRIVATE cxx_std_17)
target_include_directories(example_handle PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(example_handle PRIVATE roc::hipsparse)
if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(example_handle PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(example_handle PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(example_handle PRIVATE ${CUDA_LIBRARIES})
elseif(HIPSPARSE_ENABLE_HIP)
    target_link_libraries(example_handle PRIVATE hip::host hip::device)
endif()
set_target_properties(example_handle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# C++ samples that require older CUDA versions
if(NOT HIPSPARSE_ENABLE_CUDA OR (HIPSPARSE_ENABLE_CUDA AND CUDA_VERSION LESS 11))
    add_executable(example_csrmv)
    target_sources(example_csrmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_csrmv.cpp"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_csrmv PRIVATE cxx_std_17)
    target_include_directories(example_csrmv PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    target_link_libraries(example_csrmv PRIVATE roc::hipsparse)
    if(HIPSPARSE_ENABLE_CUDA)
        target_compile_definitions(example_csrmv PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(example_csrmv PRIVATE ${HIP_INCLUDE_DIRS})
        target_link_libraries(example_csrmv PRIVATE ${CUDA_LIBRARIES})
    elseif(HIPSPARSE_ENABLE_HIP)
        target_link_libraries(example_csrmv PRIVATE hip::host hip::device)
    endif()
    set_target_properties(example_csrmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    add_executable(example_hybmv)
    target_sources(example_hybmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_hybmv.cpp"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_hybmv PRIVATE cxx_std_17)
    target_include_directories(example_hybmv PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    target_link_libraries(example_hybmv PRIVATE roc::hipsparse)
    if(HIPSPARSE_ENABLE_CUDA)
        target_compile_definitions(example_hybmv PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(example_hybmv PRIVATE ${HIP_INCLUDE_DIRS})
        target_link_libraries(example_hybmv PRIVATE ${CUDA_LIBRARIES})
    elseif(HIPSPARSE_ENABLE_HIP)
        target_link_libraries(example_hybmv PRIVATE hip::host hip::device)
    endif()
    set_target_properties(example_hybmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
endif()

# Fortran samples (only if not on Windows, using HIP, and building with library)
if(NOT WIN32 AND HIPSPARSE_ENABLE_HIP AND NOT HIPSPARSE_BUILD_CLIENTS_ONLY)
    # Fortran example: example_fortran_auxiliary
    add_executable(example_fortran_auxiliary)
    target_sources(example_fortran_auxiliary PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_auxiliary.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_auxiliary PRIVATE cxx_std_17)
    target_include_directories(example_fortran_auxiliary PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_auxiliary hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_auxiliary PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_auxiliary PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_auxiliary PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_auxiliary PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_csrsv2
    add_executable(example_fortran_csrsv2)
    target_sources(example_fortran_csrsv2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrsv2.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_csrsv2 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_csrsv2 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_csrsv2 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_csrsv2 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_csrsv2 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_csrsv2 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_csrsv2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_bsric02
    add_executable(example_fortran_bsric02)
    target_sources(example_fortran_bsric02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsric02.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_bsric02 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_bsric02 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_bsric02 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_bsric02 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_bsric02 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_bsric02 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_bsric02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_csric02
    add_executable(example_fortran_csric02)
    target_sources(example_fortran_csric02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csric02.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_csric02 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_csric02 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_csric02 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_csric02 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_csric02 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_csric02 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_csric02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_bsrilu02
    add_executable(example_fortran_bsrilu02)
    target_sources(example_fortran_bsrilu02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsrilu02.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_bsrilu02 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_bsrilu02 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_bsrilu02 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_bsrilu02 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_bsrilu02 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_bsrilu02 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_bsrilu02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_csrilu02
    add_executable(example_fortran_csrilu02)
    target_sources(example_fortran_csrilu02 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrilu02.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_csrilu02 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_csrilu02 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_csrilu02 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_csrilu02 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_csrilu02 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_csrilu02 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_csrilu02 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_csrsm2
    add_executable(example_fortran_csrsm2)
    target_sources(example_fortran_csrsm2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_csrsm2.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_csrsm2 PRIVATE cxx_std_17)
    target_include_directories(example_fortran_csrsm2 PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_csrsm2 hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_csrsm2 PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_csrsm2 PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_csrsm2 PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_csrsm2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_dotci
    add_executable(example_fortran_dotci)
    target_sources(example_fortran_dotci PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_dotci.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_dotci PRIVATE cxx_std_17)
    target_include_directories(example_fortran_dotci PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_dotci hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_dotci PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_dotci PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_dotci PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_dotci PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_roti
    add_executable(example_fortran_roti)
    target_sources(example_fortran_roti PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_roti.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_roti PRIVATE cxx_std_17)
    target_include_directories(example_fortran_roti PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_roti hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_roti PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_roti PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_roti PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_roti PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_spmv
    add_executable(example_fortran_spmv)
    target_sources(example_fortran_spmv PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_spmv.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_spmv PRIVATE cxx_std_17)
    target_include_directories(example_fortran_spmv PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_spmv hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_spmv PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_spmv PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_spmv PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_spmv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
    
    # Fortran example: example_fortran_bsrmm
    add_executable(example_fortran_bsrmm)
    target_sources(example_fortran_bsrmm PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_bsrmm.f90"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_fortran_bsrmm PRIVATE cxx_std_17)
    target_include_directories(example_fortran_bsrmm PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    )
    add_dependencies(example_fortran_bsrmm hipsparse_fortran)
    # Fortran linking C++: Need explicit C++ runtime dependencies
    target_link_libraries(example_fortran_bsrmm PRIVATE 
        roc::hipsparse 
        hip::host 
        hip::device
    )
    if(TARGET Threads::Threads)
        target_link_libraries(example_fortran_bsrmm PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(example_fortran_bsrmm PRIVATE m ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(example_fortran_bsrmm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
endif()

# Documentation examples (HIP only) - explicitly listed to avoid file(GLOB) and foreach
if(HIPSPARSE_ENABLE_HIP)
    # Level1 examples
    add_executable(example_hipsparse_axpyi)
    target_sources(example_hipsparse_axpyi PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level1/example_hipsparse_axpyi.cpp"
        ${HIPSPARSE_SAMPLES_COMMON}
    )
    target_compile_features(example_hipsparse_axpyi PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_axpyi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_axpyi PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_axpyi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_doti)
    target_sources(example_hipsparse_doti PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level1/example_hipsparse_doti.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_doti PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_doti PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_doti PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_doti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gthr)
    target_sources(example_hipsparse_gthr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level1/example_hipsparse_gthr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gthr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gthr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gthr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gthr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_roti)
    target_sources(example_hipsparse_roti PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level1/example_hipsparse_roti.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_roti PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_roti PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_roti PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_roti PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_sctr)
    target_sources(example_hipsparse_sctr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level1/example_hipsparse_sctr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_sctr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_sctr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_sctr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_sctr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Level2 examples
    add_executable(example_hipsparse_bsrmv)
    target_sources(example_hipsparse_bsrmv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_bsrmv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsrmv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsrmv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsrmv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsrmv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_bsrsv2)
    target_sources(example_hipsparse_bsrsv2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_bsrsv2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsrsv2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsrsv2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsrsv2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsrsv2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrmv)
    target_sources(example_hipsparse_csrmv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_csrmv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrmv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrmv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrmv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrmv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrsv2)
    target_sources(example_hipsparse_csrsv2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_csrsv2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrsv2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrsv2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrsv2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrsv2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gemvi)
    target_sources(example_hipsparse_gemvi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_gemvi.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gemvi PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gemvi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gemvi PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gemvi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_hybmv)
    target_sources(example_hipsparse_hybmv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level2/example_hipsparse_hybmv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_hybmv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_hybmv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_hybmv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_hybmv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Level3 examples
    add_executable(example_hipsparse_bsrmm)
    target_sources(example_hipsparse_bsrmm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level3/example_hipsparse_bsrmm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsrmm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsrmm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsrmm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsrmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_bsrsm)
    target_sources(example_hipsparse_bsrsm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level3/example_hipsparse_bsrsm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsrsm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsrsm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsrsm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsrsm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrmm)
    target_sources(example_hipsparse_csrmm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level3/example_hipsparse_csrmm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrmm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrmm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrmm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrsm2)
    target_sources(example_hipsparse_csrsm2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level3/example_hipsparse_csrsm2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrsm2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrsm2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrsm2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrsm2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gemmi)
    target_sources(example_hipsparse_gemmi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/level3/example_hipsparse_gemmi.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gemmi PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gemmi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gemmi PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gemmi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Extra examples
    add_executable(example_hipsparse_csrgeam)
    target_sources(example_hipsparse_csrgeam PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/extra/example_hipsparse_csrgeam.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrgeam PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrgeam PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrgeam PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrgeam PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrgeam2)
    target_sources(example_hipsparse_csrgeam2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/extra/example_hipsparse_csrgeam2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrgeam2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrgeam2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrgeam2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrgeam2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrgemm)
    target_sources(example_hipsparse_csrgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/extra/example_hipsparse_csrgemm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrgemm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrgemm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrgemm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrgemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrgemm2)
    target_sources(example_hipsparse_csrgemm2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/extra/example_hipsparse_csrgemm2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrgemm2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrgemm2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrgemm2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrgemm2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Precond examples
    add_executable(example_hipsparse_bsric02)
    target_sources(example_hipsparse_bsric02 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_bsric02.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsric02 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsric02 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsric02 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsric02 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_bsrilu02)
    target_sources(example_hipsparse_bsrilu02 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_bsrilu02.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsrilu02 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsrilu02 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsrilu02 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsrilu02 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csric02)
    target_sources(example_hipsparse_csric02 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_csric02.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csric02 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csric02 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csric02 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csric02 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrilu02)
    target_sources(example_hipsparse_csrilu02 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_csrilu02.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrilu02 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrilu02 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrilu02 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrilu02 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gtsv)
    target_sources(example_hipsparse_gtsv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_gtsv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gtsv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gtsv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gtsv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gtsv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gtsv_interleaved_batch)
    target_sources(example_hipsparse_gtsv_interleaved_batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_gtsv_interleaved_batch.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gtsv_interleaved_batch PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gtsv_interleaved_batch PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gtsv_interleaved_batch PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gtsv_interleaved_batch PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gtsv_no_pivot)
    target_sources(example_hipsparse_gtsv_no_pivot PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/precond/example_hipsparse_gtsv_no_pivot.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gtsv_no_pivot PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gtsv_no_pivot PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gtsv_no_pivot PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gtsv_no_pivot PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Conversion examples
    add_executable(example_hipsparse_bsr2csr)
    target_sources(example_hipsparse_bsr2csr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_bsr2csr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_bsr2csr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_bsr2csr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_bsr2csr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_bsr2csr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_coo2csr)
    target_sources(example_hipsparse_coo2csr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_coo2csr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_coo2csr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_coo2csr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_coo2csr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_coo2csr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_coosort_by_column)
    target_sources(example_hipsparse_coosort_by_column PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_coosort_by_column.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_coosort_by_column PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_coosort_by_column PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_coosort_by_column PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_coosort_by_column PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_coosort_by_row)
    target_sources(example_hipsparse_coosort_by_row PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_coosort_by_row.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_coosort_by_row PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_coosort_by_row PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_coosort_by_row PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_coosort_by_row PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_create_identity_permutation)
    target_sources(example_hipsparse_create_identity_permutation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_create_identity_permutation.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_create_identity_permutation PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_create_identity_permutation PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_create_identity_permutation PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_create_identity_permutation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_cscsort)
    target_sources(example_hipsparse_cscsort PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_cscsort.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_cscsort PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_cscsort PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_cscsort PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_cscsort PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2bsr)
    target_sources(example_hipsparse_csr2bsr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2bsr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2bsr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2bsr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2bsr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2bsr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2csc)
    target_sources(example_hipsparse_csr2csc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2csc.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2csc PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2csc PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2csc PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2csc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2csc_ex2)
    target_sources(example_hipsparse_csr2csc_ex2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2csc_ex2.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2csc_ex2 PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2csc_ex2 PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2csc_ex2 PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2csc_ex2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2csr_compress)
    target_sources(example_hipsparse_csr2csr_compress PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2csr_compress.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2csr_compress PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2csr_compress PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2csr_compress PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2csr_compress PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2dense)
    target_sources(example_hipsparse_csr2dense PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2dense.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2dense PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2dense PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2dense PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2dense PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2gebsr)
    target_sources(example_hipsparse_csr2gebsr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2gebsr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2gebsr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2gebsr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2gebsr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2gebsr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csr2hyb)
    target_sources(example_hipsparse_csr2hyb PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csr2hyb.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csr2hyb PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csr2hyb PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csr2hyb PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csr2hyb PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_csrsort)
    target_sources(example_hipsparse_csrsort PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_csrsort.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_csrsort PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_csrsort PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_csrsort PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_csrsort PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_dense2csr)
    target_sources(example_hipsparse_dense2csr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_dense2csr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_dense2csr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_dense2csr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_dense2csr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_dense2csr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gebsr2csr)
    target_sources(example_hipsparse_gebsr2csr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_gebsr2csr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gebsr2csr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gebsr2csr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gebsr2csr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gebsr2csr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gebsr2gebsc)
    target_sources(example_hipsparse_gebsr2gebsc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_gebsr2gebsc.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gebsr2gebsc PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gebsr2gebsc PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gebsr2gebsc PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gebsr2gebsc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gebsr2gebsr)
    target_sources(example_hipsparse_gebsr2gebsr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_gebsr2gebsr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gebsr2gebsr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gebsr2gebsr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gebsr2gebsr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gebsr2gebsr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_nnz_compress)
    target_sources(example_hipsparse_nnz_compress PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_nnz_compress.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_nnz_compress PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_nnz_compress PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_nnz_compress PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_nnz_compress PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_prune_dense2csr)
    target_sources(example_hipsparse_prune_dense2csr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_prune_dense2csr.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_prune_dense2csr PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_prune_dense2csr PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_prune_dense2csr PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_prune_dense2csr PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_prune_dense2csr_by_percentage)
    target_sources(example_hipsparse_prune_dense2csr_by_percentage PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/conversion/example_hipsparse_prune_dense2csr_by_percentage.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_prune_dense2csr_by_percentage PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_prune_dense2csr_by_percentage PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_prune_dense2csr_by_percentage PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_prune_dense2csr_by_percentage PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    # Generic examples
    add_executable(example_hipsparse_axpby)
    target_sources(example_hipsparse_axpby PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_axpby.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_axpby PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_axpby PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_axpby PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_axpby PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_dense_to_sparse)
    target_sources(example_hipsparse_dense_to_sparse PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_dense_to_sparse.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_dense_to_sparse PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_dense_to_sparse PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_dense_to_sparse PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_dense_to_sparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_gather)
    target_sources(example_hipsparse_gather PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_gather.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_gather PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_gather PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_gather PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_gather PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_rot)
    target_sources(example_hipsparse_rot PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_rot.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_rot PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_rot PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_rot PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_rot PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_scatter)
    target_sources(example_hipsparse_scatter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_scatter.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_scatter PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_scatter PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_scatter PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_scatter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_sddmm)
    target_sources(example_hipsparse_sddmm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_sddmm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_sddmm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_sddmm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_sddmm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_sddmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_sparse_to_dense)
    target_sources(example_hipsparse_sparse_to_dense PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_sparse_to_dense.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_sparse_to_dense PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_sparse_to_dense PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_sparse_to_dense PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_sparse_to_dense PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spgemm)
    target_sources(example_hipsparse_spgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spgemm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spgemm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spgemm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spgemm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spgemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spgemm_reuse)
    target_sources(example_hipsparse_spgemm_reuse PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spgemm_reuse.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spgemm_reuse PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spgemm_reuse PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spgemm_reuse PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spgemm_reuse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spmm)
    target_sources(example_hipsparse_spmm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spmm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spmm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spmm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spmm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spmv)
    target_sources(example_hipsparse_spmv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spmv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spmv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spmv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spmv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spmv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spsm)
    target_sources(example_hipsparse_spsm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spsm.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spsm PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spsm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spsm PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spsm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spsv)
    target_sources(example_hipsparse_spsv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spsv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spsv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spsv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spsv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spsv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    
    add_executable(example_hipsparse_spvv)
    target_sources(example_hipsparse_spvv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/documentation_examples/generic/example_hipsparse_spvv.cpp" ${HIPSPARSE_SAMPLES_COMMON})
    target_compile_features(example_hipsparse_spvv PRIVATE cxx_std_17)
    target_include_directories(example_hipsparse_spvv PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_link_libraries(example_hipsparse_spvv PRIVATE roc::hipsparse hip::host)
    set_target_properties(example_hipsparse_spvv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endif()
