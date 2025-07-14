# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../library/src/hcc_detail/rocsparselt")

# TensileLite compile definitions
target_compile_definitions(hipsparselt PRIVATE ROCSPARSELT_TENSILE_LAZY_LOAD=1)

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/handle.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/Debug.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/status.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/utility.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/rocsparselt_auxiliary.cpp"
)

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_compress.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_prune.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_spmm.cpp"
)

target_include_directories(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/include"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

if(NOT HIPSPARSELT_ENABLE_TENSILELITE)
    # There is some additional complexity in here that I am willfully ignoring for now
    # include(src/hcc_detail/rocsparselt/src/spmm/hip/CMakeLists.txt)
    # include(src/hcc_detail/rocsparselt/src/spmm/kernels/CMakeLists.txt)

    target_sources(hipsparselt PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_compress.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_prune.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_spmm.cpp"
    )
else()
    set(tensile_source_files "${_CMAKE_CURRENT_SOURCE_DIR}/src/tensile_host.cpp")
    list(APPEND hipsparselt_source_backend ${tensile_source_files})

    # Not needed I don't think
    # target_include_directories(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/Tensile")

    if(HIPSPARSELT_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
        target_link_libraries(hipsparselt PRIVATE TensileHost)
    else()
        # TODO: Shouldn't the TensileHost be propogating these properties when linked against?
        # For static builds, get TensileHost properties and apply them
        get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
        if(TensileHost_INCLUDES)
            target_include_directories(hipsparselt PRIVATE ${TensileHost_INCLUDES})
        endif()
        
        get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
        if(TensileHost_DEFINES)
            target_compile_definitions(hipsparselt PRIVATE ${TensileHost_DEFINES})
        endif()
        
        add_dependencies(hipsparselt TensileHost)
    endif()

    add_dependencies(hipsparselt TensileLibrary)
endif()