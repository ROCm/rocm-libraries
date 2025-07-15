# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../library/src/hcc_detail/rocsparselt")

option(ROCSPARSELT_ENABLE_TENSILELITE "Enable tensilelite" ON)

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
    # "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_compress.cpp"
    # "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_prune.cpp"
    # "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_spmm.cpp"
)

target_include_directories(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/include"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

if(NOT ROCSPARSELT_ENABLE_TENSILELITE)
    message(FATAL_ERROR "ROCSPARSELT_ENABLE_TENSILELITE can only be ON in the next-cmake build system")
    # TODO: There is some additional complexity in here that I am willfully ignoring for now
    # TODO: If these files aren't needed then they can be deleted?
    # include(src/hcc_detail/rocsparselt/src/spmm/hip/CMakeLists.txt)
    # include(src/hcc_detail/rocsparselt/src/spmm/kernels/CMakeLists.txt)

    # TODO: If this isn't needed can these files be deleted?
    # target_sources(hipsparselt PRIVATE
    #     "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_compress.cpp"
    #     "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_prune.cpp"
    #     "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/hip/rocsparselt_spmm.cpp"
    # )
else()
    target_sources(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/tensile_host.cpp")
    target_include_directories(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/Tensile")
    
    set(tensile_source_files "${_CMAKE_CURRENT_SOURCE_DIR}/src/tensile_host.cpp")
    list(APPEND hipsparselt_source_backend ${tensile_source_files})

    # TODO: Not needed I don't think
    # target_include_directories(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/Tensile")

    if(HIPSPARSELT_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
        target_link_libraries(hipsparselt PRIVATE tensilelite::tensilelite-host)
    else()
        # TODO: Shouldn't the tensilelite-host be propogating these properties when linked against?
        # For static builds, get tensilelite-host properties and apply them
        get_target_property(tensilelite-host_INCLUDES tensilelite::tensilelite-host INCLUDE_DIRECTORIES)
        if(tensilelite-host_INCLUDES)
            target_include_directories(hipsparselt PRIVATE ${tensilelite-host_INCLUDES})
        endif()
        
        get_target_property(tensilelite-host_DEFINES tensilelite::tensilelite-host COMPILE_DEFINITIONS)
        if(tensilelite-host_DEFINES)
            target_compile_definitions(hipsparselt PRIVATE ${tensilelite-host_DEFINES})
        endif()
        
        add_dependencies(hipsparselt tensilelite::tensilelite-host)
    endif()
endif()