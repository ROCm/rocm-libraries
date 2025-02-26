# ########################################################################
# Copyright 2024 Advanced Micro Devices, Inc.
# ########################################################################

function(add_hipstdpar_test TEST TEST_TYPE INTERPOSE_ALLOC)
    set(TEST_SOURCE "test_${TEST}.cpp")
    if(USE_HIPCXX)
        set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE HIP)
    endif()
    set(TEST_TARGET "${TEST}_${TEST_TYPE}.hip")

    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_compile_options(${TEST_TARGET}
        PRIVATE
            --hipstdpar
            --hipstdpar-path=${HIPSTDPAR_LOCATION}
            --hipstdpar-thrust-path=${THRUST_LOCATION}
            --hipstdpar-prim-path=${ROCPRIM_LOCATION})
    if(INTERPOSE_ALLOC)
        target_compile_options(${TEST_TARGET}
            PRIVATE
                --hipstdpar-interpose-alloc
        )
    endif()
    target_link_libraries(${TEST_TARGET}
        PRIVATE
            --hipstdpar
            Threads::Threads
    )

    if(TARGET TBB::tbb OR TARGET tbb)
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                TBB::tbb
        )
    endif()
    
    if (NOT WIN32)
      foreach(gpu_target ${GPU_TARGETS})
          target_link_libraries(${TEST_TARGET}
              PRIVATE
                  --offload-arch=${gpu_target}
          )
      endforeach()
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/hipstdpar"
    )
    if(AMDGPU_TEST_TARGETS)
    foreach(AMDGPU_TARGET IN LISTS AMDGPU_TEST_TARGETS)
        add_test("${AMDGPU_TARGET}-${TEST_TARGET}" ${TEST_TARGET})
        set_tests_properties("${AMDGPU_TARGET}-${TEST_TARGET}"
            PROPERTIES
                RESOURCE_GROUPS "1,${AMDGPU_TARGET}:1"
                LABELS "hip;${AMDGPU_TARGET}"
        )
    endforeach()
    else()
        add_relative_test(${TEST_TARGET} ${TEST_TARGET})
        set_tests_properties(${TEST_TARGET}
            PROPERTIES
                LABELS "hip"
        )
    endif()

    rocm_install(TARGETS ${TEST_TARGET} COMPONENT tests)
    if (WIN32 AND NOT DEFINED DLLS_COPIED)
      set(DLLS_COPIED "YES")
      set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
      # for now adding in all .dll as dependency chain is not cmake based on win32
      file( GLOB third_party_dlls
            LIST_DIRECTORIES ON
            CONFIGURE_DEPENDS
            ${HIP_DIR}/bin/*.dll
            ${CMAKE_SOURCE_DIR}/rtest.*
      )
      foreach( file_i ${third_party_dlls})
        add_custom_command( TARGET ${TEST_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test )
      endforeach( file_i )
    endif()
endfunction()

# Make sure that hipstdpar tests are compiled for C++17, without changing the global state
set(ROCTHRUST_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD 17)

# Dependencies
find_package(Threads REQUIRED)

# Define where to find rocThrust, hipstdpar and rocPRIM headers
set(THRUST_LOCATION ${PROJECT_SOURCE_DIR})
set(HIPSTDPAR_LOCATION ${THRUST_LOCATION}/thrust/system/hip/hipstdpar)
set(ROCPRIM_LOCATION ${ROCPRIM_INCLUDE_DIR})

# Add tests
add_hipstdpar_test("algorithms" "compile" OFF)
add_hipstdpar_test("interpose" "compile" ON)

# Restore global state
set(CMAKE_CXX_STANDARD ${ROCTHRUST_CMAKE_CXX_STANDARD})
