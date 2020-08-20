# ########################################################################
# Copyright 2020 Advanced Micro Devices, Inc.
# ########################################################################


# link host.cpp to device.cu

set(EXAMPLE_SOURCE host.cpp device.cu device.h)
set(EXAMPLE_TARGET example_thrust_linking)
set_source_files_properties(device.cu
    PROPERTIES
        LANGUAGE CXX
)
add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
        rocthrust
        roc::rocprim_hip
)
foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${EXAMPLE_TARGET}
        INTERFACE
            --cuda-gpu-arch=${amdgpu_target}
    )
endforeach()
set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cpp_integration"
)
