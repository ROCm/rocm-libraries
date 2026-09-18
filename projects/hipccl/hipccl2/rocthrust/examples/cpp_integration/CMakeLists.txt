# ########################################################################
# Copyright 2020-2025 Advanced Micro Devices, Inc.
# ########################################################################


# link host.cpp to device.cu

set(EXAMPLE_SOURCE host.cpp device.cu device.h)
set(EXAMPLE_TARGET example_thrust_linking)
set_source_files_properties(device.cu
    PROPERTIES
        LANGUAGE CXX
)
add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

target_compile_options(${EXAMPLE_TARGET} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
        rocthrust
        roc::rocprim_hip
)
foreach(gpu_target ${GPU_TARGETS})
    target_link_libraries(${EXAMPLE_TARGET}
        INTERFACE
            --cuda-gpu-arch=${gpu_target}
    )
endforeach()
set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cpp_integration"
)
