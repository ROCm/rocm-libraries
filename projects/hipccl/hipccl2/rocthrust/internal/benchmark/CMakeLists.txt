# ########################################################################
# Copyright 2020 Advanced Micro Devices, Inc.
# ########################################################################

function(add_thrust_benchmark BENCHMARK)
    set(BENCHMARK_SOURCE "${BENCHMARK}.cu")
    set(BENCHMARK_TARGET "benchmark_thrust_${BENCHMARK}")
    set_source_files_properties(${BENCHMARK_SOURCE} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})

    target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
            rocthrust
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(${BENCHMARK_TARGET}
            PRIVATE
                --amdgpu-target=${amdgpu_target}
        )
    endforeach()
    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/"
    )
endfunction()

# ****************************************************************************
# Benchmarks
# ****************************************************************************
message (STATUS "Building benchmarks")

add_thrust_benchmark("bench")
