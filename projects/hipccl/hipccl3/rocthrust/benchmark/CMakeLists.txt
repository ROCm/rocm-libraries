# ########################################################################
# Copyright 2024-2026 Advanced Micro Devices, Inc.
# ########################################################################

message(STATUS "Configuring benchmarks")

# Force CMake to find Git immediately so ${GIT_EXECUTABLE} is populated
find_package(Git REQUIRED)

# Get the current branch, or fall back to DETACHED
execute_process(
  COMMAND ${GIT_EXECUTABLE} symbolic-ref -q --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE BRANCH_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(BRANCH_NAME STREQUAL "")
  set(BRANCH_NAME "DETACHED")
endif()

# Get the short commit hash
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

function(add_thrust_benchmark BENCHMARK_SOURCE)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_LIST_DIR}/bench ${CMAKE_CURRENT_LIST_DIR}/${BENCHMARK_SOURCE})
    string(REPLACE ".cu" "" bench_name ${rel_path})
    string(REPLACE "/" "_" bench_name ${bench_name})

    set(BENCHMARK_TARGET benchmark_thrust_${bench_name})

    if(USE_HIPCXX)
        set_source_files_properties(${BENCHMARK_SOURCE}
            PROPERTIES
                LANGUAGE HIP
        )
    else()
        set_source_files_properties(${BENCHMARK_SOURCE}
            PROPERTIES
                LANGUAGE CXX
        )
    endif()

    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
    target_compile_options(${BENCHMARK_TARGET} PRIVATE ${COMPILE_OPTIONS})

    # NOTE(hipccl3): PRIVATE, not INTERFACE - same defect as ../test/CMakeLists.txt
    # (see the note there). ${BENCHMARK_TARGET} is an executable, so INTERFACE
    # usage requirements never reach its own compile line. Latent rather than
    # actively broken, since this branch only runs with
    # GRAFT_THRUST_ONTO_BINARIES=ON.
    if(GRAFT_THRUST_ONTO_BINARIES)
        target_link_libraries(${BENCHMARK_TARGET} PRIVATE CCCL::CCCL)
    else()
        target_link_libraries(${BENCHMARK_TARGET}
            PRIVATE
                rocthrust
                roc::rocprim_hip
        )
    endif()

    target_link_libraries(${BENCHMARK_TARGET} PRIVATE roc::rocrand)

    # NOTE(hipccl3): two extra ".." vs upstream
    # (rocm-libraries/projects/rocthrust) - this copy lives two directory
    # levels deeper (rocm-libraries/projects/hipccl/hipccl3/rocthrust), so the
    # relative path to the shared rocm-libraries/shared/primbench directory
    # needs two more "..".
    target_include_directories(${BENCHMARK_TARGET} PRIVATE
      "${PROJECT_SOURCE_DIR}/benchmark"
      "${PROJECT_SOURCE_DIR}/../../../../shared/primbench")

    target_compile_definitions(${BENCHMARK_TARGET} PUBLIC BRANCH_NAME="${BRANCH_NAME}")
    target_compile_definitions(${BENCHMARK_TARGET} PUBLIC COMMIT_HASH="${COMMIT_HASH}")

    if(WIN32)
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE PRIMBENCH_NO_MONITORING)
    elseif(HIP_COMPILER STREQUAL "nvcc")
        target_link_libraries(${BENCHMARK_TARGET} PRIVATE nvidia-ml)
    else()
        target_link_libraries(${BENCHMARK_TARGET} PRIVATE amd_smi)
    endif()

    foreach(gpu_target ${GPU_TARGETS})
        target_link_libraries(${BENCHMARK_TARGET}
            INTERFACE
                --cuda-gpu-arch=${gpu_target}
        )
    endforeach()

    # NOTE(hipccl3): ${PROJECT_BINARY_DIR}, not ${CMAKE_BINARY_DIR}, so a
    # nested build writes to <build>/rocthrust/benchmark rather than sharing
    # one directory with rocPRIM and hipCUB. PROJECT_BINARY_DIR equals
    # CMAKE_BINARY_DIR for a top-level project, so standalone builds land in
    # the same place as before. rocThrust's own targets are already prefixed
    # (benchmark_thrust_*) so it has no filename clash with the others - this
    # is for a consistent <build>/<component>/benchmark layout across all
    # three components.
    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/benchmark/
    )
    rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
endfunction()

add_thrust_benchmark(bench/adjacent_difference/basic.cu)
add_thrust_benchmark(bench/adjacent_difference/custom.cu)
add_thrust_benchmark(bench/adjacent_difference/in_place.cu)
add_thrust_benchmark(bench/copy/basic.cu)
add_thrust_benchmark(bench/copy/if.cu)
add_thrust_benchmark(bench/equal/basic.cu)
add_thrust_benchmark(bench/fill/basic.cu)
add_thrust_benchmark(bench/for_each/basic.cu)
add_thrust_benchmark(bench/inner_product/basic.cu)
add_thrust_benchmark(bench/merge/basic.cu)
add_thrust_benchmark(bench/partition/basic.cu)
add_thrust_benchmark(bench/reduce/basic.cu)
add_thrust_benchmark(bench/reduce/by_key.cu)
add_thrust_benchmark(bench/scan/exclusive/by_key.cu)
add_thrust_benchmark(bench/scan/exclusive/max.cu)
add_thrust_benchmark(bench/scan/exclusive/sum.cu)
add_thrust_benchmark(bench/scan/inclusive/by_key.cu)
add_thrust_benchmark(bench/scan/inclusive/max.cu)
add_thrust_benchmark(bench/scan/inclusive/sum.cu)
add_thrust_benchmark(bench/set_operations/difference.cu)
add_thrust_benchmark(bench/set_operations/difference_by_key.cu)
add_thrust_benchmark(bench/set_operations/intersection.cu)
add_thrust_benchmark(bench/set_operations/intersection_by_key.cu)
add_thrust_benchmark(bench/set_operations/symmetric_difference.cu)
add_thrust_benchmark(bench/set_operations/symmetric_difference_by_key.cu)
add_thrust_benchmark(bench/set_operations/union.cu)
add_thrust_benchmark(bench/set_operations/union_by_key.cu)
add_thrust_benchmark(bench/shuffle/basic.cu)
add_thrust_benchmark(bench/sort/keys.cu)
add_thrust_benchmark(bench/sort/keys_custom.cu)
add_thrust_benchmark(bench/sort/pairs.cu)
add_thrust_benchmark(bench/sort/pairs_custom.cu)
add_thrust_benchmark(bench/tabulate/basic.cu)
add_thrust_benchmark(bench/transform/basic.cu)
add_thrust_benchmark(bench/transform_reduce/sum.cu)
add_thrust_benchmark(bench/unique/basic.cu)
add_thrust_benchmark(bench/unique/by_key.cu)
add_thrust_benchmark(bench/vectorized_search/basic.cu)
add_thrust_benchmark(bench/vectorized_search/lower_bound.cu)
add_thrust_benchmark(bench/vectorized_search/upper_bound.cu)
