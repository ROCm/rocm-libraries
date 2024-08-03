# ########################################################################
# Copyright 2024 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust benchmarks
# ###########################

# Common functionality for configuring rocThrust's benchmarks

function(find_rocrand)
    # rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
    if(NOT DOWNLOAD_ROCRAND)
        find_package(rocrand QUIET)
    endif()
    if(NOT rocrand_FOUND)
        message(STATUS "Downloading and building rocrand.")
        set(ROCRAND_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand CACHE PATH "")

        download_project(
        PROJ                rocrand
        GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocRAND.git
        GIT_TAG             develop
        INSTALL_DIR         ${ROCRAND_ROOT}
        CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
        LOG_DOWNLOAD        TRUE
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        BUILD_PROJECT       TRUE
        UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
        )
        find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT})
    endif()
endfunction()

# Registers a .cu as C++ rocThrust benchmark
function(add_thrust_benchmark BENCHMARK_NAME BENCHMARK_SOURCE NOT_INTERNAL)
    set(BENCHMARK_TARGET "benchmark_thrust_${BENCHMARK_NAME}")
    set_source_files_properties(${BENCHMARK_SOURCE}
        PROPERTIES
            LANGUAGE CXX
    )
    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})

    target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
            rocthrust
            roc::rocprim_hip
    )
    # Internal benchmark does not use Google Benchmark nor rocRAND.
    # This can be omited when that benchmark is removed.
    if(NOT_INTERNAL)
        find_rocrand()
        target_link_libraries(${BENCHMARK_TARGET}
            PRIVATE
                roc::rocrand
                benchmark::benchmark
    )
    endif()
    foreach(gpu_target ${GPU_TARGETS})
        target_link_libraries(${BENCHMARK_TARGET}
            INTERFACE
                --cuda-gpu-arch=${gpu_target}
        )
    endforeach()

    # Separate normal from internal benchmarks
    if(NOT_INTERNAL)
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/benchmarks/")
    else()
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/benchmarks/internal/")
    endif()

    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    )
    rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
endfunction()
