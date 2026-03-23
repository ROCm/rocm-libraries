# Try to find an installed GTest first. QUIET suppresses the "not found" message
# so we can fall back to FetchContent without noisy warnings.
find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        SYSTEM
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(race-emulator-tests
    e2e_hip_general.cpp
    e2e_hip_kittens.cpp
    e2e_iree.cpp
    e2e_tensilelite.cpp
    race_tests.cpp
    vector_instruction_tests.cpp
    scalar_instruction_tests.cpp
    memory_instruction_tests.cpp
    control_flow_instruction_tests.cpp
    parsing_tests.cpp
    profiler_tests.cpp
    interval_set_tests.cpp
    util_tests.cpp
    float_types_tests.cpp
)

# RELEASE_BUILD is used by tests that are too slow for debug builds.
# Tests check `#ifndef RELEASE_BUILD` and call GTEST_SKIP() when absent.
target_compile_definitions(race-emulator-tests PRIVATE
    TEST_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/asm"
    $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:RELEASE_BUILD>
)

target_link_libraries(race-emulator-tests PRIVATE race-emulator GTest::gtest_main)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(race-emulator-tests PRIVATE -Wall -Wextra -pedantic)
endif()

include(GoogleTest)
gtest_discover_tests(race-emulator-tests)

# GPU hardware validation tests (requires hipcc and a GPU to run).
# Set RACE_EMULATOR_BUILD_HW_TESTING to a GPU target (e.g. gfx942) to enable.
if(RACE_EMULATOR_BUILD_HW_TESTING)
    set(_hw_target ${RACE_EMULATOR_BUILD_HW_TESTING})
    set(_hw_test_dir gpu_validation/${_hw_target})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_hw_test_dir})
        message(FATAL_ERROR
            "No HW validation tests for target '${_hw_target}'. "
            "Expected directory: ${_hw_test_dir}")
    endif()
    # Auto-detect the HIP compiler from the ROCm installation so that
    # users only need -DCMAKE_PREFIX_PATH=/opt/rocm (or similar) rather
    # than also having to set -DCMAKE_HIP_COMPILER explicitly.
    if(NOT CMAKE_HIP_COMPILER)
        find_program(_rocm_clangpp clang++
            PATHS "${ROCM_PATH}/lib/llvm/bin"
                  "${ROCM_PATH}/llvm/bin"
                  "/opt/rocm/lib/llvm/bin"
                  "/opt/rocm/llvm/bin"
            NO_DEFAULT_PATH)
        if(_rocm_clangpp)
            set(CMAKE_HIP_COMPILER "${_rocm_clangpp}" CACHE FILEPATH
                "HIP compiler (auto-detected from ROCm)")
        endif()
    endif()
    enable_language(HIP)
    find_package(hip REQUIRED)

    # Assemble .s files into .co code objects for GPU loading.
    # Uses ROCm clang to produce AMDGPU ELF code objects.
    find_program(ROCM_CLANG clang
        PATHS "${ROCM_PATH}/lib/llvm/bin"
              "${ROCM_PATH}/llvm/bin"
              "/opt/rocm/lib/llvm/bin"
              "/opt/rocm/llvm/bin"
        NO_DEFAULT_PATH)
    if(NOT ROCM_CLANG)
        find_program(ROCM_CLANG clang)
    endif()
    set(_asm_co_outputs)
    file(GLOB _asm_sources ${CMAKE_CURRENT_SOURCE_DIR}/asm/${_hw_target}/*.s)
    foreach(_asm_src ${_asm_sources})
        get_filename_component(_asm_name ${_asm_src} NAME_WE)
        set(_co_output ${CMAKE_CURRENT_BINARY_DIR}/${_asm_name}.co)
        add_custom_command(
            OUTPUT ${_co_output}
            COMMAND ${ROCM_CLANG} -target amdgcn-amd-amdhsa
                -mcpu=${_hw_target} -x assembler ${_asm_src} -o ${_co_output}
            DEPENDS ${_asm_src}
            COMMENT "Assembling ${_asm_name}.s -> ${_asm_name}.co"
        )
        list(APPEND _asm_co_outputs ${_co_output})
    endforeach()
    add_custom_target(race-emulator-asm-co ALL
        DEPENDS ${_asm_co_outputs}
        COMMENT "Assembling GPU validation code objects"
    )

    file(GLOB _hw_test_sources ${_hw_test_dir}/*.hip)
    add_executable(race-emulator-hw-tests ${_hw_test_sources})
    add_dependencies(race-emulator-hw-tests race-emulator-asm-co)
    set_source_files_properties(${_hw_test_sources} PROPERTIES LANGUAGE HIP)
    set_target_properties(race-emulator-hw-tests PROPERTIES
        HIP_ARCHITECTURES ${_hw_target})
    target_compile_definitions(race-emulator-hw-tests PRIVATE
        TEST_CO_DIR="${CMAKE_CURRENT_BINARY_DIR}"
        TEST_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/asm"
    )
    target_link_libraries(race-emulator-hw-tests PRIVATE
        race-emulator GTest::gtest_main hip::device)
    target_compile_features(race-emulator-hw-tests PRIVATE cxx_std_20)
    gtest_discover_tests(race-emulator-hw-tests)
endif()
