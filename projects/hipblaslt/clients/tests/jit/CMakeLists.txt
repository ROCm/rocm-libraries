# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu.
enable_testing()
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
# Committed bundles; data/README.md lists them and how to regenerate them.
set(hipblaslt_jit_data "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(hipblaslt_jit_splitk "${hipblaslt_jit_data}/gfx950/splitk")
string(REGEX MATCH "(^|;)gfx950(:[^;]*)?(;|$)" hipblaslt_jit_gfx950 "${GPU_TARGETS}")

# Each test writes to its own directory in an emptied scratch directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reset-scratch.cmake"
     "file(REMOVE_RECURSE \"${hipblaslt_jit_scratch}\")\n"
     "file(MAKE_DIRECTORY \"${hipblaslt_jit_scratch}\")\n")
add_test(NAME jit-scratch
         COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/reset-scratch.cmake")
set_tests_properties(jit-scratch PROPERTIES FIXTURES_SETUP hipblaslt-jit-scratch)

function(hipblaslt_jit_test name label)
    add_test(NAME ${name} COMMAND ${ARGN})
    set_tests_properties(${name} PROPERTIES
        LABELS ${label} FIXTURES_REQUIRED hipblaslt-jit-scratch)
endfunction()

# hipblaslt_jit_python_test(name label script args...) runs a script here.
function(hipblaslt_jit_python_test name label script)
    hipblaslt_jit_test(${name} ${label} "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${ARGN})
    set_tests_properties(${name} PROPERTIES ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
endfunction()

function(hipblaslt_jit_stage)
    foreach(test_target ${ARGN})
        target_compile_features(${test_target} PRIVATE cxx_std_17)
        set_target_properties(${test_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    endforeach()
endfunction()

# Public API only, so it also builds without JIT.
add_executable(hipblaslt-jit-heuristic-test heuristic_test.cpp)
target_link_libraries(hipblaslt-jit-heuristic-test
    PRIVATE roc::hipblaslt hip::device Threads::Threads)

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_include_directories(hipblaslt-jit-source-bundle-test PRIVATE "${hipblaslt_jit_source}")
target_link_libraries(hipblaslt-jit-source-bundle-test PRIVATE ZLIB::ZLIB)
hipblaslt_jit_test(jit-source-bundle jit-cpu
    $<TARGET_FILE:hipblaslt-jit-source-bundle-test> "${hipblaslt_jit_scratch}/source-bundle")
hipblaslt_jit_stage(hipblaslt-jit-heuristic-test hipblaslt-jit-source-bundle-test)

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_stage(hipblaslt-jit-disabled-test)
    hipblaslt_jit_test(jit-disabled jit-cpu $<TARGET_FILE:hipblaslt-jit-disabled-test>)
    return()
endif()

# Compiles the Jit stages directly; runs without a GPU.
add_executable(hipblaslt-jit-component-test
    jit_component_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-component.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-catalog-knowledge.cpp")
target_include_directories(hipblaslt-jit-component-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-component-test PRIVATE HIPBLASLT_ENABLE_JIT)
target_link_libraries(hipblaslt-jit-component-test PRIVATE
    roc::hipblaslt hip::host Threads::Threads)
hipblaslt_jit_test(jit-component jit-cpu
    $<TARGET_FILE:hipblaslt-jit-component-test> "${hipblaslt_jit_scratch}/component")

# Compiles HIPBLASLT_JIT_DEBUG parsing and the line sink directly.
add_executable(hipblaslt-jit-debug-test
    jit_debug_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp")
target_include_directories(hipblaslt-jit-debug-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_compile_definitions(hipblaslt-jit-debug-test PRIVATE HIPBLASLT_ENABLE_JIT)
target_link_libraries(hipblaslt-jit-debug-test PRIVATE Threads::Threads)
hipblaslt_jit_test(jit-debug jit-cpu
    $<TARGET_FILE:hipblaslt-jit-debug-test> "${hipblaslt_jit_scratch}/debug")

# Compiles the comgr builder directly; runs without a GPU unless --gpu is given.
add_executable(hipblaslt-jit-code-object-test
    code_object_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-code-object-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-code-object-test PRIVATE amd_comgr hip::host Threads::Threads)
hipblaslt_jit_test(jit-code-object jit-cpu
    $<TARGET_FILE:hipblaslt-jit-code-object-test>
    --target gfx950 --out "${hipblaslt_jit_scratch}/code-object" --bundle "${hipblaslt_jit_splitk}")
hipblaslt_jit_test(jit-code-object-gpu jit-gpu
    $<TARGET_FILE:hipblaslt-jit-code-object-test>
    --gpu --out "${hipblaslt_jit_scratch}/code-object-gpu" --bundle "${hipblaslt_jit_splitk}")
hipblaslt_jit_stage(hipblaslt-jit-component-test hipblaslt-jit-debug-test
                    hipblaslt-jit-code-object-test)

if(HIPBLASLT_JIT_TESTING)
    add_executable(hipblaslt-jit-mock-backend-test mock_backend_test.cpp)
    target_include_directories(hipblaslt-jit-mock-backend-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/include")
    target_link_libraries(hipblaslt-jit-mock-backend-test PRIVATE roc::hipblaslt hip::device)

    # The public C and C++ GEMM APIs on solutions replayed through the mock backend.
    add_executable(hipblaslt-jit-api-test jit_api_test.cpp)
    target_include_directories(hipblaslt-jit-api-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-api-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_stage(hipblaslt-jit-mock-backend-test hipblaslt-jit-api-test)

    # The committed bundles hold gfx950 code.
    if(hipblaslt_jit_gfx950)
        set(mock $<TARGET_FILE:hipblaslt-jit-mock-backend-test>)
        set(api $<TARGET_FILE:hipblaslt-jit-api-test>)
        hipblaslt_jit_test(jit-mock-backend jit-gpu ${mock} "${hipblaslt_jit_splitk}")
        hipblaslt_jit_test(jit-mock-backend-library jit-gpu
            ${mock} "${hipblaslt_jit_splitk}" --library)
        set_tests_properties(jit-mock-backend-library PROPERTIES ENVIRONMENT
            HIPBLASLT_JIT_LIBRARY_PATH=${hipblaslt_jit_scratch}/mock-backend-library)
        foreach(case "splitk;splitk;--k;512"
                     "streamk;streamk;--k;4096;--workspace-fallback;1"
                     "amax;amax;--amax;1"
                     "alpha-zero;amax;--amax;1;--alpha-zero;1")
            list(POP_FRONT case name bundle)
            hipblaslt_jit_test(jit-api-${name} jit-gpu
                ${api} --replay "${hipblaslt_jit_data}/gfx950/${bundle}" ${case})
        endforeach()
        set_tests_properties(jit-api-streamk PROPERTIES ENVIRONMENT
            "TENSILE_PERSISTENT_FIXED_GRID=16;TENSILE_PERSISTENT_DYNAMIC_GRID=0;TENSILE_DB=64")
        hipblaslt_jit_python_test(jit-bundle-failures jit-gpu test_bundle_failures.py
            ${api} "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/bundle-failures" --replay)
        hipblaslt_jit_python_test(jit-helper-failures jit-gpu test_helper_failures.py
            ${api} "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/helper-failures" --replay)

        # The heuristic routes, on what the test backend replays.
        if(hipblaslt_jit_process_backend STREQUAL "hipblaslt-jit-test-backend.cpp")
            foreach(route fallback-c fallback-cpp forced cache-hit distinct unsupported
                          concurrent null-algo capture capture-query report partial-fill override
                          provider-order debug-timing debug-progress debug-off debug-file)
                hipblaslt_jit_python_test(jit-heuristic-${route} jit-gpu test_heuristic.py
                    $<TARGET_FILE:hipblaslt-jit-heuristic-test> ${route}
                    "${hipblaslt_jit_scratch}/heuristic-${route}" --backend test
                    --replay "${hipblaslt_jit_data}/gfx950/rank-1"
                    --replay "${hipblaslt_jit_data}/gfx950/rank-2"
                    --replay "${hipblaslt_jit_splitk}")
            endforeach()
        endif()
        # Several backends: mock backends that replace the build's.
        foreach(route both order optin unavailable count domain exclude fellshort capture)
            hipblaslt_jit_python_test(jit-heuristic-multi-${route} jit-gpu test_heuristic.py
                $<TARGET_FILE:hipblaslt-jit-heuristic-test> multi-${route}
                "${hipblaslt_jit_scratch}/heuristic-multi-${route}")
        endforeach()
    endif()
endif()

# Compiles the JIT solution library directly; runs without a GPU.
if(NOT HIPBLASLT_ENABLE_YAML)
    # The tuning knowledge reader, matcher and library lookup, on fixture databases it writes.
    add_executable(hipblaslt-jit-knowledge-test
        jit_knowledge_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-catalog-knowledge.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-knowledge.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-tuning-knowledge.cpp")
    target_include_directories(hipblaslt-jit-knowledge-test PRIVATE "${hipblaslt_jit_source}"
        "${hipblaslt_jit_source}/rocblaslt/include" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-knowledge-test PRIVATE
        HIPBLASLT_ENABLE_JIT TENSILE_MSGPACK)
    target_link_libraries(hipblaslt-jit-knowledge-test PRIVATE
        roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} ZLIB::ZLIB hip::device
        Threads::Threads)
    hipblaslt_jit_test(jit-knowledge jit-cpu $<TARGET_FILE:hipblaslt-jit-knowledge-test>
        "${hipblaslt_jit_scratch}/knowledge")
    hipblaslt_jit_stage(hipblaslt-jit-knowledge-test)

    add_executable(hipblaslt-jit-library-test
        jit_library_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")
    target_include_directories(hipblaslt-jit-library-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-library-test PRIVATE
        HIPBLASLT_ENABLE_JIT HIPBLASLT_JIT_LIBRARY_TESTING)
    target_link_libraries(hipblaslt-jit-library-test PRIVATE
        roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} ZLIB::ZLIB amd_comgr
        hip::device Threads::Threads)
    hipblaslt_jit_test(jit-library jit-cpu $<TARGET_FILE:hipblaslt-jit-library-test>
        "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/library")
    hipblaslt_jit_test(jit-library-concurrency jit-cpu $<TARGET_FILE:hipblaslt-jit-library-test>
        "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/library-concurrency"
        --writers 8 --per-writer 4)

    # Fails when the committed bundles no longer match the layout versions in
    # GlobalParameters.py, the builder's code-object version or the host library.
    set(parameters "${PROJECT_SOURCE_DIR}/tensilelite/Tensile/Common/GlobalParameters.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${parameters}")
    file(READ "${parameters}" parameters)
    foreach(key KernArgsVersion PersistentLoopArgsVersion)
        if(NOT parameters MATCHES "\"${key}\": *([0-9]+)")
            message(FATAL_ERROR "No ${key} in GlobalParameters.py")
        endif()
        set(${key} ${CMAKE_MATCH_1})
    endforeach()
    add_executable(hipblaslt-jit-bundle-freshness-test
        jit_bundle_freshness_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp")
    target_include_directories(hipblaslt-jit-bundle-freshness-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-bundle-freshness-test PRIVATE
        HIPBLASLT_ENABLE_JIT
        HIPBLASLT_JIT_KERNARGS_VERSION=${KernArgsVersion}
        HIPBLASLT_JIT_PERSISTENT_LOOP_ARGS_VERSION=${PersistentLoopArgsVersion})
    target_link_libraries(hipblaslt-jit-bundle-freshness-test PRIVATE
        roc::hipblaslt roc::tensilelite-host ZLIB::ZLIB amd_comgr hip::device Threads::Threads)
    hipblaslt_jit_test(jit-bundle-freshness jit-cpu
        $<TARGET_FILE:hipblaslt-jit-bundle-freshness-test>
        "${hipblaslt_jit_data}" "${hipblaslt_jit_scratch}/bundle-freshness")
    hipblaslt_jit_stage(hipblaslt-jit-library-test hipblaslt-jit-bundle-freshness-test)
endif()
