# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-mode.cpp"
)

if(HIPBLASLT_ENABLE_JIT)
    find_package(ZLIB REQUIRED)
    find_package(amd_comgr CONFIG REQUIRED)
    target_link_libraries(hipblaslt PRIVATE ZLIB::ZLIB amd_comgr)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_JIT)
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-code-object.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-fs.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-msgpack.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-library.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-builder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-catalog-knowledge.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-knowledge.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-origami-predictor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-tuning-knowledge.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-problem-type.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-component.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-loader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-process-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-report.cpp")
    # Heuristic queries generate with the one makeDefaultProcessBackend provider linked.
    set(hipblaslt_jit_process_backend hipblaslt-jit-no-backend.cpp)
    if(HIPBLASLT_JIT_TESTING)
        target_sources(hipblaslt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-mock-backend.cpp")
        set(hipblaslt_jit_process_backend hipblaslt-jit-test-backend.cpp)
    endif()
    include("${PROJECT_SOURCE_DIR}/cmake/hipblaslt-jit-hipkittens.cmake")
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/${hipblaslt_jit_process_backend}")
endif()

if(HIPBLASLT_BUILD_TESTING)
    add_subdirectory("${PROJECT_SOURCE_DIR}/clients/tests/jit"
                     "${PROJECT_BINARY_DIR}/clients/tests/jit")
endif()
