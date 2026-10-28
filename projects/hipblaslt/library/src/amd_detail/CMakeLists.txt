# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
)

if(HIPBLASLT_ENABLE_JIT)
    find_package(amd_comgr CONFIG REQUIRED)
    target_link_libraries(hipblaslt PRIVATE amd_comgr)
    target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_JIT)
    target_sources(hipblaslt PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-code-object.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-fs.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-msgpack.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-library.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-builder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-problem-type.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-component.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-replay-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-jit-mode.cpp")
    include("${PROJECT_SOURCE_DIR}/cmake/hipblaslt-jit-hipkittens.cmake")
    # Replay reads this tree. The path is compiled in, and only a test build
    # has that tree. An installed library must not point at the build
    # machine's sources. HipKittens does not replace replay: a build can
    # contain both, either, or neither.
    option(HIPBLASLT_JIT_ENABLE_REPLAY "Build the JIT replay backend." ON)
    if(HIPBLASLT_JIT_ENABLE_REPLAY AND HIPBLASLT_BUILD_TESTING
       AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/clients/tests/jit/data")
        target_compile_definitions(hipblaslt PRIVATE
            "HIPBLASLT_JIT_REPLAY_DATA=\"${PROJECT_SOURCE_DIR}/clients/tests/jit/data\"")
    endif()
endif()

# TheRock configures with testing on and rejects find_package(Catch2): Catch2
# is in the super-project, and hipBLASLt does not declare it. The JIT test
# lanes do not set HIPBLASLT_ENABLE_THEROCK, so they still build these tests.
if(HIPBLASLT_BUILD_TESTING AND NOT HIPBLASLT_ENABLE_THEROCK)
    add_subdirectory("${PROJECT_SOURCE_DIR}/clients/tests/jit"
                     "${PROJECT_BINARY_DIR}/clients/tests/jit")
endif()
