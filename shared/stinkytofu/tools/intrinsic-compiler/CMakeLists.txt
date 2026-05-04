# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Intrinsic Compiler Tool
#
# This tool compiles intrinsic definitions (.intrinsic) to optimized bitcode (.st.bc)
# It runs AFTER the main library is built, so it can link against libstinkytofu.

set(SOURCES
    intrinsic-compiler.cpp
)

add_executable(intrinsic-compiler "${SOURCES}")

# Link against the main stinkytofu library (for CodeGenerator, passes, etc.)
target_link_libraries(intrinsic-compiler PRIVATE stinkytofu)

target_include_directories(intrinsic-compiler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/serialization/asm"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/include"
    "${PROJECT_BINARY_DIR}"          # For generated files like hardware/gfxIsa.inc
    "${PROJECT_BINARY_DIR}/include"  # For other generated files
)

# Custom command to compile intrinsics to .st.bc
# This runs AFTER the main library is built
set(INTRINSIC_DEF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/ir/logical/Intrinsics.intrinsic")
set(INTRINSIC_BC_FILE "${CMAKE_BINARY_DIR}/intrinsics.st.bc")

add_custom_command(
    OUTPUT "${INTRINSIC_BC_FILE}"
    COMMAND $<TARGET_FILE:intrinsic-compiler> "${INTRINSIC_DEF_FILE}" "${INTRINSIC_BC_FILE}"
    DEPENDS intrinsic-compiler "${INTRINSIC_DEF_FILE}"
    COMMENT "Compiling intrinsics to ${INTRINSIC_BC_FILE}..."
    VERBATIM
)

# Custom target for intrinsic compilation
add_custom_target(intrinsics_compiled
    DEPENDS "${INTRINSIC_BC_FILE}"
    COMMENT "Compile intrinsic patterns"
)

# Install the intrinsic-compiler tool
install(TARGETS intrinsic-compiler
        RUNTIME DESTINATION bin
        COMPONENT tools)

# Install the .st.bc file
install(FILES "${INTRINSIC_BC_FILE}"
        DESTINATION lib
        COMPONENT runtime)

##===----------------------------------------------------------------------===##
# Test Loader Program
##===----------------------------------------------------------------------===##

set(TEST_LOADER_SOURCES
    test_loader.cpp
)

add_executable(test_loader "${TEST_LOADER_SOURCES}")

# Link against the main stinkytofu library
target_link_libraries(test_loader PRIVATE stinkytofu)

target_include_directories(test_loader PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

# Make test_loader depend on intrinsics_compiled
add_dependencies(test_loader intrinsics_compiled)

##===----------------------------------------------------------------------===##
# Auto-Load Demo
##===----------------------------------------------------------------------===##

set(DEMO_AUTO_LOAD_SOURCES
    demo_auto_load.cpp
)

add_executable(demo_auto_load "${DEMO_AUTO_LOAD_SOURCES}")

# Link against the main stinkytofu library
target_link_libraries(demo_auto_load PRIVATE stinkytofu)

target_include_directories(demo_auto_load PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

# Make demo depend on intrinsics_compiled
add_dependencies(demo_auto_load intrinsics_compiled)
