# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.16)

project(stinkytofu)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable assertions even in Release builds
# Remove NDEBUG from Release build types
foreach(flag_var
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "(^| )-DNDEBUG($| )" " " ${flag_var} "${${flag_var}}")
    string(STRIP "${${flag_var}}" ${flag_var})
endforeach()

add_compile_options(-fno-rtti -fno-exceptions)

# Only set PARENT_SCOPE variables if we have a parent scope
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STINKYTOFU_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
endif()

##===----------------------------------------------------------------------===##
# Architectures
##===----------------------------------------------------------------------===##
include(cmake/StinkytofuArchList.cmake)

set(STINKYTOFU_ARCHS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of backends to build, or \"all\".")

if (STINKYTOFU_ARCHS_TO_BUILD STREQUAL "all" )
    set(STINKYTOFU_ARCHS_TO_BUILD ${STINKYTOFU_ALL_ARCHS})
endif()

foreach(arch ${STINKYTOFU_ARCHS_TO_BUILD})
    list(APPEND STINKYTOFU_ARCHS "${arch}")
    set(STINKYTOFU_ENUM_ARCHS "${STINKYTOFU_ENUM_ARCHS}STINKYTOFU_ARCH(${arch})\n")

    # Set architecture variable for #cmakedefine
    string(TOUPPER "${arch}" ARCH_UPPER)
    set(STINKYTOFU_ARCH_${ARCH_UPPER} 1)
endforeach(arch)

message(STATUS "Using archs: ${STINKYTOFU_ARCHS}")

configure_file(hardware/include/Config/Archs.def.in include/Config/Archs.def @ONLY)
configure_file(include/Config.h.in include/Config/Config.h)

# Note: Stinkytofu library should be built with:
# (1) No RTTI
# (2) No exceptions
#
# => clang/gcc: -fno-rtti -fno-exceptions
# => MSVC: /GR- /EHs-c-
set(STINKYTOFU_SOURCE_FILES
    stinkytofu.cpp
    ir/python/PyLogicalModule.cpp
    ir/logical/LogicalToFunctionConverter.cpp
    ir/IRSerializer.cpp
    ir/IntrinsicLibrary.cpp
    ir/IntrinsicRegistry.cpp
    ir/IntrinsicPatternConverter.cpp
    # ir/passes/IntrinsicExpansionPass.cpp  # TODO: Fix API compatibility
    ir/logical/passes/ToStinkyAsmPass.cpp
    ir/logical/passes/CompositeInstructionLoweringPass.cpp
    ir/logical/passes/LogicalPeepholePass.cpp
    ir/asm/CFGBuilderPass.cpp
    ir/asm/PeepholeOptimizationPass.cpp
    ir/asm/DeadCodeEliminationPass.cpp
    ir/asm/RedundantMovEliminationPass.cpp
    ir/asm/DelayAluInsertionPass.cpp
    ir/asm/WaitCntLegalizationPass.cpp
    ir/asm/OptimizationPipeline.cpp
    ir/asm/ScheduleLastLRsPass.cpp
    ir/asm/StinkyDAGSchedulerPass.cpp
    ir/asm/StinkyConfigurableWaitCntPass.cpp
    ir/asm/ScheduleFirstLRsPass.cpp
    ir/asm/StinkyAsmIR.cpp
    ir/asm/StinkyAsmPrinter.cpp
    ir/asm/StinkyAsmEmitter.cpp
    ir/asm/StinkySignature.cpp
    ir/asm/StinkyAsmModule.cpp
    ir/asm/IRLexer.cpp
    ir/asm/IRParser.cpp
    ir/asm/PatternParser.cpp
    hardware/ArchHelper.cpp
)

list(TRANSFORM STINKYTOFU_SOURCE_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/")

# Note: Do not add rocisa sources to the stinkytofu library target.
# They are built as a separate rocisa target defined in rocisa/CMakeLists.txt.
#
# This is a temporary solution until:
# (1) stinkytofu designs a new rocisa IR to replace the original ones, or
# (2) stinkytofu completely replaces rocisa IR with stinkyAsm IR.
set(ROCISA_TO_STINKYTOFU_SOURCE_FILES
    ir/rocisa/ToStinkyAsmPass.cpp
    ir/rocisa/FromStinkyAsmPass.cpp
    ir/rocisa/AllHwMappings.cpp
    ir/rocisa/ToStinkyTofuUtils.cpp
)
list(TRANSFORM ROCISA_TO_STINKYTOFU_SOURCE_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/")

# Only set PARENT_SCOPE variables if we have a parent scope
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCISA_TO_STINKYTOFU_SOURCES ${ROCISA_TO_STINKYTOFU_SOURCE_FILES} PARENT_SCOPE)
endif()

# Add subdirectories first to define gfxisa, tablegen, and tablegen_generated targets
add_subdirectory(hardware)
add_subdirectory(tools/tablegen)
add_subdirectory(tools/stinkytofu-opt)
# Note: intrinsic-compiler is added later after libstinkytofu is built

# Option to build unit tests
option(STINKYTOFU_BUILD_TESTS "Build StinkyTofu unit tests" ON)
if(STINKYTOFU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Option to build examples
# option(STINKYTOFU_BUILD_EXAMPLES "Build StinkyTofu examples" ON)
# if(STINKYTOFU_BUILD_EXAMPLES)
#     add_subdirectory(examples)
# endif()

# Option to build Python bindings
option(STINKYTOFU_BUILD_PYTHON "Build StinkyTofu Python bindings" ON)
if(STINKYTOFU_BUILD_PYTHON)
    add_subdirectory(python_module)
endif()

# Now create stinkytofu library which depends on tablegen_generated
# Add generated source files
set(STINKYTOFU_GENERATED_SOURCES
    "${TABLEGEN_OUTPUT_DIR}/ir/logical/LogicalOpcode.cpp"
)

# Mark generated files so CMake knows they will be created during build
set_source_files_properties(${STINKYTOFU_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

add_library(stinkytofu SHARED "${STINKYTOFU_SOURCE_FILES}" "${STINKYTOFU_GENERATED_SOURCES}")
target_include_directories(stinkytofu
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> # For generated files
    PRIVATE
        # Make the generated files' directories available to the library
        # This allows source files to #include the generated .inc files
        "${TABLEGEN_OUTPUT_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/hardware/include"
)

# Make stinkytofu depend on the generated ISA definition files from tablegen
# This ensures the build order is: gfxisa -> tablegen -> run tablegen -> stinkytofu
add_dependencies(stinkytofu tablegen_generated)

##===----------------------------------------------------------------------===##
# Intrinsic Compiler Tool
##===----------------------------------------------------------------------===##
# Add intrinsic-compiler subdirectory AFTER stinkytofu library is built
# This tool compiles Intrinsics.def to intrinsics.st.bc
add_subdirectory(tools/intrinsic-compiler)

##===----------------------------------------------------------------------===##
# Install targets
##===----------------------------------------------------------------------===##

# Install the stinkytofu library
install(TARGETS stinkytofu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install public header files with stinkytofu/ namespace
# Exclude ir/python/ (Python-specific, internal headers)
install(DIRECTORY include/
    DESTINATION include/stinkytofu
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "ir/python" EXCLUDE
)

# Install generated header files
install(FILES
    ${PROJECT_BINARY_DIR}/ir/logical/LogicalInstructions_generated.hpp
    DESTINATION include/stinkytofu/ir/logical
)
