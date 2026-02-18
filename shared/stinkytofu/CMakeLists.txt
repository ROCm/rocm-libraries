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

configure_file(hardware/hardware.def.in include/Config/Archs.def @ONLY)
configure_file(include/stinkytofu/Config.h.in include/stinkytofu/Config/Config.h)

# Note: Stinkytofu library should be built with:
# (1) No RTTI
# (2) No exceptions
#
# => clang/gcc: -fno-rtti -fno-exceptions
# => MSVC: /GR- /EHs-c-
set(STINKYTOFU_SOURCE_FILES
    core/IRBase.cpp
    core/BasicBlock.cpp
    core/Function.cpp
    core/PassManager.cpp
    bindings/python/LogicalModule.cpp
    bindings/python/Module.cpp
    ir/logical/LogicalToFunctionConverter.cpp
    ir/logical/LogicalOpcode.cpp
    serialization/logical/IRSerializer.cpp
    ir/logical/IntrinsicLibrary.cpp
    ir/logical/IntrinsicRegistry.cpp
    ir/logical/IntrinsicPatternConverter.cpp
    transforms/logical/IntrinsicExpansionPass.cpp
    transforms/logical/ToStinkyAsmPass.cpp
    transforms/logical/CompositeInstructionLoweringPass.cpp
    transforms/logical/LogicalPeepholePass.cpp
    transforms/asm/CFGBuilderPass.cpp
    transforms/asm/PeepholeOptimizationPass.cpp
    transforms/asm/DeadCodeEliminationPass.cpp
    transforms/asm/RedundantMovEliminationPass.cpp
    transforms/asm/DelayAluInsertionPass.cpp
    transforms/asm/LegalizationUtils.cpp
    pipeline/OptimizationPipeline.cpp
    transforms/asm/ScheduleLastLRsPass.cpp
    transforms/asm/StinkyDAGSchedulerPass.cpp
    transforms/asm/StinkyConfigurableWaitCntPass.cpp
    transforms/asm/ScheduleFirstLRsPass.cpp
    ir/asm/StinkyAsmIR.cpp
    serialization/asm/StinkyAsmPrinter.cpp
    serialization/asm/StinkyAsmEmitter.cpp
    ir/asm/StinkySignature.cpp
    serialization/asm/IRLexer.cpp
    serialization/asm/IRParser.cpp
    serialization/asm/IRConverter.cpp
    serialization/asm/PatternParser.cpp
    analysis/asm/AsmVerifierPass.cpp
    analysis/logical/IRVerifierPass.cpp
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
    conversion/rocisa/AllHwMappings.cpp
    conversion/rocisa/ToStinkyTofuUtils.cpp
)
list(TRANSFORM ROCISA_TO_STINKYTOFU_SOURCE_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/")

# Only set PARENT_SCOPE variables if we have a parent scope
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCISA_TO_STINKYTOFU_SOURCES ${ROCISA_TO_STINKYTOFU_SOURCE_FILES} PARENT_SCOPE)
endif()

# Add subdirectories first to define gfxisa, tablegen, and tablegen_generated targets
add_subdirectory(hardware)
add_subdirectory(tools/tablegen)
# gfxisa needs generated instruction .inc files (hardware/generated/*.inc)
target_include_directories(gfxisa PUBLIC ${TABLEGEN_OUTPUT_DIR})
add_dependencies(gfxisa instruction_generated)
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
    "${TABLEGEN_OUTPUT_DIR}/stinkytofu/ir/logical/LogicalOpcode.cpp"
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
)

# Make stinkytofu depend on the generated ISA definition files from tablegen
# This ensures the build order is: gfxisa -> tablegen -> run tablegen -> stinkytofu
add_dependencies(stinkytofu tablegen_generated)

# Backend-related sources are added by the backend subdirectory
add_subdirectory(src/pipeline/backend)

##===----------------------------------------------------------------------===##
# Intrinsic Compiler Tool
##===----------------------------------------------------------------------===##
# Add intrinsic-compiler subdirectory AFTER stinkytofu library is built
# This tool compiles Intrinsics.intrinsic to intrinsics.st.bc
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
    ${PROJECT_BINARY_DIR}/stinkytofu/ir/logical/LogicalInstructions_generated.hpp
    DESTINATION include/stinkytofu/ir/logical
)
