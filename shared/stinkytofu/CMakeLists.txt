# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

project(stinkytofu)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-fno-rtti -fno-exceptions)

set(STINKYTOFU_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# Note: Stinkytofu library should be built with:
# (1) No RTTI
# (2) No exceptions
#
# => clang/gcc: -fno-rtti -fno-exceptions
# => MSVC: /GR- /EHs-c-
set(STINKYTOFU_SOURCE_FILES
    stinkytofu.cpp
    ir/asm/StinkyClusterDSReadPass.cpp
    ir/asm/ScheduleLastLRsPass.cpp
    ir/asm/StinkyDAGSchedulerPass.cpp
    ir/asm/StinkyUnrollInsertWaitCntPass.cpp
    ir/asm/ScheduleFirstLRsPass.cpp
    ir/asm/StinkyAsmIR.cpp
    ir/asm/StinkyAsmPrinter.cpp
    ir/asm/IRLexer.cpp
    ir/asm/IRParser.cpp
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
)
list(TRANSFORM ROCISA_TO_STINKYTOFU_SOURCE_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/")
set(ROCISA_TO_STINKYTOFU_SOURCES ${ROCISA_TO_STINKYTOFU_SOURCE_FILES} PARENT_SCOPE)

# Add subdirectories first to define gfxisa, tablegen, and tablegen_generated targets
add_subdirectory(hardware)
add_subdirectory(tools/tablegen)
add_subdirectory(tools/stinkytofu-opt)
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(unittests)
endif()

# Now create stinkytofu library which depends on tablegen_generated
add_library(stinkytofu SHARED "${STINKYTOFU_SOURCE_FILES}")
target_include_directories(stinkytofu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"

    # Make the generated files' directories available to the library
    # This allows source files to #include the generated .inc files
    "${TABLEGEN_OUTPUT_DIR}"

    "${CMAKE_CURRENT_SOURCE_DIR}/hardware/include"
)

# Make stinkytofu depend on the generated ISA definition files from tablegen
# This ensures the build order is: gfxisa -> tablegen -> run tablegen -> stinkytofu
add_dependencies(stinkytofu tablegen_generated)
