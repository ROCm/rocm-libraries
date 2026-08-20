# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

# Common test main entry point
set(TEST_MAIN
    StinkyTofuTestMain.cpp
)
list(TRANSFORM TEST_MAIN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Common include directories for all tests
set(TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblaslt/tensilelite/rocisa/rocisa/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src          # For private headers (IRLexer.hpp, etc.)
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/serialization/asm   # For IRLexer.hpp and parser/lexer private headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../hardware/include
    ${CMAKE_CURRENT_SOURCE_DIR}/unit  # For TestHelpers.hpp
    ${TABLEGEN_OUTPUT_DIR}
)

# Common link libraries for all tests.
# unit_tests links the static lib so all symbols (including non-exported ones)
# are accessible for white-box coverage testing. When BUILD_SHARED_LIBS=OFF,
# stinkytofu itself is already static so no separate target is needed.
if(BUILD_SHARED_LIBS)
    set(_stinkytofu_test_lib stinkytofu_static)
else()
    set(_stinkytofu_test_lib stinkytofu)
endif()
set(TEST_LINK_LIBS
    GTest::gtest
    Threads::Threads
    ${_stinkytofu_test_lib}
)

# =============================================================================
# Unit Tests
# =============================================================================

# Assembly IR Tests
set(UNIT_ASM_TEST_FILES
    unit/asm/WaitCntInsertionTest.cpp
    unit/asm/DAGSchedulerPassTest.cpp
    unit/asm/ExecMaskGroupingTest.cpp
    unit/asm/DeadCodeEliminationPassTest.cpp
    unit/asm/DefUseChainTest.cpp
    unit/asm/InsertPhiTest.cpp
    unit/asm/EstimateAsmCyclesTest.cpp
    unit/asm/RedundantMovEliminationPassTest.cpp
    unit/asm/OptimizationPipelineTest.cpp
    unit/asm/ScopeAdaptorTest.cpp
    unit/asm/ModuleAdaptorsTest.cpp
    unit/asm/PeepholeOptimizationPassTest.cpp
    unit/asm/AccumulateInstructionSizePassTest.cpp
    unit/asm/SwInstructionPrefetchRelStaticPassTest.cpp
    unit/asm/SwInstructionPrefetchRelDynamicPassTest.cpp
    unit/asm/SwInstructionPrefetchAbsStaticPassTest.cpp
    unit/asm/SwInstructionPrefetchAbsDynamicPassTest.cpp
    unit/asm/StinkyAsmEmitterTest.cpp
    unit/asm/StinkyModifiersTest.cpp
    unit/asm/LegalizationUtilsTest.cpp
    unit/asm/VirtualRegisterTest.cpp
    # Parser/Lexer Tests (added for critical testing gaps)
    unit/asm/IRLexerTest.cpp
    unit/asm/IRParserTest.cpp
    unit/asm/PatternParserTest.cpp
    unit/asm/FuzzTest.cpp
    unit/asm/HwInstDescTest.cpp
    unit/asm/ReadWriteOperandTest.cpp
    unit/asm/BarrierTest.cpp
    unit/asm/LoopRegionRemarkPassTest.cpp
    unit/asm/BranchHelpersTest.cpp
    unit/asm/SetpcSwappcCfgTest.cpp
    unit/asm/LongBranchLoweringTest.cpp
    unit/asm/MemTokenConsistencyCheckPassTest.cpp
    unit/asm/InsertClusterBarrierPassTest.cpp
    unit/asm/VerifyInstrumentationTest.cpp
    unit/asm/DebugPrintInstrumentation_test.cpp
    unit/asm/AccumulateInstructionSizePassRunTest.cpp
    unit/asm/StinkyRemoveWaitCntPassTest.cpp
    unit/asm/StinkyWmmaVgprReorderPassTest.cpp
    unit/asm/RemoveInstructionPassTest.cpp
    unit/asm/RemoveDscntPassTest.cpp
    unit/asm/FlattenCalleesPassTest.cpp
    unit/asm/Gfx1250HazardPassTest.cpp
    unit/asm/Gfx1250HazardProfileTest.cpp
    unit/asm/BasicBlockTest.cpp
    unit/asm/DumpStinkyModulePassTest.cpp
    unit/asm/StinkySignatureTest.cpp
    unit/asm/RederiveExpertScopePassTest.cpp
    unit/asm/HWModelTest.cpp
)

# High-Level IR Tests
set(UNIT_LOGICAL_TEST_FILES
    unit/logical/LogicalCommutativeTest.cpp
    unit/logical/LogicalHexLiteralTest.cpp
    unit/logical/LogicalPeepholePassTest.cpp
    unit/logical/LogicalBuilderTest.cpp
    unit/logical/LogicalModuleTest.cpp
    unit/logical/LogicalModuleOwnershipTest.cpp
    unit/logical/LogicalOpcodeTest.cpp
    unit/logical/LogicalToAsmPipelineTest.cpp
    unit/logical/LogicalToAsmMultiArchTest.cpp
    unit/logical/LogicalIntrinsicFlowTest.cpp
    unit/logical/LogicalIRVerifierPassTest.cpp
)

# Core Infrastructure Tests
set(UNIT_CORE_TEST_FILES
    unit/core/AnalysisManagerTest.cpp
    unit/core/BackendRegistryTest.cpp
    unit/core/ComgrProbeTest.cpp
    unit/core/PassBuilderTest.cpp
    unit/core/BackendTest.cpp
    unit/core/VersionTest.cpp
)

# Common/Shared Tests
set(UNIT_COMMON_TEST_FILES
    unit/ErrorHandlingTest.cpp
    unit/FunctionDestructionTest.cpp
    unit/IntrusiveListTest.cpp
    unit/ir/IntrinsicExpansionPassTest.cpp
    unit/ir/RegisterWidthValidationTest.cpp
)

# Combine all unit tests
set(UNIT_TEST_FILES
    ${UNIT_CORE_TEST_FILES}
    ${UNIT_ASM_TEST_FILES}
    ${UNIT_LOGICAL_TEST_FILES}
    ${UNIT_COMMON_TEST_FILES}
)
list(TRANSFORM UNIT_TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(unit_tests ${TEST_MAIN} ${UNIT_TEST_FILES})
target_link_libraries(unit_tests PRIVATE stinkytofu-warnings)
target_link_libraries(unit_tests PUBLIC ${TEST_LINK_LIBS})
target_include_directories(unit_tests PRIVATE ${TEST_INCLUDE_DIRS})
add_dependencies(unit_tests tablegen_generated)

# gtest 1.18 uses std::any, whose MSVC header needs static RTTI; overriding the
# library-wide -fno-rtti just for the gtest-linking test targets on Windows.
if(WIN32)
    target_compile_options(unit_tests PRIVATE -frtti)
endif()

# Example-plugin tests are gated by STINKYTOFU_BUILD_EXAMPLES: the plugin target
# only exists when examples are built, so the integration test compiles only then.
# Future example tests slot in under this same flag.
#
# Static build: plugin is an OBJECT lib linked directly into unit_tests.
# Shared build: plugin is a MODULE loaded at runtime via loadPlugin(). Because
# the plugin links stinkytofu.dll, the test binary must also use the shared lib
# so both share the same registry instance. Plugin tests therefore live in
# api_tests (which links stinkytofu shared), not unit_tests (which links
# stinkytofu_static). Only the STINKYTOFU_BUILD_EXAMPLES compile definition is
# added to unit_tests so non-plugin tests in the same file still compile.
if(STINKYTOFU_BUILD_EXAMPLES)
    if(BUILD_SHARED_LIBS)
        # Shared build: plugin links stinkytofu.dll so the plugin test must run
        # inside api_tests (which also links stinkytofu.dll). unit_tests links
        # stinkytofu_static — two separate registry instances — so the plugin test
        # would not see its registered passes. Leave STINKYTOFU_BUILD_EXAMPLES
        # undefined here; PassBuilderTest.cpp's plugin section is compiled out.
    else()
        target_compile_definitions(unit_tests PRIVATE STINKYTOFU_BUILD_EXAMPLES)
        target_include_directories(unit_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../examples/plugins
        )
        target_link_libraries(unit_tests PRIVATE stinkytofu-plugin-helloworld)
    endif()
endif()

gtest_discover_tests(unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    PROPERTIES LABELS "unit_tests"
)

# Static link-level guard for #8342: assert libstinkytofu has no DT_NEEDED on a
# plugin. This is an ELF property that a runtime gtest cannot observe (a broken
# libstinkytofu would fail to load before any test starts), so it is checked by
# inspecting the built .so directly. Linux/ELF only and only meaningful for a
# shared libstinkytofu.
if(UNIX AND NOT APPLE AND BUILD_SHARED_LIBS AND CMAKE_READELF)
    add_test(
        NAME LibStinkytofuHasNoPluginDependency
        COMMAND ${CMAKE_COMMAND}
            -DLIB=$<TARGET_FILE:stinkytofu>
            -DREADELF=${CMAKE_READELF}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckNoPluginDependency.cmake
    )
    set_tests_properties(LibStinkytofuHasNoPluginDependency
        PROPERTIES LABELS "unit_tests")
endif()

# =============================================================================
# API Export Tests
# Links stinkytofu as a SHARED library (or static when BUILD_SHARED_LIBS=OFF).
# Purpose: verify that every symbol used by real consumers (rocisa,
# stinkytofu-opt) is properly exported. A linker failure here means a symbol
# was accidentally un-exported.
# Only meaningful when BUILD_SHARED_LIBS=ON; skipped otherwise since all
# symbols are always visible in a static build.
# =============================================================================
if(BUILD_SHARED_LIBS)
    add_executable(api_tests
        ${TEST_MAIN}
        ${CMAKE_CURRENT_SOURCE_DIR}/api/ApiExportTest.cpp
        # Address identity of the per-arch HWModel objects. Meaningful only against
        # the shared lib; the unit_tests copy links statically and cannot fail.
        ${CMAKE_CURRENT_SOURCE_DIR}/api/HWModelIdentityTest.cpp
    )
    target_link_libraries(api_tests PRIVATE stinkytofu-warnings)
    target_link_libraries(api_tests PUBLIC
        GTest::gtest
        Threads::Threads
        stinkytofu  # shared lib — only exported symbols visible
    )
    target_include_directories(api_tests PRIVATE ${TEST_INCLUDE_DIRS})
    add_dependencies(api_tests tablegen_generated)

    if(WIN32)
        target_compile_options(api_tests PRIVATE -frtti)
    endif()

    # Plugin integration tests require the shared lib so the plugin and the test
    # binary share the same stinkytofu registry instance at runtime.
    if(STINKYTOFU_BUILD_EXAMPLES)
        target_sources(api_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/core/PassBuilderTest.cpp
        )
        target_compile_definitions(api_tests PRIVATE
            STINKYTOFU_BUILD_EXAMPLES
            STINKYTOFU_PLUGIN_HELLOWORLD_PATH="$<TARGET_FILE:stinkytofu-plugin-helloworld>"
        )
        target_include_directories(api_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../examples/plugins
        )
        add_dependencies(api_tests stinkytofu-plugin-helloworld)

        # Test-only plugin that reports a deliberately wrong stinkytofu version,
        # to assert loadPlugin() rejects a version mismatch before calling
        # registerPlugin(). Headers-only, same DT_NEEDED rationale as the
        # example plugin above (see examples/plugins/CMakeLists.txt).
        add_library(stinkytofu-plugin-bad-version MODULE
            ${CMAKE_CURRENT_SOURCE_DIR}/plugins/BadVersionPlugin.cpp
        )
        if(WIN32)
            target_link_libraries(stinkytofu-plugin-bad-version PRIVATE stinkytofu)
        else()
            target_include_directories(stinkytofu-plugin-bad-version PRIVATE
                $<TARGET_PROPERTY:stinkytofu,INTERFACE_INCLUDE_DIRECTORIES>
            )
        endif()
        set_target_properties(stinkytofu-plugin-bad-version PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
        )
        target_compile_definitions(api_tests PRIVATE
            STINKYTOFU_PLUGIN_BAD_VERSION_PATH="$<TARGET_FILE:stinkytofu-plugin-bad-version>"
        )
        add_dependencies(api_tests stinkytofu-plugin-bad-version)
    endif()

    gtest_discover_tests(api_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TIMEOUT 60
        PROPERTIES LABELS "api_tests"
    )
endif()

# =============================================================================
# All Tests Target (convenience target to build all test executables)
# =============================================================================
add_custom_target(tests COMMENT "Build all test executables")
add_dependencies(tests unit_tests)
if(BUILD_SHARED_LIBS)
    add_dependencies(tests api_tests)
endif()

# Note: Future test executables can be added here, e.g.:
# add_executable(integration_tests ...)
# add_dependencies(tests integration_tests)

# =============================================================================
# Tablegen Instruction Generation Test (Proof-of-Concept)
# =============================================================================
add_executable(test_gen_instructions
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools/tablegen/test_gen_instructions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../tools/tablegen/GenInstructions.cpp
)
target_link_libraries(test_gen_instructions PRIVATE stinkytofu-warnings)

target_link_libraries(test_gen_instructions PRIVATE gfxisa)

target_include_directories(test_gen_instructions PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../hardware/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Pass the configured arch list so the test checks the same set the build declares outputs for,
# instead of a list of its own that would silently stop tracking it.
add_test(NAME GenInstructionsTest
         COMMAND $<TARGET_FILE:test_gen_instructions> ${CMAKE_CURRENT_SOURCE_DIR}/..
                 ${STINKYTOFU_ARCHS_TO_BUILD}
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

set_tests_properties(GenInstructionsTest PROPERTIES
    LABELS "unit_tests"
    TIMEOUT 30
)

add_dependencies(tests test_gen_instructions)

# =============================================================================
# ASan Repro Test (InsertAsanCheckPass end-to-end, requires a real gfx1250 GPU)
# =============================================================================
# Optional: stinkytofu itself has zero HIP/GPU-runtime dependency, so this is
# gated on finding hip and amdclang++ (neither of which the rest of this
# build requires) plus gfx1250 actually being one of the configured archs.
# Skips gracefully (SKIP_RETURN_CODE) at runtime if no gfx1250 GPU is present
# -- see AsanReproTest.cpp -- matching the plain-executable +
# SKIP_RETURN_CODE pattern already used for environment-dependent tests
# elsewhere in this repo (e.g. shared/origami/python/CMakeLists.txt).
if(NOT WIN32 AND "Gfx1250" IN_LIST STINKYTOFU_ARCHS_TO_BUILD)
    find_package(hip QUIET)

    # A pip-wheel ROCm SDK's ROCM_PATH is
    # <venv-root>/lib/pythonX.Y/site-packages/_rocm_sdk_devel: the
    # amdclang++ hardlink living directly under its own bin/ is broken (its
    # amdllvm companion isn't hardlinked alongside it -- see the
    # worktree-venv skill's documented gotcha, ROCm/TheRock#7141); the
    # working console-script trampoline lives at the venv root's bin/
    # instead. Try that first, then fall back to ROCM_PATH/bin for a plain
    # system ROCm install (where ROCM_PATH already is the root).
    if(ROCM_PATH)
        set(_stinkytofu_resolved_rocm_path "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(_stinkytofu_resolved_rocm_path "$ENV{ROCM_PATH}")
    endif()
    if(_stinkytofu_resolved_rocm_path)
        get_filename_component(_stinkytofu_rocm_venv_root
            "${_stinkytofu_resolved_rocm_path}/../../../.." ABSOLUTE)
    endif()
    # Check file existence directly rather than find_program(): find_program
    # kept resolving to the broken hardlink under ROCM_PATH/bin even with
    # HINTS + NO_DEFAULT_PATH restricted to the venv root (root
    # CMakeLists.txt's CMAKE_PREFIX_PATH append appears to take precedence
    # some other way); a direct EXISTS check sidesteps that entirely.
    if(_stinkytofu_rocm_venv_root AND EXISTS "${_stinkytofu_rocm_venv_root}/bin/amdclang++")
        set(STINKYTOFU_AMDCLANGPP_EXECUTABLE "${_stinkytofu_rocm_venv_root}/bin/amdclang++"
            CACHE FILEPATH "amdclang++ for asan-repro kernel assembly" FORCE)
    elseif(_stinkytofu_resolved_rocm_path AND EXISTS "${_stinkytofu_resolved_rocm_path}/bin/amdclang++")
        set(STINKYTOFU_AMDCLANGPP_EXECUTABLE "${_stinkytofu_resolved_rocm_path}/bin/amdclang++"
            CACHE FILEPATH "amdclang++ for asan-repro kernel assembly" FORCE)
    else()
        find_program(STINKYTOFU_AMDCLANGPP_EXECUTABLE NAMES amdclang++)
    endif()

    if(hip_FOUND AND STINKYTOFU_AMDCLANGPP_EXECUTABLE)
        set(ASAN_REPRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../examples/asan-repro")
        set(ASAN_REPRO_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/asan_repro")
        set(ASAN_REPRO_BODY_S "${ASAN_REPRO_DIR}/kernel_body.s")
        set(ASAN_REPRO_DESCRIPTOR_S "${ASAN_REPRO_DIR}/kernel_descriptor.s")
        set(ASAN_REPRO_INSTRUMENTED_S "${ASAN_REPRO_BUILD_DIR}/kernel_instrumented.s")
        set(ASAN_REPRO_CO "${ASAN_REPRO_BUILD_DIR}/kernel.co")

        add_custom_command(
            OUTPUT "${ASAN_REPRO_INSTRUMENTED_S}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ASAN_REPRO_BUILD_DIR}"
            COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/asan_repro_build_kernel.sh"
                    "$<TARGET_FILE:stinkytofu-opt>" "${ASAN_REPRO_BODY_S}"
                    "${ASAN_REPRO_DESCRIPTOR_S}" "${ASAN_REPRO_INSTRUMENTED_S}"
            DEPENDS stinkytofu-opt "${ASAN_REPRO_BODY_S}" "${ASAN_REPRO_DESCRIPTOR_S}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/asan_repro_build_kernel.sh"
            COMMENT "Running InsertAsanCheckPass on the asan-repro kernel"
            VERBATIM
        )

        add_custom_command(
            OUTPUT "${ASAN_REPRO_CO}"
            COMMAND "${STINKYTOFU_AMDCLANGPP_EXECUTABLE}" -x assembler
                    -target amdgcn-amd-amdhsa -mcpu=gfx1250
                    "${ASAN_REPRO_INSTRUMENTED_S}" -o "${ASAN_REPRO_CO}"
            DEPENDS "${ASAN_REPRO_INSTRUMENTED_S}"
            COMMENT "Assembling the asan-repro kernel for gfx1250"
            VERBATIM
        )

        add_custom_target(asan_repro_kernel_co DEPENDS "${ASAN_REPRO_CO}"
            COMMENT "Build the asan-repro gfx1250 kernel code object")

        add_executable(AsanReproTest AsanReproTest.cpp)
        target_compile_options(AsanReproTest PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
        target_link_options(AsanReproTest PRIVATE -fsanitize=address)
        target_link_libraries(AsanReproTest PRIVATE hip::host)
        target_compile_definitions(AsanReproTest PRIVATE
            ASAN_REPRO_KERNEL_CO="${ASAN_REPRO_CO}")
        add_dependencies(AsanReproTest asan_repro_kernel_co)

        add_test(NAME AsanRepro.OOBAccessDetected COMMAND AsanReproTest)
        set_tests_properties(AsanRepro.OOBAccessDetected PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "gpu;asan"
            TIMEOUT 60
        )

        add_dependencies(tests AsanReproTest)
    else()
        message(STATUS "  - Skipping AsanReproTest (hip or amdclang++ not found)")
    endif()
endif()

# =============================================================================
# Python Tests (pytest)
# =============================================================================

# Only register Python tests if the Python module is being built
if(STINKYTOFU_BUILD_PYTHON)
    # Find Python (also done in python_module, but tests are configured first)
    find_package(Python COMPONENTS Interpreter QUIET)

    if(Python_FOUND)
        # Check if pytest is available
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pytest --version
            RESULT_VARIABLE PYTEST_NOT_FOUND
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(NOT PYTEST_NOT_FOUND)
            message(STATUS "pytest found - adding Python tests to CTest")

            # Discover all test files in python_module/tests/
            file(GLOB PYTEST_FILES "${CMAKE_SOURCE_DIR}/python_module/tests/test_*.py")

            # Register each test file as a separate CTest entry
            foreach(PYTEST_FILE ${PYTEST_FILES})
                # Extract test file name without path and extension
                get_filename_component(TEST_NAME ${PYTEST_FILE} NAME_WE)

                # Add test (CTest suppresses output by default, shows on failure)
                add_test(
                    NAME python.${TEST_NAME}
                    COMMAND ${Python_EXECUTABLE} -m pytest
                            ${PYTEST_FILE}
                            -v
                            --tb=short
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                )

                # Set PYTHONPATH so pytest can find the module. The staleness-guard
                # test briefly puts a probe .def in the scanned tree, which would
                # make any concurrent `import stinkytofu` raise, so the Python tests
                # share a lock rather than overlapping.
                set_tests_properties(python.${TEST_NAME} PROPERTIES
                    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib"
                    LABELS "python;unit_tests"
                    RESOURCE_LOCK "stinkytofu_source_tree"
                )
            endforeach()

            list(LENGTH PYTEST_FILES PYTEST_FILE_COUNT)
            message(STATUS "  - Registered ${PYTEST_FILE_COUNT} Python test files")
            message(STATUS "  - Run all: ctest -L python")
            message(STATUS "  - Run one: ctest -R python.test_ir_basic")
            message(STATUS "  - Debug: pytest python_module/tests/test_ir_basic.py::TestClass::test_func -v")
        else()
            message(WARNING "pytest not found - Python tests will not be added to CTest")
            message(STATUS "  - Install pytest with: pip3 install pytest")
        endif()
    else()
        message(WARNING "Python interpreter not found - Python tests will not be added to CTest")
    endif()
else()
    message(STATUS "Python module build disabled (STINKYTOFU_BUILD_PYTHON=OFF) - skipping Python tests")
endif()

##===----------------------------------------------------------------------===##
# FileCheck tests (LLVM-style .stir pattern tests)
##===----------------------------------------------------------------------===##
file(GLOB FILECHECK_TESTS
    "${CMAKE_CURRENT_SOURCE_DIR}/filecheck/*.stir"
    "${CMAKE_CURRENT_SOURCE_DIR}/filecheck/*.s"
)
# v0-specific FileCheck tests invoke `--arch gfx1250v0`, so they are only valid
# when the Gfx1250v0 stepping is built. Register them only then, keeping a
# Gfx1250-only test build green.
if("Gfx1250v0" IN_LIST STINKYTOFU_ARCHS_TO_BUILD)
    file(GLOB FILECHECK_V0_TESTS
        "${CMAKE_CURRENT_SOURCE_DIR}/filecheck/v0/*.stir"
        "${CMAKE_CURRENT_SOURCE_DIR}/filecheck/v0/*.s"
    )
    list(APPEND FILECHECK_TESTS ${FILECHECK_V0_TESTS})
endif()
foreach(test_file ${FILECHECK_TESTS})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(
        NAME FileCheck.${test_name}
        COMMAND $<TARGET_FILE:stinkytofu-check>
                ${test_file}
                --stinkytofu-opt $<TARGET_FILE:stinkytofu-opt>
    )
endforeach()
list(LENGTH FILECHECK_TESTS FILECHECK_COUNT)
message(STATUS "  - Registered ${FILECHECK_COUNT} FileCheck tests")
