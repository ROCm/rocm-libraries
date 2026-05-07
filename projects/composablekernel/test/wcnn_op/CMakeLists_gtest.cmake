# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Find Python3 (if not already found)
if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# Find GTest include directories (shared across all variants)
set(WCNN_GTEST_INCLUDE_DIRS "")
if(TARGET gtest)
    get_target_property(WCNN_GTEST_INCLUDE_DIRS gtest INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT WCNN_GTEST_INCLUDE_DIRS)
        set(WCNN_GTEST_INCLUDE_DIRS "")
    endif()
    message(STATUS "[WCNN] Found gtest target, include dirs: ${WCNN_GTEST_INCLUDE_DIRS}")
elseif(TARGET GTest::gtest)
    get_target_property(WCNN_GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT WCNN_GTEST_INCLUDE_DIRS)
        set(WCNN_GTEST_INCLUDE_DIRS "")
    endif()
    message(STATUS "[WCNN] Found GTest::gtest target, include dirs: ${WCNN_GTEST_INCLUDE_DIRS}")
elseif(DEFINED GTEST_INCLUDE_DIR)
    set(WCNN_GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
    message(STATUS "[WCNN] Using GTEST_INCLUDE_DIR: ${WCNN_GTEST_INCLUDE_DIRS}")
else()
    message(WARNING "[WCNN] gtest not found. Tests may not compile correctly.")
endif()

# Source directory (shared)
set(WCNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# Function: add_wcnn_splited_target
# Creates a test executable with split test files for separate .s generation
#
# Parameters:
#   TARGET_NAME         - Name of the executable target
#   SOURCE_FILE         - Path to the source .cpp file
#   DISABLE_VOPD        - Whether disable VOPD instructions
#   TEST_FILTER         - Regex pattern to filter tests
# ============================================================================
function(add_wcnn_splited_target TARGET_NAME SOURCE_FILE DISABLE_VOPD)
    # Optional TEST_FILTER parameter (5th argument), defaults to ".*"
    if(ARGC GREATER 3)
        set(TEST_FILTER "${ARGV3}")
    else()
        set(TEST_FILTER ".*")
    endif()
    # Get base name from source file (e.g., "grouped_conv_fwd_wcnn" from "grouped_conv_fwd_wcnn.cpp")
    get_filename_component(BASE_NAME ${SOURCE_FILE} NAME_WE)

    # Build directory for generated files - named after the target
    set(BUILD_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})

    # Create build directory
    file(MAKE_DIRECTORY ${BUILD_GEN_DIR})

    # Generate test files at CMake configure time
    message(STATUS "[WCNN:${TARGET_NAME}] Generating test files in: ${BUILD_GEN_DIR}")
    message(STATUS "[WCNN:${TARGET_NAME}] TEST_FILTER: ${TEST_FILTER}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${WCNN_SOURCE_DIR}/split_wcnn_tests.py ${WCNN_SOURCE_DIR}/${SOURCE_FILE} ${BUILD_GEN_DIR} "${TEST_FILTER}"
        RESULT_VARIABLE GEN_RESULT
        OUTPUT_VARIABLE GEN_OUTPUT
        ERROR_VARIABLE GEN_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "[WCNN:${TARGET_NAME}] Failed to generate test files:\n${GEN_ERROR}")
    endif()

    message(STATUS "[WCNN:${TARGET_NAME}] ${GEN_OUTPUT}")

    # Find all generated test files (exclude main file)
    file(GLOB ALL_GENERATED_FILES "${BUILD_GEN_DIR}/*.cpp")
    set(TEST_SOURCE_FILES "")
    foreach(FILE ${ALL_GENERATED_FILES})
        get_filename_component(FNAME ${FILE} NAME)
        # Exclude main file
        if(NOT FNAME STREQUAL "${BASE_NAME}_main.cpp")
            list(APPEND TEST_SOURCE_FILES ${FILE})
        endif()
    endforeach()
    list(LENGTH TEST_SOURCE_FILES TEST_FILE_COUNT)
    message(STATUS "[WCNN:${TARGET_NAME}] Found ${TEST_FILE_COUNT} test files")

    if(TEST_FILE_COUNT EQUAL 0)
        message(FATAL_ERROR "[WCNN:${TARGET_NAME}] No test files generated!")
    endif()

    # Create object library for each test file
    foreach(TEST_FILE ${TEST_SOURCE_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

        set(OBJ_LIB_NAME "${TEST_NAME}_obj")

        # Create object library
        add_library(${OBJ_LIB_NAME} OBJECT ${TEST_FILE})

        # Include directories
        target_include_directories(${OBJ_LIB_NAME} PRIVATE
            ${WCNN_SOURCE_DIR}
            ${BUILD_GEN_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/library/include
        )

        # GTest include directories
        if(WCNN_GTEST_INCLUDE_DIRS)
            target_include_directories(${OBJ_LIB_NAME} SYSTEM PRIVATE ${WCNN_GTEST_INCLUDE_DIRS})
        endif()

        # Base compile options
        target_compile_options(${OBJ_LIB_NAME} PRIVATE
            -save-temps=obj
            -ffunction-sections
            -fdata-sections
            -Wno-global-constructors
            -Wno-deprecated-declarations
            -Wno-gnu-line-marker
            -mprintf-kind=buffered
        )

        # Extra compile options for this variant
        if(${DISABLE_VOPD})
            target_compile_options(${OBJ_LIB_NAME} PRIVATE -mllvm -amdgpu-enable-vopd=0)
        endif()

        # C++ standard
        target_compile_features(${OBJ_LIB_NAME} PRIVATE cxx_std_17)
    endforeach()

    # Create the executable
    add_executable(${TARGET_NAME}
        ${BUILD_GEN_DIR}/${BASE_NAME}_main.cpp
    )

    # Add all test objects to the executable
    foreach(TEST_FILE ${TEST_SOURCE_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        set(OBJ_LIB_NAME "${TEST_NAME}_obj")
        target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:${OBJ_LIB_NAME}>)
    endforeach()

    # Include directories for main
    target_include_directories(${TARGET_NAME} PRIVATE
        ${WCNN_SOURCE_DIR}
        ${BUILD_GEN_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
    )

    # Compile definitions for main file
    if(COMPILE_DEFINITIONS)
        foreach(DEF ${COMPILE_DEFINITIONS})
            target_compile_definitions(${TARGET_NAME} PRIVATE ${DEF})
        endforeach()
    endif()

    # Extra compile options for main file
    if(COMPILE_OPTIONS)
        foreach(OPT ${COMPILE_OPTIONS})
            target_compile_options(${TARGET_NAME} PRIVATE ${OPT})
        endforeach()
    endif()

    # Link libraries
    target_link_libraries(${TARGET_NAME} PRIVATE utility)

    # Link with gtest
    if(TARGET gtest)
        target_link_libraries(${TARGET_NAME} PRIVATE gtest)
    elseif(TARGET GTest::gtest)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest)
    endif()

    if(TARGET gtest_main)
        target_link_libraries(${TARGET_NAME} PRIVATE gtest_main)
    elseif(TARGET GTest::gtest_main)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    endif()

    # C++ standard
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

    # Add to CTest
    if(BUILD_TESTING)
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    endif()

    message(STATUS "[WCNN:${TARGET_NAME}] Configuration complete (${TEST_FILE_COUNT} test files)")
endfunction()

# ============================================================================
# Function: add_wcnn_monolithic_target
# Creates an executable from the original monolithic source file
# Useful for generating a single .s file with all tests AND an executable
#
# Parameters:
#   TARGET_NAME         - Name of the executable target
#   SOURCE_FILE         - Path to the source .cpp file
# ============================================================================
function(add_wcnn_monolithic_target TARGET_NAME SOURCE_FILE)
    # Create executable directly from the monolithic file
    add_executable(${TARGET_NAME}
        ${WCNN_SOURCE_DIR}/${SOURCE_FILE}
    )

    # Include directories
    target_include_directories(${TARGET_NAME} PRIVATE
        ${WCNN_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/library/include
    )

    # GTest include directories
    if(WCNN_GTEST_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${WCNN_GTEST_INCLUDE_DIRS})
    endif()

    # Compile options
    target_compile_options(${TARGET_NAME} PRIVATE
        -save-temps=obj
        -ffunction-sections
        -fdata-sections
        -Wno-global-constructors
        -Wno-deprecated-declarations
        -Wno-gnu-line-marker
    )

    # C++ standard
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

    # Link libraries
    target_link_libraries(${TARGET_NAME} PRIVATE utility)

    # Link with gtest
    if(TARGET gtest)
        target_link_libraries(${TARGET_NAME} PRIVATE gtest)
    elseif(TARGET GTest::gtest)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest)
    endif()

    if(TARGET gtest_main)
        target_link_libraries(${TARGET_NAME} PRIVATE gtest_main)
    elseif(TARGET GTest::gtest_main)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    endif()
endfunction()

# ============================================================================
# Create splited targets for generating seperate .s files for each testcase
# ============================================================================
add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn    # TARGET_NAME
    "grouped_conv_fwd_wcnn.cpp"   # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_wg    # TARGET_NAME
    "grouped_conv_fwd_wcnn.cpp"      # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_wg__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_wg_sc   # TARGET_NAME
    "grouped_conv_fwd_wcnn.cpp"        # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_wg_sc__.+" # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_wg_sc_mc   # TARGET_NAME
    "grouped_conv_fwd_wcnn.cpp"           # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_wg_sc_mc__.+" # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_fma       # TARGET_NAME
    "grouped_conv_fwd_wcnn_fma_cvt.cpp"  # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_fma__.+"     # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_fma_cvt    # TARGET_NAME
    "grouped_conv_fwd_wcnn_fma_cvt.cpp"   # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_fma_cvt__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_fma_wg    # TARGET_NAME
    "grouped_conv_fwd_wcnn_fma_cvt.cpp"  # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_fma_wg__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_fma_cvt_wg    # TARGET_NAME
    "grouped_conv_fwd_wcnn_fma_cvt.cpp"      # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_fma_cvt_wg__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_sba       # TARGET_NAME
    "grouped_conv_fwd_wcnn_sba_cvt.cpp"  # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_sba__.+"     # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_sba_cvt    # TARGET_NAME
    "grouped_conv_fwd_wcnn_sba_cvt.cpp"   # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_sba_cvt__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_sba_wg    # TARGET_NAME
    "grouped_conv_fwd_wcnn_sba_cvt.cpp"  # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_sba_wg__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_fwd_wcnn_sba_cvt_wg    # TARGET_NAME
    "grouped_conv_fwd_wcnn_sba_cvt.cpp"      # SOURCE_FILE
    OFF
    "^grouped_conv_fwd_wcnn_sba_cvt_wg__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_bwd_data_wcnn    # TARGET_NAME
    "grouped_conv_bwd_data_wcnn.cpp"   # SOURCE_FILE
    OFF
    "^grouped_conv_bwd_data_wcnn__.+"  # TEST_FILTER
)

add_wcnn_splited_target(
    test_grouped_conv_bwd_data_wcnn_wg    # TARGET_NAME
    "grouped_conv_bwd_data_wcnn.cpp"      # SOURCE_FILE
    ON
    "^grouped_conv_bwd_data_wcnn_wg__.+"  # TEST_FILTER
)

# ============================================================================
# Create monolithic targets for generating single .s files
# ============================================================================
add_wcnn_monolithic_target(
    test_grouped_conv_fwd_wcnn_monolithic  # TARGET_NAME
    "grouped_conv_fwd_wcnn.cpp"            # SOURCE_FILE
)

add_wcnn_monolithic_target(
    test_grouped_conv_fwd_wcnn_fma_cvt_monolithic  # TARGET_NAME
    "grouped_conv_fwd_wcnn_fma_cvt.cpp"            # SOURCE_FILE
)

add_wcnn_monolithic_target(
    test_grouped_conv_fwd_wcnn_sba_cvt_monolithic  # TARGET_NAME
    "grouped_conv_fwd_wcnn_sba_cvt.cpp"            # SOURCE_FILE
)

add_wcnn_monolithic_target(
    test_grouped_conv_bwd_data_wcnn_monolithic  # TARGET_NAME
    "grouped_conv_bwd_data_wcnn.cpp"            # SOURCE_FILE
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "[WCNN] ============================================")
message(STATUS "[WCNN] Configuration Summary:")
message(STATUS "[WCNN]   Splited executables:")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_wg")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_wg_sc")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_wg_sc_mc")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_fma")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_fma_cvt")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_fma_wg")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_fma_cvt_wg")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_sba")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_sba_cvt")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_sba_wg")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_sba_cvt_wg")
message(STATUS "[WCNN]     - test_grouped_conv_bwd_data_wcnn")
message(STATUS "[WCNN]     - test_grouped_conv_bwd_data_wcnn_wg")
message(STATUS "[WCNN]   Monolithic executables:")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_monolithic")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_fma_cvt_monolithic")
message(STATUS "[WCNN]     - test_grouped_conv_fwd_wcnn_sba_cvt_monolithic")
message(STATUS "[WCNN]     - test_grouped_conv_bwd_data_wcnn_monolithic")
message(STATUS "[WCNN] ============================================")
