# This script reads the test categories YAML file and applies labels to CTest

# Find Python3 for running the parser script
find_package(Python3 COMPONENTS Interpreter)

# Function to apply category labels to discovered GTest tests
function(apply_test_category_labels target_name yaml_file working_dir)
    # Execute the Python script to generate CMake code
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_test_categories.py ${yaml_file} ${target_name} ${working_dir}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Include and execute the generated CMake code
    include("${CATEGORY_CMAKE}")
endfunction()


