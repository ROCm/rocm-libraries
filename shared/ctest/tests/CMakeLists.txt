# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

get_filename_component(_rocm_libraries_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_package(Python3 COMPONENTS Interpreter)

add_test(
    NAME shared_ctest_optional_args_scope
    COMMAND
        ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_LIST_DIR}/optional_args_scope" -B
        "${CMAKE_CURRENT_BINARY_DIR}/optional_args_scope"
        -DROCM_LIBRARIES_ROOT=${_rocm_libraries_root}
)

if(Python3_FOUND)
    add_test(
        NAME shared_ctest_parse_test_categories_unit
        COMMAND
            ${Python3_EXECUTABLE} -m unittest discover -v
            -s "${CMAKE_CURRENT_LIST_DIR}"
            -p "test_parse_test_categories.py"
    )

    add_test(
        NAME shared_ctest_parse_ctest_categories_unit
        COMMAND
            ${Python3_EXECUTABLE} -m unittest discover -v
            -s "${CMAKE_CURRENT_LIST_DIR}"
            -p "test_parse_ctest_categories.py"
    )

    add_test(
        NAME shared_ctest_parse_catch2_categories_unit
        COMMAND
            ${Python3_EXECUTABLE} -m unittest discover -v
            -s "${CMAKE_CURRENT_LIST_DIR}"
            -p "test_parse_catch2_categories.py"
    )

    # Configuring a project whose category YAML the parser rejects has to fail, with the
    # parser's message. A warning would register the tests without their tier labels.
    add_test(
        NAME shared_ctest_invalid_config_is_fatal
        COMMAND
            ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_LIST_DIR}/invalid_config_is_fatal" -B
            "${CMAKE_CURRENT_BINARY_DIR}/invalid_config_is_fatal"
            -DROCM_LIBRARIES_ROOT=${_rocm_libraries_root}
    )
    set_tests_properties(
        shared_ctest_invalid_config_is_fatal
        PROPERTIES PASS_REGULAR_EXPRESSION "Invalid test categories YAML"
    )
endif()
