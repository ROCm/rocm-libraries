# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

get_filename_component(_rocm_libraries_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

find_package(Python3 COMPONENTS Interpreter)

# The nested configure runs its own find_package(Python3), so without being told
# otherwise it can select a different interpreter than this build was given -- one
# where the parser's yaml import fails, which surfaces as missing category labels
# rather than as a missing dependency. Forward the interpreter that was resolved here.
add_test(
    NAME shared_ctest_optional_args_scope
    COMMAND
        ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_LIST_DIR}/optional_args_scope" -B
        "${CMAKE_CURRENT_BINARY_DIR}/optional_args_scope"
        -DROCM_LIBRARIES_ROOT=${_rocm_libraries_root}
        -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
)

if(Python3_FOUND)
    add_test(
        NAME shared_ctest_parse_test_categories_unit
        COMMAND
            ${Python3_EXECUTABLE} -m unittest discover -v
            -s "${CMAKE_CURRENT_LIST_DIR}"
            -p "test_parse_test_categories.py"
    )
endif()
