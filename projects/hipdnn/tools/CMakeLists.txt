# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_list_engines ListEngines.cpp)
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Register the uhd_gen converter's Python tests with CTest so CI actually runs them.
# These pin the cross-language features_hash contract against the C++ FeatureExtractor;
# without this registration nothing executes them and the two sides drift silently.
# The suite self-skips (pytest.importorskip) when lightgbm/flatbuffers/numpy are
# absent, so environments without the training dependencies report skips, not failures.
# Registered unconditionally, matching how this project adds its other test
# directories (cmake/Tests.cmake calls enable_testing(); there is no BUILD_TESTING gate).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    # Probe for pytest rather than discovering its absence as a test failure:
    # `python -m pytest` exits 1 when the module is missing, which is
    # indistinguishable from a genuine test failure.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE UHD_GEN_PYTEST_PROBE
        OUTPUT_QUIET ERROR_QUIET)
    if(UHD_GEN_PYTEST_PROBE EQUAL 0)
        add_test(
            NAME uhd_gen_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/uhd_gen/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        # pytest exits 5 when it collects nothing, which is what the suite's
        # importorskip guards produce when lightgbm/flatbuffers/numpy are absent.
        set_tests_properties(uhd_gen_tests PROPERTIES SKIP_RETURN_CODE 5 LABELS "unit")
    else()
        message(WARNING
                "uhd_gen tests NOT registered: pytest is not importable by "
                "${Python3_EXECUTABLE}. These pin the features_hash contract between "
                "tools/uhd_gen and the C++ FeatureExtractor; install pytest (plus "
                "lightgbm, flatbuffers, numpy) so they run.")
    endif()
else()
    message(WARNING "uhd_gen tests NOT registered: no Python3 interpreter found")
endif()
