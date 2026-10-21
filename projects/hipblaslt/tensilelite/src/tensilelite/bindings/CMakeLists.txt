# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# nanobind requires Python to be found first.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Reuse the nanobind already in this build (e.g. from rocisa), else the one
# installed for Python, else fetch it. A second copy would clash with the first
# one's targets. NB_SUPPRESS_WARNINGS requires nanobind 2.5.
if(NOT COMMAND nanobind_add_module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --version
        OUTPUT_VARIABLE _nb_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _nb_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # nanobind's version file only accepts its own major version, so the minimum
    # is checked here and the search is limited to the reported directory.
    if(_nb_version VERSION_GREATER_EQUAL 2.5)
        find_package(nanobind CONFIG QUIET PATHS "${_nb_cmake_dir}" NO_DEFAULT_PATH)
    endif()
endif()
if(NOT COMMAND nanobind_add_module)
    # Same declaration and pin as rocisa.
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
    )
    FetchContent_MakeAvailable(nanobind)
endif()
message(STATUS "tensilelite_jit: using nanobind from ${NB_DIR}")

nanobind_add_module(tensilelite_jit NOMINSIZE NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/Bindings.cpp"
)
target_link_libraries(tensilelite_jit PRIVATE roc::tensilelite-jit)
set_target_properties(tensilelite_jit
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        BUILD_RPATH "$<TARGET_FILE_DIR:tensilelite-jit>"
)
