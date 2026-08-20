# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

foreach(required_variable
        COMPONENT_SOURCE_DIR
        COMPONENT_BINARY_DIR
        PACKAGE_SOURCE_DIR
        PACKAGE_WORK_DIR
        TEST_GENERATOR
        CTEST_COMMAND)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

set(_generator_arguments -G "${TEST_GENERATOR}")
if(TEST_GENERATOR_PLATFORM)
    list(APPEND _generator_arguments -A "${TEST_GENERATOR_PLATFORM}")
endif()
if(TEST_GENERATOR_TOOLSET)
    list(APPEND _generator_arguments -T "${TEST_GENERATOR_TOOLSET}")
endif()

set(_configure_cache_arguments)
if(TEST_CXX_COMPILER)
    list(
        APPEND
        _configure_cache_arguments
        "-DCMAKE_CXX_COMPILER:FILEPATH=${TEST_CXX_COMPILER}"
    )
endif()
if(TEST_MAKE_PROGRAM)
    list(
        APPEND
        _configure_cache_arguments
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=${TEST_MAKE_PROGRAM}"
    )
endif()
if(TEST_CONFIG)
    list(
        APPEND
        _configure_cache_arguments
        "-DCMAKE_BUILD_TYPE=${TEST_CONFIG}"
    )
endif()
if(DEFINED TEST_CXX_FLAGS)
    list(
        APPEND
        _configure_cache_arguments
        "-DCMAKE_CXX_FLAGS=${TEST_CXX_FLAGS}"
    )
endif()
if(DEFINED TEST_EXE_LINKER_FLAGS)
    list(
        APPEND
        _configure_cache_arguments
        "-DCMAKE_EXE_LINKER_FLAGS=${TEST_EXE_LINKER_FLAGS}"
    )
endif()

# Build one nested producer or consumer with the package-test CPU budget.
function(_build_project build_dir label)
    # These nested consumer builds run inside the standalone CTest job. Match
    # that job's deliberately small CPU budget instead of multiplying it.
    set(_build_command "${CMAKE_COMMAND}" --build "${build_dir}" --parallel 2)
    if(TEST_CONFIG)
        list(APPEND _build_command --config "${TEST_CONFIG}")
    endif()
    execute_process(
        COMMAND ${_build_command}
        RESULT_VARIABLE _build_result
        OUTPUT_VARIABLE _build_output
        ERROR_VARIABLE _build_error
    )
    if(NOT _build_result EQUAL 0)
        message(FATAL_ERROR
            "${label} build failed.\n${_build_output}\n${_build_error}"
        )
    endif()
endfunction()

# Install one nested producer into its task-owned prefix.
function(_install_project build_dir install_dir label)
    set(
        _install_command
        "${CMAKE_COMMAND}" --install "${build_dir}" --prefix "${install_dir}"
    )
    if(TEST_CONFIG)
        list(APPEND _install_command --config "${TEST_CONFIG}")
    endif()
    execute_process(
        COMMAND ${_install_command}
        RESULT_VARIABLE _install_result
        OUTPUT_VARIABLE _install_output
        ERROR_VARIABLE _install_error
    )
    if(NOT _install_result EQUAL 0)
        message(FATAL_ERROR
            "${label} install failed.\n"
            "${_install_output}\n${_install_error}"
        )
    endif()

    set(_private_include_dir
        "${install_dir}/include/roc/host_validation/detail"
    )
    if(EXISTS "${_private_include_dir}" OR IS_SYMLINK "${_private_include_dir}")
        message(FATAL_ERROR
            "${label} installed the private header directory "
            "${_private_include_dir}."
        )
    endif()
endfunction()

# Configure, build, and execute one explicit package-component consumer.
function(
    _run_component
    install_dir
    case_name
    component
    expected_components
)
    set(_build_dir "${PACKAGE_WORK_DIR}/${case_name}")
    file(REMOVE_RECURSE "${_build_dir}")
    set(
        _configure_command
        "${CMAKE_COMMAND}"
        -S "${PACKAGE_SOURCE_DIR}/component"
        -B "${_build_dir}"
        ${_generator_arguments}
        ${_configure_cache_arguments}
        "-DCMAKE_PREFIX_PATH=${install_dir}"
        "-DROCHostValidation_TEST_COMPONENT=${component}"
        "-DROCHostValidation_TEST_EXPECTED_COMPONENTS=${expected_components}"
        ${ARGN}
    )
    execute_process(
        COMMAND ${_configure_command}
        RESULT_VARIABLE _configure_result
        OUTPUT_VARIABLE _configure_output
        ERROR_VARIABLE _configure_error
    )
    if(NOT _configure_result EQUAL 0)
        message(FATAL_ERROR
            "${case_name} configure failed.\n"
            "${_configure_output}\n${_configure_error}"
        )
    endif()

    _build_project("${_build_dir}" "${case_name}")

    set(
        _test_command
        "${CTEST_COMMAND}" --test-dir "${_build_dir}" --output-on-failure
    )
    if(TEST_CONFIG)
        list(APPEND _test_command --build-config "${TEST_CONFIG}")
    endif()
    execute_process(
        COMMAND ${_test_command}
        RESULT_VARIABLE _test_result
        OUTPUT_VARIABLE _test_output
        ERROR_VARIABLE _test_error
    )
    if(NOT _test_result EQUAL 0)
        message(FATAL_ERROR
            "${case_name} test failed.\n${_test_output}\n${_test_error}"
        )
    endif()
endfunction()

# Assert that requesting one package component fails with a stable diagnostic.
function(
    _expect_component_failure
    install_dir
    case_name
    component
    expected_message
)
    set(_build_dir "${PACKAGE_WORK_DIR}/${case_name}")
    file(REMOVE_RECURSE "${_build_dir}")
    set(
        _configure_command
        "${CMAKE_COMMAND}"
        -S "${PACKAGE_SOURCE_DIR}/component"
        -B "${_build_dir}"
        ${_generator_arguments}
        ${_configure_cache_arguments}
        "-DCMAKE_PREFIX_PATH=${install_dir}"
        "-DROCHostValidation_TEST_COMPONENT=${component}"
        "-DROCHostValidation_TEST_EXPECTED_COMPONENTS=${component}"
        ${ARGN}
    )
    execute_process(
        COMMAND ${_configure_command}
        RESULT_VARIABLE _configure_result
        OUTPUT_VARIABLE _configure_output
        ERROR_VARIABLE _configure_error
    )
    if(_configure_result EQUAL 0)
        message(FATAL_ERROR
            "${case_name} unexpectedly configured successfully."
        )
    endif()

    set(_combined_output "${_configure_output}\n${_configure_error}")
    string(FIND "${_combined_output}" "${expected_message}" _message_position)
    if(_message_position EQUAL -1)
        message(FATAL_ERROR
            "${case_name} failed without the expected diagnostic:\n"
            "  ${expected_message}\n"
            "Actual output:\n${_combined_output}"
        )
    endif()
endfunction()

# Assert that a package lookup without explicit components is rejected.
function(_expect_no_components_failure install_dir)
    set(_build_dir "${PACKAGE_WORK_DIR}/no-components")
    file(REMOVE_RECURSE "${_build_dir}")
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}"
            -S "${PACKAGE_SOURCE_DIR}/no_components"
            -B "${_build_dir}"
            ${_generator_arguments}
            ${_configure_cache_arguments}
            "-DCMAKE_PREFIX_PATH=${install_dir}"
        RESULT_VARIABLE _configure_result
        OUTPUT_VARIABLE _configure_output
        ERROR_VARIABLE _configure_error
    )
    if(_configure_result EQUAL 0)
        message(FATAL_ERROR
            "Package lookup without components unexpectedly succeeded."
        )
    endif()
    set(_combined_output "${_configure_output}\n${_configure_error}")
    string(
        FIND
        "${_combined_output}"
        "requires at least one explicit COMPONENTS entry"
        _message_position
    )
    if(_message_position EQUAL -1)
        message(FATAL_ERROR
            "Package lookup without components failed without the expected diagnostic.\n"
            "${_combined_output}"
        )
    endif()
endfunction()

file(REMOVE_RECURSE "${PACKAGE_WORK_DIR}")
file(MAKE_DIRECTORY "${PACKAGE_WORK_DIR}")

set(_full_install_dir "${PACKAGE_WORK_DIR}/full-install")
_build_project("${COMPONENT_BINARY_DIR}" "ROCHostValidation producer")
_install_project(
    "${COMPONENT_BINARY_DIR}"
    "${_full_install_dir}"
    "ROCHostValidation producer"
)

_run_component(
    "${_full_install_dir}"
    full-core
    Core
    Core
    "-DCMAKE_MODULE_PATH=${PACKAGE_SOURCE_DIR}/dependency-search-traps"
)
_run_component(
    "${_full_install_dir}"
    full-operations
    Operations
    "Core,Operations"
)
_run_component(
    "${_full_install_dir}"
    full-tiled
    Blocked
    "Core,Operations,Blocked"
)
_run_component(
    "${_full_install_dir}"
    full-mx
    MX
    "Core,MX"
)
_run_component(
    "${_full_install_dir}"
    full-amd-gpu-layout
    AMDGPULayout
    AMDGPULayout
)

file(
    GLOB_RECURSE
    _blas_target_files
    "${_full_install_dir}/*ROCHostValidationBLASTargets.cmake"
)
if(_blas_target_files)
    list(GET _blas_target_files 0 _blas_target_file)
    file(READ "${_blas_target_file}" _blas_targets)
    if(NOT _blas_targets MATCHES "CBLAS::CBLAS")
        message(FATAL_ERROR
            "The installed BLAS target does not link through CBLAS::CBLAS."
        )
    endif()
    _run_component(
        "${_full_install_dir}"
        full-blas
        BLAS
        "Core,Operations,BLAS"
    )
    _expect_component_failure(
        "${_full_install_dir}"
        missing-cblas-dependency
        BLAS
        "the BLAS component requires CBLAS::CBLAS"
        -DCMAKE_DISABLE_FIND_PACKAGE_CBLAS=TRUE
    )
    _expect_component_failure(
        "${_full_install_dir}"
        incompatible-cblas-integer-abi
        BLAS
        "the BLAS component requires CBLAS::CBLAS"
        "-DCMAKE_MODULE_PATH=${PACKAGE_SOURCE_DIR}/fake-ilp64-cblas"
        "-DCBLAS_INCLUDE_DIR=${PACKAGE_SOURCE_DIR}/fake-ilp64-cblas"
    )
endif()

file(
    GLOB_RECURSE
    _operations_target_files
    "${_full_install_dir}/*ROCHostValidationOperationsTargets.cmake"
)
if(_operations_target_files)
    list(GET _operations_target_files 0 _operations_target_file)
    file(READ "${_operations_target_file}" _operations_targets)
    if(_operations_targets MATCHES "OpenMP::OpenMP_CXX")
        _expect_component_failure(
            "${_full_install_dir}"
            missing-operations-openmp-dependency
            Operations
            "the installed Operations target requires OpenMP::OpenMP_CXX"
            -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE
        )
    endif()
endif()

_expect_component_failure(
    "${_full_install_dir}"
    unknown-component
    NotAComponent
    "does not support component \"NotAComponent\""
)
_expect_no_components_failure("${_full_install_dir}")

# Exercise the supported add_subdirectory build from a parent project without
# depending on any product-specific consumer.
set(_source_build_dir "${PACKAGE_WORK_DIR}/source-producer")
set(_source_install_dir "${PACKAGE_WORK_DIR}/source-install")
set(
    _source_configure_command
    "${CMAKE_COMMAND}"
    -S "${PACKAGE_SOURCE_DIR}/source_build"
    -B "${_source_build_dir}"
    ${_generator_arguments}
    ${_configure_cache_arguments}
    "-DROCHostValidation_SOURCE_DIR=${COMPONENT_SOURCE_DIR}"
    "-DCMAKE_MODULE_PATH=${PACKAGE_SOURCE_DIR}/aliased-blas"
)
execute_process(
    COMMAND ${_source_configure_command}
    RESULT_VARIABLE _source_configure_result
    OUTPUT_VARIABLE _source_configure_output
    ERROR_VARIABLE _source_configure_error
)
if(NOT _source_configure_result EQUAL 0)
    message(FATAL_ERROR
        "Source build configure failed.\n"
        "${_source_configure_output}\n"
        "${_source_configure_error}"
    )
endif()
_build_project(
    "${_source_build_dir}"
    "Source producer"
)
_install_project(
    "${_source_build_dir}"
    "${_source_install_dir}"
    "Source producer"
)
file(
    GLOB_RECURSE
    _source_blas_target_files
    "${_source_install_dir}/*ROCHostValidationBLASTargets.cmake"
)
if(_source_blas_target_files)
    _run_component(
        "${_source_install_dir}"
        source-blas
        BLAS
        "Core,Operations,BLAS"
    )
endif()
_run_component(
    "${_source_install_dir}"
    source-mx
    MX
    "Core,MX"
)

# Configure a minimal producer. MX remains present because it is a normal
# component invariant; BLAS remains the intentionally optional component.
set(_minimal_build_dir "${PACKAGE_WORK_DIR}/minimal-producer")
set(_minimal_install_dir "${PACKAGE_WORK_DIR}/minimal-install")
set(
    _minimal_configure_command
    "${CMAKE_COMMAND}"
    -S "${COMPONENT_SOURCE_DIR}"
    -B "${_minimal_build_dir}"
    ${_generator_arguments}
    ${_configure_cache_arguments}
    -DHOST_VALIDATION_BUILD_TESTING=OFF
    -DHOST_VALIDATION_BUILD_PYTHON=OFF
    -DHOST_VALIDATION_BUILD_BLAS_BACKEND=OFF
)
execute_process(
    COMMAND ${_minimal_configure_command}
    RESULT_VARIABLE _minimal_configure_result
    OUTPUT_VARIABLE _minimal_configure_output
    ERROR_VARIABLE _minimal_configure_error
)
if(NOT _minimal_configure_result EQUAL 0)
    message(FATAL_ERROR
        "Minimal producer configure failed.\n"
        "${_minimal_configure_output}\n${_minimal_configure_error}"
    )
endif()

_build_project("${_minimal_build_dir}" "Minimal ROCHostValidation producer")
_install_project(
    "${_minimal_build_dir}"
    "${_minimal_install_dir}"
    "Minimal ROCHostValidation producer"
)
_run_component(
    "${_minimal_install_dir}"
    minimal-mx
    MX
    "Core,MX"
)
_expect_component_failure(
    "${_minimal_install_dir}"
    unavailable-blas-component
    BLAS
    "component \"BLAS\" is unavailable: this installation was built without "
)

message(STATUS "ROCHostValidation package component tests passed.")
