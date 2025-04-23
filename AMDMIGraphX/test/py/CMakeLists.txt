#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

include(PythonModules)
set(VENV ${CMAKE_BINARY_DIR}/test/py/venv)
set(VENV_ONNX ${CMAKE_BINARY_DIR}/test/py/venv-onnx)
set(REQUIREMENTS ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt)
set(REQUIREMENTS_ONNX ${CMAKE_CURRENT_SOURCE_DIR}/requirements-onnx.txt)
set(PYTHON_VERSION_TO_DISABLE_ONNX 3.6)
option(MIGRAPHX_DISABLE_VIRTUAL_ENV "Disable python virtual environments" OFF)
option(MIGRAPHX_DISABLE_ONNX_TESTS "Disable Onnx backend tests" OFF)


function(add_py_venv_fixture FIXTURE_NAME VIRTUAL_ENV_DIR REQUIREMENTS_FILE)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_pybind_${PYTHON_VERSION}>"
            "PYTHONMALLOC=debug"
            "MALLOC_CHECK_=3"
        )
        set(PYTHON_EXECUTABLE ${PYTHON_${PYTHON_VERSION}_EXECUTABLE})

        if(NOT TEST py_${PYTHON_VERSION}_${FIXTURE_NAME}_initialize_env)
            if (NOT (${FIXTURE_NAME} STREQUAL "onnx" AND ${PYTHON_VERSION} STREQUAL ${PYTHON_VERSION_TO_DISABLE_ONNX}))
                add_test(NAME py_${PYTHON_VERSION}_${FIXTURE_NAME}_initialize_env COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUAL_ENV_DIR}/${PYTHON_VERSION} --clear)
                set_tests_properties(py_${PYTHON_VERSION}_${FIXTURE_NAME}_initialize_env PROPERTIES FIXTURES_SETUP ${FIXTURE_NAME}_${PYTHON_VERSION}_INIT_VENV)
                set(PYTHON_EXECUTABLE ${VIRTUAL_ENV_DIR}/${PYTHON_VERSION}/bin/python)
                if(EXISTS ${REQUIREMENTS_FILE})
                    add_test(
                        NAME py_${PYTHON_VERSION}_${FIXTURE_NAME}_setup_env
                        COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${REQUIREMENTS_FILE})
                else()
                    # If there is no requirements file, then there are no packages to install in the virtual env.
                    # Just create a placeholder test for setting up the required fixture for running the tests.
                    add_test(
                        NAME py_${PYTHON_VERSION}_${FIXTURE_NAME}_setup_env
                        COMMAND ${PYTHON_EXECUTABLE} -m pip install --help)
                endif()
                set_tests_properties(py_${PYTHON_VERSION}_${FIXTURE_NAME}_setup_env PROPERTIES FIXTURES_REQUIRED ${FIXTURE_NAME}_${PYTHON_VERSION}_INIT_VENV)
                set_tests_properties(py_${PYTHON_VERSION}_${FIXTURE_NAME}_setup_env PROPERTIES FIXTURES_SETUP ${FIXTURE_NAME}_${PYTHON_VERSION}_VENV)
            endif()
        endif()
    endforeach()
endfunction()

function(add_py_test NAME SCRIPT FIXTURE_NAME VENV_DIR)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_pybind_${PYTHON_VERSION}>"
            "PYTHONMALLOC=debug"
            "MALLOC_CHECK_=3"
        )
        if(MIGRAPHX_DISABLE_VIRTUAL_ENV)
            set(PYTHON_EXECUTABLE ${PYTHON_${PYTHON_VERSION}_EXECUTABLE})
        else()
            set(PYTHON_EXECUTABLE ${VENV_DIR}/${PYTHON_VERSION}/bin/python)
        endif()
        if(NOT (${FIXTURE_NAME} STREQUAL "onnx" AND ${PYTHON_VERSION} STREQUAL ${PYTHON_VERSION_TO_DISABLE_ONNX}))
            add_test(
                NAME test_py_${PYTHON_VERSION}_${NAME}
                COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
            add_custom_target(test_py_${PYTHON_VERSION}_${NAME}
                COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN}
                COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")
            if(NOT MIGRAPHX_DISABLE_VIRTUAL_ENV)
                set_tests_properties(test_py_${PYTHON_VERSION}_${NAME} PROPERTIES FIXTURES_REQUIRED ${FIXTURE_NAME}_${PYTHON_VERSION}_VENV)
            endif()
        endif()
    endforeach()
endfunction()
add_dependencies(tests migraphx_py)
add_dependencies(check migraphx_py)

if(NOT MIGRAPHX_DISABLE_VIRTUAL_ENV)
    add_py_venv_fixture(common ${VENV} ${REQUIREMENTS})
    add_py_venv_fixture(onnx ${VENV_ONNX} ${REQUIREMENTS_ONNX})
endif()

add_py_test(ref test_cpu.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(save_load test_save_load.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(op test_op.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(shape test_shape.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(module_construct test_module_construct.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(literal test_literal.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(autocast_fp8 test_autocast_fp8.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(MIGRAPHX_ENABLE_GPU)
add_py_test(gpu_offload test_gpu_offload.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu test_gpu.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(array test_array.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(NOT MIGRAPHX_DISABLE_ONNX_TESTS)
    add_py_test(backend onnx_backend_test.py onnx ${VENV_ONNX} WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
add_py_test(gpu_async test_gpu_async.py common ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
