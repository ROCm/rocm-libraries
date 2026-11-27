# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The one shape of an installed rocKE pytest entry. Included from
# platform/CMakeLists.txt, which defines the command pieces it reads:
# _ROCKE_INSTALLED_PYTEST, _ROCKE_INSTALLED_PYTEST_WORKERS,
# _ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT and _ROCKE_INSTALLED_GPU_RESOURCE_LOCK.

#   rocke_installed_pytest_entry(<out_var> <name>
#       [TESTS <path>...]
#       [ARGUMENTS <argument>...]
#       LABELS <label>...
#       ENVIRONMENT <text>
#       TIMEOUT <seconds>
#       [WORKERS] [GPU] [COMPILER_TOOLS] [COMPILER_REQUIREMENT])
#
# Sets <out_var> to the installed-CTestTestfile text that registers <name>.
# The command is ${_ROCKE_INSTALLED_PYTEST}, the worker plugin when WORKERS is
# given, then each of TESTS and ARGUMENTS as one quoted argument, in order,
# then the GPU per-test limit when GPU is given. TESTS are the paths or node
# ids the entry collects, as written in the CTestTestfile. ENVIRONMENT is the
# text of the ENVIRONMENT property. With COMPILER_TOOLS the entry's
# environment also takes the compiler tool locations
# (_rocke_compiler_tool_environment of the installed CTestTestfile), and with
# COMPILER_REQUIREMENT the policy that makes a missing tool fail
# (_rocke_compiler_requirement_environment). GPU also takes the GPU resource
# lock, so a parallel ctest -j never runs two GPU entries at once.
function(rocke_installed_pytest_entry out_var name)
  cmake_parse_arguments(PARSE_ARGV 2 _entry
                        "WORKERS;GPU;COMPILER_TOOLS;COMPILER_REQUIREMENT"
                        "ENVIRONMENT;TIMEOUT"
                        "TESTS;ARGUMENTS;LABELS")
  if(_entry_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "rocke_installed_pytest_entry(${name}): unknown arguments "
            "${_entry_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_required IN ITEMS LABELS ENVIRONMENT TIMEOUT)
    if(NOT DEFINED _entry_${_required})
      message(FATAL_ERROR
              "rocke_installed_pytest_entry(${name}): ${_required} is required")
    endif()
  endforeach()
  set(_command "${_ROCKE_INSTALLED_PYTEST}")
  if(_entry_WORKERS)
    string(APPEND _command " ${_ROCKE_INSTALLED_PYTEST_WORKERS}")
  endif()
  foreach(_argument IN LISTS _entry_TESTS _entry_ARGUMENTS)
    string(APPEND _command " \"${_argument}\"")
  endforeach()
  if(_entry_GPU)
    string(APPEND _command " ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT}")
  endif()

  set(_text "")
  set(_environment "${_entry_ENVIRONMENT}")
  if(_entry_COMPILER_TOOLS OR _entry_COMPILER_REQUIREMENT)
    set(_environment_list "_${name}_environment")
    string(APPEND _text
           "set(${_environment_list} \"${_entry_ENVIRONMENT}\")\n"
           "list(APPEND ${_environment_list}")
    if(_entry_COMPILER_TOOLS)
      string(APPEND _text "\n     \${_rocke_compiler_tool_environment}")
    endif()
    if(_entry_COMPILER_REQUIREMENT)
      string(APPEND _text "\n     \${_rocke_compiler_requirement_environment}")
    endif()
    string(APPEND _text ")\n")
    set(_environment "\${${_environment_list}}")
  endif()

  list(JOIN _entry_LABELS ";" _labels)
  string(APPEND _text
         "add_test(${name} ${_command})\n"
         "set_tests_properties(${name} PROPERTIES\n"
         "  LABELS \"${_labels}\"\n"
         "  ENVIRONMENT \"${_environment}\"\n")
  if(_entry_GPU)
    string(APPEND _text "  RESOURCE_LOCK ${_ROCKE_INSTALLED_GPU_RESOURCE_LOCK}\n")
  endif()
  string(APPEND _text "  TIMEOUT ${_entry_TIMEOUT})\n")
  set(${out_var} "${_text}" PARENT_SCOPE)
endfunction()
