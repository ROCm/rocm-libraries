# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The one shape of an installed rocKE pytest entry: every installed entry that
# runs pytest under ${_ROCKE_INSTALLED_PYTEST} is registered through
# rocke_installed_pytest_entry. (The explicit gates of
# library/tests/ctest_lists.cmake stop on a partial run, so they run pytest
# without that command and keep their own registration.) Included from
# platform/CMakeLists.txt, which defines the command pieces it reads:
# _ROCKE_INSTALLED_PYTEST, _ROCKE_INSTALLED_PYTEST_WORKERS,
# _ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY, _ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT
# and _ROCKE_INSTALLED_GPU_RESOURCE_LOCK.

#   rocke_installed_pytest_entry(<out_var> <name>
#       [PYTEST_CONFIG <file>]
#       [TESTS <path>...]
#       [ARGUMENTS <argument>...]
#       LABELS <label>...
#       ENVIRONMENT <text>
#       TIMEOUT <seconds>
#       [SKIP_RETURN_CODE <code>]
#       [WORKERS] [GPU | GPU_OWN_TEST_LIMIT]
#       [COMPILER_TOOLS] [COMPILER_REQUIREMENT])
#
# Sets <out_var> to the installed-CTestTestfile text that registers <name>.
# The command is ${_ROCKE_INSTALLED_PYTEST}, then -c and PYTEST_CONFIG when it
# is given (the pytest configuration file of an entry that collects a directory
# with none above it), the worker plugin when WORKERS is given, then each of
# TESTS and ARGUMENTS as one quoted argument, in order, then the GPU text. TESTS
# are the paths or node ids the entry collects, as written in the
# CTestTestfile; a path listed twice stops the configure. ENVIRONMENT is the
# text of the ENVIRONMENT property. With COMPILER_TOOLS the entry's environment
# also takes the compiler tool locations (_rocke_compiler_tool_environment of
# the installed CTestTestfile), and with COMPILER_REQUIREMENT the policy that
# makes a missing tool fail (_rocke_compiler_requirement_environment).
# SKIP_RETURN_CODE is the exit code CTest reports as a skip.
#
# An entry whose tests launch GPU work gives one of two options. GPU ends the
# command with the GPU per-test limit, which includes the GPU test policy.
# GPU_OWN_TEST_LIMIT is for an entry with a per-test limit of its own, a
# --timeout=<seconds> in ARGUMENTS: it ends the command with the policy alone
# (the configure stops when ARGUMENTS has no --timeout=). Either
# takes the GPU resource lock, so a parallel ctest -j never runs two GPU
# entries at once.
function(rocke_installed_pytest_entry out_var name)
  cmake_parse_arguments(PARSE_ARGV 2 _entry
                        "WORKERS;GPU;GPU_OWN_TEST_LIMIT;COMPILER_TOOLS;COMPILER_REQUIREMENT"
                        "PYTEST_CONFIG;ENVIRONMENT;TIMEOUT;SKIP_RETURN_CODE"
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
  _rocke_refuse_repeated_tests(${name} ${_entry_TESTS})
  _rocke_installed_pytest_gpu_text(_gpu_text ${name} "${_entry_GPU}"
                                   "${_entry_GPU_OWN_TEST_LIMIT}" ${_entry_ARGUMENTS})

  set(_command "${_ROCKE_INSTALLED_PYTEST}")
  if(DEFINED _entry_PYTEST_CONFIG)
    string(APPEND _command " -c \"${_entry_PYTEST_CONFIG}\"")
  endif()
  if(_entry_WORKERS)
    string(APPEND _command " ${_ROCKE_INSTALLED_PYTEST_WORKERS}")
  endif()
  foreach(_argument IN LISTS _entry_TESTS _entry_ARGUMENTS)
    string(APPEND _command " \"${_argument}\"")
  endforeach()
  string(APPEND _command "${_gpu_text}")

  set(_compiler_options "")
  foreach(_option IN ITEMS COMPILER_TOOLS COMPILER_REQUIREMENT)
    if(_entry_${_option})
      list(APPEND _compiler_options ${_option})
    endif()
  endforeach()
  _rocke_installed_pytest_environment(_text _environment ${name} "${_entry_ENVIRONMENT}"
                                      ${_compiler_options})
  list(JOIN _entry_LABELS ";" _labels)
  string(APPEND _text
         "add_test(${name} ${_command})\n"
         "set_tests_properties(${name} PROPERTIES\n"
         "  LABELS \"${_labels}\"\n"
         "  ENVIRONMENT \"${_environment}\"\n")
  if(DEFINED _entry_SKIP_RETURN_CODE)
    string(APPEND _text "  SKIP_RETURN_CODE ${_entry_SKIP_RETURN_CODE}\n")
  endif()
  if(_gpu_text)
    string(APPEND _text "  RESOURCE_LOCK ${_ROCKE_INSTALLED_GPU_RESOURCE_LOCK}\n")
  endif()
  string(APPEND _text "  TIMEOUT ${_entry_TIMEOUT})\n")
  set(${out_var} "${_text}" PARENT_SCOPE)
endfunction()

# Set <out_var> to the text that ends the command of the pytest entry <name>:
# the GPU per-test limit when <gpu> is true, the GPU test policy alone when
# <own_limit> is true, else nothing. The remaining arguments are the entry's
# ARGUMENTS. Stop the configure when both are true, or when <own_limit> is
# true and no argument is a --timeout.
function(_rocke_installed_pytest_gpu_text out_var name gpu own_limit)
  set(_text "")
  if(gpu AND own_limit)
    message(FATAL_ERROR
            "rocke_installed_pytest_entry(${name}): give GPU or "
            "GPU_OWN_TEST_LIMIT, not both")
  elseif(gpu)
    set(_text " ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT}")
  elseif(own_limit)
    set(_own_limit ${ARGN})
    list(FILTER _own_limit INCLUDE REGEX "^--timeout=")
    if(NOT _own_limit)
      message(FATAL_ERROR
              "rocke_installed_pytest_entry(${name}): GPU_OWN_TEST_LIMIT needs "
              "the entry's --timeout=<seconds> in ARGUMENTS; without one, give GPU")
    endif()
    set(_text " ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY}")
  endif()
  set(${out_var} "${_text}" PARENT_SCOPE)
endfunction()

# Set <text_var> to the CTestTestfile text that must precede the registration
# of the pytest entry <name>, and <value_var> to the value of its ENVIRONMENT
# property. Without COMPILER_TOOLS or COMPILER_REQUIREMENT they are nothing and
# <base>; otherwise the text defines the list _<name>_environment, <base>
# followed by the compiler tool locations (COMPILER_TOOLS) and the compiler
# requirement policy (COMPILER_REQUIREMENT), and the value names that list.
function(_rocke_installed_pytest_environment text_var value_var name base)
  cmake_parse_arguments(PARSE_ARGV 4 _with "COMPILER_TOOLS;COMPILER_REQUIREMENT" "" "")
  set(_text "")
  set(_value "${base}")
  if(_with_COMPILER_TOOLS OR _with_COMPILER_REQUIREMENT)
    set(_list "_${name}_environment")
    string(APPEND _text "set(${_list} \"${base}\")\nlist(APPEND ${_list}")
    if(_with_COMPILER_TOOLS)
      string(APPEND _text "\n     \${_rocke_compiler_tool_environment}")
    endif()
    if(_with_COMPILER_REQUIREMENT)
      string(APPEND _text "\n     \${_rocke_compiler_requirement_environment}")
    endif()
    string(APPEND _text ")\n")
    set(_value "\${${_list}}")
  endif()
  set(${text_var} "${_text}" PARENT_SCOPE)
  set(${value_var} "${_value}" PARENT_SCOPE)
endfunction()

# Stop the configure when the TESTS of the pytest entry <name>, the remaining
# arguments, name a path more than once; the message names each repeat.
function(_rocke_refuse_repeated_tests name)
  set(_seen "")
  set(_repeated "")
  foreach(_test IN LISTS ARGN)
    if(NOT _test IN_LIST _seen)
      list(APPEND _seen "${_test}")
    elseif(NOT _test IN_LIST _repeated)
      list(APPEND _repeated "${_test}")
    endif()
  endforeach()
  if(_repeated)
    list(JOIN _repeated ", " _repeated)
    message(FATAL_ERROR
            "rocke_installed_pytest_entry(${name}): TESTS lists ${_repeated} "
            "more than once; list each path once")
  endif()
endfunction()
