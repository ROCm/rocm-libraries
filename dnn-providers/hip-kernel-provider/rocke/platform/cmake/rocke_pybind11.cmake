# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke_pybind11.cmake -- find the pybind11 the engine's Python modules
# (rocke_engine, rocke_library_engine) build against.

# rocke_locate_pybind11(<found_var>)
#   Unless pybind11_DIR is already set, points the pybind11_DIR cache entry at
#   the CMake package of the pybind11 module installed for Python3_EXECUTABLE
#   (finding the interpreter first when nothing has). Sets <found_var> to TRUE
#   when pybind11_DIR then names a pybind11 CMake package, FALSE otherwise. It
#   only locates; callers still find_package(pybind11) to use it.
function(rocke_locate_pybind11 found_var)
  if(NOT pybind11_DIR)
    if(NOT Python3_EXECUTABLE)
      find_package(Python3 COMPONENTS Interpreter)
    endif()
    if(Python3_EXECUTABLE)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _rocke_pybind11_cmakedir
        RESULT_VARIABLE _rocke_pybind11_cmakedir_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(_rocke_pybind11_cmakedir_result EQUAL 0 AND EXISTS "${_rocke_pybind11_cmakedir}")
        set(pybind11_DIR "${_rocke_pybind11_cmakedir}" CACHE PATH "pybind11 CMake package directory")
      endif()
    endif()
  endif()
  if(pybind11_DIR AND (EXISTS "${pybind11_DIR}/pybind11Config.cmake"
                       OR EXISTS "${pybind11_DIR}/pybind11-config.cmake"))
    set(${found_var} TRUE PARENT_SCOPE)
  else()
    set(${found_var} FALSE PARENT_SCOPE)
  endif()
endfunction()
