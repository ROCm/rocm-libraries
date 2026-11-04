# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Called with _ROCKE_LIBRARY_DIR and ROCKE_TEST_INSTALL_DIR from the test install.
# The test tree installs the committed locks; this stages only published payloads.
# Installed architecture lists below also control the GPU CTest registrations.
# Each tests/<operation>_reference package with an architecture registry is a
# reference operation; reference_common/artifact.py discovers the same set.
file(GLOB _ROCKE_REFERENCE_REGISTRIES RELATIVE "${_ROCKE_LIBRARY_DIR}/tests"
     "${_ROCKE_LIBRARY_DIR}/tests/*_reference/architectures/registry.json")
set(_ROCKE_REFERENCE_OPERATIONS "")
foreach(_registry IN LISTS _ROCKE_REFERENCE_REGISTRIES)
  string(REGEX REPLACE "_reference/.*$" "" _operation "${_registry}")
  string(TOUPPER "${_operation}" _operation_upper)
  list(APPEND _ROCKE_REFERENCE_OPERATIONS "${_operation}")
  set(_ROCKE_${_operation_upper}_INSTALLED_ARCHITECTURES "")
endforeach()
# The glob orders whole paths, so conv_bwd_reference/ precedes conv_reference/;
# order by name, as artifact.py does.
list(SORT _ROCKE_REFERENCE_OPERATIONS)
if(NOT ROCKE_INSTALL_TEST_GPU_REFERENCES)
  return()
endif()
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR
          "ROCKE_INSTALL_TEST_GPU_REFERENCES requires a Python3 interpreter. "
          "Install Python3 or configure with ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/PublishedGpuReferences.cmake")
# A list without a package would install nothing, and a package without a list
# would register no GPU test; both must fail instead of losing coverage.
get_cmake_property(_variables VARIABLES)
foreach(_variable IN LISTS _variables)
  if(_variable MATCHES "^ROCKE_PUBLISHED_(.+)_ARCHITECTURES$")
    string(TOLOWER "${CMAKE_MATCH_1}" _operation)
    if(NOT _operation IN_LIST _ROCKE_REFERENCE_OPERATIONS)
      message(FATAL_ERROR
              "${_variable} names no reference operation: "
              "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/architectures/registry.json "
              "does not exist. Add that package or remove the list from "
              "PublishedGpuReferences.cmake.")
    endif()
  endif()
endforeach()
foreach(_operation IN LISTS _ROCKE_REFERENCE_OPERATIONS)
  string(TOUPPER "${_operation}" _operation_upper)
  if(NOT DEFINED ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    message(FATAL_ERROR
            "Reference operation ${_operation} has no publication list. Add "
            "set(ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES ...) to "
            "PublishedGpuReferences.cmake, with \"\" until an architecture is published.")
  endif()
  set(_metadata "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/operation.json")
  if(NOT EXISTS "${_metadata}")
    message(FATAL_ERROR
            "Reference operation ${_operation} has no ${_metadata} recording the "
            "manifest_operation its bundles declare.")
  endif()
  foreach(_arch IN LISTS ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    set(_lock "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/architectures/${_arch}/baseline_lock.json")
    set(_archive "${_ROCKE_LIBRARY_DIR}/tests/reference_bundles/${_operation}/${_arch}.tar.gz")
    if(NOT EXISTS "${_lock}")
      message(FATAL_ERROR "No qualified ${_operation} baseline lock for ${_arch}: ${_lock}")
    endif()
    if(NOT EXISTS "${_archive}")
      message(FATAL_ERROR
              "${_operation_upper} reference archive missing. Run dvc pull for "
              "${_archive}.dvc or configure with "
              "ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
    endif()
    file(SHA256 "${_archive}" _archive_sha256)
    set(_bundle "${CMAKE_CURRENT_BINARY_DIR}/${_operation}-reference/${_arch}/${_archive_sha256}")
    # Unpack authenticates the manifest and payload against the committed lock.
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -I
              "${_ROCKE_LIBRARY_DIR}/tests/reference_common/artifact.py" unpack
              --operation "${_operation}"
              --archive "${_archive}" --bundle "${_bundle}" --lock "${_lock}"
      RESULT_VARIABLE _result ERROR_VARIABLE _error)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "${_operation_upper} reference staging failed for ${_arch}: ${_error}")
    endif()
    # The engines/test_arch_content layout lets TheRock split test payloads by GPU.
    install(DIRECTORY "${_bundle}/"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/engines/test_arch_content/rocke/${_operation}/${_arch}"
            PATTERN "__pycache__" EXCLUDE REGEX "\\.pyc$" EXCLUDE)
    list(APPEND _ROCKE_${_operation_upper}_INSTALLED_ARCHITECTURES "${_arch}")
  endforeach()
endforeach()
