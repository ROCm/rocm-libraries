# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Called with _rocke_library_dir and ROCKE_TEST_INSTALL_DIR from the test
# install. The test tree installs the committed locks; this stages only
# published payloads. Installed architecture lists below also control the GPU
# CTest registrations. Each tests/<operation>_reference package with an
# architecture registry is a reference operation; reference_common/artifact.py
# discovers the same set.
file(
  GLOB _rocke_reference_registries
  RELATIVE "${_rocke_library_dir}/tests"
  "${_rocke_library_dir}/tests/*_reference/architectures/registry.json")
set(_rocke_reference_operations "")
foreach(registry IN LISTS _rocke_reference_registries)
  string(REGEX REPLACE "_reference/.*$" "" operation "${registry}")
  list(APPEND _rocke_reference_operations "${operation}")
  set(_rocke_${operation}_installed_architectures "")
endforeach()
# The glob orders whole paths, so conv_bwd_reference/ precedes conv_reference/;
# order by name, as artifact.py does.
list(SORT _rocke_reference_operations)
if(NOT ROCKE_INSTALL_TEST_GPU_REFERENCES)
  return()
endif()
if(NOT Python3_Interpreter_FOUND)
  message(
    FATAL_ERROR
      "ROCKE_INSTALL_TEST_GPU_REFERENCES requires a Python3 interpreter. "
      "Install Python3 or configure with ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF."
  )
endif()

include("${CMAKE_CURRENT_LIST_DIR}/PublishedGpuReferences.cmake")
# A list without a package would install nothing, and a package without a list
# would register no GPU test; both must fail instead of losing coverage.
get_cmake_property(_variables VARIABLES)
foreach(variable IN LISTS _variables)
  if(variable MATCHES "^ROCKE_PUBLISHED_(.+)_ARCHITECTURES$")
    string(TOLOWER "${CMAKE_MATCH_1}" operation)
    if(NOT operation IN_LIST _rocke_reference_operations)
      message(
        FATAL_ERROR
          "${variable} names no reference operation: "
          "${_rocke_library_dir}/tests/${operation}_reference/architectures/r"
          "egistry.json "
          "does not exist. Add that package or remove the list from "
          "PublishedGpuReferences.cmake.")
    endif()
  endif()
endforeach()
foreach(operation IN LISTS _rocke_reference_operations)
  string(TOUPPER "${operation}" _operation_upper)
  if(NOT DEFINED ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    message(
      FATAL_ERROR
        "Reference operation ${operation} has no publication list. Add "
        "set(ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES ...) to "
        "PublishedGpuReferences.cmake, with \"\" until an architecture is pub"
        "lished.")
  endif()
  string(CONCAT _metadata
                "${_rocke_library_dir}/tests/${operation}_reference/operation"
                ".json")
  if(NOT EXISTS "${_metadata}")
    message(
      FATAL_ERROR
        "Reference operation ${operation} has no ${_metadata} recording the "
        "manifest_operation its bundles declare.")
  endif()
  foreach(arch IN LISTS ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    string(CONCAT _lock
                  "${_rocke_library_dir}/tests/${operation}_reference/architect"
                  "ures/${arch}/baseline_lock.json")
    string(CONCAT _archive
                  "${_rocke_library_dir}/tests/reference_bundles/${operation}/"
                  "${arch}.tar.gz")
    if(NOT EXISTS "${_lock}")
      message(
        FATAL_ERROR
          "No qualified ${operation} baseline lock for ${arch}: ${_lock}")
    endif()
    if(NOT EXISTS "${_archive}")
      message(
        FATAL_ERROR
          "${_operation_upper} reference archive missing. Run dvc pull for "
          "${_archive}.dvc or configure with "
          "ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
    endif()
    file(SHA256 "${_archive}" _archive_sha256)
    string(CONCAT _bundle
                  "${CMAKE_CURRENT_BINARY_DIR}/${operation}-reference/${arch}/"
                  "${_archive_sha256}")
    # Unpack authenticates the manifest and payload against the committed lock.
    execute_process(
      COMMAND
        "${Python3_EXECUTABLE}" -I
        "${_rocke_library_dir}/tests/reference_common/artifact.py" unpack
        --operation "${operation}" --archive "${_archive}" --bundle "${_bundle}"
        --lock "${_lock}"
      RESULT_VARIABLE _result
      ERROR_VARIABLE _error)
    if(NOT _result EQUAL 0)
      message(
        FATAL_ERROR
          "${_operation_upper} reference staging failed for ${arch}: ${_error}")
    endif()
    # The engines/test_arch_content layout lets TheRock split test payloads by
    # GPU.
    string(CONCAT _reference_destination
                  "${ROCKE_TEST_INSTALL_DIR}/engines/test_arch_content/rocke/"
                  "${operation}/${arch}")
    install(
      DIRECTORY "${_bundle}/"
      DESTINATION "${_reference_destination}"
      PATTERN "__pycache__" EXCLUDE
      REGEX "\\.pyc$" EXCLUDE)
    list(APPEND _rocke_${operation}_installed_architectures "${arch}")
  endforeach()
endforeach()
