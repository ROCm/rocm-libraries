# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Called with _ROCKE_LIBRARY_DIR and ROCKE_TEST_INSTALL_DIR from the test install.
# The test tree installs the committed locks; this stages only published payloads.
# Installed architecture lists below also control the GPU CTest registrations.
set(_ROCKE_SDPA_INSTALLED_ARCHITECTURES "")
set(_ROCKE_CONV_INSTALLED_ARCHITECTURES "")
if(NOT ROCKE_INSTALL_TEST_GPU_REFERENCES)
  return()
endif()
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR
          "ROCKE_INSTALL_TEST_GPU_REFERENCES requires a Python3 interpreter. "
          "Install Python3 or configure with ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/PublishedGpuReferences.cmake")
foreach(_operation sdpa conv)
  string(TOUPPER "${_operation}" _operation_upper)
  foreach(_arch IN LISTS ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    set(_lock "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/architectures/${_arch}/baseline_lock.json")
    set(_archive "${_ROCKE_LIBRARY_DIR}/tests/reference_bundles/${_operation}/${_arch}.tar.gz")
    if(NOT EXISTS "${_lock}")
      message(FATAL_ERROR "No qualified ${_operation} baseline lock for ${_arch}: ${_lock}")
    endif()
    if(NOT EXISTS "${_archive}")
      message(FATAL_ERROR
              "${_operation_upper} reference archive missing. Run dvc pull for "
              "dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/${_operation}/${_arch}.tar.gz.dvc "
              "or configure with ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
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
