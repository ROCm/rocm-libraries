# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Called with _ROCKE_LIBRARY_DIR and ROCKE_TEST_INSTALL_DIR from the test install.
# The test tree installs the committed locks; this stages only published payloads.
# Installed architecture lists below also control the GPU CTest registrations,
# and each architecture's installed variants scale its CTest timeout.
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

include("${CMAKE_CURRENT_LIST_DIR}/PublishedGpuReferences.cmake")
# A committed lock publishes its variant together with the variant's adapter,
# its architecture's list entry and its DVC pointer, and a listed architecture
# must be enrolled in its operation's registry.json. Every build checks that
# wiring, with references or not, so neither a half-wired variant nor a file
# that a branch opened before the variant layout restores at its old path can
# silently drop a qualification.
set(_bundles "${_ROCKE_LIBRARY_DIR}/tests/reference_bundles")
foreach(_operation IN LISTS _ROCKE_REFERENCE_OPERATIONS)
  string(TOUPPER "${_operation}" _operation_upper)
  set(_architectures "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/architectures")
  # A listed architecture's GPU CTest fails at session start unless
  # registry.json, a flat JSON array of names, enrolls it.
  file(READ "${_architectures}/registry.json" _registry)
  string(REGEX MATCHALL "\"[^\"]*\"" _enrolled "${_registry}")
  string(REPLACE "\"" "" _enrolled "${_enrolled}")
  foreach(_arch IN LISTS ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
    if(NOT _arch IN_LIST _enrolled)
      message(FATAL_ERROR
              "ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES in "
              "PublishedGpuReferences.cmake lists ${_arch}, which "
              "${_architectures}/registry.json does not enroll, so its GPU "
              "reference test would fail before running. Enroll ${_arch} there "
              "or remove it from the list.")
    endif()
  endforeach()
  file(GLOB _legacy "${_architectures}/*/baseline_lock.json")
  foreach(_lock IN LISTS _legacy)
    message(FATAL_ERROR
            "${_lock} is a pre-variant baseline lock, which installation does not "
            "read. Move it to ${_architectures}/<arch>/<variant>/baseline_lock.json, "
            "beside its variant's adapter, or delete it.")
  endforeach()
  file(GLOB _locks RELATIVE "${_architectures}" "${_architectures}/*/*/baseline_lock.json")
  foreach(_lock IN LISTS _locks)
    get_filename_component(_variant_dir "${_lock}" DIRECTORY)
    get_filename_component(_arch "${_variant_dir}" DIRECTORY)
    get_filename_component(_variant "${_variant_dir}" NAME)
    set(_lock "${_architectures}/${_lock}")
    set(_pointer "${_bundles}/${_operation}/${_variant_dir}.tar.gz.dvc")
    # TheRock splits installed bundles by the first gfx-named directory.
    if(NOT _variant MATCHES "^[a-z][a-z0-9_]*$" OR _variant MATCHES "^gfx")
      message(FATAL_ERROR
              "${_lock} publishes the invalid variant name ${_variant}. A variant "
              "is named for its kernel family's module: it matches "
              "^[a-z][a-z0-9_]*$ and does not start with gfx.")
    endif()
    if(NOT EXISTS "${_architectures}/${_variant_dir}/__init__.py")
      message(FATAL_ERROR
              "${_lock} has no adapter: ${_architectures}/${_variant_dir}/__init__.py "
              "does not exist, so no reference test runs the variant that it "
              "publishes.")
    endif()
    if(NOT _arch IN_LIST ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES)
      message(FATAL_ERROR
              "${_lock} publishes ${_arch}, which "
              "ROCKE_PUBLISHED_${_operation_upper}_ARCHITECTURES in "
              "PublishedGpuReferences.cmake does not list. List ${_arch} there "
              "or delete the lock.")
    endif()
    if(NOT EXISTS "${_pointer}")
      message(FATAL_ERROR
              "${_lock} has no DVC pointer: ${_pointer} does not exist. Commit "
              "the pointer that dvc add wrote for the variant's archive.")
    endif()
  endforeach()
endforeach()
file(GLOB _legacy "${_bundles}/*/*.tar.gz.dvc")
foreach(_pointer IN LISTS _legacy)
  message(FATAL_ERROR
          "${_pointer} is a pre-variant DVC pointer, which installation does not "
          "read. Move it to ${_bundles}/<operation>/<arch>/<variant>.tar.gz.dvc "
          "with its lock, or delete it.")
endforeach()
file(GLOB _pointers RELATIVE "${_bundles}" "${_bundles}/*/*/*.tar.gz.dvc")
foreach(_pointer IN LISTS _pointers)
  string(REGEX REPLACE "^([^/]+)/(.+)\\.tar\\.gz\\.dvc$"
         "\\1_reference/architectures/\\2/baseline_lock.json" _lock "${_pointer}")
  if(NOT EXISTS "${_ROCKE_LIBRARY_DIR}/tests/${_lock}")
    message(FATAL_ERROR
            "${_bundles}/${_pointer} tracks an archive that no lock publishes: "
            "${_ROCKE_LIBRARY_DIR}/tests/${_lock} does not exist. Commit the "
            "variant's lock with its pointer, or delete the pointer.")
  endif()
endforeach()
# The root ignore file already ignores every archive, so dvc add writes no other
# and variants share no ignore file. Deeper ignore files sit inside extracted
# bundles, which it ignores whole.
file(GLOB _ignores "${_bundles}/*/.gitignore" "${_bundles}/*/*/.gitignore")
foreach(_ignore IN LISTS _ignores)
  message(FATAL_ERROR
          "Delete ${_ignore}: ${_bundles}/.gitignore is the only ignore file for "
          "reference bundles, and it already ignores every archive and extracted "
          "bundle.")
endforeach()

if(NOT ROCKE_INSTALL_TEST_GPU_REFERENCES)
  return()
endif()
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR
          "ROCKE_INSTALL_TEST_GPU_REFERENCES requires a Python3 interpreter. "
          "Install Python3 or configure with ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
endif()

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
    # A variant's committed lock publishes it, with its own archive and bundle.
    # The architecture's GPU CTest runs every variant installed here.
    set(_variants "${_ROCKE_LIBRARY_DIR}/tests/${_operation}_reference/architectures/${_arch}")
    file(GLOB _locks CONFIGURE_DEPENDS RELATIVE "${_variants}" "${_variants}/*/baseline_lock.json")
    list(LENGTH _locks _lock_count)
    if(_lock_count EQUAL 0)
      message(FATAL_ERROR
              "No qualified ${_operation} baseline lock for ${_arch}: "
              "${_variants}/<variant>/baseline_lock.json")
    endif()
    list(SORT _locks)
    set(_ROCKE_${_operation_upper}_${_arch}_INSTALLED_VARIANTS "")
    foreach(_lock IN LISTS _locks)
      get_filename_component(_variant "${_lock}" DIRECTORY)
      set(_lock "${_variants}/${_lock}")
      set(_archive "${_ROCKE_LIBRARY_DIR}/tests/reference_bundles/${_operation}/${_arch}/${_variant}.tar.gz")
      if(NOT EXISTS "${_archive}")
        message(FATAL_ERROR
                "${_operation_upper} reference archive missing. Run dvc pull for "
                "${_archive}.dvc or configure with "
                "ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF.")
      endif()
      file(SHA256 "${_archive}" _archive_sha256)
      set(_bundle "${CMAKE_CURRENT_BINARY_DIR}/${_operation}-reference/${_arch}/${_variant}/${_archive_sha256}")
      # Unpack authenticates the manifest and payload against the committed lock.
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -I
                "${_ROCKE_LIBRARY_DIR}/tests/reference_common/artifact.py" unpack
                --operation "${_operation}" --variant "${_variant}"
                --archive "${_archive}" --bundle "${_bundle}" --lock "${_lock}"
        RESULT_VARIABLE _result ERROR_VARIABLE _error)
      if(NOT _result EQUAL 0)
        message(FATAL_ERROR
                "${_operation_upper} reference staging failed for ${_arch} ${_variant}: ${_error}")
      endif()
      # The engines/test_arch_content layout lets TheRock split test payloads by
      # GPU. It keys on the first processor-named directory, so variants go below.
      install(DIRECTORY "${_bundle}/"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/engines/test_arch_content/rocke/${_operation}/${_arch}/${_variant}"
              PATTERN "__pycache__" EXCLUDE REGEX "\\.pyc$" EXCLUDE)
      list(APPEND _ROCKE_${_operation_upper}_${_arch}_INSTALLED_VARIANTS "${_variant}")
    endforeach()
    list(APPEND _ROCKE_${_operation_upper}_INSTALLED_ARCHITECTURES "${_arch}")
  endforeach()
endforeach()
