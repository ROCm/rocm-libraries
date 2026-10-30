# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include("${CMAKE_CURRENT_LIST_DIR}/ImportedLibraryLocation.cmake")

# Build-tree commands only. Installed tests use runtime discovery after relocation.
set(_ROCKE_COMGR_TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/../tools/run_with_comgr.py")
# Anchor build-local selections before packaging or CTest changes directory.
# Do not collapse '..': traversing a directory symlink first can select a
# different compiler from the lexically normalized path.
function(rocke_normalize_comgr_path output path)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, sys; p = sys.argv[1]; print(p if os.path.isabs(p) else os.path.join(os.getcwd(), p))" "${path}"
    RESULT_VARIABLE _result OUTPUT_VARIABLE _absolute OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Could not normalize the build-local compiler path")
  endif()
  set(${output} "${_absolute}" PARENT_SCOPE)
endfunction()

# Form a build-test command prefix that supplies a compiler default while
# preserving the environment of the process that invokes CTest.
function(rocke_comgr_test_launcher output default_comgr)
  set(_launcher "")
  if(default_comgr)
    # Match configure-time relative overrides before CTest changes directory.
    rocke_normalize_comgr_path(_absolute "${default_comgr}")
    set(_launcher "${Python3_EXECUTABLE}" "${_ROCKE_COMGR_TEST_LAUNCHER}"
                  --default-comgr "${_absolute}" --)
  endif()
  set(${output} "${_launcher}" PARENT_SCOPE)
endfunction()
