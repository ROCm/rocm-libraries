if(EXPECTED_LOADER_EXPORTS)
  file(STRINGS "${EXPECTED_LOADER_EXPORTS}" expected_loader)
else()
  set(expected_loader
    rocblas_create_handle
    rocblas_destroy_handle
    rocblas_get_pointer_mode
    rocblas_get_stream
    rocblas_saxpy
    rocblas_sdot
    rocblas_set_pointer_mode
    rocblas_set_stream
    rocblas_sgemm
    rocblas_sgemm_64
    rocblas_sgemm_strided_batched)
endif()
list(SORT expected_loader)

if(NOT EXPECTED_LOADER_VERSION)
  set(EXPECTED_LOADER_VERSION ROCBLAS_ABI_5)
endif()
if(NOT EXPECTED_PROVIDER_VERSION)
  set(EXPECTED_PROVIDER_VERSION ROCM_INTERFACES_PROVIDER_1)
endif()

set(expected_narrow
  rocblas_create_handle
  rocblas_destroy_handle
  rocblas_get_pointer_mode
  rocblas_get_stream
  rocblas_saxpy
  rocblas_sdot
  rocblas_set_pointer_mode
  rocblas_set_stream
  rocblas_sgemm
  rocblas_sgemm_64
  rocblas_sgemm_strided_batched)
list(SORT expected_narrow)

# Read defined dynamic symbols and their ELF version-node associations.
function(read_exports path names_out versions_out)
  execute_process(
    COMMAND nm -D --defined-only --with-symbol-versions ${path}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE raw
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "nm failed for ${path}: ${error}")
  endif()
  string(REPLACE "\n" ";" lines "${raw}")
  set(names)
  set(pairs)
  foreach(line IN LISTS lines)
    if(line MATCHES "^[0-9A-Fa-f]+ ([A-Za-z]) (.+)$")
      set(symtype "${CMAKE_MATCH_1}")
      set(sym "${CMAKE_MATCH_2}")
      if(symtype STREQUAL "A" OR symtype STREQUAL "a")
        continue()
      endif()
      if(sym MATCHES "^([^@]+)@@?(.+)$")
        list(APPEND names "${CMAKE_MATCH_1}")
        list(APPEND pairs "${CMAKE_MATCH_1}=${CMAKE_MATCH_2}")
      else()
        list(APPEND names "${sym}")
        list(APPEND pairs "${sym}=")
      endif()
    endif()
  endforeach()
  list(SORT names)
  set(${names_out} "${names}" PARENT_SCOPE)
  set(${versions_out} "${pairs}" PARENT_SCOPE)
endfunction()

# Require every expected symbol to carry exactly the requested version node.
function(assert_versions label pairs expected_names expected_version)
  set(unversioned)
  set(wrong_version)
  foreach(sym IN LISTS expected_names)
    set(found_version "__MISSING__")
    foreach(pair IN LISTS pairs)
      if(pair MATCHES "^${sym}=(.*)$")
        set(found_version "${CMAKE_MATCH_1}")
        break()
      endif()
    endforeach()
    if(found_version STREQUAL "__MISSING__")
      message(FATAL_ERROR "${label}: expected export ${sym} not found")
    elseif(found_version STREQUAL "")
      list(APPEND unversioned "${sym}")
    elseif(NOT found_version STREQUAL "${expected_version}")
      list(APPEND wrong_version "${sym}@@${found_version}")
    endif()
  endforeach()
  if(unversioned)
    message(FATAL_ERROR
      "${label}: exports carry NO version node (unversioned): ${unversioned}\n"
      "every public symbol must carry the named node @@${expected_version}")
  endif()
  if(wrong_version)
    message(FATAL_ERROR
      "${label}: exports carry the wrong version node (expected @@${expected_version}): ${wrong_version}")
  endif()
endfunction()

read_exports("${LOADER}" actual_loader loader_pairs)
if(NOT actual_loader STREQUAL expected_loader)
  message(FATAL_ERROR "loader exports differ\nexpected=${expected_loader}\nactual=${actual_loader}")
endif()
assert_versions("loader" "${loader_pairs}" "${expected_loader}" "${EXPECTED_LOADER_VERSION}")

if(NARROW_V2_LOADER)
  read_exports("${NARROW_V2_LOADER}" actual_narrow_v2_loader narrow_v2_pairs)
  if(NOT actual_narrow_v2_loader STREQUAL expected_loader)
    message(FATAL_ERROR "narrow v2 loader exports differ\nexpected=${expected_loader}\nactual=${actual_narrow_v2_loader}")
  endif()
  assert_versions("narrow v2 loader" "${narrow_v2_pairs}" "${expected_loader}" "${EXPECTED_LOADER_VERSION}")
endif()

if(NARROW_LOADER)
  read_exports("${NARROW_LOADER}" actual_narrow_loader narrow_loader_pairs)
  if(NOT actual_narrow_loader STREQUAL expected_narrow)
    message(FATAL_ERROR "narrow loader exports differ\nexpected=${expected_narrow}\nactual=${actual_narrow_loader}")
  endif()
  assert_versions("narrow loader" "${narrow_loader_pairs}" "${expected_narrow}" "${EXPECTED_LOADER_VERSION}")
endif()

if(NOT DEFINED PROVIDER_COUNT)
  message(FATAL_ERROR
    "check_exports: PROVIDER_COUNT not supplied; provider export coverage would be silently skipped")
endif()
if(PROVIDER_COUNT LESS 1)
  message(FATAL_ERROR
    "check_exports: PROVIDER_COUNT=${PROVIDER_COUNT}; auto-derivation produced an empty provider list")
endif()
set(_idx 0)
while(_idx LESS PROVIDER_COUNT)
  set(_label "${PROVIDER_LABEL_${_idx}}")
  set(_file "${PROVIDER_FILE_${_idx}}")
  read_exports("${_file}" actual_provider provider_pairs)
  if(NOT actual_provider STREQUAL "rocm_interfaces_provider_query_v1")
    message(FATAL_ERROR "${_label} leaked exports: ${actual_provider}")
  endif()
  assert_versions("${_label}" "${provider_pairs}" "rocm_interfaces_provider_query_v1" "${EXPECTED_PROVIDER_VERSION}")
  math(EXPR _idx "${_idx} + 1")
endwhile()
