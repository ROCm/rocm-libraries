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

function(read_exports path output)
  execute_process(
    COMMAND nm -D --defined-only ${path}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE raw
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "nm failed for ${path}: ${error}")
  endif()
  string(REPLACE "\n" ";" lines "${raw}")
  set(names)
  foreach(line IN LISTS lines)
    if(line MATCHES "^[0-9A-Fa-f]+ [A-Za-z] (.+)$")
      list(APPEND names "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(SORT names)
  set(${output} "${names}" PARENT_SCOPE)
endfunction()

read_exports("${LOADER}" actual_loader)
list(SORT expected_loader)
if(NOT actual_loader STREQUAL expected_loader)
  message(FATAL_ERROR "loader exports differ\nexpected=${expected_loader}\nactual=${actual_loader}")
endif()
if(NARROW_V2_LOADER)
  read_exports("${NARROW_V2_LOADER}" actual_narrow_v2_loader)
  if(NOT actual_narrow_v2_loader STREQUAL expected_loader)
    message(FATAL_ERROR "narrow v2 loader exports differ\nexpected=${expected_loader}\nactual=${actual_narrow_v2_loader}")
  endif()
endif()

foreach(provider IN ITEMS BLAS_PROVIDER COMBINED_BLAS_PROVIDER BLASLT_PROVIDER SOLVER_PROVIDER RAND_PROVIDER ROCBLAS_BRIDGE_PROVIDER BLAS_NARROW_V2_PROVIDER)
  read_exports("${${provider}}" actual_provider)
  if(NOT actual_provider STREQUAL "rocm_interfaces_provider_query_v1")
    message(FATAL_ERROR "${provider} leaked exports: ${actual_provider}")
  endif()
endforeach()
