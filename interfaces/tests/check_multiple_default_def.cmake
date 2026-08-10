foreach(var CXX FIXTURE WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_multiple_default_def: ${var} not set")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")

set(dup_map "${WORKDIR}/abi04_dup.map")
file(WRITE "${dup_map}"
"ROCBLAS_ABI_6 { global: rocblas_sgemm; local: *; };\n"
"ROCBLAS_ABI_7 { global: rocblas_sgemm; } ROCBLAS_ABI_6;\n")

set(obj "${WORKDIR}/dup.o")
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -DABI_VER=6 -c "${FIXTURE}" -o "${obj}"
  RESULT_VARIABLE compile_rc
  OUTPUT_VARIABLE compile_out
  ERROR_VARIABLE compile_err)
if(NOT compile_rc EQUAL 0)
  message(FATAL_ERROR "check_multiple_default_def: fixture compile failed: ${compile_err}")
endif()

set(dso "${WORKDIR}/libdup.so")
set(link_args "-Wl,--version-script=${dup_map}" "-Wl,-soname,libdup.so")
if(DEFINED LINKER AND NOT LINKER STREQUAL "")
  list(PREPEND link_args "-fuse-ld=${LINKER}")
endif()
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -shared ${link_args} "${obj}" -o "${dso}"
  RESULT_VARIABLE link_rc
  OUTPUT_VARIABLE link_out
  ERROR_VARIABLE link_err)

if(NOT link_rc EQUAL 0)
  message(STATUS "check_multiple_default_def: linker REJECTS duplicate default def "
    "(rc=${link_rc}); single-default invariant enforced by the linker")
  return()
endif()

execute_process(
  COMMAND nm -D --defined-only --with-symbol-versions "${dso}"
  RESULT_VARIABLE nm_rc
  OUTPUT_VARIABLE nm_out
  ERROR_VARIABLE nm_err)
if(NOT nm_rc EQUAL 0)
  message(FATAL_ERROR "check_multiple_default_def: nm failed: ${nm_err}")
endif()

string(REGEX MATCHALL "rocblas_sgemm@@[A-Za-z0-9_]+" defaults "${nm_out}")
list(LENGTH defaults default_count)
if(default_count GREATER 1)
  message(FATAL_ERROR
    "check_multiple_default_def: ${default_count} default (@@) defs of rocblas_sgemm "
    "coexist (${defaults}); single-default invariant VIOLATED")
endif()
message(STATUS "check_multiple_default_def: link succeeded with "
  "${default_count} default (@@) def of rocblas_sgemm; single-default invariant holds")
