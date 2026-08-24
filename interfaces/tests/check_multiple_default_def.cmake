foreach(var CXX WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_multiple_default_def: ${var} not set")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")

set(ver_map "${WORKDIR}/abi04_dup.map")
file(WRITE "${ver_map}"
"ROCBLAS_ABI_6 { global: rocblas_sgemm; local: *; };\n"
"ROCBLAS_ABI_7 { global: rocblas_sgemm; } ROCBLAS_ABI_6;\n")

set(src6 "${WORKDIR}/impl6.cpp")
file(WRITE "${src6}"
"extern \"C\" int impl6(void) { return 6; }\n"
"__asm__(\".symver impl6, rocblas_sgemm@@ROCBLAS_ABI_6\");\n")

set(src7_dup "${WORKDIR}/impl7_dup.cpp")
file(WRITE "${src7_dup}"
"extern \"C\" int impl7(void) { return 7; }\n"
"__asm__(\".symver impl7, rocblas_sgemm@@ROCBLAS_ABI_7\");\n")

set(src7_ctrl "${WORKDIR}/impl7_ctrl.cpp")
file(WRITE "${src7_ctrl}"
"extern \"C\" int impl7(void) { return 7; }\n"
"__asm__(\".symver impl7, rocblas_sgemm@ROCBLAS_ABI_7\");\n")

# Compile one source file for the duplicate-default-definition link probes.
function(_compile src out)
  execute_process(
    COMMAND ${CXX} -O2 -fPIC -std=c++17 -c "${src}" -o "${out}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out_txt
    ERROR_VARIABLE err_txt)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "check_multiple_default_def: compile of ${src} failed: ${err_txt}")
  endif()
endfunction()

set(obj6 "${WORKDIR}/impl6.o")
set(obj7_dup "${WORKDIR}/impl7_dup.o")
set(obj7_ctrl "${WORKDIR}/impl7_ctrl.o")
_compile("${src6}" "${obj6}")
_compile("${src7_dup}" "${obj7_dup}")
_compile("${src7_ctrl}" "${obj7_ctrl}")

set(base_link "-Wl,--version-script=${ver_map}")
if(DEFINED LINKER AND NOT LINKER STREQUAL "")
  list(PREPEND base_link "-fuse-ld=${LINKER}")
endif()

set(dup_dso "${WORKDIR}/libdup.so")
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -shared ${base_link} "-Wl,-soname,libdup.so"
    "${obj6}" "${obj7_dup}" -o "${dup_dso}"
  RESULT_VARIABLE dup_rc
  OUTPUT_VARIABLE dup_out
  ERROR_VARIABLE dup_err)

if(dup_rc EQUAL 0)
  execute_process(
    COMMAND nm -D --defined-only --with-symbol-versions "${dup_dso}"
    OUTPUT_VARIABLE dup_nm)
  string(REGEX MATCHALL "rocblas_sgemm@@[A-Za-z0-9_]+" dup_defaults "${dup_nm}")
  list(LENGTH dup_defaults dup_default_count)
  message(FATAL_ERROR
    "check_multiple_default_def: two-default-definition DSO was NOT rejected "
    "(link rc=0, ${dup_default_count} @@ def(s): ${dup_defaults}); the single-default "
    "invariant is not enforced by the toolchain")
endif()

string(REGEX MATCH "duplicate symbol|multiple definition" dup_kind "${dup_err}")
string(FIND "${dup_err}" "rocblas_sgemm" dup_names_symbol)
if(dup_kind STREQUAL "" OR dup_names_symbol EQUAL -1)
  message(FATAL_ERROR
    "check_multiple_default_def: link failed (rc=${dup_rc}) but not for the expected "
    "duplicate-default-definition of rocblas_sgemm; refusing to treat an unrelated link "
    "failure as a pass. Linker said:\n${dup_err}")
endif()

set(ctrl_dso "${WORKDIR}/libctrl.so")
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -shared ${base_link} "-Wl,-soname,libctrl.so"
    "${obj6}" "${obj7_ctrl}" -o "${ctrl_dso}"
  RESULT_VARIABLE ctrl_rc
  OUTPUT_VARIABLE ctrl_out
  ERROR_VARIABLE ctrl_err)
if(NOT ctrl_rc EQUAL 0)
  message(FATAL_ERROR
    "check_multiple_default_def: single-default control (one @@, one @) failed to link "
    "(rc=${ctrl_rc}); the duplicate-default rejection above is not discriminating. "
    "Linker said:\n${ctrl_err}")
endif()

execute_process(
  COMMAND nm -D --defined-only --with-symbol-versions "${ctrl_dso}"
  RESULT_VARIABLE ctrl_nm_rc
  OUTPUT_VARIABLE ctrl_nm)
if(NOT ctrl_nm_rc EQUAL 0)
  message(FATAL_ERROR "check_multiple_default_def: nm on control DSO failed")
endif()
string(REGEX MATCHALL "rocblas_sgemm@@[A-Za-z0-9_]+" ctrl_defaults "${ctrl_nm}")
list(LENGTH ctrl_defaults ctrl_default_count)
string(REGEX MATCHALL "rocblas_sgemm@+[A-Za-z0-9_]+" ctrl_all "${ctrl_nm}")
list(LENGTH ctrl_all ctrl_all_count)
if(NOT ctrl_default_count EQUAL 1 OR NOT ctrl_all_count EQUAL 2)
  message(FATAL_ERROR
    "check_multiple_default_def: control DSO has ${ctrl_default_count} default (@@) and "
    "${ctrl_all_count} total versioned rocblas_sgemm def(s); expected exactly 1 @@ and 1 @")
endif()

message(STATUS
  "check_multiple_default_def: genuine two-@@ DSO REJECTED by the linker "
  "(${dup_kind} of rocblas_sgemm), while the single-@@ control links with exactly one "
  "default; single-default invariant enforced and the check is discriminating")
