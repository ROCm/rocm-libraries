foreach(var CXX FIXTURE WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_asan_version_node: ${var} not set")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")

set(asan_map "${WORKDIR}/abi_asan.map")
file(WRITE "${asan_map}"
"ROCBLAS_ABI_6 {\n  global:\n    rocblas_*;\n  local:\n    *;\n};\n")

set(obj "${WORKDIR}/asan.o")
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -fsanitize=address -g -fno-omit-frame-pointer -DABI_VER=6 -c "${FIXTURE}" -o "${obj}"
  RESULT_VARIABLE compile_rc
  OUTPUT_VARIABLE compile_out
  ERROR_VARIABLE compile_err)
if(NOT compile_rc EQUAL 0)
  message(FATAL_ERROR "check_asan_version_node: ASan fixture compile failed: ${compile_err}")
endif()

set(dso "${WORKDIR}/librocblas.so.6")
set(link_args "-Wl,--version-script=${asan_map}" "-Wl,-soname,librocblas.so.6")
if(DEFINED LINKER AND NOT LINKER STREQUAL "")
  list(PREPEND link_args "-fuse-ld=${LINKER}")
endif()
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -fsanitize=address -shared ${link_args} "${obj}" -o "${dso}"
  RESULT_VARIABLE link_rc
  OUTPUT_VARIABLE link_out
  ERROR_VARIABLE link_err)
if(NOT link_rc EQUAL 0)
  message(FATAL_ERROR "check_asan_version_node: ASan DSO link failed: ${link_err}")
endif()

execute_process(
  COMMAND nm -D --with-symbol-versions "${dso}"
  RESULT_VARIABLE nm_rc
  OUTPUT_VARIABLE nm_out
  ERROR_VARIABLE nm_err)
if(NOT nm_rc EQUAL 0)
  message(FATAL_ERROR "check_asan_version_node: nm failed: ${nm_err}")
endif()

if(NOT nm_out MATCHES "__asan_")
  message(FATAL_ERROR
    "check_asan_version_node: DSO carries no __asan_ symbols; "
    "-fsanitize=address did not instrument the build")
endif()

if(NOT nm_out MATCHES "rocblas_sgemm@@ROCBLAS_ABI_6")
  message(FATAL_ERROR
    "check_asan_version_node: rocblas_sgemm not bound as default (@@) to ROCBLAS_ABI_6 "
    "under -fsanitize=address")
endif()

set(nodeless_map "${WORKDIR}/abi_asan_nodeless.map")
file(WRITE "${nodeless_map}"
"{\n  global:\n    rocblas_*;\n  local:\n    *;\n};\n")

set(nodeless_dso "${WORKDIR}/librocblas_nodeless.so")
set(nodeless_link_args "-Wl,--version-script=${nodeless_map}" "-Wl,-soname,librocblas_nodeless.so")
if(DEFINED LINKER AND NOT LINKER STREQUAL "")
  list(PREPEND nodeless_link_args "-fuse-ld=${LINKER}")
endif()
execute_process(
  COMMAND ${CXX} -O2 -fPIC -std=c++17 -fsanitize=address -shared ${nodeless_link_args} "${obj}" -o "${nodeless_dso}"
  RESULT_VARIABLE nodeless_link_rc
  OUTPUT_VARIABLE nodeless_link_out
  ERROR_VARIABLE nodeless_link_err)
if(NOT nodeless_link_rc EQUAL 0)
  message(FATAL_ERROR "check_asan_version_node: nodeless control DSO link failed: ${nodeless_link_err}")
endif()

execute_process(
  COMMAND nm -D --with-symbol-versions "${nodeless_dso}"
  RESULT_VARIABLE nodeless_nm_rc
  OUTPUT_VARIABLE nodeless_nm_out
  ERROR_VARIABLE nodeless_nm_err)
if(NOT nodeless_nm_rc EQUAL 0)
  message(FATAL_ERROR "check_asan_version_node: nm failed on nodeless control: ${nodeless_nm_err}")
endif()

if(NOT nodeless_nm_out MATCHES "__asan_")
  message(FATAL_ERROR
    "check_asan_version_node: nodeless control carries no __asan_ symbols; "
    "it is not a genuine ASan build, so its clean result proves nothing")
endif()

if(NOT nodeless_nm_out MATCHES "rocblas_sgemm")
  message(FATAL_ERROR
    "check_asan_version_node: nodeless control does not export rocblas_sgemm; "
    "the control is broken")
endif()

if(nodeless_nm_out MATCHES "ROCBLAS_ABI_6")
  message(FATAL_ERROR
    "check_asan_version_node: nodeless ASan build shows a ROCBLAS_ABI_6 node; "
    "the positive assertion is not discriminating")
endif()

message(STATUS "check_asan_version_node: ASan-instrumented DSO binds rocblas_sgemm@@ROCBLAS_ABI_6 "
  "(__asan_ imports present); nodeless anonymous-map control (also ASan) exports rocblas_sgemm "
  "with zero version nodes, so the node survives ASan and the assertion is discriminating")
