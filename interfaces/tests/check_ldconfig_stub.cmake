foreach(var LDCONFIG DSO_DIR WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_ldconfig_stub: ${var} not set")
  endif()
endforeach()

set(pkg "${WORKDIR}/pkg")
file(REMOVE_RECURSE "${pkg}")
file(MAKE_DIRECTORY "${pkg}")

foreach(n 5 6 7)
  if(NOT EXISTS "${DSO_DIR}/librocblas.so.${n}")
    message(FATAL_ERROR "check_ldconfig_stub: missing ${DSO_DIR}/librocblas.so.${n}")
  endif()
  file(COPY "${DSO_DIR}/librocblas.so.${n}" DESTINATION "${pkg}")
endforeach()

file(CREATE_LINK "librocblas.so.6" "${pkg}/librocblas.so" SYMBOLIC)

file(READ_SYMLINK "${pkg}/librocblas.so" pre_target)
execute_process(COMMAND stat -c "%i" "${pkg}/librocblas.so"
  OUTPUT_VARIABLE pre_inode OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND ${LDCONFIG} -n "${pkg}"
  RESULT_VARIABLE ldc_rc
  OUTPUT_VARIABLE ldc_out
  ERROR_VARIABLE ldc_err)
if(NOT ldc_rc EQUAL 0)
  message(FATAL_ERROR "check_ldconfig_stub: ldconfig -n failed: ${ldc_err}")
endif()

if(NOT IS_SYMLINK "${pkg}/librocblas.so")
  message(FATAL_ERROR "check_ldconfig_stub: dev stub librocblas.so removed by ldconfig")
endif()
file(READ_SYMLINK "${pkg}/librocblas.so" post_target)
execute_process(COMMAND stat -c "%i" "${pkg}/librocblas.so"
  OUTPUT_VARIABLE post_inode OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT pre_target STREQUAL post_target OR NOT pre_inode STREQUAL post_inode)
  message(FATAL_ERROR
    "check_ldconfig_stub: dev stub changed across ldconfig -n "
    "(${pre_target}/${pre_inode} -> ${post_target}/${post_inode})")
endif()

execute_process(
  COMMAND nm -D --defined-only --with-symbol-versions "${pkg}/librocblas.so.6"
  OUTPUT_VARIABLE nm_out RESULT_VARIABLE nm_rc ERROR_VARIABLE nm_err)
if(NOT nm_rc EQUAL 0)
  message(FATAL_ERROR "check_ldconfig_stub: nm failed: ${nm_err}")
endif()
if(NOT nm_out MATCHES "ROCBLAS_ABI_6")
  message(FATAL_ERROR "check_ldconfig_stub: version node ROCBLAS_ABI_6 lost after ldconfig")
endif()

message(STATUS "check_ldconfig_stub: dev stub survives ldconfig -n "
  "(-> ${post_target}, inode stable); version node intact")
