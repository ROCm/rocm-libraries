foreach(var CXX FIXTURE MAP ROCRAND_INC WORKDIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "check_cpp_mangled_version: ${var} not set")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")

set(cxxflags -O2 -fPIC -std=c++17 -D__HIP_PLATFORM_AMD__
  "-I${ROCRAND_INC}" -I/opt/rocm/include)

set(obj "${WORKDIR}/abi05_cpp.o")
execute_process(
  COMMAND ${CXX} ${cxxflags} -c "${FIXTURE}" -o "${obj}"
  RESULT_VARIABLE compile_rc
  OUTPUT_VARIABLE compile_out
  ERROR_VARIABLE compile_err)
if(NOT compile_rc EQUAL 0)
  message(FATAL_ERROR "check_cpp_mangled_version: fixture compile failed: ${compile_err}")
endif()

set(link_common -O2 -fPIC -shared)
if(DEFINED LINKER AND NOT LINKER STREQUAL "")
  list(PREPEND link_common "-fuse-ld=${LINKER}")
endif()

set(versioned "${WORKDIR}/libabi05_versioned.so")
execute_process(
  COMMAND ${CXX} ${link_common}
    "-Wl,--version-script=${MAP}" "-Wl,-soname,librocrand_cpp.so.6"
    "${obj}" -o "${versioned}"
  RESULT_VARIABLE vlink_rc
  OUTPUT_VARIABLE vlink_out
  ERROR_VARIABLE vlink_err)
if(NOT vlink_rc EQUAL 0)
  message(FATAL_ERROR "check_cpp_mangled_version: versioned link failed: ${vlink_err}")
endif()

execute_process(
  COMMAND nm -D --defined-only --with-symbol-versions "${versioned}"
  RESULT_VARIABLE nm_rc OUTPUT_VARIABLE nm_out ERROR_VARIABLE nm_err)
if(NOT nm_rc EQUAL 0)
  message(FATAL_ERROR "check_cpp_mangled_version: nm failed: ${nm_err}")
endif()

string(REPLACE "\n" ";" nm_lines "${nm_out}")
set(method_versioned 0)
set(rtti_versioned 0)
foreach(line IN LISTS nm_lines)
  if(line MATCHES "@@ROCBLAS_ABI_6")
    if(line MATCHES "_ZN11rocrand_cpp5error" OR line MATCHES "_ZNK11rocrand_cpp5error")
      math(EXPR method_versioned "${method_versioned} + 1")
    endif()
    if(line MATCHES "_ZTVN11rocrand_cpp5error" OR line MATCHES "_ZTIN11rocrand_cpp5error")
      math(EXPR rtti_versioned "${rtti_versioned} + 1")
    endif()
  endif()
endforeach()

if(method_versioned LESS 1)
  message(FATAL_ERROR
    "check_cpp_mangled_version: 0 rocrand_cpp mangled method symbols carry "
    "@@ROCBLAS_ABI_6 (fixture failed to force out-of-line emission, or the map "
    "assigned no version node); C++ mangled-method versioning NOT proven")
endif()
if(rtti_versioned LESS 1)
  message(FATAL_ERROR
    "check_cpp_mangled_version: 0 rocrand_cpp vtable/typeinfo symbols carry "
    "@@ROCBLAS_ABI_6; C++ RTTI/vtable versioning NOT proven")
endif()

set(anon_map "${WORKDIR}/abi05_anon.map")
file(WRITE "${anon_map}"
"{\n  global:\n    _ZN11rocrand_cpp*;\n    _ZNK11rocrand_cpp*;\n"
"    _ZTVN11rocrand_cpp*;\n    _ZTIN11rocrand_cpp*;\n    _ZTSN11rocrand_cpp*;\n"
"  local:\n    *;\n};\n")

set(anon "${WORKDIR}/libabi05_anon.so")
execute_process(
  COMMAND ${CXX} ${link_common}
    "-Wl,--version-script=${anon_map}" "-Wl,-soname,librocrand_cpp_anon.so"
    "${obj}" -o "${anon}"
  RESULT_VARIABLE alink_rc OUTPUT_VARIABLE alink_out ERROR_VARIABLE alink_err)
if(NOT alink_rc EQUAL 0)
  message(FATAL_ERROR "check_cpp_mangled_version: anon control link failed: ${alink_err}")
endif()

execute_process(
  COMMAND nm -D --defined-only --with-symbol-versions "${anon}"
  RESULT_VARIABLE anm_rc OUTPUT_VARIABLE anm_out ERROR_VARIABLE anm_err)
if(NOT anm_rc EQUAL 0)
  message(FATAL_ERROR "check_cpp_mangled_version: anon nm failed: ${anm_err}")
endif()
string(REPLACE "\n" ";" anm_lines "${anm_out}")
set(mangled_present 0)
set(node_leaked 0)
foreach(line IN LISTS anm_lines)
  if(line MATCHES "_ZN11rocrand_cpp" OR line MATCHES "_ZNK11rocrand_cpp"
     OR line MATCHES "_ZTVN11rocrand_cpp" OR line MATCHES "_ZTIN11rocrand_cpp"
     OR line MATCHES "_ZTSN11rocrand_cpp")
    math(EXPR mangled_present "${mangled_present} + 1")
    if(line MATCHES "ROCBLAS_ABI_6")
      math(EXPR node_leaked "${node_leaked} + 1")
    endif()
  endif()
endforeach()

if(mangled_present LESS 1)
  message(FATAL_ERROR
    "check_cpp_mangled_version: anon control exported 0 rocrand_cpp mangled symbols; "
    "fixture forcing is broken (non-vacuity control failed)")
endif()
if(node_leaked GREATER 0)
  message(FATAL_ERROR
    "check_cpp_mangled_version: anonymous map (no version node) shows ${node_leaked} "
    "ROCBLAS_ABI_6-tagged rocrand_cpp symbols; the @@ROCBLAS_ABI_6 assertion is not "
    "discriminating (non-vacuity control failed)")
endif()

message(STATUS
  "check_cpp_mangled_version: ${method_versioned} rocrand_cpp mangled methods and "
  "${rtti_versioned} vtable/typeinfo symbols carry @@ROCBLAS_ABI_6; anonymous control "
  "exports ${mangled_present} rocrand_cpp symbols with 0 version nodes; C++ "
  "mangled-export versioning proven and discriminating")
