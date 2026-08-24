add_executable(rocm_interfaces_tests interfaces_test.cpp)
target_link_libraries(rocm_interfaces_tests PRIVATE rocm_interfaces_loader)
target_compile_definitions(rocm_interfaces_tests PRIVATE
  BLAS_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_classic_provider_recording>"
  COMBINED_BLAS_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_provider_recording>"
  BLASLT_PROVIDER_PATH="$<TARGET_FILE:rocm_blaslt_provider_recording>"
  SOLVER_PROVIDER_PATH="$<TARGET_FILE:rocm_solver_provider_recording>"
  RAND_PROVIDER_PATH="$<TARGET_FILE:rocm_rand_provider_recording>"
)
add_dependencies(rocm_interfaces_tests
  rocm_blas_provider_recording
  rocm_blas_classic_provider_recording
  rocm_blaslt_provider_recording
  rocm_solver_provider_recording
  rocm_rand_provider_recording)
add_test(NAME rocm_interfaces.unit COMMAND rocm_interfaces_tests)

add_executable(table_abi_negotiation_test table_abi_negotiation_test.cpp)
target_link_libraries(table_abi_negotiation_test PRIVATE rocm_interfaces_loader)
add_test(NAME rocm_interfaces.table_abi_negotiation COMMAND table_abi_negotiation_test)

add_executable(rocblas_shadow_test rocblas_shadow_test.cpp)
target_link_libraries(rocblas_shadow_test PRIVATE rocblas_loader_shadow)
target_compile_definitions(rocblas_shadow_test PRIVATE
  ROCBLAS_COMPLETE_BRIDGE_TEST=1
  BLAS_PROVIDER_PATH="$<TARGET_FILE:rocm_rocblas_bridge_provider_recording>")
add_test(NAME rocm_interfaces.rocblas_shadow COMMAND rocblas_shadow_test)

add_executable(rocblas_narrow_shadow_test rocblas_shadow_test.cpp)
target_link_libraries(rocblas_narrow_shadow_test PRIVATE rocblas_narrow_loader_shadow)
target_compile_definitions(rocblas_narrow_shadow_test PRIVATE
  BLAS_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_classic_provider_recording>")
add_test(NAME rocm_interfaces.rocblas_narrow_shadow COMMAND rocblas_narrow_shadow_test)

add_executable(rocblas_narrow_v2_test rocblas_narrow_v2_test.cpp)
target_include_directories(rocblas_narrow_v2_test PRIVATE ${ROCM_INTERFACES_STAGED_INCLUDE})
target_link_libraries(rocblas_narrow_v2_test PRIVATE rocblas_narrow_v2_loader_shadow)
target_compile_definitions(rocblas_narrow_v2_test PRIVATE
  __HIP_PLATFORM_AMD__=1
  BLAS_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_narrow_v2_provider_recording>")
add_test(NAME rocm_interfaces.rocblas_narrow_v2 COMMAND rocblas_narrow_v2_test)

if(ROCM_INTERFACES_BUILD_REAL_ROCBLAS_PROVIDER)
  find_library(ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY
    NAMES rocblas
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64)

  add_library(rocm_interfaces_incomplete_rocblas_backend SHARED
    incomplete_rocblas_backend.cpp)
  target_include_directories(rocm_interfaces_incomplete_rocblas_backend PRIVATE
    ${ROCM_INTERFACES_STAGED_INCLUDE})
  target_link_libraries(rocm_interfaces_incomplete_rocblas_backend PRIVATE hip::host)
  target_compile_definitions(rocm_interfaces_incomplete_rocblas_backend PRIVATE
    __HIP_PLATFORM_AMD__=1)
  set_target_properties(rocm_interfaces_incomplete_rocblas_backend PROPERTIES
    OUTPUT_NAME rocblas-incomplete-backend)

  add_executable(rocblas_real_provider_failure_test
    rocblas_real_provider_failure_test.cpp)
  target_include_directories(rocblas_real_provider_failure_test PRIVATE
    ${PROJECT_SOURCE_DIR}/protocols/include
    ${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}
    ${ROCM_INTERFACES_STAGED_INCLUDE})
  target_link_libraries(rocblas_real_provider_failure_test PRIVATE
    ${CMAKE_DL_LIBS}
    hip::host)
  target_compile_definitions(rocblas_real_provider_failure_test PRIVATE
    __HIP_PLATFORM_AMD__=1)
  add_dependencies(rocblas_real_provider_failure_test
    rocm_rocblas_bridge_provider_system
    rocm_interfaces_incomplete_rocblas_backend)
  add_test(NAME rocm_interfaces.real_provider_missing_backend
    COMMAND rocblas_real_provider_failure_test
      $<TARGET_FILE:rocm_rocblas_bridge_provider_system>
      ${CMAKE_CURRENT_BINARY_DIR}/does-not-exist/librocblas.so.5)
  add_test(NAME rocm_interfaces.real_provider_incomplete_backend
    COMMAND rocblas_real_provider_failure_test
      $<TARGET_FILE:rocm_rocblas_bridge_provider_system>
      $<TARGET_FILE:rocm_interfaces_incomplete_rocblas_backend>)

  add_library(rocm_interfaces_narrow_backend_fixture SHARED
    rocblas_narrow_backend_fixture.cpp)
  target_include_directories(rocm_interfaces_narrow_backend_fixture PRIVATE
    ${ROCM_INTERFACES_STAGED_INCLUDE})
  target_link_libraries(rocm_interfaces_narrow_backend_fixture PRIVATE hip::host)
  target_compile_definitions(rocm_interfaces_narrow_backend_fixture PRIVATE
    __HIP_PLATFORM_AMD__=1)
  set_target_properties(rocm_interfaces_narrow_backend_fixture PROPERTIES
    OUTPUT_NAME rocblas-narrow-backend-fixture)

  add_executable(rocblas_narrow_v2_real_test rocblas_narrow_v2_real_test.cpp)
  target_include_directories(rocblas_narrow_v2_real_test PRIVATE
    ${ROCM_INTERFACES_STAGED_INCLUDE})
  target_link_libraries(rocblas_narrow_v2_real_test PRIVATE
    rocblas_narrow_v2_loader_shadow
    hip::host)
  target_compile_definitions(rocblas_narrow_v2_real_test PRIVATE
    __HIP_PLATFORM_AMD__=1
    REAL_NARROW_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_narrow_v2_provider_system>"
    NARROW_BACKEND_PATH="$<TARGET_FILE:rocm_interfaces_narrow_backend_fixture>")
  add_dependencies(rocblas_narrow_v2_real_test
    rocm_blas_narrow_v2_provider_system
    rocm_interfaces_narrow_backend_fixture)
  add_test(NAME rocm_interfaces.rocblas_narrow_v2_real_vector_transform
    COMMAND rocblas_narrow_v2_real_test)

  if(ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY)
    add_executable(rocblas_real_provider_test rocblas_real_provider_test.cpp)
    target_include_directories(rocblas_real_provider_test PRIVATE
      ${PROJECT_SOURCE_DIR}/protocols/include
      ${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}
      ${ROCM_INTERFACES_STAGED_INCLUDE})
    target_link_libraries(rocblas_real_provider_test PRIVATE
      rocblas_loader_shadow
      ${CMAKE_DL_LIBS}
      hip::host)
    target_compile_definitions(rocblas_real_provider_test PRIVATE
      __HIP_PLATFORM_AMD__=1
      REAL_PROVIDER_PATH="$<TARGET_FILE:rocm_rocblas_bridge_provider_system>"
      REAL_NARROW_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_narrow_v2_provider_system>"
      REAL_ROCBLAS_PATH="${ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY}")
    add_dependencies(rocblas_real_provider_test
      rocm_rocblas_bridge_provider_system
      rocm_blas_narrow_v2_provider_system)
    add_test(NAME rocm_interfaces.rocblas_real_provider_differential
      COMMAND rocblas_real_provider_test)

    find_package(Threads REQUIRED)
    add_executable(rocblas_real_provider_concurrency_test
      rocblas_real_provider_concurrency_test.cpp)
    target_include_directories(rocblas_real_provider_concurrency_test PRIVATE
      ${ROCM_INTERFACES_STAGED_INCLUDE})
    target_link_libraries(rocblas_real_provider_concurrency_test PRIVATE
      rocblas_loader_shadow
      Threads::Threads
      hip::host)
    target_compile_definitions(rocblas_real_provider_concurrency_test PRIVATE
      __HIP_PLATFORM_AMD__=1
      REAL_PROVIDER_PATH="$<TARGET_FILE:rocm_rocblas_bridge_provider_system>"
      REAL_ROCBLAS_PATH="${ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY}")
    add_dependencies(rocblas_real_provider_concurrency_test
      rocm_rocblas_bridge_provider_system)
    add_test(NAME rocm_interfaces.rocblas_real_provider_concurrency
      COMMAND rocblas_real_provider_concurrency_test)

    if(ROCM_INTERFACES_ENABLE_GPU_TESTS)
      add_executable(rocblas_gpu_differential_test rocblas_gpu_differential_test.cpp)
      target_include_directories(rocblas_gpu_differential_test PRIVATE
        ${ROCM_INTERFACES_STAGED_INCLUDE})
      target_link_libraries(rocblas_gpu_differential_test PRIVATE
        ${CMAKE_DL_LIBS}
        hip::host)
      target_compile_definitions(rocblas_gpu_differential_test PRIVATE
        __HIP_PLATFORM_AMD__=1
        REAL_ROCBLAS_PATH="${ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY}"
        REAL_PROVIDER_PATH="$<TARGET_FILE:rocm_rocblas_bridge_provider_system>"
        REAL_NARROW_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_narrow_v2_provider_system>"
        EXHAUSTIVE_LOADER_PATH="$<TARGET_FILE:rocblas_loader_shadow>"
        NARROW_LOADER_PATH="$<TARGET_FILE:rocblas_narrow_v2_loader_shadow>")
      add_dependencies(rocblas_gpu_differential_test
        rocm_rocblas_bridge_provider_system
        rocm_blas_narrow_v2_provider_system
        rocblas_loader_shadow
        rocblas_narrow_v2_loader_shadow)
      add_test(NAME rocm_interfaces.rocblas_gpu_differential
        COMMAND rocblas_gpu_differential_test)
      set_tests_properties(rocm_interfaces.rocblas_gpu_differential PROPERTIES
        LABELS gpu
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        TIMEOUT 300)
    endif()
  else()
    message(WARNING
      "Canonical rocBLAS DSO not found: real-provider differential, installed-manifest "
      "runtime, and concurrency checks will not be registered")
  endif()
endif()

add_executable(rocblas_bridge_link_test
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_link_test.cpp")
add_dependencies(rocblas_bridge_link_test rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocblas_bridge_link_test PRIVATE
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_link_libraries(rocblas_bridge_link_test PRIVATE rocblas_loader_shadow hip::host)
target_compile_definitions(rocblas_bridge_link_test PRIVATE __HIP_PLATFORM_AMD__=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rocblas_bridge_link_test PRIVATE -Wno-deprecated-declarations)
endif()
add_test(NAME rocm_interfaces.rocblas_all_symbols COMMAND rocblas_bridge_link_test)

add_executable(rocblas_narrow_v2_link_test
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_link_test.cpp")
add_dependencies(rocblas_narrow_v2_link_test rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocblas_narrow_v2_link_test PRIVATE ${ROCM_INTERFACES_STAGED_INCLUDE})
target_link_libraries(rocblas_narrow_v2_link_test PRIVATE rocblas_narrow_v2_loader_shadow hip::host)
target_compile_definitions(rocblas_narrow_v2_link_test PRIVATE __HIP_PLATFORM_AMD__=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rocblas_narrow_v2_link_test PRIVATE -Wno-deprecated-declarations)
endif()
add_test(NAME rocm_interfaces.rocblas_narrow_v2_all_symbols COMMAND rocblas_narrow_v2_link_test)

add_executable(rocm_blas_narrow_c_header_test narrow_header_c_test.c)
target_include_directories(rocm_blas_narrow_c_header_test PRIVATE
  ${PROJECT_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_link_libraries(rocm_blas_narrow_c_header_test PRIVATE hip::host)
target_compile_definitions(rocm_blas_narrow_c_header_test PRIVATE __HIP_PLATFORM_AMD__=1)
add_test(NAME rocm_interfaces.blas_narrow_c_header
  COMMAND rocm_blas_narrow_c_header_test)

if(UNIX AND NOT APPLE)
  get_property(_exports_providers GLOBAL PROPERTY ROCM_INTERFACES_PROVIDER_TARGETS)
  set(_exports_provider_args)
  set(_exports_provider_n 0)
  foreach(_prov IN LISTS _exports_providers)
    list(APPEND _exports_provider_args
      "-DPROVIDER_LABEL_${_exports_provider_n}=${_prov}"
      "-DPROVIDER_FILE_${_exports_provider_n}=$<TARGET_FILE:${_prov}>")
    math(EXPR _exports_provider_n "${_exports_provider_n} + 1")
  endforeach()

  add_test(NAME rocm_interfaces.exports
    COMMAND ${CMAKE_COMMAND}
      -DLOADER=$<TARGET_FILE:rocblas_loader_shadow>
      -DNARROW_V2_LOADER=$<TARGET_FILE:rocblas_narrow_v2_loader_shadow>
      -DNARROW_LOADER=$<TARGET_FILE:rocblas_narrow_loader_shadow>
      -DEXPECTED_LOADER_EXPORTS=${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.exports
      -DPROVIDER_COUNT=${_exports_provider_n}
      ${_exports_provider_args}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake)

  # Completeness cross-check: recursively enumerate every MODULE target under
  # providers/ and require the independent ground truth to equal the registry
  # used by the exports check. A provider in a new subdirectory therefore cannot
  # silently escape inspection.
  function(_collect_provider_modules directory output)
    get_directory_property(_local_targets DIRECTORY "${directory}" BUILDSYSTEM_TARGETS)
    set(_result)
    foreach(_target IN LISTS _local_targets)
      get_target_property(_type ${_target} TYPE)
      if(_type STREQUAL "MODULE_LIBRARY")
        list(APPEND _result ${_target})
      endif()
    endforeach()
    get_directory_property(_subdirectories DIRECTORY "${directory}" SUBDIRECTORIES)
    foreach(_subdirectory IN LISTS _subdirectories)
      _collect_provider_modules("${_subdirectory}" _nested)
      list(APPEND _result ${_nested})
    endforeach()
    set(${output} ${_result} PARENT_SCOPE)
  endfunction()
  _collect_provider_modules("${PROJECT_SOURCE_DIR}" _prov_modules)
  list(SORT _prov_modules)
  set(_derived_sorted ${_exports_providers})
  list(SORT _derived_sorted)
  string(REPLACE ";" "\n" _prov_modules_txt "${_prov_modules}")
  string(REPLACE ";" "\n" _derived_txt "${_derived_sorted}")
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/exports_provider_groundtruth.txt
    CONTENT "${_prov_modules_txt}\n")
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/exports_provider_derived.txt
    CONTENT "${_derived_txt}\n")
  add_test(NAME rocm_interfaces.exports_provider_list_complete
    COMMAND ${CMAKE_COMMAND} -E compare_files
      ${CMAKE_CURRENT_BINARY_DIR}/exports_provider_groundtruth.txt
      ${CMAKE_CURRENT_BINARY_DIR}/exports_provider_derived.txt)
endif()

add_test(NAME rocm_interfaces.install_consumer
  COMMAND ${CMAKE_COMMAND}
    -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/install-consumer-test
    -DPROJECT_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer
    -DDEPENDENCY_PREFIX=${CMAKE_PREFIX_PATH}
    -DINSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DREAL_ROCBLAS_LIBRARY=${ROCM_INTERFACES_SYSTEM_ROCBLAS_LIBRARY}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_install.cmake)

if(PROJECT_IS_TOP_LEVEL)
  add_test(NAME rocm_interfaces.root_opt_in_build
    COMMAND ${CMAKE_COMMAND}
      -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/root-integration-test
      -DSOURCE_ROOT=${ROCM_INTERFACES_SOURCE_ROOT}
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DDEPENDENCY_PREFIX=${CMAKE_PREFIX_PATH}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_root_integration.cmake)
  set_tests_properties(rocm_interfaces.root_opt_in_build PROPERTIES
    RUN_SERIAL TRUE TIMEOUT 600)
endif()

if(UNIX AND NOT APPLE)
  # Build an ABI fixture with a selected version node, SONAME, and linker.
  function(add_abi03_provider target abi_ver soname version_script)
    add_library(${target} SHARED abi03_fixture_rocblas.cpp)
    target_compile_definitions(${target} PRIVATE ABI_VER=${abi_ver})
    target_link_options(${target} PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${version_script}"
      "LINKER:-soname,${soname}")
    set_target_properties(${target} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/abi03-dso
      PREFIX "" OUTPUT_NAME "${soname}" SUFFIX "")
  endfunction()

  add_abi03_provider(abi03_provA 6 libprovA.so.6 abi03_provA.map)
  add_abi03_provider(abi03_provB 7 libprovB.so.7 abi03_provB.map)
  add_abi03_provider(abi03_anonA 6 libanonA.so abi03_anon.map)
  add_abi03_provider(abi03_anonB 7 libanonB.so abi03_anon.map)

  add_executable(abi03_interpose_test abi03_interpose_test.cpp)
  target_link_libraries(abi03_interpose_test PRIVATE ${CMAKE_DL_LIBS})
  target_compile_definitions(abi03_interpose_test PRIVATE
    PROV_A_PATH="$<TARGET_FILE:abi03_provA>"
    PROV_B_PATH="$<TARGET_FILE:abi03_provB>"
    ANON_A_PATH="$<TARGET_FILE:abi03_anonA>"
    ANON_B_PATH="$<TARGET_FILE:abi03_anonB>")
  add_dependencies(abi03_interpose_test
    abi03_provA abi03_provB abi03_anonA abi03_anonB)

  add_test(NAME rocm_interfaces.abi03_coresidency
    COMMAND abi03_interpose_test positive)
  add_test(NAME rocm_interfaces.abi03_interpose_hazard
    COMMAND abi03_interpose_test negative)
  # Boundary of the node defense: with genuine version nodes present (A ABI_6,
  # B ABI_7), a version-aware dlvsym still reaches ABI_7, but a bare unversioned
  # dlsym(RTLD_DEFAULT) still takes the first-loaded ABI_6 - so the defense is
  # scoped to versioned relocations/dlvsym, not bare global lookups. Discriminates
  # from abi03_interpose_hazard (which uses nodeless DSOs) on lookup form alone.
  add_test(NAME rocm_interfaces.abi03_versioned_bare_lookup_uncovered
    COMMAND abi03_interpose_test versioned_bare)

  # Linked-consumer interposition proof: a consumer whose relocation carries a
  # versioned undefined reference (rocblas_sgemm@ROCBLAS_ABI_7) binds to the
  # matching major even though an ABI_6 provider is NEEDED first and therefore
  # earlier in the search scope. The plain control shares the identical link
  # line and differs only by the single .symver directive: it is interposed by
  # the first-NEEDED ABI_6 provider. One directive flips the bound major 7<->6,
  # so the pair is discriminating, not vacuous.
  add_executable(abi03_linked_consumer_versioned
    abi03_linked_consumer_versioned.cpp)
  target_link_libraries(abi03_linked_consumer_versioned
    PRIVATE abi03_provA abi03_provB)

  add_executable(abi03_linked_consumer_plain
    abi03_linked_consumer_plain.cpp)
  target_link_libraries(abi03_linked_consumer_plain
    PRIVATE abi03_provA abi03_provB)

  add_test(NAME rocm_interfaces.abi03_linked_consumer_versioned_binds
    COMMAND abi03_linked_consumer_versioned)
  add_test(NAME rocm_interfaces.abi03_linked_consumer_plain_interposed
    COMMAND abi03_linked_consumer_plain)
endif()

if(UNIX AND NOT APPLE)
  # Build a rocBLAS-shaped ABI fixture for the version-resolution matrix.
  function(add_abi04_provider target abi_ver soname version_script bsymbolic)
    set(linker "${ARGV5}")
    set(outdir "${ARGV6}")
    if(outdir STREQUAL "")
      set(outdir ${CMAKE_CURRENT_BINARY_DIR}/abi04-dso)
    endif()
    add_library(${target} SHARED abi03_fixture_rocblas.cpp)
    target_compile_definitions(${target} PRIVATE ABI_VER=${abi_ver})
    target_link_options(${target} PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${version_script}"
      "LINKER:-soname,${soname}")
    if(bsymbolic)
      target_link_options(${target} PRIVATE "LINKER:-Bsymbolic")
    endif()
    if(NOT linker STREQUAL "")
      target_link_options(${target} PRIVATE "-fuse-ld=${linker}")
    endif()
    set_target_properties(${target} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${outdir}
      PREFIX "" OUTPUT_NAME "${soname}" SUFFIX "")
  endfunction()

  add_abi04_provider(abi04_rb5 5 librocblas.so.5 abi04_rb5.map OFF)
  add_abi04_provider(abi04_rb6 6 librocblas.so.6 abi03_provA.map OFF)
  add_abi04_provider(abi04_rb7 7 librocblas.so.7 abi03_provB.map OFF)
  add_abi04_provider(abi04_rb6_bsym 6 librocblas_bsym.so.6 abi03_provA.map ON)
  add_abi04_provider(abi04_rb7_bsym 7 librocblas_bsym.so.7 abi03_provB.map ON)
  add_abi04_provider(abi04_rb5_n6 5 librocblas_n6.so.5 abi03_provA.map OFF)
  add_abi04_provider(abi04_rb6_n6 6 librocblas_n6.so.6 abi03_provA.map OFF)
  add_abi04_provider(abi04_rb7_n6 7 librocblas_n6.so.7 abi03_provA.map OFF)

  add_executable(abi04_versioning_test abi04_versioning_test.cpp)
  target_link_libraries(abi04_versioning_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(abi04_versioning_test
    abi04_rb5 abi04_rb6 abi04_rb7 abi04_rb6_bsym abi04_rb7_bsym
    abi04_rb5_n6 abi04_rb6_n6 abi04_rb7_n6)

  add_test(NAME rocm_interfaces.abi04_three_line_order
    COMMAND abi04_versioning_test three_line
      $<TARGET_FILE:abi04_rb5> $<TARGET_FILE:abi04_rb6> $<TARGET_FILE:abi04_rb7>)
  add_test(NAME rocm_interfaces.abi04_same_node_negative
    COMMAND abi04_versioning_test same_node
      $<TARGET_FILE:abi04_rb5_n6> $<TARGET_FILE:abi04_rb6_n6> $<TARGET_FILE:abi04_rb7_n6>)
  add_test(NAME rocm_interfaces.abi04_bsymbolic_inert
    COMMAND abi04_versioning_test bsymbolic
      $<TARGET_FILE:abi04_rb6_bsym> $<TARGET_FILE:abi04_rb7_bsym> $<TARGET_FILE:abi04_rb6>)

  add_test(NAME rocm_interfaces.abi04_multiple_default_def_rejected
    COMMAND ${CMAKE_COMMAND}
      -DCXX=${CMAKE_CXX_COMPILER}
      -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-dupdef
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_multiple_default_def.cmake)

  find_program(ROCM_INTERFACES_LDCONFIG ldconfig PATHS /sbin /usr/sbin)
  if(ROCM_INTERFACES_LDCONFIG)
    add_test(NAME rocm_interfaces.abi04_ldconfig_stub_preserved
      COMMAND ${CMAKE_COMMAND}
        -DLDCONFIG=${ROCM_INTERFACES_LDCONFIG}
        -DDSO_DIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-dso
        -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-ldconfig
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_ldconfig_stub.cmake)
  else()
    message(STATUS
      "abi04_ldconfig_stub: ldconfig not found, ldconfig-stub test not registered")
  endif()

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LINK_OPTIONS -fuse-ld=lld)
  check_cxx_source_compiles("int main(){return 0;}" ROCM_INTERFACES_HAVE_LLD)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)

  if(ROCM_INTERFACES_HAVE_LLD)
    set(_abi04_lld_dir ${CMAKE_CURRENT_BINARY_DIR}/abi04-dso-lld)
    add_abi04_provider(abi04_rb5_lld 5 librocblas.so.5 abi04_rb5.map OFF lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb6_lld 6 librocblas.so.6 abi03_provA.map OFF lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb7_lld 7 librocblas.so.7 abi03_provB.map OFF lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb6_bsym_lld 6 librocblas_bsym.so.6 abi03_provA.map ON lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb7_bsym_lld 7 librocblas_bsym.so.7 abi03_provB.map ON lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb5_n6_lld 5 librocblas_n6.so.5 abi03_provA.map OFF lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb6_n6_lld 6 librocblas_n6.so.6 abi03_provA.map OFF lld ${_abi04_lld_dir})
    add_abi04_provider(abi04_rb7_n6_lld 7 librocblas_n6.so.7 abi03_provA.map OFF lld ${_abi04_lld_dir})
    add_dependencies(abi04_versioning_test
      abi04_rb5_lld abi04_rb6_lld abi04_rb7_lld abi04_rb6_bsym_lld abi04_rb7_bsym_lld
      abi04_rb5_n6_lld abi04_rb6_n6_lld abi04_rb7_n6_lld)

    add_test(NAME rocm_interfaces.abi04_three_line_order_lld
      COMMAND abi04_versioning_test three_line
        $<TARGET_FILE:abi04_rb5_lld> $<TARGET_FILE:abi04_rb6_lld> $<TARGET_FILE:abi04_rb7_lld>)
    add_test(NAME rocm_interfaces.abi04_same_node_negative_lld
      COMMAND abi04_versioning_test same_node
        $<TARGET_FILE:abi04_rb5_n6_lld> $<TARGET_FILE:abi04_rb6_n6_lld> $<TARGET_FILE:abi04_rb7_n6_lld>)
    add_test(NAME rocm_interfaces.abi04_bsymbolic_inert_lld
      COMMAND abi04_versioning_test bsymbolic
        $<TARGET_FILE:abi04_rb6_bsym_lld> $<TARGET_FILE:abi04_rb7_bsym_lld> $<TARGET_FILE:abi04_rb6_lld>)

    add_test(NAME rocm_interfaces.abi04_multiple_default_def_rejected_lld
      COMMAND ${CMAKE_COMMAND}
        -DCXX=${CMAKE_CXX_COMPILER}
        -DLINKER=lld
        -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-dupdef-lld
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_multiple_default_def.cmake)

    if(ROCM_INTERFACES_LDCONFIG)
      add_test(NAME rocm_interfaces.abi04_ldconfig_stub_preserved_lld
        COMMAND ${CMAKE_COMMAND}
          -DLDCONFIG=${ROCM_INTERFACES_LDCONFIG}
          -DDSO_DIR=${_abi04_lld_dir}
          -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-ldconfig-lld
          -P ${CMAKE_CURRENT_SOURCE_DIR}/check_ldconfig_stub.cmake)
    endif()
  else()
    message(STATUS
      "abi04 lld matrix: -fuse-ld=lld unavailable, lld variants not registered")
  endif()
endif()

if(UNIX AND NOT APPLE)
  add_library(abi06_data_rb6 SHARED
    abi03_fixture_rocblas.cpp
    ${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocrand/library/src/rocrand_sobol32_precomputed.cpp)
  target_compile_definitions(abi06_data_rb6 PRIVATE ABI_VER=6 __HIP_PLATFORM_AMD__=1)
  target_include_directories(abi06_data_rb6 PRIVATE
    ${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocrand/library/include
    /opt/rocm/include)
  target_link_options(abi06_data_rb6 PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/abi06_data.map"
    "LINKER:-soname,librocblas.so.6")
  set_target_properties(abi06_data_rb6 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/abi06-dso
    PREFIX "" OUTPUT_NAME "librocblas.so.6" SUFFIX "")

  add_executable(abi06_data_test abi06_data_test.cpp)
  target_link_libraries(abi06_data_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(abi06_data_test abi06_data_rb6)

  add_test(NAME rocm_interfaces.abi06_data_version_node
    COMMAND abi06_data_test $<TARGET_FILE:abi06_data_rb6>)
endif()

if(UNIX AND NOT APPLE)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS -fsanitize=address)
  set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=address)
  check_cxx_source_compiles("int main(){return 0;}" ROCM_INTERFACES_HAVE_ASAN)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)

  if(ROCM_INTERFACES_HAVE_ASAN)
    add_test(NAME rocm_interfaces.abi04_asan_version_node_survives
      COMMAND ${CMAKE_COMMAND}
        -DCXX=${CMAKE_CXX_COMPILER}
        -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/abi03_fixture_rocblas.cpp
        -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi04-asan
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_asan_version_node.cmake)
  else()
    message(STATUS
      "abi04 asan: -fsanitize=address unavailable with the configured compiler, "
      "asan version-node test not registered")
  endif()
endif()

if(UNIX AND NOT APPLE)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-D__HIP_PLATFORM_AMD__")
  set(CMAKE_REQUIRED_INCLUDES "${ROCM_INTERFACES_STAGED_INCLUDE}" /opt/rocm/include)
  check_cxx_source_compiles("
#include <cstdio>
#include <rocrand/rocrand.hpp>
int main(){ rocrand_cpp::error e(ROCRAND_STATUS_SUCCESS); return e.what()==nullptr; }
" ROCM_INTERFACES_HAVE_ROCRAND_CPP)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_INCLUDES)

  if(ROCM_INTERFACES_HAVE_ROCRAND_CPP)
    add_test(NAME rocm_interfaces.abi05_cpp_mangled_version_node
      COMMAND ${CMAKE_COMMAND}
        -DCXX=${CMAKE_CXX_COMPILER}
        -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/abi05_fixture_rocrand_cpp.cpp
        -DMAP=${CMAKE_CURRENT_SOURCE_DIR}/abi05_rocrand_cpp.map
        -DROCRAND_INC=${ROCM_INTERFACES_STAGED_INCLUDE}
        -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi05-cpp-mangled
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_cpp_mangled_version.cmake)

    if(ROCM_INTERFACES_HAVE_LLD)
      add_test(NAME rocm_interfaces.abi05_cpp_mangled_version_node_lld
        COMMAND ${CMAKE_COMMAND}
          -DCXX=${CMAKE_CXX_COMPILER}
          -DLINKER=lld
          -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/abi05_fixture_rocrand_cpp.cpp
          -DMAP=${CMAKE_CURRENT_SOURCE_DIR}/abi05_rocrand_cpp.map
          -DROCRAND_INC=${ROCM_INTERFACES_STAGED_INCLUDE}
          -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}/abi05-cpp-mangled-lld
          -P ${CMAKE_CURRENT_SOURCE_DIR}/check_cpp_mangled_version.cmake)
    endif()
  else()
    message(STATUS
      "abi05 cpp mangled: rocrand.hpp not host-compilable with the configured "
      "compiler, C++ mangled-export version-node test not registered")
  endif()
endif()

set(_lto_guard_module ${PROJECT_SOURCE_DIR}/cmake/rocm_interfaces_lto_linker_guard.cmake)
set(_lto_guard_driver ${CMAKE_CURRENT_SOURCE_DIR}/check_lto_linker_guard.cmake)

add_test(NAME rocm_interfaces.lto_linker_guard_rejects_gnu_lld
  COMMAND ${CMAKE_COMMAND} -DCASE=gnu_lld_lto -DGUARD_MODULE=${_lto_guard_module}
    -P ${_lto_guard_driver})
set_tests_properties(rocm_interfaces.lto_linker_guard_rejects_gnu_lld
  PROPERTIES WILL_FAIL TRUE)

add_test(NAME rocm_interfaces.lto_linker_guard_accepts_clang_lld
  COMMAND ${CMAKE_COMMAND} -DCASE=clang_lld_lto -DGUARD_MODULE=${_lto_guard_module}
    -P ${_lto_guard_driver})

add_test(NAME rocm_interfaces.lto_linker_guard_accepts_gnu_bfd
  COMMAND ${CMAKE_COMMAND} -DCASE=gnu_bfd_lto -DGUARD_MODULE=${_lto_guard_module}
    -P ${_lto_guard_driver})

add_test(NAME rocm_interfaces.lto_linker_guard_accepts_gnu_lld_without_lto
  COMMAND ${CMAKE_COMMAND} -DCASE=gnu_lld_nolto -DGUARD_MODULE=${_lto_guard_module}
    -P ${_lto_guard_driver})

if(UNIX AND NOT APPLE AND ROCM_INTERFACES_SANITIZE STREQUAL "thread")
  find_package(Threads REQUIRED)
  add_executable(ops04_concurrency_test ops04_concurrency_test.cpp)
  target_link_libraries(ops04_concurrency_test
    PRIVATE rocm_interfaces_loader rocblas_loader_shadow Threads::Threads)
  target_compile_definitions(ops04_concurrency_test PRIVATE
    BLAS_CLASSIC_PROVIDER_PATH="$<TARGET_FILE:rocm_blas_classic_provider_recording>"
    ROCBLAS_BRIDGE_PROVIDER_PATH="$<TARGET_FILE:rocm_rocblas_bridge_provider_recording>")
  add_dependencies(ops04_concurrency_test
    rocm_blas_classic_provider_recording
    rocm_rocblas_bridge_provider_recording
    rocblas_loader_shadow)
  add_test(NAME rocm_interfaces.ops04_concurrency COMMAND ops04_concurrency_test)
  set_tests_properties(rocm_interfaces.ops04_concurrency PROPERTIES
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=0:exitcode=66")
endif()

if(ROCM_INTERFACES_CHECK_API_DRIFT)
  if(TARGET rocm-interfaces-check-api-snapshots)
    add_test(NAME rocm_interfaces.api_snapshot_drift
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
        --target rocm-interfaces-check-api-snapshots --config $<CONFIG>)
    set_tests_properties(rocm_interfaces.api_snapshot_drift PROPERTIES
      RUN_SERIAL TRUE TIMEOUT 600)
  else()
    message(FATAL_ERROR
      "ROCM_INTERFACES_CHECK_API_DRIFT=ON but target "
      "rocm-interfaces-check-api-snapshots does not exist. It requires "
      "ROCM_INTERFACES_BUILD_TOOLS=ON and a usable Clang (rocm-api-extract). "
      "Enable the tools or turn the drift gate off; the gate refuses to "
      "register as a silent no-op.")
  endif()
endif()
