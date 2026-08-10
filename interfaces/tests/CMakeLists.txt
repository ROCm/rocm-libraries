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
  ${CMAKE_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_link_libraries(rocm_blas_narrow_c_header_test PRIVATE hip::host)
target_compile_definitions(rocm_blas_narrow_c_header_test PRIVATE __HIP_PLATFORM_AMD__=1)
add_test(NAME rocm_interfaces.blas_narrow_c_header
  COMMAND rocm_blas_narrow_c_header_test)

if(UNIX AND NOT APPLE)
  add_test(NAME rocm_interfaces.exports
    COMMAND ${CMAKE_COMMAND}
      -DLOADER=$<TARGET_FILE:rocblas_loader_shadow>
      -DNARROW_V2_LOADER=$<TARGET_FILE:rocblas_narrow_v2_loader_shadow>
      -DEXPECTED_LOADER_EXPORTS=${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.exports
      -DBLAS_PROVIDER=$<TARGET_FILE:rocm_blas_classic_provider_recording>
      -DCOMBINED_BLAS_PROVIDER=$<TARGET_FILE:rocm_blas_provider_recording>
      -DBLASLT_PROVIDER=$<TARGET_FILE:rocm_blaslt_provider_recording>
      -DSOLVER_PROVIDER=$<TARGET_FILE:rocm_solver_provider_recording>
      -DRAND_PROVIDER=$<TARGET_FILE:rocm_rand_provider_recording>
      -DROCBLAS_BRIDGE_PROVIDER=$<TARGET_FILE:rocm_rocblas_bridge_provider_recording>
      -DBLAS_NARROW_V2_PROVIDER=$<TARGET_FILE:rocm_blas_narrow_v2_provider_recording>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake)
endif()

add_test(NAME rocm_interfaces.install_consumer
  COMMAND ${CMAKE_COMMAND}
    -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/install-consumer-test
    -DPROJECT_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer
    -DDEPENDENCY_PREFIX=${CMAKE_PREFIX_PATH}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_install.cmake)

if(UNIX AND NOT APPLE)
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
endif()
