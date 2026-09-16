add_library(rocm_rocblas_bridge_provider_system MODULE
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_forwarding_provider.cpp")
add_dependencies(rocm_rocblas_bridge_provider_system
  rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocm_rocblas_bridge_provider_system PRIVATE
  ${PROJECT_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_compile_definitions(rocm_rocblas_bridge_provider_system PRIVATE
  __HIP_PLATFORM_AMD__=1
  "ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME=\"${ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME}\"")
target_link_libraries(rocm_rocblas_bridge_provider_system PRIVATE
  ${CMAKE_DL_LIBS}
  hip::host)
set_target_properties(rocm_rocblas_bridge_provider_system PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME rocblas-provider-system
  PREFIX "lib")
target_link_options(rocm_rocblas_bridge_provider_system PRIVATE
  "LINKER:--version-script=${PROJECT_SOURCE_DIR}/providers/provider.map")
set_property(GLOBAL APPEND PROPERTY ROCM_INTERFACES_PROVIDER_TARGETS
  rocm_rocblas_bridge_provider_system)

add_library(rocm_blas_narrow_v2_provider_system MODULE
  rocblas_narrow_v2_provider.cpp)
target_include_directories(rocm_blas_narrow_v2_provider_system PRIVATE
  ${PROJECT_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_compile_definitions(rocm_blas_narrow_v2_provider_system PRIVATE
  __HIP_PLATFORM_AMD__=1
  "ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME=\"${ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME}\"")
target_link_libraries(rocm_blas_narrow_v2_provider_system PRIVATE
  ${CMAKE_DL_LIBS}
  hip::host)
set_target_properties(rocm_blas_narrow_v2_provider_system PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME rocblas-provider-narrow-v2-system
  PREFIX "lib")
target_link_options(rocm_blas_narrow_v2_provider_system PRIVATE
  "LINKER:--version-script=${PROJECT_SOURCE_DIR}/providers/provider.map")
set_property(GLOBAL APPEND PROPERTY ROCM_INTERFACES_PROVIDER_TARGETS
  rocm_blas_narrow_v2_provider_system)

set(_provider_install_dir "${CMAKE_INSTALL_LIBDIR}/rocm/interfaces/providers")
set(_bridge_manifest "${CMAKE_CURRENT_BINARY_DIR}/rocblas-system.json")
set(_narrow_manifest "${CMAKE_CURRENT_BINARY_DIR}/rocblas-narrow-v2-system.json")
file(GENERATE OUTPUT "${_bridge_manifest}" CONTENT
  "{\n  \"schema_version\": 1,\n  \"providers\": [\n    {\n      \"id\": \"system-rocblas-bridge\",\n      \"domain\": \"rocblas_bridge\",\n      \"module\": \"$<TARGET_FILE_NAME:rocm_rocblas_bridge_provider_system>\",\n      \"cohort\": \"system-rocblas\",\n      \"gfx\": [0],\n      \"priority\": 0\n    }\n  ]\n}\n")
file(GENERATE OUTPUT "${_narrow_manifest}" CONTENT
  "{\n  \"schema_version\": 1,\n  \"providers\": [\n    {\n      \"id\": \"system-rocblas-narrow-v2\",\n      \"domain\": \"blas_v2\",\n      \"module\": \"$<TARGET_FILE_NAME:rocm_blas_narrow_v2_provider_system>\",\n      \"cohort\": \"system-rocblas\",\n      \"gfx\": [0],\n      \"priority\": 0\n    }\n  ]\n}\n")

install(TARGETS
    rocm_rocblas_bridge_provider_system
    rocm_blas_narrow_v2_provider_system
  LIBRARY DESTINATION "${_provider_install_dir}" COMPONENT rocm-interfaces
  RUNTIME DESTINATION "${_provider_install_dir}" COMPONENT rocm-interfaces)
install(FILES "${_bridge_manifest}" "${_narrow_manifest}"
  DESTINATION "${_provider_install_dir}"
  COMPONENT rocm-interfaces)
