# Add a single-export recording provider and register it for export validation.
function(add_recording_provider target source output_name)
  add_library(${target} MODULE ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/protocols/include
    ${ROCM_INTERFACES_STAGED_INCLUDE}
  )
  target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__=1)
  target_link_libraries(${target} PRIVATE hip::host)
  set_target_properties(${target} PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME ${output_name}
    PREFIX "lib"
  )
  target_link_options(${target} PRIVATE
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/providers/provider.map")
  set_property(GLOBAL APPEND PROPERTY ROCM_INTERFACES_PROVIDER_TARGETS ${target})
endfunction()

add_recording_provider(rocm_blas_provider_recording blas_provider.cpp
  rocm-blas-provider-recording)
add_recording_provider(rocm_blas_classic_provider_recording blas_classic_provider.cpp
  rocblas-provider-tensile-recording)
add_recording_provider(rocm_blaslt_provider_recording blaslt_provider.cpp
  hipblaslt-provider-tensilelite-recording)
add_recording_provider(rocm_solver_provider_recording solver_provider.cpp
  rocm-solver-provider-recording)
add_recording_provider(rocm_rand_provider_recording rand_provider.cpp
  rocm-rand-provider-recording)
add_recording_provider(rocm_blas_narrow_v2_provider_recording blas_narrow_v2_provider.cpp
  rocblas-provider-narrow-v2-recording)

add_library(rocm_rocblas_bridge_provider_recording MODULE
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_provider.cpp")
add_dependencies(rocm_rocblas_bridge_provider_recording
  rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocm_rocblas_bridge_provider_recording PRIVATE
  ${PROJECT_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}
  ${ROCM_INTERFACES_STAGED_INCLUDE})
target_compile_definitions(rocm_rocblas_bridge_provider_recording PRIVATE
  __HIP_PLATFORM_AMD__=1)
target_link_libraries(rocm_rocblas_bridge_provider_recording PRIVATE hip::host)
set_target_properties(rocm_rocblas_bridge_provider_recording PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME rocblas-provider-bruteforce-recording
  PREFIX "lib")
target_link_options(rocm_rocblas_bridge_provider_recording PRIVATE
  "LINKER:--version-script=${PROJECT_SOURCE_DIR}/providers/provider.map")
set_property(GLOBAL APPEND PROPERTY ROCM_INTERFACES_PROVIDER_TARGETS
  rocm_rocblas_bridge_provider_recording)
