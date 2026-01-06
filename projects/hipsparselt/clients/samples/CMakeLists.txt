# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(SAMPLE_EXECUTABLES example_spmm_strided_batched example_prune_strip example_compress)

foreach(sample_name ${SAMPLE_EXECUTABLES})
  add_executable(${sample_name})

  target_compile_features(${sample_name} PRIVATE cxx_std_17)

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${sample_name} PRIVATE -mf16c)
    target_compile_definitions(${sample_name} PRIVATE HIPSPARSELT_INTERNAL_API)
  endif()

  target_sources(${sample_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${sample_name}.cpp")

  target_include_directories(
    ${sample_name} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  )

  target_link_libraries(${sample_name} PRIVATE roc::hipsparselt)

  if(HIPSPARSELT_ENABLE_HIP)
    target_compile_definitions(${sample_name} PRIVATE __HIP_PLATFORM_AMD__ ROCM_USE_FLOAT16)
    target_link_libraries(${sample_name} PRIVATE hip::host hip::device)
  elseif(HIPSPARSELT_ENABLE_CUDA)
    target_compile_definitions(${sample_name} PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(
      ${sample_name} PRIVATE $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
    )
    target_link_libraries(
      ${sample_name} PRIVATE ${cusparselt_LIBRARY} CUDA::cudart CUDA::cuda_driver
    )
  endif()
endforeach()
