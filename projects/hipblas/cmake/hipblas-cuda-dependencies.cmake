# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(CMakeFindDependencyMacro)

# Load hip::host for the NVIDIA backend if the caller has not already done so.
# Guard against accidentally using an AMD hip::host with a CUDA-built hipBLAS,
# which would silently select the wrong HIP header backend for the consumer.
if(TARGET hip::host)
  get_target_property(_hipblas_hip_defs hip::host INTERFACE_COMPILE_DEFINITIONS)
  if("${_hipblas_hip_defs}" MATCHES "__HIP_PLATFORM_AMD__")
    message(FATAL_ERROR "CUDA-built hipBLAS cannot be used with an AMD hip::host target")
  endif()
  unset(_hipblas_hip_defs)
else()
  find_dependency(hip CONFIG)
endif()
