# Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

IF(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fno-underscoring")
ENDIF(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
file(GLOB HIPFORT_SRC_HIP     "${CMAKE_CURRENT_SOURCE_DIR}/hipfort/*.f*")
#file(GLOB HIPFORT_SRC_amdgcn  "${CMAKE_CURRENT_SOURCE_DIR}/modules-amdgcn/*.f*")
#file(GLOB HIPFORT_SRC_nvptx   "${CMAKE_CURRENT_SOURCE_DIR}/modules-nvptx/*.f*")
file(GLOB HIPFORT_SRC_CONTRIB "${CMAKE_CURRENT_SOURCE_DIR}/modules-contrib/*.f*")

set(HIPFORT_ARCH "amdgcn")
    # amdgcn
    set(HIPFORT_LIB      "hipfort-${HIPFORT_ARCH}")
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort/${HIPFORT_ARCH})
    ADD_LIBRARY(${HIPFORT_LIB} STATIC 
        ${HIPFORT_SRC_HIP} 
        #${HIPFORT_SRC_amdgcn} 
        ${HIPFORT_SRC_CONTRIB} 
        )
    target_include_directories(${HIPFORT_LIB}
      PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipfort/amdgcn>
    )
    IF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    	target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_FPOINTER_INTERFACES)
    ENDIF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE _HIPFORT_ARCH='${HIPFORT_ARCH}')
    # Install Target hipfort-amdgcn
    rocm_install_targets(
        TARGETS ${HIPFORT_LIB}
        EXPORT hipfort-amdgcn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDE
           ${CMAKE_Fortran_MODULE_DIRECTORY}
    )

set(HIPFORT_ARCH "nvptx")
    # nvptx
    set(HIPFORT_LIB     "hipfort-${HIPFORT_ARCH}")
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort/${HIPFORT_ARCH})
    ADD_LIBRARY(${HIPFORT_LIB} STATIC 
         ${HIPFORT_SRC_HIP} 
         #${HIPFORT_SRC_nvptx} 
         ${HIPFORT_SRC_CONTRIB} 
         )
    IF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    	target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_FPOINTER_INTERFACES)
    ENDIF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_CUDA_NAMES)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE _HIPFORT_ARCH='${HIPFORT_ARCH}')
    # Install Target hipfort-nvptx
    rocm_install_targets(
        TARGETS ${HIPFORT_LIB}
        INCLUDE
           ${CMAKE_Fortran_MODULE_DIRECTORY}
    )
   
    # Install include files marking as devel component
    rocm_install(DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort
                 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		 COMPONENT devel)

#   target_link_libraries(${HIPFORT_LIB} PUBLIC 
#   /usr/local/cuda/targets/x86_64-linux/lib/libcudart_static.a)

rocm_install(
  EXPORT hipfort-amdgcn-targets
  FILE hipfort-amdgcn-targets.cmake
  NAMESPACE hipfort::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipfort
)

macro(hipfort_add_component name imported_target)
  add_library(hipfort-${name} INTERFACE)
  add_library(hipfort::${name} ALIAS hipfort-${name})
  set_target_properties(hipfort-${name}
    PROPERTIES
      EXPORT_NAME ${name}
  )
  target_link_libraries(hipfort-${name} INTERFACE hipfort-amdgcn ${imported_target})
  rocm_install(
    TARGETS
      hipfort-${name}
    EXPORT hipfort-${name}-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  rocm_install(
    EXPORT hipfort-${name}-targets
    FILE hipfort-${name}-targets.cmake
    NAMESPACE hipfort::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipfort
  )
endmacro()

find_package(hip PATHS ${ROCM_PATH} /opt/rocm)
if(HIP_PLATFORM STREQUAL "amd")
  if(hip_FOUND)
    hipfort_add_component(hip hip::host)
  else()
    message(STATUS "Skipping hipfort::hip target export")
  endif()

  find_package(rocprofiler-sdk-roctx PATHS ${ROCM_PATH} /opt/rocm)
  if(rocprofiler-sdk-roctx_FOUND)
    hipfort_add_component(roctx rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
  else()
    message(STATUS "Skipping hipfort::roctx target export")
  endif()

  find_package(rocblas PATHS ${ROCM_PATH} /opt/rocm)
  if(rocblas_FOUND)
    hipfort_add_component(rocblas roc::rocblas)
  else()
    message(STATUS "Skipping hipfort::rocblas target export")
  endif()

  find_package(hipblas PATHS ${ROCM_PATH} /opt/rocm)
  if(hipblas_FOUND)
    hipfort_add_component(hipblas roc::hipblas)
  else()
    message(STATUS "Skipping hipfort::hipblas target export")
  endif()

  find_package(rocfft PATHS ${ROCM_PATH} /opt/rocm)
  if(rocfft_FOUND)
    hipfort_add_component(rocfft roc::rocfft)
  else()
    message(STATUS "Skipping hipfort::rocfft target export")
  endif()

  find_package(hipfft PATHS ${ROCM_PATH} /opt/rocm)
  if(hipfft_FOUND)
    hipfort_add_component(hipfft hip::hipfft)
  else()
    message(STATUS "Skipping hipfort::hipfft target export")
  endif()

  find_package(rocrand PATHS ${ROCM_PATH} /opt/rocm)
  if(rocrand_FOUND)
    hipfort_add_component(rocrand roc::rocrand)
  else()
    message(STATUS "Skipping hipfort::rocrand target export")
  endif()

  find_package(hiprand PATHS ${ROCM_PATH} /opt/rocm)
  if(hiprand_FOUND)
    hipfort_add_component(hiprand hip::hiprand)
  else()
    message(STATUS "Skipping hipfort::hiprand target export")
  endif()

  find_package(rocsolver PATHS ${ROCM_PATH} /opt/rocm)
  if(rocsolver_FOUND)
    hipfort_add_component(rocsolver roc::rocsolver)
  else()
    message(STATUS "Skipping hipfort::rocsolver target export")
  endif()

  find_package(hipsolver PATHS ${ROCM_PATH} /opt/rocm)
  if(hipsolver_FOUND)
    hipfort_add_component(hipsolver roc::hipsolver)
  else()
    message(STATUS "Skipping hipfort::hipsolver target export")
  endif()

  find_package(rocsparse PATHS ${ROCM_PATH} /opt/rocm)
  if(rocsparse_FOUND)
    hipfort_add_component(rocsparse roc::rocsparse)
  else()
    message(STATUS "Skipping hipfort::rocsparse target export")
  endif()

  find_package(hipsparse PATHS ${ROCM_PATH} /opt/rocm)
  if(hipsparse_FOUND)
    hipfort_add_component(hipsparse roc::hipsparse)
  else()
    message(STATUS "Skipping hipfort::hipsparse target export")
  endif()

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    hipfort-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hipfort-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipfort
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hipfort-config-version.cmake
    VERSION "${HIPFORT_VERSION_MAJOR}.${HIPFORT_VERSION_MINOR}.${HIPFORT_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )

  rocm_install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/hipfort-config.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/hipfort-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipfort
  )
endif()
