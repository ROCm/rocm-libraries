# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(HIPSPARSELT_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(hipsparselt SHARED)
else()
    add_library(hipsparselt STATIC)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_STATIC_LIB)
endif()

add_library(roc::hipsparselt ALIAS hipsparselt)
rocm_set_soversion(hipsparselt ${hipsparselt_SOVERSION})

target_compile_features(hipsparselt PRIVATE cxx_std_17)
target_compile_features(hipsparselt PRIVATE cxx_nullptr)

if(HIPSPARSELT_ENABLE_CUDA)
    target_compile_definitions(hipsparselt PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

if(HIPSPARSELT_ENABLE_HIP)
    target_compile_definitions(
        hipsparselt
        PRIVATE __HIP_PLATFORM_AMD__
                # NOTE: There are larger conversations about whether ROCM_USE_FLOAT16 should be set
                # at the pre-processor level instead
                ROCM_USE_FLOAT16
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Enable F16C intrinsics
    target_compile_options(hipsparselt PRIVATE -mf16c)
endif()

if(HIPSPARSELT_ENABLE_ASAN)
    target_compile_options(hipsparselt PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsparselt PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

if(HIPSPARSELT_ENABLE_MARKER)
    target_compile_definitions(hipsparselt PRIVATE HIPSPARSELT_ENABLE_MARKER)
    target_link_libraries(hipsparselt PRIVATE ${ROCTX64_LIBRARY})
else()
    target_compile_definitions(hipsparselt PRIVATE DISABLE_ROCTX)
endif()

set_target_properties(
    hipsparselt PROPERTIES CXX_EXTENSIONS OFF VISIBILITY_INLINES_HIDDEN ON CXX_VISIBILITY_PRESET
                                                                           "hidden"
)

target_include_directories(
    hipsparselt
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(HIPSPARSELT_ENABLE_CUDA)
    target_include_directories(
        hipsparselt PRIVATE $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
                            $<BUILD_INTERFACE:${cusparselt_INCLUDE_DIR}>
    )
    target_link_libraries(hipsparselt PRIVATE ${cusparselt_LIBRARY} CUDA::cusparse CUDA::cudart)
endif()

if(HIPSPARSELT_ENABLE_HIP)
    target_compile_definitions(hipsparselt PRIVATE BUILD_WITH_TENSILE)
    target_compile_definitions(hipsparselt PRIVATE __HIP_HCC_COMPAT_MODE__=1)
    target_include_directories(
        hipsparselt PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}> # Needed?
                           # Don't link against hipsparse, only include the headers. Otherwise a
                           # transitive dependency on rocsparse will be introduced.
                           $<BUILD_INTERFACE:${HIPSPARSE_INCLUDE_DIRS}>
    )
    target_link_libraries(hipsparselt PRIVATE hip::host hip::device)
endif()

add_subdirectory(include)
add_subdirectory(src)
