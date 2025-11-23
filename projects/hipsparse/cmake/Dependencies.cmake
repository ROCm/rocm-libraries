# # Copyright Advanced Micro Devices, Inc., or its affiliates.
# # SPDX-License-Identifier: MIT

# include(FetchContent)

# # Determine HIP path
# if(NOT DEFINED ENV{HIP_PATH})
#     if(WIN32)
#         set(HIP_PATH "C:/hip")
#     else()
#         set(HIP_PATH "/opt/rocm")
#     endif()
# else()
#     file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_PATH)
# endif()

# # Either rocSPARSE or cuSPARSE is required
# if(NOT HIPSPARSE_ENABLE_CUDA)
#     if(WIN32)
#         find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH})
#         if(CUSTOM_ROCSPARSE)
#             set(ENV{rocsparse_DIR} ${CUSTOM_ROCSPARSE})
#             find_package(rocsparse REQUIRED CONFIG NO_CMAKE_PATH)
#         else()
#             find_package(rocsparse 4.0.1 REQUIRED CONFIG PATHS ${ROCSPARSE_PATH})
#         endif()
#     else()
#         find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)
#         find_package(rocsparse 4.0.1 REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocsparse /usr/local/rocsparse)
#     endif()
# else()
#     set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
#     find_package(HIP MODULE REQUIRED)
#     list(APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
#     find_package(CUDA REQUIRED)
# endif()
