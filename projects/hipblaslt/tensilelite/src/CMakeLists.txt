# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(TENSILELITE_ENABLE_YAML)
    add_subdirectory(llvm)
endif()

if(NOT TENSILELITE_ENABLE_YAML OR tensilelite_msgpack_target)
    add_subdirectory(msgpack)
endif()

add_subdirectory(hip)

target_sources(tensilelite-host
    PRIVATE
        ${TENSILELITE_HOST_SOURCES}
)

if(HIPBLASLT_ENABLE_LOCAL_KERNELS)
    target_sources(tensilelite-host
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/LocalKernelLibrary.cpp"
    )
endif()
