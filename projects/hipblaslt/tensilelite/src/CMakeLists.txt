# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(TENSILELITE_ENABLE_YAML)
    add_subdirectory(llvm)
else()
    add_subdirectory(msgpack)
endif()

add_subdirectory(hip)

target_sources(tensilelite-host
    PRIVATE
        ${TENSILELITE_HOST_SOURCES}
)
