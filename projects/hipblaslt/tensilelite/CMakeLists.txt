# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa/rocisa-cpp)

add_library(tensilelite-host OBJECT)
add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)

set_target_properties(tensilelite-host PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(tensilelite-host PUBLIC cxx_std_17)
target_compile_definitions(tensilelite-host
    PUBLIC
        $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
        __HIP_PLATFORM_AMD__
)

if(HIPBLASLT_ENABLE_HIP)
    target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_USE_HIP>)
endif()

if(HIPBLASLT_ENABLE_LLVM)
    target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
    target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(tensilelite-host PUBLIC TENSILE_YAML)
endif()

if(HIPBLASLT_ENABLE_MSGPACK)
    if(msgpack-cxx_FOUND)
        target_link_libraries(tensilelite-host PRIVATE msgpack-cxx)
    else()
        target_link_libraries(tensilelite-host PRIVATE msgpackc-cxx)
    endif()
    target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>) # make TENSILELITE_MSGPACK
endif()

target_include_directories(tensilelite-host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/include"
)

target_link_libraries(tensilelite-host 
    PUBLIC
        rocisa::rocisa-cpp
        hip::device
)
add_subdirectory(src)
add_subdirectory(include)

if(TENSILELITE_ENABLE_CLIENT)
    add_subdirectory(client)
endif()

if(HIPBLASLT_BUILD_TESTING OR BUILD_TESTING)
    add_subdirectory(tests)
endif()
