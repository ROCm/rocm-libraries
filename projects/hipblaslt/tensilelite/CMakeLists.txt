# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    add_library(tensilelite-host OBJECT)
    add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(tensilelite-host PUBLIC -fvisibility=hidden)
    target_compile_features(tensilelite-host PUBLIC cxx_std_17)
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
            $<BUILD_INTERFACE:TENSILE_USE_HIP>
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(HIPBLASLT_ENABLE_LLVM)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC TENSILE_YAML)
    endif()

    if(HIPBLASLT_ENABLE_MSGPACK)
        target_link_libraries(tensilelite-host PRIVATE ${hipblaslt_msgpack_target})
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
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
        PRIVATE
            hip::device
    )

    add_subdirectory(src)
    add_subdirectory(include)

    if(TENSILELITE_ENABLE_CLIENT)
        add_subdirectory(client)
    endif()

    if(TENSILELITE_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
