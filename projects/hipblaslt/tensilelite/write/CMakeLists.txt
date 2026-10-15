# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Nanobind module TensileCreateLibrary uses to write MessagePack libraries.
# rocisa fetches nanobind; this target stays separate from _rocisa.

if(NOT COMMAND nanobind_add_module)
    message(FATAL_ERROR "tensilelite_write requires nanobind_add_module (built with rocisa)")
endif()

# Top-level module: PYTHONPATH names this directory and `import tensilelite_write` loads the .so.
set(_tensilelite_write_output "${CMAKE_BINARY_DIR}/tensilelite/python")
set(_tensilelite_write_install "${CMAKE_INSTALL_LIBDIR}/hipblaslt/tensilelite")

set(_link_msgpack_host FALSE)
if(TARGET tensilelite-host AND NOT HIPBLASLT_ENABLE_YAML)
    set(_link_msgpack_host TRUE)
endif()

nanobind_add_module(tensilelite_write NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite_write.cpp")

set_target_properties(tensilelite_write PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_tensilelite_write_output}"
    RUNTIME_OUTPUT_DIRECTORY "${_tensilelite_write_output}"
)
# The tensilelite project root is already on sys.path for tests and for
# tensilelite/bin. Copying the module there makes `import tensilelite_write`
# succeed after a client or library build, with no separate step.
add_custom_command(TARGET tensilelite_write POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:tensilelite_write>
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

set_property(TARGET tensilelite_write PROPERTY THEROCK_INSTALL_RPATH_ORIGIN
    "${_tensilelite_write_install}")

if(_link_msgpack_host)
    # tensilelite-host links msgpack privately, so this module still needs
    # the header for <msgpack.hpp>.
    target_link_libraries(tensilelite_write PRIVATE tensilelite-host ${hipblaslt_msgpack_target})
    set_target_properties(tensilelite_write PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:tensilelite-host>"
        INSTALL_RPATH "$ORIGIN/../.."
    )
    # Windows resolves a module's DLLs from the module directory, not from PATH.
    if(WIN32)
        get_target_property(_tensilelite_host_type tensilelite-host TYPE)
        if(_tensilelite_host_type STREQUAL "SHARED_LIBRARY")
            add_custom_command(TARGET tensilelite_write POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:tensilelite-host>
                    $<TARGET_FILE_DIR:tensilelite_write>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:tensilelite-host>
                    "${CMAKE_CURRENT_SOURCE_DIR}/.."
            )
            install(FILES $<TARGET_FILE:tensilelite-host>
                DESTINATION "${_tensilelite_write_install}"
                COMPONENT devel)
        endif()
    endif()
else()
    # Host library is absent (gemm-libs) or compiled for YAML, which does not
    # build src/msgpack. Compile the writer and the serialization sources it
    # links into this module. A YAML host that is not generating device
    # libraries does not reach this directory.
    if(NOT TARGET ZLIB::ZLIB)
        find_package(ZLIB REQUIRED)
    endif()
    if(NOT hipblaslt_msgpack_target)
        message(FATAL_ERROR "tensilelite_write requires msgpack-cxx")
    endif()

    include(GenerateExportHeader)
    add_library(tensilelite-msgpack-writer STATIC
        ${TENSILELITE_HOST_SOURCES}
        ${TENSILELITE_HOST_MSGPACK_SOURCES}
    )
    set_target_properties(tensilelite-msgpack-writer PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    generate_export_header(tensilelite-msgpack-writer
        BASE_NAME TENSILELITEHOST
        EXPORT_MACRO_NAME TENSILELITEHOST_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tensilelitehost/export.h"
        STATIC_DEFINE TENSILELITEHOST_STATIC_DEFINE
    )
    target_compile_features(tensilelite-msgpack-writer PRIVATE cxx_std_17)
    target_compile_definitions(tensilelite-msgpack-writer
        PUBLIC
            TENSILELITEHOST_STATIC_DEFINE
        PRIVATE
            TENSILE_MSGPACK
            __HIP_HCC_COMPAT_MODE__=1
    )
    if(HIP_HAS_CLUSTER_LAUNCH)
        target_compile_definitions(tensilelite-msgpack-writer PRIVATE HIP_HAS_CLUSTER_LAUNCH)
    endif()
    target_include_directories(tensilelite-msgpack-writer
        PUBLIC
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )
    target_link_libraries(tensilelite-msgpack-writer
        PUBLIC
            rocisa::rocisa-cpp
            roc::origami
            hip::host
            ${hipblaslt_msgpack_target}
            ZLIB::ZLIB
        PRIVATE
            hip::device
            ${CMAKE_DL_LIBS}
    )
    target_link_libraries(tensilelite_write PRIVATE tensilelite-msgpack-writer)
endif()

install(TARGETS tensilelite_write
    LIBRARY DESTINATION "${_tensilelite_write_install}"
    RUNTIME DESTINATION "${_tensilelite_write_install}"
    COMPONENT devel
)
