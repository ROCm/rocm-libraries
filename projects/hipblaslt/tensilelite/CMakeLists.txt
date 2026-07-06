# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Coverage instrumentation for C++ code
option(TENSILELITE_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    find_package(ZLIB REQUIRED)

    option(TENSILELITE_BUILD_SHARED_LIBS "Build tensilelite-host as shared vs static" OFF)
    set(BUILD_SHARED_LIBS ${TENSILELITE_BUILD_SHARED_LIBS})
    add_library(tensilelite-host)
    add_library(roc::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
    rocm_set_soversion(tensilelite-host "1.0")

    if(BUILD_SHARED_LIBS AND NOT WIN32)
        target_link_options(tensilelite-host PRIVATE "LINKER:--exclude-libs,ALL")
    endif()

    target_compile_features(tensilelite-host PUBLIC cxx_std_17)

    include(GenerateExportHeader)
    generate_export_header(tensilelite-host
        BASE_NAME TENSILELITEHOST
        EXPORT_MACRO_NAME TENSILELITEHOST_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tensilelitehost/export.h"
    )
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(HIPBLASLT_ENABLE_YAML)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_YAML>)
    else()
        target_link_libraries(tensilelite-host PRIVATE ${hipblaslt_msgpack_target})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>)
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            $<BUILD_INTERFACE:roc::origami>
            $<BUILD_INTERFACE:hip::host>
        PRIVATE
            hip::device
            ZLIB::ZLIB
    )

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${HIP_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES hip::device)
    check_cxx_source_compiles("
        #include <hip/hip_runtime.h>
        int main() {
            hipLaunchAttributeID id = hipLaunchAttributeClusterDimension;
            return 0;
        }
    " HIP_HAS_CLUSTER_LAUNCH)
    if(HIP_HAS_CLUSTER_LAUNCH)
        target_compile_definitions(tensilelite-host PRIVATE HIP_HAS_CLUSTER_LAUNCH)
    endif()

    if(TENSILELITE_ENABLE_COVERAGE)
        message(STATUS "Enabling C++ code coverage instrumentation for tensilelite-host")
        target_compile_options(tensilelite-host PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(tensilelite-host PUBLIC -fprofile-instr-generate -fcoverage-mapping)
    endif()

    add_subdirectory(src)
    add_subdirectory(include)

    if(BUILD_SHARED_LIBS)
        if(HIPBLASLT_ENABLE_HOST)
            install(TARGETS tensilelite-host
                EXPORT hipblaslt-targets
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
                FILE_SET tensilelite_public_headers
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
                FILE_SET tensilelite_generated_headers
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
                FILE_SET rocisa_public_headers
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
            )
        elseif(DEFINED TENSILELITE_HOST_EXPORT_SET)
            install(TARGETS tensilelite-host
                EXPORT ${TENSILELITE_HOST_EXPORT_SET}
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
            )
        else()
            install(TARGETS tensilelite-host
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
            )
        endif()
        set_property(TARGET tensilelite-host APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()

    if(TENSILELITE_ENABLE_CLIENT)
        # mxDataGenerator bootstrap and `hipblaslt::mxdatagen` helper target
        # live in projects/hipblaslt/CMakeLists.txt and are guaranteed to be
        # available before this subdirectory is processed.
        add_subdirectory(client)
    endif()

    if(TENSILELITE_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
