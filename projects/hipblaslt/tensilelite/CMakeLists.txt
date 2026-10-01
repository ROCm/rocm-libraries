# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25.2)
    set(TENSILELITE_STANDALONE ON)
    set(TENSILELITE_OUTER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    cmake_path(SET ROCM_LIBRARIES_ROOT NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
    include("${ROCM_LIBRARIES_ROOT}/cmake/modules/default_amdclang.cmake")
else()
    set(TENSILELITE_STANDALONE OFF)
    set(TENSILELITE_OUTER_BINARY_DIR "${PROJECT_BINARY_DIR}")
endif()

set(TENSILELITE_PROJECT_VERSION "1.5.0" CACHE STRING "Semantic version string.")
project(tensilelite VERSION "${TENSILELITE_PROJECT_VERSION}" LANGUAGES CXX)

if(TENSILELITE_STANDALONE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensilelite_standalone.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensilelite_dependencies.cmake")
    set(TENSILELITE_RUNTIME_COMPONENT runtime)
    set(TENSILELITE_DEVEL_COMPONENT devel)
else()
    set(TENSILELITE_RUNTIME_COMPONENT tensilelite)
    set(TENSILELITE_DEVEL_COMPONENT tensilelite-devel)
endif()

option(TENSILELITE_BUILD_SHARED_LIBS "Build the TensileLite host library as shared vs static." ON)
set(BUILD_SHARED_LIBS "${TENSILELITE_BUILD_SHARED_LIBS}")

option(TENSILELITE_ENABLE_HOST_ASAN "Enable AddressSanitizer for host code" OFF)
option(TENSILELITE_ENABLE_HOST_TSAN "Enable ThreadSanitizer for host code" OFF)
option(TENSILELITE_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
mark_as_advanced(TENSILELITE_ENABLE_HOST_ASAN)
mark_as_advanced(TENSILELITE_ENABLE_HOST_TSAN)
mark_as_advanced(TENSILELITE_ENABLE_COVERAGE)

if(TENSILELITE_ENABLE_HOST_ASAN AND TENSILELITE_ENABLE_HOST_TSAN)
    message(FATAL_ERROR "ASAN and TSAN cannot be enabled simultaneously")
endif()

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    find_package(ZLIB REQUIRED)

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

    if(TENSILELITE_ENABLE_HOST_ASAN)
        # Apply sanitizer only to host code, not device code
        # Use SHELL: to pass flags that should not be applied to device offload
        target_compile_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=address>>
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fno-omit-frame-pointer>>
        )
        target_link_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:SHELL:-Xarch_host -fsanitize=address>
        )
    endif()

    if(TENSILELITE_ENABLE_HOST_TSAN)
        # Apply sanitizer only to host code, not device code
        # Use SHELL: to pass flags that should not be applied to device offload
        target_compile_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=thread>>
        )
        target_link_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:SHELL:-Xarch_host -fsanitize=thread>
        )
    endif()

    if(HIPBLASLT_ENABLE_YAML)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_YAML>)
    else()
        target_link_libraries(tensilelite-host PRIVATE ${tensilelite_msgpack_target})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>)
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # mxDataGenerator supplies the gfx950 pre-swizzled MX scale padding helper behind
    # the StrideScaleA1/StrideScaleB1 custom-kernel semantics. Treat it as optional:
    # HIPBLASLT_ENABLE_MXDATAGENERATOR is OFF on Windows, and shared/mxdatagenerator
    # is absent from checkouts that vendor only projects/hipblaslt, so probe for it
    # instead of hard-requiring it.
    set(TENSILELITE_HOST_MXDATAGENERATOR ${HIPBLASLT_ENABLE_MXDATAGENERATOR})
    set(_mxdatagen_target roc::mxDataGenerator)
    if(TENSILELITE_HOST_MXDATAGENERATOR AND NOT TARGET roc::mxDataGenerator)
        set(_mxdatagen_src "${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/mxdatagenerator")
        if(HIPBLASLT_ENABLE_THEROCK)
            find_package(mxDataGenerator QUIET)
            if(NOT mxDataGenerator_FOUND)
                set(TENSILELITE_HOST_MXDATAGENERATOR OFF)
            endif()
        elseif(EXISTS "${_mxdatagen_src}/lib/include/mxDataGenerator/PreSwizzle.hpp")
            # Consume the header-only helper directly rather than add_subdirectory()ing
            # mxDataGenerator: its CMakeLists emits install() rules unconditionally and
            # EXCLUDE_FROM_ALL does not suppress them, so vendoring it here republishes
            # its 25 headers and 3 package files from whatever install tree embeds this
            # subdirectory. hipSPARSELt embeds it, which put those files in TheRock's
            # blas artifact alongside support's own copy (rocm-libraries#11105).
            find_package(OpenMP REQUIRED)
            add_library(tensilelite-mxdatagenerator-headers INTERFACE)
            target_include_directories(tensilelite-mxdatagenerator-headers
                INTERFACE $<BUILD_INTERFACE:${_mxdatagen_src}/lib/include>)
            # LINK_ONLY: OpenMP::OpenMP_CXX carries -fopenmp in its INTERFACE_COMPILE_OPTIONS.
            # Propagating that into tensilelite-host's compiles is not merely unnecessary
            # (the pre-swizzle helper is correct serially) -- it is actively harmful. On the
            # CI image, -fopenmp re-adds ${ROCM_PATH}/lib/llvm/include as an *implicit system*
            # include dir to locate omp.h. That demotes the explicit -I for the same directory,
            # and <llvm/...> includes then fall through to the system LLVM 18 tree in
            # /usr/include/llvm. tensilelite-host would compile LLVM YAMLIO against LLVM 18
            # headers while linking ROCm's LLVM 24 libraries -- an ABI mismatch that silently
            # corrupts llvm::yaml::Input, making TensileLibrary.yaml fail to parse with
            # "unknown key 'solutions'". Keep the link requirement, drop the compile options.
            target_link_libraries(tensilelite-mxdatagenerator-headers
                INTERFACE $<LINK_ONLY:OpenMP::OpenMP_CXX>)
            set(_mxdatagen_target tensilelite-mxdatagenerator-headers)
        else()
            set(TENSILELITE_HOST_MXDATAGENERATOR OFF)
        endif()
    endif()
    if(NOT TENSILELITE_HOST_MXDATAGENERATOR)
        message(STATUS "tensilelite-host: mxDataGenerator not available; "
                       "pre-swizzled MX scale strides (StrideScaleA1/B1) are disabled")
    endif()

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
            $<BUILD_INTERFACE:hip::host>
        PRIVATE
            hip::device
            ZLIB::ZLIB
    )

    if(TENSILELITE_HOST_MXDATAGENERATOR)
        # Consume mxDataGenerator's include directories, but take the target itself as a
        # LINK_ONLY dependency so its INTERFACE_COMPILE_OPTIONS do not reach our compiles.
        # roc::mxDataGenerator links OpenMP::OpenMP_CXX (shared/mxdatagenerator/CMakeLists.txt),
        # which carries -fopenmp. On the CI image -fopenmp makes clang re-add
        # ${ROCM_PATH}/lib/llvm/include as an *implicit system* include dir (to find omp.h).
        # That demotes the explicit -I for the same directory, and <llvm/...> includes then
        # fall through to the system LLVM 18 tree under /usr/include/llvm. tensilelite-host
        # would compile llvm::yaml against LLVM 18 headers while linking ROCm's LLVM 24
        # libraries: an ABI mismatch that silently corrupts llvm::yaml::Input, so every
        # TensileLibrary.yaml fails to load with "unknown key 'solutions'".
        # The pre-swizzle helper is correct serially, so dropping -fopenmp here is safe.
        get_target_property(_mxdatagen_incs ${_mxdatagen_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_mxdatagen_incs)
            target_include_directories(tensilelite-host PRIVATE ${_mxdatagen_incs})
        endif()
        set_property(TARGET tensilelite-host APPEND PROPERTY
            LINK_LIBRARIES $<LINK_ONLY:${_mxdatagen_target}>)
        target_compile_definitions(tensilelite-host PRIVATE HIPBLASLT_ENABLE_MXDATAGENERATOR)
    endif()

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
        target_link_options(tensilelite-host PUBLIC $<BUILD_INTERFACE:-fprofile-instr-generate -fcoverage-mapping>)
    endif()

    add_subdirectory(src)
    add_subdirectory(include)

    install(TARGETS tensilelite-host
        EXPORT tensilelite-targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET tensilelite_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET tensilelite_generated_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET rocisa_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )

    set(_tensilelite_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/tensilelite")
    install(EXPORT tensilelite-targets
        NAMESPACE roc::
        DESTINATION "${_tensilelite_config_dir}"
        COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )

    get_target_property(_tensilelite_host_type tensilelite-host TYPE)
    if(_tensilelite_host_type STREQUAL "STATIC_LIBRARY")
        set(TENSILELITE_CONFIG_STATIC ON)
    else()
        set(TENSILELITE_CONFIG_STATIC OFF)
    endif()
    set(TENSILELITE_CONFIG_YAML "${HIPBLASLT_ENABLE_YAML}")
    set(TENSILELITE_CONFIG_MSGPACK_TARGET "${tensilelite_msgpack_target}")
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensilelite-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config.cmake"
        INSTALL_DESTINATION "${_tensilelite_config_dir}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config-version.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config-version.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipblaslt_codegen.cmake"
        DESTINATION "${_tensilelite_config_dir}"
        COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )
    unset(_tensilelite_config_dir)
    unset(_tensilelite_host_type)

    if(BUILD_SHARED_LIBS)
        set_property(TARGET tensilelite-host APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()

    if(TENSILELITE_ENABLE_CLIENT)
        add_subdirectory(client)
    endif()

    if(TENSILELITE_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

if(TENSILELITE_STANDALONE)
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    rocm_get_patch_version(_tensilelite_patch_version)
    set(_tensilelite_moved_from "1.5.0")
    if(_tensilelite_patch_version)
        string(APPEND _tensilelite_moved_from ".${_tensilelite_patch_version}")
    endif()
    if(TENSILELITE_BUILD_SHARED_LIBS)
        set(CPACK_DEBIAN_RUNTIME_PACKAGE_REPLACES "hipblaslt (<< ${_tensilelite_moved_from})")
        set(CPACK_DEBIAN_RUNTIME_PACKAGE_BREAKS "hipblaslt (<< ${_tensilelite_moved_from})")
        set(CPACK_RPM_RUNTIME_PACKAGE_CONFLICTS "hipblaslt < ${_tensilelite_moved_from}")
    endif()
    set(CPACK_DEBIAN_DEVEL_PACKAGE_REPLACES "hipblaslt-dev (<< ${_tensilelite_moved_from}), hipblaslt-static-dev (<< ${_tensilelite_moved_from})")
    set(CPACK_DEBIAN_DEVEL_PACKAGE_BREAKS "hipblaslt-dev (<< ${_tensilelite_moved_from}), hipblaslt-static-dev (<< ${_tensilelite_moved_from})")
    set(CPACK_RPM_DEVEL_PACKAGE_CONFLICTS "hipblaslt-devel < ${_tensilelite_moved_from}, hipblaslt-static-devel < ${_tensilelite_moved_from}")
    rocm_create_package(
        NAME tensilelite-host
        DESCRIPTION "TensileLite host library for GEMM solution selection and kernel launch"
        MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
        LDCONFIG
    )
elseif(TENSILELITE_ENABLE_HOST AND (ROCM_LIBS_SUPERBUILD OR NOT HIPBLASLT_IS_SUBPROJECT))
    block(SCOPE_FOR VARIABLES)
        set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")
        set(CPACK_RPM_PACKAGE_LICENSE "MIT")
        set(CPACK_OUTPUT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackConfig.cmake")
        set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackSourceConfig.cmake")
        set(CPACK_INSTALL_CMAKE_PROJECTS "${PROJECT_BINARY_DIR};tensilelite;ALL;/")
        if(tensilelite_bundled_origami_component)
            list(APPEND CPACK_INSTALL_CMAKE_PROJECTS
                "${TENSILELITE_OUTER_BINARY_DIR}/origami;Origami;ALL;/")
        endif()

        set(ROCM_PACKAGE_COMPONENTS "")
        set(ROCM_PACKAGE_COMPONENT_DEPENDENCIES "")
        rocm_get_patch_version(_tensilelite_patch_version)
        set(_tensilelite_package_version "${PROJECT_VERSION}")
        set(_tensilelite_moved_from "1.5.0")
        if(_tensilelite_patch_version)
            string(APPEND _tensilelite_package_version ".${_tensilelite_patch_version}")
            string(APPEND _tensilelite_moved_from ".${_tensilelite_patch_version}")
        endif()

        if(ENABLE_ASAN_PACKAGING)
            rocm_package_setup_component(tensilelite)
            set(CPACK_RPM_TENSILELITE_PACKAGE_NAME "tensilelite-host-asan")
            set(CPACK_DEBIAN_TENSILELITE_PACKAGE_NAME "tensilelite-host-asan")
            if(ROCM_DEP_ROCMCORE)
                rocm_package_add_dependencies(DEPENDS "rocm-core-asan")
            endif()
            install(FILES "${CPACK_RESOURCE_FILE_LICENSE}"
                DESTINATION "share/doc/tensilelite-host-asan"
                COMPONENT tensilelite)
            set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tensilelite-license)
        else()
            rocm_package_setup_component(tensilelite-devel)
            if(TENSILELITE_BUILD_SHARED_LIBS)
                rocm_package_setup_component(tensilelite
                    LIBRARY_NAME tensilelite
                    PACKAGE_NAME host
                    PARENT tensilelite-devel)
                set(_tensilelite_devel_package tensilelite-host)
                set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tensilelite)
                set(CPACK_DEBIAN_TENSILELITE_PACKAGE_RECOMMENDS
                    "tensilelite-host-dev (>=${_tensilelite_package_version})")
                rocm_find_program_version(rpmbuild GREATER_EQUAL 4.12.0 QUIET)
                if(rpmbuild_VERSION_OK)
                    set(CPACK_RPM_TENSILELITE_PACKAGE_SUGGESTS
                        "tensilelite-host-devel >= ${_tensilelite_package_version}")
                endif()
            else()
                set(_tensilelite_devel_package tensilelite-host-static)
                set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tensilelite-devel)
            endif()
            set(CPACK_RPM_TENSILELITE-DEVEL_PACKAGE_NAME "${_tensilelite_devel_package}-devel")
            set(CPACK_DEBIAN_TENSILELITE-DEVEL_PACKAGE_NAME "${_tensilelite_devel_package}-dev")
            set(_tensilelite_deb_devel_owners "hipblaslt-dev (<< ${_tensilelite_moved_from}), hipblaslt-static-dev (<< ${_tensilelite_moved_from})")
            set(_tensilelite_rpm_devel_owners "hipblaslt-devel < ${_tensilelite_moved_from}, hipblaslt-static-devel < ${_tensilelite_moved_from}")
            set(CPACK_DEBIAN_TENSILELITE-DEVEL_PACKAGE_REPLACES "${_tensilelite_deb_devel_owners}")
            set(CPACK_DEBIAN_TENSILELITE-DEVEL_PACKAGE_BREAKS "${_tensilelite_deb_devel_owners}")
            set(CPACK_RPM_TENSILELITE-DEVEL_PACKAGE_CONFLICTS "${_tensilelite_rpm_devel_owners}")
            if(tensilelite_bundled_origami_component)
                rocm_package_setup_component(origami-devel PARENT tensilelite-devel)
                set(CPACK_RPM_ORIGAMI-DEVEL_PACKAGE_NAME "origami-static-devel")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_NAME "origami-static-dev")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_REPLACES "${_tensilelite_deb_devel_owners}")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_BREAKS "${_tensilelite_deb_devel_owners}")
                set(CPACK_RPM_ORIGAMI-DEVEL_PACKAGE_CONFLICTS "${_tensilelite_rpm_devel_owners}")
            endif()
        endif()

        if(TENSILELITE_BUILD_SHARED_LIBS)
            set(CPACK_DEBIAN_TENSILELITE_PACKAGE_REPLACES "hipblaslt (<< ${_tensilelite_moved_from})")
            set(CPACK_DEBIAN_TENSILELITE_PACKAGE_BREAKS "hipblaslt (<< ${_tensilelite_moved_from})")
            set(CPACK_RPM_TENSILELITE_PACKAGE_CONFLICTS "hipblaslt < ${_tensilelite_moved_from}")

            # rocm-cmake's packaging defaults set CPACK_RPM_PACKAGE_AUTOREQPROV to
            # Off for every ROCm library, so no package currently Provides or
            # Requires a soname. Re-enable rpmbuild's ELF scanning for this one
            # component: tensilelite-host then Provides
            # libtensilelite-host.so.1()(64bit), derived from the binary rather
            # than hand-written, so the capability tracks SOVERSION automatically.
            # This is scoped to the component and does not change any other
            # package's dependency surface.
            set(CPACK_RPM_TENSILELITE_PACKAGE_AUTOREQPROV "ON")
        endif()

        set(ROCM_USE_DEV_COMPONENT OFF)
        set(BUILD_SHARED_LIBS ON)
        if(ENABLE_ASAN_PACKAGING)
            set(ENABLE_ASAN_PACKAGING OFF)
            set(ROCM_DEP_ROCMCORE OFF)
        endif()
        rocm_create_package(
            NAME tensilelite-host
            DESCRIPTION "TensileLite host library for GEMM solution selection and kernel launch"
            MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
            LDCONFIG
            HEADER_ONLY
        )
    endblock()
endif()
