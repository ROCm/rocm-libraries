# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
rocm_get_patch_version(_tensilelite_patch_version)
set(_tensilelite_moved_from "1.5.0")
if(_tensilelite_patch_version)
    string(APPEND _tensilelite_moved_from ".${_tensilelite_patch_version}")
endif()

if(TENSILELITE_STANDALONE)
    rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "origami-static-dev")
    rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "origami-static-devel")

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
        set(CPACK_OUTPUT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackConfig.cmake")
        set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackSourceConfig.cmake")
        set(CPACK_INSTALL_CMAKE_PROJECTS "${PROJECT_BINARY_DIR};tensilelite;ALL;/")
        if(TENSILELITE_BUNDLED_ORIGAMI)
            list(APPEND CPACK_INSTALL_CMAKE_PROJECTS
                "${TENSILELITE_BUNDLED_ORIGAMI_BINARY_DIR};Origami;ALL;/")
        endif()

        set(ROCM_PACKAGE_COMPONENTS "")
        set(ROCM_PACKAGE_COMPONENT_DEPENDENCIES "")
        set(_tensilelite_package_version "${PROJECT_VERSION}")
        if(_tensilelite_patch_version)
            string(APPEND _tensilelite_package_version ".${_tensilelite_patch_version}")
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
            if(TENSILELITE_BUNDLED_ORIGAMI)
                rocm_package_setup_component(origami-devel PARENT tensilelite-devel)
                set(CPACK_RPM_ORIGAMI-DEVEL_PACKAGE_NAME "origami-static-devel")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_NAME "origami-static-dev")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_REPLACES "${_tensilelite_deb_devel_owners}")
                set(CPACK_DEBIAN_ORIGAMI-DEVEL_PACKAGE_BREAKS "${_tensilelite_deb_devel_owners}")
                set(CPACK_RPM_ORIGAMI-DEVEL_PACKAGE_CONFLICTS "${_tensilelite_rpm_devel_owners}")
            endif()
        endif()

        if(TENSILELITE_BUILD_SHARED_LIBS)
            if(ENABLE_ASAN_PACKAGING)
                set(_tensilelite_previous_runtime_package "hipblaslt-asan")
            else()
                set(_tensilelite_previous_runtime_package "hipblaslt")
            endif()
            set(CPACK_DEBIAN_TENSILELITE_PACKAGE_REPLACES "${_tensilelite_previous_runtime_package} (<< ${_tensilelite_moved_from})")
            set(CPACK_DEBIAN_TENSILELITE_PACKAGE_BREAKS "${_tensilelite_previous_runtime_package} (<< ${_tensilelite_moved_from})")
            set(CPACK_RPM_TENSILELITE_PACKAGE_CONFLICTS "${_tensilelite_previous_runtime_package} < ${_tensilelite_moved_from}")
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
