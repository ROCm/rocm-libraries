# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    find_package(ZLIB REQUIRED)

    # tensilelite-host is now a single idiomatic library (libtensilelitehost.so when
    # shared) co-exported through hipBLASLt's package, instead of an OBJECT library
    # absorbed into both libhipblaslt.so and libhipsparselt.so. Building it once and
    # sharing it is what structurally prevents the cross-DSO C++ interposition that
    # the old all-hidden OBJECT scheme worked around (PR #6917).
    # Default to hipBLASLt's own shared/static choice (HIPBLASLT_BUILD_SHARED_LIBS,
    # defined before this subdirectory), NOT the global BUILD_SHARED_LIBS -- the
    # latter is only synced to HIPBLASLT_BUILD_SHARED_LIBS later in the parent
    # CMakeLists, after this subdirectory is added, so it is still empty here and
    # would silently fall back to STATIC (defeating the single-shared-library dedup).
    option(TENSILELITE_BUILD_SHARED_LIBS
        "Build tensilelite-host as a shared library." "${HIPBLASLT_BUILD_SHARED_LIBS}")
    # ON in hipBLASLt's standalone build (it owns the install/export); hipSPARSELt
    # sets it OFF in its add_subdirectory(hipblaslt) context so the install rules
    # do not double-fire from the fork.
    option(TENSILELITE_ENABLE_INSTALL
        "Install and co-export tensilelite-host through hipBLASLt's package." ON)
    if(TENSILELITE_BUILD_SHARED_LIBS)
        set(TENSILELITE_HOST_LIBRARY_TYPE SHARED)
    else()
        set(TENSILELITE_HOST_LIBRARY_TYPE STATIC)
    endif()
    add_library(tensilelite-host ${TENSILELITE_HOST_LIBRARY_TYPE})
    add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)
    add_library(roc::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            SOVERSION 1
    )

    target_compile_features(tensilelite-host PUBLIC cxx_std_17)

    # Generate tensilelitehost/export.h defining TENSILELITEHOST_EXPORT. Public API
    # declarations are annotated with it so that, under the hidden global preset, the
    # public surface is exported from the shared library and nothing else is. This
    # replaces the inverted PR #6917 scheme (TENSILE_API = hidden marker +
    # TENSILE_HIDDEN_BEGIN/END region wrappers).
    include(GenerateExportHeader)
    generate_export_header(tensilelite-host
        BASE_NAME TENSILELITEHOST
        EXPORT_MACRO_NAME TENSILELITEHOST_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tensilelitehost/export.h"
        STATIC_DEFINE TENSILELITEHOST_STATIC
    )
    if(NOT TENSILELITE_BUILD_SHARED_LIBS)
        target_compile_definitions(tensilelite-host PUBLIC TENSILELITEHOST_STATIC)
    endif()
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
            $<BUILD_INTERFACE:TENSILE_USE_HIP>
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

    # tensilelite-host is an internal co-exported shared library: its symbols are
    # exported (so libhipblaslt.so / libhipsparselt.so resolve them at runtime) but
    # its headers are NOT a public install surface (they were never installed
    # pre-Phase-2, and their include closure is essentially the whole Tensile tree
    # -- a narrow public-header API would require decoupling the entry-point headers
    # from their implementation includes, a separate effort). In-tree consumers
    # (hipBLASLt, hipSPARSELt, the tensilelite client/tests) include these via the
    # build interface below; no INSTALL_INTERFACE include is published.
    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
            # Three public headers (hip/Hip{Hardware,SolutionAdapter,Utils}.hpp)
            # include <hip/hip_runtime.h> for host-side hip API, so the public
            # interface requires hip::host. hip::device stays PRIVATE to satisfy the
            # exported-target invariant (hipblaslt-config.cmake.in:15-19).
            hip::host
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

    add_subdirectory(src)
    add_subdirectory(include)

    # G4 (header self-containment): VERIFY_INTERFACE_HEADER_SETS is intentionally
    # NOT enabled. CMake's header-set verification compiles each header as `-x c++`,
    # but TensileLite's headers are HIP-language headers -- they transitively pull in
    # <hip/hip_fp8.h> etc. and use HIP builtins (__hip_fp8_e4m3, __half,
    # __amd_floatx16_storage_t) that only exist under `-x hip` (tensilelite-host's
    # own .cpp compile with -x hip). A c++-mode verify cannot compile them, and
    # wiring a permanently-failing all_verify_interface_header_sets into `check`
    # would break CI. Self-containment is instead enforced by the real build: every
    # header is included (in HIP mode) by the library's translation units.

    if(TENSILELITE_ENABLE_INSTALL)
        # Co-export the shared library through hipBLASLt's existing package: same
        # hipblaslt-targets export set + namespace, so find_package(hipblaslt) also
        # yields roc::tensilelite-host (a runtime dependency of roc::hipblaslt) and
        # libtensilelite-host.so installs alongside libhipblaslt.so for DT_NEEDED
        # resolution. NOT a new top-level package. Library-only (no FILE_SET): the
        # headers are an in-tree build interface, not a public install surface.
        install(TARGETS tensilelite-host
            EXPORT hipblaslt-targets
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
        )
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
