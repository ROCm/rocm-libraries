# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    # tensilelite-host is now a single idiomatic library (libtensilelitehost.so when
    # shared) co-exported through hipBLASLt's package, instead of an OBJECT library
    # absorbed into both libhipblaslt.so and libhipsparselt.so. Building it once and
    # sharing it is what structurally prevents the cross-DSO C++ interposition that
    # the old all-hidden OBJECT scheme worked around (PR #6917).
    # Track hipBLASLt's own shared/static choice when it owns the host build, but
    # default to SHARED otherwise. Subtleties this guards against:
    #  - The global BUILD_SHARED_LIBS is synced to HIPBLASLT_BUILD_SHARED_LIBS only
    #    LATER in the parent (after this add_subdirectory), so it is empty here.
    #  - HIPBLASLT_BUILD_SHARED_LIBS itself is defined only under
    #    if(HIPBLASLT_ENABLE_HOST). hipSPARSELt consumes tensilelite-host with
    #    HIPBLASLT_ENABLE_HOST OFF, so that variable is UNDEFINED there -- falling
    #    back to it would build tensilelite-host STATIC and re-absorb it into
    #    libhipsparselt.so, defeating the single-shared-library dedup. Default ON
    #    keeps the dedup working for every consumer; a fully-static build can still
    #    set TENSILELITE_BUILD_SHARED_LIBS=OFF explicitly.
    if(DEFINED HIPBLASLT_BUILD_SHARED_LIBS)
        set(_tensilelite_shared_default ${HIPBLASLT_BUILD_SHARED_LIBS})
    else()
        set(_tensilelite_shared_default ON)
    endif()
    option(TENSILELITE_BUILD_SHARED_LIBS
        "Build tensilelite-host as a shared library." ${_tensilelite_shared_default})
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
    )
    # Full SOVERSION + VERSION symlink chain via the ROCm idiom (matches hipBLASLt /
    # hipSPARSELt / origami), rather than a bare SOVERSION property.
    rocm_set_soversion(tensilelite-host "1.0")

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
    # TENSILE_USE_HIP removed: it is unconditionally on (no build option ever
    # disabled it) and the non-HIP fallbacks it gated were dead, non-compilable
    # code (e.g. the opaque DistinctType `Half` could not satisfy the
    # unconditional float->Half cast in KernelArguments.hpp). The public DataTypes_*
    # headers now define the HIP types unconditionally, so the macro no longer
    # gates any public type and need not cross the install interface.
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
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

    # tensilelite-host is a co-exported shared library. Its public headers -- the
    # curated FILE_SETs declared in include/CMakeLists.txt (the transitive include
    # closure of the entry-point headers out-of-tree consumers actually use) -- are
    # installed, and install(TARGETS ... FILE_SET ...) adds the install destination
    # to the INSTALL_INTERFACE include path automatically, but ONLY when hipBLASLt
    # owns the host build (single header owner; the hipSPARSELt fork ships
    # library-only). The build-interface include dirs below serve in-tree consumers
    # (hipBLASLt, hipSPARSELt, the tensilelite client/tests) in every configuration.
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

    # G4 (header self-containment): the public surface is now a real PUBLIC FILE_SET
    # (tensilelite_public_headers, see include/CMakeLists.txt), but
    # VERIFY_INTERFACE_HEADER_SETS is still intentionally NOT enabled. CMake's
    # header-set verification compiles each header as `-x c++`, but TensileLite's
    # headers are HIP-language headers -- they transitively pull in <hip/hip_fp8.h>
    # etc. and use HIP builtins (__hip_fp8_e4m3, __half, __amd_floatx16_storage_t)
    # that only exist under `-x hip` (tensilelite-host's own .cpp compile with
    # -x hip). A c++-mode verify cannot compile them, and wiring a
    # permanently-failing all_verify_interface_header_sets into `check` would break
    # CI. Self-containment is instead enforced by the real build: every header is
    # included (in HIP mode) by the library's translation units.

    if(TENSILELITE_ENABLE_INSTALL)
        # Install tensilelite-host so consumers resolve libtensilelite-host.so at
        # runtime, and co-export it through the owning package (NOT a new top-level
        # package). Header ownership is single: only the hipBLASLt-owned build
        # (HIPBLASLT_ENABLE_HOST) installs the public header FILE_SETs; a fork that
        # builds host with HOST OFF (hipSPARSELt) ships library-only, so the installed
        # Tensile/*.hpp have one owner and never collide in a shared ROCm prefix.
        #
        # Export set: hipBLASLt's own (hipblaslt-targets) when it owns the build;
        # otherwise the fork names its set via TENSILELITE_HOST_EXPORT_SET (hipSPARSELt
        # -> hipsparselt-targets) so its exported library, which links the SHARED
        # tensilelite-host, carries that dependency. The fork's package config must
        # then find_dependency(origami) (tensilelite-host links it PUBLIC).
        if(HIPBLASLT_ENABLE_HOST)
            # Owner build: co-export through hipblaslt-targets WITH the public
            # headers. install(TARGETS ... FILE_SET ...) also adds the include
            # DESTINATION to the target's INSTALL_INTERFACE include path, so a
            # post-install find_package(hipblaslt) consumer compiles against the
            # TensileLite public API.
            install(TARGETS tensilelite-host
                EXPORT hipblaslt-targets
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
                FILE_SET tensilelite_public_headers
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
                FILE_SET tensilelite_generated_headers
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
            )
        elseif(DEFINED TENSILELITE_HOST_EXPORT_SET)
            # Fork build (HOST OFF): library-only co-export. The public headers are
            # attached PRIVATE in this configuration (include/CMakeLists.txt), so
            # there is no interface file set to install -- avoiding a colliding
            # header install in the shared prefix.
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
