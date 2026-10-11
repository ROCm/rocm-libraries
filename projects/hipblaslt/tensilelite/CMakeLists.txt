# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

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
            $<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=address>
            $<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fno-omit-frame-pointer>
        )
        target_link_options(tensilelite-host PUBLIC
            SHELL:-Xarch_host -fsanitize=address
        )
    endif()

    if(TENSILELITE_ENABLE_HOST_TSAN)
        # Apply sanitizer only to host code, not device code
        # Use SHELL: to pass flags that should not be applied to device offload
        target_compile_options(tensilelite-host PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=thread>
        )
        target_link_options(tensilelite-host PUBLIC
            SHELL:-Xarch_host -fsanitize=thread
        )
    endif()

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
        target_link_libraries(tensilelite-host PRIVATE $<LINK_ONLY:${_mxdatagen_target}>)
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
        target_link_options(tensilelite-host PUBLIC -fprofile-instr-generate -fcoverage-mapping)
    endif()

    add_subdirectory(src)
    add_subdirectory(include)

    # The shared library ships in its own "tensilelite" component (package
    # tensilelite-host) rather than hipblaslt's runtime component. hipSPARSELt
    # needs this one ~6.5 MB .so; bundling it in hipblaslt's runtime package
    # would force every hipSPARSELt install to pull ~1.6 GB of Tensile kernel
    # binaries it does not use. See AIHPBLAS-4434.
    # Interim: the static archive, headers and the roc::tensilelite-host
    # imported target still ship through hipblaslt-devel and the
    # hipblaslt-targets export set. A standalone tensilelite package moves
    # them to its own export set, config file and devel package.
    install(TARGETS tensilelite-host
        EXPORT hipblaslt-targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tensilelite
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT tensilelite
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel
        FILE_SET tensilelite_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
        FILE_SET tensilelite_generated_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
        FILE_SET rocisa_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT devel
    )

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
