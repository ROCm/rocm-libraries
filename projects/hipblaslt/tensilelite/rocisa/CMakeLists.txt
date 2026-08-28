# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)

    include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/modules/default_amdclang.cmake)

    project(rocisa LANGUAGES CXX)
    set(ROCISA_STANDALONE ON)
else()
    set(ROCISA_STANDALONE OFF)
endif()

option(ROCISA_USE_STABLE_ABI
       "Build rocisa as a Python stable ABI extension (requires Python >= 3.12 and CMake >= 3.26)."
       OFF)

option(ROCISA_ENABLE_COVERAGE "Enable code coverage instrumentation for rocisa" OFF)

# Compiles StinkyTofu's example HelloWorldPass plugin directly into _rocisa (LLVM-style
# static-plugin pattern: object code linked into the host, registered via an explicit
# call rather than dlopen). This is independent of whether stinkytofu itself was built
# shared or static, and of whether it was found via find_package or add_subdirectory —
# see the rocisa.registerExamplePlugin() binding below. Used to exercise the compiled-in
# plugin path from test_pass_plugin.py regardless of stinkytofu's own build config.
# Named after the single plugin it embeds: it does not enable StinkyTofu's example
# suite (that is stinkytofu's own STINKYTOFU_BUILD_EXAMPLES).
option(ROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN
    "Compile StinkyTofu's example HelloWorldPass plugin directly into rocisa (for testing static stinkytofu builds)"
    OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rocisa-cpp INTERFACE)
add_library(rocisa::rocisa-cpp ALIAS rocisa-cpp)
target_sources(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include/enum.hpp>
)
target_include_directories(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Build the Python extension when bundling deps (hipblaslt subdir build)
# or when building standalone.
if(HIPBLASLT_BUNDLE_PYTHON_DEPS OR ROCISA_STANDALONE)
    if(ROCISA_STANDALONE)
        # scikit-build-core pre-installs nanobind (from build-system.requires)
        # and injects its CMake path, so find_package works without FetchContent.
        # nanobind requires Python to be found first.
        if(ROCISA_USE_STABLE_ABI)
            find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
        else()
            find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
        endif()
        find_package(nanobind REQUIRED CONFIG)
    else()
        include(FetchContent)
        FetchContent_Declare(
          nanobind
          GIT_REPOSITORY https://github.com/wjakob/nanobind.git
          GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
        )
        FetchContent_MakeAvailable(nanobind)
        if(ROCISA_USE_STABLE_ABI)
            find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
        endif()
    endif()

    # HIP is required to build _rocisa (via hip::host).  Set the search prefix
    # so find_package works both in developer environments and in the temporary
    # build sandbox created by scikit-build-core / uv.
    if(ROCM_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    endif()
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip REQUIRED)

    # Locate origami: prefer the monorepo sibling, fall back to installed package.
    if(NOT TARGET roc::origami)
        set(_origami_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami")
        if(IS_DIRECTORY "${_origami_sibling}")
            add_subdirectory("${_origami_sibling}" origami EXCLUDE_FROM_ALL)
        else()
            find_package(origami REQUIRED)
        endif()
    endif()

    set(ROCISAINST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/extension.cpp")
    set(ROCISAFUNC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/functions.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/argument.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_math.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_cast.cpp")
    set(ROCISAPASS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/pass.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/graph.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/composite.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/cycle.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/remove.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/macro_inline.cpp")

    set(_rocisa_nb_flags NOMINSIZE NB_SUPPRESS_WARNINGS)
    if(ROCISA_USE_STABLE_ABI)
        list(APPEND _rocisa_nb_flags STABLE_ABI)
    endif()

    nanobind_add_module(_rocisa ${_rocisa_nb_flags}
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/helper.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/macro.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/register.cpp"
                            ${ROCISAINST_SOURCE}
                            ${ROCISAPASS_SOURCE}
                            ${ROCISAFUNC_SOURCE})
    target_include_directories(_rocisa PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include"
    )

    if(ROCISA_ENABLE_COVERAGE)
        target_compile_options(_rocisa PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(_rocisa PRIVATE -fprofile-instr-generate)
    endif()

    # Locate stinkytofu: prefer installed package, fall back to monorepo sibling.
    if(NOT TARGET stinkytofu::stinkytofu)
        find_package(stinkytofu QUIET)
        if(NOT stinkytofu_FOUND)
            set(_stinkytofu_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
            if(IS_DIRECTORY "${_stinkytofu_sibling}")
                set(STINKYTOFU_ENABLE_WERROR ON)
                # Build the StinkyTofu Python bindings only for gfx1250.
                if("gfx1250" IN_LIST GPU_TARGETS)
                    set(STINKYTOFU_BUILD_PYTHON ON CACHE BOOL
                        "Build StinkyTofu Python bindings" FORCE)
                else()
                    set(STINKYTOFU_BUILD_PYTHON OFF CACHE BOOL
                        "Build StinkyTofu Python bindings" FORCE)
                endif()
                add_subdirectory("${_stinkytofu_sibling}" stinkytofu EXCLUDE_FROM_ALL)
                set(_stinkytofu_from_subdirectory TRUE)
            else()
                message(FATAL_ERROR "stinkytofu not found via find_package or sibling directory")
            endif()
        endif()
    endif()

    if(NOT DEFINED _stinkytofu_from_subdirectory)
        get_target_property(_st_imported stinkytofu::stinkytofu IMPORTED)
        if(_st_imported)
            set(_stinkytofu_from_subdirectory FALSE)
        else()
            set(_stinkytofu_from_subdirectory TRUE)
        endif()
    endif()

    # A static stinkytofu is linked straight into _rocisa, so none of the runtime-loading
    # machinery below applies to it: nothing to vendor into the wheel, no RPATH to point
    # at it, no DLL to copy, and no dlopen'able example plugin. Read the target type
    # rather than BUILD_SHARED_LIBS so this is also right for an imported (find_package)
    # stinkytofu, whose type is independent of this build's BUILD_SHARED_LIBS.
    get_target_property(_st_target_type stinkytofu::stinkytofu TYPE)
    if(_st_target_type STREQUAL "SHARED_LIBRARY")
        set(_stinkytofu_shared TRUE)
    else()
        set(_stinkytofu_shared FALSE)
    endif()

    target_link_libraries(_rocisa PUBLIC roc::origami)
    target_link_libraries(_rocisa PRIVATE hip::host stinkytofu::stinkytofu)

    # rocisa<->stinkytofu conversion glue. These sources live in the stinkytofu
    # source tree but are rocisa-side bindings (they include rocisa headers and
    # define init_stinkytofu), so they compile into _rocisa, not libstinkytofu.
    # rocisa owns them directly here rather than relying on stinkytofu to inject
    # them, so the de-bundled (find_package) build links them too.
    set(_stinkytofu_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
    target_sources(_rocisa PRIVATE
        "${_stinkytofu_src_dir}/src/conversion/rocisa/AllHwMappings.cpp"
        "${_stinkytofu_src_dir}/src/conversion/rocisa/ToStinkyTofuUtils.cpp"
    )

    # Compile StinkyTofu's example HelloWorldPass plugin straight into _rocisa, same
    # rationale as the conversion glue above: this bypasses whatever CMake target
    # topology stinkytofu itself used (static/shared, find_package/add_subdirectory),
    # so rocisa.registerExamplePlugin() (bound in ToStinkyTofuUtils.cpp) is available
    # uniformly. Separate from stinkytofu's own STINKYTOFU_BUILD_EXAMPLES option.
    if(ROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN)
        target_sources(_rocisa PRIVATE
            "${_stinkytofu_src_dir}/examples/plugins/HelloWorldPass.cpp"
        )
        target_include_directories(_rocisa PRIVATE
            "${_stinkytofu_src_dir}/examples/plugins"
        )
        target_compile_definitions(_rocisa PRIVATE ROCISA_HAVE_HELLOWORLD_STATIC_PLUGIN)
    endif()

    if(_stinkytofu_from_subdirectory)
        # Source-built stinkytofu via add_subdirectory: generated headers are in
        # the stinkytofu/ subdirectory of the binary dir.
        target_include_directories(_rocisa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
    endif()
    # _build_info.py is intentionally NOT in install() rules for production
    # wheels/packages, acting as the source-build sentinel. Dev/editable installs
    # pass -DROCISA_INCLUDE_BUILD_INFO=ON to include it; that flag also marks a
    # dev build, where libstinkytofu lives in a temp install prefix off the
    # loader's default search path.
    option(ROCISA_INCLUDE_BUILD_INFO "Install _build_info.py for staleness detection" OFF)

    # Directory containing libstinkytofu, resolved by CMake for the active
    # config. $<TARGET_FILE_DIR:...> handles both the add_subdirectory target
    # (build output dir) and the imported package target (IMPORTED_LOCATION_<CONFIG>)
    # without hand-matching configs.
    set(_st_lib_dir "$<TARGET_FILE_DIR:stinkytofu::stinkytofu>")
    set_target_properties(_rocisa PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocisa"
        BUILD_RPATH "${_st_lib_dir}"
    )
    if(_stinkytofu_from_subdirectory AND _stinkytofu_shared)
        # Source-built sibling: no installed libstinkytofu exists, so the wheel carries
        # its own copy next to _rocisa and loads it from there.
        set_target_properties(_rocisa PROPERTIES INSTALL_RPATH "$ORIGIN")
        # Windows has no RPATH; copy stinkytofu.dll next to _rocisa.pyd so the loader
        # finds it at runtime.
        if(WIN32)
            add_custom_command(TARGET _rocisa POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:stinkytofu::stinkytofu>
                    $<TARGET_FILE_DIR:_rocisa>
            )
        endif()
    elseif(_stinkytofu_shared AND ROCISA_INCLUDE_BUILD_INFO)
        # Dev/editable find_package build: libstinkytofu is installed to a temp
        # prefix off the loader's default search path, so bake its dir into the
        # install RPATH from the link line. Gated on the dev-build signal so
        # released ROCm builds stay RPATH-free and rely on libstinkytofu being
        # resident in rocm/lib.
        set_target_properties(_rocisa PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
    # In a released build, installed stinkytofu is a standard ROCm library on the
    # loader's search path (like libyaml behind PyYAML); _rocisa's DT_NEEDED
    # resolves it from there, so no wheel-local copy or RPATH is needed.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/__init__.py"
        COPYONLY
    )
    # STINKYTOFU_ROOT_DIR feeds @STINKYTOFU_ROOT_DIR@ in _build_info.py (the
    # staleness check scans stinkytofu sources too, since the conversion glue is
    # compiled into _rocisa). It is set via PARENT_SCOPE only in the
    # add_subdirectory build; in the de-bundled find_package build it is unset,
    # which would leave the path empty and make the check scan the CWD. Point it
    # at the real stinkytofu source tree.
    if(NOT STINKYTOFU_ROOT_DIR)
        cmake_path(ABSOLUTE_PATH _stinkytofu_src_dir NORMALIZE OUTPUT_VARIABLE STINKYTOFU_ROOT_DIR)
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/_build_info.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_build_info.py"
        @ONLY
    )
    if(ROCISA_INCLUDE_BUILD_INFO)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_build_info.py" DESTINATION rocisa)
    endif()

    # Windows: emit the resolved runtime DLL dependencies so __init__.py can
    # register their directories via os.add_dll_directory(). Since Python 3.8 the
    # extension loader ignores PATH, and in a source/integrated build _rocisa's
    # deps (origami, hip, comgr, stinkytofu) live in scattered per-subproject
    # directories. TARGET_RUNTIME_DLLS resolves the full set (direct + transitive)
    # from the linked/imported targets. Build-tree only, like _build_info.py; not
    # installed -- packaged builds resolve deps via the merged install layout.
    if(WIN32)
        file(GENERATE
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_dll_dirs.py"
            CONTENT "DEP_DLLS = [p for p in r'''$<TARGET_RUNTIME_DLLS:_rocisa>'''.split(';') if p]\n"
        )
    endif()

    # Stage the standalone stinkytofu Python module next to _rocisa so it
    # resolves from the build tree at runtime.
    if(TARGET stinkytofu_python)
        add_dependencies(_rocisa stinkytofu_python)

        set_target_properties(stinkytofu_python PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu"
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )

        configure_file(
            "${_stinkytofu_src_dir}/python_module/stinkytofu/__init__.py"
            "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/__init__.py"
            COPYONLY
        )
        # The staged _build_info.py drives the standalone _stinkytofu.so guard.
        configure_file(
            "${_stinkytofu_src_dir}/python_module/stinkytofu/_build_info.py.in"
            "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/_build_info.py"
            @ONLY
        )

        # Stage intrinsics.st.bc next to _stinkytofu.so for runtime loading.
        if(TARGET intrinsics_compiled)
            add_dependencies(_rocisa intrinsics_compiled)

            set(_intrinsics_bc_src "${CMAKE_BINARY_DIR}/intrinsics.st.bc")
            set(_intrinsics_bc_dst "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/intrinsics.st.bc")
            add_custom_command(
                OUTPUT "${_intrinsics_bc_dst}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_intrinsics_bc_src}" "${_intrinsics_bc_dst}"
                DEPENDS intrinsics_compiled "${_intrinsics_bc_src}"
                COMMENT "Staging intrinsics.st.bc next to _stinkytofu.so"
                VERBATIM
            )
            add_custom_target(rocisa_stage_intrinsics ALL
                DEPENDS "${_intrinsics_bc_dst}")
            add_dependencies(_rocisa rocisa_stage_intrinsics)
        endif()
    endif()

    if(ROCISA_STANDALONE)
        # scikit-build-core wheel install rules: place _rocisa and __init__.py into the
        # rocisa/ package directory in the wheel.
        install(TARGETS _rocisa DESTINATION rocisa)
        if(_stinkytofu_from_subdirectory AND _stinkytofu_shared)
            # Source-built: vendor libstinkytofu into the wheel — no installed copy exists.
            # A static stinkytofu is already inside _rocisa, and TARGET_SONAME_FILE below
            # is only valid for SHARED targets.
            install(FILES $<TARGET_FILE:stinkytofu::stinkytofu> DESTINATION rocisa)
            if(NOT WIN32)
                install(FILES $<TARGET_SONAME_FILE:stinkytofu::stinkytofu> DESTINATION rocisa)
            endif()
        endif()
        # Installed stinkytofu ships in its own ROCm package; it is not bundled here.
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py" DESTINATION rocisa)
    endif()

    # Example plugin — the target only exists when stinkytofu is configured with
    # -DSTINKYTOFU_BUILD_EXAMPLES=ON (default OFF; opted in by the source-built standalone
    # wheel via tasks.py). Integrated/ROCm builds leave it OFF, so the target is absent and
    # this block is skipped — the example never lands in a production ROCm package; there,
    # the installed stinkytofu package ships and locates its own plugin.
    #
    # In the source-built wheel, libstinkytofu is vendored into rocisa/, so the plugin must
    # sit at rocisa/${STINKYTOFU_PLUGIN_INSTALL_RELDIR} to match PassBuilder::examplePluginPath()'s
    # lookup relative to libstinkytofu. Consuming stinkytofu's exported reldir keeps the install
    # path in lockstep with the runtime lookup — no hardcoded duplicate.
    # No INSTALL_RPATH is needed: on Linux the plugin has no DT_NEEDED on libstinkytofu (it
    # is headers-only) and the host is already resident when loadPlugin() dlopens it.
    # Only installable for shared builds (MODULE target); static builds use OBJECT.
    if(TARGET stinkytofu-plugin-helloworld AND _stinkytofu_shared AND _stinkytofu_from_subdirectory)
        set_target_properties(stinkytofu-plugin-helloworld PROPERTIES EXCLUDE_FROM_ALL FALSE)
        add_dependencies(_rocisa stinkytofu-plugin-helloworld)
        install(TARGETS stinkytofu-plugin-helloworld
            DESTINATION "rocisa/${STINKYTOFU_PLUGIN_INSTALL_RELDIR}")
    endif()
endif()
