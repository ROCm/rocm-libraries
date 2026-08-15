# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# ============================================================================
# rocke AOT catalog families -- kernel authoring + build
# ============================================================================
# This subsystem is OWNED BY THE KERNEL-AUTHORING TEAMS. Each aot_catalog/<arch>/
# <family>/ folder is a fully self-contained unit: a checked-in family.json (the
# spec: kernels + constraints + ABI/grid), a co-located producer script that
# COMPILES the family's .co from that spec via rocke, a per-family CMakeLists.txt,
# and -- co-located here too -- that family's C++ acceptance tests. See README.md.
#
# The AOT catalog *engine* (src/engines/aot_catalog_engine/) is a separate,
# throwaway consumer: it only LOADS the .co this subsystem produces (described by
# family.json) at runtime and has ZERO build- or link-time dependency on rocke.
# Authoring lives here; loading lives there. Adding/removing a kernel family is a
# rocke-side change -- the engine never moves.
#
#   * rocke-AOT family: the per-family CMakeLists calls rocke_add_aot_family(),
#     which runs the producer at build time (via the rocke build interpreter,
#     ${ROCKE_PYENV_PYTHON}) to compile the .co from the spec. We do NOT check
#     those .co into git (they are churning build products); TheRock packages the
#     built output via install().
#   * prebuilt-.co / AITER-ASM family (future): the CMakeLists would just stage
#     checked-in .co -- no producer. (Add rocke_add_prebuilt_family() when the
#     first such family lands; see README.md.)
#
# The catalog build is only configured when BOTH HIPKERNELPROVIDER_ENABLE_ROCKE
# and ENABLE_AOT_CATALOG_ENGINE are ON (the parent rocke/CMakeLists.txt gates the
# add_subdirectory on ENABLE_AOT_CATALOG_ENGINE, and rocke/ itself is only added
# when ENABLE_ROCKE). When rocke is off the engine still builds against an empty
# catalog and its GPU parity tests skip. A build with only prebuilt families would
# still need no rocke producer.
#
# Families are AUTO-DISCOVERED: any aot_catalog/<arch>/<family>/CMakeLists.txt is
# picked up, so a kernel author adds a new arch/kernel set just by dropping in a
# folder (e.g. gfx942/sdpa_dense/).

# The shared catalog paths are set by the top-level hip-kernel-provider
# CMakeLists (single source of truth, also read by the runtime engine):
#   HIPDNN_AOT_CATALOG_BUILD_DIR   -- build-tree catalog root <root>/<arch>/<family>/
#   HIPDNN_AOT_CATALOG_INSTALL_DIR -- install (relative) catalog root
if(NOT DEFINED HIPDNN_AOT_CATALOG_BUILD_DIR)
    message(FATAL_ERROR "aot-catalog: HIPDNN_AOT_CATALOG_BUILD_DIR not set by parent")
endif()

# ---------------------------------------------------------------------------
# rocke availability (resolved once; per-family CMakeLists read _AOT_ROCKE_OK).
#
# The producer runs under the rocke build interpreter (${ROCKE_PYENV_PYTHON}, the
# build-local editable venv from rocke/CMakeLists.txt), so it needs neither
# PYTHONPATH surgery nor a separate rocke source-dir cache var. If the pyenv was
# disabled (ROCKE_BUILD_PYENV=OFF) the families are skipped with a warning; the
# engine still builds and its GPU parity tests skip on an empty catalog.
# ---------------------------------------------------------------------------
set(ROCKE_COMGR_LIB "" CACHE FILEPATH
    "Optional path to libamd_comgr.so for the rocke producer (else it uses \
ROCKE_COMGR_LIB from the environment or /opt/rocm).")

set(_AOT_ROCKE_OK TRUE)
if(NOT ROCKE_PYENV_PYTHON)
    set(_AOT_ROCKE_OK FALSE)
    message(WARNING
        "aot-catalog: the rocke build interpreter (ROCKE_PYENV_PYTHON) is not "
        "available -- rocke-AOT kernel families will be SKIPPED (engine still "
        "builds; GPU parity tests skip on the empty catalog). Enable ROCKE_BUILD_PYENV "
        "(the default) to build kernels.")
endif()

# The producer compiles .co in-process via libamd_comgr.so (rocke's comgr runtime),
# NOT hipcc. Resolve it the SAME way the producer will at build time -- explicit
# ROCKE_COMGR_LIB, else $ROCKE_COMGR_LIB from the environment, else the ROCm default
# -- so a box that has the pyenv but NO comgr SKIPS the families (build still
# succeeds) instead of hard-failing when the producer throws mid-build.
if(_AOT_ROCKE_OK)
    set(_aot_comgr "")
    if(ROCKE_COMGR_LIB AND EXISTS "${ROCKE_COMGR_LIB}")
        set(_aot_comgr "${ROCKE_COMGR_LIB}")
    elseif(DEFINED ENV{ROCKE_COMGR_LIB} AND EXISTS "$ENV{ROCKE_COMGR_LIB}")
        set(_aot_comgr "$ENV{ROCKE_COMGR_LIB}")
    else()
        find_library(_AOT_COMGR_LIB NAMES amd_comgr
            HINTS "$ENV{ROCM_PATH}/lib" "$ENV{ROCM_PATH}/lib64" /opt/rocm/lib /opt/rocm/lib64)
        if(_AOT_COMGR_LIB)
            set(_aot_comgr "${_AOT_COMGR_LIB}")
        endif()
    endif()

    # Windows: the producer loads comgr via ctypes.CDLL, which needs the RUNTIME
    # amd_comgr.dll -- but the resolution above (find_library especially) yields the
    # IMPORT library amd_comgr.lib, which ctypes cannot load. The hermetic TheRock
    # build failed exactly here ("cannot load amd_comgr.dll ...; tried
    # [.../stage/lib/amd_comgr.lib]") because we forwarded the .lib as the override.
    # Swap a resolved .lib for the sibling DLL: ROCm/TheRock stage the DLL in a bin/
    # dir beside the lib/ dir. If no loadable DLL turns up, drop to the skip path
    # below rather than fail mid-producer.
    if(WIN32 AND _aot_comgr MATCHES "\\.lib$")
        get_filename_component(_comgr_libdir "${_aot_comgr}" DIRECTORY)
        find_file(_AOT_COMGR_DLL NAMES amd_comgr.dll
            HINTS "${_comgr_libdir}" "${_comgr_libdir}/../bin"
                  "${_comgr_libdir}/../../dist/bin"
                  "$ENV{ROCM_PATH}/bin" "$ENV{HIP_PATH}/bin")
        if(_AOT_COMGR_DLL)
            set(_aot_comgr "${_AOT_COMGR_DLL}")
        else()
            set(_aot_comgr "")
        endif()
    endif()

    if(NOT _aot_comgr)
        set(_AOT_ROCKE_OK FALSE)
        message(WARNING
            "aot-catalog: the rocke pyenv is available but libamd_comgr.so was not "
            "(checked ROCKE_COMGR_LIB, \$ROCKE_COMGR_LIB, and the ROCm lib dirs) -- "
            "rocke-AOT families SKIPPED (engine still builds; GPU parity tests skip on "
            "the empty catalog). Set -DROCKE_COMGR_LIB=<path/to/libamd_comgr.so> to build kernels.")
    endif()

    # ---------------------------------------------------------------------------
    # Resolve comgr's TRANSITIVE dependency dirs (its LLVM/Clang/LLD shared libs).
    #
    # The amd_comgr CMake package's imported target carries ONLY comgr's own include
    # + library location -- NOTHING about its LLVM deps -- so forwarding the comgr
    # path is necessary but NOT sufficient. In a hermetic split-stage build (TheRock)
    # comgr and its LLVM libs live in SEPARATE stage dirs, so the producer's ctypes
    # loader fails ENOENT on an existing libamd_comgr (the deps, not comgr, are what
    # the loader can't find). C++ consumers in this repo (e.g. shared/rocroller) avoid
    # this by linking the LLVM libs so the dep dir is baked into RPATH; our producer is
    # a Python interpreter and can't, so we must put those dirs on its loader path.
    #
    # We can't read the comgr package config's dep dirs (it has none), and the LLVM
    # stage dir isn't derivable from a find_package target, so DISCOVER it: probe a
    # curated set of dirs near comgr (hermetic sibling stages + the ROCm llvm subdir)
    # and keep those that actually contain LLVM/Clang shared libs. Self-contained --
    # no TheRock-side var required. comgr's own dir is always included (co-staged deps).
    # ---------------------------------------------------------------------------
    set(_aot_comgr_dep_dirs "")
    if(_aot_comgr)
        get_filename_component(_comgr_dir "${_aot_comgr}" DIRECTORY)          # .../amd-comgr/stage/lib
        get_filename_component(_comgr_stage "${_comgr_dir}" DIRECTORY)        # .../amd-comgr/stage
        get_filename_component(_comgr_proj "${_comgr_stage}" DIRECTORY)       # .../amd-comgr
        get_filename_component(_compiler_root "${_comgr_proj}" DIRECTORY)     # .../compiler (hermetic)

        if(WIN32)
            set(_dep_probe "LLVM*.dll" "libLLVM*.dll" "clang-cpp*.dll" "libclang*.dll" "lld*.dll")
        else()
            set(_dep_probe "libLLVM*.so*" "libclang-cpp*.so*" "liblld*.so*")
        endif()

        # comgr's own dir first (co-staged deps), then sibling compiler stages that in
        # a hermetic tree hold amd-llvm's shared libs, then a real ROCm install layout.
        set(_dep_candidates
            "${_comgr_dir}"
            "${_comgr_dir}/../bin"
            "${_compiler_root}/amd-llvm/stage/lib"
            "${_compiler_root}/amd-llvm/dist/lib"
            "${_compiler_root}/amd-llvm/build/lib"
            "${_compiler_root}/amd-llvm/lib"
            "${_compiler_root}/amd-llvm/stage/bin"
            "${_compiler_root}/amd-llvm/dist/bin"
            "$ENV{ROCM_PATH}/lib" "$ENV{ROCM_PATH}/lib/llvm/lib" "$ENV{ROCM_PATH}/bin"
            "$ENV{HIP_PATH}/bin"
            /opt/rocm/lib /opt/rocm/lib/llvm/lib)
        foreach(_d IN LISTS _dep_candidates)
            if(IS_DIRECTORY "${_d}")
                set(_found_here FALSE)
                foreach(_p IN LISTS _dep_probe)
                    file(GLOB _hit "${_d}/${_p}")
                    if(_hit)
                        set(_found_here TRUE)
                        break()
                    endif()
                endforeach()
                # Always keep comgr's own dir (co-staged private deps); keep others only
                # when they actually contain LLVM/Clang libs.
                if(_found_here OR _d STREQUAL _comgr_dir)
                    get_filename_component(_dabs "${_d}" ABSOLUTE)
                    list(APPEND _aot_comgr_dep_dirs "${_dabs}")
                endif()
            endif()
        endforeach()

        # comgr / LLVM further depend on TheRock's BUNDLED sysdeps
        # (librocm_sysdeps_*.so.* -- zlib/zstd/etc.), which live in a rocm_sysdeps/lib
        # SUBDIR reached only via an $ORIGIN-relative RPATH. That subdir is on NEITHER
        # the curated dirs above NOR LD_LIBRARY_PATH (which is not recursive), so a comgr
        # whose LLVM deps now resolve STILL fails ENOENT on e.g. librocm_sysdeps_z.so.1.
        # Recursively locate the sysdeps under the comgr + amd-llvm project trees (bounded
        # to those two projects, NOT all of compiler/) and add their dirs -- one sweep
        # covers every bundled sysdeps lib. Non-hermetic/local builds simply find none.
        if(WIN32)
            set(_sysdep_glob "rocm_sysdeps_*.dll" "librocm_sysdeps_*.dll")
        else()
            set(_sysdep_glob "librocm_sysdeps_*.so*")
        endif()
        set(_sysdep_roots "${_comgr_proj}" "${_compiler_root}/amd-llvm")
        foreach(_root IN LISTS _sysdep_roots)
            if(IS_DIRECTORY "${_root}")
                foreach(_g IN LISTS _sysdep_glob)
                    file(GLOB_RECURSE _sysdep_hits "${_root}/${_g}")
                    foreach(_h IN LISTS _sysdep_hits)
                        get_filename_component(_hd "${_h}" DIRECTORY)
                        get_filename_component(_hd "${_hd}" ABSOLUTE)
                        list(APPEND _aot_comgr_dep_dirs "${_hd}")
                    endforeach()
                endforeach()
            endif()
        endforeach()

        if(_aot_comgr_dep_dirs)
            list(REMOVE_DUPLICATES _aot_comgr_dep_dirs)
            message(STATUS "aot-catalog: comgr dependency dirs -> ${_aot_comgr_dep_dirs}")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# _aot_arch_requested(<arch> <out_var>) -- is <arch> in this build's target set?
#
# TheRock/ROCm pass the requested GPU targets in GPU_TARGETS (AMDGPU_TARGETS is
# the legacy alias); a family builds only when its arch is among them, so a
# gfx942-only build does NOT compile the gfx1151 kernels (and vice versa). Both
# the family arch and each target are normalized to their gfxNNN base -- feature
# suffixes like ":xnack-" are dropped -- mirroring the runtime device match in
# hip_kernel_provider_common::getDeviceString. When NEITHER list is set (e.g. a
# standalone dev build that never specified targets), ALL families build: the
# original behavior, so nothing silently disappears.
#
# Any future family adder (e.g. rocke_add_prebuilt_family) should gate on this too.
# ---------------------------------------------------------------------------
function(_aot_arch_requested arch out_var)
    set(_targets "${GPU_TARGETS}")
    if(NOT _targets)
        set(_targets "${AMDGPU_TARGETS}")
    endif()
    if(NOT _targets)
        set(${out_var} TRUE PARENT_SCOPE) # no target list -> build everything
        return()
    endif()
    string(REGEX REPLACE ":.*$" "" _arch_base "${arch}")
    foreach(_t IN LISTS _targets)
        string(REGEX REPLACE ":.*$" "" _t_base "${_t}")
        if(_t_base STREQUAL _arch_base)
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} FALSE PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# rocke_add_aot_family(ARCH <arch> FAMILY <family> PRODUCER <script.py>)
#
# Runs the co-located producer at build time (under ${ROCKE_PYENV_PYTHON}) to emit
# the family's .co into ${HIPDNN_AOT_CATALOG_BUILD_DIR}/<arch>/<family>/, stages the
# checked-in family.json beside them, and installs the folder for packaging. No-op
# (with a STATUS note) when the arch is not in this build's GPU_TARGETS or when the
# rocke pyenv/comgr is unavailable.
# ---------------------------------------------------------------------------
function(rocke_add_aot_family)
    cmake_parse_arguments(F "" "ARCH;FAMILY;PRODUCER" "" ${ARGN})
    if(NOT F_ARCH OR NOT F_FAMILY OR NOT F_PRODUCER)
        message(FATAL_ERROR "rocke_add_aot_family: ARCH, FAMILY and PRODUCER are required")
    endif()

    set(_src ${CMAKE_CURRENT_SOURCE_DIR})
    set(_out ${HIPDNN_AOT_CATALOG_BUILD_DIR}/${F_ARCH}/${F_FAMILY})

    # Skip archs this build didn't ask for BEFORE touching rocke -- an unwanted
    # arch must not build even where rocke is available.
    _aot_arch_requested(${F_ARCH} _arch_ok)
    if(NOT _arch_ok)
        message(STATUS
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} "
            "(arch not in GPU_TARGETS/AMDGPU_TARGETS for this build)")
        return()
    endif()

    if(NOT _AOT_ROCKE_OK)
        message(STATUS "aot-catalog: skipping rocke family ${F_ARCH}/${F_FAMILY} (rocke unavailable)")
        return()
    endif()

    # Guard a partial/half-added family folder. The folder was auto-discovered by
    # its CMakeLists alone, but the build-time custom command below needs BOTH the
    # producer script and the checked-in family.json; if either is missing the
    # command fails cryptically mid-build (python can't find the producer, or the
    # family.json copy errors). Warn and skip -- mirroring the rocke-absence handling
    # above -- so an incomplete folder never breaks the rest of the catalog build.
    if(NOT EXISTS ${_src}/${F_PRODUCER})
        message(WARNING
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} -- producer script "
            "'${F_PRODUCER}' not found in ${_src} (partial family folder?)")
        return()
    endif()
    if(NOT EXISTS ${_src}/family.json)
        message(WARNING
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} -- family.json not found "
            "in ${_src} (partial family folder?)")
        return()
    endif()

    # Env for the producer: quiet the C++-engine fallback (the pure-Python lowerer
    # is used when the rocke_engine extension is absent), and -- crucially -- tell
    # the producer where comgr AND its transitive deps are. The producer runs under
    # ${ROCKE_PYENV_PYTHON}, a bare build-local venv with NEITHER the rocm-sdk wheel
    # NOR ROCM_PATH/HIP_PATH on its environment, so rocke.runtime.comgr's own
    # discovery has nothing to go on. We forward:
    #   * ROCKE_COMGR_LIB -- the resolved comgr path (the .so/.dll to dlopen).
    #   * LD_LIBRARY_PATH (POSIX) -- comgr's dep dirs, so the loader resolves comgr's
    #     LLVM/Clang deps. Forwarding the comgr path alone is NOT enough: comgr and
    #     its LLVM libs live in separate stage dirs in a hermetic build, so dlopen of
    #     an EXISTING libamd_comgr fails ENOENT on the unresolved deps. (POSIX only:
    #     the Windows path separator is ';', which collides with CMake list syntax.)
    #   * ROCKE_COMGR_DEP_DIRS ('|'-joined) -- same dep dirs for the Windows loader,
    #     which registers them via os.add_dll_directory (LD_LIBRARY_PATH is a no-op
    #     there). '|' avoids both the CMake list sep and the Windows path sep.
    #   * ROCM_PATH -- the ROCm root, when known.
    set(_env ROCKE_CPP_QUIET_FALLBACK=1)
    if(_aot_comgr)
        list(APPEND _env ROCKE_COMGR_LIB=${_aot_comgr})
    endif()
    if(_aot_comgr_dep_dirs)
        if(NOT WIN32)
            string(JOIN ":" _ld_extra ${_aot_comgr_dep_dirs})
            list(APPEND _env "LD_LIBRARY_PATH=${_ld_extra}:$ENV{LD_LIBRARY_PATH}")
        endif()
        string(JOIN "|" _dep_joined ${_aot_comgr_dep_dirs})
        list(APPEND _env "ROCKE_COMGR_DEP_DIRS=${_dep_joined}")
    endif()
    if(ROCM_PATH)
        list(APPEND _env ROCM_PATH=${ROCM_PATH})
    endif()

    set(_stamp ${CMAKE_CURRENT_BINARY_DIR}/${F_FAMILY}.stamp)
    add_custom_command(
        OUTPUT ${_stamp}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_out}
        COMMAND ${CMAKE_COMMAND} -E env ${_env}
                ${ROCKE_PYENV_PYTHON} ${_src}/${F_PRODUCER} ${_out}
        COMMAND ${CMAKE_COMMAND} -E copy ${_src}/family.json ${_out}/family.json
        COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
        # The rocke pyenv stamp gates the interpreter being ready (editable installs
        # done); rebuild when the producer or the spec change.
        DEPENDS ${ROCKE_PYENV_STAMP} ${_src}/${F_PRODUCER} ${_src}/family.json
        COMMENT "aot-catalog: building ${F_ARCH}/${F_FAMILY} kernels via rocke"
        VERBATIM
    )
    add_custom_target(aot_family_${F_ARCH}_${F_FAMILY} ALL DEPENDS ${_stamp}
        COMMENT "aot-catalog: ${F_ARCH}/${F_FAMILY} family (kernels + family.json)")

    # Record the target so consumers (e.g. the test binary in src/tests) can depend
    # on the whole set -- an isolated `ninja hip_kernel_provider_tests` must build the
    # catalog first, else the GPU parity tests silently skip on an empty build-tree
    # catalog. src/tests reads this GLOBAL property (populated here during
    # add_subdirectory(rocke), before add_subdirectory(src)).
    set_property(GLOBAL APPEND PROPERTY
        AOT_CATALOG_FAMILY_TARGETS aot_family_${F_ARCH}_${F_FAMILY})

    # cmake-lint: disable=E1122
    install(DIRECTORY ${_out}/
            DESTINATION ${HIPDNN_AOT_CATALOG_INSTALL_DIR}/${F_ARCH}/${F_FAMILY}
            FILES_MATCHING PATTERN "*.co" PATTERN "family.json")
endfunction()

# ---------------------------------------------------------------------------
# rocke_add_aot_family_test(ARCH <arch> SOURCES <test.cpp> [<test2.cpp> ...])
#
# A family declares its OWN tests here, co-located in the family folder, so the
# family folder is a fully self-contained unit: family.json (the spec) + the
# CMakeLists that builds its .co + the tests that validate those .co, all in one
# place. Deleting aot_catalog/<arch>/<family>/ removes the kernels, the spec AND
# the tests together -- nothing about that family is left dangling elsewhere.
#
# The named sources are appended (as absolute paths) to the
# AOT_CATALOG_FAMILY_TEST_SOURCES global property, which src/tests/CMakeLists.txt
# compiles into hip_kernel_provider_tests. The tests #include the engine's private
# headers, so src/tests -- NOT this subsystem -- owns the compile (it adds the
# engine include dir + the AOT_CATALOG_TEST_DIR compile def); this keeps rocke free
# of any build-graph dependency on the throwaway engine. Gated on the same
# _aot_arch_requested() check as the kernel build, so a family's tests compile only
# when its arch is in this build's target set.
#
# CONTRACT: a family's tests must depend ONLY on that family -- no test here may
# reference another family's kernels or symbols, so removing any one family can
# never break another's tests. Cross-family or engine-substrate tests (e.g. the
# tune-cache and SDPA-decode unit tests) do NOT belong in a family folder; they
# live at the engine test level (src/tests/engines/aot_catalog_engine/).
# ---------------------------------------------------------------------------
function(rocke_add_aot_family_test)
    cmake_parse_arguments(T "" "ARCH" "SOURCES" ${ARGN})
    if(NOT T_ARCH OR NOT T_SOURCES)
        message(FATAL_ERROR "rocke_add_aot_family_test: ARCH and SOURCES are required")
    endif()

    _aot_arch_requested(${T_ARCH} _arch_ok)
    if(NOT _arch_ok)
        return() # arch not in this build's target set -> its tests don't compile
    endif()

    foreach(_s IN LISTS T_SOURCES)
        set_property(GLOBAL APPEND PROPERTY
            AOT_CATALOG_FAMILY_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${_s})
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# Auto-discover every aot_catalog/<arch>/<family>/CMakeLists.txt.
# ---------------------------------------------------------------------------
file(GLOB _family_cmakes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*/*/CMakeLists.txt)
foreach(_fc ${_family_cmakes})
    get_filename_component(_fdir ${_fc} DIRECTORY)
    add_subdirectory(${_fdir})
endforeach()
