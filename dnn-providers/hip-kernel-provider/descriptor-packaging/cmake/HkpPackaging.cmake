# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build-time hip UKD -> compile -> prune -> kpack packaging for the
# hip-kernel-provider. All functions are provider-internal and namespaced hkp_*.
# hkp = Hip Kernel-provider Packaging; the hkp_/HKP_ prefixes mark internal
# symbols of this kpack-packaging module.

include_guard(GLOBAL)

# Captured at include time so it survives into functions: inside a function
# CMAKE_CURRENT_LIST_DIR reflects the invoking listfile, not this module.
set(HKP_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(HKP_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(HKP_PYTHON_ROOT "${HKP_PKG_DIR}/python")
set(HKP_TOOL "${HKP_PKG_DIR}/tools/hkp_pack.py")
set(HKP_WHEEL_DIGEST_TOOL "${HKP_PKG_DIR}/tools/hkp_wheel_digest.py")
set(HKP_FIXTURES "${HKP_PKG_DIR}/tests/fixtures")

include(KpackPython)

# ---------------------------------------------------------------------------
# hkp_resolve_kpack(<out_var> <python_exe>)
#   Resolve the rocm_kpack python dir, or hard-fail: this pipeline cannot pack
#   without it, so there is no skip path.
#
#   Also verifies <python_exe> can import it. Resolution only proves the
#   directory exists; the import still fails when the interpreter differs from
#   the one the tree's compiled msgpack/zstandard extensions were built for.
#   Probing here reports that at configure time instead of mid-build.
# ---------------------------------------------------------------------------
function(hkp_resolve_kpack out_var python_exe)
    kpack_resolve_python_dir(_python_dir)
    if("${_python_dir}" STREQUAL "")
        kpack_unset_reason(_reason)
        message(FATAL_ERROR "hkp: ${_reason}. rocm_kpack is required to pack "
            "descriptors; there is no skip path.")
    endif()
    kpack_check_python_deps("${python_exe}" "${_python_dir}" _missing)
    if(_missing)
        string(REPLACE ";" ", " _missing_csv "${_missing}")
        message(FATAL_ERROR
            "hkp: ${python_exe} cannot import ${_missing_csv} (rocm_kpack "
            "needs zstandard>=0.20.0 and msgpack). If the resolved tree was "
            "staged for a different Python, install the dependencies for this "
            "interpreter or point -DPython3_EXECUTABLE at the one they were "
            "built for.")
    endif()
    set(${out_var} "${_python_dir}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_selected_arches(<out_var>)
#   Normalize GPU_TARGETS (or AMDGPU_TARGETS) into a bare gfx arch list,
#   stripping feature suffixes (gfx942:xnack-) and dropping anything that is not
#   a concrete gfx name. Empty result is legal (install nothing, non-error). No
#   intersection with a fixed fixture set: the tool compiles from authored
#   sources for whatever arch is requested.
#
#   The only consumer of GPU_TARGETS in dnn-providers/. The sibling kpack
#   producer, src/engines/asm_sdpa_engine/CMakeLists.txt, declares an explicit
#   list instead because it globs prebuilt .co files; this step compiles from
#   source and can target any real gfx, so it reads GPU_TARGETS.
#
#   Elsewhere in this repo a gfxNNX-style label is a selector matched against a
#   concrete arch (shared/ctest/parse_test_categories.py,
#   test/therock/test_runner.py, .github/scripts/amdgpu_family_matrix.py); here
#   the value reaches hipcc's --offload-arch, where a family name is unusable
#   rather than coarse. Hence drop-with-warning, not passthrough, and no
#   family-to-arch expansion table.
# ---------------------------------------------------------------------------
function(hkp_selected_arches out_var)
    set(_targets "")
    set(_source "")
    if(DEFINED GPU_TARGETS AND GPU_TARGETS)
        set(_targets ${GPU_TARGETS})
        set(_source "GPU_TARGETS")
    elseif(DEFINED AMDGPU_TARGETS AND AMDGPU_TARGETS)
        set(_targets ${AMDGPU_TARGETS})
        set(_source "AMDGPU_TARGETS")
    endif()

    set(_selected "")
    foreach(_arch IN LISTS _targets)
        string(REGEX REPLACE ":.*$" "" _bare "${_arch}")
        if(NOT _bare)
            continue()
        endif()
        if(NOT _bare MATCHES "^gfx[0-9a-f]+$")
            message(WARNING
                "hkp: ignoring '${_arch}' from ${_source}; it is not a concrete gfx "
                "architecture and cannot be passed to hipcc --offload-arch. Nothing "
                "is packed for it. Name real gfx architectures in ${_source} to pack "
                "for them.")
            continue()
        endif()
        list(APPEND _selected "${_bare}")
    endforeach()
    list(REMOVE_DUPLICATES _selected)
    set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_wire_pack_target(NAME <label> SOURCE_ROOT <dir>
#               ENABLE_ROCKE <bool> ARCHES <list> HIPCC <path>
#               ROCM_KPACK_DIR <dir> OUT_ROOT <dir>
#               [REQUIRE_HIPCC <bool>]
#               [ROCKE_INTERP <path>] [ROCKE_COMGR_LIB <path>]
#               [ROCKE_WHEEL_STAMP <path>])
#   Wire the compile -> prune -> pack DAG for ONE authored source root.
#
#   The root is walked recursively. Each descriptor's authored
#   subpath is preserved into the packed tree. Producer selection is per-UKD on
#   kernel_source.kind, never per-folder, so one root feeds all producers into
#   ONE kpack per arch.
#
#   OUT_ROOT is where the packer writes: one output folder, wiped and filled by
#   this invocation alone. No two invocations may share a destination. One
#   source root may be invoked more than once, into different output roots.
#   Installation is not wired here -- a root delivers into arch_content/ or
#   test_arch_content/ in the build tree, and those two trees are installed
#   wholesale by hip-kernel-provider/CMakeLists.txt.
#
#   What actually differs between roots is declared, not forked into a second
#   function:
#
#     REQUIRE_HIPCC ON makes a missing hipcc a configure error, OFF warns and
#                   skips. A release must not silently ship nothing; a test
#                   fixture must not break a developer's build.
#
#   ENABLE_ROCKE says whether the rocKE producer may run.
#   When ON the tool runs under ROCKE_INTERP (wheel-provisioned so
#   `import rocke`/`kernels` resolve) and ROCKE_COMGR_LIB, if set,
#   is forwarded to the tool environment. ROCKE_WHEEL_STAMP is the wheel
#   content digest: the pack step depends on it so that editing a kernel under
#   rocke/library rebuilds the wheel, changes the digest, and RESTAGES the
#   packaged artifacts. Without that edge the kpacks would keep shipping kernels
#   compiled from stale wheel contents.
#
#   Empty ARCHES wires nothing.
# ---------------------------------------------------------------------------
function(hkp_wire_pack_target)
    set(_one NAME SOURCE_ROOT ENABLE_ROCKE ARCHES HIPCC ROCM_KPACK_DIR
        OUT_ROOT REQUIRE_HIPCC ROCKE_INTERP ROCKE_COMGR_LIB
        ROCKE_WHEEL_STAMP)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${_one}" "")

    if(NOT ARG_ARCHES)
        message(WARNING
            "hkp: no GPU architectures are selected, so '${ARG_NAME}' root "
            "(${ARG_SOURCE_ROOT}) packs nothing. Set GPU_TARGETS or AMDGPU_TARGETS.")
        return()
    endif()
    if(NOT IS_DIRECTORY "${ARG_SOURCE_ROOT}")
        message(WARNING
            "hkp: source root '${ARG_NAME}' not found at ${ARG_SOURCE_ROOT}; "
            "nothing is packed for it.")
        return()
    endif()
    if(NOT ARG_OUT_ROOT)
        message(WARNING
            "hkp: no output directory given, so '${ARG_NAME}' root "
            "(${ARG_SOURCE_ROOT}) has nowhere to write and is skipped.")
        return()
    endif()
    if(NOT ARG_HIPCC)
        # Fatal for a shipping root, skip for a fixture: see REQUIRE_HIPCC above.
        if(ARG_REQUIRE_HIPCC)
            message(FATAL_ERROR
                "hkp: source root '${ARG_NAME}' requires hipcc but it was not "
                "found (searched hipcc, hipcc.bat, hipcc.bin.exe). Ensure the "
                "ROCm bin dir is on PATH or CMAKE_PROGRAM_PATH.")
        endif()
        message(WARNING
            "hkp: hipcc not found (searched hipcc, hipcc.bat, hipcc.bin.exe); "
            "source root '${ARG_NAME}' cannot be compiled and will not be "
            "staged. Tests that load it will skip. Put the ROCm bin dir on PATH "
            "to build it.")
        return()
    endif()

    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-${ARG_NAME}-intermediate")
    # Kept in the binary dir rather than beside the output: both the output and the
    # intermediate roots are wiped before each pack, which would take a stamp inside
    # either of them with it.
    set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/hkp-${ARG_NAME}-descriptors.stamp")

    # The rocKE producer needs the wheel-provisioned interpreter so its UKDs
    # import; a hip-only pack runs under the base interpreter (hip compiles shell
    # out to hipcc and are interpreter-agnostic).
    set(_interp "${Python3_EXECUTABLE}")
    set(_interp_what "base interpreter (hip-only, root '${ARG_NAME}')")
    set(_interp_dep "")
    set(_wheel_dep "")
    if(ARG_ENABLE_ROCKE)
        set(_interp "${ARG_ROCKE_INTERP}")
        set(_interp_what "rocKE wheel interpreter (root '${ARG_NAME}')")
        set(_interp_dep "${ARG_ROCKE_INTERP}")
        # Record the wheel digest in each rocKE UKD's provenance, so a shipped
        # kernel names the wheel that produced it, and depend on it so editing a
        # kernel restages. Only meaningful when the rocKE producer is on; the hip
        # half has no wheel.
        set(_wheel_dep "${ARG_ROCKE_WHEEL_STAMP}")
    endif()

    # Tool environment. Two backend pins belong here, alongside the in-process
    # ones the producer sets:
    #
    #   ROCKE_BACKEND=python   -- belt to the producer's backend= kwarg. The
    #     kwarg is not threaded down; compile_kernel MUTATES os.environ around
    #     the call because lower_kernel_via_backend calls resolve_backend() with
    #     no argument. Setting the env var directly makes the pin survive that
    #     indirection changing.
    #   ROCKE_CPP_STRICT=1     -- turns a silent cpp->python degradation into a
    #     hard BackendError at the point of failure. It does not fire on an
    #     explicit python request, so the two pins compose.
    #
    # ROCKE_CPP_QUIET_FALLBACK is deliberately unset: silencing that warning
    # hides the degradation these pins exist to catch.
    #
    # ROCKE_COMGR_LIB overrides a shadowed System32 amd_comgr on Windows; forward
    # it when set (runtime resolution, no find_library).
    set(_tool_env "ROCKE_BACKEND=python" "ROCKE_CPP_STRICT=1")
    if(ARG_ROCKE_COMGR_LIB)
        list(APPEND _tool_env "ROCKE_COMGR_LIB=${ARG_ROCKE_COMGR_LIB}")
    endif()

    # The authored root is a tree: glob recursively so a descriptor added in any
    # child folder retriggers the pack step. The packer itself walks recursively
    # so a flat glob here would drop the dependency edge for every nested descriptor.
    file(GLOB_RECURSE _source_inputs CONFIGURE_DEPENDS
         "${ARG_SOURCE_ROOT}/*")

    # Editing the tool's own sources must retrigger the pack step, else the
    # artifacts go stale against the current pipeline code. The resolved
    # rocm_kpack package counts too: kpack_resolver.py imports it and it decides
    # the archive format, so a packer change there must invalidate the stamp.
    file(GLOB _tool_sources CONFIGURE_DEPENDS
         "${HKP_PYTHON_ROOT}/hkp_pack/*.py"
         "${ARG_ROCM_KPACK_DIR}/rocm_kpack/*.py")

    hkp_require_kpack_runtime("${_interp}" "the ${_interp_what}")

    string(REPLACE ";" "," _arch_csv "${ARG_ARCHES}")

    set(_wheel_stamp_arg "")
    if(_wheel_dep)
        set(_wheel_stamp_arg --rocke-wheel-stamp "${_wheel_dep}")
    endif()

    set(_tool_cmd "${CMAKE_COMMAND}" -E env ${_tool_env} "${_interp}"
        "${HKP_TOOL}")

    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${ARG_OUT_ROOT}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_inter_root}"
        COMMAND ${_tool_cmd}
                --source-root "${ARG_SOURCE_ROOT}"
                --out-root "${ARG_OUT_ROOT}"
                --arches "${_arch_csv}"
                --hipcc "${ARG_HIPCC}"
                --inter-root "${_inter_root}"
                --kpack-python-dir "${ARG_ROCM_KPACK_DIR}"
                ${_wheel_stamp_arg}
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${HKP_TOOL}" ${_source_inputs} ${_tool_sources}
                ${_interp_dep} ${_wheel_dep}
        COMMENT "hkp: packing root '${ARG_NAME}' for ${ARG_ARCHES}"
        VERBATIM)

    add_custom_target(hkp_packaging_${ARG_NAME} ALL
                      DEPENDS "${_stamp}"
                      COMMENT "hkp: descriptor packaging (${ARG_NAME})")
endfunction()

# ---------------------------------------------------------------------------
# hkp_probe_comgr_resolvable(<out_ok> <out_detail>)
#   Configure-time gate for the rocKE producer, scoped to what is knowable at
#   configure time.
#
#   An earlier version imported `rocke, kernels` from the SOURCE tree under the
#   base interpreter. That validated a different mechanism than the build uses
#   (the build imports from wheels installed in the hkp venv), so it could pass
#   and the build then fail. The import check now runs where it belongs -- in
#   the provisioned venv, as the last step of hkp_rocke_wheel_python_interp --
#   because neither the venv nor the wheels exist until the build runs.
#
#   Scope caveat: rocKE treats ROCKE_COMGR_LIB as a candidate, not an assertion
#   (`runtime/comgr.py` iterates candidates and the first loadable path wins),
#   so a bogus override falls through to the system comgr and this probe still
#   reports success. The probe answers "can this machine lower a kernel at all",
#   which is the common misconfiguration and is knowable at configure time; a
#   wrong-but-loadable library is caught by the provenance the packer records.
#
#   The probe reads the resolver from the source tree deliberately: it asks
#   about the machine's comgr, not about the wheels.
# ---------------------------------------------------------------------------
function(hkp_probe_comgr_resolvable out_ok out_detail)
    set(_rocke_root "${HKP_PKG_DIR}/../rocke")
    if(WIN32)
        set(_sep ";")
    else()
        set(_sep ":")
    endif()
    set(_pp "${_rocke_root}/platform/python${_sep}${_rocke_root}/library")
    # Probe under the SAME override the build will use, so configure and build
    # ask the same question. Without this a machine that only resolves comgr via
    # the override would fail configure despite being correctly configured.
    set(_probe_env "PYTHONPATH=${_pp}")
    if(HIPKERNELPROVIDER_ROCKE_COMGR_LIB)
        list(APPEND _probe_env
             "ROCKE_COMGR_LIB=${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${_probe_env}
                "${Python3_EXECUTABLE}" -c
                "from rocke.runtime import comgr; comgr._resolve_lib()"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(_rc EQUAL 0)
        set(${out_ok} TRUE PARENT_SCOPE)
        set(${out_detail} "" PARENT_SCOPE)
    else()
        set(${out_ok} FALSE PARENT_SCOPE)
        string(STRIP "${_err}${_out}" _detail)
        set(${out_detail} "${_detail}" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# hkp_rocke_wheel_stamp(<out_stamp>)
#   Maintain a content digest of the rocke wheels, rewritten ONLY when the
#   wheels' bytes change.
#
#   ROCKE_WHEEL_VERSION is pinned at 0.1.0 and never bumps, so the wheel
#   filenames are constant and `pip wheel` rewrites both files every build.
#   Keying the venv and the pack step on wheel mtime would therefore recompile
#   every kernel for every arch on every build, even when the wheels are
#   byte-identical. Keying on this stamp instead means a rebuild that produces
#   identical wheels leaves the stamp's mtime untouched, and Ninja's restat
#   (which CMake emits for add_custom_command OUTPUT edges) prunes everything
#   downstream.
#
#   Declared as BYPRODUCTS rather than OUTPUT precisely because the script may
#   legitimately not write it; an OUTPUT that the command sometimes leaves alone
#   makes Ninja rerun the edge every build.
# ---------------------------------------------------------------------------
function(hkp_rocke_wheel_stamp out_stamp)
    set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-wheels.sha256")
    set(_platform_wheel
        "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    set(_library_wheel
        "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")

    add_custom_target(hkp_rocke_wheel_digest ALL
        BYPRODUCTS "${_stamp}"
        COMMAND "${Python3_EXECUTABLE}" "${HKP_WHEEL_DIGEST_TOOL}"
                --stamp "${_stamp}"
                --wheel "${_platform_wheel}"
                --wheel "${_library_wheel}"
        DEPENDS "${_platform_wheel}" "${_library_wheel}"
                "${HKP_WHEEL_DIGEST_TOOL}"
        COMMENT "hkp: digesting rocke wheels"
        VERBATIM)

    set(${out_stamp} "${_stamp}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_require_kpack_runtime(<interp> <what>)
#   rocm_kpack is reached by putting a source tree on sys.path, so pip never
#   resolves the msgpack/zstandard it declares. Any interpreter that runs the
#   pack step therefore needs them present independently, and a hip-only pack
#   runs under the BASE interpreter where nothing provisions anything.
#
#   Checked at configure time because the failure is otherwise a mid-build
#   ImportError from inside a dependency, which reads as a packer bug rather
#   than a missing dependency on the build machine.
#
#   Only interpreters that ALREADY EXIST can be probed. The rocKE wheel venv is
#   an add_custom_command OUTPUT, so on a clean tree it is not created until the
#   build runs and probing it here would fail every configure with a message
#   blaming absent dependencies -- advice that cannot be followed, because there
#   is no interpreter to install them into. That venv installs these same two
#   packages itself and re-affirms the import after provisioning, so skipping it
#   here loses no coverage.
# ---------------------------------------------------------------------------
function(hkp_require_kpack_runtime interp what)
    if(NOT EXISTS "${interp}")
        # Provisioned during the build (the rocKE wheel venv), which validates
        # its own imports once it exists.
        return()
    endif()

    execute_process(
        COMMAND "${interp}" -c "import msgpack, zstandard"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        string(STRIP "${_err}" _err)
        message(FATAL_ERROR
            "hkp: ${what} cannot import rocm_kpack's runtime dependencies "
            "(msgpack, zstandard), so the pack step would fail mid-build. "
            "rocm_kpack is used from a source tree, so pip never installs the "
            "dependencies it declares -- install them into the interpreter at "
            "${interp}:\n"
            "    ${interp} -m pip install 'msgpack>=1.0.0' 'zstandard>=0.20.0'\n"
            "Python said: ${_err}")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# hkp_rocke_wheel_python_interp(<out_interp> <wheel_stamp>)
#   Provision a build-local interpreter carrying the rocke + rocke_library
#   wheels (built by the rocke-wheels target, which requires
#   HIPKERNELPROVIDER_ENABLE_ROCKE). The production tool imports rocke/kernels
#   from these installed wheels rather than the editable dev venv. The venv
#   interpreter is an add_custom_command OUTPUT so the production command can
#   depend on it for build ordering.
#
#   The venv is HERMETIC:
#     - no --system-site-packages: the dev venv inherits it to pick up the
#       system ROCm torch, but torch is not a build dependency. Inheriting the
#       system environment is how a build silently starts depending on whatever
#       happens to be installed on the machine.
#     - no `pip install --upgrade pip`: unconditional network access on every
#       provisioning run, to install two local files.
#     - --no-index: hermeticity enforced by the build rather than assumed.
#     - --no-deps: rocke declares numpy>=1.24 and rocke-library declares rocke.
#       Verified that the whole build path -- import rocke, import kernels,
#       build_attention_dense, and the comgr entry rocke.helpers.compile_kernel
#       -- works with neither installed; numpy is imported only by examples/,
#       heuristics/, benchmark/ and runtime/, which lowering never touches. The
#       dependency goes deliberately unsatisfied: nothing vendored, nothing
#       fetched. Should a future kernel import numpy at build time, the failure
#       is a loud ImportError naming the module rather than a silent pull from
#       an index.
#     - --force-reinstall: pip treats a same-name/same-version wheel as already
#       satisfied and leaves the OLD bytes in place. Since the version never
#       bumps, this flag is what makes a changed wheel actually land.
#
#   Depends on the wheel digest stamp, not the wheels, so a byte-identical
#   rebuild does not reprovision.
#
#   The rocke import check runs HERE, as the last step, rather than at configure
#   time: the venv and the wheels are both add_custom_command outputs that do
#   not exist until the build runs, so there is nothing to probe at configure
#   time. Running it in the provisioned venv also means it validates exactly
#   what the pack step will import, which the configure-time source-tree probe
#   did not.
# ---------------------------------------------------------------------------
function(hkp_rocke_wheel_python_interp out_interp wheel_stamp)
    set(_venv "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-venv")
    if(WIN32)
        set(_venv_py "${_venv}/Scripts/python.exe")
    else()
        set(_venv_py "${_venv}/bin/python")
    endif()

    set(_platform_wheel
        "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    set(_library_wheel
        "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")

    # rocm_kpack's runtime dependencies. The packer reaches rocm_kpack by path on
    # sys.path rather than by pip-installing it, so nothing resolves the
    # `msgpack>=1.0.0` / `zstandard>=0.20.0` it declares in its own
    # pyproject.toml -- and `import rocm_kpack.kpack` fails without them. The
    # previous venv used --system-site-packages and inherited whatever the host
    # happened to have; making the venv hermetic removed that accident, so the
    # dependency has to be declared here instead of relied upon.
    #
    # These come from the index, unlike the rocke wheels: they are third-party
    # packages with no local artifact to install from. The install is scoped to
    # exactly these two pinned-floor names, so the venv stays reproducible in
    # everything that describes OUR code.
    add_custom_command(
        OUTPUT "${_venv_py}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_venv}"
        COMMAND "${Python3_EXECUTABLE}" -m venv --copies "${_venv}"
        COMMAND "${_venv_py}" -m pip install -q
                "msgpack>=1.0.0" "zstandard>=0.20.0"
        COMMAND "${_venv_py}" -m pip install -q
                --no-index --no-deps --force-reinstall
                "${_platform_wheel}" "${_library_wheel}"
        # Probe what the pack step will actually import, in the interpreter it
        # will actually use -- rocke/kernels AND the kpack stack.
        COMMAND "${_venv_py}" -c
                "import rocke, kernels, msgpack, zstandard"
        DEPENDS "${wheel_stamp}" "${HKP_WHEEL_DIGEST_TOOL}"
        COMMENT "hkp: provisioning hermetic rocke wheel interpreter"
        VERBATIM)

    add_custom_target(hkp_rocke_wheel_python_interp ALL DEPENDS "${_venv_py}"
                      COMMENT "hkp: rocke wheel python interpreter")
    add_dependencies(hkp_rocke_wheel_python_interp hkp_rocke_wheel_digest)
    set(${out_interp} "${_venv_py}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Gate production packaging on ONE source root plus explicit per-producer
#   switches. The root names a location; the switches say which producers may
#   run.
#
#   Root empty = production packaging dormant. Root set but not a directory =
#   fatal. Root set with neither switch on = fatal. Configure hard-fails when an
#   ON switch's configure-discoverable toolchain is missing (hip -> hipcc;
#   rocke -> the ENABLE_ROCKE + wheel-env + importable conjunction). The tests
#   are wired regardless.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_rocm_kpack_dir "${Python3_EXECUTABLE}")
    hkp_selected_arches(_arches)

    # hipcc is the perl/bat driver that honors --genco; on Windows it is
    # hipcc.exe or hipcc.bat. hipcc.bin.exe is the raw clang driver and is only
    # a last-resort fallback.
    find_program(HKP_HIPCC NAMES hipcc hipcc.bat hipcc.bin.exe)

    set(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT "" CACHE PATH
        "The authored source root the production pack step compiles from. \
Walked recursively; child folders under it scope the content (hip/, rocKE/, \
per-integration folders) and each descriptor's authored subpath is preserved \
into the staged and installed trees. Empty leaves production packaging dormant.")
    set(HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP OFF CACHE BOOL
        "Allow the hip producer to run over the production source root. \
Requires a configure-discoverable hipcc.")
    set(HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE OFF CACHE BOOL
        "Allow the rocKE producer to run over the production source root. \
Requires HIPKERNELPROVIDER_ENABLE_ROCKE, the rocke wheel-env, and importable \
rocke/kernels packages.")

    set(_source_root "")
    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but is "
                "not a directory: ${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
        endif()
        set(_source_root "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
    endif()

    # A root with no producer enabled would pack nothing and install an empty
    # tree — the silent-empty-package failure mode. Name both switches so the
    # fix is obvious.
    if(_source_root
       AND NOT HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP
       AND NOT HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but no "
            "producer is enabled — set HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP "
            "and/or HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE, otherwise the "
            "pack step would ship an empty tree.")
    endif()

    # The hip producer requires the configure-discoverable hipcc.
    if(_source_root AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP AND NOT HKP_HIPCC)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP is ON but hipcc was "
            "not found (searched hipcc, hipcc.bat, hipcc.bin.exe). Ensure the "
            "ROCm bin dir is on PATH or CMAKE_PROGRAM_PATH.")
    endif()

    set(HIPKERNELPROVIDER_ROCKE_COMGR_LIB "" CACHE PATH
        "Explicit libamd_comgr for the rocKE producer to load. Forwarded into \
ROCKE_COMGR_LIB for the pack step and the ctest entries. Needed on Windows, \
where a System32 amd_comgr.dll can shadow the ROCm one; empty lets rocke \
resolve normally. Note rocke treats this as the first CANDIDATE, not an \
assertion: an unloadable path silently falls through to the next candidate.")

    # The rocKE producer requires the full conjunction: ENABLE_ROCKE (which
    # builds the wheels), the wheel-env available, and rocke/kernels importable.
    # Any missing piece is a configure hard-fail naming what is missing.
    set(_rocke_interp "")
    # ROCKE_COMGR_LIB is rocke's RUNTIME environment variable (comgr.py:66,99,
    # core.cpp:471). Nothing in this repository ever set() or option()s it, so
    # reading it here was reading an always-empty variable and the forwarding
    # below was unreachable. Take the value from a cache variable of our own and
    # forward THAT into the environment rocke reads.
    set(_rocke_comgr_lib "${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}")
    set(_enable_rocke OFF)
    if(_source_root AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
        set(_enable_rocke ON)
        if(NOT HIPKERNELPROVIDER_ENABLE_ROCKE)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but "
                "HIPKERNELPROVIDER_ENABLE_ROCKE is OFF — enable it so the "
                "rocke/kernels wheels are built and importable.")
        endif()
        if(NOT ROCKE_WHEEL_DIR)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but the "
                "rocke wheel-env is not available (ROCKE_WHEEL_DIR unset).")
        endif()
        hkp_probe_comgr_resolvable(_comgr_ok _comgr_detail)
        if(NOT _comgr_ok)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but comgr "
                "could not be resolved, so no rocKE kernel can be lowered. Set "
                "HIPKERNELPROVIDER_ROCKE_COMGR_LIB to an explicit "
                "libamd_comgr, or make one discoverable. Resolver said:\n"
                "${_comgr_detail}")
        endif()
        hkp_rocke_wheel_stamp(_rocke_wheel_stamp)
        hkp_rocke_wheel_python_interp(_rocke_interp "${_rocke_wheel_stamp}")
    endif()

    # Production descriptors.
    if(_source_root)
        hkp_wire_pack_target(
            NAME product
            SOURCE_ROOT "${_source_root}"
            ENABLE_ROCKE "${_enable_rocke}"
            ARCHES "${_arches}"
            HIPCC "${HKP_HIPCC}"
            ROCM_KPACK_DIR "${_rocm_kpack_dir}"
            OUT_ROOT "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}"
            REQUIRE_HIPCC ON
            ROCKE_INTERP "${_rocke_interp}"
            ROCKE_COMGR_LIB "${_rocke_comgr_lib}"
            ROCKE_WHEEL_STAMP "${_rocke_wheel_stamp}")
    else()
        message(STATUS
            "hkp: no production source root set "
            "(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT empty); production "
            "packaging dormant (tests still run against the fixtures).")
    endif()

    # Test descriptors. The conv_fwd source root is packed twice, once into each test
    # root, so both test binaries see the same authored descriptors.
    hkp_wire_pack_target(
        NAME unit_conv
        SOURCE_ROOT "${HIPKERNELPROVIDER_SHARED_CONV_SOURCE_DIR}"
        ENABLE_ROCKE OFF
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${HIPKERNELPROVIDER_UNIT_CONV_BUILD_DIR}"
        REQUIRE_HIPCC OFF)

    hkp_wire_pack_target(
        NAME integration_conv
        SOURCE_ROOT "${HIPKERNELPROVIDER_SHARED_CONV_SOURCE_DIR}"
        ENABLE_ROCKE OFF
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${HIPKERNELPROVIDER_INTEGRATION_CONV_BUILD_DIR}"
        REQUIRE_HIPCC OFF)

    hkp_wire_pack_target(
        NAME integration_pointwise
        SOURCE_ROOT "${HIPKERNELPROVIDER_INTEGRATION_POINTWISE_SOURCE_DIR}"
        ENABLE_ROCKE OFF
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${HIPKERNELPROVIDER_INTEGRATION_POINTWISE_BUILD_DIR}"
        REQUIRE_HIPCC OFF)

    hkp_wire_pack_target(
        NAME archive_fixture
        SOURCE_ROOT "${HIPKERNELPROVIDER_ARCHIVE_FIXTURE_SOURCE_DIR}"
        ENABLE_ROCKE OFF
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${HIPKERNELPROVIDER_ARCHIVE_FIXTURE_BUILD_DIR}"
        REQUIRE_HIPCC OFF)

    hkp_register_tests("${_rocm_kpack_dir}" "${HKP_HIPCC}"
                       "${HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP}"
                       "${_rocke_comgr_lib}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_register_tests(<rocm_kpack_dir> <hipcc> <hip_enabled> <rocke_comgr_lib>)
#   Register the pytest suite as two build-tree ctest entries running disjoint
#   sets: a quick entry (`-m quick`, the no-compile subset) and a standard entry
#   (`-m "not quick"`, the rest). Tier labels come from HKP_PACK_test_categories,
#   whose cascade runs each test once per tier with no overlap. When
#   Python3_EXECUTABLE cannot import pytest the entries register DISABLED so
#   they list as skipped, not absent.
#
#   Configure hard-fails on a missing hipcc only when the hip producer is
#   enabled (a tests-only ingestor build configures clean on a bare box; the
#   compile-dependent tests self-skip via the hipcc/rocke fixtures, and CI
#   hard-gates them via the REQUIRE_* env vars forwarded below).
# ---------------------------------------------------------------------------
function(hkp_register_tests rocm_kpack_dir hipcc hip_enabled rocke_comgr_lib)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()
    if(hip_enabled AND NOT hipcc)
        message(FATAL_ERROR
            "hkp: the hip production producer is enabled but hipcc was not found.")
    endif()

    # These were read from the environment (conftest.py:71,109) but declared
    # nowhere, so the only way to arm the CI gate was an env var nothing
    # documented. Declaring them as cache BOOLs makes the gate discoverable and
    # settable with -D, symmetric with every other knob here.
    #
    # What they buy is durability, not coverage: the comgr and hipcc tiers pass
    # today because the toolchains happen to be present. Their fixtures SKIP
    # when a probe fails, so a ROCm wheel bump that moves or drops comgr would
    # turn the tier green-by-skipping with nobody told. ON converts that skip
    # into a hard failure.
    set(HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC OFF CACHE BOOL
        "Fail (rather than skip) the hipcc-dependent packaging tests when hipcc \
is unavailable. Set ON in CI so the tier cannot silently stop running.")
    set(HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR OFF CACHE BOOL
        "Fail (rather than skip) the comgr-dependent rocKE packaging tests when \
rocke/kernels or libamd_comgr are unavailable. Set ON in CI so the tier cannot \
silently stop running.")

    # Runs under Python3_EXECUTABLE, the interpreter hkp_resolve_kpack proved
    # can import rocm_kpack. Bare PATH `python` may be a different one. The
    # ENVIRONMENT paths are configure-time absolutes, valid because these
    # entries run only in the build tree on the configuring machine.
    #
    # conftest.py reads HIPKERNELPROVIDER_ROCM_KPACK_DIR, so that is the name
    # forwarded here regardless of which variable resolved it.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}")
    if(hipcc)
        list(APPEND _pyenv "HKP_HIPCC=${hipcc}")
    endif()
    if(rocm_kpack_dir)
        list(APPEND _pyenv "HIPKERNELPROVIDER_ROCM_KPACK_DIR=${rocm_kpack_dir}")
    endif()
    # Forward the rocke comgr override + the CI hard-gate flags so the
    # comgr-dependent tier actually runs (not silently skips) where provisioned.
    if(rocke_comgr_lib)
        list(APPEND _pyenv "ROCKE_COMGR_LIB=${rocke_comgr_lib}")
    endif()
    if(HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC)
        list(APPEND _pyenv
            "HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC=${HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC}")
    endif()
    if(HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR)
        list(APPEND _pyenv
            "HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR=${HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR}")
    endif()

    # Without pytest, register the entries as DISABLED so they appear in the
    # ctest listing as skipped rather than silently absent.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _pytest_rc
        OUTPUT_QUIET ERROR_QUIET)
    set(_disabled "")
    if(NOT _pytest_rc EQUAL 0)
        message(STATUS
            "hkp: pytest not importable by ${Python3_EXECUTABLE}; registering "
            "descriptor-packaging pytest tests as DISABLED.")
        set(_disabled DISABLED TRUE)
    endif()

    add_test(NAME hip-kernel-provider-hkp-pack-quick
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m quick -v)
    set_tests_properties(hip-kernel-provider-hkp-pack-quick PROPERTIES
        ENVIRONMENT "${_pyenv}"
        ${_disabled})

    add_test(NAME hip-kernel-provider-hkp-pack
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m "not quick" -v)
    set_tests_properties(hip-kernel-provider-hkp-pack PROPERTIES
        ENVIRONMENT "${_pyenv}"
        ${_disabled})

    # Both entries are add_test()'d in this scope just above, so the YAML's
    # test_patterns match them via the directory-property loop. EXPLICIT_TESTS is
    # avoided: apply_ctest_category_labels joins it with ';', which execute_process
    # re-splits into separate argv, leaking a second name into the parser's
    # positional install-file slot.
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
       AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endfunction()
