# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build-time hip UKD -> compile -> prune -> kpack packaging for the
# hip-kernel-provider. All functions are provider-internal and namespaced hkp_*.
# hkp = Hip Kernel-provider Packaging.

include_guard(GLOBAL)

# Captured at include time so it survives into functions: inside a function
# CMAKE_CURRENT_LIST_DIR reflects the invoking listfile, not this module.
set(HKP_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(HKP_PYTHON_ROOT "${HKP_PKG_DIR}/python")
set(HKP_TOOL "${HKP_PKG_DIR}/tools/hkp_pack.py")
set(HKP_WHEEL_DIGEST_TOOL "${HKP_PKG_DIR}/tools/hkp_wheel_digest.py")
set(HKP_FIXTURES "${HKP_PKG_DIR}/tests/fixtures")

# The file every pack writes at the top of its output root to mark that root complete.
# One name for all of them: a caller installing a staged tree excludes it with a single
# pattern that never needs a clause per pack. Distinctive enough that the pattern cannot
# match a descriptor -- no authored or emitted file carries this name.
#
# Cached rather than plain, because this module is included from a subdirectory and the
# install() rules that must exclude the stamp are written by the parent, which a plain
# variable set here never reaches.
set(HKP_PACK_STAMP_NAME ".hkp-packed.stamp" CACHE INTERNAL
    "Name of the completion stamp each pack writes inside its output root")

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
# hkp_selected_arches(<out_var> <out_source_var>)
#   Normalize GPU_TARGETS (or AMDGPU_TARGETS) into a bare gfx arch list,
#   stripping feature suffixes (gfx942:xnack-) and dropping anything that is not
#   a concrete gfx name. <out_source_var> receives the name of the variable the
#   targets came from, or empty when neither is set, so a caller can name it in a
#   diagnostic. No intersection with a fixed fixture set: the tool compiles from
#   authored sources for whatever arch is requested.
#
#   The only consumer of GPU_TARGETS in dnn-providers/. The sibling kpack
#   producer, src/engines/asm_sdpa_engine/CMakeLists.txt, declares an explicit
#   list instead because it globs prebuilt .co files; this step compiles from
#   source and can target any real gfx, so it reads GPU_TARGETS.
#
#   Elsewhere in this repo a gfxNNX-style label is a selector matched against a
#   concrete arch (shared/ctest/parse_test_categories.py,
#   test/therock/test_runner.py); here the value reaches hipcc's --offload-arch,
#   where a family name is unusable rather than coarse. Hence drop-with-warning,
#   not passthrough, and no family-to-arch expansion table.
# ---------------------------------------------------------------------------
function(hkp_selected_arches out_var out_source_var)
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
    set(${out_source_var} "${_source}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_wire_pack_target(NAME <label> SOURCE_ROOT <dir>
#               ARCHES <list> HIPCC <path>
#               ROCM_KPACK_DIR <dir> OUT_ROOT <dir>
#               ROCKE_INTERP <path> ROCKE_READY <path> ROCKE_PYTHON_DIR <dir>
#               ROCKE_WHEEL_STAMP <path> [ROCKE_COMGR_LIB <path>]
#               [PACK_JOBS <n>])
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
#   A source root that is not a directory, or a missing output root, is a
#   configure error: each one makes the pack step write nothing, and a consumer
#   cannot tell that apart from a broken layout.
#
#   What actually differs between roots is declared, not forked into a second
#   function:
#
#   Every root runs under the supplied ROCKE_INTERP with ROCKE_PYTHON_DIR
#   prepended to PYTHONPATH, so `import rocke`/`kernels` resolve from the private
#   wheels wherever a UKD names them. Producer selection stays per-UKD on
#   kernel_source.kind, including roots holding only hip descriptors.
#
#   ROCKE_READY is the private directory's wheel-install stamp. The pack step
#   depends on it rather than only the interpreter, so changing a kernel under
#   rocke/library restages the pack. ROCKE_WHEEL_STAMP is the wheel content
#   digest, recorded into each rocKE UKD's provenance so a shipped kernel names
#   the wheel that produced it. ROCKE_COMGR_LIB, if set, is forwarded to the
#   tool environment.
#
#   PACK_JOBS caps the worker processes one pack may spawn. Omitted, the packer
#   sizes itself against the machine, which fits a root large enough to repay the
#   startup cost. Every root here is a separate custom target with no ordering
#   edge between them, so the generator runs them at once and unbounded pools
#   multiply. 1 selects the packer's serial path.
#
#   NAME is also the source label the packer writes into every descriptor's
#   provenance. The function records NAME and the absolute SOURCE_ROOT in a
#   global registry, which hkp_verify_embedded_sources() reads to resolve a
#   descriptor's authored location.
# ---------------------------------------------------------------------------
function(hkp_wire_pack_target)
    set(_one NAME SOURCE_ROOT ARCHES HIPCC ROCM_KPACK_DIR
        OUT_ROOT ROCKE_INTERP ROCKE_READY ROCKE_PYTHON_DIR ROCKE_COMGR_LIB
        ROCKE_WHEEL_STAMP PACK_JOBS)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${_one}" "")

    if(NOT IS_DIRECTORY "${ARG_SOURCE_ROOT}")
        message(FATAL_ERROR
            "hkp: source root '${ARG_NAME}' is not a directory: "
            "${ARG_SOURCE_ROOT}")
    endif()
    if(NOT ARG_OUT_ROOT)
        message(FATAL_ERROR
            "hkp: root '${ARG_NAME}' (${ARG_SOURCE_ROOT}) has no OUT_ROOT, so "
            "the pack step has nowhere to write.")
    endif()

    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-${ARG_NAME}-intermediate")
    # Inside the output root, so the stamp shares the fate of the tree it vouches for.
    # A stamp kept anywhere else witnesses only the pack's own run: it can say "the pack
    # finished", never "the output is still there". Whatever empties the tree -- a partial
    # restore, a stray clean, a disk that filled -- takes the stamp with it, and the next
    # build packs again instead of reading a stamp that outlived its descriptors.
    #
    # Dot-prefixed to match the convention the packer already uses for the in-progress
    # shard directories it does not ship.
    set(_stamp "${ARG_OUT_ROOT}/${HKP_PACK_STAMP_NAME}")

    # All roots use the supplied interpreter and private wheels, including
    # hip-only roots: producer selection is per descriptor, not per root.
    set(_interp "${ARG_ROCKE_INTERP}")
    set(_interp_what "rocKE wheel interpreter (root '${ARG_NAME}')")
    set(_interp_dep "${ARG_ROCKE_READY}")
    set(_wheel_dep "${ARG_ROCKE_WHEEL_STAMP}")

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
    if(ARG_PACK_JOBS)
        list(APPEND _tool_env "HKP_PACK_JOBS=${ARG_PACK_JOBS}")
    endif()
    if(ARG_ROCKE_COMGR_LIB)
        list(APPEND _tool_env "ROCKE_COMGR_LIB=${ARG_ROCKE_COMGR_LIB}")
    endif()

    # The authored root is a tree: glob recursively so a descriptor added in any
    # child folder retriggers the pack step. The packer itself walks recursively
    # so a flat glob here would drop the dependency edge for every nested descriptor.
    #
    # A descriptor REMOVED from the tree does not retrigger it. CONFIGURE_DEPENDS
    # re-globs and CMake re-runs, but a shorter DEPENDS list makes no input newer
    # and changes no command, so the edge stays clean and the wipe below never
    # fires -- the staged copy of a deleted descriptor survives an incremental
    # build. A clean configure is always correct. Putting the input set into the
    # edge, as a digest of the sorted glob, would close it.
    file(GLOB_RECURSE _source_inputs CONFIGURE_DEPENDS
         "${ARG_SOURCE_ROOT}/*")

    # Editing the tool's own sources must retrigger the pack step, else the
    # artifacts go stale against the current pipeline code. The resolved
    # rocm_kpack package counts too: kpack_resolver.py imports it and it decides
    # the archive format, so a packer change there must invalidate the stamp.
    # Deleting one of these sources does not retrigger it either, for the reason
    # the authored-root glob above records.
    file(GLOB _tool_sources CONFIGURE_DEPENDS
         "${HKP_PYTHON_ROOT}/hkp_pack/*.py"
         "${ARG_ROCM_KPACK_DIR}/rocm_kpack/*.py")

    hkp_require_kpack_runtime("${_interp}" "the ${_interp_what}")

    string(REPLACE ";" "," _arch_csv "${ARG_ARCHES}")

    set(_wheel_stamp_arg "")
    if(_wheel_dep)
        set(_wheel_stamp_arg --rocke-wheel-stamp "${_wheel_dep}")
    endif()

    set(_tool_cmd "${CMAKE_COMMAND}" -E env ${_tool_env}
        --modify "PYTHONPATH=path_list_prepend:${ARG_ROCKE_PYTHON_DIR}" --
        "${_interp}" "${HKP_TOOL}")

    # The wipe removes the stamp along with the tree, because the stamp lives inside it.
    # So no stamp exists from the moment a pack begins until it completes: a pack that
    # dies after the wipe -- a compiler failure, a killed job, an interrupted build --
    # leaves an empty tree AND no stamp, and the next build packs again rather than
    # reading the edge as up to date and letting the embedding check walk nothing and
    # pass at zero descriptors.
    #
    # It does not by itself cover a tree emptied while its stamp survives: the build
    # reads the edge as up to date and the embedding check walks nothing and passes,
    # because a root with no descriptors is otherwise a legal pass. The rule that catches
    # it -- a stamped root must hold at least one descriptor -- is stamped_root_failures()
    # in hkp_verify_embedded_sources.py.
    #
    # The output root is created before the stamp is written, because a pack that emits
    # nothing never creates it and `touch` does not create parents. Such a root holds
    # the stamp alone, and installs as an empty directory: the install rules exclude the
    # stamp file, not the directory it sits in.
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
                --source-label "${ARG_NAME}"
                ${_wheel_stamp_arg}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUT_ROOT}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${HKP_TOOL}" ${_source_inputs} ${_tool_sources}
                ${_interp_dep} ${_wheel_dep}
        COMMENT "hkp: packing root '${ARG_NAME}' for ${ARG_ARCHES}"
        VERBATIM)

    add_custom_target(hkp_packaging_${ARG_NAME} ALL
                      DEPENDS "${_stamp}"
                      COMMENT "hkp: descriptor packaging (${ARG_NAME})")
    if(TARGET hkp_rocke_wheel_python_interp)
        # Every root shares one private wheel directory. Keep a single producer:
        # parallel consumers must not clear/repopulate it while another packs.
        add_dependencies(hkp_packaging_${ARG_NAME} hkp_rocke_wheel_python_interp)
    endif()
    set_property(GLOBAL PROPERTY HKP_PACK_STAMP_${ARG_NAME} "${_stamp}")

    # The key manifest normalises each registered path the same lexical way, so
    # the two spellings agree and the verify step compares them exactly.
    get_filename_component(_abs_source_root "${ARG_SOURCE_ROOT}" ABSOLUTE)
    set_property(GLOBAL PROPERTY HKP_PACK_SOURCE_ROOT_${ARG_NAME} "${_abs_source_root}")
    set_property(GLOBAL APPEND PROPERTY HKP_PACK_LABELS "${ARG_NAME}")
endfunction()

# ---------------------------------------------------------------------------
# _hkp_key_manifest_args(<out_arg> <out_dep> <target>)
#   Resolve the key manifest <target> published, as a command argument and a
#   dependency.
#
#   embed_kernel_sources() records the path on the target. Reading it back is
#   what stops a consumer in another directory scope from spelling the same rule
#   a second time and naming a file nothing writes -- which reads as an empty
#   table and passes, exactly as a target that embeds nothing does.
#
#   A target that never called embed_kernel_sources() has no property and gets
#   no flag, which is a fact about the target rather than about a directory. A
#   target with kernels registered but no manifest is neither case: the check
#   has been ordered before the embedding.
# ---------------------------------------------------------------------------
function(_hkp_key_manifest_args out_arg out_dep target)
    get_target_property(_manifest ${target} KERNELEMBEDDING_KEY_MANIFEST)
    if(NOT _manifest)
        get_target_property(_declared ${target} KERNELEMBEDDING_KERNEL_FILES)
        if(_declared)
            message(FATAL_ERROR
                    "hkp_verify_embedded_sources: target '${target}' has kernels "
                    "registered for embedding but publishes no key manifest. Call "
                    "embed_kernel_sources(TARGET ${target} ...) before verifying it.")
        endif()
        set(${out_arg} "" PARENT_SCOPE)
        set(${out_dep} "" PARENT_SCOPE)
        return()
    endif()

    set(${out_arg} --key-manifest "${_manifest}" PARENT_SCOPE)
    # Naming an absent file as a dependency asks the generator for a rule that
    # produces it. The table is written at configure time, so it is absent only
    # when the target registered no kernel between the two calls.
    set(_dep "")
    if(EXISTS "${_manifest}")
        set(_dep "${_manifest}")
    endif()
    set(${out_dep} "${_dep}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_verify_embedded_sources(TARGET <t> STAGED_DESCRIPTOR_ROOTS <roots>
#                             PACK_NAMES <names>)
#   Add a build step that checks <t> against the staged descriptors it serves.
#
#   Every value of STAGED_DESCRIPTOR_ROOTS is a packer output tree. None of them
#   is an authored source tree.
#
#   Every embedded_source descriptor under STAGED_DESCRIPTOR_ROOTS names a kernel
#   source. The step reads the key table embed_kernel_sources() wrote for <t>
#   and fails the build when a named source is absent from it, or when the file
#   registered under a key is not the file at the authored location the
#   descriptor's provenance records.
#
#   A descriptor resolves its own source root from its provenance.source_label,
#   through the registry hkp_wire_pack_target() fills. The step joins that root
#   with the descriptor's rel_dir and source_file, and compares the whole path
#   against the registered one. Every wired label goes to every call site, so a
#   descriptor written by a pack no PACK_NAMES value lists still resolves.
#
#   PACK_NAMES lists the pack roots that write STAGED_DESCRIPTOR_ROOTS. Each one
#   contributes its stamp file twice: as a dependency, so packing a root reruns
#   the check, and as an argument, so the step also fails when a stamped pack
#   root holds no descriptor at all. A name whose root is not wired contributes
#   neither, so a dormant root -- production, with no source root set -- is not
#   held to that rule.
#
#   An absent root, an empty root, a root with no embedded_source descriptor and
#   an empty key table each pass. A root emptied after its pack stamped it does not.
#
#   The comparison runs one way, from a staged descriptor to the table. A key no
#   descriptor names is not an error, and neither is a descriptor no pack stages.
#   The tool's docstring records why neither reverse direction can be turned on
#   while the descriptor and the embedding declaration are written independently.
# ---------------------------------------------------------------------------
function(hkp_verify_embedded_sources)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET" "STAGED_DESCRIPTOR_ROOTS;PACK_NAMES")

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
                "hkp_verify_embedded_sources: unrecognised argument(s): "
                "${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "hkp_verify_embedded_sources called without a TARGET!")
    endif()
    if(NOT TARGET ${ARG_TARGET})
        message(FATAL_ERROR
                "hkp_verify_embedded_sources: the target ${ARG_TARGET} does not exist "
                "yet. Call it after the target is created.")
    endif()
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR
                "hkp_verify_embedded_sources: Python3_EXECUTABLE is empty. The "
                "descriptor packaging finds the interpreter, so add it before the "
                "targets it verifies.")
    endif()

    # Resolved from the defining listfile: the callers are sibling directories that
    # never see this module's include-time variables.
    set(_tool "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/hkp_verify_embedded_sources.py")
    set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/hkp-verify-${ARG_TARGET}.stamp")

    _hkp_key_manifest_args(_manifest_arg _manifest_dep "${ARG_TARGET}")

    set(_root_args "")
    foreach(_root IN LISTS ARG_STAGED_DESCRIPTOR_ROOTS)
        list(APPEND _root_args --staged-descriptor-root "${_root}")
    endforeach()

    # Every wired label, at every call site. A label the registry knows but no
    # property backs contributes nothing, the same rule the stamp lookup follows.
    set(_source_root_args "")
    get_property(_labels GLOBAL PROPERTY HKP_PACK_LABELS)
    foreach(_label IN LISTS _labels)
        get_property(_label_root GLOBAL PROPERTY HKP_PACK_SOURCE_ROOT_${_label})
        if(_label_root)
            list(APPEND _source_root_args --source-root "${_label}=${_label_root}")
        endif()
    endforeach()

    # The stamp file, not the packaging target: a target-level edge orders the two
    # steps but leaves the check stale after a repack.
    set(_pack_stamps "")
    set(_stamp_args "")
    set(_pack_targets "")
    foreach(_pack IN LISTS ARG_PACK_NAMES)
        get_property(_pack_stamp GLOBAL PROPERTY HKP_PACK_STAMP_${_pack})
        if(_pack_stamp)
            list(APPEND _pack_stamps "${_pack_stamp}")
            # The same stamp again as an argument, so the tool holds the root it
            # sits in to the non-empty rule.
            list(APPEND _stamp_args --pack-stamp "${_pack_stamp}")
        endif()
        if(TARGET hkp_packaging_${_pack})
            list(APPEND _pack_targets hkp_packaging_${_pack})
        endif()
    endforeach()

    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND "${Python3_EXECUTABLE}" "${_tool}"
                --target "${ARG_TARGET}"
                ${_manifest_arg}
                ${_root_args}
                ${_stamp_args}
                ${_source_root_args}
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${_tool}" ${_manifest_dep} ${_pack_stamps}
        COMMENT "hkp: verifying embedded kernel sources (${ARG_TARGET})"
        VERBATIM)

    add_custom_target(hkp_verify_${ARG_TARGET} ALL
                      DEPENDS "${_stamp}"
                      COMMENT "hkp: embedded source verification (${ARG_TARGET})")
    if(_pack_targets)
        # The stamps are written from another directory, where a file-level edge
        # alone leaves generators that build per directory without a rule for them.
        add_dependencies(hkp_verify_${ARG_TARGET} ${_pack_targets})
    endif()
    add_dependencies(${ARG_TARGET} hkp_verify_${ARG_TARGET})
endfunction()

# ---------------------------------------------------------------------------
# hkp_imported_library_location(<target> <out_path>)
#   Full path of the loadable library behind an imported <target>, or empty.
#
#   The configuration is chosen deliberately rather than by taking whichever
#   one the package exported first: the configuration being built when that is
#   known, else a documented order, else any exported configuration that
#   resolves. A package may also export a configuration whose file is not on
#   disk, so each candidate is checked and the scan continues past one that
#   fails -- stopping at the first exported entry yields nothing in that case
#   even when another configuration would have worked.
#
#   IMPORTED_IMPLIB is deliberately never consulted: the caller hands this path
#   to rocke, which ctypes.CDLLs it, and a Windows import library is not
#   loadable. Returning empty lets rocke resolve normally, which a dead path
#   would not.
# ---------------------------------------------------------------------------
function(hkp_imported_library_location target out_path)
    set(_derived "")
    # A multi-config generator does not know the configuration at configure
    # time, so CMAKE_BUILD_TYPE may be empty and the rest of the order decides.
    # NOCONFIG is what an export() with no build type emits.
    set(_preferred "")
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _preferred)
    endif()
    list(APPEND _preferred RELEASE RELWITHDEBINFO MINSIZEREL DEBUG NOCONFIG)

    # IN_LIST against a NOTFOUND property is a hard error, which is what the
    # guard prevents.
    get_target_property(_cfgs ${target} IMPORTED_CONFIGURATIONS)
    if(_cfgs)
        foreach(_cfg IN LISTS _preferred)
            if(NOT _cfg IN_LIST _cfgs)
                continue()
            endif()
            get_target_property(_loc ${target} IMPORTED_LOCATION_${_cfg})
            if(_loc AND EXISTS "${_loc}")
                set(_derived "${_loc}")
                break()
            endif()
        endforeach()
    endif()
    # Nothing preferred resolved: any real library beats none.
    if(NOT _derived AND _cfgs)
        foreach(_cfg IN LISTS _cfgs)
            get_target_property(_loc ${target} IMPORTED_LOCATION_${_cfg})
            if(_loc AND EXISTS "${_loc}")
                set(_derived "${_loc}")
                break()
            endif()
        endforeach()
    endif()
    # A package that exports no configurations at all.
    if(NOT _derived)
        get_target_property(_loc ${target} IMPORTED_LOCATION)
        if(_loc AND EXISTS "${_loc}")
            set(_derived "${_loc}")
        endif()
    endif()
    set(${out_path} "${_derived}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_default_rocke_comgr_lib()
#   Give HIPKERNELPROVIDER_ROCKE_COMGR_LIB a package-derived default in the
#   CALLER's scope, so a correctly configured build needs no explicit path.
#   Does nothing when the variable already holds a value.
#
#   The result is a plain (non-cache) variable on purpose. It has to be visible
#   both to hkp_probe_comgr_resolvable, whose assertion only runs when the
#   override is non-empty, and to the pack step and ctest entries -- one value,
#   or configure validates something the build does not use.
#
#   Every path out of here says what it decided. Staying empty is safe, because
#   rocke then resolves comgr itself, but it is indistinguishable at a glance
#   from "no override was wanted" -- and on Windows rocke's own search can reach
#   a System32 amd_comgr.dll, which is the failure this default exists to avoid.
# ---------------------------------------------------------------------------
function(hkp_default_rocke_comgr_lib)
    if(HIPKERNELPROVIDER_ROCKE_COMGR_LIB)
        return()
    endif()
    # hip's config supplies the target on Linux but not on Windows, so the
    # package is searched for only when it is genuinely absent.
    if(NOT TARGET amd_comgr)
        find_package(amd_comgr CONFIG QUIET)
    endif()
    if(NOT TARGET amd_comgr)
        message(STATUS
            "hkp: no amd_comgr target and no amd_comgr CONFIG package, so "
            "HIPKERNELPROVIDER_ROCKE_COMGR_LIB stays empty and rocke will "
            "resolve comgr by its own search.")
        return()
    endif()
    hkp_imported_library_location(amd_comgr _derived)
    if(_derived)
        set(HIPKERNELPROVIDER_ROCKE_COMGR_LIB "${_derived}" PARENT_SCOPE)
        message(STATUS
            "hkp: HIPKERNELPROVIDER_ROCKE_COMGR_LIB derived from the "
            "amd_comgr package: ${_derived}")
    else()
        get_target_property(_cfgs amd_comgr IMPORTED_CONFIGURATIONS)
        if(NOT _cfgs)
            set(_cfgs "<none>")
        endif()
        message(STATUS
            "hkp: the amd_comgr target exists but exports no library path that "
            "is present on disk, so HIPKERNELPROVIDER_ROCKE_COMGR_LIB stays "
            "empty and rocke will resolve comgr by its own search. Exported "
            "configurations: ${_cfgs}")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# hkp_probe_comgr_resolvable(<out_ok> <out_detail>)
#   Configure-time gate for the rocKE producer, scoped to what is knowable at
#   configure time.
#
#   This probe does NOT check that `rocke`/`kernels` import. That check belongs
#   after private wheel installation in hkp_rocke_wheel_python_interp: the
#   private import directory is populated at build time, and the build imports
#   from those wheels rather than from the source tree.
#
#   An explicitly-set ROCKE_COMGR_LIB is checked as an ASSERTION, which rocKE
#   itself does not do: `_candidate_lib_paths` puts the override first and
#   `_load_lib` falls through to the next candidate when it does not load. The
#   override exists for Windows, where a System32 amd_comgr.dll can shadow the
#   ROCm one -- so falling through lands on the shadowing DLL, i.e. the override
#   fails open into the exact failure it was set to prevent. Comparing what
#   loaded against what was asked for turns that into a configure error.
#
#   The probe reads the resolver from the source tree deliberately: it asks
#   about the machine's comgr, not about the wheels.
# ---------------------------------------------------------------------------
function(hkp_probe_comgr_resolvable out_ok out_detail)
    set(_rocke_root "${HKP_PKG_DIR}/../rocke")
    # Joined with the platform's own PYTHONPATH separator. The assignment reaches
    # `cmake -E env` as one argv element because the expansion at the call site below is
    # quoted; a quoted argument never splits on a semicolon, so the Windows separator
    # needs no escaping here. Escaping it would put a literal backslash in the child's
    # first sys.path entry.
    if(WIN32)
        set(_sep ";")
    else()
        set(_sep ":")
    endif()
    set(_pp "${_rocke_root}/platform/python${_sep}${_rocke_root}/library")
    # Probe under the SAME override the build will use, so configure and build
    # ask the same question. Without this a machine that only resolves comgr via
    # the override would fail configure despite being correctly configured.
    set(_probe_extra_env "")
    if(HIPKERNELPROVIDER_ROCKE_COMGR_LIB)
        set(_probe_extra_env
            "ROCKE_COMGR_LIB=${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}")
    endif()
    # ctypes records the path it opened on the loaded handle, so comparing that
    # against the override is what distinguishes "the override loaded" from
    # "something else did". Compared through realpath: the ROCm layout reaches
    # one library through several symlinked names.
    set(_probe_py "import os, sys
from rocke.runtime import comgr
lib = comgr._resolve_lib()
want = os.environ.get(\"ROCKE_COMGR_LIB\")
got = getattr(lib, \"_name\", None)
if want and (not got or os.path.realpath(got) != os.path.realpath(want)):
    sys.exit(f\"comgr loaded {got!r} instead of the requested {want!r}\")
")
    # Each assignment must reach `-E env` as ONE argv element: the Windows
    # PYTHONPATH separator is also CMake's list separator, so an unquoted
    # expansion splits it and `-E env` takes the tail as the executable.
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_pp}" ${_probe_extra_env} --
                "${Python3_EXECUTABLE}" -c "${_probe_py}"
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
#   Keying wheel installation and packing on wheel mtime would therefore recompile
#   every kernel for every arch on every build, even when the wheels are
#   byte-identical. Keying on this stamp instead means a rebuild that produces
#   identical wheels leaves the stamp's mtime untouched, and Ninja's restat
#   prunes everything downstream.
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

    # The wheels' OUTPUT rules are declared in rocke/, so the file-level DEPENDS
    # above crosses a directory boundary -- a shape generators are not obliged to
    # resolve. The target-level edge states the ordering directly. Guarded because
    # rocke-wheels exists only under ROCKE_BUILD_PYENV; with it OFF the wheels are
    # supplied inputs whose existence hkp_require_ingestor_toolchain has checked.
    if(TARGET rocke-wheels)
        add_dependencies(hkp_rocke_wheel_digest rocke-wheels)
    endif()

    set(${out_stamp} "${_stamp}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_require_kpack_runtime(<interp> <what>)
#   rocm_kpack is reached by putting a source tree on sys.path, so pip never
#   resolves the msgpack/zstandard it declares. The supplied interpreter needs
#   them present independently for every pack, including hip-only roots.
#
#   Checked at configure time because the failure is otherwise a mid-build
#   ImportError from inside a dependency, which reads as a packer bug rather
#   than a missing dependency on the build machine.
# ---------------------------------------------------------------------------
function(hkp_require_kpack_runtime interp what)
    if(NOT EXISTS "${interp}")
        message(FATAL_ERROR
            "hkp: ${what} does not exist: ${interp}. Set Python3_EXECUTABLE "
            "to an existing interpreter with pip, msgpack and zstandard supplied.")
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
# hkp_rocke_wheel_python_interp(<out_interp> <out_ready> <out_python_dir> <wheel_stamp>)
#   Install the exact local rocke + rocke_library wheels into a build-owned import
#   directory, using the supplied Python3_EXECUTABLE and its existing pip/runtime
#   dependencies. ROCKE_BUILD_PYENV ON supplies wheels via rocke-wheels; OFF uses
#   ROCKE_WHEEL_DIR. Neither mode installs anything into the supplied environment.
#
#   Wheel-content changes trigger replacement. Clear the owned directory
#   first so removed modules cannot survive same-version replacement. Readiness
#   lives inside it and is published only after imports succeed: deleting the
#   tree or interrupting a refresh cannot leave a valid external stamp.
#
#   --no-index and --no-deps restrict pip to the two local inputs. Missing runtime
#   dependencies are errors, not permission to acquire them. The supplied Python
#   retains its normal startup behavior, including .pth and enabled user-site
#   processing; a scoped PYTHONPATH prepend selects the private wheels.
# ---------------------------------------------------------------------------
function(hkp_rocke_wheel_python_interp out_interp out_ready out_python_dir wheel_stamp)
    set(_python_dir "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-python")
    set(_ready "${_python_dir}/.installed")
    set(_platform_wheel
        "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    set(_library_wheel
        "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")

    hkp_require_kpack_runtime("${Python3_EXECUTABLE}" "the supplied interpreter")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pip --version
        RESULT_VARIABLE _pip_rc
        OUTPUT_QUIET
        ERROR_VARIABLE _pip_err)
    if(NOT _pip_rc EQUAL 0)
        string(STRIP "${_pip_err}" _pip_err)
        message(FATAL_ERROR
            "hkp: ${Python3_EXECUTABLE} cannot run pip. Supply pip in this "
            "interpreter's environment, or set Python3_EXECUTABLE to an existing "
            "interpreter with pip, msgpack and zstandard. Packaging does not "
            "bootstrap pip or acquire runtime dependencies.\nPython said: ${_pip_err}")
    endif()

    set(_import_env "ROCKE_BACKEND=python" "ROCKE_CPP_STRICT=1")
    if(HIPKERNELPROVIDER_ROCKE_COMGR_LIB)
        list(APPEND _import_env "ROCKE_COMGR_LIB=${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}")
    endif()
    add_custom_command(
        OUTPUT "${_ready}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_python_dir}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_python_dir}"
        COMMAND "${Python3_EXECUTABLE}" -m pip --disable-pip-version-check install
                --no-index --no-deps --no-cache-dir --target "${_python_dir}"
                "${_platform_wheel}" "${_library_wheel}"
        COMMAND "${CMAKE_COMMAND}" -E env ${_import_env}
                --modify "PYTHONPATH=path_list_prepend:${_python_dir}" --
                "${Python3_EXECUTABLE}" -c "import rocke, kernels, msgpack, zstandard"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ready}"
        DEPENDS "${Python3_EXECUTABLE}" "${wheel_stamp}" "${HKP_WHEEL_DIGEST_TOOL}"
        COMMENT "hkp: installing local rocke wheels into the private import directory"
        VERBATIM)

    add_custom_target(hkp_rocke_wheel_python_interp ALL DEPENDS "${_ready}"
                      COMMENT "hkp: preparing rocke wheel imports")
    add_dependencies(hkp_rocke_wheel_python_interp hkp_rocke_wheel_digest)
    set(${out_interp} "${Python3_EXECUTABLE}" PARENT_SCOPE)
    set(${out_ready} "${_ready}" PARENT_SCOPE)
    set(${out_python_dir} "${_python_dir}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_require_ingestor_toolchain(<out_arches>)
#   Assert what the ingestor needs to pack anything -- hipcc, a non-empty gfx
#   list, and the rocke wheel supply -- and return the architecture list. Set
#   HKP_HIPCC as a side effect.
#
#   Without hipcc or a gfx list the packer creates no output root at all, and a
#   consumer of a packed root reports that as a broken layout rather than as a
#   missing prerequisite. A missing wheel supply surfaces later still, inside the
#   private wheel installation or import. Fail here for all three instead,
#   and name the remedy.
#
#   The wheel check covers SUPPLY, not importability. The private directory
#   hkp_rocke_wheel_python_interp populates does not exist until the build runs;
#   imports are asserted there under the environment the pack step will use.
# ---------------------------------------------------------------------------
function(hkp_require_ingestor_toolchain out_arches)
    # hipcc is the perl/bat driver that honors --genco; on Windows it is
    # hipcc.exe or hipcc.bat. hipcc.bin.exe is the raw clang driver and is only
    # a last-resort fallback.
    #
    # The default name-major search is what holds that ordering: every directory is
    # tried for hipcc before hipcc.bin.exe is tried anywhere, so the fallback wins
    # only when no real driver exists anywhere on the path. NAMES_PER_DIR would
    # demote this list to a tiebreak within one directory and let an early
    # hipcc.bin.exe beat a later hipcc.
    find_program(HKP_HIPCC NAMES hipcc hipcc.bat hipcc.bin.exe)
    if(NOT HKP_HIPCC)
        message(FATAL_ERROR
            "hkp: HIPDNN_ENABLE_KERNEL_INGESTOR is ON and requires hipcc to "
            "compile the kernels it packs, but hipcc was not found (searched "
            "hipcc, hipcc.bat, hipcc.bin.exe). Put the ROCm bin directory on "
            "PATH or CMAKE_PROGRAM_PATH, or set "
            "HIPDNN_ENABLE_KERNEL_INGESTOR=OFF.")
    endif()

    if(NOT ROCKE_WHEEL_DIR OR NOT ROCKE_WHEEL_VERSION)
        message(FATAL_ERROR
            "hkp: HIPDNN_ENABLE_KERNEL_INGESTOR is ON and requires the rocke "
            "wheels to pack rocKE kernels, but ROCKE_WHEEL_DIR "
            "(${ROCKE_WHEEL_DIR}) and ROCKE_WHEEL_VERSION "
            "(${ROCKE_WHEEL_VERSION}) are not both set. Leave "
            "ROCKE_BUILD_PYENV=ON to have the build produce the wheels and set "
            "both variables, or with ROCKE_BUILD_PYENV=OFF set ROCKE_WHEEL_DIR "
            "and ROCKE_WHEEL_VERSION to the wheels you supply.")
    endif()

    # ROCKE_BUILD_PYENV=ON makes the wheels build outputs, absent until the build
    # runs. Only with it OFF are they inputs, and only then can they be checked.
    if(NOT ROCKE_BUILD_PYENV)
        set(_platform_wheel
            "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
        set(_library_wheel
            "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
        if(NOT EXISTS "${_platform_wheel}" OR NOT EXISTS "${_library_wheel}")
            message(FATAL_ERROR
                "hkp: HIPDNN_ENABLE_KERNEL_INGESTOR is ON and ROCKE_BUILD_PYENV "
                "is OFF, so the rocke wheels must be supplied, but "
                "ROCKE_WHEEL_DIR (${ROCKE_WHEEL_DIR}) does not hold both of:\n"
                "    rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl\n"
                "    rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl\n"
                "Point ROCKE_WHEEL_DIR at a directory holding both, correct "
                "ROCKE_WHEEL_VERSION, or set ROCKE_BUILD_PYENV=ON to build them.")
        endif()
    endif()

    hkp_selected_arches(_arches _arch_source)
    if(_arches)
        set(${out_arches} "${_arches}" PARENT_SCOPE)
        return()
    endif()
    if(_arch_source)
        message(FATAL_ERROR
            "hkp: HIPDNN_ENABLE_KERNEL_INGESTOR is ON and requires at least one "
            "concrete gfx architecture to pack for, but ${_arch_source} "
            "(${${_arch_source}}) resolves to an empty architecture list. Name "
            "concrete gfx architectures in ${_arch_source}.")
    endif()
    message(FATAL_ERROR
        "hkp: HIPDNN_ENABLE_KERNEL_INGESTOR is ON and requires at least one "
        "concrete gfx architecture to pack for, but neither GPU_TARGETS nor "
        "AMDGPU_TARGETS is set, so the architecture list is empty. Set "
        "GPU_TARGETS to the gfx architectures to pack for.")
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Gate production packaging on ONE source root. The root names a location;
#   producer selection is per-UKD on kernel_source.kind, so both producers are
#   available to every root.
#
#   This function runs only under HIPDNN_ENABLE_KERNEL_INGESTOR, and asserts
#   that option's prerequisites first through hkp_require_ingestor_toolchain.
#
#   rocKE is REQUIRED, for the test roots as much as for production, so it is
#   resolved once here for every root rather than selected per root. Unresolvable
#   comgr is fatal at configure -- there is no build in which some roots pack and
#   others do not. Private wheels are installed even for a hip-only build, so the
#   acquisition path production ships through is the one every pack exercises.
#   The pytest producer suite separately imports rocke from the source tree.
#
#   Root empty = production packaging dormant. Root set but not a directory =
#   fatal. The tests are wired regardless.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_rocm_kpack_dir "${Python3_EXECUTABLE}")
    hkp_require_ingestor_toolchain(_arches)

    set(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT "" CACHE PATH
        "The authored source root the production pack step compiles from. \
Walked recursively; child folders under it scope the content (hip/, rocKE/, \
per-integration folders) and each descriptor's authored subpath is preserved \
into the staged and installed trees. Empty leaves production packaging dormant.")
    set(_source_root "")
    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but is "
                "not a directory: ${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
        endif()
        set(_source_root "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
    endif()

    set(HIPKERNELPROVIDER_ROCKE_COMGR_LIB "" CACHE PATH
        "Explicit libamd_comgr for the rocKE producer to load. Forwarded into \
ROCKE_COMGR_LIB for the pack step and the ctest entries. Leave it empty for \
the normal case: the path is then derived from the amd_comgr package, and it \
stays empty only when no usable location can be derived, which configure \
reports and which leaves rocke to resolve comgr by its own search. Set it to \
override that -- needed on Windows, where a System32 amd_comgr.dll can shadow \
the ROCm one. rocke treats the value as the first CANDIDATE and falls through \
when it does not load, so configure asserts that the library which loaded is \
the one named here.")

    # Runs before the copy below and before the probe, so the derived value is
    # what both of them see. An explicitly-set value is left alone.
    hkp_default_rocke_comgr_lib()

    # ROCKE_COMGR_LIB is rocke's runtime environment variable, not a CMake variable: the
    # value comes from our own cache entry and is forwarded into the environment rocke
    # reads.
    set(_rocke_comgr_lib "${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}")

    # Resolved once, for every root. hkp_rocke_wheel_python_interp declares both a
    # custom command OUTPUT and a target, so a second call is a duplicate of each.
    hkp_probe_comgr_resolvable(_comgr_ok _comgr_detail)
    if(NOT _comgr_ok)
        message(FATAL_ERROR
            "hkp: comgr could not be resolved, so no rocKE kernel can be "
            "lowered and no descriptor root can be packed. comgr ships with "
            "ROCm and is required. Set HIPKERNELPROVIDER_ROCKE_COMGR_LIB to an "
            "explicit libamd_comgr, or make one discoverable. Resolver said:\n"
            "${_comgr_detail}")
    endif()
    hkp_rocke_wheel_stamp(_rocke_wheel_stamp)
    hkp_rocke_wheel_python_interp(_rocke_interp _rocke_ready _rocke_python_dir
                                 "${_rocke_wheel_stamp}")

    # One list for every root, so "every root is wired to rocKE identically" is
    # structural rather than six sites that have to agree. COMGR_LIB is appended
    # only when set: an empty element does not survive unquoted expansion, and
    # losing one would shift every following keyword into the wrong slot.
    set(_rocke_args
        ROCKE_INTERP "${_rocke_interp}"
        ROCKE_READY "${_rocke_ready}"
        ROCKE_PYTHON_DIR "${_rocke_python_dir}"
        ROCKE_WHEEL_STAMP "${_rocke_wheel_stamp}")
    if(_rocke_comgr_lib)
        list(APPEND _rocke_args ROCKE_COMGR_LIB "${_rocke_comgr_lib}")
    endif()

    # Production descriptors.
    if(_source_root)
        hkp_wire_pack_target(
            NAME product
            SOURCE_ROOT "${_source_root}"
            ARCHES "${_arches}"
            HIPCC "${HKP_HIPCC}"
            ROCM_KPACK_DIR "${_rocm_kpack_dir}"
            OUT_ROOT "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}"
            ${_rocke_args})
    else()
        # A tree left over from an earlier configuration that did pack keeps
        # being loaded: the engine selects the plugin-relative directory on
        # existence alone, and nothing else removes it once the pack target and
        # its install rule are gone.
        if(HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR)
            file(REMOVE_RECURSE "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}")
        endif()
        message(STATUS
            "hkp: no production source root set "
            "(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT empty); production "
            "packaging dormant (tests still run against the fixtures).")
    endif()

    # Test descriptors, one pack per authored set. The shared root is packed into both
    # test roots, so both test binaries see the same authored descriptors; the two need
    # distinct NAMEs.
    set(_authored "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_SOURCE_ROOT}")
    set(_unit "${HIPKERNELPROVIDER_UNIT_BUILD_DIR}")
    set(_integration "${HIPKERNELPROVIDER_INTEGRATION_BUILD_DIR}")

    hkp_wire_pack_target(
        NAME unit_shared
        SOURCE_ROOT "${_authored}/${HIPKERNELPROVIDER_TEST_SET_SHARED}"
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${_unit}/${HIPKERNELPROVIDER_TEST_SET_SHARED}"
        ${_rocke_args}
        PACK_JOBS 1)

    hkp_wire_pack_target(
        NAME unit
        SOURCE_ROOT "${_authored}/${HIPKERNELPROVIDER_TEST_SET_UNIT}"
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${_unit}/${HIPKERNELPROVIDER_TEST_SET_UNIT}"
        ${_rocke_args}
        PACK_JOBS 1)

    hkp_wire_pack_target(
        NAME integration_shared
        SOURCE_ROOT "${_authored}/${HIPKERNELPROVIDER_TEST_SET_SHARED}"
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${_integration}/${HIPKERNELPROVIDER_TEST_SET_SHARED}"
        ${_rocke_args}
        PACK_JOBS 1)

    hkp_wire_pack_target(
        NAME integration
        SOURCE_ROOT "${_authored}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}"
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${_integration}/${HIPKERNELPROVIDER_TEST_SET_INTEGRATION}"
        ${_rocke_args}
        # The only test root with enough distinct hip variants to build a worker
        # pool, so it is the one that exercises the parallel path in a real
        # build. Falls back to the serial path if that root ever drops below two.
        PACK_JOBS 2)

    hkp_wire_pack_target(
        NAME archive_fixture
        SOURCE_ROOT "${_authored}/${HIPKERNELPROVIDER_TEST_SET_ARCHIVE_FIXTURE}"
        ARCHES "${_arches}"
        HIPCC "${HKP_HIPCC}"
        ROCM_KPACK_DIR "${_rocm_kpack_dir}"
        OUT_ROOT "${_integration}/${HIPKERNELPROVIDER_TEST_SET_ARCHIVE_FIXTURE}"
        ${_rocke_args}
        PACK_JOBS 1)

    hkp_register_tests("${_rocm_kpack_dir}" "${HKP_HIPCC}" "${_rocke_comgr_lib}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_register_tests(<rocm_kpack_dir> <hipcc> <rocke_comgr_lib>)
#   Register the pytest suite as two build-tree ctest entries running disjoint
#   sets: a quick entry (`-m quick`, the no-compile subset) and a standard entry
#   (`-m "not quick"`, the rest). Tier labels come from HKP_PACK_test_categories,
#   whose cascade runs each test once per tier with no overlap. Configuration
#   fails when Python3_EXECUTABLE cannot import pytest.
#
#   hipcc is a requirement of the whole ingestor, so the hipcc-dependent tests
#   are hard-gated: their fixture fails on a missing hipcc rather than skipping.
# ---------------------------------------------------------------------------
function(hkp_register_tests rocm_kpack_dir hipcc rocke_comgr_lib)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()

    # Runs under Python3_EXECUTABLE, the interpreter hkp_resolve_kpack proved
    # can import rocm_kpack. Bare PATH `python` may be a different one. The
    # ENVIRONMENT paths are configure-time absolutes, valid because these
    # entries run only in the build tree on the configuring machine.
    #
    # conftest.py reads HIPKERNELPROVIDER_ROCM_KPACK_DIR, so that is the name
    # forwarded here regardless of which variable resolved it.
    #
    # HKP_HIPCC names the hipcc that configure found.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}"
        "HKP_HIPCC=${hipcc}")
    if(rocm_kpack_dir)
        list(APPEND _pyenv "HIPKERNELPROVIDER_ROCM_KPACK_DIR=${rocm_kpack_dir}")
    endif()
    # Forward the rocke comgr override so the comgr-dependent tier resolves the
    # same library the pack step does.
    if(rocke_comgr_lib)
        list(APPEND _pyenv "ROCKE_COMGR_LIB=${rocke_comgr_lib}")
    endif()

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _pytest_rc
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _pytest_rc EQUAL 0)
        message(FATAL_ERROR
            "hkp: pytest is not importable by ${Python3_EXECUTABLE}, so the "
            "descriptor-packaging tests cannot run. Install pytest for that "
            "interpreter, or configure with "
            "-DHIPKERNELPROVIDER_ENABLE_TESTS=OFF.")
    endif()

    add_test(NAME hip-kernel-provider-hkp-pack-quick
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m quick -v)
    set_tests_properties(hip-kernel-provider-hkp-pack-quick PROPERTIES
        ENVIRONMENT "${_pyenv}")

    add_test(NAME hip-kernel-provider-hkp-pack
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m "not quick" -v)
    set_tests_properties(hip-kernel-provider-hkp-pack PROPERTIES
        ENVIRONMENT "${_pyenv}")

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
