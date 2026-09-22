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
#               ROCKE_INTERP <path> ROCKE_READY <path>
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
#   Every root runs under ROCKE_INTERP, the wheel-provisioned interpreter, so
#   `import rocke`/`kernels` resolve wherever a UKD names them. Producer
#   selection stays per-UKD on kernel_source.kind, so a root holding no rocKE
#   descriptor never invokes that producer and pays only the interpreter.
#
#   ROCKE_READY is the venv's wheel-install stamp, and is what the pack step
#   depends on rather than the interpreter itself: the interpreter's own rule
#   carries no content dependency, so an edge to it would not restage when a
#   kernel under rocke/library changes. ROCKE_WHEEL_STAMP is the wheel content
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
#   NAME is the source label written into every descriptor's provenance. NAME, the
#   absolute SOURCE_ROOT, OUT_ROOT and ARCHES go into a global registry read by
#   hkp_verify_embedded_sources() and hkp_register_census_tests().
# ---------------------------------------------------------------------------
function(hkp_wire_pack_target)
    set(_one NAME SOURCE_ROOT ARCHES HIPCC ROCM_KPACK_DIR
        OUT_ROOT ROCKE_INTERP ROCKE_READY ROCKE_COMGR_LIB
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

    # Every root runs under the wheel interpreter, including hip-only ones that do
    # not need it: hip compiles shell out to hipcc and are interpreter-agnostic,
    # so the cost is the interpreter and nothing else. Selecting per root is what
    # let a rocKE descriptor land in a root that could not import rocke, where it
    # was not skipped but attempted -- surfacing as a mid-build ImportError rather
    # than as a configuration error.
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
    file(GLOB_RECURSE _source_inputs CONFIGURE_DEPENDS
         "${ARG_SOURCE_ROOT}/*")

    # Editing the tool's own sources must retrigger the pack step, else the
    # artifacts go stale against the current pipeline code. The resolved
    # rocm_kpack package counts too: kpack_resolver.py imports it and it decides
    # the archive format, so a packer change there must invalidate the stamp.
    file(GLOB _tool_sources CONFIGURE_DEPENDS
         "${HKP_PYTHON_ROOT}/hkp_pack/*.py"
         "${ARG_ROCM_KPACK_DIR}/rocm_kpack/*.py")

    # The globs above carry each input as its own edge, which covers an added or
    # edited file but not a REMOVED one: a shorter DEPENDS list makes no input
    # newer and changes no command, so the edge would stay clean, the wipe below
    # would never fire, and the staged copy of a deleted descriptor would survive
    # an incremental build.
    #
    # This manifest puts the input SET into the edge. Its content changes when a
    # path leaves either glob, which makes it newer than the stamp and forces the
    # pack. COPYONLY rewrites only when the content differs, so an unchanged tree
    # does not repack on every configure. It lives in the binary dir rather than
    # under ARG_OUT_ROOT because the pack command wipes that tree -- a dependency
    # deleted by the command it guards would make every build repack.
    set(_input_manifest "${CMAKE_CURRENT_BINARY_DIR}/hkp-${ARG_NAME}-inputs.txt")
    set(_manifest_paths ${_source_inputs} ${_tool_sources})
    list(SORT _manifest_paths)
    string(REPLACE ";" "\n" _manifest_body "${_manifest_paths}")
    file(WRITE "${_input_manifest}.in" "${_manifest_body}\n")
    configure_file("${_input_manifest}.in" "${_input_manifest}" COPYONLY)

    hkp_require_kpack_runtime("${_interp}" "the ${_interp_what}")

    string(REPLACE ";" "," _arch_csv "${ARG_ARCHES}")

    set(_wheel_stamp_arg "")
    if(_wheel_dep)
        set(_wheel_stamp_arg --rocke-wheel-stamp "${_wheel_dep}")
    endif()

    set(_tool_cmd "${CMAKE_COMMAND}" -E env ${_tool_env} "${_interp}"
        "${HKP_TOOL}")

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
                "${_input_manifest}"
                ${_interp_dep} ${_wheel_dep}
        COMMENT "hkp: packing root '${ARG_NAME}' for ${ARG_ARCHES}"
        VERBATIM)

    add_custom_target(hkp_packaging_${ARG_NAME} ALL
                      DEPENDS "${_stamp}"
                      COMMENT "hkp: descriptor packaging (${ARG_NAME})")
    if(TARGET hkp_rocke_wheel_python_interp)
        # Every root shares one venv. A file-level edge alone leaves generators
        # that build per directory copying the provisioning recipe into each pack
        # target, so a parallel fresh build can reprovision the venv while
        # another pack is using it.
        add_dependencies(hkp_packaging_${ARG_NAME} hkp_rocke_wheel_python_interp)
    endif()
    set_property(GLOBAL PROPERTY HKP_PACK_STAMP_${ARG_NAME} "${_stamp}")

    # The key manifest normalises each registered path the same lexical way, so
    # the two spellings agree and the verify step compares them exactly.
    get_filename_component(_abs_source_root "${ARG_SOURCE_ROOT}" ABSOLUTE)
    set_property(GLOBAL PROPERTY HKP_PACK_SOURCE_ROOT_${ARG_NAME} "${_abs_source_root}")

    # Where this root's shards land and which arches it was wired for.
    # hkp_register_census_tests() reads both to address this root's OWN shard.
    set_property(GLOBAL PROPERTY HKP_PACK_OUT_ROOT_${ARG_NAME} "${ARG_OUT_ROOT}")
    set_property(GLOBAL PROPERTY HKP_PACK_ARCHES_${ARG_NAME} "${ARG_ARCHES}")

    set_property(GLOBAL APPEND PROPERTY HKP_PACK_LABELS "${ARG_NAME}")
endfunction()

# ---------------------------------------------------------------------------
# _hkp_record_dormant_pack(<name>)
#   Record <name> as known-but-deliberately-unwired, in the registry
#   hkp_wire_pack_target() fills. A consumer must tell wired, dormant and unknown apart,
#   and absence from HKP_PACK_LABELS alone collapses the last two. The name is the whole
#   record: nothing was packed, so there is no OUT_ROOT, arch list or stamp.
# ---------------------------------------------------------------------------
function(_hkp_record_dormant_pack name)
    set_property(GLOBAL APPEND PROPERTY HKP_PACK_DORMANT_LABELS "${name}")
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
#   in the provisioned venv (last step of hkp_rocke_wheel_python_interp):
#   neither the venv nor the wheels exist at configure time, and the build
#   imports from the wheels, not from the source tree.
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
#   Keying the venv and the pack step on wheel mtime would therefore recompile
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
# hkp_rocke_wheel_python_interp(<out_interp> <out_ready> <wheel_stamp>)
#   Provision a build-local interpreter carrying the rocke + rocke_library
#   wheels. With ROCKE_BUILD_PYENV ON they are built by the rocke-wheels target,
#   which rides HIPKERNELPROVIDER_ENABLE_ROCKE; with it OFF there is no such
#   target and the wheels are supplied through ROCKE_WHEEL_DIR. Every pack step
#   imports rocke/kernels from these wheels rather than from the editable dev
#   venv, in either provenance.
#
#   Not ROCKE_PYENV_PYTHON: production ships wheels, so the packs must test wheels.
#
#   TWO rules, with deliberately different dependency sets:
#
#     Rule A produces the interpreter. Expensive -- it reaches the index for
#     msgpack/zstandard -- and carries NO content dependency, so it runs once per
#     build tree and never again.
#     Rule B produces <out_ready>, reinstalling the wheels into that venv. Cheap
#     and offline, and the only rule keyed on wheel content.
#
#   Merged, as they once were, one edited rocKE kernel tore the whole venv down
#   and re-provisioned it over the network. Pack steps therefore depend on
#   <out_ready>, never on <out_interp>.
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
#   Rule B depends on the wheel digest stamp, not the wheels, so a byte-identical
#   rebuild does not reinstall.
#
#   The rocke import check runs in RULE B, as its last step, rather than at
#   configure time: the venv and the wheels are both add_custom_command outputs
#   that do not exist until the build runs, so there is nothing to probe at
#   configure time. Running it there also means it validates exactly the wheels
#   just installed, in the interpreter the pack step will use.
# ---------------------------------------------------------------------------
function(hkp_rocke_wheel_python_interp out_interp out_ready wheel_stamp)
    set(_venv "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-venv")
    if(WIN32)
        set(_venv_py "${_venv}/Scripts/python.exe")
    else()
        set(_venv_py "${_venv}/bin/python")
    endif()
    set(_ready "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-venv.installed")

    # `cmake --fresh` removes CMakeCache.txt and CMakeFiles/ only, so the venv
    # would survive one -- and Rule A, having no content dependency, would never
    # rebuild it. Keying on a cache variable makes --fresh mean what it says:
    # absent cache, absent marker, wipe. It must be CACHE; a normal variable does
    # not survive a configure and would wipe the venv on every one.
    #
    # This is the documented remedy for the staleness Rule A accepts: a changed
    # Python3_EXECUTABLE, or a raised msgpack/zstandard floor, leaves the old venv
    # in place until someone reconfigures fresh.
    if(NOT DEFINED HKP_ROCKE_VENV_GENERATION)
        file(REMOVE_RECURSE "${_venv}")
        set(HKP_ROCKE_VENV_GENERATION 1 CACHE INTERNAL
            "Marks hkp-rocke-venv as belonging to this cache generation")
    endif()

    set(_platform_wheel
        "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    set(_library_wheel
        "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")

    # Rule A -- the venv itself, carrying rocm_kpack's runtime dependencies (see
    # hkp_require_kpack_runtime). Those two come from the index, unlike the rocke
    # wheels: they are third-party packages with no local artifact to install
    # from, and they are what makes this the expensive rule. Scoped to exactly
    # these two pinned-floor names, so the venv stays reproducible in everything
    # that describes OUR code.
    #
    # No wheel dependency, so editing a rocKE kernel never reaches this rule.
    add_custom_command(
        OUTPUT "${_venv_py}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_venv}"
        COMMAND "${Python3_EXECUTABLE}" -m venv --copies "${_venv}"
        COMMAND "${_venv_py}" -m pip install -q
                "msgpack>=1.0.0" "zstandard>=0.20.0"
        COMMENT "hkp: provisioning hermetic rocke wheel interpreter"
        VERBATIM)

    # Rule B -- the wheels in it. Offline, and the only rule keyed on their content.
    add_custom_command(
        OUTPUT "${_ready}"
        COMMAND "${_venv_py}" -m pip install -q
                --no-index --no-deps --force-reinstall
                "${_platform_wheel}" "${_library_wheel}"
        # Probe what the pack step will actually import, in the interpreter it
        # will actually use -- rocke/kernels AND the kpack stack.
        COMMAND "${_venv_py}" -c
                "import rocke, kernels, msgpack, zstandard"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ready}"
        DEPENDS "${_venv_py}" "${wheel_stamp}" "${HKP_WHEEL_DIGEST_TOOL}"
        COMMENT "hkp: installing rocke wheels into the pack interpreter"
        VERBATIM)

    add_custom_target(hkp_rocke_wheel_python_interp ALL DEPENDS "${_ready}"
                      COMMENT "hkp: rocke wheel python interpreter")
    add_dependencies(hkp_rocke_wheel_python_interp hkp_rocke_wheel_digest)
    set(${out_interp} "${_venv_py}" PARENT_SCOPE)
    set(${out_ready} "${_ready}" PARENT_SCOPE)
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
#   venv provisioning's pip install or import. Fail here for all three instead,
#   and name the remedy.
#
#   The wheel check covers SUPPLY, not importability. The interpreter that
#   imports rocke/kernels is the venv hkp_rocke_wheel_python_interp provisions,
#   an add_custom_command OUTPUT that does not exist until the build runs; the
#   import is asserted there, in the interpreter the pack step will use.
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
# _hkp_resolve_production_root(<out_var> <out_is_default>)
#   Declare the overridable production source root and resolve it to a path or to empty.
#   Empty is the dormant case and not an error; a value that is set but is not a
#   directory is fatal.
#
#   <out_is_default> reports whether the resolved root is still the built-in default,
#   which callers that turn "nothing to ship" into an error read more gently than a root
#   this build named. A cache entry records no author, so is-default is decided by
#   comparing against the default path.
# ---------------------------------------------------------------------------
function(_hkp_resolve_production_root out_var out_is_default)
    set(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT
        "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}" CACHE PATH
        "The authored source root the production pack step compiles from, \
defaulting to the provider's in-tree shipped descriptors. Walked recursively; child \
folders under it scope the content (hip/, rocKE/, per-integration folders) and each \
descriptor's authored subpath is preserved into the staged and installed trees. A root \
holding no descriptor, like an empty value, leaves production packaging dormant.")

    set(${out_var} "" PARENT_SCOPE)
    if("${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}" STREQUAL
       "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}")
        set(${out_is_default} TRUE PARENT_SCOPE)
    else()
        set(${out_is_default} FALSE PARENT_SCOPE)
    endif()

    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but is "
                "not a directory: ${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
        endif()
        set(${out_var} "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_path_is_hidden(<out_var> <root> <path>)
#   TRUE when any segment of <path> below <root> is dot-prefixed, which is how
#   load_flat_input() decides a file is not authored content. Shared so the two functions
#   below cannot drift apart.
# ---------------------------------------------------------------------------
function(_hkp_path_is_hidden out_var root path)
    set(${out_var} FALSE PARENT_SCOPE)
    file(RELATIVE_PATH _rel "${root}" "${path}")
    string(REPLACE "/" ";" _segments "${_rel}")
    foreach(_segment IN LISTS _segments)
        if(_segment MATCHES "^\\.")
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_root_has_kdp(<out_var> <root>)
#   TRUE when <root> holds at least one non-hidden *.kdp.json. An empty <root> is FALSE
#   rather than an error, which is what leaves packaging dormant. CONFIGURE_DEPENDS so
#   adding the first KDP re-runs configure and wires the target. Dot-prefixed segments
#   are skipped as load_flat_input() skips them, so a `.git/` under a user-supplied root
#   is not content.
# ---------------------------------------------------------------------------
function(_hkp_root_has_kdp out_var root)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT root)
        return()
    endif()

    file(GLOB_RECURSE _kdps CONFIGURE_DEPENDS "${root}/*.kdp.json")
    foreach(_kdp IN LISTS _kdps)
        _hkp_path_is_hidden(_kdp_hidden "${root}" "${_kdp}")
        if(NOT _kdp_hidden)
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_root_covers_any_arch(<out_var> <root> <arches>)
#   TRUE when at least one non-hidden *.kdp.json under <root> would survive
#   arch_matches() for at least one arch in <arches>. Mirrors that predicate exactly: an
#   absent `arch` key and an empty `arch` array are both wildcards, anything else is
#   exact string membership in the wired arch list. Consulted for the default root alone
#   (hkp_add_packaging below).
#
#   Only FALSE is authoritative. kdp_survives() tests arch_matches() first, so a root no
#   arch matches is provably empty; TRUE claims nothing beyond "not provably empty",
#   since a matching KDP can still prune on its UKD entries. Every ambiguous case
#   therefore resolves to TRUE -- an unparseable KDP, or an `arch` that is not an array,
#   counts as covering, so the root stays wired and the packer reports what is wrong
#   with it.
#
#   CONFIGURE_DEPENDS for the same reason as _hkp_root_has_kdp.
# ---------------------------------------------------------------------------
function(_hkp_root_covers_any_arch out_var root arches)
    # cmake-lint: disable=E1120
    #   cmake-lint carries no argument spec for foreach(... RANGE ...) and reports
    #   every spelling of it as missing a positional argument. The index loop below
    #   is valid CMake.
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT root)
        return()
    endif()

    file(GLOB_RECURSE _kdps CONFIGURE_DEPENDS "${root}/*.kdp.json")
    foreach(_kdp IN LISTS _kdps)
        _hkp_path_is_hidden(_kdp_hidden "${root}" "${_kdp}")
        if(_kdp_hidden)
            continue()
        endif()

        # CONFIGURE_DEPENDS re-globs when the SET of files changes, but this answer turns
        # on their CONTENTS: without a content dependency, editing a KDP's `arch` leaves
        # the previous verdict standing, so a root that starts declaring this build's
        # architecture stays dormant with no configure to say otherwise.
        set_property(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_kdp}")

        file(READ "${_kdp}" _kdp_json)

        # One error variable covers two ambiguous cases that both resolve to TRUE: the
        # absent-key wildcard, and an unparseable file the packer must be the one to
        # report.
        string(JSON _arch_type ERROR_VARIABLE _arch_err TYPE "${_kdp_json}" arch)
        if(_arch_err OR NOT _arch_type STREQUAL "ARRAY")
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()

        string(JSON _arch_len ERROR_VARIABLE _len_err LENGTH "${_kdp_json}" arch)
        if(_len_err OR _arch_len EQUAL 0)
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()

        math(EXPR _arch_last "${_arch_len} - 1")
        foreach(_i RANGE ${_arch_last})
            string(JSON _arch ERROR_VARIABLE _get_err GET "${_kdp_json}" arch ${_i})
            if(_get_err OR _arch IN_LIST arches)
                set(${out_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_resolve_rocke_args(out_args out_comgr_lib)
#   Resolve the rocKE toolchain once and return the keyword list every pack target is
#   wired with, plus the comgr library the ctest entries forward. Called once for all
#   roots: hkp_rocke_wheel_python_interp declares a custom command OUTPUT and a target,
#   which a second call would duplicate.
# ---------------------------------------------------------------------------
function(_hkp_resolve_rocke_args out_args out_comgr_lib)
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
    hkp_rocke_wheel_python_interp(_rocke_interp _rocke_ready "${_rocke_wheel_stamp}")

    # One list for every root, so "every root is wired to rocKE identically" is
    # structural rather than six sites that have to agree. COMGR_LIB is appended
    # only when set: an empty element does not survive unquoted expansion, and
    # losing one would shift every following keyword into the wrong slot.
    set(_rocke_args
        ROCKE_INTERP "${_rocke_interp}"
        ROCKE_READY "${_rocke_ready}"
        ROCKE_WHEEL_STAMP "${_rocke_wheel_stamp}")
    if(_rocke_comgr_lib)
        list(APPEND _rocke_args ROCKE_COMGR_LIB "${_rocke_comgr_lib}")
    endif()

    set(${out_args} "${_rocke_args}" PARENT_SCOPE)
    set(${out_comgr_lib} "${_rocke_comgr_lib}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Gate production packaging on ONE source root; producer selection is per-UKD on
#   kernel_source.kind, so both producers are available to every root. Runs only under
#   HIPDNN_ENABLE_KERNEL_INGESTOR, whose prerequisites it asserts first through
#   hkp_require_ingestor_toolchain. rocKE is required for the test roots as much as for
#   production, so it is resolved once here for every root; unresolvable comgr is fatal
#   at configure.
#
#   The root defaults to the provider's in-tree descriptor root, which currently holds no
#   descriptor, so production packaging is dormant unless the root is pointed at a
#   populated one. Root empty, or holding no descriptor = dormant. The default root also
#   goes dormant when no descriptor under it declares an architecture this build packs
#   for; a named root in the same state is the packer's hard failure. Root set but not a
#   directory = fatal. The tests are wired regardless.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_rocm_kpack_dir "${Python3_EXECUTABLE}")
    hkp_require_ingestor_toolchain(_arches)

    _hkp_resolve_production_root(_source_root _source_root_is_default)

    _hkp_resolve_rocke_args(_rocke_args _rocke_comgr_lib)

    # A KDP is what arch pruning consumes, so a root holding none has nothing to ship.
    # Standalone UKD/UMD/UED/UDD/KMD/UHD files, kernel sources and READMEs do not make
    # a pack.
    _hkp_root_has_kdp(_product_has_content "${_source_root}")
    if(NOT _source_root)
        set(_product_dormant_reason "empty-root")
    else()
        set(_product_dormant_reason "no-kdp")
    endif()

    # Arch coverage is consulted for the DEFAULT root alone, which builds inherit without
    # asking for it; a named root reaches the packer and fails there. Safe in one
    # direction only: arch_matches() runs first inside kdp_survives(), so "no KDP
    # declares an arch this build packs for" proves no KDP survives, and a root this
    # misses stays wired for the packer to report.
    if(_product_has_content AND _source_root_is_default)
        _hkp_root_covers_any_arch(_product_covers_arch "${_source_root}" "${_arches}")
        if(NOT _product_covers_arch)
            set(_product_has_content FALSE)
            set(_product_dormant_reason "no-arch")
        endif()
    endif()

    # Production descriptors.
    if(_source_root AND _product_has_content)
        hkp_wire_pack_target(
            NAME product
            SOURCE_ROOT "${_source_root}"
            ARCHES "${_arches}"
            HIPCC "${HKP_HIPCC}"
            ROCM_KPACK_DIR "${_rocm_kpack_dir}"
            OUT_ROOT "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}"
            ${_rocke_args}
            PACK_JOBS 2)
    else()
        # Every dormant reason passes through here, so none can reach a message(STATUS)
        # while leaving 'product' looking misspelled to hkp_register_census_tests().
        _hkp_record_dormant_pack(product)

        # A tree left from an earlier configuration that did pack keeps being loaded:
        # the engine selects the plugin-relative directory on existence alone, and
        # nothing else removes it once the pack target and its install rule are gone.
        if(HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR)
            file(REMOVE_RECURSE "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}")
        endif()
        # One message per reason. The arch line names the arch list because that is the
        # value to change to make packing happen.
        if(_product_dormant_reason STREQUAL "empty-root")
            message(STATUS
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is empty; production "
                "packaging dormant (tests still run against the fixtures).")
        elseif(_product_dormant_reason STREQUAL "no-arch")
            message(STATUS
                "hkp: the default production root '${_source_root}' declares no "
                "descriptor for any architecture this build packs for (${_arches}), "
                "so every descriptor under it would prune; production packaging "
                "dormant (tests still run against the fixtures).")
        else()
            message(STATUS
                "hkp: no *.kdp.json under '${_source_root}'; production packaging "
                "dormant (tests still run against the fixtures).")
        endif()
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

    # Both entries are add_test()'d in this scope, so the YAML's test_patterns match them
    # via the directory-property loop. EXPLICIT_TESTS is avoided:
    # apply_ctest_category_labels joins it with ';', which execute_process re-splits into
    # separate argv, leaking a second name into the parser's positional install-file slot.
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
       AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endfunction()


# ---------------------------------------------------------------------------
# _hkp_join_census_cases(<out-var> [<case>...])
#
# Packs an EXPECTED_CASES list into the comma-separated string the binary reads. Comma
# rather than the semicolon CMake lists use: the ENVIRONMENT test property is itself a
# semicolon-separated list of VAR=VALUE.
# ---------------------------------------------------------------------------
function(_hkp_join_census_cases _outvar)
    set(_cases "${ARGN}")
    foreach(_case IN LISTS _cases)
        if(_case MATCHES ",")
            message(FATAL_ERROR
                "hkp: expected census case '${_case}' contains a comma, which is "
                "the separator the pin is delivered with, so the binary would read "
                "it as two names. A GTest case name cannot hold one; this is a typo.")
        endif()
    endforeach()
    list(JOIN _cases "," _joined)
    set(${_outvar} "${_joined}" PARENT_SCOPE)
endfunction()


# ---------------------------------------------------------------------------
# _hkp_add_census_test(<name> <target> <gtest-filter> <environment> <pass-regex>)
#
# One CTest entry of a census family. Entry and controls go through here so a drifting
# command or label cannot leave a control no longer controlling its entry.
#
# Empty <pass-regex> is the census entry itself, verdicted on exit status. Nonempty is a
# control, and names the diagnostic the census prints when it refuses; asserting only
# "exited nonzero" would also be satisfied by a binary that never launched.
# PASS_REGULAR_EXPRESSION REPLACES the exit-status check rather than adding to it, so a
# control carries it ALONE; CTest still fails a test that times out or dies on a signal.
# The entry instead carries FAIL_REGULAR_EXPRESSION on the diagnostic prefix, catching a
# run that prints a refusal yet exits zero.
# ---------------------------------------------------------------------------
function(_hkp_add_census_test _name _target _filter _environment _pass_regex)
    add_test(
        NAME "${_name}"
        COMMAND "$<TARGET_FILE:${_target}>"
                "--gtest_filter=${_filter}")
    # TEST_ENVIRONMENT (ASAN symbolizer path, HSA_XNACK, the MIOpen cache redirect) goes
    # FIRST and the census-specific entries LAST: CTest resolves a repeated variable to
    # its last occurrence, so a census value wins any collision with an ambient one.
    set(_merged_environment "")
    if(DEFINED TEST_ENVIRONMENT)
        list(APPEND _merged_environment ${TEST_ENVIRONMENT})
    endif()
    list(APPEND _merged_environment ${_environment})

    # A census run is one host-only process -- no device, no compile -- so it lands in
    # seconds. 300 absorbs a sanitizer build's slowdown, well inside ctest's 1500 s
    # default.
    set_tests_properties("${_name}" PROPERTIES
        ENVIRONMENT "${_merged_environment}"
        LABELS "unit_test;hip-kernel-provider;host"
        TIMEOUT 300)

    # PATH prepends (Windows ASAN runtime / ROCm / build DLL dirs) go through
    # ENVIRONMENT_MODIFICATION so the runtime PATH is extended rather than replaced, which
    # a literal PATH= entry in ENVIRONMENT cannot do. It needs its own guard: it is
    # Windows-only under BUILD_ADDRESS_SANITIZER, while TEST_ENVIRONMENT is also defined
    # for both THEROCK_SANITIZER ASAN flavours.
    if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
        set_tests_properties("${_name}" PROPERTIES
            ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
    endif()

    if(_pass_regex)
        set_tests_properties("${_name}" PROPERTIES
            PASS_REGULAR_EXPRESSION "${_pass_regex}")
    else()
        set_tests_properties("${_name}" PROPERTIES
            FAIL_REGULAR_EXPRESSION "Census: ")
    endif()
endfunction()


# ---------------------------------------------------------------------------
# _hkp_add_census_entry(<target> <suite> <arch> <shard> <joined-cases>)
#
# One suite's census at one architecture, together with the controls that make it
# observable as a gate. <joined-cases> comes from _hkp_join_census_cases(); empty
# suppresses both the pin and the control that watches it.
# ---------------------------------------------------------------------------
function(_hkp_add_census_entry _target _suite _arch _shard _cases)
    set(_name "hip-kernel-provider-hkp-census-${_arch}-${_suite}")
    # HIPDNN_DESCRIPTOR_RUNTIME_DIR is pinned empty because descriptorSearchDirectories()
    # APPENDS it to the explicit root rather than being overridden by one: left ambient,
    # an export at a multi-arch tree draws a refusal that the root spans shards. Empty is
    # what the loader already treats as absent.
    #
    # Split at the expected arch so the control that varies only that value rebuilds the
    # rest from the same string.
    set(_env_without_arch "HIPDNN_TEST_CENSUS_SUITE=${_suite};HIPDNN_DESCRIPTOR_RUNTIME_DIR=")
    set(_common "${_env_without_arch};HIPDNN_TEST_EXPECTED_ARCH=${_arch}")
    set(_pin "")
    if(_cases)
        set(_pin ";HIPDNN_TEST_CENSUS_EXPECTED_CASES=${_cases}")
    endif()

    _hkp_add_census_test("${_name}" "${_target}" "${_suite}.*"
                         "${_common};HIPDNN_DESCRIPTOR_DIR=${_shard}${_pin}" "")

    # Control: every case in the suite goes unvisited -- the filter's negative half
    # cancels its positive half, so only the listener's per-iteration completion check
    # turns this red. Neither the filter nor the regex names a case.
    _hkp_add_census_test("${_name}-control-unvisited" "${_target}"
                         "${_suite}.*-${_suite}.*"
                         "${_common};HIPDNN_DESCRIPTOR_DIR=${_shard}${_pin}"
                         "did not complete .* successfully")

    # Control: the explicit root does not exist. The shard name is a sentinel no pack
    # rule writes, and the regex is the preflight's own refusal, so a fallback to the
    # binary's compiled-in root cannot satisfy it.
    _hkp_add_census_test("${_name}-control-absent-root" "${_target}" "${_suite}.*"
                         "${_common};HIPDNN_DESCRIPTOR_DIR=${_shard}-hkp-census-control-absent${_pin}"
                         "Census requires a nonempty HIPDNN_TEST_EXPECTED_ARCH and an existing explicit HIPDNN_DESCRIPTOR_DIR")

    # Control: the loaded packs carry a stamp other than the expected one. Identical to
    # the entry but for the expected arch: 'gfxhkpcensuscontrol' fails
    # hkp_selected_arches()'s ^gfx[0-9a-f]+$ filter, the only path by which an arch
    # reaches a shard name. The regex is the stamp comparison's own wording.
    _hkp_add_census_test("${_name}-control-unexpected-stamp" "${_target}" "${_suite}.*"
                         "${_env_without_arch};HIPDNN_TEST_EXPECTED_ARCH=gfxhkpcensuscontrol;HIPDNN_DESCRIPTOR_DIR=${_shard}${_pin}"
                         "packs loaded from this root carry the stamps")

    # Control: the pin names a case the suite does not register; the sentinel is not
    # registered by the current suite. The regex is the name check's own wording, which
    # separates this from -control-unvisited: both exit nonzero, only this prints it.
    if(_cases)
        _hkp_add_census_test("${_name}-control-unregistered-case" "${_target}"
                             "${_suite}.*"
                             "${_common};HIPDNN_DESCRIPTOR_DIR=${_shard};HIPDNN_TEST_CENSUS_EXPECTED_CASES=${_cases},HkpCensusControlCaseThatIsNeverRegistered"
                             "is expected but not registered, so the suite has lost a case")
    endif()
endfunction()


# ---------------------------------------------------------------------------
# _hkp_require_census_declaration(suites target pack_name)
#   Fail configure when a census declaration names suites but omits the TARGET or
#   PACK_NAME that would run them. Each condition is independently fatal.
# ---------------------------------------------------------------------------
function(_hkp_require_census_declaration suites target pack_name)
    if(NOT target)
        message(FATAL_ERROR
            "hkp: census suites are declared (${suites}) without a TARGET, so "
            "no binary could run them.")
    endif()
    if(NOT pack_name)
        message(FATAL_ERROR
            "hkp: census suites are declared (${suites}) without a PACK_NAME, "
            "so no shard could be named.")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR
            "hkp: census suites are declared (${suites}) but the target "
            "${target} does not exist, so no census could be registered. This "
            "call must run after that target is created.")
    endif()
endfunction()


# ---------------------------------------------------------------------------
# The emitted-bundle census. Each generated engine ships a GTest suite that reads what
# loaded through discoverDescriptorSets() and loadValidatedDescriptorSets<Handle>(), and
# compares the loaded pack/kernel identities, runtime source kind and SDK version against
# the inventory its generation emitted: a pack whose symbols do not register drops its
# descriptors at load, and the census sees them missing.
#
# The architecture is supplied EXPLICITLY, from the registry (HKP_PACK_ARCHES_<name>)
# rather than from a probe: a host census must not depend on which card is in the
# machine. Each suite gets an entry per selected arch against that arch's own shard under
# HKP_PACK_OUT_ROOT_<name>, so a suite is declarable only where it reads exactly one
# pack's shard. Call this once per packed target, beside hkp_verify_embedded_sources();
# a missing PACK_NAME, TARGET or recorded arch list is fatal.
#
# A PACK_NAME the registry knows only as DORMANT registers nothing and says so at STATUS;
# an unknown name stays fatal. EXPECTED_CASES optionally pins ONE suite's case-name set
# -- names, never a count, because a case added and a case lost cancel in a count -- and
# supplying the keyword with no names is fatal.
# ---------------------------------------------------------------------------
function(hkp_register_census_tests)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()

    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;PACK_NAME" "SUITES;EXPECTED_CASES")
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "hkp_register_census_tests: unrecognised argument(s): "
            "${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_SUITES)
        return()
    endif()
    if("EXPECTED_CASES" IN_LIST ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR
            "hkp: census suites are declared (${ARG_SUITES}) with an EXPECTED_CASES "
            "keyword that names no case. An empty pin admits every case set, which "
            "reads as a pinned suite while checking nothing. List the suite's cases, "
            "or drop the keyword.")
    endif()

    _hkp_require_census_declaration("${ARG_SUITES}" "${ARG_TARGET}" "${ARG_PACK_NAME}")

    get_property(_labels GLOBAL PROPERTY HKP_PACK_LABELS)
    get_property(_dormant_labels GLOBAL PROPERTY HKP_PACK_DORMANT_LABELS)
    if(NOT ARG_PACK_NAME IN_LIST _labels)
        # Registering nothing and saying so keeps a generated integration's census call
        # valid across every configuration; hkp_add_packaging() already named the reason.
        if(ARG_PACK_NAME IN_LIST _dormant_labels)
            message(STATUS
                "hkp: pack target '${ARG_PACK_NAME}' is dormant in this configuration, "
                "so it stages no shard and the census suites declared against it "
                "(${ARG_SUITES}) are not registered. The dormancy message earlier in "
                "this configure names the reason.")
            return()
        endif()
        message(FATAL_ERROR
            "hkp: census suites are declared (${ARG_SUITES}) at pack target "
            "'${ARG_PACK_NAME}', which no hkp_wire_pack_target() call wired, so "
            "there is no shard to census. Wired roots: ${_labels}. Roots left "
            "dormant by this configuration: ${_dormant_labels}.")
    endif()

    get_property(_out_root GLOBAL PROPERTY HKP_PACK_OUT_ROOT_${ARG_PACK_NAME})
    get_property(_arches GLOBAL PROPERTY HKP_PACK_ARCHES_${ARG_PACK_NAME})
    if(NOT _arches)
        message(FATAL_ERROR
            "hkp: census suites are declared (${ARG_SUITES}) at pack target "
            "'${ARG_PACK_NAME}', which was wired with an empty architecture list, "
            "so no shard exists to census. Set GPU_TARGETS/AMDGPU_TARGETS.")
    endif()

    # A pin names ONE suite's cases. Spread over several it would demand that each
    # register the same set, so the mistake would read as a broken suite rather than as a
    # misplaced argument.
    list(LENGTH ARG_SUITES _suite_count)
    if(ARG_EXPECTED_CASES AND NOT _suite_count EQUAL 1)
        message(FATAL_ERROR
            "hkp: EXPECTED_CASES pins one suite's case-name set, but this call "
            "declares ${_suite_count} suites (${ARG_SUITES}). Split the call so each "
            "pinned suite carries its own list.")
    endif()

    _hkp_join_census_cases(_census_expected_cases ${ARG_EXPECTED_CASES})

    foreach(_suite IN LISTS ARG_SUITES)
        # The entry name carries the arch and the suite and nothing of the pack, so the
        # same suite declared at a second pack target would ask CTest for one name twice,
        # and the second registration would silently take the first one's shard.
        get_property(_owner GLOBAL PROPERTY HKP_CENSUS_SUITE_OWNER_${_suite})
        if(_owner)
            message(FATAL_ERROR
                "hkp: census suite '${_suite}' is declared at two pack targets, "
                "'${_owner}' and '${ARG_PACK_NAME}'. A census entry is named for its "
                "suite and arch alone, so the two collide. Declare the suite at the "
                "one pack whose shard it reads.")
        endif()
        set_property(GLOBAL PROPERTY HKP_CENSUS_SUITE_OWNER_${_suite} "${ARG_PACK_NAME}")

        foreach(_census_arch IN LISTS _arches)
            _hkp_add_census_entry("${ARG_TARGET}" "${_suite}" "${_census_arch}"
                                  "${_out_root}/${_census_arch}"
                                  "${_census_expected_cases}")
        endforeach()
    endforeach()

    # The loop above add_test()'d every entry in THIS directory scope -- a CMake function
    # opens none of its own -- so the YAML's regex patterns reach them through the
    # parser's directory-property enumeration; see hkp_register_tests() for why
    # EXPLICIT_TESTS is not used. Tier expansion is the parser's, which is why
    # _hkp_add_census_test()'s literal LABELS string cannot carry it. The call sits after
    # the loop because every return above it registers nothing.
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
       AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endfunction()
