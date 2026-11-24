# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HkpSourceDigest.cmake")

# ============================================================================
# hkp_asm_sdpa_revision(<out-revision> <engine-dir> [CONFIGURE_DEPENDS] [MANIFEST <file>])
# ============================================================================
# The ASM SDPA selector revision: the provider prefix and 16 hex digits over everything that
# decides which forward or backward kernel runs and how it is launched. The L1 models
# estimate both directions, record this revision, and the loader refuses a mismatch, so it
# must hash exactly those inputs: a missing input lets a stale model silently pick the wrong
# engine; an extra one expires models for unrelated changes. This is the one place that
# writes the revision's form: the build passes <out-revision> to AsmSdpaEngine.cpp as
# HKP_ASM_SDPA_REVISION, and the manifest records the same string.
#
# Retraining. Any edit to an input, a comment or a clang-tidy fix included, moves the
# revision and expires both shipped models (descriptors/predict_engine/<arch>/): the loader
# refuses them, and ShippedL1ModelsRecordTheRevisionThisBuildComputes in
# src/tests/engines/asm_sdpa_engine/TestAsmSdpaEngine.cpp fails the build. The change that
# moves the revision retrains both, for each of gfx942 (MI300X) and gfx950 (MI355X). The
# steps are flows of the hipDNN UHD pipeline, which is an external service: this repository
# defines no flow. By hand they are the local hipdnn-uhd skill (projects/hipdnn/tools/ai/
# skills/hipdnn-uhd/: generate.md, data.md, validate.md) over the `uhd_gen` trainer
# (projects/hipdnn/tools/uhd_gen/README.md), with the commands named below:
#   1. Measure at the new revision: collect a corpus of ASM_SDPA_ENGINE on that SKU (flow
#      uhd-collect; `uhd_gen generate --collect-only --role predict_engine ...`), or carry
#      the previous revision's rows forward (flow uhd-dataset-carry-forward; pipeline only)
#      where re-measured graphs show the change moved no timing.
#   2. Train with role predict_engine, metric tflops and the architecture's id from
#      AsmSdpaEngine::L1_MODEL_IDS, so the compiled-in id still binds (flow uhd-generate;
#      `uhd_gen generate --no-promote --role predict_engine --metric tflops --uhd-id <id>
#      ...`), then validate it on that SKU (flow uhd-validate; validate.md, Stage V).
#   3. Apply the model: the packet of the pipeline run, or `uhd_gen promote` by hand. It
#      writes asm_sdpa_engine_tflops.uhd.json and .bin under descriptors/predict_engine/
#      <arch>/.
# A change that provably leaves every kernel choice alone (a file moved, or an input added
# that is byte-identical at the commit the models were trained at) may instead relabel both
# models: edit `trained_against.selector_revision` in each asm_sdpa_engine_tflops.uhd.json
# to the new revision (the build prints it, and the selector dump names it). The .bin stays
# as it is: the .uhd.json refers to it by `mlp_data.hash`, and the revision is not in it.
# ShippedL1ModelsRecordTheRevisionThisBuildComputes passing is the confirmation, with the
# proof in the commit message. The selector dumps of the two commits (cmake/
# HkpSelectorManifest.cmake) name the inputs that changed.
#
# CONFIGURE_DEPENDS makes every input a configure dependency so an edit or a new kernel
# reconfigures. Test fixture trees omit it: they are scratch copies, rewritten each time the
# fixture build step runs.
# MANIFEST writes the selector dump of these inputs (HkpSourceDigest.cmake).
function(hkp_asm_sdpa_revision _out_revision _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "MANIFEST" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()
    hkp_asm_sdpa_inputs(_inputs "${_engine_dir}" ${_track})
    hkp_asm_sdpa_revision_prefix(_prefix)

    # Keyed by relative path: the same kernel name exists under both MI300/ and MI308/.
    set(_manifest "")
    if(_arg_MANIFEST)
        set(_manifest MANIFEST "${_arg_MANIFEST}" ENGINE ASM_SDPA_ENGINE PREFIX "${_prefix}")
    endif()
    hkp_source_digest(_digest "${_engine_dir}" ${_manifest} ${_inputs})
    set(${_out_revision} "${_prefix}${_digest}" PARENT_SCOPE)
    if(_arg_CONFIGURE_DEPENDS)
        # The calling directory's property: a function runs in its caller's directory.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_inputs})
    endif()
endfunction()

# The form every ASM SDPA revision, and so every `trained_against.selector_revision` of a
# shipped model, starts with.
function(hkp_asm_sdpa_revision_prefix _out_prefix)
    set(${_out_prefix} "hip-kernel-provider/asm-sdpa/" PARENT_SCOPE)
endfunction()

# hkp_asm_sdpa_inputs(<out-list> <engine-dir> [CONFIGURE_DEPENDS])
# Every file the revision hashes, as absolute paths: hkp_asm_sdpa_sources() under
# <engine-dir>, then hkp_asm_sdpa_kernel_files().
function(hkp_asm_sdpa_inputs _out_inputs _engine_dir)
    hkp_asm_sdpa_sources(_sources)
    set(_inputs "")
    foreach(_source IN LISTS _sources)
        if(NOT EXISTS "${_engine_dir}/${_source}")
            message(FATAL_ERROR
                "asm_sdpa selector revision: input ${_source} does not exist under "
                "${_engine_dir}. A moved or renamed source must move here too, or the "
                "revision silently stops naming it.")
        endif()
        list(APPEND _inputs "${_engine_dir}/${_source}")
    endforeach()
    hkp_asm_sdpa_kernel_files(_kernels "${_engine_dir}" ${ARGN})
    list(APPEND _inputs ${_kernels})
    set(${_out_inputs} "${_inputs}" PARENT_SCOPE)
endfunction()

# hkp_asm_sdpa_kernel_files(<out-list> <engine-dir> [CONFIGURE_DEPENDS])
# Forward and backward kernels and CSVs for every arch directory present (same scope as
# codegen.py), not just configured arches, as absolute paths. Globbed over the whole tree
# so a new arch or kernel directory is caught by the CONFIGURE_DEPENDS glob.
function(hkp_asm_sdpa_kernel_files _out_files _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()
    file(GLOB_RECURSE _kernel_files ${_track}
        "${_engine_dir}/asm/asm_kernels/*.csv"
        "${_engine_dir}/asm/asm_kernels/*.co")
    set(_kernels "")
    foreach(_file IN LISTS _kernel_files)
        file(RELATIVE_PATH _name "${_engine_dir}" "${_file}")
        if(_name MATCHES "^asm/asm_kernels/[^/]+/fmha_v3_(fwd|bwd)/")
            list(APPEND _kernels "${_file}")
        endif()
    endforeach()
    set(${_out_files} "${_kernels}" PARENT_SCOPE)
endfunction()

# hkp_asm_sdpa_sources(<out-list>)
# The sources that change which kernel runs or what it is given, in either direction, as
# paths relative to the engine directory. The revision hashes them, and the fixtures probe
# every one of them, so a source added here is covered by the test with no second list.
function(hkp_asm_sdpa_sources _out_sources)
    set(${_out_sources}
        # Turns the CSVs into the config tables the builders search.
        asm/asm_kernels/codegen.py
        # Reads each kernel's max_flat_workgroup_size out of its code object for codegen.py;
        # the forward builder declines a config whose launch block exceeds it.
        asm/asm_kernels/co_metadata.py
        # Which build of a gfx942 forward kernel runs: the MI300 or the MI308 one
        # (getFwdKernelTocKey).
        asm/AsmKernelPath.hpp
        # The backward dq_acc layout and the 32-bit offset checks that choose between pssk
        # = 0 and pssk = 1 kernels and gate backward applicability; the workspace size.
        plans/SdpaKernelUtils.hpp
        # Applicability and the (arch, dtype, head dims, mask, mode, bf16 rounding) ->
        # kernel lookup.
        plans/SdpaFwdPlanBuilder.cpp
        plans/SdpaFwdPlanBuilder.hpp
        plans/SdpaBwdPlanBuilder.cpp
        plans/SdpaBwdPlanBuilder.hpp
        # Mask classification, a config key of both directions.
        plans/SdpaPlanUtils.hpp
        # Launch geometry and the problem parameters it is derived from.
        plans/SdpaFwdPlan.cpp
        plans/SdpaFwdPlan.hpp
        plans/SdpaFwdParams.hpp
        plans/SdpaFwdLaunchParams.hpp
        plans/SdpaBwdPlan.cpp
        plans/SdpaBwdPlan.hpp
        plans/SdpaBwdParams.hpp
        plans/SdpaBwdLaunchParams.hpp
        # The kernel arguments, and the layout the kernels read them through.
        plans/SdpaFwdArgsBuilder.hpp
        plans/SdpaBwdArgsBuilder.hpp
        asm/SdpaFwdKernelArgs.hpp
        asm/SdpaBwdKernelArgs.hpp
        asm/SgprPadding.hpp
        PARENT_SCOPE)
endfunction()

# hkp_asm_sdpa_not_inputs(<out-list>)
# The files of the engine tree that the revision deliberately leaves out, relative to the
# engine directory. descriptors/ and the kernel directories (asm/asm_kernels/<arch>/
# fmha_v3_fwd and fmha_v3_bwd, whose CSVs and code objects are inputs and whose SOURCE.md
# are provenance) are outside this scope. Every other file is in this list or in
# hkp_asm_sdpa_sources(): TestAsmSdpaSelectorRevision fails on one that is in neither, so a
# new file is a visible decision, and probes each of these as a change that must not move
# the revision. Promoting one means moving it to hkp_asm_sdpa_sources() and retraining.
function(hkp_asm_sdpa_not_inputs _out_files)
    set(${_out_files}
        # They load the kernel the inputs above chose.
        asm/AsmKpackArchive.hpp
        plans/SdpaModuleCache.hpp
        # They find the plugin directory the kernel archives are read from.
        asm/PluginModuleDir.cpp
        asm/PluginModuleDir.hpp
        # It packs the same .co bytes the revision hashes.
        asm/asm_kernels/pack.py
        # Provenance and docs cannot move a throughput number: the AITER pin of the kernels
        # that ship, and the notes on them.
        asm/asm_kernels/aiter_pins.json
        asm/asm_kernels/README.md
        # Build wiring and this recipe: the inputs and the files they are read from are
        # named here, not decided by an edit of these.
        CMakeLists.txt
        AsmSdpaSelectorRevision.cmake
        # AsmSdpaEngine.cpp and AsmSdpaEngine.hpp hold the engine details and behavior
        # notes, the L1 binding, the prediction plumbing (which asks the builders whether
        # they serve a graph), the workspace-limit decline, and the dispatch to the first
        # applicable plan builder. Forward and backward builders accept disjoint graphs, and
        # their registration order lives in core/Container.cpp, so nothing there picks a
        # kernel or how it is launched. Applicability itself is the builders', which are
        # inputs: widening it must still expire models.
        AsmSdpaEngine.cpp
        AsmSdpaEngine.hpp
        PARENT_SCOPE)
endfunction()

# hkp_asm_sdpa_unclassified_files(<out-list> <engine-dir>)
# The files of the engine tree (outside descriptors/ and the kernel directories) that are
# neither in hkp_asm_sdpa_sources() nor in hkp_asm_sdpa_not_inputs(), relative to
# <engine-dir>: a new file that nobody decided about.
function(hkp_asm_sdpa_unclassified_files _out_files _engine_dir)
    hkp_asm_sdpa_sources(_sources)
    hkp_asm_sdpa_not_inputs(_not_inputs)
    file(GLOB_RECURSE _files RELATIVE "${_engine_dir}" "${_engine_dir}/*")
    set(_unclassified "")
    foreach(_rel IN LISTS _files)
        if(_rel MATCHES "^descriptors/" OR _rel MATCHES "^asm/asm_kernels/[^/]+/fmha_v3_(fwd|bwd)/")
            continue()
        endif()
        if(NOT _rel IN_LIST _sources AND NOT _rel IN_LIST _not_inputs)
            list(APPEND _unclassified "${_rel}")
        endif()
    endforeach()
    set(${_out_files} "${_unclassified}" PARENT_SCOPE)
endfunction()
