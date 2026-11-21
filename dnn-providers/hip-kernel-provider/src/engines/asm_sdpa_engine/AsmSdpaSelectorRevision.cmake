# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HkpSourceDigest.cmake")

# ============================================================================
# hkp_asm_sdpa_revision(<out-revision> <engine-dir> [CONFIGURE_DEPENDS] [MANIFEST <file>])
# ============================================================================
# The ASM SDPA selector revision: 16 hex digits over everything that decides which forward
# or backward kernel runs and how it is launched. The L1 models estimate both directions,
# record this revision, and the loader refuses a mismatch, so it must hash exactly those
# inputs: a missing input lets a stale model silently pick the wrong engine; an extra one
# expires models for unrelated changes.
#
# CONFIGURE_DEPENDS makes every input a configure dependency so an edit or a new kernel
# reconfigures. Test fixture trees omit it: they are rewritten on each configure.
# MANIFEST writes the selector dump of these inputs (HkpSourceDigest.cmake).
function(hkp_asm_sdpa_revision _out_revision _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "MANIFEST" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()

    # Forward and backward kernels and CSVs for every arch directory present (same scope as
    # codegen.py), not just configured arches. Globbed over the whole tree so a new arch or
    # kernel directory is caught by the CONFIGURE_DEPENDS glob.
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

    # Sources that change which kernel runs or what it is given, in either direction.
    set(_sources
        # Turns the CSVs into the config tables the builders search.
        asm/asm_kernels/codegen.py
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
        asm/SgprPadding.hpp)
    # Not inputs:
    #  - AsmKernelPath.hpp, AsmKpackArchive.hpp, SdpaModuleCache.hpp, SdpaKernelUtils.hpp
    #    (load and launch plumbing), pack.py (it packs the same .co bytes hashed above), and
    #    CMakeLists.txt.
    #  - AsmSdpaEngine.cpp and AsmSdpaEngine.hpp. They hold the engine details and behavior
    #    notes, the L1 binding, the prediction plumbing (which asks the builders whether
    #    they serve a graph), the workspace-limit decline, and the dispatch to the first
    #    applicable plan builder. Forward and backward builders accept disjoint graphs, and
    #    their registration order lives in core/Container.cpp, so nothing there picks a
    #    kernel or how it is launched. Applicability itself is the builders', which are
    #    inputs: widening it must still expire models.
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
    list(APPEND _inputs ${_kernels})

    # Keyed by relative path: the same kernel name exists under both MI300/ and MI308/.
    set(_manifest "")
    if(_arg_MANIFEST)
        # The engine's registered name and the prefix of AsmSdpaEngine.cpp's
        # SELECTOR_REVISION.
        set(_manifest MANIFEST "${_arg_MANIFEST}" ENGINE ASM_SDPA_ENGINE
                      PREFIX "hip-kernel-provider/asm-sdpa/")
    endif()
    hkp_source_digest(_revision "${_engine_dir}" ${_manifest} ${_inputs})
    set(${_out_revision} "${_revision}" PARENT_SCOPE)
    if(_arg_CONFIGURE_DEPENDS)
        # The calling directory's property: a function runs in its caller's directory.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_inputs})
    endif()
endfunction()
