# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

# ============================================================================
# hkp_asm_sdpa_fwd_revision(<out-revision> <engine-dir> [CONFIGURE_DEPENDS])
# ============================================================================
# The ASM SDPA forward selector revision: 16 hex digits over everything that decides which
# forward kernel runs and how it is launched. L1 models record it and the loader refuses
# a mismatch, so it must hash exactly those inputs: a missing input lets a stale model
# silently pick the wrong engine; an extra one expires models for unrelated changes.
#
# CONFIGURE_DEPENDS makes every input a configure dependency so an edit or a new kernel
# reconfigures. Test fixture trees omit it: they are rewritten on each configure.
function(hkp_asm_sdpa_fwd_revision _out_revision _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()

    # Forward kernels and CSVs for every arch directory present (same scope as codegen.py),
    # not just configured arches. Globbed over the whole tree so a new arch or forward
    # directory is caught by the CONFIGURE_DEPENDS glob.
    file(GLOB_RECURSE _kernel_files ${_track}
        "${_engine_dir}/asm/asm_kernels/*.csv"
        "${_engine_dir}/asm/asm_kernels/*.co")
    set(_kernels "")
    foreach(_file IN LISTS _kernel_files)
        file(RELATIVE_PATH _name "${_engine_dir}" "${_file}")
        if(_name MATCHES "^asm/asm_kernels/[^/]+/fmha_v3_fwd/")
            list(APPEND _kernels "${_file}")
        endif()
    endforeach()

    # Forward sources that change which kernel runs or what it is given. Backward-only
    # sources are excluded: they cannot move a forward throughput number.
    set(_sources
        # Turns the CSVs into the config table the forward builder searches.
        asm/asm_kernels/codegen.py
        # Engine applicability and the forward/backward dispatch decision.
        AsmSdpaEngine.cpp
        AsmSdpaEngine.hpp
        # Applicability and the (arch, dtype, head dims, mask, mode, bf16 rounding) ->
        # kernel lookup.
        plans/SdpaFwdPlanBuilder.cpp
        plans/SdpaFwdPlanBuilder.hpp
        # Mask classification. Shared with backward, but its MaskType is a forward config
        # key, so it decides which forward kernel a graph gets.
        plans/SdpaPlanUtils.hpp
        # Launch geometry and the problem parameters it is derived from.
        plans/SdpaFwdPlan.cpp
        plans/SdpaFwdPlan.hpp
        plans/SdpaFwdParams.hpp
        plans/SdpaFwdLaunchParams.hpp
        # The kernel arguments, and the layout the kernel reads them through (SgprPadding
        # is shared with backward, but it lays out the forward arguments too).
        plans/SdpaFwdArgsBuilder.hpp
        asm/SdpaFwdKernelArgs.hpp
        asm/SgprPadding.hpp)
    # Not inputs: AsmKernelPath.hpp, SdpaModuleCache.hpp, SdpaKernelUtils.hpp (load and
    # launch plumbing), pack.py (nothing loads .kpack yet), and CMakeLists.txt.
    set(_inputs "")
    foreach(_source IN LISTS _sources)
        if(NOT EXISTS "${_engine_dir}/${_source}")
            message(FATAL_ERROR
                "asm_sdpa forward selector revision: input ${_source} does not exist under "
                "${_engine_dir}. A moved or renamed forward source must move here too, or "
                "the revision silently stops naming it.")
        endif()
        list(APPEND _inputs "${_engine_dir}/${_source}")
    endforeach()
    list(APPEND _inputs ${_kernels})

    # Sorted and keyed by relative path: GLOB order is filesystem order, and the same
    # kernel name exists under both MI300/ and MI308/.
    list(SORT _inputs)
    set(_digest "")
    foreach(_input IN LISTS _inputs)
        file(RELATIVE_PATH _name "${_engine_dir}" "${_input}")
        if(_input MATCHES "\\.co$")
            # A code object is bytes; git never converts it.
            file(SHA256 "${_input}" _one)
        else()
            # Hash text as LF so CRLF and LF checkouts of one commit agree. file(READ)
            # drops CR only on Windows; the REPLACE covers CRLF files on Linux.
            file(READ "${_input}" _text)
            string(REPLACE "\r\n" "\n" _text "${_text}")
            string(SHA256 _one "${_text}")
        endif()
        string(APPEND _digest "${_name}:${_one}\n")
    endforeach()
    string(SHA256 _revision "${_digest}")
    string(SUBSTRING "${_revision}" 0 16 _revision)

    set(${_out_revision} "${_revision}" PARENT_SCOPE)
    if(_arg_CONFIGURE_DEPENDS)
        # The calling directory's property: a function runs in its caller's directory.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_inputs})
    endif()
endfunction()
