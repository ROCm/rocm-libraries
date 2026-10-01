# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

# ============================================================================
# hkp_asm_sdpa_fwd_revision(<out-revision> <engine-dir> [CONFIGURE_DEPENDS])
# ============================================================================
# The ASM SDPA forward selector revision: 16 lowercase hex digits naming everything that
# decides which forward kernel runs and how it is launched. An L1 model for this engine
# predicts the throughput of that dispatch, and the loader refuses a model whose recorded
# `selector_revision` is not the one the provider reports, so this value is the expiry
# rule for every shipped AITER model. It has to name exactly what changes those
# measurements, and nothing else:
#
#   * too wide -- the provider version, or a git hash. `0.2.0 -> 0.2.1` for a fix that
#     cannot touch this engine expires both shipped models, and the only symptom is
#     UNAVAILABLE on every engine-selection query.
#   * too narrow -- a hand-bumped string, or a digest missing an input. It expires
#     nothing when selection changes, which is the silent direction: a stale L1 estimate
#     changes which ENGINE runs.
#
# Its own function, in its own file, because the engine's tests compute it over fixture
# trees to prove the two properties no single build can show: a CRLF and an LF checkout
# agree, and each input below does (or does not) move it.
#
# <engine-dir> is the asm_sdpa_engine source directory. CONFIGURE_DEPENDS is for the build
# that reports the value: the digest is computed at configure time, so it makes every
# hashed file a configure dependency of the calling directory and every glob below a
# CONFIGURE_DEPENDS glob. Without that, an edit -- or a kernel added -- after configure
# leaves the build reporting a revision for sources it no longer compiles. Fixture trees
# omit it: they are rewritten by each configure, and a glob recorded mid-mutation would
# never verify and so would reconfigure every build.
function(hkp_asm_sdpa_fwd_revision _out_revision _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()

    # The forward kernel inventory and the CSVs that describe it. Same scope as codegen.py
    # (`<arch>/fmha_v3_fwd/**/*.csv` for EVERY arch directory present, not the configured
    # arch list), because those CSVs are the config table forward selection searches, and a
    # .co whose bytes change is a different kernel even under an identical CSV row. Other
    # files there (SOURCE.md) are provenance notes and select nothing. Globbed over the
    # whole kernel tree and filtered, so a new arch directory or a new forward directory is
    # caught by the same CONFIGURE_DEPENDS glob as a new kernel (develop's fp8 .co files).
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

    # The forward sources, each because it changes which kernel runs or what it is given.
    # Backward-only sources are absent by design: a backward change cannot move a forward
    # throughput number, and expiring the models for one is the "too wide" failure.
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
    # Not inputs: asm/AsmKernelPath.hpp (which directory, not which kernel),
    # plans/SdpaModuleCache.hpp and plans/SdpaKernelUtils.hpp (module load and launch
    # plumbing shared with backward), pack.py (.kpack archives nothing loads yet), and this
    # engine's CMakeLists.txt (build wiring; the recipe itself lives in this file, whose
    # changes alter the value by construction).
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

    # Sorted, and each entry named by its path under <engine-dir>: GLOB order is
    # filesystem order, and the same kernel name exists under both MI300/ and MI308/, so
    # an unsorted list or a bare file name would make two checkouts of identical content
    # disagree.
    list(SORT _inputs)
    set(_digest "")
    foreach(_input IN LISTS _inputs)
        file(RELATIVE_PATH _name "${_engine_dir}" "${_input}")
        if(_input MATCHES "\\.co$")
            # A code object is bytes; git never converts it.
            file(SHA256 "${_input}" _one)
        else()
            # Text is hashed as git stores it (LF). A Windows checkout (core.autocrlf) has
            # CRLF where Linux has LF; hashing the raw bytes made the same commit report two
            # revisions, so a model trained on one platform was refused on the other.
            # file(READ) already drops the CR on Windows (it reads in text mode) but not on
            # Linux, which can also hold a CRLF file; the REPLACE makes both read LF.
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
