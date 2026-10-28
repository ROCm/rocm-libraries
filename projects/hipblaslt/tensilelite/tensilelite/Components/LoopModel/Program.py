# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build and cache the finalized GIR program for a LoopModel kernel."""

from ...Common.GlobalParameters import globalParameters
from ...LoopModel.checks import validate_loopir
from ...LoopModel.decoder import decode_theta
from ...LoopModel.loopir import render as render_loopir
from ...LoopModel.render import render_geometry
from ...Lowering import gir_text
from ...Lowering.loopir_to_gir import lower_to_gir
from ...Lowering.gir import check_plan
from ...Lowering.gir.analyses import Gl2PrefetchRegions
from ...Lowering.gir.frame_contract import install_frame_contract
from ...Lowering.gir.passes import pipeline as gir_pipeline
from ...Lowering.gir.passes import run_pipeline
from ...Lowering.gir.passes.manual_schedule import (
    ManualSchedulePass,
    schedule_cache_tag,
)

from .Theta import loopModelTheta, namingSolution

def _valuAllocation(writer):
    """Each operand's `ValuAlloc` as vgprAllocationImplClassic sized it.

    Empty before allocation has run, which leaves the dump on theta's footprint.
    """
    return dict(getattr(writer.states, "loopModelValuAlloc", {}) or {})


def loopModelGirProgram(writer, kernel):
    """Build and cache the finalized GIR program for one kernel."""
    key = getattr(writer.states, "kernelName", None) or id(kernel)
    true_sia0 = int(kernel.get("ScheduleIterAlg", 0) or 0) == 0 \
        and int(kernel.get("_StinkyTofuOptLevel", 0) or 0) == 0
    dump_schedule = globalParameters.get("GIRScheduleDumpPath", "") if true_sia0 else ""
    load_schedule = globalParameters.get("GIRScheduleLoadPath", "") if true_sia0 else ""
    schedule_tag = schedule_cache_tag(str(key), dump_schedule, load_schedule)
    verify_ir = bool(globalParameters.get("LoopModelIRVerify", True))
    include_dependencies = verify_ir or bool(
        globalParameters.get("OutputLoopIR", False)
    )
    cache = getattr(writer, "_loopModelGirCache", None)
    if cache is not None and cache[0] == key:
        cached_verified = cache[2] if len(cache) > 2 else True
        cached_dependencies = cache[3] if len(cache) > 3 else True
        cached_schedule = cache[4] if len(cache) > 4 else ""
        if (
            (not verify_ir or cached_verified)
            and (not include_dependencies or cached_dependencies)
            and cached_schedule == schedule_tag
        ):
            return cache[1]

    theta, _S = loopModelTheta(writer, kernel)
    with namingSolution(writer, kernel):
        # The explicit checker below is the single verification point. Avoid
        # repeating its structural/profile checks in decode and lowering.
        decoded = decode_theta(theta, verify=False)
        if verify_ir:
            ir_violations = validate_loopir(
                decoded.program,
                decoded.profile,
            )
            if ir_violations:
                raise ValueError(
                    "UseLoopModel: LoopIR validation failed (%d violation(s)) — "
                    "repair the frontend, do not bypass the checker.\n  %s"
                    % (
                        len(ir_violations),
                        "\n  ".join(ir_violations[:8]),
                    )
                )
        prog = lower_to_gir(
            decoded.program,
            decoded.profile,
            include_dependencies=include_dependencies,
            verify=False,
        )
        prog.meta["valu_alloc"] = {
            name: {
                "per_buffer": int(alloc.perBuffer),
                "total": int(alloc.total),
            }
            for name, alloc in (writer.states.loopModelValuAlloc or {}).items()
            if int(alloc.total) > 0
        }
        # PrefetchGL2 is a target-side GIR placement parameter, not LoopIR.
        prog.params[Gl2PrefetchRegions.PARAM] = kernel["PrefetchGL2"]
        passes = gir_pipeline()
        if dump_schedule or load_schedule:
            manual = ManualSchedulePass(
                key,
                dump_path=dump_schedule,
                load_path=load_schedule,
            )
            insertion = next(
                (index for index, item in enumerate(passes)
                 if item.__class__.__name__ == "CollectPendingMarksPass"),
                len(passes),
            )
            passes.insert(insertion, manual)
        run_pipeline(
            prog,
            passes=passes,
            verify=verify_ir,
        )
        kinds = {}
        output_ir = bool(globalParameters.get("OutputLoopIR", False))
        if output_ir:
            for block in prog.blocks.values():
                for operation in block.body:
                    for _endpoint, _counter, kind, _scope in (
                        getattr(operation, "deps", ()) or ()
                    ):
                        kinds[kind] = kinds.get(kind, 0) + 1
            # Cache before the semantic check so a failing kernel remains renderable when
            # OutputLoopIR is diagnosing that exact failure. The failed entry is explicitly
            # unverified, so no later verified request can reuse it as successful codegen.
            writer._loopModelMainloopCache = (key, decoded.program, kinds)
            writer._loopModelGirCache = (
                key,
                prog,
                False,
                include_dependencies,
                schedule_tag,
            )
        if verify_ir:
            violations = check_plan(prog)
            if violations:
                raise ValueError(
                    "UseLoopModel: the GIR plan fails its own semantic check (%d violation(s)) — "
                    "a decoder/lowering defect to repair, do NOT gate it off.\n  %s"
                    % (len(violations), "\n  ".join(violations[:8])))
    # Verification completed (or was not requested); publish the normal reusable cache state.
    writer._loopModelMainloopCache = (key, decoded.program, kinds)
    writer._loopModelGirCache = (
        key,
        prog,
        verify_ir,
        include_dependencies,
        schedule_tag,
    )
    return prog


def loopModelGirProgramCached(writer):
    """Return the cached program for the current named kernel, if any."""
    cache = getattr(writer, "_loopModelGirCache", None)
    if not cache:
        return None
    # Only the name is verifiable here; ids can be recycled by CPython.
    name = getattr(writer.states, "kernelName", None)
    return cache[1] if (name and cache[0] == name) else None


def loopModelIrTextCached(writer):
    """Render target geometry and the Program that produced the GIR."""
    name = getattr(writer.states, "kernelName", None)
    theta_cache = getattr(writer, "_loopModelThetaCache", None)
    mainloop_cache = getattr(writer, "_loopModelMainloopCache", None)
    if not (
        name
        and theta_cache
        and mainloop_cache
        and theta_cache[0] == name
        and mainloop_cache[0] == name
    ):
        return "# LoopModel IR unavailable: this kernel has no cached target-aware theta/IR\n"
    try:
        theta, _depths = theta_cache[1]
        program = mainloop_cache[1]
        if len(mainloop_cache) > 2:
            kinds = mainloop_cache[2]
        else:
            kinds = {}
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(kinds.items())
        ) or "none"
        return (
            render_geometry(theta, _valuAllocation(writer))
            + f"\n  derived_dependences: {summary}"
            + "\n\n"
            + render_loopir(program)
            + "\n"
        )
    except Exception as exc:
        return f"# LoopModel IR unavailable (cached render error): {exc}\n"


def loopModelGirTextCached(writer):
    """Render the GIR emitted by the cached program."""
    return gir_text(loopModelGirProgramCached(writer))


def attachFrameContract(writer, stModule, kernel):
    """Attach this kernel's frame contract to the StinkyTofu module."""
    install_frame_contract(loopModelGirProgram(writer, kernel), stModule)
