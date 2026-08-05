# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke-serve`` command line.

Invoke as a module, which is how every other entry point in this tree resolves
its imports (see ``rocke/BUILDING.md`` -- the library packages are
editable-installed, so nothing here touches ``sys.path``)::

    python -m serve probe --arch gfx950
    python -m serve plan  request.json [result.json]
    python -m serve run   request.json result.json

``run`` is the contract the external orchestrator calls. It always writes a
result file, including on failure: the caller reads that file to find out what
happened, so exiting without one would turn a diagnosable rejection into a
silent non-answer.

Exit codes are for the shell, not for the caller's decision logic -- 0 served,
2 declined, 1 malformed. The result file is the actual answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .protocol import ProtocolError, ServeRequest, make_result

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DECLINED = 2


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if path is None:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _aggregate_speedup(measurements: list[dict[str, Any]]) -> float | None:
    """Total time saved across the cohort, not the average of its ratios.

    A mean of per-shape speedups would let a rare shape with a large ratio
    outvote the shape the workload actually spends its time in. Summing weighted
    times and dividing once gives the number the caller is really asking for:
    how much less time the workload would spend in this kernel.
    """
    base_total = 0.0
    rocke_total = 0.0
    for m in measurements:
        baseline = m.get("baseline") or {}
        rocke = m.get("rocke") or {}
        if not (baseline.get("ran") and rocke.get("ran")):
            continue
        weight = float(m.get("call_count") or 1) or 1.0
        base_total += weight * float(baseline["latency_ms"])
        rocke_total += weight * float(rocke["latency_ms"])
    if base_total <= 0 or rocke_total <= 0:
        return None
    return base_total / rocke_total


def _aggregate_correctness(measurements: list[dict[str, Any]]) -> bool | None:
    verdicts = [
        bool(m["verify"]["passed"])
        for m in measurements
        if (m.get("verify") or {}).get("ran")
    ]
    return all(verdicts) if verdicts else None


def _report(
    request: ServeRequest,
    plans: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    reasons: list[str],
) -> str:
    served = [p for p in plans if p.get("servable")]
    lines = [
        "# rocKE attention kernel generation",
        "",
        f"- arch: `{request.arch}`",
        f"- shapes received: {len(plans)}",
        f"- shapes rocKE serves: {len(served)}",
    ]
    if request.advisory:
        lines.append(
            "- **advisory**: shapes were synthesized from model configuration, "
            "not observed in a running workload"
        )
    lines.append("")
    if served:
        lines.append("## Planned kernels")
        lines.append("")
        for plan in served:
            lines.append(
                f"- `{plan['signature']}` -> **{plan['candidate']}** "
                f"({plan.get('path') or plan.get('algorithm')} path), "
                f"kernel `{plan.get('kernel_name')}`"
            )
        lines.append("")
    declined = [p for p in plans if not p.get("servable")]
    if declined:
        lines.append("## Declined shapes")
        lines.append("")
        for plan in declined:
            lines.append(f"- `{plan['signature']}`: {plan.get('reason', '')}")
        lines.append("")
    if measurements:
        lines.append("## Measured")
        lines.append("")
        for m in measurements:
            if not m.get("ran"):
                lines.append(
                    f"- `{m.get('signature')}`: not measured -- {m.get('reason')}"
                )
                continue
            verify = m.get("verify") or {}
            verdict = (
                ("pass" if verify.get("passed") else "FAIL")
                if verify.get("ran")
                else f"not verified ({verify.get('reason', 'skipped')})"
            )
            speedup = m.get("speedup")
            speed = f"{speedup:.4f}x vs Triton baseline" if speedup else "no baseline"
            lines.append(f"- `{m.get('signature')}`: correctness {verdict}; {speed}")
        lines.append("")
    for reason in reasons:
        lines.append(f"> {reason}")
    return "\n".join(lines)


def _load_request(path: Path) -> ServeRequest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProtocolError(f"cannot read request {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"request {path} is not valid JSON: {exc}") from exc
    return ServeRequest.from_dict(raw)


def _coverage(registry: Any) -> dict[str, Any]:
    """Describe the registry, with or without the declarative capability model.

    ``coverage()`` arrived alongside capability-based gating. Where it is
    absent the same manifest is derived from the candidate list, which carries
    the identity fields on every dispatcher revision.
    """
    native = getattr(registry, "coverage", None)
    if callable(native):
        return native()
    return {
        "family": registry.family,
        "candidates": [
            {
                "name": c.name,
                "algorithm": c.algorithm,
                "spec_id": c.spec_id,
                "abi_version": c.abi_version,
                "priority": c.priority,
                "capability": None,
            }
            for c in registry.candidates()
        ],
    }


def _candidates_for_arch(registry: Any, arch: str) -> tuple[list[str] | None, str]:
    """Name the candidates claiming ``arch``, or say why that is unanswerable.

    Request-free arch coverage is a property of declared capabilities. Without
    them the only gate is a support predicate that needs a full request, so the
    honest answer is no answer rather than one synthesized from a made-up shape.
    """
    native = getattr(registry, "for_arch", None)
    if callable(native):
        return [c.name for c in native(arch)], ""
    return None, (
        "this dispatcher declares no candidate capabilities, so arch coverage "
        "cannot be answered without a request"
    )


def cmd_probe(args: argparse.Namespace) -> int:
    """Report what rocKE can serve here, without needing a request or a device."""
    from dispatch.attention import ATTENTION_REGISTRY

    from .runner import torch_gpu_available

    gpu_ok, gpu_reason = torch_gpu_available()
    arch = str(args.arch or "").strip().lower()
    payload: dict[str, Any] = {
        "schema": "hyperloom.rocke.serve_probe/v1",
        "family": "attention_unified",
        "measured_lanes_available": gpu_ok,
        "measured_lanes_reason": gpu_reason,
        "coverage": _coverage(ATTENTION_REGISTRY),
    }
    if arch:
        names, reason = _candidates_for_arch(ATTENTION_REGISTRY, arch)
        payload["arch"] = arch
        payload["candidates_for_arch"] = names
        payload["candidates_for_arch_reason"] = reason
    _write(Path(args.output) if args.output else None, payload)
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    from .planner import plan_all

    try:
        request = _load_request(Path(args.request))
    except ProtocolError as exc:
        _write(
            Path(args.output) if args.output else None,
            make_result(status="error", reasons=[str(exc)]),
        )
        return EXIT_ERROR

    plans = plan_all(request.entries, arch=request.arch)
    served = [p for p in plans if p.get("servable")]
    result = make_result(
        status="ok" if served else "declined",
        plans=plans,
        report=_report(request, plans, [], []),
        reasons=[] if served else ["no shape in this request is servable by rocKE"],
    )
    _write(Path(args.output) if args.output else None, result)
    return EXIT_OK if served else EXIT_DECLINED


def cmd_run(args: argparse.Namespace) -> int:
    from .planner import plan_all

    out_path = Path(args.output)
    try:
        request = _load_request(Path(args.request))
    except ProtocolError as exc:
        _write(out_path, make_result(status="error", reasons=[str(exc)]))
        return EXIT_ERROR

    plans = plan_all(request.entries, arch=request.arch)
    served = [p for p in plans if p.get("servable")]
    artifact_dir = request.output_dir or str(out_path.parent)
    if not served:
        result = make_result(
            status="declined",
            plans=plans,
            artifact_path=artifact_dir,
            report=_report(request, plans, [], []),
            reasons=["no shape in this request is servable by rocKE"],
        )
        _write(out_path, result)
        return EXIT_DECLINED

    reasons: list[str] = []
    measurements: list[dict[str, Any]] = []
    if args.plan_only:
        reasons.append("measured lanes disabled by --plan-only")
    else:
        from .runner import measure_plan, torch_gpu_available

        gpu_ok, gpu_reason = torch_gpu_available()
        if not gpu_ok:
            reasons.append(f"measured lanes skipped: {gpu_reason}")
        else:
            for plan in served[: args.max_shapes]:
                measurement = measure_plan(
                    plan,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    seed=args.seed,
                    do_verify=not args.no_verify,
                    do_baseline=not args.no_baseline,
                )
                measurement["call_count"] = plan.get("call_count") or 0
                measurements.append(measurement)

    if request.advisory:
        reasons.append(
            "shapes are advisory (synthesized from model configuration, not observed)"
        )

    speedup = _aggregate_speedup(measurements)
    correctness = _aggregate_correctness(measurements)
    # "planned" is not a lesser "ok": it says rocKE serves these shapes but
    # produced no evidence about them. Collapsing the two would let a caller
    # read an absent measurement as a completed one that found nothing.
    measured = speedup is not None or correctness is not None
    result = make_result(
        status="ok" if measured else "planned",
        plans=plans,
        measurements=measurements,
        micro_speedup=speedup,
        correctness_passed=correctness,
        artifact_path=artifact_dir,
        report=_report(request, plans, measurements, reasons),
        reasons=reasons,
    )
    _write(out_path, result)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocke-serve", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="report servable coverage; needs no request")
    probe.add_argument("--arch", default="", help="restrict coverage to one gfx target")
    probe.add_argument("--output", default="", help="write JSON here instead of stdout")
    probe.set_defaults(func=cmd_probe)

    plan = sub.add_parser("plan", help="dispatch only; no device required")
    plan.add_argument("request")
    plan.add_argument("output", nargs="?", default="")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="plan, then verify and measure where possible")
    run.add_argument("request")
    run.add_argument("output")
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-shapes", type=int, default=8)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument("--no-verify", action="store_true")
    run.add_argument("--no-baseline", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
