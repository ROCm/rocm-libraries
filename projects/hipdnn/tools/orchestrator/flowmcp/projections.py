# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Derivations: turning a run manifest and a flow declaration into results.

`schema.py` says what a result looks like, `resources.py` does the I/O and the
safety checks, and this module derives one from the other. Every function here
takes data and returns data.

**Nothing in this module may contain the name of a flow, a step, a loop group,
an output key or an artifact file.** The three derivations that make that
possible:

* :func:`step_kind` classifies a step from its own declaration -- a prompt in
  and a declared result file out is an agent invocation, everything else is a
  tool run judged by its exit code. It never compares a tool name, so a flow
  driving a different agent CLI, or two of them, classifies correctly.
* :func:`artifacts_for` discovers files rather than naming them: outputs the
  flow itself declared as paths, the evidence under each step's own directory,
  and a walk of the run directory.
* :func:`progress_for` derives its denominator from the flow's own node list
  and omits it rather than inventing one.

Flow objects are used structurally rather than imported, so this module stays
stdlib-only and the hermetic suite can exercise it against a synthetic flow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import resources, schema

#: Step statuses that mean the step is still in flight. Everything else the
#: engine records -- including a skipped step and a timed-out one -- is a
#: settled outcome and counts towards progress.
_IN_FLIGHT = ("pending", "running")


def _is_group(node: Any) -> bool:
    """Whether a flow node is a loop group rather than a bare step."""
    return hasattr(node, "steps")


def _nodes(flow: Any) -> Sequence[Any]:
    return getattr(flow, "nodes", ()) or ()


def step_kind(step: Any) -> str:
    """`agent` when the step declares the prompt/result contract, else `tool`.

    That contract *is* what an agent invocation is in this system: a prompt goes
    in on stdin or from a prompt file, and a structured result file is expected
    back. A step invoked with argv and judged by its exit code is a tool run.
    The tool's name is never consulted, deliberately: a flow whose agent is
    called something else, and a flow with a step whose tool is called something
    agent-like, both classify correctly.
    """
    has_prompt = getattr(step, "stdin", None) is not None or bool(
        getattr(step, "prompt_file", None)
    )
    declares_result = bool(getattr(step, "result_file", None))
    return "agent" if has_prompt and declares_result else "tool"


def walk_steps(flow: Any) -> Iterator[tuple[Any, str | None]]:
    """Every declared step in flow order, paired with its group or `None`."""
    for node in _nodes(flow):
        if _is_group(node):
            for step in node.steps:
                yield step, node.id
        else:
            yield node, None


def loop_groups(flow: Any) -> list[Any]:
    return [node for node in _nodes(flow) if _is_group(node)]


def output_types(step: Any) -> dict[str, str]:
    """Declared type per output name.

    Only what the flow declared. The engine also records implicit values for
    every step; they carry no declared type and so are not reported as if they
    had one -- which is what keeps "declared as a path" meaningful.
    """
    return {spec.name: spec.type for spec in getattr(step, "outputs", ()) or ()}


def declared_budget(flow: Any) -> int | None:
    """The largest iteration budget the flow declares, or `None` for no loop.

    Across however many groups the flow has: this is the bound a requested
    budget is clamped to, and a flow that declares no loop has no bound to
    clamp against because the number could not mean anything.
    """
    budgets = [group.loop.max_iterations for group in loop_groups(flow)]
    return max(budgets) if budgets else None


def group_budget(group: Any, max_iterations: int | None) -> int:
    """The budget a group will actually run under."""
    if max_iterations is not None:
        return max_iterations
    return int(group.loop.max_iterations)


# -- flow declarations ------------------------------------------------------


def input_specs_for(flow: Any) -> list[dict[str, Any]]:
    return [
        schema.input_spec(
            name=name,
            description=spec.description or None,
            type=spec.type,
            required=bool(spec.required),
            default=spec.default,
            exists=bool(spec.exists),
        )
        for name, spec in (getattr(flow, "inputs", {}) or {}).items()
    ]


def step_specs_for(flow: Any) -> list[dict[str, Any]]:
    return [
        schema.step_spec(
            id=step.id,
            group=group,
            tool=step.tool,
            kind=step_kind(step),
            timeout=step.timeout,
            outputs=[
                schema.output_spec(name=spec.name, type=spec.type)
                for spec in getattr(step, "outputs", ()) or ()
            ],
        )
        for step, group in walk_steps(flow)
    ]


def loop_specs_for(flow: Any) -> list[dict[str, Any]]:
    return [
        schema.loop_spec(
            id=group.id,
            max_iterations=group.loop.max_iterations,
            until=group.loop.until.text,
            on_exhausted=group.loop.on_exhausted,
            on_step_failure=group.loop.on_step_failure,
            step_ids=[step.id for step in group.steps],
        )
        for group in loop_groups(flow)
    ]


def flow_summary_for(flow: Any) -> dict[str, Any]:
    """One flow described entirely by its own declaration."""
    steps = list(walk_steps(flow))
    return schema.flow_summary(
        name=flow.name,
        description=flow.description or None,
        path=str(flow.path),
        step_count=len(steps),
        tools=sorted({step.tool for step, _ in steps}),
        steps=step_specs_for(flow),
        loops=loop_specs_for(flow),
        inputs=input_specs_for(flow),
        load_error=None,
    )


def unloadable_flow_summary(path: Path | str, error: str) -> dict[str, Any]:
    """A flow that failed to load still appears, with the reason attached.

    Dropping it would present a shorter list as if it were the whole one, and
    the operator who has to fix the YAML is the one reading this.
    """
    path = Path(path)
    return schema.flow_summary(
        name=path.stem,
        description=None,
        path=str(path),
        step_count=0,
        tools=[],
        steps=[],
        loops=[],
        inputs=[],
        load_error=error,
    )


# -- run projections --------------------------------------------------------


def _step_index(flow: Any) -> dict[tuple[str | None, str], Any]:
    return {(group, step.id): step for step, group in walk_steps(flow)}


def _relative_dir(raw: Any, run_dir: Path) -> str | None:
    """A step's recorded directory, relative to the run it belongs to.

    The engine records an absolute path; a consumer addresses artifacts by
    `run://` and needs the relative one. A directory outside the run is
    reported as the engine wrote it rather than silently dropped.
    """
    if not raw:
        return None
    candidate = Path(str(raw))
    try:
        resolved = candidate.resolve()
    except OSError:
        return str(raw)
    if resolved.is_relative_to(run_dir):
        return resolved.relative_to(run_dir).as_posix()
    return str(raw)


def run_steps_for(
    run_json: Mapping[str, Any], flow: Any, run_dir: Path | str
) -> list[dict[str, Any]]:
    """Every step record, enriched with what the flow declared about it."""
    root = Path(run_dir).resolve()
    declared = _step_index(flow) if flow is not None else {}
    projected: list[dict[str, Any]] = []
    for record in run_json.get("steps") or []:
        step = declared.get((record.get("group"), record.get("id")))
        projected.append(
            schema.run_step(
                id=record.get("id", ""),
                group=record.get("group"),
                iteration=record.get("iteration"),
                tool=record.get("tool"),
                kind=step_kind(step) if step is not None else None,
                status=record.get("status", "unknown"),
                exit_code=record.get("exit_code"),
                timed_out=bool(record.get("timed_out")),
                duration_s=float(record.get("duration_s") or 0.0),
                dir=_relative_dir(record.get("dir"), root),
                outputs=record.get("outputs") or {},
                output_types=output_types(step) if step is not None else {},
                error=record.get("error"),
            )
        )
    return projected


def run_loops_for(run_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every loop group's recorded state: zero, one, or several.

    The engine records a loop when the group finishes, and it renders the exit
    condition as written. The measured rendering exists only in the engine's
    supplementary log, so `untilMeasured` carries the condition verbatim when
    the manifest has no measurement to show -- never a re-worded one.
    """
    projected: list[dict[str, Any]] = []
    for record in run_json.get("loops") or []:
        until = str(record.get("until", ""))
        projected.append(
            schema.run_loop(
                id=record.get("id", ""),
                iterations=int(record.get("iterations") or 0),
                budget=int(record.get("budget") or 0),
                satisfied=bool(record.get("satisfied")),
                until=until,
                until_measured=str(record.get("until_measured") or until),
                budget_source=str(record.get("budget_source") or ""),
            )
        )
    return projected


def _positions(
    flow: Any, max_iterations: int | None
) -> list[tuple[str, str | None, int | None]]:
    """Every (step, group, iteration) the flow could execute, in order."""
    plan: list[tuple[str, str | None, int | None]] = []
    for node in _nodes(flow):
        if _is_group(node):
            for iteration in range(group_budget(node, max_iterations)):
                for step in node.steps:
                    plan.append((step.id, node.id, iteration))
        else:
            plan.append((node.id, None, None))
    return plan


def current_step_for(
    run_json: Mapping[str, Any], flow: Any, *, max_iterations: int | None = None
) -> dict[str, Any] | None:
    """The position the run is working on, derived from the flow's own order.

    A cursor rather than a lookup: recorded steps are matched against the flow's
    execution order, skipping positions the run passed over -- an abandoned
    iteration, a step whose `when` was false -- and the next unmatched position
    is where the run is. A finished loop group is stepped past, because its
    record says it will not iterate again.
    """
    if flow is None or run_json.get("status") != "running":
        return None
    plan = _positions(flow, max_iterations)
    finished = {record.get("id") for record in run_json.get("loops") or []}
    cursor = 0
    for record in run_json.get("steps") or []:
        key = (record.get("id"), record.get("group"), record.get("iteration"))
        while cursor < len(plan) and plan[cursor] != key:
            cursor += 1
        if cursor < len(plan):
            cursor += 1
    while cursor < len(plan) and plan[cursor][1] in finished:
        cursor += 1
    if cursor >= len(plan):
        return None
    step_id, group, iteration = plan[cursor]
    return schema.current_step(id=step_id, group=group, iteration=iteration)


def progress_for(
    run_json: Mapping[str, Any], flow: Any, *, max_iterations: int | None = None
) -> tuple[int, int | None]:
    """`(settled step records, upper bound)` -- and the bound may be unknowable.

    The numerator needs nothing but the run. The denominator sums each loop
    group's step count times its budget, plus one for every step outside a
    group, which is the only form that survives a flow with several groups of
    different sizes. It is an upper bound that moves: a loop satisfied early
    runs fewer iterations, and a skipped step never runs at all.

    When the flow cannot be read -- a run found by directory scan whose YAML has
    since moved -- `total` is `None`. An absent denominator is honest; an
    invented one is not.
    """
    progress = sum(
        1
        for record in run_json.get("steps") or []
        if record.get("status") not in _IN_FLIGHT
    )
    if flow is None:
        return progress, None
    total = 0
    for node in _nodes(flow):
        if _is_group(node):
            total += len(node.steps) * group_budget(node, max_iterations)
        else:
            total += 1
    return progress, total


# -- artifact discovery -----------------------------------------------------


def _artifact_paths(
    run_json: Mapping[str, Any], flow: Any, run_dir: Path
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Candidate artifacts, most specific origin first.

    Three sources, none of them a file name:

    1. Outputs the flow itself declared as paths. `type: path` is the flow
       author's own statement that the value is a file, and it is the only
       artifact signal anything needs -- a string output whose value happens to
       look like a path is not one.
    2. The evidence under each step's own recorded directory, enumerated.
    3. A walk of the run directory, which is how everything else turns up:
       the engine's own per-run files, and whatever the flow wrote.
    """
    declared = _step_index(flow) if flow is not None else {}
    for record in run_json.get("steps") or []:
        step = declared.get((record.get("group"), record.get("id")))
        if step is None:
            continue
        types = output_types(step)
        for name, value in (record.get("outputs") or {}).items():
            if types.get(name) != "path" or not isinstance(value, str) or not value:
                continue
            candidate = Path(value)
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            # A `run://` URI addresses one run directory. An output written
            # elsewhere is real but unaddressable here, and inventing a URI for
            # it would break containment.
            if not resolved.is_relative_to(run_dir) or not resolved.is_file():
                continue
            yield resolved, {
                "source": "output",
                "step_id": record.get("id"),
                "iteration": record.get("iteration"),
                "output_name": name,
                "label": name,
            }

    for record in run_json.get("steps") or []:
        raw = record.get("dir")
        if not raw:
            continue
        step_dir = Path(str(raw))
        try:
            entries = sorted(step_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            resolved = entry.resolve()
            if not resolved.is_relative_to(run_dir):
                continue
            yield resolved, {
                "source": "step",
                "step_id": record.get("id"),
                "iteration": record.get("iteration"),
                "output_name": None,
                "label": entry.name,
            }

    for relative in resources.walk_run_dir(run_dir):
        yield (run_dir / relative), {
            "source": "run",
            "step_id": None,
            "iteration": None,
            "output_name": None,
            "label": Path(relative).name,
        }


def artifacts_for(
    run_json: Mapping[str, Any], flow: Any, *, run_id: str, run_dir: Path | str
) -> list[dict[str, Any]]:
    """Everything a run produced, discovered rather than enumerated."""
    root = Path(run_dir).resolve()
    feedback: Path | None = None
    recorded = run_json.get("feedback_path")
    if recorded:
        try:
            feedback = Path(str(recorded)).resolve()
        except OSError:
            feedback = None

    found: dict[Path, dict[str, Any]] = {}
    for path, origin in _artifact_paths(run_json, flow, root):
        # First origin wins: an output the flow declared says more about a file
        # than the walk that would also have found it.
        if path in found:
            continue
        relative = path.relative_to(root).as_posix()
        found[path] = schema.artifact(
            source=origin["source"],
            step_id=origin["step_id"],
            iteration=origin["iteration"],
            output_name=origin["output_name"],
            label=origin["label"],
            uri=schema.format_run_uri(run_id, relative),
            mime_type=resources.mime_for(path),
            # The engine's own cross-iteration channel, identified by the path
            # the run recorded rather than by matching a file name -- so a flow
            # that declares an output of its own with a similar name is
            # unaffected, and a run whose channel is named differently still
            # gets the hint.
            role="feedback" if feedback is not None and path == feedback else None,
        )
    return list(found.values())
