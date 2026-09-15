# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The wire contract: tool schemas, result builders, and the `run://` URI.

Stdlib only. This module holds data and pure functions, so the hermetic test
suite covers the whole contract without the MCP SDK installed and without a
flow, a registry or a filesystem.

**A declared shape and the value built against it cannot drift.** Every result
object is described exactly once, as a JSON Schema whose `properties` are its
field set. `_shaped()` then builds the value *from that schema*: it rejects a
missing key and an unknown key, and it emits the keys in the declared order. A
field added to a shape without a corresponding builder argument fails loudly the
first time the builder runs, rather than producing a result that quietly no
longer validates against the `outputSchema` the server advertised.

**Nothing here names a flow, a step, a loop group, an output key or an artifact
filename**, and nothing may be added that does. Flow vocabulary is data passing
through these builders, never contract: the same code serves a two-step flow
with one loop group and a fifteen-step flow with three. The only file
name that appears is the engine's own run manifest, which is the engine's
layout rather than any flow's -- see `MANIFEST_NAME`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

#: The engine's run manifest, written by `Engine._checkpoint` for every run
#: regardless of flow. Naming it is not a bake: it is part of the run directory
#: layout the engine itself defines, and it is the one resource whose URI a
#: client is expected to know without enumerating anything.
MANIFEST_NAME = "run.json"

#: URI scheme for anything inside a run directory: `run://<runId>/<path>`.
RUN_URI_SCHEME = "run"


class SchemaError(ValueError):
    """A result was built with a field set that does not match its declared shape."""


# -- JSON Schema fragments --------------------------------------------------


def _string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def _integer(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": description, **extra}


def _number(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "number", "description": description, **extra}


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The same schema, with `null` admitted as a value.

    Used wherever the honest answer is "not known yet" or "does not apply" --
    an exit code before the process exits, a group id for a step outside every
    loop. A placeholder would be a lie and a consumer cannot tell it apart from
    a real value.
    """
    declared = dict(schema)
    kind = declared.get("type")
    if kind is None:
        return declared
    kinds = list(kind) if isinstance(kind, list) else [kind]
    if "null" not in kinds:
        kinds.append("null")
    declared["type"] = kinds
    return declared


def _array(items: Mapping[str, Any], description: str) -> dict[str, Any]:
    return {"type": "array", "items": dict(items), "description": description}


def _map(description: str) -> dict[str, Any]:
    """An open object. The keys are the flow's vocabulary and are never enumerated."""
    return {"type": "object", "additionalProperties": True, "description": description}


def _any(description: str) -> dict[str, Any]:
    """Any JSON value -- a declared default, an extracted output."""
    return {"description": description}


def _object(
    properties: Mapping[str, Mapping[str, Any]], description: str
) -> dict[str, Any]:
    """A closed result object: every declared field is present in every value.

    `required` is the full property set because `_shaped()` guarantees it. A
    consumer validating `structuredContent` against the advertised
    `outputSchema` therefore never has to branch on absence, only on `null`.
    """
    return {
        "type": "object",
        "description": description,
        "properties": {key: dict(value) for key, value in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


def _shaped(schema: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Build a result object from its own declared schema.

    This is the anti-drift mechanism: the schema's `properties` are the single
    definition of the field set, and a value that does not match it exactly is a
    programming error raised here rather than a malformed document shipped to a
    client that validates.
    """
    declared = schema["properties"]
    missing = [key for key in declared if key not in values]
    unknown = [key for key in values if key not in declared]
    if missing or unknown:
        raise SchemaError(
            f"result does not match its declared shape: "
            f"missing {missing or 'nothing'}, unknown {unknown or 'nothing'}"
        )
    return {key: values[key] for key in declared}


# -- Shared shapes ----------------------------------------------------------

#: How a step's declaration classifies structurally. `agent` is a step whose
#: flow declaration carries the prompt/result contract -- a prompt in, a
#: declared result file out; everything else is `tool`, invoked with argv and
#: judged by its exit code. Derived from the flow's own structure, never from a
#: list of tool names, so a flow driving a different agent CLI or two of them
#: classifies correctly with no change here.
STEP_KINDS = ("agent", "tool")

#: The reconciled run status (`flow_status`), derived from the run manifest and
#: the supervisor's process record together.
RUN_STATES = ("running", "ok", "failed", "cancelled", "crashed", "unknown")

#: Where a discovered artifact came from: a step output the flow declared as a
#: path, a file under a step's recorded directory, or a file in the run root.
ARTIFACT_SOURCES = ("output", "step", "run")

OUTPUT_SPEC = _object(
    {
        "name": _string("Output key as the flow declares it."),
        "type": _string(
            "Declared type. A value typed as a path is the flow's own statement "
            "that the output is a file; that is the only artifact signal anything "
            "needs."
        ),
    },
    "One output a step declares, with the type the flow gave it.",
)

INPUT_SPEC = _object(
    {
        "name": _string("Input key as the flow declares it."),
        "description": _nullable(_string("Prose from the flow, for the form label.")),
        "type": _string("Declared type; drives the form control and validation."),
        "required": _boolean("Whether a launch must supply a value."),
        "default": _any("Declared default, or null when there is none."),
        "exists": _boolean("Whether a path-typed value is existence-checked."),
    },
    "One input a flow declares.",
)

STEP_SPEC = _object(
    {
        "id": _string("Step id as the flow declares it."),
        "group": _nullable(
            _string("Owning loop group, or null for a step outside every loop.")
        ),
        "tool": _string("Registered tool the step invokes."),
        "kind": _string("Structural classification.", enum=list(STEP_KINDS)),
        "timeout": _nullable(_number("Per-step deadline in seconds, when declared.")),
        "outputs": _array(OUTPUT_SPEC, "Outputs the step declares, in flow order."),
    },
    "One step of a flow, as declared. Every node appears, grouped or not.",
)

LOOP_SPEC = _object(
    {
        "id": _string("Loop group id as the flow declares it."),
        "maxIterations": _integer("Iteration budget the flow declares."),
        "until": _string("Exit condition as written."),
        "onExhausted": _string("What the flow does when the budget runs out."),
        "onStepFailure": _string("What the flow does when a step in the group fails."),
        "stepIds": _array(_string("Step id."), "Steps in the group, in order."),
    },
    "One loop group of a flow, as declared. A flow may have none, one or several.",
)

FLOW_SUMMARY = _object(
    {
        "name": _string("Flow name."),
        "description": _nullable(_string("Prose from the flow.")),
        "path": _string("Absolute path to the flow file."),
        "stepCount": _integer("Number of declared steps."),
        "tools": _array(_string("Tool name."), "Distinct tools the flow invokes."),
        "steps": _array(STEP_SPEC, "Every step, in flow order."),
        "loops": _array(LOOP_SPEC, "Every loop group, in flow order; may be empty."),
        "inputs": _array(INPUT_SPEC, "Every declared input."),
        "loadError": _nullable(
            _string("Why the flow failed to load; other fields are best-effort.")
        ),
    },
    "One flow as declared, described entirely by its own structure.",
)

RUN_LOOP = _object(
    {
        "id": _string("Loop group id."),
        "iterations": _integer("Iterations run so far."),
        "budget": _integer("Iteration budget actually applied."),
        "satisfied": _boolean("Whether the exit condition was met."),
        "until": _string("Exit condition as written."),
        "untilMeasured": _string(
            "The condition as written and as measured, in the engine's own "
            "rendering. Displayed verbatim; never re-worded."
        ),
        "budgetSource": _string("Where the applied budget came from."),
    },
    "One loop group's live state.",
)

RUN_STEP = _object(
    {
        "id": _string("Step id."),
        "group": _nullable(_string("Owning loop group, or null when ungrouped.")),
        "iteration": _nullable(
            _integer("Zero-based iteration, or null when ungrouped.")
        ),
        "tool": _nullable(_string("Tool the step invoked, as the run recorded it.")),
        "kind": _nullable(_string("Structural classification.", enum=list(STEP_KINDS))),
        "status": _string("Status the engine recorded, including a skipped step."),
        "exitCode": _nullable(_integer("Process exit code once the step has run.")),
        "timedOut": _boolean("Whether the step hit its deadline."),
        "durationS": _number("Wall-clock seconds the step took."),
        "dir": _nullable(_string("Step directory, relative to the run directory.")),
        "outputs": _map("Extracted outputs, keyed by the flow's own output names."),
        "outputTypes": _map(
            "Declared type per output name. This is how a consumer knows which "
            "values are files without knowing what any of them mean."
        ),
        "error": _nullable(_string("Failure detail the engine recorded.")),
    },
    "One step record from the run manifest, enriched with what the flow declared.",
)

CURRENT_STEP = _object(
    {
        "id": _string("Step id."),
        "group": _nullable(_string("Owning loop group, or null when ungrouped.")),
        "iteration": _nullable(
            _integer("Zero-based iteration, or null when ungrouped.")
        ),
    },
    "The step the run is executing right now.",
)

ARTIFACT = _object(
    {
        "source": _string("How the entry was discovered.", enum=list(ARTIFACT_SOURCES)),
        "stepId": _nullable(_string("Producing step, when the entry came from one.")),
        "iteration": _nullable(_integer("Producing iteration, when there was one.")),
        "outputName": _nullable(
            _string("Output key, when the entry is a path-typed output.")
        ),
        "label": _string("Display label; the output key or the file name."),
        "uri": _string("Addressable `run://` URI."),
        "mimeType": _string("Mapped from the file extension."),
        "role": _nullable(
            _string(
                "Structural hint. Set by comparing against the run's recorded "
                "feedback path, never by matching a file name."
            )
        ),
    },
    "One discovered artifact. Discovered, never named -- see the discovery sources.",
)


# -- Result builders --------------------------------------------------------


def output_spec(*, name: str, type: str) -> dict[str, Any]:
    return _shaped(OUTPUT_SPEC, {"name": name, "type": type})


def input_spec(
    *,
    name: str,
    description: str | None,
    type: str,
    required: bool,
    default: Any,
    exists: bool,
) -> dict[str, Any]:
    return _shaped(
        INPUT_SPEC,
        {
            "name": name,
            "description": description,
            "type": type,
            "required": required,
            "default": default,
            "exists": exists,
        },
    )


def step_spec(
    *,
    id: str,
    group: str | None,
    tool: str,
    kind: str,
    timeout: float | None,
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _shaped(
        STEP_SPEC,
        {
            "id": id,
            "group": group,
            "tool": tool,
            "kind": kind,
            "timeout": timeout,
            "outputs": list(outputs),
        },
    )


def loop_spec(
    *,
    id: str,
    max_iterations: int,
    until: str,
    on_exhausted: str,
    on_step_failure: str,
    step_ids: Sequence[str],
) -> dict[str, Any]:
    return _shaped(
        LOOP_SPEC,
        {
            "id": id,
            "maxIterations": max_iterations,
            "until": until,
            "onExhausted": on_exhausted,
            "onStepFailure": on_step_failure,
            "stepIds": list(step_ids),
        },
    )


def flow_summary(
    *,
    name: str,
    description: str | None,
    path: str,
    step_count: int,
    tools: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
    loops: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    load_error: str | None = None,
) -> dict[str, Any]:
    return _shaped(
        FLOW_SUMMARY,
        {
            "name": name,
            "description": description,
            "path": path,
            "stepCount": step_count,
            "tools": list(tools),
            "steps": list(steps),
            "loops": list(loops),
            "inputs": list(inputs),
            "loadError": load_error,
        },
    )


def run_loop(
    *,
    id: str,
    iterations: int,
    budget: int,
    satisfied: bool,
    until: str,
    until_measured: str,
    budget_source: str,
) -> dict[str, Any]:
    return _shaped(
        RUN_LOOP,
        {
            "id": id,
            "iterations": iterations,
            "budget": budget,
            "satisfied": satisfied,
            "until": until,
            "untilMeasured": until_measured,
            "budgetSource": budget_source,
        },
    )


def run_step(
    *,
    id: str,
    group: str | None,
    iteration: int | None,
    tool: str | None,
    kind: str | None,
    status: str,
    exit_code: int | None,
    timed_out: bool,
    duration_s: float,
    dir: str | None,
    outputs: Mapping[str, Any],
    output_types: Mapping[str, str],
    error: str | None,
) -> dict[str, Any]:
    return _shaped(
        RUN_STEP,
        {
            "id": id,
            "group": group,
            "iteration": iteration,
            "tool": tool,
            "kind": kind,
            "status": status,
            "exitCode": exit_code,
            "timedOut": timed_out,
            "durationS": duration_s,
            "dir": dir,
            "outputs": dict(outputs),
            "outputTypes": dict(output_types),
            "error": error,
        },
    )


def current_step(
    *, id: str, group: str | None, iteration: int | None
) -> dict[str, Any]:
    return _shaped(CURRENT_STEP, {"id": id, "group": group, "iteration": iteration})


def artifact(
    *,
    source: str,
    label: str,
    uri: str,
    mime_type: str,
    step_id: str | None = None,
    iteration: int | None = None,
    output_name: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    return _shaped(
        ARTIFACT,
        {
            "source": source,
            "stepId": step_id,
            "iteration": iteration,
            "outputName": output_name,
            "label": label,
            "uri": uri,
            "mimeType": mime_type,
            "role": role,
        },
    )


# -- Tool results -----------------------------------------------------------

FLOW_LIST_RESULT = _object(
    {"flows": _array(FLOW_SUMMARY, "Every flow found, in directory order.")},
    "Everything the server is willing to run.",
)

FLOW_INPUTS_RESULT = _object(
    {
        "flow": _string("Resolved flow name."),
        "path": _string("Absolute path to the flow file."),
        "inputs": _array(INPUT_SPEC, "Every input the flow declares."),
    },
    "One flow's declared inputs -- enough to build a form for it.",
)

FLOW_VALIDATE_RESULT = _object(
    {
        "ok": _boolean("Whether the flow is runnable as configured."),
        "flow": _string("Resolved flow name."),
        "stepCount": _integer("Number of declared steps."),
        "inputCount": _integer("Number of declared inputs."),
        "tools": _array(_string("Tool name."), "Distinct tools the flow invokes."),
        "missingTools": _array(
            _string("Tool name."), "Tools the flow invokes that the registry lacks."
        ),
        "resolvedTools": _map("Executable path per tool name, when resolvable."),
        "errors": _array(_string("Message."), "Why the flow is not runnable."),
    },
    "Preflight outcome: reference checks, tool coverage and executable resolution.",
)

FLOW_LAUNCH_RESULT = _object(
    {
        "runId": _string(
            "Run identity, shared by the manifest and the directory leaf."
        ),
        "runDir": _string("Absolute run directory."),
        "runJsonUri": _string("URI of the authoritative run manifest."),
        "flow": _string("Resolved flow name."),
        "status": _string("State at launch.", enum=list(RUN_STATES)),
        "pid": _nullable(_integer("Worker process id.")),
        "startedAt": _string("UTC launch timestamp."),
        "maxIterations": _nullable(
            _integer(
                "Iteration budget actually applied after clamping to what the flow "
                "itself declares; null when the flow declares no loop."
            )
        ),
        "warnings": _array(
            _string("Message."),
            "Non-fatal adjustments, such as a requested budget being clamped.",
        ),
    },
    "Launch acknowledgement. Returns immediately; the run outlives the call.",
)

FLOW_CANCEL_RESULT = _object(
    {
        "runId": _string("Run identity."),
        "cancelled": _boolean("Whether this call killed a live process tree."),
        "state": _string("Process state after the attempt.", enum=list(RUN_STATES)),
        "killedPid": _nullable(_integer("Worker process id that was killed.")),
        "message": _string("What happened, for display."),
    },
    "Cancellation outcome. Cancelling an unknown or finished run is not an error.",
)

FLOW_STATUS_RESULT = _object(
    {
        "runId": _string("Run identity."),
        "runDir": _string("Absolute run directory."),
        "flow": _string("Flow name the run recorded."),
        "flowPath": _string("Flow file the run recorded."),
        "status": _string(
            "Reconciled status: the manifest and the process record together.",
            enum=list(RUN_STATES),
        ),
        "engineStatus": _nullable(_string("Verbatim status from the run manifest.")),
        "processState": _nullable(
            _string("Verbatim state from the supervisor's record; null when none.")
        ),
        "error": _nullable(_string("Failure detail from the run manifest.")),
        "startedAt": _nullable(_string("UTC start timestamp.")),
        "durationS": _number("Wall-clock seconds so far."),
        "exitCode": _nullable(_integer("Worker exit code once it has exited.")),
        "provenance": _map("Hashes and revision the engine recorded."),
        "inputs": _map("Bound input values, keyed by the flow's input names."),
        "vars": _map("Resolved flow vars, keyed by the flow's var names."),
        "loops": _array(RUN_LOOP, "One entry per loop group the run has."),
        "steps": _array(RUN_STEP, "Every step record, in order."),
        "currentStep": _nullable(CURRENT_STEP),
        "artifacts": _array(ARTIFACT, "Everything discovered under the run directory."),
        "runJsonUri": _string("URI of the authoritative run manifest."),
        "logTail": _array(
            _string("Log line."),
            "Supplementary human prose. Never parsed for state.",
        ),
    },
    "The merged view of a run: step-level truth from the manifest, process-level "
    "truth from the supervisor.",
)


def flow_list_result(flows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _shaped(FLOW_LIST_RESULT, {"flows": list(flows)})


def flow_inputs_result(
    *, flow: str, path: str, inputs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return _shaped(
        FLOW_INPUTS_RESULT, {"flow": flow, "path": path, "inputs": list(inputs)}
    )


def flow_validate_result(
    *,
    ok: bool,
    flow: str,
    step_count: int,
    input_count: int,
    tools: Sequence[str],
    missing_tools: Sequence[str],
    resolved_tools: Mapping[str, str],
    errors: Sequence[str],
) -> dict[str, Any]:
    return _shaped(
        FLOW_VALIDATE_RESULT,
        {
            "ok": ok,
            "flow": flow,
            "stepCount": step_count,
            "inputCount": input_count,
            "tools": list(tools),
            "missingTools": list(missing_tools),
            "resolvedTools": dict(resolved_tools),
            "errors": list(errors),
        },
    )


def flow_launch_result(
    *,
    run_id: str,
    run_dir: str,
    flow: str,
    status: str,
    pid: int | None,
    started_at: str,
    max_iterations: int | None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    return _shaped(
        FLOW_LAUNCH_RESULT,
        {
            "runId": run_id,
            "runDir": run_dir,
            "runJsonUri": manifest_uri(run_id),
            "flow": flow,
            "status": status,
            "pid": pid,
            "startedAt": started_at,
            "maxIterations": max_iterations,
            "warnings": list(warnings),
        },
    )


def flow_cancel_result(
    *,
    run_id: str,
    cancelled: bool,
    state: str,
    killed_pid: int | None,
    message: str,
) -> dict[str, Any]:
    return _shaped(
        FLOW_CANCEL_RESULT,
        {
            "runId": run_id,
            "cancelled": cancelled,
            "state": state,
            "killedPid": killed_pid,
            "message": message,
        },
    )


def flow_status_result(
    *,
    run_id: str,
    run_dir: str,
    flow: str,
    flow_path: str,
    status: str,
    engine_status: str | None,
    process_state: str | None,
    error: str | None,
    started_at: str | None,
    duration_s: float,
    exit_code: int | None,
    provenance: Mapping[str, Any],
    inputs: Mapping[str, Any],
    vars: Mapping[str, Any],
    loops: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    current_step: Mapping[str, Any] | None,
    artifacts: Sequence[Mapping[str, Any]],
    log_tail: Sequence[str] = (),
) -> dict[str, Any]:
    return _shaped(
        FLOW_STATUS_RESULT,
        {
            "runId": run_id,
            "runDir": run_dir,
            "flow": flow,
            "flowPath": flow_path,
            "status": status,
            "engineStatus": engine_status,
            "processState": process_state,
            "error": error,
            "startedAt": started_at,
            "durationS": duration_s,
            "exitCode": exit_code,
            "provenance": dict(provenance),
            "inputs": dict(inputs),
            "vars": dict(vars),
            "loops": list(loops),
            "steps": list(steps),
            "currentStep": dict(current_step) if current_step is not None else None,
            "artifacts": list(artifacts),
            "runJsonUri": manifest_uri(run_id),
            "logTail": list(log_tail),
        },
    )


# -- Tool definitions -------------------------------------------------------

_NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}

_FLOW_ARGUMENT = _string("Flow name, or a path that stays inside the flows directory.")

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "flow_list",
        "description": (
            "Enumerate the runnable flows, each described by its own declared "
            "structure: steps, loop groups, inputs and output types. Everything a "
            "caller needs to present a flow it has never seen."
        ),
        "inputSchema": _NO_ARGUMENTS,
        "outputSchema": FLOW_LIST_RESULT,
    },
    {
        "name": "flow_inputs",
        "description": (
            "One flow's declared inputs, with type, requiredness and default -- "
            "enough to build its form."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"flow": _FLOW_ARGUMENT},
            "required": ["flow"],
            "additionalProperties": False,
        },
        "outputSchema": FLOW_INPUTS_RESULT,
    },
    {
        "name": "flow_validate",
        "description": (
            "Check a flow without running it: reference resolution, tool coverage "
            "and, when inputs are supplied, executable resolution. Reports what "
            "would fail at launch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "flow": _FLOW_ARGUMENT,
                "inputs": _map(
                    "Optional; supplying values enables tool-resolution preflight."
                ),
                "profile": _string("Tool profile to resolve executables against."),
            },
            "required": ["flow"],
            "additionalProperties": False,
        },
        "outputSchema": FLOW_VALIDATE_RESULT,
    },
    {
        "name": "flow_launch",
        "description": (
            "Start a run and return immediately after preflight. The run executes "
            "in a separate process for minutes to hours; monitor it with "
            "flow_status or by subscribing to its run manifest."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "flow": _FLOW_ARGUMENT,
                "inputs": _map(
                    "Values for the inputs the flow declares; path-typed values "
                    "must be absolute."
                ),
                "maxIterations": _integer(
                    "Requested loop budget. Clamped to the budget the flow itself "
                    "declares, or to the operator's ceiling when one is set. "
                    "Rejected when the flow declares no loop.",
                    minimum=1,
                ),
                "profile": _string("Tool profile to resolve executables against."),
                "label": _string("Free text echoed in status; unused by the engine."),
            },
            "required": ["flow", "inputs"],
            "additionalProperties": False,
        },
        "outputSchema": FLOW_LAUNCH_RESULT,
    },
    {
        "name": "flow_cancel",
        "description": (
            "Tree-kill a running run. The last manifest checkpoint is preserved, "
            "so the evidence written up to that point survives."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"runId": _string("Run identity from flow_launch.")},
            "required": ["runId"],
            "additionalProperties": False,
        },
        "outputSchema": FLOW_CANCEL_RESULT,
    },
    {
        "name": "flow_status",
        "description": (
            "The merged state of a run: step and loop progress from the run "
            "manifest, process state from the supervisor, and every artifact "
            "discovered under the run directory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "runId": _string("Run identity from flow_launch."),
                "logTail": _integer(
                    "Supplementary human log lines to include; not state.",
                    minimum=0,
                    maximum=500,
                    default=50,
                ),
            },
            "required": ["runId"],
            "additionalProperties": False,
        },
        "outputSchema": FLOW_STATUS_RESULT,
    },
)

TOOL_BY_NAME: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in TOOLS}


# -- `run://` URIs ----------------------------------------------------------


def format_run_uri(run_id: str, path: str) -> str:
    """`run://<runId>/<path>`, POSIX separators, unencoded.

    Example: `run://20260915T171233Z-a1b4/run.json`. The path is whatever the
    run directory holds at that relative location; this function neither knows
    nor checks what that is.
    """
    if not run_id:
        raise ValueError("a run URI needs a run id")
    relative = str(path).replace("\\", "/").strip("/")
    if not relative:
        raise ValueError(f"a run URI needs a path inside run '{run_id}'")
    return f"{RUN_URI_SCHEME}://{run_id}/{relative}"


def manifest_uri(run_id: str) -> str:
    """The run's authoritative manifest -- the one URI a client may assume."""
    return format_run_uri(run_id, MANIFEST_NAME)


def parse_run_uri(uri: str) -> tuple[str, str]:
    """Split a run URI into `(runId, relative path)`.

    Both separator forms resolve identically: a client expanding the resource
    template with plain `{path}` percent-encodes every `/`, which would make
    each nested artifact unaddressable, so an encoded separator is accepted on
    the way in. Only the unencoded form is ever emitted.

    Containment is *not* checked here -- this is a parser. The path it returns
    may still escape the run directory, and the caller resolves it against the
    real directory before opening anything.
    """
    parts = urlsplit(str(uri))
    if parts.scheme != RUN_URI_SCHEME:
        raise ValueError(f"not a {RUN_URI_SCHEME}:// URI: {uri!r}")
    run_id = unquote(parts.netloc)
    if not run_id:
        raise ValueError(f"run URI has no run id: {uri!r}")
    relative = unquote(parts.path).replace("\\", "/").strip("/")
    if not relative:
        raise ValueError(f"run URI has no path: {uri!r}")
    return run_id, relative


def is_manifest_uri(uri: str) -> bool:
    """Whether a URI addresses a run's manifest, in either separator form.

    The subscription surface is exactly this: the manifest changes under a
    reader, every other artifact is written once.
    """
    try:
        _, relative = parse_run_uri(uri)
    except ValueError:
        return False
    return relative == MANIFEST_NAME
