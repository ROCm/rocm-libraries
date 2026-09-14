# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Flow schema: run inputs, steps, and the bounded repair loop.

Everything here is validated before a single process starts. The expensive failure mode
this guards against is a four-hour agent run that dies on a typo in the step *after* it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import ConfigError
from .refs import iter_refs

IMPLICIT_OUTPUTS = (
    "exit_code",
    "stdout",
    "stderr",
    "stdout_path",
    "stderr_path",
    "duration_s",
    "workdir",
)
RUN_FIELDS = ("id", "name", "dir", "feedback_path")
LOOP_FIELDS = ("iteration", "attempt", "attempt_dir", "feedback_path", "max_iterations")
EXTRACTOR_KINDS = ("regex", "json", "json_file", "file", "glob", "tail", "lines")
OUTPUT_KEYS = set(EXTRACTOR_KINDS) | {"type", "group", "path", "from"}
OUTPUT_TYPES = ("string", "int", "float", "bool", "path", "json")
INPUT_TYPES = ("string", "text", "int", "float", "bool", "path")
STEP_KEYS = {
    "id",
    "tool",
    "args",
    "env",
    "cwd",
    "timeout",
    "stdin",
    "prompt_file",
    "result_file",
    "result_schema",
    "outputs",
    "assert",
    "when",
    "expect_exit",
    "retries",
    "continue_on_error",
    "after",
}
LOOP_KEYS = {"max_iterations", "until", "on_exhausted", "feedback_from"}
FLOW_KEYS = {"version", "name", "description", "inputs", "vars", "steps"}


@dataclass(frozen=True)
class InputSpec:
    name: str
    description: str = ""
    type: str = "string"
    required: bool = False
    default: Any = None
    exists: bool = True


@dataclass(frozen=True)
class OutputSpec:
    name: str
    kind: str
    argument: Any
    type: str = "string"
    group: int = 1
    path: str | None = None
    source: str = "stdout"


@dataclass
class Step:
    id: str
    tool: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout: float | None = None
    stdin: str | None = None
    prompt_file: str | None = None
    result_file: str | None = None
    result_schema: dict[str, Any] = field(default_factory=dict)
    outputs: list[OutputSpec] = field(default_factory=list)
    asserts: list[dict[str, str]] = field(default_factory=list)
    when: str | None = None
    expect_exit: tuple[int, ...] = (0,)
    retries: int = 0
    continue_on_error: bool = False
    after: tuple[str, ...] = ()

    @property
    def output_names(self) -> set[str]:
        return {spec.name for spec in self.outputs} | set(IMPLICIT_OUTPUTS)


@dataclass(frozen=True)
class LoopSpec:
    max_iterations: int
    until: str
    on_exhausted: str = "fail"
    feedback_from: str | None = None


@dataclass
class LoopGroup:
    id: str
    loop: LoopSpec
    steps: list[Step]


Node = Step | LoopGroup


@dataclass
class Flow:
    path: Path
    name: str
    description: str
    inputs: dict[str, InputSpec]
    vars: dict[str, Any]
    nodes: list[Node]

    @classmethod
    def load(cls, path: str | Path) -> "Flow":
        path = Path(path).resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError as error:
            raise ConfigError(f"cannot read flow {path}: {error}") from None
        except yaml.YAMLError as error:
            raise ConfigError(f"{path} is not valid YAML: {error}") from None
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: top level must be a mapping")
        _reject_unknown(raw, FLOW_KEYS, f"{path}: top level")

        name = str(raw.get("name") or path.stem)
        inputs = {
            key: _input(key, spec, path)
            for key, spec in (raw.get("inputs") or {}).items()
        }
        nodes = [_node(entry, path) for entry in (raw.get("steps") or [])]
        if not nodes:
            raise ConfigError(f"{path}: flow declares no steps")

        flow = cls(
            path=path,
            name=name,
            description=str(raw.get("description") or ""),
            inputs=inputs,
            vars=dict(raw.get("vars") or {}),
            nodes=nodes,
        )
        _check_unique_ids(flow)
        return flow

    def all_steps(self) -> Iterable[Step]:
        for node in self.nodes:
            if isinstance(node, LoopGroup):
                yield from node.steps
            else:
                yield node

    def tools_used(self) -> set[str]:
        return {step.tool for step in self.all_steps()}


# -- parsing ----------------------------------------------------------------


def _input(name: str, spec: Any, path: Path) -> InputSpec:
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{path}: input '{name}' must be a mapping")
    _reject_unknown(
        spec,
        {"description", "type", "required", "default", "exists"},
        f"{path}: input '{name}'",
    )
    kind = str(spec.get("type", "string"))
    if kind not in INPUT_TYPES:
        raise ConfigError(
            f"{path}: input '{name}' has type '{kind}'; expected one of {', '.join(INPUT_TYPES)}"
        )
    return InputSpec(
        name=name,
        description=str(spec.get("description") or ""),
        type=kind,
        required=bool(spec.get("required", False)),
        default=spec.get("default"),
        exists=bool(spec.get("exists", True)),
    )


def _node(entry: Any, path: Path) -> Node:
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{path}: each step must be a mapping")
    if "loop" in entry:
        return _loop_group(entry, path)
    return _step(entry, path)


def _loop_group(entry: Mapping[str, Any], path: Path) -> LoopGroup:
    _reject_unknown(entry, {"id", "loop", "steps"}, f"{path}: loop group")
    group_id = _require(entry, "id", path, "loop group")
    raw_loop = entry.get("loop") or {}
    if not isinstance(raw_loop, Mapping):
        raise ConfigError(f"{path}: loop '{group_id}': 'loop' must be a mapping")
    _reject_unknown(raw_loop, LOOP_KEYS, f"{path}: loop '{group_id}'")

    until = raw_loop.get("until")
    if not until:
        raise ConfigError(
            f"{path}: loop '{group_id}' has no 'until'. A loop without a measured exit "
            f"condition cannot tell success from an empty run."
        )
    try:
        max_iterations = int(raw_loop.get("max_iterations", 3))
    except (TypeError, ValueError):
        raise ConfigError(
            f"{path}: loop '{group_id}': max_iterations must be an integer"
        ) from None
    if max_iterations < 1:
        raise ConfigError(f"{path}: loop '{group_id}': max_iterations must be >= 1")
    on_exhausted = str(raw_loop.get("on_exhausted", "fail"))
    if on_exhausted not in ("fail", "continue"):
        raise ConfigError(
            f"{path}: loop '{group_id}': on_exhausted must be 'fail' or 'continue'"
        )

    steps = [_step(item, path) for item in (entry.get("steps") or [])]
    if not steps:
        raise ConfigError(f"{path}: loop '{group_id}' contains no steps")
    return LoopGroup(
        id=str(group_id),
        loop=LoopSpec(
            max_iterations=max_iterations,
            until=str(until),
            on_exhausted=on_exhausted,
            feedback_from=(
                str(raw_loop["feedback_from"])
                if raw_loop.get("feedback_from")
                else None
            ),
        ),
        steps=steps,
    )


def _step(entry: Mapping[str, Any], path: Path) -> Step:
    _reject_unknown(entry, STEP_KEYS, f"{path}: step {entry.get('id', '<unnamed>')!r}")
    step_id = str(_require(entry, "id", path, "step"))
    tool = str(_require(entry, "tool", path, f"step '{step_id}'"))

    if entry.get("stdin") is not None and entry.get("prompt_file") is not None:
        raise ConfigError(
            f"{path}: step '{step_id}' sets both 'stdin' and 'prompt_file'; they are the "
            f"same channel, so only one may be given"
        )
    args = entry.get("args") or []
    if isinstance(args, str) or not isinstance(args, Sequence):
        raise ConfigError(f"{path}: step '{step_id}': args must be a list of strings")

    result_schema = entry.get("result_schema") or {}
    if result_schema and not entry.get("result_file"):
        raise ConfigError(
            f"{path}: step '{step_id}' declares result_schema without result_file"
        )
    if not isinstance(result_schema, Mapping):
        raise ConfigError(f"{path}: step '{step_id}': result_schema must be a mapping")
    _reject_unknown(
        result_schema, {"required"}, f"{path}: step '{step_id}' result_schema"
    )

    outputs = [
        _output(step_id, name, spec, path)
        for name, spec in (entry.get("outputs") or {}).items()
    ]
    for spec in outputs:
        if (
            spec.kind == "json_file"
            and spec.argument == "result"
            and not entry.get("result_file")
        ):
            raise ConfigError(
                f"{path}: step '{step_id}' output '{spec.name}' reads the step's result "
                f"file, but the step declares no result_file"
            )

    asserts = []
    for item in entry.get("assert") or []:
        if not isinstance(item, Mapping) or "that" not in item:
            raise ConfigError(f"{path}: step '{step_id}': each assert needs a 'that'")
        _reject_unknown(item, {"that", "message"}, f"{path}: step '{step_id}' assert")
        asserts.append(
            {"that": str(item["that"]), "message": str(item.get("message") or "")}
        )

    expect = entry.get("expect_exit", [0])
    if isinstance(expect, int):
        expect = [expect]
    try:
        expect_exit = tuple(int(code) for code in expect)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{path}: step '{step_id}': expect_exit must be integers"
        ) from None

    timeout = entry.get("timeout")
    after = entry.get("after") or []
    if isinstance(after, str):
        after = [after]

    return Step(
        id=step_id,
        tool=tool,
        args=[str(item) for item in args],
        env={str(key): str(value) for key, value in (entry.get("env") or {}).items()},
        cwd=(str(entry["cwd"]) if entry.get("cwd") else None),
        timeout=(float(timeout) if timeout is not None else None),
        stdin=(str(entry["stdin"]) if entry.get("stdin") is not None else None),
        prompt_file=(str(entry["prompt_file"]) if entry.get("prompt_file") else None),
        result_file=(str(entry["result_file"]) if entry.get("result_file") else None),
        result_schema={
            "required": [str(key) for key in (result_schema.get("required") or [])]
        },
        outputs=outputs,
        asserts=asserts,
        when=(str(entry["when"]) if entry.get("when") else None),
        expect_exit=expect_exit,
        retries=int(entry.get("retries", 0)),
        continue_on_error=bool(entry.get("continue_on_error", False)),
        after=tuple(str(item) for item in after),
    )


def _output(step_id: str, name: str, spec: Any, path: Path) -> OutputSpec:
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{path}: step '{step_id}' output '{name}' must be a mapping")
    _reject_unknown(spec, OUTPUT_KEYS, f"{path}: step '{step_id}' output '{name}'")
    kinds = [key for key in EXTRACTOR_KINDS if key in spec]
    if len(kinds) != 1:
        raise ConfigError(
            f"{path}: step '{step_id}' output '{name}' must name exactly one extractor "
            f"({', '.join(EXTRACTOR_KINDS)}); got {len(kinds)}"
        )
    kind = kinds[0]
    out_type = str(spec.get("type", "string"))
    if out_type not in OUTPUT_TYPES:
        raise ConfigError(
            f"{path}: step '{step_id}' output '{name}': type '{out_type}' is not one of "
            f"{', '.join(OUTPUT_TYPES)}"
        )
    source = str(spec.get("from", "stdout"))
    if source not in ("stdout", "stderr"):
        raise ConfigError(
            f"{path}: step '{step_id}' output '{name}': 'from' must be stdout or stderr"
        )
    if kind in ("json", "json_file") and not spec.get("path"):
        raise ConfigError(
            f"{path}: step '{step_id}' output '{name}': a {kind} extractor needs 'path' "
            f'(e.g. path: "$.result")'
        )
    return OutputSpec(
        name=name,
        kind=kind,
        argument=spec[kind],
        type=out_type,
        group=int(spec.get("group", 1)),
        path=(str(spec["path"]) if spec.get("path") else None),
        source=source,
    )


# -- validation -------------------------------------------------------------


def _check_unique_ids(flow: Flow) -> None:
    seen: set[str] = set()
    for node in flow.nodes:
        ids = [node.id] if isinstance(node, LoopGroup) else [node.id]
        if isinstance(node, LoopGroup):
            ids += [step.id for step in node.steps]
        for identifier in ids:
            if identifier in seen:
                raise ConfigError(f"{flow.path}: duplicate step id '{identifier}'")
            seen.add(identifier)


def validate_refs(flow: Flow, machine_vars: Mapping[str, Any]) -> None:
    """Every reference must name something that exists *and has already run*.

    Forward references are the interesting case: they parse fine and read fine, and
    then resolve to nothing at the moment they matter.
    """
    known_vars = set(machine_vars) | set(flow.vars)
    shadowed = sorted(set(machine_vars) & set(flow.vars))
    if shadowed:
        raise ConfigError(
            f"{flow.path}: flow vars shadow machine vars from the tool registry: "
            f"{', '.join(shadowed)}. Rename them; a machine path must mean one thing."
        )

    available: dict[str, set[str]] = {}
    for node in flow.nodes:
        if isinstance(node, LoopGroup):
            loop_scope: dict[str, set[str]] = {}
            for step in node.steps:
                _check_step_refs(
                    flow,
                    step,
                    {**available, **loop_scope},
                    known_vars,
                    in_loop=True,
                    loop_steps={s.id for s in node.steps},
                )
                loop_scope[step.id] = step.output_names
            # `until` and `feedback_from` are evaluated after an iteration, so every
            # step of that iteration is in scope for them.
            scope = {**available, **loop_scope}
            _check_refs_in(
                flow,
                node.loop.until,
                scope,
                known_vars,
                True,
                {s.id for s in node.steps},
                f"loop '{node.id}' until",
            )
            if node.loop.feedback_from:
                _check_refs_in(
                    flow,
                    node.loop.feedback_from,
                    scope,
                    known_vars,
                    True,
                    {s.id for s in node.steps},
                    f"loop '{node.id}' feedback_from",
                )
            available.update(loop_scope)
        else:
            _check_step_refs(
                flow, node, available, known_vars, in_loop=False, loop_steps=set()
            )
            available[node.id] = node.output_names


def _check_step_refs(
    flow: Flow,
    step: Step,
    available: Mapping[str, set[str]],
    known_vars: set[str],
    in_loop: bool,
    loop_steps: set[str],
) -> None:
    for missing in step.after:
        if missing not in available:
            raise ConfigError(
                f"{flow.path}: step '{step.id}' declares after: [{missing}], which does "
                f"not run before it"
            )
    fields: list[Any] = [
        step.args,
        step.env,
        step.cwd,
        step.stdin,
        step.prompt_file,
        step.result_file,
        step.when,
    ]
    fields += [spec.argument for spec in step.outputs]
    fields += [spec.path for spec in step.outputs if spec.path]
    for value in fields:
        _check_refs_in(
            flow,
            value,
            available,
            known_vars,
            in_loop,
            loop_steps,
            f"step '{step.id}'",
            has_result_file=bool(step.result_file),
        )

    # Asserts run after the step's own outputs have been extracted, so they -- and only
    # they -- may read this step's outputs. That is the whole point: an assert exists to
    # check what this step just produced.
    own_scope = {**available, step.id: step.output_names}
    for item in step.asserts:
        _check_refs_in(
            flow,
            item["that"],
            own_scope,
            known_vars,
            in_loop,
            loop_steps,
            f"step '{step.id}' assert",
            has_result_file=bool(step.result_file),
        )

    # A prompt file is where most references actually live, so it is checked here too.
    # Otherwise a typo in a prompt surfaces only when that step launches -- which for an
    # agent flow can be an hour in.
    if step.prompt_file and not list(iter_refs(step.prompt_file)):
        prompt = (flow.path.parent / step.prompt_file).resolve()
        if not prompt.is_file():
            raise ConfigError(
                f"{flow.path}: step '{step.id}': prompt_file {prompt} does not exist"
            )
        _check_refs_in(
            flow,
            prompt.read_text(encoding="utf-8"),
            available,
            known_vars,
            in_loop,
            loop_steps,
            f"step '{step.id}' prompt {prompt.name}",
            has_result_file=bool(step.result_file),
        )


def _check_refs_in(
    flow: Flow,
    value: Any,
    available: Mapping[str, set[str]],
    known_vars: set[str],
    in_loop: bool,
    loop_steps: set[str],
    where: str,
    has_result_file: bool = False,
) -> None:
    for ref in iter_refs(value):
        parts = [part for part in ref.split(".") if part]
        if not parts:
            raise ConfigError(f"{flow.path}: {where}: empty reference")
        head, rest = parts[0], parts[1:]
        if head == "inputs":
            if len(rest) != 1 or rest[0] not in flow.inputs:
                declared = ", ".join(sorted(flow.inputs)) or "(none)"
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}' is not a declared input; declared: {declared}"
                )
        elif head == "vars":
            if len(rest) != 1 or rest[0] not in known_vars:
                declared = ", ".join(sorted(known_vars)) or "(none)"
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}' is not a known var; known: {declared}"
                )
        elif head == "run":
            if len(rest) != 1 or rest[0] not in RUN_FIELDS:
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}'; run fields are {', '.join(RUN_FIELDS)}"
                )
        elif head == "step":
            if rest != ["result_file"]:
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}'; the only step field is step.result_file"
                )
            if not has_result_file:
                raise ConfigError(
                    f"{flow.path}: {where}: uses '${{step.result_file}}' but declares no result_file"
                )
        elif head == "loop":
            if not in_loop:
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}' is only available inside a loop"
                )
            if rest and rest[0] == "previous":
                if len(rest) != 4 or rest[2] != "outputs":
                    raise ConfigError(
                        f"{flow.path}: {where}: '${{{ref}}}' must be "
                        f"'loop.previous.<step-id>.outputs.<name>'"
                    )
                if rest[1] not in loop_steps:
                    raise ConfigError(
                        f"{flow.path}: {where}: '${{{ref}}}' names '{rest[1]}', which is "
                        f"not a step of this loop"
                    )
            elif len(rest) != 1 or rest[0] not in LOOP_FIELDS:
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}'; loop fields are {', '.join(LOOP_FIELDS)}"
                )
        elif head == "steps":
            if len(rest) != 3 or rest[1] != "outputs":
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}' must be 'steps.<id>.outputs.<name>'"
                )
            step_id, name = rest[0], rest[2]
            if step_id not in available:
                order = ", ".join(available) or "(none)"
                raise ConfigError(
                    f"{flow.path}: {where}: '${{{ref}}}' reads step '{step_id}', which does "
                    f"not run before it; available: {order}"
                )
            if name not in available[step_id]:
                declared = ", ".join(sorted(available[step_id]))
                raise ConfigError(
                    f"{flow.path}: {where}: step '{step_id}' declares no output '{name}'; "
                    f"it has: {declared}"
                )
        elif head not in ("env", "platform"):
            raise ConfigError(
                f"{flow.path}: {where}: unknown namespace in '${{{ref}}}'"
            )


# -- run inputs -------------------------------------------------------------


def bind_inputs(
    flow: Flow, pairs: Sequence[str] = (), inputs_file: str | Path | None = None
) -> dict[str, Any]:
    """Turn `--input k=v` / `--inputs-file f.yaml` into typed, checked values."""
    supplied: dict[str, Any] = {}
    if inputs_file:
        raw = yaml.safe_load(Path(inputs_file).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{inputs_file}: must be a mapping of input -> value")
        supplied.update(raw)
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--input expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        supplied[key.strip()] = value

    unknown = sorted(set(supplied) - set(flow.inputs))
    if unknown:
        declared = ", ".join(sorted(flow.inputs)) or "(none)"
        raise ConfigError(
            f"{flow.path}: unknown input(s) {', '.join(unknown)}; declared: {declared}"
        )

    bound: dict[str, Any] = {}
    missing: list[str] = []
    for name, spec in flow.inputs.items():
        if name in supplied:
            bound[name] = _coerce(spec, supplied[name])
        elif spec.default is not None:
            bound[name] = _coerce(spec, spec.default)
        elif spec.required:
            missing.append(f"  {name}: {spec.description or '(no description)'}")
        else:
            bound[name] = ""
    if missing:
        raise ConfigError(
            "missing required input(s):\n"
            + "\n".join(missing)
            + "\n\nsupply them with --input <name>=<value> (or =@file to read a file)"
        )
    return bound


def _coerce(spec: InputSpec, value: Any) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        source = Path(value[1:]).expanduser()
        if not source.is_file():
            raise ConfigError(f"input '{spec.name}': {source} does not exist")
        value = source.read_text(encoding="utf-8")
    if spec.type in ("string", "text"):
        return str(value)
    if spec.type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"input '{spec.name}' must be an integer, got {value!r}"
            ) from None
    if spec.type == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"input '{spec.name}' must be a number, got {value!r}"
            ) from None
    if spec.type == "bool":
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    resolved = Path(str(value)).expanduser().resolve()
    if spec.exists and not resolved.exists():
        raise ConfigError(f"input '{spec.name}': {resolved} does not exist")
    return str(resolved)


def _require(mapping: Mapping[str, Any], key: str, path: Path, where: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ConfigError(f"{path}: {where} has no '{key}'")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
