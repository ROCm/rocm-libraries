# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The driver: step execution, the bounded repair loop, and the run record.

Evidence is written as it is produced -- rendered prompt, argv, logs, extracted outputs,
`run.json` after every step -- so a run that dies in hour three is still readable.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import refs
from .errors import ConfigError, LoopExhausted, OrchestratorError, StepError
from .flow import Flow, LoopGroup, Step
from .outputs import Artifacts, LazyText, extract, load_result
from .process import build_env, launch
from .toolreg import ToolRegistry

FEEDBACK_SEED = "first attempt - no prior failures\n"

#: An agent CLI in `--output-format json` mode emits its whole session as one line, with
#: the answer buried in an escaped string. The raw log stays byte-exact -- it is the
#: evidence -- and a readable sibling is written next to it when it parses.
PRETTY_LOG_LIMIT = 8 * 1024 * 1024


def _prettify_json_log(log_path: Path) -> None:
    try:
        if not log_path.is_file() or log_path.stat().st_size > PRETTY_LOG_LIMIT:
            return
        text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text or text[0] not in "{[":
            return
        document = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return
    # Long embedded strings (an agent's final message, a diff) are the whole point of
    # reading this file, so they are unescaped into real lines rather than left as one
    # \n-riddled blob.
    pretty = json.dumps(_unescape_long_strings(document), indent=2, ensure_ascii=False)
    log_path.with_suffix(".pretty.json").write_text(pretty, encoding="utf-8")


def _unescape_long_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unescape_long_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unescape_long_strings(item) for item in value]
    if isinstance(value, str) and "\n" in value:
        return value.splitlines()
    return value


@dataclass
class StepRecord:
    id: str
    status: str
    group: str | None = None
    iteration: int | None = None
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    argv: list[str] = field(default_factory=list)
    cwd: str | None = None
    dir: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class Invocation:
    """Everything a launch needs, resolved once. `plan()` and the real run build this
    the same way, so `--dry-run` describes the process that will actually start rather
    than an approximation of it."""

    argv: list[str]
    cwd: str | None
    env: dict[str, str]
    stdin: str | None
    result_file: str | None
    timeout: float | None


def _new_run_id() -> str:
    """Timestamp for humans, random suffix for correctness. Two runs starting in the
    same second is ordinary -- a sweep over graphs does it constantly -- and a
    second-resolution id silently points both at one directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(2)}"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git_revision(start: Path) -> str | None:
    """Best effort: not every checkout is a git checkout, and a run must never fail
    because provenance was unavailable."""
    try:
        finished = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip() or None


@dataclass
class LoopRecord:
    id: str
    iterations: int
    satisfied: bool
    until: str
    budget: int = 0
    budget_source: str = ""


@dataclass
class RunReport:
    run_id: str
    run_dir: Path
    status: str
    steps: list[StepRecord]
    loops: list[LoopRecord]
    error: str | None = None


class Engine:
    def __init__(
        self,
        flow: Flow,
        registry: ToolRegistry,
        inputs: dict[str, Any],
        *,
        run_root: Path | None = None,
        run_dir: Path | None = None,
        max_iterations: int | None = None,
        tee: bool = False,
        only: str | None = None,
        start_from: str | None = None,
        profile: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.flow = flow
        self.registry = registry
        self.inputs = inputs
        self.max_iterations = max_iterations
        self.tee = tee
        self.only = only
        self.start_from = start_from
        self.profile = profile
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        if max_iterations is not None and max_iterations < 1:
            raise ConfigError(
                f"--max-iterations must be at least 1, got {max_iterations}"
            )
        self.run_id = _new_run_id()
        #: An explicitly named run directory may be created for us; a generated one may
        #: not already exist. Either way nothing in it is ever overwritten.
        self.explicit_run_dir = run_dir is not None
        root = run_dir or (run_root or Path.cwd() / "runs") / flow.name / self.run_id
        self.run_dir = Path(root).resolve()
        self.feedback_path = self.run_dir / "feedback.md"
        # Flow vars may be written in terms of the run (e.g. a per-run scratch dir) or
        # of an input, so they are resolved here rather than taken literally.
        self.vars = self._resolve_flow_vars(registry.vars, flow.vars)
        self.completed: dict[str, dict[str, Any]] = {}
        self.records: list[StepRecord] = []
        self.loops: list[LoopRecord] = []
        self.started = time.time()
        self.provenance: dict[str, Any] = {}

    def _resolve_flow_vars(
        self, machine_vars: dict[str, Any], flow_vars: dict[str, Any]
    ) -> dict[str, Any]:
        resolved = dict(machine_vars)
        pending = dict(flow_vars)
        while pending:
            progressed = False
            for name in list(pending):
                resolver = refs.Resolver(
                    inputs=self.inputs,
                    vars=resolved,
                    run={
                        "id": self.run_id,
                        "name": self.flow.name,
                        "dir": str(self.run_dir),
                        "feedback_path": str(self.feedback_path),
                    },
                )
                try:
                    resolved[name] = refs.render_value(str(pending[name]), resolver)
                except ConfigError:
                    continue
                del pending[name]
                progressed = True
            if not progressed:
                raise ConfigError(
                    f"{self.flow.path}: flow vars cannot be resolved (cycle or unknown "
                    f"reference): {', '.join(sorted(pending))}"
                )
        return resolved

    # -- public ------------------------------------------------------------

    def preflight(self) -> dict[str, str]:
        """Resolve every tool the flow names, before anything launches."""
        resolved: dict[str, str] = {}
        for name in sorted(self.flow.tools_used()):
            tool = self.registry.get(name)
            resolved[name] = str(self.registry.resolve_exe(tool))
        return resolved

    def run(self) -> RunReport:
        self.preflight()
        self._create_run_dir()
        (self.run_dir / "inputs.json").write_text(
            json.dumps(self.inputs, indent=2, default=str), encoding="utf-8"
        )
        self.feedback_path.write_text(FEEDBACK_SEED, encoding="utf-8")
        self.started = time.time()
        self.provenance = self._collect_provenance()
        self._checkpoint()
        status, error = "ok", None
        try:
            for node in self._selected_nodes():
                if isinstance(node, LoopGroup):
                    self._run_loop(node)
                else:
                    self._run_step(node, self.run_dir / node.id)
        except (StepError, LoopExhausted, ConfigError) as failure:
            status, error = "failed", str(failure)
        self._checkpoint(status, error)
        return RunReport(
            run_id=self.run_id,
            run_dir=self.run_dir,
            status=status,
            steps=self.records,
            loops=self.loops,
            error=error,
        )

    def plan(self) -> list[dict[str, Any]]:
        """Resolve argv/env/cwd/prompt for every step without launching anything."""
        planned: list[dict[str, Any]] = []
        for node in self._selected_nodes():
            steps = node.steps if isinstance(node, LoopGroup) else [node]
            group = node.id if isinstance(node, LoopGroup) else None
            budget = self._budget(node)[0] if isinstance(node, LoopGroup) else None
            for step in steps:
                resolver = self._resolver(
                    step,
                    loop=(
                        {
                            "iteration": 0,
                            "attempt": 1,
                            "attempt_dir": str(
                                self.run_dir / (group or "") / "iter-00"
                            ),
                            "feedback_path": str(self.feedback_path),
                            "max_iterations": budget,
                        }
                        if isinstance(node, LoopGroup)
                        else None
                    ),
                    lenient=True,
                )
                invocation, _ = self._invocation(step, resolver)
                planned.append(
                    {
                        "id": step.id,
                        "group": group,
                        "tool": step.tool,
                        "argv": invocation.argv,
                        "cwd": invocation.cwd,
                        # Only what this flow changes about the environment. Echoing the
                        # whole inherited environment back would bury the tool wiring
                        # that is the reason to look.
                        "env": {
                            key: value
                            for key, value in invocation.env.items()
                            if os.environ.get(key) != value
                        },
                        "timeout": invocation.timeout,
                        "result_file": invocation.result_file,
                        "stdin": invocation.stdin,
                    }
                )
        return planned

    # -- resolution --------------------------------------------------------

    def _invocation(
        self, step: Step, resolver: refs.Resolver
    ) -> tuple[Invocation, refs.Resolver]:
        """Resolve one step's process. Returns the child resolver too, because
        `${step.result_file}` only exists once the result file has been resolved."""
        # The prompt tells the agent where to write its result, so result_file has to be
        # resolved before the prompt is rendered -- otherwise the contract renders empty
        # exactly where it matters most.
        result_file = (
            refs.render(step.result_file, resolver) if step.result_file else None
        )
        if result_file:
            resolver = resolver.child(step={"result_file": result_file})
        tool = self.registry.get(step.tool)
        invocation = Invocation(
            argv=[str(self.registry.resolve_exe(tool))]
            + [refs.render(arg, resolver) for arg in step.args],
            cwd=refs.render(step.cwd, resolver) if step.cwd else None,
            env=build_env(
                tool.env,
                {key: refs.render(value, resolver) for key, value in step.env.items()},
                [refs.render(item, resolver) for item in tool.path_prepend],
            ),
            stdin=self._stdin_text(step, resolver),
            result_file=result_file,
            timeout=step.timeout,
        )
        return invocation, resolver

    def _budget(self, group: LoopGroup) -> tuple[int, str]:
        # Where the budget came from matters in the failure message: "ran 1 iteration
        # without satisfying until" reads like a broken loop when it is really a
        # one-pass debugging run that was never allowed to iterate.
        if self.max_iterations is not None:
            return self.max_iterations, "--max-iterations"
        return group.loop.max_iterations, f"{self.flow.path.name} max_iterations"

    def _create_run_dir(self) -> None:
        """A run directory is created, never joined. Previous evidence is not ours to
        overwrite, and `--run-dir` aimed at a finished run is a mistake worth reporting
        rather than a silent merge of two runs' logs."""
        if self.explicit_run_dir:
            if self.run_dir.exists() and any(self.run_dir.iterdir()):
                raise ConfigError(
                    f"run directory {self.run_dir} already contains a run. Point "
                    f"--run-dir at a new or empty directory; re-running never "
                    f"overwrites previous evidence."
                )
            self.run_dir.mkdir(parents=True, exist_ok=True)
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ConfigError(f"run directory {self.run_dir} already exists") from None

    def _collect_provenance(self) -> dict[str, Any]:
        """What this run was produced *from*. Two runs whose flow or prompts differed
        are otherwise indistinguishable from two runs of the same thing."""
        prompts: dict[str, str] = {}
        for step in self.flow.all_steps():
            if not step.prompt_file or list(refs.iter_refs(step.prompt_file)):
                continue
            path = (self.flow.path.parent / step.prompt_file).resolve()
            if path.is_file():
                prompts[step.id] = _sha256(path)
        configs = {"flow": self.flow.path, "tools": self.registry.path}
        return {
            "config_sha256": {
                name: _sha256(path)
                for name, path in configs.items()
                if Path(path).is_file()
            },
            "prompt_sha256": prompts,
            "revision": _git_revision(self.flow.path.parent),
        }

    # -- internals ---------------------------------------------------------

    def _selected_nodes(self) -> Iterable[Any]:
        nodes = list(self.flow.nodes)
        if self.start_from:
            ids = [node.id for node in nodes]
            if self.start_from not in ids:
                raise ConfigError(
                    f"--from names '{self.start_from}', which is not a top-level step"
                )
            nodes = nodes[ids.index(self.start_from) :]
        if self.only:
            nodes = [node for node in nodes if node.id == self.only]
            if not nodes:
                raise ConfigError(
                    f"--only names '{self.only}', which is not a top-level step"
                )
        return nodes

    def _resolver(
        self,
        step: Step,
        *,
        loop: dict[str, Any] | None = None,
        previous: dict[str, dict[str, Any]] | None = None,
        lenient: bool = False,
        result_file: str | None = None,
    ) -> refs.Resolver:
        return refs.Resolver(
            inputs=self.inputs,
            vars=self.vars,
            run={
                "id": self.run_id,
                "name": self.flow.name,
                "dir": str(self.run_dir),
                "feedback_path": str(self.feedback_path),
            },
            step={"result_file": result_file or ""},
            loop=loop,
            steps=dict(self.completed),
            previous=previous,
            lenient=lenient,
        )

    def _run_loop(self, group: LoopGroup) -> None:
        budget, budget_source = self._budget(group)
        outer = dict(self.completed)
        previous: dict[str, dict[str, Any]] | None = None
        satisfied = False
        iteration = 0
        for iteration in range(budget):
            attempt_dir = self.run_dir / group.id / f"iter-{iteration:02d}"
            loop_ctx = {
                "iteration": iteration,
                "attempt": iteration + 1,
                "attempt_dir": str(attempt_dir),
                "feedback_path": str(self.feedback_path),
                "max_iterations": budget,
            }
            self.log(f"[{group.id}] iteration {iteration + 1}/{budget}")
            # Every iteration starts from the outputs that existed before the loop. A
            # step that is skipped or fails this time round must not still be readable
            # through `${steps...}` from the iteration where it succeeded: that is how
            # an exit condition gets satisfied by a value this iteration never produced.
            # Last iteration's values remain available, but only via `${loop.previous}`.
            self.completed = dict(outer)
            aborted: str | None = None
            for step in group.steps:
                try:
                    self._run_step(
                        step,
                        attempt_dir / step.id,
                        loop=loop_ctx,
                        previous=previous,
                        group=group.id,
                        iteration=iteration,
                    )
                except StepError as failure:
                    # `on_step_failure: fail` (the default) stops the run here. `retry`
                    # abandons the rest of this iteration and starts the next one: the
                    # cycle is the unit of retry, because re-running one step whose
                    # inputs have not changed just repeats the same failure at the same
                    # cost.
                    if group.loop.on_step_failure != "retry":
                        raise
                    aborted = str(failure)
                    self.log(f"[{group.id}] iteration abandoned: {aborted}")
                    break

            resolver = self._resolver(group.steps[-1], loop=loop_ctx, previous=previous)
            measured = self._measured(group.loop.until, resolver)
            if aborted is None and group.loop.until.evaluate(resolver):
                satisfied = True
                self.log(
                    f"[{group.id}] done after {iteration + 1} iteration(s): "
                    f"{measured} is true"
                )
                break
            if aborted is None:
                self.log(
                    f"[{group.id}] iteration {iteration + 1}/{budget} is not done yet: "
                    f"{measured} is false - looping to repair it"
                )
                self._append_feedback(group, iteration, resolver)
            else:
                self._write_feedback(
                    iteration,
                    f"The previous attempt did not complete: {aborted}\n\n"
                    f"Treat this as a failed round. Produce a complete result this time, "
                    f"including every file the instructions require.",
                )
            previous = {
                key: dict(value)
                for key, value in self.completed.items()
                if key not in outer
            }

        self.loops.append(
            LoopRecord(
                id=group.id,
                iterations=iteration + 1,
                satisfied=satisfied,
                until=group.loop.until.text,
                budget=budget,
                budget_source=budget_source,
            )
        )
        self._checkpoint()
        if not satisfied and group.loop.on_exhausted == "fail":
            hint = (
                " The budget was 1, so the loop was never allowed to act on the "
                "feedback it collected - raise it to iterate."
                if budget == 1
                else ""
            )
            raise LoopExhausted(
                f"loop '{group.id}' used its full budget of {budget} iteration(s) "
                f"(from {budget_source}) and never reached its goal {measured}."
                f"{hint} Evidence: {self.run_dir / group.id}"
            )

    def _measured(self, condition: refs.Condition, resolver: refs.Resolver) -> str:
        """A condition as written *and* as measured: `${...critical_count} == 0 (0 == 0)`.

        The text alone says what the loop wants and never says what it got, which reads
        as an unexplained verdict in the log of a run that is about to spend another
        hour of agent time.
        """
        try:
            rendered = refs.render(condition.text, resolver.child(lenient=True))
        except ConfigError:
            return condition.text
        return f"{condition.text} ({rendered})"

    def _append_feedback(
        self, group: LoopGroup, iteration: int, resolver: refs.Resolver
    ) -> None:
        if not group.loop.feedback_from:
            return
        text = refs.render(group.loop.feedback_from, resolver).strip()
        if text:
            self._write_feedback(iteration, text)

    def _write_feedback(self, iteration: int, text: str) -> None:
        existing = self.feedback_path.read_text(encoding="utf-8")
        if existing.strip() == FEEDBACK_SEED.strip():
            existing = ""
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.feedback_path.write_text(
            f"{existing}\n## iteration {iteration} ({stamp})\n\n{text}\n",
            encoding="utf-8",
        )

    def _run_step(
        self,
        step: Step,
        step_dir: Path,
        *,
        loop: dict[str, Any] | None = None,
        previous: dict[str, dict[str, Any]] | None = None,
        group: str | None = None,
        iteration: int | None = None,
    ) -> StepRecord:
        # One record per step, created before anything can fail and filled in as the
        # step proceeds. The earlier shape built a populated record, raised, and rebuilt
        # an empty one in the handler, so every failure -- the case an operator actually
        # reads -- reported exit_code=None, timed_out=False, duration 0 and no argv.
        record = StepRecord(
            id=step.id,
            status="pending",
            group=group,
            iteration=iteration,
            dir=str(step_dir),
        )
        self.records.append(record)
        resolver = self._resolver(step, loop=loop, previous=previous)
        try:
            if step.when and not step.when.evaluate(resolver):
                record.status = "skipped"
                self.log(
                    f"  - {step.id}: skipped, its `when` is false: "
                    f"{self._measured(step.when, resolver)}"
                )
                # A skipped step produces nothing this iteration. It gets an empty
                # bucket rather than inheriting whatever it produced last time.
                self.completed[step.id] = {}
                self._checkpoint()
                return record
            self._attempt(step, step_dir, resolver, record)
        except StepError as failure:
            record.status = "timed_out" if record.timed_out else "failed"
            record.error = str(failure)
            self.log(f"  - {step.id}: FAILED - {failure}")
            self._checkpoint()
            if not step.continue_on_error:
                raise
            # Tolerated failure still has to be readable downstream: `continue_on_error`
            # exists so a later step can branch on what happened here. `_attempt`
            # registered the process metadata as soon as the process returned; this only
            # covers a step that failed before it ever launched.
            self.completed.setdefault(step.id, {})
            return record
        except OrchestratorError as failure:
            record.status = "failed"
            record.error = str(failure)
            self._checkpoint()
            raise
        record.status = "ok"
        self._checkpoint()
        return record

    def _attempt(
        self, step: Step, step_dir: Path, resolver: refs.Resolver, record: StepRecord
    ) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        invocation, resolver = self._invocation(step, resolver)
        record.argv = invocation.argv
        record.cwd = invocation.cwd

        if invocation.result_file:
            result_path = Path(invocation.result_file)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.unlink(missing_ok=True)

        (step_dir / "cmd.txt").write_text(
            " ".join(shlex.quote(part) for part in invocation.argv) + "\n",
            encoding="utf-8",
        )
        (step_dir / "argv.json").write_text(
            json.dumps(
                {
                    "argv": invocation.argv,
                    "cwd": invocation.cwd,
                    "timeout": invocation.timeout,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if invocation.stdin is not None:
            (step_dir / "stdin.txt").write_text(invocation.stdin, encoding="utf-8")

        self.log(f"  - {step.id}: {step.tool} ...")
        record.status = "running"
        self._checkpoint()
        proc = launch(
            invocation.argv,
            cwd=invocation.cwd,
            env=invocation.env,
            stdout_path=step_dir / "stdout.log",
            stderr_path=step_dir / "stderr.log",
            stdin_text=invocation.stdin,
            timeout=invocation.timeout,
            tee=self.tee,
        )
        record.exit_code = proc.exit_code
        record.timed_out = proc.timed_out
        record.duration_s = proc.duration_s
        self.log(
            f"  - {step.id}: exit {proc.exit_code} in {proc.duration_s}s "
            f"-> {step_dir.relative_to(self.run_dir) if step_dir.is_relative_to(self.run_dir) else step_dir}"
        )
        _prettify_json_log(step_dir / "stdout.log")

        artifacts = Artifacts(
            stdout_path=step_dir / "stdout.log",
            stderr_path=step_dir / "stderr.log",
            workdir=Path(invocation.cwd) if invocation.cwd else Path.cwd(),
            result_path=(
                Path(invocation.result_file) if invocation.result_file else None
            ),
        )
        # Registered before the exit-code and timeout checks: a step that failed is
        # still a step that ran, and both the failure record and any downstream
        # condition reading its exit code depend on this metadata existing.
        values: dict[str, Any] = {
            "exit_code": proc.exit_code,
            "duration_s": proc.duration_s,
            "stdout_path": str(artifacts.stdout_path),
            "stderr_path": str(artifacts.stderr_path),
            "workdir": str(artifacts.workdir),
            # Paths eagerly, text on demand. A build log is evidence on disk; holding
            # every megabyte of it in orchestration state on the chance that some later
            # condition reads it is what the streaming design exists to avoid.
            "stdout": LazyText(artifacts.stdout_path),
            "stderr": LazyText(artifacts.stderr_path),
        }
        self.completed[step.id] = values
        record.outputs = _recorded(values)

        if proc.timed_out:
            raise StepError(
                f"step '{step.id}' exceeded its {step.timeout}s timeout and was killed "
                f"(logs: {step_dir})"
            )
        if proc.exit_code not in step.expect_exit:
            raise StepError(
                f"step '{step.id}' exited {proc.exit_code}, expected one of "
                f"{list(step.expect_exit)} (logs: {step_dir})"
            )

        result = (
            load_result(
                step.id,
                Path(invocation.result_file),
                step.result_schema.get("required", []),
            )
            if invocation.result_file
            else None
        )

        for spec in step.outputs:
            argument = (
                refs.render(str(spec.argument), resolver)
                if isinstance(spec.argument, str)
                else spec.argument
            )
            values[spec.name] = extract(
                spec, argument, artifacts, result, proc.exit_code
            )
        record.outputs = _recorded(values)
        (step_dir / "result.json").write_text(
            json.dumps(record.outputs, indent=2, default=str), encoding="utf-8"
        )

        assert_resolver = self._resolver(
            step,
            loop=resolver.loop,
            previous=resolver.previous,
            result_file=invocation.result_file,
        )
        for item in step.asserts:
            if not item.that.evaluate(assert_resolver):
                raise StepError(
                    f"step '{step.id}': assertion failed - "
                    f"{item.message or item.that.text}; measured: "
                    f"{self._measured(item.that, assert_resolver)}"
                )

    def _stdin_text(self, step: Step, resolver: refs.Resolver) -> str | None:
        if step.stdin is not None:
            return refs.render(step.stdin, resolver)
        if step.prompt_file:
            path = Path(refs.render(step.prompt_file, resolver))
            if not path.is_absolute():
                path = (self.flow.path.parent / path).resolve()
            if not path.is_file():
                raise ConfigError(
                    f"step '{step.id}': prompt_file {path} does not exist"
                )
            return refs.render(path.read_text(encoding="utf-8"), resolver)
        return None

    def _checkpoint(self, status: str = "running", error: str | None = None) -> None:
        """Write `run.json` now, atomically.

        The manifest is the run's report, and a run that dies in its third hour still
        has to leave one. Writing it only at the end means the evidence for the failure
        mode most worth reading is the evidence that never gets written.
        """
        payload = {
            "run_id": self.run_id,
            "flow": self.flow.name,
            "flow_path": str(self.flow.path),
            "tools_path": str(self.registry.path),
            "profile": self.profile,
            "status": status,
            "error": error,
            "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(time.time() - self.started, 3),
            "provenance": self.provenance,
            "inputs": self.inputs,
            "vars": self.vars,
            "loops": [vars(loop) for loop in self.loops],
            "steps": [vars(record) for record in self.records],
        }
        temporary = self.run_dir / "run.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, self.run_dir / "run.json")


def _recorded(values: dict[str, Any]) -> dict[str, Any]:
    """The manifest view of a step's outputs: everything except the log text, which is
    already on disk and is referenced by path."""
    return {
        key: value for key, value in values.items() if key not in ("stdout", "stderr")
    }
