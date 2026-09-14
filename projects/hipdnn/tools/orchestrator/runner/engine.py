# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The driver: step execution, the bounded repair loop, and the run record.

Evidence is written as it is produced -- rendered prompt, argv, logs, extracted outputs,
`run.json` after every step -- so a run that dies in hour three is still readable.
"""
from __future__ import annotations

import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import refs
from .errors import ConfigError, LoopExhausted, StepError
from .flow import Flow, LoopGroup, Step
from .outputs import Artifacts, extract, load_result
from .process import build_env, launch
from .toolreg import ToolRegistry

FEEDBACK_SEED = "first attempt - no prior failures\n"


@dataclass
class StepRecord:
    id: str
    status: str
    group: str | None = None
    iteration: int | None = None
    attempt: int = 1
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    argv: list[str] = field(default_factory=list)
    cwd: str | None = None
    dir: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class LoopRecord:
    id: str
    iterations: int
    satisfied: bool
    until: str


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
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = run_dir or (run_root or Path.cwd() / "runs") / flow.name / self.run_id
        self.run_dir = Path(root).resolve()
        self.feedback_path = self.run_dir / "feedback.md"
        # Flow vars may be written in terms of the run (e.g. a per-run scratch dir) or
        # of an input, so they are resolved here rather than taken literally.
        self.vars = self._resolve_flow_vars(registry.vars, flow.vars)
        self.completed: dict[str, dict[str, Any]] = {}
        self.records: list[StepRecord] = []
        self.loops: list[LoopRecord] = []

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
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "inputs.json").write_text(
            json.dumps(self.inputs, indent=2, default=str), encoding="utf-8"
        )
        self.feedback_path.write_text(FEEDBACK_SEED, encoding="utf-8")
        started = time.time()
        status, error = "ok", None
        try:
            for node in self._selected_nodes():
                if isinstance(node, LoopGroup):
                    self._run_loop(node)
                else:
                    self._run_step(node, self.run_dir / node.id)
        except (StepError, LoopExhausted, ConfigError) as failure:
            status, error = "failed", str(failure)
        report = RunReport(
            run_id=self.run_id,
            run_dir=self.run_dir,
            status=status,
            steps=self.records,
            loops=self.loops,
            error=error,
        )
        self._write_manifest(report, started)
        return report

    def plan(self) -> list[dict[str, Any]]:
        """Resolve argv/env/cwd/prompt for every step without launching anything."""
        planned: list[dict[str, Any]] = []
        for node in self._selected_nodes():
            steps = node.steps if isinstance(node, LoopGroup) else [node]
            group = node.id if isinstance(node, LoopGroup) else None
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
                            "max_iterations": node.loop.max_iterations,
                        }
                        if isinstance(node, LoopGroup)
                        else None
                    ),
                    lenient=True,
                )
                # The prompt tells the agent where to write its result, so the plan has
                # to resolve result_file before rendering the prompt -- otherwise
                # --dry-run shows an empty path exactly where the contract lives.
                result_file = (
                    refs.render(step.result_file, resolver)
                    if step.result_file
                    else None
                )
                if result_file:
                    resolver = resolver.child(step={"result_file": result_file})
                tool = self.registry.get(step.tool)
                argv = [str(self.registry.resolve_exe(tool))] + [
                    refs.render(arg, resolver) for arg in step.args
                ]
                planned.append(
                    {
                        "id": step.id,
                        "group": group,
                        "tool": step.tool,
                        "argv": argv,
                        "cwd": refs.render(step.cwd, resolver) if step.cwd else None,
                        "env": {
                            key: refs.render(value, resolver)
                            for key, value in step.env.items()
                        },
                        "timeout": step.timeout,
                        "result_file": result_file,
                        "stdin": self._stdin_text(step, resolver),
                    }
                )
        return planned

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
        budget = self.max_iterations or group.loop.max_iterations
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
            iteration_outputs: dict[str, dict[str, Any]] = {}
            for step in group.steps:
                record = self._run_step(
                    step,
                    attempt_dir / step.id,
                    loop=loop_ctx,
                    previous=previous,
                    group=group.id,
                    iteration=iteration,
                )
                if record.status in ("ok", "skipped"):
                    iteration_outputs[step.id] = self.completed.get(step.id, {})

            resolver = self._resolver(group.steps[-1], loop=loop_ctx, previous=previous)
            if refs.evaluate(group.loop.until, resolver):
                satisfied = True
                self.log(
                    f"[{group.id}] until satisfied after {iteration + 1} iteration(s)"
                )
                break
            self.log(f"[{group.id}] until not satisfied: {group.loop.until}")
            self._append_feedback(group, iteration, resolver)
            previous = {key: dict(value) for key, value in iteration_outputs.items()}

        self.loops.append(
            LoopRecord(
                id=group.id,
                iterations=iteration + 1,
                satisfied=satisfied,
                until=group.loop.until,
            )
        )
        if not satisfied and group.loop.on_exhausted == "fail":
            raise LoopExhausted(
                f"loop '{group.id}' ran {iteration + 1} iteration(s) without satisfying "
                f"`until: {group.loop.until}`. Evidence: {self.run_dir / group.id}"
            )

    def _append_feedback(
        self, group: LoopGroup, iteration: int, resolver: refs.Resolver
    ) -> None:
        if not group.loop.feedback_from:
            return
        text = refs.render(group.loop.feedback_from, resolver).strip()
        if not text:
            return
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
        resolver = self._resolver(step, loop=loop, previous=previous)
        if step.when and not refs.evaluate(step.when, resolver):
            record = StepRecord(
                id=step.id,
                status="skipped",
                group=group,
                iteration=iteration,
                dir=str(step_dir),
            )
            self.log(f"  - {step.id}: skipped (when: {step.when})")
            self.completed.setdefault(step.id, {})
            self.records.append(record)
            return record

        last_error: Exception | None = None
        for attempt in range(step.retries + 1):
            attempt_dir = (
                step_dir
                if attempt == 0
                else step_dir.parent / f"{step.id}.retry-{attempt}"
            )
            try:
                record = self._attempt(
                    step, attempt_dir, resolver, group, iteration, attempt + 1
                )
                self.records.append(record)
                return record
            except StepError as failure:
                last_error = failure
                self.records.append(
                    StepRecord(
                        id=step.id,
                        status="failed",
                        group=group,
                        iteration=iteration,
                        attempt=attempt + 1,
                        dir=str(attempt_dir),
                        error=str(failure),
                    )
                )
                if attempt == step.retries:
                    break

        assert last_error is not None
        if step.continue_on_error:
            self.completed.setdefault(step.id, {})
            return self.records[-1]
        raise last_error

    def _attempt(
        self,
        step: Step,
        step_dir: Path,
        resolver: refs.Resolver,
        group: str | None,
        iteration: int | None,
        attempt: int,
    ) -> StepRecord:
        step_dir.mkdir(parents=True, exist_ok=True)
        tool = self.registry.get(step.tool)
        exe = self.registry.resolve_exe(tool)

        result_file = (
            refs.render(step.result_file, resolver) if step.result_file else None
        )
        if result_file:
            resolver = resolver.child(step={"result_file": result_file})
            Path(result_file).parent.mkdir(parents=True, exist_ok=True)
            Path(result_file).unlink(missing_ok=True)

        argv = [str(exe)] + [refs.render(arg, resolver) for arg in step.args]
        cwd = refs.render(step.cwd, resolver) if step.cwd else None
        env = build_env(
            tool.env,
            {key: refs.render(value, resolver) for key, value in step.env.items()},
            [refs.render(item, resolver) for item in tool.path_prepend],
        )
        stdin_text = self._stdin_text(step, resolver)

        (step_dir / "cmd.txt").write_text(
            " ".join(shlex.quote(part) for part in argv) + "\n", encoding="utf-8"
        )
        (step_dir / "argv.json").write_text(
            json.dumps({"argv": argv, "cwd": cwd, "timeout": step.timeout}, indent=2),
            encoding="utf-8",
        )
        if stdin_text is not None:
            (step_dir / "stdin.txt").write_text(stdin_text, encoding="utf-8")

        retry_note = f" (retry {attempt - 1})" if attempt > 1 else ""
        self.log(f"  - {step.id}: {step.tool}{retry_note} ...")
        proc = launch(
            argv,
            cwd=cwd,
            env=env,
            stdout_path=step_dir / "stdout.log",
            stderr_path=step_dir / "stderr.log",
            stdin_text=stdin_text,
            timeout=step.timeout,
            tee=self.tee,
        )
        self.log(
            f"  - {step.id}: exit {proc.exit_code} in {proc.duration_s}s "
            f"-> {step_dir.relative_to(self.run_dir) if step_dir.is_relative_to(self.run_dir) else step_dir}"
        )

        record = StepRecord(
            id=step.id,
            status="ok",
            group=group,
            iteration=iteration,
            attempt=attempt,
            exit_code=proc.exit_code,
            timed_out=proc.timed_out,
            duration_s=proc.duration_s,
            argv=argv,
            cwd=cwd,
            dir=str(step_dir),
        )
        if proc.timed_out:
            record.status = "timed_out"
            raise StepError(
                f"step '{step.id}' exceeded its {step.timeout}s timeout and was killed "
                f"(logs: {step_dir})"
            )
        if proc.exit_code not in step.expect_exit:
            record.status = "failed"
            raise StepError(
                f"step '{step.id}' exited {proc.exit_code}, expected one of "
                f"{list(step.expect_exit)} (logs: {step_dir})"
            )

        artifacts = Artifacts(
            stdout_path=step_dir / "stdout.log",
            stderr_path=step_dir / "stderr.log",
            workdir=Path(cwd) if cwd else Path.cwd(),
            result_path=Path(result_file) if result_file else None,
        )
        result = (
            load_result(
                step.id, Path(result_file), step.result_schema.get("required", [])
            )
            if result_file
            else None
        )

        values: dict[str, Any] = {
            "exit_code": proc.exit_code,
            "duration_s": proc.duration_s,
            "stdout_path": str(artifacts.stdout_path),
            "stderr_path": str(artifacts.stderr_path),
            "workdir": str(artifacts.workdir),
            "stdout": artifacts.text("stdout"),
            "stderr": artifacts.text("stderr"),
        }
        for spec in step.outputs:
            argument = (
                refs.render(str(spec.argument), resolver)
                if isinstance(spec.argument, str)
                else spec.argument
            )
            values[spec.name] = extract(
                spec, argument, artifacts, result, proc.exit_code
            )

        self.completed[step.id] = values
        record.outputs = {
            key: value
            for key, value in values.items()
            if key not in ("stdout", "stderr")
        }
        (step_dir / "result.json").write_text(
            json.dumps(record.outputs, indent=2, default=str), encoding="utf-8"
        )

        assert_resolver = self._resolver(
            step,
            loop=resolver.loop,
            previous=resolver.previous,
            result_file=result_file,
        )
        for item in step.asserts:
            if not refs.evaluate(item["that"], assert_resolver):
                record.status = "failed"
                message = item["message"] or item["that"]
                raise StepError(f"step '{step.id}': assertion failed - {message}")
        return record

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

    def _write_manifest(self, report: RunReport, started: float) -> None:
        payload = {
            "run_id": report.run_id,
            "flow": self.flow.name,
            "flow_path": str(self.flow.path),
            "tools_path": str(self.registry.path),
            "profile": self.profile,
            "status": report.status,
            "error": report.error,
            "started": datetime.fromtimestamp(started, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(time.time() - started, 3),
            "inputs": self.inputs,
            "vars": self.vars,
            "loops": [vars(loop) for loop in report.loops],
            "steps": [vars(record) for record in report.steps],
        }
        (self.run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
