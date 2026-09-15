# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Launching, watching, cancelling and reconciling runs.

The supervisor owns everything about a run that the engine cannot know, and
nothing the engine does know.

**Two writers, two files.** The worker's engine owns `<runDir>/run.json`; the
supervisor owns `<run-root>/.supervisor/<runId>.json`. Neither ever writes the
other's file, and that separation is what makes "cancelled" and "crashed"
expressible at all -- a tree-killed worker leaves a manifest frozen at its last
checkpoint still saying `running`, and only the supervisor knows why.

**The record lives outside the run directory, and that is load-bearing.**
`Engine._create_run_dir` refuses an explicit run directory that exists and is
non-empty, so a state file written inside it before the worker starts would
fail every launch.

**Liveness is never inferred from a recorded pid.** Pids are reused. A run is
known to be live only while this process is waiting on its child; for anything
else -- a record left behind by a previous server session, a run found by
scanning, a run launched by a second client -- the manifest is the only
evidence, and when the manifest still says `running` the answer is `unknown`.

Record fields: `runId, flow, flowPath, runDir, pid, state, exitCode, startedAt,
endedAt, cancelledAt, label, maxIterations, argv, stderr`. `state` is one of
`running`, `exited`, `cancelled`, `failed-to-start`, `detached`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from runner.engine import Engine, _new_run_id
from runner.errors import OrchestratorError
from runner.flow import Flow, bind_inputs, validate_refs
from runner.process import _new_group, kill_tree
from runner.toolreg import ToolRegistry

from . import projections, resources, schema

#: The orchestrator root -- the working directory a worker is spawned in, so
#: that `-m flowmcp.worker` resolves.
ROOT = Path(__file__).resolve().parents[1]

#: Process records live here, one per run, beside the runs rather than inside
#: them. The leading dot keeps the directory out of every run scan.
RECORD_DIR_NAME = ".supervisor"

#: Covers checkpoints that produce no log line: the engine checkpoints at the
#: start and end of a run independently of logging, so a purely log-driven
#: watcher would miss both.
POLL_INTERVAL_S = 2.0

#: Supplementary prose kept per run, for `flow_status`'s log tail. Never parsed
#: for state.
LOG_BUFFER = 500

PROCESS_RUNNING = "running"
PROCESS_EXITED = "exited"
PROCESS_CANCELLED = "cancelled"
PROCESS_DETACHED = "detached"
PROCESS_FAILED_TO_START = "failed-to-start"

#: Statuses the engine only writes when it is finished with a run. A terminal
#: manifest outranks whatever the supervisor last observed about the process.
_TERMINAL_ENGINE = ("ok", "failed")


class SupervisorError(Exception):
    """An operational failure: a bad flow, a missing input, a cap reached."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def merge_status(process_state: str | None, engine_status: str | None) -> str:
    """Reconcile the two writers into the one status a client is told.

    The rules, in the order they are decided:

    * Cancellation is a fact only the supervisor holds, and it wins outright.
    * A run this process is not waiting on -- detached at startup, discovered by
      scan, or launched by another client -- is believed only when its manifest
      is terminal. Otherwise it is `unknown`, never `running`: a live foreign
      run and an abandoned one are indistinguishable, and the recorded pid
      proves nothing.
    * A terminal manifest beats a stale `running` record, which is what a
      server restart and the leave-it-running path both produce.
    * A process that exited leaving no terminal checkpoint crashed; one that
      left no manifest at all failed before it could write one.
    """
    if process_state == PROCESS_CANCELLED:
        return "cancelled"
    if process_state in (None, PROCESS_DETACHED):
        return engine_status if engine_status in _TERMINAL_ENGINE else "unknown"
    if process_state == PROCESS_RUNNING:
        return engine_status if engine_status in _TERMINAL_ENGINE else "running"
    if process_state == PROCESS_FAILED_TO_START:
        return "failed"
    if engine_status is None:
        return "failed"
    if engine_status in _TERMINAL_ENGINE:
        return engine_status
    return "crashed"


@dataclass
class _Run:
    """One run this process launched and is waiting on."""

    run_id: str
    run_dir: Path
    flow: Any
    max_iterations: int | None
    process: subprocess.Popen
    record: dict[str, Any]
    #: Step children relayed by the worker while they were alive. The POSIX
    #: cancellation path needs them: `launch()` puts every agent in its own
    #: session, so `killpg` on the worker's group does not reach them.
    agent_pids: list[int] = field(default_factory=list)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=LOG_BUFFER))
    manifest_stat: tuple[int, int] | None = None
    progress_seen: tuple[Any, ...] | None = None
    stderr_text: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    threads: list[threading.Thread] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        *,
        tools_path: Path | str,
        flows_dir: Path | str,
        run_root: Path | str,
        profile: str | None = None,
        max_concurrent: int = 2,
        max_iterations_ceiling: int | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
        python: str | None = None,
        cwd: Path | str | None = None,
        worker_module: str = "flowmcp.worker",
        poll_interval_s: float = POLL_INTERVAL_S,
    ) -> None:
        self.tools_path = Path(tools_path).resolve()
        self.flows_dir = Path(flows_dir).resolve()
        self.run_root = Path(run_root).resolve()
        self.record_dir = self.run_root / RECORD_DIR_NAME
        self.profile = profile
        self.max_concurrent = max_concurrent
        self.max_iterations_ceiling = max_iterations_ceiling
        self.python = python or sys.executable
        self.cwd = Path(cwd) if cwd else ROOT
        #: The child entry point, resolved against `cwd`. Named rather than
        #: hardcoded so a caller can stand a different one in its place.
        self.worker_module = worker_module
        self.poll_interval_s = poll_interval_s
        self._emit_event = emit
        self._runs: dict[str, _Run] = {}
        self._subscriptions: set[str] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._poller: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def set_emitter(self, emit: Callable[[dict[str, Any]], None] | None) -> None:
        """Where observed events go. The adapter needs a live session to have
        one, and it only has one after this object exists."""
        self._emit_event = emit

    def start(self) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.reconcile()
        self._poller = threading.Thread(
            target=self._poll, name="flowmcp-poll", daemon=True
        )
        self._poller.start()

    def close(self) -> None:
        self._stop.set()
        if self._poller is not None:
            self._poller.join(timeout=self.poll_interval_s * 2)
        for run in list(self._runs.values()):
            for thread in run.threads:
                thread.join(timeout=1)

    def reconcile(self) -> list[str]:
        """Detach every record left `running` by a previous server session.

        Such a record belongs to a worker this process is not waiting on, so the
        one honest thing to say about it is that we are no longer watching. The
        recorded pid is not probed: it may belong to something else entirely by
        now.
        """
        self.record_dir.mkdir(parents=True, exist_ok=True)
        detached: list[str] = []
        for path in sorted(self.record_dir.glob("*.json")):
            record = resources.read_json(path)
            if not record or record.get("state") != PROCESS_RUNNING:
                continue
            record["state"] = PROCESS_DETACHED
            self._write_record(record)
            detached.append(str(record.get("runId")))
        return detached

    # -- subscriptions ------------------------------------------------------

    def subscribe(self, uri: str) -> str:
        """Register interest in a run's manifest. The only subscribable URI.

        Every other artifact is written once and does not change under a reader,
        so a subscription to one would be a promise of updates that never come.
        """
        if not schema.is_manifest_uri(uri):
            raise SupervisorError(
                f"{uri!r} is not subscribable; subscribe to a run's "
                f"{schema.MANIFEST_NAME}"
            )
        self._subscriptions.add(str(uri))
        return str(uri)

    def unsubscribe(self, uri: str) -> str:
        self._subscriptions.discard(str(uri))
        return str(uri)

    def is_subscribed(self, uri: str) -> bool:
        return str(uri) in self._subscriptions

    # -- flows --------------------------------------------------------------

    def list_flows(self) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for path in resources.flow_files(self.flows_dir):
            try:
                summaries.append(projections.flow_summary_for(Flow.load(path)))
            except OrchestratorError as failure:
                summaries.append(
                    projections.unloadable_flow_summary(path, str(failure))
                )
        return schema.flow_list_result(summaries)

    def flow_inputs(self, flow: str) -> dict[str, Any]:
        path = resources.resolve_flow(flow, self.flows_dir)
        loaded = Flow.load(path)
        return schema.flow_inputs_result(
            flow=loaded.name,
            path=str(path),
            inputs=projections.input_specs_for(loaded),
        )

    def validate(
        self,
        flow: str,
        inputs: Mapping[str, Any] | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        path = resources.resolve_flow(flow, self.flows_dir)
        errors: list[str] = []
        missing: list[str] = []
        resolved: dict[str, str] = {}
        loaded = None
        registry = None
        try:
            loaded = Flow.load(path)
            registry = ToolRegistry.load(self.tools_path, profile or self.profile)
            validate_refs(loaded, registry.vars)
            missing = sorted(loaded.tools_used() - set(registry.tools))
        except OrchestratorError as failure:
            errors.append(str(failure))
        if loaded is not None and registry is not None and not errors and not missing:
            if inputs is not None:
                try:
                    bound = self._bind(loaded, inputs)
                    resolved = Engine(
                        loaded,
                        registry,
                        bound,
                        run_root=self.run_root,
                        profile=profile or self.profile,
                    ).preflight()
                except OrchestratorError as failure:
                    errors.append(str(failure))
        return schema.flow_validate_result(
            ok=not errors and not missing,
            flow=loaded.name if loaded is not None else path.stem,
            step_count=len(list(loaded.all_steps())) if loaded is not None else 0,
            input_count=len(loaded.inputs) if loaded is not None else 0,
            tools=sorted(loaded.tools_used()) if loaded is not None else [],
            missing_tools=missing,
            resolved_tools=resolved,
            errors=errors,
        )

    # -- launching ----------------------------------------------------------

    def _bind(self, flow: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
        pairs = [
            f"{name}={value}" for name, value in inputs.items() if value is not None
        ]
        return bind_inputs(flow, pairs)

    def _clamp(
        self, flow: Any, requested: int | None
    ) -> tuple[int | None, list[str]]:
        """Bound a requested iteration budget by what the flow itself declares.

        The engine treats `max_iterations` as an unconditional override in both
        directions, so forwarding a request raw would let a caller *raise* a
        flow's budget -- the opposite of a cost guardrail. The bound is the
        flow's own largest declared budget; an operator's ceiling is the only
        thing that can lift it.
        """
        declared = projections.declared_budget(flow)
        if requested is None:
            return None, []
        if declared is None:
            raise SupervisorError(
                f"{flow.name} declares no loop; maxIterations does not apply"
            )
        bound = declared
        if self.max_iterations_ceiling is not None:
            bound = max(declared, self.max_iterations_ceiling)
        if requested > bound:
            return bound, [
                f"maxIterations {requested} lowered to {bound}, the largest "
                f"budget available for {flow.name}"
            ]
        return requested, []

    def _live_count(self) -> int:
        return sum(
            1 for run in self._runs.values() if run.process.poll() is None
        )

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def launch(
        self,
        *,
        flow: str,
        inputs: Mapping[str, Any],
        max_iterations: int | None = None,
        profile: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        path = resources.resolve_flow(flow, self.flows_dir)
        resolved_profile = profile or self.profile
        try:
            loaded = Flow.load(path)
            registry = ToolRegistry.load(self.tools_path, resolved_profile)
            validate_refs(loaded, registry.vars)
            missing = sorted(loaded.tools_used() - set(registry.tools))
            if missing:
                raise SupervisorError(
                    f"{loaded.path.name}: uses tool(s) the registry does not "
                    f"declare: {', '.join(missing)}"
                )
            bound = self._bind(loaded, inputs)
        except OrchestratorError as failure:
            raise SupervisorError(str(failure)) from None

        applied, warnings = self._clamp(loaded, max_iterations)
        with self._lock:
            live = self._live_count()
            if live >= self.max_concurrent:
                raise SupervisorError(
                    f"Concurrency cap reached ({live} running). Cancel a run or "
                    f"raise --max-concurrent."
                )

        run_id = _new_run_id()
        run_dir = self.run_root / loaded.name / run_id
        try:
            Engine(
                loaded,
                registry,
                bound,
                run_dir=run_dir,
                run_id=run_id,
                max_iterations=applied,
                profile=resolved_profile,
            ).preflight()
        except OrchestratorError as failure:
            raise SupervisorError(str(failure)) from None

        argv = [self.python, "-m", self.worker_module]
        started_at = _now()
        record = {
            "runId": run_id,
            "flow": loaded.name,
            "flowPath": str(loaded.path),
            "runDir": str(run_dir),
            "pid": None,
            "state": PROCESS_RUNNING,
            "exitCode": None,
            "startedAt": started_at,
            "endedAt": None,
            "cancelledAt": None,
            "label": label,
            "maxIterations": applied,
            "argv": argv,
            "stderr": None,
        }
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(self.cwd),
                env=self._env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                # The worker is spawned as a group leader so POSIX `killpg`
                # reaches it. That, and nothing more, is what this buys: every
                # agent the worker starts gets its own session.
                **_new_group(),
            )
        except OSError as failure:
            record["state"] = PROCESS_FAILED_TO_START
            record["endedAt"] = _now()
            record["stderr"] = str(failure)
            self._write_record(record)
            raise SupervisorError(f"could not start a worker: {failure}") from None

        record["pid"] = process.pid
        self._write_record(record)

        spec = {
            "flow": str(loaded.path),
            "toolsPath": str(self.tools_path),
            "profile": resolved_profile,
            "inputs": bound,
            "runId": run_id,
            "runDir": str(run_dir),
            "maxIterations": applied,
        }
        assert process.stdin is not None
        process.stdin.write(json.dumps(spec, default=str))
        process.stdin.close()

        run = _Run(
            run_id=run_id,
            run_dir=run_dir,
            flow=loaded,
            max_iterations=applied,
            process=process,
            record=record,
        )
        with self._lock:
            self._runs[run_id] = run
        run.threads = [
            threading.Thread(
                target=self._pump_stderr, args=(run,), daemon=True,
                name=f"flowmcp-err-{run_id}",
            ),
            threading.Thread(
                target=self._pump_frames, args=(run,), daemon=True,
                name=f"flowmcp-out-{run_id}",
            ),
        ]
        for thread in run.threads:
            thread.start()
        self._emit({"type": "resource_list_changed"})
        return schema.flow_launch_result(
            run_id=run_id,
            run_dir=str(run_dir),
            flow=loaded.name,
            status="running",
            pid=process.pid,
            started_at=started_at,
            max_iterations=(
                applied if applied is not None else projections.declared_budget(loaded)
            ),
            warnings=warnings,
        )

    # -- watching -----------------------------------------------------------

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._emit_event is not None:
            self._emit_event(dict(event))

    def _pump_stderr(self, run: _Run) -> None:
        stream = run.process.stderr
        if stream is None:
            return
        try:
            run.stderr_text = stream.read() or ""
        except (OSError, ValueError):
            run.stderr_text = ""

    def _pump_frames(self, run: _Run) -> None:
        """Consume the worker's private channel until it closes, then settle up."""
        stream = run.process.stdout
        if stream is not None:
            while True:
                try:
                    line = stream.readline()
                except (OSError, ValueError):
                    break
                if not line:
                    break
                self._on_frame(run, line)
        self._finalize(run)

    def _on_frame(self, run: _Run, line: str) -> None:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(frame, dict):
            return
        kind = frame.get("t")
        if kind == "pid":
            pid = frame.get("pid")
            if isinstance(pid, int):
                with run.lock:
                    run.agent_pids.append(pid)
        elif kind == "log":
            text = str(frame.get("text", ""))
            run.log_lines.append(text)
            self._emit(
                {
                    "type": "message",
                    "level": "info",
                    "logger": f"flow/{run.run_id}",
                    "data": text,
                }
            )
            # A log line is not state, but it is a reliable cue that state may
            # have moved: the engine logs around every step transition.
            self._observe(run)

    def _finalize(self, run: _Run) -> None:
        for thread in run.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)
        code = run.process.wait()
        with run.lock:
            record = run.record
            if record.get("state") != PROCESS_CANCELLED:
                record["state"] = PROCESS_EXITED
            record["exitCode"] = code
            record["endedAt"] = _now()
            if run.stderr_text.strip():
                record["stderr"] = run.stderr_text.strip()[-4000:]
            self._write_record(record)
        self._observe(run, force=True)
        self._emit({"type": "resource_list_changed"})

    def _observe(self, run: _Run, force: bool = False) -> None:
        """Emit an update when the manifest moved -- and only to subscribers."""
        manifest = run.run_dir / schema.MANIFEST_NAME
        try:
            stat = manifest.stat()
            key: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if key == run.manifest_stat and not force:
            return
        run.manifest_stat = key
        if key is None:
            return
        uri = schema.manifest_uri(run.run_id)
        if self.is_subscribed(uri):
            self._emit({"type": "resource_updated", "uri": uri})
        self._emit_progress(run, manifest)

    def _emit_progress(self, run: _Run, manifest: Path) -> None:
        run_json = resources.read_json(manifest)
        if run_json is None:
            return
        progress, total = projections.progress_for(
            run_json, run.flow, max_iterations=run.max_iterations
        )
        current = projections.current_step_for(
            run_json, run.flow, max_iterations=run.max_iterations
        )
        marker = (progress, total, None if current is None else tuple(current.values()))
        if marker == run.progress_seen:
            return
        run.progress_seen = marker
        self._emit(
            {
                "type": "progress",
                "token": run.run_id,
                "progress": progress,
                "total": total,
                "message": self._progress_message(run, current, progress, total),
            }
        )

    def _progress_message(
        self,
        run: _Run,
        current: Mapping[str, Any] | None,
        progress: int,
        total: int | None,
    ) -> str:
        """Prose built entirely from the run's own values."""
        if current is None:
            done = f"{progress} of {total}" if total is not None else str(progress)
            return f"{done} step(s) settled"
        group = current.get("group")
        if group is None:
            return f"{current['id']} running"
        budget = next(
            (
                projections.group_budget(node, run.max_iterations)
                for node in projections.loop_groups(run.flow)
                if node.id == group
            ),
            None,
        )
        iteration = int(current.get("iteration") or 0) + 1
        span = f"{iteration}/{budget}" if budget else str(iteration)
        return f"{group} iteration {span} — {current['id']} running"

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            for run in list(self._runs.values()):
                if run.process.poll() is None:
                    self._observe(run)

    # -- cancellation -------------------------------------------------------

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None or run.process.poll() is not None:
            state = self._merged_for(run_id)
            return schema.flow_cancel_result(
                run_id=run_id,
                cancelled=False,
                state=state,
                killed_pid=None,
                message=(
                    f"Run {run_id} is not running here; nothing was killed "
                    f"(state: {state})."
                ),
            )
        process = run.process
        pid = process.pid
        # Recorded before the kill: the reader thread settles the record the
        # moment the pipe closes, and a cancellation observed as a plain exit
        # is exactly the distinction this record exists to preserve.
        with run.lock:
            run.record["state"] = PROCESS_CANCELLED
            run.record["cancelledAt"] = _now()
            self._write_record(run.record)

        kill_tree(process)
        if os.name != "nt":
            # `launch()` gives every agent its own session, so the worker's
            # process group does not contain them. On Windows `taskkill /T`
            # walks parent-pid links and has already reached them.
            with run.lock:
                relayed = list(run.agent_pids)
            for agent in relayed:
                _terminate(agent)
        alive = not _await_exit(process)
        with run.lock:
            run.record["exitCode"] = process.poll()
            self._write_record(run.record)
        return schema.flow_cancel_result(
            run_id=run_id,
            cancelled=not alive,
            state="cancelled",
            killed_pid=pid,
            message=(
                f"Run cancelled; the last {schema.MANIFEST_NAME} checkpoint is "
                f"preserved."
                if not alive
                else f"Kill signalled but worker {pid} is still alive."
            ),
        )

    # -- status -------------------------------------------------------------

    def _record_path(self, run_id: str) -> Path:
        return self.record_dir / f"{run_id}.json"

    def _write_record(self, record: Mapping[str, Any]) -> None:
        """Same atomic discipline as the manifest: write beside, then replace."""
        self.record_dir.mkdir(parents=True, exist_ok=True)
        path = self._record_path(str(record["runId"]))
        temporary = path.parent / (path.name + ".tmp")
        temporary.write_text(
            json.dumps(dict(record), indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)

    def read_record(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        if run is not None:
            with run.lock:
                return dict(run.record)
        return resources.read_json(self._record_path(run_id))

    def _run_dir_for(self, run_id: str, record: Mapping[str, Any] | None) -> Path | None:
        if record and record.get("runDir"):
            candidate = Path(str(record["runDir"]))
            if candidate.is_dir():
                return candidate.resolve()
        return resources.find_run_dir(self.run_root, run_id)

    def _merged_for(self, run_id: str) -> str:
        record = self.read_record(run_id)
        run_dir = self._run_dir_for(run_id, record)
        if record is None and run_dir is None:
            return "unknown"
        run_json = (
            resources.read_json(run_dir / schema.MANIFEST_NAME) if run_dir else None
        )
        return merge_status(
            record.get("state") if record else None,
            run_json.get("status") if run_json else None,
        )

    def _flow_for(self, run_id: str, run_json: Mapping[str, Any] | None) -> Any:
        run = self._runs.get(run_id)
        if run is not None:
            return run.flow
        if not run_json or not run_json.get("flow_path"):
            return None
        try:
            return Flow.load(str(run_json["flow_path"]))
        except (OrchestratorError, OSError):
            return None

    def status(self, run_id: str, log_tail: int = 50) -> dict[str, Any]:
        record = self.read_record(run_id)
        run_dir = self._run_dir_for(run_id, record)
        if record is None and run_dir is None:
            raise SupervisorError(f"unknown run {run_id!r}")
        run_json = (
            resources.read_json(run_dir / schema.MANIFEST_NAME) if run_dir else None
        )
        engine_status = run_json.get("status") if run_json else None
        process_state = record.get("state") if record else None
        status = merge_status(process_state, engine_status)
        flow = self._flow_for(run_id, run_json)
        run = self._runs.get(run_id)
        applied = record.get("maxIterations") if record else None

        error = run_json.get("error") if run_json else None
        if error is None and status == "failed" and record:
            # No manifest means the failure happened before the first
            # checkpoint; the worker's stderr is the only account of it.
            error = record.get("stderr")

        artifacts: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        loops: list[dict[str, Any]] = []
        current = None
        if run_json is not None and run_dir is not None:
            steps = projections.run_steps_for(run_json, flow, run_dir)
            loops = projections.run_loops_for(run_json)
            current = projections.current_step_for(
                run_json, flow, max_iterations=applied
            )
            artifacts = projections.artifacts_for(
                run_json, flow, run_id=run_id, run_dir=run_dir
            )

        tail: list[str] = []
        if run is not None and log_tail:
            tail = list(run.log_lines)[-log_tail:]

        return schema.flow_status_result(
            run_id=run_id,
            run_dir=str(run_dir) if run_dir else str(record.get("runDir", "")),
            flow=str((run_json or {}).get("flow") or (record or {}).get("flow") or ""),
            flow_path=str(
                (run_json or {}).get("flow_path") or (record or {}).get("flowPath") or ""
            ),
            status=status,
            engine_status=engine_status,
            process_state=process_state,
            error=error,
            started_at=(run_json or {}).get("started")
            or (record or {}).get("startedAt"),
            duration_s=float((run_json or {}).get("duration_s") or 0.0),
            exit_code=record.get("exitCode") if record else None,
            provenance=(run_json or {}).get("provenance") or {},
            inputs=(run_json or {}).get("inputs") or {},
            vars=(run_json or {}).get("vars") or {},
            loops=loops,
            steps=steps,
            current_step=current,
            artifacts=artifacts,
            log_tail=tail,
        )

    # -- resources ----------------------------------------------------------

    def listed_runs(self) -> dict[str, tuple[Path, str | None]]:
        """Runs this session launched, plus the newest found under the run root."""
        listed: dict[str, tuple[Path, str | None]] = {}
        for run_id, run in self._runs.items():
            listed[run_id] = (run.run_dir, run.flow.name)
        for path in resources.newest_run_dirs(self.run_root):
            listed.setdefault(path.name, (path, path.parent.name))
        return listed

    def list_resources(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_id, (run_dir, flow_name) in self.listed_runs().items():
            entries.extend(resources.resource_entries(run_id, run_dir, flow_name))
        return entries

    def read_resource(self, uri: str) -> dict[str, Any]:
        try:
            run_id, relative = schema.parse_run_uri(uri)
        except ValueError as failure:
            raise resources.PathRefused(str(failure)) from None
        run_dir = self._run_dir_for(run_id, self.read_record(run_id))
        if run_dir is None:
            raise resources.RunNotFound(f"no run directory for {run_id!r}")
        path = resources.resolve_in_run(run_dir, relative)
        if not path.is_file():
            raise resources.RunNotFound(f"{uri} does not exist")
        return {
            "uri": schema.format_run_uri(
                run_id, path.relative_to(run_dir).as_posix()
            ),
            "mimeType": resources.mime_for(path),
            "text": resources.read_text(path),
        }


def _await_exit(process: subprocess.Popen, timeout: float = 10.0) -> bool:
    """Wait for a process to actually be gone. Death is verified, not assumed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    try:
        process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    return process.poll() is not None


def _terminate(pid: int, timeout: float = 5.0) -> bool:
    """Signal one relayed descendant and confirm it is gone."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return True
            time.sleep(0.05)
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False
