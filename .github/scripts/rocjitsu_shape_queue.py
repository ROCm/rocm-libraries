# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Drain a largest-first shape queue with independent serial emulator clients."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from rocjitsu_pytest import descendants, supervise, write_json


def stop_group(proc):
    if proc.poll() is not None:
        return
    for pid in descendants(proc.pid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


class Queue:
    def __init__(self, plan, directory, command, timeout):
        self.directory = directory
        self.command = command
        self.timeout = timeout
        self.lock = threading.Lock()
        self.pending = iter(plan["jobs"])
        self.records = {
            job["nodeid"]: dict(outcome="incomplete", phases={}) for job in plan["jobs"]
        }
        self.junit = ET.Element("testsuites")
        self.active = 0
        self.peak_active = 0
        self.claimed = 0
        for job in plan["excluded"]:
            self.records[job["nodeid"]] = dict(
                outcome="skipped",
                phases={
                    "setup": dict(outcome="skipped", seconds=0, detail=job["reason"]),
                    "teardown": dict(outcome="passed", seconds=0),
                },
            )
            suite = ET.SubElement(
                self.junit, "testsuite", name="size-gate", tests="1", skipped="1"
            )
            case = ET.SubElement(suite, "testcase", name=job["nodeid"], time="0")
            ET.SubElement(case, "skipped", message=job["reason"])
        (directory / "junit").mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self):
        counts = {key: 0 for key in ("passed", "failed", "skipped", "incomplete")}
        for record in self.records.values():
            counts[record["outcome"]] += 1
        write_json(
            self.directory / "progress.json",
            dict(
                selected=len(self.records),
                finished=len(self.records) - counts["incomplete"],
                **counts,
                tests=self.records,
            ),
        )
        destination = self.directory / "junit/tensilelite.xml"
        temporary = destination.with_suffix(".tmp")
        ET.ElementTree(self.junit).write(
            temporary, encoding="utf-8", xml_declaration=True
        )
        temporary.replace(destination)
        write_json(
            self.directory / "queue-state.json",
            dict(active=self.active, peak_active=self.peak_active),
        )

    def claim(self):
        with self.lock:
            job = next(self.pending, None)
            if job is not None:
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                self.records[job["nodeid"]]["started_unix"] = time.time()
                self.records[job["nodeid"]]["queue_position"] = self.claimed
                self.claimed += 1
                self.save()
            return job

    def run_job(self, job, worker):
        directory = self.directory / "jobs" / job["id"]
        directory.mkdir(parents=True)
        jobpath = directory / "job.json"
        write_json(jobpath, job)
        test_command = [
            *self.command,
            "--rocjitsu-shape-job",
            str(jobpath),
            "--rocjitsu-report-dir",
            str(directory),
            "--basetemp",
            str(directory / "tmp"),
            "--junit-xml",
            str(directory / "junit.xml"),
        ]
        # A per-job subreaper also owns helpers that start separate sessions.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--execute-job",
            "--report-dir",
            str(directory),
            "--per-job-seconds",
            str(self.timeout),
            "--",
            *test_command,
        ]
        start = time.time()
        timed_out = False
        with (directory / "test.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                try:
                    status = proc.wait(timeout=self.timeout + 30)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    status = 124
            finally:
                stop_group(proc)
        execution = directory / "job-execution.json"
        if execution.exists():
            timed_out = timed_out or json.loads(execution.read_text())["timed_out"]
        result = dict(
            exit_code=status,
            timed_out=timed_out,
            worker=worker,
            started_unix=start,
            finished_unix=time.time(),
            wall_seconds=time.time() - start,
        )
        progress = directory / "progress.json"
        junit = directory / "junit.xml"
        phases = (
            json.loads(progress.read_text()).get("tests", {})
            if progress.exists()
            else {}
        )
        valid = len(phases) == 1 and all(
            "teardown" in entry.get("phases", {}) for entry in phases.values()
        )
        try:
            suites = list(ET.parse(junit).getroot()) if junit.exists() else []
            cases = [case for suite in suites for case in suite.iter("testcase")]
        except ET.ParseError:
            suites, cases = [], []
        if status == 0 and valid and len(cases) == 1:
            outcome = next(iter(phases.values()))["outcome"]
            if outcome not in ("passed", "skipped"):
                result["exit_code"] = 1
                outcome = "failed"
        else:
            result["exit_code"] = status or 1
            outcome = "failed"
        if outcome == "failed" and not any(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        ):
            suites = [
                ET.Element("testsuite", name="shape-job", tests="1", failures="1")
            ]
            case = ET.SubElement(
                suites[0],
                "testcase",
                name=job["nodeid"],
                time=str(result["wall_seconds"]),
            )
            ET.SubElement(
                case,
                "failure",
                message=f"client job exit {result['exit_code']}; see jobs/{job['id']}/test.log",
            )
        write_json(directory / "result.json", result)
        reported_phases = next(iter(phases.values()))["phases"] if valid else {}
        if outcome == "failed" and not any(
            phase["outcome"] == "failed" for phase in reported_phases.values()
        ):
            detail = (
                f"Timeout (>{self.timeout}s)"
                if timed_out
                else f"job exit {result['exit_code']}; missing or inconsistent pytest results"
            )
            reported_phases["call"] = dict(
                outcome="failed", seconds=result["wall_seconds"], detail=detail
            )
        reported_phases.setdefault("teardown", dict(outcome="passed", seconds=0))
        with self.lock:
            self.active -= 1
            record = dict(
                outcome=outcome,
                queue_position=self.records[job["nodeid"]]["queue_position"],
                phases=reported_phases,
                **result,
            )
            self.records[job["nodeid"]] = record
            self.junit.extend(suites)
            self.save()
            print(
                f"{outcome.upper()} worker={worker} seconds={result['wall_seconds']:.2f} {job['nodeid']}",
                flush=True,
            )
        if outcome != "failed":
            # Keep logs and result records, but avoid uploading successful code objects.
            shutil.rmtree(directory / "tmp", ignore_errors=True)

    def worker(self, worker):
        while (job := self.claim()) is not None:
            self.run_job(job, worker)

    def run(self, workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.worker, worker) for worker in range(workers)]
            for future in futures:
                future.result()
        return int(
            any(
                record["outcome"] in ("failed", "incomplete")
                for record in self.records.values()
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-job-seconds", type=float, default=600)
    parser.add_argument("--execute-job", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or args.workers < 1
        or not math.isfinite(args.per_job_seconds)
        or args.per_job_seconds <= 0
    ):
        parser.error(
            "command, positive workers and finite per-job-seconds are required"
        )
    if args.execute_job:
        result = supervise(command, args.per_job_seconds, grace=5)
        write_json(args.report_dir / "job-execution.json", result)
        return result["exit_code"]
    if args.plan is None:
        parser.error("--plan is required when running the queue")
    plan = json.loads(args.plan.read_text())
    if plan.get("version") != 1 or not plan["jobs"]:
        parser.error("a nonempty version-1 shape plan is required")
    if plan["jobs"] != sorted(
        plan["jobs"], key=lambda job: (-job["work"], job["nodeid"])
    ):
        parser.error("shape jobs must be sorted largest first")
    if len({job["nodeid"] for job in plan["jobs"] + plan["excluded"]}) != len(
        plan["jobs"] + plan["excluded"]
    ):
        parser.error("shape node ids must be unique")
    return Queue(plan, args.report_dir.resolve(), command, args.per_job_seconds).run(
        args.workers
    )


if __name__ == "__main__":
    sys.exit(main())
