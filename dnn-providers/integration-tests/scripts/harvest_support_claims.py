#!/usr/bin/env python3
"""Harvest unclaimed support from CI logs of the bundle integration tests.

RFC 0015: a run of hipdnn_integration_tests prints a SUPPORT CLAIM SUMMARY
block after its tests.  Its unclaimed_support list names graphs that reached
the depth their bundle requires on this lane (arch + platform + engine) but
that no support.json sidecar claims yet.  This script reads CI job logs,
extracts those blocks, and prints the claim cells to add; with --write it
adds them to the sidecars.

Log handling
------------
1. GitHub timestamps and ANSI colours are stripped.  Lines are split into
   streams by their CTest "N: " prefix; unprefixed lines form one more stream,
   which is where CTest reprints the output of a failed test.  The same
   block therefore often appears twice; identical blocks count once.
2. Within a stream, each block is paired with the gtest results printed
   before it.  A block with no "[==========] ... ran." line before it, JSON
   that does not parse, schema_version != 1, a mode other than enforcing or
   warning_only, an invalid run arch, platform or engine,
   counters_consistent != true, or a harness_defects count other than 0 is
   rejected whole.
3. Each unclaimed_support entry becomes one cell per case: (bundle, case,
   arch, platform, engine).  "reached: verified" means the comparison ran,
   not that it passed, so a cell whose gtest FAILED or was SKIPPED in the
   same stream is dropped.  So is a cell whose reached depth is below its
   required depth, or either depth is missing.

Runs
----
4. A run is one CI workflow run.  Each RUN argument is a run read from disk:
   a job log, or a directory of *.log job logs.  Each --run ID is a run
   fetched from GitHub with the gh CLI: the logs of its jobs whose name
   matches --jobs and that succeeded or failed.  With neither, the latest
   completed scheduled run of the develop nightly is fetched.
5. A log with a rejected block contributes nothing, and the exit status is
   1.  A log with no block, and a matching job that neither succeeded nor
   failed, is listed under Missing: its engines are absent from the
   harvest.  A job that timed out, for one, ends before its block.  A cell
   is claimed when it is kept in at least --min-runs runs and dropped in
   none; a single drop anywhere vetoes it.
6. claim_failures and failed_in_use are listed for a human; they are never
   claims.

Writing
-------
7. With --lane, the claims are added to the current sidecars of their
   bundles, and each sidecar that would change is named; --write writes
   them.  --write needs --lane, so one change covers one lane.  Claims are
   only added: a sidecar becomes its old claims plus the new ones.
8. The sidecars must be up to date.  The tip of develop is fetched from
   GitHub with git, and a sidecar that lacks any claim its develop version
   has is refused: rebase onto develop first.
9. Nothing is written if a log was rejected or any sidecar is refused (see
   support_sidecar.py).  verify_support_claims.py checks the result.

Exit status: 0 on success, 1 if any log was rejected or a sidecar was
refused, 2 on a usage error or when a run or develop cannot be fetched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
)

from support_sidecar import Claim, SidecarError, load, parse, render, sidecar_path

VALID_PLATFORMS = {"linux", "windows"}
SCHEMA_VERSION = 1
MODES = ("enforcing", "warning_only")  # modeNames in SupportClaimReport.cpp
BUNDLE_PREFIX = "integration-test-bundles/"

# VerificationDepth in src/harness/bundle/VerificationOutcome.hpp.
DEPTHS = {
    "not-reached": 0,
    "applicable": 1,
    "buildable": 2,
    "executed": 3,
    "verified": 4,
}

DROP_NOT_PASSED = "gtest failed or skipped"
DROP_SHORTFALL = "reached below required"
DROP_NO_DEPTH = "depth missing or unknown"
DROP_BAD_PATH = "bundle outside integration-test-bundles/"
DROP_MIN_RUNS = "kept in fewer than --min-runs runs"

REPO = "ROCm/rocm-libraries"
DEVELOP = (f"https://github.com/{REPO}.git", "develop")
RUN_URL = f"https://github.com/{REPO}/actions/runs"
NIGHTLY_WORKFLOW = "therock-multi-arch-ci-nightly.yml"
DEFAULT_JOBS = r"Test \w+provider\b"  # every provider's engines, on every GPU
FETCHED_CONCLUSIONS = ("success", "failure")
NIGHTLY_PAGE = 50
INTEGRATION_TESTS = Path(__file__).resolve().parent.parent

_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ?")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CTEST_PREFIX = re.compile(r"^(\d+): ")
_SUMMARY_HEADER = re.compile(r"^==== SUPPORT CLAIM SUMMARY \((.+)\) ====$")
_GTEST_NOT_PASSED = re.compile(r"^\[\s+(?:FAILED|SKIPPED)\s+\] ([^\s,]+\.[^\s,]+)")
_GTEST_RAN = re.compile(r"^\[==========\] \d+ tests? from \d+ test suites? ran\.")
_ARCH = re.compile(r"^gfx[0-9a-f]+$")
_LANE = re.compile(r"^gfx[0-9a-f]+/(?:linux|windows)$")


@dataclass
class Block:
    """One SUPPORT CLAIM SUMMARY block and the gtest results printed before it."""

    log: str
    stream: str
    summary: Optional[dict] = None
    not_passed: FrozenSet[str] = frozenset()
    error: Optional[str] = None


@dataclass
class _StreamState:
    not_passed: Set[str] = field(default_factory=set)
    saw_ran_line: bool = False


def split_streams(text: str) -> Dict[str, List[str]]:
    """Group log lines by CTest prefix, in order.  Unprefixed lines go under ""."""
    streams: Dict[str, List[str]] = {}
    for raw in text.splitlines():
        line = _ANSI.sub("", _TIMESTAMP.sub("", raw))
        match = _CTEST_PREFIX.match(line)
        key = match.group(1) if match else ""
        if match:
            line = line[match.end() :]
        streams.setdefault(key, []).append(line)
    return streams


def _validate_summary(parsed: object) -> str | dict:
    """Returns the inner summary object, or a reason to reject the block."""
    if not isinstance(parsed, dict) or not isinstance(
        parsed.get("support_claim_summary"), dict
    ):
        return "no support_claim_summary object"
    summary = parsed["support_claim_summary"]
    if summary.get("schema_version") != SCHEMA_VERSION:
        return (
            f"schema_version {summary.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if summary.get("mode") not in MODES:
        return f"mode {summary.get('mode')!r} is not enforcing or warning_only"
    run = summary.get("run")
    if not isinstance(run, dict):
        return "no run object"
    if not isinstance(run.get("arch"), str) or not _ARCH.match(run["arch"]):
        return f"run.arch {run.get('arch')!r} is not a gfx target"
    if run.get("platform") not in VALID_PLATFORMS:
        return f"run.platform {run.get('platform')!r} is not linux or windows"
    if not isinstance(run.get("engine"), str) or not run["engine"]:
        return f"run.engine {run.get('engine')!r} is empty"
    if summary.get("counters_consistent") is not True:
        return "counters_consistent is not true"
    defects = summary.get("harness_defects")
    if not isinstance(defects, dict) or "missed_query" not in defects:
        return "harness_defects has no missed_query count"
    for name, count in sorted(defects.items()):
        if type(count) is not int or count != 0:
            return f"harness_defects.{name} is {count!r}, not 0"
    return summary


def _parse_block(lines: List[str], start: int) -> str | dict:
    text = "\n".join(lines[start:])
    offset = len(text) - len(text.lstrip())
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, offset)
    except json.JSONDecodeError as exc:
        return f"summary JSON does not parse: {exc.msg} (line {exc.lineno})"
    return _validate_summary(parsed)


def extract_blocks(text: str, log: str = "") -> List[Block]:
    """Every SUPPORT CLAIM SUMMARY block in one job log, in stream order."""
    blocks: List[Block] = []
    for stream, lines in split_streams(text).items():
        state = _StreamState()
        for index, line in enumerate(lines):
            stripped = line.strip()
            match = _GTEST_NOT_PASSED.match(stripped)
            if match:
                state.not_passed.add(match.group(1))
                continue
            if _GTEST_RAN.match(stripped):
                state.saw_ran_line = True
                continue
            if not _SUMMARY_HEADER.match(stripped):
                continue

            block = Block(
                log=log, stream=stream, not_passed=frozenset(state.not_passed)
            )
            if not state.saw_ran_line:
                block.error = "no gtest '[==========] ... ran.' line before the summary"
            else:
                parsed = _parse_block(lines, index + 1)
                if isinstance(parsed, str):
                    block.error = parsed
                else:
                    block.summary = parsed
            blocks.append(block)
            state = _StreamState()
    return blocks


class Cell(NamedTuple):
    """One claim a sidecar writer may add."""

    bundle: str
    case: str
    arch: str
    platform: str
    engine: str

    @property
    def lane(self) -> str:
        return f"{self.arch}/{self.platform}"


@dataclass
class BlockCells:
    kept: Set[Cell] = field(default_factory=set)
    dropped: Dict[Cell, str] = field(default_factory=dict)  # cell -> DROP_* reason
    claim_failures: List[dict] = field(default_factory=list)
    failed_in_use: List[dict] = field(default_factory=list)

    @property
    def drops(self) -> Counter:
        return Counter(self.dropped.values())


class MalformedSummary(ValueError):
    pass


def _sanitize(text: str) -> str:
    """sanitizeForGtest in src/harness/bundle/BundleDiscovery.hpp, byte by byte."""
    return "".join(
        chr(b) if chr(b).isascii() and (chr(b).isalnum() or chr(b) == "_") else "_"
        for b in text.encode("utf-8")
    )


def gtest_name(bundle: str, case: str) -> Optional[str]:
    """The "Suite.Test" name the harness registers for a bundle graph.

    Suite: the bundle's directories under integration-test-bundles/, each
    sanitized, joined with "_".  Test: the sanitized case id for a sweep,
    else the sanitized file stem.  None when the path is not under
    integration-test-bundles/.
    """
    if not bundle.startswith(BUNDLE_PREFIX):
        return None
    parts = PurePosixPath(bundle[len(BUNDLE_PREFIX) :]).parts
    if len(parts) < 2 or any(p in ("..", ".") for p in parts):
        return None
    suite = "_".join(_sanitize(p) for p in parts[:-1])
    test = _sanitize(case) if case else _sanitize(PurePosixPath(parts[-1]).stem)
    return f"{suite}.{test}"


def _lane_of(entry: dict, run: dict) -> Dict[str, str]:
    """An entry's engine/arch/platform: its own when present, else the run's."""
    lane = {}
    for key in ("engine", "arch", "platform"):
        value = entry.get(key, run[key])
        if not isinstance(value, str) or not value:
            raise MalformedSummary(f"entry {key} {value!r} is not a string")
        lane[key] = value
    if not _ARCH.match(lane["arch"]):
        raise MalformedSummary(f"entry arch {lane['arch']!r} is not a gfx target")
    if lane["platform"] not in VALID_PLATFORMS:
        raise MalformedSummary(f"entry platform {lane['platform']!r} is invalid")
    return lane


def _entries(summary: dict, key: str) -> List[dict]:
    entries = summary.get(key)
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("bundle"), str) for e in entries
    ):
        raise MalformedSummary(f"{key} is not a list of entries with a bundle")
    return entries


def _with_lane(entry: dict, run: dict) -> dict:
    return {**entry, **_lane_of(entry, run)}


def block_cells(block: Block) -> BlockCells:
    """Splits a parsed block's unclaimed_support into kept and dropped cells."""
    summary = block.summary
    if summary is None:
        raise ValueError("block_cells needs a parsed block")
    run = summary["run"]
    result = BlockCells()

    for entry in _entries(summary, "unclaimed_support"):
        cases = entry.get("cases", [""])
        if not isinstance(cases, list) or not all(isinstance(c, str) for c in cases):
            raise MalformedSummary(f"cases of {entry['bundle']} is not a string list")
        lane = _lane_of(entry, run)
        reached = DEPTHS.get(entry.get("reached"))
        required = DEPTHS.get(entry.get("required"))

        for case in cases:
            cell = Cell(
                entry["bundle"], case, lane["arch"], lane["platform"], lane["engine"]
            )
            name = gtest_name(cell.bundle, case)
            if name is None:
                reason = DROP_BAD_PATH
            elif reached is None or required is None:
                reason = DROP_NO_DEPTH
            elif reached < required:
                reason = DROP_SHORTFALL
            elif name in block.not_passed:
                reason = DROP_NOT_PASSED
            else:
                result.kept.add(cell)
                continue
            result.dropped.setdefault(cell, reason)

    result.claim_failures = [
        _with_lane(e, run) for e in _entries(summary, "claim_failures")
    ]
    result.failed_in_use = [
        _with_lane(e, run) for e in _entries(summary, "failed_in_use")
    ]
    return result


@dataclass
class LogReport:
    path: str
    blocks: int = 0
    unique: int = 0
    lanes: Set[str] = field(default_factory=set)
    error: Optional[str] = None

    def status(self) -> str:
        if self.error is not None:
            return f"REJECTED: {self.error}"
        if not self.blocks:
            return "no block"
        noun = "block" if self.blocks == 1 else "blocks"
        lanes = " ".join(sorted(self.lanes))
        return f"{self.blocks} {noun} ({self.unique} unique) {lanes}"


@dataclass
class Harvest:
    runs: int
    min_runs: int
    lane: Optional[str]
    logs: List[LogReport]
    claims: List[Cell]
    drops: Dict[str, Counter]  # lane -> DROP_* reason -> distinct cells
    claim_failures: List[dict]
    failed_in_use: List[dict]
    missing: List[dict] = field(default_factory=list)  # see missing_logs


class Job(NamedTuple):
    """A job of a fetched CI run whose name matches --jobs."""

    url: str
    name: str
    conclusion: str


def missing_logs(logs: Sequence[LogReport], jobs: Sequence[Job]) -> List[dict]:
    """Every accepted log with no block, then every job whose log was not read."""
    by_url = {job.url: job for job in jobs}

    def entry(log: str) -> dict:
        job = by_url.get(log)
        if job is None:
            return {"log": log}
        return {"log": log, "conclusion": job.conclusion, "name": job.name}

    missing = [entry(log.path) for log in logs if log.error is None and not log.blocks]
    read = {log.path for log in logs}
    return missing + [entry(job.url) for job in jobs if job.url not in read]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stream_name(stream: str) -> str:
    return f"stream {stream}" if stream else "unprefixed stream"


def read_log(path: str, text: str) -> Tuple[LogReport, List[BlockCells]]:
    """The cells of every block in one log, or none if any block is rejected."""
    report = LogReport(path)
    blocks = extract_blocks(text, path)
    cells: List[BlockCells] = []
    for block in blocks:
        try:
            if block.error is not None:
                raise MalformedSummary(block.error)
            cells.append(block_cells(block))
        except MalformedSummary as exc:
            report.error = f"{_stream_name(block.stream)}: {exc}"
            return report, []
        run = block.summary["run"]
        report.lanes.add(f"{run['arch']}/{run['platform']}")
    report.blocks = len(blocks)
    report.unique = len({_canonical([b.summary, sorted(b.not_passed)]) for b in blocks})
    return report, cells


def _entry_lane(entry: dict) -> str:
    return f"{entry['arch']}/{entry['platform']}"


def harvest(
    runs: Sequence[Sequence[Tuple[str, str]]],
    min_runs: int = 1,
    lane: Optional[str] = None,
) -> Harvest:
    """Merges runs, each a list of (log path, log text), into claims."""
    logs: List[LogReport] = []
    runs_kept: Counter = Counter()  # cell -> runs that kept it
    vetoed: Dict[Cell, str] = {}
    failures: Dict[str, dict] = {}
    in_use: Dict[str, dict] = {}

    for run in runs:
        kept: Set[Cell] = set()
        for path, text in run:
            report, block_list = read_log(path, text)
            logs.append(report)
            for cells in block_list:
                kept |= cells.kept
                for cell, reason in cells.dropped.items():
                    vetoed.setdefault(cell, reason)
                for entry in cells.claim_failures:
                    failures.setdefault(_canonical(entry), entry)
                for entry in cells.failed_in_use:
                    in_use.setdefault(_canonical(entry), entry)
        runs_kept.update(kept)

    def wanted(cell_lane: str) -> bool:
        return lane is None or cell_lane == lane

    drops: Dict[str, Counter] = {}
    for cell, reason in vetoed.items():
        if wanted(cell.lane):
            drops.setdefault(cell.lane, Counter())[reason] += 1

    claims: List[Cell] = []
    for cell, count in runs_kept.items():
        if cell in vetoed or not wanted(cell.lane):
            continue
        if count < min_runs:
            drops.setdefault(cell.lane, Counter())[DROP_MIN_RUNS] += 1
        else:
            claims.append(cell)

    def selected(entries: Dict[str, dict]) -> List[dict]:
        return [entries[k] for k in sorted(entries) if wanted(_entry_lane(entries[k]))]

    return Harvest(
        runs=len(runs),
        min_runs=min_runs,
        lane=lane,
        logs=logs,
        claims=sorted(claims, key=lambda c: (c.lane, c.engine, c.bundle, c.case)),
        drops=drops,
        claim_failures=selected(failures),
        failed_in_use=selected(in_use),
    )


def _cell_json(cell: Cell) -> dict:
    result = cell._asdict()
    if not cell.case:
        del result["case"]
    return result


def _lanes(result: Harvest) -> List[str]:
    """Every lane seen in an accepted log, or only --lane."""
    seen = {lane for log in result.logs for lane in log.lanes}
    seen |= set(result.drops) | {c.lane for c in result.claims}
    return sorted(seen if result.lane is None else seen & {result.lane})


def render_json(result: Harvest) -> str:
    claims_by_lane = Counter(c.lane for c in result.claims)
    lanes = _lanes(result)
    return json.dumps(
        {
            "runs": result.runs,
            "min_runs": result.min_runs,
            "lane": result.lane,
            "logs": [
                {
                    "path": log.path,
                    "blocks": log.blocks,
                    "unique_blocks": log.unique,
                    "lanes": sorted(log.lanes),
                    "error": log.error,
                }
                for log in result.logs
            ],
            "lanes": {
                lane: {
                    "claims": claims_by_lane[lane],
                    "drops": dict(sorted(result.drops.get(lane, {}).items())),
                }
                for lane in lanes
            },
            "missing": result.missing,
            "claims": [_cell_json(c) for c in result.claims],
            "claim_failures": result.claim_failures,
            "failed_in_use": result.failed_in_use,
        },
        indent=2,
    )


def _entry_line(entry: dict) -> str:
    case = f" [{entry['case']}]" if entry.get("case") else ""
    verdict = f" {entry['verdict']}:" if entry.get("verdict") else ""
    return (
        f"  {_entry_lane(entry)} {entry['engine']} {entry['bundle']}{case}"
        f"{verdict} {entry.get('reason', '')}".rstrip()
    )


def render_table(result: Harvest) -> str:
    lane = f"  lane: {result.lane}" if result.lane else ""
    out = [f"runs: {result.runs}  min-runs: {result.min_runs}{lane}", "", "Logs"]
    width = max((len(log.path) for log in result.logs), default=0)
    out += [f"  {log.path:<{width}}  {log.status()}" for log in result.logs]

    out += ["", f"Missing ({len(result.missing)})"]
    out += [
        "  " + "  ".join(m[k] for k in ("log", "conclusion", "name") if k in m)
        for m in result.missing
    ]

    out += ["", "Lanes"]
    claims_by_lane = Counter(c.lane for c in result.claims)
    lanes = _lanes(result)
    if not lanes:
        out.append("  (none)")
    for name in lanes:
        drops = ", ".join(
            f"{n} {reason}" for reason, n in sorted(result.drops.get(name, {}).items())
        )
        out.append(
            f"  {name:<18} {claims_by_lane[name]:>5} claims"
            + (f"; dropped: {drops}" if drops else "")
        )

    out += ["", f"Claims ({len(result.claims)})"]
    out += [
        f"  {c.lane} {c.engine} {c.bundle}" + (f" [{c.case}]" if c.case else "")
        for c in result.claims
    ]
    for title, entries in (
        ("Claim failures", result.claim_failures),
        ("Failed in use", result.failed_in_use),
    ):
        out += ["", f"{title} ({len(entries)})"]
        out += [_entry_line(e) for e in entries]
    return "\n".join(out)


Gh = Callable[[List[str]], bytes]


class FetchError(RuntimeError):
    """A CI run, its job logs or develop could not be fetched."""


def run_gh(args: List[str]) -> bytes:
    """Runs `gh api ARGS`, whose first item is the endpoint, and returns stdout."""
    try:
        done = subprocess.run(["gh", "api", *args], capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise FetchError("the GitHub CLI (gh) is not installed") from exc
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip()
        raise FetchError(f"gh api {args[0]}: {message}")
    return done.stdout


def latest_nightly(gh: Gh) -> int:
    """Returns the ID of the latest completed scheduled nightly run on develop.

    GitHub's branch, event and status filters on this endpoint return stale
    and varying run lists, so none is sent: the newest NIGHTLY_PAGE runs are
    listed and checked here.
    """
    endpoint = (
        f"repos/{REPO}/actions/workflows/{NIGHTLY_WORKFLOW}/runs"
        f"?per_page={NIGHTLY_PAGE}"
    )
    for run in json.loads(gh([endpoint])).get("workflow_runs") or []:
        if (
            run.get("event") == "schedule"
            and run.get("head_branch") == "develop"
            and run.get("status") == "completed"
        ):
            return int(run["id"])
    raise FetchError(
        f"no completed scheduled {NIGHTLY_WORKFLOW} run on develop"
        f" among its newest {NIGHTLY_PAGE} runs"
    )


def fetch_run(
    run_id: int, jobs: Pattern[str], gh: Gh
) -> Tuple[List[Tuple[str, str]], List[Job]]:
    """Returns the jobs of RUN_ID that match JOBS, and (job URL, log text) of
    each one that succeeded or failed.

    A job that neither succeeded nor failed has no complete log and is not
    fetched.
    """
    status = json.loads(gh([f"repos/{REPO}/actions/runs/{run_id}"])).get("status")
    if status != "completed":
        raise FetchError(f"run {run_id} is {status}, not completed")
    listing = gh(
        [
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
            "--paginate",
            "--jq",
            ".jobs[] | {id, name, conclusion}",
        ]
    )
    matched = [
        job
        for job in (json.loads(line) for line in listing.decode().splitlines() if line)
        if jobs.search(job["name"])
    ]
    if not matched:
        raise FetchError(f"run {run_id} has no job matching {jobs.pattern!r}")

    found, logs = [], []
    for job in sorted(matched, key=lambda j: j["id"]):
        url = f"{RUN_URL}/{run_id}/job/{job['id']}"
        found.append(Job(url, job["name"], str(job["conclusion"])))
        if job["conclusion"] in FETCHED_CONCLUSIONS:
            text = gh([f"repos/{REPO}/actions/jobs/{job['id']}/logs"])
            logs.append((url, text.decode("utf-8", "replace")))
    if not logs:
        raise FetchError(f"run {run_id} has no succeeded or failed matching job")
    return logs, found


def _lane_arg(text: str) -> str:
    if not _LANE.match(text):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not arch/platform, e.g. gfx942/linux"
        )
    return text


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return value


def _regex(text: str) -> Pattern[str]:
    try:
        return re.compile(text)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a regex: {exc}")


def _read_run(path: Path) -> List[Tuple[str, str]]:
    files = sorted(path.glob("*.log")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"{path}: no *.log files")
    return [(str(f), f.read_text(encoding="utf-8", errors="replace")) for f in files]


Git = Callable[[List[str]], bytes]


def run_git(args: List[str]) -> bytes:
    """Runs `git ARGS` and returns stdout."""
    try:
        done = subprocess.run(["git", *args], capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise FetchError("git is not installed") from exc
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip()
        raise FetchError(f"git {' '.join(args)}: {message}")
    return done.stdout


def develop_claims(
    paths: Sequence[Path], root: Path, git: Git
) -> Dict[Path, Set[Claim]]:
    """The claims each sidecar under root has on the tip of develop.

    The fetch only sets FETCH_HEAD; no branch or remote-tracking ref moves.
    """
    in_root = ["-C", str(root)]
    git([*in_root, "fetch", "--quiet", "--no-tags", *DEVELOP])
    rev = git([*in_root, "rev-parse", "FETCH_HEAD"]).decode().strip()
    names = {path.relative_to(root).as_posix(): path for path in paths}
    listed = git([*in_root, "ls-tree", "-z", "--name-only", rev, "--", *names])
    claims: Dict[Path, Set[Claim]] = {path: set() for path in paths}
    for name in filter(None, listed.decode("utf-8").split("\0")):
        text = git([*in_root, "cat-file", "blob", f"{rev}:./{name}"])
        claims[names[name]] = parse(names[name], text.decode("utf-8"))
    return claims


class Change(NamedTuple):
    text: str  # the new content of the sidecar
    added: int  # claims it gains


def plan_claims(cells: Sequence[Cell], root: Path, git: Git) -> Dict[Path, Change]:
    """The change adding cells makes to each sidecar under root it changes.

    A sidecar that lacks a claim its develop version has is behind develop,
    and is refused (SidecarError) like a missing bundle or a sidecar that
    support_sidecar.load refuses.
    """
    new: Dict[Path, Set[Claim]] = {}
    for cell in cells:
        if not (root / cell.bundle).is_file():
            raise SidecarError(f"{cell.bundle}: no such bundle under {root}")
        path = root / sidecar_path(cell.bundle)
        claim = Claim(cell.case, cell.engine, cell.arch, cell.platform)
        new.setdefault(path, set()).add(claim)
    if not new:
        return {}
    current = {path: load(path) for path in new}
    for path, claims in sorted(develop_claims(sorted(new), root, git).items()):
        missing = len(claims - current[path])
        if missing:
            raise SidecarError(
                f"{path.relative_to(root)}: lacks {missing} of develop's claims;"
                " rebase onto develop"
            )
    plan = {}
    for path in sorted(new):
        added = new[path] - current[path]
        if added:
            plan[path] = Change(render(path, current[path] | added), len(added))
    return plan


def main(
    argv: Optional[Sequence[str]] = None,
    gh: Gh = run_gh,
    root: Path = INTEGRATION_TESTS,
    git: Git = run_git,
) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs",
        nargs="*",
        metavar="RUN",
        help="one CI run: a job log, or a directory of *.log job logs",
    )
    parser.add_argument(
        "--run",
        dest="run_ids",
        type=_positive_int,
        action="append",
        default=[],
        metavar="ID",
        help="one CI run fetched from GitHub by its run ID (repeatable)",
    )
    parser.add_argument(
        "--jobs",
        type=_regex,
        default=re.compile(DEFAULT_JOBS),
        metavar="REGEX",
        help=f"fetch the jobs whose name matches this (default: {DEFAULT_JOBS!r})",
    )
    parser.add_argument(
        "--min-runs",
        type=_positive_int,
        default=1,
        help="claim a cell only when kept in this many runs (default: 1)",
    )
    parser.add_argument(
        "--lane", type=_lane_arg, help="only this arch/platform, e.g. gfx942/linux"
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--write",
        action="store_true",
        help="add the claims to the sidecars (needs --lane)",
    )
    args = parser.parse_args(argv)

    if args.write and args.lane is None:
        print("error: --write needs --lane", file=sys.stderr)
        return 2
    repeated = sorted(i for i, n in Counter(args.run_ids).items() if n > 1)
    if repeated:
        print(f"error: --run given twice for {repeated}", file=sys.stderr)
        return 2
    count = len(args.runs) + len(args.run_ids) or 1
    if args.min_runs > count:
        print(
            f"error: --min-runs {args.min_runs} exceeds the {count} runs given",
            file=sys.stderr,
        )
        return 2
    try:
        run_ids = args.run_ids
        if not args.runs and not run_ids:
            run_ids = [latest_nightly(gh)]
            print(f"latest nightly: run {run_ids[0]}", file=sys.stderr)
            print(f"  {RUN_URL}/{run_ids[0]}", file=sys.stderr)
        runs = [_read_run(Path(p)) for p in args.runs]
        jobs: List[Job] = []
        for run_id in run_ids:
            logs, found = fetch_run(run_id, args.jobs, gh)
            runs.append(logs)
            jobs += found
    except (OSError, FetchError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = harvest(runs, args.min_runs, args.lane)
    result.missing = missing_logs(result.logs, jobs)
    print(render_json(result) if args.format == "json" else render_table(result))
    if result.missing:
        n = len(result.missing)
        print(
            f"warning: {n} {'log has' if n == 1 else 'logs have'} no support"
            " block; their engines are not harvested (see Missing)",
            file=sys.stderr,
        )
    if any(log.error is not None for log in result.logs):
        if args.write:
            print("error: a log was rejected; wrote nothing", file=sys.stderr)
        return 1
    if args.lane is None:
        return 0
    wrote_nothing = "; wrote nothing" if args.write else ""
    try:
        plan = plan_claims(result.claims, root, git)
    except FetchError as exc:
        print(f"error: {exc}{wrote_nothing}", file=sys.stderr)
        return 2
    except (SidecarError, OSError) as exc:
        print(f"error: {exc}{wrote_nothing}", file=sys.stderr)
        return 1
    verb = "would change"
    if args.write:
        verb = "changed"
        try:
            for path, change in plan.items():
                path.write_bytes(change.text.encode("utf-8"))
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    for path, change in plan.items():
        print(f"{verb} {path.relative_to(root)} (+{change.added})", file=sys.stderr)
    added = sum(change.added for change in plan.values())
    print(f"{len(plan)} sidecars {verb}, +{added} claims", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
