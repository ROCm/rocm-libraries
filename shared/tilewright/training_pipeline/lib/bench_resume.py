"""Cross-run resume of a bench stage: stage03, and the stages that bench
through it (stage03b, 03c, 03d and stage08's yaml datasets).

A journal beside the stage's outputs (`resume/journal.json`) records, as the
stage goes, what it benches (`inputs`, a digest of its lines and settings,
and `layout`, its units), the GPU model it benches on, each run (one
invocation: the host's boot, its GPUs and the segment each benches) and how
the stage ended; `resume/units/<key>.json` holds the state of each unit.

A run that an interruption or an AC cycle cut off, a journal without an
end, is continued by the next invocation over the same inputs and code:
each unit after its last finished shape, the shape cut off mid-run benched
again. So is a run that failed: stage03 benches again only what failed
(`failed`). A run that passed stands, so the same invocation again benches
nothing. Another code continues a run only on the operator's word, which
the run records with a note (`begin`'s `change`). Outputs a stage does not
continue move aside (`set_aside`); measured data is never deleted.

A segment is the data of one canary baseline. The first run benches segment
1. A later run benches on, on each GPU, the segment of its baseline
(`Journal.begin`: the GPU's own, else for a GPU new to the stage the last
run's), until its canary disagrees with that baseline (stage03
`Canary.boundary`); the GPU then starts a new segment with a baseline of
its own (`Journal.new_segment`). One stage's data holds one GPU model (arch,
chip id and CU count): readings from before and after an interruption and
from several GPUs of the model, never of another.

Every bench process names its GPU (uuid, bdf), the host's boot, its run and
segment, the stage's code and whether the canary times it in a PROCESS
header line of its log (lib/bench_log.py), which stage04 stamps on the rows
it reads behind it. A stage that cannot continue a run cut off mid-way keeps
a journal too and refuses to start over one (`refusal`)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import bench_log

SCHEMA = "bench_resume_v1"
JOURNAL = Path("resume") / "journal.json"
UNITS = Path("resume") / "units"
SUPERSEDED = "superseded"
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
# A test's stand-in for the host's boot (an AC cycle is a new boot).
BOOT_ID_ENV = "TILEWRIGHT_BOOT_ID"
SYSFS_ENV = "TILEWRIGHT_SYSFS_ROOT"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def boot_id() -> str:
    """The host's boot id: a reboot or an AC cycle changes it."""
    given = os.environ.get(BOOT_ID_ENV)
    if given:
        return given
    try:
        return BOOT_ID_FILE.read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def lines_hash(lines: Iterable[Any]) -> str:
    """The digest of the lines a stage benches."""
    return hashlib.sha256("\n".join(str(v) for v in lines).encode()).hexdigest()


def file_digest(path: Path) -> Optional[str]:
    """The sha256 of the file at `path`, None when it cannot be read."""
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def gpu_model(gpu: Any) -> Dict[str, Any]:
    """The model of a KFD GPU (`lib.kfd.Gpu`): its target version (arch),
    device id (chip) and CU count, from its KFD node's properties; values
    sysfs does not give are None."""
    props: Dict[str, int] = {}
    node = getattr(gpu, "node", None)
    root = Path(os.environ.get(SYSFS_ENV) or "/sys")
    path = root / "class/kfd/kfd/topology/nodes" / str(node) / "properties"
    try:
        lines = path.read_text().splitlines() if node is not None else []
    except OSError:
        lines = []
    for line in lines:
        key, _, value = line.partition(" ")
        if value.strip().lstrip("-").isdigit():
            props[key] = int(value)
    simds, per_cu = props.get("simd_count"), props.get("simd_per_cu")
    return {
        "gfx_target_version": props.get("gfx_target_version"),
        "device_id": props.get("device_id"),
        "cu_count": simds // per_cu if simds and per_cu else None,
    }


def gpu_key(identity: Mapping[str, Any], device: int) -> str:
    """What names a GPU across runs: its uuid, else (sysfs gives none) its
    device index."""
    uuid = identity.get("uuid")
    return str(uuid) if uuid else f"device{int(device)}"


def set_aside(out_dir: Path, paths: Iterable[Path]) -> Optional[Path]:
    """Move `paths` (files or directories below `out_dir`) to
    `out_dir/superseded/<utc>/`, keeping their relative paths: the outputs of
    an earlier invocation the stage does not continue from. None when there
    is nothing to move."""
    found = [Path(p) for p in paths if Path(p).exists()]
    if not found:
        return None
    base = Path(out_dir) / SUPERSEDED / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    n = 0
    while base.exists():
        n += 1
        base = base.with_name(f"{base.name.split('.')[0]}.{n}")
    for p in found:
        dest = base / p.relative_to(out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
    return base


def set_aside_all(out_dir: Path) -> Optional[Path]:
    """`set_aside` of everything in `out_dir` but what earlier invocations
    set aside."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    return set_aside(
        out_dir, [p for p in sorted(out_dir.iterdir()) if p.name != SUPERSEDED]
    )


def _write(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


class Journal:
    """A stage's resume journal in `out_dir` (see the module docstring)."""

    def __init__(self, out_dir: Path, data: Dict[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.data = data
        self.lock = threading.Lock()
        self.identities: Dict[int, Dict[str, Any]] = {}
        self.segments: Dict[int, int] = {}

    @classmethod
    def load(cls, out_dir: Path) -> Optional["Journal"]:
        """The journal in `out_dir`, None without a readable one."""
        try:
            data = json.loads((Path(out_dir) / JOURNAL).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None
        return cls(out_dir, data)

    @classmethod
    def new(cls, out_dir: Path, inputs: str, code: Optional[str] = None) -> "Journal":
        """A journal for a stage over `inputs` (their digest) with `code` (the
        digest of the stage's own code)."""
        units = Path(out_dir) / UNITS
        if units.is_dir():
            shutil.rmtree(units)
        data = {
            "schema": SCHEMA,
            "inputs": inputs,
            "code": code,
            "gpu_model": None,
            "runs": [],
            "layout": None,
        }
        return cls(out_dir, {**data, "n_segments": 0, "end": None})

    @property
    def inputs(self) -> Optional[str]:
        return self.data.get("inputs")

    @property
    def layout(self) -> Optional[List[Dict[str, Any]]]:
        return self.data.get("layout")

    @property
    def ended(self) -> bool:
        """Whether the stage's last run ended, passed or failed."""
        return self.data.get("end") is not None

    @property
    def passed(self) -> bool:
        return self.ended and self.data["end"].get("rc") == 0

    @property
    def failed(self) -> bool:
        """Whether the stage's last run ended with a failure."""
        return self.ended and self.data["end"].get("rc") != 0

    @property
    def code(self) -> Optional[str]:
        """The digest of the code of the stage's last run; None in a journal
        from before it was recorded apart from the inputs."""
        return self.data.get("code")

    @property
    def run(self) -> int:
        """The index of the stage's current (last) run, from 1."""
        return len(self.data.get("runs") or [])

    def last_run(self) -> str:
        """The last run in words, for messages."""
        runs = self.data.get("runs") or []
        if not runs:
            return "no run recorded"
        r = runs[-1]
        return f"run {r['run']}, started {r['utc']} on boot {r['boot_id']}"

    def units(self) -> Dict[str, Dict[str, Any]]:
        """{key: state} of every unit recorded."""
        out: Dict[str, Dict[str, Any]] = {}
        for p in sorted((self.out_dir / UNITS).glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
        return out

    def start(self) -> None:
        """A new run of a stage without units or GPU model (`refusal`)."""
        self._open_run({})

    def begin(
        self,
        models: Mapping[int, Mapping[str, Any]],
        identities: Mapping[int, Mapping[str, Any]],
        layout: List[Dict[str, Any]],
        code: Optional[str] = None,
        change: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """A new run on the stage's devices (`models`, `identities` by
        device index) for the units of `layout`, with `code`, the digest of
        the stage's code, and `change`, the code change the operator let the
        run go on across ({from, to, note}); each device benches its GPU's
        segment of an earlier run, else (a GPU new to the stage) the last
        run's highest one, else (the first run) segment 1. ValueError when
        the devices are of several models or of another model than the
        data's."""
        kinds = sorted({json.dumps(dict(m), sort_keys=True) for m in models.values()})
        if len(kinds) > 1:
            raise ValueError(
                f"the stage's GPUs are of {len(kinds)} models ({'; '.join(kinds)}): "
                f"one stage's data holds one GPU model"
            )
        model = json.loads(kinds[0]) if kinds else {}
        then = self.data.get("gpu_model")
        if then is not None and then != model:
            raise ValueError(
                f"the data in {self.out_dir} was benched on {json.dumps(then, sort_keys=True)} "
                f"and the stage's GPUs are {json.dumps(model, sort_keys=True)}: one stage's "
                f"data never mixes GPU models; move the directory aside to start anew"
            )
        self.data["gpu_model"] = model
        self.data["layout"] = layout
        self.identities = {int(d): dict(v) for d, v in identities.items()}
        own = self.gpu_segments()
        last = [int(v["segment"]) for v in self._last_devices().values()]
        if not self.data.get("n_segments"):
            self.data["n_segments"] = 1
        devices: Dict[str, Dict[str, Any]] = {}
        for d, ident in sorted(self.identities.items()):
            key = gpu_key(ident, d)
            seg = own.get(key, max(last) if last else 1)
            self.segments[d] = seg
            devices[str(d)] = {
                **ident,
                "gpu": key,
                "segment": seg,
                "first_segment": seg,
            }
        if code is not None:
            self.data["code"] = code
        extra: Dict[str, Any] = {"code": code} if code is not None else {}
        if change is not None:
            extra["code_change"] = dict(change)
        self._open_run(devices, extra)

    def _open_run(
        self,
        devices: Dict[str, Dict[str, Any]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        runs = self.data.setdefault("runs", [])
        runs.append(
            {
                "run": len(runs) + 1,
                "boot_id": boot_id(),
                "utc": _utc(),
                "devices": devices,
                **(extra or {}),
            }
        )
        self.data["end"] = None
        _write(self.out_dir / JOURNAL, self.data)

    def note_run(self, key: str, value: Any) -> None:
        """Record `value` as `key` of the current run."""
        with self.lock:
            self.data["runs"][-1][key] = value
            _write(self.out_dir / JOURNAL, self.data)

    def archive_unit(self, key: str) -> None:
        """Keep the record of unit `key` out of the units a later run reads,
        in `units/history/`: the unit starts anew."""
        path = self.out_dir / UNITS / f"{key}.json"
        if path.is_file():
            dest = self.out_dir / UNITS / "history" / f"{key}.run{self.run}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dest)

    def _last_devices(self) -> Dict[str, Dict[str, Any]]:
        runs = self.data.get("runs") or []
        return dict(runs[-1].get("devices") or {}) if runs else {}

    def gpu_segments(self) -> Dict[str, int]:
        """{GPU key: the segment it benched last} over the earlier runs."""
        out: Dict[str, int] = {}
        for r in self.data.get("runs") or []:
            for v in (r.get("devices") or {}).values():
                if v.get("gpu") is not None and v.get("segment") is not None:
                    out[str(v["gpu"])] = int(v["segment"])
        return out

    def earlier_gpus(self, segment: int) -> List[str]:
        """The GPUs of the last earlier run that benched `segment` there."""
        runs = self.data.get("runs") or []
        if len(runs) < 2:
            return []
        devices = runs[-2].get("devices") or {}
        return sorted(
            str(v["gpu"])
            for v in devices.values()
            if int(v.get("segment") or 0) == int(segment)
        )

    def segment_of(self, device: Optional[int]) -> Optional[int]:
        """The segment `device` benches in this run."""
        with self.lock:
            return self.segments.get(int(device)) if device is not None else None

    def gpu_of(self, device: int) -> str:
        return gpu_key(self.identities.get(int(device), {}), int(device))

    def new_segment(self, device: int, note: Mapping[str, Any]) -> int:
        """Start a new segment on `device`, whose canary disagreed with its
        baseline (`note`, the verdict); returns its id."""
        with self.lock:
            seg = int(self.data.get("n_segments") or 1) + 1
            self.data["n_segments"] = seg
            self.segments[int(device)] = seg
            entry = self.data["runs"][-1]["devices"][str(int(device))]
            entry.update(segment=seg, canary=dict(note))
            _write(self.out_dir / JOURNAL, self.data)
        return seg

    def note(self, device: int, verdict: Mapping[str, Any]) -> None:
        """Record the canary's verdict on `device`'s baseline in this run."""
        with self.lock:
            self.data["runs"][-1]["devices"][str(int(device))]["canary"] = dict(verdict)
            _write(self.out_dir / JOURNAL, self.data)

    def end(self, rc: int) -> None:
        """The stage's run ended with `rc`: a later invocation does not
        continue it."""
        with self.lock:
            self.data["end"] = {"rc": int(rc), "utc": _utc(), "run": self.run}
            _write(self.out_dir / JOURNAL, self.data)

    def record(self, key: str, state: Mapping[str, Any]) -> None:
        """Record the state of unit `key` (atomically, one file per unit)."""
        with self.lock:
            _write(self.out_dir / UNITS / f"{key}.json", dict(state))

    def header(self, device: Optional[int], canary: bool) -> str:
        """The PROCESS header line (with its line breaks) of a bench process
        on `device` in this run, timed by the canary or not."""
        ident = self.identities.get(int(device), {}) if device is not None else {}
        fields = {
            "gpu_uuid": ident.get("uuid"),
            "gpu_bdf": ident.get("bdf"),
            "boot_id": boot_id(),
            "run": self.run,
            "segment": self.segment_of(device),
            "canary": "on" if canary else "off",
            "code": (self.code or "")[:12] or None,
        }
        return "\n" + bench_log.process_header(fields) + "\n"


def add_code_change_arguments(ap: Any) -> None:
    """--resume-across-code-change and --note of a stage that benches
    through stage03, which hands them to it (`code_change_args`)."""
    ap.add_argument(
        "--resume-across-code-change",
        action="store_true",
        help="with --note: its stage03 runs go on across a change of stage03's "
        "code, keeping the shapes they finished",
    )
    ap.add_argument(
        "--note",
        default=None,
        help="with --resume-across-code-change: why the code change does not "
        "change what the benches measure",
    )


def code_change_error(args: Any) -> Optional[str]:
    """Why the code-change arguments of `args` do not go together, if so."""
    if (
        getattr(args, "resume_across_code_change", False)
        and not (getattr(args, "note", None) or "").strip()
    ):
        return (
            "--resume-across-code-change needs --note: why the code change does "
            "not change what the benches measure"
        )
    return None


def code_change_args(args: Any) -> List[str]:
    """The stage03 options of the code-change arguments of `args`."""
    if not getattr(args, "resume_across_code_change", False):
        return []
    return ["--resume-across-code-change", "--note", str(args.note)]


def refusal(out_dir: Path, why: str) -> Optional[str]:
    """Why a stage that cannot continue a run cut off mid-way (`why`) does
    not start in `out_dir`, where a run of it did not end; None when none
    is there."""
    journal = Journal.load(out_dir)
    if journal is None or journal.ended or not journal.run:
        return None
    return (
        f"{out_dir} holds a run of this stage that did not end ({journal.last_run()}), "
        f"and the stage does not continue one: {why}; move it aside, or pass "
        f"--restart, which moves it to {SUPERSEDED}/"
    )
