# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage06: deploy a trained model into a rocm-libraries checkout.

Packs `<train-dir>/models.pt` into one model file (MLREC_v2, or MLREC_v3 for
a bundle trained with the v3 catalog) that carries the model constants of the
config's arch (`lib.hardware.arch_constants`) and stages it
under `<out-dir>/staged_for_deploy/`. With --apply-patch the file is installed
into `shared/tilewright/weights/hipblaslt/<t>/<t>/` for every deploy target
`t` (`deploy.targets`, default: the config's arch) and the library stem is
registered in that directory's `tilewright_index`. With --build-after-apply
the hipBLASLt build target that copies the models into the build
(`hipblaslt-tilewright-models`) then places them next to the Tensile library,
where hipBLASLt loads them from, and the copies are verified.

Installing is all-or-nothing: every file is written under a temporary name,
synced and renamed into place, indexes last; a failure at any point,
including a failed build or a copy the build placed wrongly, restores the
previous files in the checkout and the index and model copies in the build
tree. A build without a Tensile library directory for any deploy target is
recorded as built but unverified (`colocation: "unverified"`).

The model is loaded back with the tilewright Python module before anything is
staged; with --apply-patch the module is required.

Deploy gate (--gate-csv-dir): the model file, exactly the bytes that would be
installed, is evaluated (lib/evaluate.py) on enriched GEMMs its bundle never
trained on (stage04b's gate/ of the round). With --gate-target T it is
installed only when its selection efficiency there reaches T; with
--gate-fallback not-worse a model below T is still installed when no
--gate-baseline-train-dir is given (nothing deployed yet), or when it is at
least as good as that baseline bundle on the same gate GEMMs. With
--rebench-misses every miss of the model (and of the baseline) is re-benched
on the GPUs and charged by its measured ratio to the winner
(lib/miss_rebench.py); the decision uses those numbers and gate.json keeps
the configured rule's as `pre_rebench`. A model the
gate keeps out is still staged; `deploy_record.json` says `deployed: false`
and why. Origami's numbers on the gate GEMMs are recorded and printed only
with --origami-reference.

Outputs (run directory only):
    staged_for_deploy/   the files to install, per-target index fragments and
                         INSTALL.md
    manifest.json        cells, splits and file hash of the model
    summary.txt          the split tree
    gate.json            with --gate-csv-dir: the model's own numbers on the
                         gate GEMMs, overall and per cell, and the baseline's
    deploy_record.json   what was written where; stage07 checks the runtime
                         against it
    build.log            build output, one section per step
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import yaml  # noqa: E402

from lib import evaluate as ev  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import grid  # noqa: E402
from lib import hardware  # noqa: E402
from lib import miss_rebench as mr  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from run_orchestrator import (  # noqa: E402
    FEATURE_DIMS,
    ConfigError,
    cfg_get,
    deploy_targets,
)

WEIGHTS_REL = Path("shared") / "tilewright" / "weights" / "hipblaslt"
INDEX_NAME = "tilewright_index"
INDEX_HEADER = "# tilewright per-library weights index: <logic_stem>\\t<weights.bin>"
BIN_SUFFIX = ".tilewright.bin"
MODELS_TARGET = "hipblaslt-tilewright-models"


class DeployError(Exception):
    pass


# ── tilewright_index ────────────────────────────────────────────────────────


@dataclass
class IndexFile:
    comments: List[str]
    entries: List[Tuple[str, str]]
    dropped: List[str]

    def as_dict(self) -> Dict[str, str]:
        return dict(self.entries)


def parse_index(text: str) -> IndexFile:
    """Parse a tilewright_index the way the engine does: a '#' starts a
    comment, the first two whitespace-separated tokens are stem and file,
    and the first line naming a stem is the one that counts. Later
    duplicates and one-token lines are returned in `dropped`."""
    comments: List[str] = []
    entries: List[Tuple[str, str]] = []
    dropped: List[str] = []
    seen = set()
    for raw in text.splitlines():
        line = raw[:-1] if raw.endswith("\r") else raw
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            if line.strip() and line.rstrip() not in comments:
                comments.append(line.rstrip())
            continue
        if len(tokens) < 2 or tokens[0] in seen:
            dropped.append(line)
            continue
        seen.add(tokens[0])
        entries.append((tokens[0], tokens[1]))
    return IndexFile(comments, entries, dropped)


def render_index(comments: List[str], entries: Mapping[str, str]) -> str:
    lines = list(comments) or [INDEX_HEADER]
    lines += [f"{stem}\t{entries[stem]}" for stem in sorted(entries)]
    return "\n".join(lines) + "\n"


def _plain_file_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name[0] != "."


def choose_weights_name(entries: Mapping[str, str], stem: str) -> str:
    """Keep the file name the stem already maps to unless another stem
    shares that file; otherwise `<stem>.tilewright.bin`."""
    old = entries.get(stem)
    if (
        old
        and _plain_file_name(old)
        and old.endswith(BIN_SUFFIX)
        and sum(1 for b in entries.values() if b == old) == 1
    ):
        return old
    name = stem + BIN_SUFFIX
    if any(b == name for s, b in entries.items() if s != stem):
        raise DeployError(f"{name} is already used by another stem in the index")
    return name


# ── atomic file installation ────────────────────────────────────────────────


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp(directory: Path, name: str, data: bytes) -> Path:
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o666 & ~_umask())
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(_write_temp(path.parent, path.name, data), path)
    _fsync_dir(path.parent)


class Transaction:
    """File replacements and removals that can be undone as a whole."""

    def __init__(self, spill_dir: Path) -> None:
        self.spill_dir = spill_dir
        self._undo: List[Callable[[], None]] = []
        self._backups: List[Path] = []
        self._created_dirs: List[Path] = []
        self._touched_dirs: set = set()

    def ensure_dir(self, path: Path) -> None:
        missing = []
        p = path
        while not p.exists():
            missing.append(p)
            p = p.parent
        for d in reversed(missing):
            d.mkdir()
            self._created_dirs.append(d)

    def _backup(self, dst: Path) -> Optional[Path]:
        if not dst.exists():
            return None
        bak = dst.parent / f".{dst.name}.{os.getpid()}.bak"
        try:
            bak.unlink(missing_ok=True)
            os.link(dst, bak)
        except OSError:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            bak = self.spill_dir / f"{len(self._backups)}_{dst.name}"
            shutil.copy2(dst, bak)
        self._backups.append(bak)
        return bak

    def _restore(self, dst: Path, bak: Optional[Path]) -> None:
        if bak is None:
            dst.unlink(missing_ok=True)
        elif bak.parent == dst.parent:
            os.replace(bak, dst)
        else:
            os.replace(_write_temp(dst.parent, dst.name, bak.read_bytes()), dst)
        self._touched_dirs.add(dst.parent)

    def put(self, dst: Path, data: bytes) -> None:
        self.ensure_dir(dst.parent)
        tmp = _write_temp(dst.parent, dst.name, data)
        try:
            bak = self._backup(dst)
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self._touched_dirs.add(dst.parent)
        self._undo.append(lambda: self._restore(dst, bak))

    def remove(self, dst: Path) -> None:
        bak = self._backup(dst)
        if bak is None:
            return
        dst.unlink()
        self._touched_dirs.add(dst.parent)
        self._undo.append(lambda: self._restore(dst, bak))

    def guard(self, dst: Path) -> None:
        """Make a rollback restore `dst` as it is now (remove it when absent),
        whatever else writes it in between. The backup is a copy: another
        writer may rewrite the file in place."""
        bak: Optional[Path] = None
        if dst.exists():
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            bak = self.spill_dir / f"{len(self._backups)}_{dst.name}"
            shutil.copy2(dst, bak)
            self._backups.append(bak)
        self._touched_dirs.add(dst.parent)
        self._undo.append(lambda: self._restore(dst, bak))

    def sync(self) -> None:
        for d in sorted(self._touched_dirs):
            if d.exists():
                _fsync_dir(d)

    def _discard_backups(self) -> None:
        for b in self._backups:
            b.unlink(missing_ok=True)
        self._backups.clear()

    def commit(self) -> None:
        self.sync()
        self._undo.clear()
        self._created_dirs.clear()
        self._discard_backups()

    def rollback(self) -> None:
        for fn in reversed(self._undo):
            try:
                fn()
            except OSError as e:
                ui.err("rollback", f"could not restore a file: {e!r}")
        self._undo.clear()
        for d in reversed(self._created_dirs):
            try:
                d.rmdir()
            except OSError:
                pass
        self._created_dirs.clear()
        self.sync()
        self._discard_backups()


@contextlib.contextmanager
def locked_dirs(dirs: List[Path]) -> Iterator[None]:
    """Exclusive flock on every directory, taken in sorted order, so two
    deploys into one checkout cannot interleave their index updates."""
    fds = []
    try:
        for d in sorted(set(dirs)):
            fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        for fd in reversed(fds):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ── model ───────────────────────────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle(train_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import torch
    except ImportError as e:
        raise DeployError(
            f"reading models.pt needs torch ({e}); pip install -r requirements.txt"
        ) from None

    models_pt = train_dir / "models.pt"
    if not models_pt.is_file():
        raise DeployError(f"no models.pt in {train_dir}")
    try:
        bundle = torch.load(str(models_pt), map_location="cpu", weights_only=True)
    except Exception as e:
        raise DeployError(f"cannot read {models_pt}: {e}") from None
    cells_json: Dict[str, Any] = {}
    cj = train_dir / "cells.json"
    if cj.is_file():
        with cj.open() as f:
            cells_json = json.load(f)
    if not isinstance(bundle, dict) or not bundle.get("models"):
        raise DeployError(f"{models_pt} holds no trained cells")
    return bundle, cells_json


def engine_module(required: bool) -> Optional[Any]:
    """The tilewright module; None when it is missing and not `required`."""
    try:
        return ev.tilewright_module()
    except RuntimeError as e:
        if required:
            raise DeployError(
                f"--apply-patch checks the model with the engine before it is "
                f"installed: {e}"
            ) from None
        return None


def check_catalog(
    bundle: Mapping[str, Any], cells_json: Mapping[str, Any], tw: Optional[Any]
) -> None:
    """Refuse a bundle trained on a feature catalog the engine does not
    compute."""
    names = [list(bundle.get(k) or []) for k in ("q_names", "i_names", "x_names")]
    if str(bundle.get("catalog") or "v2") != "v2":
        check_v3_catalog(bundle, cells_json, names, tw)
        return
    catalog = [
        fs.query_feature_names(),
        fs.item_feature_names(),
        fs.interaction_feature_names(),
    ]
    dims = tuple(len(n) for n in names)
    if tuple(len(c) for c in catalog) != FEATURE_DIMS:
        raise DeployError(
            f"lib/features.py dims {tuple(len(c) for c in catalog)} differ from "
            f"the engine's {FEATURE_DIMS}"
        )
    if dims != FEATURE_DIMS:
        raise DeployError(f"bundle feature dims {dims} differ from {FEATURE_DIMS}")
    if names != catalog:
        raise DeployError("bundle feature names differ from lib/features.py")
    want = fs.feature_names_hash()
    for where, value in (
        ("cells.json", cells_json.get("feature_names_hash")),
        ("models.pt", bundle.get("feature_names_hash")),
    ):
        if value is not None and str(value) != want:
            raise DeployError(
                f"{where} feature hash {value} differs from the catalog's {want}"
            )
    if tw is None:
        return
    engine = tw.feature_catalog_hash()
    if engine != want:
        raise DeployError(
            f"tilewright module feature hash {engine} differs from the catalog's {want}"
        )


def check_v3_catalog(
    bundle: Mapping[str, Any],
    cells_json: Mapping[str, Any],
    names: List[List[str]],
    tw: Optional[Any],
) -> None:
    if bundle.get("catalog") != "v3":
        raise DeployError(f"unknown bundle catalog {bundle.get('catalog')!r}")
    try:
        catalog = fs.Catalog(
            "v3", fs.parse_attribute_spec(bundle.get("attribute_spec") or [])
        )
    except ValueError as e:
        raise DeployError(f"bundle attribute spec: {e}") from None
    if names != [list(n) for n in catalog.names()]:
        raise DeployError(
            "bundle feature names differ from lib/features.py's v3 catalog with "
            "the bundle's attribute spec"
        )
    want = catalog.hash()
    value = cells_json.get("feature_names_hash")
    if value is not None and str(value) != want:
        raise DeployError(f"cells.json feature hash {value} differs from {want}")
    if tw is None:
        return
    hashes = getattr(
        tw, "feature_catalog_hashes", lambda: [tw.feature_catalog_hash()]
    )()
    if want not in list(hashes):
        raise DeployError(
            f"the tilewright module computes catalogs {list(hashes)}, not {want}; "
            f"rebuild it from this checkout"
        )


def engine_check(data: bytes, n_cells: int, n_splits: int, tw: Any) -> None:
    """Load the bytes with the engine and compare its cell and split counts."""
    try:
        model = tw.load_model_from_memory(data)
    except Exception as e:
        raise DeployError(f"the engine rejects the model: {e}") from None
    info = tw.describe(model)
    if (int(info.n_cells), int(info.n_splits)) != (n_cells, n_splits):
        raise DeployError(
            f"the engine sees {info.n_cells} cells / {info.n_splits} splits, "
            f"the file has {n_cells} / {n_splits}"
        )


def coverage(labels: List[str]) -> Tuple[List[str], int]:
    """(split-tree leaves with no trained cell in their lineage, number of
    base cells without any model)."""
    have = set(labels)
    tree = sc.split_tree_from_labels(have)
    holes: List[str] = []
    bare = 0
    for base in grid.all_cell_labels():
        leaves = sc.leaves_under(base, tree)
        missing = [lf for lf in leaves if sc.resolve_model_cell(lf, have) is None]
        holes += missing
        if len(missing) == len(leaves):
            bare += 1
    return holes, bare


# ── plan ────────────────────────────────────────────────────────────────────


@dataclass
class Target:
    dir: str
    stem: str
    weights_name: str = ""
    weights_dir: Optional[Path] = None
    index_text: Optional[str] = None
    superseded: Optional[str] = None


def _index_entries(weights_dir: Optional[Path]) -> Tuple[Optional[str], IndexFile]:
    if weights_dir is None:
        return None, parse_index("")
    path = weights_dir / INDEX_NAME
    text = path.read_text() if path.is_file() else None
    return text, parse_index(text or "")


def plan_targets(
    targets: List[Tuple[str, str]], repo: Optional[Path], warn: bool = True
) -> List[Target]:
    out = []
    for d, stem in targets:
        t = Target(dir=d, stem=stem)
        if repo is not None:
            t.weights_dir = repo / WEIGHTS_REL / d / d
        _, idx = _index_entries(t.weights_dir)
        entries = idx.as_dict()
        t.weights_name = choose_weights_name(entries, stem)
        old = entries.get(stem)
        entries[stem] = t.weights_name
        t.index_text = render_index(idx.comments, entries)
        if old and old != t.weights_name and old not in entries.values():
            t.superseded = old
        if not warn:
            out.append(t)
            continue
        for line in idx.dropped:
            ui.warn("index", f"{d}: dropping duplicate or malformed line {line!r}")
        if t.weights_dir is not None:
            for s, b in entries.items():
                if s != stem and not (t.weights_dir / b).exists():
                    ui.warn("index", f"{d}: {s} lists missing file {b}")
        out.append(t)
    return out


# ── staging ─────────────────────────────────────────────────────────────────


def install_md(
    *,
    stem_by_dir: List[Target],
    arch: str,
    dtype: str,
    n_cells: int,
    n_splits: int,
    file_format: str = "MLREC_v2",
) -> str:
    files = "\n".join(
        f"    {WEIGHTS_REL / t.dir / t.dir / t.weights_name}\n"
        f"    {WEIGHTS_REL / t.dir / t.dir / (INDEX_NAME + '.fragment')}"
        for t in stem_by_dir
    )
    stems = "\n".join(f"    {t.dir}: {t.stem}" for t in stem_by_dir)
    return f"""# tilewright model bundle

One {file_format} model for {arch} ({n_cells} cells, {n_splits} splits, {dtype}
weights), registered for these hipBLASLt libraries:

{stems}

## Files

{files}

## Installing by hand

For every target directory above, copy the `.tilewright.bin` file to the same
path in the rocm-libraries checkout and add the line of
`{INDEX_NAME}.fragment` to that directory's `{INDEX_NAME}`, replacing any line
that names the same library stem. The engine uses the first line that names a
stem.

The `.tilewright.bin` files are committed as plain binary files (see
`shared/tilewright/.gitattributes`).

## Using the model

Build the `{MODELS_TARGET}` target of the hipBLASLt build directory
(it is part of the default build); it copies each `{INDEX_NAME}` and its
models next to the Tensile library of the same architecture, where hipBLASLt
looks for them:

    cmake --build <hipBLASLt build directory> --target {MODELS_TARGET}

hipBLASLt ranks candidate kernels with the model when `TENSILE_USE_TILEWRIGHT=1`
is set; without it selection is unchanged.
"""


def stage_bundle(
    staging: Path,
    data: bytes,
    targets: List[Target],
    with_index: bool,
    install_text: str,
) -> Dict[str, str]:
    if staging.exists():
        shutil.rmtree(staging)
    staged: Dict[str, str] = {}
    for t in targets:
        d = staging / WEIGHTS_REL / t.dir / t.dir
        d.mkdir(parents=True, exist_ok=True)
        (d / t.weights_name).write_bytes(data)
        (d / (INDEX_NAME + ".fragment")).write_text(f"{t.stem}\t{t.weights_name}\n")
        if with_index and t.index_text is not None:
            (d / INDEX_NAME).write_text(t.index_text)
        staged[t.dir] = str(d / t.weights_name)
    (staging / "INSTALL.md").write_text(install_text)
    return staged


# ── build ───────────────────────────────────────────────────────────────────


def run_appending(cmd: List[str], log_path: Path, title: str, quiet: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("a", buffering=1, errors="replace") as log:
        log.write(
            f"\n==== {datetime.now().isoformat(timespec='seconds')} {title}\n"
            f"$ {' '.join(cmd)}\n"
        )
        if not quiet:
            ui.info("build", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                errors="replace",
                bufsize=1,
            )
        except OSError as e:
            log.write(f"failed to start: {e!r}\n")
            ui.err("build", f"{title}: failed to start ({e!r})")
            return 127
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if not quiet:
                sys.stdout.write(line)
        rc = proc.wait()
        log.write(f"==== rc={rc} after {time.time() - t0:.1f}s\n")
    if rc != 0:
        ui.err("build", f"{title} failed (rc={rc}); see {log_path}")
    return rc


def build_library_dir(build_dir: Path, target: Target) -> Path:
    return build_dir / "Tensile" / "library" / target.dir


def verify_colocated(
    build_dir: Path, targets: List[Target], sha: str
) -> Tuple[List[str], List[str]]:
    """(problems, checked target dirs) of the weights the build placed next
    to its libraries. A target the build has no library directory for is
    skipped."""
    problems = []
    checked = []
    for t in targets:
        lib_dir = build_library_dir(build_dir, t)
        if not lib_dir.is_dir():
            ui.info("build", f"{lib_dir} does not exist; {t.dir} is not built here")
            continue
        checked.append(t.dir)
        idx_path = lib_dir / INDEX_NAME
        entries = (
            parse_index(idx_path.read_text()).as_dict() if idx_path.is_file() else {}
        )
        if entries.get(t.stem) != t.weights_name:
            problems.append(f"{idx_path} does not map {t.stem} to {t.weights_name}")
            continue
        placed = lib_dir / t.weights_name
        if not placed.is_file() or sha256_file(placed) != sha:
            problems.append(f"{placed} is missing or differs from the deployed file")
    if not checked:
        ui.warn(
            "build",
            f"{build_dir} has no Tensile library directory for any deploy target; "
            f"the copies are unverified and this build cannot load the model",
        )
    return problems, checked


# ── deploy gate ─────────────────────────────────────────────────────────────

GATE_FALLBACKS = ("none", "not-worse")


@dataclass
class Gate:
    """Where and how the model is measured before it is installed (see the
    module docstring)."""

    csv_dirs: List[Path]
    target: Optional[float] = None
    fallback: str = "none"
    baseline_train_dir: Optional[Path] = None
    plan: Optional[Path] = None
    hardware_device: int = 0
    library_dir: Optional[Path] = None
    pick_measurement: str = "warmup"
    bootstrap: int = 0
    rebench: Optional[mr.MissBench] = None


def _num(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return float(x)
    return None


def _cell_rows(results: List[ev.GemmResult]) -> Dict[str, Dict[str, Any]]:
    """Per leaf of the model's split tree: the keys stage04b writes per cell,
    for the model on the gate GEMMs."""
    by_leaf: Dict[str, List[ev.GemmResult]] = {}
    for r in results:
        by_leaf.setdefault(r.leaf, []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for leaf in sorted(by_leaf):
        s = ev.summarize(by_leaf[leaf])
        has_ori = s["n_origami"] > 0
        out[leaf] = {
            "sel_eff_new": _num(s["sel_eff"]),
            "n_eval": s["n_evaluated"],
            "n_miss": s["n_miss"],
            "winner_us_geomean": _num(s["winner_us_geomean"]),
            "pick_us_geomean": _num(s["pick_us_geomean"]),
            "paired_sel_eff_new": (
                _num(s["paired"]["model_sel_eff"]) if has_ori else None
            ),
            "origami_sel_eff_new": _num(s["origami_sel_eff"]) if has_ori else None,
            "origami_n_eval": s["n_origami"],
            "origami_pick_us_geomean": (
                _num(s["origami_pick_us_geomean"]) if has_ori else None
            ),
        }
    return out


def _gate_gemms(gate: Gate, quiet: bool) -> List[Dict[str, Any]]:
    import stage05_train as _train

    dirs = [str(d) for d in gate.csv_dirs if any(Path(d).glob("chunk_*.csv"))]
    if not dirs:
        return []
    cells, _ = _train.load_enriched_chunks(dirs, quiet)
    gemms = [g for gs in cells.values() for g in gs]
    if gate.plan is not None:
        try:
            ev.attach_plan(gemms, json.loads(Path(gate.plan).read_text()))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise DeployError(f"cannot read --gate-plan {gate.plan}: {e!r}") from None
    return gemms


def measure_gate(
    data: bytes,
    labels: List[str],
    *,
    cfg: Mapping[str, Any],
    arch: str,
    dtype: str,
    stem: str,
    gate: Gate,
    tw: Any,
    quiet: bool,
) -> Dict[str, Any]:
    """The model file `data` on the gate GEMMs, and the baseline bundle on
    the same GEMMs; DeployError when the measurement cannot be made."""
    if tw is None:
        raise DeployError(
            "the deploy gate evaluates the model with the engine; install the "
            "tilewright module (pip install shared/tilewright/python)"
        )
    try:
        hw = hardware.device_hardware(cfg, gate.hardware_device)
    except (ValueError, RuntimeError, OSError) as e:
        raise DeployError(f"deploy gate: no device values ({e})") from None
    gemms = _gate_gemms(gate, quiet)
    library = (
        ev.library_pool(gate.library_dir, stem, cfg, gate.hardware_device)
        if gate.library_dir is not None
        else None
    )
    pools = ev.pools_for(gemms, library)

    def run(model: bytes, model_labels: List[str], what: str) -> mr.Measured:
        def evaluate(rule: str) -> ev.Evaluation:
            out = ev.evaluate_model(
                model,
                gemms,
                hardware=hw,
                pools=pools,
                labels=model_labels,
                tw=tw,
                pick_measurement=rule,
            )
            return ev.require_consistent_routing(out, what)

        try:
            return mr.measured_evaluation(
                evaluate, gemms, gate.rebench, gate.pick_measurement, what
            )
        except (ev.RoutingMismatchError, RuntimeError, ValueError) as e:
            raise DeployError(f"deploy gate: {e}") from None

    measured = run(data, labels, "gate") if gemms else None
    results = measured.evaluation.results if measured is not None else []
    overall = ev.summarize(results, bootstrap=gate.bootstrap)
    m: Dict[str, Any] = {
        "csv_dirs": [str(d) for d in gate.csv_dirs],
        "weight_dtype": dtype,
        "pick_measurement": (
            ev.REBENCH_PICK_MEASUREMENT if gate.rebench else gate.pick_measurement
        ),
        "n_gemms": len(gemms),
        "n_evaluated": overall["n_evaluated"],
        "n_miss": overall["n_miss"],
        "sel_eff": _num(overall["sel_eff"]),
        "n_cells": len({r.leaf for r in results}),
        "overall": overall,
        "per_cell": _cell_rows(results),
        "baseline": None,
    }
    if gate.rebench is not None:
        m.update(_rebench_numbers(measured, gate.pick_measurement))
    if gate.baseline_train_dir is not None:
        base: Dict[str, Any] = {"train_dir": str(gate.baseline_train_dir)}
        try:
            bundle, _ = load_bundle(Path(gate.baseline_train_dir))
            base_data = mlrec.write_model(
                bundle, None, arch, hardware.arch_constants(arch), dtype
            )
        except (DeployError, mlrec.FormatError, ValueError) as e:
            base["error"] = str(e)
        else:
            if gemms:
                theirs = run(base_data, sorted(bundle["models"]), "baseline")
                other = theirs.evaluation
                both = [
                    (a, b)
                    for a, b in zip(results, other.results)
                    if a.sel_eff is not None and b.sel_eff is not None
                ]
                if gate.rebench is not None:
                    base.update(_rebench_numbers(theirs, gate.pick_measurement))
                base.update(
                    n=len(both),
                    sel_eff=_num(ev.geomean(b.sel_eff for _, b in both)),
                    bundle_sel_eff=_num(ev.geomean(a.sel_eff for a, _ in both)),
                    sha256=sha256_bytes(base_data),
                )
        m["baseline"] = base
    return m


def _rebench_numbers(measured: Optional[mr.Measured], rule: str) -> Dict[str, Any]:
    """The configured rule's numbers and the re-bench counts of a gate
    evaluation whose misses were re-benched."""
    if measured is None or measured.record is None:
        return {"pre_rebench": None, "rebench": None}
    rec = measured.record
    return {
        "pre_rebench": mr.pre_rebench(ev.summarize(measured.pre.results), rule),
        "n_rebenched": rec["n_rebenched"],
        "n_rebench_failed": rec["n_rebench_failed"],
        "n_context_skipped": rec["n_context_skipped"],
        "rebench": rec,
    }


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _gate_record(m: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The gate's headline numbers for the deploy record."""
    if m is None:
        return None
    keys = (
        "sel_eff",
        "n_gemms",
        "n_evaluated",
        "n_miss",
        "n_cells",
        "pick_measurement",
        "pre_rebench",
        "n_rebenched",
        "n_rebench_failed",
        "n_context_skipped",
        "target",
        "fallback",
        "passed",
        "reason",
        "baseline",
        "csv_dirs",
    )
    out = {k: m.get(k) for k in keys}
    if isinstance(out.get("baseline"), dict):
        out["baseline"] = {k: v for k, v in out["baseline"].items() if k != "rebench"}
    out["file"] = "gate.json"
    return _json_ready(out)


def gate_decision(m: Mapping[str, Any], gate: Gate) -> Tuple[bool, str]:
    """(install the model, why) under the gate's target and fallback."""
    se = m.get("sel_eff") if m.get("n_evaluated") else None
    if gate.target is None:
        return True, "no target"
    if se is not None and se >= gate.target:
        return True, f"{se * 100:.2f}% on the gate GEMMs reaches the target"
    below = (
        f"{se * 100:.2f}% on the gate GEMMs is below the target"
        if se is not None
        else "no gate GEMM measured"
    )
    if gate.fallback != "not-worse":
        return False, below
    if gate.baseline_train_dir is None:
        return True, f"{below}; no bundle was deployed before"
    base = m.get("baseline") or {}
    mine, theirs = _num(base.get("bundle_sel_eff")), _num(base.get("sel_eff"))
    if mine is not None and theirs is not None and mine >= theirs:
        return True, (
            f"{below}, but not worse than the deployed bundle on the same "
            f"{base.get('n')} GEMMs ({mine * 100:.2f}% vs {theirs * 100:.2f}%)"
        )
    if mine is not None and theirs is not None:
        return False, (
            f"{below}, and worse than the deployed bundle on the same "
            f"{base.get('n')} GEMMs ({mine * 100:.2f}% vs {theirs * 100:.2f}%)"
        )
    return False, f"{below}, and not comparable with the deployed bundle"


def print_gate(m: Mapping[str, Any], origami_reference: bool) -> None:
    se = m.get("sel_eff")
    ui.info(
        "gate",
        f"model on {m['n_evaluated']} gate GEMMs ({m['n_cells']} cells, "
        f"{m['n_miss']} misses): " + (f"{se * 100:.2f}%" if se is not None else "n/a"),
    )
    pre = m.get("pre_rebench")
    if pre:
        pse = pre.get("sel_eff")
        ui.info(
            "gate",
            f"misses re-benched: {m['n_rebenched']} measured, "
            f"{m['n_rebench_failed']} failed, {m['n_context_skipped']} context "
            f"GEMMs skipped; under {pre['pick_measurement']} "
            + (f"{pse * 100:.2f}%" if pse is not None else "n/a"),
        )
    base = m.get("baseline") or {}
    if base.get("n"):
        ui.info(
            "gate",
            f"deployed bundle on the same {base['n']} GEMMs: "
            f"{base['sel_eff'] * 100:.2f}% (this model "
            f"{base['bundle_sel_eff'] * 100:.2f}%)",
        )
    elif base.get("error"):
        ui.warn("gate", f"baseline {base['train_dir']} not measured: {base['error']}")
    p = (m.get("overall") or {}).get("paired") or {}
    if origami_reference and p.get("n"):
        ui.info(
            "gate",
            f"[reference] Origami on the same {p['n']} GEMMs: "
            f"{p['origami_sel_eff'] * 100:.2f}% (this model "
            f"{p['model_sel_eff'] * 100:.2f}%)",
        )


# ── stage entry point ───────────────────────────────────────────────────────


def deploy(
    *,
    train_dir: Path,
    config_yaml: Path,
    out_dir: Path,
    library_stem: Optional[str] = None,
    weight_dtype: Optional[str] = None,
    repo: Optional[Path] = None,
    apply_patch: bool = False,
    build_after_apply: bool = False,
    build_dir: Optional[Path] = None,
    build_jobs: Optional[int] = None,
    build_target: str = MODELS_TARGET,
    gate: Optional[Gate] = None,
    origami_reference: bool = False,
    quiet: bool = False,
) -> int:
    try:
        return _deploy(
            train_dir=train_dir,
            config_yaml=config_yaml,
            out_dir=out_dir,
            library_stem=library_stem,
            weight_dtype=weight_dtype,
            repo=repo,
            apply_patch=apply_patch,
            build_after_apply=build_after_apply,
            build_dir=build_dir,
            build_jobs=build_jobs,
            build_target=build_target,
            gate=gate,
            origami_reference=origami_reference,
            quiet=quiet,
        )
    except (DeployError, ConfigError, ValueError, mlrec.FormatError) as e:
        ui.err("deploy", str(e))
        return 1


def _deploy(
    *,
    train_dir: Path,
    config_yaml: Path,
    out_dir: Path,
    library_stem: Optional[str],
    weight_dtype: Optional[str],
    repo: Optional[Path],
    apply_patch: bool,
    build_after_apply: bool,
    build_dir: Optional[Path],
    build_jobs: Optional[int],
    build_target: str,
    gate: Optional[Gate],
    origami_reference: bool,
    quiet: bool,
) -> int:
    train_dir, out_dir = train_dir.resolve(), out_dir.resolve()
    repo = repo.resolve() if repo is not None else None
    build_dir = build_dir.resolve() if build_dir is not None else None
    with config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    arch = str(cfg.get("arch") or "")
    if not arch:
        raise DeployError(f"{config_yaml} sets no arch")
    hbl = dict(cfg.get("hipblaslt") or {})
    if library_stem:
        hbl["library_stem"] = library_stem
    if not hbl.get("library_stem"):
        raise DeployError(
            "no library stem: set hipblaslt.library_stem or pass --library-stem"
        )
    cfg = dict(cfg, hipblaslt=hbl)
    stem = str(hbl["library_stem"])
    dtype = (weight_dtype or str(cfg_get(cfg, "deploy.weight_dtype"))).lower()
    mlrec.weight_dtype_code(dtype)
    if apply_patch and repo is None:
        raise DeployError("--apply-patch needs --rocm-libraries-to-deploy")
    if repo is not None and not (repo / "shared" / "tilewright").is_dir():
        raise DeployError(f"{repo} is not a rocm-libraries checkout")
    if build_after_apply and not apply_patch:
        ui.warn("deploy", "--build-after-apply ignored without --apply-patch")
        build_after_apply = False
    if build_after_apply and build_dir is None:
        assert repo is not None
        build_dir = repo / "projects" / "hipblaslt" / "build" / "release"

    out_dir.mkdir(parents=True, exist_ok=True)
    if not quiet:
        ui.banner(f"stage06  deploy  config_id={cfg.get('config_id')}  arch={arch}")
        ui.info("cfg", f"train_dir    : {train_dir}")
        ui.info("cfg", f"library stem : {stem}")
        ui.info("cfg", f"weight dtype : {dtype}")
        ui.info("cfg", f"checkout     : {repo or '(none: staging only)'}")

    tw = engine_module(required=apply_patch or gate is not None)
    bundle, cells_json = load_bundle(train_dir)
    check_catalog(bundle, cells_json, tw)
    labels = sorted(bundle["models"])
    data = mlrec.write_model(bundle, None, arch, hardware.arch_constants(arch), dtype)
    parsed = mlrec.read_model(data)
    n_cells, n_splits = len(parsed["cells"]), len(parsed["splits"])
    if [c["label"] for c in parsed["cells"]] != labels:
        raise DeployError("the written model does not round-trip its cell labels")
    engine_ok = tw is not None
    if engine_ok:
        engine_check(data, n_cells, n_splits, tw)
    sha = sha256_bytes(data)
    if not quiet:
        ui.ok(
            "pack",
            f"{parsed['format']} {dtype}: {len(data):,} bytes, {n_cells} cells, "
            f"{n_splits} splits, sha256 {sha[:16]}"
            + (
                ""
                if engine_ok
                else " (tilewright module not installed: engine check skipped)"
            ),
        )
    holes, bare = coverage(labels)
    if holes:
        served = (
            "the model scores them with its nearest trained base cell"
            if parsed["version"] == 3
            else "problems routed there use the analytical ranking"
        )
        ui.warn(
            "coverage",
            f"{len(holes)} split-tree leaves have no trained cell in their lineage "
            f"({bare} of {len(grid.all_cell_labels())} base cells have no model); "
            f"{served}, e.g. {holes[:3]}",
        )

    measured: Optional[Dict[str, Any]] = None
    passed, why = True, "no gate"
    if gate is not None:
        measured = measure_gate(
            data,
            labels,
            cfg=cfg,
            arch=arch,
            dtype=dtype,
            stem=stem,
            gate=gate,
            tw=tw,
            quiet=quiet,
        )
        passed, why = gate_decision(measured, gate)
        measured.update(
            target=gate.target, fallback=gate.fallback, passed=passed, reason=why
        )
        write_atomic(
            out_dir / "gate.json",
            (json.dumps(_json_ready(measured), indent=2) + "\n").encode(),
        )
        if not quiet:
            print_gate(measured, origami_reference)
            (ui.ok if passed else ui.warn)(
                "gate", ("deploying: " if passed else "not deploying: ") + why
            )

    targets = plan_targets(deploy_targets(cfg), repo)
    staging = out_dir / "staged_for_deploy"
    staged = stage_bundle(
        staging,
        data,
        targets,
        with_index=repo is not None,
        install_text=install_md(
            stem_by_dir=targets,
            arch=arch,
            dtype=dtype,
            n_cells=n_cells,
            n_splits=n_splits,
            file_format=parsed["format"],
        ),
    )
    tree = {r.cell: r for r in parsed["splits"]}
    manifest = {
        "config_id": cfg.get("config_id"),
        "arch": arch,
        "weight_dtype": dtype,
        "format": parsed["format"],
        "feature_catalog_hash": parsed["feature_catalog_hash"],
        "feature_dims": dict(zip(("q", "i", "x"), parsed["dims"])),
        "file": {
            "size": len(data),
            "sha256": sha,
            "payload_crc32": f"0x{parsed['payload_crc32']:08x}",
        },
        "targets": [
            {"dir": t.dir, "library_stem": t.stem, "weights_file": t.weights_name}
            for t in targets
        ],
        "n_cells": n_cells,
        "n_splits": n_splits,
        "cells": [
            {
                "label": c["label"],
                "embed_dim": c["embed_dim"],
                "hidden_dim": c["hidden_dim"],
                "inter_hidden": c["inter_hidden"],
                "n_signatures": len(c["smart_k_signatures"]),
                "n_view_keys": len(c.get("view_key_hashes") or []),
            }
            for c in parsed["cells"]
        ],
        "splits": [
            {
                "cell": r.cell,
                "axis": r.axis,
                "threshold": r.threshold,
                "lo": r.lo_label,
                "hi": r.hi_label,
            }
            for r in parsed["splits"]
        ],
    }
    if parsed["version"] == 3:
        manifest["feature_spec_hash"] = f"{parsed['feature_spec_hash']:016x}"
        manifest["attribute_spec"] = [a.to_dict() for a in parsed["attribute_spec"]]
        manifest["contexts"] = parsed["contexts"]
        manifest["serving"] = parsed["serving"]
        manifest["clip"] = parsed["clip"]
    write_atomic(
        out_dir / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()
    )
    summary = [
        f"config_id     : {cfg.get('config_id')}",
        f"arch          : {arch}",
        f"weight dtype  : {dtype}",
        f"cells, splits : {n_cells}, {n_splits}",
        "split tree    :",
    ]
    summary += [
        "    " + ln for ln in (sc.format_split_tree(tree) or "(none)").splitlines()
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")

    record: Dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "train_dir": str(train_dir),
        "arch": arch,
        "weight_dtype": dtype,
        "feature_catalog_hash": parsed["feature_catalog_hash"],
        "sha256": sha,
        "size": len(data),
        "payload_crc32": parsed["payload_crc32"],
        "n_cells": n_cells,
        "n_splits": n_splits,
        "staged_model": staged[targets[0].dir],
        "repo": str(repo) if repo else None,
        "engine_checked": engine_ok,
        "deployed": passed,
        "gate": _gate_record(measured),
        "applied": False,
        "built": False,
        "colocation": None,
        "build_dir": str(build_dir) if build_dir else None,
        "targets": [
            {
                "dir": t.dir,
                "library_stem": t.stem,
                "weights_file": t.weights_name,
                "staged": staged[t.dir],
                "installed": (
                    str(t.weights_dir / t.weights_name) if t.weights_dir else None
                ),
            }
            for t in targets
        ],
    }

    def _write_record() -> None:
        write_atomic(
            out_dir / "deploy_record.json",
            (json.dumps(record, indent=2) + "\n").encode(),
        )

    if not passed:
        record["not_deployed"] = why
        _write_record()
        if not quiet:
            ui.ok("stage", f"staged in {staging}; the gate kept it out: {why}")
        return 0
    if not apply_patch:
        _write_record()
        if not quiet:
            ui.ok("stage", f"staged in {staging}; nothing written to a checkout")
        return 0

    assert repo is not None
    weight_dirs = [t.weights_dir for t in targets if t.weights_dir is not None]
    txn = Transaction(out_dir / ".rollback")
    for d in weight_dirs:
        txn.ensure_dir(d)
    try:
        with locked_dirs(weight_dirs):
            targets = plan_targets(deploy_targets(cfg), repo, warn=False)
            for t in targets:
                assert t.weights_dir is not None
                txn.put(t.weights_dir / t.weights_name, data)
            for t in targets:
                assert t.weights_dir is not None and t.index_text is not None
                txn.put(t.weights_dir / INDEX_NAME, t.index_text.encode())
            for t in targets:
                if t.superseded and _plain_file_name(t.superseded):
                    assert t.weights_dir is not None
                    txn.remove(t.weights_dir / t.superseded)
            txn.sync()
            if build_after_apply:
                assert build_dir is not None
                for t in targets:
                    txn.guard(build_library_dir(build_dir, t) / INDEX_NAME)
                    txn.guard(build_library_dir(build_dir, t) / t.weights_name)
                cmd = ["cmake", "--build", str(build_dir), "--target", build_target]
                if build_jobs:
                    cmd += ["--parallel", str(build_jobs)]
                rc = run_appending(cmd, out_dir / "build.log", build_target, quiet)
                if rc != 0:
                    raise DeployError(f"building {build_target} failed (rc={rc})")
                problems, checked = verify_colocated(build_dir, targets, sha)
                if problems:
                    raise DeployError("; ".join(problems))
                record["built"] = bool(checked)
                record["colocation"] = "verified" if checked else "unverified"
                record["colocated_targets"] = checked
            txn.commit()
    except BaseException as e:
        ui.err("deploy", f"restoring the previous files ({e})")
        txn.rollback()
        record["error"] = str(e)
        record["deployed"] = False
        _write_record()
        raise
    record["applied"] = True
    for t, rt in zip(targets, record["targets"]):
        rt["weights_file"] = t.weights_name
        rt["installed"] = str(t.weights_dir / t.weights_name) if t.weights_dir else None
        if t.superseded:
            rt["removed"] = t.superseded
    _write_record()
    if not quiet:
        for t in targets:
            ui.ok("write", f"{t.weights_dir / t.weights_name}  (index: {t.stem})")
            if t.superseded:
                ui.info(
                    "write",
                    f"removed {t.superseded}, which no index entry uses any more",
                )
        ui.ok(
            "done",
            "deployed"
            + (
                " and co-located in the build"
                if record["built"]
                else (
                    " (build ran; copies unverified)"
                    if record["colocation"] == "unverified"
                    else ""
                )
            ),
        )
    return 0


def _miss_bench(args: argparse.Namespace) -> Optional[mr.MissBench]:
    """The gate's MissBench of --rebench-misses (None without it)."""
    if not args.rebench_misses or not args.gate_csv_dir:
        return None
    cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    try:
        n_cu: Optional[int] = hardware.device_hardware(cfg, args.hardware_device).n_cu
    except (ValueError, RuntimeError, OSError):
        n_cu = None
    return mr.from_args(args, cfg, args.out_dir / "rebench", n_cu)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="stage06: deploy a trained model into a rocm-libraries checkout"
    )
    ap.add_argument(
        "--train-dir", type=Path, required=True, help="stage05 output (models.pt)"
    )
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="staging directory")
    ap.add_argument(
        "--library-stem", default=None, help="default: hipblaslt.library_stem"
    )
    ap.add_argument(
        "--weight-dtype",
        choices=("fp32", "bf16", "int8", "int4"),
        default=None,
        help="default: deploy.weight_dtype",
    )
    ap.add_argument("--rocm-libraries-to-deploy", type=Path, default=None)
    ap.add_argument(
        "--apply-patch", action="store_true", help="install into the checkout"
    )
    ap.add_argument(
        "--build-after-apply",
        action="store_true",
        help="build the hipBLASLt target that copies the models into the build",
    )
    ap.add_argument(
        "--build-target", default=MODELS_TARGET, help=f"default: {MODELS_TARGET}"
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="hipBLASLt build directory (default: <checkout>/projects/hipblaslt/build/release)",
    )
    ap.add_argument("--build-jobs", type=int, default=None)
    g = ap.add_argument_group("deploy gate")
    g.add_argument(
        "--gate-csv-dir",
        type=Path,
        action="append",
        default=None,
        help="enriched GEMMs the bundle never trained on (repeatable); the "
        "model is measured on them before it is installed",
    )
    g.add_argument(
        "--gate-target",
        type=float,
        default=None,
        help="install only when the model reaches this selection efficiency "
        "on the gate GEMMs",
    )
    g.add_argument(
        "--gate-fallback",
        choices=GATE_FALLBACKS,
        default="none",
        help="not-worse: below the target, install when nothing was deployed "
        "before (no --gate-baseline-train-dir) or when the model is not worse "
        "than that bundle on the same gate GEMMs",
    )
    g.add_argument(
        "--gate-baseline-train-dir",
        type=Path,
        default=None,
        help="stage05 dir of the bundle deployed before; measured on the "
        "same gate GEMMs",
    )
    g.add_argument(
        "--gate-plan",
        type=Path,
        default=None,
        help="stage01b plan.json of the gate GEMMs",
    )
    g.add_argument("--hardware-device", type=int, default=0)
    g.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library dir holding <library-stem>.dat[.zlib]: the gate's "
        "kernel pool (default: the kernels measured in the gate data)",
    )
    g.add_argument(
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default="warmup",
        help="warmup: a skip row counts with its warm-up latency and a kernel off the plan's list is a miss; timed: both are misses; any: a skip row counts "
        "and an off-list kernel is passed over (lib/evaluate.py)",
    )
    g.add_argument("--bootstrap", type=int, default=1000)
    mr.add_arguments(ap)
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print Origami's numbers on the gate GEMMs (an external "
        "reference; they never decide the deploy)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    gate = None
    try:
        rebench = _miss_bench(args)
    except (OSError, ValueError) as e:
        ui.err("gate", str(e))
        return 1
    if args.gate_csv_dir:
        gate = Gate(
            csv_dirs=list(args.gate_csv_dir),
            target=args.gate_target,
            fallback=args.gate_fallback,
            baseline_train_dir=args.gate_baseline_train_dir,
            plan=args.gate_plan,
            hardware_device=args.hardware_device,
            library_dir=args.library_dir,
            pick_measurement=args.pick_measurement,
            bootstrap=args.bootstrap,
            rebench=rebench,
        )
    return deploy(
        train_dir=args.train_dir,
        config_yaml=args.config_yaml,
        out_dir=args.out_dir,
        library_stem=args.library_stem,
        weight_dtype=args.weight_dtype,
        repo=args.rocm_libraries_to_deploy,
        apply_patch=args.apply_patch,
        build_after_apply=args.build_after_apply,
        build_dir=args.build_dir,
        build_jobs=args.build_jobs,
        build_target=args.build_target,
        gate=gate,
        origami_reference=args.origami_reference,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
