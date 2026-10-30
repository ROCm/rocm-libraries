# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage06: deploy a trained model into a rocm-libraries checkout.

Packs `<train-dir>/models.pt` into one model file (MLREC_v2, or MLREC_v3 for
a bundle trained with the v3 catalog) that carries the model constants of the
config's arch (`lib.hardware.arch_constants`) and stages it
under `<out-dir>/staged_for_deploy/`. With --apply-patch the file is installed
into the `Tilewright` directory of every deploy target `t` (`deploy.targets`,
default: the config's arch): `<arch>/<t>/Tilewright/` of hipBLASLt's library
logic (`lib/weights_layout.py`), where `<arch>` is `t`, or for a build alias
the architecture it builds, and the library stem is registered in that
directory's `tilewright_index`. The checkout must hold the logic of `<arch>`.
With --build-after-apply the hipBLASLt build target that copies the models
into the build (`hipblaslt-tilewright-models`) then places them next to the
Tensile library, where hipBLASLt loads them from, and the copies are
verified.

The kernels of the served pool (the library's table and bank) that the
config's blocklist names (lib/correctness.py: they compute wrong results) are
listed, one name per line, in the integration-side sidecar
`<library stem>.tilewright.blocklist` staged and installed next to the model
of every target (`blocklist_names`); TensileLite leaves them out of the
pools it ranks with the model. A deploy without blocked kernels removes the
sidecar a former deploy of that stem installed, in the checkout and the
build. The deploy record names the blocklist the deploy applied
(`blocklist.digest`, lib/correctness.py `names_digest`), the blocklists that
filtered the bundle's labels (`blocklist.trained_with`, stage05's record,
which is warned about when it differs) and each target's sidecar
(`targets[].blocklist_sha256`).

Installing is all-or-nothing: every file is written under a temporary name,
synced and renamed into place, indexes last; a failure at any point,
including a failed build or a copy the build placed wrongly, restores the
previous files in the checkout and the index and model copies in the build
tree. A build without a Tensile library directory for any deploy target is
recorded as built but unverified (`colocation: "unverified"`).

The model is loaded back with the tilewright Python module before anything is
staged; with --apply-patch the module is required.

An MLREC_v3 model records the neutral provenance of a published model
(`deploy.provenance: neutral`, the default; lib/provenance.py
`published_provenance`): the library stem as its `config_id`, the bundle's
training data hash and its launch model, and none of the training config's
id, the source revision or the date, which stay in models.pt and
metrics.json; `deploy.provenance: full` records the bundle's provenance as
stage05 wrote it. The deploy record names what the file records
(`provenance`).

An MLREC_v3 model is packed with three serving options stage06 decides
(`deployed_serving`): the bank flag, when the pool the bundle was trained on
held the bank (stage05 records it; for a bundle that predates the record,
when the config's kernel pool has a bank, lib/library_pool.py), the
trained-key gate with the view keys the bundle records as trained, when
`train.trained_key_gate` is on (the default), and the alignment veto of
`deploy.alignment_veto` (lib/feasibility.py `deployed_alignment_veto`), when
the config sets it. A bundle that records no trained keys is packed without
the gate. The veto is a serving rule of the staged file: no round's bundle,
no stage04b evaluation and no planner reads it, and with
--score-without-alignment-veto, which the orchestrator passes for a round,
the deploy gate, the audit record and the per-cell gate score the model, and
the bundles they compare it with, without it, so that no decision of a
run's rounds, and none of the data they collect, depends on it; without the
option (a final retrain, a deploy outside a run) they score the bytes
staged, veto included, as the audit set's evaluations do (lib/audit.py
`model_bytes`). The deploy record names the veto the staged file carries
(`alignment_veto`), its sha256 without the veto (`veto_free_sha256`, by which
a later repair finds the bundle) and whether the gates scored it without
(`scored_without_alignment_veto`). With `deploy.drift_fallback`
({keep: <share in (0, 1]>}; absent: none) it is packed with the engine's
drift fallback as well (lib/drift_fallback.py): a cell whose pool holds
fewer than that share of its trained whitelisted view keys ranks its
whitelist and the cascade over its other keys as one tier. Only the
deployed bytes carry it, never a round's bundle, so what a run benches and
trains is as without it. The deploy refuses it for a bundle without a
view-key whitelist tier, without a cascade, with launch keys or with a cell
that lists signatures, and, before it measures or stages anything
(`drift_preflight`), for a model the pool of the library it deploys into
(--library-dir, else the config's bench library) would already serve
degraded: that library lacks trained whitelist tier keys of the model,
whose tier is built for it again first (scripts/build_whitelist_tier.py).
The gate, the audit and the per-cell gate measure the deployed bytes as
they rank over the library pool (lib/evaluate.py, `allow_drift`) and name
the cells they find degraded, a baseline's or those of an audit pool that
leaves out more kernels (gate.json `drift_degraded`); the deploy record
keeps the preflight's counts (`drift_fallback`) and, for a bundle with a
whitelist tier, those over the pool the model is served from
(`whitelist_tier.drift_fallback`).

Deploy gate (--gate-csv-dir): the model file, exactly the bytes that would be
installed (less the alignment veto with --score-without-alignment-veto), is
evaluated (lib/evaluate.py) on enriched GEMMs its bundle never trained on
(stage04b's gate/ of the round). With --gate-target T it is
installed only when its selection efficiency there reaches T; with
--gate-fallback not-worse a model below T is still installed when no
--gate-baseline-train-dir is given (nothing deployed yet), or when it is at
least as good as that baseline bundle on the same gate GEMMs. With
--rebench-misses every miss of the model (and of the baseline) is re-benched
on the GPUs and charged by its measured ratio to the winner
(lib/miss_rebench.py); the decision uses those numbers and gate.json keeps
the configured rule's as `pre_rebench`, and the re-bench record with each
pick and re-benched winner named by signature and kernel id. A model the
gate keeps out is still staged; `deploy_record.json` says `deployed: false`
and why. Origami's numbers on the gate GEMMs are recorded and printed only
with --origami-reference.

Per-cell gate (--cell-gate-baseline, with --audit-dir): the bundle the run
ships is scored on the audit set beside this one, and this one is kept out
when it regresses against it (`lib.audit.cell_gate`): on the
--cell-gate-groups, a cell significantly worse beyond --cell-tolerance or
confidently fallen below --cell-floor, family-wise at --cell-gate-alpha
over their cells (--audit-bootstrap sign flips); over every group, more
than --tail-budget new picks below --tail-floor that a repeated
measurement confirms, each GEMM of each group counted on its own (a pick
the shipped bundle had at or above the floor; with --tail-collapse below
1, only the drops to at most that fraction of its efficiency), however
many collapses it heals. cell_gate.json names the cells and GEMMs, the
healed collapses and those both bundles share. When the audit set cannot
measure both bundles the other gates decide. --cell-gate-record-only
measures it without keeping the bundle out. The deploy record keeps the
deploy gate's own verdict before the per-cell gate's (`deploy_gate`), so
that a bundle the per-cell gate alone keeps out can be repaired: the
orchestrator repairs the base cells it flagged (gate.repair,
lib/cell_merge.py `RoundRepair`; full.final_retrain.repair for a final
retrain) and runs this stage again on the repaired bundle, whose
metrics.json `repair` the deploy record carries (`repair`).

Outputs (run directory only):
    staged_for_deploy/   the files to install, per-target index fragments and
                         INSTALL.md
    manifest.json        cells, splits and file hash of the model
    summary.txt          the split tree
    gate.json            with --gate-csv-dir: the model's own numbers on the
                         gate GEMMs, overall and per cell, and the baseline's
    cell_gate.json       with --cell-gate-baseline: the per-cell gate's
                         findings per audit group, the baseline it compared
                         with and the verdict
    deploy_record.json   what was written where; stage07 checks the runtime
                         against it. `models_sha256`: the sha256 of the
                         <train-dir>/models.pt it packed. With
                         --library-dir also the member order of every
                         candidate key with several members
                         in the library the model is served from (the
                         deploy build's when it holds the library, else
                         --library-dir's), as a digest per key
                         (`member_order`), and the keys whose order differs
                         from the build the bundle trained with
                         (<train-dir>/member_order.json of stage05, for a
                         carried bundle that of the round it came from),
                         which are warned about: the runtime serves a key's
                         first member whose predicates pass.
                         `identity_rule`: the kernel identity rule the
                         bundle's keys follow (lib/kernel_identity.py; null
                         for a bundle from before it was recorded), which
                         the integration must apply too. `launch`: how
                         the benches of the bundle's training data launched
                         the kernels (stage05's `bench_launch`: the launch
                         heuristics, the launch overrides set, the
                         workspaces; lib/launch_settings.py) and, with a
                         build directory, that build's launch heuristic
                         (null when unknown), which is warned about when
                         its launch-rule sources differ from the training
                         benches' builds. `whitelist_tier`: the bundle's
                         whitelist tier (stage05 --whitelist-tier-k,
                         lib/whitelist_tier.py) and, with a library, how it
                         meets the pool the model is served from
                         (`served_pool`); tier keys that pool lacks are
                         warned about
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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence
from typing import Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import yaml  # noqa: E402

from lib import audit as auditlib  # noqa: E402
from lib import bank_pool  # noqa: E402
from lib import cell_merge as cm  # noqa: E402
from lib import correctness as cx  # noqa: E402
from lib import drift_fallback  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import feasibility as feas  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import grid  # noqa: E402
from lib import hardware  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import miss_rebench as mr  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import provenance as prov  # noqa: E402
from lib import repeats  # noqa: E402
from lib import run_bundles  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import tilewright_index as twi  # noqa: E402
from lib import ui  # noqa: E402
from lib import weights_layout as wl  # noqa: E402
from lib import whitelist_tier as wt  # noqa: E402
from lib.bundle import bundle_file  # noqa: E402
from run_orchestrator import (  # noqa: E402
    FEATURE_DIMS,
    ConfigError,
    cfg_get,
    deploy_targets,
    derive_paths,
)

# The Tilewright directory of a deploy target, relative to the checkout.
weights_rel = wl.rel_dir
INDEX_NAME = twi.INDEX_NAME
INDEX_HEADER = (
    "# tilewright per-library weights index: "
    "<logic_stem>\\t<weights.bin>[\\tchip=0x<PCI chip ID>]"
)
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
    # Rows for the devices of one PCI chip ID: (stem, file, chip ID).
    chip_rows: List[Tuple[str, str, int]] = field(default_factory=list)
    # Rows that make the engine load no model for their stem: (line number
    # from 1, stem, line).
    malformed_rows: List[Tuple[int, str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, str]:
        return dict(self.entries)


def parse_index(text: str) -> IndexFile:
    """Parse a tilewright_index the way the engine does
    (lib/tilewright_index.py): a '#' starts a comment, the first two
    whitespace-separated tokens are stem and file, a third `chip=0x<id>`
    makes the row one for the devices of that PCI chip ID (`chip_rows`),
    and the first line naming a stem (and chip ID) is the one that counts.
    Later duplicates, one-token lines and malformed rows are returned in
    `dropped`, malformed rows also in `malformed_rows`."""
    idx = twi.parse(text)
    return IndexFile(
        idx.comments,
        [(s, f) for s, f, c in idx.rows if c is None],
        idx.dropped,
        [(s, f, c) for s, f, c in idx.rows if c is not None],
        idx.malformed_rows,
    )


def render_index(
    comments: List[str],
    entries: Mapping[str, str],
    chip_rows: Sequence[Tuple[str, str, int]] = (),
) -> str:
    lines = list(comments) or [INDEX_HEADER]
    lines += twi.render_rows(
        [(s, f, None) for s, f in entries.items()] + list(chip_rows)
    )
    return "\n".join(lines) + "\n"


def _plain_file_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name[0] != "."


def choose_weights_name(
    entries: Mapping[str, str], stem: str, chip_files: Sequence[str] = ()
) -> str:
    """Keep the file name the stem already maps to unless another stem, or a
    row for the devices of one chip ID (`chip_files` holds their files),
    shares that file; otherwise `<stem>.tilewright.bin`."""
    old = entries.get(stem)
    if (
        old
        and _plain_file_name(old)
        and old.endswith(BIN_SUFFIX)
        and sum(1 for b in entries.values() if b == old) == 1
        and old not in chip_files
    ):
        return old
    name = stem + BIN_SUFFIX
    if any(b == name for s, b in entries.items() if s != stem) or name in chip_files:
        raise DeployError(f"{name} is already used by another row of the index")
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


def deployed_serving(
    bundle: Mapping[str, Any], cfg: Mapping[str, Any], what: str, quiet: bool
) -> Dict[str, Any]:
    """`bundle` with the serving options an MLREC_v3 model deploys with: the
    bank flag (as stage05 recorded it, else whether the config's planner has
    a bank), the trained-key gate (`train.trained_key_gate`, when the
    bundle records trained view keys), the alignment veto
    (`deploy.alignment_veto`; DeployError for a bundle served with the
    non-temporal rule) and the drift fallback (`deploy.drift_fallback`;
    DeployError for a bundle that cannot take it).
    Other bundles are returned as they are, and DeployError when the config
    asks for the drift fallback."""
    keep = drift_keep_ppm(cfg)
    if str(bundle.get("catalog") or "v2") != "v3":
        if keep is not None:
            raise DeployError(
                f"deploy.drift_fallback: {what} is not an MLREC_v3 bundle; the drift "
                f"fallback is an MLREC_v3 serving option"
            )
        return dict(bundle)
    serving = dict(bundle.get("serving") or {})
    serving.pop("drift_keep_ppm", None)
    if keep is not None:
        why = drift_fallback.refusal(serving, (bundle.get("models") or {}).values())
        if why is not None:
            raise DeployError(
                f"deploy.drift_fallback: {what} cannot take the drift fallback: {why}"
            )
        serving["drift_keep_ppm"] = keep
    if "bank" not in serving:
        serving["bank"] = bank_pool.bank_source(cfg).enabled
    gate = bool(cfg_get(cfg, "train.trained_key_gate"))
    keys = bundle.get("trained_view_keys") or []
    if gate and not keys and not quiet:
        ui.warn(
            "gate",
            f"{what} records no trained view keys (trained before the gate "
            f"existed): it is packed without the trained-key gate",
        )
    serving["trained_key_gate"] = gate and bool(keys)
    try:
        veto = feas.deployed_alignment_veto(cfg)
    except ValueError as e:
        raise DeployError(str(e)) from None
    serving.pop("alignment_veto", None)
    if veto is not None:
        if serving.get("non_temporal_rule"):
            raise DeployError(
                f"deploy.alignment_veto: {what} is served with the non-temporal "
                f"rule, which the alignment veto is never served with"
            )
        serving["alignment_veto"] = veto
    return dict(bundle, serving=serving)


def drift_keep_ppm(cfg: Mapping[str, Any]) -> Optional[int]:
    """The keep_ppm of `deploy.drift_fallback`, or None when it is absent
    (lib/drift_fallback.py `deployed_keep_ppm`); DeployError for a share the
    engine cannot hold exactly."""
    try:
        return drift_fallback.deployed_keep_ppm(cfg)
    except ValueError as e:
        raise DeployError(f"deploy.drift_fallback: {e}") from None


def drift_record(
    bundle: Mapping[str, Any], kernels: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """The drift fallback's trigger for a bundle packed with it
    (`deployed_serving`) over a pool's `kernels`: `lib.drift_fallback.
    drift_summary`, with the degraded cells by label."""
    models = bundle.get("models") or {}
    labels = sorted(models)
    spec = fs.parse_attribute_spec(bundle.get("attribute_spec") or [])
    summary = drift_fallback.drift_summary(
        {
            "serving": bundle.get("serving"),
            "trained_view_keys": bundle.get("trained_view_keys"),
            "cells": [models[label] for label in labels],
        },
        drift_fallback.pool_hashes(ev.pool_from_kernels(kernels), spec),
    )
    return {**summary, "degraded": [labels[i] for i in summary["degraded"]]}


def drift_preflight(
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    stem: str,
    library_dir: Optional[Path],
    device: int,
) -> Optional[Dict[str, Any]]:
    """For a bundle packed with the drift fallback: `drift_record` over the
    pool of the library it deploys into (`lib.bank_pool.config_pool`, the
    pool the runtime serves: the table and the bank the config serves,
    without the kernels the config's blocklist names), with `library_dir`.
    DeployError when that pool degrades a cell, naming the cells: the model
    would be served degraded from the start, and its tier is to be built for
    that library first. None without the fallback."""
    keep = (bundle.get("serving") or {}).get("drift_keep_ppm")
    if keep is None:
        return None
    if library_dir is None:
        raise DeployError(
            "deploy.drift_fallback: a model served with the drift fallback is checked "
            "against the library it deploys into; give --library-dir"
        )
    try:
        pool = bank_pool.config_pool(cfg, library_dir, stem, device)
    except (OSError, ValueError, RuntimeError) as e:
        raise DeployError(
            f"deploy.drift_fallback: cannot read the pool of {library_dir}: {e}"
        ) from None
    rec = drift_record(bundle, pool.kernels)
    cells = rec["degraded"]
    if cells:
        raise DeployError(
            f"deploy.drift_fallback: the pool of {library_dir} lacks {rec['missing']} "
            f"trained whitelist tier keys of {len(cells)} cells "
            f"({', '.join(cells[:4])}{', ...' if len(cells) > 4 else ''}), which the "
            f"model would serve degraded from the start (keep_ppm {keep}): build the "
            f"tier for this pool with scripts/build_whitelist_tier.py, or deploy "
            f"without deploy.drift_fallback"
        )
    return {**rec, "library_dir": str(library_dir)}


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
    text = twi.read_text(path) if path.is_file() else None
    return text, parse_index(text or "")


def refuse_malformed_rows(index: Path, idx: IndexFile) -> None:
    """DeployError while the index file `index` (parsed: `idx`) has a
    malformed row: the engine loads no model for that stem on any device,
    and a rewrite without the row would serve the stem's other rows on the
    devices the row may have been meant for."""
    if idx.malformed_rows:
        rows = "; ".join(
            f"line {n} for {stem}: {line!r}" for n, stem, line in idx.malformed_rows
        )
        raise DeployError(
            f"{index} has malformed rows ({rows}), for whose stems the engine "
            f"loads no model; a row is '<stem> <file>', optionally followed by "
            f"'chip=0x<one to four hex digits, not all zero>'. Correct or "
            f"remove them, then deploy again"
        )


def plan_targets(
    targets: List[Tuple[str, str]], repo: Optional[Path], warn: bool = True
) -> List[Target]:
    out = []
    for d, stem in targets:
        t = Target(dir=d, stem=stem)
        if repo is not None:
            logic = repo / wl.LOGIC_REL / wl.logic_arch(d)
            if not logic.is_dir():
                raise DeployError(
                    f"{logic} does not exist: the checkout has no hipBLASLt "
                    f"library logic for deploy target {d}, whose Tilewright "
                    f"directory would be {repo / weights_rel(d)}"
                )
            t.weights_dir = repo / weights_rel(d)
        _, idx = _index_entries(t.weights_dir)
        if t.weights_dir is not None:
            refuse_malformed_rows(t.weights_dir / INDEX_NAME, idx)
        entries = idx.as_dict()
        chip_files = [f for _, f, _ in idx.chip_rows]
        t.weights_name = choose_weights_name(entries, stem, chip_files)
        old = entries.get(stem)
        entries[stem] = t.weights_name
        t.index_text = render_index(idx.comments, entries, idx.chip_rows)
        if (
            old
            and old != t.weights_name
            and old not in entries.values()
            and old not in chip_files
        ):
            t.superseded = old
        if not warn:
            out.append(t)
            continue
        for line in idx.dropped:
            ui.warn("index", f"{d}: dropping duplicate or one-field line {line!r}")
        for s in dict.fromkeys(s for s, _, _ in idx.chip_rows):
            if s not in entries:
                ui.warn(
                    "index",
                    f"{d}: {s} has rows for PCI chip IDs only; an engine that "
                    f"predates the chip ID serves the model of its first row on "
                    f"every device",
                )
        if t.weights_dir is not None:
            for s, b in entries.items():
                if s != stem and not (t.weights_dir / b).exists():
                    ui.warn("index", f"{d}: {s} lists missing file {b}")
            for s, b, c in idx.chip_rows:
                if not (t.weights_dir / b).exists():
                    ui.warn(
                        "index",
                        f"{d}: {s} ({twi.chip_field(c)}) lists missing file {b}",
                    )
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
    blocked: Sequence[str] = (),
) -> str:
    files = "\n".join(
        f"    {weights_rel(t.dir) / t.weights_name}\n"
        f"    {weights_rel(t.dir) / (INDEX_NAME + '.fragment')}"
        for t in stem_by_dir
    )
    if blocked:
        files += "\n" + "\n".join(
            f"    {weights_rel(t.dir) / cx.sidecar_name(t.stem)}" for t in stem_by_dir
        )
    stems = "\n".join(f"    {t.dir}: {t.stem}" for t in stem_by_dir)
    if blocked:
        sidecar = f"""
Copy each `{cx.SIDECAR_SUFFIX}` file next to its model as well: it names,
one per line, the kernels of the library that compute wrong results, which
TensileLite never offers to the model. It replaces the file a former deploy
of the same stem left there; a bundle without one needs that file removed.
"""
    else:
        stale = "\n".join(
            f"    {weights_rel(t.dir) / cx.sidecar_name(t.stem)}" for t in stem_by_dir
        )
        sidecar = f"""
This bundle ships no `{cx.SIDECAR_SUFFIX}` file. Remove the one a former
deploy of the same stem left next to the model, in the checkout and in any
hipBLASLt build that holds a copy of it:

{stale}

TensileLite leaves the kernels such a file names out of every pool it ranks
with the model, kernels this model was trained to pick among them.
"""
    return f"""# tilewright model bundle

One {file_format} model for {arch} ({n_cells} cells, {n_splits} splits, {dtype}
weights), registered for these hipBLASLt libraries:

{stems}

## Files

{files}

## Installing by hand

For every target directory above, copy the `.tilewright.bin` file to the same
path in the rocm-libraries checkout and add the line of
`{INDEX_NAME}.fragment` to that directory's `{INDEX_NAME}`, replacing the
line that names the same library stem without a chip ID, above the stem's
lines that end in `chip=0x<id>`: those serve the devices of that PCI chip ID
and stay. The engine uses a stem's first line for the device's chip ID, else
its first line without one.

The `.tilewright.bin` files are committed as plain binary files (see
`projects/hipblaslt/.gitattributes`). A Tilewright directory holds no
`.yaml` file: TensileCreateLibrary would read it as library logic.
{sidecar}
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
    blocked: Sequence[str] = (),
) -> Dict[str, str]:
    if staging.exists():
        shutil.rmtree(staging)
    staged: Dict[str, str] = {}
    for t in targets:
        d = staging / weights_rel(t.dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / t.weights_name).write_bytes(data)
        (d / (INDEX_NAME + ".fragment")).write_text(f"{t.stem}\t{t.weights_name}\n")
        if with_index and t.index_text is not None:
            (d / INDEX_NAME).write_text(t.index_text)
        if blocked:
            (d / cx.sidecar_name(t.stem)).write_text(cx.sidecar_text(t.stem, blocked))
        staged[t.dir] = str(d / t.weights_name)
    (staging / "INSTALL.md").write_text(install_text)
    return staged


def _config_library_dir(cfg: Mapping[str, Any], stem: str) -> Optional[Path]:
    """The bench build's library directory of the config, when it holds
    `stem`."""
    try:
        d = derive_paths(cfg).library_dir
    except (KeyError, TypeError):
        return None
    return (
        d if any((d / f"{stem}{x}").is_file() for x in (".dat", ".dat.zlib")) else None
    )


def blocklist_names(
    cfg: Mapping[str, Any], stem: str, library_dir: Optional[Path], device: int
) -> List[str]:
    """The names of the kernels of the served pool (table and bank, read
    from `library_dir`) that the config's blocklist names; every name of the
    blocklist when the library cannot be read."""
    blocklist = cx.config_blocklist(cfg)
    if not blocklist:
        return []
    if library_dir is not None:
        try:
            pool = bank_pool.config_pool(
                cfg, library_dir, stem, device, apply_blocklist=False
            )
            return sorted(
                {str(k["kernel_name"]) for k in pool.kernels if blocklist.blocks(k)}
            )
        except (OSError, ValueError, RuntimeError) as e:
            ui.warn(
                "blocklist",
                f"cannot read the pool of {stem} in {library_dir} ({e}); the "
                f"sidecar lists every kernel of the blocklist",
            )
    return sorted(blocklist.names)


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
    build_dir: Path, targets: List[Target], sha: str, blocked: Sequence[str] = ()
) -> Tuple[List[str], List[str]]:
    """(problems, checked target dirs) of the weights the build placed next
    to its libraries, and of their blocklist sidecars when `blocked` names
    kernels. A target the build has no library directory for is skipped."""
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
            parse_index(twi.read_text(idx_path)).as_dict() if idx_path.is_file() else {}
        )
        if entries.get(t.stem) != t.weights_name:
            problems.append(f"{idx_path} does not map {t.stem} to {t.weights_name}")
            continue
        placed = lib_dir / t.weights_name
        if not placed.is_file() or sha256_file(placed) != sha:
            problems.append(f"{placed} is missing or differs from the deployed file")
        sidecar = lib_dir / cx.sidecar_name(t.stem)
        if blocked and (
            not sidecar.is_file() or cx.read_sidecar(sidecar) != sorted(set(blocked))
        ):
            problems.append(
                f"{sidecar} is missing or differs from the deployed blocklist (a "
                f"hipBLASLt whose build does not copy {cx.SIDECAR_SUFFIX} files "
                f"next to its libraries serves the blocked kernels)"
            )
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


def _gate_gemms(
    gate: Gate,
    quiet: bool,
    identity: Optional[Any] = None,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
    blocklist: Optional[cx.Blocklist] = None,
) -> List[Dict[str, Any]]:
    """The gate GEMMs, their solution indices those of `identity`'s library
    (lib/kernel_identity.py) when given, repeated rows merged by
    `aggregation` (lib/repeats.py), without rows of kernels `blocklist`
    names."""
    import stage05_train as _train

    dirs = [str(d) for d in gate.csv_dirs if any(Path(d).glob("chunk_*.csv"))]
    if not dirs:
        return []
    cells, _ = _train.load_enriched_chunks(
        dirs, quiet, identity=identity, aggregation=aggregation, blocklist=blocklist
    )
    gemms = [g for gs in cells.values() for g in gs]
    if gate.plan is not None:
        try:
            ev.attach_plan(gemms, ev.read_plan(gate.plan), identity)
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
    alignment_veto: bool = True,
) -> Dict[str, Any]:
    """The model file `data` on the gate GEMMs, and the baseline bundle on
    the same GEMMs (packed as stage06 stages it, without the alignment veto
    unless `alignment_veto`); DeployError when the measurement cannot be
    made."""
    if tw is None:
        raise DeployError(
            "the deploy gate evaluates the model with the engine; install the "
            "tilewright module (pip install shared/tilewright/python)"
        )
    try:
        hw = hardware.device_hardware(cfg, gate.hardware_device)
    except (ValueError, RuntimeError, OSError) as e:
        raise DeployError(f"deploy gate: no device values ({e})") from None
    identity = (
        ev.library_identity(gate.library_dir, stem, cfg, gate.hardware_device)
        if gate.library_dir is not None
        else None
    )
    gemms = _gate_gemms(
        gate,
        quiet,
        identity,
        repeats.config_aggregation(cfg),
        cx.config_blocklist(cfg),
    )
    library = (
        ev.library_pool(gate.library_dir, stem, cfg, gate.hardware_device)
        if gate.library_dir is not None
        else None
    )
    if library is None and "drift_keep_ppm" in mlrec.read_header(data)["serving"]:
        raise DeployError(
            "deploy gate: a model served with the drift fallback is gated over the "
            "library pool it serves; give the gate a library directory (--library-dir)"
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
                allow_drift=library is not None,
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
    if measured is not None and measured.evaluation.degraded_cells:
        m["drift_degraded"] = measured.evaluation.degraded_cells
    if gate.rebench is not None:
        m.update(_rebench_numbers(measured, gate.pick_measurement, identity))
    if gate.baseline_train_dir is not None:
        base: Dict[str, Any] = {"train_dir": str(gate.baseline_train_dir)}
        try:
            bundle, _ = load_bundle(Path(gate.baseline_train_dir))
            bundle = deployed_serving(bundle, cfg, "the baseline bundle", quiet)
            base_data = mlrec.write_model(
                bundle, None, arch, hardware.arch_constants(arch), dtype
            )
            if not alignment_veto:
                base_data = auditlib.without_alignment_veto(base_data)
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
                    base.update(
                        _rebench_numbers(theirs, gate.pick_measurement, identity)
                    )
                base.update(
                    n=len(both),
                    sel_eff=_num(ev.geomean(b.sel_eff for _, b in both)),
                    bundle_sel_eff=_num(ev.geomean(a.sel_eff for a, _ in both)),
                    sha256=sha256_bytes(base_data),
                )
                if other.degraded_cells:
                    base["drift_degraded"] = other.degraded_cells
        m["baseline"] = base
    return m


def _rebench_numbers(
    measured: Optional[mr.Measured],
    rule: str,
    identity: Optional[ki.LibraryIdentity] = None,
) -> Dict[str, Any]:
    """The configured rule's numbers and the re-bench counts of a gate
    evaluation whose misses were re-benched, with its re-bench record named
    by `identity`, the library its solution indices name
    (`lib.miss_rebench.named_record`)."""
    if measured is None or measured.record is None:
        return {"pre_rebench": None, "rebench": None}
    rec = measured.record
    return {
        "pre_rebench": mr.pre_rebench(ev.summarize(measured.pre.results), rule),
        "n_rebenched": rec["n_rebenched"],
        "n_rebench_failed": rec["n_rebench_failed"],
        "n_context_skipped": rec["n_context_skipped"],
        "rebench": mr.named_record(rec, identity),
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


# ── per-cell non-regression gate ────────────────────────────────────────────

CELL_GATE_FILE = "cell_gate.json"


@dataclass
class CellGate:
    """The per-cell non-regression gate (lib.audit.cell_gate): the bundle
    against the bundle the run ships (`baseline_dir`, a round or a final
    retrain directory, whose stage06 staged model is scored on the same
    audit set), with the thresholds of `settings`
    (lib.audit.CELL_GATE_DEFAULTS); `record_only` measures and records it
    without keeping the bundle out."""

    baseline_dir: Path
    settings: Dict[str, Any]
    record_only: bool = False

    @property
    def baseline_name(self) -> str:
        d = Path(self.baseline_dir)
        if run_bundles.round_index(d) is not None:
            return f"{d.name}'s bundle"
        return f"the final retrain in {d.name}"


def measure_cell_gate(
    audited: Optional[Mapping[str, Any]],
    gate: CellGate,
    spec: auditlib.Spec,
    cfg: Mapping[str, Any],
    *,
    arch: str,
    dtype: str,
    tw: Any,
    alignment_veto: bool = True,
) -> Dict[str, Any]:
    """`lib.audit.cell_gate` of the bundle's audit record `audited` against
    the baseline's, which is scored here with `spec` (leaving out the shapes
    of the rounds it trained on, `lib.audit.bundle_trained_rounds`; without
    the alignment veto unless `alignment_veto`, `lib.audit.model_bytes`),
    with `baseline`: where its model came from and its record's status.
    `passed` is None, with why, when the audit set did not measure both."""
    base_dir = Path(gate.baseline_dir)
    path, source = auditlib.bundle_model(base_dir)
    out: Dict[str, Any] = {
        "settings": dict(gate.settings),
        "record_only": gate.record_only,
        "baseline": {
            "dir": str(base_dir),
            "name": gate.baseline_name,
            "model": str(path),
            "model_source": source,
        },
        "passed": None,
    }

    def unmeasured(why: str) -> Dict[str, Any]:
        out["reason"] = why
        return out

    if audited is None or not auditlib.is_measured(audited):
        st = (audited or {}).get("status_text") or (audited or {}).get("status")
        return unmeasured(f"the audit set did not measure this bundle ({st})")
    if not path.is_file():
        return unmeasured(f"the model of {gate.baseline_name} ({path}) does not exist")
    try:
        data = auditlib.model_bytes(
            path, arch, dtype, cfg, alignment_veto=alignment_veto
        )
        base = auditlib.evaluate(
            data,
            replace(spec, trained_rounds=auditlib.bundle_trained_rounds(base_dir)),
            cfg,
            hardware=hardware.device_hardware(cfg, spec.hardware_device),
            tw=tw,
        )
    except Exception as e:  # noqa: BLE001
        return unmeasured(
            f"{gate.baseline_name} could not be scored on the audit set "
            f"({type(e).__name__}: {e})"
        )
    out["baseline"].update(
        sha256=base.get("model_sha256"),
        status=base.get("status"),
        status_text=base.get("status_text"),
    )
    if not auditlib.is_measured(base):
        return unmeasured(
            f"the audit set did not measure {gate.baseline_name} "
            f"({base.get('status_text') or base.get('status')})"
        )
    result = auditlib.cell_gate(
        dict(base, what=base_dir.name),
        dict(audited, what="this bundle"),
        gate.settings,
        spec.bootstrap,
        official=spec.official,
    )
    out.update(result)
    return out


def cell_gate_decision(
    passed: bool, why: str, cells: Mapping[str, Any], gate: CellGate
) -> Tuple[bool, str]:
    """(install the model, why) once the per-cell gate `cells` has measured
    it: it keeps out a bundle with a finding unless `gate.record_only`; one
    it could not measure is left to the other gates."""
    name = gate.baseline_name
    lead = "" if why == "no gate" else f"{why}; "
    verdict = cells.get("passed")
    if verdict is True:
        return passed, f"{lead}per cell not worse than {name}: {cells['reason']}"
    if verdict is None:
        return passed, (
            f"{lead}the per-cell gate could not compare it with {name}: "
            f"{cells.get('reason')}"
        )
    if gate.record_only:
        return passed, (
            f"{lead}the per-cell gate against {name} would keep it out (recorded "
            f"only): {cells['reason']}"
        )
    return False, f"{lead}the per-cell gate against {name} keeps it out: " + str(
        cells["reason"]
    )


def _official_headline(audited: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The official exclusive number of the bundle's audit record (its
    protocol and headline), with v2's alongside when that is another."""
    g, proto = auditlib.group_view(audited, auditlib.EXCLUSIVE, audited.get("official"))
    if g is None:
        return None
    out: Dict[str, Any] = {"protocol": proto, "headline": auditlib.headline_of(g)}
    if proto != "v2":
        g2 = auditlib.evaluated_group(
            auditlib.protocol_record(audited, "v2"), auditlib.EXCLUSIVE
        )
        out["v2"] = auditlib.headline_of(g2) if g2 is not None else None
    return out


def _cell_gate_record(cells: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The per-cell gate's verdict and counts for the deploy record."""
    if cells is None:
        return None
    keys = (
        "passed",
        "reason",
        "record_only",
        "settings",
        "official",
        "baseline",
        "cell_rules",
        "n_groups",
        "n_cells_tested",
        "n_regressed",
        "n_floor",
        "n_tail",
        "n_tail_new",
        "n_tail_healed",
        "n_tail_persisting",
        "tail_groups",
        "n_reported",
        "not_compared",
    )
    return _json_ready({**{k: cells.get(k) for k in keys}, "file": CELL_GATE_FILE})


def repair_record(train_dir: Path) -> Optional[Dict[str, Any]]:
    """What a repair changed in the bundle of `train_dir` (its metrics.json
    `repair`: lib/collapse_repair.py `summary`), with the note that the
    audit GEMMs that flagged its cells also judged it
    (`selected_by_audit`, `selection_note`); None for a bundle no repair
    changed."""
    try:
        metrics = json.loads((Path(train_dir) / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    rep = metrics.get("repair") if isinstance(metrics, dict) else None
    if not isinstance(rep, dict):
        return None
    return _json_ready(
        dict(rep, selected_by_audit=True, selection_note=cm.SELECTION_NOTE)
    )


def print_cell_gate(cells: Mapping[str, Any], gate: CellGate) -> None:
    verdict = cells.get("passed")
    say = ui.ok if verdict is True else ui.warn
    state = {True: "passed", False: "failed", None: "not measured"}[verdict]
    say(
        "cells",
        f"per-cell gate against {gate.baseline_name}"
        + (" (recorded only)" if gate.record_only else "")
        + f": {state}: {cells.get('reason')}",
    )


# ── stage entry point ───────────────────────────────────────────────────────


def member_order_record(
    data: bytes,
    train_dir: Path,
    cfg: Mapping[str, Any],
    stem: str,
    library_dir: Path,
    device: int,
    quiet: bool,
) -> Optional[Dict[str, Any]]:
    """The member order (`lib.kernel_identity.member_orders`, keys grouped
    as the model `data` groups them) of the pool `library_dir` serves, and
    the keys whose order differs from the build the bundle trained with
    (stage05's member_order.json in `train_dir`), which are warned about;
    None when the library cannot be read."""
    try:
        pool = ev.library_pool(library_dir, stem, cfg, device)
        identity = ev.library_identity(library_dir, stem, cfg, device)
    except (OSError, ValueError, RuntimeError) as e:
        ui.warn("members", f"cannot read the member order of {library_dir}: {e}")
        return None
    key_of, _ = auditlib.candidate_keys(pool, mlrec.read_header(data))
    now = ki.member_orders(pool, lambda k: key_of[int(k["sol_idx_global"])])
    out: Dict[str, Any] = {
        "library_dir": str(library_dir),
        "library_digest": identity.digest,
        "keys": now,
        "training": None,
        "differs": [],
    }
    try:
        path = bundle_file(train_dir, ki.MEMBER_ORDER_FILE)
        trained = json.loads(path.read_text())
        then = dict(trained["keys"])
    except (OSError, ValueError, KeyError, TypeError):
        if not quiet:
            ui.info(
                "members",
                f"{train_dir} records no member order (its stage05 read no "
                f"library pool, or ran before member orders were recorded); the "
                f"deploy cannot compare it",
            )
        return out
    keys = set(then) | set(now)
    differs = sorted(k for k in keys if then.get(k) != now.get(k))
    out["training"] = {"library_digest": trained.get("library_digest")}
    out["differs"] = differs
    if differs:
        ui.warn(
            "members",
            f"{len(differs)} of the {len(keys)} candidate keys with several members "
            f"order them differently in {library_dir} (library digest "
            f"{identity.digest}) than in the build the bundle trained with "
            f"({trained.get('library_digest')}), e.g. {differs[0]}: the runtime "
            f"serves a key's first member whose predicates pass, so it can serve "
            f"another kernel of those keys than the one the model trained on",
        )
    return out


def tier_record(
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    stem: str,
    library_dir: Optional[Path],
    device: int,
    quiet: bool,
) -> Optional[Dict[str, Any]]:
    """The deploy record's `whitelist_tier`: the bundle's tier
    (lib/whitelist_tier.py; None without one) and, with the library the
    model is served from, how the tier meets its pool (`served_pool`,
    `whitelist_tier.pool_check`); tier keys the pool lacks are warned
    about."""
    record = bundle.get(wt.RECORD)
    if not record:
        return None
    out: Dict[str, Any] = {**dict(record), "served_pool": None}
    if library_dir is None:
        return out
    try:
        pool = bank_pool.config_pool(cfg, library_dir, stem, device)
    except (OSError, ValueError, RuntimeError) as e:
        ui.warn("tier", f"cannot read the pool of {library_dir}: {e}")
        return out
    served = wt.pool_check(bundle, ev.pool_from_kernels(pool.kernels))
    out["served_pool"] = served
    keep = (bundle.get("serving") or {}).get("drift_keep_ppm")
    if keep is not None:
        rec = drift_record(bundle, pool.kernels)
        out["drift_fallback"] = rec
        degraded = rec["degraded"]
        if degraded and not quiet:
            ui.warn(
                "drift_fallback",
                f"the pool of {library_dir} lacks trained whitelist tier keys of "
                f"{len(degraded)} cells, which the drift fallback serves degraded "
                f"(keep_ppm {keep}): build the tier for this pool with "
                f"scripts/build_whitelist_tier.py",
            )
    built = (record.get("pool") or {}).get("keys_digest")
    if served and served["tier_keys_missing"] and not quiet:
        bare = served["cells_without_tier_keys"]
        ui.warn(
            "tier",
            f"{served['tier_keys_missing']} of the {record.get('n_keys')} "
            f"whitelist tier keys are not in the pool of {library_dir}, another "
            f"than the tier was built for: cells rank the tier keys the pool "
            f"has, {len(bare)} keep none and serve through the cascade. Build "
            f"the tier for this pool with scripts/build_whitelist_tier.py",
        )
    elif served and built and served["keys_digest"] != built and not quiet:
        ui.info(
            "tier",
            f"the pool of {library_dir} holds keys the whitelist tier was not "
            f"chosen from: they rank where none of a cell's tier keys is "
            f"feasible, and behind the tier for a caller that asks for more "
            f"kernels than it holds",
        )
    return out


def launch_record(
    bundle: Mapping[str, Any], build_dir: Optional[Path], quiet: bool
) -> Dict[str, Any]:
    """The deploy record's `launch` (see the module docstring)."""
    benched = bundle.get("bench_launch")
    built = launch_settings.heuristic(build_dir) if build_dir is not None else None
    digest = launch_settings.comparable_digest(built)
    known = list((benched or {}).get("heuristics") or ())
    matches = None if digest is None or not known else digest in known
    if matches is False and not quiet:
        ui.warn(
            "launch",
            f"the launch-rule sources of {build_dir} (launch heuristic "
            f"{launch_settings.short(digest)}) differ from those of the bundle's "
            f"training benches ({launch_settings.describe_digests(benched)}): the "
            f"runtime may launch the picks otherwise than they were measured",
        )
    return {"benched": benched, "build_heuristic": digest, "build_matches": matches}


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
    audit: Optional[auditlib.Spec] = None,
    origami_reference: bool = False,
    quiet: bool = False,
    library_dir: Optional[Path] = None,
    hardware_device: int = 0,
    cell_gate: Optional[CellGate] = None,
    score_without_alignment_veto: bool = False,
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
            audit=audit,
            origami_reference=origami_reference,
            quiet=quiet,
            library_dir=library_dir,
            hardware_device=hardware_device,
            cell_gate=cell_gate,
            score_without_alignment_veto=score_without_alignment_veto,
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
    audit: Optional[auditlib.Spec],
    origami_reference: bool,
    quiet: bool,
    library_dir: Optional[Path],
    hardware_device: int,
    cell_gate: Optional[CellGate] = None,
    score_without_alignment_veto: bool = False,
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
    if repo is not None and not (repo / wl.LOGIC_REL).is_dir():
        raise DeployError(
            f"{repo} is not a rocm-libraries checkout with hipBLASLt's library "
            f"logic ({wl.LOGIC_REL})"
        )
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
    bundle = deployed_serving(bundle, cfg, "the bundle", quiet)
    preflight = drift_preflight(
        bundle,
        cfg,
        stem,
        library_dir or _config_library_dir(cfg, stem),
        hardware_device,
    )
    labels = sorted(bundle["models"])
    provenance_mode = str(cfg_get(cfg, "deploy.provenance"))
    provenance = prov.deployed_provenance(bundle.get("provenance"), cfg)
    data = mlrec.write_model(
        bundle, None, arch, hardware.arch_constants(arch), dtype, provenance
    )
    parsed = mlrec.read_model(data)
    n_cells, n_splits = len(parsed["cells"]), len(parsed["splits"])
    if [c["label"] for c in parsed["cells"]] != labels:
        raise DeployError("the written model does not round-trip its cell labels")
    engine_ok = tw is not None
    if engine_ok:
        engine_check(data, n_cells, n_splits, tw)
    sha = sha256_bytes(data)
    veto = (parsed.get("serving") or {}).get("alignment_veto")
    # The bytes the gates and the audit score: without the veto for a
    # round's bundle (--score-without-alignment-veto), else the staged ones.
    plain = auditlib.without_alignment_veto(data) if veto else data
    scored = plain if score_without_alignment_veto else data
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
        if parsed["version"] == 3:
            serving = parsed["serving"]
            trained = parsed["trained_view_keys"]
            ui.info(
                "serve",
                f"trained on the table{' and the bank' if serving['bank'] else ''}; "
                + (
                    f"trained-key gate over {len(trained):,} view keys"
                    if trained
                    else "no trained-key gate"
                )
                + (
                    "; family-wise non-temporal rule"
                    if serving["non_temporal_family_rule"]
                    else ""
                ),
            )
            if bundle.get(wt.RECORD):
                ui.info(
                    "serve",
                    f"whitelist tier: {wt.record_text(bundle[wt.RECORD])}; "
                    f"cascade {serving['cascade_keys']} where none suits",
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
            scored,
            labels,
            cfg=cfg,
            arch=arch,
            dtype=dtype,
            stem=stem,
            gate=gate,
            tw=tw,
            quiet=quiet,
            alignment_veto=not score_without_alignment_veto,
        )
        passed, why = gate_decision(measured, gate)
    deploy_gate = {"passed": passed, "reason": why}

    audited: Optional[Dict[str, Any]] = None
    if audit is not None:
        audited = auditlib.run_for_stage(
            out_dir / auditlib.RESULT_FILE,
            scored,
            replace(audit, library_stem=audit.library_stem or stem),
            cfg,
            hardware=hardware.device_hardware(cfg, audit.hardware_device),
            tw=tw,
            what="stage06: the candidate bundle",
            source=str(train_dir),
            quiet=quiet,
        )

    cells: Optional[Dict[str, Any]] = None
    if cell_gate is not None:
        if audit is None:
            raise DeployError(
                "the per-cell gate measures on the audit set: --audit-dir"
            )
        cells = measure_cell_gate(
            audited,
            cell_gate,
            replace(audit, library_stem=audit.library_stem or stem),
            cfg,
            arch=arch,
            dtype=dtype,
            tw=tw,
            alignment_veto=not score_without_alignment_veto,
        )
        passed, why = cell_gate_decision(passed, why, cells, cell_gate)
        write_atomic(
            out_dir / CELL_GATE_FILE,
            (json.dumps(_json_ready(cells), indent=2) + "\n").encode(),
        )
        if not quiet:
            print_cell_gate(cells, cell_gate)
    if gate is not None:
        assert measured is not None
        measured.update(
            target=gate.target,
            fallback=gate.fallback,
            passed=passed,
            reason=why,
            cell_gate=_cell_gate_record(cells),
        )
        write_atomic(
            out_dir / "gate.json",
            (json.dumps(_json_ready(measured), indent=2) + "\n").encode(),
        )
        if not quiet:
            print_gate(measured, origami_reference)
    if not quiet and (gate is not None or cells is not None):
        (ui.ok if passed else ui.warn)(
            "gate", ("deploying: " if passed else "not deploying: ") + why
        )

    targets = plan_targets(deploy_targets(cfg), repo)
    blocked = blocklist_names(
        cfg, stem, library_dir or _config_library_dir(cfg, stem), hardware_device
    )
    if blocked and not quiet:
        ui.info(
            "blocklist",
            f"{len(blocked)} kernel(s) of the pool compute wrong results; the "
            f"sidecar {cx.sidecar_name(stem)} names them",
        )
    applied = cx.config_blocklist(cfg)
    trained = bundle.get("blocklist")
    if (
        isinstance(trained, dict)
        and trained.get("digest") != cx.names_digest(applied)
        and not quiet
    ):
        ui.warn(
            "blocklist",
            f"the bundle's labels were filtered by another blocklist "
            f"({trained.get('kernels')} kernel(s)) than this deploy's "
            f"({len(applied or ())}): the sidecar leaves out other kernels "
            f"than its training did",
        )
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
            blocked=blocked,
        ),
        blocked=blocked,
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
        manifest["n_trained_keys"] = len(parsed["trained_view_keys"] or [])
        manifest["clip"] = parsed["clip"]
        manifest["whitelist_tier"] = bundle.get(wt.RECORD)
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

    served_from = library_dir
    if build_dir is not None:
        built = build_library_dir(build_dir, targets[0])
        if any((built / f"{stem}{x}").is_file() for x in (".dat", ".dat.zlib")):
            served_from = built
    members = (
        member_order_record(
            data, train_dir, cfg, stem, served_from, hardware_device, quiet
        )
        if served_from is not None
        else None
    )
    tier = tier_record(bundle, cfg, stem, served_from, hardware_device, quiet)
    record: Dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "train_dir": str(train_dir),
        "arch": arch,
        "weight_dtype": dtype,
        "feature_catalog_hash": parsed["feature_catalog_hash"],
        "sha256": sha,
        "models_sha256": sha256_file(train_dir / "models.pt"),
        "size": len(data),
        "payload_crc32": parsed["payload_crc32"],
        "provenance": {
            "mode": provenance_mode,
            "recorded": dict(parsed.get("provenance") or {}),
        },
        "n_cells": n_cells,
        "n_splits": n_splits,
        "staged_model": staged[targets[0].dir],
        "repo": str(repo) if repo else None,
        "engine_checked": engine_ok,
        "deployed": passed,
        "deploy_gate": deploy_gate,
        "gate": _gate_record(measured),
        "cell_gate": _cell_gate_record(cells),
        "repair": repair_record(train_dir),
        "audit": (
            None
            if audited is None
            else {
                "status": audited.get("status"),
                "reason": audited.get("reason"),
                "status_text": audited.get("status_text"),
                "headline": audited.get("headline"),
                "official": _official_headline(audited),
            }
        ),
        "member_order": members,
        "identity_rule": bundle.get("identity_rule"),
        "whitelist_tier": tier,
        "drift_fallback": preflight,
        "launch": launch_record(bundle, build_dir, quiet),
        "blocklist": {
            "kernels": len(blocked),
            "sha256": (
                sha256_bytes(cx.sidecar_text(stem, blocked).encode())
                if blocked
                else None
            ),
            "digest": cx.names_digest(applied),
            "trained_with": bundle.get("blocklist"),
        },
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
                "blocklist": cx.sidecar_name(t.stem) if blocked else None,
                "blocklist_sha256": (
                    sha256_bytes(cx.sidecar_text(t.stem, blocked).encode())
                    if blocked
                    else None
                ),
            }
            for t in targets
        ],
    }
    if veto:
        record.update(
            alignment_veto=dict(veto),
            veto_free_sha256=sha256_bytes(plain),
            scored_without_alignment_veto=bool(score_without_alignment_veto),
        )

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
                sidecar = t.weights_dir / cx.sidecar_name(t.stem)
                if blocked:
                    txn.put(sidecar, cx.sidecar_text(t.stem, blocked).encode())
                else:
                    txn.remove(sidecar)
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
                    txn.guard(build_library_dir(build_dir, t) / cx.sidecar_name(t.stem))
                cmd = ["cmake", "--build", str(build_dir), "--target", build_target]
                if build_jobs:
                    cmd += ["--parallel", str(build_jobs)]
                rc = run_appending(cmd, out_dir / "build.log", build_target, quiet)
                if rc != 0:
                    raise DeployError(f"building {build_target} failed (rc={rc})")
                if not blocked:
                    for t in targets:
                        stale = build_library_dir(build_dir, t) / cx.sidecar_name(
                            t.stem
                        )
                        stale.unlink(missing_ok=True)
                problems, checked = verify_colocated(build_dir, targets, sha, blocked)
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
        help="stage02 plan.json of the gate GEMMs",
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
    auditlib.add_arguments(ap)
    c = ap.add_argument_group("per-cell non-regression gate (lib.audit.cell_gate)")
    c.add_argument(
        "--cell-gate-baseline",
        type=Path,
        default=None,
        help="directory of the bundle the run ships (a round, or a final "
        "retrain): its staged model is scored on the --audit-dir audit set "
        "beside this bundle, which is kept out when a cell regresses against it",
    )
    defaults = auditlib.CELL_GATE_DEFAULTS
    c.add_argument(
        "--cell-tolerance",
        type=float,
        default=defaults["cell_tolerance"],
        help="a cell significantly worse by more than this keeps the bundle out",
    )
    c.add_argument(
        "--cell-floor",
        type=float,
        default=defaults["cell_floor"],
        help="a cell confidently fallen below it from at or above it keeps the "
        "bundle out (0: off)",
    )
    c.add_argument(
        "--cell-gate-alpha",
        type=float,
        default=defaults["alpha"],
        help="family-wise level of the cell rules over the cells they judge "
        "(their sign flips are --audit-bootstrap)",
    )
    c.add_argument(
        "--tail-floor",
        type=float,
        default=defaults["tail_floor"],
        help="a pick below it is in the tail (0: the tail rule off)",
    )
    c.add_argument(
        "--tail-collapse",
        type=float,
        default=defaults["tail_collapse"],
        help="a new pick below --tail-floor counts when it is at most this "
        "fraction of the shipped pick's efficiency (1: every step across the "
        "floor)",
    )
    c.add_argument(
        "--tail-budget",
        type=int,
        default=defaults["tail_budget"],
        help="new picks below --tail-floor the bundle may add over every group, "
        "never offset by the collapses it heals",
    )
    c.add_argument(
        "--cell-gate-groups",
        choices=auditlib.CELL_GATE_GROUPS,
        default=defaults["groups"],
        help="the audit groups whose cells the cell rules judge (the tail rule "
        "judges every group)",
    )
    c.add_argument(
        "--cell-gate-record-only",
        action="store_true",
        help="measure and record the per-cell gate without keeping the bundle out",
    )
    ap.add_argument(
        "--score-without-alignment-veto",
        action="store_true",
        help="the deploy gate, the audit record and the per-cell gate score the "
        "model and the bundles they compare it with without the alignment veto "
        "(deploy.alignment_veto), as a run scores its rounds' bundles; the "
        "staged file carries it",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print Origami's numbers on the gate GEMMs (an external "
        "reference; they never decide the deploy)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.cell_gate_baseline is not None:
        if args.audit_dir is None:
            ap.error("--cell-gate-baseline measures on the audit set: give --audit-dir")
        if not 0 < args.cell_gate_alpha < 1:
            ap.error("--cell-gate-alpha is in (0, 1)")
        if not 0 < args.tail_collapse <= 1 or args.tail_budget < 0:
            ap.error("--tail-collapse is in (0, 1] and --tail-budget at least 0")
        if args.audit_bootstrap < auditlib.min_flips(args.cell_gate_alpha):
            ap.error(
                f"--audit-bootstrap {args.audit_bootstrap} leaves the per-cell gate's "
                f"cell tests unable to reach --cell-gate-alpha "
                f"{args.cell_gate_alpha:g}: give at least "
                f"{auditlib.min_flips(args.cell_gate_alpha)}"
            )
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
    audit = None
    if args.audit_dir is not None:
        audit = auditlib.Spec(
            audit_dir=args.audit_dir,
            library_dir=args.library_dir,
            library_stem=args.library_stem or "",
            hardware_device=args.hardware_device,
            pick_measurement=args.pick_measurement,
            target=args.audit_target,
            bootstrap=args.audit_bootstrap,
            trained_rounds=auditlib.trained_rounds(
                args,
                auditlib.rounds_until(args.train_dir.resolve().parent, inclusive=True),
            ),
            official=args.audit_official,
        )
    cell_gate = None
    if args.cell_gate_baseline is not None:
        cell_gate = CellGate(
            baseline_dir=args.cell_gate_baseline,
            settings={
                "cell_tolerance": args.cell_tolerance,
                "cell_floor": args.cell_floor,
                "alpha": args.cell_gate_alpha,
                "tail_floor": args.tail_floor,
                "tail_collapse": args.tail_collapse,
                "tail_budget": args.tail_budget,
                "groups": args.cell_gate_groups,
            },
            record_only=args.cell_gate_record_only,
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
        audit=audit,
        origami_reference=args.origami_reference,
        quiet=args.quiet,
        library_dir=args.library_dir,
        hardware_device=args.hardware_device,
        cell_gate=cell_gate,
        score_without_alignment_veto=args.score_without_alignment_veto,
    )


if __name__ == "__main__":
    sys.exit(main())
