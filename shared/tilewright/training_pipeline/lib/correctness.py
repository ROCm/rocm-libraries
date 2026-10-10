# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernels that compute wrong results, and where the pipeline leaves them
out.

A blocklist names kernels by the name their library's logic file gives them
(`.dat` `name`: the solution name hipblaslt-bench prints and TensileLite's
`ContractionSolution::name()` returns; `kernel_name` of
`lib.dat.kernel_dat_info`), with the kernel id (lib/kernel_identity.py) and
signature where it was resolved against a library, and why it is blocked. A
kernel id ends in a hash of that name as the identity rule reads it (its
canonical name), so an artifact that names kernels by id alone (an enriched
folder's kernels.json, a plan) is checked by that hash
(`Blocklist.blocks_id`). A name blocks the kernels of its canonical name,
an alias group whole, since they run the same code. Names are
framework-specific: they live in the pipeline's records and in the
integration-side sidecar, never in a model.

Sources, the config's `correctness` section:

  lists      files of kernels known to compute wrong results: a list, one
             kernel per line as `<class><TAB><name>` or the name alone (`#`
             starts a comment), or a blocklist the gate wrote (JSON). An
             entry without a path separator or suffix names a list shipped
             in `blocklists/<entry>.tsv`.
  blocklist  the gate's blocklist of the run (`blocklist.json` of
             stages/stage00_correctness_gate.py, lib/correctness_gate.py),
             which the orchestrator sets when `correctness.enabled`: a copy
             per invocation, which its config snapshot names and its stage
             markers record (`blocklist_record`).

`config_blocklist` is their union; the run pins the contents of the lists
(`list_digests`), and stages 4 and 5 record the digest of the one they
applied (`applied_record`), which the deploy record repeats. A bundle
deployed earlier serves with the blocklist of its own deploy, which stages 7
and 8 validate it with (`bundle_blocklist`). A blocked kernel

  * is not in any pool built from the config (`lib.bank_pool.config_pool`):
    the planner's kernel lists, stage05's pool and served candidates, the
    deploy gate, the audit set's evaluations and stages 7 and 8 rank the
    pool without it, as the runtime does with the sidecar below;
  * has no rows from stage04 on, so it is never a GEMM's winner or a label;
  * has its rows dropped by the loaders that are given the blocklist
    (stage05's `load_enriched_chunks`, `lib.evaluate.load_timings`,
    `lib.run_analysis.load_bench`), for folders enriched before it was
    blocked;
  * is listed by stage06 in the sidecar `<library stem>.tilewright.blocklist`
    (`SIDECAR_SUFFIX`) it stages and installs next to the model for every
    deploy target: the kernel names of the served pool (the library's table
    and bank) that are blocked, those of every member of a blocked alias
    group too, one per line. TensileLite's tilewright adapter reads the
    sidecar of a library stem from the directory of its tilewright_index and
    leaves those kernels out of the pools it ranks, so a key whose served
    member is blocked is served by its next member and a key whose members
    are all blocked is no candidate, as in the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import dat
from . import kernel_identity as ki
from . import ui
from .fs import atomic_write_text

SCHEMA = "tilewright_blocklist_v1"
BLOCKLIST_FILE = "blocklist.json"
SIDECAR_SUFFIX = ".tilewright.blocklist"
LIST_SUFFIX = ".tsv"
LISTS_DIR = Path(__file__).resolve().parent.parent / "blocklists"


@dataclass(frozen=True)
class Entry:
    """One blocked kernel: its `name`, why (`reason`: a class of a list, or
    what the gate saw), its kernel id and signature where known, and the
    file or check it comes from (`source`)."""

    name: str
    reason: str = ""
    kernel_id: Optional[str] = None
    signature: Optional[str] = None
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kernel_id": self.kernel_id,
            "signature": self.signature,
            "reason": self.reason,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], source: str = "") -> "Entry":
        name = str(d.get("name") or "")
        if not name:
            raise ValueError("a blocklist entry has no name")
        kid = d.get("kernel_id")
        sig = d.get("signature")
        return cls(
            name=name,
            reason=str(d.get("reason") or ""),
            kernel_id=str(kid) if kid else None,
            signature=str(sig) if sig else None,
            source=str(d.get("source") or source),
        )


class Blocklist:
    """Blocked kernels, the first entry of a name kept. A name blocks every
    kernel of its canonical name (`lib.dat.canonical_name`), its alias
    group (lib/kernel_identity.py) whole."""

    def __init__(self, entries: Iterable[Entry] = ()) -> None:
        kept: Dict[str, Entry] = {}
        for e in entries:
            kept.setdefault(e.name, e)
        self.entries: Tuple[Entry, ...] = tuple(kept.values())
        self.names: FrozenSet[str] = frozenset(kept)
        self.canonical: FrozenSet[str] = frozenset(dat.canonical_name(n) for n in kept)
        self.hashes: FrozenSet[str] = frozenset(
            ki.name_hash(n) for n in self.names | self.canonical
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def blocks_name(self, name: Any) -> bool:
        return bool(name) and (
            str(name) in self.names or dat.canonical_name(name) in self.canonical
        )

    def blocks_id(self, kid: Any) -> bool:
        """Whether a kernel id names a blocked kernel (by its name hash)."""
        if not kid or ":" not in str(kid):
            return False
        return str(kid).rsplit(":", 1)[1] in self.hashes

    def blocks(self, kernel: Mapping[str, Any]) -> bool:
        """Whether a kernel (`kernel_dat_info`, or a pool entry of
        `lib.evaluate.pool_from_kernels`, which carries its kernel id) is
        blocked."""
        name = kernel.get("kernel_name")
        if name:
            return (
                str(name) in self.names or ki.canonical_name(kernel) in self.canonical
            )
        return self.blocks_id(kernel.get("kernel_id"))

    def kept(self, kernels: Iterable[Mapping[str, Any]]) -> List[Any]:
        return [k for k in kernels if not self.blocks(k)]

    def blocked_indices(self, ids: Mapping[int, str]) -> Set[int]:
        """The solution indices of `ids` ({index: kernel id}, such as
        `LibraryIdentity.ids` or a record's) that name blocked kernels."""
        return {int(s) for s, kid in ids.items() if self.blocks_id(kid)}

    def merged(self, other: "Blocklist") -> "Blocklist":
        return Blocklist(self.entries + other.entries)

    def to_doc(self, **extra: Any) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            **extra,
            "entries": [e.to_dict() for e in self.entries],
        }


def shipped_list(entry: Any) -> bool:
    """Whether a `correctness.lists` entry names a shipped list (no path
    separator, no suffix) rather than a file."""
    text = str(entry)
    return "/" not in text and os.sep not in text and "." not in text


def list_path(entry: Any) -> Path:
    """The file of a `correctness.lists` entry: a path, or the name of a
    shipped list (`LISTS_DIR/<name>.tsv`)."""
    if shipped_list(entry):
        return LISTS_DIR / f"{entry}{LIST_SUFFIX}"
    return Path(str(entry))


def read_list(path: Any) -> Blocklist:
    """The kernels a list or a blocklist file names. ValueError (naming the
    file) when it cannot be read or is malformed."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise ValueError(f"cannot read the blocklist {path}: {e}") from None
    if text.lstrip().startswith("{"):
        try:
            doc = json.loads(text)
            if doc.get("schema") != SCHEMA:
                raise ValueError(f"not a {SCHEMA} file")
            return Blocklist(
                Entry.from_dict(d, str(path)) for d in doc.get("entries") or []
            )
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"{path}: {e}") from None
    entries = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("\t") if f.strip()]
        if len(fields) > 2 or any(" " in f for f in fields[-1:]):
            raise ValueError(
                f"{path}:{n}: expected `<class><TAB><name>` or a name, got {raw!r}"
            )
        reason = fields[0] if len(fields) == 2 else ""
        entries.append(Entry(name=fields[-1], reason=reason, source=str(path)))
    return Blocklist(entries)


def write_blocklist(path: Path, blocklist: Blocklist, **extra: Any) -> None:
    atomic_write_text(
        Path(path), json.dumps(blocklist.to_doc(**extra), indent=1) + "\n"
    )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        return f"unreadable ({type(e).__name__})"


def list_digests(entries: Iterable[Any]) -> Dict[str, str]:
    """{`correctness.lists` entry: sha256 of the file it names}: what a run
    pins of its lists, whose contents decide which kernels leave its pool."""
    return {str(e): _sha256(list_path(e)) for e in entries}


def names_digest(blocklist: Optional[Blocklist]) -> Optional[str]:
    """sha256 of the kernel names `blocklist` blocks, sorted, one per line;
    None when it blocks none. Blocklists with one digest leave out the same
    kernels, whatever files named them."""
    if not blocklist:
        return None
    text = "".join(f"{n}\n" for n in sorted(blocklist.names))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def applied_record(blocklist: Optional[Blocklist]) -> Dict[str, Any]:
    """What a stage records of the blocklist it applied: its `digest`
    (`names_digest`) and the number of `kernels` it names."""
    return {"digest": names_digest(blocklist), "kernels": len(blocklist or ())}


def blocklist_record(path: Any) -> Dict[str, Any]:
    """The blocklist file `path` as a stage marker records it: the file, its
    sha256 and how many kernels it names. ValueError when it cannot be
    read."""
    return {
        "file": str(path),
        "sha256": _sha256(Path(path)),
        "kernels": len(read_list(path)),
    }


def lists_blocklist(cfg: Optional[Mapping[str, Any]]) -> Blocklist:
    """The kernels the config's `correctness.lists` name, without the
    invocation's blocklist: what holds before the canary and the gate write
    it. ValueError when a list cannot be read."""
    out = Blocklist()
    for entry in ((cfg or {}).get("correctness") or {}).get("lists") or []:
        out = out.merged(read_list(list_path(entry)))
    return out


def config_files(cfg: Optional[Mapping[str, Any]]) -> List[Path]:
    """The blocklist files of a config: its `correctness.lists`, then its
    `correctness.blocklist`."""
    sec = (cfg or {}).get("correctness") or {}
    files = [list_path(e) for e in sec.get("lists") or []]
    if sec.get("blocklist"):
        files.append(Path(str(sec["blocklist"])))
    return files


_CACHE: Dict[Tuple[Any, ...], Blocklist] = {}
_WARNED: List[bool] = []


def config_blocklist(cfg: Optional[Mapping[str, Any]]) -> Optional[Blocklist]:
    """The union of the config's blocklist files; None when it names none.
    ValueError when one is missing or cannot be read. A config whose gate is
    on but names no gate blocklist (a run's own config rather than the
    snapshot the orchestrator resolved, which names it) applies its lists
    alone, with a warning."""
    sec = (cfg or {}).get("correctness") or {}
    if sec.get("enabled") and not sec.get("blocklist") and not _WARNED:
        _WARNED.append(True)
        ui.warn(
            "blocklist",
            "correctness.enabled, but the config names no gate blocklist "
            "(correctness.blocklist): only correctness.lists apply; read a run "
            "with the config snapshot of its run directory (configs/)",
        )
    files = config_files(cfg)
    if not files:
        return None
    key = []
    for p in files:
        try:
            st = p.stat()
        except OSError as e:
            raise ValueError(
                f"the config's correctness section names {p}, which cannot be "
                f"read: {e}"
            ) from None
        key.append((str(p.resolve()), st.st_size, st.st_mtime_ns))
    hit = _CACHE.get(tuple(key))
    if hit is not None:
        return hit
    out = Blocklist()
    for p in files:
        out = out.merged(read_list(p))
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[tuple(key)] = out
    return out


def resolved(blocklist: Blocklist, kernels: Sequence[Mapping[str, Any]]) -> Blocklist:
    """`blocklist`'s entries of the kernels of `kernels` (`kernel_dat_info`
    entries), each with its kernel id and signature."""
    by_name = {str(k.get("kernel_name") or ""): k for k in kernels}
    out = []
    for e in blocklist.entries:
        k = by_name.get(e.name)
        if k is None:
            continue
        out.append(
            Entry(
                name=e.name,
                reason=e.reason,
                kernel_id=ki.kernel_id(k),
                signature=ki.signature_label(k),
                source=e.source,
            )
        )
    return Blocklist(out)


# ── the integration-side sidecar ─────────────────────────────────────────────


def sidecar_name(stem: str) -> str:
    return f"{stem}{SIDECAR_SUFFIX}"


def sidecar_text(stem: str, names: Iterable[str]) -> str:
    """The sidecar of library `stem`: a comment, then one kernel name per
    line, sorted."""
    head = (
        f"# Kernels of {stem} (and its bank) that tilewright never offers: "
        f"one name per line, as the library's logic file names them."
    )
    return "\n".join([head, *sorted(set(names))]) + "\n"


def read_sidecar(path: Any) -> List[str]:
    """The kernel names of a sidecar: its lines without blanks and `#`
    comments, as TensileLite reads them."""
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def sidecar_blocklist(path: Any) -> Blocklist:
    """The kernels a sidecar names (`read_sidecar`), as a Blocklist.
    ValueError when it cannot be read."""
    try:
        names = read_sidecar(path)
    except OSError as e:
        raise ValueError(f"cannot read the blocklist sidecar {path}: {e}") from None
    return Blocklist(Entry(name=n, reason="sidecar", source=str(path)) for n in names)


def deployed_blocklist(record: Mapping[str, Any], stem: str) -> Optional[Blocklist]:
    """The blocklist the bundle of stage06's deploy record `record` serves
    library `stem` with: the sidecar staged next to its model for that stem
    (`sidecar_blocklist`; an empty Blocklist when it staged none). None when
    the record predates sidecars (it has no `blocklist`) or has no target of
    `stem`. ValueError when the sidecar cannot be read."""
    if "blocklist" not in record:
        return None
    for t in record.get("targets") or []:
        if t.get("library_stem") != stem:
            continue
        if not t.get("blocklist"):
            return Blocklist()
        model = Path(str(t.get("staged") or record.get("staged_model") or ""))
        return sidecar_blocklist(model.parent / str(t["blocklist"]))
    return None


def bundle_blocklist(train_dir: Any, stem: str) -> Tuple[Optional[Blocklist], str]:
    """(the blocklist the bundle of stage05 directory `train_dir` serves
    library `stem` with, where it was read): the sidecar its stage06 staged
    (`deployed_blocklist` of `<bundle>/stage06/deploy_record.json`), else
    the blocklist its stage05 applied (`config_blocklist` of the config
    snapshot its metrics.json names; an empty one when that names none).
    (None, why) when neither can be read."""
    import yaml

    from .bundle import bundle_file

    train_dir = Path(train_dir)
    record = train_dir.parent / "stage06" / "deploy_record.json"
    try:
        rec = json.loads(record.read_text())
        found = deployed_blocklist(rec, stem) if isinstance(rec, dict) else None
    except (OSError, ValueError) as e:
        if record.is_file():
            return None, str(e)
        found = None
    if found is not None:
        return found, f"the sidecar its deploy staged ({record})"
    try:
        metrics = json.loads(bundle_file(train_dir, "metrics.json").read_text())
        snapshot = Path(str(metrics["args"]["config_yaml"]))
        cfg = yaml.safe_load(snapshot.read_text()) or {}
        return (
            config_blocklist(cfg) or Blocklist(),
            f"the config its stage05 ran with ({snapshot})",
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        return None, f"neither its deploy nor its stage05 config tells ({e})"
