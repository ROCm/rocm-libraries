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
an alias group whole, since they run the same code, and every kernel whose
name equals it once both drop the tokens builds of TensileLite spell
differently for one kernel (`blocklist_key`), as TensileLite reads the
sidecar: a list or sidecar written from one build names the same kernels in
another. A kernel id cannot be matched that way, since its hash covers those
tokens, so where the library is known its solutions are matched through
their kernels' names (`Blocklist.blocked_indices` with the kernels of a
`lib.kernel_identity.LibraryIdentity`). Names are framework-specific: they
live in the pipeline's records and in the integration-side sidecar, never in
a model.

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
    leaves those kernels out of the pools it ranks (matching names by
    `blocklist_key`, as a Blocklist does), so a key whose served
    member is blocked is served by its next member and a key whose members
    are all blocked is no candidate, as in the pipeline.

A guarded entry (`Entry.guards`, from the gate's static check under
`correctness.guard_unguarded_custom`) blocks its kernel only for problems
whose free sizes its guards deny (`Guard`: the free size M or N and the
least value admitted): such a kernel stays in the pools built from the
config, marked with its guards (`POOL_GUARDS`), and every stage that plans,
benches, labels or ranks a problem leaves it out where a guard denies it
(`guards_admit`), as TensileLite does with the guards sidecar
`<library stem>.tilewright.guards` (`GUARDS_SUFFIX`) that stage06 stages
beside the blocklist sidecar for a model it flags as trained with guards.
The blocklist sidecar still names the kernel, so an integration or a model
without guards leaves it out whole (`Blocklist.whole`). A name some entry
blocks whole is never narrowed by a guard of another.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
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
from .feasibility import GUARD_FREE_SIZES, guard_bounds, guard_denies
from .fs import atomic_write_text

SCHEMA = "tilewright_blocklist_v1"
BLOCKLIST_FILE = "blocklist.json"
SIDECAR_SUFFIX = ".tilewright.blocklist"
GUARDS_SUFFIX = ".tilewright.guards"
# The key of a pool kernel's guards (`Guard.to_dict` each) in the pools that
# keep guarded kernels (lib/library_pool.py `without_blocked`).
POOL_GUARDS = "guards"
LIST_SUFFIX = ".tsv"
LISTS_DIR = Path(__file__).resolve().parent.parent / "blocklists"
# The name tokens, an abbreviated parameter and its value, that builds of
# TensileLite spell differently for one kernel: IA (InterleaveAlpha), which
# older builds leave out of names, and PKA (PreloadKernArgs), which a build
# turns off when its compiler cannot preload kernel arguments. TensileLite's
# TilewrightRanker::BlocklistIgnoredTokens are the same.
IGNORED_NAME_TOKENS: Tuple[str, ...] = ("IA", "PKA")
_IGNORED_TOKEN = re.compile(
    "(?:" + "|".join(map(re.escape, IGNORED_NAME_TOKENS)) + ")n?[0-9]+"
)


def blocklist_key(name: Any) -> str:
    """`name` without its IGNORED_NAME_TOKENS: every `_`-separated part
    that is one of them followed by its value (decimal digits, after an `n`
    when negative) is dropped with its separator, as TensileLite's
    TilewrightRanker::blocklistKey drops it. A blocklist name names every
    kernel whose name has its key."""
    text = str(name or "")
    if not any(token in text for token in IGNORED_NAME_TOKENS):
        return text
    return _dropped(text)


@functools.lru_cache(maxsize=1 << 14)
def _dropped(text: str) -> str:
    return "_".join(p for p in text.split("_") if not _IGNORED_TOKEN.fullmatch(p))


@dataclass(frozen=True)
class Guard:
    """The least free size `free` (M, the free size of A, or N, that of B)
    a guarded kernel is served for: `minimum`, at least 2."""

    free: str
    minimum: int

    def __post_init__(self) -> None:
        if self.free not in GUARD_FREE_SIZES:
            raise ValueError(f"a guard reads M or N, not {self.free!r}")
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise ValueError(
                f"a guard's minimum must be a whole number: {self.minimum!r}"
            )
        if not 2 <= self.minimum < 10**9:
            raise ValueError(
                f"a guard's minimum must be from 2 to 999999999: {self.minimum}"
            )

    @property
    def index(self) -> int:
        """The free index the guard reads (0: M, 1: N)."""
        return GUARD_FREE_SIZES.index(self.free)

    def admits(self, m: int, n: int) -> bool:
        return (int(m), int(n))[self.index] >= self.minimum

    def text(self) -> str:
        return f"{self.free} >= {self.minimum}"

    def to_dict(self) -> Dict[str, Any]:
        return {"free": self.free, "minimum": self.minimum}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Guard":
        if not isinstance(d, Mapping):
            raise ValueError(f"a guard is a mapping of free and minimum, not {d!r}")
        return cls(free=str(d.get("free")), minimum=d.get("minimum"))  # type: ignore


def guards_admit(guards: Iterable[Any], m: int, n: int) -> bool:
    """Whether every guard of `guards` (Guard objects or their dicts, as a
    pool kernel's POOL_GUARDS holds them) admits a problem whose free sizes
    are `m` (of A) and `n` (of B), as TensileLite decides it
    (lib/feasibility.py `guard_denies`)."""
    if not guards:
        return True
    return not guard_denies(m, n, guard_bounds(guards))


def kernel_admits(kernel: Mapping[str, Any], m: int, n: int) -> bool:
    """Whether a pool kernel's guards (POOL_GUARDS; none: every problem)
    admit free sizes `m` and `n`."""
    return guards_admit(kernel.get(POOL_GUARDS) or (), m, n)


def _guards_of(raw: Any) -> Tuple[Guard, ...]:
    """The guards of an entry's `guards` field, sorted by free size; one per
    free size (ValueError otherwise)."""
    out: Dict[str, Guard] = {}
    for g in raw or ():
        guard = g if isinstance(g, Guard) else Guard.from_dict(g)
        if guard.free in out:
            raise ValueError(f"an entry guards {guard.free} twice")
        out[guard.free] = guard
    return tuple(out[f] for f in GUARD_FREE_SIZES if f in out)


@dataclass(frozen=True)
class Entry:
    """One blocked kernel: its `name`, why (`reason`: a class of a list, or
    what the gate saw), its kernel id and signature where known, the file or
    check it comes from (`source`), and its `guards` when it is blocked only
    below them (none: blocked whole)."""

    name: str
    reason: str = ""
    kernel_id: Optional[str] = None
    signature: Optional[str] = None
    source: str = ""
    guards: Tuple[Guard, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "kernel_id": self.kernel_id,
            "signature": self.signature,
            "reason": self.reason,
            "source": self.source,
        }
        if self.guards:
            out["guards"] = [g.to_dict() for g in self.guards]
        return out

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
            guards=_guards_of(d.get("guards")),
        )


class Blocklist:
    """Blocked kernels, the first entry of a name kept: an entry that blocks
    it whole wins over a guarded one, and guarded entries of one name merge
    into the first with the stricter guards (`_merged_guards`), whatever
    their order. A name blocks every kernel of its canonical name
    (`lib.dat.canonical_name`), its alias group (lib/kernel_identity.py)
    whole, and every kernel whose name has the same `blocklist_key`; a
    kernel id only by its name hash. A guarded entry (`Entry.guards`) blocks
    those kernels only where its guards deny a problem (`guards`, `admits`);
    `blocks` and its kin answer for the entries that block whole."""

    def __init__(self, entries: Iterable[Entry] = ()) -> None:
        kept: Dict[str, Entry] = {}
        for e in entries:
            have = kept.get(e.name)
            if have is None or (have.guards and not e.guards):
                kept[e.name] = e
            elif have.guards and e.guards:
                kept[e.name] = replace(
                    have, guards=_merged_guards(have.guards, e.guards)
                )
        self.entries: Tuple[Entry, ...] = tuple(kept.values())
        self.names: FrozenSet[str] = frozenset(kept)
        self.canonical: FrozenSet[str] = frozenset(dat.canonical_name(n) for n in kept)
        whole = [n for n, e in kept.items() if not e.guards]
        whole_names = set(whole) | {dat.canonical_name(n) for n in whole}
        self.keys: FrozenSet[str] = frozenset(blocklist_key(n) for n in whole_names)
        self.hashes: FrozenSet[str] = frozenset(ki.name_hash(n) for n in whole_names)
        # {name key: guards} of the guarded entries no entry blocks whole.
        self.guard_keys: Dict[str, Tuple[Guard, ...]] = {}
        self.guard_hashes: Dict[str, Tuple[Guard, ...]] = {}
        for name, e in kept.items():
            if not e.guards:
                continue
            for n in (name, dat.canonical_name(name)):
                key = blocklist_key(n)
                if key not in self.keys:
                    self.guard_keys[key] = _merged_guards(
                        self.guard_keys.get(key, ()), e.guards
                    )
                h = ki.name_hash(n)
                if h not in self.hashes:
                    self.guard_hashes[h] = _merged_guards(
                        self.guard_hashes.get(h, ()), e.guards
                    )

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def guarded_entries(self) -> Tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.guards)

    def has_guards(self) -> bool:
        return bool(self.guard_keys)

    def whole(self) -> "Blocklist":
        """Every entry blocking its kernels whole: what an integration or a
        model without guards serves with (the blocklist sidecar's lines)."""
        if not self.guarded_entries:
            return self
        return Blocklist(replace(e, guards=()) for e in self.entries)

    def guards(self, kernel: Mapping[str, Any]) -> Tuple[Guard, ...]:
        """The guards of a kernel (`kernel_dat_info`, or a pool entry that
        carries its kernel id) that no entry blocks whole; none for a kernel
        no guarded entry names."""
        if not self.guard_keys or self.blocks(kernel):
            return ()
        name = kernel.get("kernel_name")
        if name:
            found: Tuple[Guard, ...] = ()
            for n in (str(name), ki.canonical_name(kernel)):
                found = _merged_guards(found, self.guard_keys.get(blocklist_key(n), ()))
            return found
        kid = kernel.get("kernel_id")
        if not kid or ":" not in str(kid):
            return ()
        return self.guard_hashes.get(str(kid).rsplit(":", 1)[1], ())

    def admits(self, kernel: Mapping[str, Any], m: int, n: int) -> bool:
        """Whether a kernel serves a problem whose free sizes are `m` and
        `n`: no entry blocks it whole and its guards admit them."""
        return not self.blocks(kernel) and guards_admit(self.guards(kernel), m, n)

    def _keyed(self, name: str, canonical: str) -> bool:
        return blocklist_key(name) in self.keys or blocklist_key(canonical) in self.keys

    def blocks_name(self, name: Any) -> bool:
        return bool(name) and self._keyed(str(name), dat.canonical_name(name))

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
            return self._keyed(str(name), ki.canonical_name(kernel))
        return self.blocks_id(kernel.get("kernel_id"))

    def kept(self, kernels: Iterable[Mapping[str, Any]]) -> List[Any]:
        """The kernels no entry blocks whole; a guarded one as a copy that
        carries its guards (POOL_GUARDS)."""
        out: List[Any] = []
        for k in kernels:
            if self.blocks(k):
                continue
            guards = self.guards(k)
            out.append(
                dict(k, **{POOL_GUARDS: [g.to_dict() for g in guards]}) if guards else k
            )
        return out

    def blocked_indices(
        self,
        ids: Mapping[int, str],
        kernels: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Set[int]:
        """The solution indices of `ids` ({index: kernel id}, such as
        `LibraryIdentity.ids` or a record's) that name blocked kernels. An
        index whose kernel `kernels` ({index: `kernel_dat_info`}, such as
        `LibraryIdentity.kernels`) holds by name is matched by that name as
        `blocks` matches it, so a list written from another build's names
        blocks it as TensileLite does; an index known by its kernel id alone
        is matched by the id's name hash, which only the names of the build
        that wrote the id give."""
        known = kernels or {}
        out: Set[int] = set()
        for s, kid in ids.items():
            kernel = known.get(int(s))
            if self.blocks_id(kid) or (
                kernel is not None
                and bool(kernel.get("kernel_name"))
                and self.blocks(kernel)
            ):
                out.add(int(s))
        return out

    def guarded_indices(
        self,
        ids: Mapping[int, str],
        kernels: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Dict[int, Tuple[Guard, ...]]:
        """{solution index: its guards} of the indices of `ids` that name
        guarded kernels no entry blocks whole, matched as `blocked_indices`
        matches."""
        if not self.guard_keys:
            return {}
        known = kernels or {}
        out: Dict[int, Tuple[Guard, ...]] = {}
        for s, kid in ids.items():
            kernel = known.get(int(s))
            if kernel is not None and kernel.get("kernel_name"):
                guards = self.guards(kernel)
            else:
                guards = self.guards({"kernel_id": kid})
            if guards and not self.blocks_id(kid):
                out[int(s)] = guards
        return out

    def merged(self, other: "Blocklist") -> "Blocklist":
        return Blocklist(self.entries + other.entries)

    def to_doc(self, **extra: Any) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            **extra,
            "entries": [e.to_dict() for e in self.entries],
        }


def _merged_guards(a: Tuple[Guard, ...], b: Tuple[Guard, ...]) -> Tuple[Guard, ...]:
    """The guards of both, the larger minimum of a free size kept."""
    best: Dict[str, Guard] = {}
    for g in (*a, *b):
        if g.free not in best or g.minimum > best[g.free].minimum:
            best[g.free] = g
    return tuple(best[f] for f in GUARD_FREE_SIZES if f in best)


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
    """sha256 of the kernel names `blocklist` blocks, sorted, one per line,
    a guarded name followed by a tab and its guards (`Guard.text`, `; `
    between them); None when it blocks none. Blocklists with one digest
    leave out the same kernels, whatever files named them."""
    if not blocklist:
        return None
    guarded = {e.name: e.guards for e in blocklist.entries if e.guards}
    text = "".join(
        (
            f"{n}\t{'; '.join(g.text() for g in guarded[n])}\n"
            if n in guarded
            else f"{n}\n"
        )
        for n in sorted(blocklist.names)
    )
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def applied_record(blocklist: Optional[Blocklist]) -> Dict[str, Any]:
    """What a stage records of the blocklist it applied: its `digest`
    (`names_digest`) and the number of `kernels` it names, and of those
    guarded ones the number it serves above their guards (`guarded`, only
    when there are any)."""
    out: Dict[str, Any] = {
        "digest": names_digest(blocklist),
        "kernels": len(blocklist or ()),
    }
    if blocklist is not None and blocklist.guarded_entries:
        out["guarded"] = len(blocklist.guarded_entries)
    return out


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


def guards_name(stem: str) -> str:
    return f"{stem}{GUARDS_SUFFIX}"


def guards_by_key(
    guards: Mapping[str, Sequence[Guard]],
) -> Dict[str, Tuple[Guard, ...]]:
    """`guards` ({name: guards}) with one name per `blocklist_key`, the first
    in sorted order, holding the guards of every name of its key merged (the
    larger minimum of a free size): TensileLite and `read_guards` match a
    guards line's name by its key and refuse a key and free size guarded
    twice."""
    out: Dict[str, Tuple[str, Tuple[Guard, ...]]] = {}
    for name in sorted(guards):
        key = blocklist_key(name)
        first, have = out.get(key, (name, ()))
        out[key] = (first, _merged_guards(have, tuple(guards[name])))
    return {name: g for name, g in out.values() if g}


def sidecar_guards(
    blocklist: Optional[Blocklist], names: Iterable[str]
) -> Dict[str, Tuple[Guard, ...]]:
    """{name: guards} of the names of `names` (a sidecar's) that a guarded
    entry of `blocklist` narrows (`Blocklist.guards` by name), one name per
    `blocklist_key` (`guards_by_key`)."""
    if blocklist is None or not blocklist.has_guards():
        return {}
    out: Dict[str, Tuple[Guard, ...]] = {}
    for n in names:
        guards = blocklist.guards({"kernel_name": n})
        if guards:
            out[str(n)] = guards
    return guards_by_key(out)


def guards_text(stem: str, guards: Mapping[str, Sequence[Guard]]) -> str:
    """The guards sidecar of library `stem`: a comment, then per guard a
    kernel name, its free size and the minimum, tab-separated, sorted by
    name and free size, one name per `blocklist_key` (`guards_by_key`), as
    TensileLite reads them."""
    head = (
        f"# Kernels of {stem} (and its bank) that tilewright offers only for "
        f"problems whose free size reaches a minimum, for a model trained with "
        f"kernel guards: <name> <M or N> <minimum>, one guard per line."
    )
    keyed = guards_by_key(guards)
    lines = [
        f"{name}\t{g.free}\t{g.minimum}"
        for name in sorted(keyed)
        for g in sorted(keyed[name], key=lambda g: g.index)
    ]
    return "\n".join([head, *lines]) + "\n"


def read_guards(path: Any) -> Dict[str, Tuple[Guard, ...]]:
    """{kernel name: guards} of a guards sidecar, read as TensileLite reads
    it: every line but blank ones and `#` comments holds a name, M or N and
    a minimum of at least 2, separated by blanks, and a name and free size
    appear once (by `blocklist_key`). ValueError naming the line otherwise,
    or when the file cannot be read."""
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise ValueError(f"cannot read the guards {path}: {e}") from None
    out: Dict[str, List[Guard]] = {}
    seen: Set[Tuple[str, str]] = set()
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip(" \t\r")
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        guard: Optional[Guard] = None
        if (
            len(fields) == 3
            and fields[1] in GUARD_FREE_SIZES
            and fields[2].isdigit()
            and fields[2].isascii()
            and len(fields[2]) <= 9
        ):
            try:
                guard = Guard(fields[1], int(fields[2]))
            except ValueError:
                guard = None
        if guard is None:
            raise ValueError(
                f"the guards {path} line {number} is not `<kernel name> <M or N> "
                f"<minimum of at least 2>`"
            )
        if (blocklist_key(fields[0]), guard.free) in seen:
            raise ValueError(
                f"the guards {path} line {number} guards the {guard.free} of "
                f"{fields[0]} again"
            )
        seen.add((blocklist_key(fields[0]), guard.free))
        out.setdefault(fields[0], []).append(guard)
    return {n: _guards_of(gs) for n, gs in out.items()}


def narrowed(
    blocklist: Blocklist, guards: Mapping[str, Sequence[Guard]], source: str = ""
) -> Blocklist:
    """`blocklist` as a model trained with kernel guards is served with the
    guards `guards` ({kernel name: guards}, `read_guards`): every entry of a
    name they guard (by `blocklist_key`) blocks it only below them, and a
    guarded name no entry holds is blocked below its guards too."""
    guarded = {blocklist_key(n): tuple(g) for n, g in guards.items()}
    entries = [
        (
            replace(e, guards=guarded[blocklist_key(e.name)])
            if blocklist_key(e.name) in guarded
            else e
        )
        for e in blocklist.entries
    ]
    named = {blocklist_key(e.name) for e in blocklist.entries}
    entries += [
        Entry(name=n, reason="guards sidecar", source=source, guards=tuple(g))
        for n, g in guards.items()
        if blocklist_key(n) not in named
    ]
    return Blocklist(entries)


def sidecar_blocklist(path: Any, guards_path: Any = None) -> Blocklist:
    """The kernels a sidecar names (`read_sidecar`), as a Blocklist; with
    `guards_path`, the guards sidecar of a model trained with kernel guards,
    each kernel it guards blocked only below its guards, whether or not the
    blocklist names it, as TensileLite serves such a model (`narrowed`).
    ValueError when either cannot be read."""
    try:
        names = read_sidecar(path) if path is not None else []
    except OSError as e:
        raise ValueError(f"cannot read the blocklist sidecar {path}: {e}") from None
    plain = Blocklist(Entry(name=n, reason="sidecar", source=str(path)) for n in names)
    if guards_path is None:
        return plain
    return narrowed(plain, read_guards(guards_path), str(guards_path))


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
        model = Path(str(t.get("staged") or record.get("staged_model") or ""))
        guards = model.parent / str(t["guards"]) if t.get("guards") else None
        if not t.get("blocklist") and guards is None:
            return Blocklist()
        return sidecar_blocklist(
            model.parent / str(t["blocklist"]) if t.get("blocklist") else None,
            guards,
        )
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
