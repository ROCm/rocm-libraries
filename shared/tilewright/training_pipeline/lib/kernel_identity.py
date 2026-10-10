# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel identity across hipBLASLt builds.

TensileCreateLibrary numbers the solutions of a build globally: every
solution of every architecture and library file, in sorted order. Adding or
removing any library file or architecture renumbers the solutions of the
others, so a solution index (`sol_idx_global`) names a kernel only within
one build. The pipeline stores solution indices in its CSVs, plans and
records; each such artifact also records the kernels they named, and a stage
checks that record before it joins the artifact with the library it reads
now.

  kernel id       `<signature>:<name hash>`: the 8-field signature
                  (features.SIGNATURE_FIELDS) written
                  MT<m>x<n>x<depthU>_MI<m>x<n>x<k>_NTA<a>_NTB<b>, and 12 hex
                  digits of sha256 over the solution name (`.dat` `name`, the
                  `--Solution name` the bench prints), which tell the kernels
                  of one signature apart
  file digest     16 hex digits of sha256 over one library file: its stem,
                  its (solution index, kernel id) pairs in ascending index and
                  its Prediction table, the order in which the runtime ranks
                  the pool
  index digest    the same without the Prediction table: what a solution
                  index names, which a rebuild that only reorders the table
                  keeps
  library digest  the same over the file digests of the files a stage read
                  (`library_identity`)

`record(identity, indices)` is the JSON object an artifact keeps under
RECORD_KEY: the library digest, every file's digest and index digest and,
per file, the kernel id of every index of it the artifact names. A recorded
file names other kernels now when its index digest differs (its file
digest, in a record written before index digests were). Before a join,
`Remap.of(record, identity, what)` compares the record with the library the
stage reads now, file by file. Indices of a file whose index digest is
unchanged still name their kernels, and when no recorded file the stage
reads names other kernels the indices are kept as they are, those of
recorded files the stage does not read included, except an index a file the
stage reads holds another kernel under, which is unresolved. Otherwise each
recorded index of a file the stage reads maps to the index its kernel id has
now; a kernel the library no longer holds, or holds under several indices,
is unresolved, and so is every index of a recorded file the stage does not
read. Joins that cannot translate indices (bench lines about to run) call
`Remap.require_same`, which raises BuildChangedError.

An enriched folder written before identities were recorded is checked
instead (`folder_remap`, `LegacyCheck`): its rows carry each kernel's engine
config and its kernels.json the kernel attributes, and a solution index
whose config or attributes differ from the library's kernel at that index
names another kernel now; its rows are left out. Plans written before
identities were recorded carry neither and are taken as they are.

Benches record the identity of the library files their indices name
(`bench_record` reads it from a log directory's stage summary), so a stage
reading their logs can tell whether its library is still the one the bench
loaded.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

from . import ui
from .dat import (
    kernel_attributes_of_doc,
    kernel_dat_info,
    library_logic_path,
    prediction_table,
    read_tensile_logic,
)
from .features import SIGNATURE_FIELDS

SCHEMA = "kernel_identity_v1"
RECORD_KEY = "kernel_identity"
INDEX_DIGESTS = "index_digests"
NAME_HASH_DIGITS = 12
DIGEST_DIGITS = 16
# Stage summaries next to a bench's logs/ that hold its RECORD_KEY.
BENCH_SUMMARIES = ("stage03_summary.json", "rebench_summary.json")
# stage05's record of `member_orders` in its output directory.
MEMBER_ORDER_FILE = "member_order.json"


class BuildChangedError(RuntimeError):
    """A persisted artifact names kernels by solution indices of another
    hipBLASLt build than the library a stage reads now."""


def signature_text(kernel: Mapping[str, Any]) -> str:
    """MT<m>x<n>x<depthU>_MI<m>x<n>x<k>_NTA<a>_NTB<b> of a kernel's
    signature fields."""
    m, n, du, mi_m, mi_n, mi_k, nta, ntb = (int(kernel[f]) for f in SIGNATURE_FIELDS)
    return f"MT{m}x{n}x{du}_MI{mi_m}x{mi_n}x{mi_k}_NTA{nta}_NTB{ntb}"


def signature_label(kernel: Mapping[str, Any]) -> str:
    """How the report names a kernel: its signature, e.g. `MT256x256x64
    MI16x16x32 NTA0 NTB4`."""
    return signature_text(kernel).replace("_", " ")


def label_of_id(kid: str) -> str:
    """The `signature_label` of a kernel id."""
    return str(kid).split(":", 1)[0].replace("_", " ")


_SIGNATURE_TEXT = re.compile(
    r"MT(\d+)x(\d+)x(\d+)_MI(\d+)x(\d+)x(\d+)_NTA(\d+)_NTB(\d+)"
)


def signature_of_id(kid: str) -> Optional[Tuple[int, ...]]:
    """The signature fields (SIGNATURE_FIELDS order) of a kernel id, None
    when it is not one."""
    m = _SIGNATURE_TEXT.fullmatch(str(kid).split(":", 1)[0])
    return tuple(int(v) for v in m.groups()) if m else None


def naming(
    identity: Optional["LibraryIdentity"], sid: Any, prefix: str = ""
) -> Dict[str, Optional[str]]:
    """{`<prefix>signature`, `<prefix>kernel_id`} of solution index `sid` in
    `identity`'s library, which persisted outputs write next to an index;
    None where there is no identity or it holds no such index."""
    kid = None
    if identity is not None and sid is not None:
        try:
            kid = identity.ids.get(int(sid))
        except (TypeError, ValueError):
            kid = None
    return {
        f"{prefix}signature": label_of_id(kid) if kid else None,
        f"{prefix}kernel_id": kid,
    }


def name_hash(name: str) -> str:
    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:NAME_HASH_DIGITS]


_MT_NAME_RE = re.compile(r"_MT(\d+x\d+x\d+)_")


def label_of_name(name: str) -> str:
    """A kernel named by its solution name alone, where no library file
    gives its signature: the macro tile the name spells (`MT<m>x<n>x<depthU>`,
    `?` without one) and the name hash. The name does not spell the rest of
    the signature (the matrix instruction's K, for one)."""
    m = _MT_NAME_RE.search(str(name))
    return f"{'MT' + m.group(1) if m else '?'} (name {name_hash(name)})"


def kernel_id(kernel: Mapping[str, Any]) -> str:
    """The kernel id of a `lib.dat.kernel_dat_info` entry (see the module
    docstring)."""
    return f"{signature_text(kernel)}:{name_hash(kernel.get('kernel_name') or '')}"


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:DIGEST_DIGITS]


@dataclass(frozen=True)
class LibraryIdentity:
    """The kernels of some library files: `ids` (solution index -> kernel
    id) and `kernels` (solution index -> `kernel_dat_info`) over all of
    them, the first file's kernel when several hold an index; per file
    (`stems`, in the order read) its digest (`files`), index digest
    (`index_digests`) and kernel ids (`file_ids`); and the library
    `digest`."""

    digest: str
    ids: Mapping[int, str]
    stems: Tuple[str, ...] = ()
    files: Mapping[str, str] = field(default_factory=dict)
    file_ids: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    index_digests: Mapping[str, str] = field(default_factory=dict)
    kernels: Mapping[int, Mapping[str, Any]] = field(
        default_factory=dict, compare=False, repr=False
    )
    _by_id: Dict[str, Optional[int]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def index_of(self, kid: str) -> Optional[int]:
        """The solution index of kernel id `kid`; None when the files hold
        it under no index or under several."""
        if not self._by_id and self.ids:
            for sid, k in sorted(self.ids.items()):
                self._by_id[k] = None if k in self._by_id else sid
        return self._by_id.get(kid)


def _file_digest(stem: str, ids: Mapping[int, str], table: Optional[Sequence[int]]):
    lines = [SCHEMA, f"stem\t{stem}"]
    lines += [f"{sid}\t{ids[sid]}" for sid in sorted(ids)]
    if table is not None:
        lines.append("table\t" + ",".join(str(int(i)) for i in table))
    return _hex("\n".join(lines) + "\n")


def _library_digest(files: Mapping[str, str]) -> str:
    return _hex("\n".join([SCHEMA] + [f"{s}\t{files[s]}" for s in sorted(files)]))


def identity_of_files(
    files: Sequence[Tuple[str, Iterable[Mapping[str, Any]], Optional[Sequence[int]]]],
) -> LibraryIdentity:
    """The identity of library files given as (stem, `kernel_dat_info`
    entries of its solutions, its Prediction table or None)."""
    ids: Dict[int, str] = {}
    kernels: Dict[int, Mapping[str, Any]] = {}
    digests: Dict[str, str] = {}
    index_digests: Dict[str, str] = {}
    file_ids: Dict[str, Dict[int, str]] = {}
    for stem, entries, table in files:
        own: Dict[int, str] = {}
        for k in entries:
            sid = k.get("sol_idx_global")
            if sid is None or int(sid) in own:
                continue
            own[int(sid)] = kernel_id(k)
            if int(sid) not in ids:
                ids[int(sid)], kernels[int(sid)] = own[int(sid)], k
        digests[str(stem)] = _file_digest(str(stem), own, table)
        index_digests[str(stem)] = _file_digest(str(stem), own, None)
        file_ids[str(stem)] = own
    return LibraryIdentity(
        _library_digest(digests),
        ids,
        tuple(digests),
        digests,
        file_ids,
        index_digests,
        kernels,
    )


_CACHE: Dict[Tuple[Any, ...], LibraryIdentity] = {}


def library_identity(
    dat_dir: Any, stems: Sequence[str], tolerate_tables: bool = False
) -> LibraryIdentity:
    """The identity of the logic files `<dat_dir>/<stem>.dat[.zlib]` of
    `stems`. FileNotFoundError / ValueError, naming the file, when one is
    missing or unreadable or holds a solution TensileLite cannot read; with
    `tolerate_tables` a Prediction table that cannot be followed leaves the
    table out of its file's digest instead."""
    stems = list(dict.fromkeys(str(s) for s in stems))
    paths = [library_logic_path(dat_dir, s) for s in stems]
    key = tuple(
        (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths
    ) + (bool(tolerate_tables),)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    files = []
    for stem, path in zip(stems, paths):
        data = read_tensile_logic(path)
        if data is None:
            raise ValueError(f"unreadable Tensile logic file: {path}")
        try:
            kernels = [
                kernel_dat_info(s)
                for s in data.get("solutions") or []
                if s.get("index") is not None
            ]
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        try:
            table = prediction_table(data)
        except ValueError as e:
            if not tolerate_tables:
                raise ValueError(f"{path}: {e}") from None
            table = None
        files.append((stem, kernels, table))
    out = identity_of_files(files)
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out


# ── records ──────────────────────────────────────────────────────────────────


def record(
    identity: LibraryIdentity, indices: Optional[Iterable[int]] = None
) -> Dict[str, Any]:
    """The RECORD_KEY object of an artifact naming the solution `indices`
    of `identity`'s library (every solution of it without `indices`)."""
    chosen = set(identity.ids) if indices is None else {int(i) for i in indices}
    owner: Dict[int, str] = {}
    for stem in identity.stems:
        for sid in identity.file_ids.get(stem, {}):
            owner.setdefault(sid, stem)
    return {
        "schema": SCHEMA,
        "digest": identity.digest,
        "files": dict(identity.files),
        INDEX_DIGESTS: dict(identity.index_digests),
        "kernel_ids": {
            stem: {
                str(sid): identity.ids[sid]
                for sid in sorted(chosen)
                if owner.get(sid) == stem
            }
            for stem in identity.stems
        },
    }


def read_record(doc: Any) -> Optional[Dict[str, Any]]:
    """The RECORD_KEY object of an artifact, or None when it has none.
    ValueError when it has one this module cannot read."""
    if not isinstance(doc, Mapping) or doc.get(RECORD_KEY) is None:
        return None
    rec = doc[RECORD_KEY]
    if (
        not isinstance(rec, Mapping)
        or rec.get("schema") != SCHEMA
        or not isinstance(rec.get("files"), Mapping)
        or not isinstance(rec.get(INDEX_DIGESTS, {}), Mapping)
        or not isinstance(rec.get("kernel_ids"), Mapping)
        or not all(isinstance(v, Mapping) for v in rec["kernel_ids"].values())
    ):
        raise ValueError(f"not a {SCHEMA} kernel identity record")
    return dict(rec)


def names_other_kernels(
    rec: Mapping[str, Any], identity: LibraryIdentity, stem: str
) -> bool:
    """Whether recorded file `stem`, which `identity` reads, holds other
    kernels under its solution indices now: its index digest differs, or
    its file digest in a record without index digests."""
    then = (rec.get(INDEX_DIGESTS) or {}).get(stem)
    if then is not None:
        return identity.index_digests.get(stem) != str(then)
    return identity.files.get(stem) != str(rec["files"][stem])


def recorded_ids(rec: Optional[Mapping[str, Any]]) -> Optional[Dict[int, str]]:
    """{solution index: kernel id} of a record, every file's."""
    if rec is None:
        return None
    return {
        int(s): str(k) for ids in rec["kernel_ids"].values() for s, k in ids.items()
    }


def _stems_text(stems: Sequence[str]) -> str:
    return ", ".join(stems[:3]) + (
        f" and {len(stems) - 3} more" if len(stems) > 3 else ""
    )


# ── enriched folders without a record ────────────────────────────────────────


@dataclass
class LegacyCheck:
    """The kernels an enriched folder without a record names, compared with
    the library's at their solution index (`configs`: the engine config,
    lib/evaluate.py CONFIG_FIELDS, of each, or the `fields` given; `ids`:
    their kernel ids): an index whose config (`check_row`, the folder's
    rows) or attributes (`of`, its kernels.json) differ from the library's
    kernel there names another kernel now (`differ`, with what differs). An
    index the library does not hold is not checked. `of_kernel_ids` checks
    against kernel ids alone instead of a library."""

    configs: Mapping[int, Tuple[int, ...]]
    ids: Mapping[int, str]
    differ: Dict[int, str] = field(default_factory=dict)
    agree: Set[int] = field(default_factory=set)
    fields: Tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        identity: LibraryIdentity,
        attributes: Mapping[int, Mapping[str, int]],
    ) -> "LegacyCheck":
        from .evaluate import CONFIG_FIELDS

        out = cls(
            {
                sid: tuple(int(k[f]) for f in CONFIG_FIELDS)
                for sid, k in identity.kernels.items()
            },
            identity.ids,
        )
        out.check_attributes(
            {sid: k.get("attributes") for sid, k in identity.kernels.items()},
            attributes,
        )
        return out

    @classmethod
    def of_kernel_ids(
        cls,
        ids: Mapping[int, str],
        then: Mapping[int, Optional[Mapping[str, int]]],
        attributes: Mapping[int, Mapping[str, int]],
    ) -> "LegacyCheck":
        """The check against kernel ids (`ids`; the folder's rows are
        compared with their signatures) and the attributes that came with
        them (`then`, compared with the folder's `attributes`)."""
        configs = {}
        for sid, kid in ids.items():
            sig = signature_of_id(kid)
            if sig is not None:
                configs[int(sid)] = sig
        out = cls(configs, ids, fields=tuple(SIGNATURE_FIELDS))
        out.check_attributes(then, attributes)
        return out

    def check_attributes(
        self,
        then: Mapping[int, Optional[Mapping[str, int]]],
        attributes: Mapping[int, Mapping[str, int]],
    ) -> None:
        """Mark each index whose attributes (`attributes`, the folder's
        kernels.json) differ from those `then` gives it."""
        for sid, attrs in attributes.items():
            now = then.get(int(sid))
            if now is None:
                continue
            for name, value in attrs.items():
                if name in now and int(now[name]) != int(value):
                    self.differ[int(sid)] = f"a kernel with {name} {value}"
                    break

    def decided(self, sid: int) -> bool:
        return sid in self.agree or sid in self.differ or sid not in self.configs

    def check_row(self, sid: int, row: Mapping[str, Any]) -> None:
        """Compare the engine config of a row of solution `sid` with the
        library's kernel there (columns the row lacks are not compared)."""
        from .evaluate import CONFIG_FIELDS

        cur = self.configs.get(sid)
        if cur is None or sid in self.agree or sid in self.differ:
            return
        fields = self.fields or CONFIG_FIELDS
        then: Dict[str, int] = {}
        for f in fields:
            try:
                then[f] = int(row.get(f, ""))
            except (TypeError, ValueError):
                continue
        now = dict(zip(fields, cur))
        if all(v == now[f] for f, v in then.items()):
            self.agree.add(sid)
            return
        self.differ[sid] = (
            signature_text(then)
            if all(f in then for f in SIGNATURE_FIELDS)
            else "a kernel of another engine config"
        )

    def describe(self, sid: int) -> str:
        return (
            f"solution {sid} held {self.differ[sid]} and holds "
            f"{self.ids.get(sid, 'none')} now"
        )


@dataclass
class Found:
    """What a copy of a Remap (`Remap.fresh`) found reading part of the
    artifact's rows (`Remap.found`): the kernels of an enriched folder
    without a record that its rows showed to differ from the library's or
    to agree with them, and the rows it left out."""

    differ: Dict[int, str] = field(default_factory=dict)
    agree: Set[int] = field(default_factory=set)
    rows_left_out: int = 0


# ── remaps ───────────────────────────────────────────────────────────────────


@dataclass
class Remap:
    """How the solution indices of an artifact (`what`) map to the current
    library. `recorded` when the artifact has a record; `mapping` holds
    every recorded index with its current one (None: unresolved) when some
    index does not keep its kernel, else it is empty. `changed` lists the
    recorded (index, kernel id) pairs whose index names another kernel or
    none now (moved, retuned or removed; while `files_changed` is empty,
    the indices of `files_outside` a file the stage reads holds another
    kernel under), `files_changed` the recorded files the stage reads that
    hold other kernels under their indices (`names_other_kernels`),
    `files_outside` the recorded files it does not read. `legacy` checks an
    enriched folder without a record."""

    what: str
    digest_now: str
    recorded: bool = False
    digest_then: Optional[str] = None
    n_recorded: int = 0
    mapping: Dict[int, Optional[int]] = field(default_factory=dict)
    changed: List[Tuple[int, str]] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    files_outside: List[str] = field(default_factory=list)
    legacy: Optional[LegacyCheck] = None
    rows_left_out: int = 0
    kernels_then: Dict[int, str] = field(default_factory=dict, repr=False)

    @classmethod
    def of(
        cls, rec: Optional[Mapping[str, Any]], current: LibraryIdentity, what: str
    ) -> "Remap":
        out = cls(what, current.digest)
        ids = recorded_ids(rec)
        if ids is None:
            return out
        assert rec is not None
        out.recorded, out.digest_then, out.n_recorded = True, rec["digest"], len(ids)
        out.kernels_then = ids
        files = [str(s) for s in rec["files"]]
        out.files_outside = [s for s in files if s not in current.files]
        out.files_changed = [
            s
            for s in files
            if s in current.files and names_other_kernels(rec, current, s)
        ]
        covered = {
            int(sid)
            for stem, named in rec["kernel_ids"].items()
            if stem in current.files
            for sid in named
        }
        if not out.files_changed:
            out.changed = [
                (sid, kid)
                for sid, kid in sorted(ids.items())
                if sid not in covered and current.ids.get(sid, kid) != kid
            ]
            if out.changed:
                taken = {sid for sid, _ in out.changed}
                out.mapping = {sid: (None if sid in taken else sid) for sid in ids}
            return out
        moved: Dict[int, Optional[int]] = {}
        for sid, kid in sorted(ids.items()):
            if current.ids.get(sid) != kid:
                out.changed.append((sid, kid))
            if sid not in covered:
                moved[sid] = None
            else:
                moved[sid] = (
                    sid if current.ids.get(sid) == kid else current.index_of(kid)
                )
        if any(new != sid for sid, new in moved.items()):
            out.mapping = moved
        return out

    @property
    def same(self) -> bool:
        """Every index the artifact names keeps its kernel (as far as an
        enriched folder without a record was checked so far)."""
        return not self.mapping and not (self.legacy and self.legacy.differ)

    @property
    def unresolved(self) -> List[int]:
        return sorted(s for s, new in self.mapping.items() if new is None)

    @property
    def n_moved(self) -> int:
        return sum(1 for s, new in self.mapping.items() if new is not None and new != s)

    def index(self, sid: int) -> Optional[int]:
        """The current index of recorded solution index `sid`; None when its
        kernel is unresolved, or when the build changed and the record does
        not name `sid`. Negative ids (rows without a solution index) stay."""
        sid = int(sid)
        if sid < 0:
            return sid
        if self.legacy is not None:
            return None if sid in self.legacy.differ else sid
        if not self.mapping:
            return sid
        return self.mapping.get(sid)

    def indices(self, sids: Iterable[int]) -> List[int]:
        """`index` of each of `sids`, unresolved ones left out."""
        out = (self.index(s) for s in sids)
        return [s for s in out if s is not None]

    def fresh(self) -> "Remap":
        """A copy to read part of the artifact's rows with (in another
        process, say), holding only what `row_index` needs: it cannot
        explain. `found` says what it found, which `absorb` takes back."""
        legacy = None
        if self.legacy is not None:
            legacy = replace(
                self.legacy,
                ids={},
                differ=dict(self.legacy.differ),
                agree=set(self.legacy.agree),
            )
        return replace(
            self, legacy=legacy, rows_left_out=0, kernels_then={}, changed=[]
        )

    def found(self, since: "Remap") -> Found:
        """What this copy of `since` (`since.fresh()`) found beyond what
        `since` holds."""
        if self.legacy is None or since.legacy is None:
            return Found(rows_left_out=self.rows_left_out)
        return Found(
            {
                s: why
                for s, why in self.legacy.differ.items()
                if s not in since.legacy.differ
            },
            self.legacy.agree - since.legacy.agree,
            self.rows_left_out,
        )

    def absorb(self, found: Found) -> None:
        if self.legacy is not None:
            self.legacy.differ.update(found.differ)
            self.legacy.agree |= found.agree - set(self.legacy.differ)
        self.rows_left_out += found.rows_left_out

    def needs_row(self, sid: int) -> bool:
        """An enriched folder without a record has not checked `sid` yet."""
        return self.legacy is not None and sid >= 0 and not self.legacy.decided(sid)

    def row_index(self, sid: int, row: Mapping[str, Any]) -> Optional[int]:
        """`index` of the solution index of an enriched row (`row` maps the
        CSV columns), checking the row's engine config first when the folder
        has no record; counts the rows it leaves out."""
        if self.needs_row(sid):
            assert self.legacy is not None
            self.legacy.check_row(sid, row)
        out = self.index(sid)
        if out is None:
            self.rows_left_out += 1
        return out

    def _files_text(self) -> str:
        parts = []
        if self.files_changed:
            parts.append(f"library file(s) {_stems_text(self.files_changed)} changed")
        if self.files_outside:
            parts.append(
                f"it names kernels of {_stems_text(self.files_outside)}, which "
                f"this stage does not read"
            )
        return "; ".join(parts)

    def explain(self) -> str:
        if self.legacy is not None:
            return self._explain_legacy()
        if not self.recorded:
            return (
                f"{self.what} records no kernel identities (written before they "
                f"were recorded): its solution indices are taken to name the "
                f"kernels of the current library"
            )
        if self.same:
            if self.digest_then == self.digest_now:
                return f"{self.what}: same library ({self.digest_now})"
            what = self._files_text() or "the library files differ"
            return (
                f"{self.what}: {what} (library digest {self.digest_then} -> "
                f"{self.digest_now}), but every solution index it names in the "
                f"files this stage reads still names the same kernel"
            )
        if not self.files_changed:
            sid, kid = self.changed[0]
            return (
                f"{self.what}: {self._files_text()}, and the files it reads hold "
                f"other kernels under {len(self.changed)} of the solution indices "
                f"it names there (e.g. solution {sid}, which named {kid})"
            )
        text = (
            f"{self.what}: the hipBLASLt build changed since it was written "
            f"({self._files_text()}; library digest {self.digest_then} -> "
            f"{self.digest_now}): {len(self.changed)} of the {self.n_recorded} "
            f"solution indices it names now name another kernel or none"
        )
        moved = sorted(
            (s, n) for s, n in self.mapping.items() if n is not None and n != s
        )
        if moved:
            s, n = moved[0]
            text += (
                f"; {len(moved)} of its kernels are at other solution indices "
                f"now (e.g. {self.kernels_then[s]}: solution {s} then, {n} now)"
            )
        gone = self.unresolved
        if gone:
            text += (
                f"; {len(gone)} of its kernels are not in the library this stage "
                f"reads under one solution index (e.g. "
                f"{self.kernels_then[gone[0]]})"
            )
        return text

    def _explain_legacy(self) -> str:
        assert self.legacy is not None
        n = len(self.legacy.agree) + len(self.legacy.differ)
        if not self.legacy.differ:
            return (
                f"{self.what} records no kernel identities; the {n} kernels it "
                f"names match the current library's at their solution indices "
                f"by engine config and attributes"
            )
        first = min(self.legacy.differ)
        return (
            f"{self.what} records no kernel identities, and {len(self.legacy.differ)} "
            f"of the {n} kernels it names differ from the current library's at "
            f"their solution indices (engine config or attributes; e.g. "
            f"{self.legacy.describe(first)}): it was enriched with another "
            f"hipBLASLt build, and its {self.rows_left_out} row(s) of those "
            f"kernels are left out"
        )

    def require_same(self) -> None:
        """BuildChangedError unless the artifact's indices still name its
        kernels."""
        if not self.same:
            raise BuildChangedError(self.explain())

    def report(self) -> None:
        """Warn once when the artifact is read with other indices or rows
        left out; an enriched folder without a record whose kernels all
        match is noted."""
        if not self.digest_now or (self.recorded and self.same):
            return
        if self.legacy is not None and self.same:
            _say(ui.info, self.explain())
            return
        text = self.explain()
        if self.recorded:
            text += "; read with its kernels' current indices"
            if self.unresolved:
                text += f", {len(self.unresolved)} kernel(s) of it left out"
        _say(ui.warn, text)


_REPORTED: set = set()


def _say(fn: Callable[[str, str], None], text: str) -> None:
    if text not in _REPORTED:
        _REPORTED.add(text)
        fn("identity", text)


def report_once(text: str, warn: bool = True) -> None:
    """Print `text` once per process, as a warning or else as a note."""
    _say(ui.warn if warn else ui.info, text)


def warn_unrecorded(what: str) -> None:
    """Warn once that artifact `what` records no kernel identities, so its
    solution indices are taken as they are."""
    report_once(Remap(what, "").explain())


def remap_of(doc: Any, identity: Optional[LibraryIdentity], what: str) -> Remap:
    """The Remap of the solution indices of an artifact (`doc`, the JSON
    object holding its record) to `identity`'s library; no remap without
    `identity`. A remap, and an artifact without a record, are reported
    once."""
    if identity is None:
        return Remap(what, "")
    remap = Remap.of(read_record(doc), identity, what)
    remap.report()
    return remap


def folder_remap(doc: Any, identity: Optional[LibraryIdentity], what: str) -> Remap:
    """`remap_of` for an enriched folder whose kernels.json is `doc` (None
    without one): without a record its kernels are checked against
    `identity` (`LegacyCheck`, by the attributes of `doc` here and by the
    rows' engine config in `Remap.row_index`); `Remap.report` after its rows
    were read says what the check found."""
    if identity is None:
        return Remap(what, "")
    rec = read_record(doc)
    if rec is not None:
        return remap_of(doc, identity, what)
    try:
        attributes = kernel_attributes_of_doc(doc) if doc is not None else {}
    except ValueError:
        attributes = {}
    return Remap(what, identity.digest, legacy=LegacyCheck.of(identity, attributes))


# ── bench records ────────────────────────────────────────────────────────────


def bench_record(log_dir: Any) -> Optional[Dict[str, Any]]:
    """The RECORD_KEY object a bench left next to its logs (`log_dir` is
    `<stage dir>/logs`, its summary one of BENCH_SUMMARIES), or None.
    ValueError for a record this module cannot read."""
    for name in BENCH_SUMMARIES:
        path = Path(log_dir).parent / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rec = read_record(doc)
        if rec is not None:
            return rec
    return None


def changed_files(rec: Mapping[str, Any], identity: LibraryIdentity) -> List[str]:
    """The files of a record that `identity` does not read, or that hold
    other kernels under their solution indices now
    (`names_other_kernels`)."""
    return [
        str(s)
        for s in rec["files"]
        if str(s) not in identity.files or names_other_kernels(rec, identity, str(s))
    ]


# ── members of a key ─────────────────────────────────────────────────────────


def member_orders(
    pool: Sequence[Mapping[str, Any]], key_of: Callable[[Mapping[str, Any]], Any]
) -> Dict[str, str]:
    """{key: digest of its members' kernel ids in pool order} of every key of
    `key_of` (a tuple of ints) with several members in `pool` (runtime
    order, entries of `kernel_dat_info` or `lib.evaluate.pool_from_kernels`):
    the runtime serves a key's first member whose predicates pass. A key is
    written as its members' signature, `|`, then its values."""
    members: Dict[Tuple[int, ...], List[Mapping[str, Any]]] = {}
    for k in pool:
        members.setdefault(tuple(int(v) for v in key_of(k)), []).append(k)
    return {
        f"{signature_text(ks[0])}|{','.join(str(v) for v in key)}": _hex(
            "\n".join(k.get("kernel_id") or kernel_id(k) for k in ks)
        )
        for key, ks in sorted(members.items())
        if len(ks) > 1
    }
