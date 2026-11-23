# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Registered never-train shape lists: evaluation sets that never train,
tune or validate tilewright, and the GEMM corpus they are drawn from. They
may overlap the sets tilewright is validated on.

A list is recognised whole by its fingerprint (`fingerprint`): the sha256
of its distinct (M, N, K, batch), sorted and written one "m,n,k,batch"
line each, the lines joined by newlines, whatever its format, order,
repeats or other fields. A CSV is recognised by the columns of those sets
(`column_refusal`): `legacy_category` or `refined_category`, or a `source`
column holding `log_sample` or `model`. A subset or a mixture of such a
set has another fingerprint and is not recognised.

The loaders of shape lists refuse one with `ExternalSet`, a ValueError:
lib/held_out.py (held-out lists, CSVs included), lib/bench_yaml.py
`parse_shapes_yaml` (bench yamls: round shapes, histories, audit sets),
stage07's bench yamls and stage08's datasets (a bench yaml before it is
benched, every dataset once its GEMMs are read); and preflight names the
lists of a run's config that are one (`config_errors`)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

ShapeKey = Tuple[int, int, int, int]

# fingerprint -> what the list is
FINGERPRINTS = {
    "3432ad47d68c59d76c99707ac25bebcaa31bbc39cd911e1b75c4c47b83f5957d": (
        "a registered never-train set"
    ),
    "48a0bb491ef815ea602abb56794e9281fd72cafa9a08090b09f4687e156fc8e3": (
        "a reduced registered never-train set"
    ),
    "ffe49717a4b2da1978161912228708a9d51e731b2df9cc13a5e722e6ab2481a8": (
        "the GEMM corpus of the registered never-train sets"
    ),
}
# The columns of the registered never-train sets' CSVs, and the values of
# their `source` column.
SIGNATURE_COLUMNS = ("legacy_category", "refined_category")
SIGNATURE_SOURCES = frozenset({"log_sample", "model"})
NEVER = "which never trains, tunes or validates tilewright"


class ExternalSet(ValueError):
    """A shape list that is a registered never-train set."""


def fingerprint(gemms: Iterable[Sequence[Any]]) -> str:
    """sha256 of the distinct (M, N, K, batch) of `gemms`, sorted, one
    "m,n,k,batch" line each, joined by newlines."""
    keys = sorted({tuple(int(v) for v in list(g)[:4]) for g in gemms})
    text = "\n".join(",".join(str(v) for v in k) for k in keys)
    return hashlib.sha256(text.encode()).hexdigest()


def refusal(gemms: Iterable[Sequence[Any]]) -> Optional[str]:
    """Why the GEMMs `gemms` never serve tilewright: they are one of the
    sets FINGERPRINTS names; None otherwise."""
    what = FINGERPRINTS.get(fingerprint(gemms))
    return None if what is None else f"its GEMMs are {what}, {NEVER}"


def column_refusal(
    fieldnames: Optional[Iterable[str]], rows: Iterable[Mapping[str, Any]] = ()
) -> Optional[str]:
    """Why a CSV of the columns `fieldnames` and the rows `rows` is one of
    those sets: a signature column, or a `source` column holding a
    signature value; None otherwise."""
    names = {str(c).strip().lower(): c for c in fieldnames or []}
    hits = [c for c in SIGNATURE_COLUMNS if c in names]
    if hits:
        return (
            f"its column(s) {', '.join(hits)} are those of the registered "
            f"never-train sets, {NEVER}"
        )
    source = names.get("source")
    if source is None:
        return None
    for row in rows:
        value = str(row.get(source) or "").strip()
        if value in SIGNATURE_SOURCES:
            return (
                f"its source column holds {value!r}, a source of the registered "
                f"never-train sets, {NEVER}"
            )
    return None


def check(gemms: Iterable[Sequence[Any]], where: Any) -> None:
    """ExternalSet, naming `where`, when `gemms` are such a set."""
    why = refusal(gemms)
    if why is not None:
        raise ExternalSet(f"{where}: {why}")


def check_csv(
    fieldnames: Optional[Iterable[str]], rows: Iterable[Mapping[str, Any]], where: Any
) -> None:
    """ExternalSet, naming `where`, when the CSV is such a set
    (`column_refusal`)."""
    why = column_refusal(fieldnames, rows)
    if why is not None:
        raise ExternalSet(f"{where}: {why}")


def entry_gemms(entries: Iterable[Any]) -> List[ShapeKey]:
    """(M, N, K, batch_count) of the bench yaml entries (mappings) that name
    M, N and K; batch_count 1 when absent."""
    out: List[ShapeKey] = []
    for e in entries:
        if not isinstance(e, Mapping):
            continue
        try:
            out.append(
                (
                    int(e["M"]),
                    int(e["N"]),
                    int(e["K"]),
                    int(e.get("batch_count", 1) or 1),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _yaml_gemms(path: Path) -> List[ShapeKey]:
    """The GEMMs of a bench yaml: its flow entries (`- {...}` lines), or the
    entries of the whole file when it has none."""
    import yaml

    from .bench_yaml import shape_of_line

    text = Path(path).read_text()
    flow = [ln for ln in text.splitlines() if ln.strip().startswith("- {")]
    if flow:
        return [s for s in (shape_of_line(ln) for ln in flow) if s is not None]
    data = yaml.safe_load(text) or []
    return entry_gemms([data] if isinstance(data, Mapping) else data)


def _csv_dir_refusal(path: Path) -> Optional[str]:
    for chunk in sorted(Path(path).glob("chunk_*.csv")):
        with chunk.open(newline="") as f:
            reader = csv.DictReader(f)
            why = column_refusal(reader.fieldnames, reader)
        if why is not None:
            return f"{chunk.name}: {why}"
    return None


def config_errors(cfg: Mapping[str, Any]) -> List[str]:
    """Preflight: the shape lists of the config that are such a set, the
    `audit.held_out` lists, `stage07.bench_yamls` and the `stage08.datasets`
    that are bench yamls or directories of chunk_*.csv files (stage08 checks
    a directory of bench logs once it has read its GEMMs). A list that
    cannot be read is reported by the checks of its key."""
    import yaml

    from . import held_out as hol

    def section(name: str) -> Mapping[str, Any]:
        sec = cfg.get(name)
        return sec if isinstance(sec, Mapping) else {}

    errors: List[str] = []
    for entry in section("audit").get("held_out") or []:
        try:
            hol.read(entry)
        except ExternalSet as e:
            errors.append(f"audit.held_out: {e}")
        except (OSError, ValueError):
            pass
    named = [
        ("stage07.bench_yamls", p) for p in section("stage07").get("bench_yamls") or []
    ]
    for ds in section("stage08").get("datasets") or []:
        if isinstance(ds, Mapping) and ds.get("path"):
            named.append((f"stage08.datasets {ds.get('name')}", ds["path"]))
    for key, entry in named:
        p = Path(str(entry))
        try:
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
                why = refusal(_yaml_gemms(p))
            elif p.is_dir():
                why = _csv_dir_refusal(p)
            else:
                continue
        except (OSError, ValueError, csv.Error, yaml.YAMLError):
            continue
        if why is not None:
            errors.append(f"{key}: {p}: {why}")
    return errors
