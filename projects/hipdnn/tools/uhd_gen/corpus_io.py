# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reading a corpus in whichever of RFC 0019.13 §8.3's two formats it arrived in.

§8.3 collects as CSV and publishes as Parquet, so every command taking `--input`
(`train`, `evaluate`, `knobs`, `merge`) has to read both, and the suffix decides which.
That rule lives here once rather than in each of them: the two formats disagree about
exactly one thing -- who decides a column's type -- and a copy of the decision per
command is a chance for two commands to answer differently about the same file.

The disagreement is real and it is silent. `pd.read_csv` infers, so a device id spelled
`0123` comes back as the integer 123 unless the reader pins it; Parquet carries whatever
type the writer froze, so the same id can come back `int64` from the dataset and `str`
from the CSV it was published from. Every consumer of identity here compares or joins on
it -- `evaluate.resolve_grouping`, `immediate._text`, `merge`'s device report -- so the
pinning happens at the read, before anything can branch on the difference.

Deliberately importing nothing from this package: `train`, `evaluate`, `knobs` and
`merge` all import it, and `knobs` already imports `evaluate`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = ["IDENTITY_COLUMNS", "IDENTITY_DTYPES", "pin_identity_dtypes", "read_corpus_frame"]

#: The columns that name a thing rather than measure one: the graph and the board, in
#: both the §8.3 envelope's spelling (`benchmark`/`device`) and the immediate corpus's
#: (`graph_id`/`device_id`). They are identities, never quantities -- nothing computes
#: with them -- so a numeric-looking one is still a name and is read as text.
IDENTITY_COLUMNS = ("benchmark", "device", "graph_id", "device_id")

#: The `dtype=` map for the CSV reader, which is the only reader that can be told in
#: advance. Parquet and JSON are pinned after the fact by `pin_identity_dtypes`.
IDENTITY_DTYPES = {name: str for name in IDENTITY_COLUMNS}


def pin_identity_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Read every identity column present as text, in place.

    Missing stays missing. An empty CSV field arrives as NaN and a Parquet null arrives
    as None, and neither is the *string* "nan": a row that carries no device identity is
    a distinct fact from one whose device is named, and `evaluate.resolve_grouping`
    reports on exactly that distinction.
    """
    for column in IDENTITY_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column]
        frame[column] = values.where(values.isna(), values.astype(str))
    return frame


def read_corpus_frame(path: Path) -> pd.DataFrame:
    """Read a corpus, the suffix deciding the reader.

    `.parquet` is the dataset `tools/results_import` publishes (§8.3) and the route a
    shipped model should come by; `.csv` is the collected corpus read directly, the
    escape hatch for a quick local run on which none of §8.3's checks have been applied;
    `.json` is the same rows as records, for fixtures and hand-written corpora. A JSON
    object rather than an array is one record, which is what a single measurement looks
    like.
    """
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame([content] if isinstance(content, dict) else content)
    else:
        frame = pd.read_csv(path, dtype=IDENTITY_DTYPES)
    return pin_identity_dtypes(frame)
