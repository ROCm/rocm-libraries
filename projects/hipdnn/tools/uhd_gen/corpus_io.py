# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read a corpus as CSV (collected), Parquet (published, §8.3) or JSON, by suffix.

Identity columns are pinned to text at the read so every format agrees: `read_csv` would
otherwise turn a device id `0123` into 123. Imports nothing from this package (cycles).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = [
    "IDENTITY_COLUMNS",
    "graph_key",
    "IDENTITY_DTYPES",
    "pin_identity_dtypes",
    "read_corpus_frame",
]

#: Columns that name a thing rather than measure one, in both the §8.3 envelope's and
#: the immediate corpus's spelling; read as text even when numeric-looking.
IDENTITY_COLUMNS = ("benchmark", "device", "graph_id", "graph_key", "device_id")


def graph_key(response: dict) -> str:
    """The content key `hipdnn_bench` reports for the graph a response measured: the one
    identity a measurement is stored, deduplicated and held out under."""
    key = response.get("graph_key")
    if not isinstance(key, str) or not key:
        raise ValueError(
            "hipdnn_bench reported no graph_key, so the graph cannot be identified by "
            "content; rebuild hipdnn_bench from this checkout"
        )
    return key


#: `dtype=` map for the CSV reader; Parquet and JSON are pinned by `pin_identity_dtypes`.
IDENTITY_DTYPES = {name: str for name in IDENTITY_COLUMNS}


def pin_identity_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Read every identity column present as text, in place; missing values stay missing."""
    for column in IDENTITY_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column]
        frame[column] = values.where(values.isna(), values.astype(str))
    return frame


def read_corpus_frame(path: Path) -> pd.DataFrame:
    """Read a corpus, the suffix deciding the reader; a JSON object is one record."""
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame([content] if isinstance(content, dict) else content)
    else:
        # round_trip: pandas' default parser is off by an ulp for about one value in ten,
        # so a written corpus would not read back as the same features.
        frame = pd.read_csv(path, dtype=IDENTITY_DTYPES, float_precision="round_trip")
    return pin_identity_dtypes(frame)
