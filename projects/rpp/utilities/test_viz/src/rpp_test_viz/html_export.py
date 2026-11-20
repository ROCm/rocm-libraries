"""Writes capture bundles into one self-contained HTML viewer (no server, no display needed)."""

from __future__ import annotations

import base64
import gzip
import json
from importlib import resources
from pathlib import Path

import numpy as np

from .bundle import Case

# Converted to float32 only where that is exact; everything else (u4/i4/u8/i8/f8) goes as float64.
_EXACT_IN_F32 = {"u1", "i1", "u2", "i2", "f2", "f4"}


def _pack(a: np.ndarray) -> str:
    """gzip + base64; the page inflates it with DecompressionStream when the case is opened."""
    return base64.b64encode(
        gzip.compress(np.ascontiguousarray(a).tobytes(), 6)
    ).decode()


def _values(a: np.ndarray) -> tuple[str, str]:
    kind = f"{a.dtype.kind}{a.dtype.itemsize}"
    target = np.float32 if kind in _EXACT_IN_F32 else np.float64
    data = a.astype(np.dtype(target).newbyteorder("<"))
    return ("f4" if target is np.float32 else "f8"), _pack(data)


def _payload(case: Case, index: int) -> dict:
    cmp = case.load(index)
    ref_type, ref = _values(cmp.reference)
    act_type, act = _values(cmp.actual)
    return {
        "case": case.name,
        "index": index,
        "count": len(case.comparisons),
        "passed": case.passed,
        "failure": case.failure,
        "kind": cmp.kind,
        "axes": cmp.axes,
        "shape": list(cmp.reference.shape),
        "dtypes": [str(cmp.reference.dtype), str(cmp.actual.dtype)],
        "compared": cmp.compared,
        "failed": cmp.failed,
        "ref": [ref_type, ref],
        "act": [act_type, act],
        "diff": _pack(cmp.diff.astype("<f4")),
        "status": _pack(cmp.status.astype(np.uint8)),
    }


def write_html(cases: list[Case], out: Path) -> int:
    """Returns the number of bytes written."""
    entries = [
        _payload(case, k) for case in cases for k in range(len(case.comparisons))
    ]
    data = json.dumps(entries, separators=(",", ":")).replace("</", "<\\/")
    template = resources.files(__package__).joinpath("html_viewer.html").read_text()
    html = template.replace("/*__RPP_TEST_VIZ_DATA__*/[]", data)
    out.write_text(html)
    return len(html.encode())
