# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""
Structured data schemas for GEMM optimization workflows.

This module defines typed representations used across the project,
including GEMM operation specifications and related configuration
entities.

This moduleprovides runtime data structures with validation and type safety.

Contents:
    - GemmType: Dataclass representing the logical GEMM operation type
        (transposes, data types, compute type).
    - GemmConfig: Bundles a GEMM type (GemmType) with the concrete set of
        problem sizes.
    - RunState: Persistent run metadata stored in workdir/run_state.json.
"""

import json

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import ClassVar, List, Tuple

from .constants import DTYPE, GEMM_LOG_FIELDS
from .utils import get_utc_timestamp, compute_file_sha256

TRANSPOSE_TYPES = ("N", "T")


def _compute_type_for_workload_log(compute_type: str) -> str:
    """Normalize compute_type to hipBLASLt workload-log form (e.g. c_f32_r)."""
    ct = compute_type.lstrip("c_")
    if ct.startswith("c_"):
        return ct
    if "32" in ct or "64" in ct:
        return "c_" + ct
    return ct


@dataclass(frozen=True)
class GemmType:
    """Specification of a GEMM operation type with field validation.

    Represents the invariant aspects of a GEMM problem: transposition
    options together with the hipBLASLt logical dtype keys and/or the
    Tensile dtype triple.  Either set may be supplied directly, but for
    consistency between the two views prefer the from_hipblaslt /
    from_tensile constructors, which fill in the other set automatically.

    Attributes:
        transA (str | bool): Transpose mode for A ("N" / "T";
            booleans are accepted and normalized to strings).
        transB (str | bool): Transpose mode for B ("N" / "T").
        a_type (str | None): hipBLASLt dtype key for matrix A
            (e.g. "f16_r"); validated against constants.DTYPE.
        b_type (str | None): hipBLASLt dtype key for matrix B.
        c_type (str | None): hipBLASLt dtype key for matrix C/D.
        compute_type (str | None): hipBLASLt accumulation compute type
            (stored without the leading c_).
        data_type (str): Tensile DataType letter or pair (e.g. "H"
            or "H8"); the canonical name fragment used in YAML.
        dest_data_type (str): Tensile DestDataType letter.
        compute_data_type (str): Tensile ComputeDataType letter.

    Note:
        a_type/b_type/c_type/compute_type must be all provided together or
        all left as None. The Tensile triple
        (data_type/dest_data_type/compute_data_type) must always be set to
        non-empty strings.
    """

    _TENSILE_LETTER_TO_HIPBLASLT: ClassVar[dict[str, str]] = {v: k for k, v in DTYPE.items()}

    transA: str | bool
    transB: str | bool
    a_type: str | None = None
    b_type: str | None = None
    c_type: str | None = None
    compute_type: str | None = None
    data_type: str = ""
    dest_data_type: str = ""
    compute_data_type: str = ""

    @staticmethod
    def format_gemm_name(
        data_type: str,
        dest_data_type: str,
        compute_data_type: str,
        transA: str,
        transB: str,
    ) -> str:
        """Tensile-style basename fragment DataDestCompute_transAtransB."""
        return f"{data_type}{dest_data_type}{compute_data_type}_{transA}{transB}"

    @property
    def gemm_name(self) -> str:
        """Same as format_gemm_name for this instance's dtype and transpose fields."""
        return self.format_gemm_name(
            self.data_type,
            self.dest_data_type,
            self.compute_data_type,
            str(self.transA),
            str(self.transB),
        )

    @staticmethod
    def _hipblaslt_to_tensile(
        a_type: str,
        b_type: str,
        c_type: str,
        compute_type: str,
    ) -> Tuple[str, str, str]:
        """Map hipBLASLt dtype keys to Tensile (data_type, dest_data_type, compute_data_type)."""
        Da, Db, Dc, Dcomp = (
            DTYPE[a_type],
            DTYPE[b_type],
            DTYPE[c_type],
            DTYPE[compute_type],
        )
        data_type = Da
        if a_type != b_type:
            data_type = f"{Da}{Db}"
        dest_data_type = Dc
        compute_data_type = Dcomp

        if Dcomp == "X":
            if (Da != Db != "S") or Dc != "S":
                raise NotImplementedError(
                    f"TF32 not implemented for a_type={a_type}, b_type={b_type}, c_type={c_type}"
                )
            data_type = "X"
            dest_data_type = "S"
            compute_data_type = "S"

        return data_type, dest_data_type, compute_data_type

    @classmethod
    def from_hipblaslt(
        cls,
        transA: str | bool,
        transB: str | bool,
        a_type: str,
        b_type: str,
        c_type: str,
        compute_type: str,
    ) -> "GemmType":
        """Construct from hipBLASLt dtype keys (also fills the Tensile triple)."""
        ct = compute_type.lstrip("c_")
        dt, dd, cd = cls._hipblaslt_to_tensile(a_type, b_type, c_type, ct)
        return cls(
            transA,
            transB,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            compute_type=ct,
            data_type=dt,
            dest_data_type=dd,
            compute_data_type=cd,
        )

    @classmethod
    def from_tensile(
        cls,
        transA: str | bool,
        transB: str | bool,
        data_type: str,
        dest_data_type: str,
        compute_data_type: str,
    ) -> "GemmType":
        """Construct from Tensile triple strings (also derives hipBLASLt logical keys)."""
        dt = str(data_type).strip()
        dd = str(dest_data_type).strip()
        cd = str(compute_data_type).strip()
        a_t, b_t, c_t, comp = cls._tensile_triple_to_hipblaslt(dt, dd, cd)
        return cls(
            transA,
            transB,
            a_type=a_t,
            b_type=b_t,
            c_type=c_t,
            compute_type=comp,
            data_type=dt,
            dest_data_type=dd,
            compute_data_type=cd,
        )

    @staticmethod
    def _tensile_triple_to_hipblaslt(
        data_type: str,
        dest_data_type: str,
        compute_data_type: str,
    ) -> Tuple[str, str, str, str]:
        """Map a Tensile triple to hipBLASLt (a_type, b_type, c_type, compute_type).

        compute_type has no c_ prefix. Raises ValueError if the triple is unknown
        or inconsistent with the round-trip through _hipblaslt_to_tensile.
        """
        dt = str(data_type).strip()
        dd = str(dest_data_type).strip()
        cd = str(compute_data_type).strip()
        m = GemmType._TENSILE_LETTER_TO_HIPBLASLT

        if dt == "X" and dd == "S" and cd == "S":
            return "xf32_r", "f32_r", "f32_r", "xf32_r"

        if len(dt) == 1:
            try:
                a_type = b_type = m[dt]
            except KeyError as e:
                raise ValueError(f"Unknown Tensile DataType letter {dt!r}") from e
        elif len(dt) == 2:
            try:
                a_type = m[dt[0]]
                b_type = m[dt[1]]
            except KeyError as e:
                raise ValueError(f"Unknown Tensile DataType code {dt!r}") from e
        else:
            raise ValueError(
                f"Tensile DataType must be 1 or 2 letters for this mapper, got {dt!r}"
            )

        try:
            c_type = m[dd]
        except KeyError as e:
            raise ValueError(f"Unknown Tensile DestDataType letter {dd!r}") from e

        try:
            compute_type = m[cd]
        except KeyError as e:
            raise ValueError(f"Unknown Tensile ComputeDataType letter {cd!r}") from e

        check_dt, check_dd, check_cd = GemmType._hipblaslt_to_tensile(
            a_type, b_type, c_type, compute_type
        )
        if (check_dt, check_dd, check_cd) != (dt, dd, cd):
            raise ValueError(
                f"Tensile triple {dt!r}/{dd!r}/{cd!r} is inconsistent with derived hipBLASLt types"
            )

        return a_type, b_type, c_type, compute_type

    def __post_init__(self):
        # Bool to T/N
        if isinstance(self.transA, bool):
            object.__setattr__(self, "transA", "T" if self.transA else "N")

        if isinstance(self.transB, bool):
            object.__setattr__(self, "transB", "T" if self.transB else "N")

        # Validate transpose values
        if self.transA not in TRANSPOSE_TYPES:
            raise ValueError(f"Invalid transA '{self.transA}', must be one of {TRANSPOSE_TYPES}")

        if self.transB not in TRANSPOSE_TYPES:
            raise ValueError(f"Invalid transB '{self.transB}', must be one of {TRANSPOSE_TYPES}")

        # hipBLASLt logical types are optional, but must be all-set or all-None
        logical = (self.a_type, self.b_type, self.c_type, self.compute_type)
        if any(x is not None for x in logical) and not all(x is not None for x in logical):
            raise ValueError(
                "a_type, b_type, c_type, and compute_type must be all set or all None"
            )

        # Validate hipBLASLt data types exist in DTYPE mapping (when present)
        if all(x is not None for x in logical):
            object.__setattr__(self, "compute_type", self.compute_type.lstrip("c_"))  # type: ignore[union-attr]
            for field_name in ("a_type", "b_type", "c_type", "compute_type"):
                value = getattr(self, field_name)
                if value not in DTYPE:
                    raise ValueError(
                        f"Invalid {field_name} '{value}'. Must be one of: {list(DTYPE.keys())}"
                    )

        # Tensile triple must always be present
        if not (self.data_type and self.dest_data_type and self.compute_data_type):
            raise ValueError(
                "data_type, dest_data_type, and compute_data_type must be non-empty strings"
            )

    def workload_log_type_fields(self) -> dict:
        """Workload-log transpose and dtype fields only (no M, N, K, batch_count).

        Requires hipBLASLt logical types set on this instance. d_type mirrors c_type.
        """
        return {
            "transA": str(self.transA).upper(),
            "transB": str(self.transB).upper(),
            "a_type": self.a_type,
            "b_type": self.b_type,
            "c_type": self.c_type,
            "d_type": self.c_type,
            "compute_type": _compute_type_for_workload_log(self.compute_type),
        }


@dataclass(frozen=True)
class GemmConfig:
    """Full GEMM optimization configuration.

    Bundles a GEMM logical type (GemmType) with the concrete set of
    problem sizes.

    Attributes:
        gemm_type (GemmType): Logical GEMM description
            (transpose flags, data types).
        sizes (List[List[int]]): List of GEMM sizes, each formatted as
            [M, N, batch_count, K].
    """

    gemm_type: GemmType
    sizes: List[List[int]]

    def __post_init__(self):
        # Validate that sizes is a list of lists of length 4
        if not isinstance(self.sizes, list) or not self.sizes:
            raise ValueError("sizes must be a non-empty list")

        for s in self.sizes:
            if not isinstance(s, list) or len(s) != 4 or not all(isinstance(x, int) and x > 0 for x in s):
                raise ValueError(f"Each size must be a list of four positive integers [M, N, batch_count, K], got: {s}")

    def workload_log_rows(self) -> List[dict]:
        """One hipBLASLt-shaped row per size (keys match constants.GEMM_LOG_FIELDS)."""
        base = self.gemm_type.workload_log_type_fields()
        rows: List[dict] = []
        for m, n, b, kk in self.sizes:
            row = {
                **base,
                "batch_count": int(b),
                "M": int(m),
                "N": int(n),
                "K": int(kk),
            }
            if set(row) != set(GEMM_LOG_FIELDS) or len(row) != len(GEMM_LOG_FIELDS):
                raise ValueError("row keys must match GEMM_LOG_FIELDS exactly")
            rows.append(row)
        return rows


@dataclass
class RunState:
    """
    Persistent run metadata stored in workdir/run_state.json.

    Tracks:
    - which input file the run is associated with.
    - SHA-256 hash of the input file.
    - creation timestamp.
    - last modification timestamp.
    - whether configuration and optimization steps are complete.
    """

    input_sha256: str
    input_path: str
    created_at: str
    last_modified: str
    configured: bool = False
    optimized: bool = False

    @classmethod
    def create(cls, input_path: str | Path) -> "RunState":
        """Create a new RunState from an input file."""
        input_path = str(input_path)
        sha = compute_file_sha256(input_path)
        timestamp = get_utc_timestamp()
        return cls(
            input_sha256=sha,
            input_path=input_path,
            created_at=timestamp,
            last_modified=timestamp,
        )

    def dump(self, path: str | Path) -> None:
        """Serialize RunState to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "last_modified", get_utc_timestamp())

        with path.open("w") as f:
            json.dump(
                asdict(self),
                f,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        """Load RunState from a JSON file."""
        with Path(path).open("r") as f:
            data = json.load(f)
        return cls(**data)

    def verify(self, current_input: str | Path) -> None:
        """Validate that the current input matches the one used to create this run.

        Args:
            current_input (str | Path): Input path being checked against the
                value recorded in this RunState.

        Raises:
            ValueError: If current_input does not match the stored
                input_path, or if the file's SHA-256 does not match
                the stored input_sha256.
        """
        current_input = str(current_input)

        if current_input != self.input_path:
            raise ValueError(f"Workdir belongs to input '{self.input_path}', " f"but got '{current_input}'")

        current_sha = compute_file_sha256(current_input)
        if current_sha != self.input_sha256:
            raise ValueError(f"Input file {current_input!r} contents changed (hash mismatch)")
