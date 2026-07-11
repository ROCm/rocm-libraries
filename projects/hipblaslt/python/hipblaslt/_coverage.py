# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse hipBLASLt header enums to drive API-surface coverage tests."""
import os
import re
from pathlib import Path

# header enum typedef -> set of integer values intentionally not bound.
# Extend here rather than silently skipping new gaps.
ALLOWED_MISSING = {
    # Epilogue variants bound here are the basic set (DEFAULT, RELU, BIAS,
    # RELU_BIAS, GELU, GELU_BIAS, SIGMOID_EXT). The remaining values are
    # gradient epilogues (DGELU*), auxiliary-output variants (*_AUX*), bias-
    # gradient variants (BGRADA/BGRADB), and extended activation epilogues
    # (SWISH_EXT, CLAMP_EXT, SIGMOID_BIAS_EXT). These are not yet surfaced by
    # the public Python API; add them here so the harness stays green until
    # enums.cpp is extended.
    "hipblasLtEpilogue_t": {
        130,    # HIPBLASLT_EPILOGUE_RELU_AUX
        134,    # HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
        160,    # HIPBLASLT_EPILOGUE_GELU_AUX
        164,    # HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
        192,    # HIPBLASLT_EPILOGUE_DGELU
        208,    # HIPBLASLT_EPILOGUE_DGELU_BGRAD
        256,    # HIPBLASLT_EPILOGUE_BGRADA
        512,    # HIPBLASLT_EPILOGUE_BGRADB
        65536,  # HIPBLASLT_EPILOGUE_SWISH_EXT
        65540,  # HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
        131072, # HIPBLASLT_EPILOGUE_CLAMP_EXT
        131076, # HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT
        131200, # HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT
        131204, # HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT
        262148, # HIPBLASLT_EPILOGUE_SIGMOID_BIAS_EXT
    },
    # ScaleMode: VEC16_UE4M3 (1) and the "not supported yet" entries
    # VEC128_32F (4), BLK128x128_32F (5), and the sentinel END (6) are not
    # bound. BLK32_UE8M0_32_8_EXT (1001) is version-gated (>1.2.x) and is
    # not present in the installed 1.2.2 header at all.
    "hipblasLtMatmulMatrixScale_t": {
        1,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 — not supported yet
        4,  # HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F — not supported yet
        5,  # HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F — not supported yet
        6,  # HIPBLASLT_MATMUL_MATRIX_SCALE_END — sentinel, not a real mode
    },
    # MatmulDescAttr: only the subset needed for basic GEMM dispatch is bound.
    # The unbound members are: BIAS_DATA_TYPE (4), C_SCALE_POINTER (7),
    # EPILOGUE_AUX_* (9-12), POINTER_MODE (13), AMAX_D_POINTER (14),
    # EPILOGUE_AUX_DATA_TYPE (22), COMPUTE_INPUT_TYPE_*_EXT (100, 101),
    # EPILOGUE_ACT_ARG*_EXT (102, 103), and the sentinel MAX (104).
    "hipblasLtMatmulDescAttributes_t": {
        4,   # HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
        7,   # HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER
        9,   # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
        10,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
        11,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
        12,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE
        13,  # HIPBLASLT_MATMUL_DESC_POINTER_MODE
        14,  # HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
        22,  # HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
        100, # HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT
        101, # HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT
        102, # HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT
        103, # HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT
        104, # HIPBLASLT_MATMUL_DESC_MAX — sentinel
    },
}


def find_header():
    """Locate hipblaslt.h, preferring the installed ROCm header.

    The Python extension is compiled against the *installed* ROCm SDK, so the
    installed header is the authoritative source for value checking.  The
    in-tree header is used as a fallback for developer environments where no
    ROCm SDK is installed (e.g. CI without a GPU node or header-only checks).

    Search order:
    1. ``$ROCM_PATH/include/hipblaslt/hipblaslt.h`` (or ``/opt/rocm``).
    2. Walk up from this file's location to find the in-tree header at
       ``library/include/hipblaslt/hipblaslt.h`` (developer build fallback).

    Returns a :class:`pathlib.Path` to the first found header.
    Raises :class:`FileNotFoundError` if none exist.
    """
    candidates = []

    # Installed ROCm header — preferred because the extension is compiled
    # against the installed SDK, so values are guaranteed to match.
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates.append(Path(rocm) / "include" / "hipblaslt" / "hipblaslt.h")

    # In-tree fallback: walk up from python/hipblaslt/_coverage.py.
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / "library" / "include" / "hipblaslt" / "hipblaslt.h"
        if p.exists():
            candidates.append(p)
            break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"hipblaslt.h not found; searched: {[str(c) for c in candidates]}"
    )


def header_enum_values(header_path, enum_type):
    """Parse a ``typedef enum { ... } enum_type;`` block from *header_path*.

    Returns a ``{member_name: int_value}`` dict.  Simple ``= N`` and
    ``= 0xHEX`` assignments are resolved; members without an explicit value
    receive the previous value plus one (C auto-increment semantics).

    Block comments (``/* ... */``) and line comments (``// ...``) are stripped
    before name/value extraction.

    Parameters
    ----------
    header_path : str or pathlib.Path
        Path to ``hipblaslt.h``.
    enum_type : str
        The C typedef name, e.g. ``"hipblasLtEpilogue_t"``.

    Returns
    -------
    dict[str, int]
    """
    text = Path(header_path).read_text()

    # Match: typedef enum { ... } enum_type;
    # Use [^}]* instead of .*? to avoid spanning across multiple enum blocks
    # when the file contains several consecutive typedef enums (re.DOTALL would
    # allow .*? to skip over the closing brace of an earlier enum).
    pattern = re.compile(
        r"typedef\s+enum\s*\{(?P<body>[^}]*)\}\s*"
        + re.escape(enum_type)
        + r"\s*;",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"enum {enum_type!r} not found in {header_path}")

    body = match.group("body")

    # Strip block comments then line comments before tokenising.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", "", body)

    values: dict[str, int] = {}
    current = -1

    for raw_token in body.split(","):
        token = raw_token.strip()
        if not token:
            continue

        # Match "NAME" or "NAME = VALUE" (decimal or hex, optional sign).
        m = re.match(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(0[xX][0-9a-fA-F]+|-?\d+))?",
            token,
        )
        if not m:
            continue

        name = m.group(1)
        if m.group(2) is not None:
            current = int(m.group(2), 0)
        else:
            current += 1

        values[name] = current

    return values
