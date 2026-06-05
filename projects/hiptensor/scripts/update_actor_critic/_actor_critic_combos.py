"""
_actor_critic_combos.py - Shared definitions for actor-critic probe/patch scripts.

Describes every ActorCriticSelection and ActorCriticSelectionUnaryOps
specialization in contraction_selection.cpp, plus helpers for building
YAML probe configs and locating the matching C++ struct blocks.

Not intended to be run directly.
"""

import math
import re
import textwrap

# ---------------------------------------------------------------------------
# Type-combination table
#
# Each row: (key_suffix, op, data_types_yaml, cpp_A, cpp_compute)
# Complex rows carry an extra field: explicit ContractionOpId_t string.
# The D/B/E C++ template types are always the same as A in all specializations.
# ---------------------------------------------------------------------------

_TYPE_DEFS = [
    # key_suffix                            op          data_types_yaml [A,B,D,E,compute]                                                                   cpp_A            cpp_compute
    ("f16_f16_f16_f16_scale_f16",         "scale",    ["HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "NONE_TYPE",        "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F"],  "_Float16",      "_Float16"),
    ("f16_f16_f16_f16_bilinear_f16",      "bilinear", ["HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F"],  "_Float16",      "_Float16"),
    ("f16_f16_f16_f16_scale_f32",         "scale",    ["HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "NONE_TYPE",        "HIPTENSOR_R_16F",  "HIPTENSOR_R_32F"],  "_Float16",      "float"),
    ("f16_f16_f16_f16_bilinear_f32",      "bilinear", ["HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_16F",  "HIPTENSOR_R_32F"],  "_Float16",      "float"),
    ("bf16_bf16_bf16_bf16_scale_bf16",    "scale",    ["HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "NONE_TYPE",        "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF"], "hip_bfloat16",  "hip_bfloat16"),
    ("bf16_bf16_bf16_bf16_bilinear_bf16", "bilinear", ["HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF"], "hip_bfloat16",  "hip_bfloat16"),
    ("bf16_bf16_bf16_bf16_scale_f32",     "scale",    ["HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "NONE_TYPE",        "HIPTENSOR_R_16BF", "HIPTENSOR_R_32F"],  "hip_bfloat16",  "float"),
    ("bf16_bf16_bf16_bf16_bilinear_f32",  "bilinear", ["HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_16BF", "HIPTENSOR_R_32F"],  "hip_bfloat16",  "float"),
    ("f32_f32_f32_f32_scale_f16",         "scale",    ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "NONE_TYPE",        "HIPTENSOR_R_32F",  "HIPTENSOR_R_16F"],  "float",         "_Float16"),
    ("f32_f32_f32_f32_bilinear_f16",      "bilinear", ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_16F"],  "float",         "_Float16"),
    ("f32_f32_f32_f32_scale_bf16",        "scale",    ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "NONE_TYPE",        "HIPTENSOR_R_32F",  "HIPTENSOR_R_16BF"], "float",         "hip_bfloat16"),
    ("f32_f32_f32_f32_bilinear_bf16",     "bilinear", ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_16BF"], "float",         "hip_bfloat16"),
    ("f32_f32_f32_f32_scale_f32",         "scale",    ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "NONE_TYPE",        "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F"],  "float",         "float"),
    ("f32_f32_f32_f32_bilinear_f32",      "bilinear", ["HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F",  "HIPTENSOR_R_32F"],  "float",         "float"),
    ("f64_f64_f64_f64_scale_f32",         "scale",    ["HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "NONE_TYPE",        "HIPTENSOR_R_64F",  "HIPTENSOR_R_32F"],  "double",        "float"),
    ("f64_f64_f64_f64_bilinear_f32",      "bilinear", ["HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_32F"],  "double",        "float"),
    ("f64_f64_f64_f64_scale_f64",         "scale",    ["HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "NONE_TYPE",        "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F"],  "double",        "double"),
    ("f64_f64_f64_f64_bilinear_f64",      "bilinear", ["HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F",  "HIPTENSOR_R_64F"],  "double",        "double"),
    # Complex types use SCALE_COMPLEX / BILINEAR_COMPLEX in the C++ template,
    # and have no ActorCriticSelectionUnaryOps variants.
    ("cf32_scale",                        "scale",    ["HIPTENSOR_C_32F",  "HIPTENSOR_C_32F",  "NONE_TYPE",        "HIPTENSOR_C_32F",  "HIPTENSOR_C_32F"],  "hipFloatComplex",  "hipFloatComplex",  "SCALE_COMPLEX"),
    ("cf32_bilinear",                     "bilinear", ["HIPTENSOR_C_32F",  "HIPTENSOR_C_32F",  "HIPTENSOR_C_32F",  "HIPTENSOR_C_32F",  "HIPTENSOR_C_32F"],  "hipFloatComplex",  "hipFloatComplex",  "BILINEAR_COMPLEX"),
    ("cf64_scale",                        "scale",    ["HIPTENSOR_C_64F",  "HIPTENSOR_C_64F",  "NONE_TYPE",        "HIPTENSOR_C_64F",  "HIPTENSOR_C_64F"],  "hipDoubleComplex", "hipDoubleComplex", "SCALE_COMPLEX"),
    ("cf64_bilinear",                     "bilinear", ["HIPTENSOR_C_64F",  "HIPTENSOR_C_64F",  "HIPTENSOR_C_64F",  "HIPTENSOR_C_64F",  "HIPTENSOR_C_64F"],  "hipDoubleComplex", "hipDoubleComplex", "BILINEAR_COMPLEX"),
]

# Layouts: name + YAML Memory Layout value.
# The layout is controlled via the YAML "Memory Layout" field, which causes the
# test binary to generate explicit strides before calling the library API.
# Do NOT use the HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR environment variable —
# that only affects the library's internal default and is ignored when the test
# generates strides explicitly from the Memory Layout field.
LAYOUTS: list[tuple[str, str]] = [
    ("col_major", "HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR"),
    ("row_major", "HIPTENSOR_MEMORY_LAYOUT_ROW_MAJOR"),
]

ALL_RANKS: list[int] = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _op_id(op: str, is_complex: bool) -> str:
    if is_complex:
        return "BILINEAR_COMPLEX" if op == "bilinear" else "SCALE_COMPLEX"
    return "BILINEAR" if op == "bilinear" else "SCALE"


def _make_cpp_match(struct_name: str, cpp_A: str, cpp_compute: str,
                    op_id: str) -> str:
    """Regex matching the template<> struct header for one specialization."""
    ws = r"\s*"
    comma = ws + r"," + ws
    return (
        rf"struct\s+{struct_name}<"
        rf"{re.escape(cpp_A)}{comma}"          # A
        rf"{re.escape(cpp_A)}{comma}"          # B
        rf"{re.escape(cpp_A)}{comma}"          # D
        rf"{re.escape(cpp_A)}{comma}"          # E
        rf"ContractionOpId_t::{op_id}{comma}"
        rf"{re.escape(cpp_compute)}>"
    )


def _regular_binary(op: str, is_complex: bool) -> str:
    if is_complex:
        return ("complex_bilinear_contraction_test" if op == "bilinear"
                else "complex_scale_contraction_test")
    return ("bilinear_contraction_test" if op == "bilinear"
            else "scale_contraction_test")


def _unary_binary(op: str) -> str:
    return ("bilinear_contraction_with_unary_ops_test" if op == "bilinear"
            else "scale_contraction_with_unary_ops_test")


# ---------------------------------------------------------------------------
# Public: TYPE_COMBINATIONS
#
# List of dicts, one per ActorCritic[UnaryOps] specialization, with keys:
#   key          unique string, used as the primary key in the JSON
#   op           "bilinear" | "scale"
#   data_types   list of 5 YAML data-type strings [A, B, D, E, compute]
#   test_binary  binary name (without path); ranked ones get _mXnXkX appended
#   ranked       True  → binary name has a _m{N}n{N}k{N} rank suffix
#                False → single binary covers all ranks via YAML
#   unary        True  → ActorCriticSelectionUnaryOps
#   cpp_match    regex matching the template<> struct ... header line(s)
# ---------------------------------------------------------------------------

TYPE_COMBINATIONS: list[dict] = []

for row in _TYPE_DEFS:
    key_suffix  = row[0]
    op          = row[1]
    data_types  = row[2]
    cpp_A       = row[3]
    cpp_compute = row[4]
    is_complex  = key_suffix.startswith("cf")
    # Complex rows carry an explicit op_id override at index 5
    cpp_op_id   = row[5] if is_complex else _op_id(op, is_complex)  # type: ignore[misc]

    TYPE_COMBINATIONS.append(dict(
        key=key_suffix,
        op=op,
        data_types=data_types,
        test_binary=_regular_binary(op, is_complex),
        ranked=True,
        unary=False,
        cpp_match=_make_cpp_match("ActorCriticSelection", cpp_A,
                                  cpp_compute, cpp_op_id),
    ))

    if not is_complex:
        TYPE_COMBINATIONS.append(dict(
            key=f"unary_{key_suffix}",
            op=op,
            data_types=data_types,
            test_binary=_unary_binary(op),
            ranked=False,
            unary=True,
            cpp_match=_make_cpp_match("ActorCriticSelectionUnaryOps", cpp_A,
                                      cpp_compute, cpp_op_id),
        ))


# ---------------------------------------------------------------------------
# Public: YAML generation
# ---------------------------------------------------------------------------

def _rank_lengths_and_modes(rank: int) -> tuple[
        list[int], list[int], list[int], list[int], list[int], list[int]]:
    # Mode numbering convention matching the hipTensor test suite:
    #   M modes: [0 .. rank-1]
    #   N modes: [rank .. 2*rank-1]
    #   K modes: [2*rank .. 3*rank-1]
    # A = M + K,  B = N + K,  E = M + N
    m_modes = list(range(0, rank))
    n_modes = list(range(rank, 2 * rank))
    k_modes = list(range(2 * rank, 3 * rank))
    a_modes = m_modes + k_modes
    b_modes = n_modes + k_modes
    e_modes = m_modes + n_modes

    # Choose per-mode dimension so that tensor A has roughly 16 M elements,
    # giving enough GPU work for a meaningful timing comparison while staying
    # well within GPU memory at all ranks (1-6).
    #   rank 1 → 2 modes in A → dim = 4096  (A ~ 16 M)
    #   rank 2 → 4 modes in A → dim =  64   (A ~ 16 M)
    #   rank 3 → 6 modes in A → dim =  16   (A ~ 16 M)
    #   rank 4 → 8 modes in A → dim =   8   (A ~ 16 M)
    #   rank 5 → 10 modes in A → dim =   5  (A ~ 10 M)
    #   rank 6 → 12 modes in A → dim =   4  (A ~ 16 M)
    target = 16 * 1024 * 1024  # ~16 M elements in A
    n_a_modes = 2 * rank
    dim = max(4, int(math.pow(target, 1.0 / n_a_modes)))

    a_len = [dim] * len(a_modes)
    b_len = [dim] * len(b_modes)
    e_len = [dim] * len(e_modes)
    return a_len, a_modes, b_len, b_modes, e_len, e_modes


def make_probe_yaml(combo: dict, rank: int, memory_layout: str) -> str:
    """Minimal YAML config for DEFAULT_PATIENT + HEURISTICS_TRACE logging.

    memory_layout: one of 'HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR' or
                   'HIPTENSOR_MEMORY_LAYOUT_ROW_MAJOR'.
    """
    data_types = combo["data_types"]
    a_len, a_modes, b_len, b_modes, e_len, e_modes = _rank_lengths_and_modes(rank)

    # Unary-ops tests need a non-identity op to exercise the unary dispatch path
    ops_line = (
        "  - [HIPTENSOR_OP_RELU, HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY]"
        if combo["unary"] else
        "  - [HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY]"
    )

    return textwrap.dedent(f"""\
        ---
        Log Level: [ HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE ]
        Tensor Data Types:
          - [ {', '.join(data_types)} ]
        Algorithm Types:
          - HIPTENSOR_ALGO_DEFAULT_PATIENT
        Operators:
        {ops_line}
        Worksize Prefs:
          - HIPTENSOR_WORKSPACE_MAX
        Alphas:
          - [1]
        Betas:
          - [1]
        Lengths:
          - [{', '.join(map(str, a_len))}, {', '.join(map(str, b_len))}, {', '.join(map(str, e_len))}]
        Strides:
          - []
        Modes:
          - [{', '.join(map(str, a_modes))}, {', '.join(map(str, b_modes))}, {', '.join(map(str, e_modes))}]
        Memory Layout:
          - {memory_layout}
        ...
        """)


# ---------------------------------------------------------------------------
# Public: log parsing
# ---------------------------------------------------------------------------

_PERF_RE = re.compile(
    r"BRUTE_FORCE_KERNEL_PERF\]\s+"
    r"KernelId:\s*\d+,\s*KernelName:\s*(.+?),\s*AvgTime:\s*([\d.]+)\s*ms"
)


def parse_best_kernel(output: str) -> tuple[str | None, float | None]:
    """
    Parse all BRUTE_FORCE_KERNEL_PERF lines and return the kernel name and
    time (ms) of the one with the lowest AvgTime.  Returns (None, None) if
    no lines were found.
    """
    best_name: str | None = None
    best_time: float = float("inf")
    for line in output.splitlines():
        m = _PERF_RE.search(line)
        if m:
            name = m.group(1).strip()
            t    = float(m.group(2))
            if t < best_time:
                best_time = t
                best_name = name
    return (best_name, best_time if best_name is not None else None)
