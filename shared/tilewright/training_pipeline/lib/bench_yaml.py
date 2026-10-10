# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt-bench yaml writer and line-oriented reader.

A bench yaml holds one flow mapping per line, `- {function: matmul, ...}`,
describing one GEMM with all its bench knobs inline. Reading is regex based
(no yaml load) so large shape files stay cheap to process.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .fs import atomic_write_text
from .shapes import Shape, footprint_bytes, known_bytes_per_element


def strides_for(
    transA: str, transB: str, m: int, n: int, k: int
) -> Tuple[int, int, int, int]:
    """Default leading dimensions for hipBLASLt's column-major matrices."""
    lda = m if transA == "N" else k
    ldb = k if transB == "N" else n
    return lda, ldb, m, m


DEFAULT_LINE_KNOBS = {
    "alpha": 1,
    "beta": 0,
    "initialization": "trig_float",
    "use_gpu_timer": 1,
    "flush": "true",
    "rotating": 512,
    "print_kernel_info": 1,
    "requested_solution_num": -1,
}

# ── adaptive timing ──────────────────────────────────────────────────────────
#
# Every bench line has `adaptive: true`: hipblaslt-bench times each kernel
# until the relative standard error of its mean drops under `noise_threshold`
# (after `min_iters` iterations and `measure_time` ms, at most
# `max_measure_time` ms), instead of running a fixed iteration count. The yaml
# keys are the `Arguments` struct field names (hipblaslt_arguments.hpp), not
# the `--adaptive_*` command-line spellings: an unknown yaml key is silently
# ignored by the bench. Every key is written, so that a line does not depend
# on the bench's own defaults.
ADAPTIVE_DEFAULTS: Dict[str, Any] = {
    "warmup_time": 1.0,
    "sample_time": 0.5,
    "measure_time": 1.5,
    "max_measure_time": 8.0,
    "min_iters": 3,
    "max_iters": 0,
    "noise_threshold": 0.05,
    "stability_threshold": 0.05,
    "stability_window": 32,
    "stability_interval": 16,
}
ADAPTIVE_KEYS = tuple(ADAPTIVE_DEFAULTS)

# The bench rejects `adaptive` combined with iters / cold_iters.
ITER_KEYS = ("iters", "cold_iters")

# ── long lines ───────────────────────────────────────────────────────────────
#
# A shorter sample averages fewer calls of a short kernel and reads it
# noisier, but a kernel of a few hundred microseconds or more takes few
# calls per sample either way, so its line can warm up, sample and stop
# sooner without reading noisier. `bench.adaptive_long` (opt-in) gives a
# line the knobs LONG_KNOB_KEYS of the block over `bench.adaptive` when its
# GEMM cannot run faster than `min_roofline_us`: its roofline lower bound,
# the larger of its arithmetic at the data type's peak matrix rate
# (`peak_tflops`) and of reading A and B and writing D once at the memory
# bandwidth (`bandwidth_gbs`). The rule needs only the shape and the
# config, so every stage gives a GEMM the same knobs. The orchestrator
# fills the rates it is not given from the bench device when a run starts
# (lib/hardware.py `roofline_rates`); a rate left unknown leaves its term
# out, and so does a data type of unknown size the memory term, which keeps
# the bound a lower bound.
ADAPTIVE_LONG_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "min_roofline_us": 100.0,
    "warmup_time": 0.5,
    "sample_time": 0.25,
    "max_measure_time": 4.0,
    "peak_tflops": None,
    "bandwidth_gbs": None,
}
LONG_KNOB_KEYS = ("warmup_time", "sample_time", "max_measure_time")


def adaptive_long_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The config's `bench.adaptive_long` over ADAPTIVE_LONG_DEFAULTS."""
    blk = (config.get("bench") or {}).get("adaptive_long") or {}
    return {k: blk.get(k, d) for k, d in ADAPTIVE_LONG_DEFAULTS.items()}


def roofline_us(
    shape: Tuple[int, int, int, int],
    hl: Mapping[str, Any],
    peak_tflops: Optional[float],
    bandwidth_gbs: Optional[float],
) -> Optional[float]:
    """The roofline lower bound of a GEMM's latency in us (see the note
    above) for the data types of the config's `hipblaslt` section `hl`;
    None when no term is known (no peak, and no bandwidth or a data type of
    unknown size)."""
    m, n, k, b = (int(x) for x in shape)
    terms = []
    if peak_tflops:
        terms.append(2.0 * m * n * k * b / (float(peak_tflops) * 1e6))
    sizes = [
        known_bytes_per_element(hl.get("a_type")),
        known_bytes_per_element(hl.get("b_type", hl.get("a_type"))),
        known_bytes_per_element(hl.get("d_type", hl.get("c_type"))),
    ]
    if bandwidth_gbs and None not in sizes:
        size_a, size_b, size_d = (float(x) for x in sizes)
        nbytes = b * (m * k * size_a + n * k * size_b + m * n * size_d)
        terms.append(nbytes / (float(bandwidth_gbs) * 1e3))
    return max(terms) if terms else None


def is_long_line(shape: Tuple[int, int, int, int], config: Mapping[str, Any]) -> bool:
    """Whether `bench.adaptive_long` gives `shape`'s line its knobs.
    ValueError when the block is on with neither rate."""
    st = adaptive_long_settings(config)
    if not st["enabled"]:
        return False
    bound = roofline_us(
        shape, config.get("hipblaslt") or {}, st["peak_tflops"], st["bandwidth_gbs"]
    )
    if bound is None:
        raise ValueError(
            "bench.adaptive_long needs peak_tflops or bandwidth_gbs, the latter "
            "with hipblaslt data types of known size; the orchestrator reads "
            "the rates from the bench device when a run starts"
        )
    return bound >= float(st["min_roofline_us"])


def adaptive_settings(
    config: Mapping[str, Any], shape: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, Any]:
    """The adaptive timing fields of a bench line: the config's
    `bench.adaptive` values over ADAPTIVE_DEFAULTS, and for the line of
    `shape` the knobs of `bench.adaptive_long` over them when it is a long
    line (`is_long_line`).

        bench:
          adaptive:
            measure_time: 2.0      # ms per kernel
            ...
    """
    blk = (config.get("bench") or {}).get("adaptive") or {}
    out = {k: blk.get(k, d) for k, d in ADAPTIVE_DEFAULTS.items()}
    if shape is not None and is_long_line(shape, config):
        long_knobs = adaptive_long_settings(config)
        out.update({k: long_knobs[k] for k in LONG_KNOB_KEYS})
    return out


def line_is_adaptive(line: str) -> bool:
    m = re.search(r"(?<![\w])adaptive:\s*(\w+)", line)
    return m is not None and m.group(1).lower() in ("true", "1")


def adaptive_line(line: str, config: Mapping[str, Any]) -> str:
    """`line` timed adaptively: a line with fixed iteration counts (written
    before every bench was adaptive) gets `adaptive: true` and the config's
    adaptive fields for its shape in their place; an adaptive line is
    returned as it is."""
    if line_is_adaptive(line):
        return line
    for key in ITER_KEYS:
        line = drop_line_knob(line, key)
    line = set_line_knob(line, "adaptive", "true")
    for key, value in adaptive_settings(config, shape_of_line(line)).items():
        line = set_line_knob(line, key, value)
    return line


# ── kernel order ─────────────────────────────────────────────────────────────
#
# A kernel timed early in a long line runs before the device has settled at
# its power limit and reads faster than the kernels after it, so a line in a
# ranker's order flatters the kernels the ranker put first. Bench lines list
# their kernels in a seeded shuffle of what they bench instead
# (`bench.kernel_order: shuffled`, seed `bench.order_seed`); `listed` keeps
# the order they were listed in.
KERNEL_ORDERS = ("shuffled", "listed")
DEFAULT_ORDER_SEED = 42


def kernel_order_seed(config: Mapping[str, Any]) -> Optional[int]:
    """The seed of the shuffle bench lines list their kernels in, or None
    with `bench.kernel_order: listed`."""
    bench = config.get("bench") or {}
    if bench.get("kernel_order", "shuffled") == "listed":
        return None
    return int(bench.get("order_seed", DEFAULT_ORDER_SEED))


# ── sacrificial kernel ───────────────────────────────────────────────────────
#
# The skip screen judges every kernel by one cold call. The first listed
# kernel's call comes right after the bench initialised the problem's data,
# on a device that has been idle, and reads several times too slow, so the
# screen drops it and its warm-up becomes its label. A screened line
# therefore lists its first kernel twice (`bench.sacrificial_kernel`): the
# copy at position 0 takes that call, and every reader of the log sets its
# row aside (lib/bench_log.py).


def sacrificial_kernel(config: Mapping[str, Any]) -> bool:
    """Screened bench lines lead with a sacrificial copy of their first
    kernel (`bench.sacrificial_kernel`, on by default)."""
    return bool((config.get("bench") or {}).get("sacrificial_kernel", True))


# ── warm-up line ─────────────────────────────────────────────────────────────
#
# A bench process loads a library file's code object when it first runs one
# of the file's kernels, and the screen reads the cold call of that kernel
# several times too slow, after the prepass's untimed call too, and drops
# it. The sacrificial copy takes that call only for the file of a line's
# first kernel: the first kernel of every other file stays exposed. So
# every bench process of stage03 starts with a throwaway warm-up line
# (`bench.warmup_line`): a small fixed problem (`size` cubed, batch 1) of
# the stage's problem type, timed and screened as the stage's lines are,
# over `kernels` of the kernels they list, one of each library file in turn
# and at least one of each (`warmup_kernels`), which loads the files before
# the first line. Its output is discarded; the stage's platform.json records
# the line.
WARMUP_LINE_DEFAULTS: Dict[str, Any] = {"enabled": True, "kernels": 4, "size": 512}


def warmup_line_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The config's `bench.warmup_line` over WARMUP_LINE_DEFAULTS."""
    blk = (config.get("bench") or {}).get("warmup_line")
    blk = blk if isinstance(blk, Mapping) else {}
    return {k: blk.get(k, d) for k, d in WARMUP_LINE_DEFAULTS.items()}


def warmup_kernels(
    lines: Iterable[str],
    count: int,
    files: Optional[Sequence[Iterable[int]]] = None,
) -> List[int]:
    """Up to `count` of the distinct kernels `lines` list, for the warm-up
    line. With `files`, the solution indices of each library file the bench
    loads, one kernel of each file in turn (a file's in the order the lines
    first list them, the kernels of no file last as one more file), and at
    least one of each file the lines list, beyond `count` if need be;
    without, kernels spread evenly over the listed indices in index order,
    which reaches the files that hold a fair share of them, since a library
    build numbers each file's solutions consecutively. Empty when no line
    lists a kernel."""
    listed: Dict[int, None] = {}
    for line in lines:
        for k in line_solution_list(line) or ():
            listed.setdefault(int(k), None)
    count = int(count)
    if not listed or count < 1:
        return []
    if files is None:
        ranked = sorted(listed)
        if count >= len(ranked):
            return ranked
        if count == 1:
            return ranked[:1]
        picks = (round(i * (len(ranked) - 1) / (count - 1)) for i in range(count))
        return [ranked[p] for p in dict.fromkeys(picks)]
    groups: List[List[int]] = [[] for _ in range(len(files) + 1)]
    owner: Dict[int, int] = {}
    for i, ids in enumerate(files):
        for sid in ids:
            owner.setdefault(int(sid), i)
    for k in listed:
        groups[owner.get(k, len(files))].append(k)
    count = max(count, sum(1 for g in groups if g))
    out: List[int] = []
    for depth in range(max(len(g) for g in groups)):
        for g in groups:
            if depth < len(g) and len(out) < count:
                out.append(g[depth])
    return out


def warmup_line(
    template: str,
    kernels: Sequence[int],
    *,
    count: int,
    size: int,
    config: Mapping[str, Any],
) -> str:
    """`template`, a bench line of the stage, restamped as its warm-up line:
    the problem `size` x `size` x `size`, batch 1, with the template's
    types, layout, screen and other knobs; leading dimensions, strides and
    the rotating buffer of that problem; the config's adaptive fields (none
    of a long line's); `kernels` as its list, or without, the first `count`
    the bench takes for the problem (`requested_solution_num`); no entry
    name. Newline-terminated."""
    line = drop_line_knob(template.rstrip("\n"), ENTRY_NAME_KEY)
    m = n = k = int(size)
    lda, ldb, ldc, ldd = strides_for(
        line_knob(line, "transA") or "N", line_knob(line, "transB") or "N", m, n, k
    )
    for key, value in (
        ("M", m),
        ("N", n),
        ("K", k),
        ("batch_count", 1),
        ("lda", lda),
        ("ldb", ldb),
        ("ldc", ldc),
        ("ldd", ldd),
    ):
        line = set_line_knob(line, key, value)
    for key in ("stride_a", "stride_b", "stride_c", "stride_d"):
        if line_knob(line, key) is not None:
            line = set_line_knob(line, key, 0)
    for key in ITER_KEYS:
        line = drop_line_knob(line, key)
    line = set_line_knob(line, "adaptive", "true")
    for key, value in adaptive_settings(config).items():
        line = set_line_knob(line, key, value)
    rotating = _int_or_none(line_knob(line, "rotating"))
    if rotating:
        hl: Dict[str, Any] = {t: line_knob(line, t) for t in TYPE_COLUMNS[:4]}
        for key in SCALE_COLUMNS:
            mode = _int_or_none(line_knob(line, key))
            if mode is not None:
                hl[key] = mode
        sized = rotating_mb_for(
            Shape(m, n, k, 1), {"hipblaslt": hl, "bench": config.get("bench") or {}}
        )
        if sized:
            line = set_line_knob(line, "rotating", min(rotating, sized))
    if kernels:
        line = drop_line_knob(line, "algo_method")
        line = set_line_knob(line, "requested_solution_num", -1)
        line = set_line_knob(line, "solution_index_list", format_solution_list(kernels))
    else:
        line = drop_line_knob(line, "solution_index_list")
        line = set_line_knob(line, "requested_solution_num", int(count))
    return line + "\n"


def _int_or_none(text: Optional[str]) -> Optional[int]:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def line_skip_ratio(line: str) -> float:
    """The line's `skip_slow_solution_ratio`; 0 (no screen) without one."""
    m = re.search(rf"(?<![\w])skip_slow_solution_ratio:\s*({_SCALAR_VALUE})", line)
    try:
        return float(m.group(1)) if m else 0.0
    except ValueError:
        return 0.0


def _default_scale_for_dtype(dtype: str) -> int:
    """Default scaleA / scaleB mode when the config sets none: 1 (Scalar, the
    `UseScaleAB: Scalar` F8 libraries) for f8 operands, 0 otherwise. Mode 2
    would be Vector, which selects a different kernel family."""
    return 1 if "f8_r" in dtype else 0


# ── rotating buffer sizing ───────────────────────────────────────────────────
#
# `rotating` is a rotating-buffer budget in MiB; the bench fills it with as
# many copies of the problem as fit and cycles through them so consecutive
# runs read cold data. A fixed budget turns into millions of copies of a
# small problem, so the budget is sized to the problem instead:
#     rotating_mib = ceil(footprint * bench.rotating_target_blocks / MiB)
# clamped to [1, bench.rotating_mb]. Never 0: without rotation, candidates
# benched back to back on the same buffers read them warm, and the kernel
# measured last looks fastest.
_MIB = 1024 * 1024


def _scale_modes(hl: Dict[str, Any]) -> Tuple[int, int]:
    a_t = str(hl.get("a_type", "bf16_r"))
    b_t = str(hl.get("b_type", a_t))
    scale_a = int(hl["scaleA"]) if "scaleA" in hl else _default_scale_for_dtype(a_t)
    scale_b = int(hl["scaleB"]) if "scaleB" in hl else _default_scale_for_dtype(b_t)
    return scale_a, scale_b


def _problem_footprint(shape: Shape, hl: Dict[str, Any]) -> int:
    scale_a, scale_b = _scale_modes(hl)
    return footprint_bytes(
        shape.m,
        shape.n,
        shape.k,
        shape.batch,
        a_type=hl.get("a_type"),
        b_type=hl.get("b_type"),
        c_type=hl.get("c_type"),
        d_type=hl.get("d_type"),
        scale_a=scale_a,
        scale_b=scale_b,
    )


def rotating_mb_for(shape: Shape, config: Dict[str, Any]) -> int:
    """Rotating-buffer MiB for one shape (see the sizing note above)."""
    bench = config.get("bench") or {}
    cap = int(bench.get("rotating_mb", 512))
    target = int(bench.get("rotating_target_blocks", 16))
    if cap <= 0 or target <= 0:
        return 0
    want = _problem_footprint(shape, config["hipblaslt"]) * target
    want_mib = -(-want // _MIB)
    return int(min(cap, max(1, want_mib)))


def render_bench_line(
    shape: Shape, config: Dict[str, Any], knobs: Optional[Dict[str, Any]] = None
) -> str:
    """One hipblaslt-bench yaml line for `shape`, with dtype / layout from
    `config.hipblaslt` and `knobs` merged over DEFAULT_LINE_KNOBS.

    Precedence: `bench.skip_slow_solution_ratio` from the config overrides
    `knobs`, and at 0 the line carries no ratio (the bench's default: no
    screen); `rotating` is sized per shape unless `knobs` sets it; iteration
    counts in `knobs` are dropped, and `adaptive: true` with every adaptive
    field for the shape (`adaptive_settings`) is added. Strides are emitted
    as 0 so the bench derives them. scaleA / scaleB come from
    `config.hipblaslt` or default by dtype (1 for f8, else 0)."""
    hl = config["hipblaslt"]
    transA, transB = hl["transA"], hl["transB"]
    m, n, k, b = shape.m, shape.n, shape.k, shape.batch
    lda, ldb, ldc, ldd = strides_for(transA, transB, m, n, k)
    scale_a, scale_b = _scale_modes(hl)
    k_dict = dict(DEFAULT_LINE_KNOBS)
    if knobs:
        k_dict.update(knobs)
    # The screen drops a candidate when warm_up * ratio > best warm-up so far;
    # 0 disables it and every candidate gets fully timed. A config snapshot
    # states 0 where its source left the ratio out, and both write the same
    # line.
    ssr = (config.get("bench") or {}).get("skip_slow_solution_ratio")
    if ssr is not None and float(ssr) > 0:
        k_dict["skip_slow_solution_ratio"] = ssr
    elif ssr is not None:
        k_dict.pop("skip_slow_solution_ratio", None)
    if not (knobs and "rotating" in knobs):
        k_dict["rotating"] = rotating_mb_for(shape, config)
    for key in ITER_KEYS:
        k_dict.pop(key, None)
    k_dict["adaptive"] = "true"
    k_dict.update(adaptive_settings(config, shape.as_tuple()))
    kv = ", ".join(f"{key}: {v}" for key, v in k_dict.items())
    return (
        f"- {{function: matmul, transA: {transA}, transB: {transB}, "
        f"a_type: {hl['a_type']}, b_type: {hl['b_type']}, "
        f"c_type: {hl['c_type']}, d_type: {hl['d_type']}, "
        f"scale_type: {hl['scale_type']}, compute_type: {hl['compute_type']}, "
        f"scaleA: {scale_a}, scaleB: {scale_b}, "
        f"M: {m}, N: {n}, K: {k}, "
        f"ldc: {ldc}, ldd: {ldd}, lda: {lda}, ldb: {ldb}, "
        f"stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, "
        f"batch_count: {b}, {kv}}}"
    )


def write_bench_yaml(
    shapes: Iterable[Shape],
    config: Dict[str, Any],
    out_path: Path,
    knobs: Optional[Dict[str, Any]] = None,
) -> int:
    """Write one bench line per shape (atomically); returns the line count."""
    lines = [render_bench_line(s, config, knobs) for s in shapes]
    atomic_write_text(Path(out_path), "".join(line + "\n" for line in lines))
    return len(lines)


# The columns of a bench row that name the problem type, and the compute
# types hipblaslt-bench prints other than as its yaml name without `c_`
# (hipblas_computetype_to_string).
TYPE_COLUMNS = ("a_type", "b_type", "c_type", "d_type", "compute_type")
SCALE_COLUMNS = ("scaleA", "scaleB")
_PRINTED_COMPUTE = {"c_f32_fast_f16_r": "f32_f16_r", "c_f32_fast_bf16_r": "f32_bf16_r"}
_UNPRINTED_COMPUTE = ("pedantic",)
# hipblaslt_scaling_format names, for a bench that prints them.
SCALING_FORMATS = {
    "none": 0,
    "scalar": 1,
    "vector": 2,
    "block_32_ue8m0": 3,
    "block_16_ue8m0": 4,
    "block_32_ue4m3": 5,
    "block_16_ue4m3": 6,
    "block_32_ue5m3": 7,
    "block_16_ue5m3": 8,
}


def printed_types(hl: Mapping[str, Any]) -> Dict[str, Any]:
    """What a bench row prints in TYPE_COLUMNS and SCALE_COLUMNS for a line
    of the config's `hipblaslt` section (`render_bench_line`): the data
    types, the compute type as the bench names it (left out when it prints
    none), and the scaleA / scaleB modes as integers."""
    out: Dict[str, Any] = {k: str(hl[k]) for k in TYPE_COLUMNS[:4]}
    compute = str(hl["compute_type"])
    if not any(word in compute for word in _UNPRINTED_COMPUTE):
        out["compute_type"] = _PRINTED_COMPUTE.get(
            compute, compute[2:] if compute.startswith("c_") else compute
        )
    out["scaleA"], out["scaleB"] = _scale_modes(dict(hl))
    return out


def scale_mode_code(text: Any) -> Optional[int]:
    """The scaling-format mode a bench row prints (a number or a name), or
    None when it is neither."""
    s = str(text).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return SCALING_FORMATS.get(s.lower())


# ── reading ──────────────────────────────────────────────────────────────────

_KEY_RE = {
    key: re.compile(rf"(?<![\w]){key}:\s*(\d+)")
    for key in ("M", "N", "K", "batch_count")
}


def read_bench_lines(path: Path) -> List[str]:
    """Every `- {...}` entry line of a bench yaml, newline-terminated, in
    file order (which is the bench's execution order)."""
    out: List[str] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip().startswith("-"):
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def shape_of_line(line: str) -> Optional[Tuple[int, int, int, int]]:
    """(M, N, K, batch_count) of one bench line; batch_count defaults to 1 as
    in the bench. None when M, N or K is missing."""
    vals = {}
    for key, pat in _KEY_RE.items():
        m = pat.search(line)
        if m:
            vals[key] = int(m.group(1))
    if not all(k in vals for k in ("M", "N", "K")):
        return None
    return (vals["M"], vals["N"], vals["K"], vals.get("batch_count", 1))


# ── knob rewriting ───────────────────────────────────────────────────────────
#
# `solution_index_list` values are flow sequences (`[1, 2, 3]`), the only
# values holding commas; the patterns below match them whole.

_LIST_VALUE = r"\[[^\]]*\]"
_SCALAR_VALUE = r"[^,}\[]*"


def _knob_re(key: str) -> re.Pattern:
    return re.compile(
        rf"(?<![\w]){re.escape(key)}:\s*(?:{_LIST_VALUE}|{_SCALAR_VALUE})"
    )


def set_line_knob(line: str, key: str, value: Any) -> str:
    """`line` with `key: value` replaced, or appended when absent."""
    nl = "\n" if line.endswith("\n") else ""
    body = line.rstrip("\n")
    pat = _knob_re(key)
    if pat.search(body):
        body = pat.sub(lambda _: f"{key}: {value}", body, count=1)
    else:
        body = body.rstrip().rstrip("}") + f", {key}: {value}}}"
    return body + nl


def line_knob(line: str, key: str) -> Optional[str]:
    """The value of `key` in `line` as written, None when absent."""
    m = _knob_re(key).search(line)
    return m.group(0).split(":", 1)[1].strip() if m else None


def drop_line_knob(line: str, key: str) -> str:
    """`line` without `key` (unchanged when absent)."""
    pat = re.compile(
        rf",\s*(?<![\w]){re.escape(key)}:\s*(?:{_LIST_VALUE}|{_SCALAR_VALUE})"
    )
    return pat.sub("", line, count=1)


def format_solution_list(ids: Iterable[int]) -> str:
    return "[" + ", ".join(str(int(i)) for i in ids) + "]"


def line_solution_list(line: str) -> Optional[List[int]]:
    """The `solution_index_list` of a bench line, or None without one."""
    m = re.search(rf"(?<![\w])solution_index_list:\s*({_LIST_VALUE})", line)
    if m is None:
        return None
    inner = m.group(1)[1:-1].strip()
    return [int(t) for t in inner.split(",") if t.strip()] if inner else []


def line_has_prescreen(line: str) -> bool:
    """The line asks the bench to warm up every candidate before timing any
    (`skip_screen_prepass`)."""
    m = re.search(r"(?<![\w])skip_screen_prepass:\s*(\w+)", line)
    return m is not None and m.group(1).lower() in ("true", "1")


def parse_shapes_yaml(path: Path) -> List[Tuple[int, int, int, int]]:
    """(M, N, K, batch) of every entry of a bench yaml, in file order."""
    out: List[Tuple[int, int, int, int]] = []
    for line in read_bench_lines(path):
        shape = shape_of_line(line)
        if shape is not None:
            out.append(shape)
    return out


# ── entry names ──────────────────────────────────────────────────────────────
#
# hipblaslt-bench converts its --yaml with clients/hipblaslt_gentest.py, which
# writes an entry only when its arguments differ from every earlier entry's:
# of entries equal in every argument, the bench runs the first alone. `name`
# is an argument the bench does not read (at most 63 bytes), so entries named
# apart all run.
ENTRY_NAME_KEY = "name"


def name_entry(line: str, name: str) -> str:
    """`line`, newline-terminated, with the entry name `name`."""
    return set_line_knob(line.rstrip("\n"), ENTRY_NAME_KEY, name) + "\n"


# ── canary ───────────────────────────────────────────────────────────────────
#
# `bench.canary`: whether stage03 times the canary around the shapes of
# every bench process, how far a reading may exceed its GPU's reference
# before the block is benched again, the canary problems per reading, and
# how long a block whose canary drifted waits before it is benched again.
CANARY_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "tolerance": 0.10,
    "timings": 3,
    "backoff_s": 120.0,
}

# hipblaslt-bench refuses `algo_method: 1` for scale modes from this one up,
# the block-scaled (MX) ones.
_BLOCK_SCALING_MODE = 3
_SCALE_RE = re.compile(r"(?<![\w])scale[AB]:\s*(\d+)")


def least_work_line(lines: Iterable[str]) -> Optional[str]:
    """The line of least work (M * N * K * batch; the first of equal ones),
    None when no line names its shape."""
    best: Optional[Tuple[int, str]] = None
    for line in lines:
        shape = shape_of_line(line)
        if shape is None:
            continue
        work = shape[0] * shape[1] * shape[2] * shape[3]
        if best is None or work < best[0]:
            best = (work, line)
    return None if best is None else best[1]


def canary_needs_kernel(line: str) -> bool:
    """A block-scaled line that lists no kernel: its canary cannot take the
    library's first kernel (`algo_method: 1`) and needs one named."""
    return not line_solution_list(line) and any(
        int(m) >= _BLOCK_SCALING_MODE for m in _SCALE_RE.findall(line)
    )


def canary_line(line: str, kernel: Optional[int] = None) -> Optional[str]:
    """`line` restamped as a canary: it times one kernel alone, without a
    screen, and prints the kernel's name. The kernel is the first the line
    lists; on a line without a list, the solution index `kernel` when given,
    else the first the library holds for the problem (`algo_method: 1` with
    `requested_solution_num: 1`); no ranking picks either. None for a line
    that `canary_needs_kernel` when `kernel` is None."""
    listed = line_solution_list(line)
    out = drop_line_knob(line, "skip_screen_prepass")
    out = set_line_knob(out, "skip_slow_solution_ratio", 0)
    out = set_line_knob(out, "print_kernel_info", 1)
    if listed or kernel is not None:
        first = listed[0] if listed else int(kernel)
        out = drop_line_knob(out, "algo_method")
        return set_line_knob(out, "solution_index_list", format_solution_list([first]))
    if canary_needs_kernel(line):
        return None
    out = set_line_knob(out, "requested_solution_num", 1)
    return set_line_knob(out, "algo_method", 1)
