# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Input schema and expansion of the skinny GEMM sweep (sweep_skinny_gemm.py).

Pure Python plus NumPy: nothing here needs the bindings or a GPU, so the input
can be parsed, validated and counted anywhere.

An input file has three top-level keys, all but `problems` optional:

  settings:  iters, warmup, refine_top, refine_iters, top, verify_full_max_elems,
             seed.
  defaults:  useBeta, betaModes, alpha, beta, spk, transpose and sweep, applied
             to every problem entry and overridden by the entry's own keys.
  problems:  a list of entries with name, m, n, k and the optional keys
             transpose, spk, ldA, ldB, ldD, ldC (BYTES, the kernel ABI), useBeta,
             betaModes, alpha, beta and sweep.

m, n and k are an int or a list of ints: the entry covers their cartesian
product. With transpose every (n, m, k) swapped shape is covered too. spk is an
int, "k" (one split, the default) or a list of those; useBeta is a bool or a
list of bools. The explicit ld values apply to every shape of the entry; the
defaults are those of skinny_gemm_support.makeProblem without padding.

A `sweep` is a mapping (or a list of mappings, whose configurations are
unioned) from the compile-time knobs of SkinnyGEMMConfig to a scalar, a list or a
range mapping {min, max, step} or {min, max, mul} (inclusive, arithmetic or
geometric): bm, tn, lanesK, waves, prefetch, upScales, grid3d, group, numXcds,
blocks, mfma and lds. A knob that is not swept takes the default of DEFAULT_SWEEP,
which is the one of the SkinnyGEMMConfig struct. When an entry's sweep and the
default sweep are merged, the entry's value of a knob replaces the default's; a
list of sweeps on either side is combined pairwise.

upScales is the support module's up_scales: 0 rings the scales (UP_ITERS 0), 1
loads the scale vector up front, with UP_ITERS derived from the split extents.
partialSteps and upIters are never swept: they are derived from the shape and
spk exactly as skinny_gemm_support.Problem derives them. grid3d true
configurations are launched on the 3-D grid and force group = numXcds = 1,
which is what support.useGrid3d requires (the kernel ignores them there).
grid3d false configurations are persistent and sweep group, numXcds and blocks;
every block count is capped to the tile count and duplicates are dropped.

mfma is a swept boolean knob (default false) that selects the scaled-MFMA kernel.
With it bm is 16 or 32, tn counts the 16-column MFMA tiles of a wave, waves split
K and prefetch is the ring depth in 256-K iterations; lanesK must be 64 and
upScales 0, and any k is supported. The bm values of an mfma configuration are
those of the sweep that are 16 or 32, or both when the sweep has none (the
default bm of 1 or a VALU list such as [1, 2, 4, 8] does not apply to mfma), so
a sweep over {mfma: [false, true]} can share its bm list with the VALU
configurations; other invalid combinations are counted as constraint failures.

lds is a swept boolean knob (default false) that, with mfma, stages the operands
through a per-wave LDS ring (prefetch is then the ring depth in LDS). lds
without mfma is a constraint failure, and so is an mfma configuration whose
LDS ring or registers exceed the budget: the emitter rejects it when the kernel
is built, which is recorded as a failure to build like the other register-budget
rejections.

useBeta defaults to [false, true]: every problem is swept with the kernel that
never reads C and with the one that adds beta * C. alpha is a number (pinned),
"random" or {random: {min, max, signed}}; the default is random in [0.25, 2].
beta is tested in several modes, each its own case with its own ranking and
verification and the same kernels, which are built once: betaModes (default
[zero, random]) lists them. zero is beta 0.0, the exactly-zero fast path of the
useBeta kernel, with C poisoned as in the support module; random is a nonzero
draw in [-2, -0.25] and [0.25, 2] (or the range of {random: {min, max, signed}}
given as beta). The modes apply when beta matters: with useBeta, or with several
splits (the split-K reduce applies beta * C whatever useBeta is); otherwise the
problem runs once with betaMode none. A beta number pins a single case (betaMode
zero for 0.0, else pinned). A draw is rounded to 3 decimals, redrawn when within
0.05 of 1, and made by a generator seeded with settings.seed (CLI --seed), the
name, m, n, k and useBeta of the problem (alpha is shared by the modes), so it is
reproducible and independent of the problem order and of the GPU assignment. The
resolved alpha, beta and betaMode are in the problem record of the results and of
the failed configurations.

Settings: iters and warmup are the timed and untimed launches of the screening
pass, refine_top and refine_iters the configurations re-timed and their
launches, top the winners reported per problem, seed the seed of the alpha and
beta draws. Every winner is verified against the host reference, whatever the
size: with (m + n) * k up to verify_full_max_elems (default 2^31, about 1 s of
host time per 10^9 elements) the reference is dense and the whole result is
compared ("full"), above it the reference is computed for a sample of the output
elements ("sampled(N)"), failed (verify mismatch), or no (--no-verify). In refine+top
mode only the winners are verified, not the other screened candidates. In one-shot
mode every screened configuration is verified once.
"""

import itertools
import zlib
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import yaml

import skinny_gemm_support as support
from skinny_gemm_support import Problem

SETTING_DEFAULTS = {
    "iters": 20,
    "warmup": 3,
    "refine_top": 10,
    "refine_iters": 100,
    "top": 5,
    "verify_full_max_elems": 2**31,
    "seed": 20240601,
}
# Settings that may be zero; the others must be positive.
ZERO_SETTINGS = ("warmup", "seed")
# Ignored when the sweep runs with --one-shot; may be zero in the input YAML then.
ONE_SHOT_ZERO_SETTINGS = ("refine_top", "refine_iters", "top")
DEFAULT_KEYS = (
    "useBeta",
    "betaModes",
    "alpha",
    "beta",
    "spk",
    "transpose",
    "sweep",
)
PROBLEM_KEYS = DEFAULT_KEYS + ("name", "m", "n", "k", "ldA", "ldB", "ldD", "ldC")
TOP_KEYS = ("settings", "defaults", "problems")
SWEEP_KEYS = (
    "bm",
    "tn",
    "lanesK",
    "waves",
    "prefetch",
    "upScales",
    "grid3d",
    "group",
    "numXcds",
    "blocks",
    "mfma",
    "lds",
)
# Knobs that take booleans.
BOOL_KEYS = ("grid3d", "mfma", "lds")
# The skinny block sizes of the mfma kernel.
MFMA_BMS = (16, 32)
# The defaults of the SkinnyGEMMConfig struct, and a block count that covers the
# 256 CUs of an MI355X.
DEFAULT_SWEEP = {
    "bm": [1],
    "tn": [1],
    "lanesK": [64],
    "waves": [4],
    "prefetch": [2],
    "upScales": [0],
    "grid3d": [True],
    "group": [1],
    "numXcds": [1],
    "blocks": [256],
    "mfma": [False],
    "lds": [False],
}
RANGE_KEYS = ("min", "max", "step", "mul")


class SweepConfigError(ValueError):
    """The input file is malformed."""


@dataclass(frozen=True)
class ScalarSpec:
    """How alpha or beta of a problem is chosen: pinned to a value, or drawn
    from a seeded generator with a magnitude in [low, high] and, if signed, a
    random sign."""

    value: float | None = None
    low: float = 0.0
    high: float = 0.0
    signed: bool = False

    @property
    def pinned(self) -> bool:
        return self.value is not None


BETA_MODES = ("zero", "random")
# alpha in [0.25, 2]; beta in [-2, -0.25] and [0.25, 2]: never 0, so neither
# the zero-beta path of the kernels nor a vanishing product is hit by accident.
RANDOM_ALPHA = ScalarSpec(None, 0.25, 2.0, False)
RANDOM_BETA = ScalarSpec(None, 0.25, 2.0, True)
# Draws closer to 1 than this are redrawn: scaling by about 1 hides a missing
# alpha or beta in the verification.
TRIVIAL_MARGIN = 0.05
MAX_REDRAWS = 100
DRAW_DECIMALS = 3


def check_keys(where: str, mapping: dict, allowed: tuple) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise SweepConfigError(
            f"{where}: unknown key(s) {unknown}, allowed: {sorted(allowed)}"
        )


def as_mapping(where: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise SweepConfigError(f"{where}: expected a mapping, got {value!r}")
    return value


def as_int(where: str, value: Any, minimum: int | None = None) -> int:
    # bool is an int subclass, so it is rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SweepConfigError(f"{where}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise SweepConfigError(f"{where}: must be at least {minimum}, got {value}")
    return value


def as_float(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def as_bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise SweepConfigError(f"{where}: expected true or false, got {value!r}")
    return value


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def unique(values: list) -> list:
    return list(dict.fromkeys(values))


def expand_range(where: str, spec: dict) -> list[int]:
    check_keys(where, spec, RANGE_KEYS)
    if "min" not in spec or "max" not in spec:
        raise SweepConfigError(f"{where}: a range needs min and max")
    if ("step" in spec) == ("mul" in spec):
        raise SweepConfigError(f"{where}: a range needs exactly one of step and mul")
    low = as_int(f"{where}.min", spec["min"])
    high = as_int(f"{where}.max", spec["max"])
    if "step" in spec:
        return list(range(low, high + 1, as_int(f"{where}.step", spec["step"], 1)))
    factor = as_int(f"{where}.mul", spec["mul"], 2)
    if low < 1:
        raise SweepConfigError(f"{where}: a geometric range needs min >= 1")
    values = []
    while low <= high:
        values.append(low)
        low *= factor
    return values


def expand_values(where: str, key: str, raw: Any) -> list:
    """The values of one swept knob: a scalar, a list or a range."""
    if isinstance(raw, dict):
        if key in BOOL_KEYS:
            raise SweepConfigError(f"{where}: {key} takes booleans, not a range")
        values = expand_range(where, raw)
    elif key in BOOL_KEYS:
        values = [as_bool(where, value) for value in as_list(raw)]
    else:
        values = [as_int(where, value) for value in as_list(raw)]
    if not values:
        raise SweepConfigError(f"{where}: no values")
    return unique(values)


def sweep_mappings(where: str, raw: Any) -> list[dict]:
    """A sweep as a list of mappings; no sweep is one empty mapping."""
    if raw is None:
        return [{}]
    mappings = raw if isinstance(raw, list) else [raw]
    if not mappings:
        raise SweepConfigError(f"{where}: empty list of sweeps")
    for index, mapping in enumerate(mappings):
        as_mapping(f"{where}[{index}]", mapping)
        check_keys(f"{where}[{index}]", mapping, SWEEP_KEYS)
    return mappings


def merge_sweeps(where: str, defaults: Any, entry: Any) -> list[dict[str, list]]:
    """The expanded sweeps of an entry: the entry's knobs over the defaults',
    every default sweep combined with every entry sweep."""
    merged = []
    for base in sweep_mappings("defaults.sweep", defaults):
        for own in sweep_mappings(f"{where}.sweep", entry):
            knobs = {**base, **own}
            merged.append(
                {
                    key: expand_values(f"{where}.sweep.{key}", key, raw)
                    for key, raw in knobs.items()
                }
            )
    return merged


class Case(NamedTuple):
    """One problem of the sweep: a shape, split size and beta mode with its
    knob sweeps. The ld values are in bytes."""

    name: str
    m: int
    n: int
    k: int
    spk: int
    ld_a: int
    ld_b: int
    ld_d: int
    ld_c: int
    use_beta: bool
    alpha: float
    beta: float
    beta_mode: str
    sweeps: tuple

    @property
    def label(self) -> str:
        return (
            f"{self.name} m={self.m} n={self.n} k={self.k} spk={self.spk} "
            f"useBeta={int(self.use_beta)} beta={self.beta_mode}"
        )

    def record(self) -> dict:
        return {
            "name": self.name,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "spk": self.spk,
            "ldA": self.ld_a,
            "ldB": self.ld_b,
            "ldD": self.ld_d,
            "ldC": self.ld_c,
            "useBeta": self.use_beta,
            "alpha": self.alpha,
            "beta": self.beta,
            "betaMode": self.beta_mode,
        }


@dataclass
class SweepInput:
    settings: dict
    cases: list[Case]


def parse_settings(raw: Any, *, one_shot: bool = False) -> dict:
    settings = dict(SETTING_DEFAULTS)
    if raw is None:
        return settings
    as_mapping("settings", raw)
    check_keys("settings", raw, tuple(SETTING_DEFAULTS))
    for key, value in raw.items():
        if key in ZERO_SETTINGS or (one_shot and key in ONE_SHOT_ZERO_SETTINGS):
            minimum = 0
        else:
            minimum = 1
        settings[key] = as_int(f"settings.{key}", value, minimum)
    return settings


def parse_defaults(raw: Any) -> dict:
    if raw is None:
        return {}
    as_mapping("defaults", raw)
    check_keys("defaults", raw, DEFAULT_KEYS)
    return raw


def int_list(where: str, raw: Any) -> list[int]:
    values = [as_int(where, value, 1) for value in as_list(raw)]
    if not values:
        raise SweepConfigError(f"{where}: no values")
    return unique(values)


def spk_list(where: str, raw: Any) -> list[int | str]:
    values = [
        "k" if value == "k" else as_int(where, value, 1) for value in as_list(raw)
    ]
    if not values:
        raise SweepConfigError(f"{where}: no values")
    return unique(values)


def shapes_of(where: str, entry: dict) -> list[tuple[int, int, int]]:
    ms, ns, ks = (int_list(f"{where}.{key}", entry[key]) for key in ("m", "n", "k"))
    shapes = list(itertools.product(ms, ns, ks))
    if as_bool(f"{where}.transpose", entry.get("transpose", False)):
        shapes += [(n, m, k) for m, n, k in shapes]
    return unique(shapes)


def check_ld(where: str, value: Any, multiple: int) -> int | None:
    if value is None:
        return None
    value = as_int(where, value, 1)
    if value % multiple:
        raise SweepConfigError(f"{where}: must be a multiple of {multiple} bytes")
    return value


def parse_scalar(where: str, raw: Any, random_spec: ScalarSpec) -> ScalarSpec | None:
    """alpha or beta of an entry: a number (pinned), "random" (the default
    range random_spec) or {random: {min, max, signed}}; None when not given."""
    if raw is None:
        return None
    if raw == "random":
        return random_spec
    if isinstance(raw, dict):
        check_keys(where, raw, ("random",))
        body = as_mapping(f"{where}.random", raw.get("random"))
        check_keys(f"{where}.random", body, ("min", "max", "signed"))
        low = as_float(f"{where}.random.min", body.get("min", random_spec.low))
        high = as_float(f"{where}.random.max", body.get("max", random_spec.high))
        signed = as_bool(
            f"{where}.random.signed", body.get("signed", random_spec.signed)
        )
        if not 0 < low <= high:
            raise SweepConfigError(f"{where}.random: needs 0 < min <= max")
        return ScalarSpec(None, low, high, signed)
    return ScalarSpec(as_float(where, raw))


def draw_scalar(spec: ScalarSpec, seed: int, case_key: tuple, stream: int) -> float:
    """The value of a spec: the pinned one, or a draw that depends only on the
    seed and on the problem (name, m, n, k, useBeta), not on the order of the
    problems or on the GPU that times them."""
    if spec.pinned:
        return spec.value
    name, *numbers = case_key
    entropy = [seed, zlib.crc32(name.encode()), *numbers, stream]
    rng = np.random.default_rng(entropy)
    for _ in range(MAX_REDRAWS):
        magnitude = round(float(rng.uniform(spec.low, spec.high)), DRAW_DECIMALS)
        negative = spec.signed and rng.random() < 0.5
        if abs(magnitude - 1.0) >= TRIVIAL_MARGIN:
            break
    return -magnitude if negative else magnitude


def beta_modes_of(where: str, raw: Any) -> list[str]:
    modes = as_list(raw)
    for mode in modes:
        if mode not in BETA_MODES:
            raise SweepConfigError(
                f"{where}: expected one of {BETA_MODES}, got {mode!r}"
            )
    if not modes:
        raise SweepConfigError(f"{where}: no modes")
    return unique(modes)


def beta_variants(
    spec: ScalarSpec | None, modes: list[str], relevant: bool, seed: int, key: tuple
) -> list[tuple[str, float]]:
    """The (mode, beta) pairs of a problem: the pinned beta alone; else, when
    the beta matters, one per mode, zero (the kernel's zero fast path, C
    uninitialized) and random; else one without a beta."""
    if spec is not None and spec.pinned:
        return [("zero" if spec.value == 0 else "pinned", spec.value)]
    if not relevant:
        return [("none", 0.0)]
    random_spec = spec or RANDOM_BETA
    draw = draw_scalar(random_spec, seed, key, 1)
    return [("zero", 0.0) if mode == "zero" else ("random", draw) for mode in modes]


def parse_entry(index: int, entry: Any, defaults: dict, seed: int) -> list[Case]:
    where = f"problems[{index}]"
    as_mapping(where, entry)
    check_keys(where, entry, PROBLEM_KEYS)
    for key in ("name", "m", "n", "k"):
        if key not in entry:
            raise SweepConfigError(f"{where}: missing key {key}")
    if not isinstance(entry["name"], str):
        raise SweepConfigError(f"{where}.name: expected a string")
    merged = {**defaults, **entry}
    sweeps = tuple(merge_sweeps(where, defaults.get("sweep"), entry.get("sweep")))
    lds = {
        "ld_a": check_ld(f"{where}.ldA", entry.get("ldA"), 4),
        "ld_b": check_ld(f"{where}.ldB", entry.get("ldB"), 4),
        "ld_d": check_ld(f"{where}.ldD", entry.get("ldD"), 2),
        "ld_c": check_ld(f"{where}.ldC", entry.get("ldC"), 2),
    }
    spks = spk_list(f"{where}.spk", merged.get("spk", "k"))
    use_betas = [
        as_bool(f"{where}.useBeta", value)
        for value in as_list(merged.get("useBeta", [False, True]))
    ]
    modes = beta_modes_of(
        f"{where}.betaModes", merged.get("betaModes", list(BETA_MODES))
    )
    alpha_spec = parse_scalar(f"{where}.alpha", merged.get("alpha"), RANDOM_ALPHA)
    beta_spec = parse_scalar(f"{where}.beta", merged.get("beta"), RANDOM_BETA)
    cases = []
    for m, n, k in shapes_of(where, merged):
        for spk, use_beta in itertools.product(spks, unique(use_betas)):
            spk = k if spk == "k" else spk
            key = (entry["name"], m, n, k, int(use_beta))
            alpha = draw_scalar(alpha_spec or RANDOM_ALPHA, seed, key, 0)
            # The kernel reads beta only with useBeta, and the split-K reduce
            # reads it whatever useBeta is.
            relevant = use_beta or spk < k
            for mode, beta in beta_variants(beta_spec, modes, relevant, seed, key):
                cases.append(
                    Case(
                        entry["name"],
                        m,
                        n,
                        k,
                        spk,
                        lds["ld_a"] or support.alignUp(k, 4),
                        lds["ld_b"] or support.alignUp(k, 4),
                        lds["ld_d"] or 2 * m,
                        lds["ld_c"] or 2 * m,
                        use_beta,
                        alpha,
                        beta,
                        mode,
                        sweeps,
                    )
                )
    return cases


def parse_input(
    data: Any, seed: int | None = None, *, one_shot: bool = False
) -> SweepInput:
    as_mapping("input", data)
    check_keys("input", data, TOP_KEYS)
    if "problems" not in data:
        raise SweepConfigError("input: missing key problems")
    entries = data["problems"]
    if not isinstance(entries, list) or not entries:
        raise SweepConfigError("problems: expected a non-empty list")
    defaults = parse_defaults(data.get("defaults"))
    settings = parse_settings(data.get("settings"), one_shot=one_shot)
    if seed is not None:
        settings["seed"] = as_int("seed", seed, 0)
    cases = []
    for index, entry in enumerate(entries):
        cases += parse_entry(index, entry, defaults, settings["seed"])
    if not one_shot and settings["refine_top"] < settings["top"]:
        raise SweepConfigError("settings: refine_top must be at least top")
    return SweepInput(settings, cases)


def load_input(
    path: str, seed: int | None = None, *, one_shot: bool = False
) -> SweepInput:
    with open(path) as stream:
        return parse_input(yaml.safe_load(stream), seed, one_shot=one_shot)


class RawConfig(NamedTuple):
    """The compile-time knobs of one configuration before the support module
    derives the rest; blocks is None for the 3-D grid."""

    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_scales: int
    grid3d: bool
    group: int
    xcds: int
    blocks: int | None
    mfma: bool = False
    lds: bool = False


def capped_blocks(case: Case, knobs: tuple, blocks: list[int]) -> list[int]:
    """The block counts capped to the tile count of the configuration, without
    duplicates: extra blocks would find no tile."""
    bm, tn, lanes_k, waves, mfma = knobs
    bn = support.blockCols(waves, lanes_k, tn, mfma)
    tiles = support.countTiles(case.m, case.n, case.k, case.spk, bm, bn)
    return unique([min(count, tiles) for count in blocks])


def mfma_bms(bms: list[int]) -> list[int]:
    """The bm values of the mfma configurations of a sweep over bms: its 16 and
    32, else both."""
    return [bm for bm in bms if bm in MFMA_BMS] or list(MFMA_BMS)


def raw_configs(case: Case, sweep: dict[str, list]):
    """The configurations of a sweep: the 3-D grid ones with the plain tile
    order, the persistent ones over group, numXcds and capped blocks."""
    values = {key: sweep.get(key, DEFAULT_SWEEP[key]) for key in SWEEP_KEYS}
    shape = itertools.product(
        values["mfma"],
        values["lds"],
        values["tn"],
        values["lanesK"],
        values["waves"],
        values["prefetch"],
        values["upScales"],
        values["grid3d"],
    )
    for mfma, lds, tn, lanes_k, waves, pf, up_scales, grid3d in shape:
        for bm in mfma_bms(values["bm"]) if mfma else values["bm"]:
            head = (bm, tn, lanes_k, waves, pf, up_scales)
            if grid3d:
                yield RawConfig(*head, True, 1, 1, None, mfma=mfma, lds=lds)
                continue
            blocks = capped_blocks(
                case, (bm, tn, lanes_k, waves, mfma), values["blocks"]
            )
            for group, xcds, count in itertools.product(
                values["group"], values["numXcds"], blocks
            ):
                yield RawConfig(*head, False, group, xcds, count, mfma=mfma, lds=lds)


def raw_record(case: Case, raw: RawConfig) -> dict:
    return {
        "bm": raw.bm,
        "tn": raw.tn,
        "lanesK": raw.lanes_k,
        "waves": raw.waves,
        "prefetch": raw.pf,
        "upScales": raw.up_scales,
        "grid3d": raw.grid3d,
        "group": raw.group,
        "numXcds": raw.xcds,
        "spk": case.spk,
        "blocks": raw.blocks,
        "mfma": raw.mfma,
        "lds": raw.lds,
    }


def config_record(p: Problem) -> dict:
    """The full knob set of a problem, with the derived upIters and
    partialSteps, plus its spk and block count."""
    return {
        "bm": p.bm,
        "tn": p.tn,
        "lanesK": p.lanes_k,
        "waves": p.waves,
        "prefetch": p.pf,
        "upIters": p.up_iters,
        "partialSteps": p.partial_steps,
        "group": p.group,
        "numXcds": p.xcds,
        "grid3d": bool(p.grid3d),
        "useBeta": bool(p.use_beta),
        "spk": p.spk,
        "blocks": p.blocks,
        "mfma": bool(p.mfma),
        "lds": bool(p.lds),
    }


def build_problem(case: Case, raw: RawConfig) -> Problem:
    """The support Problem of a configuration, every knob explicit. The ld
    values of Problem are in elements and its padding is in bytes of A and B."""
    padding = (
        case.ld_a - support.alignUp(case.k, 4),
        case.ld_b - support.alignUp(case.k, 4),
        0,
    )
    return support.makeProblem(
        case.m,
        case.n,
        case.k,
        spk=case.spk,
        bm=raw.bm,
        tn=raw.tn,
        lanes_k=raw.lanes_k,
        waves=raw.waves,
        pf=raw.pf,
        group=raw.group,
        xcds=raw.xcds,
        up_scales=raw.up_scales,
        blocks=raw.blocks,
        padding=padding,
        ldD=case.ld_d // 2,
        ldC=case.ld_c // 2,
        alpha=case.alpha,
        beta=case.beta,
        use_beta=case.use_beta,
        mfma=raw.mfma,
        lds=raw.lds,
    )


def problem_key(p: Problem) -> tuple:
    """Configurations with the same kernel, block count and split size are the
    same launch."""
    return (support.gemmKernelName(p), p.blocks, p.spk)


@dataclass
class ConfigFailure:
    stage: str
    reason: str
    config: dict


@dataclass
class ExpandedCase:
    case: Case
    problems: list[Problem] = field(default_factory=list)
    failures: list[ConfigFailure] = field(default_factory=list)
    swept: int = 0


def first_line(error: BaseException) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else type(error).__name__


def expand_case(case: Case) -> ExpandedCase:
    """The configurations of a case: unique ones that satisfy the support
    constraints as problems, the others as constraint failures."""
    expanded = ExpandedCase(case)
    seen_raw = set()
    seen_problems = set()
    for sweep in case.sweeps:
        for raw in raw_configs(case, sweep):
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            expanded.swept += 1
            try:
                p = build_problem(case, raw)
            except (AssertionError, ValueError) as err:
                config = raw_record(case, raw)
                expanded.failures.append(
                    ConfigFailure("constraint", first_line(err), config)
                )
                continue
            key = problem_key(p)
            if key in seen_problems:
                expanded.swept -= 1
                continue
            seen_problems.add(key)
            expanded.problems.append(p)
    return expanded
