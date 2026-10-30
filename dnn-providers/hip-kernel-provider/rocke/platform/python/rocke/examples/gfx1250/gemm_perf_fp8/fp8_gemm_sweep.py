#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Staged fp8e4m3 -> bf16 RCR Universal GEMM sweep for gfx1250.

Ported from the bf16 sweep harness, with the operand type, the correctness
gate and three structural fixes changed.  The staging is the point: a full
cross product of tile geometry x traits is tens of thousands of builds, and
most of them are answering a question the cheap stage already answered.

Stages
------
``tile``
    Sweep tile / warp geometry only, with every trait pinned to the config's
    ``screening_traits`` block.  Verify the top ``tile_finalists`` and pass
    only the verified ones forward.

``trait``
    Sweep the full trait space over those finalists.  No verification here --
    this is the widest stage and verification costs a blocking launch plus a
    device-to-host copy per candidate.  Correctness is re-established in
    ``final`` before any number is reported.

``final``
    Walk the trait ranking, verify, and re-time each survivor in a *fresh
    process* until ``final_timed`` have passed.  Fresh-process timing is what
    makes the number trustworthy; the in-process stages are for ranking.

``lever``
    A/B the champion against each registered lever.  This is the stage that
    answers "did the thing we just added help", independent of the search.

Rigor is deliberately inverse to candidate count.  Stage 2 screens thousands
of kernels with one attempt and no reference; stage 3 pays for isolation and
repetition on a handful.  Flattening that -- running everything carefully --
costs an order of magnitude more and tells you nothing extra.

The screening pin is load-bearing
---------------------------------
``screening_traits`` decides which traits stage 1 holds fixed while it ranks
geometry.  A bad pin silently eliminates the winner before traits are ever
swept: if a pinned value happens to collapse performance for one region of
the tile space, every tile in that region is screened out in stage 1 and
never reaches stage 2.  The bf16 harness pinned ``traits[...][0]`` -- the
first element of each search list -- which made list *order* a hidden tuning
parameter that no reader would guess.  Here the pin is its own config block,
so it is stated rather than implied.

A lever ratio is an attribution, not a speedup
----------------------------------------------
Every arm of the ``lever`` stage is timed on *one* geometry: the champion
``final`` chose.  That champion was selected by ``tile``/``trait`` searching
at the config's default traits, so it is the *default's* best tile, not each
arm's.  An arm whose optimum lies elsewhere in the tile space is therefore
measured somewhere it would never have chosen, and is understated -- by much
more than the effect being measured, when the two prefer different tiles.

So a lever ratio answers "holding geometry fixed, did this knob help".  That
is the right question for attributing one change, and the wrong number to
quote as a speedup.  For best-vs-best, run two full ``tile,trait,final``
sweeps and compare their winners, pinning the trait under test in
``screening_traits`` for one of them.  That pin reaches every candidate:
``enumerate_trait_configs`` rebuilds each spec with ``replace(base.trait,
...)``, overriding only the keys it iterates, so anything else set on the
base survives into stage 2.

The same asymmetry is why a directory named for an arm can mislead.  A config
whose ``screening_traits`` omits the trait under test ran its whole search at
the default; only the lever arm differed.  Check the stage's specs, not the
directory name, before reading a historical result as that arm's.

Exactness
---------
Inputs are integers in -5..5, each exact in e4m3 (four significand bits).
With ``|sum| <= 25 * K`` the fp32 accumulator is exact for any K below ~670k,
so summation order cannot change the result and the only rounding left is the
kernel's final deterministic RNE to bf16, which the reference reproduces.
The comparison is therefore exact (``err > 0.0``), not a tolerance, and the
harness refuses a K that would break the argument.  A tolerance here would
silently downgrade the gate without changing any output.

Results go outside the repo
---------------------------
``--out`` (or ``$ROCKE_PERF_OUT``) is required and is deliberately absent
from the committed configs: a tracked config carries the *search space*, not
a path to somebody's results directory.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import datetime
import itertools
import json
import math
import os
import shutil
import socket
import statistics
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rocke.helpers import compile_kernel
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.runtime.hip_module import Runtime
from rocke.sweep import BuildRecord, build_all_instances

CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_CONFIG = CONFIG_DIR / "sq4k.json"

STAGES = ("tile", "trait", "final", "lever")

# The exact-accumulate argument holds while |sum| <= 25*K stays under 2^24.
_MAX_EXACT_K = (1 << 24) // 25

_WORKER_RESULT_PREFIX = "ROCKE_SWEEP_WORKER_RESULT="

# HIP error codes that latch onto the context rather than being returned once:
# an illegal access (700), a device-side abort (710) and an unspecified launch
# failure (719) leave every later call in the process returning the same code.
# Once one is seen the remaining candidates in the stage cannot be trusted, so
# the stage stops and keeps what it already measured instead of reporting a
# cascade of identical bogus failures.
_STICKY_HIP_ERRORS = ("hipError(700)", "hipError(710)", "hipError(719)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge. Scalars and lists replace; dicts merge.

    Lists replace rather than concatenate so a derived config can *narrow* a
    sweep (the common case: pin one pipeline out of two), which appending
    could not express.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: Path = DEFAULT_CONFIG, _seen: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """Load a config, resolving an optional ``extends`` chain.

    ``"extends": "sq4k.json"`` names a parent relative to this file's own
    directory; the child is deep-merged over it. That is what lets a per-depth
    or per-pipeline variant be three lines instead of a 60-line copy, which is
    how nine untracked variants ended up outside the repo.
    """
    path = Path(path)
    seen = list(_seen or [])
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in seen + [resolved])
        raise SystemExit(f"'extends' cycle: {chain}")
    seen.append(resolved)

    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"config not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    parent_path = (path.parent / str(parent_ref)).resolve()
    return _deep_merge(load_config(parent_path, _seen=seen), data)


def _parse_set_value(text: str) -> Any:
    """Parse a ``--set`` value as JSON, falling back to a bare string.

    So ``=true``/``=2``/``=null``/``=[0,48]`` arrive typed, while
    ``=intrawave`` still works without quoting it past the shell.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_set_overrides(
    config: Dict[str, Any], assignments: Sequence[str]
) -> List[str]:
    """Apply repeatable ``--set a.b.c=value`` assignments in order."""
    applied: List[str] = []
    for raw in assignments or ():
        if "=" not in raw:
            raise SystemExit(f"--set expects key.path=value, got {raw!r}")
        dotted, _, text = raw.partition("=")
        keys = [k for k in dotted.strip().split(".") if k]
        if not keys:
            raise SystemExit(f"--set has an empty key path: {raw!r}")
        value = _parse_set_value(text)
        section = config
        for key in keys[:-1]:
            nxt = section.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                section[key] = nxt
            section = nxt
        section[keys[-1]] = value
        applied.append(f"{'.'.join(keys)}={value!r}")
    return applied


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Where results land.  Never read from the committed config."""
    raw = args.out or os.environ.get("ROCKE_PERF_OUT")
    if not raw:
        raise SystemExit(
            "no output directory: pass --out or set ROCKE_PERF_OUT. "
            "Results are kept outside the repository on purpose."
        )
    return Path(raw).expanduser()


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _make_spec(
    config: Dict[str, Any],
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    traits: Dict[str, Any],
) -> UniversalGemmSpec:
    """One spec from a tile geometry plus a fully-resolved trait dict."""
    target = config["target"]
    problem = config["problem"]
    warp_tile_m, warp_tile_n, warp_tile_k = target["warp_tile"]
    return UniversalGemmSpec(
        name=f"{target['arch']}_fp8_sweep",
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=int(warp_tile_m),
            warp_tile_n=int(warp_tile_n),
            warp_tile_k=int(warp_tile_k),
        ),
        trait=TraitSpec(
            # A guard is dead code when the tile grid lands exactly on the
            # extent, and pad_n forfeits the cshuffle wide store, so each one
            # follows the shape rather than being forced on.
            pad_m=int(problem["m"]) % tile_m != 0,
            pad_n=int(problem["n"]) % tile_n != 0,
            pad_k=int(problem["k"]) % tile_k != 0,
            **traits,
        ),
        data=DataSpec(
            dtype_a=str(target["dtype_ab"]),
            dtype_b=str(target["dtype_ab"]),
            dtype_c=str(target["dtype_c"]),
            dtype_acc="fp32",
            layout=str(target["layout"]),
        ),
        wave_size=int(target["wave_size"]),
    )


def _spec_identity(spec: UniversalGemmSpec, *, arch: str) -> str:
    import hashlib

    payload = {
        "arch": arch,
        "tile": asdict(spec.tile),
        "trait": asdict(spec.trait),
        "data": asdict(spec.data),
        "wave_size": spec.wave_size,
        "block_size": spec.block_size,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _spec_from_dict(data: Dict[str, Any]) -> UniversalGemmSpec:
    return UniversalGemmSpec(
        name=str(data["name"]),
        tile=TileSpec(**data["tile"]),
        trait=TraitSpec(**data["trait"]),
        data=DataSpec(**data["data"]),
        wave_size=int(data["wave_size"]),
        block_size=int(data["block_size"]),
    )


def _dedupe_valid(
    specs: Sequence[UniversalGemmSpec], *, arch: str
) -> List[UniversalGemmSpec]:
    seen: Dict[str, UniversalGemmSpec] = {}
    for spec in specs:
        ok = is_valid_spec(spec, arch=arch)
        # is_valid_spec returns (bool, reason) on this branch; tolerate a bare
        # bool so the harness survives a signature change rather than silently
        # admitting every spec.
        if isinstance(ok, tuple):
            ok = ok[0]
        if not ok:
            continue
        seen.setdefault(_spec_identity(spec, arch=arch), spec)
    return list(seen.values())


# Derived from the shape, so a config that sets them is a mistake rather than
# an override -- ``_make_spec`` already passes them positionally.
_SHAPE_DERIVED_TRAITS = ("pad_m", "pad_n", "pad_k")


def _trait_field_names() -> frozenset:
    """Every field ``TraitSpec`` actually has, as the validation allowlist."""
    return frozenset(TraitSpec.__dataclass_fields__)


def resolve_traits(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """Split the ``traits`` section into the stage-1 pin and the sweep space.

    A scalar value is pinned and never swept; a list is swept, and its stage-1
    value comes from the explicit ``screening`` sub-block. The pin is explicit
    rather than "the first list element" on purpose: ``lds_k_pad`` is swept
    over ``[0, 16, 48, 80, 112]`` while pinning 48, because screening at pad=0
    collapses the big-block region and silently removes it from the search.
    Making the pin positional would make list order load-bearing and invisible.

    Returns ``(pin, swept)`` where ``pin`` is a complete trait dict suitable
    for ``_make_spec`` and ``swept`` maps each swept key to its candidate list.

    Raises ``SystemExit`` on anything unusable. That is the point of this
    function: the previous shape accepted keys it then ignored -- adding
    ``tdm_depth`` to the sweep produced specs that all carried the pin's depth
    -- so a config could report a confident ranking over identical kernels.
    """
    raw = config.get("traits")
    if raw is None:
        raise SystemExit(
            "config has no 'traits' section. The older 'screening_traits' + "
            "'trait_config' pair was replaced by one 'traits' block: a scalar "
            "value is pinned, a list is swept, and swept keys take their "
            "stage-1 value from a 'screening' sub-block."
        )
    if not isinstance(raw, dict):
        raise SystemExit("'traits' must be an object")

    raw = dict(raw)
    screening = dict(raw.pop("screening", {}) or {})
    if not isinstance(screening, dict):
        raise SystemExit("'traits.screening' must be an object")

    known = _trait_field_names()
    pin: Dict[str, Any] = {}
    swept: Dict[str, List[Any]] = {}
    for key, value in raw.items():
        if key in _SHAPE_DERIVED_TRAITS:
            raise SystemExit(
                f"'traits.{key}' is derived from the problem shape and the "
                f"tile, not configured. Remove it."
            )
        if key not in known:
            near = sorted(k for k in known if k.startswith(key[:4]))
            hint = f" Did you mean one of: {', '.join(near)}?" if near else ""
            raise SystemExit(f"'traits.{key}' is not a TraitSpec field.{hint}")
        if isinstance(value, list):
            if not value:
                raise SystemExit(f"'traits.{key}' is an empty list")
            swept[key] = list(value)
        else:
            pin[key] = value

    # A one-value list is a pin, not a sweep. This is what narrowing from the
    # command line produces -- ``--epilogues cshuffle`` rewrites the list to a
    # single element -- and it would otherwise collide with the inherited
    # ``screening`` entry, which still names the value that was dropped. The
    # entry is removed rather than rejected because with one candidate there
    # is nothing left to choose.
    narrowed = [k for k, v in swept.items() if len(v) == 1]
    for key in narrowed:
        pin[key] = swept.pop(key)[0]

    # A scalar in ``traits`` IS the pin, so a ``screening`` entry for the same
    # key is redundant and ignored rather than rejected. That is what makes
    # narrowing work from the command line: ``--epilogues cshuffle`` and
    # ``--set traits.epilogue=cshuffle`` both leave the inherited
    # ``screening.epilogue`` behind, naming a value the sweep no longer
    # visits, and erroring there would make the obvious invocation fail.
    for key in list(screening):
        if key in pin and key not in swept:
            screening.pop(key)

    for key in screening:
        if key not in swept:
            raise SystemExit(
                f"'traits.screening.{key}' names a trait that is not declared. "
                f"Add 'traits.{key}', or remove the screening entry."
            )
    missing = sorted(set(swept) - set(screening))
    if missing:
        raise SystemExit(
            "every swept trait needs an explicit stage-1 value in "
            "'traits.screening'; missing: " + ", ".join(missing)
        )
    for key, value in screening.items():
        if value not in swept[key]:
            raise SystemExit(
                f"'traits.screening.{key}' = {value!r} is not among the swept "
                f"values {swept[key]!r}. The pin must be a value the sweep "
                f"also visits, or stage 1 ranks geometry under a kernel stage "
                f"2 never builds."
            )
        pin[key] = value
    return pin, swept


#: Default margin for :func:`select_finalists`, in percent. 12 is the measured
#: spread of the stage-2 gain across geometries on the square sweep
#: (1.022x-1.143x), i.e. how far the trait stage can reorder what stage 1
#: ranked. Anything closer to the cut than this is inside the noise of the
#: ranking that produced it.
DEFAULT_FINALIST_MARGIN_PCT = 12.0


def _finalist_budget(selection: Dict[str, Any]) -> int:
    """How many verified tiles the trait stage can be given at most."""
    n = int(selection["tile_finalists"])
    margin = float(selection.get("finalist_margin_pct", DEFAULT_FINALIST_MARGIN_PCT))
    # With a margin the pool can exceed tile_finalists, but not without bound:
    # cap the verification walk so a pathological run cannot verify everything.
    return n if margin <= 0 else 2 * n


def select_finalists(
    ranked: Sequence[Dict[str, Any]], selection: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Top ``tile_finalists``, plus anything within ``finalist_margin_pct``.

    A fixed top-N cut assumes stage 1's ranking is stage 2's ranking. It is
    not: stage 1 times every geometry under one pinned trait config, and the
    gain from each geometry's *own* best traits varies -- measured 1.022x to
    1.143x across the square sweep's finalists, reordering 13 of 20. The cut
    itself sat at a 0.9% margin between rank 20 and rank 21, far inside that
    spread, so which geometries survived was close to arbitrary.

    Keeping everything within a margin of rank-N makes the cost adaptive: a
    well-separated boundary keeps exactly N, a crowded one keeps more --
    crowded being precisely when the cut is unsafe. ``finalist_margin_pct: 0``
    restores the fixed top-N.
    """
    n = int(selection["tile_finalists"])
    margin = float(selection.get("finalist_margin_pct", DEFAULT_FINALIST_MARGIN_PCT))
    head = list(ranked)[:n]
    if margin <= 0 or len(ranked) <= n or not head:
        return head
    cutoff_ms = head[-1].get("best_ms")
    if not cutoff_ms:
        return head
    limit = cutoff_ms * (1.0 + margin / 100.0)
    extra = [r for r in list(ranked)[n:] if r.get("best_ms") and r["best_ms"] <= limit]
    return head + extra


def enumerate_tile_configs(config: Dict[str, Any]) -> List[UniversalGemmSpec]:
    """Stage 1: geometry varies, traits are held at the declared screening pin."""
    tile = config["tile_config"]
    pin, _ = resolve_traits(config)
    specs = [
        _make_spec(config, tm, tn, tk, wm, wn, pin)
        for tm in tile["tile_m"]
        for tn in tile["tile_n"]
        for tk in tile["tile_k"]
        for wm in tile["warp_m"]
        for wn in tile["warp_n"]
    ]
    return _dedupe_valid(specs, arch=config["target"]["arch"])


def _trait_combos(swept: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """The cartesian product of the swept traits, minus known-dead pairings.

    ``dtl_prefetch`` without ``direct_to_lds`` is rejected by
    ``is_valid_spec``, so pairing them out here keeps the grid from inflating
    with combinations that can only be discarded.
    """
    keys = sorted(swept)
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*(swept[k] for k in keys))
    ]
    return [
        c for c in combos if not (c.get("dtl_prefetch") and not c.get("direct_to_lds"))
    ]


def enumerate_trait_configs(
    config: Dict[str, Any], finalists: Sequence[UniversalGemmSpec]
) -> List[UniversalGemmSpec]:
    """Stage 2: the full trait space, over each surviving geometry.

    Generic over whatever ``traits`` declares as a list. The previous version
    was a fixed nest of six loops, so the other 15 ``TraitSpec`` fields could
    be *written* in the config and were then silently ignored -- a sweep could
    ask for four TDM depths and rank four copies of one kernel.
    """
    _, swept = resolve_traits(config)
    combos = _trait_combos(swept)
    specs: List[UniversalGemmSpec] = []
    for base in finalists:
        for combo in combos:
            specs.append(replace(base, trait=replace(base.trait, **combo)))
    return _dedupe_valid(specs, arch=config["target"]["arch"])


def _kernel_fingerprint(spec: UniversalGemmSpec, arch: str) -> str:
    """Hash of the *compiled* kernel, blind to its name.

    Compiled rather than IR because inertness has two sources and only one is
    visible in the IR. ``lds_swizzle`` and ``ab_load_elem_bytes`` change (or
    fail to change) what the emitter writes, so IR would catch them.
    ``waves_per_eu`` lowers to an ``"amdgpu-waves-per-eu"`` *attribute*: the IR
    always differs, and whether it means anything is the backend's decision --
    on an LDS-pinned tile it does not, and all values compile to identical
    machine code. Only comparing the HSACO sees both.

    The kernel is renamed first because several traits are spelled into the
    kernel name, which would otherwise make every variant differ by name bytes
    alone. Costs about 0.1s per call.
    """
    import hashlib

    kernel = build_universal_gemm(spec, arch=arch)
    kernel.name = "PROBE"
    return hashlib.sha1(compile_kernel(kernel, arch=arch).hsaco).hexdigest()


def warn_if_sweep_is_inert(
    config: Dict[str, Any], finalists: Sequence[UniversalGemmSpec]
) -> List[str]:
    """Flag swept traits whose values emit *the same kernel*.

    An inert trait is worse than a useless one: the sweep builds, times and
    ranks several copies of one kernel, then reports a confident winner among
    them. Three have been found by hand on this target, each only after a
    result looked odd --

      * ``lds_swizzle`` under TDM: the mover owns the LDS write, so there is
        no per-lane column left for the emitter to permute;
      * ``waves_per_eu`` on an LDS-pinned tile: one workgroup is resident
        whatever the hint says, so the allocator has nothing to trade;
      * ``ab_load_elem_bytes`` under TDM: the descriptor replaces the per-lane
        global load, so there is no load width to widen.

    Nothing in the config says so in any of those cases, and the ranking looks
    normal. Rather than keep rediscovering instances, compare the emitted IR:
    vary one trait at a time over a single base spec and group the values by
    what they actually emit. Costs ``sum(len(values))`` lowerings -- 13 for
    the shipped square config, a fraction of a second -- and no GPU.
    """
    _, swept = resolve_traits(config)
    if not swept or not finalists:
        return []
    arch = config["target"]["arch"]

    # Inertness can be geometry-dependent -- ``waves_per_eu`` is a no-op only
    # where LDS already pins occupancy to one workgroup, which is true of the
    # big tiles and false of the small ones. A single base would answer for
    # whichever end it happened to pick, so sample across the size range.
    ordered = sorted(
        finalists, key=lambda s: s.tile.tile_m * s.tile.tile_n * s.tile.tile_k
    )
    bases = [ordered[0], ordered[-1]]  # two compiles per value, not three
    seen, probes = set(), []
    for b in bases:
        tag = (
            b.tile.tile_m,
            b.tile.tile_n,
            b.tile.tile_k,
            b.tile.warp_m,
            b.tile.warp_n,
        )
        if tag not in seen:
            seen.add(tag)
            probes.append(b)

    warnings: List[str] = []
    for key, values in sorted(swept.items()):
        inert_on, duplicate_sets = 0, []
        degenerate = None
        for base in probes:
            groups: Dict[str, List[Any]] = {}
            for value in values:
                candidate = replace(base, trait=replace(base.trait, **{key: value}))
                try:
                    if not is_valid_spec(candidate, arch=arch)[0]:
                        continue
                    digest = _kernel_fingerprint(candidate, arch)
                except Exception:
                    return warnings  # diagnostics must never break the run
                groups.setdefault(digest, []).append(value)
            valid = sum(len(v) for v in groups.values())
            if valid < 2:
                degenerate = valid
                continue
            dups = [vals for vals in groups.values() if len(vals) > 1]
            if dups:
                inert_on += 1
                duplicate_sets = dups  # representative; they agree in practice

        if degenerate is not None and not duplicate_sets:
            warnings.append(
                f"sweeping {key!r} is degenerate: only {degenerate} of "
                f"{len(values)} value(s) are valid on this target, so there is "
                f"nothing to compare. Pin it instead of sweeping it."
            )
        elif duplicate_sets:
            shown = "; ".join(
                " / ".join(repr(v) for v in vals) for vals in duplicate_sets
            )
            where = (
                "every geometry sampled"
                if inert_on == len(probes)
                else f"{inert_on} of {len(probes)} geometries sampled"
            )
            warnings.append(
                f"sweeping {key!r} emits the SAME kernel for: {shown} -- on "
                f"{where}. Those arms differ only in the kernel name, so timing "
                f"differences between them are noise and ranking them is "
                f"meaningless. Either the trait does nothing in this "
                f"configuration, or something else here already pins it."
            )
    return warnings


def warn_if_sweep_changes_validity(
    config: Dict[str, Any], finalists: Sequence[UniversalGemmSpec]
) -> List[str]:
    """Flag swept traits that change which *geometries* are admissible.

    Stage 1 ranks geometry under the screening pin, so a trait that alters
    ``is_valid_spec``'s verdict is judged on tiles chosen for a different
    value of itself. ``tdm_depth`` is the clearest case: each ring stage costs
    an A/B region, so depth 4 admits materially fewer tiles than depth 2 and
    its arm is offered geometry it cannot build.

    This does not block the run -- sweeping depth is still useful for a quick
    look -- it names the trap so the result is read correctly, and points at
    the one-config-per-value pattern for a fair comparison.
    """
    pin, swept = resolve_traits(config)
    arch = config["target"]["arch"]
    tile = config["tile_config"]
    geometries = [
        (tm, tn, tk, wm, wn)
        for tm in tile["tile_m"]
        for tn in tile["tile_n"]
        for tk in tile["tile_k"]
        for wm in tile["warp_m"]
        for wn in tile["warp_n"]
    ]

    def admitted(traits: Dict[str, Any]) -> int:
        n = 0
        for g in geometries:
            try:
                spec = _make_spec(config, *g, traits)
            except Exception:
                continue
            if is_valid_spec(spec, arch=arch)[0]:
                n += 1
        return n

    base_n = admitted(pin)
    warnings: List[str] = []
    for key, values in sorted(swept.items()):
        counts = {}
        for v in values:
            counts[v] = admitted({**pin, key: v})
        if len(set(counts.values())) > 1:
            spread = ", ".join(f"{v!r}->{n}" for v, n in counts.items())
            warnings.append(
                f"sweeping {key!r} changes the admissible tile pool ({spread}; "
                f"screening pin admits {base_n}). Stage 1 ranked geometry under "
                f"{key}={pin.get(key)!r}, so the other arms are judged on tiles "
                f"selected for that value. Run one config per {key} value for a "
                f"comparison that is fair on the geometry axis."
            )
    return warnings


def apply_lever(spec: UniversalGemmSpec, lever: Dict[str, Any]) -> UniversalGemmSpec:
    """Overlay one lever's trait/data overrides onto a spec.

    A lever is whatever distinguishes an optimization from the baseline.  An
    empty override set is the baseline itself, which is why ``baseline`` is a
    registered lever rather than a special case -- it goes through the same
    build, the same verification and the same timing path as every other arm,
    so a difference between arms cannot be an artifact of how they were run.
    """
    trait_over = dict(lever.get("trait", {}))
    data_over = dict(lever.get("data", {}))
    out = spec
    if trait_over:
        out = replace(out, trait=replace(out.trait, **trait_over))
    if data_over:
        out = replace(out, data=replace(out.data, **data_over))
    return out


def required_padded_n(n: int, specs: Sequence[UniversalGemmSpec]) -> int:
    if not specs:
        return n
    return max(math.ceil(n / spec.tile.tile_n) * spec.tile.tile_n for spec in specs)


# ---------------------------------------------------------------------------
# Machine state
#
# Every retracted conclusion in this workload traced to one gap: the harness
# recorded no machine state, so "was that run clean?" had to be reconstructed
# afterwards from spot checks taken at unrelated times -- and twice the answer
# came back different from what had already been written down. A timing row
# without the state it was measured under is not evidence.
#
# What is worth recording is narrow. Contention is the confound that matters,
# and the only reliable signal for it is the KFD process list: ``gpu_busy_percent``
# does NOT work here -- a saturating GEMM moves it by about one point, so it
# cannot distinguish an idle GPU from a loaded one.
#
# A faulted run also leaves its process alive holding VRAM; it does not exit.
# The corpse makes the *next* run fault immediately, which reads as a fresh
# failure. Seen three times, once for eight hours. A "before" snapshot naming
# a stale sweep is the cheapest way to catch that at the time rather than
# later.
# ---------------------------------------------------------------------------


def _kfd_processes() -> Tuple[List[Dict[str, Any]], str]:
    """Processes currently holding the GPU, and where the answer came from.

    ``rocm-smi --showpids`` first; ``/sys/class/kfd/kfd/proc`` when rocm-smi is
    absent or unparseable. Returns ``([], "unavailable")`` rather than raising:
    a missing GPU must not take down a host-only run.
    """
    smi = shutil.which("rocm-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--showpids"], capture_output=True, text=True, timeout=30
            ).stdout
            procs: List[Dict[str, Any]] = []
            for line in out.splitlines():
                parts = line.split()
                # Rows are "<pid> <name> <gpus> <vram> <sdma> <cu>"; the header
                # and the banner lines do not start with a bare integer.
                if len(parts) >= 4 and parts[0].isdigit():
                    procs.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "vram_bytes": int(parts[3]) if parts[3].isdigit() else None,
                        }
                    )
            return procs, "rocm-smi"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    proc_dir = Path("/sys/class/kfd/kfd/proc")
    if proc_dir.is_dir():
        return (
            [
                {"pid": int(p.name), "name": None, "vram_bytes": None}
                for p in proc_dir.iterdir()
                if p.name.isdigit()
            ],
            "sysfs",
        )
    return [], "unavailable"


def _is_ours(pid: int) -> bool:
    """Whether ``pid`` belongs to this run (us, or a worker we forked)."""
    if pid == os.getpid():
        return True
    try:
        # field 6 of /proc/<pid>/stat is the session id; our forked workers
        # share ours. comm can contain spaces, so split on the closing paren.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[3]) == os.getsid(0)
    except (OSError, IndexError, ValueError):
        return False


def machine_state(label: str) -> Dict[str, Any]:
    """A snapshot to attach to a measurement, so it can be judged later."""
    procs, source = _kfd_processes()
    foreign = [p for p in procs if not _is_ours(p["pid"])]
    return {
        "label": label,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "kfd_source": source,
        "kfd_processes": procs,
        "kfd_count": len(procs),
        "foreign_kfd_count": len(foreign),
        # Only meaningful when the source actually answered. "unavailable"
        # means unknown, which must not be recorded as clean.
        "clean": (len(foreign) == 0) if source != "unavailable" else None,
    }


def warn_if_contended(state: Dict[str, Any]) -> None:
    """Say so at the time. A warning after the fact is archaeology."""
    if state.get("clean") is False:
        others = ", ".join(
            f"{p['pid']}({p['name']})"
            for p in state["kfd_processes"]
            if not _is_ours(p["pid"])
        )
        print(
            f"WARNING [{state['label']}]: {state['foreign_kfd_count']} foreign KFD "
            f"process(es) on the GPU -- {others}. Timings from this run are not "
            f"comparable. A stale 'fp8_gemm_sweep' here is a corpse from a "
            f"previous faulted run; clear it before trusting anything.",
            flush=True,
        )
    elif state.get("clean") is None:
        print(
            f"WARNING [{state['label']}]: could not read the KFD process list, so "
            f"contention is unknown -- not the same as clean.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Problem setup, timing and the exact gate
# ---------------------------------------------------------------------------


def _as_u8_buffer(array) -> ctypes.Array:
    return (ctypes.c_ubyte * int(array.nbytes)).from_buffer(array)


@dataclass
class PreparedProblem:
    """Device buffers plus (optionally) the exact host reference.

    A/B are e4m3 bytes, C is raw bf16 halves carried as ``uint16`` so no
    float16 view can lie about the encoding on the way to or from the device.
    """

    rt: Runtime
    shape: Tuple[int, int, int]
    padded_n: int
    a_dev: int
    b_dev: int
    c_dev: int
    c_host: Any
    reference: Any

    @classmethod
    def create(
        cls,
        shape: Tuple[int, int, int],
        padded_n: int,
        *,
        with_reference: bool,
    ) -> "PreparedProblem":
        import numpy as np

        from rocke.instances.common.manifest_runner.gemm import _fp8e4m3_encode

        m, n, k = shape
        if 25 * k >= (1 << 24):
            raise ValueError(
                f"K={k} overflows the exact-fp32-accumulate argument "
                f"(25*K must stay below 2^24, so K <= {_MAX_EXACT_K}); "
                "lower K or replace the exact gate with a justified tolerance"
            )
        rng = np.random.default_rng(0xC0FFEE)
        a_f32 = rng.integers(-5, 6, size=(m, k), dtype=np.int16).astype(np.float32)
        b_f32 = rng.integers(-5, 6, size=(n, k), dtype=np.int16).astype(np.float32)
        a = _fp8e4m3_encode(np, a_f32)
        b = np.zeros((padded_n, k), dtype=np.uint8)
        b[:n] = _fp8e4m3_encode(np, b_f32)
        c = np.empty((m, n), dtype=np.uint16)

        reference = None
        if with_reference:
            from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16

            # The reference reproduces the kernel's final RNE to bf16; every
            # step before it is exact, so this is an equality, not a bound.
            reference = _f32_from_bf16(np, _bf16_from_f32(np, a_f32 @ b_f32.T))

        rt = Runtime()
        a_dev = rt.alloc(a.nbytes)
        b_dev = rt.alloc(b.nbytes)
        c_dev = rt.alloc(c.nbytes)
        rt.memcpy_h2d(a_dev, _as_u8_buffer(a), a.nbytes)
        rt.memcpy_h2d(b_dev, _as_u8_buffer(b), b.nbytes)
        rt.memset(c_dev, 0, c.nbytes)
        return cls(rt, shape, padded_n, a_dev, b_dev, c_dev, c, reference)

    @property
    def args(self) -> bytes:
        m, n, k = self.shape
        return struct.pack("<QQQiii", self.a_dev, self.b_dev, self.c_dev, m, n, k)

    def close(self) -> None:
        try:
            self.rt.sync()
        except Exception:  # best effort: a sticky fault must not mask results
            pass
        for ptr in (self.a_dev, self.b_dev, self.c_dev):
            try:
                self.rt.free(ptr)
            except Exception:
                pass


def _launch_geometry(
    spec: UniversalGemmSpec, shape: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Grid is (N_tiles, M_tiles), matching the verified manifest runner."""
    m, n, _k = shape
    gx = (n + spec.tile.tile_n - 1) // spec.tile.tile_n
    gy = (m + spec.tile.tile_m - 1) // spec.tile.tile_m
    return (gx, gy, 1), (spec.block_size, 1, 1)


def _time_function(
    rt: Runtime, fn, grid, block, args: bytes, *, warmup: int, iters: int
) -> float:
    for _ in range(warmup):
        rt.launch(fn, grid, block, args)
    rt.wait_stream(0)
    begin = rt.event()
    finish = rt.event()
    try:
        begin.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        finish.record()
        finish.synchronize()
        elapsed = begin.elapsed_to(finish) / iters
    finally:
        begin.destroy()
        finish.destroy()
        rt.wait_stream(0)
    return elapsed


def _is_sticky(exc: BaseException) -> bool:
    return any(code in str(exc) for code in _STICKY_HIP_ERRORS)


def _total_build_ms(record: BuildRecord) -> float:
    return float(record.ir_build_ms + record.ir_lower_ms + record.comgr_ms)


def _result_header(
    record: BuildRecord, spec: UniversalGemmSpec, *, arch: str
) -> Dict[str, Any]:
    """The build-side half of a row, filled before any launch is attempted.

    Split out so a candidate that takes the process down with it is still
    reported with the same shape as one that ran.
    """
    return {
        "id": _spec_identity(spec, arch=arch),
        "name": record.name,
        "spec": record.spec_dict,
        "build_ok": record.ok,
        "build_error": record.error,
        "build_ms": _total_build_ms(record),
        "build_timings_ms": {
            "ir_build": record.ir_build_ms,
            "ir_lower": record.ir_lower_ms,
            "comgr": record.comgr_ms,
        },
        "hsaco": record.hsaco_path,
        "hsaco_bytes": record.hsaco_bytes,
        "elf_meta": record.elf_meta,
        "verified": False,
    }


def benchmark_record(
    problem: PreparedProblem,
    record: BuildRecord,
    *,
    arch: str,
    warmup: int,
    iters: int,
    attempts: int,
) -> Dict[str, Any]:
    spec = _spec_from_dict(record.spec_dict)
    result = _result_header(record, spec, arch=arch)
    if not record.ok:
        result["error"] = record.error or "build failed"
        return result

    module = None
    try:
        module = problem.rt.load_module(Path(record.hsaco_path).read_bytes())
        fn = module.get_function(record.name)
        grid, block = _launch_geometry(spec, problem.shape)
        samples = [
            _time_function(
                problem.rt, fn, grid, block, problem.args, warmup=warmup, iters=iters
            )
            for _ in range(attempts)
        ]
        ms = statistics.median(samples)
        m, n, k = problem.shape
        result.update(
            {
                "samples_ms": samples,
                "median_ms": ms,
                "best_ms": min(samples),
                "tflops": 2.0 * m * n * k / (ms * 1.0e9),
            }
        )
    except Exception as exc:
        result.setdefault("error", f"{type(exc).__name__}: {exc}")
        if _is_sticky(exc):
            result["device_fault"] = True
    finally:
        # A sticky error is re-raised by every later HIP call, so an unguarded
        # drain raises a *second* exception out of the finally and discards the
        # first -- which is how a faulting sweep dies with a traceback that
        # never names the candidate responsible. Cleanup is best effort; the
        # diagnosis stays in ``result``.
        cleanups = [lambda: problem.rt.wait_stream(0)]
        if module is not None:
            cleanups.append(module.unload)
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as exc:
                result.setdefault("error", f"{type(exc).__name__}: {exc}")
                if _is_sticky(exc):
                    result["device_fault"] = True
    return result


def verify_record(
    problem: PreparedProblem, record: BuildRecord, result: Dict[str, Any]
) -> bool:
    """Exact compare.  See the module docstring for why this is not a tolerance."""
    import numpy as np

    from rocke.dispatch.gemm.binding import _f32_from_bf16

    if problem.reference is None or not record.ok:
        return False
    spec = _spec_from_dict(record.spec_dict)
    module = None
    try:
        module = problem.rt.load_module(Path(record.hsaco_path).read_bytes())
        fn = module.get_function(record.name)
        grid, block = _launch_geometry(spec, problem.shape)
        problem.rt.memset(problem.c_dev, 0, problem.c_host.nbytes)
        problem.rt.launch_blocking(fn, grid, block, problem.args)
        problem.rt.memcpy_d2h(
            _as_u8_buffer(problem.c_host), problem.c_dev, problem.c_host.nbytes
        )
        actual = _f32_from_bf16(np, problem.c_host)
        error = np.abs(actual - problem.reference)
        incorrect = int(np.count_nonzero(error > 0.0))
        result.update(
            {
                "verified": incorrect == 0,
                "max_abs_diff": float(error.max()),
                "incorrect": incorrect,
                "elements": int(actual.size),
            }
        )
        return incorrect == 0
    except Exception as exc:
        result["verify_error"] = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        if module is not None:
            try:
                module.unload()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------


def rank_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (r for r in results if "median_ms" in r and not r.get("error")),
        key=lambda r: float(r["median_ms"]),
    )


def _record_by_id(
    records: Sequence[BuildRecord], *, arch: str
) -> Dict[str, BuildRecord]:
    return {_spec_identity(_spec_from_dict(r.spec_dict), arch=arch): r for r in records}


def _record_from_result(result: Dict[str, Any]) -> BuildRecord:
    timings = result.get("build_timings_ms", {})
    spec = result["spec"]
    return BuildRecord(
        name=result["name"],
        spec_dict=spec,
        ok=bool(result.get("build_ok")),
        error=str(result.get("build_error", "")),
        hsaco_path=str(result["hsaco"]),
        hsaco_bytes=int(result.get("hsaco_bytes", 0)),
        block_m=int(spec["tile"]["tile_m"]),
        block_n=int(spec["tile"]["tile_n"]),
        block_k=int(spec["tile"]["tile_k"]),
        threads_per_block=int(spec["block_size"]),
        ir_build_ms=float(timings.get("ir_build", 0.0)),
        ir_lower_ms=float(timings.get("ir_lower", 0.0)),
        comgr_ms=float(timings.get("comgr", 0.0)),
        elf_meta=dict(result.get("elf_meta", {})),
    )


_CSV_FIELDS = (
    "rank",
    "id",
    "name",
    "median_ms",
    "best_ms",
    "tflops",
    "verified",
    "max_abs_diff",
    "incorrect",
    "build_ms",
    "hsaco_bytes",
    "device_fault",
    "error",
    "verify_error",
)


def _write_results(
    output_dir: Path, stem: str, results: Sequence[Dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(list(results), indent=2, sort_keys=True)
    )
    ranked_ids = {r["id"]: i + 1 for i, r in enumerate(rank_results(results))}
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({"rank": ranked_ids.get(result["id"], ""), **result})


def _load_stage(output_dir: Path, stem: str, needed_by: str) -> List[Dict[str, Any]]:
    path = output_dir / f"{stem}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is required when --stages includes {needed_by!r}; "
            f"run the {stem!r} stage first or point --out at a run that has."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Build + benchmark driver
# ---------------------------------------------------------------------------


def build_and_benchmark(
    specs: Sequence[UniversalGemmSpec],
    *,
    cache_dir: Path,
    shape: Tuple[int, int, int],
    padded_n: int,
    arch: str,
    isa: str,
    workers: int,
    warmup: int,
    iters: int,
    attempts: int,
    with_reference: bool,
    stage: str = "stage",
) -> Tuple[
    List[BuildRecord], List[Dict[str, Any]], PreparedProblem, List[Dict[str, Any]]
]:
    records = build_all_instances(
        specs, cache_dir=cache_dir, arch=arch, isa=isa, parallel=workers
    )
    # Taken after the builds and before the first launch, so it describes the
    # GPU the measurements actually ran on rather than the one the process
    # started against.
    states: List[Dict[str, Any]] = [machine_state(f"{stage}:before")]
    warn_if_contended(states[0])

    problem = PreparedProblem.create(shape, padded_n, with_reference=with_reference)
    results: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        result = benchmark_record(
            problem,
            record,
            arch=arch,
            warmup=warmup,
            iters=iters,
            attempts=attempts,
        )
        results.append(result)
        if result.get("device_fault"):
            # Every later HIP call in this process returns the same latched
            # code, so continuing would manufacture identical bogus failures
            # for every remaining candidate. Stop with what is real.
            #
            # Snapshot here, not later: this is the only moment that can
            # distinguish "the GPU was already dirty" from "we faulted it".
            fault_state = machine_state(f"{stage}:at_fault")
            fault_state["faulted_candidate"] = record.name
            fault_state["faulted_index"] = index + 1
            fault_state["candidates_total"] = len(records)
            states.append(fault_state)
            print(
                f"device fault at candidate {index + 1}/{len(records)} "
                f"({record.name}); stopping this stage and keeping "
                f"{len(results)} measured rows",
                flush=True,
            )
            break
    # An "after" snapshot catches someone joining mid-run, which a "before"
    # alone cannot: the run would look clean and the timings would not be.
    states.append(machine_state(f"{stage}:after"))
    warn_if_contended(states[-1])
    return records, results, problem, states


# ---------------------------------------------------------------------------
# Out-of-process final timing
# ---------------------------------------------------------------------------


def _independent_final_timings(
    output_dir: Path,
    record: BuildRecord,
    shape: Tuple[int, int, int],
    padded_n: int,
    *,
    warmup: int,
    iters: int,
    samples: int,
) -> List[float]:
    """Re-time in fresh processes, discarding the first.

    A fresh process is the only way to shed a warm module cache, a latched
    HIP error, and whatever allocator state the in-process ranking left
    behind. The first run of a fresh process is discarded per the runbook's
    benchmark hygiene, so ``samples + 1`` processes are spawned.
    """
    payload = {
        "record": asdict(record),
        "shape": list(shape),
        "padded_n": padded_n,
        "warmup": warmup,
        "iters": iters,
    }
    worker_input = output_dir / "final_worker.json"
    worker_input.write_text(json.dumps(payload, indent=2))

    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = (
        package_root
        if not env.get("PYTHONPATH")
        else package_root + os.pathsep + env["PYTHONPATH"]
    )

    out: List[float] = []
    for attempt in range(samples + 1):
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-record",
                str(worker_input),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        lines = [
            line
            for line in proc.stdout.splitlines()
            if line.startswith(_WORKER_RESULT_PREFIX)
        ]
        if proc.returncode != 0 or not lines:
            raise RuntimeError(
                f"final timing worker failed ({proc.returncode}): "
                f"{proc.stdout[-800:]}{proc.stderr[-800:]}"
            )
        ms = float(json.loads(lines[-1][len(_WORKER_RESULT_PREFIX) :])["median_ms"])
        if attempt:  # discard the cold first process
            out.append(ms)
    return out


def _run_worker(path: Path) -> int:
    payload = json.loads(path.read_text())
    record = BuildRecord(**payload["record"])
    shape = tuple(int(v) for v in payload["shape"])
    problem = PreparedProblem.create(
        shape, int(payload["padded_n"]), with_reference=False
    )
    try:
        spec = _spec_from_dict(record.spec_dict)
        module = problem.rt.load_module(Path(record.hsaco_path).read_bytes())
        fn = module.get_function(record.name)
        grid, block = _launch_geometry(spec, shape)
        ms = _time_function(
            problem.rt,
            fn,
            grid,
            block,
            problem.args,
            warmup=int(payload["warmup"]),
            iters=int(payload["iters"]),
        )
        module.unload()
    finally:
        problem.close()
    print(_WORKER_RESULT_PREFIX + json.dumps({"median_ms": ms}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv_of(kind):
    def parse(text: str):
        return [kind(piece) for piece in text.split(",") if piece != ""]

    return parse


def _flag(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _nullable_int(text: str) -> Optional[int]:
    return None if text.strip().lower() in {"none", "null", ""} else int(text)


_CONFIG_OVERRIDES: Tuple[Tuple[str, Tuple[str, ...], Any, str], ...] = (
    ("--m", ("problem", "m"), int, "problem M"),
    ("--n", ("problem", "n"), int, "problem N"),
    ("--k", ("problem", "k"), int, "problem K"),
    ("--tile-m", ("tile_config", "tile_m"), _csv_of(int), "block tile M values"),
    ("--tile-n", ("tile_config", "tile_n"), _csv_of(int), "block tile N values"),
    ("--tile-k", ("tile_config", "tile_k"), _csv_of(int), "block tile K values"),
    ("--warp-m", ("tile_config", "warp_m"), _csv_of(int), "warp grid M values"),
    ("--warp-n", ("tile_config", "warp_n"), _csv_of(int), "warp grid N values"),
    ("--pipelines", ("traits", "pipeline"), _csv_of(str), "pipelines"),
    ("--schedulers", ("traits", "scheduler"), _csv_of(str), "schedulers"),
    ("--epilogues", ("traits", "epilogue"), _csv_of(str), "epilogues"),
    (
        "--waves-per-eu",
        ("traits", "waves_per_eu"),
        _csv_of(_nullable_int),
        "waves-per-EU values ('none' for the default)",
    ),
    (
        "--lds-swizzle",
        ("traits", "lds_swizzle"),
        _csv_of(_flag),
        "LDS swizzle flags",
    ),
    ("--lds-k-pad", ("traits", "lds_k_pad"), _csv_of(int), "LDS K padding"),
    (
        "--direct-to-lds",
        ("traits", "direct_to_lds"),
        _csv_of(_flag),
        "DirectToLDS load path",
    ),
    (
        "--dtl-prefetch",
        ("traits", "dtl_prefetch"),
        _csv_of(_flag),
        "DirectToLDS prefetch ping-pong",
    ),
    (
        "--tile-finalists",
        ("selection", "tile_finalists"),
        int,
        "geometries kept after stage 1",
    ),
    (
        "--final-timed",
        ("selection", "final_timed"),
        int,
        "candidates re-timed out of process",
    ),
    (
        "--final-samples",
        ("selection", "final_samples"),
        int,
        "fresh-process samples per finalist (one more is run and discarded)",
    ),
    ("--workers", ("benchmark", "workers"), int, "parallel build workers"),
    ("--warmup", ("benchmark", "warmup"), int, "warmup launches per candidate"),
    ("--iters", ("benchmark", "iters"), int, "timed launches per sample"),
    ("--attempts", ("benchmark", "attempts"), int, "in-process samples, median taken"),
)


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    applied: List[str] = []
    for flag, path, _kind, _help in _CONFIG_OVERRIDES:
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is None:
            continue
        section = config
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = value
        applied.append(f"{'.'.join(path)}={value}")
    return applied


def _parse_stages(text: Optional[str]) -> List[str]:
    if not text or text == "all":
        return list(STAGES)
    chosen = [piece.strip() for piece in text.split(",") if piece.strip()]
    unknown = [piece for piece in chosen if piece not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {', '.join(STAGES)}")
    return [stage for stage in STAGES if stage in chosen]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="results directory; falls back to $ROCKE_PERF_OUT. Required. "
        "Kept out of the committed config on purpose.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=f"comma-separated subset of: {', '.join(STAGES)} (default: all). "
        "A later stage reads the earlier stage's JSON from --out, so a run "
        "can be resumed or a single stage repeated.",
    )
    parser.add_argument(
        "--levers",
        type=str,
        default=None,
        help="comma-separated lever names for the 'lever' stage "
        "(default: every lever in the config)",
    )
    for flag, path, kind, help_text in _CONFIG_OVERRIDES:
        parser.add_argument(
            flag,
            type=kind,
            default=None,
            help=f"{help_text} [overrides {'.'.join(path)}]",
        )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY.PATH=VALUE",
        help="override any config key, repeatable; value is parsed as JSON "
        "with a bare-string fallback (e.g. --set traits.tdm_depth=[1,2] "
        "--set traits.screening.epilogue=default)",
    )
    parser.add_argument("--worker-record", type=Path, help=argparse.SUPPRESS)
    return parser


def flush_machine_state(output_dir: Path, states: List[Dict[str, Any]]) -> None:
    """Write ``machine_state.json`` now, not at the end of a successful run.

    The snapshots exist to explain a run that went wrong, and a run that goes
    wrong is exactly the one that does not reach the end of ``main``. A device
    fault latches onto the HIP context, so the next allocation raises and the
    process dies with the ``<stage>:at_fault`` snapshot still only in memory --
    which is what happened the one time a sweep actually faulted: no
    ``machine_state.json``, no ``summary.json``, and an absent file is
    indistinguishable from "this run predates the feature".

    Called after every stage and from ``main``'s ``finally``, so the file on
    disk always reflects the snapshots taken so far. Best-effort by design: a
    failure to write diagnostics must never replace the error being diagnosed.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "machine_state.json").write_text(
            json.dumps(states, indent=2, sort_keys=True, default=str)
        )
    except Exception as exc:  # diagnostics must not mask the real failure
        print(f"  WARNING: could not write machine_state.json: {exc}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_record:
        return _run_worker(args.worker_record)

    config = load_config(args.config)
    overrides = _apply_overrides(config, args)
    overrides += apply_set_overrides(config, args.set_overrides)
    stages = _parse_stages(args.stages)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "hsaco_cache"

    target = config["target"]
    problem = config["problem"]
    selection = config["selection"]
    benchmark = config["benchmark"]
    arch = str(target["arch"])
    isa = str(target["isa"])
    shape = (int(problem["m"]), int(problem["n"]), int(problem["k"]))

    print(f"stages : {', '.join(stages)}")
    print(f"shape  : {shape[0]}x{shape[1]}x{shape[2]}")
    print(f"out    : {output_dir}")
    if overrides:
        print("overrides: " + "  ".join(overrides))

    # Every snapshot taken during this run, in order. Attached to summary.json
    # and written standalone, so a result can always be judged against the
    # machine it was measured on rather than a spot check taken later.
    machine_states: List[Dict[str, Any]] = [machine_state("run:start")]
    warn_if_contended(machine_states[0])
    # Flushed after every stage, not only at the end: a stage that faults takes
    # the process down before ``main`` returns, and the snapshot explaining it
    # is the one that would be lost. See ``flush_machine_state``.
    flush_machine_state(output_dir, machine_states)

    summary: Dict[str, Any] = {
        "arch": arch,
        "dtype_ab": target["dtype_ab"],
        "dtype_c": target["dtype_c"],
        "layout": target["layout"],
        "shape": list(shape),
        "stages": stages,
        "config_overrides": overrides,
        "resolved_config": config,
    }

    tile_specs = enumerate_tile_configs(config)
    padded_n = required_padded_n(shape[1], tile_specs)

    # ---- stage 1: geometry -------------------------------------------------
    if "tile" in stages:
        print(
            f"tile   : {len(tile_specs)} candidates; " f"B rows {shape[1]}->{padded_n}",
            flush=True,
        )
        tile_records, tile_results, tile_problem, tile_states = build_and_benchmark(
            tile_specs,
            cache_dir=cache_dir,
            shape=shape,
            padded_n=padded_n,
            arch=arch,
            isa=isa,
            workers=int(benchmark["workers"]),
            warmup=int(benchmark["warmup"]),
            iters=int(benchmark["iters"]),
            attempts=int(benchmark["attempts"]),
            with_reference=True,
            stage="tile",
        )
        machine_states.extend(tile_states)
        flush_machine_state(output_dir, machine_states)
        try:
            record_map = _record_by_id(tile_records, arch=arch)
            # Verify DOWN the ranking until enough have passed, rather than
            # verifying a fixed top-N and keeping whatever survives. Under the
            # old form a wrong-but-fast kernel inside the top N shrank the
            # finalist pool instead of being replaced, because rank N+1 was
            # never verified -- and fast-but-wrong is exactly what a defective
            # trait produces, so the loss was silent and systematic.
            want = _finalist_budget(selection)
            walked = rejected = 0
            for result in rank_results(tile_results):
                if walked >= want or rejected >= want:
                    break
                record = record_map.get(result["id"])
                if record is None:
                    continue
                if verify_record(tile_problem, record, result):
                    walked += 1
                else:
                    rejected += 1
            if rejected:
                print(
                    f"  NOTE: {rejected} tile candidate(s) failed verification "
                    f"and were replaced from further down the ranking.",
                    flush=True,
                )
        finally:
            tile_problem.close()
        _write_results(output_dir, "tile", tile_results)
    else:
        tile_results = _load_stage(output_dir, "tile", "trait")

    ranked_tiles = rank_results(tile_results)
    finalists = [
        _spec_from_dict(r["spec"])
        for r in select_finalists(
            [r for r in ranked_tiles if r.get("verified")], selection
        )
    ]
    if "trait" in stages and not finalists:
        raise SystemExit(
            "no tile finalist passed verification -- every geometry either "
            "failed to build, faulted, or produced a wrong result. Check "
            "tile.csv before widening the search."
        )
    sweep_warnings = warn_if_sweep_changes_validity(config, finalists)
    sweep_warnings += warn_if_sweep_is_inert(config, finalists)
    for text in sweep_warnings:
        print(f"  WARNING: {text}", flush=True)

    summary["tile"] = {
        "candidate_count": len(tile_results),
        "verified_finalists": len(finalists),
        "finalist_margin_pct": float(
            selection.get("finalist_margin_pct", DEFAULT_FINALIST_MARGIN_PCT)
        ),
        "sweep_warnings": sweep_warnings,
    }

    # ---- stage 2: traits ---------------------------------------------------
    if "trait" in stages:
        trait_specs = enumerate_trait_configs(config, finalists)
        print(
            f"trait  : {len(trait_specs)} candidates "
            f"over {len(finalists)} geometries",
            flush=True,
        )
        trait_records, trait_results, trait_problem, trait_states = build_and_benchmark(
            trait_specs,
            cache_dir=cache_dir,
            shape=shape,
            padded_n=padded_n,
            arch=arch,
            isa=isa,
            workers=int(benchmark["workers"]),
            warmup=int(benchmark["warmup"]),
            iters=int(benchmark["iters"]),
            attempts=int(benchmark["attempts"]),
            # Screening only: correctness is re-established in 'final'.
            with_reference=False,
            stage="trait",
        )
        machine_states.extend(trait_states)
        flush_machine_state(output_dir, machine_states)
        trait_problem.close()
        _write_results(output_dir, "trait", trait_results)
        summary["trait"] = {"candidate_count": len(trait_results)}
    elif {"final", "lever"} & set(stages):
        trait_results = _load_stage(output_dir, "trait", "final")
        trait_records = [_record_from_result(r) for r in trait_results]
        summary["trait"] = {"candidate_count": len(trait_results)}

    # ---- stage 3: verified, fresh-process timing ---------------------------
    champion: Optional[Dict[str, Any]] = None
    if "final" in stages:
        final_problem = PreparedProblem.create(shape, padded_n, with_reference=True)
        record_map = _record_by_id(trait_records, arch=arch)
        robust: List[Dict[str, Any]] = []
        try:
            for result in rank_results(trait_results):
                record = record_map.get(result["id"])
                if record is None or not verify_record(final_problem, record, result):
                    continue
                result["independent_samples_ms"] = _independent_final_timings(
                    output_dir,
                    record,
                    shape,
                    padded_n,
                    warmup=int(benchmark["warmup"]),
                    iters=int(benchmark["iters"]),
                    samples=int(selection.get("final_samples", 5)),
                )
                result["final_median_ms"] = statistics.median(
                    result["independent_samples_ms"]
                )
                robust.append(result)
                if len(robust) >= int(selection["final_timed"]):
                    break
        finally:
            final_problem.close()
        _write_results(output_dir, "trait", trait_results)
        if not robust:
            raise SystemExit("no trait candidate passed final verification")
        champion = min(robust, key=lambda r: float(r["final_median_ms"]))
        summary["final"] = {
            "robust_finalist_count": len(robust),
            "champion_id": champion["id"],
            "champion_spec": champion["spec"],
        }

    # ---- stage 4: lever A/B against the champion ---------------------------
    if "lever" in stages:
        if champion is None:
            prior = _load_stage(output_dir, "summary", "lever")
            champion = {"spec": prior["final"]["champion_spec"]}
        levers = config.get("levers", {})
        names = (
            [n.strip() for n in args.levers.split(",")] if args.levers else list(levers)
        )
        unknown = [n for n in names if n not in levers]
        if unknown:
            raise SystemExit(f"unknown lever(s) {unknown}; have {list(levers)}")
        base_spec = _spec_from_dict(champion["spec"])
        arms = {name: apply_lever(base_spec, levers[name]) for name in names}
        print(f"lever  : {len(arms)} arms over the champion geometry", flush=True)

        arm_records = build_all_instances(
            list(arms.values()),
            cache_dir=cache_dir,
            arch=arch,
            isa=isa,
            parallel=int(benchmark["workers"]),
        )
        lever_problem = PreparedProblem.create(shape, padded_n, with_reference=True)
        lever_results: List[Dict[str, Any]] = []
        try:
            by_id = _record_by_id(arm_records, arch=arch)
            for name, spec in arms.items():
                record = by_id.get(_spec_identity(spec, arch=arch))
                if record is None:
                    lever_results.append({"lever": name, "error": "build missing"})
                    continue
                row = _result_header(record, spec, arch=arch)
                row["lever"] = name
                row["description"] = levers[name].get("description", "")
                # Every arm is verified, not just the baseline: an arm that
                # got faster by computing the wrong thing is the failure mode
                # an A/B harness exists to catch.
                if not verify_record(lever_problem, record, row):
                    lever_results.append(row)
                    continue
                row["independent_samples_ms"] = _independent_final_timings(
                    output_dir,
                    record,
                    shape,
                    padded_n,
                    warmup=int(benchmark["warmup"]),
                    iters=int(benchmark["iters"]),
                    samples=int(selection.get("final_samples", 5)),
                )
                row["final_median_ms"] = statistics.median(
                    row["independent_samples_ms"]
                )
                lever_results.append(row)
        finally:
            lever_problem.close()
        _write_results(output_dir, "lever", lever_results)

        timed = [r for r in lever_results if "final_median_ms" in r]
        base = next((r for r in timed if r["lever"] == "baseline"), None)
        if base is not None:
            for row in timed:
                row["speedup_vs_baseline"] = float(base["final_median_ms"]) / float(
                    row["final_median_ms"]
                )
            _write_results(output_dir, "lever", lever_results)
        summary["lever"] = {
            "arms": [
                {
                    "lever": r["lever"],
                    "verified": r.get("verified", False),
                    "speedup_vs_baseline": r.get("speedup_vs_baseline"),
                }
                for r in lever_results
            ]
        }

    machine_states.append(machine_state("run:end"))
    warn_if_contended(machine_states[-1])

    # A run is only as trustworthy as its dirtiest snapshot. Recording the
    # verdict next to the rows means a later reader does not have to re-derive
    # it -- which is the step that went wrong before.
    judged = [s for s in machine_states if s.get("clean") is not None]
    summary["machine_states"] = machine_states
    summary["machine_clean"] = all(s["clean"] for s in judged) if judged else None
    flush_machine_state(output_dir, machine_states)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    print(f"\nwrote {output_dir / 'summary.json'}")
    print(
        f"wrote {output_dir / 'machine_state.json'}  "
        f"({len(machine_states)} snapshots, clean={summary['machine_clean']})"
    )
    if summary["machine_clean"] is False:
        print(
            "  NOTE: the GPU was contended during this run. Treat the timings "
            "as indicative only; re-run on an idle GPU before concluding "
            "anything from them."
        )
    if "lever" in summary:
        print("\nlever A/B (ratio vs baseline; absolute timings in lever.csv):")
        for arm in summary["lever"]["arms"]:
            ratio = arm["speedup_vs_baseline"]
            shown = f"{ratio:.4f}x" if ratio else "n/a"
            flag = "" if arm["verified"] else "  [UNVERIFIED]"
            print(f"  {arm['lever']:<24} {shown}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
