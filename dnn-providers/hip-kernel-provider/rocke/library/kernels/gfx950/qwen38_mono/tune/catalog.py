# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The registered knob spaces of the true monokernel.

One factory per knob owner; each returns a :class:`~.knobs.KnobAdapter` for
one variant (the problem fields fixed). Domains list every built value of
every knob (losing levers included: they stay default-off knobs and enter
every search, DESIGN 9.2 step 4). Knobs that are not levers are excluded
with the reason (``adapter.excluded``): measurement-only builds (stamps,
``w_null``), test-only mutants (``mutate``, ``mutant``, ``hooks``) and layout
constants. The mutants are what the evaluator's must-fail gate uses.

| factory | owner | where |
|---|---|---|
| :func:`framework` | ``FrameworkKnobs`` + ``MailboxKnobs`` | ``framework.py``, ``rocke.helpers.mailbox`` |
| :func:`toy_chain` | ``ToyChainSpec`` (+ framework) | ``toy_chain.py`` |
| :func:`fused_moe` | ``FusedKnobs`` | ``moe_fused.py`` |
| :func:`moe_port` | ``MoeKnobs`` | ``moe.py`` |
| :func:`b3` | ``B3Knobs`` | ``stage_b3.py`` |
| :func:`stage_fn` | ``StageKnobs`` | ``stage_fn.py`` |
| :func:`mx_gemv` | ``MxDecodeGemvSpec`` | ``kernels/gfx950/mx_decode_gemv.py`` |
| :func:`ccl` | ``CclKnobs`` | ``ccl/layout.py`` |
| :func:`sched` | ATOM ``ATOM_Q38_SCHED`` levers | ATOM ``qwen4_exp_mono/kernels/sched.py`` |
| :func:`direct` | ``ROCKE_CODEOBJ_BACKEND`` + ``ROCKE_DIRECT_LEVERS`` | Rust direct compiler track |

Accuracy-gated levers (DESIGN rev3 12.3, N1-N11) are registered in
:data:`N_LEVERS`; a lever joins its owner's space (as a ``tolerance`` knob)
the moment its field exists on the owner (:func:`attach_n_levers`), so the
tracks only add the field and the registry entry's ``path`` / ``values``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocke.dispatch.tuning.search.params import BOOL, CATEGORICAL, ORDINAL

from .knobs import TOLERANCE, DataclassKnobs, Knob, KnobAdapter, LeverStringKnobs

__all__ = [
    "N_LEVERS",
    "NLever",
    "attach_n_levers",
    "b3",
    "ccl",
    "direct",
    "direct_env",
    "framework",
    "framework_knobs",
    "fused_moe",
    "moe_port",
    "mx_gemv",
    "sched",
    "stage_fn",
    "toy_chain",
    "variant_features",
]


def _bool(path, **kw) -> Knob:
    return Knob(path, (False, True), BOOL, **kw)


def _boolon(path, **kw) -> Knob:
    """A bool whose shipped default is True (level 0 = True)."""
    return Knob(path, (True, False), CATEGORICAL, **kw)


def _cat(path, values, **kw) -> Knob:
    return Knob(path, tuple(values), CATEGORICAL, **kw)


def _ord(path, values, **kw) -> Knob:
    return Knob(path, tuple(sorted(values)), ORDINAL, **kw)


# ------------------------------------------------------------- mailbox / fw
def mailbox_knobs(prefix: str = "mailbox.", *, impl_default: str = "asm") -> list[Knob]:
    """``MailboxKnobs`` levers. ``scope`` is fixed (``system`` is a peer-GPU
    requirement, not a lever)."""
    impls = ("asm", "ir") if impl_default == "asm" else ("ir", "asm")
    return [
        _cat(f"{prefix}impl", impls, note="ir: Python engine (tagged_poll)"),
        _cat(f"{prefix}pause", ("s_nop", "s_sleep1", "none")),
        _ord(f"{prefix}poll_max", (4, 8, 12, 16)),
        _cat(f"{prefix}fence_impl", ("asm", "ir")),
        _bool(f"{prefix}put_nt"),
        _cat(f"{prefix}numa_copies", (1, 2), structural=True, note="NUMA dual-copy mailboxes"),
    ]


def framework_knobs(prefix: str = "") -> list[Knob]:
    """``FrameworkKnobs`` levers (+ its mailbox). Measurement-only
    ``stamps`` / ``stamps_lds`` and the residency constant ``lds_min_words``
    are excluded."""
    return mailbox_knobs(f"{prefix}mailbox.") + [
        _boolon(f"{prefix}fresh_ids"),
        _cat(f"{prefix}lds_barrier", ("bare", "lds_only")),
        _ord(f"{prefix}head_sleep", (0, 32, 64, 128, 192)),
        _boolon(f"{prefix}phase_barriers"),
        _ord(f"{prefix}waves_per_eu", (0, 1, 2)),
        _bool(f"{prefix}step_first"),
        _cat(f"{prefix}step_load", ("vmem", "smem")),
        _ord(f"{prefix}kernarg_preload", (0, 3, 5, 8), note="Python engine"),
        _bool(f"{prefix}lds_touch"),
    ]


FRAMEWORK_EXCLUDED = {
    "stamps": "measurement only (stamped builds bias timing, DESIGN 9.1)",
    "stamps_lds": "applies to stamped builds only",
    "lds_min_words": "residency constant (one CTA per CU)",
    "numa_flip_arg": "a kernel-argument name; the toy chain adds it gated on numa_copies",
    "mailbox.scope": "system scope is a peer-GPU requirement (CCL), not a lever",
}


def framework(variant: str = "layer") -> DataclassKnobs:
    from ..framework import FrameworkKnobs

    return DataclassKnobs(
        "framework", FrameworkKnobs(), framework_knobs(), variant=variant, excluded=FRAMEWORK_EXCLUDED
    )


# --------------------------------------------------------------- toy chain
def toy_chain(depth: int, work: int, *, layers: int = 1, producers: int = 160) -> DataclassKnobs:
    """The framework's toy chain at one shape (the T9 gate's lever space)."""
    from ..framework import FrameworkKnobs
    from ..toy_chain import ToyChainSpec

    base = ToyChainSpec(depth=depth, work=work, layers=layers, producers=producers, knobs=FrameworkKnobs())
    knobs = [
        _cat("hop", ("a2a", "reduce_task", "last_arriver"), structural=True),
        _cat("consume", ("wave0", "waves")),
        _cat("reduce", ("swizzle", "dpp")),
        _cat("placement", ("atom", "sequential", "xcd_blocked")),
        _cat("prefetch", ("off", "l2_touch")),
        _cat("early", ("off", "entry")),
    ]
    if layers > 1:
        knobs.append(_boolon("parity"))
    knobs += framework_knobs("knobs.")
    knobs.append(
        _cat("knobs.numa_flip_arg", ("", "s_down"), gate="knobs.mailbox.numa_copies", note="calibrated flips")
    )

    def fresh_rule(built, cfg):
        # FrameworkKnobs docstring: fresh_ids=False miscompiles hop-only
        # chains at depth >= 3 (llvm22 assembler error, llvm23 fault).
        if not built.knobs.fresh_ids and built.work == 0 and built.depth >= 3:
            return False, "fresh_ids=False miscompiles hop-only chains at depth >= 3"
        return True, ""

    return DataclassKnobs(
        "toy_chain",
        base,
        knobs,
        variant=f"d{depth}w{work}l{layers}",
        rules=(fresh_rule,),
        excluded={**{f"knobs.{k}": v for k, v in FRAMEWORK_EXCLUDED.items()}, "parity": "layers == 1"},
    )


# --------------------------------------------------------------- fused MoE
def fused_moe(rows: int, seq: bool = False, *, arena_words: int | None = None, numa_area: bool = True) -> DataclassKnobs:
    """``FusedKnobs`` of the fused expert-major MoE at ``rows`` request rows.

    ``arena_words``: the mono LDS arena (``stage_moe.mono_lds_words``; needs
    ATOM) the body must fit; ``None`` skips that rule. ``numa_area``: the
    serving build has a NUMA area (a ``numa`` schedule lever), so the
    ``numa2`` homes are legal."""
    from .. import moe_fused as MF
    from .. import abi as A

    base = MF.FusedKnobs()
    knobs = [
        _cat("map", ("line", "groups")),
        _cat("iod_split", (1, 0)),
        _ord("sh_ctas", (8, 12, 16, 20, 24, 32)),
        _ord("sh_base", (148, 152, 156, 160, 192, 224, 240)),
        _cat("dn_issue", ("with_gu", "after_epi", "after_gu", "pipe", "lds")),
        _ord("dn_inflight", tuple(range(0, 9))),
        _ord("dn_lds", (1, 2, 4, 8)),
        _cat("combine_base", (0, -1)),
        _cat("extras", ("first", "late")),
        _cat("route_gate", ("none", "pre_poll")),
        _cat("route_pause", ("", "s_nop", "s_sleep1", "none")),
        _cat("tw_at", ("late", "route", "epi")),
        _cat("line_impl", ("rt", "const")),
        _cat("slots", ("table", "arith", "wave")),
        _cat("w_cpol", (18, 0, 1, 2, 3, 16, 17, 19)),
        _boolon("a_pre"),
        _boolon("gu_interleave"),
        _cat("gu_apipe", (1, 0)),
        _cat("m3_ept", (8, 4, 2)),  # 2: probe P15 (the router over every CTA; rows > 1 with m3_rows = rows)
        _cat("guard", ("none", "loads")),
        _cat("xchg_pre", ("hint", "none")),
        _cat("router_knobs", ("fused", "mono")),
        _cat("combine", ("task", "sh", "last")),
        _cat("spec", ("off", "prev")),
        _ord("spec_loads", (4, 8, 16, 32, 64), gate="spec"),
        _cat("numa", (0, 1), structural=True),
        _cat("lg_copy", ("canon", "numa2")),
        _cat("xchg_home", ("single", "numa2")),
        _cat("ys_home", ("single", "numa2")),
        _cat("line_order", ("slot", "router_dn")),
        _ord("sh_join", (0, 1, 2, 4, 6, 8)),
    ] + mailbox_knobs("mailbox.", impl_default="ir")[:2]
    # probe P15 (the N3b router form; lost alone on the served base and the quick-track pick, kept for the combination
    # lookback, PLAN 13): the per-wave redundant top-10; P10's m3_rows. Appended last: earlier knobs keep their positions.
    knobs.append(_cat("route", ("cta", "wave")))
    if rows > 1:
        knobs.append(_cat("m3_rows", (0, rows)))

    def normalize(cfg):
        if cfg.get("dn_issue", "with_gu") != "lds":
            cfg.pop("dn_lds", None)
        return cfg

    def plan_rule(built, cfg):
        if not numa_area and built.uses_area:
            return False, "numa2 homes need a NUMA area (a numa schedule lever)"
        try:
            plan = MF.FusedPlan(rows, A.EXPERTS, built)
            words = MF.lds_layout(plan)[2]
        except (ValueError, AssertionError, KeyError) as e:
            return False, f"plan: {e}"
        if arena_words is not None and words > arena_words:
            return False, f"LDS arena {words} words > the mono arena's {arena_words}"
        return True, ""

    return DataclassKnobs(
        "fused_moe",
        base,
        knobs,
        variant=f"r{rows}{'s' if seq else ''}",
        rules=(plan_rule,),
        normalize=normalize,
        excluded={
            "router": "structural problem choice (split suffix behind a FlyDSL prefix)",
            "w_null": "measurement only (WaveScope W0)",
            "phase_stamps": "measurement only",
            "mutate": "test-only must-fail mutants (combine_swap, xtag_or, router_kdrop, router_ksum, route_k11)",
            "spec_keep": "test only (oracle bench)",
            "numa_hdr": "layout constant",
        },
    )


# ---------------------------------------------------------------- MoE port
def moe_port(rows: int) -> DataclassKnobs:
    from .. import moe as MO

    knobs = [
        _bool("dedupe"),
        _cat("union_order", ("id", "first")),
        _cat("m4_waves", (8, 4)),
        _cat("m4_map", ("mono", "balanced")),
        _cat("m3_ept", (8, 4)),
        _cat("w_cpol", (0, 1, 2, 3, 16, 17, 18, 19)),
        _bool("shared_first"),
        _cat("m5_issue", ("mono", "early", "late")),
        _cat("m5_ranks", ("lds", "pre")),
        _bool("a_pre"),
        _cat("m4_guard", ("branch", "none")),
        _bool("sched_issue"),
        _bool("m4_interleave"),
        _cat("m4_apipe", (0, 1)),
        _cat("slots", ("table", "arith")),
        _cat("m5_extras", ("after", "first", "m4")),
        _bool("shared_early"),
        _cat("route_pause", ("", "s_nop", "s_sleep1", "none")),
        _cat("route_gate", ("none", "pre_poll")),
        _cat("m4_weights", ("route", "late")),
    ] + mailbox_knobs()[:2]
    return DataclassKnobs(
        "moe_port",
        MO.MoeKnobs(),
        knobs,
        variant=f"r{rows}",
        excluded={
            "stamps": "measurement only",
            "compute": "MFMA route (op_sel variants are numerics-equal; not a lever yet)",
            "cvt": "convert route (fixed)",
            "m3_base": "placement: covered by the schedule's place.* lever",
            "m4_base": "placement: covered by the schedule's place.* lever",
            "m5_base": "placement: covered by the schedule's place.* lever",
            "m4_j": "derived by the balanced map",
            "m3_rows": "derived from rows",
        },
    )


# ---------------------------------------------------------------------- B3
B3_VARIANTS = ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s", "r4g2")


def b3(variant: str = "r1", *, numa: bool = False) -> DataclassKnobs:
    """``B3Knobs`` (gated delta rule). ``numa``: the base build addresses the
    NUMA area (served with the ``numa=1`` schedule lever)."""
    from .. import stage_b3 as SB

    base = SB.B3Knobs(numa=1) if numa else SB.B3Knobs()
    knobs = [
        _cat("split", SB.SPLITS, structural=True),
        _cat("head_map", SB.HEAD_MAPS),
        _cat("wait", SB.WAITS),
        _cat("stores", SB.STORES),
        _bool("state_nt"),
        _cat("barrier", SB.BARRIERS),
        _cat("shfl4", SB.SHFL4),
        _cat("exp", SB.EXPS, numerics=TOLERANCE, note="hw exp: not bitwise"),
        _cat("cvt", SB.CVTS),
        _cat("state_issue", SB.STATE_ISSUES),
        _cat("norms", SB.NORMS),
        _cat("core_put", SB.CORE_PUTS),
        _cat("xchg", SB.XCHGS),
        _cat("table", SB.TABLES),
        _cat("hoist", SB.HOISTS),
        _cat("pre", SB.PRES),
    ] + mailbox_knobs()[:2]
    return DataclassKnobs(
        "b3",
        base,
        knobs,
        variant=variant + ("_numa" if numa else ""),
        excluded={"numa": "fixed per serving build (the schedule's numa lever)", "numa_hdr": "layout constant"},
    )


# ------------------------------------------------------------- stage fn ABI
def stage_fn() -> DataclassKnobs:
    from ..stage_fn import StageKnobs

    knobs = [
        _cat("mailbox_impl", ("asm", "ir")),
        _cat("xshfl", ("dpp2", "dpp", "lds")),
        _cat("tbl", ("smem", "vmem")),
    ]
    return DataclassKnobs(
        "stage_fn", StageKnobs(), knobs, excluded={"fn": "device-function emission (inline=always is required)"}
    )


# --------------------------------------------------------------- MXFP4 GEMV
def mx_gemv(n: int, k: int, *, rows_max: int = 4, epilogue: str = "fused", act: str = "bf16") -> DataclassKnobs:
    """The MXFP4 decode GEMV (lm_head / draft heads) at one shape."""
    from ... import mx_decode_gemv as MX

    base = MX.MxDecodeGemvSpec(n=n, k=k, rows_max=rows_max, epilogue=epilogue, act=act)
    acts = (act,) + tuple(a for a in MX.ACTS if a != act)
    knobs = [
        _cat("act", acts, numerics=TOLERANCE, note="activation format (different numerics class)"),
        _cat("korder", MX.KORDERS),
        _cat("x_src", MX.X_SRCS),
        _ord("waves", (1, 2, 4, 8, 16)),
        _ord("nt", (1, 2, 4, 8)),
        _ord("dg", (1, 2, 3, 4)),
        _ord("ctas", sorted({64, 128, 192, 243, 256, MX.balanced_ctas(n)})),
        _cat("cta_map", MX.CTA_MAPS),
        _cat("cpol", tuple(MX.CPOLS)),
        _ord("xpad", (0, 16, 32, 64)),
        _ord("waves_per_eu", (0, 1, 2, 4)),
        _ord("setprio", (0, 1, 2, 3)),
        _cat("sched", MX.SCHEDS),
        _cat("sload", ("x4", "bcast")),
        _cat("wload", MX.WLOADS),
        _bool("smask"),
        _ord("ksplit", (1, 2, 4, 8)),
        _cat("pro", MX.PROS),
        _cat("fsync", ("counter64", "counter", "epoch")),
    ]

    def valid(built, cfg):
        return MX.is_valid_spec(built)

    return DataclassKnobs(
        "mx_gemv",
        base,
        knobs,
        variant=f"n{n}k{k}r{rows_max}{epilogue}",
        rules=(valid,),
        excluded={"n": "problem", "k": "problem", "rows_max": "problem", "epilogue": "problem"},
    )


# --------------------------------------------------------------------- CCL
def ccl(op: str = "allreduce") -> DataclassKnobs:
    from ..ccl import layout as CL

    base = CL.CclKnobs()
    knobs = [
        _cat("form", (base.form,) + tuple(f for f in CL.FORMS if f != base.form), structural=True),
        _cat("ar", CL.ARS),
        _ord("slots", (2, 3, 4, 6, 8)),
        _cat("homing", CL.HOMINGS),
        _cat("push_cpol", CL.PUSH_CPOLS),
        _cat("pause", CL.PAUSES),
        _ord("waves", (1, 2, 4, 8, 16)),
        _cat("carry", ("auto", "mem")),
        _cat("poll_cpol", CL.POLL_CPOLS),
        _cat("sync", CL.SYNCS),
        _cat("gsync", CL.GSYNCS, gate="sync"),
        _ord("unit_kib", CL.UNIT_KIBS),
        _cat("ar_dtype", CL.AR_DTYPES, numerics=TOLERANCE, note="bf16 narrows f32 data"),
        _cat("publish", CL.PUBLISHES),
        _bool("self_local"),
    ]
    return DataclassKnobs(
        "ccl",
        base,
        knobs,
        variant=op,
        excluded={
            "header": "safety (launch-header exchange stays on in served builds)",
            "hooks": "test build",
            "mutant": "must-fail mutants",
            "diag": "diagnostics",
        },
    )


# ------------------------------------------------------- ATOM schedule levers
_HOLDS = (0, 16, 32, 64, 96, 128, 192)


def sched(kind: str = "delta", variant: str = "r1", *, validate=None) -> LeverStringKnobs:
    """ATOM's schedule levers (``ATOM_Q38_SCHED`` group ``<kind>.<variant>:``)
    for one layer kind and build variant. ``validate``: ATOM's
    ``sched.parse_levers`` when ATOM is importable."""
    from ..atom_layout import ORDERS

    stages = ORDERS[kind]
    narrow = not variant.startswith("r16") and not variant.startswith("w")
    knobs = []
    if narrow:
        knobs.append(_cat("numa", (0, 1, 2, 3), structural=True, note="2/3: ablations (copy 0 / remote)"))
    knobs.append(Knob("head_sleep", (0, 64, 128, 192), ORDINAL))
    for s in stages:
        if s != "a1":
            knobs.append(_ord(f"hold.{s}", _HOLDS))
    knobs.append(_ord("pause", (0, 1, 4)))
    knobs.append(_ord("pipe", (0, 1, 2, 3)))
    for s in stages:
        knobs.append(_ord(f"prio.{s}", (0, 2)))
    if kind.startswith("delta") and narrow:
        knobs.append(_cat("b3pf", (0, 1)))
    if narrow:
        for s in stages:
            if s not in ("a1", "a2", "m5"):
                knobs.append(_cat(f"dyn.{s}", (0, 1)))
    return LeverStringKnobs(
        "sched",
        knobs,
        style="kv",
        base={"head_sleep": 64},
        validate=validate,
        variant=f"{kind}.{variant}",
        excluded={
            "place.<stage>": "placement overrides: no legal set per stage yet (lost in the ledger); add per track",
            "mutate.<stage>": "test-only must-fail mutant (the A/B gate's)",
        },
    )


# ------------------------------------------------------ Rust direct compiler
DIRECT_LEVERS = ("storewait", "nopause", "nofence", "dppmirror", "dppcombine", "ldsquads", "vmemfirst")


def direct() -> LeverStringKnobs:
    """The code-object backend and the Rust direct compiler's levers (each
    read only with ``backend=direct``)."""
    knobs = [_cat("backend", ("comgr", "direct"), structural=True)]
    knobs += [_bool(name, gate="backend") for name in DIRECT_LEVERS]
    return LeverStringKnobs("direct", knobs, style="kv")


def direct_env(knobs: Mapping[str, object]) -> dict:
    """The environment of a :func:`direct` config (``ROCKE_CODEOBJ_BACKEND``,
    ``ROCKE_DIRECT_LEVERS``); empty for the COMGR default. The evaluator
    puts it in its cache fingerprint (the engine's own environment hash
    does not see these variables)."""
    if knobs.get("backend", "comgr") != "direct":
        return {}
    flags = [n for n in DIRECT_LEVERS if knobs.get(n)]
    env = {"ROCKE_CODEOBJ_BACKEND": "direct"}
    if flags:
        env["ROCKE_DIRECT_LEVERS"] = "+".join(flags)
    return env


# ----------------------------------------------- accuracy-gated levers (N*)
@dataclass(frozen=True)
class NLever:
    """An accuracy-gated lever of DESIGN rev3 12.3. ``owner``: the adapter
    name it joins; ``path`` / ``values``: its knob (values[0] = off) once the
    field exists; ``gate``: the tolerance ladder steps; ``mutants``: the
    must-fail mutants of 12.4."""

    id: str
    owner: str
    path: str
    values: tuple
    what: str
    gate: str = "T0-T2"
    mutants: tuple = ()


N_LEVERS: tuple[NLever, ...] = (
    NLever("N1", "framework", "hc_split", ("off", "ksplit", "reassoc", "producer"), "HC chain K-split / norm-after-GEMV / producer partial-down",
           "T0-T2", ("dropped partial", "doubled partial", "wrong rrms")),
    NLever("N2", "b3", "out_proj", ("c1", "producer"), "out_proj K-split into B3 / Q7", "T0-T2", ("dropped partial",)),
    NLever("N3", "fused_moe", "router_mfma", ("off", "m2_epilogue", "nsplit"), "MFMA router + on-device top-k", "T0 flips, T2"),
    NLever("N4", "framework", "qsa_dense", ("off", "dense", "splitkv"), "QSA dense under budget / split-KV online softmax", "T0, T2, T4"),
    NLever("N5", "b3", "chunkwise", (False, True), "chunkwise (WY/UT) gated delta rule at verify rows", "T0 state, T2, T4"),
    NLever("N6", "framework", "gemv_reblock", ("off", "wave_rows", "ksplit8"), "re-blocked dense GEMVs (B1, Q1, P1)", "T0-T2"),
    NLever("N7", "framework", "act_format", ("mxfp8", "bf16"), "bf16 hand-off activations", "T0-T4"),
    NLever("N8", "mx_gemv", "vocab_subset", (0, 16384, 32768), "vocab-pruned draft head", "T3"),
    NLever("N10", "framework", "q3", ("f64", "mfma"), "MFMA QSA index scores", "T0, T4"),
    NLever("N11", "b3", "state_dtype", ("f32", "bf16"), "bf16 recurrent-state storage", "T0, T4"),
)
"""N9 is booked inside N1 / G3 and has no knob of its own."""


def attach_n_levers(adapter: DataclassKnobs) -> DataclassKnobs:
    """``adapter`` plus every registered N lever of its owner whose field
    exists on the base object (as a ``tolerance`` knob). Returns the same
    adapter when none exists yet."""
    extra = []
    have = {k.path for k in adapter.knobs}
    for lv in N_LEVERS:
        if lv.owner != adapter.name or lv.path in have:
            continue
        if hasattr(adapter.base, lv.path) and getattr(adapter.base, lv.path) == lv.values[0]:
            kind = BOOL if set(lv.values) == {False, True} else CATEGORICAL
            extra.append(Knob(lv.path, lv.values, kind, numerics=TOLERANCE, note=f"{lv.id}: {lv.what}"))
    if not extra:
        return adapter
    return DataclassKnobs(
        adapter.name,
        adapter.base,
        adapter.knobs + tuple(extra),
        variant=adapter.variant,
        arch=adapter.arch,
        rules=adapter.rules,
        normalize=adapter._normalize,
        excluded=adapter.excluded,
    )


# ------------------------------------------------------ variant descriptors
def variant_features(name: str) -> dict:
    """Numeric features of a variant name for transfer (nearest variant):
    ``r4s`` -> rows 4, seq 1; ``r4g2`` -> rows 4, group 2; ``d8w8l1`` ->
    depth 8, work 8, layers 1; ``n248320k2560r4fused`` -> n, k, rows."""
    import re

    out: dict[str, float] = {}
    s = name.split(".")[-1].replace("_numa", "")
    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", s)
    if m:
        out["rows"] = float(m.group(1))
        out["seq"] = 1.0 if m.group(2) else 0.5
        out["group"] = float(m.group(3) or 1)
        out["numa"] = 2.0 if "_numa" in name else 1.0
        return out
    m = re.fullmatch(r"d(\d+)w(\d+)l(\d+)", s)
    if m:
        return {"depth": float(m.group(1)), "work": float(m.group(2)) + 1.0, "layers": float(m.group(3))}
    m = re.fullmatch(r"n(\d+)k(\d+)r(\d+)\w*", s)
    if m:
        return {"n": float(m.group(1)), "k": float(m.group(2)), "rows": float(m.group(3))}
    return out


def all_adapters_cpu() -> list[KnobAdapter]:
    """One adapter per owner at a representative variant (inventory and the
    catalog test: every listed value builds alone or is refused for a
    stated reason)."""
    out: list[KnobAdapter] = [
        framework(),
        toy_chain(8, 8),
        b3("r1"),
        b3("r4s", numa=True),
        moe_port(4),
        stage_fn(),
        mx_gemv(248320, 2560),
        ccl(),
        sched("delta", "r1"),
        sched("qsa", "r4s"),
        direct(),
    ]
    try:
        out.append(fused_moe(1))
        out.append(fused_moe(4, True))
    except Exception:  # noqa: BLE001 - reported by the catalog test
        pass
    return out


def replace_knob(adapter: DataclassKnobs, path: str, **changes) -> DataclassKnobs:
    """``adapter`` with one knob's definition changed (e.g. a narrower
    domain for a quick search)."""
    knobs = tuple(dataclasses.replace(k, **changes) if k.path == path else k for k in adapter.knobs)
    return DataclassKnobs(
        adapter.name, adapter.base, knobs, variant=adapter.variant, arch=adapter.arch,
        rules=adapter.rules, normalize=adapter._normalize, excluded=adapter.excluded,
    )  # fmt: skip


def restrict(adapter: DataclassKnobs, paths: Sequence[str]) -> DataclassKnobs:
    """``adapter`` limited to ``paths`` (gates kept only when inside)."""
    keep = [k for k in adapter.knobs if k.path in paths]
    keep = [dataclasses.replace(k, gate=k.gate if k.gate in paths else None) for k in keep]
    return DataclassKnobs(
        adapter.name, adapter.base, keep, variant=adapter.variant, arch=adapter.arch,
        rules=adapter.rules, normalize=adapter._normalize, excluded=adapter.excluded,
    )  # fmt: skip
