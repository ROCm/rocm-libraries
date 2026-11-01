# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knobs of the true-mono persistent platform (WP1).

Three groups, each a frozen dataclass whose defaults are the plan of record
(``D0.md`` D0.2 / D0.6 / D0.8) and whose :meth:`suffix` names only deviations
from those defaults:

* :class:`PersistKnobs`: how a persistent (looped) layer body is built: P6's
  rules R1-R5 (``args``, ``fresh``, ``loop_carry``, ``machine_licm``,
  ``wgid``), ``inline``, and the launch-level sync defaults of D0.8 (per-CTA
  epoch arrivals, counter stripes, residency ticket form, poll deadline and
  the launch-fatal error word).
* :class:`EdgeKnobs` / :class:`EdgePlan`: one hand-off edge (producer stage ->
  consumer stage): protocol, NUMA polarity, poll implementation, pause,
  deadline, placement and queue (``XCD-SCHEDULING.md`` levers 1, 4, 5, 11 and
  P4 / P5's protocol findings). A plan maps edge names (``"m5>a1"``) to knobs;
  an unnamed edge gets the plan's default.
* :class:`RustScope`: per-stage lever sets for the Rust direct code-object
  backend (D0.6, ``rust-track/INTEGRATION.md`` section 2): one true-mono kernel
  holds B3, MoE and head-side stages whose best lever sets differ (``nofence``
  helps MoE and hurts the head).

The framework (:mod:`..framework`) takes them through new
``FrameworkKnobs`` fields that default to ``None`` / empty: an existing build
is byte-identical (tests).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

__all__ = [
    "ARGS",
    "ARRIVALS",
    "D0_DEFAULTS",
    "EDGE_PLACEMENTS",
    "EDGE_POLARITIES",
    "EDGE_POLLS",
    "EDGE_PROTOCOLS",
    "EDGE_QUEUES",
    "FRESH",
    "INLINES",
    "LEGACY",
    "MLICM",
    "TICKETS",
    "WGIDS",
    "EdgeKnobs",
    "EdgePlan",
    "PersistKnobs",
    "RUST_DIRECT_FAMILIES",
    "RUST_FRAMEWORK_DIRECT_FAMILIES",
    "RustScope",
    "require_launch_fatal",
    "rust_family",
    "rust_scope_backend",
]

INLINES = ("always", "never")
ARGS = ("const", "lds", "kernarg")
FRESH = ("all", "ids", "off")
MLICM = ("off", "on")
WGIDS = ("lds", "sgpr")
ARRIVALS = ("cta", "wave", "xcd")
TICKETS = ("xcd", "one", "off")


@dataclass(frozen=True)
class PersistKnobs:
    """How a persistent layer loop is built (P6 R1-R5, D0.2, D0.8).

    inline: ``always`` (``alwaysinline`` + ``convergent`` from the driver to
      the body to the stage function: rocKE stage emitters are inlined by
      construction) | ``never`` (one call per body; P6: fails the scratch gate
      on every module and loses 1.105-1.130 per layer: a registered, refused
      knob in rocKE, see :meth:`check`).
    args (R1): ``const`` (each stage reloads its arguments with scalar loads
      from a read-only per-layer table) | ``lds`` (the driver copies the
      layer's argument row into LDS once; P6: never wins) | ``kernarg`` (the
      kernel's own arguments: ``layers=1`` builds, the shipped behaviour).
    fresh (R2): ``all`` (fresh workgroup / wave / lane ids and ``s_getreg``
      reads per stage) | ``ids`` (wave and workgroup ids only: the
      framework's ``fresh_ids``) | ``off``.
    loop_carry (R3): the loop carries only the schedule index (True) or also
      held per-layer values (False, a losing knob).
    machine_licm (R4): ``off`` (``-disable-machine-licm`` on the round module;
      free on hardware, P6) | ``on``.
    wgid (R5): ``lds`` (the workgroup id is written to an LDS word at launch
      and read back per stage, so the hardware SGPR dies at entry) | ``sgpr``.
    arrivals: epoch arrivals per ``cta`` (one atomic per CTA from its last
      wave, P9 ``ectr=cta``: 0.884-0.910 by itself) | ``wave`` (every wave
      adds; the losing form) | ``xcd`` (per-XCD counter words).
    stripes: counter words per XCD of a grid sync (P4: 8 per XCD).
    ticket: residency ticket form: ``xcd`` (per-XCD start: 0.25-0.40 of
      FlyTape's) | ``one`` (FlyTape's single START word) | ``off``.
    deadline: poll / sync iteration bound (0 = unbounded polls; P4: bounded
      polls cost 1.04-1.14x, so off by default in ``pass`` / ``round`` and on
      in ``loop`` / debug builds). With a deadline, a timeout aborts the CTA
      (no later arrival, P4 safety rule 1) and sets the launch-fatal error
      word (rule 2), which the host checks after the launch.
    fatal_word: emit the launch-fatal error word on every timeout path.
    kv_bulk: in-launch KV rows: ``sc1`` (P5 default: ``sc1`` stores, ``vmcnt(0)``,
      ready pair; readers ``sc1``-load the launch-written rows) | ``bulk``
      (plain stores + ``buffer_wbl2 sc1`` + ready pair; ``buffer_inv sc1``).
    scalar_reread: a word written in this launch and re-read through the
      scalar path goes through ``s_dcache_inv`` (``dinv``, P5 C-B5) or is
      carried in the mailbox payload (``reg``, the design default).
    xhold: ``xpre`` (next-layer weight issue after a CTA's last own load of
      the current layer) | ``layer``.
    unroll: never unroll the layer loop (D0.8); 1 is the only built value.
    """

    inline: str = "always"
    args: str = "const"
    fresh: str = "all"
    loop_carry: bool = True
    machine_licm: str = "off"
    wgid: str = "lds"
    arrivals: str = "cta"
    stripes: int = 8
    ticket: str = "xcd"
    deadline: int = 0
    fatal_word: bool = True
    kv_bulk: str = "sc1"
    scalar_reread: str = "reg"
    xhold: str = "xpre"
    unroll: int = 1

    def __post_init__(self) -> None:
        for name, allowed in (
            ("inline", INLINES), ("args", ARGS), ("fresh", FRESH), ("machine_licm", MLICM),
            ("wgid", WGIDS), ("arrivals", ARRIVALS), ("ticket", TICKETS),
            ("kv_bulk", ("sc1", "bulk")), ("scalar_reread", ("reg", "dinv")),
            ("xhold", ("xpre", "layer")),
        ):  # fmt: skip
            if getattr(self, name) not in allowed:
                raise ValueError(f"PersistKnobs.{name} must be one of {allowed}")
        if self.stripes not in (1, 2, 4, 8):
            raise ValueError("stripes per XCD must be 1, 2, 4 or 8")
        if self.deadline < 0:
            raise ValueError("deadline must be >= 0")
        if self.unroll != 1:
            raise ValueError("the layer loop is never unrolled (D0.8)")

    def check(self) -> None:
        """Refuse knob values rocKE cannot build (explicitly, never silently).

        ``inline="never"``: rocKE composes stage emitters inline; a call per
        body would need a device-function call path in the kernel lowering,
        which the Python engine does not have. P6 measured the knob in the
        FlyDSL rig (a decisive loss: scratch gate failure on every module,
        1.105-1.130 per layer), so it stays registered and refused here."""
        if self.inline == "never":
            raise NotImplementedError(
                "inline='never' is not built in rocKE (P6: loses 1.105-1.130 per layer and "
                "fails the scratch gate); kept as a registered knob of the FlyDSL rig"
            )

    def suffix(self) -> str:
        d = PersistKnobs()
        short = {
            "inline": "in", "args": "ar", "fresh": "fr", "loop_carry": "lc",
            "machine_licm": "ml", "wgid": "wg", "arrivals": "ea", "stripes": "st",
            "ticket": "tk", "deadline": "dl", "fatal_word": "fw", "kv_bulk": "kv",
            "scalar_reread": "sr", "xhold": "xh", "unroll": "ur",
        }  # fmt: skip
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if v != getattr(d, f.name):
                parts.append(f"{short[f.name]}{int(v) if isinstance(v, bool) else v}")
        return "_".join(parts)


# ---------------------------------------------------------------------- edges

# protocol of one hand-off edge:
#   pairs     tagged (payload, tag) pairs, sc1 (the default; P1 / P4)
#   bulk      plain stores, vmcnt(0), buffer_wbl2 sc1, ready pair; consumer
#             buffer_inv sc1 + plain loads (P4: loses at <= 4 rows)
#   bulk_sc1  sc1 stores, vmcnt(0), ready pair; consumer sc1 loads (P5)
#   xcd_local the intra-XCD protocol: producer and consumer on one XCD, no
#             buffer_wbl2 (P5: same-XCD readers were correct without it);
#             needs an XCD-affine placement (``placement="xcd_affine"``)
#   fence     per-wave agent fences (the losing 2-layer stack; knob only)
EDGE_PROTOCOLS = ("pairs", "bulk", "bulk_sc1", "xcd_local", "fence")
# which NUMA copy a consumer polls (dual-copy mailboxes): local (the calibrated
# home of its IOD; the hybrid's default), remote (P4's microbench winner, which
# did not transfer in situ), none (single copy)
EDGE_POLARITIES = ("local", "remote", "none")
# poll implementation: asm (opaque loop, vmcnt(0) per retry), ir (the
# tagged_poll op), irf (tagged_poll with the reload fix), deadline (asm with an
# iteration bound + error word)
EDGE_POLLS = ("asm", "ir", "irf", "deadline")
# placement of the edge's producer / consumer tasks: atom (the mono's
# schedule), xcd_affine (XCD-SCHEDULING lever 1), island (lever 4: the chain
# on an XCD-local island), hier (lever 5: hierarchical reduction, last level on
# the consumer's XCD)
EDGE_PLACEMENTS = ("atom", "xcd_affine", "island", "hier")
# work queue: static (the TaskTable), xcd_queue (lever 11: XCD-local dynamic
# queue), global (the global dynamic ticket: lost 1.012-1.079; knob only)
EDGE_QUEUES = ("static", "xcd_queue", "global")
EDGE_PAUSES = ("s_nop", "s_sleep1", "none")


@dataclass(frozen=True)
class EdgeKnobs:
    """One hand-off edge's protocol / placement / queue (all default-off
    levers except the plan-of-record defaults: ``pairs``, ``local``, ``asm``).
    ``island``: CUs of the island (0 = not an island edge)."""

    protocol: str = "pairs"
    polarity: str = "local"
    poll: str = "asm"
    pause: str = "s_nop"
    deadline: int = 0
    placement: str = "atom"
    queue: str = "static"
    island: int = 0

    def __post_init__(self) -> None:
        for name, allowed in (
            ("protocol", EDGE_PROTOCOLS), ("polarity", EDGE_POLARITIES), ("poll", EDGE_POLLS),
            ("pause", EDGE_PAUSES), ("placement", EDGE_PLACEMENTS), ("queue", EDGE_QUEUES),
        ):  # fmt: skip
            if getattr(self, name) not in allowed:
                raise ValueError(f"EdgeKnobs.{name} must be one of {allowed}")
        if self.protocol == "xcd_local" and self.placement not in ("xcd_affine", "island"):
            raise ValueError("protocol 'xcd_local' needs an XCD-affine placement (xcd_affine / island)")
        if self.poll == "deadline" and self.deadline <= 0:
            raise ValueError("poll 'deadline' needs deadline > 0")
        if self.placement == "island" and self.island <= 0:
            raise ValueError("placement 'island' needs island > 0 CUs")
        if self.placement != "island" and self.island:
            raise ValueError("island CUs set without placement 'island'")

    @property
    def mailbox_impl(self) -> str:
        """The ``MailboxKnobs.impl`` the poll needs (``deadline`` polls are asm)."""
        return "asm" if self.poll in ("asm", "deadline") else self.poll

    @property
    def needs_release(self) -> bool:
        """Whether the producer must write back its L2 (``buffer_wbl2 sc1``)."""
        return self.protocol in ("bulk", "fence")

    def suffix(self) -> str:
        d = EdgeKnobs()
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if v != getattr(d, f.name):
                parts.append(f"{f.name[:2]}{v}")
        return "".join(parts)


def _edge_name(producer: str, consumer: str) -> str:
    return f"{producer}>{consumer}"


@dataclass(frozen=True)
class EdgePlan:
    """Per-edge knobs: ``edges`` maps ``"producer>consumer"`` (or a region
    name, for edges keyed by the scratch region they use) to
    :class:`EdgeKnobs`; every other edge uses ``default``."""

    default: EdgeKnobs = field(default_factory=EdgeKnobs)
    edges: tuple[tuple[str, EdgeKnobs], ...] = ()

    @staticmethod
    def of(mapping: Mapping[str, EdgeKnobs] | None = None, default: EdgeKnobs | None = None) -> EdgePlan:
        items = tuple(sorted((mapping or {}).items()))
        for k, v in items:
            if not isinstance(v, EdgeKnobs):
                raise TypeError(f"edge {k!r}: expected EdgeKnobs, got {type(v).__name__}")
        return EdgePlan(default or EdgeKnobs(), items)

    def get(self, key: str) -> EdgeKnobs:
        for k, v in self.edges:
            if k == key:
                return v
        return self.default

    def edge(self, producer: str, consumer: str) -> EdgeKnobs:
        """Knobs of the edge ``producer -> consumer`` (exact key, then the
        consumer-side wildcard ``"*>consumer"``, then ``producer>*``, then the
        default)."""
        for key in (_edge_name(producer, consumer), _edge_name("*", consumer), _edge_name(producer, "*")):
            for k, v in self.edges:
                if k == key:
                    return v
        return self.default

    def validate(self, known: set[str] | frozenset[str]) -> None:
        """Every named edge must name known stages / regions (``known``):
        a typo would otherwise silently fall back to the default."""
        for k, _ in self.edges:
            parts = k.split(">")
            names = [p for p in parts if p != "*"]
            if len(parts) > 2 or any(n not in known for n in names):
                raise ValueError(f"edge {k!r} names unknown stages / regions (known: {sorted(known)})")

    def suffix(self) -> str:
        d = EdgeKnobs()
        parts = []
        if self.default != d:
            parts.append("ed" + self.default.suffix())
        for k, v in self.edges:
            s = v.suffix()
            if s:
                parts.append(k.replace(">", "2").replace("*", "x") + s)
        return "_".join(parts)

    @property
    def is_default(self) -> bool:
        return self.default == EdgeKnobs() and all(v == EdgeKnobs() for _, v in self.edges)


# ----------------------------------------------------------------- rust scope

# rust-track/INTEGRATION.md section 1: the measured lever sets
RUST_LEVER_SETS: dict[str, tuple[str, ...]] = {
    "b3": (
        "dppcombine", "dppcombinef", "dppmirror", "ldspairs", "ldsquads", "litreuse", "litshare",
        "minmax", "nopause", "selxor", "storewait", "subadd", "vmemfirst", "wgfence", "xorcmp",
        "scalesep", "dsrebase", "loadorder",
    ),
    "moe_r1": (
        "dppmirror", "ldsquads", "litreuse", "litshare", "minmax", "nofence", "selxor", "subadd",
        "wgfence", "xorcmp", "scalesep", "dsrebase", "loadorder",
    ),
    # fused MoE rows 4: the r1 set measured neutral (G1 1.005 / 1.009): not a direct family (WP-R3 keeps tuning it)
    "moe_r4": (
        "dppmirror", "ldsquads", "litreuse", "litshare", "minmax", "nofence", "selxor", "subadd",
        "wgfence", "xorcmp", "scalesep", "dsrebase", "loadorder",
    ),
    "head": (),
    "none": (),
}  # fmt: skip
# levers measured to hurt a family (never applied there)
RUST_FORBIDDEN: dict[str, tuple[str, ...]] = {"head": ("nofence", "loadorder")}
# on by default inside the direct backend (INTEGRATION.md 5 items 1-2; D0.6); the direct backend itself applies them
# (``rocke.helpers.compile.DIRECT_DEFAULT_LEVERS``, which must equal this: tested)
RUST_BACKEND_DEFAULTS = ("acqfence", "bufoffany")
# D0.6 / WP-R3: the families whose direct objects (with their lever set) won on BOTH rocke nodes under the DESIGN 9.1
# win rule (rust-track LEDGER: B3 r1 / r2s / r3s / r4s / r4 / r4g2 tuned and r1 / r4s mono with the B3 sets; fused MoE
# rows 1 with the MoE r1 set). Direct is the default backend there in NEW standalone true-mono builds only (D0 7.4 m5:
# existing builds stay COMGR): :func:`rust_family`. Framework layer kernels use RUST_FRAMEWORK_DIRECT_FAMILIES below.
# The head and fused MoE rows 4 stay COMGR until a set wins on both nodes (a losing family is still buildable direct:
# ``"direct"``).
RUST_DIRECT_FAMILIES = ("b3", "moe_r1")
# WP-R3 fix round: the families whose stages, hosted in a FRAMEWORK layer kernel, gave direct objects that won on both
# rocke nodes. The standalone WINs above do not carry over: the framework's own code (mailbox polls, the loop driver's
# SALU address code) is unscoped and lost 6-13% direct in the only framework-hosted timing (Q2, every lever set).
# ``FrameworkKnobs(code_object_backend="auto")`` reads this set, so it stays COMGR until a framework-hosted direct
# build of a family (``rt_fw.py``: fw_b3 / fw_moe) wins on both nodes and the family is added here.
RUST_FRAMEWORK_DIRECT_FAMILIES: tuple[str, ...] = ()


@dataclass(frozen=True)
class RustScope:
    """Per-stage direct-backend lever sets (D0.6).

    ``stages`` maps a stage name to a lever set name (``RUST_LEVER_SETS``) or
    an explicit ``+``-joined lever list; ``family`` names the set's family
    for the forbidden-lever check (``"head"``: ``nofence`` / ``loadorder``
    refused). The framework tags every op a scoped stage emits with
    ``attrs["rocke.scope"] = <stage index>`` and the kernel with
    ``attrs["direct_scopes"]``; the LLVM lowering ignores both (COMGR objects
    are unchanged: tested), the direct backend reads them."""

    stages: tuple[tuple[str, str], ...] = ()
    families: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def of(stages: Mapping[str, str], families: Mapping[str, str] | None = None) -> RustScope:
        sc = RustScope(tuple(sorted(stages.items())), tuple(sorted((families or {}).items())))
        sc.levers_by_stage()  # validates
        return sc

    def levers(self, stage: str) -> tuple[str, ...] | None:
        for k, v in self.stages:
            if k == stage:
                ls = RUST_LEVER_SETS.get(v)
                if ls is None:
                    ls = tuple(x for x in v.replace(",", "+").split("+") if x)
                fam = dict(self.families).get(stage, v if v in RUST_LEVER_SETS else "")
                bad = sorted(set(ls) & set(RUST_FORBIDDEN.get(fam, ())))
                if bad:
                    raise ValueError(f"stage {stage!r} ({fam}): levers {bad} are measured losses there")
                return tuple(ls)
        return None

    def levers_by_stage(self) -> dict[str, tuple[str, ...]]:
        return {k: self.levers(k) for k, _ in self.stages}

    @property
    def enabled(self) -> bool:
        return bool(self.stages)

    def family(self, stage: str) -> str:
        """The family of ``stage``'s set: ``families[stage]``, else the set's
        name when it is a named set, else ``""`` (an explicit lever list)."""
        for k, v in self.stages:
            if k == stage:
                return dict(self.families).get(stage, v if v in RUST_LEVER_SETS else "")
        return ""


def rust_scope_backend(scope: RustScope | None, stages, families: tuple[str, ...] | None = None) -> str:
    """D0.6's backend of a framework kernel with ``stages`` (stage names) under
    ``scope``: ``"direct"`` when every stage is scoped to a family of
    ``families`` (default :data:`RUST_FRAMEWORK_DIRECT_FAMILIES`: the families
    whose framework-hosted direct builds won; empty today, so ``"comgr"``),
    else ``"comgr"`` (one stage of a family that has not won, e.g. the head,
    keeps the whole kernel on COMGR)."""
    stages = tuple(stages)
    fams = RUST_FRAMEWORK_DIRECT_FAMILIES if families is None else tuple(families)
    if scope is None or not scope.enabled or not stages:
        return "comgr"
    ok = all(scope.levers(s) is not None and scope.family(s) in fams for s in stages)
    return "direct" if ok else "comgr"


def rust_family(kernel, family: str):
    """D0.6 for a standalone kernel of one family (a new true-mono B3 or fused
    MoE build): when ``family`` is in :data:`RUST_DIRECT_FAMILIES`, mark it for
    the direct backend with the family's set as ``attrs["direct_levers"]``;
    otherwise leave it unchanged (COMGR). Returns ``kernel``. The IR, its LLVM
    lowering and the name do not change (the attribute is stripped by the direct
    backend and never lowered)."""
    if family not in RUST_LEVER_SETS:
        raise ValueError(f"unknown lever-set family {family!r}; one of {sorted(RUST_LEVER_SETS)}")
    if family in RUST_DIRECT_FAMILIES:
        from rocke.helpers.compile import request_code_object_backend

        kernel.attrs["direct_levers"] = "+".join(RUST_LEVER_SETS[family])
        request_code_object_backend(kernel, "direct")
    return kernel


# ------------------------------------------------------------------- presets

# The plan of record (D0.2, D0.8): what a new true-mono persistent build uses.
D0_DEFAULTS = PersistKnobs()
# The shipped framework's behaviour expressed as PersistKnobs (layers=1 graph
# launches: kernel arguments, framework fresh ids, hardware workgroup SGPR).
LEGACY = PersistKnobs(args="kernarg", fresh="ids", machine_licm="on", wgid="sgpr")


def require_launch_fatal(fk) -> None:
    """Path B rule (WP1 fix round): a ``layers=1`` layer kernel whose edges
    cross launches must be able to raise the launch-fatal word, so the host
    can check it after every launch (``sync.check_launch``). ``fk`` (a
    ``FrameworkKnobs``) needs ``err_arg`` and a deadline poll on every edge
    (the default and every named edge). Raises ``ValueError`` otherwise."""
    if not getattr(fk, "err_arg", ""):
        raise ValueError("Path B launch without the launch-fatal word: set FrameworkKnobs.err_arg")
    plan = getattr(fk, "edges", None)
    if plan is None:
        raise ValueError("Path B launch without deadline edges: set FrameworkKnobs.edges (poll='deadline')")
    bad = ([] if plan.default.poll == "deadline" else ["<default>"]) + [k for k, v in plan.edges if v.poll != "deadline"]
    if bad:
        raise ValueError(f"Path B edges without a deadline poll (no launch-fatal word on timeout): {bad}")
