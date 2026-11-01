# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bytes model of one large-batch decode step of Qwen3.8-Flash-Next, per rank, per stage (WP-L0c of ``DESIGN-LB.md``).

Host data only (no IR, no device). One decode step at K=0: every one of ``rows`` rows emits one token. For each stage
of the LB stage list (``DESIGN-LB.md`` 4.2: S1..S10 per GDN layer, the QSA and PLE replacements, the pass end) the
model counts the bytes that must cross HBM on one rank:

* ``weight``: weight bytes the stage reads (each weight read once per step; at large batch every row shares one read);
* ``state``: recurrent / conv state read + written (GDN: ``[v, k]`` per (row, value head); conv: the window);
* ``kv``: QSA KV cache read (the sparse-attended tokens: all of them up to the indexer budget, then the budget) plus
  the compressed indexer keys scanned past the budget, plus the per-row appends;
* ``scratch``: MoE f32 partials written and read back (MALL-resident at these sizes; reported, not in the floor).

``hbm = weight + state + kv`` is the stage's floor traffic. Routed experts: ``U`` active of ``E`` per layer, by default
``E (1 - (1 - k/E)^rows)`` (independent uniform top-k routing, FlyTape's random-weight workload); a measured ``U`` can
be passed instead.

Precision regimes (``DESIGN-LB.md`` 0.3 and the BASELINE caveats):

* ``matched``    FlyTape's ``mxfp4`` preset: routed + shared experts MXFP4; dense projections, HC mixers, router,
                 ``ple_proj`` and the head bf16; conv state bf16 with 4 slots per channel (FlyTape's layout).
* ``checkpoint`` the LB kernel at ATOM's numerics: every dense linear MXFP4 (W4A8), the HC mix weights and
                 ``lm_head`` stored MXFP4 losslessly (the checkpoint keeps them bf16 but MXFP4-rounded), router bf16,
                 experts MXFP4; conv window 3 slots bf16 (ATOM's ``[3 + k, conv_dim]`` at k = 0).
* ``atom``       what ATOM's native path reads: as ``checkpoint`` but the HC mix weights and ``lm_head`` bf16.

``state_bytes`` (4 = f32, 2 = the ``state=bf16`` lever N11) sets the GDN state element size. MXFP4 = 4 bits + one E8M0
scale per 32 elements = 0.53125 B per weight. Tensor parallelism follows ``DESIGN-LB.md`` 7.1 (FlyTape's sharding):
GDN / QSA heads, MoE intermediate split (MXFP4 320 + 320), head vocab-parallel, everything else replicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MXFP4_BPW = 0.5 + 1.0 / 32.0  # 4-bit element + one E8M0 byte per 32 elements
BF16_BPW = 2.0
FP8_BPW = 1.0
REGIMES = ("matched", "checkpoint", "atom")


@dataclass(frozen=True)
class Shape:
    """Qwen3.8-Flash-Next text decoder (``F/models/qwen4/config.py`` defaults = the released checkpoint)."""

    hidden: int = 2560
    layers: int = 48
    full_attention_interval: int = 4
    vocab: int = 248320
    hc_count: int = 4
    hc_lowrank: int = 320
    gr_down_rows: int = 324  # down rows + 4 inject rows
    gdn_key_heads: int = 16
    gdn_value_heads: int = 48
    gdn_head_dim: int = 128
    conv_kernel: int = 4
    qsa_heads: int = 24
    qsa_kv_heads: int = 2
    qsa_head_dim: int = 256
    indexer_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress: int = 4
    indexer_key_bytes: int = 128 + 4  # FP8 E4M3 compressed key + one scale word
    experts: int = 512
    topk: int = 10
    moe_inter: int = 640
    shared_inter: int = 640
    ple_layer: int = 1  # zero-based
    ngram_heads: int = 16
    ngram_head_dim: int = 160
    ple_proj_rows: int = 4 * 2560 + 2560  # key_proj (hc_hidden) + value_proj (hidden)

    @property
    def hc_hidden(self) -> int:
        return self.hc_count * self.hidden

    @property
    def num_qsa(self) -> int:
        return self.layers // self.full_attention_interval

    @property
    def num_gdn(self) -> int:
        return self.layers - self.num_qsa

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def qsa_index_rows(self) -> int:
        return (self.indexer_heads + 1) * self.indexer_head_dim


QWEN38 = Shape()


@dataclass
class Stage:
    """Bytes of one stage per occurrence (one layer) and how many occurrences a step has."""

    name: str
    family: str  # hc, gdn, qsa, router, moe, ple, head, embed
    count: int
    weight: float = 0.0
    state: float = 0.0
    kv: float = 0.0
    scratch: float = 0.0
    kind: str = "read"  # bandwidth class for floors: read (stream), rw (state read + write), seg (expert segments)
    seg_bytes: float = 0.0  # kind == "seg": bytes of one active expert's segment
    segs: float = 0.0  # kind == "seg": active segments per occurrence (U)
    note: str = ""

    @property
    def hbm(self) -> float:
        return self.weight + self.state + self.kv

    @property
    def hbm_total(self) -> float:
        return self.count * self.hbm


def active_experts(rows: int, experts: int = 512, topk: int = 10) -> float:
    """Expected distinct experts hit by ``rows`` tokens routed top-``topk`` uniformly without replacement."""
    if rows <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - topk / experts) ** rows)


def _div(n: int, tp: int, what: str) -> int:
    if n % tp:
        raise ValueError(f"{what} ({n}) does not divide across {tp} ranks")
    return n // tp


@dataclass
class StepBytes:
    tp: int
    rows: int
    ctx: int
    regime: str
    state_bytes: int
    u: float
    stages: list[Stage] = field(default_factory=list)

    def total(self, attr: str = "hbm") -> float:
        return sum(s.count * getattr(s, attr) for s in self.stages)

    def by_family(self, attr: str = "hbm") -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.stages:
            out[s.family] = out.get(s.family, 0.0) + s.count * getattr(s, attr)
        return out

    def categories(self) -> dict[str, float]:
        """The DESIGN-LB section 3 split: routed experts, GDN state r+w, dense (everything else incl. KV / conv)."""
        exp = sum(s.count * s.weight for s in self.stages if s.name == "experts")
        gstate = sum(s.count * s.state for s in self.stages if s.name == "gdn_recur")
        return {
            "experts": exp,
            "gdn_state": gstate,
            "dense": self.total() - exp - gstate,
        }


def step_bytes(
    tp: int,
    rows: int,
    ctx: int = 128,
    regime: str = "checkpoint",
    state_bytes: int = 4,
    u: float | None = None,
    shape: Shape = QWEN38,
) -> StepBytes:
    """Per-rank, per-stage HBM bytes of one decode step at ``rows`` rows, context ``ctx`` tokens, ``tp`` ranks."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} not in {REGIMES}")
    if tp not in (1, 2, 4):
        raise ValueError("tp must be 1, 2 or 4")
    c = shape
    H = c.hidden
    dense = BF16_BPW if regime == "matched" else MXFP4_BPW
    hcw = BF16_BPW if regime in ("matched", "atom") else MXFP4_BPW
    headw = hcw
    router = BF16_BPW
    expw = MXFP4_BPW
    conv_slots = 4 if regime == "matched" else 3
    lk, lv = (
        _div(c.gdn_key_heads, tp, "GDN key heads"),
        _div(c.gdn_value_heads, tp, "GDN value heads"),
    )
    lh, lkv = _div(c.qsa_heads, tp, "QSA heads"), max(1, c.qsa_kv_heads // tp)
    inter = _div(c.moe_inter, tp, "MoE intermediate")
    sinter = _div(c.shared_inter, tp, "shared intermediate")
    dk = c.gdn_head_dim
    conv_local = 2 * lk * dk + lv * dk
    gdn_in_rows = conv_local + lv * dk + 2 * lv
    qsa_in_rows = lh * 2 * c.qsa_head_dim + 2 * lkv * c.qsa_head_dim + c.qsa_index_rows
    if u is None:
        u = active_experts(rows, c.experts, c.topk)
    expert_bytes = (2 * inter * H + H * inter) * expw
    shared_bytes = (2 * sinter * H + H * sinter) * expw
    hc_down = c.gr_down_rows * c.hc_hidden * hcw
    hc_up = c.hc_hidden * c.hc_lowrank * hcw
    out = StepBytes(tp, rows, ctx, regime, state_bytes, u)
    S = out.stages
    G, Q = c.num_gdn, c.num_qsa
    L = c.layers
    # per-layer stages shared by both kinds
    for pos in ("attn", "mlp"):
        S.append(
            Stage(
                f"hc_{pos}_down",
                "hc",
                L,
                weight=hc_down,
                note="S1/S6: gr_down 324 x 10240",
            )
        )
        S.append(
            Stage(
                f"hc_{pos}_up", "hc", L, weight=hc_up, note="S2/S7: gr_up 10240 x 320"
            )
        )
    # GDN layers
    S.append(
        Stage(
            "gdn_in",
            "gdn",
            G,
            weight=gdn_in_rows * H * dense,
            state=rows * conv_local * conv_slots * 2 * 2,
            kind="read",
            note=f"S3: {gdn_in_rows} x {H}; conv window r+w",
        )
    )
    S.append(
        Stage(
            "gdn_recur",
            "gdn",
            G,
            state=rows * lv * dk * dk * state_bytes * 2,
            kind="rw",
            note="S4: [v, k] state per (row, value head), read + write",
        )
    )
    S.append(
        Stage(
            "gdn_out", "gdn", G, weight=H * lv * dk * dense, note=f"S5: {H} x {lv * dk}"
        )
    )
    # QSA layers
    attended = min(ctx, c.indexer_budget)
    kv_tok = lkv * c.qsa_head_dim * 2 * 2  # k and v, bf16
    scan = (
        (ctx // c.indexer_compress) * c.indexer_key_bytes
        if ctx > c.indexer_budget
        else 0
    )
    S.append(
        Stage(
            "qsa_in",
            "qsa",
            Q,
            weight=qsa_in_rows * H * dense,
            note=f"{qsa_in_rows} x {H}",
        )
    )
    S.append(
        Stage(
            "qsa_core",
            "qsa",
            Q,
            kv=rows * (attended * kv_tok + scan + kv_tok + c.indexer_key_bytes),
            note="KV of the attended tokens + indexer key scan past the budget + append",
        )
    )
    S.append(
        Stage(
            "qsa_out",
            "qsa",
            Q,
            weight=H * lh * c.qsa_head_dim * dense,
            note=f"{H} x {lh * c.qsa_head_dim}",
        )
    )
    # MoE (every layer)
    S.append(
        Stage(
            "router",
            "router",
            L,
            weight=(c.experts + 1) * H * router,
            note="S8: 513 x 2560 bf16",
        )
    )
    S.append(
        Stage(
            "experts",
            "moe",
            L,
            weight=u * expert_bytes,
            kind="seg",
            seg_bytes=expert_bytes,
            segs=u,
            scratch=2 * rows * (c.topk + 1) * H * 4,
            note="S9: U active experts (gate_up + down)",
        )
    )
    S.append(
        Stage(
            "shared_expert",
            "moe",
            L,
            weight=shared_bytes,
            note="S9 shared expert (dense units)",
        )
    )
    S.append(
        Stage(
            "combine",
            "moe",
            L,
            note="S10: partials only (scratch counted under experts)",
        )
    )
    # PLE (one layer)
    S.append(
        Stage(
            "ple",
            "ple",
            1,
            weight=c.ple_proj_rows * H * dense,
            kv=rows * c.ngram_heads * c.ngram_head_dim * FP8_BPW,
            note="ple_proj 12800 x 2560 + n-gram rows gathered",
        )
    )
    # pass end
    S.append(Stage("final_mix", "head", 1, weight=hc_down + hc_up, note="final HC mix"))
    S.append(
        Stage(
            "lm_head",
            "head",
            1,
            weight=(-(-c.vocab // tp)) * H * headw,
            note="vocab-parallel head",
        )
    )
    S.append(
        Stage("embed", "embed", 1, weight=rows * H * BF16_BPW, note="embedding rows")
    )
    return out


def design_lb_table() -> dict[tuple, dict[str, float]]:
    """DESIGN-LB.md section 3 (p128, random routing): (tp, rows) -> GB per rank. The model's acceptance reference."""
    t = {
        (1, 8): (9.37, 1.81, 2.37, 8.23, 13.6, 19.4),
        (1, 16): (17.4, 3.62, 2.37, 8.23, 23.4, 29.3),
        (1, 32): (30.0, 7.25, 2.37, 8.23, 39.8, 45.6),
        (1, 64): (46.0, 14.5, 2.37, 8.23, 63.1, 69.0),
        (1, 128): (59.0, 29.0, 2.37, 8.23, 90.9, 96.8),
        (2, 8): (4.68, 0.91, 1.43, 4.86, 7.0, 10.5),
        (2, 16): (8.68, 1.81, 1.43, 4.86, 12.0, 15.4),
        (2, 32): (15.0, 3.62, 1.43, 4.86, 20.1, 23.6),
        (2, 64): (23.0, 7.25, 1.43, 4.86, 31.8, 35.2),
        (2, 128): (29.5, 14.5, 1.43, 4.86, 45.7, 49.1),
    }
    keys = ("experts", "gdn_state", "dense_mx", "dense_bf16", "lb_total", "ft_total")
    return {k: dict(zip(keys, v)) for k, v in t.items()}


#: FlyTape's own per-rank estimate at TP2 B128 MXFP4 (``research/flytape.md`` 2b: state 14.5 + experts 29 + dense 4.8).
FLYTAPE_TP2_B128_GB = 48.0


def weights_only(tp: int, regime: str, shape: Shape = QWEN38) -> dict[str, float]:
    """Per-tensor-group weight bytes per occurrence at ``tp`` (the FlyTape research table's per-phase MB, TP1 bf16)."""
    sb = step_bytes(tp, 1, 128, regime, shape=shape)
    return {
        s.name: s.weight
        for s in sb.stages
        if s.weight and s.name not in ("embed", "experts")
    }


def validate(tol_design: float = 0.02, tol_flytape: float = 0.03) -> list[str]:
    """The WP-L0c acceptance checks; returns the list of failures (empty = pass).

    1. Every row of DESIGN-LB section 3 within ``tol_design`` (experts, GDN state, dense MXFP4 / bf16, both totals);
       the dense columns there exclude conv state / KV (p128) so they are compared on weights only.
    2. FlyTape's per-rank TP2 B128 estimate (48 GB = GDN state r+w + routed experts + dense weights) within
       ``tol_flytape`` of the matched regime's same components (the full floor adds the conv window and the KV:
       about 1.2 % more at p128).
    3. Per-tensor weights of FlyTape's research table (TP1 bf16): gdn_in 84.4 MB, gdn_out 31.5, hc down 6.6,
       hc up 6.55, router 2.6, qsa_in 71.4, lm_head 1.27 GB, ple 65.5 MB within 1 %.
    """
    fails: list[str] = []

    def close(a: float, b: float, tol: float) -> bool:
        return abs(a - b) <= tol * abs(b)

    for (tp, rows), ref in design_lb_table().items():
        lb = step_bytes(tp, rows, 128, "checkpoint")
        ft = step_bytes(tp, rows, 128, "matched")
        cat = lb.categories()
        dense_w = lambda sb: sum(
            s.count * s.weight for s in sb.stages if s.name != "experts"
        )
        got = {
            "experts": cat["experts"],
            "gdn_state": cat["gdn_state"],
            "dense_mx": dense_w(lb),
            "dense_bf16": dense_w(ft),
            "lb_total": lb.total(),
            "ft_total": ft.total(),
        }
        for k, v in got.items():
            if not close(v / 1e9, ref[k], tol_design):
                fails.append(
                    f"design tp{tp} B{rows} {k}: model {v / 1e9:.3f} GB vs {ref[k]}"
                )
    sb = step_bytes(2, 128, 128, "matched")
    ft = (
        sum(
            s.count * (s.weight + (s.state if s.name == "gdn_recur" else 0.0))
            for s in sb.stages
        )
        / 1e9
    )
    if not close(ft, FLYTAPE_TP2_B128_GB, tol_flytape):
        fails.append(
            f"flytape tp2 B128: model {ft:.2f} GB vs claim {FLYTAPE_TP2_B128_GB}"
        )
    w = weights_only(1, "matched")
    for name, mb in (
        ("gdn_in", 84.4),
        ("gdn_out", 31.5),
        ("hc_attn_down", 6.6),
        ("hc_attn_up", 6.55),
        ("router", 2.6),
        ("qsa_in", 71.4),
        ("lm_head", 1271.0),
        ("ple", 65.5),
    ):
        if not close(w[name] / 1e6, mb, 0.01 if name != "router" else 0.02):
            fails.append(f"tensor {name}: model {w[name] / 1e6:.2f} MB vs {mb}")
    return fails
