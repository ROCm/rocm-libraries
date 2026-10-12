# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host plan, predicate, workspace and declaration of the unified backward.

* every reason code is reachable, in the declared order (a request failing two
  checks reports the earlier one);
* ``BLOCK_TABLE_NOT_INJECTIVE`` before the atomic page mode is declared on an
  arch, and the request served (atomic page mode) after;
* the workspace formula against a hand table;
* plans: the launch sequence of each page mode, the main grid (direct mode
  covers ``roundup(max_seqlen_kv, page)`` keys), the page class of each page
  size, head packing for short spans, ``g_split > 1`` forcing the atomic
  page mode, kernargs equal to every stage's ABI list;
* every page size that is a multiple of 16 resolves to a legal instance;
* the declaration (version 1, ABI tags, classes, statuses).
"""

from __future__ import annotations

import dataclasses

import pytest

from kernels.common.attention_bwd_unified import (
    ATTN_BWD_UNIFIED_ABI,
    AttnBwdUnifiedSpec,
    validate_attn_bwd_unified_spec,
)
from kernels.common.attention_bwd_unified_plan import (
    FEATURE_GROUPS,
    REASON_CODES,
    TUNED_PAGE_TILES,
    AttnBwdUnifiedPolicy,
    AttnBwdUnifiedRequest,
    attn_bwd_unified_declaration,
    attn_bwd_unified_plan,
    attn_bwd_unified_support,
    attn_bwd_unified_tile_choice,
    attn_bwd_unified_tile_configs,
    attn_bwd_unified_workspace_bytes,
    attn_bwd_unified_workspace_layout,
    unified_stage_params,
)

CDNA = ("gfx942", "gfx950")


def _req(**kw) -> AttnBwdUnifiedRequest:
    base = dict(
        num_seqs=3,
        h_q=8,
        h_kv=2,
        d_qk=128,
        d_v=128,
        total_q=100,
        max_seqlen_q=60,
        max_seqlen_kv=300,
        page=16,
        num_blocks=50,
        max_blocks_per_seq=20,
    )
    base.update(kw)
    return AttnBwdUnifiedRequest(**base)


_ALL = AttnBwdUnifiedPolicy(validated_archs=frozenset(CDNA + ("gfx1151", "gfx1201")))

# One request per reason code (on gfx942 unless the code needs another arch).
CODE_REQUESTS = {
    "ARCH_UNSUPPORTED": ("gfx90a", {}),
    "DROPOUT": ("gfx942", dict(dropout=0.1)),
    "BIAS_IN_BACKWARD": ("gfx942", dict(has_bias=True)),
    "DBIAS": ("gfx942", dict(want_dbias=True)),
    "ALIBI": ("gfx942", dict(alibi=True)),
    "FP8": ("gfx942", dict(fp8=True)),
    "DETERMINISTIC": ("gfx942", dict(deterministic=True)),
    "UNSUPPORTED_FEATURE": ("gfx942", dict(qq_bias=True)),
    "SCALE_DEVICE_TENSOR": ("gfx942", dict(scale_is_device_tensor=True)),
    "DTYPE": ("gfx942", dict(k_dtype="fp16")),
    "OUT_DTYPE": ("gfx942", dict(dq_dtype="fp32")),
    "LSE_FORMAT": ("gfx942", dict(lse_dtype="fp16")),
    "DQK_NE_DV": ("gfx942", dict(d_v=64)),
    "HEAD_DIM_256_BWD": ("gfx942", dict(d_qk=256, d_v=256)),
    "HEAD_DIM": ("gfx942", dict(d_qk=96, d_v=96)),
    "GQA_RATIO": ("gfx942", dict(h_q=6, h_kv=4)),
    "BAND_BOUND": ("gfx942", dict(left_bound=-2)),
    "STRIDE_FORMAT": ("gfx942", dict(strides={"q": (1, 2)})),
    "STRIDE_D": ("gfx942", dict(strides={"q": (1024, 128, 2)})),
    "STRIDE_NEGATIVE": ("gfx942", dict(strides={"o": (-1024, 128, 1)})),
    "OUTPUT_OVERLAP": ("gfx942", dict(strides={"dq": (128, 64, 1)})),
    "STRIDE_ALIGN": ("gfx942", dict(tensor_alignment=1)),
    "INDEX_RANGE": ("gfx942", dict(num_blocks=2**28, page=16)),
    "GRID_LIMIT": ("gfx942", dict(num_seqs=70000)),
    "SEQ_LEN_FORMAT": ("gfx942", dict(seq_lens_dtype="int64")),
    "RAGGED_FORMAT": ("gfx942", dict(cu_seqlens_dtype="int64")),
    "KV_CACHE_LAYOUT": ("gfx942", dict(strides={"dk": (16 * 256 + 64, 256, 128, 1)})),
    "PAGE_SIZE": ("gfx942", dict(page=24, max_blocks_per_seq=20)),
    "BLOCK_TABLE_FORMAT": ("gfx942", dict(block_table_dtype="int64")),
    "SINKS": ("gfx942", dict(sinks=True)),
    "SOFTCAP": ("gfx942", dict(softcap=30.0)),
    "EMPTY": ("gfx942", dict(total_q=0)),
}


@pytest.mark.parametrize("code", sorted(CODE_REQUESTS))
def test_every_code_is_reachable(code):
    arch, kw = CODE_REQUESTS[code]
    v = attn_bwd_unified_support(_req(**kw), arch, policy=_ALL)
    assert (v.ok, v.code) == (False, code), v


def test_policy_codes():
    # not validated: the shipped policy has no hardware run on RDNA
    v = attn_bwd_unified_support(_req(), "gfx1151")
    assert v.code == "ARCH_NOT_VALIDATED"
    # not injective before the atomic page mode is declared, served after
    pol = AttnBwdUnifiedPolicy(declared_groups=frozenset({"direct_pages"}))
    assert attn_bwd_unified_support(_req(), "gfx942", policy=pol).code == (
        "BLOCK_TABLE_NOT_INJECTIVE"
    )
    assert attn_bwd_unified_support(
        _req(block_table_injective=True), "gfx942", policy=pol
    ).ok
    assert attn_bwd_unified_support(_req(), "gfx942").ok
    # injective tables need the direct group
    pol = AttnBwdUnifiedPolicy(declared_groups=frozenset({"atomic_pages"}))
    v = attn_bwd_unified_support(_req(block_table_injective=True), "gfx942", policy=pol)
    assert v.code == "NOT_YET_DECLARED"
    # an instance that failed the resource gate
    ok = attn_bwd_unified_plan(_req(), "gfx942")
    base = ok.specs["main"].kernel_name().rsplit("_", 1)[0]
    pol = AttnBwdUnifiedPolicy(instance_unavailable=frozenset({base}))
    assert attn_bwd_unified_support(_req(), "gfx942", policy=pol).code == (
        "INSTANCE_UNAVAILABLE"
    )


def test_codes_are_covered_and_ordered():
    reached = set(CODE_REQUESTS) | {
        "ARCH_NOT_VALIDATED",
        "BLOCK_TABLE_NOT_INJECTIVE",
        "INSTANCE_UNAVAILABLE",
        "NOT_YET_DECLARED",
    }
    assert reached == set(REASON_CODES)
    # first failure wins: a request built to fail two checks never reports the
    # later one (it reports the earlier, or an even earlier check that the
    # combination trips, e.g. a zero extent that empties a tensor)
    order = [c for c in REASON_CODES if c in CODE_REQUESTS and c != "ARCH_UNSUPPORTED"]
    for i, early in enumerate(order):
        for late in order[i + 1 :]:
            kw = {**CODE_REQUESTS[late][1], **CODE_REQUESTS[early][1]}
            if (
                "strides" in CODE_REQUESTS[late][1]
                and "strides" in CODE_REQUESTS[early][1]
            ):
                kw["strides"] = {
                    **CODE_REQUESTS[late][1]["strides"],
                    **CODE_REQUESTS[early][1]["strides"],
                }
            v = attn_bwd_unified_support(_req(**kw), "gfx942", policy=_ALL)
            assert not v.ok
            assert REASON_CODES.index(v.code) <= REASON_CODES.index(early) or (
                # an empty dQ cannot overlap: the overlap rule skips it
                v.code == "EMPTY"
                and (early, late) == ("OUTPUT_OVERLAP", "EMPTY")
            ), (early, late, v)


def test_workspace_hand_table():
    r = _req()
    # dq 4*100*8*128, lse2 / dsum A(4*800), page workspace 4*(50*16)*2*128 each
    assert attn_bwd_unified_workspace_bytes(r, page_mode="direct") == (
        409600 + 3328 + 3328
    )
    assert attn_bwd_unified_workspace_bytes(r, page_mode="atomic") == (
        409600 + 3328 + 3328 + 2 * 819200
    )
    lay = attn_bwd_unified_workspace_layout(r)
    assert list(lay) == ["WS_DQ", "WS_LSE2", "WS_DSUM", "WS_PDK", "WS_PDV"]
    assert all(off % 256 == 0 for off, _ in lay.values())


@pytest.mark.parametrize("arch", CDNA)
def test_plans_per_mode(arch):
    for injective in (True, False):
        req = _req(block_table_injective=injective, page=128, max_blocks_per_seq=3)
        p = attn_bwd_unified_plan(req, arch)
        stages = [s.stage for s in p.launches]
        main = p.launches[stages.index("main")]
        kn = p.specs["main"].block_n
        if injective:
            assert stages == ["prep", "main", "convert"]
            assert p.runtime["page_mode"] == "direct"
            # roundup(300, 128) = 384 keys
            assert main.grid[0] == -(-384 // kn)
            assert main.pointers["WS_PDK"] is None and main.pointers["DKC"]
        else:
            assert stages == [
                "prep", "page_zero", "main", "convert", "page_convert", "page_convert",
            ]  # fmt: skip
            assert main.grid[0] == -(-300 // kn)
            assert main.pointers["DKC"] is None and main.pointers["WS_PDK"]
            assert p.launches[-1].pointers["DST"] == "dv_cache"
        assert main.grid[1:] == (2, 3)
        for step in p.launches:
            names = [n for n, _ in unified_stage_params(step.stage)]
            assert sorted(list(step.pointers) + list(step.scalars)) == sorted(names)


@pytest.mark.parametrize("arch", CDNA)
def test_page_classes(arch):
    """Rules (2) - (4) of the tile choice (no tuned table)."""
    want = {16: "sub", 32: "sub", 48: "fallback", 64: "multi", 80: "fallback",
            128: "multi", 256: "multi"}  # fmt: skip
    untuned = AttnBwdUnifiedPolicy()
    for page, tag in want.items():
        req = _req(page=page, max_blocks_per_seq=-(-300 // page))
        p = attn_bwd_unified_plan(req, arch, policy=untuned)
        assert p.route_tag == tag, (page, p.route_tag)
        s = p.specs["main"]
        if tag == "sub":
            assert s.sub_page == page
        else:
            assert s.sub_page is None and page % s.block_n == 0
        if tag == "fallback":
            assert (s.waves, s.block_n) == (1, 16)


def test_tuned_page_tiles():
    """Rule (1): every tuned entry names a shipped configuration that serves
    its page, and the shipped policy plans it."""
    for (arch, d, page), name in TUNED_PAGE_TILES.items():
        cfgs = attn_bwd_unified_tile_configs(arch, d, 8)
        assert name in cfgs, (arch, d, page, name)
        cls, knobs = cfgs[name]
        req = _req(d_qk=d, d_v=d, page=page, max_blocks_per_seq=-(-300 // page))
        p = attn_bwd_unified_plan(req, arch)
        s = p.specs["main"]
        want = validate_attn_bwd_unified_spec(
            AttnBwdUnifiedSpec(head_size=d, **knobs), arch
        )
        assert (s.block_n, s.waves, s.sub_page) == (
            want.block_n,
            want.waves,
            want.sub_page,
        )
        assert p.route_tag == ("sub" if cls == "sub" else "multi")


def test_runtime_rules():
    pol = AttnBwdUnifiedPolicy(tuning=lambda req, arch, pc: {"g_split": 2})
    p = attn_bwd_unified_plan(_req(block_table_injective=True), "gfx942", policy=pol)
    assert p.runtime["page_mode"] == "atomic" and p.runtime["g_split"] == 2
    assert p.launches[2].grid[1] == 4
    pol = AttnBwdUnifiedPolicy(tuning=lambda req, arch, pc: {"page_mode": "direct"})
    with pytest.raises(ValueError, match="injective"):
        attn_bwd_unified_plan(_req(), "gfx942", policy=pol)
    pol = AttnBwdUnifiedPolicy(tuning=lambda req, arch, pc: {"lb_order": "mirror"})
    with pytest.raises(ValueError, match="mirror"):
        attn_bwd_unified_plan(_req(), "gfx942", policy=pol)
    pol = AttnBwdUnifiedPolicy(tuning=lambda req, arch, pc: {"bogus": 1})
    with pytest.raises(ValueError, match="unknown knobs"):
        attn_bwd_unified_plan(_req(), "gfx942", policy=pol)
    # short query spans of a GQA group: head packing
    p = attn_bwd_unified_plan(_req(max_seqlen_q=2, d_qk=64, d_v=64), "gfx942")
    assert p.specs["main"].head_pack and p.runtime["pack_heads"] == 4
    p = attn_bwd_unified_plan(_req(), "gfx942")
    assert not p.specs["main"].head_pack and p.runtime["pack_heads"] == 0


@pytest.mark.parametrize("arch", CDNA)
def test_static_scale_follows_the_scale_placement(arch):
    """The main kernel is compiled for the plan's scale placement (the general
    family's waste rule: unit multipliers are not applied), a hook may pin
    "runtime", any other pin is refused, and every multiplier the kernel does
    not apply is 1.0."""
    from kernels.common.attention_bwd import static_unit_mults
    from kernels.common.attention_bwd_plan import SCALE_PLACEMENTS

    for injective in (True, False):
        for placement in SCALE_PLACEMENTS:
            pol = AttnBwdUnifiedPolicy(
                tuning=lambda req, a, pc, p=placement: {"scale_placement": p}
            )
            p = attn_bwd_unified_plan(
                _req(block_table_injective=injective), arch, policy=pol
            )
            s = p.specs["main"]
            assert s.static_scale == placement
            main = next(x for x in p.launches if x.stage == "main")
            unit = static_unit_mults(s.static_scale, s.general_spec().dkv_mode)
            assert all(main.scalars[n] == 1.0 for n in unit), (placement, unit)
        pin = AttnBwdUnifiedPolicy(
            tuning=lambda req, a, pc: {"static_scale": "runtime"}
        )
        p = attn_bwd_unified_plan(
            _req(block_table_injective=injective), arch, policy=pin
        )
        assert p.specs["main"].static_scale == "runtime"
    bad = AttnBwdUnifiedPolicy(tuning=lambda req, a, pc: {"static_scale": "convert"})
    with pytest.raises(ValueError, match="does not match"):
        attn_bwd_unified_plan(_req(), "gfx942", policy=bad)
    # the default placement keeps the default (unsalted) kernel name
    plain = attn_bwd_unified_plan(_req(), "gfx942").specs["main"]
    assert ("static_scale", "fold_ds") not in plain.non_default_fields()


@pytest.mark.parametrize(
    "kw",
    [
        dict(xcd_remap=True),
        dict(lb_interleave=True),
        dict(ptr_noalias="workspace"),
        dict(dkv_slab_hoist=True),
        dict(lb_mirror=True),
    ],
)
def test_general_shell_knobs_are_refused(kw):
    """Knobs of the general main-kernel shell (CTA decode, kernarg attributes,
    strided dK / dV slabs) are not built by the unified shell: only their
    default is legal."""
    with pytest.raises(ValueError, match="not built for the unified family"):
        validate_attn_bwd_unified_spec(AttnBwdUnifiedSpec(head_size=64, **kw), "gfx942")


def test_the_general_prefetch_ring_is_refused():
    """The Q / dO prefetch ring is built by the general main-kernel shell and
    handed to the kv-tile body; the unified shell refuses it. The barrier
    merge stages its own 2-slot ring inside the body and is inherited."""
    for arch, kw in (
        ("gfx942", dict(ring_depth=2)),
        ("gfx950", dict(ring_depth=3)),
        ("gfx950", dict(global_path="dma", transpose_source="tr_read")),
    ):
        with pytest.raises(ValueError, match="prefetch ring"):
            validate_attn_bwd_unified_spec(AttnBwdUnifiedSpec(head_size=64, **kw), arch)
    s = validate_attn_bwd_unified_spec(
        AttnBwdUnifiedSpec(head_size=64, barrier_merge="on"), "gfx942"
    )
    assert (s.ring_depth, s.barrier_merge) == (2, "on")


@pytest.mark.parametrize("arch", CDNA + ("gfx1151", "gfx1201"))
@pytest.mark.parametrize("d", (32, 64, 128))
def test_every_legal_page_has_an_instance(arch, d):
    for sv in (8, 1):
        for page in range(16, 513, 16):
            tag, knobs = attn_bwd_unified_tile_choice(arch, d, "bf16", sv, page)
            s = validate_attn_bwd_unified_spec(
                AttnBwdUnifiedSpec(head_size=d, stage_vec=sv, **knobs), arch
            )
            if s.sub_page is not None:
                assert s.sub_page == page and tag == "sub"
            else:
                assert page % s.block_n == 0


def test_declaration():
    for arch in CDNA + ("gfx1151",):
        dec = attn_bwd_unified_declaration(arch)
        assert dec["version"] == 1
        assert dec["abi"]["main"] == ATTN_BWD_UNIFIED_ABI
        assert dec["abi"]["prep"] == "rocke.attn_bwd.v3"
        assert dec["declared"] == (arch in CDNA)
        assert dec["reason_codes"] == list(REASON_CODES)
        assert set(dec["feature_groups"]) == set(FEATURE_GROUPS)
        assert len(dec["perf_classes"]) == 3 * 2 * 4 * 2 * 3 * 2
        assert {c["status"] for c in dec["perf_classes"]} == {"unverified"}
    pc = ("gfx942", "unified", 64, "bf16", "causal_br", "mha", "multi", "atomic")
    dec = attn_bwd_unified_declaration(
        "gfx942", policy=AttnBwdUnifiedPolicy(perf_status={pc: "verified"})
    )
    rows = [c for c in dec["perf_classes"] if c["status"] == "verified"]
    assert len(rows) == 1 and rows[0]["page_class"] == "multi"


def test_request_is_hashable_and_frozen():
    r = _req(strides={"q": [1024, 128, 1]})
    assert r.strides["q"] == (1024, 128, 1)
    assert hash(r) == hash(_req(strides={"q": (1024, 128, 1)}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.page = 32
