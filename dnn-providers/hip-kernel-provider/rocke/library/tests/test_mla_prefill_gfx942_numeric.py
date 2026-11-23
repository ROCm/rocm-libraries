# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 bf16 MLA prefill forward kernel.

Runs :mod:`builders.mla.verify_score_probe_gfx942` in ``--mode full`` for its
scenario table and checks ``(out, lse)`` against the fp32 numpy oracle
(:func:`builders.mla.ref_mla_attn.ref_mla_prefill`, bottom-right causal). The
bars, per live sequence, are ``out`` ``max_abs <= 4e-2``
(:data:`~builders.mla.verify_score_probe_gfx942.BF16_ATTN_ATOL`, the bf16 bar
the other rocKE attention kernels use) *and* ``max_abs / peak(|ref|) <= 2e-2``
(``OUT_REL_PEAK_TOL``), and ``lse`` ``max_abs <= 1.7e-2`` (``LSE_ATOL``). The
peak-relative bar exists because ``out`` shrinks as S_k grows, so a lost key
tile can stay under 4e-2 absolute; ``test_negative_control_*`` pins that. Fully
masked rows are held to ``out == 0`` / ``lse == -inf`` exactly.

Specs come from the MLA dispatcher, so each case compiles the layout that ships
for it: two heads per workgroup for even head counts, the single-head layout for
odd ones, and agpr_acc for batches averaging >= 320 query rows per sequence
(``sk4096_h128``). Two extra lanes cover what dispatch only picks on some
shapes: agpr_acc forced on for short / ragged batches, and the unfused
``absorbed_query`` composition.

The numeric tests are marked ``gpu`` and skip without a gfx942 device and ROCm
torch (torch also runs the ``absorbed_query`` lane's head projections); the
scenario-table and dispatch checks run on CPU. Run:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_mla_prefill_gfx942_numeric.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready() -> bool:
    """True on a gfx942 box with ROCm torch (gated on ``gcnArchName``, the ISA
    target, as the dense numeric lane does)."""
    if not torch.cuda.is_available():
        return False
    return "gfx942" in torch.cuda.get_device_properties(0).gcnArchName.lower()


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)

# Scenario names come from the probe's table; spelled out here so the test ids
# are collectable without importing kernel code on a CPU box.
_ALL_SCENARIOS = [
    "square_h64",
    "square_h128",
    "sq_lt_sk_h64",
    "sq_lt_sk_h128",
    "page_tail_h64",
    "page_tail_h128",
    "mixed_varlen_h64",
    "mixed_varlen_h128",
    "sk512_h64",
    "sk4096_h128",
    "ragged_long_h64",
    "single_head_h5",
    "six_heads_h6",
    "many_seqs_h64",
]

# agpr_acc forced on: masked rows + page tails, a long single sequence, and the
# ragged long batch -- shapes dispatch runs on the 16-row layout.
_AGPR_FORCED = ["mixed_varlen_h64", "sk512_h64", "ragged_long_h64"]

# absorbed_query: one short, one masked/ragged, the long cases, H=6, and the
# 300-sequence batch. The single-head layout does not implement absorbed_query,
# so H=5 is not listed.
_ABSORBED = [
    "square_h64",
    "mixed_varlen_h128",
    "sk512_h64",
    "sk4096_h128",
    "ragged_long_h64",
    "six_heads_h6",
    "many_seqs_h64",
]


def _probe():
    from builders.mla import verify_score_probe_gfx942 as probe

    return probe


def _arch() -> str:
    from rocke.runtime.hip_module import get_device_arch

    return get_device_arch() or "gfx942"


def _run(name: str, *, absorbed_query: bool = False, agpr=None):
    probe = _probe()
    sc = probe.SCENARIOS[name]
    return probe.run_case(
        name,
        sc.heads,
        sc.q_lens,
        sc.k_lens,
        arch=_arch(),
        mode="full",
        seed=0,
        tol=probe.BF16_ATTN_ATOL,
        dry_run=False,
        absorbed_query=absorbed_query,
        agpr=agpr,
    )


def _agpr_flavor_ok() -> bool:
    from rocke.core.lower_llvm import _is_modern_flavor, _resolve_llvm_flavor

    return _is_modern_flavor(_resolve_llvm_flavor())


def test_scenario_lists_match_probe_table():
    """The ids above are the probe's table, so no scenario is silently dropped."""
    probe = _probe()
    assert sorted(_ALL_SCENARIOS) == sorted(probe.SCENARIOS)
    assert set(_AGPR_FORCED) <= set(probe.SCENARIOS)
    assert set(_ABSORBED) <= set(probe.SCENARIOS)


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", _ALL_SCENARIOS)
def test_mla_prefill_numeric_dispatched(name):
    assert _run(name) is True, f"{name}: see captured stdout for per-seq residuals"


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", _AGPR_FORCED)
def test_mla_prefill_numeric_agpr_forced(name):
    if not _agpr_flavor_ok():
        pytest.skip("agpr_acc needs the llvm22/llvm23 flavor")
    assert _run(name, agpr=True) is True, f"{name} agpr_acc: see captured stdout"


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", _ABSORBED)
def test_mla_prefill_numeric_absorbed_query(name):
    assert (
        _run(name, absorbed_query=True) is True
    ), f"{name} absorbed_query: see captured stdout"


def test_absorbed_lane_runs_the_spec_dispatch_selects():
    """The absorbed lane asks dispatch for ``absorbed_query`` on the request,
    so the spec it runs is the dispatched one, not a post-dispatch swap; the
    single-head layout has no absorbed kernel, so dispatch refuses H=5."""
    probe = _probe()
    for name in _ABSORBED:
        sc = probe.SCENARIOS[name]
        spec = probe._dispatched_spec(
            sc.heads, sc.q_lens, sc.k_lens, "gfx942", absorbed_query=True
        )
        assert spec.absorbed_query and spec.heads_per_wg == 2, name
        assert spec.fwd_kernel_name().endswith("_aq"), name
    sc = probe.SCENARIOS["single_head_h5"]
    spec = probe._dispatched_spec(
        sc.heads, sc.q_lens, sc.k_lens, "gfx942", absorbed_query=True
    )
    ok, why = probe.supports_mla_prefill(spec, arch="gfx942")
    assert not ok and "absorbed_query" in why


def test_packed_absorbed_refusal_is_a_failure_not_a_skip():
    """An even-head absorbed request dispatch refuses (here the 576-wide query
    extent) must surface its reason, not fall back to the single-head SKIP."""
    probe = _probe()
    with pytest.raises(ValueError, match="query extent"):
        probe._dispatched_spec(128, [30000], [30000], "gfx942", absorbed_query=True)
    result = probe.run_case(
        "extent_h128",
        128,
        [30000],
        [30000],
        mode="full",
        arch="gfx942",
        seed=0,
        tol=probe.BF16_ATTN_ATOL,
        dry_run=False,
        absorbed_query=True,
    )
    assert result is False


@requires_gfx942_gpu
@pytest.mark.gpu
def test_negative_control_one_zeroed_page_fails_the_bars():
    """A kernel that loses one 16-key tile must fail the verifier.

    ragged_long_h64's sequence 2 (S_q=40, S_k=2048) is launched with its first
    page zeroed in ``c_kv`` / ``k_rope`` while the oracle reads the intact
    inputs (input-side only, no kernel change). Measured on 304-CU gfx942:
    out max_abs 3.9e-2 -- under the 4e-2 absolute bar alone -- but
    max_abs / peak 1.5e-1 and lse max_abs 2.0e-2, so each new bar catches it.
    """
    probe = _probe()
    sc = probe.SCENARIOS["ragged_long_h64"]

    def run(**bars):
        return probe.run_case(
            "ragged_long_h64",
            sc.heads,
            sc.q_lens,
            sc.k_lens,
            arch=_arch(),
            mode="full",
            seed=0,
            dry_run=False,
            corrupt=(2, 0),
            **bars,
        )

    assert run(tol=probe.BF16_ATTN_ATOL) is False
    # Each new bar alone, with the other two opened up.
    assert run(tol=1.0, lse_tol=1.0) is False, "peak-relative out bar missed it"
    assert run(tol=1.0, rel_tol=1.0) is False, "lse bar missed it"


@requires_gfx942_gpu
@pytest.mark.gpu
def test_rows_per_sequence_dispatch_agpr_and_odd_heads_single_head():
    """The numeric cases above exercise the layouts they claim to.

    sk4096_h128 is one 512-query sequence, at least 320 rows per sequence, which
    dispatch serves with agpr_acc on any CU count; every other two-heads
    scenario averages fewer rows and runs the 16-row layout (agpr_acc is covered
    for those by the forced lane). single_head_h5 has an odd head count, which
    dispatch cannot pack two to a workgroup.
    """
    probe = _probe()
    arch = _arch()

    def spec(name):
        sc = probe.SCENARIOS[name]
        return probe._dispatched_spec(sc.heads, sc.q_lens, sc.k_lens, arch)

    if _agpr_flavor_ok():
        assert spec("sk4096_h128").agpr_acc
    for name in _ALL_SCENARIOS:
        if name != "sk4096_h128":
            assert not spec(name).agpr_acc, name
    single = spec("single_head_h5")
    assert (single.heads_per_wg, single.q_tiles_per_wg, single.agpr_acc) == (
        1,
        1,
        False,
    )
    assert spec("six_heads_h6").heads_per_wg == 2
