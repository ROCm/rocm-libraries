"""The offline defence for the tiled engine's `spec_resolution` launch surface.

That surface is declared `guard: none` in `gfx950_attention_tiled.profile.yaml`, and
deliberately so: `prepare()` trusts the KMD's resolved values verbatim rather than
re-deriving what the dispatcher would have chosen. Re-deriving in C++ would be a SECOND
implementation of a rule that already exists in Python -- the exact drift the launch
surface table exists to prevent, and catastrophic for this kernel specifically, because
the D256 cohort's seven-field override fold silently produces a DIFFERENT binary when
applied off-cohort.

So the defence is offline, and this is it: the shipped descriptors must be what rocKE's
own production resolver returns for the same shapes. If that ever stops being true, a
descriptor claims a binary the builder would not emit, and nothing at build, pack,
validate or match time notices.

Run:
    PYTHONPATH=<gen>/tools python -m pytest <gen>/tests/test_tiled_dispatch_parity.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_GEN = pathlib.Path(__file__).resolve().parent.parent
_REPO = _GEN.parents[3]  # <gen>/tools/IngestorGenerator -> repo root
_PROVIDER = _REPO / "dnn-providers" / "hip-kernel-provider"

# The rocKE library, and this integration's adapter.
for _p in (
    str(_GEN / "tools"),
    str(_PROVIDER / "rocke" / "library"),
    str(_PROVIDER / "rocke" / "platform" / "python"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SHAPES = _GEN / "configs" / "gfx950_attention_tiled.shipping-shapes.json"
_KDP = (
    _PROVIDER
    / "descriptor-packaging"
    / "examples"
    / "descriptors"
    / "rocKE"
    / "gfx950_attention_tiled"
    / "gfx950_attention_tiled.kdp.json"
)


def _require_inputs():
    """Both inputs are COMMITTED to this branch, so absence is a broken checkout
    rather than a reason to skip. A skipping test proves nothing, and these skipped
    silently on the first run because of a wrong parent-count in the path above."""
    missing = [str(p) for p in (_SHAPES, _KDP) if not p.is_file()]
    if missing:
        raise AssertionError(
            "tiled corpus/descriptors missing from the checkout: " + ", ".join(missing)
        )


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="rocKE's parity corpus imports torch; skipped where it is absent",
)


@requires_torch
def test_every_shipped_descriptor_matches_the_production_resolver():
    _require_inputs()
    """The shipped metadata IS what `_tiled_spec_from_problem` resolves.

    This is the whole point of generating through `dispatch_parity.py` rather than
    transcribing: a rule gets APPLIED rather than read. Transcribing is what missed a
    derived field on a sibling integration, where the descriptors took the dataclass
    default and the default was the opposite of the dispatcher's answer on most of a
    shipped set -- with nothing failing, and the only symptom a performance number
    misattributed three times.
    """
    import tiled_parity_adapter as tpa

    shapes = json.loads(_SHAPES.read_text())
    shipped = {
        k["name"]: k["metadata"]
        for k in json.loads(_KDP.read_text())["kernelDescriptors"]
    }
    assert shipped, "no shipped descriptors to check against"

    # Fields the descriptor carries that come straight from the resolved spec. Compared
    # by NAME so a spec field that stops reaching the catalog fails here.
    spec_fields = [
        "head_size",
        "block_size",
        "num_query_heads",
        "num_kv_heads",
        "num_seqs",
        "sliding_window",
        "num_warps",
        "block_m_per_warp",
        "tile_size",
        "waves_per_eu",
        "use_kq_lds_pad",
        "kq_lds_pad_halves",
        "use_mfma32_skip_legacy_qreg",
        "use_k_single_buffer",
        "use_q_direct_reg",
        "softmax_interleave_mode",
        "use_mask_phase_split",
    ]

    checked = 0
    for shape in shapes:
        fields = {k: v for k, v in shape.items() if not k.startswith("_")}
        fields["arch"] = "gfx950"
        request = tpa.TiledAttentionRequest(**fields)
        spec = tpa.tiled_spec_for_request(request)
        ok, _ = tpa.supports_tiled_2d_for_spec(spec, arch="gfx950")
        if not ok:
            continue  # a declined shape ships no descriptor, by construction

        # Find the shipped descriptor whose metadata this spec should equal. Keyed on
        # the resolved values themselves, since the generator derives the name.
        want = {f: getattr(spec, f) for f in spec_fields}
        matches = [
            name
            for name, md in shipped.items()
            if all(
                md.get(f) == (int(v) if isinstance(v, bool) else v)
                for f, v in want.items()
            )
        ]
        assert matches, (
            f"the resolver produced a spec no shipped descriptor describes:\n"
            f"  shape    {shape.get('_provenance', {}).get('graph', '?')}\n"
            f"  resolved {want}\n"
            f"A descriptor that disagrees with the resolver names a binary the builder "
            f"would not emit -- and nothing downstream compares the two halves."
        )
        checked += 1

    assert checked, "no servable shape was checked; the corpus or predicate changed"


@requires_torch
def test_the_d256_override_cohort_is_folded_by_the_resolver_not_by_us():
    """`_spec_gfx950_generic` ALREADY applies `_d256_gfx950_spec_overrides()`.

    The composition is idempotent ON the D256 cohort and produces a DIFFERENT binary --
    seven fields, a different kernel name -- off it. So the descriptor must bake the
    resolver's output unmodified, and this test is what notices if the resolver stops
    folding them (at which point every D256 descriptor would silently describe the
    wrong binary).
    """
    import tiled_parity_adapter as tpa

    # A D256 bf16 prefill shape: the cohort predicate is head_size==256, bf16, no fp8,
    # no window, no softcap/sinks/alibi/qq_bias, and max_seqlen_q > 1.
    request = tpa.TiledAttentionRequest(
        total_q=4096,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=256,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        arch="gfx950",
    )
    spec = tpa.tiled_spec_for_request(request)

    from kernels.common.attention_unified import _d256_gfx950_spec_overrides

    overrides = _d256_gfx950_spec_overrides()
    assert overrides, "the override dict is empty; the cohort mechanism moved"

    disagreeing = {
        field: (getattr(spec, field), value)
        for field, value in overrides.items()
        if hasattr(spec, field) and getattr(spec, field) != value
    }
    assert not disagreeing, (
        "the resolver did NOT fold the D256 overrides for an on-cohort shape: "
        f"{disagreeing}. Descriptors generated from it would describe a different "
        "binary than the one the builder emits."
    )


@requires_torch
def test_the_scope_gate_declines_three_d_routed_shapes():
    """D4's scope is enforced by US, not by rocKE's predicate.

    `supports_native_unified_attention_tiled` answers for the tiled FAMILY: measured,
    it returns (True, 'supported') for 3D-routed decode shapes. Without the adapter's
    `select_path() == "2d"` gate the parity set would carry descriptors for a path this
    engine does not ship.
    """
    import tiled_parity_adapter as tpa

    # Long-context, small-batch decode -- the regime `use_2d_kernel` routes to 3D.
    request = tpa.TiledAttentionRequest(
        total_q=1,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=32768,
        dtype="bf16",
        arch="gfx950",
    )
    spec = tpa.tiled_spec_for_request(request)
    problem = tpa.problem_for_spec(spec)
    assert (
        problem is not None and problem.select_path() == "3d"
    ), "this shape no longer routes to 3D, so it no longer tests the scope gate"

    ok, why = tpa.supports_tiled_2d_for_spec(spec, arch="gfx950")
    assert not ok, "the 2D-only scope gate did not decline a 3D-routed shape"
    assert "3d" in why.lower(), f"declined, but not for the scope reason: {why}"
