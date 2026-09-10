# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An arm may only carry shapes the engine will actually serve at that setting.

A knob value can be legal for some shapes and illegal for others, and there are two
different ways a shape can be out of reach. The spec constructor rejects an outright
illegal combination and raises, which `_arm` has always caught. The other kind
constructs perfectly well and is simply NOT SUPPORTED -- an unsupported head_size,
a `use_cfvst` combination the kernel declines -- and only the engine's own
eligibility predicate knows.

That second kind is invisible unless the predicate is asked about the FINAL spec:
after promotion to the builder's class and after this arm's overrides. Asking
before the overrides answers a question about a different kernel, and the arm then
ships a variant nothing will ever serve while the sweep reads as a clean comparison
against parity -- the shape of the failure that once put 180 unbuildable descriptors
on a device because no host gate had constructed the spec.

Nothing here imports rocKE: the dispatcher's resolution, the builder's spec class
and the engine's predicate are all supplied as stubs, because what is under test is
WHEN the predicate is asked, not what any particular kernel answers.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import knob_sweep  # noqa: E402
from dispatch_parity import Resolution  # noqa: E402

_STUB_MODULE = '''
def supports(spec, arch=None):
    """Declines exactly one FINAL combination: head_size 64 at the short sequence.

    Shaped like a real eligibility predicate rather than a blanket refusal, so the
    arm under test keeps a supported control -- an exclusion that removed every
    shape would pass a check that only counted the survivors.
    """
    if spec.head_size == 64 and spec.seqlen_q == 512:
        return False, "head_size 64 is unsupported at seqlen_q 512"
    return True, ""
'''


@dataclasses.dataclass(frozen=True)
class _Spec:
    head_size: int
    seqlen_q: int


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """A profile whose predicate is an importable stub, plus two resolved shapes."""
    (tmp_path / "stub_engine.py").write_text(_STUB_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("stub_engine", None)

    profile = {
        "slug": "test_engine",
        "source": "kernels/test.py",
        "builder": "build_test",
        "engine": {"name": "test:Engine"},
        "arch": "gfx942",
        "kmd_fields": [
            {"name": "head_size", "type": "int", "default_value": 128},
            {"name": "seqlen_q", "type": "int", "default_value": 512},
        ],
        "metadata_fields": ["head_size", "seqlen_q"],
        "specialization": {
            "metadata_fields": [],
            "matcher_only_fields": ["head_size", "seqlen_q"],
            "bindings": {},
            "vocabulary": {},
        },
        "predicate": {"module": "stub_engine", "function": "supports"},
    }
    resolutions = [
        Resolution({"seqlen_q": 512}, spec=_Spec(head_size=128, seqlen_q=512)),
        Resolution({"seqlen_q": 4096}, spec=_Spec(head_size=128, seqlen_q=4096)),
    ]
    return profile, resolutions


class TestSupportIsCheckedOnTheFinalSpec:
    def test_the_baseline_arm_carries_every_resolved_shape(self, sweep):
        """The control. Without overrides both shapes are supported, so an
        exclusion below is caused by the perturbation and not by a predicate that
        refuses this corpus outright."""
        profile, resolutions = sweep
        config, unbuildable = knob_sweep._arm(resolutions, profile, {})
        assert unbuildable == []
        assert len(config["packs"][0]["kernels"]) == 2

    def test_a_shape_unsupported_only_after_the_override_is_excluded_and_named(
        self, sweep
    ):
        """`head_size=64` constructs for both shapes and is declined for one of
        them. The declined shape must not reach the arm, and the caller must be
        able to say which one and why -- an arm covering a subset of the corpus is
        measurable, a silently narrowed one is not."""
        profile, resolutions = sweep
        config, unbuildable = knob_sweep._arm(resolutions, profile, {"head_size": 64})

        assert [index for index, _reason in unbuildable] == [0]
        assert "unsupported" in unbuildable[0][1]
        assert "head_size 64 is unsupported at seqlen_q 512" in unbuildable[0][1]

        kernels = config["packs"][0]["kernels"]
        assert len(kernels) == 1, "the supported control must survive the arm"
        assert kernels[0]["kernel_source"]["spec"]["seqlen_q"] == 4096
        assert kernels[0]["kernel_source"]["spec"]["head_size"] == 64

    def test_a_profile_with_no_predicate_still_builds_every_arm(self, sweep):
        """A profile that declares no eligibility API has not said anything about
        support, and this tool must not invent an answer: every constructible shape
        stays in the arm."""
        profile, resolutions = sweep
        profile.pop("predicate")
        _config, unbuildable = knob_sweep._arm(resolutions, profile, {"head_size": 64})
        assert unbuildable == []


class TestTheDeclarationReachesTheEmittedConfig:
    def test_the_arm_carries_the_specialization_block(self, sweep):
        """An arm is a generator config like any other. Dropping the declaration
        here would emit descriptors with no specialization_contract, and the
        machine that receives the archive would have nothing to check the compiled
        bytes against."""
        profile, resolutions = sweep
        config, _unbuildable = knob_sweep._arm(resolutions, profile, {})
        assert config["specialization"] == profile["specialization"]

    def test_a_missing_declaration_is_refused_rather_than_emitted_empty(self, sweep):
        profile, resolutions = sweep
        profile.pop("specialization")
        with pytest.raises(Exception, match="specialization"):
            knob_sweep._arm(resolutions, profile, {})

    def test_a_callback_resolved_knob_with_no_declared_readout_is_refused(self, sweep):
        """The waiver that must not exist. A knob this profile settles with its own
        policy callback has to name the builder-owned attribute or zero-argument
        accessor that answers the same question on the object the compiler hands the
        builder. There is no naming convention to infer one from, copying the
        formula into the declaration would certify the compile against something
        other than what built it, and moving the knob to matcher_only_fields would
        claim the compiler does not specialize on a field it does."""
        profile, resolutions = sweep
        profile["policies"] = {
            "use_cfvst": {"module": "stub_engine", "function": "supports"}
        }
        with pytest.raises(Exception, match="use_cfvst"):
            knob_sweep._arm(resolutions, profile, {})
