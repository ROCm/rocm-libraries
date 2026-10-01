# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An arm may only carry shapes the engine will actually serve at that setting.

An unsupported knob combination constructs fine and only the engine's eligibility
predicate knows, so the predicate must be asked about the FINAL spec: after
promotion to the builder's class and after the arm's overrides. The dispatcher,
spec class and predicate are stubs, because what is under test is WHEN it is asked.
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
        """Control: with no overrides both shapes are supported, so an exclusion below
        comes from the perturbation."""
        profile, resolutions = sweep
        config, unbuildable, _gained = knob_sweep._arm(resolutions, profile, {})
        assert unbuildable == []
        assert len(config["packs"][0]["kernels"]) == 2

    def test_a_shape_unsupported_only_after_the_override_is_excluded_and_named(
        self, sweep
    ):
        """Naming the excluded shape is what makes a narrowed arm measurable."""
        profile, resolutions = sweep
        config, unbuildable, _gained = knob_sweep._arm(
            resolutions, profile, {"head_size": 64}
        )

        assert [index for index, _reason in unbuildable] == [0]
        assert "unsupported" in unbuildable[0][1]
        assert "head_size 64 is unsupported at seqlen_q 512" in unbuildable[0][1]

        kernels = config["packs"][0]["kernels"]
        assert len(kernels) == 1, "the supported control must survive the arm"
        assert kernels[0]["kernel_source"]["spec"]["seqlen_q"] == 4096
        assert kernels[0]["kernel_source"]["spec"]["head_size"] == 64

    def test_a_profile_with_no_predicate_still_builds_every_arm(self, sweep):
        """A profile declaring no eligibility API has said nothing about support, and
        this tool must not invent an answer."""
        profile, resolutions = sweep
        profile.pop("predicate")
        _config, unbuildable, _gained = knob_sweep._arm(
            resolutions, profile, {"head_size": 64}
        )
        assert unbuildable == []


class TestTheDeclarationReachesTheEmittedConfig:
    def test_the_arm_carries_the_specialization_block(self, sweep):
        """Without the declaration the arm emits descriptors with no
        specialization_contract, leaving the receiving machine nothing to check the
        compiled bytes against."""
        profile, resolutions = sweep
        config, _unbuildable, _gained = knob_sweep._arm(resolutions, profile, {})
        assert config["specialization"] == profile["specialization"]

    def test_a_missing_declaration_is_refused_rather_than_emitted_empty(self, sweep):
        profile, resolutions = sweep
        profile.pop("specialization")
        with pytest.raises(Exception, match="specialization"):
            knob_sweep._arm(resolutions, profile, {})

    def test_a_callback_resolved_knob_with_no_declared_readout_is_refused(self, sweep):
        """A knob settled by the policy callback must name the builder-owned attribute
        or zero-argument accessor answering the same question on the object the compiler
        hands the builder; copying the formula in certifies the compile against
        something else."""
        profile, resolutions = sweep
        profile["policies"] = {
            "use_cfvst": {"module": "stub_engine", "function": "supports"}
        }
        with pytest.raises(Exception, match="use_cfvst"):
            knob_sweep._arm(resolutions, profile, {})


_SHIPPED_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx950_attention_dense.profile.yaml"
)
_D128 = dict(nhead_q=8, nhead_k=8, hdim_q=128, hdim_v=128, dtype="bf16", mask_type=0)


class TestCoverageShapesReachTheArmsThatServeThem:
    """942:S6-2: a tile arm exists for shapes the dispatcher's own tile cannot
    serve (Skv % 64 == 32 needs block_n=32). The sweep resolved every shape at the
    dispatcher's choice first, so those shapes never reached any arm and every
    arm reported the same served count."""

    def _sweep(self, monkeypatch, tmp_path, capsys):
        import importlib
        import json

        from dispatch_parity import ParityError, _bind_provider, _load_profile

        profile = _load_profile(str(_SHIPPED_PROFILE))
        monkeypatch.setattr(sys, "path", list(sys.path))
        try:
            _bind_provider(profile["provider_root"])
        except ParityError as exc:
            pytest.skip(f"the rocKE tree is not present in this tree ({exc})")
        try:
            importlib.import_module(profile["dispatch"]["module"])
        except ImportError as exc:
            pytest.skip(f"the rocKE library cannot be imported here ({exc})")
        shapes = tmp_path / "shapes.json"
        shapes.write_text(
            json.dumps(
                [
                    {**_D128, "batch": 1, "seqlen_q": 512, "seqlen_k": 512},
                    {**_D128, "batch": 1, "seqlen_q": 512, "seqlen_k": 288},
                ]
            )
        )
        argv = ["--profile", str(_SHIPPED_PROFILE), "--shapes", str(shapes)]
        rc = knob_sweep.main([*argv, "--isolate", "--out-dir", str(tmp_path / "arms")])
        return rc, capsys.readouterr().out, tmp_path / "arms"

    def test_the_block_n_32_arm_serves_the_shape_the_parity_tile_refuses(
        self, monkeypatch, tmp_path, capsys
    ):
        import yaml

        rc, out, arms = self._sweep(monkeypatch, tmp_path, capsys)
        assert rc == 0, out
        assert "coverage shapes   1" in out
        assert "COVERAGE: serves 1 of 1" in out

        def seqlens(arm):
            config = yaml.safe_load((arms / arm).read_text())
            return sorted(
                k["kernel_source"]["spec"]["seqlen_kv"]
                for k in config["packs"][0]["kernels"]
            )

        assert seqlens("arm_block_n_32.yaml") == [288, 512]
        assert seqlens("arm_parity.yaml") == [512], "parity serves only its own tile"
        assert seqlens("arm_block_n_64.yaml") == [512]
