# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stage-1 parity: the set the dispatcher itself would resolve.

The defect this tool exists to prevent is not a crash. A field the dispatcher
DERIVES from the request reads like an ordinary local variable, so a human
transcribing "the constants" copies the constants and misses the rule. The
descriptor then takes the dataclass default, which was the OPPOSITE of the
dispatcher's answer on most of a shipped set, and nothing failed: descriptors
validated, the desk check was clean, correctness passed on device. The only
symptom was a performance number, misattributed three times before the cause was
found.

So the assertions here are about AGREEMENT WITH A RULE, not about a tool running.
The central one recomputes `work >= num_persistent` independently and requires
every emitted descriptor to match it -- if the tool ever silently reverts to a
default, that test fails and no other one would.

These run against the real rocKE dispatcher and skip cleanly when it is not
importable, because the thing under test IS "we asked the library". A mocked
dispatcher would assert that the mock was called.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_PARITY = _TOOLS / "dispatch_parity.py"
_GATE = _TOOLS / "verify_variant_sets.py"
_GENERATE = _TOOLS.parent / "generate.py"
_PROFILE = _TOOLS.parent / "configs" / "gfx942_attention_dense.profile.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[5]

_BLOCK_M = 256  # baked; the kernel faults at other values


def _rocke_importable() -> bool:
    provider = _REPO_ROOT / "dnn-providers/hip-kernel-provider"
    return (provider / "rocke/library/dispatch/attention/gfx942.py").exists()


pytestmark = pytest.mark.skipif(
    not _rocke_importable(),
    reason="rocKE library not present; stage-1 parity asks the real dispatcher",
)


def _shapes() -> list[dict]:
    """A corpus spanning both sides of the persistent threshold, plus two shapes
    the kernel must refuse -- one per refusal LAYER."""
    out = []
    for batch in (1, 2):
        for heads_q, heads_kv in ((32, 8), (16, 16)):
            for seqlen in (512, 4096):
                for head_size in (64, 128):
                    for mask in (0, 1):
                        out.append(
                            {
                                "batch": batch,
                                "nhead_q": heads_q,
                                "nhead_k": heads_kv,
                                "seqlen_q": seqlen,
                                "seqlen_k": seqlen,
                                "hdim_q": head_size,
                                "hdim_v": head_size,
                                "dtype": "bf16",
                                "mask_type": mask,
                            }
                        )
    return out


def _unservable() -> list[dict]:
    base = {
        "batch": 1,
        "nhead_q": 32,
        "nhead_k": 8,
        "seqlen_q": 4096,
        "seqlen_k": 4096,
        "hdim_q": 128,
        "hdim_v": 128,
        "dtype": "bf16",
        "mask_type": 1,
    }
    return [
        {**base, "hdim_q": 96, "hdim_v": 96},  # head_size not in {64,128}
        {**base, "seqlen_q": 1},  # decode: Sq % BLOCK_M != 0
    ]


def _emitted_kernels(config_path) -> list[dict]:
    """The kernels an emitted config stands for, as plain dicts.

    Read through the config loader rather than off the YAML, because the tool emits
    the COMPACT `variants` form -- a shape list crossed with named knob sets -- and
    what these tests are about is the variant set that reaches a descriptor, not the
    syntax it was written in. Expanding through the loader is also the only reading
    that stays honest if the compact form ever gains a feature: a test that parsed
    the YAML itself would quietly stop seeing some of the kernels.
    """
    sys.path.insert(0, str(_TOOLS.parent))
    from codegen.config_loader import load_config

    return [
        {
            "name": kernel.name,
            "metadata": dict(kernel.metadata),
            "kernel_source": {"spec": dict(kernel.kernel_source.spec)},
        }
        for kernel in load_config(config_path).packs[0].kernels
    ]


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    """Run the tool once; every test reads the same emitted config."""
    work = tmp_path_factory.mktemp("parity")
    shapes = work / "shapes.json"
    shapes.write_text(json.dumps(_shapes() + _unservable()))
    config = work / "parity.yaml"
    result = subprocess.run(
        [
            sys.executable,
            str(_PARITY),
            "--profile",
            str(_PROFILE),
            "--shapes",
            str(shapes),
            "--out",
            str(config),
            "--report-knobs",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"dispatcher unavailable: {result.stderr.strip()[:200]}")
    import yaml

    return {
        "work": work,
        "stdout": result.stdout,
        "config_path": config,
        "config": yaml.safe_load(config.read_text()),
        "kernels": _emitted_kernels(config),
        "n_shapes": len(_shapes()),
    }


class TestDerivedFieldsSurvive:
    """The rule must be applied, not the default taken."""

    def test_persistent_matches_the_dispatchers_own_rule(self, parity):
        """Recomputed independently: nqb * Hq * B >= num_persistent.

        This is the assertion the whole tool exists for. Transcribing constants by
        hand got this backwards on most of a shipped set while every gate stayed
        green.
        """
        kernels = parity["kernels"]
        assert kernels, "no kernels emitted"
        disagreed = []
        for kernel in kernels:
            spec = kernel["kernel_source"]["spec"]
            nqb = -(-int(spec["seqlen_q"]) // _BLOCK_M)
            work = nqb * int(spec["num_query_heads"]) * int(spec["batch"])
            expected = work >= int(spec["num_persistent"])
            if bool(spec["persistent"]) != expected:
                disagreed.append(kernel["name"])
        assert not disagreed, (
            f"{len(disagreed)} descriptors disagree with the dispatcher's own "
            f"persistent rule, e.g. {disagreed[0]}"
        )

    def test_both_sides_of_the_persistent_rule_are_present(self, parity):
        """Otherwise the test above passes on a constant."""
        values = {
            bool(k["kernel_source"]["spec"]["persistent"]) for k in parity["kernels"]
        }
        assert values == {True, False}, (
            "corpus must straddle the persistent threshold or the rule check is "
            "vacuous"
        )

    def test_num_persistent_is_the_arch_value_not_the_shared_default(self, parity):
        """304 is gfx942's CU count; the shared dataclass default 256 is gfx950's."""
        values = {
            int(k["kernel_source"]["spec"]["num_persistent"]) for k in parity["kernels"]
        }
        assert values == {304}, f"expected the gfx942 CU count, got {values}"

    def test_waves_per_eu_comes_from_the_kernels_policy(self, parity):
        """Policy-resolved per (head_size, dtype), so it must not be one value."""
        values = {
            int(k["kernel_source"]["spec"]["waves_per_eu"]) for k in parity["kernels"]
        }
        assert len(values) > 1, (
            f"waves_per_eu is policy-owned and varies by head_size; got {values} -- "
            f"a single value means it was defaulted, not resolved"
        )


class TestMetadataDescribesTheBinary:
    def test_dtype_is_written_in_the_matchers_vocabulary(self, parity):
        """The spec says bf16; the matcher compares BF16."""
        for kernel in parity["kernels"]:
            assert kernel["metadata"]["dtype"] in ("BF16", "FP16")
            assert kernel["kernel_source"]["spec"]["dtype"] in ("bf16", "fp16")

    def test_a_policy_owned_tristate_is_resolved_not_omitted(self, parity):
        """`use_exp2_fast` is absent from the dispatcher's shared spec, but the
        binary still has a definite setting, so metadata must state it."""
        values = {k["metadata"].get("use_exp2_fast") for k in parity["kernels"]}
        assert None not in values, "a policy knob was left for the KMD default"
        assert values <= {0, 1}, f"unresolved or bogus values: {values}"
        assert len(values) > 1, (
            "the policy is seqlen-dependent; one value across a corpus that spans "
            "the threshold means it was pinned rather than asked"
        )

    def test_every_emitted_kernel_name_is_distinct(self, parity):
        """Colliding names are not caught anywhere downstream.

        The config loader checks PACK name uniqueness, not kernel names, and
        de-duplication keys on metadata rather than name -- so two variants sharing a
        name ship as separate descriptors that cannot be told apart in a log, a
        winner record, or a failure message.
        """
        names = [k["name"] for k in parity["kernels"]]
        assert len(names) == len(set(names)), "emitted kernel names collide"


class TestNamingIsOpAgnostic:
    """The tool must not assume attention's field names.

    The first version abbreviated a hardcoded list of them. On any other op it found
    none, and every variant collapsed onto one string -- two distinct conv variants
    both named `conv_fwd_dtfp16`, silently. This is a unit test rather than a
    pipeline one because the whole point is a kernel this repo's dispatcher does not
    serve.
    """

    def _name(self, slug, spec, index):
        import sys

        sys.path.insert(0, str(_TOOLS))
        from dispatch_parity import _kernel_name

        return _kernel_name(slug, spec, index)

    def test_a_conv_shaped_spec_names_its_variants_apart(self):
        import dataclasses

        @dataclasses.dataclass
        class ConvSpec:
            n: int
            c: int
            h: int
            w: int
            k: int
            dtype: str

        specs = [
            ConvSpec(1, 64, 56, 56, 64, "fp16"),
            ConvSpec(1, 128, 28, 28, 128, "fp16"),
        ]
        names = [self._name("conv_fwd", s, i) for i, s in enumerate(specs)]
        assert len(set(names)) == len(names), f"conv names collide: {names}"
        assert (
            "c64" in names[0] and "c128" in names[1]
        ), "the name must carry the fields that actually vary"

    def test_two_specs_differing_only_in_a_bool_are_named_apart(self):
        """A flag reads as present-or-absent, not as 0/1, but must still separate."""
        import dataclasses

        @dataclasses.dataclass
        class FlagSpec:
            size: int
            fused: bool

        names = [
            self._name("op", FlagSpec(64, True), 0),
            self._name("op", FlagSpec(64, False), 1),
        ]
        assert names[0] != names[1]

    def test_a_non_dataclass_spec_still_yields_a_unique_name(self):
        """Degenerate input must not produce a collision either."""

        class Opaque:
            pass

        names = [self._name("op", Opaque(), i) for i in range(3)]
        assert len(set(names)) == 3


class TestBothRefusalLayersAreReported:
    def test_construction_rejections_are_counted_separately(self, parity):
        """Spec CONSTRUCTION raises before any predicate runs. A support check that
        only calls the predicate reports these as servable and ships a wrong
        denominator."""
        assert "rejected          2" in parity["stdout"], parity["stdout"]

    def test_servable_count_excludes_them(self, parity):
        assert f"servable          {parity['n_shapes']}" in parity["stdout"]


class TestKnobPartition:
    def test_constant_knobs_are_named_as_non_axes(self, parity):
        """The mechanical form of "which knobs may be exposed"."""
        assert "CONSTANT -- shipped values, NOT tuning axes" in parity["stdout"]
        for knob in ("block_n", "lazy_rescale", "interleave"):
            assert knob in parity["stdout"]

    def test_shape_fields_and_the_two_real_axes_vary(self, parity):
        varies = parity["stdout"].split("VARIES", 1)[1].split("CONSTANT", 1)[0]
        for knob in ("waves_per_eu", "persistent", "seqlen_q", "head_size"):
            assert knob in varies, f"{knob} should vary across dispatch decisions"


class TestEndToEnd:
    def test_the_emitted_config_generates_and_passes_the_gate(self, parity):
        """The one that matters: parity -> generate -> gate, no hand editing.

        Stage 1 claims to be ONE command's worth of work. If the emitted config
        needs a human to finish it before the generator will read it, that claim is
        false, and this catches it.
        """
        bundle = parity["work"] / "bundle"
        generated = subprocess.run(
            [
                sys.executable,
                str(_GENERATE),
                "--config",
                str(parity["config_path"]),
                "--output-dir",
                str(bundle),
            ],
            cwd=_GENERATE.parent,
            capture_output=True,
            text=True,
        )
        assert generated.returncode == 0, generated.stdout + generated.stderr

        gated = subprocess.run(
            [sys.executable, str(_GATE), "A", str(bundle), "--profile", str(_PROFILE)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert gated.returncode == 0, gated.stdout + gated.stderr
        assert "GATE PASSED" in gated.stdout
        assert (
            "NOT CHECKED" not in gated.stdout
        ), "the parity profile must satisfy every gate property, not narrow the gate"


def _run_parity(work, *extra, name="out.yaml"):
    shapes = work / "shapes.json"
    if not shapes.exists():
        shapes.write_text(json.dumps(_shapes() + _unservable()))
    out = work / name
    result = subprocess.run(
        [
            sys.executable,
            str(_PARITY),
            "--profile",
            str(_PROFILE),
            "--shapes",
            str(shapes),
            "--out",
            str(out),
            *extra,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result, out


class TestTheShippingCrossProduct:
    """Stage 4a-3 builds the shipping package from the knobs that EARNED a slot.

    The runbook documented this step against a tool that could not do it: the base
    set is dispatcher-resolved, one spec per shape, so it cannot be expressed as a
    pack `axes:` block (axes cross one kernel_template). Following the step
    literally produced a config identical to the parity set, silently, and the
    survivors had to be enumerated by hand -- which is the transcription this whole
    tool exists to prevent.
    """

    def test_the_base_set_is_multiplied_by_the_surviving_knob(self, tmp_path):
        base_result, base_out = _run_parity(tmp_path, name="base.yaml")
        if base_result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="ship.yaml"
        )
        assert result.returncode == 0, result.stderr
        import yaml

        base = _emitted_kernels(base_out)
        ship = _emitted_kernels(out)
        assert len(ship) == 2 * len(base), (
            "the shipping set must be the dispatcher's set crossed with the "
            "survivors, not a re-derivation of it"
        )
        assert {k["metadata"]["use_exp2_fast"] for k in ship} == {
            0,
            1,
        }, "both arms of a surviving knob must ship; one value is not a sweep"

    def test_no_knobs_is_byte_identical_to_the_parity_set(self, tmp_path):
        """The parity path is the stage-1 deliverable and must not move because a
        later stage gained a flag."""
        a, out_a = _run_parity(tmp_path, name="a.yaml")
        if a.returncode != 0:
            pytest.skip("dispatcher unavailable")
        _, out_b = _run_parity(tmp_path, "--knobs", "{}", name="b.yaml")
        assert (
            out_a.read_bytes() == out_b.read_bytes()
        ), "an empty --knobs mapping is the parity set, exactly"

    def test_every_crossed_variant_is_named_apart(self, tmp_path):
        """Two variants of one shape differ only in the pinned knob. If the name
        does not encode it they collide; the loader rejects that, so a set this tool
        emits must encode the knob rather than lean on the rejection."""
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="n.yaml"
        )
        if result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        import yaml

        names = [k["name"] for k in _emitted_kernels(out)]
        assert len(names) == len(set(names))

    def test_an_undeclared_knob_is_refused_not_crossed(self, tmp_path):
        """An undeclared metadata field drops the WHOLE pack at
        resolveDescriptorSets(). Emitting the cross-product and discovering that at
        load time costs a build; refusing here costs nothing."""
        result, _ = _run_parity(
            tmp_path, "--knobs", '{"not_a_metadata_field": [1]}', name="bad.yaml"
        )
        assert result.returncode == 2
        assert "metadata_fields does not declare" in result.stderr

    def test_an_empty_knob_list_fails_loudly(self, tmp_path):
        """An empty axis's cross-product is empty: it would emit ZERO kernels."""
        result, _ = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": []}', name="empty.yaml"
        )
        assert result.returncode == 2
        assert "non-empty list" in result.stderr

    def test_malformed_knobs_json_names_the_problem(self, tmp_path):
        result, _ = _run_parity(tmp_path, "--knobs", "not json", name="bad2.yaml")
        assert result.returncode == 2
        assert "not valid JSON" in result.stderr

    def test_a_pinned_knob_overrides_the_policy_resolved_value(self, tmp_path):
        """Sweeping a policy-owned knob is exactly the case where the author is
        overriding the policy on purpose. If the policy value won instead, both
        arms would carry the same setting and the sweep would measure nothing."""
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="p.yaml"
        )
        if result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        import yaml

        kernels = _emitted_kernels(out)
        by_shape = {}
        for kernel in kernels:
            by_shape.setdefault(kernel["name"].rsplit(".", 1)[0], set()).add(
                kernel["metadata"]["use_exp2_fast"]
            )
        assert all(
            v == {0, 1} for v in by_shape.values()
        ), "every shape must appear under both pinned values"


class TestAPinnedKnobReachesTheBinary:
    """The arms must be different KERNELS, not one kernel under two catalog names.

    `--knobs` used to write the pinned value into `metadata` and into the spec only
    `if knob in variant_spec`. The dispatcher returns the SHARED spec, and every
    arch-private knob -- `use_exp2_fast`, `block_m`, the LDS pads -- is absent from
    it, which is precisely the set most worth sweeping. So the guard skipped exactly
    those: both arms carried the same spec, `hkp_pack` compiled ONE binary, and the
    two descriptors differed only in the catalog key the matcher compares.

    Nothing downstream called that an error. `verify_variant_sets.py` did flag it
    ("mislabel their binary"), but the runbook's own 4a-3 worked example produced it,
    so the gate read as the tool being wrong. The measurable symptom is the worst
    kind: the sweep reports ~1.000x and the knob is recorded as "no effect", when its
    other side was never compiled.
    """

    def test_both_arms_build_distinct_specs(self, tmp_path):
        """The regression test proper: N shapes x K arms must be N*K binaries."""
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="spec.yaml"
        )
        if result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        kernels = _emitted_kernels(out)
        specs = {
            json.dumps(k["kernel_source"]["spec"], sort_keys=True) for k in kernels
        }
        assert len(specs) == len(kernels), (
            f"{len(kernels)} descriptors collapse to {len(specs)} distinct specs -- "
            f"the pinned knob did not reach the spec, so the arms share a binary"
        )

    def test_the_pinned_value_is_in_the_spec_not_only_the_metadata(self, tmp_path):
        """Metadata is what the matcher compares; the spec is what gets compiled.
        Agreeing on one while diverging on the other is the tri-state trap."""
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="layers.yaml"
        )
        if result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        for kernel in _emitted_kernels(out):
            spec_value = kernel["kernel_source"]["spec"].get("use_exp2_fast")
            assert spec_value is not None, (
                f"{kernel['name']}: pinned knob absent from the spec, so the "
                f"builder's own policy -- not the pin -- decides the binary"
            )
            assert int(bool(spec_value)) == kernel["metadata"]["use_exp2_fast"], (
                f"{kernel['name']}: spec says {spec_value!r} and metadata says "
                f"{kernel['metadata']['use_exp2_fast']!r}; the matcher would select "
                f"this descriptor for a binary built the other way"
            )

    def test_the_emitted_arms_pass_the_variant_set_gate(self, tmp_path):
        """End to end, because the unit assertions above are reconstructions and
        the gate is what actually ships. This failed before the fix."""
        result, out = _run_parity(
            tmp_path, "--knobs", '{"use_exp2_fast": [0, 1]}', name="gated.yaml"
        )
        if result.returncode != 0:
            pytest.skip("dispatcher unavailable")
        tree = tmp_path / "tree"
        generated = subprocess.run(
            [
                sys.executable,
                str(_GENERATE),
                "--config",
                str(out),
                "--output-dir",
                str(tree),
            ],
            cwd=_GENERATE.parent,
            capture_output=True,
            text=True,
        )
        assert generated.returncode == 0, generated.stdout + generated.stderr
        gated = subprocess.run(
            [
                sys.executable,
                str(_GATE),
                "--profile",
                str(_PROFILE),
                "arms",
                str(tree / "descriptors"),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert gated.returncode == 0, gated.stdout + gated.stderr
        assert "GATE PASSED" in gated.stdout

    def test_a_knob_the_builder_cannot_take_is_refused(self, tmp_path):
        """The other half. A metadata-only field can never change the binary, so
        crossing on it manufactures duplicate kernels; refuse instead of emitting
        a set whose arms are identical by construction."""
        import yaml

        profile = yaml.safe_load(_PROFILE.read_text())
        profile["metadata_fields"] = list(profile["metadata_fields"]) + ["role"]
        profile["kmd_fields"] = list(profile["kmd_fields"]) + [
            {"name": "role", "type": "string"}
        ]
        # provider_root is repo-relative and the tool runs from _REPO_ROOT, so it
        # survives being written to a temp path unchanged.
        doctored = tmp_path / "role.profile.yaml"
        doctored.write_text(yaml.safe_dump(profile))
        shapes = tmp_path / "shapes.json"
        shapes.write_text(json.dumps(_shapes()))
        result = subprocess.run(
            [
                sys.executable,
                str(_PARITY),
                "--profile",
                str(doctored),
                "--shapes",
                str(shapes),
                "--knobs",
                '{"role": ["segment", "reduce"]}',
                "--out",
                str(tmp_path / "role.yaml"),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "does not accept" in result.stderr

    def test_a_real_spec_knob_is_not_refused(self, tmp_path):
        """Control for the refusal above: it must reject metadata-only fields
        WITHOUT rejecting the ordinary case, or 4a-3 stops working entirely."""
        result, _ = _run_parity(
            tmp_path, "--knobs", '{"block_n": [64, 32]}', name="ok.yaml"
        )
        assert result.returncode == 0, result.stdout + result.stderr
