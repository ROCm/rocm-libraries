# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The variant-set gate checks every launch of a multi-launch (`sources`) descriptor.

The bundle under fixtures/verify_multi_launch/ is generated (see its README): one
engine, two kernels (`decode.fp16_d64`, `decode.fp16_d128`), each with the launches
`decode` and `merge` (role `merge`) of one builder. Each case copies it, mutates the
copy and runs the gate in structural mode; the full-mode cases over a real pack are in
test_verify_variant_sets_multi_launch_rocke.py.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_variant_sets as gate_module  # noqa: E402

BUNDLE = Path(__file__).parent / "fixtures" / "verify_multi_launch" / "bundle"
KDP = "verify_multi_launch.kdp.json"
D64, D128 = "decode.fp16_d64", "decode.fp16_d128"


def _tree(tmp_path: Path, name: str, edit=None, keep=(D64, D128)) -> str:
    """A copy of the bundle holding only the kernels named in `keep`, with `edit`
    applied to the list of their descriptors."""
    root = tmp_path / name
    shutil.copytree(BUNDLE, root)
    kdp = json.loads((root / KDP).read_text())
    descriptors = [d for d in kdp["kernelDescriptors"] if d["name"] in keep]
    if edit is not None:
        edit({d["name"]: d for d in descriptors})
    kdp["kernelDescriptors"] = descriptors
    (root / KDP).write_text(json.dumps(kdp))
    return str(root)


def _gate(capsys, *labelled_roots: tuple[str, str]):
    code = gate_module.main(
        ["--mode", "structural", *(x for pair in labelled_roots for x in pair)]
    )
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _lines(output: str, text: str) -> list[str]:
    return [line for line in output.splitlines() if text in line]


def _set_specs(descriptor: dict, **per_launch: dict) -> None:
    for launch, spec in per_launch.items():
        descriptor["sources"][launch]["spec"] = spec


def test_the_committed_bundle_passes_structurally(tmp_path, capsys):
    code, output = _gate(capsys, ("g", str(BUNDLE)))
    assert code == 0, output
    assert "descriptors=    2 distinct-binaries=    2" in output


class TestNesting:
    def test_equal_launch_tuples_nest(self, tmp_path, capsys):
        small = _tree(tmp_path, "small", keep=(D64,))
        big = _tree(tmp_path, "big")
        code, output = _gate(capsys, ("small", small), ("big", big))
        assert code == 0, output
        assert "small binaries subset of big: True" in output

    def test_tuples_differing_in_one_launch_do_not_nest(self, tmp_path, capsys):
        def other_merge(by_name):
            _set_specs(
                by_name[D64],
                merge={**by_name[D64]["sources"]["merge"]["spec"], "n": 1},
            )

        small = _tree(tmp_path, "small", keep=(D64,))
        big = _tree(tmp_path, "big", other_merge, keep=(D64,))
        code, output = _gate(capsys, ("small", small), ("big", big))
        assert code == 1
        assert "small binaries subset of big: False  MISSING 1" in output

    def test_a_role_swap_between_launches_does_not_nest(self, tmp_path, capsys):
        """Both launches keep their builder and spec, so only the role tells which
        kernel of the module each one launches."""

        def swap(by_name):
            sources = by_name[D64]["sources"]
            sources["decode"]["role"] = "merge"
            del sources["merge"]["role"]

        small = _tree(tmp_path, "small", keep=(D64,))
        big = _tree(tmp_path, "big", swap, keep=(D64,))
        code, output = _gate(capsys, ("small", small), ("big", big))
        assert code == 1
        assert "small binaries subset of big: False" in output

    @pytest.mark.parametrize("order", ["single-then-multi", "multi-then-single"])
    def test_a_single_and_a_multi_launch_descriptor_never_match(
        self, tmp_path, capsys, order
    ):
        """The single-launch form carries one launch's source verbatim, so only the
        key's shape separates it from the multi-launch descriptor."""

        def single(by_name):
            d = by_name[D64]
            d["kernel_source"] = d.pop("sources")["decode"]
            d["version"] = "1.0"

        multi = _tree(tmp_path, "multi", keep=(D64,))
        one = _tree(tmp_path, "one", single, keep=(D64,))
        pairs = (
            [("a", one), ("b", multi)]
            if order == "single-then-multi"
            else [("a", multi), ("b", one)]
        )
        code, output = _gate(capsys, *pairs)
        assert code == 1
        assert "a binaries subset of b: False  MISSING 1" in output


class TestSpecializationTwins:
    """`head_size` is the bundle's compiled-specialization knob. A descriptor pins it
    when any launch spec does."""

    @staticmethod
    def _launch_specs(by_name, decode, merge):
        _set_specs(by_name[D64], decode=decode, merge=merge)

    def _gate(self, tmp_path, capsys, small_specs, big_specs):
        small = _tree(
            tmp_path,
            "small",
            lambda by: self._launch_specs(by, *small_specs),
            keep=(D64,),
        )
        big = _tree(
            tmp_path,
            "big",
            lambda by: self._launch_specs(by, *big_specs),
            keep=(D64,),
        )
        return _gate(capsys, ("small", small), ("big", big))

    LEFT = {"variant": "a"}
    PINNED = {"variant": "a", "head_size": 64}

    def test_a_missing_twin_is_reported_once_for_a_two_launch_tuple(
        self, tmp_path, capsys
    ):
        code, output = self._gate(
            tmp_path,
            capsys,
            small_specs=(self.LEFT, self.LEFT),
            big_specs=(self.PINNED, self.PINNED),
        )
        assert code == 1
        assert len(_lines(output, "specialization twin missing")) == 1, output

    def test_a_descriptor_has_a_spec_to_settle_when_any_launch_has_one(
        self, tmp_path, capsys
    ):
        """`decode` carries an empty spec and `merge` an unpinned one: only a
        descriptor whose every launch spec is empty has nothing to settle."""
        code, output = self._gate(
            tmp_path,
            capsys,
            small_specs=({}, self.LEFT),
            big_specs=(self.PINNED, self.PINNED),
        )
        assert code == 1
        assert len(_lines(output, "specialization twin missing")) == 1, output

    def test_one_launch_pinning_the_knob_means_the_descriptor_is_no_twin_candidate(
        self, tmp_path, capsys
    ):
        code, output = self._gate(
            tmp_path,
            capsys,
            small_specs=(self.LEFT, self.PINNED),
            big_specs=(self.PINNED, self.PINNED),
        )
        assert code == 1  # the tuples still differ, which is plain nesting
        assert not _lines(output, "specialization twin missing"), output

    def test_a_candidate_pinned_in_one_launch_is_pinned(self, tmp_path, capsys):
        """The bigger set's descriptor pins the knob in `decode` only, so nothing in
        it is left to the kernel and the twin is still missing."""
        code, output = self._gate(
            tmp_path,
            capsys,
            small_specs=(self.LEFT, self.LEFT),
            big_specs=(self.PINNED, self.LEFT),
        )
        assert code == 1
        assert len(_lines(output, "specialization twin missing")) == 1, output

    def test_a_candidate_leaving_the_knob_to_every_launch_covers_the_twin(
        self, tmp_path, capsys
    ):
        code, output = self._gate(
            tmp_path,
            capsys,
            small_specs=(self.LEFT, self.LEFT),
            big_specs=(self.LEFT, {"variant": "b"}),
        )
        assert not _lines(output, "specialization twin missing"), output


class TestMetadataOnlyChecksRunOncePerDescriptor:
    def test_a_sentinel_counts_one_descriptor(self, tmp_path, capsys):
        def sentinel(by_name):
            by_name[D64]["metadata"]["head_size"] = gate_module.SENTINEL

        root = _tree(tmp_path, "g", sentinel)
        code, output = _gate(capsys, ("g", root))
        assert code == 1
        assert _lines(output, "descriptors ship the unset sentinel") == [
            f"      ! 1 descriptors ship the unset sentinel, e.g. {D64}"
        ]

    def test_a_tuple_collision_counts_one_pair(self, tmp_path, capsys):
        def collide(by_name):
            by_name[D128]["metadata"] = copy.deepcopy(by_name[D64]["metadata"])

        root = _tree(tmp_path, "g", collide)
        code, output = _gate(capsys, ("g", root))
        assert code == 1
        (line,) = _lines(output, "loader-tuple collisions")
        assert line.startswith("      ! 1 loader-tuple collisions")


class TestPerLaunchSpecChecks:
    def test_a_launch_whose_spec_contradicts_the_metadata_is_named(
        self, tmp_path, capsys
    ):
        """`seqlen_q` is a plain (not compiled-specialization) field, so the spec of
        each launch has to agree with the metadata. Only `merge` does not."""

        def corrupt(by_name):
            d = by_name[D64]
            d["metadata"]["seqlen_q"] = 1
            _set_specs(
                d,
                decode={**d["sources"]["decode"]["spec"], "seqlen_q": 1},
                merge={**d["sources"]["merge"]["spec"], "seqlen_q": 2},
            )

        root = _tree(tmp_path, "g", corrupt)
        code, output = _gate(capsys, ("g", root))
        assert code == 1
        (line,) = _lines(output, "contradicts")
        assert f"{D64} launch 'merge': seqlen_q" in line
        assert "launch 'decode'" not in line

    def test_launches_that_agree_with_the_metadata_pass(self, tmp_path, capsys):
        def agree(by_name):
            d = by_name[D64]
            d["metadata"]["seqlen_q"] = 1
            for launch in d["sources"].values():
                launch["spec"]["seqlen_q"] = 1

        root = _tree(tmp_path, "g", agree)
        code, output = _gate(capsys, ("g", root))
        assert code == 0, output

    def test_a_descriptor_with_both_kernel_source_and_sources_is_refused(
        self, tmp_path, capsys
    ):
        def both(by_name):
            d = by_name[D64]
            d["kernel_source"] = copy.deepcopy(d["sources"]["decode"])

        root = _tree(tmp_path, "g", both)
        code, output = _gate(capsys, ("g", root))
        assert code == 1
        assert "exactly one of" in output
