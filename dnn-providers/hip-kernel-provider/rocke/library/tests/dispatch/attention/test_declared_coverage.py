# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declared-coverage invariants for the attention registry.

The counterpart to the platform-side ``test_declared_coverage.py``, covering the
checks that need no device: that every candidate declares what it serves, that
none admits an architecture it did not declare, and that the two candidates
which legitimately span both wave sizes are the ones we think they are.

Reachability -- that each declared arch actually resolves to its candidate -- is
covered per candidate by the wiring tests beside this file, which pin the
resolved device arch that the cohort heuristics consult.
"""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget, known_arches

from dispatch.attention import (
    ATTENTION_REGISTRY,
    AttentionRequest,
    attention_candidates,
)

# Candidates that serve a *path* rather than one kernel: the concrete backend is
# chosen downstream by attention_unified on the running device (wave64 MFMA on
# gfx942/gfx950, wave32 WMMA on gfx1250, arch-neutral scalar elsewhere), so a
# single wave size does not apply to them. Every other candidate names one
# concrete kernel and must not straddle.
_PATH_CANDIDATES = frozenset({"attention_unified_2d", "attention_unified_3d"})


def _requests(arch: str):
    for dtype in ("fp16", "bf16"):
        for hdim in (64, 128, 256):
            for block in (16, 32, 64):
                for sq, sk in ((1, 4096), (512, 512), (2048, 2048)):
                    for algorithm in ("auto", "attention_dense"):
                        yield AttentionRequest(
                            batch=2,
                            nhead_q=16,
                            nhead_k=2,
                            seqlen_q=sq,
                            seqlen_k=sk,
                            hdim_q=hdim,
                            hdim_v=hdim,
                            arch=arch,
                            dtype=dtype,
                            kv_block_size=block,
                            algorithm=algorithm,
                        )


class TestDeclaredCoverage(unittest.TestCase):
    def test_every_candidate_declares_a_capability(self):
        for candidate in attention_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.capability)

    def test_no_candidate_admits_an_architecture_it_did_not_declare(self):
        for candidate in attention_candidates():
            undeclared = set(known_arches()) - set(candidate.capability.arches)
            for arch in sorted(undeclared):
                with self.subTest(candidate=candidate.name, arch=arch):
                    for req in _requests(arch):
                        self.assertFalse(
                            candidate.admits(req)[0],
                            f"{candidate.name} wrongly admits {arch}",
                        )

    def test_specialized_candidates_span_a_single_wave_size(self):
        for candidate in attention_candidates():
            if candidate.name in _PATH_CANDIDATES:
                continue
            with self.subTest(candidate=candidate.name):
                waves = {
                    ArchTarget.from_gfx(a).wave_size
                    for a in candidate.capability.arches
                }
                self.assertEqual(len(waves), 1, f"{candidate.name} straddles {waves}")

    def test_the_path_candidates_are_the_only_wave_straddling_ones(self):
        """Pins the exception itself, so a new straddling candidate has to be
        argued for here rather than quietly inheriting the exemption."""
        straddling = {
            c.name
            for c in attention_candidates()
            if len({ArchTarget.from_gfx(a).wave_size for a in c.capability.arches}) > 1
        }
        self.assertEqual(straddling, set(_PATH_CANDIDATES))

    def test_for_arch_agrees_with_what_each_candidate_declares(self):
        for arch in known_arches():
            with self.subTest(arch=arch):
                self.assertEqual(
                    {c.name for c in ATTENTION_REGISTRY.for_arch(arch)},
                    {
                        c.name
                        for c in attention_candidates()
                        if arch in c.capability.arches
                    },
                )

    def test_manifest_names_the_declared_targets(self):
        by_name = {c["name"]: c for c in ATTENTION_REGISTRY.coverage()["candidates"]}
        self.assertEqual(
            by_name["attention_unified_2d"]["capability"]["arches"],
            list(known_arches()),
        )
        self.assertEqual(
            by_name["attention_gfx942_dense_pipe"]["capability"]["arches"], ["gfx942"]
        )
        self.assertEqual(
            by_name["attention_gfx950_dense"]["capability"]["arches"], ["gfx950"]
        )
        self.assertEqual(
            by_name["attention_gfx950_d256"]["capability"]["arches"], ["gfx950"]
        )
        self.assertEqual(
            by_name["attention_d256_decode"]["capability"]["arches"],
            ["gfx942", "gfx950"],
        )

    def test_the_unified_paths_keep_serving_every_known_arch(self):
        """The gfx1250 live prefill benchmark dispatches through unified_2d and
        runs the WMMA backend behind it; RDNA reaches the scalar backend the
        same way. Narrowing these to the wave64 MFMA targets would drop both."""
        for arch in known_arches():
            with self.subTest(arch=arch):
                names = {c.name for c in ATTENTION_REGISTRY.for_arch(arch)}
                self.assertIn("attention_unified_2d", names)
                self.assertIn("attention_unified_3d", names)


class TestDeclaredHeadSizes(unittest.TestCase):
    def test_the_d256_candidates_declare_their_head_size_as_data(self):
        for name in ("attention_gfx950_d256", "attention_d256_decode"):
            with self.subTest(candidate=name):
                shapes = ATTENTION_REGISTRY.get(name).capability.shapes
                self.assertIn(
                    {"dims": ["hdim_q"], "allowed": [256]},
                    [s.as_dict() for s in shapes],
                )

    def test_the_unified_paths_declare_the_backend_head_sizes(self):
        shapes = ATTENTION_REGISTRY.get("attention_unified_2d").capability.shapes
        self.assertIn(
            {"dims": ["hdim_q"], "allowed": [64, 128, 256]},
            [s.as_dict() for s in shapes],
        )


if __name__ == "__main__":
    unittest.main()
