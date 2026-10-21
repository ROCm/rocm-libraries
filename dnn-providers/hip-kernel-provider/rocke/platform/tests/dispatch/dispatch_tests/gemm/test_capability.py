# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declared-coverage invariants for the GEMM registries.

These are the family-level checks from ARCHITECTURE.md section 10. They are
parametrized over the registry rather than over a fixed candidate list, so a
kernel added years from now inherits them without anyone remembering to.
"""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget, known_arches, listed_arch_for
from rocke.core.arch.target import _load_generic_targets
from rocke.dispatch import GemmRequest
from rocke.dispatch.gemm.bf16_rcr import GEMM_BF16_REGISTRY
from rocke.dispatch.gemm.fp16_rcr import GEMM_FP16_REGISTRY

_REGISTRIES = (("fp16", GEMM_FP16_REGISTRY), ("bf16", GEMM_BF16_REGISTRY))

# Wide enough that every registered tile divides some entry, and skinny enough
# to reach the decode candidate.
_DIMS = (1, 2, 16, 32, 64, 128, 256, 512, 1024, 4096)
_SHAPES = tuple((m, n, k) for m in _DIMS for n in _DIMS for k in _DIMS)


def _requests(arch: str, dtype: str):
    for m, n, k in _SHAPES:
        yield GemmRequest(M=m, N=n, K=k, arch=arch, dtype=dtype)


class TestDeclaredCoverage(unittest.TestCase):
    def test_every_candidate_declares_a_capability(self):
        for dtype, registry in _REGISTRIES:
            for candidate in registry.candidates():
                with self.subTest(dtype=dtype, candidate=candidate.name):
                    self.assertIsNotNone(candidate.capability)

    def test_no_candidate_admits_an_architecture_it_did_not_declare(self):
        """The cross-architecture misroute, pinned for every arch we know of."""
        for dtype, registry in _REGISTRIES:
            for candidate in registry.candidates():
                undeclared = {
                    arch
                    for arch in known_arches()
                    if listed_arch_for(arch, candidate.capability.arches) is None
                }
                for arch in sorted(undeclared):
                    with self.subTest(candidate=candidate.name, arch=arch):
                        for req in _requests(arch, dtype):
                            self.assertFalse(
                                candidate.admits(req)[0],
                                f"{candidate.name} wrongly admits {arch}",
                            )

    def test_every_candidate_spans_a_single_wave_size(self):
        """What an ``arch_families`` field looked like it gave and did not.

        gfx1250 is CDNA-family at wave32, so a family label cannot stand in for
        this; the declared arch list is checked against the arch catalog.
        """
        for dtype, registry in _REGISTRIES:
            for candidate in registry.candidates():
                with self.subTest(dtype=dtype, candidate=candidate.name):
                    waves = {
                        ArchTarget.from_gfx(a).wave_size
                        for a in candidate.capability.arches
                    }
                    self.assertEqual(
                        len(waves),
                        1,
                        f"{candidate.name} spans wave sizes {sorted(waves)}; "
                        "split it into one candidate per wave size",
                    )

    def test_every_declared_architecture_is_reachable(self):
        """Catches the opposite error: coverage claimed but never served.

        A declared arch that no request can reach makes the coverage manifest
        overstate what is dispatchable, which is the failure a manifest exists
        to prevent.
        """
        for dtype, registry in _REGISTRIES:
            for candidate in registry.candidates():
                for arch in candidate.capability.arches:
                    with self.subTest(candidate=candidate.name, arch=arch):
                        self.assertTrue(
                            any(
                                candidate.admits(req)[0]
                                for req in _requests(arch, dtype)
                            ),
                            f"{candidate.name} declares {arch} but admits no "
                            "request there",
                        )

    def test_declared_dtype_matches_the_family(self):
        for dtype, registry in _REGISTRIES:
            for candidate in registry.candidates():
                with self.subTest(dtype=dtype, candidate=candidate.name):
                    self.assertEqual(candidate.capability.dtypes, (dtype,))

    def test_for_arch_agrees_with_what_each_candidate_declares(self):
        for _dtype, registry in _REGISTRIES:
            for arch in known_arches():
                with self.subTest(family=registry.family, arch=arch):
                    self.assertEqual(
                        {c.name for c in registry.for_arch(arch)},
                        {
                            c.name
                            for c in registry.candidates()
                            if listed_arch_for(arch, c.capability.arches)
                        },
                    )

    def test_coverage_manifest_lists_the_declared_arches(self):
        """The lists stay as declared: a generic's members are served through
        the generic's entry, never by being added to a list."""
        manifest = GEMM_FP16_REGISTRY.coverage()
        by_name = {c["name"]: c for c in manifest["candidates"]}
        self.assertEqual(
            by_name["universal_gemm_fp16_cdna_cshuffle"]["capability"]["arches"],
            ["gfx942", "gfx950"],
        )
        self.assertEqual(
            by_name["universal_gemm_fp16_rdna_wmma"]["capability"]["arches"],
            ["gfx11-generic", "gfx1151", "gfx1201"],
        )

    def test_every_generic_member_gets_exactly_its_generics_candidates(self):
        """A list that names a generic target covers each member pointing there.

        Members have rows of their own, so a request keeps the processor's name.
        A member no list names must reach exactly the candidates, and the specs,
        a request for its generic reaches; a member a list names (gfx1151,
        gfx1201) reaches those plus its own.
        """
        shapes = ((1, 4096, 4096), (64, 64, 64), (512, 1024, 256), (4096, 4096, 4096))
        for dtype, registry in _REGISTRIES:
            named = {a for c in registry.candidates() for a in c.capability.arches}
            for generic, members in _load_generic_targets().items():
                for member in members:
                    with self.subTest(dtype=dtype, member=member):
                        own = {
                            c.name
                            for c in registry.candidates()
                            if member in c.capability.arches
                        }
                        self.assertEqual(
                            {c.name for c in registry.for_arch(member)},
                            {c.name for c in registry.for_arch(generic)} | own,
                        )
                    if member in named:
                        continue
                    for m, n, k in shapes:
                        with self.subTest(dtype=dtype, member=member, shape=(m, n, k)):
                            mine = GemmRequest(M=m, N=n, K=k, arch=member, dtype=dtype)
                            its = GemmRequest(M=m, N=n, K=k, arch=generic, dtype=dtype)
                            served = registry.supported(its)
                            self.assertEqual(
                                [c.name for c in registry.supported(mine)],
                                [c.name for c in served],
                            )
                            for c in served:
                                self.assertEqual(
                                    c.select_spec(mine), c.select_spec(its)
                                )


class TestDecodeShapeGateAsData(unittest.TestCase):
    """The bf16 decode candidate's skinny-M gate moved from code to data."""

    def _decode(self):
        return GEMM_BF16_REGISTRY.get("universal_gemm_bf16_cdna_decode")

    def test_the_m_bound_is_declared_rather_than_hand_coded(self):
        shapes = self._decode().capability.shapes
        self.assertEqual([s.as_dict() for s in shapes], [{"dims": ["M"], "max": 32}])

    def test_skinny_m_is_admitted_and_wide_m_is_not(self):
        decode = self._decode()
        skinny = GemmRequest(M=2, N=4096, K=4096, arch="gfx950", dtype="bf16")
        wide = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", dtype="bf16")
        self.assertTrue(decode.admits(skinny)[0])
        ok, why = decode.admits(wide)
        self.assertFalse(ok)
        self.assertIn("M=4096 > max 32", why)


if __name__ == "__main__":
    unittest.main()
