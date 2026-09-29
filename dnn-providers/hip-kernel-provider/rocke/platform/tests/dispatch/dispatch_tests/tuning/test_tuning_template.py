# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tuned-family template end to end, on a toy kernel.

The smallest family built on :mod:`rocke.dispatch.tuning`: a request, a
kernel spec, a tuned-spec wrapper, a :class:`KnobSpace` subclass, and one
candidate per variant. ``rocke/dispatch/ARCHITECTURE.md`` section 14 walks
through the same pieces; attention (``library/dispatch/attention``) is the
production user.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelId,
    PinRefused,
    make_kernel_id,
)
from rocke.dispatch.tuning import (
    KnobSpace,
    gated,
    make_tuned_candidate,
    normalize_knobs,
    values,
    waves_per_eu_sweep_values,
)
from rocke.dispatch.tuning.testing import TuningContractError, assert_tuning_contract

ABI = "rocke-toy/v1"


# -------------------------------------------------------------- the request
@dataclass(frozen=True)
class ToyRequest:
    m: int
    arch: str = "gfx950"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    tuning_id: str = "auto"
    tuning_knobs: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "tuning_knobs", normalize_knobs(self.tuning_knobs))

    def normalized(self) -> dict:
        return asdict(self)

    def dims(self):
        return {"m": int(self.m)}

    def features(self):
        return frozenset()


# ----------------------------------------------------------- the kernel spec
@dataclass(frozen=True)
class ToyKernel:
    m: int
    tile: int
    unroll: int = 1
    prefetch: bool = False
    prefetch_depth: int = 1  # read only while prefetch is on
    swizzle: Optional[bool] = None  # None: resolved_swizzle() decides
    waves_per_eu: int = 2

    def __post_init__(self):
        if self.unroll not in (1, 2, 4):
            raise ValueError(f"unroll must be 1, 2 or 4, got {self.unroll}")

    def resolved_swizzle(self) -> bool:
        return self.swizzle if self.swizzle is not None else self.tile >= 128


@dataclass(frozen=True)
class ToyTunedSpec:
    arch: str
    candidate_name: str
    tuning_id: str
    variant_id: str
    config_key: str
    knobs: Tuple[Tuple[str, object], ...]
    kernel: ToyKernel

    def identity(self) -> dict:
        return {"arch": self.arch, "kernel": asdict(self.kernel)}

    def launch_grid(self):
        return (-(-self.kernel.m // self.kernel.tile), 1, 1)

    def launch_block(self):
        return (256, 1, 1)


# ------------------------------------------------------------- the space
AXES = (
    values("unroll", 1, (1, 2, 4)),
    gated("prefetch", {"prefetch_depth": (1, 2, 3)}),
    values("swizzle", None, (True, False)),
)


@dataclass(frozen=True)
class ToySpace(KnobSpace):
    def axes(self, base):
        return AXES

    def build(self, base, knobs, waves_per_eu):
        extra = {} if waves_per_eu is None else {"waves_per_eu": int(waves_per_eu)}
        return replace(base, **knobs, **extra)

    def inert(self, base, kernel):
        if kernel.swizzle is not None and kernel.swizzle == replace(
            kernel, swizzle=None
        ).resolved_swizzle():
            return {"swizzle": "restates its policy"}
        return {}

    def validate(self, base, kernel):
        if kernel.unroll * kernel.tile > 256:
            return False, "unroll x tile exceeds the register budget"
        return True, "ok"

    def defaults(self, base, kernel):
        # Everything the knobs are a delta against, minus the problem (m).
        return {k: v for k, v in asdict(base).items() if k != "m"}

    def default_waves(self, base):
        return base.waves_per_eu

    def waves(self, base, level):
        return waves_per_eu_sweep_values(base.waves_per_eu, level)

    def wrap(self, base, kernel, knobs, key, tid):
        return ToyTunedSpec(
            arch=self.arch,
            candidate_name=self.candidate_name,
            tuning_id=tid,
            variant_id=self.variant_id,
            config_key=key,
            knobs=knobs,
            kernel=kernel,
        )


# ------------------------------------------------------------ the candidates
def _candidate(tile: int, space_type=ToySpace):
    variant_id = f"tile{tile}"
    name = f"toy_gfx950_{variant_id}"

    def base(req: ToyRequest) -> ToyKernel:
        if int(req.m) <= 0:
            raise ValueError("m must be positive")
        return ToyKernel(m=int(req.m), tile=tile)

    return make_tuned_candidate(
        name=name,
        family="toy",
        algorithm="toy_tuned",
        spec_id=f"gfx950_{variant_id}",
        abi_version=ABI,
        priority=30,
        capability=Capability(arches=("gfx950",), dtypes=("bf16",)),
        space=space_type(
            abi=ABI, arch="gfx950", path="toy", variant_id=variant_id, candidate_name=name
        ),
        base=base,
        request_errors=lambda req: [],
        signature=lambda spec: (),
        build=lambda spec, arch: spec.kernel,
        bind_torch=lambda request, spec, tensors, **kw: None,
        grid=lambda spec, req: spec.launch_grid(),
    )


def _registry(space_type=ToySpace) -> CandidateRegistry:
    registry = CandidateRegistry("toy")
    for tile in (64, 128):
        registry.register(_candidate(tile, space_type))
    return registry


def _kid(req, candidate, spec) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="toy")


class TestToyFamily(unittest.TestCase):
    def test_every_candidate_keeps_the_contract(self):
        for candidate in _registry().candidates():
            with self.subTest(candidate=candidate.name):
                counts = assert_tuning_contract(
                    candidate,
                    [ToyRequest(m=1024), ToyRequest(m=100)],
                    other_requests=[ToyRequest(m=4096)],
                    default_knobs={"unroll": 1, "prefetch_depth": 3},
                    refused_knobs=[{"unroll": 3}, {"tile": 32}, {"not_a_knob": 1}],
                )
                self.assertGreater(counts["replays"], 0)
                self.assertGreater(counts["portable"], 0)

    def test_restated_policy_and_gateless_sub_knobs_are_dropped(self):
        candidate = _registry().get("toy_gfx950_tile128")
        req = ToyRequest(m=1024, algorithm="toy_tuned", spec_id="gfx950_tile128")
        default = candidate.select_spec(req)
        for knobs in ({"swizzle": True}, {"prefetch_depth": 2}):
            with self.subTest(knobs=knobs):
                self.assertEqual(
                    candidate.select_spec(replace(req, tuning_knobs=knobs)), default
                )
        kept = candidate.select_spec(
            replace(req, tuning_knobs={"prefetch": True, "prefetch_depth": 2})
        )
        self.assertEqual(dict(kept.knobs), {"prefetch": True, "prefetch_depth": 2})

    def test_opt_in_and_the_validator_gate(self):
        candidate = _registry().get("toy_gfx950_tile128")
        self.assertFalse(candidate.admits(ToyRequest(m=1024))[0])
        req = ToyRequest(m=1024, algorithm="toy_tuned", spec_id="gfx950_tile128")
        ok, why = candidate.admits(replace(req, tuning_knobs={"unroll": 4}))
        self.assertFalse(ok)
        self.assertIn("register budget", why)

    def test_sweep_results_pin_their_spec(self):
        registry = _registry()
        results = list(registry.iter_dispatch_all(ToyRequest(m=1024), kernel_id=_kid))
        self.assertTrue(results)
        for result in results:
            self.assertEqual(result.kernel_id.tuning_id, result.spec.tuning_id)
            self.assertEqual(result.request.tuning_knobs, result.spec.knobs)
            self.assertEqual(result.candidate.select_spec(result.request), result.spec)
        hashes = {r.kernel_id.request_hash for r in results}
        self.assertEqual(len(hashes), len(results))


class TestStalePins(unittest.TestCase):
    def test_a_pin_that_no_longer_resolves_raises_and_never_falls_back(self):
        registry = _registry()
        req = ToyRequest(m=1024, algorithm="toy_tuned", spec_id="gfx950_tile64")
        stored = registry.get("toy_gfx950_tile64").select_spec(
            replace(req, tuning_knobs={"unroll": 2})
        )
        pin = replace(req, tuning_id=stored.tuning_id, tuning_knobs=stored.knobs)
        self.assertEqual(registry.select(pin).select_spec(pin), stored)
        for label, stale, reason in (
            ("removed", replace(pin, spec_id="gfx950_tile32"), "no registered candidate"),
            ("mismatch", replace(pin, tuning_knobs={"unroll": 4}), "canonicalize to"),
            ("unknown", replace(pin, tuning_knobs=(), tuning_id="tile64_wpe2@" + "0" * 16),
             "unknown tuning_id"),
        ):
            with self.subTest(label), self.assertRaises(PinRefused) as raised:
                registry.select(stale)
            self.assertIn(reason, str(raised.exception))

    def test_bad_knob_values_fail_when_the_request_is_built(self):
        with self.assertRaisesRegex(TypeError, "JSON scalar"):
            ToyRequest(m=8, tuning_knobs={"unroll": [2]})


class _IgnoresKnobs(ToySpace):
    """A broken space: pinned knobs are silently ignored."""

    def canonicalize(self, base, knobs):
        return super().canonicalize(base, {})


class TestTheKitCatchesBrokenSpaces(unittest.TestCase):
    def test_ignoring_pinned_knobs_is_caught(self):
        candidate = _registry(_IgnoresKnobs).get("toy_gfx950_tile64")
        with self.assertRaises(TuningContractError):
            assert_tuning_contract(candidate, [ToyRequest(m=1024)])


if __name__ == "__main__":
    unittest.main()
