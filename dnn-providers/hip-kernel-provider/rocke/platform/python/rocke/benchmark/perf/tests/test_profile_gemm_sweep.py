# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the GEMM-sweep example's pure wiring (no GPU, no rocKE)."""
import tempfile
import unittest
from types import SimpleNamespace

from rocke.benchmark.perf import schema
from rocke.benchmark.perf.tool import store, selfcheck
from rocke.benchmark.perf.examples import profile_gemm_sweep as ex


def _fake_record(
    cmd,
    arch,
    *,
    match=None,
    label=None,
    op="unknown",
    shape=None,
    warmup=0,
    warn=None,
    _busy=[100],
):
    """Stand-in for harness.profile: identity from label, counters from a counter."""
    counters = {"busy_cycles": _busy[0], "total_clocks": 1000}
    return {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": label or "sym", "op": op, "shape": shape or {}},
        "wall": {},
        "counters": counters,
        "resources": {},
        "derived": {"busy_fraction": _busy[0] / 1000},
        "captured_counters": sorted(counters),
        "verify": {},
    }


class TestLaunchCmd(unittest.TestCase):
    def test_launch_cmd_shape_triple(self):
        v = ex.Variant(
            "ck", "sym", "/t/a.hsaco", "/t/m.json", {"M": 512, "N": 256, "K": 128}
        )
        cmd = ex._launch_cmd(v)
        self.assertIn("rocke.run_manifest", cmd)
        self.assertEqual(cmd[-3:], ["--shape", "512,256,128", "--verify"])
        self.assertIn("/t/a.hsaco", cmd)


def _planned(cache_key, shape=(8, 16, 32)):
    return SimpleNamespace(
        cache_key=cache_key,
        shape=SimpleNamespace(M=shape[0], N=shape[1], K=shape[2]),
    )


def _build(cache_key, *, ok=True, error=""):
    return SimpleNamespace(
        cache_key=cache_key,
        ok=ok,
        error=error,
        kernel_name=f"kernel_{cache_key}",
        hsaco_path=f"/{cache_key}.hsaco",
        manifest_path=f"/{cache_key}.json",
    )


class TestBuildValidation(unittest.TestCase):
    def test_all_builds_succeed(self):
        plan = SimpleNamespace(variants=[_planned("a"), _planned("b")])
        variants = ex._validated_variants(plan, [_build("a"), _build("b")])
        self.assertEqual([variant.cache_key for variant in variants], ["a", "b"])

    def test_partial_build_failure_is_reported(self):
        plan = SimpleNamespace(variants=[_planned("a"), _planned("b")])
        with self.assertRaises(RuntimeError) as caught:
            ex._validated_variants(
                plan, [_build("a"), _build("b", ok=False, error="compiler failed")]
            )
        self.assertIn("1 of 2", str(caught.exception))
        self.assertIn("b: compiler failed", str(caught.exception))

    def test_all_builds_failed_is_reported(self):
        plan = SimpleNamespace(variants=[_planned("a"), _planned("b")])
        with self.assertRaisesRegex(RuntimeError, "2 of 2"):
            ex._validated_variants(
                plan,
                [
                    _build("a", ok=False, error="first failed"),
                    _build("b", ok=False, error="second failed"),
                ],
            )

    def test_missing_build_record_is_reported(self):
        plan = SimpleNamespace(variants=[_planned("a")])
        with self.assertRaisesRegex(RuntimeError, "a: missing build record"):
            ex._validated_variants(plan, [])

    def test_empty_plan_is_reported_separately(self):
        plan = SimpleNamespace(variants=[])
        with self.assertRaisesRegex(RuntimeError, "no applicable variants"):
            ex._validated_variants(plan, [])


class TestMain(unittest.TestCase):
    def test_build_failure_exits_nonzero(self):
        original = ex.profile_sweep

        def fail_sweep(*args, **kwargs):
            raise RuntimeError("1 of 1 planned variants failed to build")

        ex.profile_sweep = fail_sweep
        try:
            with self.assertRaisesRegex(SystemExit, "profile_gemm_sweep: 1 of 1"):
                ex.main(["--arch", "gfx950", "--shape", "8x8x8"])
        finally:
            ex.profile_sweep = original


class TestProfileVariants(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self._orig = ex._harness.profile
        ex._harness.profile = _fake_record

    def tearDown(self):
        ex._harness.profile = self._orig
        self._tmp.cleanup()

    def test_one_record_per_variant_stored(self):
        variants = [
            ex.Variant(
                "ck_a", "sym_a", "/t/a.hsaco", "/t/a.json", {"M": 8, "N": 8, "K": 8}
            ),
            ex.Variant(
                "ck_b", "sym_b", "/t/b.hsaco", "/t/b.json", {"M": 16, "N": 16, "K": 16}
            ),
        ]
        recs = ex.profile_variants(variants, "gfx950", repeats=3, cache=self.cache)
        self.assertEqual(len(recs), 2)
        self.assertEqual(len(store.load(cache=self.cache)), 2)  # one per variant
        # identity label = cache_key
        ids = {schema.identity(r)[2] for r in store.load(cache=self.cache)}
        self.assertEqual(ids, {"ck_a", "ck_b"})

    def test_self_check_across_two_runs(self):
        variants = [
            ex.Variant(
                "ck_a", "sym_a", "/t/a.hsaco", "/t/a.json", {"M": 8, "N": 8, "K": 8}
            )
        ]
        # run 1 (busy=100), run 2 (busy=200 -> regression)
        ex._harness.profile = lambda *a, **k: _fake_record(*a, **k, _busy=[100])
        ex.profile_variants(variants, "gfx950", repeats=1, cache=self.cache)
        ex._harness.profile = lambda *a, **k: _fake_record(*a, **k, _busy=[200])
        ex.profile_variants(variants, "gfx950", repeats=1, cache=self.cache)

        ident = schema.identity(store.load(cache=self.cache)[-1])
        result = selfcheck.check_history(store.load(cache=self.cache), ident)
        self.assertEqual(result["verdict"], "regressed")
        self.assertAlmostEqual(result["pct_change"], 100.0)


class TestParseShape(unittest.TestCase):
    def test_parse_shape(self):
        self.assertEqual(ex._parse_shape("512x256x128"), (512, 256, 128, ""))
        self.assertEqual(ex._parse_shape("64x64x64:tiny"), (64, 64, 64, "tiny"))

    def test_parse_shape_bad(self):
        with self.assertRaises(SystemExit):
            ex._parse_shape("nope")


if __name__ == "__main__":
    unittest.main()
