#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the gfx1250 preshuffle GEMM reject rules.

The rules live in two copies (tile_engine gemm_validation_utils.py and
dispatcher codegen_common.py, which cannot import each other); this file pins
them identical and checks the codegen emits nothing for rejected configs.
test_preshuffle_pipeline_matrix.py covers the accepted sweep end to end.

Run: python3 -m pytest tests/test_preshuffle_gfx1250_pipelines.py -v
"""

import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
CK_DIR = DISPATCHER_DIR.parent
TE_GEMM_DIR = CK_DIR / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

import codegen_common as cc  # noqa: E402
from unified_gemm_codegen import GemmVariant, UnifiedGemmCodegen  # noqa: E402


def _load_te_validation():
    spec = importlib.util.spec_from_file_location(
        "te_gemm_validation_utils", TE_GEMM_DIR / "gemm_validation_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TE = _load_te_validation()
ALL_PIPELINES = ("preshufflev2",) + cc.PRESHUFFLE_GFX1250_PIPELINES + ("compv3",)
TILE = dict(
    tile_m=128, tile_n=128, tile_k=64, warp_m=1, warp_n=4, warp_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
)


def _traits(pipeline):
    """The scheduler and epilogue the pipeline accepts."""
    if pipeline.startswith("comp_tdm"):
        return "intrawave", "tdm"
    return ("intrawave" if pipeline.startswith("comp_") else "default"), "cshuffle"


def _full_pads(pipeline):
    """comp_async on gfx1250 is emitted only fully padded, the rest unpadded."""
    return pipeline == "comp_async"


def _single_config_json(pipeline, pad=False, warp_tile_k=32):
    scheduler, epilogue = _traits(pipeline)
    return {
        "tile_config": {k: [v] for k, v in {**TILE, "warp_tile_k": warp_tile_k}.items()},
        "trait_config": {
            "pipeline": [pipeline], "scheduler": [scheduler], "epilogue": [epilogue],
            "pad_m": [pad], "pad_n": [pad], "pad_k": [pad], "persistent": [False],
        },
    }


def _generate(pipeline, gpu_target, layout="rcr", pad=None, datatype="fp16", warp_tile_k=32):
    pad = _full_pads(pipeline) if pad is None else pad
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cfg = root / "cfg.json"
        cfg.write_text(json.dumps(_single_config_json(pipeline, pad, warp_tile_k)))
        gen = UnifiedGemmCodegen(
            root / "out", datatype=datatype, layout=layout, gpu_target=gpu_target,
            config_file=cfg, variants=[GemmVariant.PRESHUFFLE],
        )
        result = gen.generate_all(parallel=False)
        return [Path(k).read_text() for k in result["kernels"]]


class TestRejectHelperParity(unittest.TestCase):
    """The TE and dispatcher reject helpers must agree everywhere."""

    def test_constants_match(self):
        self.assertEqual(cc.TDM_PAD_REJECT_REASON, TE.TDM_PAD_REJECT_REASON)
        self.assertEqual(cc.PRESHUFFLE_PAD_NK_REJECT_REASON, TE.PRESHUFFLE_PAD_NK_REJECT_REASON)
        self.assertEqual(set(cc.PRESHUFFLE_GFX1250_PIPELINES),
                         set(TE.GEMM_PRESHUFFLE_GFX1250_PIPELINES))
        self.assertEqual(set(cc.PACKED_B_PIPELINES), set(TE.GEMM_PACKED_B_PIPELINES))

    def test_helpers_agree_on_full_grid(self):
        grid = itertools.product(
            ALL_PIPELINES,
            ("", "gfx950", "gfx1250", "gfx1250:sramecc+"),
            ("", "rcr", "rrr"),
            itertools.product((False, True, "false", "True"), repeat=3),
            (None, 2, 4),
            ("", "fp16", "fp8"),
            ("", "default", "intrawave"),
            ("", "default", "cshuffle", "tdm"),
            (False, True),
            (None, 32, 128),
        )
        for pipeline, arch, layout, pads, waves, dtype, sched, epi, persistent, wtk in grid:
            args = (pipeline, arch, layout, *pads, waves, dtype, sched, epi, persistent, wtk)
            self.assertEqual(cc.preshuffle_pipeline_reject_reason(*args),
                             TE.preshuffle_pipeline_reject_reason(*args), args)


class TestRejectRules(unittest.TestCase):
    reject = staticmethod(cc.preshuffle_pipeline_reject_reason)

    def _ok(self, pipeline, **kw):
        scheduler, epilogue = _traits(pipeline)
        pad = _full_pads(pipeline)
        kw = {"gpu_target": "gfx1250", "layout": "rcr", "num_waves": 4,
              "scheduler": scheduler, "epilogue": epilogue,
              "pad_m": pad, "pad_n": pad, "pad_k": pad, **kw}
        return self.reject(pipeline, **kw)

    def test_accepted_on_gfx1250_rcr(self):
        for pipeline, dtype in itertools.product(cc.PRESHUFFLE_GFX1250_PIPELINES,
                                                 ("fp16", "bf16")):
            self.assertEqual(self._ok(pipeline, dtype=dtype), "", pipeline)

    def test_preshufflev2_keeps_its_old_contract(self):
        for arch, layout in itertools.product(("gfx942", "gfx950"), ("rcr", "rrr")):
            self.assertEqual(self.reject("preshufflev2", arch, layout, True, True, True, 8), "")
        self.assertEqual(self.reject("preshufflev2", "gfx1250", "rrr", True, False, False, 8), "")
        self.assertIn("scheduler", self.reject("preshufflev2", scheduler="intrawave"))
        self.assertIn("tdm epilogue", self.reject("preshufflev2", epilogue="tdm"))

    def test_gfx1250_rcr_only(self):
        for pipeline in cc.PRESHUFFLE_GFX1250_PIPELINES:
            self.assertIn("requires gfx1250", self._ok(pipeline, gpu_target="gfx950"))
            self.assertIn("requires layout", self._ok(pipeline, layout="rrr"))

    def test_scheduler_epilogue_and_persistent(self):
        for pipeline in cc.PRESHUFFLE_GFX1250_PIPELINES:
            scheduler, epilogue = _traits(pipeline)
            other = "default" if scheduler == "intrawave" else "intrawave"
            self.assertIn("scheduler", self._ok(pipeline, scheduler=other))
            self.assertIn("epilogue", self._ok(pipeline, epilogue="default"))
            self.assertIn("persistent", self._ok(pipeline, persistent=True))

    def test_tdm_requires_no_padding(self):
        for pipeline in ("preshuffle_tdm",) + cc.TDM_PIPELINES:
            for pads in ((True, False, False), (False, True, False), (False, False, True)):
                self.assertEqual(self._ok(pipeline, pad_m=pads[0], pad_n=pads[1], pad_k=pads[2]),
                                 cc.TDM_PAD_REJECT_REASON)

    def test_string_pad_flags(self):
        for pipeline in ("preshuffle_tdm",) + cc.TDM_PIPELINES:
            self.assertEqual(self._ok(pipeline, pad_m="false", pad_n="False", pad_k="false"), "")
            self.assertEqual(self._ok(pipeline, pad_m="true"), cc.TDM_PAD_REJECT_REASON)

    def test_gfx1250_packed_b_rejects_pad_n_pad_k(self):
        self.assertEqual(self._ok("preshufflev2", pad_m=True), "")
        for pads in ({"pad_n": True}, {"pad_k": True}, {"pad_k": "true"}):
            self.assertEqual(self._ok("preshufflev2", **pads),
                             cc.PRESHUFFLE_PAD_NK_REJECT_REASON, pads)
        # Off gfx1250 and at trait level without an arch, preshufflev2 pads stay allowed.
        for arch in ("", "gfx942", "gfx950"):
            self.assertEqual(self._ok("preshufflev2", gpu_target=arch, pad_n=True), "", arch)

    def test_comp_tdm_v2_requires_four_waves(self):
        self.assertIn("4 waves", self._ok("comp_tdm_v2", num_waves=2))

    def test_comp_async_requires_full_padding(self):
        for pads in ({"pad_m": False}, {"pad_n": False}, {"pad_k": "false"}):
            self.assertEqual(self._ok("comp_async", **pads),
                             cc.GFX1250_COMP_ASYNC_PAD_REJECT_REASON, pads)

    def test_comp_async_8bit_warp_tile_k(self):
        for dtype in cc.GFX1250_COMP_ASYNC_8BIT_DTYPES:
            self.assertEqual(self._ok("comp_async", dtype=dtype, warp_tile_k=64),
                             cc.GFX1250_COMP_ASYNC_8BIT_WARP_TILE_K_REJECT_REASON)
            self.assertEqual(self._ok("comp_async", dtype=dtype, warp_tile_k=128), "")
            self.assertEqual(self._ok("comp_tdm", dtype=dtype, warp_tile_k=64), "")
        self.assertEqual(self._ok("comp_async", dtype="fp16", warp_tile_k=32), "")


class TestCodegen(unittest.TestCase):
    def test_rejected_configs_emit_nothing(self):
        cases = [("preshuffle_tdm", "gfx942", "rcr", False),
                 ("preshuffle_tdm", "gfx1250", "rrr", False),
                 ("preshuffle_tdm", "gfx1250", "rcr", True),
                 ("comp_tdm", "gfx942", "rcr", False),
                 ("comp_async", "gfx1250", "rrr", False),
                 ("comp_tdm_v2", "gfx1250", "rcr", True),
                 ("comp_async", "gfx1250", "rcr", False),
                 ("preshufflev2", "gfx1250", "rcr", True)]
        for pipeline, arch, layout, pad in cases:
            with self.subTest(pipeline=pipeline, arch=arch, layout=layout, pad=pad):
                self.assertEqual(_generate(pipeline, arch, layout, pad), [])
        for dtype in cc.GFX1250_COMP_ASYNC_8BIT_DTYPES:
            with self.subTest(pipeline="comp_async", dtype=dtype):
                self.assertEqual(_generate("comp_async", "gfx1250", datatype=dtype,
                                           warp_tile_k=64), [])

    def test_accepted_configs_emit_one_kernel(self):
        # Only preshufflev2 / preshuffle_tdm read host-packed B.
        for pipeline in cc.PRESHUFFLE_GFX1250_PIPELINES:
            with self.subTest(pipeline=pipeline):
                kernels = _generate(pipeline, "gfx1250")
                self.assertEqual(len(kernels), 1)
                packed = str(pipeline in cc.PACKED_B_PIPELINES).lower()
                self.assertIn(f"static constexpr bool Preshuffle = {packed};", kernels[0])


if __name__ == "__main__":
    unittest.main()
