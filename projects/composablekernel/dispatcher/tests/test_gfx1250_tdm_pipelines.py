#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU tests for the gfx1250 comp_async / comp_tdm / comp_tdm_v2 pipelines and the
tdm epilogue in the dispatcher: arch filter rules, codegen type maps, LDS
budgets, generated kernel headers, the Python sweep gate and the C++ enums.

Run: python3 -m pytest tests/test_gfx1250_tdm_pipelines.py -v
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CODEGEN_DIR = DISPATCHER_DIR / "codegen"
INCLUDE_DIR = DISPATCHER_DIR / "include"
sys.path.insert(0, str(CODEGEN_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from arch_filter import ArchFilter  # noqa: E402
from arch_specs_generated import get_lds_limit  # noqa: E402
from codegen_common import CommonTypeMappings  # noqa: E402

TDM_TILE = dict(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)


class TestArchFilter(unittest.TestCase):
    def _valid(self, arch, **kw):
        args = dict(TDM_TILE)
        args.update(kw)
        return ArchFilter(arch).is_kernel_valid(**args)

    def test_tdm_valid_on_gfx1250(self):
        for pipe in ("comp_tdm", "comp_tdm_v2"):
            self.assertTrue(self._valid("gfx1250", pipeline=pipe, epilogue="tdm"))

    def test_tdm_rejected_off_gfx1250(self):
        for arch in ("gfx942", "gfx950", "gfx90a"):
            for pipe in ("comp_tdm", "comp_tdm_v2"):
                self.assertFalse(
                    self._valid(arch, pipeline=pipe, epilogue="tdm"), (arch, pipe)
                )

    def test_tdm_v2_requires_four_waves(self):
        self.assertFalse(
            self._valid("gfx1250", pipeline="comp_tdm_v2", epilogue="tdm", warp_m=4)
        )
        self.assertTrue(
            self._valid("gfx1250", pipeline="comp_tdm", epilogue="tdm", warp_m=4)
        )

    def test_tdm_pipeline_epilogue_pairing(self):
        self.assertFalse(self._valid("gfx1250", pipeline="compv3", epilogue="tdm"))
        self.assertFalse(
            self._valid("gfx1250", pipeline="comp_tdm", epilogue="cshuffle")
        )

    def test_tdm_rejects_interwave(self):
        self.assertFalse(
            self._valid(
                "gfx1250", pipeline="comp_tdm", epilogue="tdm", scheduler="interwave"
            )
        )

    def test_comp_async_not_gated_by_shared_filter(self):
        # grouped_conv uses comp_async on gfx950, so the shared arch filter must
        # not reject it; the GEMM gfx1250 gate lives in unified_gemm_codegen.
        self.assertTrue(
            self._valid(
                "gfx950",
                pipeline="comp_async",
                epilogue="cshuffle",
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            )
        )

    def test_existing_pipelines_unchanged(self):
        for arch in ("gfx942", "gfx950"):
            self.assertTrue(
                self._valid(
                    arch,
                    pipeline="compv3",
                    epilogue="cshuffle",
                    warp_tile_m=32,
                    warp_tile_n=32,
                    warp_tile_k=16,
                )
            )


class TestTypeMappings(unittest.TestCase):
    def test_pipeline_maps(self):
        m = CommonTypeMappings
        self.assertEqual(m.PIPELINE_TO_CK["comp_tdm"], "GemmPipelineAgBgCrCompTDMV1")
        self.assertEqual(m.PIPELINE_TO_CK["comp_tdm_v2"], "GemmPipelineAgBgCrCompTDMV2")
        self.assertEqual(
            m.PIPELINE_TO_BASE["comp_tdm"], "BaseGemmPipelineAgBgCrCompTDM"
        )
        self.assertEqual(
            m.PIPELINE_TO_BASE["comp_tdm_v2"], "BaseGemmPipelineAgBgCrCompTDM"
        )
        self.assertEqual(m.PIPELINE_TO_DISPATCHER["comp_async"], "Pipeline::CompAsync")
        self.assertEqual(m.PIPELINE_TO_DISPATCHER["comp_tdm"], "Pipeline::CompTDMV1")
        self.assertEqual(m.PIPELINE_TO_DISPATCHER["comp_tdm_v2"], "Pipeline::CompTDMV2")
        self.assertEqual(m.EPILOGUE_TO_DISPATCHER["tdm"], "Epilogue::Tdm")

    def test_legacy_maps_unchanged(self):
        m = CommonTypeMappings
        self.assertEqual(m.PIPELINE_TO_CK["compv3"], "GemmPipelineAgBgCrCompV3")
        self.assertEqual(m.PIPELINE_TO_DISPATCHER["mem"], "Pipeline::Mem")
        self.assertEqual(m.EPILOGUE_TO_DISPATCHER["cshuffle"], "Epilogue::CShuffle")


class TestLdsBudget(unittest.TestCase):
    def test_tdm_shares_comp_async_budget(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            for pipe in ("comp_tdm", "comp_tdm_v2"):
                self.assertEqual(
                    get_lds_limit(arch, pipe), get_lds_limit(arch, "comp_async")
                )

    def test_gfx1250_budget(self):
        self.assertEqual(get_lds_limit("gfx1250", "comp_tdm"), 163840)


def _run_codegen(outdir, arch, variant, cfg_path, layout="rcr"):
    cmd = [
        sys.executable,
        str(CODEGEN_DIR / "unified_gemm_codegen.py"),
        "--output-dir",
        str(outdir),
        "--datatype",
        "fp16",
        "--layout",
        layout,
        "--gpu-target",
        arch,
        "--config",
        str(cfg_path),
        "--variants",
        variant,
        "--no-parallel",
    ]
    subprocess.run(cmd, check=True, capture_output=True, cwd=str(CODEGEN_DIR))
    return sorted(p.name for p in Path(outdir).glob("*.hpp"))


class TestUnifiedCodegenGolden(unittest.TestCase):
    CONFIG = {
        "tile_config": {
            "tile_m": [128],
            "tile_n": [128],
            "tile_k": [64],
            "warp_m": [2, 4],
            "warp_n": [2],
            "warp_k": [1],
            "warp_tile_m": [16],
            "warp_tile_n": [16],
            "warp_tile_k": [32],
        },
        "trait_config": {
            "pipeline": ["comp_tdm", "comp_tdm_v2", "comp_async", "compv3"],
            "epilogue": ["tdm", "cshuffle"],
            "scheduler": ["intrawave"],
            # TDM only accepts pads False; pad_k True keeps comp_async and
            # compv3 padded coverage and proves TDM drops the padded combos.
            "pad_m": [False],
            "pad_n": [False],
            "pad_k": [False, True],
            "persistent": [False, True],
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tdm_codegen_"))
        cls.cfg = cls.tmp / "cfg.json"
        cls.cfg.write_text(json.dumps(cls.CONFIG))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _gen(self, arch, variant, layout="rcr"):
        out = self.tmp / f"{arch}_{variant}_{layout}"
        return out, _run_codegen(out, arch, variant, self.cfg, layout)

    def test_gfx1250_standard_kernel_set(self):
        out, names = self._gen("gfx1250", "standard")
        tdm = [n for n in names if "_comp_tdm_tdm_" in n]
        tdm_v2 = [n for n in names if "_comp_tdm_v2_tdm_" in n]
        async_ = [n for n in names if "_comp_async_" in n]
        # comp_tdm: 2 wave layouts, non-persistent, unpadded only.
        self.assertEqual(len(tdm), 2)
        # comp_tdm_v2: 4 waves only.
        self.assertEqual(len(tdm_v2), 1)
        self.assertIn("2x2x1", tdm_v2[0])
        # comp_async: 2 wave layouts x persistent on/off x pad_k, cshuffle only.
        self.assertEqual(len(async_), 8)
        self.assertTrue(
            any("_comp_async_cshuffle_intrawave_False_False_True_" in n for n in names)
        )
        for n in tdm + tdm_v2:
            self.assertIn("_tdm_intrawave_False_False_False_False_", n)
        self.assertFalse(
            any("_compv3_tdm_" in n or "_comp_async_tdm_" in n for n in names)
        )

    def test_gfx1250_tdm_header_contents(self):
        out, names = self._gen("gfx1250", "standard")
        name = next(n for n in names if "_comp_tdm_tdm_" in n and "2x2x1" in n)
        text = (out / name).read_text()
        self.assertIn('#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"', text)
        self.assertIn("GemmPipelineAgBgCrCompTDMV1<", text)
        self.assertIn("BaseGemmPipelineAgBgCrCompTDM<", text)
        self.assertIn("TdmEpilogue<", text)
        self.assertNotIn("CShuffleEpilogue<", text)
        self.assertIn("static constexpr bool DoubleSmemBuffer = true;", text)
        self.assertIn("args.k_batch != 1", text)
        wrapper = (
            out / "dispatcher_wrappers" / f"dispatcher_wrapper_{name}"
        ).read_text()
        self.assertIn("Pipeline::CompTDMV1", wrapper)
        self.assertIn("Epilogue::Tdm", wrapper)
        self.assertIn("key.algorithm.double_buffer = true;", wrapper)

    def test_gfx1250_compv3_header_has_no_tdm(self):
        out, names = self._gen("gfx1250", "standard")
        name = next(n for n in names if "_compv3_" in n)
        text = (out / name).read_text()
        self.assertNotIn("tdm_epilogue.hpp", text)
        self.assertNotIn("TDM", text)
        self.assertIn("static constexpr bool DoubleSmemBuffer = false;", text)

    def test_gfx942_only_compv3(self):
        for variant in ("standard", "batched"):
            _, names = self._gen("gfx942", variant)
            self.assertTrue(names)
            self.assertTrue(all("_compv3_cshuffle_" in n for n in names), names)

    def test_gfx1250_batched_has_tdm(self):
        _, names = self._gen("gfx1250", "batched")
        self.assertTrue(any("_comp_tdm_tdm_" in n for n in names))
        self.assertTrue(any("_comp_tdm_v2_tdm_" in n for n in names))

    def test_gfx1250_batched_comp_async_epilogue_double_smem(self):
        out, names = self._gen("gfx1250", "batched")
        name = next(n for n in names if "_comp_async_cshuffle_" in n)
        text = (out / name).read_text()
        self.assertIn(
            "UniversalGemmProblem::TransposeC, 1, false, 1, 1, DoubleSmemBuffer", text
        )
        name = next(n for n in names if "_compv3_cshuffle_" in n)
        self.assertNotIn(", DoubleSmemBuffer>", (out / name).read_text())

    def test_comp_async_requires_cshuffle_intrawave(self):
        cfg = json.loads(json.dumps(self.CONFIG))
        cfg["trait_config"]["pipeline"] = ["comp_async"]
        cfg["trait_config"]["epilogue"] = ["cshuffle", "default", "tdm"]
        cfg["trait_config"]["scheduler"] = ["intrawave", "interwave"]
        path = self.tmp / "async_cfg.json"
        path.write_text(json.dumps(cfg))
        names = _run_codegen(self.tmp / "async_only", "gfx1250", "standard", path)
        self.assertTrue(names)
        for n in names:
            self.assertIn("_comp_async_cshuffle_intrawave_", n)

    def test_tdm_rejects_any_pad(self):
        for pad in ("pad_m", "pad_n", "pad_k"):
            cfg = json.loads(json.dumps(self.CONFIG))
            cfg["trait_config"]["pad_m"] = [False]
            cfg["trait_config"]["pad_n"] = [False]
            cfg["trait_config"]["pad_k"] = [False]
            cfg["trait_config"][pad] = [True]
            path = self.tmp / f"tdm_{pad}.json"
            path.write_text(json.dumps(cfg))
            for variant in ("standard", "batched"):
                names = _run_codegen(
                    self.tmp / f"tdm_{pad}_{variant}", "gfx1250", variant, path
                )
                self.assertFalse(any("_comp_tdm" in n for n in names), (pad, names))
                self.assertTrue(any("_comp_async_" in n for n in names), pad)

    def test_comp_async_rc_layout_only(self):
        for layout in ("rrr", "crr", "ccr"):
            _, names = self._gen("gfx1250", "standard", layout)
            self.assertFalse(any("_comp_async_" in n for n in names), layout)
            # TDM is not layout-gated.
            self.assertTrue(any("_comp_tdm_tdm_" in n for n in names), layout)
            self.assertTrue(any("_comp_tdm_v2_tdm_" in n for n in names), layout)

    def test_gfx1250_reject_reason_strings(self):
        import unified_gemm_codegen as ugc

        self.assertIn("pad_m=pad_n=pad_k=False", ugc.TDM_PAD_REJECT_REASON)
        self.assertIn(
            "A row-major and B col-major", ugc.GFX1250_COMP_ASYNC_LAYOUT_REJECT_REASON
        )

    def test_multi_d_no_gfx1250_pipelines(self):
        _, names = self._gen("gfx1250", "multi_d")
        self.assertTrue(names)
        self.assertFalse(any("comp_async" in n or "comp_tdm" in n for n in names))

    def test_grouped_no_gfx1250_pipelines(self):
        _, names = self._gen("gfx1250", "grouped")
        self.assertTrue(names)
        self.assertFalse(any("comp_async" in n or "comp_tdm" in n for n in names))

    def test_stream_k_no_comp_async(self):
        _, names = self._gen("gfx1250", "stream_k")
        self.assertTrue(names)
        self.assertFalse(any("comp_async" in n or "comp_tdm" in n for n in names))


class TestGemmUtilsSweepGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gemm_utils
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"gemm_utils not importable: {exc}")
        cls.gate = staticmethod(gemm_utils._gfx1250_pipeline_supported)

    def _ok(
        self,
        pipe,
        epi="cshuffle",
        arch="gfx1250",
        sched="intrawave",
        persist=False,
        waves=(2, 2, 1),
        variant="standard",
        **kw,
    ):
        return self.gate(pipe, sched, epi, persist, *waves, arch, variant, **kw)

    def test_legacy_passthrough(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            self.assertTrue(self._ok("compv3", arch=arch))
            self.assertTrue(self._ok("compv4", arch=arch, sched="interwave"))

    def test_tdm_rules(self):
        self.assertTrue(self._ok("comp_tdm", "tdm"))
        self.assertTrue(self._ok("comp_tdm", "tdm", variant="batched"))
        self.assertTrue(self._ok("comp_tdm_v2", "tdm"))
        self.assertFalse(self._ok("comp_tdm", "tdm", arch="gfx950"))
        self.assertFalse(self._ok("comp_tdm", "cshuffle"))
        self.assertFalse(self._ok("comp_tdm", "tdm", persist=True))
        self.assertFalse(self._ok("comp_tdm", "tdm", sched="interwave"))
        for variant in ("multi_d", "grouped", "multi_abd", "stream_k"):
            self.assertFalse(self._ok("comp_tdm", "tdm", variant=variant), variant)
        self.assertFalse(self._ok("comp_tdm_v2", "tdm", waves=(4, 2, 1)))
        self.assertFalse(self._ok("compv3", "tdm"))

    def test_comp_async_rules(self):
        self.assertTrue(self._ok("comp_async"))
        self.assertTrue(self._ok("comp_async", persist=True))
        self.assertTrue(self._ok("comp_async", variant="batched"))
        self.assertFalse(self._ok("comp_async", arch="gfx942"))
        for variant in ("multi_d", "grouped", "multi_abd", "stream_k"):
            self.assertFalse(self._ok("comp_async", variant=variant), variant)
        self.assertFalse(self._ok("comp_async", sched="interwave"))
        self.assertFalse(self._ok("comp_async", "tdm"))
        self.assertFalse(self._ok("comp_async", "default"))
        self.assertFalse(self._ok("comp_async", "cshuffle", sched="interwave"))

    def test_tdm_pad_rule(self):
        for pipe in ("comp_tdm", "comp_tdm_v2"):
            self.assertTrue(
                self._ok(pipe, "tdm", pad_m=False, pad_n=False, pad_k=False)
            )
            for pad in ("pad_m", "pad_n", "pad_k"):
                self.assertFalse(self._ok(pipe, "tdm", **{pad: True}), (pipe, pad))
        # Padding does not affect comp_async or legacy pipelines.
        self.assertTrue(self._ok("comp_async", pad_m=True, pad_n=True, pad_k=True))
        self.assertTrue(self._ok("compv3", pad_m=True, pad_n=True, pad_k=True))

    def test_comp_async_layout_rule(self):
        self.assertTrue(self._ok("comp_async", layout="rcr"))
        self.assertTrue(self._ok("comp_async", layout=""))
        for layout in ("rrr", "crr", "ccr"):
            self.assertFalse(self._ok("comp_async", layout=layout), layout)
            self.assertTrue(self._ok("comp_tdm", "tdm", layout=layout), layout)
            self.assertTrue(self._ok("compv3", layout=layout), layout)

    def test_expand_sweep_tdm_unpadded(self):
        import gemm_utils

        cfg = json.loads(json.dumps(TestUnifiedCodegenGolden.CONFIG))
        cfg["trait_config"]["pad_m"] = [False, True]
        cfg["trait_config"]["pad_n"] = [False, True]
        cfg["trait_config"]["pad_k"] = [False, True]
        # expand_sweep reads the Tile Engine {"values": [...]} schema.
        cfg = {
            section: {k: {"values": v} for k, v in entries.items()}
            for section, entries in cfg.items()
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.json"
            path.write_text(json.dumps(cfg))
            rcr = gemm_utils.expand_sweep(str(path), "gfx1250", "fp16", "rcr")
            rrr = gemm_utils.expand_sweep(str(path), "gfx1250", "fp16", "rrr")
        tdm = [c for c in rcr if c.pipeline in ("comp_tdm", "comp_tdm_v2")]
        self.assertTrue(tdm)
        for c in tdm:
            self.assertEqual((c.pad_m, c.pad_n, c.pad_k), (False, False, False))
        pads = {(c.pad_m, c.pad_n, c.pad_k) for c in rcr if c.pipeline == "comp_async"}
        self.assertEqual(len(pads), 8)
        self.assertFalse(any(c.pipeline == "comp_async" for c in rrr))
        self.assertTrue(any(c.pipeline == "comp_tdm" for c in rrr))


_CPP_PROBE = r"""
#include "ck_tile/dispatcher/arch_specs_generated.hpp"
#include <cstdio>
using namespace ck_tile::dispatcher;
using arch_specs::GpuArch;

// Existing enumerator values must not move (serialized keys depend on them).
static_assert(static_cast<int>(Pipeline::Mem) == 0, "");
static_assert(static_cast<int>(Pipeline::CompV3) == 3, "");
static_assert(static_cast<int>(Pipeline::PreShuffleV2) == 8, "");
static_assert(static_cast<int>(Pipeline::Wavelet) == 9, "");
static_assert(static_cast<int>(Pipeline::CompAsync) == 10, "");
static_assert(static_cast<int>(Pipeline::CompTDMV1) == 11, "");
static_assert(static_cast<int>(Pipeline::CompTDMV2) == 12, "");
static_assert(static_cast<int>(Epilogue::CShuffle) == 2, "");
static_assert(static_cast<int>(Epilogue::BiasActivation) == 5, "");
static_assert(static_cast<int>(Epilogue::Tdm) == 6, "");

int main()
{
    int fails = 0;
#define CHECK(c) do { if(!(c)) { std::printf("FAIL: %s\n", #c); ++fails; } } while(0)
    CHECK(to_string(Pipeline::CompAsync) == "comp_async");
    CHECK(to_string(Pipeline::CompTDMV1) == "comp_tdm");
    CHECK(to_string(Pipeline::CompTDMV2) == "comp_tdm_v2");
    CHECK(to_string(Epilogue::Tdm) == "tdm");
    CHECK(string_to_pipeline("comp_async") == Pipeline::CompAsync);
    CHECK(string_to_pipeline("comp_tdm") == Pipeline::CompTDMV1);
    CHECK(string_to_pipeline("comp_tdm_v2") == Pipeline::CompTDMV2);
    CHECK(string_to_pipeline("compv3") == Pipeline::CompV3);
    CHECK(string_to_pipeline("no_such_pipeline") == Pipeline::Mem);
    CHECK(string_to_epilogue("tdm") == Epilogue::Tdm);
    CHECK(string_to_epilogue("cshuffle") == Epilogue::CShuffle);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_1250, Pipeline::CompTDMV1) == 163840);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_1250, Pipeline::CompTDMV2) == 163840);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_1250, Pipeline::CompAsync) == 163840);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_950, Pipeline::CompTDMV1) == 81920);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_942, Pipeline::CompTDMV1) == 32768);
    CHECK(arch_specs::get_lds_capacity(GpuArch::GFX_942, Pipeline::CompV3) ==
          arch_specs::get_lds_capacity(GpuArch::GFX_942, Pipeline::Mem));
    CHECK(arch_specs::is_trait_unsupported(Pipeline::CompTDMV1, Epilogue::Tdm, Scheduler::Interwave));
    CHECK(arch_specs::is_trait_unsupported(Pipeline::CompAsync, Epilogue::CShuffle, Scheduler::Interwave));
    CHECK(!arch_specs::is_trait_unsupported(Pipeline::CompTDMV1, Epilogue::Tdm, Scheduler::Intrawave));
    return fails == 0 ? 0 : 1;
}
"""


@unittest.skipIf(shutil.which("g++") is None, "g++ not available")
class TestCppEnums(unittest.TestCase):
    def test_enum_values_strings_and_lds(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "probe.cpp"
            exe = Path(td) / "probe"
            src.write_text(_CPP_PROBE)
            build = subprocess.run(
                ["g++", "-std=c++17", "-I", str(INCLUDE_DIR), str(src), "-o", str(exe)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
