#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests of comp_tdm_scheduled in tile_engine and the dispatcher.

The tuning token, the tile_engine listing and generation, the dispatcher codegen and the
compile flags. No GPU or hipcc is required.
"""

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_DISP = Path(__file__).resolve().parent.parent
_CK = _DISP.parent
_GEMM = _CK / "tile_engine" / "ops" / "gemm"
_MX = _GEMM / "mx_gemm"
for _p in (_DISP / "python", _DISP / "codegen", _GEMM, _MX):
    sys.path.insert(0, str(_p))

import gemm_validation_utils as vu  # noqa: E402
import mx_gemm_utils as mx  # noqa: E402
import unified_mx_gemm_codegen as codegen  # noqa: E402

sched = vu.load_mx_tdm_scheduled()
BUILDER = _MX / "mx_gemm_instance_builder.py"
TILE = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 16,
    "warp_tile_k": 128,
}


class TestTuning(unittest.TestCase):
    def test_token_round_trip(self):
        tunings = sched.expand_tuning_config(
            {
                "cluster_dim": {"values": [[4, 4], [2, 1, 1]]},
                "gl2_prefetch_distance": {"values": [4, 2]},
                "split_b_lds_read": {"values": [True, False]},
                "interleave_lds_regions": {"values": [True, False]},
                "store_width": {"values": [0, 4]},
            }
        )
        self.assertEqual(len(tunings), 32)
        for t in tunings:
            token = sched.encode_tuning(t)
            self.assertNotRegex(token, r"[\s;|\"'$]")
            self.assertEqual(sched.decode_tuning(token), t)

    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"gl2": 4})
        # The tile-count parity is the pipeline's TailNumber, not an instance trait.
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"odd_tiles": 1})

    def test_instance_rules(self):
        def reason(t):
            return sched.validate_instance(
                sched.spec_from_tile(TILE, t, out_dtype="fp16"), t
            )

        self.assertEqual(reason({}), "ok")
        self.assertEqual(reason({"prefetch_lds_read": False}), "split_b_read")
        self.assertEqual(reason({"store_width": 16}), "store_width")
        self.assertEqual(reason({"store_width": 3}), "store_width")
        self.assertEqual(reason({"tail_steps": 4}), "tail_steps")
        self.assertEqual(reason({"cluster_dim": [8, 4]}), "cluster_size")

    def test_cpp_spelling(self):
        t = {
            "cluster_dim": [2, 2],
            "interleave_lds_regions": False,
        }
        spec = sched.spec_from_tile(TILE, t, out_dtype="fp16")
        self.assertEqual(sched.cpp_cluster_sequence(t), "ck_tile::sequence<2, 2, 1>")
        self.assertEqual(
            sched.cpp_cluster_sequence({"cluster_dim": [1, 4]}),
            "ck_tile::sequence<4, 1, 1>",
        )
        self.assertEqual(
            sched.cpp_policy_type(t),
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, false>",
        )
        self.assertEqual(
            sched.cpp_kernel_traits(spec, {"edge": False, "tail_steps": 2}),
            "ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 2, 8>",
        )
        self.assertEqual(
            sched.instance_name_suffix(spec, {**t, "edge": False, "tail_steps": 2}),
            "_c2x2_lsi0_noedge_tail2",
        )

    def test_instance_runs_problem(self):
        tile = sched.tile_from_spec(sched.MxTdmScheduledSpec())
        spec = sched.spec_from_tile(tile, {}, M=1000, N=1000, K=4096, out_dtype="fp16")
        self.assertEqual(sched.instance_unsupported_reason(spec, {}), "")
        self.assertEqual(
            sched.instance_unsupported_reason(spec, {"edge": False}), "edge"
        )
        # Any tile count: 4096 / 512 is even, 4608 / 512 odd.
        odd = dataclasses.replace(spec, K=4096 + 512)
        self.assertEqual(sched.instance_unsupported_reason(odd, {}), "")
        tail = dataclasses.replace(spec, K=4096 + 256)
        self.assertEqual(sched.instance_unsupported_reason(tail, {}), "tail_steps")
        self.assertEqual(sched.instance_unsupported_reason(tail, {"tail_steps": 2}), "")


class TestValidationUtils(unittest.TestCase):
    def test_registered_for_gfx1250_only(self):
        self.assertIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx1250"])
        self.assertNotIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx950"])
        self.assertIn(
            "gfx1250",
            vu.mx_tdm_scheduled_reject_reason(
                *TILE.values(), "fp4", "fp4", "fp16", "gfx950"
            ),
        )

    def test_traits(self):
        ok = vu.is_trait_combination_valid
        self.assertTrue(
            ok(
                "comp_tdm_scheduled",
                "direct",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertTrue(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "cshuffle",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "direct",
                "intrawave",
                True,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "direct",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                True,
                False,
                False,
            )
        )
        self.assertFalse(ok("comp_tdm", "direct", "intrawave", False, "mx_gemm", "rcr"))

    def test_tiles(self):
        def valid(**tile):
            dims = [{**TILE, **tile}[k] for k in TILE]
            return vu.is_tile_config_valid(
                *dims,
                "fp4",
                "fp4",
                "fp16",
                "comp_tdm_scheduled",
                "rcr",
                "gfx1250",
                "mx_gemm",
            )

        self.assertTrue(valid())
        # Valid without SplitBLdsRead only: kept for the tuning_config product to decide.
        self.assertTrue(valid(tile_n=64, warp_m=1, warp_n=4))
        self.assertTrue(
            valid(tile_m=128, tile_n=16, warp_m=4, warp_n=1, warp_tile_m=16)
        )
        dims = [{**TILE, "tile_n": 64, "warp_m": 1, "warp_n": 4}[k] for k in TILE]
        self.assertIn(
            "split_b_read",
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {}
            ),
        )
        self.assertEqual(
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {"split_b_lds_read": False}
            ),
            "",
        )
        self.assertFalse(valid(tile_k=1024))  # LDS
        self.assertFalse(valid(warp_tile_n=32))
        self.assertFalse(valid(warp_k=2))
        ok, reason = vu.validate_gemm_mx(
            *TILE.values(), "fp4", "fp4", "fp16", "comp_tdm_scheduled", "rcr", "gfx1250"
        )
        self.assertTrue(ok, reason)


def _builder(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(BUILDER), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )


class TestTileEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _config(
        self,
        tuning_config=None,
        pipelines=("comp_tdm_scheduled",),
        epilogues=("direct",),
    ):
        config = json.loads(
            (_MX / "configs" / "comp_tdm_scheduled_gfx1250.json").read_text()
        )
        config["trait_config"]["pipeline"]["values"] = list(pipelines)
        config["trait_config"]["epilogue"]["values"] = list(epilogues)
        if tuning_config is None:
            config.pop("tuning_config")
        else:
            config["tuning_config"] = tuning_config
        path = self.dir / "config.json"
        path.write_text(json.dumps(config))
        return path

    def _list(self, config):
        _builder(
            "--working_path",
            str(self.dir),
            "--datatype",
            "fp4",
            "--layout",
            "rcr",
            "--config_json",
            str(config),
            "--gpu_target",
            "gfx1250",
            "--list_kernels",
        )
        lines = (self.dir / "mx_gemm_kernel_list.txt").read_text().splitlines()
        self.assertEqual(
            int((self.dir / "mx_gemm_kernel_count.txt").read_text()), len(lines)
        )
        return [line.split("|") for line in lines]

    def test_shipped_config(self):
        rows = self._list(_MX / "configs" / "comp_tdm_scheduled_gfx1250.json")
        self.assertEqual(len(rows), 10)
        names = [r[0] for r in rows]
        self.assertEqual(len(set(names)), len(names))
        for name, tile, trait, token, suffix in rows:
            self.assertEqual(
                trait, "comp_tdm_scheduled_direct_intrawave_False_False_False_False"
            )
            self.assertEqual(name, f"mx_gemm_fp4_rcr_{trait}_{tile}{suffix}")
            tuning = sched.decode_tuning(token)
            spec = sched.spec_from_tile(conv_tile(tile), tuning, out_dtype="fp16")
            self.assertEqual(sched.validate_instance(spec, tuning), "ok")
            self.assertEqual(sched.instance_name_suffix(spec, tuning), suffix)

    def test_tunings_filtered(self):
        rows = self._list(
            self._config(
                {
                    "prefetch_lds_read": {"values": [False, True]},
                    "split_b_lds_read": {"values": [True, False]},
                }
            )
        )
        tunings = {sched.encode_tuning(sched.decode_tuning(r[3])) for r in rows}
        for t in tunings:
            t = sched.decode_tuning(t)
            self.assertFalse(not t["prefetch_lds_read"] and t["split_b_lds_read"])
        self.assertEqual(len(tunings), 3)

    def test_mixed_pipelines(self):
        config = self._config(
            None, ("comp_tdm", "comp_tdm_scheduled"), ("tdm", "direct")
        )
        rows = self._list(config)
        self.assertTrue(any(len(r) == 3 and "_comp_tdm_tdm_" in r[0] for r in rows))
        self.assertTrue(
            all(len(r) == 5 for r in rows if "_comp_tdm_scheduled_" in r[0])
        )
        self.assertTrue(all(r[4] == "_c4x4" for r in rows if len(r) == 5))

    def test_gen_single(self):
        rows = self._list(_MX / "configs" / "comp_tdm_scheduled_gfx1250.json")
        name, tile, trait, token, suffix = next(
            r for r in rows if r[4].startswith("_c2x2")
        )
        _builder(
            "--working_path",
            str(self.dir),
            "--datatype",
            "fp4",
            "--layout",
            "rcr",
            "--config_json",
            str(_MX / "configs" / "comp_tdm_scheduled_gfx1250.json"),
            "--gpu_target",
            "gfx1250",
            "--gen_single",
            "--kernel_name",
            name,
            "--tile_config",
            tile,
            "--trait_combo",
            trait,
            "--tuning",
            token,
        )
        header = (
            self.dir / f"mx_gemm_single_{name[len('mx_gemm_') :]}.hpp"
        ).read_text()
        self.assertIn(f'KERNEL_NAME = "{name}"', header)
        self.assertIn("static constexpr bool KMajorScales = true;", header)
        self.assertIn("ck_tile::sequence<2, 2, 1>", header)
        self.assertIn("ck_tile::GemmTile2DNMajorPartitioner", header)
        self.assertIn("Kernel::MakeKernelArgs(args)", header)
        self.assertIn("using CDataType = ck_tile::fp16_t;", header)

    def test_invalid_single_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _builder(
                "--working_path",
                str(self.dir),
                "--datatype",
                "fp4",
                "--layout",
                "rcr",
                "--config_json",
                str(_MX / "configs" / "comp_tdm_scheduled_gfx1250.json"),
                "--gpu_target",
                "gfx1250",
                "--gen_single",
                "--kernel_name",
                "x",
                "--tile_config",
                "256x256x512_2x2x1_32x16x128",
                "--trait_combo",
                "comp_tdm_scheduled_direct_intrawave_False_False_False_False",
                "--tuning",
                sched.encode_tuning({"prefetch_lds_read": False}),
            )


def conv_tile(tile_str):
    groups = [[int(v) for v in g.split("x")] for g in tile_str.split("_")]
    keys = ("tile", "warp", "warp_tile")
    return {
        f"{k}_{d}": groups[i][j]
        for i, k in enumerate(keys)
        for j, d in enumerate("mnk")
    }


def _cfg(**extra):
    return {
        "datatype": "fp4",
        "layout": "rcr",
        "gpu_target": "gfx1250",
        "pipeline": "comp_tdm_scheduled",
        "tile_config": dict(TILE),
        **extra,
    }


class TestDispatcher(unittest.TestCase):
    def test_names(self):
        base = "mx_gemm_fp4_rcr_comp_tdm_scheduled_direct_intrawave_False_False_False_False_256x256x512_2x2x1_32x16x128"
        self.assertEqual(codegen.kernel_name(_cfg()), base + "_c4x4")
        self.assertEqual(
            codegen.kernel_name(
                _cfg(cluster_dim=[2, 2], gl2_prefetch_distance=2, tail_steps=1)
            ),
            base + "_c2x2_pf2_tail1",
        )

    def test_rejects(self):
        with self.assertRaisesRegex(ValueError, "split_b_read"):
            codegen.kernel_name(_cfg(prefetch_lds_read=False))
        with self.assertRaisesRegex(ValueError, "only for comp_tdm_scheduled"):
            codegen.kernel_name(_cfg(pipeline="comp_tdm", cluster_dim=[2, 2]))
        with self.assertRaisesRegex(ValueError, "epilogue"):
            codegen.kernel_name(_cfg(epilogue="cshuffle"))
        with self.assertRaisesRegex(ValueError, "gfx950"):
            codegen.kernel_name(_cfg(gpu_target="gfx950"))

    def test_header(self):
        with tempfile.TemporaryDirectory() as out:
            path = codegen.generate_kernel(Path(out), _cfg(cluster_dim=[2, 2]))
            header = path.read_text()
        self.assertTrue(path.name.endswith("_c2x2.hpp"))
        self.assertIn("using Cluster = ck_tile::sequence<2, 2, 1>;", header)
        self.assertIn(
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, true>",
            header,
        )
        self.assertIn("KMajorScales = true", header)
        self.assertIn("ck_tile::MxDirectStoreEpilogue<", header)

    def test_tdm_epilogue_header(self):
        with tempfile.TemporaryDirectory() as out:
            path = codegen.generate_kernel(
                Path(out), _cfg(epilogue="tdm", store_width=1)
            )
            header = path.read_text()
        self.assertIn("_comp_tdm_scheduled_tdm_", path.name)
        self.assertIn("ck_tile::MxTdmStoreEpilogue<Pipeline>", header)
        self.assertNotIn("MxDirectStoreEpilogue", header)

    def test_kernel_config(self):
        cfg = mx.MxGemmKernelConfig(
            datatype="fp4",
            gpu_target="gfx1250",
            pipeline="comp_tdm_scheduled",
            tile_m=256,
            tile_n=256,
            tile_k=512,
            warp_tile_m=32,
            warp_tile_n=16,
            tuning={"cluster_dim": (2, 2), "gl2_prefetch_distance": 2},
        ).to_codegen_config()
        self.assertEqual(cfg["epilogue"], "direct")
        self.assertEqual(cfg["cluster_dim"], [2, 2])
        self.assertTrue(codegen.kernel_name(cfg).endswith("_c2x2_pf2"))

    def test_tuning_config(self):
        path = _MX / "configs" / "comp_tdm_scheduled_tuning_gfx1250.json"
        tuning_config = json.loads(path.read_text())["tuning_config"]
        self.assertEqual(
            set(tuning_config), {*sched.TUNING_DEFAULTS, *sched.SHAPE_DEFAULTS}
        )
        cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
        self.assertTrue(cfgs)
        self.assertEqual({c.pipeline for c in cfgs}, {"comp_tdm_scheduled"})
        self.assertEqual({c.epilogue for c in cfgs}, {"direct", "tdm"})
        names = [c.name for c in cfgs]
        self.assertEqual(len(set(names)), len(names))
        problems = json.loads(
            (_MX / "configs" / "comp_tdm_scheduled_problems.json").read_text()
        )["problems"]
        self.assertTrue(problems)
        for p in problems:
            self.assertEqual(set(p), {"M", "N", "K"})

    def test_tuned_instance_configs(self):
        # Example 56's maf, medium and mab256, each with its one problem.
        expect = {
            "maf": ("256x256x512_2x2x1_32x16x128_c4x4_noedge_w8", (4096, 4096, 65536)),
            "medium": (
                "128x128x1024_2x2x1_16x16x128_c4x4_hplrnone_noedge_w4",
                (2048, 2048, 65536),
            ),
            "mab": (
                "256x16x512_4x1x1_16x16x128_c1x4_pf0_lp0_hplrnone_noedge_w1",
                (65536, 16, 65536),
            ),
        }
        for case, (suffix, mnk) in expect.items():
            with self.subTest(case=case):
                path = _MX / "configs" / f"comp_tdm_scheduled_{case}_gfx1250.json"
                cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
                self.assertEqual(len(cfgs), 1)
                self.assertTrue(cfgs[0].name.endswith(suffix), cfgs[0].name)
                problems = json.loads(
                    (
                        _MX / "configs" / f"comp_tdm_scheduled_{case}_problems.json"
                    ).read_text()
                )["problems"]
                self.assertEqual([(p["M"], p["N"], p["K"]) for p in problems], [mnk])


class TestCompileFlags(unittest.TestCase):
    def test_sched_flags_only_for_comp_tdm_scheduled(self):
        flags = mx._MX_TDM_SCHEDULED_FLAGS
        for pipeline, expect in (("comp_tdm_scheduled", True), ("comp_tdm", False)):
            run_result = mock.Mock(returncode=0, stderr="")
            with mock.patch.object(mx, "_mx_codegen_flags", return_value=()):
                with mock.patch.object(
                    mx.subprocess, "run", return_value=run_result
                ) as run:
                    self.assertTrue(
                        mx._compile_kernel(
                            Path("k.hpp"), Path("k.so"), "gfx1250", pipeline
                        )
                    )
            cmd = run.call_args.args[0]
            for flag in flags[1::2]:
                self.assertEqual(flag in cmd, expect, (pipeline, flag))


if __name__ == "__main__":
    unittest.main()
