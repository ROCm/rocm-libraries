# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for isa_report.py and compile_one.py on a checked-in hand-written
gfx1250 assembly fixture. No GPU and no compiler are required.

Run: python3 -m pytest test_isa_report.py -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

import compile_one  # noqa: E402
import isa_report  # noqa: E402

_FIXTURE = _HERE / "test_data" / "isa_report_tiny_gfx1250.s"
_TILE = "32x32x64_2x2x1_16x16x32"

# Minimal llvm-objdump -d text: a one-block loop whose back edge is encoded as
# simm16 65533 (-3 dwords from PC + 4).
_OBJDUMP = """
0000000000001000 <k>:
\ts_mov_b32 s2, 4                                            // 000000001000: BE820084
\tv_wmma_f32_16x16x32_bf16 v[8:15], v[0:7], v[16:23], v[8:15] // 000000001004: CC5A4008 1C222100
\ts_cbranch_scc1 65533                                      // 00000000100C: BFA2FFFD <k+0x4>
\ts_endpgm                                                   // 000000001010: BFB00000
"""


def run_main(*argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = isa_report.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


def fixture_report(*extra):
    rc, out, _ = run_main(_FIXTURE, "--json", *extra)
    assert rc == 0
    return json.loads(out)["kernels"]


class TestParsing(unittest.TestCase):
    def test_classify_and_shapes(self):
        self.assertEqual(isa_report.classify("v_wmma_f32_16x16x32_bf16"), "mma")
        self.assertEqual(isa_report.classify("v_mfma_f32_32x32x8_f16"), "mma")
        self.assertEqual(
            isa_report.classify("global_load_async_to_lds_b128"), "async_to_lds"
        )
        self.assertEqual(isa_report.classify("ds_read_b128"), "ds_load")
        self.assertEqual(isa_report.classify("s_waitcnt"), "wait")
        self.assertEqual(isa_report.normalize_opcode("v_mov_b32_e32"), "v_mov_b32")
        self.assertEqual(isa_report.mma_macs("v_wmma_f32_16x16x32_bf16"), 8192)
        self.assertEqual(isa_report.mma_macs("v_add_f32"), 0)

    def test_metadata(self):
        meta = isa_report.parse_metadata(_FIXTURE.read_text())
        self.assertEqual(list(meta), ["tiny_gemm_kernel"])
        k = meta["tiny_gemm_kernel"]
        self.assertEqual(k["vgpr_count"], 128)
        self.assertEqual(k["vgpr_spill_count"], 3)
        self.assertEqual(k["private_segment_fixed_size"], 16)
        self.assertEqual(k["group_segment_fixed_size"], 139264)
        self.assertEqual(k["max_flat_workgroup_size"], 256)

    def test_remarks(self):
        text = (
            "remark: a.hpp:1:0: Function Name: kern [-Rpass-analysis=kernel-resource-usage]\n"
            "remark: a.hpp:1:0:     Occupancy [waves/SIMD]: 2 [-Rpass-analysis=x]\n"
        )
        self.assertEqual(isa_report.parse_remarks(text), {"kern": {"occupancy": 2}})

    def test_objdump_back_edge(self):
        (func,) = isa_report.parse_objdump(_OBJDUMP)
        self.assertEqual(func.name, "k")
        self.assertEqual(func.branches, [(2, 1)])
        blocks, loops = isa_report.find_loops(func)
        self.assertEqual(len(loops), 1)
        header, body = loops[0]
        self.assertEqual(blocks[header], (1, 3))


class TestFixtureReport(unittest.TestCase):
    def test_only_kernels_by_default(self):
        names = [r["name"] for r in fixture_report()]
        self.assertEqual(names, ["tiny_gemm_kernel"])
        names = [r["name"] for r in fixture_report("--all-functions")]
        self.assertEqual(names, ["tiny_gemm_kernel", "tiny_helper"])

    def test_resources(self):
        res = fixture_report()[0]["resources"]
        self.assertEqual(res["vgpr_msb_count"], 2)
        self.assertEqual(res["lds_capacity"], 327680)
        self.assertEqual(res["lds_pct"], 42.5)
        self.assertEqual(res["lds_blocks_per_cu"], 2)

    def test_loops(self):
        r = fixture_report()[0]
        # The out-of-line block branching back to the header is not a loop.
        self.assertEqual(len(r["loops"]), 1)
        hot = r["hot_loop"]
        self.assertEqual(hot["range"], [4, 20])
        self.assertEqual(hot["blocks"], 3)
        self.assertEqual(r["prologue_insts"], 4)
        self.assertEqual(
            hot["opcodes"],
            {
                "ds_load_b128": 2,
                "global_load_async_to_lds_b128": 1,
                "scratch_store_b32": 1,
                "tensor_load_to_lds": 1,
                "v_wmma_f32_16x16x32_bf16": 2,
            },
        )
        self.assertEqual(
            hot["waits"],
            {"s_wait_dscnt 0x0": 1, "s_wait_dscnt 0x1": 1, "s_wait_tensorcnt 0x0": 1},
        )
        self.assertEqual(hot["barrier_syncs"], 1)
        self.assertEqual(hot["classes"]["barrier"], 2)
        self.assertEqual(hot["classes"]["vgpr_msb"], 1)

    def test_k_normalisation(self):
        hot = fixture_report("--tile", _TILE)[0]["hot_loop"]
        # 2 x 16x16x32 MACs over a 16x16 per-wave tile -> 64 K per iteration.
        self.assertEqual(hot["k_per_iter"], 64)
        self.assertEqual(hot["per_k_tile"]["opcodes"]["v_wmma_f32_16x16x32_bf16"], 2)
        hot = fixture_report("--tile", "32x32x128_2x2x1")[0]["hot_loop"]
        self.assertEqual(hot["per_k_tile"]["opcodes"]["ds_load_b128"], 4)
        self.assertEqual(hot["per_k_tile"]["barrier_syncs"], 2)

    def test_epilogue(self):
        ep = fixture_report()[0]["epilogue"]
        self.assertEqual(ep["stores"], {"ds_store_b64": 1, "global_store_b128": 2})
        self.assertEqual(ep["barrier_syncs"], 1)
        self.assertEqual(
            ep["waits"], {"s_wait_asynccnt 0x0": 1, "s_waitcnt_vscnt null, 0x0": 1}
        )
        self.assertEqual(ep["opcodes"]["scratch_load_b32"], 1)

    def test_text_output_and_check(self):
        rc, out, _ = run_main(_FIXTURE, "--tile", _TILE)
        self.assertEqual(rc, 0)
        self.assertIn("(hot)", out)
        self.assertIn("ds_load_b128=2", out)
        self.assertIn("stores: ds_store_b64=1 global_store_b128=2", out)
        rc, _, err = run_main(_FIXTURE, "--check")
        self.assertEqual(rc, 1)
        self.assertIn("vgpr_spill_count=3", err)
        rc, _, _ = run_main(_FIXTURE, "--kernel", "no_such_kernel")
        self.assertEqual(rc, 2)

    def test_ref_diff(self):
        rc, out, _ = run_main(_FIXTURE, "--ref", _FIXTURE)
        self.assertEqual(rc, 0)
        delta = out[out.index("== delta") :]
        self.assertNotIn("hot ", delta)
        self.assertNotIn("resource ", delta)

    def test_lds_capacity_override(self):
        r = fixture_report("--lds-capacity", "65536")[0]
        self.assertEqual(r["resources"]["lds_capacity"], 65536)
        self.assertIn(
            "group_segment_fixed_size=139264 > 65536", ", ".join(isa_report.check(r))
        )


class TestCompileOne(unittest.TestCase):
    def test_arch_defines(self):
        self.assertEqual(compile_one.target_id("gfx1250"), "0x1250")
        self.assertEqual(compile_one.target_id("gfx90a"), "0x090A")
        self.assertEqual(compile_one.target_id("gfx90a:xnack-"), "0x090A")
        self.assertEqual(compile_one.target_id("gfx11-generic"), "0x11FF")
        self.assertEqual(compile_one.target_id("gfx1250-strict"), "0x1250")
        with self.assertRaises(ValueError):
            compile_one.target_id("gfx9-4-generic")
        d = compile_one.arch_defines("gfx1250")
        for want in (
            "CK_USE_GFX1250",
            "CK_GFX1250_SUPPORT",
            "CK_GFX12_SUPPORT",
            "CK_TILE_USE_WMMA=1",
            "USE_NEW_UNIFIED_FRAMEWORK=0",
            "CK_CMAKE_GPU_TARGET_IDS=0x1250",
        ):
            self.assertIn(want, d)
        d = compile_one.arch_defines("gfx942")
        self.assertIn("CK_TILE_USE_WMMA=0", d)
        self.assertIn("CK_USE_FNUZ_FP8", d)
        self.assertNotIn("CK_USE_GFX1250", d)
        self.assertIn("CK_USE_GFX94", d)
        self.assertIn("CK_ENABLE_TF32", d)
        d = compile_one.arch_defines("gfx950")
        for want in ("CK_USE_GFX94", "CK_USE_GFX950", "CK_ENABLE_TF32"):
            self.assertIn(want, d)
        d = compile_one.arch_defines("gfx90a:xnack-")
        self.assertIn("CK_CMAKE_GPU_TARGET_IDS=0x090A", d)
        self.assertNotIn("CK_ENABLE_TF32", d)

    def test_dry_run_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "gemm_universal_single_x_256x256x64_2x4x1_16x16x32.hpp"
            header.write_text("// empty\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = compile_one.main(
                    [
                        "--out-dir",
                        tmp,
                        "--rocm-path",
                        "/opt/rocm-test",
                        "--header",
                        str(header),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 0)
            cmd = out.getvalue()
            self.assertIn("/opt/rocm-test/lib/llvm/bin/clang++", cmd)
            self.assertIn("--offload-arch=gfx1250", cmd)
            self.assertIn("--cuda-device-only", cmd)
            self.assertIn("-Rpass-analysis=kernel-resource-usage", cmd)
            self.assertIn(str(compile_one._DEFAULT_CK_ROOT / "include"), cmd)
            self.assertEqual(
                compile_one.tile_of(header.name), "256x256x64_2x4x1_16x16x32"
            )


if __name__ == "__main__":
    unittest.main()
