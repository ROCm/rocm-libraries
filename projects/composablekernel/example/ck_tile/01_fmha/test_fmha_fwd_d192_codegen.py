# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from codegen.ops.fmha_fwd import get_fwd_blobs, write_fwd_api

_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_D192_PIPELINE = "qr_tdm_d192_v128"
_D192_ESM2_FILENAME = re.compile(
    r"^fmha_fwd_d192_bf16_.*_qr_tdm_d192_v128_.*_gfx125\.cpp$"
)
_D192_BATCH_NMASK_WWM_FAST_FILENAME = re.compile(
    r"^fmha_fwd_d192_bf16_batch_.*_qr_tdm_d192_v128_.*_nmask_.*_gfx125\.cpp$"
)
_D192_BATCH_NMASK_FILENAME = (
    "fmha_fwd_d192_bf16_batch_b128x128x32x128x32x192_"
    "r4x1x1_r4x1x1_w16x16x32_w16x16x32_o1_qr_tdm_d192_v128_"
    "vr_pddv_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
    "ntrload_nsink_gfx125.cpp"
)
_D192_BATCH_MASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_nmask_", "_mask_")
_D192_GROUP_NMASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_batch_", "_group_")
_D192_SELECTOR = (
    "is_gfx125_d192_tdm_enabled() && a.hdim_q == 192 && "
    "a.hdim_v == 128 && a.max_seqlen_q >= 128"
)


def _generate(receipt, optdim_list=None):
    return get_fwd_blobs(
        targets=["gfx1250"],
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        receipt=receipt,
        optdim_list=[256] if optdim_list is None else optdim_list,
        mask_impl="simplified",
    )


class TestGfx125D192Codegen(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_scopes_wwm_fast_to_batch_nmask(self):
        source_flags = Path(__file__).with_name("cmake") / "fmha_fwd_source_flags.cmake"
        d256_source = (
            "fmha_fwd_d256_bf16_batch_b64x64x32x256x32x256_"
            "r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_"
            "vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
            "ntrload_nsink_gfx125.cpp"
        )
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_MASK_FILENAME,
            _D192_GROUP_NMASK_FILENAME,
            d256_source,
        )

        def configure(wwm_fast):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for source in sources:
                    (root / source).touch()
                source_list = "\n  ".join(
                    f'"${{CMAKE_CURRENT_LIST_DIR}}/{x}"' for x in sources
                )
                (root / "CMakeLists.txt").write_text(
                    f"""cmake_minimum_required(VERSION 3.20)
project(fmha_source_flags LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FMHA_FWD_GFX1250_D192_TDM_ESM2 ON)
set(FMHA_FWD_GFX1250_D192_BATCH_NMASK_WWM_FAST {wwm_fast})
include(\"{source_flags}\")
set(sources
  {source_list}
)
add_library(fmha_source_flags OBJECT ${{sources}})
foreach(source IN LISTS sources)
  get_filename_component(source_name \"${{source}}\" NAME)
  ck_tile_fmha_fwd_get_source_compile_options(
    \"${{source_name}}\" source_compile_options)
  if(source_compile_options)
    set_property(SOURCE \"${{source}}\" APPEND PROPERTY COMPILE_OPTIONS
      ${{source_compile_options}})
  endif()
endforeach()
"""
                )
                subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commands = json.loads(
                    (root / "build" / "compile_commands.json").read_text()
                )
                return {
                    Path(entry["file"]).name: entry["command"] for entry in commands
                }

        enabled = configure("ON")
        self.assertIn(
            "-amdgpu-expert-scheduling-mode", enabled[_D192_BATCH_NMASK_FILENAME]
        )
        self.assertIn("-wwm-regalloc=fast", enabled[_D192_BATCH_NMASK_FILENAME])
        for source in (_D192_BATCH_MASK_FILENAME, _D192_GROUP_NMASK_FILENAME):
            self.assertIn("-amdgpu-expert-scheduling-mode", enabled[source])
            self.assertNotIn("-wwm-regalloc=fast", enabled[source])
        self.assertNotIn("-amdgpu-expert-scheduling-mode", enabled[d256_source])
        self.assertNotIn("-wwm-regalloc=fast", enabled[d256_source])

        disabled = configure("OFF")
        self.assertIn(
            "-amdgpu-expert-scheduling-mode", disabled[_D192_BATCH_NMASK_FILENAME]
        )
        self.assertNotIn("-wwm-regalloc=fast", disabled[_D192_BATCH_NMASK_FILENAME])

    def test_receipts_emit_only_the_dedicated_pipeline_for_d192(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt)
                candidates = [
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                ]

                self.assertTrue(candidates)
                for kernel in candidates:
                    self.assertEqual(kernel.F_arch.name, "gfx125")
                    self.assertEqual(kernel.F_dtype, "bf16")
                    self.assertEqual(kernel.F_hdim, 192)
                    self.assertEqual(kernel.F_tile.F_bn1, 128)
                    self.assertEqual(kernel.F_tile.F_bk0max, 192)
                    self.assertEqual(kernel.F_pipeline.F_vlayout, "row")
                    self.assertEqual(kernel.F_pipeline.F_logits, "f")
                    self.assertEqual(kernel.F_pipeline.F_bias, "no")
                    self.assertEqual(kernel.F_pipeline.F_dropout, "f")
                    self.assertEqual(kernel.F_pipeline.F_qscale, "no")
                    self.assertEqual(kernel.F_pipeline.F_skip, "f")
                    self.assertEqual(kernel.F_pipeline.F_sink, "f")
                    self.assertRegex(kernel.filename, _D192_ESM2_FILENAME)
                    should_use_wwm_fast = (
                        kernel.F_mode == "batch" and "_nmask_" in kernel.filename
                    )
                    if should_use_wwm_fast:
                        self.assertRegex(
                            kernel.filename, _D192_BATCH_NMASK_WWM_FAST_FILENAME
                        )
                    else:
                        self.assertNotRegex(
                            kernel.filename, _D192_BATCH_NMASK_WWM_FAST_FILENAME
                        )

                for kernel in kernels:
                    if kernel not in candidates:
                        self.assertNotRegex(kernel.filename, _D192_ESM2_FILENAME)
                        self.assertNotRegex(
                            kernel.filename, _D192_BATCH_NMASK_WWM_FAST_FILENAME
                        )

                self.assertFalse(
                    any(
                        kernel.F_hdim == 192
                        and kernel.F_tile.F_bn1 == 128
                        and kernel.F_pipeline.tag == "qr_tdm"
                        for kernel in kernels
                    )
                )

    def test_d128_keeps_the_existing_qr_tdm_pipeline(self):
        expected_inventory = {
            100: Counter({("batch", "qr"): 8, ("batch", "qr_tdm"): 8}),
            200: Counter({("group", "qr"): 4}),
            600: Counter(
                {
                    ("batch", "qr"): 8,
                    ("batch", "qr_tdm"): 8,
                    ("group", "qr"): 4,
                }
            ),
        }
        expected_name_digests = {
            100: "f6dcb11ad4779f1bc52edbacd4fd24597d3a523a9d3ef795b6c62a2f67cba515",
            200: "404347bedbd8332ddb9d2eea46d6b0bb6327265d18e3a1f8e4fa0c9f060f4c57",
            600: "2188afcf6c4c104ce3d7d476694e09e540dcbe7fa0a99d96dd8fba24944b26bc",
        }

        for receipt, expected in expected_inventory.items():
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                actual = Counter(
                    (kernel.F_mode, kernel.F_pipeline.tag) for kernel in kernels
                )
                self.assertEqual(actual, expected)
                names = "\n".join(sorted(kernel.name for kernel in kernels)).encode()
                self.assertEqual(
                    hashlib.sha256(names).hexdigest(),
                    expected_name_digests[receipt],
                )
                self.assertFalse(
                    any(kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels)
                )

        _, batch_kernels = _generate(100, [128])
        representative = next(
            kernel
            for kernel in batch_kernels
            if kernel.name == "fmha_fwd_d128_bf16_batch_b64x64x32x128x32x128_"
            "r4x1x1_r4x1x1_w16x16x32_w16x16x32_"
            "qr_tdm_vr_npad_nlogits_nbias_nmask_nlse_ndropout_"
            "nskip_nqscale_ntrload_nsink"
        )
        self.assertEqual(
            hashlib.sha256(representative.render().encode()).hexdigest(),
            "bb08faae2f17aedee9e76bba2c3a493d673e32703f3ed2c60b2634f756f39c29",
        )

    def test_generated_api_pairs_candidate_with_generic_complement(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                api_pool, kernels = _generate(receipt)
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertIn("QRKSVS_TDM_D192_V128", api)
                self.assertIn(_D192_SELECTOR, api)
                self.assertIn(f"!({_D192_SELECTOR})", api)
                self.assertIn(
                    "value[0] == '1' && value[1] == '\\0'",
                    api,
                )

                candidate = next(
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                )
                source = candidate.render()
                self.assertIn("BlockFmhaPipelineQRKSVSTdmD192V128", source)
                self.assertIn("QRKSVS_TDM_D192_V128", source)
                self.assertIn(_D192_PIPELINE, candidate.name)

    def test_filtered_unsupported_feature_keeps_ungated_generic_fallback(self):
        filters = {
            "logits": "fmha_fwd_d*_bf16_*_logits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*",
            "bias": "fmha_fwd_d*_bf16_*_nlogits*_bias*_nmask*_nlse*_ndropout*_nskip*_nqscale*",
            "dropout": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_dropout*_nskip*_nqscale*",
            "skip": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_skip*_nqscale*",
            "sink": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*_sink*",
        }
        for feature, kernel_filter in filters.items():
            with self.subTest(feature=feature):
                api_pool, kernels = get_fwd_blobs(
                    targets=["gfx1250"],
                    kernel_filter=kernel_filter,
                    receipt=100,
                    optdim_list=[256],
                    mask_impl="simplified",
                )
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertTrue(kernels)
                self.assertFalse(
                    any(kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels)
                )
                self.assertNotIn("QRKSVS_TDM_D192_V128", api)
                self.assertNotIn(_D192_SELECTOR, api)

        _, qscale_kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter="fmha_fwd_d*_bf16_*_pertensor*",
            receipt=100,
            optdim_list=[256],
            mask_impl="simplified",
        )
        self.assertFalse(qscale_kernels)

    def test_candidate_only_filters_force_the_generic_complement(self):
        for receipt in (100, 200, 600):
            for kernel_filter in ("*d192_bf16*", f"*{_D192_PIPELINE}*"):
                with self.subTest(receipt=receipt, kernel_filter=kernel_filter):
                    api_pool, kernels = get_fwd_blobs(
                        targets=["gfx1250"],
                        kernel_filter=kernel_filter,
                        receipt=receipt,
                        optdim_list=[256],
                        mask_impl="simplified",
                    )
                    candidate_count = sum(
                        kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels
                    )
                    fallback_count = sum(
                        kernel.F_hdim == 256 and kernel.F_pipeline.tag == "qr"
                        for kernel in kernels
                    )
                    self.assertGreater(candidate_count, 0)
                    self.assertGreater(fallback_count, 0)

                    with tempfile.TemporaryDirectory() as output_dir:
                        output_path = Path(output_dir)
                        write_fwd_api(api_pool, output_path)
                        api = (output_path / "fmha_fwd_api.cpp").read_text()
                    self.assertIn(_D192_SELECTOR, api)
                    self.assertIn(f"!({_D192_SELECTOR})", api)

    def test_non_gfx125_targets_never_emit_the_dedicated_pipeline(self):
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_D192_PIPELINE}*",
                    receipt=100,
                    optdim_list=[256],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)


if __name__ == "__main__":
    unittest.main()
