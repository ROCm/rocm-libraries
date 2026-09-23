# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from codegen.ops.fmha_fwd import (
    FMHA_FWD_API_FOOTER,
    FMHA_FWD_API_HEADER,
    FmhaFwdApiPool,
    get_fwd_blobs,
    write_fwd_api,
)

_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_D192_PIPELINE = "qr_tdm_d192_v128"
_FAMILY_PIPELINES = frozenset((_D192_PIPELINE, "qr_tdm_v128"))
_OPT_IN_ENV = "CK_TILE_FMHA_GFX125_D192_TDM"
_FORCE_FALLBACK_ENV = "CK_TILE_FMHA_GFX125_TDM_V128_FORCE_FALLBACK"
_HOST_CXX = shutil.which("c++")
_RENDERED_CALL = re.compile(
    r"using trait_ = (fmha_fwd_traits_<[^;\n]+>);\s*"
    r"return fmha_fwd_<trait_, ([^>]+)>\(s, a\);"
)
_D192_ESM2_FILENAME = re.compile(
    r"^fmha_fwd_d192_bf16_.*_qr_tdm_d192_v128_.*_gfx125\.cpp$"
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


def _is_family_kernel(kernel):
    return kernel.F_pipeline.tag in _FAMILY_PIPELINES


# These stubs provide types and unique launch IDs only. All selection predicates,
# branch ordering, environment parsing, and v3/v2 fallback come from the renderer.
_HOST_HIP_STUB = """#pragma once
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { unsigned multiProcessorCount = 80; };
inline int hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline int hipGetDeviceProperties(hipDeviceProp_t*, int) { return hipSuccess; }
"""

_HOST_FMHA_STUB = """#pragma once
#include <cstdlib>
#include <string>
namespace ck_tile {
using index_t = int;
struct stream_config {};
struct gfx125_t {};
inline std::string get_device_name() {
    const char* value = std::getenv("FMHA_TEST_DEVICE");
    return value == nullptr ? "gfx1250" : value;
}
template <bool> struct SimplifiedGenericAttentionMask {};
enum class BlockFmhaPipelineEnum { QRKSVS, QRKSVS_TDM, QRKSVS_TDM_D192_V128, QRKSVS_TDM_V128 };
enum class BlockAttentionBiasEnum { NO_BIAS, ELEMENTWISE_BIAS, ALIBI };
enum class BlockAttentionQuantScaleEnum { NO_SCALE, PERTENSOR, BLOCKSCALE, KV_BLOCKSCALE, MX };
}
enum class mask_enum { no_mask, mask_top_left, mask_bottom_right, window_generic };
enum class bias_enum { no_bias, elementwise_bias, alibi };
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx };
struct FmhaFwdBf16 {};
inline constexpr ck_tile::index_t fmha_fwd_largest_n_tile_size = 128;
template <int, typename, bool, int, int, int, int, int, int, bool, auto, bool,
          typename, auto, bool, bool, auto, bool, bool, bool, bool, bool, bool, bool, int = -1>
struct fmha_fwd_traits_ {};
struct fmha_fwd_traits {
    int hdim_q = 192, hdim_v = 128;
    std::string data_type = "bf16";
    bool is_group_mode = false, is_v_rowmajor = true, has_logits_soft_cap = false;
    mask_enum mask_type = mask_enum::no_mask;
    bias_enum bias_type = bias_enum::no_bias;
    bool has_lse = false, has_dropout = false;
    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    bool skip_min_seqlen_q = false, has_sink = false;
};
struct fmha_fwd_args {
    int hdim_q = 192, hdim_v = 128, seqlen_q = 128, max_seqlen_q = 128, seqlen_k = 128;
    unsigned batch = 2, nhead_q = 8, nhead_k = 2;
    const void* cu_seqlen_k_ptr = nullptr;
};
template <typename Trait, typename Arch> struct dispatch_id;
template <typename Trait, typename Arch>
float fmha_fwd_(const ck_tile::stream_config&, const fmha_fwd_args&) {
    return dispatch_id<Trait, Arch>::value;
}
"""

_HOST_MAIN = """
#include <iostream>
int main(int argc, char** argv) {
    fmha_fwd_traits t;
    fmha_fwd_args a;
    int mask, bias, scale, arg_q, arg_v, max_q;
    if(!(std::cin >> t.hdim_q >> t.hdim_v >> a.seqlen_q >> a.seqlen_k
         >> t.is_group_mode >> t.is_v_rowmajor >> t.has_logits_soft_cap >> mask >> bias
         >> t.has_lse >> t.has_dropout >> scale >> t.skip_min_seqlen_q >> t.has_sink
         >> t.data_type >> arg_q >> arg_v >> max_q >> a.nhead_q >> a.nhead_k)) return 2;
    t.mask_type = static_cast<mask_enum>(mask);
    t.bias_type = static_cast<bias_enum>(bias);
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    a.hdim_q = arg_q < 0 ? t.hdim_q : arg_q;
    a.hdim_v = arg_v < 0 ? t.hdim_v : arg_v;
    a.max_seqlen_q = max_q < 0 ? a.seqlen_q : max_q;
    std::cout << fmha_fwd(t, a, {}) << '\\n';
    if(argc >= 2) {
        if(std::string(argv[1]) == "__UNSET__")
            unsetenv("CK_TILE_FMHA_GFX125_D192_TDM");
        else if(std::string(argv[1]) != "__KEEP__")
            setenv("CK_TILE_FMHA_GFX125_D192_TDM", argv[1], 1);
        if(argc == 3) {
            if(std::string(argv[2]) == "__UNSET__")
                unsetenv("CK_TILE_FMHA_GFX125_TDM_V128_FORCE_FALLBACK");
            else
                setenv("CK_TILE_FMHA_GFX125_TDM_V128_FORCE_FALLBACK", argv[2], 1);
        }
        std::cout << fmha_fwd(t, a, {}) << '\\n';
    }
}
"""


class _CompiledDispatcher:
    def __init__(
        self, test, pool, kernels, filter_fn=None, mutate=None, default_enabled=None
    ):
        directory = tempfile.TemporaryDirectory(prefix="fmha-dispatch-host-")
        test.addCleanup(directory.cleanup)
        root = Path(directory.name)
        if filter_fn is None:
            write_fwd_api(pool, root)
            api = (root / "fmha_fwd_api.cpp").read_text()
        else:
            api = (
                FMHA_FWD_API_HEADER
                + pool.render("fmha_fwd_v2", filter_fn=filter_fn)
                + pool.render("fmha_fwd_v3", filter_fn=lambda trait: False)
                + FMHA_FWD_API_FOOTER
            )
        if default_enabled is not None:
            # Exercise the planned R9 constant without duplicating selection logic.
            setting = r"constexpr bool kTdmV128DefaultEnabled = (?:true|false);"
            test.assertEqual(len(re.findall(setting, api)), 1)
            api = re.sub(
                setting,
                f"constexpr bool kTdmV128DefaultEnabled = {str(default_enabled).lower()};",
                api,
            )
        signatures = list(dict.fromkeys(_RENDERED_CALL.findall(api)))
        ids = {signature: index + 1 for index, signature in enumerate(signatures)}
        self.kernel_ids = []
        for kernel in kernels:
            single = FmhaFwdApiPool()
            single.register_traits(kernel.api_trait())
            (signature,) = _RENDERED_CALL.findall(single.render("unused"))
            if signature in ids:
                self.kernel_ids.append((kernel, ids[signature]))
        test.assertEqual(set(ids.values()), {value for _, value in self.kernel_ids})
        specializations = "\n".join(
            f"template <> struct dispatch_id<{trait}, {arch}> {{ "
            f"static constexpr int value = {value}; }};"
            for (trait, arch), value in ids.items()
        )
        (root / "hip").mkdir()
        (root / "hip" / "hip_runtime.h").write_text(_HOST_HIP_STUB)
        (root / "fmha_fwd.hpp").write_text(_HOST_FMHA_STUB + specializations)
        (root / "fmha_fwd_api.cpp").write_text(
            (api if mutate is None else mutate(api)) + _HOST_MAIN
        )
        self.binary = root / "dispatcher"
        compile_result = subprocess.run(
            [
                _HOST_CXX,
                "-std=c++17",
                "-O0",
                "-I",
                str(root),
                str(root / "fmha_fwd_api.cpp"),
                "-o",
                str(self.binary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        test.assertEqual(compile_result.returncode, 0, compile_result.stderr)

    def expected_id(self, **fields):
        def value(kernel, name):
            if name in ("mode", "hdim"):
                return getattr(kernel, "F_" + name)
            if name == "bm0":
                return kernel.F_tile.F_bm0
            if name == "pipeline":
                return kernel.F_pipeline.tag
            return getattr(kernel.F_pipeline, "F_" + name)

        matches = {
            index
            for kernel, index in self.kernel_ids
            if all(value(kernel, key) == expected for key, expected in fields.items())
        }
        if len(matches) != 1:
            raise AssertionError(
                f"expected one exact rendered trait for {fields}: {matches}"
            )
        return matches.pop()

    def run(
        self,
        env="1",
        next_env=None,
        device="gfx1250",
        force=None,
        next_force=None,
        **changes,
    ):
        case = dict(
            hdim_q=192,
            hdim_v=128,
            seqlen_q=128,
            seqlen_k=128,
            group=0,
            row=1,
            logits=0,
            mask=0,
            bias=0,
            lse=0,
            dropout=0,
            qscale=0,
            skip=0,
            sink=0,
            dtype="bf16",
            arg_q=-1,
            arg_v=-1,
            max_seqlen_q=-1,
            nhead_q=8,
            nhead_k=2,
        )
        if changes.keys() - case.keys():
            raise ValueError(
                f"unknown dispatcher fields: {changes.keys() - case.keys()}"
            )
        case.update(changes)
        environment = dict(
            os.environ,
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            FMHA_TEST_DEVICE=device,
        )
        environment.pop(_OPT_IN_ENV, None)
        environment.pop(_FORCE_FALLBACK_ENV, None)
        if env is not None:
            environment[_OPT_IN_ENV] = env
        if force is not None:
            environment[_FORCE_FALLBACK_ENV] = force
        command = [str(self.binary)]
        if next_force is not None:
            command.extend(["__KEEP__" if next_env is None else next_env, next_force])
        elif next_env is not None:
            command.append(next_env)
        result = subprocess.run(
            command,
            input=" ".join(map(str, case.values())) + "\n",
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return [int(value) for value in result.stdout.split()]


class TestGfx125D192Codegen(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_deprecated_wwm_option_defaults_off_and_warns(self):
        cmake_source = Path(__file__).with_name("CMakeLists.txt").read_text()
        option = "FMHA_FWD_GFX1250_D192_BATCH_NMASK_WWM_FAST"
        begin = cmake_source.index(f"option({option}\n")
        end = cmake_source.index("include(${CMAKE_CURRENT_LIST_DIR}/cmake/", begin)
        declaration = cmake_source[begin:end]
        for cached, expected in ((None, "OFF"), ("OFF", "OFF"), ("ON", "ON")):
            with (
                self.subTest(cached=cached),
                tempfile.TemporaryDirectory() as directory,
            ):
                script = Path(directory) / "option.cmake"
                initial = (
                    ""
                    if cached is None
                    else f'set({option} {cached} CACHE BOOL "legacy")\n'
                )
                script.write_text(
                    "cmake_minimum_required(VERSION 3.20)\n"
                    + initial
                    + declaration
                    + f'if(NOT "${{{option}}}" STREQUAL "{expected}")\n'
                    + '  message(FATAL_ERROR "unexpected option default/cache value")\nendif()\n'
                )
                result = subprocess.run(
                    ["cmake", "-P", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                self.assertEqual(
                    f"{option} is ignored" in result.stderr, cached == "ON"
                )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_uses_default_wwm_with_stale_fast_option(self):
        source_flags = Path(__file__).with_name("cmake") / "fmha_fwd_source_flags.cmake"
        d256_source = (
            "fmha_fwd_d256_bf16_batch_b64x64x32x256x32x256_"
            "r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_"
            "vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
            "ntrload_nsink_gfx125.cpp"
        )
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_NMASK_FILENAME.replace("_nlse_", "_lse_"),
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

        for option in ("ON", "OFF"):
            with self.subTest(wwm_fast=option):
                commands = configure(option)
                for source in sources:
                    self.assertNotIn("-wwm-regalloc=fast", commands[source])
                    self.assertEqual(
                        "-amdgpu-expert-scheduling-mode" in commands[source],
                        source != d256_source,
                    )

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

                for kernel in kernels:
                    if kernel not in candidates:
                        self.assertNotRegex(kernel.filename, _D192_ESM2_FILENAME)

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
            100: Counter({("batch", "qr"): 8, ("batch", "qr_tdm"): 16}),
            200: Counter({("group", "qr"): 4, ("group", "qr_tdm"): 8}),
            600: Counter(
                {
                    ("batch", "qr"): 8,
                    ("batch", "qr_tdm"): 16,
                    ("group", "qr"): 4,
                    ("group", "qr_tdm"): 8,
                }
            ),
        }
        expected_name_digests = {
            100: "999a673328eab8ccbc069002fe3e6e9997304dce30daf87763e8ef5df7005072",
            200: "6bde6006a611b82f9ce93a50357e72d29a2ac962203ba4da90ec5bd78e1695d4",
            600: "729e4a358f382817d167f36c0c876555e0b8a3b525bb0b4c62cb3726889674b0",
        }

        for receipt, expected in expected_inventory.items():
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                non_family = [
                    kernel for kernel in kernels if not _is_family_kernel(kernel)
                ]
                self.assertTrue(
                    [kernel for kernel in kernels if _is_family_kernel(kernel)]
                )
                actual = Counter(
                    (kernel.F_mode, kernel.F_pipeline.tag) for kernel in non_family
                )
                self.assertEqual(actual, expected)
                names = "\n".join(sorted(kernel.name for kernel in non_family)).encode()
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
            "3306beb9a5d0a371ee3d6b4ac4a60baf4a14f41bf08a5168dc83f78c7259bc3c",
        )

    def test_generated_api_places_exact_candidate_before_legacy_buckets(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                api_pool, kernels = _generate(receipt)
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertIn("QRKSVS_TDM_D192_V128", api)
                self.assertIn(_D192_SELECTOR, api)
                self.assertNotIn(f"!({_D192_SELECTOR})", api)
                self.assertNotIn("t.hdim_q <= 192", api)
                self.assertLess(api.index(_D192_SELECTOR), api.index("t.hdim_q <= 256"))
                self.assertIn(
                    "value[0] == '1' && value[1] == '\\0'",
                    api,
                )

                candidate = next(
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                )
                self.assertEqual(candidate.api_trait().dispatch_bucket, ("192", 128))
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

    def test_candidate_only_filters_retain_ungated_generic_fallback(self):
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
                    self.assertNotIn(f"!({_D192_SELECTOR})", api)

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


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125D192Dispatch(unittest.TestCase):
    """CPU selection baseline, not kernel validity or final default-on R4 acceptance."""

    def compile(
        self,
        receipt=600,
        optdims=None,
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        filter_fn=None,
        mutate=None,
    ):
        pool, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=kernel_filter,
            receipt=receipt,
            optdim_list=[128, 256] if optdims is None else optdims,
            mask_impl="simplified",
        )
        return _CompiledDispatcher(self, pool, kernels, filter_fn, mutate), kernels

    @staticmethod
    def candidate_id(dispatch, mode, mask="s_no", lse="f"):
        return dispatch.expected_id(
            mode=mode, hdim=192, pipeline=_D192_PIPELINE, mask=mask, lse=lse, sink="f"
        )

    @staticmethod
    def fallback_id(dispatch, mode, mask="s_no", lse="f", padded="t", sink="f"):
        return dispatch.expected_id(
            mode=mode,
            hdim=256,
            pipeline="qr",
            mask=mask,
            lse=lse,
            spad=padded,
            dpad=padded,
            sink=sink,
        )

    def test_real_dispatch_priority_exact_shape_short_q_and_opt_in(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            fallback = self.fallback_id(dispatch, mode)
            for env in (None, "", "0", "1", "01", "true", "1x", " 1", "1 "):
                with self.subTest(mode=mode, env=env):
                    self.assertEqual(
                        dispatch.run(env=env, group=group),
                        [candidate if env == "1" else fallback],
                    )
            for length in (128, 129, 257):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length), [candidate]
                )
            for length in (0, 1, 64, 127):
                self.assertEqual(dispatch.run(group=group, seqlen_q=length), [fallback])
            for q, v in ((160, 128), (191, 128), (193, 128), (192, 127), (192, 192)):
                with self.subTest(mode=mode, shape=(q, v)):
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=q, hdim_v=v), [fallback]
                    )
            generic256 = self.fallback_id(dispatch, mode, padded="t" if group else "f")
            self.assertEqual(
                dispatch.run(group=group, hdim_q=256, hdim_v=256), [generic256]
            )

    def test_real_dispatch_opt_in_is_cached_per_process(self):
        dispatch, _ = self.compile()
        candidate = self.candidate_id(dispatch, "batch")
        fallback = self.fallback_id(dispatch, "batch")
        self.assertEqual(dispatch.run(env="1", next_env="0"), [candidate, candidate])
        self.assertEqual(
            dispatch.run(env="1", next_env="__UNSET__"), [candidate, candidate]
        )
        self.assertEqual(dispatch.run(env=None, next_env="1"), [fallback, fallback])
        self.assertEqual(dispatch.run(env="01", next_env="1"), [fallback, fallback])

    def test_family_only_and_empty_filtered_dispatch(self):
        family, _ = self.compile(
            filter_fn=lambda trait: trait.pipeline_tag == _D192_PIPELINE
        )
        candidate = self.candidate_id(family, "batch")
        self.assertEqual(family.run(), [candidate])
        for arguments in ({"env": None}, {"seqlen_q": 127}, {"hdim_q": 128}):
            self.assertEqual(family.run(**arguments), [-1])
        empty, _ = self.compile(filter_fn=lambda trait: False)
        self.assertEqual(empty.run(), [-1])

    def test_mismatched_trait_argument_dimensions_never_select_family(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            for dimensions in (
                dict(hdim_q=128, arg_q=192),
                dict(hdim_q=256, arg_q=192),
                dict(hdim_v=256, arg_v=128),
                dict(arg_q=128),
                dict(arg_v=256),
            ):
                with self.subTest(mode=mode, dimensions=dimensions):
                    self.assertNotEqual(
                        dispatch.run(group=group, **dimensions), [candidate]
                    )

    def test_real_dispatch_d128_legacy_priority_and_sequence_boundary(self):
        dispatch, kernels = self.compile()
        self.assertTrue(
            [
                kernel
                for kernel in kernels
                if kernel.F_hdim == 128 and _is_family_kernel(kernel)
            ]
        )
        for mode, group, pipeline in (("batch", 0, "qr_tdm"), ("group", 1, "qr_tdm")):
            for length, bm0 in ((1, 64), (127, 64), (128, 64), (2047, 64), (2048, 128)):
                expected = dispatch.expected_id(
                    mode=mode,
                    hdim=128,
                    pipeline=pipeline,
                    bm0=bm0,
                    spad="t" if group else "f",
                    mask="s_no",
                    lse="f",
                    sink="f",
                    dpad="f",
                )
                for env in (None, "1"):
                    selected = expected
                    if env == "1" and length >= 128:
                        selected = dispatch.expected_id(
                            mode=mode,
                            hdim=128,
                            pipeline="qr_tdm_v128",
                            mask="s_no",
                            lse="f",
                            sink="f",
                        )
                    self.assertEqual(
                        dispatch.run(
                            env=env,
                            group=group,
                            hdim_q=128,
                            hdim_v=128,
                            seqlen_q=length,
                        ),
                        [selected],
                    )

    def test_real_dispatch_mask_lse_and_unsupported_features(self):
        kernel_filter = (
            "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
        )
        dispatch, _ = self.compile(kernel_filter=kernel_filter)
        for mode, group in (("batch", 0), ("group", 1)):
            for mask in (0, 1, 2, 3):
                for lse in (0, 1):
                    fields = dict(
                        mask="s_no" if mask == 0 else "s_mask", lse="t" if lse else "f"
                    )
                    candidate = self.candidate_id(dispatch, mode, **fields)
                    fallback = self.fallback_id(dispatch, mode, **fields)
                    self.assertEqual(
                        dispatch.run(group=group, mask=mask, lse=lse), [candidate]
                    )
                    self.assertEqual(
                        dispatch.run(env=None, group=group, mask=mask, lse=lse),
                        [fallback],
                    )
            sink = self.fallback_id(dispatch, mode, sink="t")
            self.assertEqual(dispatch.run(group=group, sink=1), [sink])
        for changes in (
            dict(row=0),
            dict(logits=1),
            dict(bias=1),
            dict(dropout=1),
            dict(qscale=1),
            dict(skip=1),
            dict(dtype="fp16"),
            dict(hdim_q=257),
            dict(hdim_v=257),
            dict(device="gfx942"),
        ):
            with self.subTest(unsupported=changes):
                self.assertEqual(dispatch.run(**changes), [-1])

    def test_real_dispatch_missing_candidate_does_not_gate_generic(self):
        for render_filter in (True, False):
            with self.subTest(render_filter=render_filter):
                dispatch, _ = self.compile(
                    kernel_filter=(
                        _SUPPORTED_FEATURE_FILTER
                        if render_filter
                        else _SUPPORTED_FEATURE_FILTER.replace("d*_", "d256_", 1)
                    ),
                    filter_fn=(
                        (lambda trait: trait.pipeline_tag != _D192_PIPELINE)
                        if render_filter
                        else None
                    ),
                )
                for mode, group in (("batch", 0), ("group", 1)):
                    fallback = self.fallback_id(dispatch, mode)
                    self.assertEqual(dispatch.run(group=group), [fallback])
                    self.assertEqual(dispatch.run(env=None, group=group), [fallback])

    def test_real_dispatch_candidate_only_filter_has_feature_matched_closure(self):
        for receipt in (0, 100, 200, 600):
            with self.subTest(receipt=receipt):
                dispatch, kernels = self.compile(
                    receipt=receipt, kernel_filter=f"*{_D192_PIPELINE}*"
                )
                candidates = {
                    kernel.api_trait().gfx125_d192_feature_key
                    for kernel in kernels
                    if _is_family_kernel(kernel)
                }
                fallbacks = {
                    kernel.api_trait().gfx125_d192_feature_key
                    for kernel in kernels
                    if not _is_family_kernel(kernel)
                }
                self.assertTrue(candidates)
                self.assertEqual(candidates, fallbacks)
                self.assertEqual(len({kernel.name for kernel in kernels}), len(kernels))
                for mode in sorted({kernel.F_mode for kernel in kernels}):
                    group = int(mode == "group")
                    self.assertEqual(
                        dispatch.run(group=group), [self.candidate_id(dispatch, mode)]
                    )
                    fallback = self.fallback_id(dispatch, mode)
                    self.assertEqual(dispatch.run(env="0", group=group), [fallback])
                    self.assertEqual(
                        dispatch.run(group=group, seqlen_q=127), [fallback]
                    )
                    self.assertEqual(dispatch.run(group=group, hdim_q=160), [fallback])
                    self.assertEqual(dispatch.run(group=group, logits=1), [-1])

    def test_real_dispatch_receipt_and_optdim_baseline(self):
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text()
        common_dims = list(
            map(int, re.search(r"--optdim\s+([0-9,]+)", cmake)[1].split(","))
        )
        self.assertEqual(common_dims, [32, 64, 80, 128, 256])
        for receipt in (0, 100, 200, 600):
            for optdims in ([128], [192], [256], [-1], common_dims):
                with self.subTest(receipt=receipt, optdims=optdims):
                    dispatch, kernels = self.compile(receipt=receipt, optdims=optdims)
                    modes = sorted({kernel.F_mode for kernel in kernels})
                    family = [kernel for kernel in kernels if _is_family_kernel(kernel)]
                    self.assertTrue(family)
                    self.assertEqual(
                        any(k.F_hdim == 192 for k in family), optdims != [128]
                    )
                    for mode in modes:
                        group = int(mode == "group")
                        if optdims == [128]:
                            self.assertEqual(dispatch.run(group=group), [-1])
                        else:
                            self.assertEqual(
                                dispatch.run(group=group),
                                [self.candidate_id(dispatch, mode)],
                            )
                            self.assertEqual(
                                dispatch.run(env="0", group=group),
                                [self.fallback_id(dispatch, mode)],
                            )
                    if receipt in (100, 200):
                        self.assertEqual(dispatch.run(group=int(receipt == 100)), [-1])

    def test_compiled_negative_control_detects_incorrect_short_q_gate(self):
        def mutate(api):
            self.assertIn("a.max_seqlen_q >= 128", api)
            return api.replace("a.max_seqlen_q >= 128", "a.max_seqlen_q >= 0")

        dispatch, _ = self.compile(mutate=mutate)
        actual = dispatch.run(seqlen_q=127)
        self.assertNotEqual(actual, [self.fallback_id(dispatch, "batch")])
        self.assertEqual(actual, [self.candidate_id(dispatch, "batch")])


if __name__ == "__main__":
    unittest.main()
