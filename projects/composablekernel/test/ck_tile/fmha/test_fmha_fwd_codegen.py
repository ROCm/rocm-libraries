# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA codegen and real rendered-dispatch regression tests (CPU only)."""

import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from dataclasses import fields, replace
import sys

_CK_ROOT = Path(__file__).resolve().parents[3]
_FMHA_EXAMPLE_DIR = _CK_ROOT / "example" / "ck_tile" / "01_fmha"
sys.path.insert(0, str(_FMHA_EXAMPLE_DIR))

from codegen.ops.fmha_fwd_splitkv import get_fwd_splitkv_blobs  # noqa: E402
from codegen.ops.fmha_fwd import (  # noqa: E402
    CppConstraint,
    FMHA_FWD_API_FOOTER,
    FMHA_FWD_API_HEADER,
    FmhaFwdApiPool,
    KernelComponentFactoryGfx125,
    KernelContext,
    ProblemContext,
    get_fwd_blobs,
    write_fwd_api,
)


def _is_plain_fp8_sched(k):
    return k.F_pipeline.tag == "qr_tdm_fp8_sched" and k.F_pipeline.F_packgqa == "f"


def _fp8_sched_kind(k):
    # Plain sched, or packed GQA on the full tile or on the 32-row one-wavefront tile.
    if k.F_pipeline.F_packgqa == "f":
        return k.F_pipeline.tag
    return "packgqa_32row" if k.F_tile.F_bm0 == 32 else "packgqa"


def _fp8_sched_dispatch_order(api):
    # (bm0, packed GQA) of each scheduled FP8 trait in dispatch order.
    pattern = (
        r"using trait_ = fmha_fwd_traits_<\d+, \w+, \w+, (\d+), "
        r"[^;]*?QRKSVS_TDM_SCHED_FP8,[^;]*?(, true)?>;"
    )
    return [(int(m[1]), m[2] is not None) for m in re.finditer(pattern, api)]


_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_D192_PIPELINE = "qr_tdm_sched"
_HOST_CXX = shutil.which("c++")
_ROCM_ROOT = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
_HIP_HOST_CXX = _ROCM_ROOT / "llvm" / "bin" / "clang++"
_RENDERED_CALL = re.compile(
    r"using trait_ = (fmha_fwd_traits_<[^;\n]+>);\s*"
    r"return fmha_fwd_<trait_, ([^>]+)>\(s, a\);"
)
_D192_ESM2_FILENAME = re.compile(r"^fmha_fwd_d192_bf16_.*_qr_tdm_sched_.*_gfx125\.cpp$")
_D192_BATCH_NMASK_FILENAME = (
    "fmha_fwd_d192_bf16_batch_b128x64x32x128x32x192_"
    "r4x1x1_r4x1x1_w16x16x32_w16x16x32_o2_qr_tdm_sched_"
    "vr_pddv_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
    "ntrload_nsink_gfx125.cpp"
)
_D192_BATCH_MASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_nmask_", "_mask_")
_D192_GROUP_NMASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_batch_", "_group_")
_D192_BATCH_N128_FILENAME = _D192_BATCH_NMASK_FILENAME.replace(
    "b128x64x32", "b128x128x32"
).replace("_o2_", "_o1_")
_D192_SELECTOR = (
    "a.hdim_q == 192 && a.hdim_v == 128 && a.sink_ptr == nullptr && "
    "a.max_seqlen_q >= 128"
)


def _generate(receipt, optdim_list=None):
    return get_fwd_blobs(
        targets=["gfx1250"],
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        receipt=receipt,
        optdim_list=[192, 256] if optdim_list is None else optdim_list,
        mask_impl="simplified",
    )


def _is_family_kernel(kernel):
    return kernel.F_pipeline.tag == _D192_PIPELINE


# These stubs provide types and unique launch IDs only. All selection predicates,
# branch ordering, and v3/v2 fallback come from the renderer.
_HOST_HIP_STUB = """#pragma once
#include <cstdlib>
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { unsigned multiProcessorCount = 256; };
inline int hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline int hipGetDeviceProperties(hipDeviceProp_t* props, int) {
    const char* value = std::getenv("FMHA_TEST_NUM_CUS");
    if(value != nullptr) props->multiProcessorCount = std::atoi(value);
    return hipSuccess;
}
"""

_HOST_FMHA_STUB = """#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
namespace ck_tile {
using index_t = int;
struct stream_config {};
struct gfx125_t {};
inline std::string get_device_name() {
    const char* value = std::getenv("FMHA_TEST_DEVICE");
    return value == nullptr ? "gfx1250" : value;
}
template <bool> struct SimplifiedGenericAttentionMask {};
enum class BlockFmhaPipelineEnum { QRKSVS, QRKSVS_TDM, QRKSVS_TDM_SCHED, QRKSVS_TDM_PINGPONG, QRKSVS_TDM_SCHED_FP8, QRKSVS_TDM_CR };
enum class BlockAttentionQuantScaleEnum { NO_SCALE, PERTENSOR, BLOCKSCALE, KV_BLOCKSCALE, MX, PERHEAD };
}
enum class mask_enum { no_mask, mask_top_left, mask_bottom_right, window_generic };
enum class bias_enum { no_bias, elementwise_bias, alibi };
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx, perhead };
struct FmhaFwdBf16 {};
struct FmhaFwdFp8 {};
struct FmhaFwdFp16 {};
struct FmhaMasks {
    struct NoMask {};
    struct GenericMask {};
    struct CausalMask {};
};
inline constexpr ck_tile::index_t fmha_fwd_largest_n_tile_size = 128;
template <int, typename, bool, int, int, int, int, int, int, bool, auto, bool,
          typename, auto, bool, bool, auto, bool, bool, bool, bool, bool, bool, bool, int = -1,
          bool = false>
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
    int stride_q = 192, nhead_stride_q = 24576;
    int window_size_left = -1, window_size_right = -1;
    int batch = 2, nhead_q = 8, nhead_k = 2;
    const void* sink_ptr = nullptr;
    const void* seqlen_k_ptr = nullptr;
    const void* cu_seqlen_k_ptr = nullptr;
    const void* o_ptr = nullptr;
    int stride_o = 1024, nhead_stride_o = 128, batch_stride_o = 262144;
    const void *q_ptr = nullptr, *k_ptr = nullptr, *v_ptr = nullptr, *lse_ptr = nullptr;
    const void *seqstart_q_ptr = nullptr, *seqstart_k_ptr = nullptr;
    const void *seqlen_q_ptr = nullptr, *cu_seqlen_q_ptr = nullptr;
    int stride_k = 1024, stride_v = 1024;
    int nhead_stride_k = 128, nhead_stride_v = 128, nhead_stride_lse = 128;
    int batch_stride_q = 131072, batch_stride_k = 131072, batch_stride_v = 131072;
    int batch_stride_lse = 1024, mask_type = 0;
    float scale_s = 0.08838835f;
    const void* bias_ptr = reinterpret_cast<const void*>(4096);
    int stride_bias = 1024, nhead_stride_bias = 131072, batch_stride_bias = 1048576;
    float logits_soft_cap = 8.f;
    int sink_size = 0, min_seqlen_q = 0;
    int block_scale_size_q = 0, block_scale_size_kv = 0;
};
inline bool fmha_fwd_can_merge_head_groups(const fmha_fwd_args&, int, bool, bool, bool = false) {
    return false;
}
template <typename Trait, typename Arch> struct dispatch_id;
template <typename Trait, typename Arch>
float fmha_fwd_(const ck_tile::stream_config&, const fmha_fwd_args&) {
    return dispatch_id<Trait, Arch>::value;
}
"""

_HOST_MAIN = """
#include <iostream>
int main() {
    fmha_fwd_traits t;
    fmha_fwd_args a;
    int mask, bias, scale, arg_q, arg_v, max_q, has_cu_sk, has_sk_ptr, has_sink_ptr, o_offset, bshd;
    int lse_offset, lse_head_stride, lse_batch_stride, has_seqstart, has_sq_ptr, has_cu_sq;
    int input_offset, input_padding, input_parts;
    if(!(std::cin >> t.hdim_q >> t.hdim_v >> a.seqlen_q >> a.seqlen_k
         >> t.is_group_mode >> t.is_v_rowmajor >> t.has_logits_soft_cap >> mask >> bias
         >> t.has_lse >> t.has_dropout >> scale >> t.skip_min_seqlen_q >> t.has_sink
         >> t.data_type >> arg_q >> arg_v >> max_q >> a.batch >> a.nhead_q
         >> a.nhead_k >> a.window_size_left >> a.window_size_right
         >> has_cu_sk >> has_sk_ptr >> has_sink_ptr >> o_offset >> a.stride_o
         >> a.nhead_stride_o >> a.batch_stride_o >> bshd
         >> lse_offset >> lse_head_stride >> lse_batch_stride >> has_seqstart
         >> has_sq_ptr >> has_cu_sq >> a.scale_s >> input_offset >> input_padding
         >> input_parts >> a.block_scale_size_q >> a.block_scale_size_kv)) return 2;
    t.mask_type = static_cast<mask_enum>(mask);
    t.bias_type = static_cast<bias_enum>(bias);
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    a.hdim_q = arg_q < 0 ? t.hdim_q : arg_q;
    a.hdim_v = arg_v < 0 ? t.hdim_v : arg_v;
    a.max_seqlen_q = max_q < 0 ? a.seqlen_q : max_q;
    a.cu_seqlen_k_ptr = has_cu_sk ? &a : nullptr;
    a.seqstart_q_ptr = a.seqstart_k_ptr = has_seqstart ? &a : nullptr;
    a.seqlen_k_ptr = has_sk_ptr ? &a : nullptr;
    a.seqlen_q_ptr = has_sq_ptr ? &a : nullptr;
    a.cu_seqlen_q_ptr = has_cu_sq ? &a : nullptr;
    a.sink_ptr = has_sink_ptr ? &a : nullptr;
    a.sink_size = t.has_sink ? 65 : 0;
    a.o_ptr = reinterpret_cast<const void*>(std::uintptr_t{4096} + o_offset);
    if(bshd) {
        a.q_ptr = reinterpret_cast<const void*>(std::uintptr_t{4096} + ((input_parts & 1) ? input_offset : 0));
        a.k_ptr = reinterpret_cast<const void*>(std::uintptr_t{4096} + ((input_parts & 2) ? input_offset : 0));
        a.v_ptr = reinterpret_cast<const void*>(std::uintptr_t{4096} + ((input_parts & 4) ? input_offset : 0));
        a.stride_q = a.nhead_q * a.hdim_q + ((input_parts & 1) ? input_padding : 0);
        a.stride_k = a.nhead_k * a.hdim_q + ((input_parts & 2) ? input_padding : 0);
        a.stride_v = a.nhead_k * a.hdim_v + ((input_parts & 4) ? input_padding : 0);
        a.nhead_stride_q = a.nhead_stride_k = a.hdim_q;
        a.nhead_stride_v = a.hdim_v;
        a.batch_stride_q = a.stride_q * a.seqlen_q;
        a.batch_stride_k = a.stride_k * a.seqlen_k;
        a.batch_stride_v = a.stride_v * a.seqlen_k;
        if(bshd == 2) {
            a.stride_q = a.stride_k = a.hdim_q;
            a.stride_v = a.hdim_v;
            a.nhead_stride_q = a.seqlen_q * a.hdim_q;
            a.nhead_stride_k = a.seqlen_k * a.hdim_q;
            a.nhead_stride_v = a.seqlen_k * a.hdim_v;
        }
        a.lse_ptr = t.has_lse ? reinterpret_cast<const void*>(std::uintptr_t{4096} + lse_offset) : nullptr;
        a.nhead_stride_lse = lse_head_stride < 0 ? a.seqlen_q : lse_head_stride;
        a.batch_stride_lse = lse_batch_stride < 0 ? a.nhead_q * a.seqlen_q : lse_batch_stride;
        a.mask_type = mask;
    }
    std::cout << fmha_fwd(t, a, {}) << '\\n';
}
"""


class _CompiledDispatcher:
    def __init__(self, test, pool, kernels, filter_fn=None, mutate=None):
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
        signatures = list(dict.fromkeys(_RENDERED_CALL.findall(api)))
        ids = {signature: index + 1 for index, signature in enumerate(signatures)}
        self.kernel_ids = []
        for kernel in kernels:
            single = FmhaFwdApiPool()
            single.register_traits(kernel.api_trait())
            (signature,) = dict.fromkeys(
                _RENDERED_CALL.findall(single.render("unused"))
            )
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
        (root / "fmha_fwd.hpp").write_text(
            _HOST_FMHA_STUB
            + f'\n#include "{_FMHA_EXAMPLE_DIR / "fmha_fwd_cr_support.hpp"}"\n'
            + specializations
        )
        (root / "fmha_fwd_api.cpp").write_text(
            (api if mutate is None else mutate(api)) + _HOST_MAIN
        )
        self.binary = root / "dispatcher"
        compile_result = subprocess.run(
            [
                _HOST_CXX,
                "-std=c++17",
                "-O0",
                # As in the CK build: a dispatch condition that compares ck_tile::index_t
                # with the unsigned num_cus fails there.
                "-Wsign-compare",
                "-Werror",
                "-I",
                str(root),
                "-I",
                str(_CK_ROOT / "include"),
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
            if name in ("mode", "hdim", "dtype"):
                return getattr(kernel, "F_" + name)
            if name in ("bm0", "bn0", "occupancy"):
                return getattr(kernel.F_tile, "F_" + name)
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

    def run(self, device="gfx1250", num_cus=256, **changes):
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
            batch=2,
            nhead_q=8,
            nhead_k=2,
            window_size_left=-1,
            window_size_right=-1,
            has_cu_sk=0,
            has_sk_ptr=0,
            has_sink_ptr=0,
            o_offset=0,
            stride_o=1024,
            nhead_stride_o=128,
            batch_stride_o=262144,
            bshd=0,
            lse_offset=0,
            lse_head_stride=-1,
            lse_batch_stride=-1,
            has_seqstart=0,
            has_sq_ptr=0,
            has_cu_sq=0,
            scale_s=0.08838835,
            input_offset=0,
            input_padding=0,
            input_parts=7,
            block_scale_size_q=0,
            block_scale_size_kv=0,
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
            FMHA_TEST_NUM_CUS=str(num_cus),
        )
        result = subprocess.run(
            [str(self.binary)],
            input=" ".join(map(str, case.values())) + "\n",
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return [int(value) for value in result.stdout.split()]


class TestGfx125TdmCrCodegen(unittest.TestCase):
    def test_lse_keeps_original_dispatch_including_python_receipt(self):
        for receipt in (0, 5):
            with self.subTest(receipt=receipt):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"],
                    "fmha_fwd_*_lse_*",
                    receipt,
                    [32, 64, 96, 128, 160, 192],
                    "simplified",
                )
                self.assertTrue(kernels)
                self.assertTrue(all(k.F_pipeline.F_lse == "t" for k in kernels))
                self.assertFalse(any(k.F_pipeline.tag == "qr_tdm_cr" for k in kernels))
                # Compile the real dispatcher with launch-ID stubs. Limit this
                # check to one score/sequence feature at a time; the policy
                # assertion above covers the entire LSE inventory.
                selected_kernels = []
                selected_pool = FmhaFwdApiPool()
                for kernel in kernels:
                    p = kernel.F_pipeline
                    feature_count = (
                        (p.F_logits == "t")
                        + (p.F_bias != "no")
                        + (p.F_sink == "t")
                        + (p.F_skip == "t")
                    )
                    if (
                        kernel.F_dtype not in ("fp16", "bf16")
                        or p.F_dropout != "f"
                        or p.F_qscale != "no"
                    ):
                        continue
                    if feature_count and (kernel.F_hdim != 128 or feature_count != 1):
                        continue
                    selected_kernels.append(kernel)
                    selected_pool.register_traits(kernel.api_trait())
                dispatch = _CompiledDispatcher(self, selected_pool, selected_kernels)
                features = [dict(hdim_q=d, hdim_v=d) for d in (32, 64, 96, 128, 160)]
                features += [dict(hdim_q=65, hdim_v=65), dict(bias=1), dict(bias=2)]
                if receipt == 0:
                    features += [
                        dict(hdim_q=192, hdim_v=128),
                        dict(logits=1),
                        dict(sink=1),
                        dict(skip=1),
                    ]
                for dtype, group, length, mask, feature in itertools.product(
                    ("fp16", "bf16"), (0, 1), (257, 8192), (0, 2), features
                ):
                    if receipt == 5 and group and feature.get("bias") == 1:
                        continue  # The Python receipt exposes elementwise bias in batch only.
                    args = dict(
                        dtype=dtype,
                        hdim_q=128,
                        hdim_v=128,
                        group=group,
                        has_seqstart=group,
                        bshd=1,
                        seqlen_q=length,
                        seqlen_k=length,
                        mask=mask,
                        window_size_right=0 if mask else -1,
                        lse=1,
                    )
                    args.update(feature)
                    args.update(
                        stride_o=8 * args["hdim_v"],
                        nhead_stride_o=args["hdim_v"],
                        batch_stride_o=length * 8 * args["hdim_v"],
                    )
                    with self.subTest(receipt=receipt, **args):
                        selected = dispatch.run(**args)[0]
                        self.assertGreater(selected, 0)
                        kernel = next(
                            k for k, i in dispatch.kernel_ids if i == selected
                        )
                        self.assertNotEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                        self.assertEqual(kernel.F_pipeline.F_lse, "t")

    def test_expanded_default_dispatch_preserves_cr_geometry(self):
        _, generated = get_fwd_blobs(
            ["gfx1250"], "*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        kernels = []
        pool = FmhaFwdApiPool()
        for kernel in generated:
            p = kernel.F_pipeline
            if (
                kernel.F_dtype not in ("fp16", "bf16")
                or p.tag not in ("qr_tdm", "qr_tdm_sched", "qr_tdm_cr")
                or p.F_dropout != "f"
                or p.F_qscale != "no"
                or p.F_lse != "f"
            ):
                continue
            features = (
                (p.F_logits == "t")
                + (p.F_bias != "no")
                + (p.F_sink == "t")
                + (p.F_skip == "t")
            )
            if features and (kernel.F_hdim != 128 or features != 1):
                continue
            kernels.append(kernel)
            pool.register_traits(kernel.api_trait())
        mixed = _CompiledDispatcher(self, pool, kernels)
        cr = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag == "qr_tdm_cr"
        )
        names = [{i: k.name for k, i in d.kernel_ids} for d in (mixed, cr)]
        features = [
            dict(hdim_q=q, hdim_v=v)
            for q, v in (
                (32, 32),
                (96, 96),
                (160, 160),
                (17, 31),
                (65, 95),
                (129, 159),
                (161, 127),
            )
        ]
        features += [
            dict(hdim_q=128, hdim_v=128, **extra)
            for extra in (
                dict(bias=1),
                dict(bias=2),
                dict(logits=1),
                dict(sink=1),
                dict(group=1, has_seqstart=1, skip=1),
                dict(has_sink_ptr=1),
            )
        ]
        for dtype, group, sq, mask, feature in itertools.product(
            ("bf16", "fp16"), (0, 1), (257, 8192), (0, 2), features
        ):
            # Real sink and skip fields are ignored by the ordinary dense/batch API.
            if feature.get("sink") and not mask:
                continue
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                seqlen_q=sq,
                seqlen_k=sq + 2,
                bshd=1,
                mask=mask,
                lse=0,
                window_size_right=0 if mask else -1,
            )
            args.update(feature)
            args.update(
                stride_o=8 * args["hdim_v"],
                nhead_stride_o=args["hdim_v"],
                batch_stride_o=sq * 8 * args["hdim_v"],
            )
            with self.subTest(**args):
                expected = cr.run(**args)[0]
                self.assertGreater(expected, 0)
                selected = mixed.run(**args)[0]
                self.assertGreater(selected, 0)
                self.assertEqual(names[0][selected], names[1][expected])
                rejected = mixed.run(**args, device="gfx1251")[0]
                if rejected > 0:
                    self.assertNotIn("qr_tdm_cr", names[0][rejected])

    def test_sink_and_skip_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 128], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, group, sink, skip, virtual, mask in itertools.product(
            ("fp16", "bf16"), (0, 1), (0, 1), (0, 1), (0, 1), (0, 2)
        ):
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                hdim_q=32,
                hdim_v=32,
                bshd=1,
                seqlen_q=257,
                seqlen_k=515,
                sink=sink,
                skip=skip,
                has_sink_ptr=virtual,
                mask=mask,
                lse=0,
                stride_o=256,
                nhead_stride_o=32,
                batch_stride_o=257 * 256,
                window_size_left=31,
                window_size_right=0 if mask else -1,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(
                    kernel.F_pipeline.F_sink, "t" if sink and mask else "f"
                )
                self.assertEqual(
                    kernel.F_pipeline.F_skip, "t" if group and skip else "f"
                )
                self.assertEqual(kernel.F_pipeline.F_spad, "t")
                self.assertEqual(kernel.F_pipeline.F_skpad, "t")

    def test_score_feature_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, group, mask, (dq, dv), (logits, bias) in itertools.product(
            ("fp16", "bf16"),
            (0, 1),
            (0, 1, 2, 3),
            ((32, 32), (65, 95), (128, 128), (160, 160)),
            ((0, 1), (0, 2), (1, 0)),
        ):
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                hdim_q=dq,
                hdim_v=dv,
                bshd=1,
                logits=logits,
                bias=bias,
                seqlen_q=257,
                seqlen_k=515,
                mask=mask,
                lse=0,
                window_size_left=31 if mask == 3 else -1,
                window_size_right=0 if mask else -1,
                stride_o=8 * dv + 1,
                nhead_stride_o=dv,
                batch_stride_o=257 * (8 * dv + 1),
                o_offset=2,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.F_logits, "t" if logits else "f")
                self.assertEqual(
                    kernel.F_pipeline.F_bias, ("no", "bias", "alibi")[bias]
                )
                self.assertEqual(kernel.F_pipeline.F_spad, "t")
                self.assertEqual(kernel.F_pipeline.F_skpad, "t")

        # Native long causal ALiBi uses M128/N64. Other modes retain their
        # existing short, window, group, real-sink and padded-head geometries.
        for dtype, (sq, group, left, sink, dq, mask, expected_m) in itertools.product(
            ("bf16", "fp16"),
            (
                (2048, 0, -1, 0, 160, 2, 64),
                (2049, 0, -1, 0, 160, 2, 128),
                (8192, 0, 63, 0, 160, 2, 64),
                (8192, 1, -1, 0, 160, 2, 64),
                (8192, 0, -1, 1, 160, 2, 64),
                (8192, 0, -1, 0, 159, 2, 128),
                (8192, 0, -1, 0, 160, 0, 128),
            ),
        ):
            with self.subTest(
                dtype=dtype, sq=sq, group=group, left=left, sink=sink, dq=dq, mask=mask
            ):
                selected = dispatch.run(
                    dtype=dtype,
                    hdim_q=dq,
                    hdim_v=160,
                    seqlen_q=sq,
                    seqlen_k=sq,
                    group=group,
                    has_seqstart=group,
                    bshd=1,
                    bias=2,
                    mask=mask,
                    lse=0,
                    sink=sink,
                    window_size_left=left,
                    window_size_right=0 if mask else -1,
                )[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(
                    (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), (expected_m, 64)
                )

    def test_head_dimensions_and_padding(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, group, mask, (dq, dv) in itertools.product(
            ("fp16", "bf16"),
            (0, 1),
            (0, 2),
            (
                (1, 1),
                (31, 17),
                (32, 32),
                (33, 63),
                (64, 64),
                (65, 95),
                (96, 96),
                (127, 113),
                (128, 128),
                (129, 159),
                (160, 160),
                (161, 127),
                (192, 128),
                (31, 128),
                (96, 160),
                (160, 96),
                (32, 160),
            ),
        ):
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                bshd=1,
                hdim_q=dq,
                hdim_v=dv,
                mask=mask,
                lse=0,
                window_size_right=0 if mask else -1,
                seqlen_q=257,
                seqlen_k=515,
                stride_o=8 * dv + 1,
                nhead_stride_o=dv,
                batch_stride_o=257 * (8 * dv + 1),
                input_offset=2,
                input_padding=1,
                o_offset=2,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                self.assertGreaterEqual(kernel.F_tile.F_bk0max, dq)
                self.assertGreaterEqual(kernel.F_tile.F_bn1, dv)
                if dq != kernel.F_tile.F_bk0max or dv != kernel.F_tile.F_bn1:
                    self.assertEqual(kernel.F_pipeline.F_dpad, "t")
                    self.assertEqual(kernel.F_pipeline.F_dvpad, "t")
                else:
                    self.assertEqual(kernel.F_pipeline.F_dpad, "f")
                    self.assertEqual(kernel.F_pipeline.F_dvpad, "f")
        for dq, dv in (
            (0, 32),
            (32, 0),
            (193, 128),
            (192, 129),
            (256, 256),
            (512, 512),
        ):
            with self.subTest(dq=dq, dv=dv):
                self.assertEqual(dispatch.run(hdim_q=dq, hdim_v=dv, bshd=1), [-1])

    def test_exact_heads_precede_padded_fallbacks(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        covered = {}
        for dtype, group, mask, (batch, sq), (dq, dv) in itertools.product(
            ("fp16", "bf16"),
            (0, 1),
            (0, 1, 2, 3),
            ((2, 257), (8, 257), (2, 8192)),
            ((32, 32), (64, 64), (96, 96), (128, 128), (160, 160), (192, 128)),
        ):
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                bshd=1,
                hdim_q=dq,
                hdim_v=dv,
                mask=mask,
                lse=0,
                batch=batch,
                window_size_left=31 if mask == 3 else -1,
                window_size_right=0 if mask else -1,
                seqlen_q=sq,
                seqlen_k=sq,
                stride_o=8 * dv,
                nhead_stride_o=dv,
                batch_stride_o=sq * 8 * dv,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.F_dpad, "f")
                self.assertEqual(kernel.F_pipeline.F_dvpad, "f")
                covered.setdefault((dtype, group, dq), set()).add(kernel.F_tile.F_bm0)
        for key, tiles in covered.items():
            with self.subTest(coverage=key):
                # Native D32 reuses M64 with N64/N128. Packed D192 M128
                # is qualified for BF16; FP16 retains its M64 fallback.
                expected = (
                    {64} if key[2] == 32 or key == ("fp16", 1, 192) else {64, 128}
                )
                self.assertEqual(tiles, expected)

    def test_virtual_sink_dense_geometry(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 96], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, (dq, dv), length, virtual, mask, group, (
            bias,
            logits,
        ) in itertools.product(
            ("fp16", "bf16"),
            ((32, 32), (65, 95)),
            (257, 2048, 2049, 8192),
            (0, 1),
            (0, 2),
            (0, 1),
            ((0, 0), (1, 0), (0, 1)),
        ):
            args = dict(
                dtype=dtype,
                bias=bias,
                logits=logits,
                group=group,
                has_seqstart=group,
                hdim_q=dq,
                hdim_v=dv,
                bshd=1,
                seqlen_q=length,
                seqlen_k=length,
                has_sink_ptr=virtual,
                mask=mask,
                window_size_right=0 if mask else -1,
                stride_o=8 * dv,
                nhead_stride_o=dv,
                batch_stride_o=length * 8 * dv,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                if length <= 2048:
                    expected = (64, 64)
                elif virtual and not mask and not group and not (bias or logits):
                    expected = (128, 64)
                else:
                    expected = (64, 128) if dq == 32 else (128, 128)
                self.assertEqual((kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected)

    def test_native_d64_virtual_sink_geometry(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*qr_tdm_cr*_nlogits_nbias*_nskip*_nsink*",
            0,
            [64],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, length, (batch, heads), group, virtual, mask in itertools.product(
            ("bf16", "fp16"),
            (2048, 2049, 4096, 4097, 8192, 8193),
            ((2, 4), (1, 9), (1, 32), (1, 33)),
            (0, 1),
            (0, 1),
            (0, 2),
        ):
            with self.subTest(
                dtype=dtype,
                length=length,
                batch=batch,
                heads=heads,
                group=group,
                virtual=virtual,
                mask=mask,
            ):
                selected = dispatch.run(
                    dtype=dtype,
                    hdim_q=64,
                    hdim_v=64,
                    batch=batch,
                    nhead_q=heads,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=length,
                    seqlen_k=length,
                    group=group,
                    has_seqstart=group,
                    has_sink_ptr=virtual,
                    mask=mask,
                    window_size_right=0 if mask else -1,
                    stride_o=heads * 64,
                    nhead_stride_o=64,
                    batch_stride_o=length * heads * 64,
                )[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                small = batch * heads <= 32 and (
                    length <= 2048
                    or (mask and (not virtual or group))
                    or (virtual and not mask and not group and length <= 4096)
                )
                if dtype == "bf16" and group and mask and not virtual:
                    small = small and batch * heads * ((length + 127) // 128) <= 256
                cooperative = (
                    virtual
                    and mask
                    and not group
                    and batch * heads <= 8
                    and 2048 < length <= 8192
                )
                expected = (
                    (64, 128) if cooperative else (64, 64) if small else (128, 128)
                )
                self.assertEqual((kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected)
                self.assertEqual(kernel.F_pipeline.F_dpad, "f")
        for keys, left, right in itertools.product(
            (2047, 2048, 8192, 8193), (-1, 31), (0, 1)
        ):
            with self.subTest(keys=keys, left=left, right=right):
                selected = dispatch.run(
                    dtype="bf16",
                    hdim_q=64,
                    hdim_v=64,
                    batch=1,
                    nhead_q=8,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=4096,
                    seqlen_k=keys,
                    has_sink_ptr=1,
                    mask=1,
                    window_size_left=left,
                    window_size_right=right,
                    stride_o=512,
                    nhead_stride_o=64,
                    batch_stride_o=4096 * 512,
                )[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                cooperative = left < 0 and right == 0 and 2048 <= keys <= 8192
                self.assertEqual(
                    kernel.F_tile.F_bn0 == 128 and kernel.F_tile.F_bm0 == 64,
                    cooperative,
                )

    def test_native_d192_virtual_sink_dense_geometry(self):
        _, generated = get_fwd_blobs(
            ["gfx1250"], "*_nlogits_nbias*_nskip*_nsink*", 0, [192], "simplified"
        )
        # Include the other pipelines so the common dispatch ordering is tested
        # when a native problem has no virtual sink.
        kernels = [
            k
            for k in generated
            if k.F_dtype in ("bf16", "fp16")
            and k.F_pipeline.F_dropout == "f"
            and k.F_pipeline.F_qscale == "no"
        ]
        pool = FmhaFwdApiPool()
        for kernel in kernels:
            pool.register_traits(kernel.api_trait())
        dispatch = _CompiledDispatcher(self, pool, kernels)
        without_virtual_tile = _CompiledDispatcher(
            self,
            pool,
            kernels,
            filter_fn=lambda t: not (
                t.pipeline_tag == "qr_tdm_cr"
                and t.bm0 == 128
                and t.dpad == "f"
                and t.dvpad == "f"
                and t.mask == "no"
            ),
        )
        for dtype, (sq, sk), heads, group, virtual in itertools.product(
            ("bf16", "fp16"),
            ((8191, 8192), (8192, 8191), (8192, 8192), (8193, 8192), (8192, 16384)),
            (4, 32, 33),
            (0, 1),
            (0, 1),
        ):
            with self.subTest(
                dtype=dtype,
                sq=sq,
                sk=sk,
                heads=heads,
                group=group,
                virtual=virtual,
                lse=0,
            ):
                args = dict(
                    dtype=dtype,
                    hdim_q=192,
                    hdim_v=128,
                    batch=1,
                    nhead_q=heads,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    group=group,
                    has_seqstart=group,
                    has_sink_ptr=virtual,
                    mask=0,
                    lse=0,
                    stride_o=heads * 128,
                    nhead_stride_o=128,
                    batch_stride_o=sq * heads * 128,
                )
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                if not virtual:
                    previous = without_virtual_tile.run(**args)[0]
                    self.assertGreater(previous, 0)
                    previous_kernel = next(
                        k for k, i in without_virtual_tile.kernel_ids if i == previous
                    )
                    self.assertEqual(kernel.name, previous_kernel.name)
                    continue
                expected = (
                    128
                    if virtual and not group and min(sq, sk) >= 8192 and heads <= 32
                    else 64
                )
                self.assertEqual(kernel.F_tile.F_bm0, expected)
                self.assertEqual(kernel.F_tile.F_bn0, 64)
                self.assertEqual(kernel.F_pipeline.F_dpad, "f")

    def test_padded_m128_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, group, mask, (dq, dv) in itertools.product(
            ("fp16", "bf16"),
            (0, 1),
            (0, 2),
            ((31, 17), (33, 63), (65, 95), (127, 113), (129, 159), (161, 127)),
        ):
            batch, length = (2, 8192) if dq == 161 else (8, 8192)
            args = dict(
                dtype=dtype,
                group=group,
                has_seqstart=group,
                bshd=1,
                batch=batch,
                hdim_q=dq,
                hdim_v=dv,
                mask=mask,
                lse=0,
                window_size_right=0 if mask else -1,
                seqlen_q=length,
                seqlen_k=length,
                stride_o=8 * dv,
                nhead_stride_o=dv,
                batch_stride_o=length * 8 * dv,
            )
            with self.subTest(**args):
                selected = dispatch.run(**args)[0]
                self.assertGreater(selected, 0)
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.F_dpad, "t")
                self.assertEqual(kernel.F_pipeline.F_dvpad, "t")
                self.assertGreaterEqual(kernel.F_tile.F_bk0max, dq)
                self.assertGreaterEqual(kernel.F_tile.F_bn1, dv)
                self.assertEqual(
                    kernel.F_tile.F_bm0,
                    64 if dq == 161 and group else 128,
                )

    def test_long_alibi_tile_selection(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        # Per-sequence heads matter for sparse groups; total packed Q must not
        # substitute for the maximum logical query length at either boundary.
        for dtype, group in itertools.product(("bf16", "fp16"), (0, 1)):
            for dq, dv, heads, length, mask, virtual, expected in (
                (31, 17, 4, 8192, 2, 0, (128, 128)),
                (31, 17, 16, 8191, 2, 0, (128, 128)),
                (31, 17, 16, 8192, 2, 0, (128, 64)),
                (33, 63, 4, 8192, 2, 0, (128, 128)),
                (33, 63, 16, 8192, 1, 0, (128, 64)),
                (65, 95, 16, 8192, 2, 0, (128, 64)),
                (97, 113, 4, 4095, 2, 0, (128, 128)),
                (97, 113, 4, 4096, 2, 0, (128, 64)),
                (64, 64, 4, 4095, 2, 0, (128, 128)),
                (64, 64, 4, 4096, 2, 0, (64, 64)),
                (64, 64, 24, 8192, 1, 0, (64, 64)),
                (31, 17, 24, 2048, 2, 0, (64, 64)),
                (97, 113, 24, 8192, 0, 0, (128, 128)),
                (31, 17, 24, 8192, 2, 1, (128, 128)),
                (64, 64, 24, 8192, 2, 1, (128, 128)),
            ):
                with self.subTest(
                    dtype=dtype,
                    group=group,
                    lse=0,
                    dq=dq,
                    heads=heads,
                    length=length,
                    mask=mask,
                    virtual=virtual,
                ):
                    selected = dispatch.run(
                        dtype=dtype,
                        group=group,
                        has_seqstart=group,
                        has_sq_ptr=group,
                        has_sk_ptr=group,
                        bshd=1,
                        batch=2,
                        hdim_q=dq,
                        hdim_v=dv,
                        nhead_q=heads,
                        nhead_k=heads // 4,
                        seqlen_q=length * (2 if group else 1),
                        seqlen_k=(length + 64) * (2 if group else 1),
                        max_seqlen_q=length,
                        mask=mask,
                        bias=2,
                        lse=0,
                        window_size_right=0 if mask else -1,
                        has_sink_ptr=virtual,
                        stride_o=heads * dv,
                        nhead_stride_o=dv,
                        batch_stride_o=length * heads * dv,
                    )[0]
                    self.assertGreater(selected, 0)
                    kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                    self.assertEqual(
                        (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected
                    )

    def test_long_native_active_work_selection(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [96, 160], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype, dim in itertools.product(("bf16", "fp16"), (96, 160)):
            b4_m = 128 if dtype == "bf16" else 64
            for group, batch, heads, sq, sk, maximum, mask, expected_m in (
                (0, 2, 8, 8192, 8256, 8192, 2, 128),
                (0, 2, 8, 8192, 257, 8192, 2, 64),
                (0, 2, 8, 8192, 257, 8192, 1, 128),
                (1, 2, 8, 16448, 16576, 8192, 2, 128),
                (1, 2, 8, 12352, 12480, 8192, 2, 128),
                (1, 2, 8, 8288, 8352, 8192, 2, 64),
                (1, 2, 8, 16448, 640, 8192, 2, 64),
                (1, 2, 8, 16448, 640, 8192, 1, 128),
                (1, 4, 8, 32896, 33152, 8192, 2, b4_m),
                (1, 4, 8, 16383, 16512, 8192, 2, 64),
                (1, 4, 8, 16384, 16512, 8192, 2, b4_m),
                (1, 4, 8, 20608, 20864, 8192, 2, b4_m),
                (1, 4, 8, 32896, 1028, 8192, 2, 64),
                (1, 4, 8, 32896, 1028, 8192, 1, b4_m),
                (1, 3, 8, 24672, 24864, 8192, 2, 64),
                (1, 4, 4, 32896, 33152, 8192, 2, 64),
                (1, 4, 12, 32896, 33152, 8192, 2, 128),
                (1, 2, 16, 16448, 16576, 8192, 2, 128),
                (1, 2, 4, 16448, 16576, 8192, 2, 64),
                (1, 4, 8, 16512, 16768, 4096, 2, 64),
                (1, 2, 8, 16448, 16576, 8191, 2, 64),
                (0, 1, 4, 32768, 32832, 32768, 2, 64),
            ):
                args = dict(
                    dtype=dtype,
                    hdim_q=dim,
                    hdim_v=dim,
                    lse=0,
                    group=group,
                    has_seqstart=group,
                    has_sq_ptr=group,
                    has_sk_ptr=group,
                    batch=batch,
                    nhead_q=heads,
                    nhead_k=heads // 4,
                    bshd=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    max_seqlen_q=maximum,
                    mask=mask,
                    window_size_right=0,
                    stride_o=heads * dim,
                    nhead_stride_o=dim,
                    batch_stride_o=maximum * heads * dim,
                )
                with self.subTest(**args):
                    selected = dispatch.run(**args)[0]
                    self.assertGreater(selected, 0)
                    kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                    self.assertEqual(kernel.F_tile.F_bm0, expected_m)
                    self.assertEqual(kernel.F_tile.F_bn0, 64)
                    if group and expected_m == 128 and batch * heads <= 32:
                        # Score transforms and virtual sinks retain their
                        # independently qualified small-grid preference.
                        for changes in (dict(bias=2), dict(has_sink_ptr=1)):
                            selected = dispatch.run(**(args | changes))[0]
                            kernel = next(
                                k for k, i in dispatch.kernel_ids if i == selected
                            )
                            self.assertEqual(kernel.F_tile.F_bm0, 64)

    def test_scalar_views_preserve_default_pipeline_preference(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_[bf][fp]16_*_nlogits_nbias*_ndropout_nskip_nqscale*",
            0,
            [32, 64, 96, 128, 160, 192],
            "simplified",
        )
        default = _CompiledDispatcher(self, pool, kernels)
        fallback = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag != "qr_tdm_cr"
        )
        cr_only = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag == "qr_tdm_cr"
        )
        # Native short and group scalar views retain their fallback. Qualified
        # long batch views, new dimensions and padded heads use CR geometry.
        for dtype, (dq, dv), mask, view, group, (sq, sk) in itertools.product(
            ("fp16", "bf16"),
            (
                (32, 32),
                (64, 64),
                (80, 80),
                (96, 96),
                (120, 120),
                (128, 128),
                (160, 160),
                (192, 128),
            ),
            (0, 2),
            ("qkv", "o"),
            (0, 1),
            ((256, 512), (8192, 8192)),
        ):
            args = dict(
                dtype=dtype,
                hdim_q=dq,
                hdim_v=dv,
                bshd=1,
                batch=2,
                nhead_q=2,
                nhead_k=1,
                seqlen_q=sq,
                seqlen_k=sk,
                group=group,
                has_seqstart=group,
                mask=mask,
                window_size_right=0 if mask else -1,
                lse=0,
                stride_o=2 * dv + (view == "o"),
                nhead_stride_o=dv,
                batch_stride_o=sq * (2 * dv + (view == "o")),
                o_offset=2 if view == "o" else 0,
                input_offset=2 if view == "qkv" else 0,
                input_padding=1 if view == "qkv" else 0,
            )
            with self.subTest(**args):
                native = (dq, dv) in ((64, 64), (128, 128), (192, 128))
                long_native = (
                    native
                    and not group
                    and sq >= 8192
                    and sk >= 8192
                    and (
                        dq == 64 or dq == 128 or (dq == 192 and (mask or view == "qkv"))
                    )
                )
                preferred = fallback if native and not long_native else cr_only
                selected = preferred.run(**args)[0]
                self.assertGreater(selected, 0)
                expected = next(
                    k.name for k, i in preferred.kernel_ids if i == selected
                )
                selected = default.run(**args)[0]
                self.assertGreater(selected, 0)
                actual = next(k.name for k, i in default.kernel_ids if i == selected)
                self.assertEqual(actual, expected)
                self.assertGreater(cr_only.run(**args)[0], 0)
                if native:
                    # The original path can multiply the masked -infinity by a
                    # nonpositive scale. Native CR must handle scalar views too.
                    for scale in (0, -0.19):
                        scaled = args | dict(scale_s=scale)
                        selected = default.run(**scaled)[0]
                        kernel = next(k for k, i in default.kernel_ids if i == selected)
                        self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                if not native:
                    # Aligned and scalar views share the qualified CR preference.
                    aligned = args | dict(
                        input_offset=0,
                        input_padding=0,
                        o_offset=0,
                        stride_o=2 * dv,
                        batch_stride_o=sq * 2 * dv,
                    )
                    selected = default.run(**aligned)[0]
                    original = cr_only.run(**aligned)[0]
                    actual = next(
                        k.name for k, i in default.kernel_ids if i == selected
                    )
                    expected = next(
                        k.name for k, i in cr_only.kernel_ids if i == original
                    )
                    self.assertEqual(actual, expected)
                elif view == "qkv" and (sq <= 2048 or mask):
                    aligned = args | dict(input_offset=0, input_padding=0)
                    selected = default.run(**aligned)[0]
                    kernel = next(k for k, i in default.kernel_ids if i == selected)
                    self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                    if dq == 192 and sq >= 8192 and not group:
                        self.assertEqual(kernel.F_tile.F_bm0, 128)
                    # Within measured preference ranges, nonpositive scales
                    # must reach the CR path that scales before masking.
                    for scale in (0, -0.19):
                        self.assertEqual(
                            default.run(**(aligned | dict(scale_s=scale))), [selected]
                        )

    def test_sched_supported_inputs_select_cr_only_without_lse(self):
        dispatchers = {}
        for pipeline in ("qr_tdm_sched", "qr_tdm_cr"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*_{pipeline}_*", 0, [64, 128, 192], "simplified"
            )
            dispatchers[pipeline] = _CompiledDispatcher(self, pool, kernels)
        sched, cr = (dispatchers[p] for p in ("qr_tdm_sched", "qr_tdm_cr"))
        covered = set()
        # Exercise generated predicates, including GQA, independent strides,
        # physical/logical lengths, both mask origins and finite windows.
        for dtype, (dq, dv), group, mask, lse, variant in itertools.product(
            ("fp16", "bf16"),
            ((64, 64), (128, 128), (192, 128)),
            (0, 1),
            (0, 1, 2, 3),
            (0, 1),
            range(8),
        ):
            sq, sk = ((128, 512), (256, 513), (512, 4096), (4096, 512))[variant % 4]
            heads, kv_heads = ((8, 8), (8, 2), (8, 1), (2, 1))[variant % 4]
            bhsd = variant == 2
            row_stride = dv if bhsd else heads * dv + 8
            if variant == 5:
                row_stride += 1
            args = dict(
                dtype=dtype,
                hdim_q=dq,
                hdim_v=dv,
                group=group,
                bshd=2 if bhsd else 1,
                batch=2,
                nhead_q=heads,
                nhead_k=kv_heads,
                seqlen_q=sq,
                seqlen_k=sk,
                mask=mask,
                lse=lse,
                stride_o=row_stride,
                nhead_stride_o=sq * dv if bhsd else dv,
                batch_stride_o=sq * row_stride * (heads if bhsd else 1),
                has_seqstart=group,
                has_sq_ptr=int(group and variant == 1),
                has_sk_ptr=int(group and variant == 1),
                has_cu_sq=int(variant == 3),
                has_cu_sk=int(variant == 3),
                window_size_left=31 if mask == 3 else -1,
                window_size_right=7 if mask == 3 else (0 if mask else -1),
                scale_s=0.19,
                o_offset=2 if variant == 4 else 0,
                input_offset=2 if variant == 6 else 0,
                input_padding=1 if variant == 7 else 0,
            )
            with self.subTest(**args):
                selected = sched.run(**args)[0]
                if selected == -1:
                    continue
                covered.add((dtype, dq, group, mask, lse))
                selected_cr = cr.run(**args)[0]
                if lse:
                    self.assertEqual(selected_cr, -1)
                    continue
                self.assertGreater(selected_cr, 0)
                kernel = next(k for k, i in cr.kernel_ids if i == selected_cr)
                self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
        self.assertEqual(len(covered), 2 * 3 * 2 * 4 * 2)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_scheduling_flags(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [32, 64, 96, 128, 160, 192], "simplified"
        )
        names = [k.filename for k in kernels]
        excluded = [
            re.sub(r"_d(32|64|96|128|160|192)_", "_d256_", names[0]),
            names[0].replace("_gfx125.", "_gfx12."),
            names[0].replace("_qr_tdm_cr_", "_qr_tdm_"),
        ]
        flags = "-mllvm;-amdgpu-expert-scheduling-mode;-mllvm;-amdgpu-sched-strategy=max-ilp;-mllvm;-amdgpu-barrier-signal-wait-latency=32"
        source_flags = _FMHA_EXAMPLE_DIR / "cmake/fmha_fwd_source_flags.cmake"
        script = f'include("{source_flags}")\n'
        for name in names + excluded:
            expected = flags if name in names else ""
            if name in names and (
                "_d64_" in name
                or ("_d32_" in name and re.search(r"_b(64x128|128x(64|128))x", name))
            ):
                expected += ";-Xclang;-target-feature;-Xclang;-real-true16"
            script += (
                f'ck_tile_fmha_fwd_get_source_compile_options("{name}" actual)\n'
                f'if(NOT "${{actual}}" STREQUAL "{expected}")\n'
                f'  message(FATAL_ERROR "{name}: ${{actual}}")\n'
                "endif()\n"
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.cmake"
            path.write_text(script)
            result = subprocess.run(
                ["cmake", "-P", str(path)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_clang23_native_bf16_cr_uses_dependency_checks(self):
        source_flags = _FMHA_EXAMPLE_DIR / "cmake/fmha_fwd_source_flags.cmake"
        with tempfile.TemporaryDirectory() as directory:
            for version in (23, 24):
                script = f'set(CMAKE_HIP_COMPILER_VERSION {version})\ninclude("{source_flags}")\n'
                for dim in (64, 128):
                    for dtype in ("bf16", "fp16"):
                        for lse in ("nlse", "lse"):
                            name = (
                                f"fmha_fwd_d{dim}_{dtype}_group_qr_tdm_cr_vr_pssk_"
                                f"nlogits_nbias_mask_{lse}_ndropout_nskip_nqscale_ntrload_nsink_gfx125.cpp"
                            )
                            expert = not (
                                version == 23 and dtype == "bf16" and lse == "nlse"
                            )
                            script += (
                                f'ck_tile_fmha_fwd_get_source_compile_options("{name}" flags)\n'
                                'list(FIND flags "-amdgpu-expert-scheduling-mode" position)\n'
                                f"if({'' if expert else 'NOT '}position EQUAL -1)\n"
                                f'  message(FATAL_ERROR "Unexpected scheduling flags: {version}, {name}: ${{flags}}")\n'
                                "endif()\n"
                            )
                path = Path(directory) / "clang.cmake"
                path.write_text(script)
                result = subprocess.run(
                    ["cmake", "-P", str(path)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_d192_packed_masked_tile_preserves_small_query_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*_bf16_group*_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink*",
            0,
            [192],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for batch, sq, sk, full, window in (
            (8, 1024, 33792, (128, 96), (128, 64)),
            (2, 4096, 4096, (128, 64), (128, 64)),
            (1, 128, 4224, (64, 64), (64, 64)),
            (1, 128, 32896, (128, 128), (64, 64)),
            (1, 512, 33280, (128, 128), (64, 64)),
            (1, 512, 4608, (64, 64), (64, 64)),
            (1, 1024, 5120, (64, 64), (64, 64)),
            (1, 2048, 2048, (64, 64), (64, 64)),
            (1, 4096, 4096, (128, 128), (128, 64)),
            # N96 requires at least two workgroups per CU and a long full prefix.
            (1, 3968, 8064, (128, 64), (128, 64)),
            (1, 4096, 8191, (128, 64), (128, 64)),
            (1, 4096, 8192, (128, 96), (128, 64)),
            (2, 2048, 6144, (128, 96), (128, 64)),
            (32, 256, 8448, (128, 96), (128, 64)),
            (64, 128, 8320, (128, 64), (128, 64)),
            # N128 windows need a nearly full M128 grid and a visible prefix.
            (1, 1280, 9472, (128, 128), (64, 64)),
            (1, 1536, 9728, (128, 128), (128, 128)),
            (1, 2048, 6144, (128, 128), (128, 128)),
            (1, 2049, 10241, (128, 64), (128, 64)),
            (1, 2048, 4095, (64, 64), (64, 64)),
            (1, 2048, 4096, (128, 128), (128, 128)),
            (2, 1024, 9216, (128, 128), (128, 128)),
        ):
            for left in (-1, 2047):
                with self.subTest(batch=batch, sq=sq, sk=sk, left=left):
                    selected = dispatch.run(
                        dtype="bf16",
                        hdim_q=192,
                        hdim_v=128,
                        group=1,
                        has_seqstart=1,
                        bshd=1,
                        batch=batch,
                        nhead_q=16,
                        nhead_k=16,
                        seqlen_q=batch * sq,
                        seqlen_k=batch * sk,
                        max_seqlen_q=sq,
                        stride_o=16 * 128,
                        nhead_stride_o=128,
                        mask=2,
                        window_size_left=left,
                        window_size_right=0,
                    )[0]
                    kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                    expected = full if left == -1 else window
                    self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                    self.assertEqual(
                        (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected
                    )

    def test_d64_packed_tiles_preserve_window_and_grid_fallbacks(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*_bf16_group*_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink*",
            0,
            [64],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for batch, sq, sk, left, expected in (
            (1, 512, 33280, -1, (64, 128)),
            (1, 512, 33280, 2047, (64, 128)),
            (1, 512, 33280, 31, (64, 64)),
            (1, 8192, 40960, -1, (128, 128)),
            (32, 256, 256, -1, (128, 64)),
            (64, 128, 128, 2047, (128, 64)),
            (64, 128, 73728, -1, (128, 64)),
        ):
            with self.subTest(batch=batch, sq=sq, sk=sk, left=left):
                selected = dispatch.run(
                    num_cus=256,
                    dtype="bf16",
                    hdim_q=64,
                    hdim_v=64,
                    group=1,
                    has_seqstart=1,
                    bshd=1,
                    batch=batch,
                    nhead_q=16,
                    nhead_k=2,
                    seqlen_q=batch * sq,
                    seqlen_k=batch * sk,
                    max_seqlen_q=sq,
                    stride_o=16 * 64,
                    nhead_stride_o=64,
                    mask=2,
                    window_size_left=left,
                    window_size_right=0,
                )[0]
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual(
                    kernel.F_pipeline.tag,
                    "qr_tdm_sched" if batch == 64 and sk == 73728 else "qr_tdm_cr",
                )
                self.assertEqual((kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected)

    def test_d128_wide_key_tile_is_limited_to_small_packed_grids(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*bf16_group*_qr_tdm_cr_*_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink*",
            0,
            [64, 128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dim, sq, sk, left, mask, expected in (
            (128, 512, 66048, -1, 2, (64, 256)),
            (128, 512, 66048, 2047, 2, (64, 256)),
            (64, 512, 66048, -1, 2, (64, 128)),
            (64, 512, 66048, 2047, 2, (64, 128)),
            (128, 2048, 67584, -1, 2, (64, 64)),
            (128, 8192, 73728, -1, 2, (128, 128)),
            (128, 512, 512, -1, 2, (64, 64)),
            (128, 512, 66048, 31, 2, (64, 64)),
            (128, 512, 66048, 2047, 3, (64, 256)),
            (128, 512, 66048, 31, 3, (64, 64)),
        ):
            with self.subTest(dim=dim, sq=sq, sk=sk, left=left, mask=mask):
                selected = dispatch.run(
                    num_cus=256,
                    dtype="bf16",
                    hdim_q=dim,
                    hdim_v=dim,
                    group=1,
                    has_seqstart=1,
                    bshd=1,
                    batch=1,
                    nhead_q=16,
                    nhead_k=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    max_seqlen_q=sq,
                    stride_o=16 * dim,
                    nhead_stride_o=dim,
                    mask=mask,
                    window_size_left=left,
                    window_size_right=0,
                )[0]
                kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                self.assertEqual((kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), expected)

    def test_packed_cr_tile_tracks_query_grid_size(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*bf16_group*_qr_tdm_cr_*_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink*",
            0,
            [64, 128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dim in (64, 128):
            for num_cus in (128, 256, 304):
                for sq in (512, 2048, 2176, 8192):
                    for left in (-1, 2047):
                        with self.subTest(dim=dim, num_cus=num_cus, sq=sq, left=left):
                            selected = dispatch.run(
                                num_cus=num_cus,
                                dtype="bf16",
                                hdim_q=dim,
                                hdim_v=dim,
                                group=1,
                                has_seqstart=1,
                                bshd=1,
                                batch=1,
                                nhead_q=16,
                                nhead_k=2,
                                seqlen_q=sq,
                                seqlen_k=sq + 8192,
                                max_seqlen_q=sq,
                                stride_o=16 * dim,
                                nhead_stride_o=dim,
                                mask=2,
                                window_size_left=left,
                                window_size_right=0,
                            )[0]
                            kernel = next(
                                k for k, i in dispatch.kernel_ids if i == selected
                            )
                            expected = (
                                64 if 16 * ((sq + 127) // 128) <= num_cus else 128
                            )
                            self.assertEqual(kernel.F_tile.F_bm0, expected)
                            expected_n = (
                                256
                                if dim == 128
                                and 16 * ((sq + 127) // 128) <= num_cus // 2
                                and sq + 8192 >= 4 * sq
                                else expected
                            )
                            if dim == 64:
                                expected_n = 128
                            self.assertEqual(kernel.F_tile.F_bn0, expected_n)

    def test_extended_dtype_and_head_dimensions(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "*qr_tdm_cr*_nlogits_nbias*_nskip*_nsink*",
            0,
            [32, 64, 96, 128, 160, 192],
            "simplified",
        )
        inventory = Counter((k.F_dtype, k.F_hdim, k.F_mode) for k in kernels)
        self.assertEqual(
            inventory,
            Counter(
                {
                    (dtype, dim, mode): (
                        (10 if mode == "batch" else (7 if dtype == "bf16" else 4))
                        if dim == 192
                        else (
                            22
                            if (dtype, dim, mode) == ("bf16", 128, "batch")
                            else (
                                10
                                if (mode == "batch" and dim in (64, 128))
                                or (dtype, dim, mode) == ("bf16", 64, "group")
                                else 9
                                if (mode == "batch" and dim in (32, 96))
                                or (dtype, dim, mode) == ("bf16", 128, "group")
                                else 8
                            )
                        )
                    )
                    for dtype in ("fp16", "bf16")
                    for dim in (32, 64, 96, 128, 160, 192)
                    for mode in ("batch", "group")
                }
            ),
        )
        for kernel in kernels:
            compact = (kernel.F_dtype, kernel.F_hdim, kernel.F_mode) == (
                "bf16",
                128,
                "batch",
            )
            if not compact:
                self.assertEqual(kernel.F_pipeline.F_spad, "t")
                self.assertEqual(kernel.F_pipeline.F_skpad, "t")
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype in ("fp16", "bf16"):
            for dq, dv in (
                (32, 32),
                (64, 64),
                (96, 96),
                (128, 128),
                (160, 160),
                (192, 128),
            ):
                ids = {
                    i
                    for k, i in dispatch.kernel_ids
                    if k.F_dtype == dtype and k.F_hdim == dq and k.F_mode == "batch"
                }
                args = dict(
                    dtype=dtype,
                    hdim_q=dq,
                    hdim_v=dv,
                    bshd=1,
                    seqlen_q=257,
                    seqlen_k=515,
                    nhead_q=8,
                    nhead_k=2,
                    stride_o=8 * dv,
                    nhead_stride_o=dv,
                    batch_stride_o=257 * 8 * dv,
                    mask=2,
                    window_size_left=31,
                    window_size_right=7,
                )
                with self.subTest(dtype=dtype, dq=dq, dv=dv):
                    selected = dispatch.run(**args)[0]
                    self.assertIn(selected, ids)
                    kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                    self.assertEqual(kernel.F_tile.F_bm0, 64)
                    self.assertEqual(dispatch.run(**(args | dict(nhead_k=3))), [-1])
                    self.assertEqual(dispatch.run(**(args | dict(batch=1 << 28))), [-1])

    def test_default_priority_and_filtered_generic_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits_nbias*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
            0,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        fallback = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag != "qr_tdm_cr"
        )
        static_lengths = (512, 1024, 2048, 4096, 8192, 16384, 32768)
        for b, h, sq, sk in tuple((32, 8, n, n) for n in static_lengths) + (
            (32, 8, 128, 128),
            (64, 8, 128, 128),
            (32, 8, 256, 256),
            (32, 8, 768, 768),
            (32, 8, 65536, 65536),
            (2, 3, 193, 257),
            (2, 3, 2049, 257),
            (32, 8, 257, 257),
            (64, 8, 513, 515),
            (1, 8, 512, 512),
            (32, 4, 512, 512),
            (32, 8, 1024, 2048),
        ):
            for mask in (0, 1):
                for lse in (0, 1):
                    args = dict(
                        hdim_q=128,
                        hdim_v=128,
                        bshd=1,
                        batch=b,
                        nhead_q=h,
                        nhead_k=h,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        stride_o=h * 128,
                        batch_stride_o=sq * h * 128,
                        mask=mask,
                        window_size_right=0 if mask else -1,
                        lse=lse,
                    )
                    with self.subTest(b=b, h=h, sq=sq, sk=sk, mask=mask, lse=lse):
                        if (
                            not lse
                            and b == 32
                            and h == 8
                            and sq == sk
                            and sq in static_lengths
                        ):
                            expected = dispatch.expected_id(
                                pipeline="qr_tdm_cr",
                                dpad="f",
                                dvpad="f",
                                bm0=128,
                                mode="batch",
                                mask="s_mask" if mask else "s_no",
                                lse="t" if lse else "f",
                                spad="f",
                                skpad="f",
                            )
                        elif not lse and (
                            (b * h <= 8 and sq <= 4096)
                            or (
                                b == 32
                                and h == 8
                                and sq == sk
                                and sq in (128, 256)
                                and mask == 1
                            )
                        ):
                            expected = dispatch.expected_id(
                                pipeline="qr_tdm_cr",
                                dpad="f",
                                dvpad="f",
                                bm0=64,
                                mode="batch",
                                mask="s_mask" if mask else "s_no",
                                lse="t" if lse else "f",
                                spad="t" if sq % 64 else "f",
                                skpad="t" if sk % 64 else "f",
                            )
                        else:
                            selected = fallback.run(**args)[0]
                            kernel = next(
                                k for k, i in fallback.kernel_ids if i == selected
                            )
                            expected = next(
                                i
                                for k, i in dispatch.kernel_ids
                                if k.name == kernel.name
                            )
                        self.assertEqual(dispatch.run(**args), [expected])

    def test_large_grid_window_cr_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_[bf][fp]16_batch_*_nlogits_nbias*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
            0,
            [128],
            "simplified",
        )
        default = _CompiledDispatcher(self, pool, kernels)
        fallback = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag != "qr_tdm_cr"
        )
        cr_only = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag == "qr_tdm_cr"
        )
        for dtype in ("fp16", "bf16"):
            args = dict(
                dtype=dtype,
                hdim_q=128,
                hdim_v=128,
                bshd=1,
                batch=64,
                nhead_q=8,
                nhead_k=8,
                seqlen_q=1024,
                seqlen_k=1024,
                stride_o=1024,
                nhead_stride_o=128,
                batch_stride_o=1024 * 1024,
                mask=1,
                window_size_left=31,
                window_size_right=7,
            )
            with self.subTest(dtype=dtype):
                # Large grids retain the existing default heuristic. CR-only
                # builds still select the eligible M64 narrow-window variant.
                actual = default.run(**args)[0]
                expected = fallback.run(**args)[0]
                self.assertEqual(
                    next(k.name for k, i in default.kernel_ids if i == actual),
                    next(k.name for k, i in fallback.kernel_ids if i == expected),
                )
                selected = cr_only.run(**args)[0]
                kernel = next(k for k, i in cr_only.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                self.assertEqual(kernel.F_tile.F_bm0, 64)

    def test_small_tensor_views_take_default_priority(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_[cs]*", 0, [64, 128, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype in ("bf16", "fp16"):
            for dq, dv in ((64, 64), (128, 128), (192, 128)):
                for lse in (0, 1):
                    with self.subTest(dtype=dtype, dq=dq, lse=lse):
                        selected = dispatch.run(
                            dtype=dtype,
                            hdim_q=dq,
                            hdim_v=dv,
                            bshd=1,
                            seqlen_q=257,
                            seqlen_k=515,
                            nhead_q=8,
                            nhead_k=2,
                            mask=2,
                            window_size_left=31,
                            window_size_right=7,
                            stride_o=8 * dv,
                            nhead_stride_o=dv,
                            batch_stride_o=257 * 8 * dv,
                            lse=lse,
                        )[0]
                        kernel = next(
                            k for k, i in dispatch.kernel_ids if i == selected
                        )
                        if lse:
                            self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_sched")
                            continue
                        self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                        self.assertEqual(kernel.F_tile.F_bm0, 64)

    def test_long_masked_tensor_view_geometry(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_[cs]*", 0, [64, 128, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype in ("bf16", "fp16"):
            for dim in (64, 128, 192):
                value_dim = min(dim, 128)
                for sq, sk, group, left in (
                    (8192, 8192, 0, -1),
                    (32768, 32768, 0, 255),
                    (8192, 8192, 1, -1),
                    (8193, 4097, 0, -1),
                    (4097, 8193, 0, 255),
                ):
                    with self.subTest(dtype=dtype, dim=dim, sq=sq, group=group):
                        selected = dispatch.run(
                            dtype=dtype,
                            hdim_q=dim,
                            hdim_v=value_dim,
                            bshd=1,
                            batch=4 if group else 1,
                            nhead_q=8,
                            nhead_k=2,
                            seqlen_q=sq,
                            seqlen_k=sk,
                            group=group,
                            has_seqstart=group,
                            mask=2,
                            window_size_left=left,
                            window_size_right=0,
                            stride_o=8 * value_dim,
                            nhead_stride_o=value_dim,
                            batch_stride_o=sq * 8 * value_dim,
                        )[0]
                        kernel = next(
                            k for k, i in dispatch.kernel_ids if i == selected
                        )
                        self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                        self.assertEqual(
                            kernel.F_tile.F_bm0,
                            128
                            if (dim == 192 and not group and min(sq, sk) >= 8192)
                            or (dtype == "bf16" and dim in (64, 128, 192) and group)
                            else 64,
                        )

    def test_long_strided_native_geometry(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_[cs]*", 0, [64, 128, 192], "simplified"
        )
        default = _CompiledDispatcher(self, pool, kernels)
        fallback = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag != "qr_tdm_cr"
        )
        # K/V alignment and available workgroups determine the long geometry.
        # Large Q-only grids also use M128; windows retain the existing geometry.
        for dtype in ("bf16", "fp16"):
            for dim, length, heads, mask, view, tile in (
                (64, 8192, 16, 2, "kv", (128, 64)),
                (64, 8192, 64, 2, "kv", (128, 128)),
                (128, 8192, 16, 2, "kv", (128, 64)),
                (128, 8192, 4, 2, "kv", (128, 128)),
                (128, 16384, 4, 2, "kv", (128, 128)),
                (128, 32768, 4, 2, "kv", (128, 64)),
                (128, 8192, 4, 2, "o", (64, 64)),
                (128, 8193, 4, 2, "o", (64, 64)),
                (128, 12288, 4, 2, "o", (64, 64)),
                (128, 12289, 4, 2, "o", (64, 64)),
                (128, 12352, 4, 2, "o", (64, 64)),
                (128, 12415, 4, 2, "o", (64, 64)),
                (128, 12416, 4, 2, "o", (128, 128)),
                (128, 12800, 4, 2, "o", (128, 128)),
                (128, 13312, 4, 2, "o", (128, 128)),
                (128, 13313, 4, 2, "o", (64, 64)),
                (128, 14336, 4, 2, "o", (128, 128)),
                (128, 8192, 6, 2, "o", (128, 128)),
                (128, 8192, 4, 2, "all", (128, 128)),
                (128, 16384, 4, 2, "o", (128, 128)),
                (128, 16385, 4, 2, "o", (64, 64)),
                (128, 20481, 4, 2, "o", (64, 64)),
                (128, 24577, 4, 2, "o", (64, 64)),
                (128, 28673, 4, 2, "o", (128, 64)),
                (128, 32768, 4, 2, "o", (128, 64)),
                (128, 32769, 4, 2, "o", (128, 64)),
                (128, 8192, 4, 0, "o", (128, 128)),
                (128, 16384, 4, 0, "o", (128, 64)),
                (128, 8192, 16, 0, "kv", (128, 64)),
                (192, 32768, 4, 0, "o", (128, 64)),
                (192, 8192, 4, 2, "o", (128, 128)),
                (192, 8193, 4, 2, "o", (64, 64)),
                (192, 8256, 4, 2, "o", (64, 64)),
                (192, 8319, 4, 2, "o", (64, 64)),
                (192, 9217, 4, 2, "o", (64, 64)),
                (192, 8193, 4, 2, "all", (128, 64)),
                (192, 16385, 4, 2, "o", (128, 64)),
                (192, 32769, 4, 2, "o", (128, 64)),
                (192, 8192, 4, 0, "o", None),
                (64, 8192, 16, 2, "q", (64, 64)),
                (64, 8192, 32, 2, "q", (128, 64)),
                (64, 32768, 16, 2, "q", (128, 64)),
                (64, 16384, 4, 2, "o", (128, 64)),
                (64, 8192, 16, 2, "window", None),
                (192, 8192, 4, 2, "q", None),
            ):
                dv = min(dim, 128)
                output_padding = int(view in ("o", "all"))
                args = dict(
                    dtype=dtype,
                    lse=0,
                    hdim_q=dim,
                    hdim_v=dv,
                    batch=heads // 2,
                    nhead_q=2,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=length,
                    seqlen_k=length,
                    mask=mask,
                    window_size_left=31 if view == "window" else -1,
                    window_size_right=0 if mask else -1,
                    input_offset=0 if view == "o" else 2,
                    input_padding=0 if view == "o" else 1,
                    input_parts=7 if view == "all" else 1 if view == "q" else 6,
                    o_offset=2 * output_padding,
                    stride_o=2 * dv + output_padding,
                    nhead_stride_o=dv,
                    batch_stride_o=length * (2 * dv + output_padding),
                )
                with self.subTest(
                    dtype=dtype,
                    dim=dim,
                    length=length,
                    heads=heads,
                    mask=mask,
                    view=view,
                    lse=0,
                ):
                    selected = default.run(**args)[0]
                    kernel = next(k for k, i in default.kernel_ids if i == selected)
                    if tile is None:
                        original = fallback.run(**args)[0]
                        self.assertEqual(
                            kernel.name,
                            next(
                                k.name for k, i in fallback.kernel_ids if i == original
                            ),
                        )
                    else:
                        self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                        self.assertEqual(
                            (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), tile
                        )

            # A narrow query-heavy grid favors N128 for misaligned K/V.
            # Check both boundaries; larger batches retain their geometry.
            for sq, sk, heads, expected_n in (
                (16383, 8192, 4, 64),
                (16384, 8192, 4, 64),
                (16385, 8192, 4, 64),
                (30719, 8192, 4, 64),
                (30720, 8192, 4, 128),
                (30723, 8193, 4, 64),
                (30724, 8193, 4, 128),
                (32768, 8192, 4, 128),
                (32897, 8192, 4, 128),
                (38912, 8192, 4, 128),
                (38913, 8192, 4, 64),
                (40960, 8192, 4, 64),
                (38399, 10240, 4, 64),
                (38400, 10240, 4, 128),
                (32768, 9000, 4, 64),
                (32768, 8192, 8, 64),
                (16383, 8192, 32, 64),
                (16384, 8192, 32, 64),
                (16383, 8192, 64, 128),
                (16384, 8192, 64, 128),
            ):
                selected = default.run(
                    dtype=dtype,
                    lse=0,
                    hdim_q=64,
                    hdim_v=64,
                    batch=heads // 2,
                    nhead_q=2,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    mask=1,
                    window_size_right=0,
                    input_offset=2,
                    input_padding=1,
                    input_parts=6,
                    stride_o=128,
                    nhead_stride_o=64,
                    batch_stride_o=sq * 128,
                )[0]
                kernel = next(k for k, i in default.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                self.assertEqual(
                    (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0), (128, expected_n)
                )

            # Top-left uses every Q row; bottom-right can have fully masked
            # leading rows. Only active work should select the smaller K tile.
            for (sq, sk), mask, view, (batch, heads) in itertools.product(
                ((32768, 8192), (8192, 32768)),
                (1, 2),
                ("kv", "o"),
                (
                    (2, 2),
                    (1, 5),
                    (5, 1),
                    (3, 2),
                    (2, 3),
                    (1, 7),
                    (7, 1),
                    (2, 4),
                    (4, 2),
                ),
            ):
                output_padding = int(view == "o")
                selected = default.run(
                    dtype=dtype,
                    lse=0,
                    hdim_q=128,
                    hdim_v=128,
                    batch=batch,
                    nhead_q=heads,
                    nhead_k=1,
                    bshd=1,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    mask=mask,
                    window_size_right=0,
                    input_offset=0 if output_padding else 2,
                    input_padding=0 if output_padding else 1,
                    input_parts=6,
                    o_offset=2 * output_padding,
                    stride_o=heads * 128 + output_padding,
                    nhead_stride_o=128,
                    batch_stride_o=sq * (heads * 128 + output_padding),
                )[0]
                kernel = next(k for k, i in default.kernel_ids if i == selected)
                self.assertEqual(kernel.F_pipeline.tag, "qr_tdm_cr")
                self.assertEqual(
                    (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0),
                    (
                        (64, 64)
                        if view == "o"
                        and (
                            (mask == 2 and (sq > sk or batch * heads <= 6))
                            or (sq < sk and batch * heads == 4)
                        )
                        else (
                            128,
                            (
                                64
                                if (mask == 1 and sq > sk)
                                or (view == "o" and mask == 2 and sq < sk)
                                else 128
                            ),
                        )
                    ),
                )

    def test_sequence_length_pointer_contract(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*qr_tdm_cr*", 0, [64, 128, 192], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dtype in ("fp16", "bf16"):
            for dq, dv in ((64, 64), (128, 128), (192, 128)):
                for group in (0, 1):
                    args = dict(
                        dtype=dtype,
                        hdim_q=dq,
                        hdim_v=dv,
                        group=group,
                        has_seqstart=group,
                        bshd=1,
                        seqlen_q=257,
                        seqlen_k=515,
                        stride_o=8 * dv,
                        nhead_stride_o=dv,
                        batch_stride_o=257 * 8 * dv,
                    )
                    for pointers in (
                        {},
                        dict(has_cu_sq=1),
                        dict(has_cu_sk=1),
                        dict(has_cu_sq=1, has_cu_sk=1),
                        dict(has_sq_ptr=1),
                        dict(has_sk_ptr=1),
                        dict(has_sq_ptr=1, has_sk_ptr=1),
                    ):
                        with self.subTest(dtype=dtype, dq=dq, group=group, **pointers):
                            selected = dispatch.run(**(args | pointers))
                            rejected = not group and (
                                pointers.get("has_sq_ptr") or pointers.get("has_sk_ptr")
                            )
                            if rejected:
                                self.assertEqual(selected, [-1])
                            else:
                                self.assertNotEqual(selected, [-1])

    def test_small_compact_geometry(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "fmha_fwd_d128_bf16_*qr_tdm_cr*", 0, [128], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for b, h, sq, sk, dense_tile, causal_tile in (
            (1, 8, 128, 128, 64, 64),
            (2, 3, 193, 257, 64, 64),
            (2, 3, 2049, 257, 64, 64),
            (4, 8, 1024, 1024, 128, 128),
            (32, 8, 128, 128, 128, 64),
            (64, 8, 128, 128, 128, 128),
            (32, 8, 256, 256, 128, 64),
            (64, 8, 256, 256, 128, 128),
            (32, 8, 1024, 1024, 128, 128),
            (1, 8, 8192, 8192, 128, 128),
            (32, 8, 32768, 32768, 128, 128),
        ):
            for mask in (0, 1):
                with self.subTest(b=b, h=h, sq=sq, sk=sk, mask=mask, lse=0):
                    selected = dispatch.run(
                        dtype="bf16",
                        hdim_q=128,
                        hdim_v=128,
                        bshd=1,
                        batch=b,
                        nhead_q=h,
                        nhead_k=h,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        stride_o=h * 128,
                        batch_stride_o=sq * h * 128,
                        mask=mask,
                        window_size_right=0 if mask else -1,
                        lse=0,
                    )[0]
                    kernel = next(k for k, i in dispatch.kernel_ids if i == selected)
                    self.assertEqual(
                        kernel.F_tile.F_bm0, causal_tile if mask else dense_tile
                    )

    def test_inventory_and_filtered_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits_nbias*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
            0,
            [128],
            "simplified",
        )
        candidates = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_cr"]
        self.assertEqual(len(candidates), 31)
        for k in candidates:
            self.assertIn(k.F_mode, ("batch", "group"))
            self.assertIn(
                (k.F_tile.F_bm0, k.F_tile.F_bn0),
                ((64, 64), (64, 256), (128, 64), (128, 128)),
            )
            self.assertIn("FmhaFwdLaunch<k_,", k.render())
            self.assertIn("ck_tile::FmhaFwdKernel<", k.render())
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda t: t.pipeline_tag == "qr_tdm_cr"
        )
        family = {i for k, i in dispatch.kernel_ids if k.F_pipeline.tag == "qr_tdm_cr"}
        for sq, sk in ((512, 512), (513, 512), (512, 513), (513, 515), (2048, 257)):
            for mask, lse in ((0, 0), (1, 0), (0, 1), (1, 1)):
                args = dict(
                    hdim_q=128,
                    hdim_v=128,
                    bshd=1,
                    batch=32,
                    nhead_q=8,
                    nhead_k=8,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    stride_o=1024,
                    batch_stride_o=sq * 1024,
                    mask=mask,
                    window_size_right=0 if mask else -1,
                    lse=lse,
                )
                if lse:
                    self.assertEqual(dispatch.run(**args), [-1])
                    continue
                expected = dispatch.expected_id(
                    pipeline="qr_tdm_cr",
                    dpad="f",
                    dvpad="f",
                    bm0=128,
                    bn0=128,
                    mode="batch",
                    mask="s_mask" if mask else "s_no",
                    lse="t" if lse else "f",
                    spad="t" if sq % 128 else "f",
                    skpad="t" if sk % 128 else "f",
                )
                self.assertEqual(dispatch.run(**args), [expected])
                for changes in (
                    dict(group=1),
                    dict(row=0),
                    dict(logits=1),
                    dict(dropout=1),
                    dict(bias=1),
                    dict(o_offset=1),
                    dict(input_offset=1),
                    dict(batch_stride_o=-1),
                    dict(has_sk_ptr=1),
                    dict(has_sq_ptr=1),
                    dict(qscale=1),
                    dict(bshd=0),
                ):
                    with self.subTest(
                        sq=sq, sk=sk, mask=mask, lse=lse, changes=changes
                    ):
                        self.assertNotIn(dispatch.run(**(args | changes))[0], family)
                self.assertNotIn(dispatch.run(device="gfx1251", **args)[0], family)
                for changes in (
                    dict(has_sink_ptr=1),
                    dict(nhead_k=2),
                    dict(nhead_k=1),
                    dict(has_cu_sk=1),
                    dict(stride_o=128),
                    dict(seqlen_k=0),
                    dict(o_offset=2),
                    dict(stride_o=1025, batch_stride_o=sq * 1025),
                    dict(batch_stride_o=sq * 1024 + 1),
                    dict(scale_s=0),
                    dict(scale_s=-0.19),
                ):
                    with self.subTest(changes=changes):
                        self.assertIn(dispatch.run(**(args | changes))[0], family)
                if mask:
                    self.assertIn(
                        dispatch.run(**(args | dict(window_size_left=128)))[0], family
                    )
                    self.assertEqual(dispatch.run(**(args | dict(mask=2))), [expected])


class TestGfx125D192Codegen(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_esm2_scope_and_default_wwm(self):
        source_flags = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        d256_source = (
            "fmha_fwd_d256_bf16_batch_b64x64x32x256x32x256_"
            "r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_"
            "vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
            "ntrload_nsink_gfx125.cpp"
        )
        fp8_sched_source = (
            "fmha_fwd_d128_fp8bf16_batch_b128x128x128x128x128x128_"
            "r4x1x1_r4x1x1_w16x16x128_w16x16x128_o2_qr_tdm_fp8_sched_"
            "vr_pssk_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_"
            "ntrload_npackgqa_nsink_gfx125.cpp"
        )
        fp8_tdm_source = fp8_sched_source.replace(
            "_qr_tdm_fp8_sched_", "_qr_tdm_"
        ).replace("_npackgqa_", "_")
        mx_sched_source = fp8_sched_source.replace("_fp8bf16_", "_mxfp8_").replace(
            "_pertensor_", "_mx_"
        )
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_N128_FILENAME,
            _D192_BATCH_N128_FILENAME.replace("_bf16_", "_fp16_"),
            _D192_BATCH_NMASK_FILENAME.replace("_nlse_", "_lse_"),
            _D192_BATCH_MASK_FILENAME,
            _D192_GROUP_NMASK_FILENAME,
            d256_source,
            fp8_sched_source,
            mx_sched_source,
            fp8_tdm_source,
        )

        def configure(hip_version=None):
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
{f"set(CMAKE_HIP_COMPILER_VERSION {hip_version})" if hip_version else ""}
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

        for hip_version in (None, "24.0.0", "23.0.0"):
            commands = configure(hip_version)
            # Clang before 24 builds MX fp8_sched instances without the expert mode.
            no_expert = (d256_source, fp8_tdm_source) + (
                (mx_sched_source,) if hip_version == "23.0.0" else ()
            )
            for source in sources:
                with self.subTest(hip_version=hip_version, source=source):
                    self.assertNotIn("-wwm-regalloc=fast", commands[source])
                    self.assertEqual(
                        "-amdgpu-expert-scheduling-mode" in commands[source],
                        source not in no_expert,
                    )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_fp8_sched_rule_scope(self):
        # Filenames come from codegen so a naming change cannot silently disable a rule.
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*fp8*", 0, [64, 128, 192, 256], "simplified"
        )

        def sched(k):
            return k.F_pipeline.tag == "qr_tdm_fp8_sched"

        rules = {
            "-amdgpu-codegenprepare-force-break-large-phis": lambda k: (
                sched(k)
                and k.F_pipeline.F_trload == "f"
                and (
                    k.F_pipeline.F_qscale == "blockscale"
                    or (
                        k.F_hdim == 64
                        and k.F_pipeline.F_qscale in ("no", "pertensor", "perhead")
                    )
                )
            ),
            "-structurizecfg-skip-uniform-regions": lambda k: (
                sched(k) and k.F_hdim in (128, 256) and k.F_dtype.startswith("mxfp8")
            ),
            "-regalloc-enable-priority-advisor=dummy": lambda k: (
                sched(k) and k.F_hdim == 256 and k.F_dtype.startswith("fp8")
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "flags.cmake"
            script.write_text(
                f'include("{helper}")\n'
                + "\n".join(
                    f'ck_tile_fmha_fwd_get_source_compile_options("{k.filename}" flags)\n'
                    f'message("{i}:${{flags}}")'
                    for i, k in enumerate(kernels)
                )
            )
            process = subprocess.run(
                ["cmake", "-P", str(script)],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
        lines = process.stderr.strip().splitlines()
        self.assertEqual(len(lines), len(kernels))
        for flag, expected in rules.items():
            seen = Counter()
            for i, (k, line) in enumerate(zip(kernels, lines)):
                self.assertTrue(line.startswith(f"{i}:"))
                want = expected(k)
                seen[(_fp8_sched_kind(k), want)] += 1
                with self.subTest(flag=flag, source=k.filename):
                    self.assertEqual(flag in line.split(";"), want)
            # Each rule has positive and negative codegen names, packed GQA on both tiles included.
            for tag in ("qr_tdm_fp8_sched", "packgqa", "packgqa_32row"):
                for want in (True, False):
                    with self.subTest(flag=flag, tag=tag, positive=want):
                        self.assertGreater(seen[(tag, want)], 0)
            self.assertEqual(seen[("qr_tdm", True)], 0)
            self.assertGreater(seen[("qr_tdm", False)], 0)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_source_flags_follow_gfx125_family_targets(self):
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        source = (
            "fmha_fwd_d256_fp8bf16_batch_b128x128x128x256x128x256_"
            "r4x1x1_r4x1x1_w16x16x128_w16x16x128_o1_qr_tdm_fp8_sched_"
            "vr_pssk_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_"
            "ntrload_npackgqa_nsink_gfx125.cpp"
        )
        for targets, applied in (
            ("gfx1250-strict", True),
            ("gfx1250", True),
            ("gfx1251", True),
            ("gfx950", False),
            ("gfx1201", False),
        ):
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "CMakeLists.txt").write_text(
                    f"""cmake_minimum_required(VERSION 3.20)
project(fmha_source_flags LANGUAGES NONE)
include(\"{helper}\")
set(source \"${{CMAKE_CURRENT_LIST_DIR}}/{source}\")
ck_tile_fmha_fwd_apply_source_compile_options(\"{targets}\" \"${{source}}\")
get_source_file_property(options \"${{source}}\" COMPILE_OPTIONS)
message(\"options:${{options}}\")
"""
                )
                process = subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                options = re.search(r"^options:(.*)$", process.stderr, re.M)
                self.assertIsNotNone(options)
                for flag in (
                    "-amdgpu-expert-scheduling-mode",
                    "-regalloc-enable-priority-advisor=dummy",
                ):
                    self.assertEqual(flag in options[1].split(";"), applied)

    def test_receipts_emit_d192_sched_and_fallback_tiles(self):
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
                    self.assertIn(
                        (kernel.F_tile.F_bn0, kernel.F_tile.F_occupancy),
                        ((64, 2), (128, 1)),
                    )
                    if kernel.F_tile.F_bn0 == 128:
                        self.assertEqual(
                            kernel.F_pipeline.F_spad,
                            "t" if kernel.F_mode == "group" else "f",
                        )
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

                self.assertTrue(
                    any(
                        kernel.F_hdim == 192
                        and kernel.F_tile.F_bn1 == 128
                        and kernel.F_pipeline.tag == "qr_tdm"
                        for kernel in kernels
                    )
                )
                self.assertEqual(
                    sum(kernel.F_tile.F_bn0 == 128 for kernel in candidates),
                    2 if receipt == 600 else 1,
                )

    def test_d128_retains_short_query_fallback_tiles(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                self.assertTrue(any(_is_family_kernel(k) for k in kernels))
                # The two N128 specializations precede the existing short-query
                # N64 fallbacks; adding CR must preserve those fallback tiles.
                fallback = [
                    k
                    for k in kernels
                    if k.F_pipeline.tag not in ("qr_tdm_sched", "qr_tdm_cr")
                ]
                self.assertTrue(fallback)
                self.assertTrue(all(k.F_hdim == 128 for k in fallback))
                self.assertTrue(any(k.F_tile.F_bm0 < 128 for k in fallback))
                self.assertTrue(all(k.F_tile.F_bn0 == 64 for k in fallback))
                self.assertTrue(any(k.F_pipeline.tag == "qr_tdm" for k in fallback))
                self.assertEqual(len(kernels), len({k.filename for k in kernels}))
                for kernel in fallback:
                    self.assertNotIn("QRKSVS_TDM_SCHED", kernel.render())

    def test_generated_api_places_exact_candidate_before_legacy_entries(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                api_pool, kernels = _generate(receipt)
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertIn("QRKSVS_TDM_SCHED", api)
                self.assertIn(_D192_SELECTOR, api)
                if receipt != 200:
                    self.assertIn(
                        "a.seqlen_k >= 512LL * (t.is_group_mode ? a.batch : 1)",
                        api,
                    )
                    self.assertIn("a.seqlen_k_ptr == nullptr", api)
                    self.assertIn("get_num_blocks(128) <= num_cus", api)
                self.assertNotIn(f"!({_D192_SELECTOR})", api)
                self.assertLess(api.index("t.hdim_q <= 192"), api.index(_D192_SELECTOR))
                self.assertLess(api.index(_D192_SELECTOR), api.index("t.hdim_q <= 256"))
                self.assertLess(
                    api.index("QRKSVS_TDM_SCHED", api.index("t.hdim_q <= 192")),
                    api.index(
                        "BlockFmhaPipelineEnum::QRKSVS_TDM,",
                        api.index("t.hdim_q <= 192"),
                    ),
                )
                self.assertNotIn("is_gfx125_tdm_v128_enabled", api)

                candidate = next(
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                )
                self.assertEqual((candidate.F_hdim, candidate.F_tile.F_bn1), (192, 128))
                source = candidate.render()
                self.assertIn("BlockFmhaPipelineQRKSVSTdmSched", source)
                self.assertIn(
                    "FmhaTdmSchedPolicyFor<fmha_pipeline_problem>, false", source
                )
                self.assertIn("QRKSVS_TDM_SCHED", source)
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
                    receipt=200 if feature == "skip" else 100,
                    optdim_list=[192, 256],
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
                self.assertNotIn("QRKSVS_TDM_SCHED", api)
                self.assertNotIn(_D192_SELECTOR, api)

        _, qscale_kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter="fmha_fwd_d*_bf16_*_pertensor*",
            receipt=100,
            optdim_list=[192, 256],
            mask_impl="simplified",
        )
        self.assertFalse(qscale_kernels)

    def test_filters_select_only_matching_sources(self):
        for receipt in (100, 200, 600):
            for kernel_filter in ("*d192_bf16*", f"*{_D192_PIPELINE}*"):
                with self.subTest(receipt=receipt, kernel_filter=kernel_filter):
                    api_pool, kernels = get_fwd_blobs(
                        targets=["gfx1250"],
                        kernel_filter=kernel_filter,
                        receipt=receipt,
                        optdim_list=[192, 256],
                        mask_impl="simplified",
                    )
                    candidate_count = sum(
                        kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels
                    )
                    self.assertGreater(candidate_count, 0)
                    self.assertTrue(
                        all(fnmatch(kernel.name, kernel_filter) for kernel in kernels)
                    )
                    self.assertTrue(all(kernel.F_hdim == 192 for kernel in kernels))
                    if kernel_filter == f"*{_D192_PIPELINE}*":
                        self.assertEqual(candidate_count, len(kernels))

                    with tempfile.TemporaryDirectory() as output_dir:
                        output_path = Path(output_dir)
                        write_fwd_api(api_pool, output_path)
                        api = (output_path / "fmha_fwd_api.cpp").read_text()
                    self.assertIn(_D192_SELECTOR, api)
                    self.assertNotIn(f"!({_D192_SELECTOR})", api)

    def test_only_gfx125_targets_emit_the_scheduled_pipeline(self):
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_D192_PIPELINE}*",
                    receipt=100,
                    optdim_list=[192],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)

        for target in ("gfx1250", "gfx1251", "gfx1252"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_D192_PIPELINE}*",
                    receipt=100,
                    optdim_list=[192],
                    mask_impl="simplified",
                )
                self.assertTrue(
                    any(kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels)
                )

    def test_n128_emits_batch_and_group_without_generic_windows(self):
        mask_options = {
            "simplified": ("s_no", "s_mask"),
            "generic": ("no", "causal"),
        }
        for dtype in ("fp16", "bf16"):
            for mask_impl, masks in mask_options.items():
                with self.subTest(dtype=dtype, mask_impl=mask_impl):
                    self.check_n128_masks(dtype, mask_impl, masks)

    def check_n128_masks(self, dtype, mask_impl, masks):
        _, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=f"*{dtype}*{_D192_PIPELINE}*",
            receipt=600,
            optdim_list=[192],
            mask_impl=mask_impl,
        )
        n128 = [
            kernel
            for kernel in kernels
            if kernel.F_pipeline.tag == _D192_PIPELINE and kernel.F_tile.F_bn0 == 128
        ]
        actual = Counter(
            (kernel.F_mode, kernel.F_pipeline.F_mask, kernel.F_pipeline.F_lse)
            for kernel in n128
        )
        expected = Counter(
            {
                (mode, mask, lse): 1
                for mode in ("batch", "group")
                for mask in masks
                for lse in ("f", "t")
            }
        )
        self.assertEqual(actual, expected)
        for kernel in n128:
            required_pad = "t" if kernel.F_mode == "group" else "f"
            self.assertEqual(kernel.F_pipeline.F_spad, required_pad)
            self.assertEqual(kernel.F_pipeline.F_skpad, required_pad)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125D192Dispatch(unittest.TestCase):
    """Check generated CPU dispatch; kernel validity is tested separately."""

    def compile(
        self,
        receipt=600,
        optdims=None,
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        filter_fn=None,
        mutate=None,
        mask_impl="simplified",
    ):
        pool, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=kernel_filter,
            receipt=receipt,
            optdim_list=[128, 192, 256] if optdims is None else optdims,
            mask_impl=mask_impl,
        )
        return _CompiledDispatcher(self, pool, kernels, filter_fn, mutate), kernels

    @staticmethod
    def candidate_id(dispatch, mode, mask="s_no", lse="f", bn0=64, dtype="bf16"):
        return dispatch.expected_id(
            mode=mode,
            dtype=dtype,
            hdim=192,
            pipeline=_D192_PIPELINE,
            mask=mask,
            lse=lse,
            bn0=bn0,
            sink="f",
        )

    @staticmethod
    def fallback_id(dispatch, mode, mask="s_no", lse="f", sink="f", bm0=128):
        return dispatch.expected_id(
            mode=mode,
            hdim=192,
            pipeline="qr_tdm",
            bm0=bm0,
            mask=mask,
            lse=lse,
            spad="t" if mode == "group" else "f",
            dpad="t",
            sink=sink,
        )

    @staticmethod
    def head_tail_id(dispatch, mode, mask="s_no", lse="f", padded="t", sink="f"):
        return dispatch.expected_id(
            mode=mode,
            hdim=256,
            pipeline="qr_tdm",
            mask=mask,
            lse=lse,
            spad="t" if mode == "group" else "f",
            dpad=padded,
            sink=sink,
        )

    def test_real_dispatch_priority_exact_shape_and_short_q(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            fallback = self.fallback_id(dispatch, mode)
            self.assertEqual(dispatch.run(group=group), [candidate])
            for length in (128, 129, 257):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length), [candidate]
                )
            for length in (0, 1, 64, 127):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length),
                    [self.fallback_id(dispatch, mode, bm0=64)],
                )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=160, hdim_v=128), [fallback]
            )
            for q, v in ((191, 128), (192, 127)):
                with self.subTest(mode=mode, shape=(q, v)):
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=q, hdim_v=v),
                        [
                            dispatch.expected_id(
                                mode=mode,
                                hdim=192,
                                pipeline="qr_tdm",
                                bm0=128,
                                mask="s_no",
                                lse="f",
                                spad="t" if group else "f",
                                dpad="t",
                                sink="f",
                            )
                        ],
                    )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=193, hdim_v=128),
                [self.head_tail_id(dispatch, mode)],
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=192, hdim_v=192),
                [
                    dispatch.expected_id(
                        mode=mode,
                        hdim=256,
                        pipeline="qr_tdm",
                        bm0=64,
                        mask="s_no",
                        lse="f",
                        spad="t" if group else "f",
                        dpad="t",
                        sink="f",
                    )
                ],
            )
            generic256 = dispatch.expected_id(
                mode=mode,
                hdim=256,
                pipeline="qr_tdm",
                bm0=64,
                mask="s_no",
                lse="f",
                spad="t" if group else "f",
                dpad="f",
                sink="f",
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=256, hdim_v=256), [generic256]
            )

    def test_n128_long_k_one_wave_selector_and_n64_fallback(self):
        dispatch, _ = self.compile(
            kernel_filter=(
                "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
            )
        )
        n64 = self.candidate_id(dispatch, "batch")
        n128 = self.candidate_id(dispatch, "batch", bn0=128)

        for sk in (128, 192, 256, 511):
            self.assertEqual(dispatch.run(seqlen_k=sk), [n64])
        for sk in (512, 513, 2048):
            self.assertEqual(dispatch.run(seqlen_k=sk), [n128])
        self.assertEqual(dispatch.run(seqlen_k=512, has_cu_sk=1), [n64])

        grid256 = dict(batch=8, nhead_q=8, seqlen_q=512, seqlen_k=512)
        self.assertEqual(dispatch.run(**grid256), [n128])
        self.assertEqual(dispatch.run(num_cus=128, **grid256), [n64])
        self.assertEqual(dispatch.run(**{**grid256, "batch": 16}), [n64])
        self.assertEqual(dispatch.run(max_seqlen_q=1024, **grid256), [n64])
        for device in ("gfx1251", "gfx1252"):
            self.assertEqual(dispatch.run(device=device, **grid256), [n128])
            self.assertEqual(dispatch.run(device=device, seqlen_k=511), [n64])

        for mask in (1, 2):
            causal = dict(
                mask=mask,
                seqlen_q=512,
                seqlen_k=512,
                window_size_left=-1,
                window_size_right=0,
            )
            self.assertEqual(
                dispatch.run(**causal),
                [self.candidate_id(dispatch, "batch", mask="s_mask", bn0=128)],
            )
            for disqualifier in (
                {"window_size_left": 128},
                {"window_size_right": 64},
                {"seqlen_q": 256},
            ):
                self.assertEqual(
                    dispatch.run(**{**causal, **disqualifier}),
                    [self.candidate_id(dispatch, "batch", mask="s_mask")],
                )
        self.assertEqual(
            dispatch.run(mask=3, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", mask="s_mask")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=512),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024),
            [self.candidate_id(dispatch, "group", bn0=128)],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024, has_cu_sk=1),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024, has_sk_ptr=1),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, batch=4, seqlen_k=2047),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, batch=4, seqlen_k=2048),
            [self.candidate_id(dispatch, "group", bn0=128)],
        )
        self.assertEqual(
            dispatch.run(lse=1, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", lse="t", bn0=128)],
        )

    def test_fp16_scheduled_dispatch_for_d128_and_d192(self):
        dispatch, kernels = self.compile(kernel_filter="*qr_tdm_sched*")
        self.assertEqual(
            {(k.F_dtype, k.F_hdim, k.F_tile.F_bn0) for k in kernels},
            {
                (dtype, hdim, bn0)
                for dtype in ("fp16", "bf16")
                for hdim, bn0 in ((128, 128), (192, 64), (192, 128))
            },
        )

        for mode, group in (("batch", 0), ("group", 1)):
            for hdim, short_bn0 in ((128, 128), (192, 64)):
                with self.subTest(mode=mode, hdim=hdim, length="short"):
                    expected = dispatch.expected_id(
                        dtype="fp16",
                        mode=mode,
                        hdim=hdim,
                        bn0=short_bn0,
                        pipeline="qr_tdm_sched",
                        mask="s_no",
                        lse="f",
                        sink="f",
                    )
                    self.assertEqual(
                        dispatch.run(
                            dtype="fp16", group=group, hdim_q=hdim, seqlen_k=128
                        ),
                        [expected],
                    )

            long_k = 1024 if group else 512
            n128 = self.candidate_id(dispatch, mode, bn0=128, dtype="fp16")
            self.assertEqual(
                dispatch.run(
                    dtype="fp16",
                    group=group,
                    hdim_q=192,
                    seqlen_q=long_k,
                    seqlen_k=long_k,
                ),
                [n128],
            )

    def test_n64_only_filtered_inventory_still_dispatches_long_k(self):
        dispatch, _ = self.compile(
            filter_fn=lambda trait: (
                trait.pipeline_tag != _D192_PIPELINE or trait.bn0 == 64
            )
        )
        self.assertEqual(
            dispatch.run(seqlen_k=512), [self.candidate_id(dispatch, "batch")]
        )

    def test_n128_generic_no_mask_and_causal_dispatch(self):
        dispatch, _ = self.compile(kernel_filter="*qr_tdm_sched*", mask_impl="generic")
        for mode, group in (("batch", 0), ("group", 1)):
            with self.subTest(mode=mode):
                length = 1024 if group else 512
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length, seqlen_k=length),
                    [self.candidate_id(dispatch, mode, mask="no", bn0=128)],
                )
                causal = dict(
                    group=group,
                    mask=1,
                    seqlen_q=length,
                    seqlen_k=length,
                    window_size_left=-1,
                    window_size_right=0,
                )
                n128 = self.candidate_id(dispatch, mode, mask="causal", bn0=128)
                self.assertEqual(dispatch.run(**causal), [n128])
                self.assertEqual(dispatch.run(**{**causal, "mask": 2}), [n128])
                generic = self.candidate_id(dispatch, mode, mask="generic")
                self.assertEqual(
                    dispatch.run(**{**causal, "mask": 3, "window_size_left": 64}),
                    [generic],
                )

    def test_family_only_and_empty_filtered_dispatch(self):
        family, _ = self.compile(
            filter_fn=lambda trait: (
                trait.pipeline_tag == _D192_PIPELINE and str(trait.hdim) == "192"
            )
        )
        candidate = self.candidate_id(family, "batch")
        self.assertEqual(family.run(), [candidate])
        for arguments in ({"seqlen_q": 127}, {"hdim_q": 128}):
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

    def test_real_dispatch_d128_priority_and_sequence_boundary(self):
        dispatch, kernels = self.compile()
        self.assertTrue(
            [
                kernel
                for kernel in kernels
                if kernel.F_hdim == 128 and _is_family_kernel(kernel)
            ]
        )
        for mode, group in (("batch", 0), ("group", 1)):
            for length, bm0 in ((1, 64), (127, 64), (128, 64), (2047, 64), (2048, 128)):
                expected = dispatch.expected_id(
                    mode=mode,
                    hdim=128,
                    pipeline="qr_tdm",
                    bm0=bm0,
                    spad="t" if group else "f",
                    mask="s_no",
                    lse="f",
                    sink="f",
                    dpad="f",
                )
                selected = expected
                if length >= 128:
                    selected = dispatch.expected_id(
                        mode=mode,
                        hdim=128,
                        pipeline="qr_tdm_sched",
                        mask="s_no",
                        lse="f",
                        sink="f",
                    )
                self.assertEqual(
                    dispatch.run(
                        group=group,
                        hdim_q=128,
                        hdim_v=128,
                        seqlen_q=length,
                    ),
                    [selected],
                )

    def test_real_dispatch_d128_o_layout_fallback(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            sched = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_sched",
                mask="s_no",
                lse="f",
                sink="f",
            )
            fallback = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm",
                bm0=128,
                spad="t" if group else "f",
                mask="s_no",
                lse="f",
                sink="f",
                dpad="f",
            )
            rejected = [
                dict(o_offset=2),
                dict(stride_o=1028),
                dict(nhead_stride_o=132),
                dict(stride_o=-1024),
                dict(stride_o=16909320),
            ]
            accepted = [dict(stride_o=16909312)]
            (rejected if group == 0 else accepted).append(dict(batch_stride_o=262148))
            for changes in rejected + accepted:
                selected = fallback if changes in rejected else sched
                self.assertEqual(
                    dispatch.run(
                        group=group, hdim_q=128, hdim_v=128, seqlen_q=2048, **changes
                    ),
                    [selected],
                    changes,
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
                    self.assertEqual(
                        dispatch.run(group=group, mask=mask, lse=lse), [candidate]
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
                if changes == dict(skip=1):
                    self.assertEqual(dispatch.run(**changes), dispatch.run())
                else:
                    self.assertEqual(dispatch.run(**changes), [-1])

        for device in ("gfx1251", "gfx1252"):
            for mode, group in (("batch", 0), ("group", 1)):
                with self.subTest(device=device, mode=mode):
                    self.assertEqual(
                        dispatch.run(device=device, group=group),
                        [self.candidate_id(dispatch, mode)],
                    )

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
                    fallback = (
                        self.fallback_id(dispatch, mode)
                        if render_filter
                        else dispatch.expected_id(
                            mode=mode,
                            hdim=256,
                            pipeline="qr_tdm",
                            bm0=64,
                            mask="s_no",
                            lse="f",
                            spad="t" if group else "f",
                            dpad="t",
                            sink="f",
                        )
                    )
                    self.assertEqual(dispatch.run(group=group), [fallback])

    def test_real_dispatch_candidate_only_filter_is_strict(self):
        for receipt in (0, 100, 200, 600):
            with self.subTest(receipt=receipt):
                dispatch, kernels = self.compile(
                    receipt=receipt, kernel_filter=f"*{_D192_PIPELINE}*"
                )
                self.assertTrue(kernels)
                self.assertTrue(all(_is_family_kernel(kernel) for kernel in kernels))
                self.assertEqual(len({kernel.name for kernel in kernels}), len(kernels))
                for mode in sorted({kernel.F_mode for kernel in kernels}):
                    group = int(mode == "group")
                    self.assertEqual(
                        dispatch.run(group=group), [self.candidate_id(dispatch, mode)]
                    )
                    self.assertEqual(dispatch.run(group=group, seqlen_q=127), [-1])
                    self.assertEqual(dispatch.run(group=group, hdim_q=160), [-1])
                    self.assertEqual(dispatch.run(group=group, logits=1), [-1])

    def test_real_dispatch_receipt_and_optdim_baseline(self):
        cmake = (_FMHA_EXAMPLE_DIR / "CMakeLists.txt").read_text()
        gfx125_dims = re.search(
            r"if\(FMHA_FWD_HAS_GFX125\)\s+set\(FMHA_FWD_OPTDIM\s+([0-9,]+)\)",
            cmake,
        )
        self.assertIsNotNone(gfx125_dims)
        common_dims = list(map(int, gfx125_dims[1].split(",")))
        self.assertEqual(common_dims, [32, 64, 96, 128, 160, 192, 256, 512])
        for receipt in (0, 100, 200, 600):
            for optdims in ([128], [192], [256], [-1], common_dims):
                with self.subTest(receipt=receipt, optdims=optdims):
                    dispatch, kernels = self.compile(receipt=receipt, optdims=optdims)
                    modes = sorted({kernel.F_mode for kernel in kernels})
                    family = [kernel for kernel in kernels if _is_family_kernel(kernel)]
                    expected_family_dims = (
                        {64, 128, 192}
                        if optdims == [-1]
                        else {64, 128, 192}.intersection(optdims)
                    )
                    self.assertEqual(
                        {kernel.F_hdim for kernel in family}, expected_family_dims
                    )
                    for mode in modes:
                        group = int(mode == "group")
                        if optdims == [128]:
                            self.assertEqual(dispatch.run(group=group), [-1])
                        elif optdims == [256]:
                            fallback = dispatch.expected_id(
                                mode=mode,
                                hdim=256,
                                pipeline="qr_tdm",
                                bm0=64,
                                mask="s_no",
                                lse="f",
                                spad="t" if group else "f",
                                dpad="t",
                                sink="f",
                            )
                            self.assertEqual(dispatch.run(group=group), [fallback])
                        else:
                            self.assertEqual(
                                dispatch.run(group=group),
                                [self.candidate_id(dispatch, mode)],
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


_SUPPORTED_DISPATCH_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*"
)


class TestGfx125V128Codegen(unittest.TestCase):
    @staticmethod
    def dimension_cases():
        cmake = (_FMHA_EXAMPLE_DIR / "CMakeLists.txt").read_text()
        default_dims = re.search(
            r"if\(FMHA_FWD_HAS_GFX125\)\s+set\(FMHA_FWD_OPTDIM\s+([0-9,]+)\)",
            cmake,
        )
        assert default_dims is not None
        return [
            [128],
            [192],
            [256],
            [128, 192],
            [-1],
            [int(value) for value in default_dims.group(1).split(",")],
        ]

    def test_sched_respects_requested_dimensions(self):
        for dimensions, expected in (
            ([-1], {64, 128, 192}),
            ([128], {128}),
            ([192], {192}),
            ([256], set()),
            ([128, 192], {128, 192}),
            ([32, 64, 80, 128, 256], {64, 128}),
            ([], set()),
            ([64], {64}),
        ):
            with self.subTest(dimensions=dimensions):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], "*_qr_tdm_sched_*", 100, dimensions, "simplified"
                )
                actual = {kernel.F_hdim for kernel in kernels}
                self.assertEqual(actual, expected)

    def test_dimension_filter_preserves_full_inventory(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for receipt in (0, 100, 200, 600):
                _, full = get_fwd_blobs(targets, None, receipt, [-1], "simplified")
                self.assertEqual(len(full), len({k.filename for k in full}))
                for dimensions in self.dimension_cases():
                    with self.subTest(
                        targets=targets, receipt=receipt, dims=dimensions
                    ):
                        _, kernels = get_fwd_blobs(
                            targets, None, receipt, dimensions, "simplified"
                        )
                        expected = [
                            k.filename
                            for k in full
                            if dimensions == [-1] or k.F_hdim in dimensions
                        ]
                        self.assertEqual([k.filename for k in kernels], expected)

    def test_sched_filter_emits_only_matching_kernels(self):
        for dtype in ("fp16", "bf16"):
            for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
                for dimensions in self.dimension_cases():
                    with self.subTest(dtype=dtype, receipt=receipt, dims=dimensions):
                        self.check_sched_filter(dtype, receipt, count, dimensions)

    def check_sched_filter(self, dtype, receipt, count, dimensions):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], f"*{dtype}*_qr_tdm_sched_*", receipt, dimensions, "simplified"
        )
        selected_dims = (
            {64, 128, 192}
            if dimensions == [-1]
            else {64, 128, 192}.intersection(dimensions)
        )
        self.assertEqual({k.F_hdim for k in kernels}, selected_dims)
        self.assertEqual(
            len(kernels),
            sum(
                (
                    {0: 12, 100: 8, 200: 4, 600: 12}[receipt]
                    if dim == 64
                    else count * (2 if dim == 192 else 1)
                )
                for dim in selected_dims
            ),
        )
        self.assertTrue(
            all(
                kernel.F_pipeline.tag == "qr_tdm_sched"
                and fnmatch(kernel.name, f"*{dtype}*_qr_tdm_sched_*")
                for kernel in kernels
            )
        )

    def test_dimension_generic_and_unsupported_filters_are_deterministic(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for dimensions in self.dimension_cases():
                for pattern in (
                    "*_bf16_*_qr_vr_*",
                    "*not_a_kernel*",
                ):
                    with self.subTest(
                        targets=targets, dims=dimensions, pattern=pattern
                    ):
                        pool, kernels = get_fwd_blobs(
                            targets, pattern, 600, dimensions, "simplified"
                        )
                        again_pool, again = get_fwd_blobs(
                            targets, pattern, 600, dimensions, "simplified"
                        )
                        self.assertFalse(
                            any(k.F_pipeline.tag == "qr_tdm_sched" for k in kernels)
                        )
                        self.assertEqual(
                            [k.filename for k in kernels], [k.filename for k in again]
                        )
                        self.assertEqual(pool.render("test"), again_pool.render("test"))
                        if targets == ["gfx1250"] or pattern == "*not_a_kernel*":
                            self.assertFalse(kernels)

    def test_d64_geometry_and_padding(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*d64_{dtype}*_qr_tdm_sched_*", 600, [64], "simplified"
            )
            self.assertEqual(len(kernels), 12)
            for kernel in kernels:
                self.assertEqual(kernel.F_tile.F_bm0, 128)
                self.assertEqual(kernel.F_tile.F_bn0, 64)
                self.assertEqual(kernel.F_tile.F_bn1, 64)
                self.assertEqual(kernel.F_tile.F_occupancy, 3)
                self.assertEqual(kernel.F_pipeline.F_dpad, "f")
                self.assertEqual(kernel.F_pipeline.F_dvpad, "f")
                trait = kernel.api_trait()
                if kernel.F_mode == "batch" and trait.skpad == "f":
                    self.assertIn("a.seqlen_k % 64 == 0", trait.skcheck)
                    self.assertIn("a.seqlen_k != 0", trait.skcheck)
                    self.assertIn("a.cu_seqlen_k_ptr == nullptr", trait.skcheck)
                else:
                    self.assertEqual(trait.spad, "t")
                    self.assertEqual(trait.skpad, "t")
            self.assertIn("a.hdim_v == 64", pool.render("test"))

    def test_fp16_bf16_d128_sched_emission(self):
        for dtype in ("fp16", "bf16"):
            for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
                with self.subTest(dtype=dtype, receipt=receipt):
                    self.check_d128_emission(dtype, receipt, count)

    def test_fp8_sched_dispatched_before_qr_tdm(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_fp8bf16_batch*", 0, [128], "simplified"
        )
        self.assertTrue(any(_is_plain_fp8_sched(k) for k in kernels))
        api = pool.render("test")
        self.assertLess(
            api.rindex("QRKSVS_TDM_SCHED_FP8"),
            api.index("BlockFmhaPipelineEnum::QRKSVS_TDM,"),
        )
        # No seqlen_q condition gates qr_tdm_fp8_sched.
        for k in kernels:
            if _is_plain_fp8_sched(k):
                self.assertNotIn("max_seqlen_q", str(k.F_pipeline.F_constraint))
        self.assertIn("a.block_scale_size_q > 0 && a.block_scale_size_kv > 0", api)
        self.assertIn("a.block_scale_size_q % 128 == 0", api)
        self.assertIn("a.block_scale_size_kv % 32 == 0", api)
        # The FP8 sched tile runs two workgroups per CU, one with an elementwise bias.
        for k in kernels:
            if _is_plain_fp8_sched(k):
                self.assertEqual(
                    k.F_tile.F_occupancy, 1 if k.F_pipeline.F_bias == "bias" else 2
                )
        # BLOCKSCALE qr_tdm and qr_tdm_fp8_sched share one kv-scale row via the lcm.
        self.assertIn("ck_tile::lcm(", pool.render_block_scale_size_kv())

    def test_mxfp8_emits_only_fp8_sched(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_mxfp8*", 0, [128], "simplified"
        )
        self.assertTrue(kernels)
        self.assertEqual({k.F_mode for k in kernels}, {"batch", "group"})
        for k in kernels:
            with self.subTest(kernel=k.name):
                self.assertEqual(k.F_pipeline.tag, "qr_tdm_fp8_sched")
                self.assertEqual(k.F_pipeline.F_vlayout, "row")
                self.assertEqual(k.F_pipeline.F_qscale, "mx")
                self.assertEqual(k.F_tile.F_occupancy, 2)
        api = pool.render("test")
        self.assertIn("reinterpret_cast<uintptr_t>(a.q_descale_ptr) % 4 == 0", api)
        self.assertIn("a.batch_stride_k_descale % 4 == 0", api)
        # No seqlen_q condition gates qr_tdm_fp8_sched.
        for k in kernels:
            self.assertNotIn("max_seqlen_q", str(k.F_pipeline.F_constraint))

    def test_fp8_sched_packgqa_decode_instances(self):
        # Packed-GQA decode on the full tile: batch only, no bias, dispatched ahead of plain sched.
        cases = (
            ("fp8bf16", {"no", "pertensor", "perhead", "blockscale"}),
            ("mxfp8bf16", {"mx"}),
        )
        for (dtype, qscales), hdim in itertools.product(cases, (64, 128, 256)):
            with self.subTest(dtype=dtype, hdim=hdim):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d{hdim}_{dtype}_*", 0, [hdim], "simplified"
                )
                packed = [k for k in kernels if _fp8_sched_kind(k) == "packgqa"]
                self.assertEqual(len(packed), len(qscales) * 8)
                for k in packed:
                    self.assertEqual(k.F_mode, "batch")
                    self.assertEqual(k.F_pipeline.F_bias, "no")
                    self.assertIn(k.F_pipeline.F_qscale, qscales)
                    self.assertIn("_packgqa_", k.name)
                    constraint = str(k.api_trait().constraint)
                    self.assertRegex(
                        constraint,
                        r"fmha_fwd_can_merge_head_groups\(a, 128, \w+, \w+\)",
                    )
                    self.assertIn(
                        "ck_tile::TileFmhaMergeHeadsTraits<ck_tile::TileFmhaTraits<",
                        k.render(),
                    )
                order = _fp8_sched_dispatch_order(pool.render("test"))
                self.assertLess(order.index((128, True)), order.index((128, False)))

    def test_fp8_sched_packgqa_one_wave_decode_instances(self):
        # Packed-GQA decode on the 32-row one-wavefront tile: batch only, dispatched first.
        for dtype, count in (("fp8bf16", 32), ("mxfp8bf16", 8)):
            with self.subTest(dtype=dtype):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d64_{dtype}_*", 0, [64], "simplified"
                )
                packed = [k for k in kernels if _fp8_sched_kind(k) == "packgqa_32row"]
                self.assertEqual(len(packed), count)
                for k in packed:
                    self.assertEqual(k.F_mode, "batch")
                    self.assertEqual((k.F_tile.F_bm0, k.F_tile.F_rm0), (32, 1))
                    self.assertIn("_packgqa_", k.name)
                    constraint = str(k.api_trait().constraint)
                    self.assertRegex(
                        constraint,
                        r"fmha_fwd_can_merge_head_groups\(a, 32, \w+, \w+, true\)",
                    )
                    self.assertIn(
                        "ck_tile::TileFmhaMergeHeadsTraits<ck_tile::TileFmhaTraits<",
                        k.render(),
                    )
                order = _fp8_sched_dispatch_order(pool.render("test"))
                self.assertLess(order.index((32, True)), order.index((128, True)))

    def test_mxfp8bf16_mirrors_mxfp8(self):
        def key(k):
            return (
                k.F_mode,
                k.F_tile.name,
                k.F_pipeline.tag,
                k.F_pipeline.F_qscale,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_bias,
            )

        traits = {}
        for dtype in ("mxfp8", "mxfp8bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
            )
            self.assertTrue(all(k.F_dtype == dtype for k in kernels))
            traits[dtype] = Counter(key(k) for k in kernels)
            if dtype == "mxfp8bf16":
                self.assertIn('t.data_type.compare("mxfp8bf16")', pool.render("test"))
                self.assertIn("FmhaFwdMxFp8Bf16", kernels[0].render())
        # Only the O type differs: same pipelines, tiles and feature variants.
        self.assertTrue(traits["mxfp8bf16"])
        self.assertEqual(traits["mxfp8bf16"], traits["mxfp8"])

    def test_fp8_and_fp8bf16_emit_fp8_sched(self):
        for dtype in ("fp8", "fp8bf16"):
            with self.subTest(dtype=dtype):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_batch*", 0, [128], "simplified"
                )
                sched = [k for k in kernels if _is_plain_fp8_sched(k)]
                self.assertEqual(
                    sorted({k.F_pipeline.F_qscale for k in sched}),
                    ["blockscale", "no", "perhead", "pertensor"],
                )
                for k in sched:
                    self.assertEqual(k.F_dtype, dtype)
                    self.assertEqual(
                        k.F_tile.F_occupancy, 1 if k.F_pipeline.F_bias == "bias" else 2
                    )
                api = pool.render("test")
                self.assertLess(
                    api.rindex("QRKSVS_TDM_SCHED_FP8"),
                    api.index("BlockFmhaPipelineEnum::QRKSVS_TDM,"),
                )

    def test_fp8_sched_emits_lse_variants(self):
        for dtype in ("fp8", "fp8bf16", "mxfp8"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_batch*", 0, [128], "simplified"
                )
                sched = [k for k in kernels if _is_plain_fp8_sched(k)]
                per_lse = 2 if dtype == "mxfp8" else 16
                self.assertEqual(
                    Counter((k.F_pipeline.F_mask, k.F_pipeline.F_lse) for k in sched),
                    Counter(
                        {
                            (mask, lse): per_lse
                            for mask in ("s_no", "s_mask")
                            for lse in ("f", "t")
                        }
                    ),
                )

    def test_fp8_sched_emits_mask_variants(self):
        for mask_impl, masks in (
            ("simplified", ("s_no", "s_mask")),
            ("generic", ("no", "causal", "generic")),
        ):
            for dtype in ("fp8", "fp8bf16", "mxfp8"):
                with self.subTest(dtype=dtype, mask_impl=mask_impl):
                    pool, kernels = get_fwd_blobs(
                        ["gfx1250"], f"*d128_{dtype}_batch*", 0, [128], mask_impl
                    )
                    sched = [k for k in kernels if _is_plain_fp8_sched(k)]
                    per_mask = 4 if dtype == "mxfp8" else 32
                    self.assertEqual(
                        Counter(k.F_pipeline.F_mask for k in sched),
                        Counter({mask: per_mask for mask in masks}),
                    )
                    if dtype != "mxfp8":
                        api = pool.render("test")
                        self.assertLess(
                            api.rindex("QRKSVS_TDM_SCHED_FP8"),
                            api.index("BlockFmhaPipelineEnum::QRKSVS_TDM,"),
                        )

    def test_fp8_sched_emits_sink_variants(self):
        for dtype in ("fp8", "fp8bf16", "mxfp8"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
                )
                sched = [k for k in kernels if _is_plain_fp8_sched(k)]

                def key(k):
                    return (
                        k.F_mode,
                        k.F_pipeline.F_qscale,
                        k.F_pipeline.F_mask,
                        k.F_pipeline.F_lse,
                    )

                by_sink = {
                    sink: Counter(key(k) for k in sched if k.F_pipeline.F_sink == sink)
                    for sink in ("f", "t")
                }
                self.assertTrue(by_sink["t"])
                self.assertEqual(by_sink["t"], by_sink["f"])
                for k in sched:
                    sink = k.F_pipeline.F_sink == "t"
                    constraint = str(k.F_pipeline.F_constraint)
                    # The virtual sink is compiled in exactly for the F_sink instances.
                    self.assertIn(
                        "FmhaTdmSchedFp8PolicyFor<fmha_pipeline_problem>, "
                        + ("true" if sink else "false"),
                        k.render(),
                    )
                    self.assertEqual("a.sink_ptr == nullptr" in constraint, not sink)
                    # Mask sink tokens are handled in-kernel, MX included.
                    self.assertNotIn("sink_size", constraint)

    def test_fp8_sched_emits_elementwise_bias_variants(self):
        for dtype in ("fp8", "fp8bf16", "mxfp8"):
            with self.subTest(dtype=dtype):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
                )
                sched = [k for k in kernels if _is_plain_fp8_sched(k)]

                def key(k):
                    return (
                        k.F_mode,
                        k.F_pipeline.F_qscale,
                        k.F_pipeline.F_mask,
                        k.F_pipeline.F_lse,
                        k.F_pipeline.F_sink,
                    )

                by_bias = {
                    bias: Counter(key(k) for k in sched if k.F_pipeline.F_bias == bias)
                    for bias in ("no", "bias", "alibi")
                }
                self.assertTrue(by_bias["no"])
                self.assertFalse(by_bias["alibi"])
                # MX + bias is not implemented; the other dtypes mirror the no-bias set.
                if dtype == "mxfp8":
                    self.assertFalse(by_bias["bias"])
                else:
                    self.assertEqual(by_bias["bias"], by_bias["no"])
                    self.assertIn("bias_enum::elementwise_bias", pool.render("test"))
                for k in sched:
                    if k.F_pipeline.F_bias == "bias":
                        self.assertEqual(k.F_tile.F_occupancy, 1)
                        self.assertIn("_o1_qr_tdm_fp8_sched_", k.name)

    def test_fp8_sched_d64_mirrors_d128(self):
        def key(k):
            return (
                k.F_mode,
                k.F_pipeline.F_qscale,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_bias,
                k.F_tile.F_occupancy,
            )

        for dtype in ("fp8", "fp8bf16", "mxfp8", "mxfp8bf16"):
            with self.subTest(dtype=dtype):
                traits = {}
                for hdim in (64, 128):
                    pool, kernels = get_fwd_blobs(
                        ["gfx1250"], f"*d{hdim}_{dtype}_*", 0, [hdim], "simplified"
                    )
                    sched = [k for k in kernels if _is_plain_fp8_sched(k)]
                    traits[hdim] = Counter(key(k) for k in sched)
                    if hdim == 64:
                        api = pool.render("test")
                        # MXFP8 has no other D64 pipeline.
                        if dtype in ("mxfp8", "mxfp8bf16"):
                            self.assertEqual(
                                len(sched),
                                sum(k.F_pipeline.F_packgqa == "f" for k in kernels),
                            )
                        else:
                            self.assertLess(
                                api.rindex("QRKSVS_TDM_SCHED_FP8"),
                                api.index("BlockFmhaPipelineEnum::QRKSVS_TDM,"),
                            )
                        for k in sched:
                            # M128 N128 tile, K64 QK WMMA, K128 PV WMMA.
                            t = k.F_tile
                            self.assertEqual(
                                (
                                    t.F_bm0,
                                    t.F_bn0,
                                    t.F_bk0,
                                    t.F_bn1,
                                    t.F_bk1,
                                    t.F_bk0max,
                                ),
                                (128, 128, 64, 64, 128, 64),
                            )
                            self.assertEqual((t.F_wk0, t.F_wk1), (64, 128))
                        if dtype.startswith("mx"):
                            # A D64 MX scale row is two bytes.
                            self.assertIn(
                                "reinterpret_cast<uintptr_t>(a.q_descale_ptr) % 2 == 0",
                                api,
                            )
                            self.assertNotIn("% 4 == 0", api)
                self.assertTrue(traits[64])
                self.assertEqual(traits[64], traits[128])
                # The D64 branch must come first, or the D128 branch shadows it.
                pool, _ = get_fwd_blobs(
                    ["gfx1250"], f"*_{dtype}_*", 0, [64, 128], "simplified"
                )
                api = pool.render("test")
                self.assertLess(
                    api.index("t.hdim_q <= 64 && t.hdim_v <= 64"),
                    api.index("t.hdim_q <= 128 && t.hdim_v <= 128"),
                )

    def test_fp8_sched_d256_mirrors_d128_at_occupancy_one(self):
        def key(k):
            return (
                k.F_mode,
                k.F_pipeline.F_qscale,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_bias,
            )

        for dtype in ("fp8", "fp8bf16", "mxfp8", "mxfp8bf16"):
            with self.subTest(dtype=dtype):
                traits = {}
                for hdim in (128, 256):
                    pool, kernels = get_fwd_blobs(
                        ["gfx1250"], f"*d{hdim}_{dtype}_*", 0, [hdim], "simplified"
                    )
                    sched = [k for k in kernels if _is_plain_fp8_sched(k)]
                    # One D128 tile per bias choice; D256 has a single tile.
                    traits[hdim] = Counter(
                        key(k)
                        for k in sched
                        if hdim == 256
                        or k.F_tile.F_occupancy
                        == (1 if k.F_pipeline.F_bias == "bias" else 2)
                    )
                    if hdim == 256:
                        self.assertEqual(
                            len(sched),
                            sum(k.F_pipeline.F_packgqa == "f" for k in kernels),
                        )
                        for k in sched:
                            t = k.F_tile
                            self.assertEqual(
                                (
                                    t.F_bm0,
                                    t.F_bn0,
                                    t.F_bk0,
                                    t.F_bn1,
                                    t.F_bk1,
                                    t.F_bk0max,
                                ),
                                (128, 128, 256, 256, 128, 256),
                            )
                            self.assertEqual(
                                (t.F_wk0, t.F_wk1, t.F_occupancy), (128, 128, 1)
                            )
                        if dtype.startswith("mx"):
                            # D256 scale rows are read as int32s.
                            self.assertIn(
                                "reinterpret_cast<uintptr_t>(a.q_descale_ptr) % 4 == 0",
                                pool.render("test"),
                            )
                self.assertTrue(traits[256])
                self.assertEqual(traits[256], traits[128])
                pool, _ = get_fwd_blobs(
                    ["gfx1250"], f"*_{dtype}_*", 0, [64, 128, 256], "simplified"
                )
                api = pool.render("test")
                self.assertLess(
                    api.index("t.hdim_q <= 128 && t.hdim_v <= 128"),
                    api.index("t.hdim_q <= 256 && t.hdim_v <= 256"),
                )

    def test_fp8_sched_d192_mirrors_d128(self):
        def key(k):
            return (
                _fp8_sched_kind(k),
                k.F_mode,
                k.F_pipeline.F_qscale,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_bias,
            )

        for dtype in ("fp8", "fp8bf16", "mxfp8", "mxfp8bf16"):
            with self.subTest(dtype=dtype):
                _, d128 = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
                )
                pool, d192 = get_fwd_blobs(
                    ["gfx1250"], f"*d192_{dtype}_*", 0, [192], "simplified"
                )
                # D192/V128 runs only the scheduled FP8 pipelines, with D128's variants.
                self.assertTrue(d192)
                fp8_sched = "qr_tdm_fp8_sched"
                self.assertTrue(all(k.F_pipeline.tag == fp8_sched for k in d192))
                self.assertEqual(
                    Counter(key(k) for k in d192),
                    Counter(key(k) for k in d128 if k.F_pipeline.tag == fp8_sched),
                )
                for k in d192:
                    t = k.F_tile
                    one_wave = _fp8_sched_kind(k) == "packgqa_32row"
                    self.assertEqual(
                        (t.F_bm0, t.F_bn0, t.F_bk0, t.F_bn1, t.F_bk1, t.F_bk0max),
                        (32 if one_wave else 128, 128, 192, 128, 128, 192),
                    )
                    # MX, bias and the packed-GQA LDS/VGPR outliers run one workgroup per CU.
                    packed_one = _fp8_sched_kind(k) == "packgqa" and (
                        (
                            dtype == "fp8"
                            and k.F_pipeline.F_qscale in ("perhead", "blockscale")
                        )
                        or (
                            k.F_pipeline.F_sink == "t"
                            and k.F_pipeline.F_mask not in ("no", "s_no")
                        )
                    )
                    occupancy = (
                        1
                        if not one_wave
                        and (
                            k.F_pipeline.F_qscale == "mx"
                            or k.F_pipeline.F_bias == "bias"
                            or packed_one
                        )
                        else 2
                    )
                    self.assertEqual(
                        (t.F_wk0, t.F_wk1, t.F_occupancy), (64, 128, occupancy)
                    )
                api = pool.render("test")
                self.assertIn("a.hdim_q == 192 && a.hdim_v == 128", api)
                self.assertIn("a.hdim_v % 128 == 0", api)
                self.assertNotIn("a.hdim_v % 192 == 0", api)
                if dtype.startswith("mx"):
                    # D192 Q/K scale rows are read as one int16 per K64 step.
                    self.assertIn(
                        "reinterpret_cast<uintptr_t>(a.q_descale_ptr) % 2 == 0", api
                    )
                pool, _ = get_fwd_blobs(
                    ["gfx1250"], f"*_{dtype}_*", 0, [128, 192, 256], "simplified"
                )
                api = pool.render("test")
                self.assertLess(
                    api.index("t.hdim_q <= 128 && t.hdim_v <= 128"),
                    api.index("t.hdim_q <= 192 && t.hdim_v <= 128"),
                )
                self.assertLess(
                    api.index("t.hdim_q <= 192 && t.hdim_v <= 128"),
                    api.index("t.hdim_q <= 256 && t.hdim_v <= 256"),
                )

    def test_d192_bias_exception_is_fp8_sched_only(self):
        # (192, 128) emits bias for the FP8 scheduled pipeline only; other dtypes keep none.
        for dtype in ("fp16", "bf16", "fp8", "fp8bf16"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d192_{dtype}_*", 0, [192], "simplified"
                )
                biased = [k for k in kernels if k.F_pipeline.F_bias != "no"]
                if dtype.startswith("fp8"):
                    self.assertTrue(biased)
                self.assertTrue(all(_is_plain_fp8_sched(k) for k in biased))

    def test_fp8_sched_group_mode_mirrors_batch(self):
        for dtype in ("fp8", "fp8bf16", "mxfp8"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
                )
                traits = {
                    mode: Counter(
                        (k.F_pipeline.F_qscale, k.F_pipeline.F_mask, k.F_pipeline.F_lse)
                        for k in kernels
                        if _is_plain_fp8_sched(k) and k.F_mode == mode
                    )
                    for mode in ("batch", "group")
                }
                self.assertTrue(traits["group"])
                self.assertEqual(traits["group"], traits["batch"])

    def check_d128_emission(self, dtype, receipt, count):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], f"*d128_{dtype}*_qr_tdm_sched_*", receipt, [128], "simplified"
        )
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        self.assertEqual(len(family), count)
        self.assertEqual(len(family), len(kernels))
        self.assertEqual(len({k.filename for k in kernels}), len(kernels))
        for kernel in family:
            self.assertEqual(kernel.F_hdim, 128)
            self.assertEqual(kernel.F_tile.F_bn0, 128)
            self.assertEqual(kernel.F_tile.F_bn1, 128)
            self.assertEqual(kernel.F_tile.F_bk0max, 128)
            self.assertEqual(kernel.F_tile.F_occupancy, 1)
            self.assertIn("BlockFmhaPipelineQRKSVSTdmSched", kernel.render())
            self.assertIn("QRKSVS_TDM_SCHED", kernel.render())
        api = pool.render("test")
        self.assertIn("a.hdim_q == 128", api)
        self.assertIn("a.max_seqlen_q >= 128", api)

    def test_family_padding_matches_validated_specializations(self):
        kernels = []
        for dtype in ("fp16", "bf16"):
            _, selected = get_fwd_blobs(
                ["gfx1250"], f"*{dtype}*", 600, [128, 192], "simplified"
            )
            kernels.extend(selected)
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        self.assertEqual(len(family), 48)
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        for kernel in family:
            with self.subTest(shape=kernel.F_hdim, name=kernel.filename):
                expected = (
                    "t" if kernel.F_hdim == 128 or kernel.F_mode == "group" else "f"
                )
                self.assertEqual(kernel.F_pipeline.F_spad, expected)
                self.assertEqual(kernel.F_pipeline.F_skpad, expected)
                problem = ProblemContext(
                    dtype=kernel.F_dtype,
                    mode=kernel.F_mode,
                    hdim=kernel.F_hdim,
                    hdim_v=128,
                )
                for qpad in ("f", "t"):
                    for kpad in ("f", "t"):
                        context = KernelContext(
                            tile=kernel.F_tile,
                            pipeline=replace(
                                kernel.F_pipeline, F_spad=qpad, F_skpad=kpad
                            ),
                            mask_impl="simplified",
                        )
                        self.assertEqual(
                            rule(problem, context),
                            qpad == expected and kpad == expected,
                        )

    def test_new_tag_is_not_emitted_for_other_shapes_or_architectures(self):
        _, kernels = get_fwd_blobs(
            ["gfx942", "gfx950"], "*_qr_tdm_sched_*", 600, [-1], "simplified"
        )
        self.assertFalse(kernels)
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*fp32*_qr_tdm_sched_*", 600, [-1], "simplified"
        )
        self.assertFalse(kernels)
        for target in ("gfx1250", "gfx1251", "gfx1252"):
            _, kernels = get_fwd_blobs(
                [target], "*_qr_tdm_sched_*", 600, [-1], "simplified"
            )
            self.assertTrue(
                any(kernel.F_pipeline.tag == "qr_tdm_sched" for kernel in kernels)
            )

    def test_family_and_legacy_entries_use_the_same_arch_check(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*", 100, [128, 192], "simplified"
        )
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        legacy = [k for k in kernels if k.F_pipeline.tag != "qr_tdm_sched"]
        self.assertEqual(
            {k.F_pipeline.tag for k in family},
            {"qr_tdm_sched"},
        )
        self.assertTrue(legacy)
        for kernel in family + legacy:
            with self.subTest(kernel=kernel.name):
                source = kernel.render()
                self.assertIn("defined(__gfx125__)", source)
                self.assertNotIn("ck_tile::get_device_name()", source)

    def test_family_tile_and_pipeline_compatibility_is_symmetric(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*d128_bf16*", 100, [128], "simplified")
        candidate = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched")
        legacy = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm")
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        problem = ProblemContext(dtype="bf16", mode="batch", hdim=128, hdim_v=128)
        context = KernelContext(
            tile=candidate.F_tile, pipeline=candidate.F_pipeline, mask_impl="simplified"
        )
        self.assertTrue(rule(problem, context))
        self.assertTrue(rule(replace(problem, dtype="fp16"), context))
        self.assertFalse(rule(problem, replace(context, tile=legacy.F_tile)))
        self.assertFalse(rule(problem, replace(context, pipeline=legacy.F_pipeline)))
        for changes in (dict(dtype="fp32"), dict(hdim=192), dict(hdim_v=192)):
            self.assertFalse(rule(replace(problem, **changes), context))

    def test_sched_tile_geometry_identity_and_inventory_isolation(self):
        factory = KernelComponentFactoryGfx125
        for dtype in ("fp16", "bf16"):
            inventory = factory.get_hdim_tile_size_dict(dtype)
            for hdim in (128, 192):
                tiles = [
                    tile
                    for tile in inventory[(hdim, 128)]
                    if factory.is_tile_supported_by_qr_tdm_sched(tile, hdim)
                ]
                self.assertEqual(len(tiles), 1 if hdim == 128 else 2)
                for tile in tiles:
                    # Runtime selection does not change tile compatibility.
                    self.assertTrue(
                        factory.is_tile_supported_by_qr_tdm_sched(
                            replace(tile, F_constraint=CppConstraint("false")), hdim
                        )
                    )
                    # Every geometry/occupancy change must reject this pairing.
                    for descriptor in fields(tile):
                        value = getattr(tile, descriptor.name)
                        if isinstance(value, int):
                            with self.subTest(
                                dtype=dtype, tile=tile.name, field=descriptor.name
                            ):
                                self.assertFalse(
                                    factory.is_tile_supported_by_qr_tdm_sched(
                                        replace(tile, **{descriptor.name: value + 1}),
                                        hdim,
                                    )
                                )
                # Callers can tune their inventory without changing later calls.
                tile = tiles[0]
                original_name = tile.name
                tile.F_bm0 += 1
                tile.F_constraint.bool_expr = "false"
                fresh = factory.get_hdim_tile_size_dict(dtype)[(hdim, 128)]
                restored = next(t for t in fresh if t.name == original_name)
                self.assertNotEqual(str(restored.F_constraint), "false")

    @unittest.skipUnless(
        _HIP_HOST_CXX.is_file() and (_ROCM_ROOT / "include").is_dir(),
        "ROCm clang and HIP headers are required for generated TU host syntax",
    )
    def test_sched_generated_occupancy_contract(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*", 200, [-1], "simplified")
        candidates = {}
        diagnostic = "qr_tdm_sched codegen occupancy must match policy tuning"
        for kernel in kernels:
            if kernel.F_pipeline.tag == "qr_tdm_sched":
                key = (
                    kernel.F_dtype,
                    kernel.F_hdim,
                    kernel.F_tile.F_bn0,
                    kernel.F_tile.F_bn1,
                )
                candidates.setdefault(key, kernel)
            else:
                self.assertNotIn(diagnostic, kernel.render())
        self.assertEqual(
            set(candidates),
            {
                (dtype, head, n, v)
                for dtype in ("bf16", "fp16")
                for head, n, v in (
                    (64, 64, 64),
                    (128, 128, 128),
                    (192, 64, 128),
                    (192, 128, 128),
                )
            },
        )
        with tempfile.TemporaryDirectory(
            prefix="fmha-occupancy-contract-"
        ) as directory:
            root = Path(directory)
            for key, kernel in candidates.items():
                for changed in (False, True):
                    with self.subTest(config=key, changed_occupancy=changed):
                        rendered = (
                            replace(
                                kernel,
                                F_tile=replace(
                                    kernel.F_tile,
                                    F_occupancy=kernel.F_tile.F_occupancy + 1,
                                ),
                            )
                            if changed
                            else kernel
                        ).render()
                        self.assertIn(diagnostic, rendered)
                        source = root / "generated.cpp"
                        source.write_text(rendered)
                        result = subprocess.run(
                            [
                                str(_HIP_HOST_CXX),
                                "-x",
                                "hip",
                                "--offload-host-only",
                                "-nogpuinc",
                                "-nogpulib",
                                "-include",
                                "__clang_hip_runtime_wrapper.h",
                                "-std=c++20",
                                "-DUSE_NEW_UNIFIED_FRAMEWORK=0",
                                "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                                "-I",
                                str(_CK_ROOT / "include"),
                                "-I",
                                str(_CK_ROOT),
                                "-I",
                                str(_CK_ROOT / "library" / "include"),
                                "-I",
                                str(_FMHA_EXAMPLE_DIR),
                                "-I",
                                str(_ROCM_ROOT / "include"),
                                "-fsyntax-only",
                                str(source),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if changed:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(diagnostic, result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_family_esm2_exact_scope(self):
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        d128 = "fmha_fwd_d128_bf16_batch_b128_qr_tdm_sched_vr_nmask_nlse_gfx125.cpp"
        d192 = "fmha_fwd_d192_bf16_batch_b128_qr_tdm_sched_vr_nmask_nlse_gfx125.cpp"
        sources = [
            d128,
            d128.replace("batch", "group"),
            d192,
            d192.replace("nmask", "mask"),
            d128.replace("qr_tdm_sched", "qr_tdm"),
            d128.replace("bf16", "fp16"),
            d128.replace("d128", "d256"),
            d128.replace("gfx125", "gfx950"),
            "prefix_" + d128,
            d128 + ".backup",
            "fmha_fwd_api.cpp",
            "mha_fwd.cu",
            # All supported scheduled head dimensions use expert scheduling mode.
            d128.replace("d128", "d64"),
            d128.replace("d128", "d64").replace("bf16", "fp16"),
            # qr_tdm_pingpong without LSE too, while plain qr_tdm (4) and the
            # qr_tdm_pingpong LSE instances keep the default
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong"),
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong").replace("_nlse_", "_lse_"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "flags.cmake"
            script.write_text(
                f'include("{helper}")\n'
                + "\n".join(
                    f'ck_tile_fmha_fwd_get_source_compile_options("{name}" flags)\n'
                    f'message("{i}:${{flags}}")'
                    for i, name in enumerate(sources)
                )
            )
            process = subprocess.run(
                ["cmake", "-P", str(script)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            lines = process.stderr.strip().splitlines()
            self.assertEqual(len(lines), len(sources))
            for i, line in enumerate(lines):
                self.assertTrue(line.startswith(f"{i}:"))
                self.assertEqual(
                    "-amdgpu-expert-scheduling-mode" in line,
                    i in (0, 1, 2, 3, 5, 12, 13, 14),
                )
                self.assertNotIn("-wwm-regalloc=fast", line)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125V128Dispatch(unittest.TestCase):
    def test_fp8_mx_selects_mx_data_type(self):
        # The entry point hands FP8 inputs with MX scales to the MX instances.
        stub = """#include <iostream>
#include <string>
namespace ck_tile { struct stream_config {}; }
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx, perhead };
struct fmha_fwd_traits { std::string data_type; quant_scale_enum qscale_type; };
struct fmha_fwd_args { int max_seqlen_q = 1; };
float fmha_fwd_v3(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&) { return -1; }
float fmha_fwd_v2(fmha_fwd_traits t, fmha_fwd_args, const ck_tile::stream_config&) {
    std::cout << t.data_type;
    return 0;
}
"""
        main = """
int main() {
    fmha_fwd_traits t;
    int scale;
    if(!(std::cin >> t.data_type >> scale)) return 2;
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    return fmha_fwd(t, {}, {}) < 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="fmha-mx-alias-") as directory:
            source = Path(directory) / "api.cpp"
            binary = Path(directory) / "api"
            source.write_text(stub + FMHA_FWD_API_FOOTER + main)
            subprocess.run(
                [_HOST_CXX, "-std=c++17", str(source), "-o", str(binary)],
                check=True,
                timeout=60,
            )
            for dtype, scale, expected in (
                ("fp8", 4, "mxfp8"),
                ("fp8bf16", 4, "mxfp8bf16"),
                ("mxfp8", 4, "mxfp8"),
                ("mxfp4", 4, "mxfp4"),
                ("fp8", 1, "fp8"),
                ("fp8bf16", 5, "fp8bf16"),
                ("bf16", 4, "bf16"),
            ):
                with self.subTest(dtype=dtype, scale=scale):
                    result = subprocess.run(
                        [str(binary)],
                        input=f"{dtype} {scale}\n",
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    self.assertEqual(result.stdout, expected)

    def test_d64_sequence_padding_and_short_q_fallback(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"],
                f"fmha_fwd_d64_{dtype}_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
                600,
                [64],
                "simplified",
            )
            dispatch = _CompiledDispatcher(self, pool, kernels)
            for sq, sk, has_cu_sk, group in (
                (127, 128, 0, 0),
                (128, 128, 0, 0),
                (129, 129, 0, 0),
                (255, 255, 0, 0),
                (128, 0, 0, 0),
                (128, 128, 1, 0),
                (255, 255, 0, 1),
            ):
                candidate = dispatch.expected_id(
                    dtype=dtype,
                    hdim=64,
                    mode="group" if group else "batch",
                    pipeline="qr_tdm" if sq < 128 else "qr_tdm_sched",
                    bm0=64 if sq < 128 else 128,
                    mask="s_no",
                    lse="t",
                    dpad="f",
                    dvpad="f",
                    sink="f",
                    spad="t" if group or has_cu_sk or sk == 0 or sk % 64 else "f",
                )
                self.assertEqual(
                    dispatch.run(
                        dtype=dtype,
                        hdim_q=64,
                        hdim_v=64,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        has_cu_sk=has_cu_sk,
                        group=group,
                        lse=1,
                    ),
                    [candidate],
                )

    def test_fp8_head_dims_below_d64_reach_qr_tdm(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_fp8_[bg]*_nlogits_nbias_nmask_nlse_ndropout_nskip_*_nsink*",
            0,
            [64, 128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: True
        )
        pertensor, blockscale = 1, 2
        for hdim_q, hdim_v, qscale, block_scale_size, pipeline, padded in (
            (32, 32, pertensor, 0, "qr_tdm", True),
            (48, 48, pertensor, 0, "qr_tdm", True),
            (64, 32, pertensor, 0, "qr_tdm", True),
            (64, 64, pertensor, 0, "qr_tdm_fp8_sched", False),
            (64, 64, blockscale, 64, "qr_tdm", False),
            (64, 64, blockscale, 128, "qr_tdm_fp8_sched", False),
        ):
            with self.subTest(hdim_q=hdim_q, hdim_v=hdim_v, qscale=qscale):
                features = dict(
                    dtype="fp8",
                    hdim=64,
                    mode="batch",
                    pipeline=pipeline,
                    qscale="pertensor" if qscale == pertensor else "blockscale",
                    dpad="t" if padded else "f",
                    dvpad="t" if padded else "f",
                    spad="f" if pipeline == "qr_tdm" else "t",
                )
                if pipeline == "qr_tdm_fp8_sched":
                    features["packgqa"] = "f"
                self.assertEqual(
                    dispatch.run(
                        dtype="fp8",
                        hdim_q=hdim_q,
                        hdim_v=hdim_v,
                        qscale=qscale,
                        block_scale_size_q=block_scale_size,
                        block_scale_size_kv=block_scale_size,
                        nhead_k=8,
                    ),
                    [dispatch.expected_id(**features)],
                )

    def test_batch_sched_only_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_sched_*", 600, [128], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
            features = dict(
                mode="batch",
                hdim=128,
                mask="s_mask" if mask else "s_no",
                lse="t" if lse else "f",
                sink="f",
            )
            candidate = dispatch.expected_id(pipeline="qr_tdm_sched", **features)
            args = dict(hdim_q=128, group=0, mask=mask, lse=lse)
            self.assertEqual(dispatch.run(**args), [candidate])

    def test_group_max_q_is_independent_of_member_length_and_head_ratio(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128, 192, 256],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dim in (128, 192):
            for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
                features = dict(
                    mode="group",
                    mask="s_mask" if mask else "s_no",
                    lse="t" if lse else "f",
                    sink="f",
                )
                candidate = dispatch.expected_id(
                    hdim=dim,
                    pipeline="qr_tdm_sched",
                    bn0=64 if dim == 192 else 128,
                    **features,
                )
                for heads, kv_heads in ((1, 1), (8, 8), (8, 2), (8, 1)):
                    for member_q, max_q in (
                        (0, 128),
                        (1, 128),
                        (127, 128),
                        (128, 128),
                        (0, 127),
                        (127, 127),
                    ):
                        fallback = dispatch.expected_id(
                            hdim=dim,
                            bm0=64 if max_q < 128 else 128,
                            pipeline="qr_tdm",
                            dpad="f" if dim == 128 else "t",
                            spad="t",
                            **features,
                        )
                        args = dict(
                            hdim_q=dim,
                            group=1,
                            mask=mask,
                            lse=lse,
                            seqlen_q=member_q,
                            max_seqlen_q=max_q,
                            nhead_q=heads,
                            nhead_k=kv_heads,
                        )
                        expected = candidate if max_q >= 128 else fallback
                        self.assertEqual(dispatch.run(**args), [expected])

    def test_default_dispatch_and_rejected_features(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128, 192, 256],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        family_ids = {
            value
            for kernel, value in dispatch.kernel_ids
            if kernel.F_pipeline.tag == "qr_tdm_sched"
        }
        fallback_ids = {
            value
            for kernel, value in dispatch.kernel_ids
            if kernel.F_pipeline.tag == "qr_tdm"
        }
        for dim in (128, 192):
            for mode, group in (("batch", 0), ("group", 1)):
                for mask in (0, 1, 2, 3):
                    for lse in (0, 1):
                        features = dict(
                            mode=mode,
                            mask="s_mask" if mask else "s_no",
                            lse="t" if lse else "f",
                            sink="f",
                        )
                        candidate = dispatch.expected_id(
                            hdim=dim,
                            pipeline="qr_tdm_sched",
                            bn0=64 if dim == 192 else 128,
                            **features,
                        )
                        args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                        self.assertEqual(dispatch.run(**args), [candidate])
                        sink_fallback = dispatch.run(**args, has_sink_ptr=1)[0]
                        self.assertGreater(sink_fallback, 0)
                        self.assertIn(sink_fallback, fallback_ids)
                        for changes in (
                            dict(seqlen_q=127),
                            dict(arg_q=160),
                            dict(arg_v=192),
                            dict(hdim_q=160),
                            dict(hdim_v=192),
                            dict(hdim_v=127),
                            dict(device="gfx942"),
                        ):
                            rejected_args = dict(args, **changes)
                            self.assertNotIn(
                                dispatch.run(**rejected_args)[0], family_ids
                            )
                        for device in ("gfx1251", "gfx1252"):
                            self.assertEqual(
                                dispatch.run(**args, device=device), [candidate]
                            )
                        for rejected in (
                            dict(logits=1),
                            dict(bias=1),
                            dict(dropout=1),
                            dict(row=0),
                            dict(qscale=1),
                            dict(skip=1),
                            dict(sink=1),
                            dict(dtype="fp16"),
                        ):
                            if rejected == dict(skip=1) and not group:
                                self.assertEqual(
                                    dispatch.run(**args, **rejected),
                                    dispatch.run(**args),
                                )
                            else:
                                self.assertNotIn(
                                    dispatch.run(**args, **rejected)[0], family_ids
                                )

    def test_d128_filtered_priority_and_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for mode, group in (("batch", 0), ("group", 1)):
            for mask in (0, 1, 2, 3):
                for lse in (0, 1):
                    features = dict(
                        mode=mode,
                        hdim=128,
                        mask="s_mask" if mask else "s_no",
                        lse="t" if lse else "f",
                        sink="f",
                    )
                    candidate = dispatch.expected_id(
                        pipeline="qr_tdm_sched", **features
                    )
                    for size in (1, 127, 128, 129, 2047, 2048, 32768):
                        fallback = dispatch.expected_id(
                            pipeline="qr_tdm",
                            spad="t" if group else "f",
                            bm0=64 if size < 128 else 128,
                            dpad="f",
                            **features,
                        )
                        args = dict(
                            group=group,
                            hdim_q=128,
                            hdim_v=128,
                            seqlen_q=size,
                            mask=mask,
                            lse=lse,
                        )
                        self.assertEqual(
                            dispatch.run(**args),
                            [candidate if size >= 128 else fallback],
                        )
            candidate = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_sched",
                mask="s_no",
                lse="f",
                sink="f",
            )
            for changes in (
                dict(arg_q=192),
                dict(arg_v=192),
                dict(hdim_q=192),
                dict(hdim_v=192),
                dict(seqlen_q=127),
            ):
                args = dict(group=group, hdim_q=128, hdim_v=128)
                args.update(changes)
                self.assertNotEqual(dispatch.run(**args), [candidate])
            for changes in (
                dict(logits=1),
                dict(bias=1),
                dict(qscale=1),
                dict(dropout=1),
                dict(skip=1),
                dict(sink=1),
                dict(row=0),
                dict(dtype="fp16"),
            ):
                if changes == dict(skip=1) and not group:
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=128, **changes),
                        dispatch.run(group=group, hdim_q=128),
                    )
                else:
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=128, **changes), [-1]
                    )

    def test_missing_d128_candidate_does_not_disable_generic(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(
            self,
            pool,
            kernels,
            filter_fn=lambda trait: trait.pipeline_tag != "qr_tdm_sched",
        )
        for mode, group in (("batch", 0), ("group", 1)):
            fallback = dispatch.expected_id(
                mode=mode,
                hdim=128,
                bm0=128,
                pipeline="qr_tdm",
                spad="t" if group else "f",
                mask="s_no",
                lse="f",
                sink="f",
                dpad="f",
            )
            self.assertEqual(dispatch.run(group=group, hdim_q=128), [fallback])

    def test_d128_pingpong_tiles(self):
        # qr_tdm_pingpong runs on its own d128 fp16/bf16 bm0=64 tile only, which no
        # other pipeline takes; it alone uses the LDS-shuffle epilogue.
        _, kernels = get_fwd_blobs(["gfx1250"], "*", 600, [128, 192], "simplified")
        pingpong = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"]
        self.assertTrue(pingpong)
        for kernel in kernels:
            owned = kernel.F_tile.F_pipeline_tag == "qr_tdm_pingpong"
            self.assertEqual(
                owned, kernel.F_pipeline.tag == "qr_tdm_pingpong", kernel.name
            )
            self.assertEqual(
                "LdsShuffle2DEpilogue" in kernel.render(), owned, kernel.name
            )
        for kernel in pingpong:
            self.assertIn(kernel.F_dtype, ("fp16", "bf16"))
            self.assertEqual(kernel.F_hdim, 128)
            self.assertIn(kernel.F_tile.F_bm0, (16, 64))
            self.assertIn("BlockFmhaPipelineQRKSVSTdmPingPong", kernel.render())

    def test_d128_pingpong_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits*_ndropout*_nskip*_nqscale*_ntrload*",
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)

        def kernel(pipeline, bm0, mask="s_no", skpad="f", bias="no", sink="f"):
            return dispatch.expected_id(
                hdim=128,
                mode="batch",
                pipeline=pipeline,
                bm0=bm0,
                mask=mask,
                skpad=skpad,
                dpad="f",
                bias=bias,
                lse="f",
                sink=sink,
            )

        def run(**changes):
            return dispatch.run(hdim_q=128, hdim_v=128, **changes)

        sched_ids = {
            index
            for k, index in dispatch.kernel_ids
            if k.F_pipeline.tag == "qr_tdm_sched"
        }
        square = dict(seqlen_q=532, seqlen_k=532)
        causal = dict(mask=1, window_size_left=-1, window_size_right=0)
        # 1280 x 2 (batch, head) pairs: 10 per num_cus (256).
        big = dict(batch=1280, nhead_q=2, nhead_k=2)

        # On a big grid below 640 rows. The unpadded instance serves any seqlen_k,
        # except with an elementwise bias, which still needs a tile multiple.
        self.assertEqual(run(**square, **big), [kernel("qr_tdm_pingpong", 64)])
        # Up to 16 rows the 16-row tile, under the same grid condition (one m-tile:
        # 1280 x 2 blocks, and 1792 x 1 at 32 rows).
        for length, bm0 in ((8, 16), (16, 16), (17, 64)):
            self.assertEqual(
                run(seqlen_q=length, seqlen_k=532, **big),
                [kernel("qr_tdm_pingpong", bm0)],
            )
        self.assertEqual(
            run(seqlen_q=32, seqlen_k=200, batch=1792, nhead_q=1, nhead_k=1),
            [kernel("qr_tdm_pingpong", 64)],
        )
        self.assertEqual(run(seqlen_q=8, seqlen_k=512), [kernel("qr_tdm", 64)])
        self.assertEqual(
            run(**square, **big, bias=1),
            [kernel("qr_tdm_pingpong", 64, skpad="t", bias="bias")],
        )
        self.assertEqual(
            run(seqlen_q=532, seqlen_k=512, bias=1, **big),
            [kernel("qr_tdm_pingpong", 64, bias="bias")],
        )
        self.assertEqual(
            run(**square, **causal, **big),
            [kernel("qr_tdm_pingpong", 64, mask="causal")],
        )
        # qr_tdm_sched below 640 rows on small grids and from 640 rows on any grid;
        # where it does not apply (a sink), qr_tdm.
        self.assertIn(run(**square)[0], sched_ids)
        # The grid threshold, 2 (batch, head) pairs per num_cus: 256 x 2 >= 2 x 256.
        self.assertEqual(
            run(**square, batch=256, nhead_q=2, nhead_k=2),
            [kernel("qr_tdm_pingpong", 64)],
        )
        self.assertIn(run(**square, batch=255, nhead_q=2, nhead_k=2)[0], sched_ids)
        # Not a block count: 16 x 16 heads of 512 rows (2048 bm0=64 blocks) stay on
        # qr_tdm_sched.
        self.assertIn(
            run(seqlen_q=512, seqlen_k=512, batch=16, nhead_q=16, nhead_k=16)[0],
            sched_ids,
        )
        self.assertIn(run(seqlen_q=640, seqlen_k=640, **big)[0], sched_ids)
        self.assertEqual(
            run(**square, **causal, has_sink_ptr=1),
            [kernel("qr_tdm", 128, mask="causal", skpad="t")],
        )
        # Below 128 rows qr_tdm_pingpong needs the big grid too; on a small one qr_tdm
        # keeps its bm0=64 tile.
        self.assertEqual(
            run(seqlen_q=100, seqlen_k=512, **big), [kernel("qr_tdm_pingpong", 64)]
        )
        self.assertEqual(run(seqlen_q=100, seqlen_k=512), [kernel("qr_tdm", 64)])


class SplitKVQrTdmTiles(unittest.TestCase):
    """The split-kv side of the gfx125 qr_tdm tile contract."""

    def _tdm_kernels(self):
        kernels = get_fwd_splitkv_blobs(
            targets=["gfx1250"],
            kernel_filter="",
            receipt=600,
            mask_impl="simplified",
            optdim_list=[-1],
        )
        return [
            k
            for k in kernels
            if getattr(getattr(k, "F_pipeline", None), "tag", None) == "qr_tdm"
        ]

    def test_every_qr_tdm_tile_keeps_bn0_64(self):
        # kReloadKProgressively is gated on kN0 == 64 and turns itself off
        # silently on any other tile, so pin it here in codegen.
        tdm = self._tdm_kernels()
        self.assertTrue(tdm, "no qr_tdm split-kv kernels were generated")
        for kernel in tdm:
            self.assertEqual(kernel.F_tile.F_bn0, 64, kernel.filename)

    def test_qr_tdm_covers_every_gfx125_split_kv_head_dim(self):
        self.assertEqual(
            sorted({kernel.F_hdim for kernel in self._tdm_kernels()}),
            [32, 64, 128, 256],
        )

    def test_qr_tdm_owns_logits_soft_cap(self):
        # The soft cap entries that dispatch on an exact head dim must be
        # qr_tdm's. The head-dim check at the end keeps the set non-empty, so
        # the loop cannot pass vacuously.
        kernels = get_fwd_splitkv_blobs(
            targets=["gfx1250"],
            kernel_filter="",
            receipt=600,
            mask_impl="simplified",
            optdim_list=[-1],
        )
        exact = [
            k
            for k in kernels
            if k.F_pipeline.F_logits == "t" and k.F_pipeline.F_dpad == "f"
        ]
        for kernel in exact:
            self.assertEqual(kernel.F_pipeline.tag, "qr_tdm", kernel.filename)
        # d32 is absent by construction: its bucket carries bk0max 64, so the
        # unpadded head-dim predicate reads 64 and can never hold under
        # hdim_q <= 32. get_fwd_splitkv_blobs drops those as unreachable.
        self.assertEqual(sorted({kernel.F_hdim for kernel in exact}), [64, 128, 256])


class SplitKVDecodeDispatch(unittest.TestCase):
    """Which split-kv instances decode a seqlen_q 1 GQA problem, and how."""

    @staticmethod
    def _kernels(target, receipt):
        return get_fwd_splitkv_blobs(
            targets=[target],
            kernel_filter="",
            receipt=receipt,
            mask_impl="simplified",
            optdim_list=[-1],
        )

    def test_only_tdm_pipelines_merge_heads_with_a_sink(self):
        # A merged row reads the sink logit of its own q head only in the TDM
        # pipelines; the others read one per kv head.
        for target in ("gfx1250", "gfx942"):
            sink = [
                k
                for k in self._kernels(target, 0)
                if k.F_pipeline.F_sink == "t" and k.F_pipeline.F_bias == "no"
            ]
            self.assertTrue(sink, target)
            for kernel in sink:
                self.assertEqual(
                    "(!true || true)" in kernel.template,
                    kernel.F_pipeline.tag in ("qr_tdm", "qr_tdm_pingpong"),
                    kernel.filename,
                )

    def test_pingpong_takes_up_to_16_rows_ahead_of_qr_tdm(self):
        kernels = self._kernels("gfx1250", 0)
        pingpong = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"]
        self.assertTrue(pingpong)
        for kernel in pingpong:
            self.assertEqual(
                (kernel.F_hdim, kernel.F_mode, kernel.F_tile.F_bm0), (128, "batch", 16)
            )
            self.assertEqual(
                (
                    kernel.F_pipeline.F_sink,
                    kernel.F_pipeline.F_bias,
                    kernel.F_pipeline.F_logits,
                    kernel.F_pipeline.F_pagedkv,
                ),
                ("f", "no", "f", "f"),
                kernel.filename,
            )
            self.assertIn("<= 16", str(kernel.F_tile.F_constraint), kernel.filename)
        # Dispatch takes the first hit in emission order: rows 17 and up fall
        # through to the b64 qr_tdm entries behind the 16-row ones.
        for dtype in ("fp16", "bf16"):
            d128 = [
                (i, k.F_pipeline.tag, k.F_tile.F_bm0)
                for i, k in enumerate(kernels)
                if (k.F_dtype, k.F_hdim, k.F_mode) == (dtype, 128, "batch")
            ]
            last_pingpong = max(i for i, tag, _ in d128 if tag == "qr_tdm_pingpong")
            first_b64 = min(i for i, tag, bm0 in d128 if tag == "qr_tdm" and bm0 == 64)
            self.assertLess(last_pingpong, first_b64, dtype)
        self.assertFalse(
            [k for k in kernels if k.F_hdim != 128 and k.F_tile.F_bm0 == 16]
        )
        for receipt in (2, 100, 200, 600):
            self.assertFalse(
                [
                    k
                    for k in self._kernels("gfx1250", receipt)
                    if k.F_pipeline.tag == "qr_tdm_pingpong"
                ],
                receipt,
            )


if __name__ == "__main__":
    unittest.main()
