# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from codegen.ops.fmha_fwd import (
    KernelComponentFactoryGfx125,
    KernelContext,
    ProblemContext,
    family_hdim_requested,
    get_fwd_blobs,
)
from test_fmha_fwd_d192_codegen import _CompiledDispatcher, _HOST_CXX


class TestGfx125V128Codegen(unittest.TestCase):
    @staticmethod
    def dimension_cases():
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text()
        common = re.search(r"set\(FMHA_FWD_CODE_GEN_COMMON_ARGS\s+(.*?)\)", cmake, re.S)
        assert common is not None
        default_dims = re.search(r"--optdim\s+([0-9,]+)", common.group(1))
        assert default_dims is not None
        return [
            [128],
            [192],
            [256],
            [128, 192],
            [-1],
            [int(value) for value in default_dims.group(1).split(",")],
        ]

    def test_family_dimension_alias_is_not_a_legacy_range(self):
        for dimensions, expected in (
            ([-1], {128, 192}),
            ([128], {128}),
            ([192], {192}),
            ([256], {192}),
            ([128, 192], {128, 192}),
            ([32, 64, 80, 128, 256], {128, 192}),
            ([], set()),
            ([64], set()),
        ):
            with self.subTest(dimensions=dimensions):
                actual = {
                    dim
                    for dim in (32, 64, 80, 128, 160, 192, 256)
                    if family_hdim_requested(dim, dimensions)
                }
                self.assertEqual(actual, expected)

    def test_dimension_partition_preserves_legacy_inventory(self):
        # Frozen pre-partition ordered filename/source digests, not the current
        # generator projected against itself. Update only for reviewed legacy changes.
        legacy_digests = {
            (1, 0): "7158cd5605f78ad75e1e0221bb7c639bebaa0c8b949bee24a90b9440655f87c0",
            (
                1,
                100,
            ): "529f7ea66bdc4773f3de8bd7111703697ccdc62a695613d35798151a9f2016e5",
            (
                1,
                200,
            ): "423b7e4c0846097eabfcffe7a7d9062874afcd2de275f1e89245c2c74d82c5d3",
            (
                1,
                600,
            ): "b12d4b5c4a13b8316dc0124d6d607bc4a8f8c568cf18dfe5a17a458b1ee20e00",
            (3, 0): "1ccebca6192b755f67784ea4efd3398bac98894f7eda4f6fe2ccbf9b939d95e2",
            (
                3,
                100,
            ): "8a9acd4dc22c024483d842d0ee5765f41f43f8e4e81e87314970295a17c04299",
            (
                3,
                200,
            ): "80db25c09ba6eb403527262755a07d4476cdf1d98782dd9915302bc0e9e13ad7",
            (
                3,
                600,
            ): "5a39b98d251bd64b0467d6a0b44a8981c467ac04239df9f7ce4b158dd6d23edc",
        }
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for receipt in (0, 100, 200, 600):
                _, full = get_fwd_blobs(targets, None, receipt, [-1], "simplified")
                legacy = "".join(
                    k.filename + "\0" + sha256(k.render().encode()).hexdigest() + "\n"
                    for k in full
                    if not k.api_trait().is_tdm_v128_family
                )
                self.assertEqual(
                    sha256(legacy.encode()).hexdigest(),
                    legacy_digests[(len(targets), receipt)],
                )
                for dimensions in self.dimension_cases():
                    with self.subTest(
                        targets=targets, receipt=receipt, dims=dimensions
                    ):
                        _, kernels = get_fwd_blobs(
                            targets, None, receipt, dimensions, "simplified"
                        )
                        expected = []
                        for kernel in full:
                            trait = kernel.api_trait()
                            if trait.is_tdm_v128_family:
                                requested = family_hdim_requested(
                                    kernel.F_hdim, dimensions
                                )
                            else:
                                requested = (
                                    dimensions == [-1]
                                    or kernel.F_hdim in dimensions
                                    or (
                                        trait.arch.name == "gfx125"
                                        and trait.dtype == "bf16"
                                        and kernel.F_hdim in (192, 256)
                                        and (192 in dimensions or 256 in dimensions)
                                    )
                                )
                            if requested:
                                expected.append(kernel.filename)
                        self.assertEqual([k.filename for k in kernels], expected)
                        self.assertEqual(len(expected), len(set(expected)))

    def test_dimension_filters_keep_only_feature_matched_closure(self):
        for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
            for dimensions in self.dimension_cases():
                with self.subTest(receipt=receipt, dims=dimensions):
                    _, kernels = get_fwd_blobs(
                        ["gfx1250"],
                        "*bf16*_qr_tdm_*v128_*",
                        receipt,
                        dimensions,
                        "simplified",
                    )
                    family = [k for k in kernels if k.api_trait().is_tdm_v128_family]
                    selected_dims = {
                        dim
                        for dim in (128, 192)
                        if family_hdim_requested(dim, dimensions)
                    }
                    self.assertEqual({k.F_hdim for k in family}, selected_dims)
                    self.assertEqual(len(family), count * len(selected_dims))
                    keys = {
                        (k.F_hdim, k.api_trait().gfx125_d192_feature_key)
                        for k in family
                    }
                    for candidate in family:
                        fallback_tags = {
                            k.F_pipeline.tag
                            for k in kernels
                            if not k.api_trait().is_tdm_v128_family
                            and k.F_hdim == (256 if candidate.F_hdim == 192 else 128)
                            and k.api_trait().gfx125_d192_feature_key
                            == candidate.api_trait().gfx125_d192_feature_key
                        }
                        expected_tags = (
                            {"qr", "qr_tdm"}
                            if (candidate.F_hdim == 128 and candidate.F_mode == "batch")
                            else {"qr"}
                        )
                        self.assertEqual(fallback_tags, expected_tags)
                    for kernel in kernels:
                        trait = kernel.api_trait()
                        if trait.is_tdm_v128_family:
                            continue
                        self.assertIn(trait.pipeline_tag, ("qr", "qr_tdm"))
                        family_dim = 192 if kernel.F_hdim == 256 else kernel.F_hdim
                        self.assertIn((family_dim, trait.gfx125_d192_feature_key), keys)

    def test_dimension_generic_and_unsupported_filters_are_deterministic(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for dimensions in self.dimension_cases():
                for pattern in (
                    "*bf16*_qr_vr_*",
                    "*fp16*_qr_tdm_*v128_*",
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
                            any(k.api_trait().is_tdm_v128_family for k in kernels)
                        )
                        self.assertEqual(
                            [k.filename for k in kernels], [k.filename for k in again]
                        )
                        self.assertEqual(pool.render("test"), again_pool.render("test"))
                        if pattern != "*bf16*_qr_vr_*":
                            self.assertFalse(kernels)

    def test_bf16_d128_family_emission_and_filtered_closure(self):
        for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
            with self.subTest(receipt=receipt):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"],
                    "*d128_bf16*_qr_tdm_v128_*",
                    receipt,
                    [128],
                    "simplified",
                )
                family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_v128"]
                self.assertEqual(len(family), count)
                self.assertEqual(len({k.filename for k in kernels}), len(kernels))
                for kernel in family:
                    self.assertEqual(kernel.F_hdim, 128)
                    self.assertEqual(kernel.F_tile.F_bn0, 128)
                    self.assertEqual(kernel.F_tile.F_bn1, 128)
                    self.assertEqual(kernel.F_tile.F_bk0max, 128)
                    self.assertEqual(kernel.F_tile.F_occupancy, 1)
                    self.assertIn("BlockFmhaPipelineQRKSVSTdmV128", kernel.render())
                    self.assertIn("QRKSVS_TDM_V128", kernel.render())
                    fallbacks = {
                        k.F_pipeline.tag
                        for k in kernels
                        if not k.api_trait().is_tdm_v128_family
                        and k.F_mode == kernel.F_mode
                        and k.F_pipeline.F_mask == kernel.F_pipeline.F_mask
                        and k.F_pipeline.F_lse == kernel.F_pipeline.F_lse
                    }
                    self.assertEqual(
                        fallbacks,
                        {"qr", "qr_tdm"} if kernel.F_mode == "batch" else {"qr"},
                    )
                api = pool.render("test")
                self.assertLess(
                    api.index("QRKSVS_TDM_V128"), api.index("t.hdim_q <= 128")
                )

    def test_family_keys_distinguish_shapes_and_features(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*bf16*", 600, [128, 192], "simplified")
        family = [k.api_trait() for k in kernels if k.api_trait().is_tdm_v128_family]
        self.assertEqual(len(family), 16)
        self.assertEqual(len({trait.tdm_v128_key for trait in family}), len(family))
        self.assertEqual({trait.hdim for trait in family}, {"128", "192"})

    def test_family_padding_matches_validated_specializations(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*bf16*", 600, [128, 192], "simplified")
        family = [k for k in kernels if k.api_trait().is_tdm_v128_family]
        self.assertEqual(len(family), 16)
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        for kernel in family:
            with self.subTest(shape=kernel.F_hdim, name=kernel.filename):
                expected = (
                    "t" if kernel.F_hdim == 128 or kernel.F_mode == "group" else "f"
                )
                self.assertEqual(kernel.F_pipeline.F_spad, expected)
                self.assertEqual(kernel.F_pipeline.F_skpad, expected)
                problem = ProblemContext(
                    dtype="bf16", mode=kernel.F_mode, hdim=kernel.F_hdim, hdim_v=128
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

    def test_new_tag_is_not_emitted_for_other_shapes_dtypes_or_architectures(self):
        for targets, dimensions in ((["gfx1250"], [192]), (["gfx942", "gfx950"], [-1])):
            _, kernels = get_fwd_blobs(
                targets, "*_qr_tdm_v128_*", 600, dimensions, "simplified"
            )
            self.assertFalse(kernels)
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*fp16*_qr_tdm_v128_*", 600, [-1], "simplified"
        )
        self.assertFalse(kernels)

    def test_candidate_identity_rejects_unsupported_features(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_v128_*", 100, [128], "simplified"
        )
        trait = next(
            k.api_trait() for k in kernels if k.F_pipeline.tag == "qr_tdm_v128"
        )
        self.assertTrue(trait.is_gfx125_tdm_v128_candidate)
        for changes in (
            dict(tr_load="t"),
            dict(dropout="t"),
            dict(logits="t"),
            dict(bias="bias"),
            dict(qscale="pertensor"),
            dict(skip="t"),
            dict(sink="t"),
            dict(dtype="fp16"),
            dict(hdim="160"),
            dict(bn1=192),
            dict(vlayout="col"),
            dict(pipeline_tag="qr_tdm_d192_v128"),
        ):
            with self.subTest(changes=changes):
                changed = replace(trait, **changes)
                self.assertFalse(changed.is_gfx125_tdm_v128_candidate)
                self.assertNotEqual(changed.tdm_v128_key, trait.tdm_v128_key)
                self.assertIn("false", str(changed.dispatch_constraint()))

    def test_family_tile_and_pipeline_compatibility_is_symmetric(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_v128_*", 100, [128], "simplified"
        )
        candidate = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm_v128")
        legacy = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm")
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        problem = ProblemContext(dtype="bf16", mode="batch", hdim=128, hdim_v=128)
        context = KernelContext(
            tile=candidate.F_tile, pipeline=candidate.F_pipeline, mask_impl="simplified"
        )
        self.assertTrue(rule(problem, context))
        self.assertFalse(rule(problem, replace(context, tile=legacy.F_tile)))
        self.assertFalse(rule(problem, replace(context, pipeline=legacy.F_pipeline)))
        wrong_tag = replace(candidate.F_pipeline, tag="qr_tdm_d192_v128")
        for tile in (candidate.F_tile, legacy.F_tile):
            self.assertFalse(
                rule(problem, replace(context, tile=tile, pipeline=wrong_tag))
            )
        for changes in (dict(dtype="fp16"), dict(hdim=192), dict(hdim_v=192)):
            self.assertFalse(rule(replace(problem, **changes), context))

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_family_esm2_precedence_and_exact_scope(self):
        helper = Path(__file__).with_name("cmake") / "fmha_fwd_source_flags.cmake"
        d128 = "fmha_fwd_d128_bf16_batch_b128_qr_tdm_v128_vr_nmask_nlse_gfx125.cpp"
        d192 = "fmha_fwd_d192_bf16_batch_b128_qr_tdm_d192_v128_vr_nmask_nlse_gfx125.cpp"
        sources = [
            d128,
            d128.replace("batch", "group"),
            d192,
            d192.replace("nmask", "mask"),
            d128.replace("qr_tdm_v128", "qr_tdm"),
            d128.replace("bf16", "fp16"),
            d128.replace("d128", "d256"),
            d128.replace("gfx125", "gfx950"),
            "prefix_" + d128,
            d128 + ".backup",
            "fmha_fwd_api.cpp",
            "mha_fwd.cu",
        ]
        for legacy, family, enabled in (
            ("ON", None, True),
            ("OFF", None, False),
            ("ON", "OFF", False),
            ("OFF", "ON", True),
        ):
            with self.subTest(legacy=legacy, family=family):
                with tempfile.TemporaryDirectory() as directory:
                    script = Path(directory) / "flags.cmake"
                    settings = f"set(FMHA_FWD_GFX1250_D192_TDM_ESM2 {legacy})\n"
                    if family is not None:
                        settings += f"set(FMHA_FWD_GFX1250_TDM_V128_ESM2 {family})\n"
                    script.write_text(
                        settings
                        + "set(FMHA_FWD_GFX1250_D192_BATCH_NMASK_WWM_FAST ON)\n"
                        + f'include("{helper}")\n'
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
                            "-amdgpu-expert-scheduling-mode" in line, enabled and i < 4
                        )
                        self.assertNotIn("-wwm-regalloc=fast", line)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125V128Dispatch(unittest.TestCase):
    def test_batch_qr_only_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_v128_*", 600, [128], "simplified"
        )
        for default_enabled in (False, True):
            dispatch = _CompiledDispatcher(
                self,
                pool,
                kernels,
                default_enabled=default_enabled,
                filter_fn=lambda trait: not (
                    trait.mode == "batch" and trait.pipeline_tag == "qr_tdm"
                ),
            )
            for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
                features = dict(
                    mode="batch",
                    hdim=128,
                    mask="s_mask" if mask else "s_no",
                    lse="t" if lse else "f",
                    sink="f",
                )
                candidate = dispatch.expected_id(pipeline="qr_tdm_v128", **features)
                fallback = dispatch.expected_id(
                    pipeline="qr", bm0=64, dpad="f", **features
                )
                args = dict(hdim_q=128, group=0, mask=mask, lse=lse)
                self.assertEqual(dispatch.run(env="1", **args), [candidate])
                self.assertEqual(dispatch.run(env="1", force="1", **args), [fallback])
                self.assertEqual(
                    dispatch.run(env=None, **args),
                    [candidate if default_enabled else fallback],
                )

    def test_one_missing_family_sibling_keeps_feature_matched_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*_v128_*", 600, [128, 256], "simplified"
        )
        for missing_dim, missing_mode in (
            (128, "batch"),
            (128, "group"),
            (192, "batch"),
            (192, "group"),
        ):
            for default_enabled in (False, True):
                dispatch = _CompiledDispatcher(
                    self,
                    pool,
                    kernels,
                    default_enabled=default_enabled,
                    filter_fn=lambda t: not (
                        t.is_tdm_v128_family
                        and t.hdim == str(missing_dim)
                        and t.mode == missing_mode
                        and t.mask == "s_mask"
                        and t.lse == "t"
                    ),
                )
                for dim in (128, 192):
                    for mode, group in (("batch", 0), ("group", 1)):
                        for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
                            features = dict(
                                mode=mode,
                                mask="s_mask" if mask else "s_no",
                                lse="t" if lse else "f",
                                sink="f",
                            )
                            fallback_fields = dict(
                                hdim=128 if dim == 128 else 256,
                                bm0=64,
                                pipeline="qr_tdm" if dim == 128 and not group else "qr",
                                dpad="t" if dim == 192 or group else "f",
                                **features,
                            )
                            if dim == 192:
                                fallback_fields["spad"] = "t"
                            fallback = dispatch.expected_id(**fallback_fields)
                            missing = (
                                dim == missing_dim
                                and mode == missing_mode
                                and mask == 1
                                and lse == 1
                            )
                            candidate = (
                                fallback
                                if missing
                                else dispatch.expected_id(
                                    hdim=dim,
                                    pipeline="qr_tdm_v128"
                                    if dim == 128
                                    else "qr_tdm_d192_v128",
                                    **features,
                                )
                            )
                            args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                            self.assertEqual(dispatch.run(env="1", **args), [candidate])
                            self.assertEqual(
                                dispatch.run(env=None, **args),
                                [candidate if default_enabled else fallback],
                            )
                            self.assertEqual(
                                dispatch.run(env="1", force="1", **args), [fallback]
                            )

    def test_group_max_q_is_independent_of_member_length_and_head_ratio(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*_v128_*", 600, [128, 256], "simplified"
        )
        for default_enabled in (False, True):
            dispatch = _CompiledDispatcher(
                self, pool, kernels, default_enabled=default_enabled
            )
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
                        pipeline="qr_tdm_v128" if dim == 128 else "qr_tdm_d192_v128",
                        **features,
                    )
                    fallback = dispatch.expected_id(
                        hdim=128 if dim == 128 else 256,
                        bm0=64,
                        pipeline="qr",
                        dpad="t",
                        spad="t",
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
                            self.assertEqual(dispatch.run(env="1", **args), [expected])
                            self.assertEqual(
                                dispatch.run(env="1", force="1", **args), [fallback]
                            )
                            self.assertEqual(
                                dispatch.run(env=None, **args),
                                [expected if default_enabled else fallback],
                            )

    def test_development_and_final_controls_use_real_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*_v128_*", 600, [128, 256], "simplified"
        )
        for default_enabled in (False, True):
            dispatch = _CompiledDispatcher(
                self, pool, kernels, default_enabled=default_enabled
            )
            family_ids = {
                value
                for kernel, value in dispatch.kernel_ids
                if kernel.api_trait().is_tdm_v128_family
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
                                pipeline="qr_tdm_v128"
                                if dim == 128
                                else "qr_tdm_d192_v128",
                                **features,
                            )
                            fallback_fields = dict(
                                hdim=128 if dim == 128 else 256,
                                pipeline="qr_tdm" if dim == 128 and not group else "qr",
                                dpad="t" if dim == 192 or group else "f",
                                **features,
                            )
                            if dim == 128:
                                fallback_fields["bm0"] = 64
                            else:
                                fallback_fields["spad"] = "t"
                            fallback = dispatch.expected_id(**fallback_fields)
                            args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                            for old in (None, "0", "1"):
                                for force in (None, "0", "1"):
                                    with self.subTest(
                                        default=default_enabled,
                                        dim=dim,
                                        mode=mode,
                                        mask=mask,
                                        lse=lse,
                                        old=old,
                                        force=force,
                                    ):
                                        enabled = force != "1" and (
                                            default_enabled or old == "1"
                                        )
                                        self.assertEqual(
                                            dispatch.run(env=old, force=force, **args),
                                            [candidate if enabled else fallback],
                                        )
                            for force in ("", "01", "true", "1x", " 1", "1 "):
                                self.assertEqual(
                                    dispatch.run(env="1", force=force, **args),
                                    [candidate],
                                )
                            for old in ("", "01", "true", "1x", " 1", "1 "):
                                self.assertEqual(
                                    dispatch.run(env=old, **args),
                                    [candidate if default_enabled else fallback],
                                )
                            self.assertEqual(
                                dispatch.run(
                                    env="1", next_env="0", next_force="1", **args
                                ),
                                [candidate, candidate],
                            )
                            self.assertEqual(
                                dispatch.run(
                                    env="1", force="1", next_force="0", **args
                                ),
                                [fallback, fallback],
                            )
                            self.assertEqual(
                                dispatch.run(
                                    env="1", force="1", next_force="__UNSET__", **args
                                ),
                                [fallback, fallback],
                            )
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
                                    dispatch.run(env="1", **rejected_args)[0],
                                    family_ids,
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
                                self.assertNotIn(
                                    dispatch.run(env="1", **args, **rejected)[0],
                                    family_ids,
                                )

    def test_d128_filtered_priority_and_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_v128_*", 600, [128], "simplified"
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
                    candidate = dispatch.expected_id(pipeline="qr_tdm_v128", **features)
                    for size in (1, 127, 128, 129, 2047, 2048, 32768):
                        fallback = dispatch.expected_id(
                            pipeline="qr" if group else "qr_tdm",
                            bm0=64 if size < 2048 else 128,
                            dpad="t" if group else "f",
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
                            dispatch.run(env="1", **args),
                            [candidate if size >= 128 else fallback],
                        )
                        self.assertEqual(dispatch.run(env=None, **args), [fallback])
            candidate = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_v128",
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
                self.assertEqual(dispatch.run(group=group, hdim_q=128, **changes), [-1])

    def test_missing_d128_candidate_does_not_disable_generic(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_v128_*", 600, [128], "simplified"
        )
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: not trait.is_tdm_v128_family
        )
        for mode, group in (("batch", 0), ("group", 1)):
            fallback = dispatch.expected_id(
                mode=mode,
                hdim=128,
                bm0=64,
                pipeline="qr" if group else "qr_tdm",
                mask="s_no",
                lse="f",
                sink="f",
                dpad="t" if group else "f",
            )
            self.assertEqual(dispatch.run(group=group, hdim_q=128), [fallback])
            self.assertEqual(
                dispatch.run(env=None, group=group, hdim_q=128), [fallback]
            )


if __name__ == "__main__":
    unittest.main()
