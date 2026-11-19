# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host checks of the paged dense kernel ABI and launcher on gfx942 and gfx950: the
paged kernel arguments, and the launch-time refusals that keep a bad cache or
block table from driving an out-of-bounds read. Tensor stand-ins replace torch,
and compile is patched out, so no torch, comgr or GPU is needed."""

import importlib
import unittest

from ._dense_paged_stand_ins import (
    CacheStandIn,
    I32DeviceTensor,
    dense_arch_spec_class,
)


class TestDensePagedLaunch(unittest.TestCase):
    def test_dense_paged_builds_and_signature(self):
        """Paged ABI: the signature carries block_tables/kv_lens/block_table_stride,
        and the paged spec builds on host."""
        for arch in ("gfx942", "gfx950"):
            ad = importlib.import_module(f"kernels.{arch}.attention_dense")
            AttentionDenseSpec = dense_arch_spec_class(ad)
            attention_dense_signature = ad.attention_dense_signature
            build_attention_dense = ad.build_attention_dense
            with self.subTest(arch=arch):
                spec = AttentionDenseSpec(
                    batch=1,
                    seqlen_q=8192,
                    seqlen_kv=8192,
                    num_query_heads=32,
                    num_kv_heads=8,
                    head_size=128,
                    causal=True,
                    dtype="fp16",
                    sliding_window=4096,
                    block_n=64,
                    paged=True,
                    block_size=16,
                    num_kv_blocks=512,
                )
                names = [p["name"] for p in attention_dense_signature(spec)]
                for req in ("block_tables", "kv_lens", "block_table_stride"):
                    self.assertIn(req, names)
                self.assertIsNotNone(build_attention_dense(spec, arch=arch))

    def test_dense_paged_launcher_rejects_kv_cache_shape_mismatch(self):
        """The launcher must reject a paged K/V cache whose shape disagrees with
        the spec that sizes the buffer-resource bound
        (num_kv_blocks*block_size*num_kv_heads*head_size). A too-small cache
        under a too-large ``num_kv_blocks`` sets an oversized hardware bound and lets
        a block-table entry drive an OOB read. Host-only: the shape check raises
        before any comgr compile / GPU launch, so no torch or GPU is required (only
        ``.shape`` is read pre-launch, so lightweight stand-ins suffice)."""
        from types import SimpleNamespace
        from unittest import mock

        for arch in ("gfx942", "gfx950"):
            ad = importlib.import_module(f"kernels.{arch}.attention_dense")
            AttentionDenseSpec = dense_arch_spec_class(ad)
            run_attention_dense_torch = ad.run_attention_dense_torch
            with self.subTest(arch=arch):
                spec = AttentionDenseSpec(
                    batch=1,
                    seqlen_q=8192,
                    seqlen_kv=8192,
                    num_query_heads=32,
                    num_kv_heads=8,
                    head_size=128,
                    causal=True,
                    dtype="fp16",
                    sliding_window=4096,
                    block_n=64,
                    paged=True,
                    block_size=16,
                    num_kv_blocks=512,
                )
                want = (
                    spec.num_kv_blocks,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )
                wrong = (
                    256,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )  # 256 != 512
                qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
                q = SimpleNamespace(shape=qshape)
                out = SimpleNamespace(shape=qshape)
                block_tables = I32DeviceTensor([list(range(512))])
                kv_lens = I32DeviceTensor([spec.seqlen_kv])

                # Negative (K): fewer physical blocks than the spec claims -> reject.
                with self.assertRaises(ValueError) as ctx:
                    run_attention_dense_torch(
                        spec=spec,
                        q=q,
                        k=CacheStandIn(shape=wrong),
                        v=CacheStandIn(shape=want),
                        out=out,
                        scale=1.0,
                        block_tables=block_tables,
                        kv_lens=kv_lens,
                    )
                msg = str(ctx.exception)
                self.assertIn("paged k cache shape", msg)
                self.assertIn("(256,", msg)  # the offending shape is reported
                self.assertIn("OOB", msg)

                # Negative (V): the same defect on the value cache is also caught.
                with self.assertRaises(ValueError):
                    run_attention_dense_torch(
                        spec=spec,
                        q=q,
                        k=CacheStandIn(shape=want),
                        v=CacheStandIn(shape=wrong),
                        out=out,
                        scale=1.0,
                        block_tables=block_tables,
                        kv_lens=kv_lens,
                    )

                # Positive control: MATCHING shapes pass the shape gate and reach
                # compile. Patch compile_kernel to a sentinel so the launcher stays
                # host-only (no comgr, no GPU) yet proves it did NOT raise our shape
                # ValueError. An empty launcher cache keeps a prior test from letting us
                # skip compile. validate_paged is off here to isolate the SHAPE gate
                # (block-table CONTENTS validation is covered by
                # test_..._validates_block_table_bounds).
                sentinel = RuntimeError("reached-compile")
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as ok_ctx:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=CacheStandIn(shape=want),
                            v=CacheStandIn(shape=want),
                            out=out,
                            scale=1.0,
                            block_tables=block_tables,
                            kv_lens=kv_lens,
                            validate_paged=False,
                        )
                    self.assertIs(ok_ctx.exception, sentinel)

    def test_dense_paged_launcher_rejects_malformed_block_metadata(self):
        """Inputs the kernel cannot read as laid out -- block_tables / kv_lens that are
        not [batch, >= pages] / [batch] int32 arrays on the K/V device, a strided K/V
        cache or a V cache off K's device -- are rejected from metadata alone, even
        with validate_paged=False; a validate_paged that is no bool is rejected too.
        Host-only: tensor stand-ins, the check raises before compile."""
        from types import SimpleNamespace

        for arch in ("gfx942", "gfx950"):
            ad = importlib.import_module(f"kernels.{arch}.attention_dense")
            spec = dense_arch_spec_class(ad)(
                batch=1,
                seqlen_q=1024,
                seqlen_kv=1024,
                num_query_heads=16,
                num_kv_heads=4,
                head_size=128,
                causal=True,
                dtype="fp16",
                sliding_window=256,
                paged=True,
                block_size=16,
                num_kv_blocks=128,
            )
            q = SimpleNamespace(shape=(1, 1024, 16, 128))
            cache = CacheStandIn((128, 16, 4, 128))
            table, lens = [list(range(64))], [1024]
            i32 = I32DeviceTensor
            for block_tables, kv_lens, k, error in (
                (i32([list(range(63))]), i32(lens), cache, "block_tables shape"),
                (i32(list(range(64))), i32(lens), cache, "block_tables shape"),
                (i32(table * 2), i32(lens), cache, "block_tables shape"),
                (i32(table), i32([]), cache, "kv_lens shape"),
                (i32(table), i32(lens * 2), cache, "kv_lens shape"),
                (
                    i32(table, dtype="torch.int64"),
                    i32(lens),
                    cache,
                    "block_tables dtype",
                ),
                (i32(table), i32(lens, dtype="torch.int64"), cache, "kv_lens dtype"),
                (i32(table, is_cuda=False), i32(lens), cache, "GPU tensor"),
                (i32(table, device="cuda:1"), i32(lens), cache, "K/V device"),
                (i32(table, inner_stride=2), i32(lens), cache, "innermost stride"),
                (
                    i32(table, dtype="torch.uint32"),
                    i32(lens),
                    cache,
                    "block_tables dtype",
                ),
                (
                    i32(table),
                    i32(lens),
                    CacheStandIn((128, 16, 4, 128), dtype="torch.bfloat16"),
                    "cache dtype",
                ),
                (
                    i32(table),
                    i32(lens),
                    CacheStandIn((128, 16, 4, 128), contiguous=False),
                    "contiguous",
                ),
            ):
                with self.subTest(arch=arch, error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        ad.run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=k,
                            out=q,
                            scale=1.0,
                            block_tables=block_tables,
                            kv_lens=kv_lens,
                            validate_paged=False,
                        )
            for v in (
                CacheStandIn((128, 16, 4, 128), device="cpu", is_cuda=False),
                CacheStandIn((128, 16, 4, 128), device="cuda:1"),
            ):
                with self.subTest(arch=arch, v_device=v.device):
                    with self.assertRaisesRegex(
                        ValueError, "paged v cache must be a GPU tensor on the K device"
                    ):
                        ad.run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=cache,
                            v=v,
                            out=q,
                            scale=1.0,
                            block_tables=i32(table),
                            kv_lens=i32(lens),
                            validate_paged=False,
                        )
            for flag in (None, 0, 1, "no"):
                with self.subTest(arch=arch, validate_paged=flag):
                    with self.assertRaisesRegex(
                        ValueError, "validate_paged must be a bool"
                    ):
                        ad.run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=cache,
                            v=cache,
                            out=q,
                            scale=1.0,
                            block_tables=i32(table),
                            kv_lens=i32(lens),
                            validate_paged=flag,
                        )

    def test_dense_paged_launcher_validates_block_table_bounds(self):
        """Gated CONTENTS check on paged block tables: an entry outside
        [0, num_kv_blocks) reads 0 via the bounds-checked cache SRD (silent wrong
        output), so validate_paged=True rejects it loudly, validate_paged=False
        skips it (sync-free hot path), and only DEREFERENCED pages are checked.
        Host-only: block_tables/kv_lens are plain Python lists (only ints + slicing
        are read), so no torch / GPU."""
        from types import SimpleNamespace
        from unittest import mock

        for arch in ("gfx942", "gfx950"):
            ad = importlib.import_module(f"kernels.{arch}.attention_dense")
            AttentionDenseSpec = dense_arch_spec_class(ad)
            run_attention_dense_torch = ad.run_attention_dense_torch
            with self.subTest(arch=arch):
                spec = AttentionDenseSpec(
                    batch=1,
                    seqlen_q=256,
                    seqlen_kv=256,
                    num_query_heads=32,
                    num_kv_heads=8,
                    head_size=128,
                    causal=True,
                    dtype="fp16",
                    sliding_window=4096,
                    block_n=64,
                    paged=True,
                    block_size=64,
                    num_kv_blocks=512,
                )
                want = (
                    spec.num_kv_blocks,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )
                qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
                q = SimpleNamespace(shape=qshape)
                out = SimpleNamespace(shape=qshape)
                k = CacheStandIn(shape=want)
                v = CacheStandIn(shape=want)
                kv_lens = I32DeviceTensor([256])  # == seqlen_kv -> 4 pages deref
                good_bt = I32DeviceTensor([[0, 1, 2, 3]])  # all in [0, 512)
                bad_bt = I32DeviceTensor([[0, 1, 600, 3]])  # 600 >= num_kv_blocks, USED
                sentinel = RuntimeError("reached-compile")

                # Negative (validate on, default): a used entry >= num_kv_blocks ->
                # reject loudly, before any compile / launch.
                with self.assertRaises(ValueError) as ctx:
                    run_attention_dense_torch(
                        spec=spec,
                        q=q,
                        k=k,
                        v=v,
                        out=out,
                        scale=1.0,
                        block_tables=bad_bt,
                        kv_lens=kv_lens,
                    )
                msg = str(ctx.exception)
                self.assertIn("600", msg)
                self.assertIn("num_kv_blocks=512", msg)

                # Only DEREFERENCED pages are checked: a bad id in a column BEYOND
                # n_pages (index >= ceil(256/64)=4) is never read, so it is ignored.
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as beyond:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=v,
                            out=out,
                            scale=1.0,
                            block_tables=I32DeviceTensor([[0, 1, 2, 3, 600]]),
                            kv_lens=kv_lens,
                        )
                    self.assertIs(beyond.exception, sentinel)

                # Gate off: the bad table is skipped entirely -> reaches compile.
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as gated:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=v,
                            out=out,
                            scale=1.0,
                            block_tables=bad_bt,
                            kv_lens=kv_lens,
                            validate_paged=False,
                        )
                    self.assertIs(gated.exception, sentinel)

                # Positive (validate on, valid table): passes the gate -> reaches
                # compile.
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as ok:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=v,
                            out=out,
                            scale=1.0,
                            block_tables=good_bt,
                            kv_lens=kv_lens,
                        )
                    self.assertIs(ok.exception, sentinel)

    def test_dense_paged_launcher_validates_kv_len_contract(self):
        """Gated CONTENTS check: the kernel visits ALL compile-time seqlen_kv
        tiles but the page-bounds mask uses the runtime kv_len, so a kv_len
        shorter than seqlen_kv leaves uncovered tiles reading page 0 (the masked
        block-table default) -> silently wrong output. validate_paged=True
        enforces kv_lens[i] == seqlen_kv; validate_paged=False skips it (hot
        path). Host-only (plain lists), no torch / GPU."""
        from types import SimpleNamespace
        from unittest import mock

        for arch in ("gfx942", "gfx950"):
            ad = importlib.import_module(f"kernels.{arch}.attention_dense")
            AttentionDenseSpec = dense_arch_spec_class(ad)
            run_attention_dense_torch = ad.run_attention_dense_torch
            with self.subTest(arch=arch):
                spec = AttentionDenseSpec(
                    batch=1,
                    seqlen_q=256,
                    seqlen_kv=256,
                    num_query_heads=32,
                    num_kv_heads=8,
                    head_size=128,
                    causal=True,
                    dtype="fp16",
                    sliding_window=4096,
                    block_n=64,
                    paged=True,
                    block_size=64,
                    num_kv_blocks=512,
                )
                want = (
                    spec.num_kv_blocks,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                )
                qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
                q = SimpleNamespace(shape=qshape)
                out = SimpleNamespace(shape=qshape)
                k = CacheStandIn(shape=want)
                v = CacheStandIn(shape=want)
                good_bt = I32DeviceTensor([[0, 1, 2, 3]])
                sentinel = RuntimeError("reached-compile")

                # Negative (validate on, default): kv_len < seqlen_kv -> reject loudly.
                with self.assertRaises(ValueError) as ctx:
                    run_attention_dense_torch(
                        spec=spec,
                        q=q,
                        k=k,
                        v=v,
                        out=out,
                        scale=1.0,
                        block_tables=good_bt,
                        kv_lens=I32DeviceTensor([128]),
                    )
                msg = str(ctx.exception)
                self.assertIn("kv_lens[0]=128", msg)
                self.assertIn("seqlen_kv=256", msg)

                # Gate off: the contract is not checked -> reaches compile.
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as gated:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=v,
                            out=out,
                            scale=1.0,
                            block_tables=good_bt,
                            kv_lens=I32DeviceTensor([128]),
                            validate_paged=False,
                        )
                    self.assertIs(gated.exception, sentinel)

                # Positive (kv_len == seqlen_kv): passes the gate -> reaches compile.
                with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
                    "rocke.helpers.compile.compile_kernel", side_effect=sentinel
                ):
                    with self.assertRaises(RuntimeError) as ok:
                        run_attention_dense_torch(
                            spec=spec,
                            q=q,
                            k=k,
                            v=v,
                            out=out,
                            scale=1.0,
                            block_tables=good_bt,
                            kv_lens=I32DeviceTensor([256]),
                        )
                    self.assertIs(ok.exception, sentinel)
