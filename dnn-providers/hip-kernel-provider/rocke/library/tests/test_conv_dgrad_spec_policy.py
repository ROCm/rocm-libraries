# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only policy tests for the implicit-GEMM dgrad spec (no GPU needed).

Covers the validator rules that replaced silently-ignored or unsafe knobs:

  - async_dma / unroll_k / chiplet_swizzle are rejected (the dgrad builder has
    no such path);
  - an accumulator tile above 256 fp32 registers per lane is rejected (it
    fills a wave64's whole register file);
  - compv4 is a schedule policy here and is not charged a second LDS buffer,
    and the LDS charge equals what the builder allocates;
  - the K-outer B fragment rule the builder enforces is in the validator too;
  - a non-default ``lds_k_pad`` reaches the kernel name.

And the stride-1 tap-outer K loop (``tap_outer_k``): default off, rejected
wherever it cannot compute the problem (it has no fallback loop), the shape
of the loop it emits, and its cache identity, label and serving rule.
"""

from __future__ import annotations

import unittest
from unittest import mock

import kernels.common.conv_implicit_gemm_dgrad as dgrad_mod
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    dgrad_lds_bytes,
    is_valid_dgrad_spec,
)

_ARCH = "gfx950"


def _problem(**kw):
    base = {
        "N": 2,
        "Hi": 14,
        "Wi": 14,
        "C": 64,
        "K": 64,
        "Y": 3,
        "X": 3,
        "pH": 1,
        "pW": 1,
    }
    base.update(kw)
    return ConvProblem(**base)


def _spec(problem=None, dtype="bf16", **kw):
    base = {
        "problem": problem if problem is not None else _problem(),
        "data": ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "lds_k_outer": True,
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _smem_bytes(kernel, spec):
    """(A+B bytes, C bytes) of the kernel's LDS allocations (16-bit dtypes)."""
    ab = c = 0
    for op in kernel.body.ops:
        if op.name != "tile.smem_alloc":
            continue
        shape = op.results[0].type.shape
        n = shape[0] * shape[1] * 2
        if spec.epilogue == "cshuffle" and tuple(shape) == (spec.tile_m, spec.tile_n):
            c += n
        else:
            ab += n
    return ab, c


class TestValidatorRules(unittest.TestCase):
    def test_ignored_knobs_are_rejected(self):
        for kw in ({"async_dma": True}, {"unroll_k": True}, {"chiplet_swizzle": True}):
            with self.subTest(**kw):
                ok, why = is_valid_dgrad_spec(_spec(lds_k_outer=False, **kw), _ARCH)
                self.assertFalse(ok)
                self.assertIn("does not implement", why)

    def test_spilling_accumulator_tile_rejected(self):
        # A 256x256 tile with 1x2 warps of the 16x16x32 atom: 512 fp32
        # accumulators per lane, the whole wave64 register file.
        big = _problem(N=4, Hi=12, Wi=12, C=384, K=384)
        bad = _spec(
            problem=big,
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=1,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        )
        ok, why = is_valid_dgrad_spec(bad, _ARCH)
        self.assertFalse(ok)
        self.assertIn("accumulator", why)
        # 256 accumulators per lane (2x2 warps) is the largest accepted.
        ok, why = is_valid_dgrad_spec(
            _spec(
                problem=big,
                tile_m=256,
                tile_n=256,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            _ARCH,
        )
        self.assertTrue(ok, why)

    def test_compv4_is_single_buffered(self):
        # A/B single-buffered fits the gfx950 LDS; a doubled charge would not.
        spec = _spec(
            problem=_problem(C=256, K=256),
            tile_m=256,
            tile_n=256,
            tile_k=128,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            pipeline="compv4",
        )
        ok, why = is_valid_dgrad_spec(spec, _ARCH)
        self.assertTrue(ok, why)
        self.assertEqual(dgrad_lds_bytes(spec), dgrad_lds_bytes(_with(spec, "mem")))

    def test_lds_charge_equals_the_allocation(self):
        p16 = _problem(Hi=16, Wi=16, C=128, K=128)
        for kw in (
            {},
            {"pipeline": "compv4"},
            {"epilogue": "default"},
            {"lds_k_outer": False},
            {"lds_k_outer": False, "lds_k_pad": 16},
            {"tile_m": 128, "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 32},
        ):
            with self.subTest(**kw):
                spec = _spec(problem=p16, **kw)
                k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
                ab, c = _smem_bytes(k, spec)
                self.assertEqual(dgrad_lds_bytes(spec), max(ab, c))

    def test_kouter_fragment_rule_is_in_the_validator(self):
        # No atom the targets carry violates it today; a fragment of 2
        # elements per lane (half a ds_read_tr16_b64) must be refused by the
        # validator rather than by the builder.
        spec = _spec()
        real = dgrad_mod._resolve_dgrad_op(spec, _ARCH)
        fake = mock.Mock(wraps=real)
        fake.b_frag_len = 2
        with mock.patch.object(dgrad_mod, "_resolve_dgrad_op", return_value=fake):
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
        self.assertFalse(ok)
        self.assertIn("B fragment length", why)
        # M-outer specs do not read through the transpose.
        with mock.patch.object(dgrad_mod, "_resolve_dgrad_op", return_value=fake):
            ok, why = is_valid_dgrad_spec(_spec(lds_k_outer=False), _ARCH)
        self.assertTrue(ok, why)

    def test_every_accepted_kouter_atom_builds(self):
        for wt, wk in ((16, 16), (16, 32), (32, 8), (32, 16)):
            spec = _spec(warp_tile_m=wt, warp_tile_n=wt, warp_tile_k=wk)
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
            with self.subTest(atom=(wt, wk)):
                self.assertTrue(ok, why)
                build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)


def _with(spec, pipeline):
    from dataclasses import replace

    return replace(spec, pipeline=pipeline)


class TestKernelName(unittest.TestCase):
    def test_lds_k_pad_reaches_the_name(self):
        self.assertNotIn("_kp", _spec(lds_k_outer=False).kernel_name())
        # The default pad, spelled out, is the default build.
        self.assertNotIn("_kp", _spec(lds_k_outer=False, lds_k_pad=8).kernel_name())
        self.assertIn("_kp16", _spec(lds_k_outer=False, lds_k_pad=16).kernel_name())
        self.assertIn("_kp0", _spec(lds_k_outer=False, lds_k_pad=0).kernel_name())
        # tile_k < 16 defaults to no pad.
        self.assertNotIn(
            "_kp",
            _spec(
                lds_k_outer=False, tile_k=8, warp_tile_k=8, lds_k_pad=0, dtype="fp16"
            ).kernel_name(),
        )


def _loop_tree(kernel):
    """Nesting of the kernel's scf.for loops, as nested lists."""

    def walk(ops):
        out = []
        for op in ops:
            inner = [o for r in getattr(op, "regions", ()) or () for o in r.ops]
            if op.name == "scf.for":
                out.append(walk(inner))
            else:
                out.extend(walk(inner))
        return out

    return walk(kernel.body.ops)


def _staging(loop_op):
    keep = ("tile.buffer_load_vN", "tile.smem_store_vN")
    return [o.name for r in loop_op.regions for o in r.ops if o.name in keep]


class TestTapOuterK(unittest.TestCase):
    def test_default_off_and_name_tag(self):
        self.assertFalse(_spec().tap_outer_k)
        self.assertNotIn("tapk", _spec().kernel_name())
        name = _spec(tap_outer_k=True).kernel_name()
        self.assertTrue(name.endswith("_tapk"), name)
        self.assertTrue(
            _spec(lds_k_outer=False, lds_k_pad=16, tap_outer_k=True)
            .kernel_name()
            .endswith("_kp16_tapk")
        )

    def test_rejected_where_it_cannot_compute(self):
        cases = (
            ({"problem": _problem(K=96)}, "kpg % tile_k"),
            ({"problem": _problem(sH=2, sW=2)}, "stride 1"),
            ({"problem": _problem(dH=2, dW=2, pH=2, pW=2)}, "dilation 1"),
            ({"split_k": 2}, "split_k"),
            ({"uniform_k_decode": True}, "uniform_k_decode"),
        )
        for kw, needle in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _spec(tap_outer_k=True, **kw)
                ok, why = is_valid_dgrad_spec(spec, _ARCH)
                self.assertFalse(ok)
                self.assertIn(needle, why)
                with self.assertRaises(ValueError):
                    spec.validate()
        # kpg = 96 is fine with tile_k = 32; grouped, 1x1 and 5x5 qualify.
        for kw in (
            {"problem": _problem(K=96), "tile_k": 32},
            {"problem": _problem(C=128, K=256, groups=2)},
            {"problem": _problem(Y=1, X=1, pH=0, pW=0)},
            {"problem": _problem(Y=5, X=5, pH=2, pW=2)},
        ):
            with self.subTest(ok=str(kw)):
                ok, why = is_valid_dgrad_spec(_spec(tap_outer_k=True, **kw), _ARCH)
                self.assertTrue(ok, why)

    def test_emits_a_nested_tap_and_chunk_loop(self):
        k = build_implicit_gemm_conv_dgrad(_spec(tap_outer_k=True), arch=_ARCH)
        # One tap loop holding one chunk loop; no search, no other K loop.
        self.assertEqual(_loop_tree(k), [[[]]])
        tap = next(op for op in k.body.ops if op.name == "scf.for")
        chunk = next(o for r in tap.regions for o in r.ops if o.name == "scf.for")
        staging = _staging(chunk)
        # Every global read of the tile precedes its first LDS write.
        first_store = staging.index("tile.smem_store_vN")
        self.assertNotIn("tile.buffer_load_vN", staging[first_store:])

    def test_record_and_tap_decode_read_no_buffer(self):
        # The folded record plus the magic table: eight scalar loads at
        # constant offsets, as without the tap-outer loop.
        def loads(spec):
            n = 0

            def walk(ops):
                nonlocal n
                for op in ops:
                    n += op.name == "memref.global_load_typed"
                    for r in getattr(op, "regions", ()) or ():
                        walk(r.ops)

            walk(build_implicit_gemm_conv_dgrad(spec, arch=_ARCH).body.ops)
            return n

        self.assertEqual(loads(_spec(tap_outer_k=True)), 8)


def _identity(**kw):
    from benchmarks.common.kernel_cache import KernelIdentity

    base = {
        "arch": "gfx950",
        "direction": "dgrad",
        "algorithm": "implicit_gemm",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_d": "bf16",
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "wave_size": 64,
        "vector_size_a": 8,
        "vector_size_b": 8,
        "vector_size_c": 8,
        "lds_k_outer": True,
        "grouped": True,
        "stride_h": 1,
        "stride_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "max_sub_gemms": 1024,
    }
    base.update(kw)
    return KernelIdentity(**base)


class TestTapOuterKIdentity(unittest.TestCase):
    def test_off_keeps_the_hash_on_is_distinct(self):
        import dataclasses
        import hashlib
        import json

        off = _identity(uniform_k_decode=True)
        # The hash an identity had before the field existed.
        fields = dataclasses.asdict(off)
        for name in (
            "ws_replicas",
            "prefetch_addr",
            "patch",
            "tap_outer_k",
            "dy_halo",
            "dy_halo_setprio",
            "dy_halo_kouter_pad",
            "halo_max_wo",
        ):
            del fields[name]
        del fields["streamk"], fields["streamk_reduction"]
        legacy = hashlib.sha1(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(off.stable_hash(), legacy)
        on = _identity(tap_outer_k=True)
        self.assertNotEqual(on.stable_hash(), off.stable_hash())
        self.assertIn("tapk", on.label())
        type(on).from_label(on.label())  # round trip
        self.assertEqual(type(on).from_label(on.label()), on)

    def test_cache_rejects_tap_outer_on_unaligned_kpg(self):
        from benchmarks.common.kernel_cache import KernelCache

        cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
        tapk = _identity(tap_outer_k=True)
        aligned = _problem(C=128, K=256, groups=2)  # kpg 128
        self.assertTrue(cache.supports_problem(tapk, aligned)[0])
        for p in (
            _problem(C=192, K=192, groups=2),  # kpg 96
            _problem(K=32),  # kpg 32 < tile_k
        ):
            ok, why = cache.supports_problem(tapk, p)
            self.assertFalse(ok)
            self.assertIn("kpg % tile_k", why)
        # The same configuration without the loop serves them.
        flat = _identity(uniform_k_decode=True)
        self.assertTrue(cache.supports_problem(flat, _problem(K=96))[0])
        # Stride and dilation are pinned to 1.
        self.assertFalse(cache.supports_problem(tapk, _problem(sH=2, sW=2))[0])
        self.assertFalse(
            cache.supports_problem(tapk, _problem(dH=2, dW=2, pH=2, pW=2))[0]
        )

    def test_aot_grid_carries_it_on_stride1_only(self):
        from kernels.conv import aot_jobs as ks
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(_ARCH)
        geos = ks._geometry_list(target, "bf16")[::97]
        jobs = list(
            ks._dgrad_jobs(_ARCH, "bf16", 64, "mma", target, 1024, geometries=geos)
        )
        tapk = [j for j in jobs if j.identity.tap_outer_k]
        self.assertTrue(tapk)
        for j in tapk:
            i = j.identity
            self.assertEqual((i.stride_h, i.dilation_h), (1, 1))
            self.assertFalse(i.uniform_k_decode)
            self.assertTrue(j.spec_kwargs["tap_outer_k"])
            self.assertEqual(ks.job_for_identity(i).identity, i)
        self.assertFalse(
            any(j.identity.tap_outer_k and j.identity.stride_h == 0 for j in jobs)
        )
        self.assertFalse(any(j.identity.pipeline == "wavelet" for j in tapk))


def _all_ops(kernel):
    out = []

    def walk(ops):
        for op in ops:
            out.append(op.name)
            for r in getattr(op, "regions", ()) or ():
                walk(r.ops)

    walk(kernel.body.ops)
    return out


_XCD_GROUPED = {"C": 256, "K": 256, "groups": 4}


class TestXcdTileOrder(unittest.TestCase):
    """XCD-contiguous tile order of grouped stride-1 kernels (gfx950).

    Build time: the target, the knob, a grouped kernel, the folded record and
    no split-K. Run time, from the kernargs: groups > 1, Y*X > 1 or
    num_pid_n > 1, and at least one workgroup per XCD.
    """

    def _ops(self, arch=_ARCH, **kw):
        if arch != _ARCH:
            # gfx942 has no K-outer transpose read; same tile otherwise.
            kw.setdefault("lds_k_outer", False)
            kw.setdefault("warp_tile_k", 8)
            kw.setdefault("tile_k", 32)
        spec = _spec(**kw)
        return _all_ops(build_implicit_gemm_conv_dgrad(spec, arch=arch)), spec

    def test_build_time_rule(self):
        grouped = _problem(**_XCD_GROUPED)
        for arch in ("gfx950", "gfx942"):
            on, spec_on = self._ops(arch, problem=grouped)
            off, spec_off = self._ops(arch, problem=grouped, xcd_tile_order=False)
            with self.subTest(arch=arch):
                self.assertTrue(spec_on.xcd_tile_order)
                self.assertNotIn("noxcd", spec_on.kernel_name())
                self.assertTrue(spec_off.kernel_name().endswith("_noxcd"))
                xcds = dgrad_mod.xcd_tile_order_xcds(spec_on, arch)
                self.assertEqual(xcds, 8 if arch == "gfx950" else 0)
                self.assertEqual(dgrad_mod.xcd_tile_order_xcds(spec_off, arch), 0)
                if xcds:
                    # The helper's select plus the run-time switch.
                    self.assertEqual(
                        on.count("arith.select") - off.count("arith.select"), 2
                    )
                else:
                    self.assertEqual(on, off)

    def test_withheld_elsewhere(self):
        flat = _XCD_GROUPED
        cases = {
            "ungrouped": {"problem": _problem()},
            "strided": {"problem": _problem(sH=2, sW=2, **flat)},
            "split-K": {"problem": _problem(**flat), "split_k": 2},
        }
        for label, kw in cases.items():
            with self.subTest(case=label):
                spec = _spec(**kw)
                self.assertEqual(dgrad_mod.xcd_tile_order_xcds(spec, _ARCH), 0)
                on, _ = self._ops(**kw)
                off, _ = self._ops(xcd_tile_order=False, **kw)
                self.assertEqual(on, off)

    def test_emitted_order_is_the_model(self):
        # emit_xcd_tile_order, evaluated op by op, against the model, over
        # grids that do and do not take the remap; the (tile, group) map is
        # a bijection of the launch grid either way.
        from rocke.core.ir import I32, IRBuilder

        names = ("p_num_pid_m", "p_num_pid_n", "p_groups", "p_Y", "p_X")
        b = IRBuilder("xcd_order")
        params = {n: b.param(n, I32) for n in names}
        bx = b.block_id_x()
        tile, group = dgrad_mod.emit_xcd_tile_order(b, bx, params, num_xcds=8)

        def div(x, y):
            return int(x / y)

        binop = {
            "arith.add": lambda x, y: x + y,
            "arith.mul": lambda x, y: x * y,
            "arith.div": div,
            "arith.mod": lambda x, y: x - div(x, y) * y,
            "arith.smax": max,
            "arith.and": lambda x, y: x and y,
            "arith.or": lambda x, y: x or y,
        }
        cmp = {
            "lt": lambda x, y: x < y,
            "gt": lambda x, y: x > y,
            "ge": lambda x, y: x >= y,
        }

        def run(env, by):
            vals = {params[n].name: env[n] for n in names}
            vals[bx.name] = env["bx"]
            for op in b.kernel.body.ops:
                ins = [vals.get(v.name) for v in op.operands]
                if op.name == "arith.constant":
                    vals[op.result.name] = op.attrs["value"]
                elif op.name == "gpu.block_id" and op.attrs["axis"] == "y":
                    vals[op.result.name] = by
                elif op.name in binop:
                    vals[op.result.name] = binop[op.name](*ins)
                elif op.name == "arith.cmp":
                    vals[op.result.name] = cmp[op.attrs["pred"]](*ins)
                elif op.name == "arith.select":
                    vals[op.result.name] = ins[1] if ins[0] else ins[2]
            return vals[tile.name], vals[group.name]

        for m, n, g, taps in (
            (7, 2, 4, 9),  # remapped
            (9, 1, 3, 9),  # one N tile, several taps: remapped
            (6, 1, 4, 1),  # one N tile, one tap: launch order
            (3, 2, 1, 9),  # ungrouped launch of a grouped binary
            (1, 1, 5, 9),  # fewer workgroups than XCDs
            (33, 3, 2, 1),  # 1x1 filter, three N tiles: remapped
        ):
            seen = set()
            remapped = False
            for gy in range(g):
                for x in range(m * n):
                    env = {
                        "p_num_pid_m": m,
                        "p_num_pid_n": n,
                        "p_groups": g,
                        "p_Y": taps,
                        "p_X": 1,
                        "bx": x,
                    }
                    got = run(env, gy)
                    want = dgrad_mod.python_xcd_tile_order(
                        x, gy, num_pid_m=m, num_pid_n=n, groups=g, taps=taps
                    )
                    self.assertEqual(got, want, (m, n, g, taps, x, gy))
                    remapped |= got != (x, gy)
                    seen.add(got)
            with self.subTest(m=m, n=n, g=g, taps=taps):
                self.assertEqual(
                    seen, {(t, gg) for t in range(m * n) for gg in range(g)}
                )
                takes = g > 1 and (taps > 1 or n > 1) and m * n * g >= 8
                self.assertEqual(remapped, takes)

    def test_never_swept(self):
        from kernels.conv import aot_jobs as ks
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(_ARCH)
        geos = ks._geometry_list(target, "bf16")[::211]
        for job in ks._dgrad_jobs(
            _ARCH, "bf16", 64, "mma", target, 1024, geometries=geos
        ):
            self.assertNotIn("xcd_tile_order", job.spec_kwargs)


def _halo_spec(Y=3, Wi=14, **kw):
    base = {
        "problem": _problem(Wi=Wi, Y=Y, X=Y, pH=Y // 2, pW=Y // 2),
        "tap_outer_k": True,
        "dy_halo": 2,
        "halo_max_wo": 16,
    }
    base.update(kw)
    return _spec(**base)


class TestDyHalo(unittest.TestCase):
    """The dY halo mode of the tap-outer loop: knobs, rejections, LDS and the
    loop it emits."""

    def test_default_off_and_name_tags(self):
        spec = _spec()
        self.assertEqual(
            (spec.dy_halo, spec.dy_halo_setprio, spec.dy_halo_kouter_pad), (0, 0, 0)
        )
        self.assertEqual(spec.halo_max_wo, 0)
        self.assertNotIn("halo", spec.kernel_name())
        name = _halo_spec(
            tile_m=128, warp_m=4, warp_n=1, dy_halo_setprio=1, dy_halo_kouter_pad=32
        ).kernel_name()
        self.assertTrue(name.endswith("_tapk_halo2_hprio1_hkp32_hwo16"), name)
        self.assertTrue(
            _halo_spec(dy_halo=1, xcd_tile_order=False)
            .kernel_name()
            .endswith("_tapk_noxcd_halo1_hwo16")
        )

    def test_rejected_where_it_cannot_compute(self):
        cases = (
            ({"tap_outer_k": False}, "tap-outer K loop"),
            ({"problem": _problem(pH=0, pW=0)}, "Ho == Hi"),
            ({"problem": _problem(Y=1, X=1, pH=0, pW=0)}, "more than one tap"),
            ({"halo_max_wo": 0}, "halo_max_wo > 0"),
            ({"halo_max_wo": 8}, "Wo <= halo_max_wo"),
            ({"dy_halo_kouter_pad": 32, "lds_k_outer": False}, "lds_k_outer"),
            ({"dy_halo_kouter_pad": 12}, "multiple of 8"),
            ({"dy_halo": 3}, "dy_halo must be"),
            ({"dy_halo_setprio": 4}, "0..3"),
            ({"dy_halo": 0, "halo_max_wo": 16}, "halo_max_wo needs"),
            ({"dy_halo": 0, "halo_max_wo": 0, "dy_halo_setprio": 1}, "setprio"),
            ({"problem": _problem(K=96)}, "kpg % tile_k"),
            ({"problem": _problem(sH=2, sW=2)}, "stride 1"),
        )
        for kw, needle in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _halo_spec(**kw)
                ok, why = is_valid_dgrad_spec(spec, _ARCH)
                self.assertFalse(ok)
                self.assertIn(needle, why)
                with self.assertRaises(ValueError):
                    spec.validate()
        for kw in (
            {},
            {"dy_halo": 1},
            {"problem": _problem(Wi=16)},
            {"problem": _problem(Wi=1, Hi=1)},
            {"problem": _problem(C=128, K=256, groups=2)},
            {"problem": _problem(Y=3, X=5, pH=1, pW=2)},
            {"lds_k_outer": False},
            {"tile_m": 256, "warp_m": 4, "warp_n": 1, "dy_halo_kouter_pad": 32},
        ):
            with self.subTest(ok=str(kw)):
                ok, why = is_valid_dgrad_spec(_halo_spec(**kw), _ARCH)
                self.assertTrue(ok, why)

    def test_lds_is_sized_for_halo_max_wo_not_the_problem(self):
        for kw in (
            {},
            {"dy_halo": 1},
            {"tile_m": 128, "warp_m": 4, "warp_n": 1, "dy_halo_kouter_pad": 32},
            {"Y": 5, "halo_max_wo": 64},
            {"lds_k_outer": False},
        ):
            with self.subTest(**kw):
                spec = _halo_spec(**kw)
                k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
                ab, c = _smem_bytes(k, spec)
                self.assertEqual(dgrad_lds_bytes(spec), max(ab, c))
                # The halo tile: tile_m + (Y-1)*halo_max_wo + (X-1) rows,
                # whole loader passes, plus the zero row.
                p = spec.problem
                need = spec.tile_m + (p.Y - 1) * spec.halo_max_wo + (p.X - 1)
                rows = spec.dy_halo_lds_rows
                self.assertGreater(rows, need)
                self.assertEqual(((rows - 1) * spec.tile_k) % (spec.block_size * 8), 0)
                narrow = _halo_spec(**{**kw, "Wi": 3})
                self.assertEqual(dgrad_lds_bytes(narrow), dgrad_lds_bytes(spec))
        wide = _halo_spec(halo_max_wo=128)
        self.assertGreater(dgrad_lds_bytes(wide), dgrad_lds_bytes(_halo_spec()))

    def test_emits_one_chunk_loop_with_unrolled_taps(self):
        for Y, mode in ((3, 2), (5, 1)):
            with self.subTest(Y=Y, dy_halo=mode):
                spec = _halo_spec(Y=Y, dy_halo=mode)
                k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
                # One chunk loop, no tap loop: the taps are unrolled in it.
                self.assertEqual(_loop_tree(k), [[]])
                ops = _all_ops(k)
                per_tap = spec.mfmas_per_warp_m * spec.mfmas_per_warp_n
                per_tap *= spec.k_atoms_per_tile_k
                self.assertEqual(ops.count("tile.mma"), Y * Y * per_tap)
                # dy_halo=2 pins each of taps 1..Y*X-1's B prefetch with two
                # scheduling barriers; dy_halo=1 has none.
                self.assertEqual(
                    ops.count("tile.sched_barrier"), 2 * (Y * Y - 1) * (mode == 2)
                )

    def test_identity_hash_and_label(self):
        plain = _identity(tap_outer_k=True)
        halo = _identity(
            tap_outer_k=True,
            filter_h=3,
            filter_w=3,
            dy_halo=2,
            dy_halo_setprio=1,
            dy_halo_kouter_pad=32,
            halo_max_wo=64,
        )
        self.assertNotEqual(halo.stable_hash(), plain.stable_hash())
        label = halo.label()
        for tag in ("halo2", "hprio1", "hkp32", "hwo64", "f3x3", "tapk"):
            self.assertIn(tag, label.split("-"))
        self.assertEqual(type(halo).from_label(label), halo)
        self.assertNotIn("halo", plain.label())

    def test_aot_grid_halo_slice(self):
        from kernels.conv import aot_jobs as ks
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(_ARCH)
        geos = [
            g
            for g in ks._geometry_list(target, "bf16")
            if g[:6] in ks.CACHE_DGRAD_HALO_TILES and g[6] in ("mem", "compv4")
        ]
        self.assertTrue(geos)
        jobs = list(
            ks._dgrad_jobs(_ARCH, "bf16", 64, "mma", target, 1024, geometries=geos)
        )
        halo = [j for j in jobs if j.identity.dy_halo]
        self.assertTrue(halo)
        for j in halo:
            i = j.identity
            self.assertEqual(i.pipeline, "mem")
            self.assertTrue(i.tap_outer_k and i.lds_k_outer and i.grouped)
            self.assertEqual((i.stride_h, i.dilation_h), (1, 1))
            self.assertIn(i.filter_h, ks.CACHE_DGRAD_HALO_FILTERS)
            self.assertEqual(i.filter_h, i.filter_w)
            self.assertIn(i.halo_max_wo, ks.CACHE_DGRAD_HALO_MAX_WO)
            p = ks._probe_problem("dgrad", j.caps)
            self.assertEqual(
                (p.Y, p.X, p.Wo, p.Ho, p.Wi),
                (i.filter_h,) * 2
                + (
                    i.halo_max_wo,
                    p.Hi,
                    i.halo_max_wo,
                ),
            )
            # The 2x2 tile carries neither setprio nor the wider pad.
            if i.warp_n == 2:
                self.assertEqual((i.dy_halo_setprio, i.dy_halo_kouter_pad), (0, 0))
        # Not built where the transpose read (K-outer) is unavailable.
        t942 = ArchTarget.from_gfx("gfx942")
        geos942 = ks._geometry_list(t942, "bf16")[::37]
        self.assertFalse(
            any(
                j.identity.dy_halo
                for j in ks._dgrad_jobs(
                    "gfx942", "bf16", 64, "mma", t942, 1024, geometries=geos942
                )
            )
        )
        # Every valid halo job rebuilds from its identity.
        valid = [j for j in halo if ks._spec_is_valid(j, _ARCH, "bf16")]
        self.assertTrue(valid)
        for j in valid[:: max(1, len(valid) // 12)]:
            self.assertEqual(ks.job_for_identity(j.identity).identity, j.identity)


if __name__ == "__main__":
    unittest.main()
