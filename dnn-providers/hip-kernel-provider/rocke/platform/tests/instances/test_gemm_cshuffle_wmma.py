# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU regression tests for the cshuffle epilogue on the WMMA path.

The WMMA path rejected every non-``default`` epilogue outright, so cshuffle
was unreachable on gfx1250 for *any* dtype -- not an fp8 gap. Enabling it
needs three things to hold, and each one fails silently rather than loudly:

1. The validity gate admits the combination at all.
2. The accumulator -> (row, col) scatter follows the **wave32** map. MFMA and
   WMMA distribute a warp tile across lanes differently, so reusing the MFMA
   scatter would stage every element at the wrong LDS cell -- a kernel that
   builds, lowers, and returns a transposed-looking result.
3. The staging tile is typed by the **C** dtype, not the A/B dtype. For fp8
   those differ (fp8 in, bf16 out); upstream conflates them in one
   ``storage_dtype``, which is correct only while A/B and C agree.

The assertions are differential against the ``default`` epilogue and against
the A/B dtype, because absolute counts here are not discriminating: the two
epilogues agree on most of the kernel, and both emit plenty of unrelated
stores. Each is mutation-checked -- reverting the corresponding part of the
port fails the matching test.
"""

from __future__ import annotations

import unittest
from collections import Counter

ARCH = "gfx1250"


def _spec(*, dtype: str = "bf16", epilogue: str = "default", pad: bool = False):
    """A minimal valid gfx1250 WMMA GEMM spec.

    ``pad`` drives ``pad_m``/``pad_n`` together; see
    :meth:`TestCshuffleWmma.test_pad_n_forfeits_the_wide_store` for why the
    default here is off.
    """
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    is_8bit = dtype.startswith(("fp8", "bf8"))
    warp_tile_k = 64 if is_8bit else 32
    return UniversalGemmSpec(
        name="cshuffle_wmma_probe",
        tile=TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=warp_tile_k,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=warp_tile_k,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue=epilogue,
            pad_m=pad,
            pad_n=pad,
            pad_k=True,
        ),
        data=DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c="bf16" if is_8bit else dtype,
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
    )


def _lower(spec) -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import (
        build_universal_gemm,
        is_valid_spec,
    )

    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        raise AssertionError(f"spec rejected: {why}")
    ir = build_universal_gemm(spec, arch=ARCH)
    return lower_kernel_to_llvm(ir, arch=ARCH)


def _store_shapes(ll: str) -> Counter:
    """Count ``store`` instructions keyed by ``"<addrspace> <type>"``.

    Address space separates the LDS staging traffic cshuffle adds from the
    global traffic it is meant to widen; the type carries the dtype question.
    """
    counts: Counter = Counter()
    for line in ll.splitlines():
        text = line.strip()
        if not text.startswith("store "):
            continue
        head = text.split()[1]
        ty = (
            text[text.index("<") : text.index(">") + 1]
            if head.startswith("<")
            else head
        )
        if "addrspace(3)" in text:
            space = "lds"
        elif "addrspace(1)" in text:
            space = "global"
        else:
            space = "other"
        counts[f"{space} {ty}"] += 1
    return counts


class TestCshuffleWmma(unittest.TestCase):
    def test_wmma_path_admits_the_cshuffle_epilogue(self):
        """The gate used to reject this for every dtype, not just fp8."""
        from rocke.instances.common.gemm_universal import is_valid_spec

        for dtype in ("bf16", "fp8e4m3"):
            with self.subTest(dtype=dtype):
                ok, why = is_valid_spec(
                    _spec(dtype=dtype, epilogue="cshuffle"), arch=ARCH
                )
                self.assertTrue(ok, f"WMMA + cshuffle rejected: {why}")

    def test_cshuffle_stays_gated_on_the_other_wmma_targets(self):
        """Admitting it everywhere was a regression, caught by the gfx1151 suite.

        The emitter is arch-neutral, so the gate is about *coverage*: the
        parity configs and the on-silicon verification are gfx1250 only.
        Removing the blanket 'default epilogue only' gate without replacing it
        with this arch test shipped an epilogue nothing had ever run there.
        """
        from dataclasses import replace

        from rocke.instances.common.gemm_universal import is_valid_spec

        # gfx1151 / gfx1201 carry the 16x16x16 atom, not gfx1250's 16x16x32,
        # so the spec has to match or it is rejected for the atom instead and
        # the assertion proves nothing.
        base = _spec(dtype="fp16", epilogue="default")
        base = replace(
            base,
            tile=replace(base.tile, tile_k=32, warp_tile_k=16),
        )
        for arch in ("gfx1151", "gfx1201"):
            with self.subTest(arch=arch):
                ok, why = is_valid_spec(base, arch=arch)
                self.assertTrue(ok, f"baseline spec invalid on {arch}: {why}")
                shuf = replace(base, trait=replace(base.trait, epilogue="cshuffle"))
                ok, why = is_valid_spec(shuf, arch=arch)
                self.assertFalse(ok, f"cshuffle must stay gated on {arch}")
                self.assertIn("epilogue", why)

    def test_cshuffle_widens_the_global_c_stores(self):
        """The point of the LDS round-trip: scalar global stores become vectors.

        Differential against ``default``, which has no staging pass. The wave32
        accumulator holds a lane's output slots one *column* apart in C, so a
        direct store is one scalar store per slot; staging through LDS lets a
        flat thread map read back contiguous runs. If the staging landed at the
        wrong cells the readback would still vectorize, which is why the
        companion dtype and parity assertions exist -- this one only pins that
        the round-trip is actually buying the store width.
        """
        for dtype in ("bf16", "fp8e4m3"):
            with self.subTest(dtype=dtype):
                base = _store_shapes(_lower(_spec(dtype=dtype, epilogue="default")))
                shuf = _store_shapes(_lower(_spec(dtype=dtype, epilogue="cshuffle")))

                scalar_global = [
                    k for k in base if k.startswith("global") and "<" not in k
                ]
                self.assertTrue(
                    scalar_global,
                    "default epilogue is expected to store C scalar-at-a-time",
                )
                for key in scalar_global:
                    self.assertLess(
                        shuf[key],
                        base[key],
                        f"cshuffle did not reduce the {key!r} store count; the "
                        f"LDS round-trip is being paid for with no store-width win",
                    )
                self.assertTrue(
                    any(k.startswith("global") and "<" in k for k in shuf),
                    "cshuffle emitted no vector global store",
                )

    def test_fp8_stages_at_the_c_dtype_not_the_ab_dtype(self):
        """The one genuinely fp8-specific part of the port.

        Upstream types the staging tile from a single ``storage_dtype``, which
        holds only while A/B and C agree -- fp8 has an 8-bit A/B and a 16-bit
        C. (Today that mistake happens to raise in ``vec_cast_f32_to``, which
        has no fp8 target; the assertion is written against the staged dtype
        rather than the raise so it still holds for a narrow C the caster does
        accept.) Asserted as the *absence* of narrow staging traffic rather
        than the presence of wide traffic, because the A/B mainloop
        legitimately stores ``<16 x i8>`` to LDS and would mask a presence
        check.
        """
        shuf = _store_shapes(_lower(_spec(dtype="fp8e4m3", epilogue="cshuffle")))
        base = _store_shapes(_lower(_spec(dtype="fp8e4m3", epilogue="default")))

        added = {k: shuf[k] - base.get(k, 0) for k in shuf if shuf[k] > base.get(k, 0)}
        lds_added = {k: v for k, v in added.items() if k.startswith("lds")}
        self.assertTrue(lds_added, "cshuffle added no LDS staging stores")
        for key in lds_added:
            self.assertNotIn(
                "i8",
                key,
                f"fp8 cshuffle stages C as {key!r}; the staging tile must be "
                f"typed by dtype_c (bf16), not by dtype_a",
            )

    def test_scatter_dispatch_on_op_family_is_load_bearing(self):
        """Guards the ``op.family`` branch in the staging scatter.

        MFMA and WMMA distribute a warp tile across lanes differently, so a
        WMMA kernel routed through the MFMA scatter stages at the wrong cells
        -- it still builds and lowers, which is why nothing else catches it.
        Proven by substitution: redirect the WMMA scatter to the MFMA one and
        the kernel must stop being the same kernel. If it raises instead, the
        dispatch is equally load-bearing and the test passes on that.
        """
        from unittest import mock

        from rocke.instances.common.gemm_universal import _emit_mfma_acc_scatter

        spec = _spec(dtype="bf16", epilogue="cshuffle")
        good = _lower(spec)

        def _as_mfma(
            b, spec_, _op, lane, accs, m_off, n_off, c_per_lane, dtype, per_cell
        ):
            # Same arguments minus ``op``, which the MFMA scatter derives from
            # the tile instead of the atom's c_layout().
            return _emit_mfma_acc_scatter(
                b,
                spec_,
                lane,
                accs,
                m_off,
                n_off,
                c_per_lane,
                dtype,
                per_cell,
                n_base_first=True,
            )

        with mock.patch(
            "rocke.instances.common.gemm_universal._emit_wmma_acc_scatter",
            _as_mfma,
        ):
            try:
                swapped = _lower(spec)
            except Exception:
                return  # routing WMMA through the MFMA scatter is not even valid

        self.assertNotEqual(
            good,
            swapped,
            "the MFMA scatter produces byte-identical IR on a WMMA spec, so "
            "the op.family dispatch in the cshuffle staging step is not "
            "load-bearing and could regress undetected",
        )

    def test_pad_n_forfeits_the_wide_store(self):
        """``pad_n`` decomposes the wide store back into guarded scalars.

        Not a defect -- a tail column needs per-element bounds -- but it means
        a cshuffle kernel built with ``pad_n`` on pays the LDS round-trip and
        buys nothing, so it is strictly worse than ``default``. Sweep harnesses
        must set ``pad_n`` from the shape rather than pinning it on, or the
        epilogue comparison is measuring a kernel no one would ship.
        """
        padded = _store_shapes(_lower(_spec(epilogue="cshuffle", pad=True)))
        unpadded = _store_shapes(_lower(_spec(epilogue="cshuffle", pad=False)))

        self.assertTrue(
            any(k.startswith("global") and "<" in k for k in unpadded),
            "cshuffle without padding should emit a vector global store",
        )
        self.assertFalse(
            any(k.startswith("global") and "<" in k for k in padded),
            "pad_n is expected to force element-granular guarded stores",
        )


if __name__ == "__main__":
    unittest.main()
