# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the optional fused conv1d / gated-RMSNorm GDN/KDA decode modes.

Covers the spec contract (fields, kernel-name tags, admission rules) and the
conditional kernel ABI. On-device numerics live in
``test_gdn_decode_fused_gfx950_numeric.py``.
"""

from __future__ import annotations

import dataclasses as dc
import unittest

from kernels.gfx950.gdn_decode import (
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_pointer_alignment,
    gdn_decode_signature,
    is_valid_spec,
)

ARCH = "gfx950"


def _fused(**kw) -> GdnDecodeSpec:
    """A BPV=1, Hk == Hv KDA spec; flags off unless ``kw`` turns them on."""
    base = dict(
        num_k_heads=16,
        num_v_heads=16,
        gate_kind="kda",
        num_warps=4,
        warp_threads_k=16,
        blocks_per_v_dim=1,
    )
    base.update(kw)
    return GdnDecodeSpec(**base)


class TestFusedSpec(unittest.TestCase):
    def test_default_name_unchanged(self):
        self.assertEqual(
            GdnDecodeSpec().kernel_name(),
            "rocke_gdn_decode_bf16_kh16_vh32_dk128_dv128_w2k16b8_l2",
        )

    def test_name_tags_only_when_on(self):
        s = _fused()
        self.assertFalse(s.kernel_name().endswith(("_cv", "_rn")))
        self.assertTrue(dc.replace(s, fuse_conv=True).kernel_name().endswith("_l2_cv"))
        self.assertTrue(
            dc.replace(s, fuse_out_norm=True).kernel_name().endswith("_l2_rn")
        )
        self.assertTrue(
            dc.replace(s, fuse_conv=True, fuse_out_norm=True)
            .kernel_name()
            .endswith("_l2_cv_rn")
        )

    def test_fused_requires_bpv1(self):
        for flag in ("fuse_conv", "fuse_out_norm"):
            with self.subTest(flag=flag):
                ok, why = is_valid_spec(
                    _fused(blocks_per_v_dim=4, **{flag: True}), arch=ARCH
                )
                self.assertFalse(ok)
                self.assertIn("blocks_per_v_dim", why)
                self.assertTrue(why.startswith("NOT_YET_IMPLEMENTED:"), why)

    def test_fused_rejects_simple(self):
        ok, why = is_valid_spec(_fused(simple=True, fuse_out_norm=True), arch=ARCH)
        self.assertFalse(ok)
        self.assertIn("simple", why)
        self.assertTrue(why.startswith("NOT_YET_IMPLEMENTED:"), why)

    def test_conv_requires_equal_heads(self):
        ok, why = is_valid_spec(
            _fused(num_k_heads=8, num_v_heads=16, fuse_conv=True), arch=ARCH
        )
        self.assertFalse(ok)
        self.assertIn("num_k_heads == num_v_heads", why)
        self.assertTrue(why.startswith("NOT_YET_IMPLEMENTED:"), why)

    def test_abi_flags_must_be_real_bools(self):
        """A truthy non-bool must be rejected, not read as True: the string
        "false" would otherwise switch on the fused ABI."""
        for flag in (
            "fuse_conv",
            "fuse_out_norm",
            "use_qk_l2norm",
            "simple",
            "fuse_gate",
        ):
            for bad in ("false", 1, 0, None):
                with self.subTest(flag=flag, bad=bad):
                    ok, why = is_valid_spec(_fused(**{flag: bad}), arch=ARCH)
                    self.assertFalse(ok)
                    self.assertIn(f"{flag} must be a bool", why)

    def test_integer_fields_must_be_integers(self):
        for field, bad in (
            ("num_warps", 4.0),
            ("head_k_dim", "128"),
            ("blocks_per_v_dim", True),
            ("num_k_heads", None),
        ):
            with self.subTest(field=field, bad=bad):
                ok, why = is_valid_spec(_fused(**{field: bad}), arch=ARCH)
                self.assertFalse(ok)
                self.assertIn(f"{field} must be an integer", why)

    def test_norm_allows_gqa(self):
        ok, why = is_valid_spec(
            _fused(num_k_heads=8, num_v_heads=16, gate_kind="gdn", fuse_out_norm=True),
            arch=ARCH,
        )
        self.assertTrue(ok, why)

    def test_fused_bpv1_valid(self):
        for gate in ("gdn", "kda"):
            with self.subTest(gate=gate):
                ok, why = is_valid_spec(
                    _fused(gate_kind=gate, fuse_conv=True, fuse_out_norm=True),
                    arch=ARCH,
                )
                self.assertTrue(ok, why)


class TestFusedSignature(unittest.TestCase):
    @staticmethod
    def names(spec):
        return [p["name"] for p in gdn_decode_signature(spec)]

    def test_unfused_unchanged(self):
        self.assertEqual(
            self.names(GdnDecodeSpec()),
            [
                "query",
                "key",
                "value",
                "a",
                "b",
                "dt_bias",
                "A_log",
                "read_indices",
                "write_indices",
                "state",
                "out",
                "batch_size",
            ],
        )

    def test_conv_and_norm(self):
        s = _fused(fuse_conv=True, fuse_out_norm=True)
        self.assertEqual(
            self.names(s),
            [
                "mixed_qkv",
                "a",
                "b",
                "dt_bias",
                "A_log",
                "read_indices",
                "write_indices",
                "state",
                "out",
                "conv_state",
                "conv_weight",
                "out_gate",
                "norm_weight",
                "batch_size",
                "qkv_stride",
                "og_stride",
                "norm_eps",
            ],
        )

    def test_norm_only(self):
        n = self.names(_fused(fuse_out_norm=True))
        self.assertEqual(n[:3], ["query", "key", "value"])
        self.assertNotIn("conv_state", n)
        self.assertEqual(n[-3:], ["batch_size", "og_stride", "norm_eps"])


def _scratch_bytes(test: unittest.TestCase, spec: GdnDecodeSpec) -> int:
    """Compile through comgr (no GPU) and return the spilled scratch bytes."""
    import tempfile
    from pathlib import Path

    try:
        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel
    except Exception as e:  # pragma: no cover - env-dependent
        test.skipTest(f"comgr toolchain unavailable: {e}")
    try:
        art = compile_kernel(
            build_gdn_decode(spec, arch=ARCH), arch=ARCH, capture_ir_text=False
        )
    except ImportError as e:  # pragma: no cover - env-dependent
        test.skipTest(f"comgr toolchain unavailable: {e}")
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(bytes(art.hsaco))
        fh.flush()
        try:
            scratch = analyze_hsaco(Path(fh.name)).resources.scratch_bytes
        except (FileNotFoundError, RuntimeError) as e:  # pragma: no cover
            test.skipTest(f"HSACO introspection tool unavailable: {e}")
    if scratch is None:  # pragma: no cover - metadata shape drift
        test.skipTest("could not parse the scratch size from the HSACO")
    return scratch


_FLAGS = ((True, False), (False, True), (True, True))


class TestFusedEmission(unittest.TestCase):
    def test_params_match_signature(self):
        for gate in ("gdn", "kda"):
            for conv, norm in _FLAGS:
                for nw in (1, 4):
                    s = _fused(
                        gate_kind=gate, num_warps=nw, fuse_conv=conv, fuse_out_norm=norm
                    )
                    with self.subTest(name=s.kernel_name()):
                        k = build_gdn_decode(s, arch=ARCH)
                        self.assertEqual(
                            [p.name for p in k.params],
                            [e["name"] for e in gdn_decode_signature(s)],
                        )

    def test_lowers_and_compiles_without_scratch(self):
        """Every fused variant lowers and compiles without spilling, including
        one wave with fuse_conv, which owns all DV rows plus their taps and
        weights (0 B on gfx950 with ROCm 7.1 and ROCm 7.13)."""
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for gate in ("gdn", "kda"):
            for st in ("bf16", "f32"):
                for conv, norm in _FLAGS:
                    for nw in (1, 2, 4, 8):
                        s = _fused(
                            gate_kind=gate,
                            state_dtype=st,
                            num_warps=nw,
                            fuse_conv=conv,
                            fuse_out_norm=norm,
                        )
                        with self.subTest(name=s.kernel_name()):
                            ll = _lower_kernel_to_llvm_python(
                                build_gdn_decode(s, arch=ARCH),
                                arch=ARCH,
                                llvm_flavor="llvm20",
                            )
                            self.assertIn("define amdgpu_kernel", ll)
                            self.assertEqual(_scratch_bytes(self, s), 0)

    def test_unfused_kernel_has_no_lds_or_barrier(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        ll = _lower_kernel_to_llvm_python(
            build_gdn_decode(_fused(), arch=ARCH), arch=ARCH, llvm_flavor="llvm20"
        )
        self.assertNotIn("addrspace(3)", ll)
        self.assertNotIn("s.barrier", ll)


def _flat_ops(kernel):
    """Every op of ``kernel`` in emission order (regions inline)."""
    out = []

    def walk(ops):
        for op in ops:
            out.append(op)
            for region in op.regions:
                walk(region.ops)

    walk(kernel.body.ops)
    return out


_GLOBAL_MEM = {
    "memref.global_load_vN",
    "memref.global_store_vN",
    "memref.global_load_typed",
    "memref.global_store_typed",
}
_ELEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4}


def _root_param(value) -> str:
    """The kernel param a global pointer was derived from."""
    while value.op is not None and value.op.name == "tile.global_ptr_add":
        value = value.op.operands[0]
    assert value.op is None, f"pointer {value} is not derived from a param"
    return value.name.lstrip("%")


def _specs_for_ir_checks():
    """Both emitters, both gate kinds, every state width and fusion mode."""
    for gate in ("gdn", "kda"):
        for st in ("bf16", "f32"):
            yield dc.replace(GdnDecodeSpec(), gate_kind=gate, state_dtype=st)
            yield dc.replace(
                GdnDecodeSpec(), gate_kind=gate, state_dtype=st, simple=True
            )
            for dtype in ("bf16", "f16"):
                for conv, norm in _FLAGS:
                    for nw in (1, 4):
                        yield _fused(
                            gate_kind=gate,
                            dtype=dtype,
                            state_dtype=st,
                            num_warps=nw,
                            fuse_conv=conv,
                            fuse_out_norm=norm,
                        )


class TestPointerAlignment(unittest.TestCase):
    """Every alignment the emitted code claims is one the host guarantees.

    A vector access claims ``align N`` and a param declares ``align N``; LLVM
    believes both and nothing on the device checks them. The host checks
    each base against ``gdn_decode_pointer_alignment``; any other pointer is
    only naturally aligned (to its element). An access that claims more than
    that -- e.g. an f32 vec8 load at its 32 B payload size on a 16 B param --
    fails here.
    """

    def test_claims_are_backed(self):
        for spec in _specs_for_ir_checks():
            kernel = build_gdn_decode(spec, arch=ARCH)
            guaranteed = gdn_decode_pointer_alignment(spec)
            with self.subTest(name=spec.kernel_name()):
                for p in kernel.params:
                    if "align" in p.attrs:
                        self.assertLessEqual(
                            p.attrs["align"], guaranteed.get(p.name, 0), p.name
                        )
                for op in _flat_ops(kernel):
                    if op.name not in _GLOBAL_MEM:
                        continue
                    name = _root_param(op.operands[0])
                    natural = _ELEM_BYTES[op.attrs["elem_type"]]
                    self.assertLessEqual(
                        op.attrs["align"],
                        max(guaranteed.get(name, 0), natural),
                        f"{op.name} on {name}",
                    )


class TestFusedBarrier(unittest.TestCase):
    """The in-place conv-tap shift is ordered after every wave's tap reads.

    Several waves read the same q/k taps that wave 0 later overwrites in the
    same slot (read slot == write slot is legal). On silicon the race shows
    only with many waves (test_gdn_decode_fused_gfx950_numeric.py), so the
    order is also pinned in the IR: some workgroup barrier must sit after the
    last conv-state load and before the first conv-state store whenever the
    workgroup has more than one wave.
    """

    @staticmethod
    def _positions(spec):
        ops = _flat_ops(build_gdn_decode(spec, arch=ARCH))
        loads, stores, syncs = [], [], []
        for i, op in enumerate(ops):
            if op.name == "tile.sync":
                syncs.append(i)
            elif op.name in _GLOBAL_MEM and _root_param(op.operands[0]) == (
                "conv_state"
            ):
                (loads if "load" in op.name else stores).append(i)
        return loads, stores, syncs

    def test_barrier_between_tap_reads_and_tap_writes(self):
        for gate in ("gdn", "kda"):
            for norm in (False, True):
                for nw in (2, 4, 8):
                    spec = _fused(
                        gate_kind=gate, num_warps=nw, fuse_conv=True, fuse_out_norm=norm
                    )
                    with self.subTest(name=spec.kernel_name()):
                        loads, stores, syncs = self._positions(spec)
                        self.assertTrue(loads and stores)
                        self.assertTrue(
                            any(max(loads) < s < min(stores) for s in syncs),
                            f"no barrier between tap reads (last op {max(loads)}) "
                            f"and tap writes (first op {min(stores)}): {syncs}",
                        )

    def test_one_wave_needs_no_barrier(self):
        for norm in (False, True):
            spec = _fused(num_warps=1, fuse_conv=True, fuse_out_norm=norm)
            with self.subTest(name=spec.kernel_name()):
                self.assertEqual(self._positions(spec)[2], [])


class TestFusedReference(unittest.TestCase):
    """Host-side contract of the fused inputs and the fp32 reference (CPU)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - env-dependent
            raise unittest.SkipTest("torch unavailable")

    @staticmethod
    def spec(**kw):
        return _fused(num_k_heads=2, num_v_heads=2, **kw)

    def test_inputs_present(self):
        from builders.gfx950.gdn.gdn_decode import make_inputs
        from kernels.gfx950.gdn_decode import CONV_WIDTH

        s = self.spec(fuse_conv=True, fuse_out_norm=True)
        inp = make_inputs(s, 3, device="cpu")
        cd = 2 * 2 * 128 + 2 * 128
        self.assertEqual(s.conv_dim, cd)
        self.assertEqual(tuple(inp["mixed_qkv"].shape), (3, cd))
        self.assertEqual(inp["qkv_stride"], cd)
        # mixed_qkv replaces query/key/value; the fixture follows the contract.
        for name in ("query", "key", "value"):
            self.assertNotIn(name, inp)
        self.assertEqual(tuple(inp["conv_state"].shape), (7, cd, CONV_WIDTH - 1))
        self.assertEqual(tuple(inp["conv_weight"].shape), (cd, CONV_WIDTH))
        self.assertEqual(tuple(inp["out_gate"].shape), (3, 2 * 128))
        self.assertEqual(tuple(inp["norm_weight"].shape), (128,))
        self.assertEqual(inp["og_stride"], 2 * 128)
        self.assertEqual(inp["norm_eps"], 1e-6)

    def test_unfused_inputs_unchanged(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs

        plain = make_inputs(self.spec(), 3, device="cpu")
        self.assertNotIn("mixed_qkv", plain)
        self.assertNotIn("out_gate", plain)
        # the norm draws come after every unfused draw: shared inputs match
        normed = make_inputs(self.spec(fuse_out_norm=True), 3, device="cpu")
        for name in ("query", "key", "value", "a", "b", "state"):
            self.assertTrue(torch.equal(plain[name], normed[name]), name)

    def test_conv_shift(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_conv_state_after

        s = self.spec(fuse_conv=True)
        inp = make_inputs(s, 3, device="cpu")
        after = ref_conv_state_after(s, inp)
        taps = inp["conv_state"].float()[inp["read_indices"].long()]
        self.assertTrue(torch.equal(after[..., 0], taps[..., 1]))
        self.assertTrue(torch.equal(after[..., 1], taps[..., 2]))
        self.assertTrue(torch.equal(after[..., 2], inp["mixed_qkv"].float()))

    def test_conv_changes_output(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32

        s = self.spec(fuse_conv=True)
        inp = make_inputs(s, 3, device="cpu")
        o_fused, _ = ref_fp32(s, inp)
        # The unfused twin reads the same packed row, unconvolved.
        hk, dk, dv = s.num_k_heads, s.head_k_dim, s.head_v_dim
        row = inp["mixed_qkv"]
        plain = dict(inp)
        plain["query"] = row[:, : hk * dk].reshape(3, 1, hk, dk)
        plain["key"] = row[:, hk * dk : 2 * hk * dk].reshape(3, 1, hk, dk)
        plain["value"] = row[:, 2 * hk * dk :].reshape(3, 1, s.num_v_heads, dv)
        o_plain, _ = ref_fp32(dc.replace(s, fuse_conv=False), plain)
        self.assertFalse(torch.allclose(o_fused, o_plain))

    def test_norm_gives_unit_rms(self):
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32

        s = self.spec(fuse_out_norm=True)
        inp = make_inputs(s, 3, device="cpu")
        inp["out_gate"] = torch.full_like(inp["out_gate"], 30.0)  # sigmoid -> 1
        inp["norm_weight"] = torch.ones_like(inp["norm_weight"])
        # The raw decode output's RMS is ~1e-4..1e-2 here, so eps must be far
        # below mean(o^2) for the normalised RMS to reach 1.
        inp["norm_eps"] = 1e-12
        o, _ = ref_fp32(s, inp)
        rms = o.float().pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_norm_gate_activation_follows_gate_kind(self):
        """GDN gates its norm with SiLU (Qwen3-Next), KDA with sigmoid (Kimi
        Linear). A constant gate g scales the normalised output by act(g):
        silu(0) = 0 zeroes the GDN output, and the KDA output at g = 3 is
        sigmoid(3) / sigmoid(0) times its output at g = 0."""
        import torch

        from builders.gfx950.gdn.gdn_decode import make_inputs, ref_fp32

        outs = {}
        for gate in ("gdn", "kda"):
            s = self.spec(gate_kind=gate, fuse_out_norm=True)
            inp = make_inputs(s, 3, device="cpu")
            for g in (0.0, 3.0):
                inp["out_gate"] = torch.full_like(inp["out_gate"], g)
                outs[gate, g] = ref_fp32(s, inp)[0]
        self.assertEqual(outs["gdn", 0.0].abs().max().item(), 0.0)
        self.assertGreater(outs["gdn", 3.0].abs().max().item(), 0.0)
        ratio = outs["kda", 3.0] / outs["kda", 0.0]
        want = torch.sigmoid(torch.tensor(3.0)) / 0.5
        self.assertTrue(torch.allclose(ratio, want.expand_as(ratio)))

    def test_out_error_accepts_bf16_rounding_of_the_reference(self):
        """The fused output is O(1..7): bf16 rounding of the exact reference
        must pass, in every fused mode."""
        from builders.gfx950.gdn.gdn_decode import TOL, make_inputs, out_error, ref_fp32

        for gate in ("gdn", "kda"):
            for conv, norm in _FLAGS:
                s = self.spec(gate_kind=gate, fuse_conv=conv, fuse_out_norm=norm)
                with self.subTest(name=s.kernel_name()):
                    ref, _ = ref_fp32(s, make_inputs(s, 16, device="cpu"))
                    self.assertLess(out_error(s, ref.bfloat16(), ref), TOL)

    def test_out_error_rejects_plausible_norm_bugs(self):
        """The scale-aware metric still catches real defects: a dropped output
        gate and a norm taken over the wrong length."""
        import torch

        from builders.gfx950.gdn.gdn_decode import TOL, make_inputs, out_error, ref_fp32

        s = self.spec(fuse_conv=True, fuse_out_norm=True)
        inp = make_inputs(s, 16, device="cpu")
        ref, _ = ref_fp32(s, inp)
        no_gate = ref / torch.sigmoid(inp["out_gate"].float().reshape(ref.shape))
        self.assertGreater(out_error(s, no_gate, ref), TOL)
        wrong_len = ref * (2.0**0.5)  # rsqrt(sum/(DV/2)) instead of rsqrt(sum/DV)
        self.assertGreater(out_error(s, wrong_len, ref), TOL)

    def test_prepare_takes_fused_inputs_without_query(self):
        """The fused fixture carries no query/key/value (mixed_qkv replaces
        them), and prepare() must not look for them."""
        from builders.gfx950.gdn.gdn_decode import make_inputs, prepare

        s = self.spec(fuse_conv=True, fuse_out_norm=True)
        values, _ = prepare(s, make_inputs(s, 3, device="cpu"), 3)
        self.assertEqual(values["out"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
