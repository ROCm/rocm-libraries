# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS alias-scope metadata (kernel attr ``lds_alias_scopes``).

Without scopes, SIInsertWaitcnts must assume an LDS read can alias any
in-flight LDS DMA (``buffer_load ... lds``) and waits ``vmcnt(0)`` first. With
the attr, every access to an smem allocation carries its own scope and is
``noalias`` against the allocations whose pooled byte ranges it does not
overlap, so a read only waits for DMA into its own allocation.
"""

from __future__ import annotations

import re
import unittest

from rocke.core.ir import BF16, F32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, lower_kernel_to_llvm


def _kernel(*, scopes: bool, overlap: bool = False):
    """DMA into ``dst`` while reading ``other``.

    ``overlap=True`` makes ``other`` die before ``late`` is allocated, so the
    pool packs ``late`` onto ``other``'s bytes and the two must not be marked
    noalias.
    """
    b = IRBuilder("lds_aa_probe")
    out = b.param("out", PtrType(F32, "global"))
    src = b.param("src", PtrType(BF16, "global"))
    if scopes:
        b.kernel.attrs["lds_alias_scopes"] = True
    dst = b.smem_alloc(BF16, [64, 64], name_hint="dst")
    other = b.smem_alloc(BF16, [64, 64], name_hint="other")
    lane = b.thread_id_x()
    b.smem_store_vN(
        other, [lane, b.const_i32(0)], b.vec_trunc_f32_to_bf16(b.zero_vec_f32(4)), 4
    )
    rsrc = b.buffer_rsrc(src, b.const_i32(1 << 20))
    addr = b.smem_ptr_add(b.smem_addr_of(dst), b.const_i64(256))
    b.async_buffer_load_lds_addr(
        rsrc, addr, b.mul(lane, b.const_i32(4)), b.const_i32(0), 1
    )
    v = b.smem_load_vN(other, lane, b.const_i32(0), dtype=BF16, n=4)
    if overlap:
        late = b.smem_alloc(BF16, [64, 64], name_hint="late")
        b.smem_store_vN(late, [lane, b.const_i32(0)], v, 4)
    else:
        b.smem_store_vN(dst, [lane, b.const_i32(0)], v, 4)
    b.global_store(out, lane, b.const_f32(0.0))
    return b.kernel


class TestLdsAliasScopes(unittest.TestCase):
    def test_off_by_default(self):
        ir = lower_kernel_to_llvm(_kernel(scopes=False), arch="gfx942")
        self.assertNotIn("!alias.scope", ir)
        self.assertNotIn("rocke.lds", ir)

    def test_dma_and_reads_are_scoped(self):
        ir = lower_kernel_to_llvm(_kernel(scopes=True), arch="gfx942")
        dma = next(
            l for l in ir.splitlines() if "buffer.load.lds(" in l and "call" in l
        )
        load = next(
            l
            for l in ir.splitlines()
            if re.search(r"= load <4 x bfloat>, ptr addrspace\(3\)", l)
        )
        self.assertIn("!alias.scope", dma)
        self.assertIn("!noalias", load)
        md = dict(re.findall(r"^(![0-9]+) = (.*)$", ir, flags=re.MULTILINE))
        dma_scope = md[re.search(r"!alias.scope (![0-9]+)", dma).group(1)]
        load_noalias = md[re.search(r"!noalias (![0-9]+)", load).group(1)]
        # The read of `other` is noalias against `dst`'s scope.
        self.assertIn(dma_scope.strip("!{}"), load_noalias)

    def test_pooled_overlap_is_not_noalias(self):
        # `dst` dies once its DMA is issued, so the pool packs `late` onto
        # `dst`'s bytes; `other` keeps its own range. Overlapping allocations
        # must never be noalias of each other, so a read of `late` still waits
        # for the DMA into `dst`.
        ir = lower_kernel_to_llvm(_kernel(scopes=True, overlap=True), arch="gfx942")
        scope = {
            m.group(2).split("@")[1].split(".")[0].rstrip("0123456789"): m.group(1)
            for m in re.finditer(
                r'^(![0-9]+) = distinct !\{!\d+, !\d+, !"([^"]+)"\}', ir, re.MULTILINE
            )
        }
        lists = dict(re.findall(r"^(![0-9]+) = (!\{.*\})$", ir, flags=re.MULTILINE))
        noalias = {}
        for line in ir.splitlines():
            m = re.search(r"!alias.scope (![0-9]+), !noalias (![0-9]+)", line)
            if m:
                own = lists[m.group(1)][2:-1]
                noalias[own] = lists[m.group(2)]
        self.assertNotIn(scope["late"], noalias[scope["dst"]])
        self.assertNotIn(scope["dst"], noalias[scope["late"]])
        self.assertIn(scope["dst"], noalias[scope["other"]])
        self.assertIn(scope["late"], noalias[scope["other"]])

    def test_python_cpp_bytes_match(self):
        try:
            import rocke_engine
        except ImportError as exc:
            self.skipTest(f"rocke_engine extension not built: {exc}")
        for overlap in (False, True):
            kernel = _kernel(scopes=True, overlap=overlap)
            for flavor in LLVM_FLAVORS:
                with self.subTest(overlap=overlap, flavor=flavor):
                    self.assertEqual(
                        rocke_engine.lower_serialized_ir(
                            serialize(kernel), arch="gfx942", flavor=flavor
                        ),
                        lower_kernel_to_llvm(kernel, arch="gfx942", llvm_flavor=flavor),
                    )


if __name__ == "__main__":
    unittest.main()
