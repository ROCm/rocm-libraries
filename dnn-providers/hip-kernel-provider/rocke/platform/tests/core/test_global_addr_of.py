# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``tile.global_addr_of`` must stay defined for an index past the allocation.

The address feeds a TDM descriptor, and a clipped fill can point past the
buffer with a zero extent -- early refill's second prologue fill does when
K < tile_k. The mover's clipping keeps the copy in bounds, but only if the
address itself is defined:

* LLVM: an ``inbounds`` GEP past the object is poison, so the GEP is plain.
* HIP: ``&ptr[index]`` past the object is undefined pointer arithmetic, so the
  address is formed in integers.

The C++ engine's LLVM text is pinned byte-for-byte by the ``gfx1250_lowering``
parity configs that build descriptors with this op.
"""

from __future__ import annotations

import re
import unittest

from rocke.core.ir import BF16, I32, IRBuilder, PtrType


def _kernel():
    b = IRBuilder("global_addr_of_probe")
    a = b.param("A", PtrType(BF16, "global"), readonly=True, align=16)
    out = b.param("out", PtrType(I32, "global"), align=16)
    k = b.param("K", I32)
    addr = b.global_addr_of(a, k)
    b.global_store(out, b.const_i32(0), b.trunc(addr, I32))
    b.ret()
    return b.kernel


class TestGlobalAddrOf(unittest.TestCase):
    def test_llvm_gep_is_not_inbounds(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        llvm = _lower_kernel_to_llvm_python(_kernel(), arch="gfx1250")
        geps = re.findall(r"%gaddr_gep[.\d]* = getelementptr[^\n]*", llvm)
        self.assertEqual(len(geps), 1, llvm)
        self.assertNotIn("inbounds", geps[0])
        self.assertIn("getelementptr bfloat, ptr addrspace(1) %A", geps[0])
        self.assertRegex(llvm, r"= ptrtoint ptr addrspace\(1\) %gaddr_gep[.\d]* to i64")

    def test_hip_forms_the_address_in_integers(self):
        from rocke.core.lower_hip import lower_kernel_to_hip

        hip = lower_kernel_to_hip(_kernel(), arch="gfx1250")
        line = next(l for l in hip.splitlines() if "sizeof(*" in l)
        self.assertIn("(int64_t)(uintptr_t)", line)
        self.assertNotRegex(hip, r"\(int64_t\)\(&\w+\[")


if __name__ == "__main__":
    unittest.main()
