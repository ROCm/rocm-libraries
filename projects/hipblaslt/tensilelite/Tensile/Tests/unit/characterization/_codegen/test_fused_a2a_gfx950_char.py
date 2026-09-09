# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 fused GEMM+A2A (FusedGemmA2A=1) codegen characterization.

Checks that a FusedGemmA2A=1 solution emits the SDMA ring and packet
instructions and that the result assembles for gfx950. The store path replaces
the ordinary global-write batch, so a solution that silently fell back to it
still satisfies every assertion except the SDMA markers.

Covers Tensile/Components/SdmaRingEmitter.py, SdmaPacketEmitter.py, and the
FusedGemmA2A=1 store path in GlobalWriteBatch.py.
"""

import os

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # Tensile/Tests/
        "common", "comm", "gfx950", "fused_a2a.yaml",
    )
)

# Emitted only by the SDMA ring/packet emitters.
_A2A_MARKERS = ("s_atomic_umax_x2", "s_atomic_cmpswap_x2", "s_bfm_b64")


def test_fused_a2a_gfx950_emits_assembly():
    """FusedGemmA2A=1: SDMA ring instructions present, assembles for gfx950."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert_assembles(src, base)
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx950" in src, f"Kernel {base!r} missing gfx950 arch marker"
        missing = [m for m in _A2A_MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing SDMA instructions {missing}"


def test_fused_a2a_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted fused kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot


class _StubSgprPool:
    """Ascending indices, honouring alignment; checkIn is a no-op."""

    def __init__(self):
        self.next = 8

    def checkOut(self, n, tag="", preventOverflow=True):
        idx = self.next
        self.next += n
        return idx

    def checkOutAligned(self, n, align, tag="", preventOverflow=True):
        self.next = -(-self.next // align) * align
        idx = self.next
        self.next += n
        return idx

    def checkIn(self, idx):
        pass


class _StubLabels:
    def __init__(self):
        self.n = 0

    def getNameInc(self, name):
        self.n += 1
        return "%s_%u" % (name, self.n)


class _StubWriter:
    """The whole KernelWriter surface emitReserveQueueSpace touches."""

    def __init__(self):
        self.sgprPool = _StubSgprPool()
        self.labels = _StubLabels()


def _reserve_size_operands(size):
    """src1 of both `cur + size` adds emitted by one emitReserveQueueSpace."""
    import re

    from rocisa.code import Module

    from Tensile.Components.SdmaRingEmitter import SdmaRingEmitter

    module = Module("reserve")
    SdmaRingEmitter(groupImm=0x100).emitReserveQueueSpace(
        module, _StubWriter(), 0, 2, 4, 6, size, 100, 102
    )
    return re.findall(
        r"^s_add_u32 s\d+, s\d+, (\S+)[^\n]*"
        r"// (?:WrapIntoRing\(cur\) \+|new lo = cur \+) size",
        str(module),
        re.M,
    )


def test_reserve_queue_space_takes_an_immediate_size():
    assert _reserve_size_operands(84) == ["84", "84"]


def test_reserve_queue_space_takes_an_sgpr_size():
    from rocisa.container import sgpr

    assert _reserve_size_operands(sgpr(99)) == ["s99", "s99"]
