# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""State-isolation helpers for tests that select a rocisa target ISA."""

from contextlib import contextmanager


@contextmanager
def preserve_rocisa_kernel_state():
    """Restore rocisa's active kernel and register tracking after a test scope.

    ``rocIsa.init`` intentionally retains the per-ISA capability cache, whereas
    ``rocIsa.setKernel`` changes the thread's instruction spelling and clears
    register-name tracking. Tests can safely retain the former while restoring
    the latter for the next pytest item.
    """
    from rocisa import rocIsa

    ri = rocIsa.getInstance()
    previous_kernel = ri.getKernel()
    previous_vgpr_idx = dict(ri.getVgprIdx())
    previous_vgpr_msb = ri.getVgprMsb()
    try:
        yield
    finally:
        previous_isa = getattr(previous_kernel, "isa", None)
        if previous_isa is None:
            # The StinkyTofu adaptor represents "never pinned" with isa=None.
            ri.setKernelInfo(previous_kernel)
        else:
            ri.setKernel(tuple(previous_isa), previous_kernel.wavefrontSize)
        # Native rocisa clears this map in setKernel; the StinkyTofu adaptor
        # exposes the live dict and currently does not. Clear handles both.
        ri.getVgprIdx().clear()
        for name, idx in previous_vgpr_idx.items():
            ri.setVgprIdx(name, idx)
        ri.setVgprMsb(previous_vgpr_msb)
