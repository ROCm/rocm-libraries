# SPDX-License-Identifier: Apache-2.0
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The `gfx950_standalone` interface of the gfx950 backward kernels: the two names `buffer_ops` and `dualwave`.

The sweep sources import a flat `gfx950_standalone` module; here every importer reaches it as
`kernels.attention_bwd_cdna.gfx950_standalone` (the import lines are rewritten). Nothing else is exported, so a
file that needs another name must add it here. There is no `sys.path` surgery: `kernels.*` resolves because the
provider's kernel root is on the path.

Two environments import this module, and they resolve `dualwave` differently:

* **Build.** `flydsl` is installed and `kernels.*` resolves, so `dualwave` is the real
  `kernels.attention_cdna.flash_attn_utils`.
* **Generator.** No flydsl, no kernel root -- by design, so that the generator imports without FlyDSL installed. `dualwave`
  falls back to `flyc_polyfill`, which carries a verbatim copy of the one name the generate-time path needs
  (`DualwaveSwpTraits`, the base class of `fmha_traits_gfx950.ParityDualwaveTraits`).

`buffer_ops` is resolved **lazily**, and deliberately so. It is device-side
only: the files that import it all import `flydsl` at module
scope too, so none of them is reachable from the generator. Importing it
eagerly here would therefore break the generator over a name the generator
never uses. Deferring it to first access keeps the failure where it belongs
-- at build time, with `kernels.common`'s own `ImportError` and traceback
intact, rather than as an `AttributeError` on a `None` placeholder.
"""

# Cheap and flydsl-free: every flydsl import in `flyc_polyfill` is
# function-local, precisely so the generator can reach the copy below.
from kernels.attention_bwd_cdna import flyc_polyfill

try:
    from kernels.attention_cdna import flash_attn_utils as dualwave
except ModuleNotFoundError:  # generator: no flydsl, no kernel root
    dualwave = flyc_polyfill
else:
    # Both definitions exist, so the copy's claim to be verbatim is checkable
    # here and nowhere else. `fields()` names/types/order, all 79 of them --
    # for this class that is a total check, not a sample. This call is why the
    # check lives on the gfx950 path rather than in `flyc_polyfill`'s own
    # module scope: gfx1201 aliases `flyc_polyfill` wholesale but never touches
    # `DualwaveSwpTraits`, and must not be failed by a class it does not use.
    flyc_polyfill.assert_dualwave_swp_traits_equivalent(dualwave.DualwaveSwpTraits)

__all__ = ["buffer_ops", "dualwave"]


def __getattr__(name):
    """Resolve `buffer_ops` on first use (PEP 562). See the module docstring."""
    if name == "buffer_ops":
        from kernels.common import buffer_ops

        globals()["buffer_ops"] = buffer_ops
        return buffer_ops
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
