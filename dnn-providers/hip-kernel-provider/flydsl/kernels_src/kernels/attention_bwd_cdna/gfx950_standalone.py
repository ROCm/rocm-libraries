# SPDX-License-Identifier: Apache-2.0
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Our own `gfx950_standalone`, authored rather than vendored.

Upstream's file of this name (`kernels/attention/parity/gfx950_standalone.py`)
is a compatibility layer between `parity/` -- which is not part of the flydsl
package -- and its parent, which is. It does that with `sys.path` surgery
relative to `Path(__file__).parents[3]`. Under `modules/flash/flyc/` that
resolves to the **AOTriton repository root**, so vendoring it verbatim would
put the wrong directory on `sys.path`. There is nothing to fix in it: it is
correct where it lives and wrong where we would put it.

So this file supplies its *interface* instead. Measured across all eight
importers among the vendored parity files, that interface is exactly two
names -- `buffer_ops` and `dualwave`. Upstream also re-exports
`kernels_common`, `layout_utils`, `mem_ops` and `utils`; nothing here takes
them, so they are not re-exported. Add one only when a vendored file imports
it.

The payoff is that **every vendored gfx950 file keeps its import lines
verbatim**: there are no gfx950 import rewrites at all, so a re-sync diff for
those files is empty. That is a better outcome than gfx1201 got, where the
equivalent shim was deleted and four import lines are rewritten on every
re-sync.

**No `sys.path` surgery here.** `python/flyc_bootstrap.py` already owns that
(it puts `$AOTRITON_FLYDSL_KERNEL_ROOT` on the path, so `kernels.*` resolves);
doing it a second time from a different anchor is how the two would drift.

Two environments have to import this module, and they resolve `dualwave`
differently:

* **Build.** `flyc_bootstrap.setup()` has run, `flydsl` is installed and
  `kernels.*` resolves, so `dualwave` is the real
  `kernels.attention.flash_attn_utils`.
* **Generator.** No flydsl, no kernel root -- by design, so that
  `.ci/build-shim.sh` stays cheap. `dualwave` falls back to `flyc_polyfill`,
  which carries a verbatim copy of the one name the generate-time path needs
  (`DualwaveSwpTraits`, the base class of
  `fmha_traits_gfx950.ParityDualwaveTraits`).

`buffer_ops` is resolved **lazily**, and deliberately so. It is device-side
only: the five vendored files that import it all import `flydsl` at module
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
    # here and nowhere else. `fields()` names/types/order, all 78 of them --
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
