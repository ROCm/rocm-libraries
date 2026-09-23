# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``TileMma`` -- the front-door MMA object (composes a design plan + an iteration driver).

The author states logical intent (a WAVE-TILE shape + dtypes) and a bound target; ``TileMma`` resolves
the concrete intrinsic and exposes the A/B/C layouts (IR-free), and calling the instance
(``mma(b, a_frag, b_frag, acc)``) walks the atom grid and returns the updated accumulator.

Under the hood it is a glass-box composition (``docs/tiling_api_contract.md``): a
:class:`~rocke.helpers.tiling.mma.plan.TileMmaPlan` (the design half -- resolution + layouts) and a
:class:`~rocke.helpers.tiling.mma.driver.TileMmaDriver` (the iteration half). The front door keeps the
simple case a two-liner; an advanced author reaches ``mma.plan`` / ``mma.driver`` (both toolbox tier) to
customise. The front door hides neither.
"""

from __future__ import annotations

from .driver import TileMmaDriver
from .plan import Tiling, TileMmaPlan

__all__ = ["Tiling", "TileMma"]


class TileMma:
    """A target-resolved wave-tile MMA operation + subtile driver (dense; the M1 surface).

    Author call: ``TileMma((16, 16, 64), a="f16", b="f16", c="f32", target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)))``. Resolution + validation happen at construction,
    fail-fast, with no IR. Calling the instance (``mma(b, a_frag, b_frag, acc)``) walks the atom grid
    internally and returns the updated accumulator ``Fragment``. Reach ``.plan`` / ``.driver`` for the
    composed pieces.
    """

    def __init__(self, shape=None, *, a=None, b=None, c=None, target,
                 tiling=None, style=None, catalog=None, atom_override=None) -> None:
        self._plan = TileMmaPlan(
            shape, a=a, b=b, c=c, target=target, tiling=tiling, style=style,
            catalog=catalog, atom_override=atom_override,
        )
        self._driver = TileMmaDriver(self._plan)

    # ---- the composed pieces (glass-box: expose, never hide) ------------------------------------
    @property
    def plan(self) -> TileMmaPlan:
        """The design half -- atom resolution + A/B/C layouts (toolbox tier)."""
        return self._plan

    @property
    def driver(self) -> TileMmaDriver:
        """The iteration half -- the atom-grid walk (toolbox tier)."""
        return self._driver

    # ---- design surface, delegated to the plan --------------------------------------------------
    @property
    def shape(self): return self._plan.shape

    @property
    def atom_shape(self): return self._plan.atom_shape

    @property
    def subtiles(self): return self._plan.subtiles

    @property
    def tiling(self): return self._plan.tiling

    @property
    def target(self): return self._plan.target

    @property
    def op_id(self): return self._plan.op_id

    @property
    def wave_size(self): return self._plan.wave_size

    @property
    def traits(self): return self._plan.traits

    @property
    def a_layout(self): return self._plan.a_layout

    @property
    def b_layout(self): return self._plan.b_layout

    @property
    def c_layout(self): return self._plan.c_layout

    @property
    def c_desc(self): return self._plan.c_desc

    @property
    def style(self): return self._plan.style

    @property
    def a_operand_desc(self): return self._plan.a_operand_desc

    @property
    def b_operand_desc(self): return self._plan.b_operand_desc

    @property
    def c_native_desc(self): return self._plan.c_native_desc

    def a_desc(self):
        return self._plan.a_desc()

    def b_desc(self):
        return self._plan.b_desc()

    def emit_op(self):
        return self._plan.emit_op()

    # ---- iteration, delegated to the driver -----------------------------------------------------
    def __call__(self, b, a_fragment, b_fragment, accumulator):
        return self._driver(b, a_fragment, b_fragment, accumulator)

    def __repr__(self) -> str:
        p = self._plan
        return (
            f"TileMma(shape={p.shape}, atom={p.atom_shape}, "
            f"a={p.a_dtype!r}, b={p.b_dtype!r}, c={p.c_dtype!r}, "
            f"target={p.target!r}, op_id={p.op_id!r})"
        )
