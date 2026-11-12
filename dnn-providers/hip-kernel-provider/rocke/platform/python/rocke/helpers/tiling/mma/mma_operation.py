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

from ....core.arch import MmaOp
from ....core.ir import Type
from ..traits import MmaTraitsCatalog
from .driver import TileMmaDriver
from .plan import Tiling, TileMmaPlan
from .styles import LayoutStyle

__all__ = ["Tiling", "TileMma"]


class TileMma:
    """A target-resolved wave-tile MMA operation + subtile driver (dense; the M1 surface).

    Author call: ``TileMma((16, 16, 64), a=F16, b=F16, c=F32, target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)))`` with the dtypes imported from ``rocke.core.ir``.
    The keywords are :class:`TileMmaPlan`'s:

    * ``a``, ``b``, ``c``, ``d`` -- the A and B operand dtypes, the accumulator taken in and the
      result returned (``d`` omitted = ``c``), each a scalar IR type.
    * ``target`` -- the gfx target the instruction is resolved for.
    * ``tiling`` -- the :class:`Tiling` knobs (atom shape, subtile order, ``mac_prio``).
    * ``style`` -- the operand-layout profile (:class:`CanonicalStyle` by default).
    * ``catalog`` -- the MMA catalog to resolve from (the committed one by default).
    * ``atom_override="<op_id>"`` -- pins one catalog instruction by name instead of selecting by
      shape and dtypes; ``shape`` and every dtype are then optional (an omitted shape is the
      instruction's own, an omitted dtype is the instruction's), and each dtype given must be one
      the instruction takes.

    Resolution + validation happen at construction, fail-fast, with no IR. Calling the instance
    (``mma(b, a_frag, b_frag, acc)``) walks the atom grid internally and returns the updated
    accumulator ``Fragment``. Reach ``.plan`` / ``.driver`` for the composed pieces.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] | None = None,
        *,
        a: Type | None = None,
        b: Type | None = None,
        c: Type | None = None,
        d: Type | None = None,
        target: str,
        tiling: Tiling | None = None,
        style: LayoutStyle | None = None,
        catalog: MmaTraitsCatalog | None = None,
        atom_override: str | None = None,
    ) -> None:
        self._plan = TileMmaPlan(
            shape,
            a=a,
            b=b,
            c=c,
            d=d,
            target=target,
            tiling=tiling,
            style=style,
            catalog=catalog,
            atom_override=atom_override,
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
    def shape(self):
        return self._plan.shape

    @property
    def atom_shape(self):
        return self._plan.atom_shape

    @property
    def subtiles(self):
        return self._plan.subtiles

    @property
    def tiling(self):
        return self._plan.tiling

    @property
    def target(self):
        return self._plan.target

    @property
    def a_dtype(self) -> Type:
        """The A operand dtype (an IR type)."""
        return self._plan.a_dtype

    @property
    def b_dtype(self) -> Type:
        """The B operand dtype (an IR type)."""
        return self._plan.b_dtype

    @property
    def c_dtype(self) -> Type:
        """The accumulator dtype the instruction takes in (an IR type)."""
        return self._plan.c_dtype

    @property
    def d_dtype(self) -> Type:
        """The result dtype the instruction returns (an IR type); :attr:`c_dtype` for every
        accumulate-in-place instruction."""
        return self._plan.d_dtype

    @property
    def op_id(self):
        return self._plan.op_id

    @property
    def wave_size(self):
        return self._plan.wave_size

    @property
    def traits(self):
        return self._plan.traits

    @property
    def a_layout(self):
        return self._plan.a_layout

    @property
    def b_layout(self):
        return self._plan.b_layout

    @property
    def c_layout(self):
        return self._plan.c_layout

    @property
    def a_desc(self):
        return self._plan.a_desc

    @property
    def b_desc(self):
        return self._plan.b_desc

    @property
    def c_desc(self):
        return self._plan.c_desc

    @property
    def style(self):
        return self._plan.style

    def emit_op(self, *, scale_format: tuple[str, str] | None = None) -> MmaOp:
        """The backend op that issues this instruction on the target, for a hand-written
        ``b.mma`` call (:meth:`TileMmaPlan.emit_op`): ``NotImplementedError`` when rocke does not
        issue it there; ``scale_format`` names a scaled instruction's (A, B) scale dtypes.
        """
        return self._plan.emit_op(scale_format=scale_format)

    # ---- iteration, delegated to the driver -----------------------------------------------------
    def __call__(self, b, a_fragment, b_fragment, accumulator):
        return self._driver(b, a_fragment, b_fragment, accumulator)

    def __repr__(self) -> str:
        p = self._plan
        return (
            f"TileMma(shape={p.shape}, atom={p.atom_shape}, "
            f"a={p.a_dtype!r}, b={p.b_dtype!r}, c={p.c_dtype!r}, "
            f"d={p.d_dtype!r}, target={p.target!r}, op_id={p.op_id!r})"
        )
