################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Python-side ``Register`` abstraction over rocisa ``RegisterContainer``.

The validator handles two parallel realizations of the same logical register:

  * **Numeric** form (``regIdx >= 0``, ``regName is None``) — the post-allocation
    form emitted during normal codegen.
  * **Symbolic** form (``regIdx == -1``, ``regName`` set) — used by pre-allocation
    passes and several CMS test fixtures. The identity is the
    ``regName.name`` plus ``regName.getTotalOffsets()``.

``Register`` collapses the form-dispatch into one place. Construction goes
through :meth:`Register.from_rocisa`. After construction the overlap /
containment / intersection predicates operate uniformly regardless of the
underlying form. Symbolic-to-numeric resolution is intentionally NOT
provided here; the future-design bead ``rocm-libraries-d0xd`` will add that
once a real consumer exists.

The wrapped ``RegisterContainer`` is preserved on ``rocisa_container`` for
callers that need to round-trip back into rocisa-locked APIs (factories,
formatters, etc.). ``Register`` does NOT replace ``RegisterContainer``; the
rocisa class is nanobind-locked and out of scope to modify.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class Register:
    """Unified abstraction over numeric and symbolic register identifiers.

    Two ``Register`` instances compare equal when they refer to the same
    logical register in the same form (numeric vs symbolic). Cross-form
    comparison is intentionally rejected (see :meth:`overlaps`).

    Attributes
    ----------
    reg_type:
        ``'v'`` (vgpr) / ``'s'`` (sgpr) / ``'acc'`` (accvgpr) / ``'m'`` (mgpr)
        / ``'scc'`` etc. Matches the rocisa ``RegisterContainer.regType``
        string verbatim.
    name:
        Symbolic name root (e.g. ``"ValuA"``) when the register is symbolic,
        otherwise ``None``.
    base:
        Absolute index for numeric registers; offset within ``name`` for
        symbolic registers (``regName.getTotalOffsets()``).
    count:
        Number of consecutive registers covered (1 for single-register
        containers, ``N`` for an ``N``-wide range). Always ``>= 1``.
    rocisa_container:
        The originating ``RegisterContainer`` (or ``None`` for sentinels
        constructed without a backing container). Preserved verbatim so
        callers can round-trip back into rocisa-locked APIs without a fresh
        factory call. Excluded from equality/hashing — equality depends only
        on the resolved (reg_type, name, base, count) tuple.
    """

    reg_type: str
    name: Optional[str]
    base: int
    count: int
    rocisa_container: Optional[Any] = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Register.count must be >= 1, got {self.count}")

    # ------------------------------------------------------------------
    # Form predicates
    # ------------------------------------------------------------------
    def is_symbolic(self) -> bool:
        """True if this register is in symbolic (un-resolved) form."""
        return self.name is not None

    def is_numeric(self) -> bool:
        """True if this register is in numeric (post-allocation) form."""
        return self.name is None

    # ------------------------------------------------------------------
    # Range accessors
    # ------------------------------------------------------------------
    def lo(self) -> int:
        """Inclusive low end of the register range."""
        return self.base

    def hi(self) -> int:
        """Exclusive high end of the register range (``base + count``)."""
        return self.base + self.count

    # ------------------------------------------------------------------
    # Overlap / containment / intersection
    # ------------------------------------------------------------------
    def overlaps(self, other: "Register") -> bool:
        """True if ``self`` shares at least one register slot with ``other``.

        Semantics:

          * Registers of different ``reg_type`` never overlap (vgpr v sgpr is
            a different namespace).
          * Two symbolic registers must share the same ``name`` root before
            comparing offsets.
          * A numeric and an unresolved symbolic register never overlap
            (different namespaces, no resolution table available).

        Symmetric: ``a.overlaps(b) == b.overlaps(a)``.
        """
        if self.reg_type != other.reg_type:
            return False
        if self.is_symbolic() != other.is_symbolic():
            return False
        if self.is_symbolic() and self.name != other.name:
            return False
        return self.lo() < other.hi() and other.lo() < self.hi()

    def contains(self, other: "Register") -> bool:
        """True if ``self``'s range fully covers ``other``'s range.

        Containment requires the same ``reg_type`` AND (for symbolic
        registers) the same ``name`` root. Reflexive: ``a.contains(a)`` is
        always ``True``. A zero-overlap pair is NOT contained.
        """
        if self.reg_type != other.reg_type:
            return False
        if self.is_symbolic() != other.is_symbolic():
            return False
        if self.is_symbolic() and self.name != other.name:
            return False
        return self.lo() <= other.lo() and other.hi() <= self.hi()

    def intersection(self, other: "Register") -> Optional["Register"]:
        """Return the ``Register`` covering the overlap with ``other``,
        or ``None`` if they don't overlap.

        Cross-type, cross-form, and different-name pairs return ``None``
        (consistent with :meth:`overlaps`). ``rocisa_container`` on the
        result is ``None`` because the intersection is a fresh logical
        range, not a captured rocisa container; the caller is responsible
        for materialising a real ``RegisterContainer`` if it needs one.
        """
        if not self.overlaps(other):
            return None
        lo = max(self.lo(), other.lo())
        hi = min(self.hi(), other.hi())
        return Register(
            reg_type=self.reg_type,
            name=self.name,  # equal to other.name on the overlap path
            base=lo,
            count=hi - lo,
        )

    # ------------------------------------------------------------------
    # Predicate / construction surface for rocisa-typed callers
    # ------------------------------------------------------------------
    @staticmethod
    def is_register(x: Any) -> bool:
        """True if ``x`` walks like a rocisa ``RegisterContainer`` — i.e.
        has ``regType`` AND ``regIdx``. Used to filter ``getParams()``
        entries that aren't registers (modifiers, ints, comments, None).
        """
        return x is not None and hasattr(x, "regType") and hasattr(x, "regIdx")

    @classmethod
    def from_rocisa(cls, rc: Any) -> "Register":
        """Build a ``Register`` from a rocisa ``RegisterContainer``.

        Numeric containers (``regIdx >= 0``) become numeric ``Register``s.
        Symbolic containers (``regIdx == -1`` and ``regName`` set) become
        symbolic ``Register``s carrying the ``regName.name`` root and the
        ``regName.getTotalOffsets()`` offset.

        Symbolic-to-numeric resolution is intentionally not provided. See
        ``rocm-libraries-d0xd`` (the c70 follow-up epic) for the future
        design once a real consumer exists.

        Raises
        ------
        ValueError
            If ``rc`` is ``None``, lacks ``regType`` / ``regIdx``, or is
            in an unrecognised half-symbolic form (``regIdx == -1`` with
            no ``regName``).
        """
        if rc is None:
            raise ValueError("Register.from_rocisa: rc is None")

        # Direct attribute access for the two non-optional rocisa fields.
        # `regType` and `regIdx` are bound on every RegisterContainer; if
        # they're missing the input isn't a RegisterContainer at all and
        # we should fail loudly rather than synthesise a bogus Register.
        try:
            reg_type = rc.regType
            reg_idx = rc.regIdx
        except AttributeError as exc:
            raise ValueError(
                f"Register.from_rocisa: container {rc!r} is missing "
                f"regType/regIdx ({exc})"
            ) from exc

        # `regNum` defaults to 1 across rocisa subclasses but is
        # technically optional on a few synthetic resource types
        # (e.g. SCC sentinels), hence getattr with default.
        count = getattr(rc, "regNum", 1) or 1

        # `regName` is optional — only present on symbolic containers.
        reg_name = getattr(rc, "regName", None)

        if reg_idx >= 0:
            # Numeric form. regName may still be set on some constructions
            # (it's harmless metadata) but the identity is the numeric idx.
            return cls(
                reg_type=reg_type,
                name=None,
                base=reg_idx,
                count=count,
                rocisa_container=rc,
            )

        if reg_idx == -1 and reg_name is not None:
            # Symbolic form.
            symbolic_root = getattr(reg_name, "name", None)
            if symbolic_root is None:
                raise ValueError(
                    f"Register.from_rocisa: symbolic container {rc!r} has "
                    f"regName with no name attribute"
                )
            offset = (
                reg_name.getTotalOffsets()
                if hasattr(reg_name, "getTotalOffsets")
                else 0
            )
            return cls(
                reg_type=reg_type,
                name=symbolic_root,
                base=offset,
                count=count,
                rocisa_container=rc,
            )

        # Malformed: regIdx < 0 but no regName, or some other shape.
        raise ValueError(
            f"Register.from_rocisa: container {rc!r} has regIdx={reg_idx} "
            f"but no regName — neither numeric nor symbolic form"
        )

    # ------------------------------------------------------------------
    # Hashable signature
    # ------------------------------------------------------------------
    def signature(self) -> Tuple[Any, ...]:
        """Stable, hashable summary used for set-based dedup of edges."""
        return (self.reg_type, self.name, self.base, self.count)
