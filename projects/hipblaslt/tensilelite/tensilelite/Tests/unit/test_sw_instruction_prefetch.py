# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the software instruction-prefetch solution parameter (gfx1250).

Guards the single-integer `SwInstructionPrefetch` bitmask that replaced the
legacy (`SwInstructionPrefetch`: bool, `SwInstructionPrefetchAbs`: bool) pair:
its valid values and Relative default, the legacy-bool alias kept for backward
compatibility, and the resolver that maps the mode to the two module-option
enables the StinkyTofu passes read.
"""
import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.ValidParameters import (
    validParameters,
    normalizeSwInstructionPrefetch,
    resolveSwInstructionPrefetch,
    SW_INSTRUCTION_PREFETCH_AUTO,
    SW_INSTRUCTION_PREFETCH_OFF,
    SW_INSTRUCTION_PREFETCH_RELATIVE,
    SW_INSTRUCTION_PREFETCH_ABSOLUTE,
)

SWP = "SwInstructionPrefetch"


def _default(name):
    """Return the default value list for a benchmark-common parameter, or None."""
    for entry in defaultBenchmarkCommonParameters:
        if name in entry:
            return entry[name]
    return None


# --------------------------------------------------------------------------- #
# Parameter definition
# --------------------------------------------------------------------------- #
def test_bitmask_domain_contains_all_modes():
    domain = validParameters[SWP]
    for mode in (-1, 0, 1, 2):
        assert mode in domain


def test_legacy_bool_still_accepted_in_domain():
    # Shipped library-logic YAMLs carry `SwInstructionPrefetch: true`; the bool
    # alias must remain a valid (deprecated) value so they keep loading.
    domain = validParameters[SWP]
    assert True in domain and False in domain
    # bool must be an accepted type (type() check is strict about bool vs int).
    assert bool in {type(v) for v in domain}
    assert int in {type(v) for v in domain}


def test_default_is_relative():
    # Default is Relative(1) so the default kernel naming token stays SIP1 (same
    # as the legacy True default), avoiding a naming/hash churn across kernels.
    # Auto(-1) remains available as an opt-in.
    assert _default(SWP) == [SW_INSTRUCTION_PREFETCH_RELATIVE]  # [1]


def test_absolute_bool_parameter_removed():
    # The separate SwInstructionPrefetchAbs knob no longer exists.
    assert "SwInstructionPrefetchAbs" not in validParameters
    assert _default("SwInstructionPrefetchAbs") is None


# --------------------------------------------------------------------------- #
# normalizeSwInstructionPrefetch: legacy bool alias
# --------------------------------------------------------------------------- #
def test_normalize_bool_alias():
    assert normalizeSwInstructionPrefetch(True) == SW_INSTRUCTION_PREFETCH_RELATIVE
    assert normalizeSwInstructionPrefetch(False) == SW_INSTRUCTION_PREFETCH_OFF


def test_normalize_int_passthrough():
    for mode in (-1, 0, 1, 2):
        assert normalizeSwInstructionPrefetch(mode) == mode


# --------------------------------------------------------------------------- #
# resolveSwInstructionPrefetch: (enableRelative, enableAbsolute)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,isGfx1250,isStreamK,expected",
    [
        # Off -> nothing.
        (SW_INSTRUCTION_PREFETCH_OFF, True, False, (False, False)),
        (SW_INSTRUCTION_PREFETCH_OFF, False, False, (False, False)),
        # Relative -> relative regardless of arch/Stream-K.
        (SW_INSTRUCTION_PREFETCH_RELATIVE, True, False, (True, False)),
        (SW_INSTRUCTION_PREFETCH_RELATIVE, False, True, (True, False)),
        # Absolute -> absolute (gating enforced downstream/rejected earlier).
        (SW_INSTRUCTION_PREFETCH_ABSOLUTE, True, False, (False, True)),
        # Auto -> Absolute only on gfx1250 non-Stream-K; Relative otherwise.
        (SW_INSTRUCTION_PREFETCH_AUTO, True, False, (False, True)),
        (SW_INSTRUCTION_PREFETCH_AUTO, True, True, (True, False)),
        (SW_INSTRUCTION_PREFETCH_AUTO, False, False, (True, False)),
        (SW_INSTRUCTION_PREFETCH_AUTO, False, True, (True, False)),
        # Legacy bool alias flows through the resolver too.
        (True, True, False, (True, False)),   # True -> Relative
        (False, True, False, (False, False)),  # False -> Off
    ],
)
def test_resolve_mapping(value, isGfx1250, isStreamK, expected):
    assert resolveSwInstructionPrefetch(value, isGfx1250, isStreamK) == expected


# --------------------------------------------------------------------------- #
# f64 is excluded from Absolute: Auto -> Relative on gfx1250, and an explicit
# Absolute request is defensively downgraded to Relative here (it is rejected
# earlier in Solution.assignProblemIndependentDerivedParameters).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,isGfx1250,isStreamK,expected",
    [
        # Auto on gfx1250 non-Stream-K would be Absolute for non-f64, but f64 -> Relative.
        (SW_INSTRUCTION_PREFETCH_AUTO, True, False, (True, False)),
        # Explicit Absolute on f64 -> defensively Relative (rejected earlier anyway).
        (SW_INSTRUCTION_PREFETCH_ABSOLUTE, True, False, (True, False)),
        # Off / Relative unaffected by the f64 flag.
        (SW_INSTRUCTION_PREFETCH_OFF, True, False, (False, False)),
        (SW_INSTRUCTION_PREFETCH_RELATIVE, True, False, (True, False)),
    ],
)
def test_resolve_mapping_f64_excluded(value, isGfx1250, isStreamK, expected):
    assert resolveSwInstructionPrefetch(value, isGfx1250, isStreamK, isF64=True) == expected


# --------------------------------------------------------------------------- #
# Behavior preservation: legacy inputs reproduce the legacy module options.
# --------------------------------------------------------------------------- #
def test_behavior_preservation_legacy_relative_default():
    # Old default: SwInstructionPrefetch=True (relative on), abs off.
    # New: the legacy bool True resolves to (relative=True, absolute=False)
    # on every architecture, matching the old EnableSwInstructionPrefetchRelStatic
    # / EnableSwInstructionPrefetchAbs pair exactly.
    for isGfx1250 in (True, False):
        for isStreamK in (True, False):
            assert resolveSwInstructionPrefetch(True, isGfx1250, isStreamK) == (True, False)


def test_behavior_preservation_legacy_both_true_maps_to_absolute():
    # Old: SwInstructionPrefetch=True AND SwInstructionPrefetchAbs=True -> abs
    # wins (backend else-if). New canonical form is Absolute(2) -> abs only.
    assert resolveSwInstructionPrefetch(
        SW_INSTRUCTION_PREFETCH_ABSOLUTE, True, False) == (False, True)


# --------------------------------------------------------------------------- #
# Solution.assignProblemIndependentDerivedParameters reject path:
# explicit Absolute(2) is only valid on gfx1250 non-Stream-K.
# --------------------------------------------------------------------------- #
GFX1250 = (12, 5, 0)
GFX942 = (9, 4, 2)


def _run_piap(state):
    """Run the reject-carrying prologue of assignProblemIndependentDerivedParameters.

    The SwInstructionPrefetch reject block runs before ProblemType is consulted,
    so a minimal state reaches it; a non-rejecting mode falls through and later
    raises on the missing ProblemType (unrelated to this feature).
    """
    from Tensile.SolutionStructs.Solution import Solution

    Solution.assignProblemIndependentDerivedParameters(state, False, {})
    return state


def _min_state(swp, isa, streamk):
    return {
        "ScheduleIterAlg": 0,
        "SwInstructionPrefetch": swp,
        "ISA": isa,
        "StreamK": streamk,
    }


def test_reject_explicit_absolute_on_streamk():
    state = _min_state(SW_INSTRUCTION_PREFETCH_ABSOLUTE, GFX1250, 1)
    _run_piap(state)
    assert state.get("Valid") is False


def test_reject_explicit_absolute_on_non_gfx1250():
    state = _min_state(SW_INSTRUCTION_PREFETCH_ABSOLUTE, GFX942, 0)
    _run_piap(state)
    assert state.get("Valid") is False


def test_explicit_absolute_ok_on_gfx1250_non_streamk():
    # Absolute is valid here, so the reject block must NOT fire (Valid stays True).
    # The method later raises on the absent ProblemType; that is unrelated.
    state = _min_state(SW_INSTRUCTION_PREFETCH_ABSOLUTE, GFX1250, 0)
    try:
        _run_piap(state)
    except Exception:
        pass
    assert state.get("Valid") is not False


def test_explicit_absolute_ok_on_gfx1250_list_isa():
    """Regression: the gfx1250 guard must tuple()-normalize ISA.

    Solution YAML I/O serializes ISA as a plain list, so a round-tripped gfx1250
    state carries ``[12, 5, 0]``. Comparing that list directly to the tuple
    ``(12, 5, 0)`` is False-negative (``[12,5,0] != (12,5,0)`` is True), which
    wrongly rejected valid gfx1250 Absolute-prefetch kernels loaded from YAML.
    """
    state = _min_state(SW_INSTRUCTION_PREFETCH_ABSOLUTE, list(GFX1250), 0)
    try:
        _run_piap(state)
    except Exception:
        pass
    assert state.get("Valid") is not False


@pytest.mark.parametrize(
    "swp,isa,streamk",
    [
        # On gfx1250 Stream-K, Auto resolves to Relative (a real, working mode).
        (SW_INSTRUCTION_PREFETCH_AUTO, GFX1250, 1),
        # On non-gfx1250 the StinkyTofu backend is skipped entirely
        # (isSupportedByStinkyTofu is gfx1250-only), so every non-Absolute mode
        # emits NO prefetch at all -- the resolver's relative-enable value is an
        # unconsumed no-op there. These cases only assert the mode is not rejected.
        (SW_INSTRUCTION_PREFETCH_AUTO, GFX942, 0),      # effectively Off (no-op)
        (SW_INSTRUCTION_PREFETCH_RELATIVE, GFX942, 1),  # effectively Off (no-op)
        (True, GFX942, 1),                              # legacy bool alias; effectively Off (no-op)
        (SW_INSTRUCTION_PREFETCH_OFF, GFX942, 1),       # Off
    ],
)
def test_no_reject_for_non_absolute_modes(swp, isa, streamk):
    state = _min_state(swp, isa, streamk)
    try:
        _run_piap(state)
    except Exception:
        pass
    assert state.get("Valid") is not False
