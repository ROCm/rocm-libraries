"""Producer correctness guards: the packer must refuse what it cannot express.

Each guard here defends against a *silent* wrong result rather than a crash --
a kernel that packs successfully while being tuned differently, validated not at
all, or lowered by a different engine than the artifact claims. Those are worse
than failures because the output looks fine.
"""

import dataclasses
import textwrap
import typing

import pytest

from hkp_pack.errors import HkpPackError
from hkp_pack.rocke_compile import (
    _check_support_predicate,
    _require_spec_arch_signature,
    build_spec,
)


@dataclasses.dataclass
class _Spec:
    n: int


# --- A. Builder signature: nothing the UKD cannot supply --------------------
@pytest.mark.quick
def test_accepts_spec_arch_signature():
    def build_ok(spec: _Spec, *, arch: str = "gfx942"):
        return ("kernel", spec, arch)

    _require_spec_arch_signature(build_ok, "build_ok")


@pytest.mark.quick
def test_rejects_keyword_only_parameter():
    """The real case: gfx942's build_attention_dense takes `tuning`.

    A defaulted keyword-only knob is invisible to the descriptor and silently
    frozen, so the kernel ships tuned differently than anyone declared. The
    tuning class's own docstring warns this is "exactly how a real +79% got
    reported as -17% in this tree."
    """

    def build_tuned(spec: _Spec, *, arch: str = "gfx942", tuning: int = 7):
        return ("kernel", spec, arch, tuning)

    with pytest.raises(HkpPackError, match="tuning") as exc:
        _require_spec_arch_signature(build_tuned, "build_tuned")
    assert "silently frozen" in str(exc.value)


@pytest.mark.quick
def test_rejects_var_keyword_parameter():
    def build_kwargs(spec: _Spec, *, arch: str = "gfx942", **rest):
        return ("kernel", spec, arch, rest)

    with pytest.raises(HkpPackError, match=r"\*args/\*\*kwargs"):
        _require_spec_arch_signature(build_kwargs, "build_kwargs")


@pytest.mark.quick
def test_rejects_extra_positional_parameter():
    def build_two(spec: _Spec, other, *, arch: str = "gfx942"):
        return ("kernel", spec, other, arch)

    with pytest.raises(HkpPackError, match="must be"):
        _require_spec_arch_signature(build_two, "build_two")


# --- B. Support predicates --------------------------------------------------
class _Mod:
    """Stand-in for a kernel module exposing a predicate."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.mark.quick
def test_rejects_spec_the_predicate_refuses():
    def is_valid_spec(spec, *, arch="gfx942"):
        return (False, "head_size 999 out of envelope")

    module = _Mod(is_valid_spec=is_valid_spec)

    with pytest.raises(HkpPackError, match="out of envelope"):
        _check_support_predicate(module, "build_x", _Spec(n=1), "gfx942")


@pytest.mark.quick
def test_accepts_spec_the_predicate_allows():
    module = _Mod(is_valid_spec=lambda spec, *, arch="gfx942": (True, ""))

    _check_support_predicate(module, "build_x", _Spec(n=1), "gfx942")


@pytest.mark.quick
def test_resolves_predicate_by_derived_name():
    # `build_attention_dense` -> `supports_attention_dense`, the shape the real
    # gfx950 module uses (it has no is_valid_spec).
    module = _Mod(
        supports_attention_dense=lambda spec, *, arch="gfx950": (False, "nope")
    )

    with pytest.raises(HkpPackError, match="nope"):
        _check_support_predicate(module, "build_attention_dense", _Spec(n=1), "gfx950")


@pytest.mark.quick
def test_resolves_predicate_by_alias_for_tiled_family():
    # The tiled family's predicates do not follow the naming convention; without
    # the alias map this builder would go unvalidated.
    module = _Mod(supports_tiled_2d=lambda spec, *, arch="gfx942": (False, "tiled no"))

    with pytest.raises(HkpPackError, match="tiled no"):
        _check_support_predicate(
            module, "build_unified_attention_2d_tiled", _Spec(n=1), "gfx942"
        )


@pytest.mark.quick
def test_skips_kwargs_only_predicate():
    """Not callable from a spec instance, so it must be skipped, not guessed at.

    The real supports_tiled_2d takes head_size/block_size/dtype individually as
    keyword-only args. Inventing a spec-field-to-kwarg mapping would validate
    something other than what gets built.
    """

    def supports_tiled_2d(*, head_size, block_size, dtype):
        raise AssertionError("must not be called from a spec instance")

    module = _Mod(supports_tiled_2d=supports_tiled_2d)

    _check_support_predicate(
        module, "build_unified_attention_2d_tiled", _Spec(n=1), "gfx942"
    )


@pytest.mark.quick
def test_missing_predicate_is_not_an_error():
    _check_support_predicate(_Mod(), "build_x", _Spec(n=1), "gfx942")


@pytest.mark.quick
def test_raising_predicate_does_not_block_packing():
    # A pre-flight check that itself breaks must not fail the pack; the builder
    # remains the real gate.
    def is_valid_spec(spec, *, arch="gfx942"):
        raise RuntimeError("predicate exploded")

    _check_support_predicate(
        _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "g"
    )


# --- C. Literal validation --------------------------------------------------
@dataclasses.dataclass
class _LiteralSpec:
    mode: typing.Literal["none", "causal"]


@pytest.mark.quick
def test_accepts_valid_literal():
    assert build_spec(_LiteralSpec, {"mode": "causal"}).mode == "causal"


@pytest.mark.quick
def test_rejects_invalid_literal():
    """A typo'd enum value must not build a kernel with the wrong semantics.

    FmhaMaskMode is a real Literal in the corpus, so {"mode": "casual"} would
    otherwise produce a silently wrong mask.
    """
    with pytest.raises(HkpPackError, match="invalid value"):
        build_spec(_LiteralSpec, {"mode": "casual"})


@pytest.mark.quick
def test_unresolvable_hint_is_a_readable_error(tmp_path):
    """A bad forward reference must not surface as a bare NameError.

    Encountered for real during review verification: one unresolvable annotation
    took down the whole spec construction with no indication of which class.
    """
    ns = {}
    exec(
        textwrap.dedent(
            """
            from __future__ import annotations
            import dataclasses

            @dataclasses.dataclass
            class Broken:
                field: DoesNotExist
            """
        ),
        ns,
    )

    with pytest.raises(HkpPackError, match="cannot resolve type hints"):
        build_spec(ns["Broken"], {"field": 1})
