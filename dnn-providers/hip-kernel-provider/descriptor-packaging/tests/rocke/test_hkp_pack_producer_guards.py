"""Producer correctness guards: the packer must refuse what it cannot express.

Each guard here defends against a *silent* wrong result rather than a crash --
a kernel that packs successfully while being tuned differently, validated not at
all, or lowered by a different engine than the artifact claims. Those are worse
than failures because the output looks fine.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import textwrap
import typing
from pathlib import Path

import pytest

from hkp_pack.errors import HkpPackError
from hkp_pack.rocke_compile import (
    _check_support_predicate,
    _require_spec_arch_signature,
    _resolve_spec_class,
    _resolve_support_predicate,
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
    """A defaulted keyword-only tuning knob must be refused.

    Such a knob is invisible to the descriptor and silently frozen, so the
    kernel ships tuned differently than anyone declared -- exactly how a real
    regression got mis-reported in this tree.
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
    """Stand-in for a kernel module exposing `build_*` builders and predicates."""

    def __init__(self, builders=("build_x",), **attrs):
        for name in builders:
            setattr(self, name, _build)
        for k, v in attrs.items():
            setattr(self, k, v)


def _build(spec: _Spec, *, arch: str = "gfx942"):
    return ("kernel", spec, arch)


@pytest.mark.quick
def test_rejects_spec_the_predicate_refuses():
    def is_valid_spec(spec: _Spec, *, arch="gfx942"):
        return (False, "head_size 999 out of envelope")

    module = _Mod(is_valid_spec=is_valid_spec)

    with pytest.raises(HkpPackError, match="out of envelope"):
        _check_support_predicate(module, "build_x", _Spec(n=1), "gfx942")


@pytest.mark.quick
def test_accepts_spec_the_predicate_allows():
    def is_valid_spec(spec: _Spec, *, arch="gfx942"):
        return (True, "")

    _check_support_predicate(
        _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
    )


@pytest.mark.quick
def test_resolves_predicate_by_derived_name():
    # `build_attention_dense` -> `supports_attention_dense`, the shape the real
    # gfx950 module uses (it has no is_valid_spec).
    def supports_attention_dense(spec: "_Spec", *, arch="gfx950"):
        return (False, "nope")

    module = _Mod(
        builders=("build_attention_dense",),
        supports_attention_dense=supports_attention_dense,
    )

    with pytest.raises(HkpPackError, match="nope"):
        _check_support_predicate(module, "build_attention_dense", _Spec(n=1), "gfx950")


@pytest.mark.quick
def test_resolves_predicate_by_alias_for_tiled_family():
    # The tiled family's predicates do not follow the naming convention; without
    # the alias map this builder would go unvalidated.
    def supports_tiled_2d_spec(spec: _Spec, *, arch="gfx942"):
        return (False, "tiled no")

    module = _Mod(
        builders=("build_unified_attention_2d_tiled",),
        supports_tiled_2d_spec=supports_tiled_2d_spec,
    )

    with pytest.raises(HkpPackError, match="tiled no"):
        _check_support_predicate(
            module, "build_unified_attention_2d_tiled", _Spec(n=1), "gfx942"
        )


@pytest.mark.quick
def test_resolves_only_the_predicate_written_for_the_builders_spec():
    """One module, several builders: each predicate judges only its own spec class.

    The shape of conv_direct_grouped / kda_chunkwise: `is_valid_spec` is written
    for another spec, and the builder's own predicate has a suffixed name.
    """

    @dataclasses.dataclass
    class _Other:
        m: int

    def is_valid_spec(spec: _Other, *, arch="gfx942"):
        return spec.m > 0, ""

    def is_valid_spec_x(spec: "_Spec", *, arch="gfx942"):
        return (False, "own predicate")

    def is_valid_unannotated(spec, *, arch="gfx942"):
        raise AssertionError("an unannotated predicate is not this builder's")

    def is_valid_spec_for_problem(spec: _Spec, problem, arch="gfx942"):
        raise AssertionError("a predicate needing a problem cannot be called")

    module = _Mod(
        is_valid_unannotated=is_valid_unannotated,
        is_valid_spec_for_problem=is_valid_spec_for_problem,
        is_valid_spec=is_valid_spec,
        is_valid_spec_x=is_valid_spec_x,
    )

    with pytest.raises(HkpPackError, match="own predicate"):
        _check_support_predicate(module, "build_x", _Spec(n=1), "gfx942")


@pytest.mark.quick
def test_skips_kwargs_only_predicate():
    """Not callable from a spec instance, so it must be skipped, not guessed at.

    The real supports_tiled_2d takes head_size/block_size/dtype individually as
    keyword-only args. Inventing a spec-field-to-kwarg mapping would validate
    something other than what gets built.
    """

    def supports_tiled_2d_spec(*, spec: _Spec, head_size, block_size, dtype):
        raise AssertionError("must not be called from a spec instance")

    module = _Mod(
        builders=("build_unified_attention_2d_tiled",),
        supports_tiled_2d_spec=supports_tiled_2d_spec,
    )

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
    def is_valid_spec(spec: _Spec, *, arch="gfx942"):
        raise RuntimeError("predicate exploded")

    _check_support_predicate(
        _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "g"
    )


@pytest.mark.quick
def test_predicate_that_cannot_read_the_spec_fails_the_pack():
    """A predicate that cannot read the spec has no verdict, so the pack fails."""

    def is_valid_spec(spec: _Spec, *, arch="gfx942"):
        return spec.kv_storage_dtype is None, ""

    with pytest.raises(HkpPackError, match="raised AttributeError at") as exc:
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )
    assert "is_valid_spec" in str(exc.value)
    assert "test_hkp_pack_producer_guards.py:" in str(exc.value)


@pytest.mark.quick
def test_predicate_with_positional_only_arch_fails_with_call_error():
    """A predicate whose arch parameter is positional-only triggers a TypeError
    at bind time, not inside the body.  The error must surface as a pack failure
    with a 'could not be called' message (not 'could not read the spec')."""

    # positional-only arch: the packer always passes arch as a keyword
    def is_valid_spec(spec: _Spec, arch: str = "gfx942", /):
        return True, ""

    with pytest.raises(HkpPackError, match="could not be called") as exc:
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )
    assert "is_valid_spec" in str(exc.value)


@pytest.mark.quick
def test_reduce_tiled_builder_has_no_predicate():
    # The reduce spec lacks the fields supports_tiled_3d_spec reads.
    def supports_tiled_3d_spec(spec: _Spec, *, arch="gfx942"):
        return spec.kv_storage_dtype is None, ""

    module = _Mod(
        builders=("build_unified_attention_reduce_tiled",),
        supports_tiled_3d_spec=supports_tiled_3d_spec,
    )

    _check_support_predicate(
        module, "build_unified_attention_reduce_tiled", _Spec(n=1), "gfx942"
    )


_HAS_ROCKE_LIBRARY = (
    Path(__file__).resolve().parents[3] / "rocke" / "library"
).is_dir()


def _library_builders():
    """``(module, builder, spec class)`` for every ``build_*`` the kernel library
    defines whose spec class resolves."""
    import kernels

    for info in pkgutil.walk_packages(kernels.__path__, "kernels."):
        module = importlib.import_module(info.name)
        for builder, fn in vars(module).items():
            if (
                not builder.startswith("build_")
                or not inspect.isfunction(fn)
                or fn.__module__ != module.__name__
            ):
                continue
            try:
                yield module, builder, _resolve_spec_class(module, fn)
            except HkpPackError:
                continue


@pytest.mark.skipif(not _HAS_ROCKE_LIBRARY, reason="rocke/library not in this tree")
def test_every_rocke_builder_resolves_a_predicate_for_its_own_spec():
    """Over the real kernel library: a resolved predicate takes the builder's spec."""
    checked = 0
    for module, builder, spec_cls in _library_builders():
        name, predicate = _resolve_support_predicate(module, builder)
        if predicate is None:
            continue
        checked += 1
        first = next(iter(inspect.signature(predicate).parameters.values()))
        hint = typing.get_type_hints(predicate).get(first.name)
        assert isinstance(hint, type) and issubclass(spec_cls, hint), (
            f"{module.__name__}.{builder} ({spec_cls.__name__}) resolved {name} "
            f"for {hint!r}"
        )
    assert checked


#: Builders `_check_support_predicate` cannot validate today, because they have no
#: spec-first predicate. Every builder already in the library is either resolvable or
#: listed here, so the test below flags only a builder added without a predicate. A
#: builder that gains one needs no edit here.
_UNVALIDATED_BUILDERS = frozenset(
    {
        "kernels.common.attention_unified.build_unified_attention_2d",
        "kernels.common.attention_unified.build_unified_attention_3d",
        "kernels.common.attention_unified.build_unified_attention_reduce",
        "kernels.common.conv_direct_grouped.build_direct_transpose_weights_dgrad",
        "kernels.common.conv_direct_grouped.build_direct_reorganize_weights",
        "kernels.common.conv_direct_grouped.build_direct_coalesced_weights_dgrad",
        "kernels.common.conv_direct_grouped.build_direct_mfma_dgrad",
        "kernels.common.conv_implicit_gemm_wgrad_two_stage.build_implicit_gemm_conv_wgrad_two_stage",
        "kernels.common.conv_wgrad_workspace_reduce.build_conv_wgrad_workspace_reduce",
        "kernels.gfx1250.attention_tiled_2d.build_unified_attention_2d_tiled",
        "kernels.gfx1250.attention_tiled_3d.build_unified_attention_3d_tiled",
        "kernels.gfx1250.attention_tiled_3d.build_unified_attention_reduce_tiled",
        "kernels.gfx942.attention_tiled_3d.build_unified_attention_reduce_tiled",
        "kernels.gfx950.attention_tiled_2d_fastkv_regp.build_unified_attention_2d_fastkv_register_p",
        "kernels.gfx950.attention_tiled_3d.build_unified_attention_reduce_tiled",
    }
)


@pytest.mark.skipif(not _HAS_ROCKE_LIBRARY, reason="rocke/library not in this tree")
def test_no_new_builder_ships_without_a_support_predicate():
    """The pack validates a spec only through a predicate it can resolve, so a builder
    added without one ships unvalidated. Give it a predicate, or list it above on
    purpose."""
    unvalidated = {
        f"{module.__name__}.{builder}"
        for module, builder, _ in _library_builders()
        if _resolve_support_predicate(module, builder)[1] is None
    }
    new = sorted(unvalidated - _UNVALIDATED_BUILDERS)
    assert not new, (
        f"builder(s) with no support predicate: {new}. Add a spec-first "
        "`supports_*_spec`/`is_valid_*` predicate (see _PREDICATE_ALIASES) or list "
        "them in _UNVALIDATED_BUILDERS."
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
