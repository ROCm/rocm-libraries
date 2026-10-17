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
    _PREDICATE_ALIASES,
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
def test_refuses_kwargs_only_predicate_without_calling_it():
    """Not callable from a spec instance, so the builder has no usable predicate:
    the pack refuses it, and never guesses a spec-field-to-kwarg mapping.

    The real supports_tiled_2d takes head_size/block_size/dtype individually as
    keyword-only args; mapping them would validate something other than what gets
    built.
    """
    calls = []

    def supports_tiled_2d_spec(*, spec: _Spec, head_size, block_size, dtype):
        calls.append(spec)

    module = _Mod(
        builders=("build_unified_attention_2d_tiled",),
        supports_tiled_2d_spec=supports_tiled_2d_spec,
    )

    with pytest.raises(HkpPackError, match="build_unified_attention_2d_tiled"):
        _check_support_predicate(
            module, "build_unified_attention_2d_tiled", _Spec(n=1), "gfx942"
        )
    assert calls == []


@pytest.mark.quick
def test_refuses_builder_with_no_predicate():
    """A builder whose spec cannot be validated is refused, not packed unchecked.
    The message names the builder, its module and the arch."""
    module = _Mod()
    module.__name__ = "kernels.fake_module"

    with pytest.raises(HkpPackError) as exc:
        _check_support_predicate(module, "build_x", _Spec(n=1), "gfx950")
    message = str(exc.value)
    assert "build_x" in message
    assert "kernels.fake_module" in message
    assert "gfx950" in message


@pytest.mark.quick
@pytest.mark.parametrize(
    "exc_type",
    [AttributeError, TypeError, KeyError, ValueError, IndexError, RuntimeError],
)
def test_predicate_that_raises_fails_the_pack(exc_type):
    """A predicate that raises has no verdict, whatever it raises: packing on would
    ship the spec without the check, so the pack fails and names where it broke."""

    def is_valid_spec(spec: _Spec, *, arch="gfx942"):
        raise exc_type("no verdict")

    with pytest.raises(HkpPackError, match=f"raised {exc_type.__name__} at") as exc:
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )
    assert "is_valid_spec" in str(exc.value)
    assert "test_hkp_pack_producer_guards.py:" in str(exc.value)


@pytest.mark.quick
def test_predicate_with_positional_only_arch_fails_with_call_error():
    """A predicate whose arch parameter is positional-only triggers a TypeError
    at bind time, not inside the body.  The error must surface as a pack failure
    with a 'could not be called' message (not a 'raised ...' body failure)."""

    # positional-only arch: the packer always passes arch as a keyword
    def is_valid_spec(spec: _Spec, arch: str = "gfx942", /):
        return True, ""

    with pytest.raises(HkpPackError, match="could not be called") as exc:
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )
    assert "is_valid_spec" in str(exc.value)


@pytest.mark.quick
def test_reduce_tiled_builder_is_refused_not_matched_to_a_tiled_predicate():
    """The reduce spec lacks the fields supports_tiled_3d_spec reads, so that
    predicate must neither be called for it nor stand in for a missing one."""
    calls = []

    def supports_tiled_3d_spec(spec: _Spec, *, arch="gfx942"):
        calls.append(spec)
        return True, ""

    module = _Mod(
        builders=("build_unified_attention_reduce_tiled",),
        supports_tiled_3d_spec=supports_tiled_3d_spec,
    )

    with pytest.raises(HkpPackError, match="build_unified_attention_reduce_tiled"):
        _check_support_predicate(
            module, "build_unified_attention_reduce_tiled", _Spec(n=1), "gfx942"
        )
    assert calls == []


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


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "rocke" / "library").is_dir(),
    reason="rocke/library not in this tree",
)
def test_predicate_aliases_resolve_to_real_predicates():
    """Each _PREDICATE_ALIASES entry must name a callable in the real library.

    The alias map exists so builders whose predicates don't follow the naming
    convention still get pre-flight validation.  If a predicate is renamed, the
    alias becomes stale and hkp_pack silently skips validation for that builder.
    This test makes stale aliases a test failure rather than a silent regression.
    """
    import kernels.gfx942.attention_tiled_2d as gfx942_2d
    import kernels.gfx942.attention_tiled_3d as gfx942_3d
    import kernels.gfx950.attention_tiled_2d as gfx950_2d
    import kernels.gfx950.attention_tiled_3d as gfx950_3d

    # Each entry: alias_key -> (predicate_name, list of modules that must have it)
    expected = {
        "build_unified_attention_2d_tiled": (
            "supports_tiled_2d_spec",
            [gfx942_2d, gfx950_2d],
        ),
        "build_gfx942_4warp_gqa": (
            "supports_gfx942_4warp_gqa_spec",
            [gfx942_2d],
        ),
        "build_unified_attention_3d_tiled": (
            "supports_tiled_3d_spec",
            [gfx942_3d, gfx950_3d],
        ),
    }
    for builder, (predicate_name, modules) in expected.items():
        # Alias must be present in the map
        assert (
            builder in _PREDICATE_ALIASES
        ), f"_PREDICATE_ALIASES is missing the entry for {builder!r}"
        assert _PREDICATE_ALIASES[builder] == predicate_name, (
            f"_PREDICATE_ALIASES[{builder!r}] = {_PREDICATE_ALIASES[builder]!r}, "
            f"expected {predicate_name!r}"
        )
        # Predicate must be a callable in every expected module
        for mod in modules:
            fn = getattr(mod, predicate_name, None)
            assert callable(fn), (
                f"_PREDICATE_ALIASES[{builder!r}] = {predicate_name!r} "
                f"but {predicate_name!r} is not callable in {mod.__name__}"
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
