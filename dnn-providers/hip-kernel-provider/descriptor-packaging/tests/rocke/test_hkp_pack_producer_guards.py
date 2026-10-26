"""Producer correctness guards: the packer must refuse what it cannot express.

Each guard here defends against a *silent* wrong result rather than a crash --
a kernel that packs successfully while being tuned differently, validated not at
all, or lowered by a different engine than the artifact claims. Those are worse
than failures because the output looks fine.
"""

import dataclasses
import json
import textwrap
import typing
from importlib import import_module
from pathlib import Path

import pytest

from hkp_pack.errors import HkpPackError
from hkp_pack.rocke_compile import (
    ROCKE_TARGET_NATIVE,
    _check_support_predicate,
    _module_from_source,
    _require_spec_arch_signature,
    _resolve_spec_class,
    build_spec,
    rocke_compile_arch,
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

    with pytest.raises(HkpPackError, match="build_tuned.*tuning"):
        _require_spec_arch_signature(build_tuned, "build_tuned")


@pytest.mark.quick
def test_rejects_var_keyword_parameter():
    def build_kwargs(spec: _Spec, *, arch: str = "gfx942", **rest):
        return ("kernel", spec, arch, rest)

    with pytest.raises(HkpPackError, match="build_kwargs.*rest"):
        _require_spec_arch_signature(build_kwargs, "build_kwargs")


@pytest.mark.quick
def test_rejects_extra_positional_parameter():
    def build_two(spec: _Spec, other, *, arch: str = "gfx942"):
        return ("kernel", spec, other, arch)

    with pytest.raises(HkpPackError, match="build_two"):
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

    with pytest.raises(HkpPackError, match="is_valid_spec.*build_x"):
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

    with pytest.raises(HkpPackError, match="supports_attention_dense"):
        _check_support_predicate(module, "build_attention_dense", _Spec(n=1), "gfx950")


@pytest.mark.quick
def test_resolves_predicate_by_alias_for_tiled_family():
    # The tiled family's predicates do not follow the naming convention; without
    # the alias map the pack of this builder fails with no resolvable predicate.
    def supports_tiled_2d_spec(spec: _Spec, *, arch="gfx942"):
        return (False, "tiled no")

    module = _Mod(
        builders=("build_unified_attention_2d_tiled",),
        supports_tiled_2d_spec=supports_tiled_2d_spec,
    )

    with pytest.raises(HkpPackError, match="supports_tiled_2d_spec"):
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

    with pytest.raises(HkpPackError, match="is_valid_spec_x"):
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

    with pytest.raises(HkpPackError, match=f"is_valid_spec.*{exc_type.__name__}"):
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )


@pytest.mark.quick
def test_predicate_with_positional_only_arch_fails_with_call_error():
    """A predicate whose arch parameter is positional-only triggers a TypeError
    at bind time, not inside the body.  The error must surface as a pack failure
    naming the predicate, and the body must never run."""
    calls = []

    # positional-only arch: the packer always passes arch as a keyword
    def is_valid_spec(spec: _Spec, arch: str = "gfx942", /):
        calls.append(spec)
        return True, ""

    with pytest.raises(HkpPackError, match="is_valid_spec.*build_x"):
        _check_support_predicate(
            _Mod(is_valid_spec=is_valid_spec), "build_x", _Spec(n=1), "gfx942"
        )
    assert calls == []


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


_DESCRIPTOR_TREE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
)


def _shipped_rocke_checks():
    """``(source, builder, target, arch, spec)`` for one spec of every distinct rocKE
    ``(source, builder, target, arch)`` the shipped descriptors pack.

    ``arch`` is each shard architecture the kernel is packed for: its own ``arch``
    list when it narrows one, otherwise its pack's.
    """
    seen = {}
    for path in sorted(_DESCRIPTOR_TREE.rglob("*.kdp.json")):
        pack = json.loads(path.read_text())
        for kernel in pack.get("kernelDescriptors", []):
            # A multi-launch kernel (RFC 0017 section 15.1) names its builder in each
            # `sources` entry instead of `kernel_source`.
            for source in kernel.get("sources", {}).values() or [
                kernel["kernel_source"]
            ]:
                if source.get("kind") != "rocke":
                    continue
                target = source.get("target", ROCKE_TARGET_NATIVE)
                for arch in kernel.get("arch") or pack.get("arch", []):
                    key = (source["source"], source["builder"], target, arch)
                    seen.setdefault(key, source["spec"])
    return [(*key, spec) for key, spec in seen.items()]


@pytest.mark.skipif(not _HAS_ROCKE_LIBRARY, reason="rocke/library not in this tree")
def test_every_shipped_rocke_builder_passes_the_pack_time_predicate_check():
    """Each rocKE builder the shipped descriptors pack resolves a support predicate
    and the predicate accepts that builder's shipped spec, through the same
    resolution and check the packer runs before it builds the kernel."""
    assert _DESCRIPTOR_TREE.is_dir(), f"no descriptor tree at {_DESCRIPTOR_TREE}"
    checks = _shipped_rocke_checks()
    assert checks, f"no rocKE kernels found under {_DESCRIPTOR_TREE}"
    for source, builder, target, arch, spec in checks:
        module = import_module(_module_from_source(source))
        spec_obj = build_spec(
            _resolve_spec_class(module, getattr(module, builder)), spec
        )
        try:
            _check_support_predicate(
                module, builder, spec_obj, rocke_compile_arch(arch, target)
            )
        except HkpPackError as exc:
            pytest.fail(f"{source}:{builder} @ {arch} ({target}): {exc}")


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
    with pytest.raises(HkpPackError, match="casual"):
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

    with pytest.raises(HkpPackError, match="Broken"):
        build_spec(ns["Broken"], {"field": 1})
