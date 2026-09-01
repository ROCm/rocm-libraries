# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Structural guard for the builder-signature contract (see rocke/AGENTS.md
# "Hard rules" and platform/dsl_docs/development/extending.md 3.6):
#
#     a builder takes the spec and the arch, and nothing else
#
# Concretely: every `build_*` defined under `library/kernels/` must accept
# `arch`, and must accept no parameter other than `spec` and `arch`. An
# arch-specific knob belongs in the spec dataclass -- as a field on an
# arch-specific subclass if it only applies to one arch -- never as a third
# builder parameter.
#
# Why this is worth a test rather than a convention:
#
# A builder that grows an extra parameter still works perfectly for everyone
# calling it by hand. It only breaks the machinery that has to describe a
# kernel WITHOUT calling it -- the descriptor format, and the packager that
# hydrates a spec from that descriptor. Those live downstream, so the failure
# surfaces as "this arch needs its own file format", months later, far from the
# commit that caused it. `gfx942`'s `build_attention_dense(spec, tuning, *,
# arch)` is exactly that story. This test moves the failure to authoring time,
# in rocke's own CI, on the commit that introduces it.
#
# The walk is over SUBMODULES, not over the top-level `kernels` namespace, on
# purpose: re-exporting a builder in `kernels/__init__.py` is optional, so a
# namespace-only walk would let a new builder in a new file escape the contract
# simply by not being re-exported.
#
# There is no allowlist. Every builder under `library/kernels/` satisfies the
# contract today, including the arch-neutral `attention_unified` ones: a builder
# that ignores `arch` still takes it, so a descriptor can name a target without
# calling the builder.
#
# The second half of this file guards the OTHER way a descriptor breaks. Holding
# the signature fixed is only half the contract: the descriptor stores the spec's
# fields, and the packager hydrates the spec back out of them --
# `Gfx942AttentionDenseSpec(**fields_from_the_file)`. A descriptor written today
# lists only the fields that existed the day it was written, so a spec field
# added later without a default makes that call raise `TypeError: missing
# required argument` for every descriptor already in the wild. `REQUIRED_FIELDS`
# freezes the set that is allowed to be mandatory -- the problem shape -- so a
# new tuning knob has to arrive defaulted.

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import typing

import kernels
import pytest

ALLOWED_PARAMS = {"spec", "arch"}

# Fields a spec is allowed to demand at construction time, frozen per class.
#
# These are problem-shape fields: a descriptor cannot omit them, because without
# them there is no kernel to describe. Everything else must carry a default, so
# that a descriptor written before the field existed still hydrates.
#
# Adding a required field here is not a rubber stamp -- it invalidates every
# descriptor and AOT pack already produced for that spec. Prefer a default. A
# default of `None` meaning "resolve through the shipping policy" is the strongest
# form: a descriptor that omits the field then auto-tracks whatever ships, rather
# than freezing the value that happened to be the default the day it was written.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "kernels.common._fmha_common.FmhaCommonSpec": ("shape",),
    "kernels.common._fmha_common.FmhaShape": (
        "head_size",
        "num_kv_heads",
        "num_query_heads",
    ),
    "kernels.common.attention_unified.UnifiedAttention2DSpec": ("problem",),
    "kernels.common.attention_unified.UnifiedAttention3DSpec": ("problem",),
    "kernels.common.attention_unified.UnifiedAttentionProblem": (
        "block_size",
        "dtype",
        "head_size",
        "max_seqlen_k",
        "max_seqlen_q",
        "num_kv_heads",
        "num_query_heads",
        "num_seqs",
        "total_q",
    ),
    "kernels.common.attention_unified.UnifiedAttentionReduceSpec": (
        "num_segments",
        "problem",
    ),
    "kernels.common.fmha_appendkv.FmhaAppendKvSpec": ("batch", "common"),
    "kernels.common.fmha_bwd.FmhaBwdSpec": ("common", "seqlen_k", "seqlen_q"),
    "kernels.common.fmha_fwd_fp8.FmhaFwdFp8Spec": ("common",),
    "kernels.common.fmha_head_grouping.FmhaFwdHeadGroupingSpec": (
        "common",
        "seqlen_k",
        "seqlen_q",
    ),
    "kernels.common.fmha_mfma.FmhaMfmaSpec": ("common", "seqlen_k", "seqlen_q"),
    "kernels.common.fmha_paged_prefill.FmhaFwdPagedPrefillSpec": (
        "batch",
        "common",
        "max_blocks_per_seq",
        "page_block_size",
    ),
    "kernels.common.fmha_splitkv_decode.FmhaFwdSplitKvDecodeSpec": (
        "batch",
        "common",
        "num_segments",
    ),
    "kernels.common.fmha_varlen.FmhaFwdVarlenSpec": (
        "batch",
        "common",
        "max_seqlen_k",
        "max_seqlen_q",
    ),
    "kernels.common.sage_attention.SageAttentionSpec": (
        "common",
        "k_scale",
        "q_scale",
        "quant_mode",
        "seqlen_k",
        "seqlen_q",
    ),
    "kernels.common.sparse_attention.JengaSparseSpec": (
        "common",
        "seqlen_k",
        "seqlen_q",
    ),
    "kernels.common.sparse_attention.VsaSparseSpec": ("common", "seqlen_k", "seqlen_q"),
    "kernels.gfx1151.wmma_fmha_fwd.WmmaFmhaFwdSpec": ("head_size", "num_query_heads"),
    "kernels.gfx1250.attention_tiled_2d.UnifiedAttention2DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx1250.attention_tiled_3d.UnifiedAttention3DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx1250.attention_tiled_3d.UnifiedAttentionReduceTiledSpec": (
        "dtype",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
    ),
    "kernels.gfx1250.wmma_attention_fwd.WmmaAttentionFwdSpec": (
        "head_size",
        "num_query_heads",
    ),
    # The gfx942 dense spec is a subclass of gfx950's and inherits its required
    # set; every gfx942-only knob it adds is defaulted, four of them to `None`.
    "kernels.gfx942.attention_dense.Gfx942AttentionDenseSpec": (
        "batch",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "seqlen_kv",
        "seqlen_q",
    ),
    "kernels.gfx942.attention_tiled_2d.UnifiedAttention2DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx942.attention_tiled_3d.UnifiedAttention3DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx942.attention_tiled_3d.UnifiedAttentionReduceTiledSpec": (
        "dtype",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
    ),
    # gfx942's dense spec is a subclass of gfx950's and inherits its required set;
    # every gfx942-only knob it adds is defaulted, four of them to `None`.
    "kernels.gfx950.attention_dense.AttentionDenseSpec": (
        "batch",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "seqlen_kv",
        "seqlen_q",
    ),
    "kernels.gfx950.attention_tiled_2d.UnifiedAttention2DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx950.attention_tiled_3d.UnifiedAttention3DTiledSpec": (
        "block_size",
        "dtype",
        "has_softcap",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
        "sliding_window",
        "use_sinks",
    ),
    "kernels.gfx950.attention_tiled_3d.UnifiedAttentionReduceTiledSpec": (
        "dtype",
        "head_size",
        "num_kv_heads",
        "num_query_heads",
        "num_segments",
    ),
    # Reached through a spec field, so a descriptor has to express it too.
    "rocke.helpers.qk_scale.QkScaleSpec": ("layout",),
}


def _discover():
    """Return ({qualname: signature}, [import errors]) for kernels' builders.

    A function is attributed to the module that DEFINES it (`__module__`), so a
    builder re-exported through several packages is still checked exactly once.
    """
    builders = {}
    errors = []
    for info in pkgutil.walk_packages(kernels.__path__, kernels.__name__ + "."):
        try:
            mod = importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(f"  {info.name}: {type(exc).__name__}: {exc}")
            continue
        for name, obj in vars(mod).items():
            if not name.startswith("build_"):
                continue
            if not inspect.isfunction(obj) or obj.__module__ != mod.__name__:
                continue
            builders[f"{mod.__name__}.{name}"] = inspect.signature(obj)
    return builders, errors


def _discover_specs():
    """Return {qualname: class} for every spec class a descriptor has to express.

    Roots are the `spec` annotations of the discovered builders; from there the
    walk follows dataclass-typed fields, because a nested spec (`FmhaCommonSpec`
    under `common`, `UnifiedAttentionProblem` under `problem`) is just as much a
    thing the descriptor stores and the packager hydrates.

    Annotations are resolved with `get_type_hints` rather than read raw: most
    modules use `from __future__ import annotations`, so `f.type` is a string,
    and the same string (`"UnifiedAttention2DTiledSpec"`) names a DIFFERENT class
    per arch. Resolving against the owning module is what keeps those distinct.

    Names starting with `_` are skipped: those are internal resolved-config
    structs (`_ResolvedTiled3D` and friends) built by the builder after it has
    the spec. No descriptor ever names one, so their fields are not a
    compatibility surface.
    """
    builders, _ = _discover()
    pending = []
    for qualname in builders:
        mod_name, _, fn_name = qualname.rpartition(".")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        spec_cls = typing.get_type_hints(fn).get("spec")
        if dataclasses.is_dataclass(spec_cls):
            pending.append(spec_cls)

    found = {}
    while pending:
        cls = pending.pop()
        if cls.__qualname__.startswith("_"):
            continue
        key = f"{cls.__module__}.{cls.__qualname__}"
        if key in found:
            continue
        found[key] = cls
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            nested = hints.get(field.name)
            if isinstance(nested, type) and dataclasses.is_dataclass(nested):
                pending.append(nested)
    return found


def _required_fields(cls) -> tuple[str, ...]:
    return tuple(
        sorted(
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        )
    )


def test_every_kernels_submodule_imports() -> None:
    """A submodule that fails to import silently shrinks this guard's coverage.

    Without this, an ImportError anywhere under `kernels/` would quietly remove
    that file's builders from the contract check and the suite would stay green.
    """
    _, errors = _discover()
    assert not errors, "kernels submodules failed to import:\n" + "\n".join(errors)


def test_builders_were_discovered() -> None:
    """Guard against the walk silently finding nothing (bad path, renamed pkg)."""
    builders, _ = _discover()
    assert builders, "no build_* functions found under kernels/ -- walk is broken"


def test_builders_take_only_spec_and_arch() -> None:
    """Every builder's parameters are a subset of {spec, arch}, and include arch.

    `arch` may be positional-or-keyword or keyword-only, and may carry a default;
    what is forbidden is a THIRD parameter, because that is the thing a kernel
    descriptor cannot express. Passing `arch` keyword-only is preferred for new
    builders but is not enforced here -- most of the tree predates that style and
    the ordering has never been the bug.
    """
    builders, _ = _discover()

    violations = []
    for qualname, sig in sorted(builders.items()):
        params = set(sig.parameters)
        extra = sorted(params - ALLOWED_PARAMS)
        problems = []
        if extra:
            problems.append(f"extra parameter(s) {extra} -- move them into the spec")
        if "arch" not in params:
            problems.append("no 'arch' parameter")
        if problems:
            violations.append(f"  {qualname}{sig}\n      {'; '.join(problems)}")

    assert not violations, (
        "builder(s) break the (spec, arch) contract.\n"
        "A builder takes the spec and the arch and nothing else; an arch-specific\n"
        "knob is a field on an arch-specific spec subclass, not a third parameter.\n"
        "A builder whose body is arch-neutral still takes `arch` -- validate it and\n"
        "ignore it; the uniform shape is what a kernel descriptor depends on.\n"
        + "\n".join(violations)
    )


def test_spec_is_the_first_parameter() -> None:
    """The spec is positional-first everywhere, so `build_x(spec)` always reads."""
    builders, _ = _discover()
    wrong = [
        f"  {q}{s}"
        for q, s in sorted(builders.items())
        if s.parameters and next(iter(s.parameters)) != "spec"
    ]
    assert not wrong, "builder(s) whose first parameter is not 'spec':\n" + "\n".join(
        wrong
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_dense_builders_share_one_signature(arch: str) -> None:
    """The case that motivated this guard: dense must not diverge by arch again.

    gfx942 and gfx950 `build_attention_dense` must be interchangeable in shape.
    The specs stay different in CONTENT -- gfx942's is a subclass carrying extra
    fields -- but the call shape must not differ, or a descriptor format has to
    branch on arch.
    """
    mod = importlib.import_module(f"kernels.{arch}.attention_dense")
    sig = inspect.signature(mod.build_attention_dense)
    assert list(sig.parameters) == ["spec", "arch"], (
        f"kernels.{arch}.attention_dense.build_attention_dense{sig} must be "
        f"(spec, *, arch)"
    )
    assert (
        sig.parameters["arch"].kind is inspect.Parameter.KEYWORD_ONLY
    ), f"'arch' must be keyword-only on the dense builders, got {sig}"


def test_specs_were_discovered() -> None:
    """Guard against the annotation walk silently finding nothing."""
    specs = _discover_specs()
    assert specs, "no spec classes reached from builder annotations -- walk is broken"


def test_every_spec_class_is_listed() -> None:
    """A new spec class has to declare its required set, not inherit silence."""
    found = set(_discover_specs())
    listed = set(REQUIRED_FIELDS)
    assert found == listed, (
        "REQUIRED_FIELDS is out of sync with the specs reachable from builders.\n"
        f"  not listed: {sorted(found - listed)}\n"
        f"  listed but gone: {sorted(listed - found)}\n"
        "Add the new spec with its required (non-defaulted) fields, keeping that\n"
        "set to the problem shape."
    )


def test_spec_required_fields_are_frozen() -> None:
    """New spec fields must be defaulted, so old descriptors still hydrate.

    The descriptor stores a spec's fields and the packager hydrates the spec back
    out of them. A descriptor written today lists only today's fields, so adding
    a field WITHOUT a default turns every existing descriptor into a `TypeError:
    missing required argument`.

    Python only catches part of this by accident: a dataclass rejects a
    non-defaulted field declared AFTER a defaulted one, so appending a required
    field to a spec whose tail is defaulted fails at import. Inserting one before
    the defaulted block is perfectly legal Python -- and breaks every descriptor
    in the wild. This test does not depend on field order.
    """
    drift = []
    for qualname, cls in sorted(_discover_specs().items()):
        expected = REQUIRED_FIELDS.get(qualname)
        if expected is None:
            continue  # reported by test_every_spec_class_is_listed
        actual = _required_fields(cls)
        if actual == expected:
            continue
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))
        detail = []
        if added:
            detail.append(f"newly required {added}")
        if removed:
            detail.append(f"no longer required {removed}")
        drift.append(f"  {qualname}: {'; '.join(detail)}")

    assert not drift, (
        "spec required-field set(s) changed.\n"
        "A field with no default cannot be omitted, so every kernel descriptor and\n"
        "AOT pack written before it existed now fails to hydrate the spec. Give the\n"
        "field a default -- the currently-shipped value, or `None` where a policy\n"
        "function resolves it, which lets an old descriptor auto-track what ships.\n"
        "If the field genuinely belongs to the problem shape, update REQUIRED_FIELDS\n"
        "in this file and say in the PR that existing descriptors are invalidated.\n"
        + "\n".join(drift)
    )
