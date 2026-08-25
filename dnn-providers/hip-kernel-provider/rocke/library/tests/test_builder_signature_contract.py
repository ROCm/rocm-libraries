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

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import kernels

ALLOWED_PARAMS = {"spec", "arch"}


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
        if list(s.parameters) and list(s.parameters)[0] != "spec"
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
