# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection policy shared by explicit gate launchers and pytest preflight."""

from __future__ import annotations

__all__ = ["preflight_is_disabled", "selection_is_partial"]

_ALLOWED_GATE_PLUGINS = frozenset({"pytest_timeout", "tests.gates.conftest"})

_PREFLIGHT_BYPASS_ARGS = ("--noconftest", "--confcutdir")
_SELECTION_ARGS = (
    "-k",
    "-m",
    "--collect-only",
    "--co",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "-o",
    "--override-ini",
    "-c",
    "--lf",
    "--last-failed",
    "--ff",
    "--failed-first",
    "--nf",
    "--new-first",
    "--setup-only",
    "--setup-plan",
    "--stepwise",
    "--sw",
    "--stepwise-skip",
    "--rocke-reference-arch",
    "--rocke-reference-operation",
    "--rocke-reference-variant",
    "--gdn-batch",
    "--gdn-spec-id",
    "--rocke-seam-arches",
)
_SHORT_SELECTION_ARGS = ("-k", "-m", "-o", "-c")


def _loads_unapproved_plugin(arguments: list[str]) -> bool:
    for index, argument in enumerate(arguments):
        plugin = None
        if argument == "-p" and index + 1 < len(arguments):
            plugin = arguments[index + 1]
        elif argument.startswith("-p") and argument != "-p":
            plugin = argument[2:]
        if (
            plugin
            and not plugin.startswith("no:")
            and plugin not in _ALLOWED_GATE_PLUGINS
        ):
            return True
    return False


def selection_is_partial(arguments: list[str]) -> bool:
    """Whether pytest arguments can suppress part or all of an explicit gate."""
    return (
        _loads_unapproved_plugin(arguments)
        or any("::" in argument for argument in arguments)
        or any(
            argument == option
            or argument.startswith(option + "=")
            or (
                option in _SHORT_SELECTION_ARGS
                and argument.startswith(option)
                and argument != option
            )
            for argument in arguments
            for option in _SELECTION_ARGS
        )
    )


def preflight_is_disabled(arguments: list[str]) -> bool:
    """Whether pytest arguments suppress an explicit gate's conftest."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
        for option in _PREFLIGHT_BYPASS_ARGS
    )
