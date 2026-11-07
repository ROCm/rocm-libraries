# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection policy shared by explicit gate launchers and pytest preflight."""

from __future__ import annotations

__all__ = ["preflight_is_disabled", "selection_is_partial"]

_PREFLIGHT_BYPASS_ARGS = ("--noconftest", "--confcutdir")
_SELECTION_ARGS = (
    "-k",
    "-m",
    "--collect-only",
    "--co",
    "--ignore",
    "--ignore-glob",
    "--deselect",
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
)
_SHORT_SELECTION_ARGS = ("-k", "-m")


def selection_is_partial(arguments: list[str]) -> bool:
    """Whether pytest arguments can suppress part or all of an explicit gate."""
    return any("::" in argument for argument in arguments) or any(
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


def preflight_is_disabled(arguments: list[str]) -> bool:
    """Whether pytest arguments suppress an explicit gate's conftest."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
        for option in _PREFLIGHT_BYPASS_ARGS
    )
