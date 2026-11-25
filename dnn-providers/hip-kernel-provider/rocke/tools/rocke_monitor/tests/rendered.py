# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What checks.render and outcome.finish print, read the same whatever the temporary folder's path is; and the next
line the tests meet most, state 2's."""

from rkmon_lib import wrapping
from rkmon_lib.checks import WIDTH


def one_line(text: str) -> str:
    """`text` on one line, its words one space apart: a row reads the same wherever its lines break. checks.render
    wraps a row at checks.WIDTH columns, so a longer temporary folder's path in it moves the breaks.
    """
    return " ".join(text.split())


def finished(*lines: str) -> str:
    """The outcome line and the next line as outcome.finish prints them: each wrapped at a check row's width,
    so a longer temporary folder's path moves their breaks the same way here as in the command.
    """
    return "".join(f"{ln}\n" for text in lines for ln in wrapping.wrap(text, WIDTH))


def rewrapped(text: str) -> str:
    """TEXT, whole lines of an outcome (the outcome line, the next line), as outcome.finish prints them: each line
    longer than a check row's width wrapped at it (finished), so an expected line that names the temporary folder
    reads the same whatever that folder's path's length. A line that fits stays as it is.
    """
    return "\n".join(
        "\n".join(wrapping.wrap(line, WIDTH)) if len(line) > WIDTH else line
        for line in text.split("\n")
    )


def state_2_next(program: str) -> str:
    """State 2's next line, a folder with no deploy settings recorded, as PROGRAM names the commands."""
    return (
        f"next: step 1, template: {program} template deploy ~/rkmon-deploy.yaml (with a file of your own: "
        f"{program} check --settings FILE)"
    )
