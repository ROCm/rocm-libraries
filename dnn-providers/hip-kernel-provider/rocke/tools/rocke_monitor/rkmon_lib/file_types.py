# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's three file types, each a header kind, its layout, its schema's file, its checks across
keys, its steps from earlier layouts, and the commands that write a new file of the kind and print a section of it in
full, as `rkmon help` shows them:

- `DEPLOY`: deploy settings, layout 2, schema/deploy_settings.schema.json; coherence.deploy; its step from
  layout 1, rkmon_lib/from_layout1.py's `deploy`; deploy step 1's command ("template deploy [--profile PROFILE]
  FILE"), and the section command's form in the command table ("template section KEY"); its template values
  that only the package knows, `package_values()`: image.tag, the package's version.
- `BUILD`: build settings, layout 2, schema/build_settings.schema.json; coherence.build; its step from layout
  1, rkmon_lib/from_layout1.py's `build`; build step 1's command ("template build FILE"), and no section
  command: its template writes every key; its template values that only the package knows, `build_values()`:
  rocm.version, the ROCm release the review gates are tuned on (build_plan.ROCM_RELEASE, the number's one home).
- `INTERNAL_STATE`: the monitor's internal state, layout 1, schema/internal_state.schema.json; no checks across keys,
  and neither command, since the monitor writes the file.

`KINDS` and `OLD_HEADERS` are the headers that make a text a settings file of either kind, deploy or build: its kind,
or its first key in the layout before layout 1 (DEPLOY_HEADER_BEFORE_LAYOUT_1, BUILD_HEADER_BEFORE_LAYOUT_1), which no
reader takes (migrate.before_layout_1). settings.header_line finds them, and the privacy check refuses a file with
one, whatever the file's name.

A file type reads its schema at its first use (`FileType.load`, rkmon_lib/settings_file.py), never on import.
"""

from __future__ import annotations

from . import build_plan, coherence, commands, from_layout1, image, steps
from .paths import PACKAGE
from .settings_file import FileType

SCHEMAS = PACKAGE / "schema"


def package_values() -> dict:
    """The deploy template's values that only the package knows: image.tag, this package's version."""
    return {"image.tag": image.package_version(PACKAGE)}


def build_values() -> dict:
    """The build template's values that only the package knows: rocm.version, the ROCm release the review gates are
    tuned on (build_plan.ROCM_RELEASE), which its description names too ({x-template}).
    """
    return {"rocm.version": build_plan.ROCM_RELEASE}


DEPLOY = FileType(
    "deploy_settings",
    2,
    SCHEMAS / "deploy_settings.schema.json",
    coherence=coherence.deploy,
    upgrades={1: from_layout1.deploy},
    template_command=steps.DEPLOY_STEPS[0].command,
    section_command=commands.find(("template", "section"), commands.TABLE).form,
    template_values=package_values,
)
BUILD = FileType(
    "build_settings",
    2,
    SCHEMAS / "build_settings.schema.json",
    coherence=coherence.build,
    upgrades={1: from_layout1.build},
    template_command=steps.BUILD_STEPS[0].command,
    template_values=build_values,
)
INTERNAL_STATE = FileType("internal_state", 1, SCHEMAS / "internal_state.schema.json")

KINDS = (DEPLOY.kind, BUILD.kind)
# The first key of a file in the layout before layout 1: a deploy settings file's, which held every setting under
# config:, the image's among them; a build settings file's.
DEPLOY_HEADER_BEFORE_LAYOUT_1 = "rkmon_settings"
BUILD_HEADER_BEFORE_LAYOUT_1 = "rkmon_build_settings"
OLD_HEADERS = (DEPLOY_HEADER_BEFORE_LAYOUT_1, BUILD_HEADER_BEFORE_LAYOUT_1)
