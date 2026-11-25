# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The settings templates' golden texts (tests/settings_template_texts.py), made from the writer: each template the
package gives you (rkmon_lib/settings_templates.py: `templates`), as rkmon_lib/settings_file.py writes it (`template`),
under its label, with the file's VERSION kept as the deploy file's image.tag. The file is one of the generated files
(tools/generated.py, FILES): when a schema's text changes, tests/test_settings_templates.py finds the writer's texts
out of step with it, and `python3 -B tools/make_package.py --refresh-generated` writes it again; review the change
and commit it.
"""

import dataclasses
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import file_types, settings_file, settings_templates  # noqa: E402

VERSION = re.compile(r'^VERSION = "([^"]*)"$', re.M)
HEAD = '''\
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The settings templates' texts, golden: the build file's, the deploy file's for each of its three profiles and its
every-key text, as rkmon_lib/settings_file.py writes them from the schemas in schema/, with VERSION, the package's
version, as the deploy file's image.tag. TEXTS has each template's text under its label, the one
rkmon_lib/settings_templates.py gives it; tests/test_settings_templates.py compares each with the text the writer
writes. `python3 -B tools/make_package.py --refresh-generated` writes this file again from the writer
(tools/template_texts.py), VERSION kept: when a schema's text changes, these texts change with it.
"""

VERSION = "{version}"
'''


def pinned(file_type, version: str):
    """FILE_TYPE, with VERSION as its templates' image.tag when it is the deploy file type, whose templates give the
    package's own version."""
    if file_type is not file_types.DEPLOY:
        return file_type
    return dataclasses.replace(
        file_type, template_values=lambda: {"image.tag": version}
    )


def texts(version: str) -> dict[str, str]:
    """Each template's label -> its text as the writer writes it, with VERSION as the deploy file's image.tag."""
    return {
        t.label: settings_file.template(
            pinned(t.file_type, version), t.profile, t.every_key
        )
        for t in settings_templates.templates()
    }


def name(label: str) -> str:
    """A template's name in the file, from its label: "deploy review-only" -> DEPLOY_REVIEW_ONLY."""
    return re.sub(r"[ -]", "_", label).upper()


def file_text(version: str) -> str:
    """The golden file whole: its header, VERSION, each template's text as a constant, then TEXTS."""
    found = texts(version)
    out = [HEAD.replace("{version}", version)]
    for label, text in found.items():
        if "\\" in text or '"""' in text or not text.endswith("\n"):
            raise ValueError(f"the {label} template cannot be a plain text constant")
        out.append(f'\n{name(label)} = """\\\n{text}"""\n')
    out.append("\nTEXTS = {\n")
    out += [f'    "{label}": {name(label)},\n' for label in found]
    out.append("}\n")
    return "".join(out)
