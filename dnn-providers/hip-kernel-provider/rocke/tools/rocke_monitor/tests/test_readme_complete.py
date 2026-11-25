# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""README.md's Settings reference names each part of the deploy settings file and leads to docs/internal_settings.md,
the one list of every other configuration key with its meaning and its default (made from the registry,
rkmon_lib/registry.py), which this checks too: a new setting fails here until the documents name it.
"""

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

# Settings whose value is a map of data, not settings of their own: the map's key must appear, its entries need not.
MAPS = {"codeowner_teams", "hot_spots", "gpu.env", "jira.accounts"}


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def code_spans(text: str) -> list[str]:
    """A document's inline code spans (fenced blocks aside)."""
    text = re.sub(r"(?ms)^```.*?^```", "", text)
    return re.findall(r"`([^`\n]+)`", text)


def setting_keys() -> list[str]:
    """Every key of the configuration (the registry's), dotted, and the groups around each advanced key (one no key of
    the deploy settings file feeds: the schema describes the others' parts); a map is one key, its entries aside
    (MAPS)."""
    from rkmon_lib import registry

    out = []
    for k in registry.keys():
        if k.owner != registry.DEPLOY:
            parts = k.name.split(".")
            out += [".".join(parts[:i]) for i in range(1, len(parts))]
        out.append(k.name)
    return list(dict.fromkeys(out))


def named(token: str, spans: list[str]) -> bool:
    """Is `token` a whole word of some code span? Neither a dotted key's last part nor its group is enough."""
    pat = re.compile(rf"(?<![\w.-]){re.escape(token)}(?![\w.-])")
    return any(pat.search(s) for s in spans)


class TestTheSettingsReference(unittest.TestCase):
    def test_every_setting(self):
        from rkmon_lib import deploy_tree

        ref = readme().split("\n## Settings reference\n", 1)[1].split("\n## ", 1)[0]
        self.assertIn("(docs/internal_settings.md)", ref)
        for part in deploy_tree.parts():
            self.assertTrue(named(part, code_spans(ref)), part)
        internal = (ROOT / "docs" / "internal_settings.md").read_text(encoding="utf-8")
        spans = code_spans(internal)
        keys = [k for k in setting_keys() if deploy_tree.file_key(k) is None]
        self.assertIn("worker.temperature", keys)
        missing = [k for k in keys if not named(k, spans)]
        self.assertEqual(
            missing, [], "configuration keys docs/internal_settings.md does not name"
        )

    def test_the_maps_are_still_keys(self):
        from rkmon_lib import registry

        self.assertEqual(
            sorted(m for m in MAPS if registry.kind(m) != registry.KEY),
            [],
            "a map the configuration no longer has as one key",
        )

    def test_the_check_catches_a_missing_name(self):
        spans = code_spans(
            "Run `rk tick-filter --rc 0` and set `review.pipes`.\n```\nrk ghost\n```\n"
        )
        self.assertTrue(named("review.pipes", spans))
        self.assertFalse(
            named("review", spans)
        )  # a group is not named by its member's key
        self.assertFalse(named("pipes", spans))
        self.assertFalse(
            any("ghost" in s for s in spans)
        )  # a fenced block is not a mention


if __name__ == "__main__":
    unittest.main()
