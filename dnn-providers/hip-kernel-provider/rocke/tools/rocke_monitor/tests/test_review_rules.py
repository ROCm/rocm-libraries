# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The review rules the package ships agree with the operator's: the drafter's rules card is scope first (every
blocker needed to merge, no cap; significant follow-ups only), the follow-up cooldown defaults to 60 minutes wherever
a default is stated, line 1 with no first-pass drafter (the verifier reviews alone) carries the verifier's effort,
and docs/internal_settings.md states each plain default as config.defaults.json has it.
"""

import datetime as dt
import json
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import bindings, config, file_types, plan, schema  # noqa: E402
import home_repository  # noqa: E402

DEFAULTS = json.loads((ROOT / "config.defaults.json").read_text(encoding="utf-8"))


# A table cell's escapes of markup (tools/generated.py's cell), to undo
ESCAPED = re.compile(r"\\([\\`*~<|\[$_])")


def internal_defaults() -> dict:
    """docs/internal_settings.md's package defaults whose description closes with one JSON literal as the default
    ("Default: VALUE."): {key: value}."""
    out = {}
    text = (ROOT / "docs" / "internal_settings.md").read_text(encoding="utf-8")
    part = text.split("\n## Package defaults\n", 1)[1].split("\n## ", 1)[0]
    for m in re.finditer(r"^\| `([\w.-]+)` \| .*Default: (.+)\. \|$", part, re.M):
        value = ESCAPED.sub(r"\1", m.group(2)).replace("&amp;", "&")
        try:
            out[m.group(1)] = json.loads(value)
        except ValueError:
            continue
    return out


def default_of(key: str):
    node = DEFAULTS
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


class TestScopeFirst(unittest.TestCase):
    def test_the_drafters_rules_card_is_scope_first(self):
        card = (home_repository.FOUND.path.parent / "drafter_rules.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "Needed to merge: every net-new blocker at head, each lit (🔴, 🟠 or 🟡), with no cap.",
            card,
        )
        self.assertIn(
            "Highly recommended follow-up (not blocking): significant items only, each lit; no nits.",
            card,
        )
        import drafter

        for text in (
            card,
            drafter.GENERIC_RULES,
            (ROOT / "worker" / "drafter.py").read_text(encoding="utf-8"),
        ):
            self.assertNotIn("at most one 🟡", text)
            self.assertNotIn("only net-new 🔴/🟠", text.lower())


class TestDefaults(unittest.TestCase):
    def test_the_follow_up_gap_is_an_hour_everywhere(self):
        key = "features.review_automation.schedule.follow_up_cooldown_minutes"
        self.assertEqual(schema.node(file_types.DEPLOY.load(), key)["default"], 60)
        self.assertEqual(config.load_defaults()["followup_min_gap_min"], 60)
        self.assertRegex(
            (ROOT / "README.md").read_text(encoding="utf-8"),
            rf"(?m)^\| `{re.escape(key)}` \| .* Default: 60\. \|$",
        )
        self.assertEqual(plan.gap_label({}), "1-h")
        # the plan's reader waits the hour too, with nothing set: a post half an hour old holds the follow-up
        # until an hour after it
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        last = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)).strftime(
            fmt
        )
        end = dt.datetime.strptime(last, fmt) + dt.timedelta(minutes=60)
        self.assertEqual(plan.gap_until({}, {}, {"n": 7}, {7: last}), end.strftime(fmt))

    def test_the_internal_settings_plain_defaults_are_config_defaults(self):
        rows = internal_defaults()
        self.assertGreater(len(rows), 30)
        wrong = []
        for key, value in rows.items():
            try:
                have = default_of(key)
            except KeyError:
                continue
            if have != value:
                wrong.append(
                    f"{key}: docs/internal_settings.md {value!r}, "
                    f"config.defaults.json {have!r}"
                )
        self.assertEqual(wrong, [])


class TestLineOne(unittest.TestCase):
    def cfg(self, **effort):
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="o/r",
            rocke_path="rk",
            operator={"github_login": "me"},
            scope={"labels": ["x"]},
            effort=dict(c.get("effort") or {}, **effort),
        )
        return config.normalize(c)

    def test_no_draft_mode_names_the_verifiers_effort(self):
        c = self.cfg()
        self.assertTrue(bindings.no_draft(c))
        self.assertTrue(
            bindings.first_line(c).endswith(", effort: high"), bindings.first_line(c)
        )
        c = self.cfg(
            verify="max"
        )  # features.review_automation.passes.verification_pass.effort
        self.assertTrue(
            bindings.first_line(c).endswith(", effort: max"), bindings.first_line(c)
        )
        self.assertIn(bindings.verifier_name(c), bindings.first_line(c))
        built = bindings.first_line(c)
        self.assertTrue(
            built.startswith("Automated review by rocke-monitor: Model - "), built
        )
        # A line 1 the configuration gives is not read: it would give no automated reviewer name
        c["review_first_line_no_draft"] = "Automated review: Model - X, effort: low"
        self.assertEqual(bindings.first_line(c), built)


if __name__ == "__main__":
    unittest.main()
