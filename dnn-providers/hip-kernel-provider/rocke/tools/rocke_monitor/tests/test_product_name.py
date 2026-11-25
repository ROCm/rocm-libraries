# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The product's name in the package's files: the rocKE monitor, its package folder rocke_monitor, its image
rocke-monitor, its code in the image /opt/rocke_monitor and the web host's rocke-monitor.service. An earlier name of
the product (EARLIER) is in a file only where ALLOWED says why: a place that reads text the monitor wrote under an
earlier name, or an older monitor's names as a deployment moves from it, and the tests and the docs of those places.
"""

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

from test_headers import package_files  # noqa: E402

# Between two words of a name: "-" or "_" (an image, a folder, a unit), a space, or a line break with what continues
# a line after it in the package's texts (a comment's #, a quote's >, a string literal's closing and opening quotes).
GAP = r"""(?:[-_ \t]|[ \t]*["']?[ \t]*\n[ \t]*(?:\#+[ \t]*|>[ \t]*|f?["'])?)"""
# An earlier name of the product, in any case: rocKE PR monitor, rocKE merge monitor, rocKE PR merge monitor, the
# merge monitor or the PR monitor alone, and the names made of them (rocke-pr-monitor, rocke_pr_monitor,
# rocke-merge-monitor).
EARLIER = re.compile(
    rf"(?<![a-z0-9])(?:(?:rocke{GAP}+)?pr{GAP}+(?:merge{GAP}+)?|(?:rocke{GAP}+)?merge{GAP}+)monitor(?![a-z0-9])",
    re.I,
)
# The places that keep an earlier name: (the file, the words of its line, why). A name anywhere else is a finding,
# and an entry that keeps no name is one too.
ALLOWED = (
    (
        r"README\.md",
        r"^A deployment of the package rocke_pr_monitor, whose image is rocke-pr-monitor, needs no update in place",
        "README.md, \"Updating\": a deployment of the earlier package, by its folder's and its image's names, moves "
        "by a new deployment",
    ),
    (
        r"rkmon_lib/settings_file\.py",
        r'^PRODUCT_NAME, EARLIER_PRODUCT_NAME = "rocKE monitor", "rocKE PR monitor"$',
        "the product's name in the title line of a settings file an earlier package wrote, which names its profile "
        "all the same",
    ),
    (
        r"rkmon_lib/from_layout1\.py",
        r'^IMAGE_NAME = "rocke-pr-monitor"$|\(in either file\) "rocke-pr-monitor"',
        "layout 1's meaning of a blank or missing image name, frozen: the one-time move gives such a deployment the "
        "image it runs",
    ),
    (
        r"schema/layout1/deploy_settings\.schema\.json",
        r'"default": "rocke-pr-monitor"|rocke-pr-monitor\.",$',
        "layout 1's schema: the image name a blank or missing one meant, which the one-time move reads",
    ),
    (
        r"tools/webhost_remote\.py",
        r'^OLDER_UNITS = \("rocke-merge-monitor\.service",\)$',
        "an earlier package's service for the page on the web host, which the install replaces, and starts again "
        "when this package's does not come up",
    ),
    (
        r"tests/data/deploy_tree_carry\.json",
        r'"image": "rocke-pr-monitor:2\.0\.1"',
        "layout 1 test data: the image an earlier deployment runs, which the one-time move carries",
    ),
    (
        r"tests/data/older_setup_config\.json",
        r'"project": "rocke-pr-monitor"',
        "an earlier setup's configuration, which the one-time move reads",
    ),
    (
        r"tests/test_docker_guard\.py",
        r'rocke_pr_monitor_secrets|"rocke-pr-monitor:latest", "rocke-pr-monitor:2026\.10\.1"',
        "the earlier package's Docker objects in production, which the guard keeps every test away from",
    ),
    (
        r"tests/test_from_layout1\.py",
        r'^IMAGE_NAME = "rocke-pr-monitor"',
        "the test's copy of layout 1's frozen image name",
    ),
    (
        r"tests/test_migrate\.py",
        r'# rocKE PR monitor: deploy settings \(profile "review-only"|assertNotIn\("rocKE PR monitor", text\)',
        "the test of a title line an earlier package wrote",
    ),
    (
        r"tests/test_smoke\.py",
        r'/opt/rocke_pr_monitor|"rocke-pr-monitor:l2test-earlier"|TarInfo\(f"rocke_pr_monitor/\{name\}"\)',
        "the test of the first-apply stage with an earlier package's image, whose code is in /opt/rocke_pr_monitor",
    ),
    (
        r"tests/test_translate_production\.py",
        r'"rocke-pr-monitor"',
        "the image production runs, which the one-time move carries",
    ),
    (
        r"tests/test_product_name\.py",
        r".",
        "this test, which names the earlier names to find them",
    ),
)


def findings() -> tuple[list[str], list[tuple[str, str]]]:
    """(each earlier name in the package's files that ALLOWED does not keep, as `file:line: name`; each entry of
    ALLOWED that keeps none, as its file and words)."""
    used, out = set(), []
    for f in package_files():
        rel = f.relative_to(ROOT).as_posix()
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.split("\n")
        for m in EARLIER.finditer(text):
            n = text.count("\n", 0, m.start())
            keep = {
                i
                for i, (where, words, _why) in enumerate(ALLOWED)
                if re.fullmatch(where, rel) and re.search(words, lines[n])
            }
            used |= keep
            if not keep:
                out.append(f"{rel}:{n + 1}: {' '.join(m.group(0).split())}")
    return out, [entry[:2] for i, entry in enumerate(ALLOWED) if i not in used]


class TestTheProductName(unittest.TestCase):
    def test_an_earlier_name_is_only_where_allowed_says_why(self):
        found, unused = findings()
        self.assertEqual(found, [])
        self.assertEqual(unused, [])

    def test_what_an_earlier_name_is(self):
        for text in (
            "the rocKE PR monitor",
            "the rocKE merge monitor's",
            "The rocKE PR merge monitor",
            "the merge monitor",
            "a PR monitor",
            "rocke-pr-monitor:latest",
            "/opt/rocke_pr_monitor/rk",
            "rocke_pr_monitor_secrets",
            "rocke-merge-monitor.service",
            'the rocKE "\n        "merge monitor checks it',
            "the rocKE PR\nmonitor",
            "    # the rocKE\n    # merge monitor",
            "> the rocKE merge\n> monitor",
        ):
            self.assertRegex(text, EARLIER)
        for text in (
            "the rocKE monitor",
            "rocke-monitor:latest",
            "/opt/rocke_monitor/rk",
            "rocke-monitor.service",
            "the merge queue",
            "merge monitoring",
            "a monitor of each merge",
            "rkmon",
        ):
            self.assertNotRegex(text, EARLIER)


if __name__ == "__main__":
    unittest.main()
