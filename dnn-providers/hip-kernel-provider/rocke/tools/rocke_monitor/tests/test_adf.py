# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Markdown to ADF (rkmon_lib/adf.py), for a ticket's description through the atlassian CLI; `rk md2adf FILE`, which
prints it and closes the file it read."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import adf  # noqa: E402

MD = """## What

The build of `rocke.core` fails on **develop** since the merge; see [the run](https://github.com/ROCm/x/actions).

- first item
  continued here
- second with https://github.com/ROCm/x/pull/1.

1. one
2. two

```python
print("hi")
```
"""


class TestConvert(unittest.TestCase):
    def test_blocks(self):
        doc = adf.convert(MD)
        self.assertEqual(doc["type"], "doc")
        self.assertEqual(
            [b["type"] for b in doc["content"]],
            ["heading", "paragraph", "bulletList", "orderedList", "codeBlock"],
        )
        self.assertEqual(doc["content"][0]["attrs"], {"level": 2})
        para = doc["content"][1]["content"]
        self.assertIn(
            {"type": "text", "text": "rocke.core", "marks": [{"type": "code"}]}, para
        )
        self.assertIn(
            {"type": "text", "text": "develop", "marks": [{"type": "strong"}]}, para
        )
        self.assertIn(
            {
                "type": "text",
                "text": "the run",
                "marks": [
                    {
                        "type": "link",
                        "attrs": {"href": "https://github.com/ROCm/x/actions"},
                    }
                ],
            },
            para,
        )
        first = doc["content"][2]["content"][0]["content"][0]["content"]
        self.assertEqual(first, [{"type": "text", "text": "first item continued here"}])
        second = doc["content"][2]["content"][1]["content"][0]["content"]
        self.assertEqual(
            second[1]["marks"][0]["attrs"]["href"], "https://github.com/ROCm/x/pull/1"
        )
        self.assertEqual(second[2]["text"], ".")
        self.assertEqual(len(doc["content"][3]["content"]), 2)
        self.assertEqual(
            doc["content"][4],
            {
                "type": "codeBlock",
                "attrs": {"language": "python"},
                "content": [{"type": "text", "text": 'print("hi")'}],
            },
        )
        json.dumps(doc)

    def test_plain_text_is_one_paragraph(self):
        self.assertEqual(
            adf.convert("a\nb\n"),
            {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "a b"}]}
                ],
            },
        )

    def test_rk_md2adf_prints_a_file_s_adf_and_closes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            md = pathlib.Path(d) / "ticket.md"
            md.write_text(MD, encoding="utf-8")
            out = io.StringIO()
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always", ResourceWarning)
                with contextlib.redirect_stdout(out):
                    self.assertEqual(adf.main([str(md)]), 0)
        self.assertEqual(json.loads(out.getvalue()), adf.convert(MD))
        self.assertEqual([str(w.message) for w in seen], [])


if __name__ == "__main__":
    unittest.main()
