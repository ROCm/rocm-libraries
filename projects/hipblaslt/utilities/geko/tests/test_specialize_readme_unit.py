# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""README example extraction pairs fences and never captures prose.

The test utility pairs identical backtick/tilde delimiters within the specialize
section and tokenizes only command candidates. Break: a backtick-only regex, or an
occurrence outside the section / in prose becoming an executable example.
"""
import pytest
import yaml

from tests.specialize_test_support import readme_fenced_blocks, readme_specialize_examples

MARKDOWN = """## Usage
### 4. Specialize
The user's --specialize call isn't a quoted shell command.
```bash
./bin/geko --specialize --workload-log traces/ --route-facts route_facts/ \\
  --hipblaslt baseline --arch gfx950 --workdir first
```
#### More inputs
~~~sh
./bin/geko --specialize --workload-log traces/ --route-facts route_facts/ \\
  --hipblaslt baseline --arch gfx950 --workdir second --keep-beta
~~~
```yaml
exclusions:
  - facts: [UseBeta]
```
### 5. Unrelated
```bash
./bin/geko --specialize --workdir outside
```
"""


def test_every_matched_fenced_block_in_specialize_section_is_found():
    blocks = readme_fenced_blocks(MARKDOWN)
    assert [language for language, _ in blocks] == ["bash", "sh", "yaml"]
    assert len(blocks) == 3
    assert yaml.safe_load(blocks[2][1]) == {"exclusions": [{"facts": ["UseBeta"]}]}


def test_only_specialize_section_commands_are_executed_not_prose_or_other_blocks():
    examples = readme_specialize_examples(MARKDOWN)
    assert len(examples) == 2
    assert [args[args.index("--workdir") + 1] for args in examples] == ["first", "second"]
    assert "--keep-beta" not in examples[0]
    assert examples[1][-1] == "--keep-beta"
    assert all(args[0] == "--specialize" for args in examples)


@pytest.mark.parametrize("opening,embedded", [("```bash", "~~~"), ("~~~bash", "```")])
def test_other_fence_kind_inside_block_does_not_terminate_it(opening, embedded):
    delimiter = opening[:3]
    markdown = ("### Specialize\n" + opening + "\n" + embedded + "\n"
                "./bin/geko --specialize --workdir inside\n" + delimiter + "\n"
                "### Following\n```bash\n./bin/geko --specialize --workdir outside\n```\n")
    blocks = readme_fenced_blocks(markdown)
    assert len(blocks) == 1
    assert blocks[0][0] == "bash"
    examples = readme_specialize_examples(markdown)
    assert examples == [["--specialize", "--workdir", "inside"]]
