# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.torch_uses: each kind of torch use is found wherever the code
makes it, text that only mentions torch is not a use, and the allowlist
comparison names every breach."""

from __future__ import annotations

from pathlib import Path

import pytest
from rocke.tests.source_allowlist import (
    AllowlistEntry,
    file_kinds,
    format_allowlist_entry,
)
from rocke.tests.torch_uses import (
    TorchUse,
    TorchUseKind,
    allowlist_breaches,
    is_config_file,
    read_allowlist,
    torch_named_identifiers,
    torch_uses_in_config,
    torch_uses_in_python,
)

IMPORT = TorchUseKind.IMPORT
IMPORTORSKIP = TorchUseKind.IMPORTORSKIP
TORCH_INTEROP = TorchUseKind.TORCH_INTEROP
TORCH_API = TorchUseKind.TORCH_API
TORCH_OBJECT = TorchUseKind.TORCH_OBJECT
CONFIG = TorchUseKind.CONFIG


@pytest.mark.parametrize(
    "source, kinds",
    [
        ("import torch\n", {IMPORT}),
        ("import torch.cuda as cuda\ncuda.synchronize()\n", {IMPORT, TORCH_API}),
        ("import torch as T\nT.cuda.synchronize()\n", {IMPORT, TORCH_API}),
        (
            "def load():\n    from torch import nn\n    return nn\n",
            {IMPORT, TORCH_OBJECT},
        ),
        ("from torch import fx\n", {IMPORT, TORCH_API}),
        ("try:\n    import torch\nexcept ImportError:\n    pass\n", {IMPORT}),
        ("__import__('torch')\n", {IMPORT}),
        (
            "import importlib\nimportlib.import_module('torch.cuda')\n",
            {IMPORT, TORCH_API},
        ),
        ("import importlib.util\nimportlib.util.find_spec('torch')\n", {IMPORT}),
        ("import sys\nloaded = sys.modules.get('torch')\n", {IMPORT}),
        ("import sys\nloaded = sys.modules['torch']\n", {IMPORT}),
        ("import pytest\ntorch = pytest.importorskip('torch')\n", {IMPORTORSKIP}),
        (
            (
                "from rocke.tests.lanes import TICKET, import_or_skip_until\n"
                "import_or_skip_until('torch', TICKET)\n"
            ),
            {IMPORTORSKIP},
        ),
        ("from rocke.runtime import torch_interop\n", {TORCH_INTEROP}),
        ("from .torch_interop import resolve_stream\n", {TORCH_INTEROP}),
        ("import rocke.runtime.torch_interop\n", {TORCH_INTEROP}),
        (
            "import rocke.runtime as rt\nrt.torch_interop.resolve_stream(0)\n",
            {TORCH_INTEROP},
        ),
        ("torch.ops.load_library(path)\n", {TORCH_API}),
        ("compiled = torch.compile(fn)\n", {TORCH_API}),
        ("graph = torch.fx.symbolic_trace(fn)\n", {TORCH_API}),
        ("def run(torch):\n    return torch.zeros(1)\n", {TORCH_OBJECT}),
        ("dtype = torch.float16\n", {TORCH_OBJECT}),
    ],
)
def test_each_kind_of_use_is_found(source, kinds):
    assert file_kinds(torch_uses_in_python(source, "<synthetic>")) == kinds


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\n",
        "FORBIDDEN = ('torch.nn.functional.scaled_dot_product_attention',)\n",
        "# torch is not needed here: a numpy reference\nx = 1\n",
        '"""Like torch.zeros, but numpy."""\n',
        "def run(module):\n    return module.torch_free\n",
        "import torchvision\nfrom torchaudio import io\n",
        "from .torch import helper\n",
        "out = run_unified_attention_torch(q, k, v)\n",
        "import pytest\nyaml = pytest.importorskip('yaml')\n",
        "import sys\nassert 'torch' not in sys.modules\n",
        "import sys\nsys.modules['torch'] = None\n",
        "import sys\nloaded = sys.modules.get('numpy')\n",
    ],
)
def test_text_that_only_mentions_torch_is_not_a_use(source):
    assert torch_uses_in_python(source, "<synthetic>") == []


def test_a_use_names_its_line_and_code():
    source = (
        "import numpy\n\n\ndef load():\n    import torch\n    return torch.zeros(1)\n"
    )
    assert torch_uses_in_python(source, "<synthetic>") == [
        TorchUse(5, IMPORT, "import torch"),
        TorchUse(6, TORCH_OBJECT, "torch.zeros"),
    ]


def test_source_that_does_not_parse_is_an_error_not_a_clean_file():
    with pytest.raises(SyntaxError):
        torch_uses_in_python("import torch\ndef broken(:\n", "<synthetic>")


def test_config_lines_naming_torch_are_uses_comments_included():
    text = (
        "numpy>=1.24\n"
        "torch>=2.4\n"
        "# Install a ROCm Torch wheel separately\n"
        "# pip install --index-url https://download.pytorch.org/whl/rocm7.0\n"
        "add_test(test_sdpa_torch_reference.py)\n"
    )
    assert torch_uses_in_config(text) == [
        TorchUse(2, CONFIG, "torch>=2.4"),
        TorchUse(3, CONFIG, "# Install a ROCm Torch wheel separately"),
    ]


@pytest.mark.parametrize(
    "name, config",
    [
        ("requirements.txt", True),
        ("requirements-dev.txt", True),
        ("pyproject.toml", True),
        ("CMakeLists.txt", True),
        ("ctest_lists.cmake", True),
        ("README.md", False),
        ("notes.txt", False),
        ("pytest.ini", False),
    ],
)
def test_config_files_are_requirements_pyproject_and_cmake(name, config):
    assert is_config_file(Path("tree") / name) is config


def test_torch_named_identifiers_are_listed_once_with_their_first_line():
    source = (
        "from rocke.dispatch import TorchBinding\n"
        "import torch\n"
        "def bind_torch(q):\n"
        "    return run_unified_attention_torch(q, lane=TORCH_IN_CI)\n"
        "bind_torch(torch.zeros(1))\n"
        "pytorch_version = torchvision = None\n"
    )
    assert torch_named_identifiers(source, "<synthetic>") == [
        (1, "TorchBinding"),
        (3, "bind_torch"),
        (4, "TORCH_IN_CI"),
        (4, "run_unified_attention_torch"),
    ]


def _allowlist(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "allowlist.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_allowlist_reads_one_line_per_file():
    entry = format_allowlist_entry("a/b.py", {TORCH_OBJECT: 4, IMPORT: 1})
    assert entry == "a/b.py: import=1 torch_object=4"


def test_the_allowlist_skips_comments_and_blank_lines(tmp_path):
    path = _allowlist(
        tmp_path, "# header\n\na/b.py: import=1 torch_object=4\nc.cmake: config=2\n"
    )
    assert read_allowlist(path) == {
        "a/b.py": AllowlistEntry({IMPORT: 1, TORCH_OBJECT: 4}),
        "c.cmake": AllowlistEntry({CONFIG: 2}),
    }


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "a/b.py import=1\n",
            r"allowlist.txt:1='a/b.py import=1', expected 'file: kind=count \.\.\.'",
        ),
        (
            "a/b.py:\n",
            r"allowlist.txt:1='a/b.py:', expected 'file: kind=count \.\.\.'",
        ),
        (
            "a/b.py: imports=1\n",
            r"allowlist.txt:1 kind='imports', expected one of \[",
        ),
        (
            "a/b.py: import=1\na/b.py: config=1\n",
            r"allowlist.txt:2 file='a/b.py', expected",
        ),
    ],
)
def test_a_malformed_allowlist_is_refused_naming_the_line(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_allowlist(_allowlist(tmp_path, text))


def _uses(*kinds: TorchUseKind) -> list[TorchUse]:
    return [TorchUse(line, kind, "code") for line, kind in enumerate(kinds, 1)]


def test_a_tree_that_matches_its_allowlist_has_no_breach():
    found = {"a.py": _uses(IMPORT, TORCH_OBJECT)}
    assert (
        allowlist_breaches(
            found, {"a.py": AllowlistEntry({IMPORT: 1, TORCH_OBJECT: 1})}
        )
        == []
    )


def test_a_file_not_on_the_list_is_a_new_use():
    breaches = allowlist_breaches({"new.py": _uses(IMPORT)}, {})
    assert breaches == ["new.py: new torch use: line 1 import: code"]


def test_a_listed_file_gaining_a_kind_is_a_new_kind_of_use():
    breaches = allowlist_breaches(
        {"a.py": _uses(IMPORT, TORCH_API)},
        {"a.py": AllowlistEntry({IMPORT: 1})},
    )
    assert breaches == ["a.py: new kind of torch use: line 2 torch_api: code"]


def test_a_listed_file_gaining_a_use_of_a_listed_kind_is_more_use():
    breaches = allowlist_breaches(
        {"a.py": _uses(IMPORT, TORCH_OBJECT, TORCH_OBJECT)},
        {"a.py": AllowlistEntry({IMPORT: 1, TORCH_OBJECT: 1})},
    )
    assert breaches == [
        (
            "a.py: more torch use than listed: torch_object 1 -> 2: line 2 "
            "torch_object: code; line 3 torch_object: code"
        )
    ]


def test_a_listed_file_without_torch_is_a_stale_entry():
    breaches = allowlist_breaches({}, {"gone.py": AllowlistEntry({IMPORT: 1})})
    assert breaches == ["gone.py: stale entry, no torch use left: delete its line"]


def test_a_listed_kind_no_longer_used_is_stale():
    breaches = allowlist_breaches(
        {"a.py": _uses(IMPORT)},
        {"a.py": AllowlistEntry({IMPORT: 1, TORCH_API: 1})},
    )
    assert breaches == [
        (
            "a.py: stale counts, fewer than listed: torch_api 1 -> 0; make the line "
            "'a.py: import=1'"
        )
    ]
