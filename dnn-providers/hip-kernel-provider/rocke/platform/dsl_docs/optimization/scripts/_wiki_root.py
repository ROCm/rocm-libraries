"""Wiki-root resolution for the rocke optimization wiki.

The wiki root is the directory that contains SKILL.md, data/, wiki/, sources/.
That is ``dsl_docs/optimization/`` — this file lives in ``scripts/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _looks_like_wiki_root(p: Path) -> bool:
    return (p / "data" / "tags.yaml").is_file() and (p / "wiki").is_dir()


def _error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_wiki_root() -> Path:
    env = os.environ.get("ROCKE_OPT_WIKI_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_wiki_root(p):
            return p
        _error(
            f"ROCKE_OPT_WIKI_ROOT={env!r} is not a rocke optimization wiki "
            "(missing data/tags.yaml or wiki/)."
        )

    default_root = Path(__file__).resolve().parent.parent
    if _looks_like_wiki_root(default_root):
        return default_root

    seen: set[Path] = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_wiki_root(candidate):
                return candidate

    _error(
        "Could not locate the rocke optimization wiki root.\n"
        "       Expected a directory containing `data/tags.yaml` and `wiki/`.\n"
        "       Run scripts from dsl_docs/optimization/, or set ROCKE_OPT_WIKI_ROOT."
    )
    return Path()


WIKI_ROOT = resolve_wiki_root()
