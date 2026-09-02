#!/usr/bin/env python3
"""Validate wiki/source frontmatter against data/schemas.yaml and data/tags.yaml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402

# Software-achieved perf claims are forbidden in this tree (rocke AGENTS.md).
# Hardware capacities (KB LDS, bank counts) are allowed.
_FORBIDDEN_PERF = re.compile(
    r"\b\d+(\.\d+)?\s*(TFLOP|tflop|GFLOP|gflop|MFU)\b"
    r"|\b\d+(\.\d+)?\s*(µs|μs)\b"
    r"|\b\d+(\.\d+)?\s*GB/s\b",
    re.IGNORECASE,
)


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def split_fm(content):
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", content, re.DOTALL)
    if not m:
        return None, content
    try:
        return yaml.safe_load(m.group(1)), m.group(2)
    except yaml.YAMLError:
        return None, content


def vocab_sets(tags):
    out = {}
    for key, vals in tags.items():
        if isinstance(vals, list):
            out[key] = set(vals)
    return out


def main():
    schemas = load_yaml(WIKI_ROOT / "data" / "schemas.yaml")
    tags = vocab_sets(load_yaml(WIKI_ROOT / "data" / "tags.yaml"))
    errors = []
    ids = {}
    pages = []

    for sub in ("wiki", "sources"):
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            rel = md.relative_to(WIKI_ROOT).as_posix()
            fm, body = split_fm(md.read_text(encoding="utf-8"))
            if not isinstance(fm, dict):
                errors.append(f"{rel}: missing or invalid frontmatter")
                continue
            pages.append((rel, fm, body))
            pid = fm.get("id")
            if not pid:
                errors.append(f"{rel}: missing id")
            elif pid in ids:
                errors.append(f"{rel}: duplicate id {pid} (also {ids[pid]})")
            else:
                ids[pid] = rel

            ptype = fm.get("type")
            schema_key = {
                "family": "wiki-family",
                "pattern": "wiki-pattern",
                "technique": "wiki-technique",
                "hardware": "wiki-hardware",
                "kernel": "wiki-kernel",
                "process": "wiki-process",
                "project": "source-project",
                "migration": "wiki-migration",
            }.get(ptype)
            schema = schemas.get(schema_key or "", {})
            for field in schema.get("required", []):
                if field not in fm or fm[field] in (None, [], ""):
                    errors.append(f"{rel}: missing required field {field}")
            prefix = schema.get("constraints", {}).get("id_prefix")
            if prefix and pid and not str(pid).startswith(prefix):
                errors.append(f"{rel}: id {pid} must start with {prefix}")

            allowed_tags = (
                tags.get("tags", set())
                | tags.get("techniques", set())
                | tags.get("hardware_features", set())
                | tags.get("kernel_types", set())
                | tags.get("operator_families", set())
                | tags.get("architecture_families", set())
            )
            for field, vocab_name in (
                ("architectures", "architectures"),
                ("architecture_families", "architecture_families"),
                ("operator_families", "operator_families"),
                ("symptoms", "symptoms"),
                ("hardware_features", "hardware_features"),
                ("kernel_types", "kernel_types"),
                ("languages", "languages"),
            ):
                vals = fm.get(field)
                if vals is None:
                    continue
                allowed = tags.get(vocab_name, set())
                for v in vals:
                    if v not in allowed:
                        errors.append(f"{rel}: unknown {field} value {v!r}")
            if fm.get("confidence") is not None:
                if fm["confidence"] not in tags.get("confidence", set()):
                    errors.append(f"{rel}: unknown confidence {fm['confidence']}")
            if fm.get("reproducibility") is not None:
                if fm["reproducibility"] not in tags.get("reproducibility", set()):
                    errors.append(
                        f"{rel}: unknown reproducibility {fm['reproducibility']}"
                    )
            arch_vocab = tags.get("architectures", set())
            for field in ("from_arch", "to_arch"):
                v = fm.get(field)
                if v is None:
                    continue
                if v not in arch_vocab:
                    errors.append(f"{rel}: unknown {field} value {v!r}")
            for v in fm.get("tags") or []:
                if v not in allowed_tags:
                    errors.append(f"{rel}: unknown tags value {v!r}")

            if _FORBIDDEN_PERF.search(body or ""):
                errors.append(
                    f"{rel}: software-achieved perf quantity (TFLOP/µs/GB/s) — keep numbers off-repo"
                )

    for rel, fm, _body in pages:
        for key in ("related", "sources", "candidate_techniques", "prerequisites"):
            for ref in fm.get(key) or []:
                if ref not in ids:
                    errors.append(f"{rel}: {key} references unknown id {ref}")

    if errors:
        print(f"{len(errors)} validation error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print(f"OK: {len(pages)} pages, {len(ids)} ids")


if __name__ == "__main__":
    main()
