#!/usr/bin/env python3
"""Regenerate queries/*.md from wiki + sources frontmatter."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402

QUERIES = WIKI_ROOT / "queries"


def extract_frontmatter(filepath):
    content = filepath.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def qlink(path):
    return f"../{path}"


def collect_all_pages():
    pages = []
    errors = []
    for search_dir in [WIKI_ROOT / "sources", WIKI_ROOT / "wiki"]:
        if not search_dir.exists():
            continue
        for md_file in sorted(search_dir.rglob("*.md")):
            fm = extract_frontmatter(md_file)
            rel = md_file.relative_to(WIKI_ROOT).as_posix()
            if not isinstance(fm, dict):
                errors.append(rel)
                continue
            fm["_path"] = rel
            pages.append(fm)
    if errors:
        print("ERROR: missing frontmatter:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    return pages


def _link(p):
    return f"[{p.get('title', p.get('id'))}]({qlink(p['_path'])})"


def generate_by_problem(pages):
    id_map = {p["id"]: p for p in pages if "id" in p}
    lines = [
        "# Query: By Problem / Symptom",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Symptom | Pattern | Candidate techniques |",
        "|---------|---------|----------------------|",
    ]
    for p in sorted(
        (x for x in pages if x.get("type") == "pattern"),
        key=lambda x: x.get("title", ""),
    ):
        symptoms = ", ".join(p.get("symptoms", []))
        techs = []
        for tid in p.get("candidate_techniques", []):
            techs.append(_link(id_map[tid]) if tid in id_map else tid)
        lines.append(f"| {symptoms} | {_link(p)} | {', '.join(techs)} |")
    return "\n".join(lines) + "\n"


def generate_by_technique(pages):
    lines = [
        "# Query: By Technique",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Technique | Scope | Architectures | Operators |",
        "|-----------|-------|---------------|-----------|",
    ]
    for t in sorted(
        (x for x in pages if x.get("type") == "technique"),
        key=lambda x: x.get("title", ""),
    ):
        scope = "arch-specific" if t.get("arch_specific") else "common"
        archs = ", ".join(
            t.get("architectures") or t.get("architecture_families") or []
        )
        ops = ", ".join(t.get("operator_families") or [])
        lines.append(f"| {_link(t)} | {scope} | {archs} | {ops} |")
    return "\n".join(lines) + "\n"


def generate_by_family(pages):
    lines = [
        "# Query: By Operator Family",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Family page | Operators | Architectures |",
        "|-------------|-----------|---------------|",
    ]
    for p in sorted(
        (x for x in pages if x.get("type") == "family"),
        key=lambda x: x.get("title", ""),
    ):
        ops = ", ".join(p.get("operator_families") or [])
        archs = ", ".join(
            p.get("architecture_families") or p.get("architectures") or []
        )
        lines.append(f"| {_link(p)} | {ops} | {archs} |")
    return "\n".join(lines) + "\n"


def generate_by_architecture(pages):
    by_arch = defaultdict(list)
    by_fam = defaultdict(list)
    for p in pages:
        for a in p.get("architectures") or []:
            by_arch[a].append(p)
        for a in p.get("architecture_families") or []:
            by_fam[a].append(p)
    lines = [
        "# Query: By Architecture",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "## Architecture families",
        "",
    ]
    for fam in sorted(by_fam):
        lines.append(f"### `{fam}`")
        lines.append("")
        lines.append(", ".join(_link(p) for p in by_fam[fam][:40]))
        lines.append("")
    lines.append("## Exact architectures")
    lines.append("")
    for arch in sorted(by_arch):
        lines.append(f"### `{arch}`")
        lines.append("")
        lines.append(", ".join(_link(p) for p in by_arch[arch][:40]))
        lines.append("")
    return "\n".join(lines) + "\n"


def generate_by_hardware_feature(pages):
    tags_path = WIKI_ROOT / "data" / "tags.yaml"
    hw_tag_set = set()
    if tags_path.exists():
        hw_tag_set = set(
            yaml.safe_load(tags_path.read_text(encoding="utf-8")).get(
                "hardware_features", []
            )
        )
    feature_pages = defaultdict(list)
    for p in pages:
        indexed = set()
        for feat in p.get("hardware_features") or []:
            feature_pages[feat].append(p)
            indexed.add(feat)
        for tag in p.get("tags") or []:
            if tag in hw_tag_set and tag not in indexed:
                feature_pages[tag].append(p)
    lines = [
        "# Query: By Hardware Feature",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Feature | Related pages |",
        "|---------|---------------|",
    ]
    for feat in sorted(feature_pages):
        links = ", ".join(_link(p) for p in feature_pages[feat])
        lines.append(f"| `{feat}` | {links} |")
    return "\n".join(lines) + "\n"


def generate_by_kernel_type(pages):
    type_pages = defaultdict(list)
    for p in pages:
        for kt in p.get("kernel_types") or []:
            type_pages[kt].append(p)
    lines = [
        "# Query: By Kernel Type",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Kernel type | Pages |",
        "|-------------|-------|",
    ]
    for kt in sorted(type_pages):
        lines.append(f"| `{kt}` | {', '.join(_link(p) for p in type_pages[kt])} |")
    return "\n".join(lines) + "\n"


def generate_by_migration(pages):
    lines = [
        "# Query: By Migration",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Migration | From | To |",
        "|-----------|------|----|",
    ]
    for p in sorted(
        (x for x in pages if x.get("type") == "migration"),
        key=lambda x: x.get("title", ""),
    ):
        lines.append(
            f"| {_link(p)} | `{p.get('from_arch', '')}` | `{p.get('to_arch', '')}` |"
        )
    return "\n".join(lines) + "\n"


def generate_by_repo(pages):
    lines = [
        "# Query: By Source Project",
        "",
        "> Auto-generated. Do not edit manually.",
        "",
        "| Project | Page | Tree |",
        "|---------|------|------|",
    ]
    for p in sorted(
        (x for x in pages if x.get("type") == "project"),
        key=lambda x: x.get("title", ""),
    ):
        tree = p.get("tree", "")
        lines.append(f"| {p.get('repo', p.get('id'))} | {_link(p)} | `{tree}` |")
    return "\n".join(lines) + "\n"


def main():
    QUERIES.mkdir(exist_ok=True)
    pages = collect_all_pages()
    print(f"Collected {len(pages)} pages")
    generators = {
        "by-problem.md": generate_by_problem,
        "by-technique.md": generate_by_technique,
        "by-family.md": generate_by_family,
        "by-architecture.md": generate_by_architecture,
        "by-hardware-feature.md": generate_by_hardware_feature,
        "by-kernel-type.md": generate_by_kernel_type,
        "by-repo.md": generate_by_repo,
        "by-migration.md": generate_by_migration,
    }
    for filename, gen in generators.items():
        (QUERIES / filename).write_text(gen(pages), encoding="utf-8")
        print(f"  Generated queries/{filename}")
    print("Done.")


if __name__ == "__main__":
    main()
