#!/usr/bin/env python3
"""Query the rocke optimization wiki.

Examples:
    python3 scripts/query.py --symptom lds-stall --architecture gfx950
    python3 scripts/query.py --operator gemm --family cdna
    python3 scripts/query.py --type family
    python3 scripts/query.py "async lds" --compact
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402

_ALIAS_CACHE = None


def load_alias_expansions():
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in variants or []:
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_frontmatter(path):
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages():
    pages = []
    for subdir in ["sources", "wiki"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append(
                {
                    "path": str(md.relative_to(WIKI_ROOT)),
                    "fm": fm,
                    "body": body or "",
                }
            )
    return pages


def detect_page_type(fm, path):
    if "type" in fm:
        return f"wiki-{fm['type']}"
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"
    return "unknown"


def score_keyword_match(fm, body, keywords):
    score = 0
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v)
        for k in (
            "tags",
            "techniques",
            "hardware_features",
            "kernel_types",
            "languages",
            "aliases",
            "symptoms",
            "operator_families",
            "architecture_families",
        )
        for v in (fm.get(k) or [])
    ).lower()
    body_lower = body.lower()
    for kw in keywords:
        best = 0
        for variant in expand_keyword(kw):
            v_l = variant.lower()
            variant_score = 0
            if v_l in title_text:
                variant_score += 10
            if v_l in tag_text:
                variant_score += 5
            variant_score += min(body_lower.count(v_l), 3)
            best = max(best, variant_score)
        score += best
    return score


def _arch_matches(fm, requested):
    variants = {v.lower() for v in expand_keyword(requested)}
    arches = [str(a).lower() for a in (fm.get("architectures") or [])]
    families = [str(a).lower() for a in (fm.get("architecture_families") or [])]
    from_to = []
    if fm.get("from_arch"):
        from_to.append(str(fm["from_arch"]).lower())
    if fm.get("to_arch"):
        from_to.append(str(fm["to_arch"]).lower())
    hay = set(arches) | set(families) | set(from_to)
    return any(v in hay or any(v in h for h in hay) for v in variants)


def filter_pages(pages, args):
    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type:
            if not (ptype.endswith(args.type) or ptype == args.type):
                continue

        if args.tag:
            all_tags = set()
            for k in (
                "tags",
                "techniques",
                "hardware_features",
                "kernel_types",
                "languages",
                "operator_families",
            ):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag)}
            if not any(str(t).lower() in tag_variants for t in all_tags):
                continue

        if args.operator:
            ops = {str(x).lower() for x in (fm.get("operator_families") or [])}
            ops |= {str(x).lower() for x in (fm.get("kernel_types") or [])}
            op_variants = {v.lower() for v in expand_keyword(args.operator)}
            if not (ops & op_variants):
                continue

        if args.family:
            fams = {str(x).lower() for x in (fm.get("architecture_families") or [])}
            fam_variants = {v.lower() for v in expand_keyword(args.family)}
            if not (fams & fam_variants):
                continue

        if args.architecture and not _arch_matches(fm, args.architecture):
            continue

        if args.symptom:
            symptoms = {str(s).lower() for s in (fm.get("symptoms") or [])}
            symptom_variants = {v.lower() for v in expand_keyword(args.symptom)}
            if not (symptoms & symptom_variants):
                continue

        if args.confidence:
            if str(fm.get("confidence", "")) != args.confidence:
                continue

        if args.arch_specific:
            if not fm.get("arch_specific"):
                continue

        out.append(p)
    return out


def format_result(page, compact=False):
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")
    if compact:
        return f"  [{ptype}] {pid}: {title}  ({path})"
    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    for k in (
        "architectures",
        "architecture_families",
        "operator_families",
        "confidence",
        "reproducibility",
        "tags",
        "symptoms",
        "candidate_techniques",
        "from_arch",
        "to_arch",
        "related",
        "sources",
        "rocke_primitive",
    ):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the rocke optimization wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument(
        "--type",
        help="page type: family, pattern, technique, hardware, kernel, process, project, migration",
    )
    parser.add_argument("--tag", help="Filter by tag")
    parser.add_argument(
        "--operator",
        help="Operator family: gemm, attention, convolution, moe, small-ops",
    )
    parser.add_argument("--family", help="Architecture family: cdna, rdna, gfx12")
    parser.add_argument("--architecture", help="Exact gfx id, e.g. gfx950")
    parser.add_argument("--symptom", help="Pattern symptom, e.g. lds-stall")
    parser.add_argument("--confidence", help="verified | source-reported | inferred")
    parser.add_argument(
        "--arch-specific",
        action="store_true",
        help="Only architecture-specific technique pages",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--paths-only", action="store_true")
    args = parser.parse_args()

    pages = filter_pages(load_all_pages(), args)

    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[: args.limit]
    if args.paths_only:
        for p in pages:
            print(p["path"])
        return
    if not pages:
        print("No matching pages.")
        return
    print(f"# {len(pages)} result(s)\n")
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
