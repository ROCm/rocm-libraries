# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ABI lanes and gate G9.

rocke/abi.h carries two counters with different blast radii:
- `ROCKE_ABI_VERSION` (binary): bump when a C caller built against the old header would be wrong about memory
  (struct layout, an enumerator inserted mid-list, a signature change).
- `ROCKE_RECIPE_ABI` (wire): bump when the engine learns a construct an older engine would misread. The construct must
  be registered in portable_ir/src/abi.py.

Each counter keeps a history comment (` * N: what changed`) above its #define. Dispatch families also carry kernel-ABI
strings (`<FAMILY>_ABI_VERSION = "name/vN"`).

Each counter is a lane, and lanes are serialized. Numbers are assigned by landing order, not chosen by the PR.
- The lane head is the first lane member, in queue order, whose other gates are green (readiness order).
- G9 passes for the head only when its claim is develop + 1 on today's develop, the history line for the new number is
  present and no earlier history line was dropped.
- Everyone else waits behind the head with a tentative reservation, and is asked to renumber only when it becomes the
  head.
- Stacked PRs move as one unit, under their bottom PR.
- PRs whose bump hunks are identical (the same commit carried by two PRs) form one claim.
- Two claims in one counter's lane are a conflict in the merge plan, even when git merges them cleanly: never in one
  wave, the earlier in plan order first. The plan reads who is in a lane
  (lane_claims), then the lanes follow the plan's order.
"""

from __future__ import annotations

import hashlib
import re
import subprocess

from . import gitops, paths

CACHE_VERSION = 6  # bump when the claim logic changes, so cached claims are recomputed (6: generated headers, exports)
ABI_H = "platform/cpp/include/rocke/abi.h"
ABI_PY = "platform/python/rocke/portable_ir/src/abi.py"
PUBLIC_HEADERS = "platform/cpp/include/rocke/"
CPP_TREE = "platform/cpp/"  # where a moved struct or function is looked for
DEFINES = {"binary": "ROCKE_ABI_VERSION", "recipe": "ROCKE_RECIPE_ABI"}
# ABI_PY's mirrors of the abi.h counters. Only a test against a built shared engine pins them together, and no CI lane
# builds one (a bump of ROCKE_ABI_VERSION that leaves BINARY_ABI behind makes online.load() refuse the engine).
MIRRORS = {"binary": "BINARY_ABI", "recipe": "RECIPE_ABI"}
LANE_TITLE = {
    "binary": "binary ABI (ROCKE_ABI_VERSION)",
    "recipe": "wire ABI (ROCKE_RECIPE_ABI)",
}
FAMILY_LINE = re.compile(r'^([+-])\s*([A-Z0-9_]*ABI_VERSION)\s*=\s*"([^"]+)"', re.M)


def parse_header(text: str) -> dict:
    """Counter values and their history entries ({n: text}) from abi.h text."""
    out = {}
    for lane, name in DEFINES.items():
        m = re.search(rf"#define\s+{name}\s+(\d+)", text)
        out[lane] = int(m.group(1)) if m else None
        hist = {}
        if m:
            block = text[: m.start()].rsplit("/*", 1)[-1]
            # The comment's closing `*/` may share the last entry's line: drop it, or an entry would read differently
            # as the last one than as an earlier one (and look like the PR's own when compared with develop).
            hist = {
                int(n): re.sub(r"\s*\*/\s*$", "", body).strip()
                for n, body in re.findall(r"^\s*\*?\s*(\d+):\s*(.*)$", block, re.M)
            }
        out[f"hist_{lane}"] = hist
    return out


def parse_mirrors(text: str) -> dict:
    """ABI_PY's mirror values ({lane: int or None}) from its text."""
    out = {}
    for lane, name in MIRRORS.items():
        m = re.search(rf"^{name}\s*=\s*(\d+)\s*$", text, re.M)
        out[lane] = int(m.group(1)) if m else None
    return out


def _show(repo: str, ref: str, path: str) -> str:
    return gitops.git(repo, "show", f"{ref}:{path}", check=False)


# The binary counter's layout canaries: `static_assert(sizeof(T) == S && ROCKE_ABI_VERSION >= N, ...)` in C/C++ files
# anywhere in rocKE's tree (an edited struct fails the build there until the counter moves). A canary never names a
# number above the define.
CANARY = re.compile(r"ROCKE_ABI_VERSION\s*>=\s*(\d+)")
CANARY_GREP = r"ROCKE_ABI_VERSION[[:space:]]*>=[[:space:]]*[0-9]+"


def _history_numbers(text: str, name: str) -> list[int]:
    """Every ` * N:` history number above `name`'s #define, duplicates kept."""
    m = re.search(rf"#define\s+{name}\s+(\d+)", text)
    if not m:
        return []
    block = text[: m.start()].rsplit("/*", 1)[-1]
    return [int(n) for n in re.findall(r"^\s*\*?\s*(\d+):", block, re.M)]


def homes(repo: str, K: str, ref: str) -> dict:
    """Each counter's homes at `ref` (a merge commit: the tree as it would land) across all files: abi.h's #define, its
    newest history line, abi.py's mirror and, for the binary counter, the highest layout canary. {lane: {"define",
    "history", "mirror", "canary", "dups", "problems"}}; `problems` empty when the homes agree.
    """
    h_text = _show(repo, ref, f"{K}/{ABI_H}")
    if not h_text.strip():
        return {}  # no abi.h in that tree: nothing to agree
    mir = parse_mirrors(_show(repo, ref, f"{K}/{ABI_PY}"))
    hdr = parse_header(h_text)
    canaries: list[tuple[int, str]] = []
    for ln in gitops.git(
        repo, "grep", "-n", "-E", CANARY_GREP, ref, "--", K, check=False
    ).splitlines():
        # <ref>:<path>:<line>:<text>
        parts = ln.split(":", 3)
        if len(parts) == 4:
            for v in CANARY.findall(parts[3]):
                canaries.append((int(v), f"{parts[1][len(K) + 1:]}:{parts[2]}"))
    out = {}
    for lane, name in DEFINES.items():
        d = hdr[lane]
        nums = _history_numbers(h_text, name)
        hist = max(nums) if nums else None
        dups = sorted({x for x in nums if nums.count(x) > 1})
        can = max(canaries) if lane == "binary" and canaries else None
        problems = []
        if d is None:
            problems.append(f"abi.h defines no {name}")
        else:
            if hist is not None and hist != d:
                problems.append(
                    f"abi.h {name} {d} but its newest history line is {hist}"
                )
            if hist is None and d > 1:
                problems.append(f"abi.h {name} {d} has no history line")
            if mir[lane] is not None and mir[lane] != d:
                problems.append(
                    f"abi.py {MIRRORS[lane]} {mir[lane]} but abi.h {name} {d}"
                )
            if can and can[0] > d:
                problems.append(
                    f"a layout canary needs {name} >= {can[0]} ({can[1]}) but abi.h has {d}"
                )
        if dups:
            problems.append(f"history number(s) {dups} appear more than once")
        out[lane] = {
            "define": d,
            "history": hist,
            "mirror": mir[lane],
            "canary": can,
            "dups": dups,
            "problems": problems,
        }
    return out


def homes_lines(h: dict) -> list[str]:
    """`homes <lane>: define N · history N · abi.py N[ · canary N] · OK`, or `... · HALF RENUMBER: <what disagrees>`."""
    if not h:
        return ["homes: no abi.h in that tree"]
    out = []
    for lane, x in h.items():
        parts = [
            f"define {x['define'] if x['define'] is not None else '-'}",
            f"history {x['history'] if x['history'] is not None else '-'}",
            f"abi.py {x['mirror'] if x['mirror'] is not None else '-'}",
        ]
        if lane == "binary":
            parts.append(f"canary {x['canary'][0] if x['canary'] else '-'}")
        verdict = (
            "HALF RENUMBER: " + "; ".join(x["problems"]) if x["problems"] else "OK"
        )
        out.append(f"homes {lane}: " + " · ".join(parts + [verdict]))
    return out


def develop_counters(
    cfg: dict, develop_ref: str = "refs/remotes/origin/develop"
) -> dict:
    return parse_header(
        _show(
            cfg["paths"]["repo"], develop_ref, f"{paths.repository_folder(cfg)}/{ABI_H}"
        )
    )


def is_abi_header(path: str, rocke_path: str) -> bool:
    """The engine's C ABI surface: rocke/*.h headers, not the generated instance_*/helper_* or *_internal.h
    builder headers (about 160 of the 190 files there)."""
    prefix = f"{rocke_path}/{PUBLIC_HEADERS}"
    if not path.startswith(prefix) or not path.endswith(".h"):
        return False
    name = path[len(prefix) :]
    return (
        "/" not in name
        and not name.startswith(("instance_", "helper_"))
        and not name.endswith("_internal.h")
    )


def header_kind(path: str, rocke_path: str) -> str | None:
    """ "public" for the C ABI headers (is_abi_header); "generated" for the instance_*/helper_* headers, whose spec and
    problem structs and exported functions cross the C API too (abi.h's history bumps for them); None for the rest
    (_internal.h builder headers, subdirectories, abi.h)."""
    prefix = f"{rocke_path}/{PUBLIC_HEADERS}"
    if (
        path == f"{rocke_path}/{ABI_H}"
        or not path.startswith(prefix)
        or not path.endswith(".h")
    ):
        return None
    if is_abi_header(path, rocke_path):
        return "public"
    name = path[len(prefix) :]
    return (
        "generated"
        if (
            "/" not in name
            and name.startswith(("instance_", "helper_"))
            and not name.endswith("_internal.h")
        )
        else None
    )


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", text, flags=re.S))


def _type_only(decl: str) -> str:
    """A field or parameter declaration without its name: `const char *name` -> `const char*`, `int a[4]` ->
    `int[4]`, `unsigned x : 3` -> `unsigned:3`, `void (*fn)(int)` -> `void(*)(int)`. A renamed field keeps its type
    and place, so it moves no byte; only types, their order and their count tell two layouts apart.
    """
    s = " ".join(decl.split())
    s = re.sub(r"\(\s*\*\s*\w+\s*\)", "(*)", s)  # a function pointer's name
    width = ""
    m = re.match(r"^(.*?)\s*:\s*(\w+)$", s)
    if m and "(" not in s:
        s, width = m.group(1), ":" + m.group(2)
    parts = [p.strip() for p in s.split(",")] if "(" not in s else [s]
    out = []
    for i, p in enumerate(parts):
        m = re.match(r"^(.*?)\b([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", p)
        if m and (
            m.group(1).strip() or i
        ):  # the first declarator needs a type before its name
            p = m.group(1).strip() + "".join(m.group(3).split())
        out.append(re.sub(r"\s*([*()\[\],])\s*", r"\1", p))
    return ",".join(out) + width


def struct_layouts(text: str) -> dict:
    """{name: fields} for each struct a C header defines: the typedef name, or the tag of a plain `struct X {…};`.
    Comments, spacing and field names are dropped (_type_only), so only the fields' types, their order and their count
    tell two layouts apart."""
    text = _strip_comments(text)
    out = {}
    for m in re.finditer(r"\b(typedef\s+)?struct\s+(\w+)?\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        tail = re.match(r"\s*(\w*)\s*;", text[i:])
        name = tail and ((tail.group(1) if m.group(1) else "") or m.group(2))
        if name:  # a nested anonymous struct names a member, not a type
            out[name] = tuple(
                _type_only(f) for f in text[m.end() : i - 1].split(";") if f.strip()
            )
    return out


def _top_level(text: str) -> str:
    """A header's text at file scope: comments, preprocessor lines and `extern "C" {` dropped, every `{…}` body
    replaced by `;` (a struct's fields, a function's body), so each top-level declaration is one `;`-ended statement.
    """
    text = _strip_comments(text)
    text = re.sub(r"(?m)^\s*#.*?(?:\\\n.*?)*$", " ", text)
    text = re.sub(r'extern\s*"C"\s*\{', " ", text)
    out, depth = [], 0
    for ch in text:
        if ch == "{":
            depth += 1
            if depth == 1:
                out.append(";")
        elif ch == "}":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _params(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [_type_only(p) for p in parts if p.strip()]


def exported_functions(text: str) -> dict:
    """{name: signature} for each `rocke_*` function a C header declares or defines with external linkage (not
    `static`, not `inline`, not a typedef): the return type and the parameter types, names dropped. All-caps tokens
    (export and attribute macros) are not part of the signature."""
    out = {}
    for stmt in _top_level(text).split(";"):
        s = " ".join(stmt.split())
        m = re.match(r"^(.*?)\b(rocke_\w+)\s*\((.*)\)$", s)
        if (
            not m
            or not m.group(1).strip()
            or re.search(r"[=(]|\b(typedef|static|inline|return)\b", m.group(1))
        ):
            continue
        ret = " ".join(
            t for t in m.group(1).split() if not re.fullmatch(r"[A-Z][A-Z0-9_]+", t)
        )
        out[m.group(2)] = (
            re.sub(r"\s*\*\s*", "*", ret) + "(" + ",".join(_params(m.group(3))) + ")"
        )
    return out


def surface_changes(repo: str, K: str, mb: str, ref: str, files: set[str]) -> dict:
    """What the PR changes in the C ABI headers it touches (header_kind public or generated): {"public": structs,
    "generated": structs, "functions": [(name, "removed" | "signature")]}. A struct or function the PR moves to another
    such header keeps its place in the ABI: it is read there at `ref`, and counts only if its layout or signature
    changed."""
    out: dict = {"public": [], "generated": [], "functions": []}
    hdrs = sorted(f for f in files if header_kind(f, K))
    if not hdrs:
        return out
    before = {f: _show(repo, mb, f) for f in hdrs}
    after = {f: _show(repo, ref, f) for f in hdrs}
    b_st = {f: struct_layouts(t) for f, t in before.items()}
    b_fn = {f: exported_functions(t) for f, t in before.items()}
    a_st: dict = {}
    a_fn: dict = {}
    for t in after.values():
        a_st.update(struct_layouts(t))
        a_fn.update(exported_functions(t))
    gone = sorted(
        {n for d in b_st.values() for n in d if n not in a_st}
        | {n for d in b_fn.values() for n in d if n not in a_fn}
    )
    if (
        gone
    ):  # moved to another header (any, _internal.h too) or kept in a source: the symbol is still exported
        p = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "grep",
                "-l",
                "-w",
                "-F",
                *[x for n in gone for x in ("-e", n)],
                ref,
                "--",
                f"{K}/{CPP_TREE}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        for hit in p.stdout.split():
            path = hit.split(":", 1)[1] if ":" in hit else hit
            if re.search(r"\.(h|hpp|c|cc|cpp)$", path) and path not in after:
                text = _show(repo, ref, path)
                a_st.update(struct_layouts(text))
                a_fn.update(exported_functions(text))
    for f in hdrs:
        kind = header_kind(f, K)
        out[kind] += [n for n, v in b_st[f].items() if a_st.get(n) != v]
        out["functions"] += [
            (n, "removed" if n not in a_fn else "signature")
            for n, v in b_fn[f].items()
            if a_fn.get(n) != v
        ]
    out["public"], out["generated"] = sorted(set(out["public"])), sorted(
        set(out["generated"])
    )
    out["functions"] = sorted(set(out["functions"]))
    return out


def _names(names: list[str], most: int = 4) -> str:
    seen = list(dict.fromkeys(names))
    return ", ".join(seen[:most]) + (
        f" (+{len(seen) - most})" if len(seen) > most else ""
    )


def _code_lines(diff: str) -> list[str]:
    """Changed non-comment, non-version lines of a C header diff."""
    out = []
    for line in diff.splitlines():
        if not line or line[0] not in "+-" or line.startswith(("+++", "---")):
            continue
        body = line[1:].strip()
        if (
            not body
            or body.startswith(("*", "/*", "//", "*/"))
            or re.match(r"#define\s+ROCKE_(ABI_VERSION|RECIPE_ABI)\b", body)
        ):
            continue
        out.append(line)
    return out


def pr_claims(
    cfg: dict, rec: dict, ref: str, develop_ref: str = "refs/remotes/origin/develop"
) -> dict:
    """What PR rec claims, relative to its merge base with develop and to develop's own history; cached per head and
    develop abi.h in rec['abi_cache']."""
    repo, K = cfg["paths"]["repo"], paths.repository_folder(cfg)
    dev_blob = gitops.git(
        repo, "rev-parse", f"{develop_ref}:{K}/{ABI_H}", check=False
    ).strip()
    cache = rec.get("abi_cache")
    if (
        cache
        and cache.get("head") == rec["head"]
        and cache.get("v") == CACHE_VERSION
        and cache.get("dev_blob") == dev_blob
    ):
        return cache
    mb = gitops.git(repo, "merge-base", develop_ref, ref).strip()
    claim = {
        "head": rec["head"],
        "v": CACHE_VERSION,
        "dev_blob": dev_blob,
        **scan(repo, K, mb, ref, develop_ref),
    }
    rec["abi_cache"] = claim
    return claim


def scan(repo: str, K: str, mb: str, ref: str, develop_ref: str | None = None) -> dict:
    """What ref claims relative to mb: counter bumps (lanes), notes G9 waits on (`rk abi-ok N` after review), info for
    the review, and family ABI strings. With develop_ref, a bump is the PR's own only where its history differs from
    develop's: `own` lists its entries, `taken` those at or under develop's counter (numbers develop already used for
    other changes: the PR must renumber), and a PR whose entries all match develop's carries develop's bump, no claim.
    """
    files = set(gitops.git(repo, "diff", "--name-only", mb, ref).splitlines())
    claim = {"lanes": {}, "families": [], "notes": [], "info": []}
    h = f"{K}/{ABI_H}"
    if h in files:
        before, after = parse_header(_show(repo, mb, h)), parse_header(
            _show(repo, ref, h)
        )
        dev = parse_header(_show(repo, develop_ref, h)) if develop_ref else None
        for lane in DEFINES:
            b, a = before[lane], after[lane]
            if a is None or b is None or a == b:
                continue
            hist_b, hist_a = before[f"hist_{lane}"], after[f"hist_{lane}"]
            dropped = sorted(k for k in hist_b if k not in hist_a)
            hunk = hashlib.sha1(f"{lane}:{a}:{hist_a.get(a, '')}".encode()).hexdigest()[
                :12
            ]
            entry = {
                "from": b,
                "to": a,
                "history_line": a in hist_a,
                "dropped_history": dropped,
                "hunk": hunk,
            }
            if dev and dev[lane] is not None:
                # A PR based on an older develop may hold numbers develop has since used for other changes: never a
                # plain "claims N".
                d, hist_d = dev[lane], dev[f"hist_{lane}"]
                own = [
                    k
                    for k in range(b + 1, a + 1)
                    if k > d or hist_a.get(k) != hist_d.get(k)
                ]
                if not own:
                    claim["info"].append(
                        f"abi.h carries develop's own {lane} ABI {a} (no bump of its own)"
                    )
                    continue
                entry.update(own=own, taken=[k for k in own if k <= d])
            claim["lanes"][lane] = entry
        code = _code_lines(gitops.git(repo, "diff", "-U0", mb, ref, "--", h))
        if code and "binary" not in claim["lanes"]:
            claim["notes"].append(
                f"abi.h code lines change without a binary bump ({len(code)} lines)"
            )
    hdrs = sorted(f for f in files if is_abi_header(f, K) and f != h)
    if hdrs and "binary" not in claim["lanes"]:
        removed = [
            line
            for line in _code_lines(
                gitops.git(repo, "diff", "-U0", mb, ref, "--", *hdrs)
            )
            if line.startswith("-")
        ]
        if removed:
            claim["notes"].append(
                f"public header lines changed or removed without a binary bump: "
                f"{', '.join(x.rsplit('/', 1)[-1] for x in hdrs[:3])} (append-only is fine)"
            )
    if "binary" not in claim["lanes"]:
        # An inserted field has no removed line (a field added mid-struct would read as "no ABI change"); compare the
        # layouts instead. The generated instance_*/helper_* headers' spec and problem structs
        # cross the C API like the public ones (abi.h's history bumps for them: 6, 8, 10, 13-15), and so do their
        # exported functions (structs, and removed accessors).
        moved = surface_changes(repo, K, mb, ref, files)
        if moved["public"]:
            claim["notes"].append(
                f"C struct layouts change without a binary bump: {_names(moved['public'])} (a field "
                f"added anywhere changes the size: bump ROCKE_ABI_VERSION)"
            )
        if moved["generated"]:
            claim["notes"].append(
                f"generated-header struct layouts change without a binary bump: "
                f"{_names(moved['generated'])} (a field added, removed, reordered or retyped moves "
                f"the size or an offset: bump ROCKE_ABI_VERSION and abi.py BINARY_ABI together)"
            )
        if moved["functions"]:
            gone = [n for n, why in moved["functions"] if why == "removed"]
            changed = [n for n, why in moved["functions"] if why == "signature"]
            claim["notes"].append(
                "exported C functions "
                + "; ".join(
                    ([f"removed: {_names(gone)}"] if gone else [])
                    + ([f"signature changed: {_names(changed)}"] if changed else [])
                )
                + " without a binary bump (a caller built against the old header fails to resolve or misreads them: "
                "bump ROCKE_ABI_VERSION and abi.py BINARY_ABI together)"
            )
    if "recipe" in claim["lanes"] and f"{K}/{ABI_PY}" not in files:
        claim["notes"].append(
            "wire ABI bump without registering the construct in portable_ir/src/abi.py"
        )
    if h in files or f"{K}/{ABI_PY}" in files:
        hdr, mir = parse_header(_show(repo, ref, h)), parse_mirrors(
            _show(repo, ref, f"{K}/{ABI_PY}")
        )
        for lane, name in MIRRORS.items():
            if None not in (hdr[lane], mir[lane]) and hdr[lane] != mir[lane]:
                claim["notes"].append(
                    f"portable_ir/src/abi.py {name} = {mir[lane]} but abi.h {DEFINES[lane]} = "
                    f"{hdr[lane]}: bump both together (the shared engine refuses to load)"
                )
    fam_files = gitops.git(
        repo, "diff", "--name-only", "-G", "ABI_VERSION", mb, ref, "--", K
    ).split()
    if fam_files:
        fam = {}
        for sign, name, value in FAMILY_LINE.findall(
            gitops.git(repo, "diff", "-U0", mb, ref, "--", *fam_files)
        ):
            fam.setdefault(name, {})["old" if sign == "-" else "new"] = value
        for name, ch in sorted(fam.items()):
            kind = (
                "new family"
                if "old" not in ch
                else (
                    "removed"
                    if "new" not in ch
                    else ("changed" if ch["old"] != ch["new"] else "moved")
                )
            )
            claim["families"].append({"name": name, "kind": kind, **ch})
    return claim


def _stack_root(rec: dict, by_ref: dict, prs: dict) -> int:
    seen, cur = set(), rec
    while cur["base"] in by_ref and cur["n"] not in seen:
        seen.add(cur["n"])
        cur = prs[str(by_ref[cur["base"]])]
    return cur["n"]


def _claims(
    lane: str, open_recs: list[dict], by_ref: dict, prs: dict
) -> dict[str, dict]:
    """Lane `lane`'s claims, {key: {"members", "hunk", "to", "taken"}}: one per stack (under its bottom PR), and PRs whose
    bump hunks are identical in one. No order: who is in a claim does not depend on the queue's order.
    """
    groups: dict[str, dict] = {}
    for r in open_recs:
        c = r.get("abi_cache") or {}
        taken = []
        if lane in DEFINES:
            cl = c.get("lanes", {}).get(lane)
            if not cl:
                continue
            key, to = f"root{_stack_root(r, by_ref, prs)}", cl["to"]
            hunk, taken = (
                cl.get("hunk") or f"#{r['n']}",
                cl.get("taken") or [],
            )  # no hunk recorded: a claim of its own
        else:
            f = next(
                (
                    f
                    for f in c.get("families", [])
                    if f["name"] == lane and f["kind"] == "changed"
                ),
                None,
            )
            if not f:
                continue
            key, to, hunk = f"root{_stack_root(r, by_ref, prs)}", f["new"], f["new"]
        # identical hunks in different stacks are one claim
        same = next(
            (k for k, g in groups.items() if g["hunk"] == hunk and k != key), None
        )
        g = groups.setdefault(
            same or key, {"members": [], "hunk": hunk, "to": to, "taken": []}
        )
        g["members"].append(r["n"])
        g["taken"] = sorted(set(g["taken"]) | set(taken))
    return groups


def lane_claims(open_recs: list[dict]) -> dict[str, list[dict]]:
    """The binary and wire lanes' claims as build_lanes groups them ({"members", "to", "taken"} each), in no order. Two
    PRs in one lane but not one claim both set the counter, so the later to land renumbers to develop + 1: the merge
    plan's ABI conflicts. The plan reads them before it orders the lanes, and who is in
    a lane never depends on that order, so the lanes can follow the plan without a loop.
    """
    prs = {str(r["n"]): r for r in open_recs}
    by_ref = {r.get("head_ref"): r["n"] for r in open_recs}
    return {
        lane: [
            {k: g[k] for k in ("members", "to", "taken")}
            for g in _claims(lane, open_recs, by_ref, prs).values()
        ]
        for lane in DEFINES
    }


def build_lanes(
    dev: dict, open_recs: list[dict], order: list[int], ready: set[int]
) -> dict:
    """Lanes for the binary and wire counters, plus family strings that change an existing value.

    order: PR numbers in queue order. ready: PRs whose non-ABI gates are green.
    Returns {lane: {"develop": v, "groups": [...], "head": group index or None}}; each group is one claim (a stack or
    PRs with an identical hunk): {"members": [n...], "root": n, "to": v, "reserve": v, "ready": bool}.
    """
    prs = {str(r["n"]): r for r in open_recs}
    by_ref = {r["head_ref"]: r["n"] for r in open_recs}
    pos = {n: i for i, n in enumerate(order)}
    lanes = {}
    lane_names = list(DEFINES) + sorted(
        {
            f["name"]
            for r in open_recs
            for f in (r.get("abi_cache") or {}).get("families", [])
            if f["kind"] == "changed"
        }
    )
    for lane in lane_names:
        glist = []
        for g in _claims(lane, open_recs, by_ref, prs).values():
            g["members"].sort(key=lambda n: pos.get(n, 10**6))
            root = min(
                g["members"],
                key=lambda n: (prs[str(n)]["base"] != "develop", pos.get(n, 10**6)),
            )
            g.update(root=root, ready=root in ready)
            glist.append(g)
        glist.sort(key=lambda g: (not g["ready"], pos.get(g["root"], 10**6), g["root"]))
        base = dev.get(lane) if lane in DEFINES else None
        for i, g in enumerate(glist):
            g["reserve"] = (base + 1 + i) if isinstance(base, int) else None
            g.pop("hunk", None)
        head = 0 if glist and glist[0]["ready"] else None
        lanes[lane] = {"develop": base, "groups": glist, "head": head}
    return lanes


def lane_lines(st: dict, n: int | None = None) -> list[str]:
    """What `rk abi` prints from the last queue computation: each lane with develop's number and its head group, the
    groups in landing order with their claims and tentative numbers, then the open PRs' ABI notes. With `n`, only PR
    n's lanes and notes (`rk facts`)."""
    out = []
    for lane, L in (st.get("abi_lanes") or {}).items():
        groups = [
            (i, g)
            for i, g in enumerate(L["groups"], 1)
            if n is None or n in g["members"]
        ]
        if n is not None and not groups:
            continue
        # "head" is the lane's ready group index: say "next to land", never "head None".
        nxt = (
            "none ready"
            if L["head"] is None
            else ", ".join(f"#{m}" for m in L["groups"][L["head"]]["members"])
        )
        out.append(f"{lane}: develop {L['develop']}, next to land: {nxt}")
        out += [
            f"  {i}. {', '.join('#' + str(m) for m in g['members'])} claims {g['to']} -> tentative {g['reserve']} "
            f"ready={g['ready']}"
            for i, g in groups
        ]
    for k, rec in sorted(st["prs"].items(), key=lambda kv: int(kv[0])):
        c = rec.get("abi_cache") or {}
        if (
            (n is None or int(k) == n)
            and rec.get("status") == "open"
            and (c.get("notes") or c.get("families") or c.get("info"))
        ):
            out.append(
                f"#{k}: notes={c.get('notes')} families={c.get('families')}"
                + (f" info={c['info']}" if c.get("info") else "")
            )
    return out


def claim_text(g: dict) -> str:
    """A lane group's claim as shown: the number it sets, and the numbers in it that develop already used for other
    changes (the PR renumbers at landing and keeps develop's history lines)."""
    taken = g.get("taken") or []
    if not taken:
        return str(g["to"])
    span = f"{taken[0]}–{taken[-1]}" if len(taken) > 1 else str(taken[0])
    return f"{g['to']} ({span} taken on develop by other changes: renumber)"


def gate_abi(rec: dict, lanes: dict, waived: set) -> tuple:
    """G9 for one PR: pass/fail/pending with a note. First in line while no lane member can land yet is pending, not a
    wait behind itself."""
    c = rec.get("abi_cache") or {}
    problems, waits, nexts = [], [], []
    for lane, L in lanes.items():
        for i, g in enumerate(L["groups"]):
            if rec["n"] not in g["members"]:
                continue
            title = LANE_TITLE.get(lane, f"{lane} string")
            stale = f"; claims {claim_text(g)}" if g.get("taken") else ""
            if (
                L["head"] != i and i == 0
            ):  # nobody ahead: it takes the number once it can land
                nexts.append(
                    f"{title} lane position 1: it takes the next number once it can land"
                    + (f"; tentative number {g['reserve']}" if g["reserve"] else "")
                    + stale
                )
                continue
            if L["head"] != i:
                ahead = L["groups"][0]
                waits.append(
                    (
                        f"{title} lane position {i + 1} (behind #{ahead['root']}); tentative number "
                        f"{g['reserve']}"
                        if g["reserve"]
                        else f"{title} lane position {i + 1}"
                    )
                    + stale
                )
                continue
            if lane in DEFINES:
                cl = c["lanes"][lane]
                if cl["to"] != L["develop"] + 1:
                    problems.append(
                        f"{title} lane head: set {DEFINES[lane]} to {L['develop'] + 1} "
                        f"(claims {cl['to']}, develop is at {L['develop']})"
                        + (
                            f"; keep develop's history lines {cl['taken']}"
                            if cl.get("taken")
                            else ""
                        )
                    )
                if not cl["history_line"]:
                    problems.append(
                        f"add the ` * {cl['to']}: …` history line above {DEFINES[lane]}"
                    )
                if cl["dropped_history"]:
                    problems.append(
                        f"restore dropped history lines {cl['dropped_history']}"
                    )
    notes = [n for n in c.get("notes", []) if (rec["n"], rec["head"]) not in waived]
    if problems:
        return "fail", "; ".join(problems)
    if waits:
        return "fail", "; ".join(waits)
    if notes or nexts:
        return "pending", "; ".join(
            nexts
            + (
                ["verify in review: " + "; ".join(notes) + " (then `rk abi-ok N`)"]
                if notes
                else []
            )
        )
    if c.get("lanes"):
        return "pass", "ABI lane head; " + ", ".join(
            f"{LANE_TITLE[k]} {v['from']}->{v['to']}" for k, v in c["lanes"].items()
        )
    return "pass", "no ABI change"
