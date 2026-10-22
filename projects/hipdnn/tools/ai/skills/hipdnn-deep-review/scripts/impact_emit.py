#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the kernel ABI and emitted-IR facts of every shipped rocKE variant at two revisions.

  run --repo DIR --base REV --head REV [--flavor F] [--timeout S] --out FILE.json

REV is any commit or tree id. Each side is `git archive`d into a temp dir (the repo is
only read) and a child interpreter with a scrubbed environment builds every shipped rocKE
variant there, lowers it to LLVM IR with the Python backend (no GPU, comgr or network),
and records the kernarg list parsed from the IR `define` line plus the SHA-256 of the IR
text. This EXECUTES rocKE code from the reviewed revisions, like building the head does.

Variants come from every `*.kdp.json` under the shipped descriptor root whose
`kernelDescriptors[].kernel_source` names a rocKE `source` + `builder`. A KDP held by
DVC (`artifacts.dvc`, no `.kdp.json`) is regenerated with that side's
`IngestorGenerator/tools/registry_enumerator.py`; when neither works the KDP is listed
under `not_inspected`, never treated as equal. A variant is its descriptor `name`; a
family is (compile arch, source, builder): a `target: generic` variant compiles once per generic
target (`gfx11-generic`), so the shard arches that share it collapse into one family whose
`members` lists them. A descriptor's own `arch` narrows the KDP-level one. `ir_changed` lists every variant whose IR digest
differs, so it includes `params_changed`. A variant that fails to build is recorded under
`failures` and takes no part in the comparison.

The flavor defaults to the top rung of `_LLVM_FLAVOR_LADDER` in the head's
`rocke/platform/python/rocke/core/lower_llvm.py`. Exit 0 when the analysis ran, whatever
it found; 1 with one line `impact_emit: <reason>` on stderr when it could not run.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROVIDER = "dnn-providers/hip-kernel-provider"
ROCKE_PLATFORM = f"{PROVIDER}/rocke/platform/python"
ROCKE_LIBRARY = f"{PROVIDER}/rocke/library"
PACKAGING = f"{PROVIDER}/descriptor-packaging/python"
DESCRIPTORS = f"{PROVIDER}/src/engines/kernel_ingestor_engine/descriptors"
GENERATOR = "projects/hipdnn/tools/IngestorGenerator"
ENUMERATOR = f"{GENERATOR}/tools/registry_enumerator.py"
MANIFESTS = f"{GENERATOR}/configs"
LOWER_LLVM = f"{ROCKE_PLATFORM}/rocke/core/lower_llvm.py"

REQUIRED_TREES = (ROCKE_PLATFORM, ROCKE_LIBRARY, PACKAGING)
OPTIONAL_TREES = (DESCRIPTORS, GENERATOR)
SCRUBBED_PREFIXES = ("ROCKE_", "HIP_", "ROCM_", "PYTHON")


class Fatal(Exception):
    """The analysis could not run; the message is the one stderr line."""


# --------------------------------------------------------------------------- git


def git(repo, *args, text=True):
    cmd = ["git", "-C", str(repo), "-c", "core.quotepath=off", *args]
    try:
        r = subprocess.run(cmd, capture_output=True)
    except OSError as e:
        raise Fatal(f"cannot run git: {e}")
    if r.returncode:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise Fatal(f"git {args[0]} failed: {err[-1] if err else r.returncode}")
    return r.stdout.decode("utf-8", "replace") if text else r.stdout


def existing_paths(repo, rev, paths):
    out = git(repo, "ls-tree", "--name-only", rev, "--", *paths)
    return set(out.splitlines())


def extract(repo, rev, dest):
    """`git archive` the trees needed from REV into DEST (regular files and dirs only)."""
    have = existing_paths(repo, rev, REQUIRED_TREES + OPTIONAL_TREES)
    missing = [p for p in REQUIRED_TREES if p not in have]
    if missing:
        raise Fatal(f"{rev} lacks {', '.join(missing)}")
    data = git(
        repo,
        "archive",
        "--format=tar",
        rev,
        "--",
        *sorted(have),
        text=False,
    )
    root = Path(dest).resolve()
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf:
            target = (root / m.name).resolve()
            if root != target and root not in target.parents:
                raise Fatal(f"{rev}: archive member escapes the extract: {m.name}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())


def default_flavor(repo, rev):
    """Top rung of `_LLVM_FLAVOR_LADDER` in REV's lower_llvm.py, parsed without importing.
    Older trees have no ladder; their newest flavor is the last entry of `LLVM_FLAVORS`.
    """
    src = git(repo, "show", f"{rev}:{LOWER_LLVM}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise Fatal(f"cannot parse {LOWER_LLVM} at {rev}: {e}; pass --flavor")
    consts, ladder, flavors = {}, None, None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names, value = [
                t for t in node.targets if isinstance(t, ast.Name)
            ], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names, value = [node.target], node.value
        else:
            continue
        for n in names:
            if n.id == "_LLVM_FLAVOR_LADDER":
                ladder = value
            elif n.id == "LLVM_FLAVORS":
                flavors = value
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                consts[n.id] = value.value
    if ladder is None or not isinstance(ladder, (ast.Tuple, ast.List)):
        names = flavors.elts if isinstance(flavors, (ast.Tuple, ast.List)) else []
        known = [
            consts.get(e.id) if isinstance(e, ast.Name) else getattr(e, "value", None)
            for e in names
        ]
        if known and isinstance(known[-1], str):
            return known[-1]
        raise Fatal(
            f"no _LLVM_FLAVOR_LADDER or LLVM_FLAVORS in {LOWER_LLVM} at {rev}; pass --flavor"
        )
    rungs = []
    for el in ladder.elts:
        if not (isinstance(el, ast.Tuple) and len(el.elts) == 2):
            continue
        major, flavor = el.elts
        if isinstance(flavor, ast.Name):
            flavor = consts.get(flavor.id)
        elif isinstance(flavor, ast.Constant):
            flavor = flavor.value
        else:
            flavor = None
        if isinstance(major, ast.Constant) and isinstance(flavor, str):
            rungs.append((major.value, flavor))
    if not rungs:
        raise Fatal(f"cannot resolve _LLVM_FLAVOR_LADDER at {rev}; pass --flavor")
    return max(rungs)[1]


# ----------------------------------------------------------------------- IR parse

_ATTR_ADDRSPACE = re.compile(r"addrspace\(\d+\)$")


def _split_top(text, sep=","):
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts]


def _balanced_close(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _param_type(text):
    """The leading type of one parameter, as written (`ptr addrspace(1)`, `<4 x float>`, `i32`)."""
    text = text.strip()
    if text and text[0] in "<[{":
        close = {"<": ">", "[": "]", "{": "}"}[text[0]]
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == text[0]
            depth -= ch == close
            if depth == 0:
                return text[: i + 1]
        return text
    words = text.split()
    if not words:
        return ""
    if words[0] == "ptr" and len(words) > 1 and _ATTR_ADDRSPACE.match(words[1]):
        return f"ptr {words[1]}"
    return words[0]


def parse_kernargs(ir_text):
    """[(name, type)] of the kernel's `define ... @k(...)` line.

    The entry point is the first `define` mentioning `amdgpu_kernel`, else the first
    `define`. Raises ValueError when the text has none.
    """
    lines = [ln for ln in ir_text.splitlines() if ln.lstrip().startswith("define")]
    if not lines:
        raise ValueError("no `define` line in the IR")
    line = next((ln for ln in lines if "amdgpu_kernel" in ln), lines[0])
    at = line.find("@")
    open_ = line.find("(", at)
    if at < 0 or open_ < 0:
        raise ValueError(f"unparsable define line: {line[:120]}")
    close = _balanced_close(line, open_)
    if close < 0:
        raise ValueError(f"unbalanced define line: {line[:120]}")
    params = []
    for raw in _split_top(line[open_ + 1 : close]):
        m = re.search(r"%(\"[^\"]*\"|[\w.$-]+)\s*$", raw)
        name = m.group(1).strip('"') if m else ""
        body = raw[: m.start()] if m else raw
        params.append([name, _param_type(body)])
    return params


# -------------------------------------------------------------------------- child
# Everything below up to `parent side` runs in the extract's scrubbed interpreter.


def _arch_list(value):
    if isinstance(value, str):
        return [value]
    return [a for a in value or [] if isinstance(a, str)]


def _discover(root):
    """(variants, not_inspected) of one extract: variants are dicts with name, arch,
    source, builder, spec, target, kdp, origin."""
    base = Path(root)
    desc = base / DESCRIPTORS
    variants, skipped = [], []
    if not desc.is_dir():
        return variants, skipped
    dvc_dirs = []
    for dirpath, dirnames, filenames in os.walk(desc):
        dirnames.sort()
        here = Path(dirpath)
        kdps = sorted(f for f in filenames if f.endswith(".kdp.json"))
        for f in kdps:
            rel = (here / f).relative_to(base).as_posix()
            try:
                doc = json.loads((here / f).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                skipped.append({"kdp": rel, "reason": f"unreadable KDP: {e}"})
                continue
            for d in doc.get("kernelDescriptors") or []:
                ks = d.get("kernel_source") or {}
                if not (ks.get("source") and ks.get("builder")):
                    continue
                # A descriptor's own `arch` narrows the KDP-level one.
                arches = _arch_list(d.get("arch")) or _arch_list(doc.get("arch"))
                if not arches:
                    skipped.append({"kdp": rel, "reason": f"{d['name']}: no `arch`"})
                    continue
                variants.append(
                    dict(
                        name=d["name"],
                        arches=arches,
                        source=ks["source"],
                        builder=ks["builder"],
                        spec=ks.get("spec") or {},
                        target=ks.get("target", "native"),
                        kdp=rel,
                        origin="kdp-in-git",
                    )
                )
        if "artifacts.dvc" in filenames and not kdps:
            dvc_dirs.append(here)
    for d in dvc_dirs:
        got, why = _regenerate(base, d)
        if why:
            kdp = (d / f"{d.name}.kdp.json").relative_to(base).as_posix()
            skipped.append({"kdp": kdp, "reason": why})
        else:
            variants.extend(got)
    return variants, skipped


def _regenerate(base, dvc_dir):
    """Variants of a DVC-held KDP from the registry enumerator, or (None, reason)."""
    rel_dir = dvc_dir.relative_to(base / DESCRIPTORS).as_posix()
    kdp = (dvc_dir / f"{dvc_dir.name}.kdp.json").relative_to(base).as_posix()
    enum = base / ENUMERATOR
    cfgs = base / MANIFESTS
    if not enum.is_file():
        return (
            None,
            "KDP is DVC-held and registry_enumerator.py is absent at this revision",
        )
    try:
        import yaml
    except ImportError:
        return None, "KDP is DVC-held and PyYAML is not importable"
    manifest = None
    for p in sorted(cfgs.glob("*.engine.yaml")) if cfgs.is_dir() else []:
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(doc, dict) and doc.get("authored_subpath") == rel_dir:
            manifest = p
            break
    if manifest is None:
        return (
            None,
            f"KDP is DVC-held and no *.engine.yaml manifest has authored_subpath {rel_dir}",
        )
    out = base / ".impact_emit" / f"{dvc_dir.name}.enum.yaml"
    out.parent.mkdir(exist_ok=True)
    try:
        r = subprocess.run(
            [sys.executable, str(enum), "--manifest", str(manifest), "--out", str(out)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(base),
            timeout=int(os.environ.get("IMPACT_EMIT_ENUM_TIMEOUT", "600")),
        )
    except subprocess.TimeoutExpired:
        return None, "registry_enumerator.py timed out"
    if r.returncode:
        tail = (r.stderr or r.stdout).strip().splitlines()
        return (
            None,
            f"registry_enumerator.py exited {r.returncode}: {tail[-1] if tail else ''}",
        )
    try:
        pack = yaml.safe_load(out.read_text(encoding="utf-8"))["packs"][0]
    except (OSError, KeyError, IndexError, TypeError, yaml.YAMLError) as e:
        return None, f"registry_enumerator.py output unusable: {type(e).__name__}: {e}"
    got = []
    for k in pack.get("kernels") or []:
        ks = k.get("kernel_source") or {}
        if not (ks.get("source") and ks.get("builder")):
            continue
        arches = _arch_list(k.get("arch")) or _arch_list(pack.get("arch"))
        if not arches:
            return None, f"registry_enumerator.py output has no arch for {k['name']}"
        got.append(
            dict(
                name=k["name"],
                arches=arches,
                source=ks["source"],
                builder=ks["builder"],
                spec=ks.get("spec") or {},
                target=ks.get("target", "native"),
                kdp=kdp,
                origin="registry_enumerator",
            )
        )
    if not got:
        return None, "registry_enumerator.py produced no rocKE kernels"
    return got, None


class _Helpers:
    """The hkp_pack / rocke helpers resolved by name; private names move between refs."""

    def __init__(self):
        self.error = None
        self.missing_optional = set()
        try:
            from hkp_pack import rocke_compile as rc

            from rocke.helpers import compile as rcompile
        except Exception as e:
            self.error = f"helpers not importable: {type(e).__name__}: {e}"
            return
        need = {}
        for name in ("_module_from_source", "_resolve_spec_class", "build_spec"):
            need[name] = getattr(rc, name, None)
        need["_lower_llvm_via_backend"] = getattr(
            rcompile, "_lower_llvm_via_backend", None
        )
        absent = [n for n, f in need.items() if not callable(f)]
        if absent:
            self.error = f"helpers missing at this revision: {', '.join(absent)}"
            return
        self.module_from_source = need["_module_from_source"]
        self.resolve_spec_class = need["_resolve_spec_class"]
        self.build_spec = need["build_spec"]
        self.lower = need["_lower_llvm_via_backend"]
        self.compile_arch = getattr(rc, "rocke_compile_arch", None)
        self.require_arch_sig = getattr(rc, "_require_spec_arch_signature", None)
        self.check_predicate = getattr(rc, "_check_support_predicate", None)


def _compile_arch(h, shard, target):
    if callable(h.compile_arch):
        return h.compile_arch(shard, target)
    if target != "native":
        raise RuntimeError(
            f"rocke_compile_arch missing; cannot resolve target {target!r}"
        )
    return shard


def _build_one(h, v, arch):
    from importlib import import_module

    module = import_module(h.module_from_source(v["source"]))
    fn = getattr(module, v["builder"])
    cls = h.resolve_spec_class(module, fn)
    if callable(h.require_arch_sig):
        h.require_arch_sig(fn, v["builder"])
    obj = h.build_spec(cls, v["spec"])
    if callable(h.check_predicate):
        h.check_predicate(module, v["builder"], obj, arch)
    kd = fn(obj, arch=arch)
    text = h.lower(kd, arch=arch, backend="python", spec=None)
    return {
        "params": parse_kernargs(text),
        "ir": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def child_main(root, out):
    # The script's own directory must not shadow rocKE modules.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    t0 = time.time()
    variants, skipped = _discover(root)
    h = _Helpers()
    fams = {}
    # One family per (compile arch, source, builder): the packer compiles once per
    # compile arch, so shard arches that share a generic target collapse into one family.
    for v in variants:
        for shard in v["arches"]:
            err = h.error
            carch = shard
            if not err:
                try:
                    carch = _compile_arch(h, shard, v["target"])
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:300]}"
            key = (carch, v["source"], v["builder"])
            fam = fams.setdefault(
                key,
                {
                    "arch": carch,
                    "source": v["source"],
                    "builder": v["builder"],
                    "kdp": v["kdp"],
                    "spec_origin": v["origin"],
                    "members": [],
                    "names": [],
                    "built": {},
                    "failures": {},
                },
            )
            if shard not in fam["members"]:
                fam["members"].append(shard)
            if v["name"] in fam["names"]:
                continue  # same variant reached through another member arch
            fam["names"].append(v["name"])
            if err:
                fam["failures"][v["name"]] = err
                continue
            try:
                fam["built"][v["name"]] = _build_one(h, v, carch)
            except Exception as e:
                fam["failures"][v["name"]] = f"{type(e).__name__}: {str(e)[:300]}"
    doc = {
        "families": list(fams.values()),
        "not_inspected": skipped,
        "seconds": round(time.time() - t0, 1),
    }
    Path(out).write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    return 0


# ------------------------------------------------------------------- parent side


def scrubbed_env(root, flavor):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(SCRUBBED_PREFIXES)
    }
    base = Path(root)
    env["PYTHONPATH"] = os.pathsep.join(
        str(base / p) for p in (ROCKE_PLATFORM, ROCKE_LIBRARY, PACKAGING)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["ROCKE_LLVM_FLAVOR"] = flavor
    env["ROCKE_PROVIDER_ROOT"] = str(base / PROVIDER)
    return env


def run_side(repo, rev, flavor, timeout, label):
    tmp = tempfile.mkdtemp(prefix="impact_emit_")
    try:
        extract(repo, rev, tmp)
        out = Path(tmp) / ".impact_emit_side.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "child",
            "--root",
            tmp,
            "--out",
            str(out),
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tmp,
                env=scrubbed_env(tmp, flavor),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise Fatal(f"{label} ({rev}) timed out after {timeout}s")
        if r.returncode or not out.is_file():
            tail = (r.stderr or r.stdout).strip().splitlines()
            raise Fatal(
                f"{label} ({rev}) child exited {r.returncode}: {tail[-1] if tail else ''}"
            )
        return json.loads(out.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare(base_side, head_side):
    def by_key(side):
        return {(f["arch"], f["source"], f["builder"]): f for f in side["families"]}

    b, h = by_key(base_side), by_key(head_side)
    out = []
    for key in sorted(set(b) | set(h)):
        fb, fh = b.get(key), h.get(key)
        names_b = fb["names"] if fb else []
        names_h = fh["names"] if fh else []
        built_b = fb["built"] if fb else {}
        built_h = fh["built"] if fh else {}
        common = [n for n in names_h if n in built_h and n in built_b]
        params_changed = [
            n for n in common if built_b[n]["params"] != built_h[n]["params"]
        ]
        ir_changed = [n for n in common if built_b[n]["ir"] != built_h[n]["ir"]]
        shown = (params_changed or common or [None])[0]
        if shown is None:
            shown = next((n for n in names_h if n in built_h), None)
        if shown is None:
            shown = next((n for n in names_b if n in built_b), None)
        pick = fh or fb
        members = sorted(
            set((fb or {}).get("members", [])) | set((fh or {}).get("members", []))
        )
        out.append(
            {
                "arch": key[0],
                "source": key[1],
                "builder": key[2],
                "kdp": pick["kdp"],
                "spec_origin": pick["spec_origin"],
                "members": members,
                "variants": {"base": len(names_b), "head": len(names_h)},
                "only_in_base": [n for n in names_b if n not in names_h],
                "only_in_head": [n for n in names_h if n not in names_b],
                "params_changed": params_changed,
                "ir_changed": ir_changed,
                "params": {
                    "base": built_b[shown]["params"] if shown in built_b else None,
                    "head": built_h[shown]["params"] if shown in built_h else None,
                },
                "failures": {
                    "base": dict(fb["failures"]) if fb else {},
                    "head": dict(fh["failures"]) if fh else {},
                },
            }
        )
    return out


def cmd_run(a):
    repo = Path(a.repo)
    if not repo.is_dir():
        raise Fatal(f"--repo {a.repo} is not a directory")
    flavor = a.flavor or default_flavor(repo, a.head)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:  # the sides are independent
        fb = pool.submit(run_side, repo, a.base, flavor, a.timeout, "base")
        fh = pool.submit(run_side, repo, a.head, flavor, a.timeout, "head")
        base_side, head_side = fb.result(), fh.result()
    families = compare(base_side, head_side)
    not_inspected = [
        {"kdp": n["kdp"], "side": side, "reason": n["reason"]}
        for side, doc in (("base", base_side), ("head", head_side))
        for n in doc["not_inspected"]
    ]
    seconds = round(time.time() - t0, 1)
    doc = {
        "schema": 1,
        "tool": "impact_emit",
        "flavor": flavor,
        "base": a.base,
        "head": a.head,
        "families": families,
        "not_inspected": not_inspected,
        "seconds": seconds,
    }
    try:
        Path(a.out).write_text(
            json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError as e:
        raise Fatal(f"cannot write {a.out}: {e}")
    for f in families:
        fails = len(f["failures"]["base"]) + len(f["failures"]["head"])
        print(
            f"{f['arch']} {f['source']}:{f['builder']} variants {f['variants']['base']}/"
            f"{f['variants']['head']} only_base {len(f['only_in_base'])} only_head "
            f"{len(f['only_in_head'])} params_changed {len(f['params_changed'])} "
            f"ir_changed {len(f['ir_changed'])} failures {fails} ({f['spec_origin']})"
        )
    for n in not_inspected:
        print(f"not inspected [{n['side']}] {n['kdp']}: {n['reason']}")
    print(f"flavor {flavor}, {len(families)} families, {seconds}s")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compare two revisions")
    r.add_argument("--repo", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--flavor")
    r.add_argument("--timeout", type=int, default=900, help="seconds per side")
    r.add_argument("--out", required=True)
    c = sub.add_parser("child")  # internal: runs inside the scrubbed interpreter
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "child":
        return child_main(args.root, args.out)
    try:
        return cmd_run(args)
    except Fatal as e:
        print(f"impact_emit: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
