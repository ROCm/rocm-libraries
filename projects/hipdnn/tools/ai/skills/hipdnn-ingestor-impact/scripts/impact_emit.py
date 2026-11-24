#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit the kernel ABI and emitted-IR facts of every shipped rocKE variant at two revisions.

  run --repo DIR --base REV --head REV [--flavor F] [--timeout S] --out FILE.json

REV is any commit or tree id. Each side is `git archive`d into a temp dir (the repo is
only read) and a child interpreter with an allow-listed environment builds every shipped rocKE
variant there, lowers it to LLVM IR with the Python backend (no GPU, comgr or network),
and records the kernarg list parsed from the IR `define` line, the SHA-256 of the IR
text, and the COMGR inputs the packer adds to that text (ISA name, codegen options,
fresh-process compile): a code object can change while its IR does not. A builder that
returns a `KernelModule` is lowered kernel by kernel, as the packer does: `params` is the
primary's list, `companions` each companion launch's list by role, and the digest covers
every kernel. A packer that pins another lowering backend fails every variant. This
EXECUTES rocKE code from the reviewed revisions, like building the head does.

Variants come from every `*.kdp.json` under the shipped descriptor root whose
`kernelDescriptors[].kernel_source` names a rocKE `source` + `builder`. A UKD with
`sources` (several launches) yields one variant per distinct (source, builder, spec,
target) of its launches, as hkp_pack compiles them: the UKD `name` when there is one, else
`<name>@<launch>[+<launch>...]` with the group's launch names sorted. A UKD whose launches all
share one other packer kind (hip, hsaco, kpack, embedded_source) has no rocKE build: it gives no
variant and is not listed, as a single hip launch is not. A UKD that cannot be read (launches that
mix kinds, no source or builder, a field that is not a string, both `kernel_source` and
`sources`) is listed under `not_inspected`. A KDP held by
DVC (`artifacts.dvc`, no `.kdp.json`) is regenerated with that side's
`IngestorGenerator/tools/registry_enumerator.py`; when neither works the KDP is listed
under `not_inspected`, never treated as equal. A variant is its descriptor `name`; a
family is (compile arch, source, builder): a `target: generic` variant compiles once per generic
target (`gfx11-generic`), so the shard arches that share it collapse into one family whose
`members` lists them. A descriptor's own `arch` narrows the KDP-level one. `ir_changed` lists every variant whose IR digest
differs, so it includes `params_changed`. A variant that fails to build is recorded under
`failures` and takes no part in the comparison.

Predicate tables (`predicate_tables`, head side). For each family's builder, and for each other
`build_*` function of its module that the builds called (a KernelModule's kernel builders), the
child takes the shipped specs (for an inner builder: the specs the builds passed it), varies one
int or bool field at a time from up to four seeds (the shipped values, 0, -1, and the negation,
double, half and 1.5 times of each shipped positive value), and asks the predicate the packer
resolves (`_resolve_support_predicate`). An accepted spec that the builder raises on is a
`builder` finding. Between two builders of one family, a value outside the sign range of every
shipped value (<= 0 where all are >= 1, < 0 where all are >= 0) of a field they share that one
predicate accepts and the other refuses is a `sign` finding on the one that accepts. A finding
the base's table also has (same check, same field) is marked `at_base`. A family's tables stop
after `IMPACT_EMIT_TABLE_SECONDS` (default 60) and say `truncated`. The probes run in a worker
thread: a builder that never returns on a spec the predicate accepts costs only its family
(and the families after it) the tables, which read `skipped: "probe timed out"`, once the
family is 5 s past its budget; the families and `not_inspected` are still written and compared.

The flavor defaults to the newest flavor the head and the base both know (the top rung
of `_LLVM_FLAVOR_LADDER` in `rocke/platform/python/rocke/core/llvm_profile.py`, or in
`lower_llvm.py` in trees from before that module, else the last
of `LLVM_FLAVORS`); with none shared it is the head's newest and the base fails loudly.
The child gets an allow-listed environment and runs in its own process group, which
`--timeout` kills as a whole. Each side's tree id (`base_tree`, `head_tree`, resolved once
with `git rev-parse REV^{tree}` and the one that is archived) is recorded so impact.py
can refuse a file made for other revisions. Exit 0 when the analysis ran, whatever
it found; 1 with one line `impact_emit: <reason>` on stderr when it could not run.
"""

import argparse
import ast
import dataclasses
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
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
#: Where newer trees define the flavor ladder; lower_llvm.py re-exports it from here.
LLVM_PROFILE = f"{ROCKE_PLATFORM}/rocke/core/llvm_profile.py"

REQUIRED_TREES = (ROCKE_PLATFORM, ROCKE_LIBRARY, PACKAGING)
OPTIONAL_TREES = (DESCRIPTORS, GENERATOR)


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


def resolve_tree(repo, rev):
    """The tree id REV names (a commit or a tree); archived and recorded so that a
    consumer can tell the file belongs to the revisions it asked about."""
    return git(repo, "rev-parse", "--verify", f"{rev}^{{tree}}").strip()


def existing_paths(repo, rev, paths):
    out = git(repo, "ls-tree", "--name-only", rev, "--", *paths)
    return set(out.splitlines())


def extract(repo, rev, dest, name=None):
    """`git archive` the trees needed from REV into DEST (regular files and dirs only).
    NAME is how errors refer to REV."""
    name = name or rev
    have = existing_paths(repo, rev, REQUIRED_TREES + OPTIONAL_TREES)
    missing = [p for p in REQUIRED_TREES if p not in have]
    if missing:
        raise Fatal(f"{name} lacks {', '.join(missing)}")
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
                raise Fatal(f"{name}: archive member escapes the extract: {m.name}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())


def _known_flavors(src, where, path=LOWER_LLVM):
    """Flavors of one flavor-defining source (PATH), oldest first, parsed without importing:
    the rungs of `_LLVM_FLAVOR_LADDER` by major version; older trees have no ladder, so
    their `LLVM_FLAVORS` in order.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise Fatal(f"cannot parse {path} at {where}: {e}; pass --flavor")
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
            return [k for k in known if isinstance(k, str)]
        raise Fatal(
            f"no _LLVM_FLAVOR_LADDER or LLVM_FLAVORS in {path} at {where}; pass --flavor"
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
        raise Fatal(f"cannot resolve _LLVM_FLAVOR_LADDER at {where}; pass --flavor")
    return [flavor for _, flavor in sorted(rungs)]


def _flavors_at(repo, rev):
    """Known flavors of REV, oldest first, read from `llvm_profile.py` when the tree has it
    and from `lower_llvm.py` before that module existed."""
    path = LLVM_PROFILE if existing_paths(repo, rev, [LLVM_PROFILE]) else LOWER_LLVM
    return _known_flavors(git(repo, "show", f"{rev}:{path}"), rev, path)


def default_flavor(repo, rev):
    """Newest flavor of REV: the top ladder rung, else the last entry of `LLVM_FLAVORS`."""
    return _flavors_at(repo, rev)[-1]


def shared_default_flavor(repo, base, head):
    """The head's newest flavor that the base also knows, so a base from before the head's
    top rung is not failed wholesale. With none shared, the head's top (it then fails
    loudly); a base whose flavors cannot be read leaves the head's top as well."""
    head_known = _flavors_at(repo, head)
    try:
        base_known = set(_flavors_at(repo, base))
    except Fatal:
        return head_known[-1]
    return next((f for f in reversed(head_known) if f in base_known), head_known[-1])


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
# Everything below up to `parent side` runs in the extract's isolated interpreter.


def _arch_list(value):
    if isinstance(value, str):
        return [value]
    return [a for a in value or [] if isinstance(a, str)]


#: Packer `kernel_source` kinds with no rocKE build: the emitter has nothing to lower for them.
_OTHER_KINDS = {"hip", "hsaco", "kpack", "embedded_source"}


def _spec_json(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def _non_string_fields(ks) -> bool:
    """Whether a launch's `source`, `builder` or `target` is not a string: it cannot be built,
    and as part of a family key it would fail the child (an unhashable list or object).
    """
    return not all(
        isinstance(ks.get(k, "native"), str) for k in ("source", "builder", "target")
    )


_NON_STRING = "has a `source`, `builder` or `target` that is not a string"


def _ukd_variants(d, arches, kdp, origin):
    """(variants, reason) of one descriptor dict, as hkp_pack enumerates it. A single
    `kernel_source` is one variant named by the descriptor (none when it lacks source and
    builder, e.g. a hip kernel). A `sources` UKD groups its launches by (source, builder,
    spec, target); each group is one variant, named by the descriptor when there is one group,
    else `<name>@<launch>[+<launch>...]` with the launch names sorted. A `sources` UKD whose
    launches all share one other packer kind (hip, hsaco, kpack, embedded_source) has no rocKE
    build and gives none, as a single hip launch does. The name is the descriptor's `name`, else
    its `id`. REASON, set when the descriptor cannot be read (a field that is not of its JSON
    type included), is for `not_inspected` and the variants are then empty.
    """
    has_ks, has_sources = "kernel_source" in d, "sources" in d
    name = d.get("name") or d.get("id") or "?"
    if has_ks and has_sources:
        return [], f"{name}: carries both `kernel_source` and `sources`"
    if has_sources:
        entries = d["sources"]
        if not isinstance(entries, dict) or not entries:
            return [], f"{name}: `sources` is not a non-empty object"
        kinds = {
            (
                ks.get("kind")
                if isinstance(ks, dict) and isinstance(ks.get("kind"), str)
                else None
            )
            for ks in entries.values()
        }
        if len(kinds) == 1 and kinds & _OTHER_KINDS:
            return [], None
        groups = {}
        for launch, ks in entries.items():
            if not isinstance(ks, dict):
                return [], f"{name}: sources entry `{launch}` is not an object"
            if ks.get("kind") != "rocke":
                return (
                    [],
                    f"{name}: sources entry `{launch}` has kind {ks.get('kind')!r}, not rocke",
                )
            if not (ks.get("source") and ks.get("builder")):
                return (
                    [],
                    f"{name}: sources entry `{launch}` lacks `source` or `builder`",
                )
            if _non_string_fields(ks):
                return [], f"{name}: sources entry `{launch}` {_NON_STRING}"
            key = (
                ks["source"],
                ks["builder"],
                _spec_json(ks.get("spec") or {}),
                ks.get("target", "native"),
            )
            groups.setdefault(key, (ks, []))[1].append(launch)
        named = [
            (ks, sorted(launches) if len(groups) > 1 else None)
            for ks, launches in groups.values()
        ]
    else:
        ks = d.get("kernel_source")
        if ks is None:
            return [], None
        if not isinstance(ks, dict):
            return [], f"{name}: `kernel_source` is not an object"
        if not (ks.get("source") and ks.get("builder")):
            return [], None
        if _non_string_fields(ks):
            return [], f"{name}: `kernel_source` {_NON_STRING}"
        named = [(ks, None)]
    if not isinstance(name, str):
        return [], f"{name}: the descriptor `name` is not a string"
    return [
        _variant(
            f"{name}@{'+'.join(launches)}" if launches else name,
            arches,
            ks,
            kdp,
            origin,
        )
        for ks, launches in named
    ], None


def _variant(name, arches, ks, kdp, origin):
    return dict(
        name=name,
        arches=arches,
        source=ks["source"],
        builder=ks["builder"],
        spec=ks.get("spec") or {},
        target=ks.get("target", "native"),
        kdp=kdp,
        origin=origin,
    )


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
                # A descriptor's own `arch` narrows the KDP-level one.
                arches = _arch_list(d.get("arch")) or _arch_list(doc.get("arch"))
                vs, reason = _ukd_variants(d, arches, rel, "kdp-in-git")
                if reason:
                    skipped.append({"kdp": rel, "reason": reason})
                elif vs and not arches:
                    skipped.append({"kdp": rel, "reason": f"{d['name']}: no `arch`"})
                else:
                    variants.extend(vs)
        if "artifacts.dvc" in filenames and not kdps:
            dvc_dirs.append(here)
    for d in dvc_dirs:
        got, why = _regenerate(base, d, skipped)
        if why:
            kdp = (d / f"{d.name}.kdp.json").relative_to(base).as_posix()
            skipped.append({"kdp": kdp, "reason": why})
        else:
            variants.extend(got)
    return variants, skipped


def _regenerate(base, dvc_dir, unreadable):
    """Variants of a DVC-held KDP from the registry enumerator, or (None, reason). A kernel
    the enumerator emitted that cannot be read is appended to UNREADABLE (not_inspected).
    """
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
        r = run_tree(
            [sys.executable, str(enum), "--manifest", str(manifest), "--out", str(out)],
            int(os.environ.get("IMPACT_EMIT_ENUM_TIMEOUT", "600")),
            cwd=str(base),
            own_group=False,
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
    got, bad = [], []
    for k in pack.get("kernels") or []:
        arches = _arch_list(k.get("arch")) or _arch_list(pack.get("arch"))
        vs, reason = _ukd_variants(k, arches, kdp, "registry_enumerator")
        if reason:
            bad.append({"kdp": kdp, "reason": reason})
        elif vs and not arches:
            return None, f"registry_enumerator.py output has no arch for {k['name']}"
        got.extend(vs)
    if not got and not bad:
        return None, "registry_enumerator.py produced no rocKE kernels"
    unreadable.extend(bad)
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
        self.rcompile = rcompile
        self.compile_arch = getattr(rc, "rocke_compile_arch", None)
        # A builder may return a KernelModule (a primary plus role-keyed companions) that the
        # packer links into one code object; a revision without the type has no such builder.
        try:
            from rocke.helpers import KernelModule
        except ImportError:
            KernelModule = None
        self.kernel_module = KernelModule
        self.require_arch_sig = getattr(rc, "_require_spec_arch_signature", None)
        self.check_predicate = getattr(rc, "_check_support_predicate", None)
        # The packer's own predicate lookup (alias, derived name, then is_valid*), which the
        # predicate tables reuse so they test the predicate the packer would consult.
        self.resolve_predicate = getattr(rc, "_resolve_support_predicate", None)
        # The packer's lowering backend; a tree without the pin is taken as Python.
        backend = getattr(rc, "_BACKEND", "python")
        if backend != "python":
            self.error = (
                f"the packer lowers with backend {backend!r}; "
                "impact_emit lowers with the Python backend only"
            )


def _comgr_inputs(rcompile, kd, arch):
    """What the packer's COMGR step takes besides the IR text: the ISA name and, per kernel,
    the codegen options and whether it compiles in a fresh process. None when this revision
    has no `_comgr_options_for_kernel`."""
    options_for = getattr(rcompile, "_comgr_options_for_kernel", None)
    if not callable(options_for):
        return None
    kernels = getattr(kd, "kernels", None)  # a KernelModule links several kernels
    kernels = list(kernels) if isinstance(kernels, (list, tuple)) else [kd]
    attr = getattr(rcompile, "COMGR_FRESH_PROCESS_ATTR", None)
    resolve = getattr(rcompile, "_resolve_compile_target", None)
    return {
        "isa": (
            resolve(arch, "amdgcn-amd-amdhsa--gfx950")[1] if callable(resolve) else None
        ),
        "options": [list(options_for(k)) for k in kernels],
        "fresh_process": [
            bool(attr and (getattr(k, "attrs", None) or {}).get(attr)) for k in kernels
        ],
    }


def _compile_arch(h, shard, target):
    if callable(h.compile_arch):
        return h.compile_arch(shard, target)
    if target != "native":
        raise RuntimeError(
            f"rocke_compile_arch missing; cannot resolve target {target!r}"
        )
    return shard


def _record_inner_builders(module, shipped):
    """Wrap MODULE's other `build_*` functions so each call records (name, spec) while the
    shipped builder runs; returns (calls, restore). A KernelModule builder reaches its kernels'
    builders through the module's globals, so this sees the spec of every companion launch.
    """
    calls, saved = [], {}
    for name, obj in list(vars(module).items()):
        if name.startswith("build_") and name != shipped and inspect.isfunction(obj):

            def recorder(*args, _name=name, _fn=obj, **kwargs):
                if args:
                    calls.append((_name, args[0]))
                return _fn(*args, **kwargs)

            saved[name] = obj
            setattr(module, name, recorder)

    def restore():
        for name, obj in saved.items():
            setattr(module, name, obj)

    return calls, restore


def _build_one(h, v, arch, capture=None):
    """Build and lower one variant. CAPTURE, a dict, receives the module, the spec object
    and the (inner builder, spec) calls the build made, for the predicate tables."""
    from importlib import import_module

    module = import_module(h.module_from_source(v["source"]))
    fn = getattr(module, v["builder"])
    cls = h.resolve_spec_class(module, fn)
    if callable(h.require_arch_sig):
        h.require_arch_sig(fn, v["builder"])
    obj = h.build_spec(cls, v["spec"])
    if callable(h.check_predicate):
        h.check_predicate(module, v["builder"], obj, arch)
    calls, restore = _record_inner_builders(module, v["builder"])
    try:
        kd = fn(obj, arch=arch)
    finally:
        restore()
    if capture is not None:
        capture.update(module=module, spec=obj, inner=calls)
    # A KernelModule is lowered kernel by kernel, as the packer's `compile_module` does; each
    # companion is its own launch with its own kernarg list.
    is_module = h.kernel_module is not None and isinstance(kd, h.kernel_module)
    launches = [("", kd.primary), *kd.companions.items()] if is_module else [("", kd)]
    texts = [h.lower(k, arch=arch, backend="python", spec=None) for _, k in launches]
    digest = hashlib.sha256()
    for (role, _), text in zip(launches, texts):
        digest.update(f"{role}\0{len(text)}\0".encode("utf-8") + text.encode("utf-8"))
    built = {
        "params": parse_kernargs(texts[0]),
        "ir": (
            digest.hexdigest()
            if is_module
            else hashlib.sha256(texts[0].encode()).hexdigest()
        ),
        "comgr": _comgr_inputs(h.rcompile, kd, arch),
    }
    if is_module:
        built["companions"] = {
            role: parse_kernargs(text)
            for (role, _), text in zip(launches[1:], texts[1:])
        }
    return built


# ------------------------------------------------------------ predicate tables (child)

#: Seeds per table, and how many distinct shipped values of one int field seed its domain.
_TABLE_SEEDS = 4
_TABLE_BASE_INTS = 6
#: Seconds a family's probes may run past their budget before the family is abandoned.
_TABLE_GRACE = 5


def _spec_key(spec):
    if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        return repr(dataclasses.asdict(spec))
    return repr(id(spec))  # not a dataclass: the table says so, it needs no key


def _domains(specs):
    """{field: (kind, shipped ints)} for the int and bool fields of the shipped specs. An int
    field may also be None in some specs (an Optional override)."""
    out = {}
    for f in dataclasses.fields(specs[0]):
        vals = [getattr(s, f.name, None) for s in specs]
        ints = sorted({x for x in vals if type(x) is int})
        if ints and all(x is None or type(x) is int for x in vals):
            out[f.name] = ("int", ints)
        elif all(type(x) is bool for x in vals):
            out[f.name] = ("bool", [])
    return out


def _candidates(kind, ints):
    """Values to try for one field: the shipped ones, 0, -1, and around each shipped positive
    value its negation, double, half and one and a half times."""
    if kind == "bool":
        return [False, True]
    pos = [x for x in ints if x > 0]
    if len(pos) > _TABLE_BASE_INTS:
        pos = sorted(set(pos[:4] + pos[-2:]))
    out = set(ints if len(ints) <= _TABLE_BASE_INTS else pos) | {0, -1}
    for x in pos:
        out |= {-x, 2 * x, x // 2, x + x // 2}
    return sorted(out)


def _seeds(specs, domains):
    """Up to _TABLE_SEEDS shipped specs, picked greedily to cover the most (field, value) pairs."""
    chosen, seen, pool = [], set(), list(specs)
    while pool and len(chosen) < _TABLE_SEEDS:
        best = max(
            pool, key=lambda s: len({(f, repr(getattr(s, f))) for f in domains} - seen)
        )
        gain = {(f, repr(getattr(best, f))) for f in domains} - seen
        if chosen and not gain:
            break
        chosen.append(best)
        seen |= gain
        pool.remove(best)
    return chosen


def _verdict(pred, spec, kw):
    res = pred(spec, **kw)
    return bool(res[0]) if isinstance(res, tuple) else bool(res)


def _where(exc):
    """`file:line` of the frame that raised EXC."""
    frames = traceback.extract_tb(exc.__traceback__)
    return f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else "?"


class _Table:
    """One builder's table: the record that goes into the emit file, and what the
    same-module sign check needs (the predicate, its first seed, the field domains)."""

    def __init__(self, name):
        self.name = name
        self.record = {
            "predicate": "",
            "rows": 0,
            "skipped": "",
            "truncated": False,
            "findings": [],
        }
        self.found = {}
        self.pred = self.kw = self.seeds = self.domains = None

    def note(self, check, field, value, seed, detail):
        f = self.found.setdefault(
            (check, field),
            {
                "check": check,
                "field": field,
                "values": [],
                "seed": ", ".join(f"{k}={getattr(seed, k)!r}" for k in self.domains)[
                    :300
                ],
                "detail": detail,
            },
        )
        if value not in f["values"] and len(f["values"]) < 8:
            f["values"].append(value)

    def accepts(self, seed, field, value):
        """The predicate's verdict on SEED with FIELD set to VALUE; a spec that cannot be
        constructed, or a predicate that raises, is no acceptance."""
        try:
            return _verdict(
                self.pred, dataclasses.replace(seed, **{field: value}), self.kw
            )
        except Exception:
            return False

    def finish(self):
        self.record["findings"] = sorted(
            self.found.values(), key=lambda f: (f["check"], f["field"])
        )
        for f in self.record["findings"]:
            f["values"].sort(key=lambda v: (type(v) is bool, v))
        return self.record


def _predicate_table(h, module, name, catalog, arch, deadline):
    """The table of one builder: its shipped specs varied one field at a time, each variation
    put to the predicate the packer would consult. A spec the predicate accepts is built; a
    builder that raises on it is a `builder` finding."""
    t = _Table(name)
    rec = t.record
    if not callable(h.resolve_predicate):
        rec["skipped"] = "hkp_pack has no _resolve_support_predicate at this revision"
        return t
    try:
        pname, pred = h.resolve_predicate(module, name)
    except Exception as e:
        rec["skipped"] = f"predicate lookup raised {type(e).__name__}: {str(e)[:200]}"
        return t
    if pred is None:
        rec["skipped"] = "no support predicate takes its spec"
        return t
    rec["predicate"] = pname
    specs = list({_spec_key(s): s for s in catalog}.values()) if catalog else []
    if not specs or not all(
        dataclasses.is_dataclass(s) and not isinstance(s, type) for s in specs
    ):
        rec["skipped"] = "no shipped spec, or a spec that is not a dataclass instance"
        return t
    t.domains = _domains(specs)
    t.seeds = _seeds(specs, t.domains)
    t.pred = pred
    t.kw = {"arch": arch} if "arch" in inspect.signature(pred).parameters else {}
    fn = getattr(module, name)
    bkw = {"arch": arch} if "arch" in inspect.signature(fn).parameters else {}
    for seed in t.seeds:
        for field, (kind, ints) in t.domains.items():
            for value in _candidates(kind, ints):
                current = getattr(seed, field)
                if value == current and type(value) is type(current):
                    continue
                if time.time() >= deadline:
                    rec["truncated"] = True
                    break
                try:
                    spec = dataclasses.replace(seed, **{field: value})
                    accepted = _verdict(pred, spec, t.kw)
                except Exception:
                    continue  # refused at construction, or the predicate has no verdict
                rec["rows"] += 1
                if not accepted:
                    continue
                try:
                    fn(spec, **bkw)
                except Exception as e:
                    t.note("builder", field, value, seed,
                           f"{pname} accepts it; {name} raises {type(e).__name__}: {str(e)[:200]} at {_where(e)}")  # fmt: skip
    return t


def _same_module_signs(tables, deadline):
    """`sign` findings between the tables of one family (a KernelModule's builders): for a
    field two of them share, a value outside the sign range of every shipped value (<= 0 where
    all are >= 1, < 0 where all are >= 0) that one predicate accepts and the other refuses.
    """
    live = [t for t in tables if t.pred is not None and t.seeds]
    for a in live:
        for b in live:
            if a is b:
                continue
            for field, (kind, ints_a) in a.domains.items():
                kind_b, ints_b = b.domains.get(field, (None, []))
                if kind != "int" or kind_b != "int":
                    continue
                low = min(ints_a[0], ints_b[0])
                if low < 0:
                    continue
                values = sorted(
                    v
                    for v in set(_candidates("int", ints_a))
                    | set(_candidates("int", ints_b))
                    if v < 0 or (low >= 1 and v == 0)
                )
                for v in values:
                    if time.time() >= deadline:
                        a.record["truncated"] = True
                        return
                    if a.accepts(a.seeds[0], field, v) and not b.accepts(
                        b.seeds[0], field, v
                    ):
                        a.note("sign", field, v, a.seeds[0],
                               f"{a.record['predicate']} accepts {field}={v}; {b.record['predicate']}, the predicate of "
                               f"{b.name} in the same module, refuses it; every shipped value is >= {low}")  # fmt: skip


def _table_names(fam, got):
    return [fam["builder"], *sorted({n for n, _ in got["inner"]})]


def _table_record(fam, t):
    return {
        "arch": fam["arch"],
        "source": fam["source"],
        "builder": fam["builder"],
        "inner": "" if t.name == fam["builder"] else t.name,
        **t.finish(),
    }


def _family_tables(h, fam, got, seconds):
    """The table records of one family: its shipped builder's and each inner builder's."""
    module = got["module"]
    inner = {}
    for name, spec in got["inner"]:
        inner.setdefault(name, []).append(spec)
    deadline = time.time() + seconds
    tables = [
        _predicate_table(h, module, name, catalog, fam["arch"], deadline)
        for name, catalog in [(fam["builder"], got["specs"]), *sorted(inner.items())]
    ]
    _same_module_signs(tables, deadline)
    return [_table_record(fam, t) for t in tables]


def _tables(h, fams, captures):
    """(records, stuck): one table per family's shipped builder and per inner builder its
    builds called. The probes run in a worker thread; a family whose probes run past its
    IMPACT_EMIT_TABLE_SECONDS budget plus _TABLE_GRACE (a builder that never returns on a spec
    the predicate accepts) is abandoned with every table not yet finished, and STUCK is True.
    """
    seconds = float(os.environ.get("IMPACT_EMIT_TABLE_SECONDS", "60"))
    plan = [(fam, captures[key]) for key, fam in fams.items() if captures.get(key)]
    done, state = [], {"at": (0, time.time()), "error": None}

    def work():
        try:
            for i, (fam, got) in enumerate(plan):
                state["at"] = (i, time.time())
                done.append(_family_tables(h, fam, got, seconds))
        except BaseException as e:  # re-raised in the caller's thread
            state["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.1)
        if time.time() - state["at"][1] > seconds + _TABLE_GRACE:
            break
    stuck = worker.is_alive()
    if state["error"] is not None:
        raise state["error"]
    finished = list(done)
    out = [r for recs in finished for r in recs]
    if stuck:
        for fam, got in plan[len(finished) :]:
            for name in _table_names(fam, got):
                t = _Table(name)
                t.record["skipped"] = "probe timed out"
                out.append(_table_record(fam, t))
    return out, stuck


def child_main(root, out):
    # The script's own directory must not shadow rocKE modules.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    t0 = time.time()
    variants, skipped = _discover(root)
    h = _Helpers()
    fams, specs, captures = {}, {}, {}
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
                # same variant reached through another member arch, unless the specs differ
                if specs[key, v["name"]] != v["spec"]:
                    fam["built"].pop(v["name"], None)
                    fam["failures"][
                        v["name"]
                    ] = f"{v['name']}: two variants share this name with different specs"
                continue
            fam["names"].append(v["name"])
            specs[key, v["name"]] = v["spec"]
            if err:
                fam["failures"][v["name"]] = err
                continue
            try:
                capture = {}
                fam["built"][v["name"]] = _build_one(h, v, carch, capture)
            except Exception as e:
                fam["failures"][v["name"]] = f"{type(e).__name__}: {str(e)[:300]}"
                continue
            got = captures.setdefault(
                key, {"module": capture["module"], "specs": [], "inner": []}
            )
            got["specs"].append(capture["spec"])
            got["inner"] += capture["inner"]
    tables, stuck = _tables(h, fams, captures)
    doc = {
        "families": list(fams.values()),
        "tables": tables,
        "not_inspected": skipped,
        "seconds": round(time.time() - t0, 1),
    }
    Path(out).write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    if stuck:
        os._exit(0)  # a probe thread still spins in a builder; the document is written
    return 0


# ------------------------------------------------------------------- parent side


#: System, temp and locale variables, plus where Python finds user-site packages (`pip
#: install --user` of Jinja2 or PyYAML): APPDATA on Windows, PYTHONUSERBASE anywhere.
_ENV_KEEP = frozenset(
    "PATH PATHEXT SYSTEMROOT SYSTEMDRIVE WINDIR COMSPEC TEMP TMP TMPDIR HOME "
    "USERPROFILE APPDATA LOCALAPPDATA PYTHONUSERBASE LANG TZ".split()
)
_ENV_KEEP_PREFIXES = ("LC_", "IMPACT_EMIT_")


def child_env(root, flavor):
    """The child's environment: an allow-list of the parent's, then the pinned variables."""
    env = {}
    for k, v in os.environ.items():
        name = k.upper() if os.name == "nt" else k
        if name in _ENV_KEEP or name.startswith(_ENV_KEEP_PREFIXES):
            env[k] = v
    base = Path(root)
    env["PYTHONPATH"] = os.pathsep.join(
        str(base / p) for p in (ROCKE_PLATFORM, ROCKE_LIBRARY, PACKAGING)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["ROCKE_LLVM_FLAVOR"] = flavor
    env["ROCKE_PROVIDER_ROOT"] = str(base / PROVIDER)
    return env


def _kill_tree(proc, own_group=True):
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
            )
        except OSError:
            pass
    elif own_group:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()  # the leader, in case the tree kill did not reach it
    except OSError:
        pass


def run_tree(cmd, timeout, *, own_group=True, **kw):
    """subprocess.run with captured text output that, on timeout, kills the child's whole
    process tree and raises TimeoutExpired without waiting for pipes a grandchild holds.
    OWN_GROUP (default) makes the child lead a new process group (POSIX session) that the
    kill takes as a whole; with False the child stays in the caller's group, so a caller
    that is itself killed as a group takes the child with it, and only the child is killed
    here (plus its tree on Windows).
    """
    if os.name == "nt":
        if own_group:
            kw["creationflags"] = kw.get("creationflags", 0) | (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )
    elif own_group:
        kw["start_new_session"] = True
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kw,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc, own_group)
        try:
            proc.wait(timeout=10)
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # give up on pipes some survivor still holds
        raise
    except BaseException:
        _kill_tree(proc, own_group)
        proc.wait()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_side(repo, rev, tree, flavor, timeout, label):
    tmp = tempfile.mkdtemp(prefix="impact_emit_")
    try:
        extract(repo, tree, tmp, rev)
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
            r = run_tree(cmd, timeout, cwd=tmp, env=child_env(tmp, flavor))
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


def _launch_params(built):
    """The kernarg lists of one built variant: the primary's, then each companion's by role."""
    return built["params"], built.get("companions")


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
            n
            for n in common
            if _launch_params(built_b[n]) != _launch_params(built_h[n])
        ]
        ir_changed = [n for n in common if built_b[n]["ir"] != built_h[n]["ir"]]
        # A side that cannot tell its COMGR inputs (None) differs from one that can.
        comgr_changed = [
            n for n in common if built_b[n].get("comgr") != built_h[n].get("comgr")
        ]
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
                "comgr_changed": comgr_changed,
                "params": {
                    "base": built_b[shown]["params"] if shown in built_b else None,
                    "head": built_h[shown]["params"] if shown in built_h else None,
                },
                # role -> kernarg list of each companion launch of a KernelModule builder
                "companions": {
                    "base": (
                        built_b[shown].get("companions") if shown in built_b else None
                    ),
                    "head": (
                        built_h[shown].get("companions") if shown in built_h else None
                    ),
                },
                "failures": {
                    "base": dict(fb["failures"]) if fb else {},
                    "head": dict(fh["failures"]) if fh else {},
                },
            }
        )
    return out


def compare_tables(base_side, head_side):
    """The head's predicate tables. `at_base` on a table says the base ran the same builder's
    table (it was shipped and tabled there); on a finding, that the base's table has a finding of
    the same check on the same field, so it predates the change. A builder the base did not
    table can carry an old predicate defect that ships for the first time."""

    def key(t):
        return (t["arch"], t["source"], t["builder"], t["inner"])

    seen = {
        key(t): {(f["check"], f["field"]) for f in t["findings"]}
        for t in base_side.get("tables", [])
        if not t["skipped"]
    }
    return [
        {
            **t,
            "at_base": key(t) in seen,
            "findings": [
                {**f, "at_base": (f["check"], f["field"]) in seen.get(key(t), set())}
                for f in t["findings"]
            ],
        }
        for t in head_side.get("tables", [])
    ]


def build_doc(
    a, flavor, base_tree, head_tree, base_side, head_side, families, tables, seconds
):
    """The emit document (schema 6) of one run, from the two sides' child documents and
    their compared families and tables."""
    return {
        "schema": 6,
        "tool": "impact_emit",
        "flavor": flavor,
        "base": a.base,
        "head": a.head,
        "base_tree": base_tree,
        "head_tree": head_tree,
        "families": families,
        "predicate_tables": tables,
        "not_inspected": [
            {"kdp": n["kdp"], "side": side, "reason": n["reason"]}
            for side, doc in (("base", base_side), ("head", head_side))
            for n in doc["not_inspected"]
        ],
        "seconds": seconds,
    }


def cmd_run(a):
    repo = Path(a.repo)
    if not repo.is_dir():
        raise Fatal(f"--repo {a.repo} is not a directory")
    # Resolve each side once; the same ids are archived and recorded.
    base_tree = resolve_tree(repo, a.base)
    head_tree = resolve_tree(repo, a.head)
    flavor = a.flavor or shared_default_flavor(repo, base_tree, head_tree)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:  # the sides are independent
        fb = pool.submit(run_side, repo, a.base, base_tree, flavor, a.timeout, "base")
        fh = pool.submit(run_side, repo, a.head, head_tree, flavor, a.timeout, "head")
        base_side, head_side = fb.result(), fh.result()
    families = compare(base_side, head_side)
    tables = compare_tables(base_side, head_side)
    seconds = round(time.time() - t0, 1)
    doc = build_doc(
        a, flavor, base_tree, head_tree, base_side, head_side, families, tables, seconds
    )
    not_inspected = doc["not_inspected"]
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
            f"ir_changed {len(f['ir_changed'])} comgr_changed {len(f['comgr_changed'])} "
            f"failures {fails} ({f['spec_origin']})"
        )
    for n in not_inspected:
        print(f"not inspected [{n['side']}] {n['kdp']}: {n['reason']}")
    for t in tables:
        who = t["builder"] + (f"[{t['inner']}]" if t["inner"] else "")
        state = t["skipped"] or (
            f"{t['rows']} rows, {len(t['findings'])} findings"
            + (", truncated" if t["truncated"] else "")
        )
        print(f"table {t['arch']} {t['source']}:{who}: {state}")
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
    c = sub.add_parser("child")  # internal: runs inside the isolated interpreter
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
