# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Engine import probe, in every review's checks. rocKE's Python backend must
run where only the platform's C++ engine module is built: a CI lane or a user without the library's engine
(`rocke_library_engine`) still chooses backend="python". A move of a lowering into a module that imports
rocke_library_engine before it chooses a backend makes backend="python" fail there, while at the merge base the same
functions imported rocke_engine first. The probe shows both sides.

**The modules**: each Python module the PR adds, changes or moves under library/ and platform/python/ (test files and
tests/ directories aside; a script, whose top level calls main(), is listed and not imported), and the merge base's
side of each: a changed module at its path, a moved one at its old path, a deleted one. At most CAP a side.

**The probe** (tools/import_probe_child.py): the head's and the merge base's rocKE trees, each written to a scratch
directory (`git archive`) and imported in a child Python (PYTHONPATH platform/python then library, no bytecode written,
HIP hidden, none of the checks' ROCKE_* environment) where `rocke_engine` imports as a stand-in, the platform engine
built, and every other rocKE engine module (`rocke_*_engine`) is not built, whatever the environment has. Each
module is imported; then its cheapest public function with a `backend` parameter (fewest required arguments, then the
first defined) is called with backend="python", arch="gfx950" and stand-in arguments (an object that answers every
attribute, call and comparison), PER_CALL_S seconds at most. At the base, each function of a name the head called is
called too, so a function the PR moves is compared with its old home. The stand-ins stop most calls soon; what counts
is whether an engine module is imported on the way (and where), and whether the call raises ImportError for an engine
that is not built.

**Leads** (the section feeds no gate): a module that imports at the base and not at the head; an entry point whose
backend="python" call needs an unbuilt engine at the head and not at the base, or that is new and does. Each says
where both sides import their engine.

Usage: python3 -I import_probe.py --repo R --head SHA --out DIR [--base REV] [--rocke-path K] [--python PY]
       [--built rocke_engine] [--budget 240]
--python: the interpreter the modules are imported in (the checks' venv, which has numpy and the rest; 3.11 or later).
Prints the section, then one last line: "IMPORTS pass|fail|skip <detail>".
       python3 -I import_probe.py --repo R --head SHA [--base REV] [--rocke-path K] --fetch
Fetches the blobs of the clone the probe reads, and runs nothing else (rkmon_lib/prtree.py, fetch_main).
"""
import argparse
import ast
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import gitops, nogpu, overlaps, prtree  # noqa: E402

BUDGET_S = 240
PER_CALL_S = 5
CAP = 60  # modules a side
SHOWN = 12  # lines per list
CHILD = pathlib.Path(__file__).resolve().parent / "import_probe_child.py"
MAINS = {
    "main",
    "_main",
    "cli",
    "run",
    "parse_args",
    "exit",
}  # a top-level call of one: the file is a script


# --- the modules ------------------------------------------------------------------------------------------------------


def side_modules(cs: list[dict], K: str) -> tuple[list[dict], list[dict]]:
    """The head's and the merge base's modules of the PR's changes (prtree.changes): {"module", "path", "change"}."""
    head, base = [], []
    for c in cs:
        for side, path in ((head, c["new"]), (base, c["old"])):
            if path is None or overlaps.TEST_PATH.search(path):
                continue
            mod = prtree.module_name(path, K)
            if mod:
                side.append({"module": mod, "path": path, "change": c})
    return head, base


def is_script(text: str) -> bool:
    """A file whose top level calls main() (or parse_args, exit): importing it would run it."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return False  # imported: the import reports the error
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if (
                f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            ) in MAINS:
                return True
    return False


def child(
    python: str,
    spec: dict,
    tree: pathlib.Path,
    work: pathlib.Path,
    tag: str,
    timeout: float,
    log: pathlib.Path,
) -> tuple[dict[str, dict], str | None]:
    """Run the child on one tree: {module: record}, and why it stopped early (None when it finished)."""
    spec_f, res_f = work / f"{tag}_spec.json", work / f"{tag}_results.jsonl"
    spec_f.write_text(json.dumps(spec))
    cwd, tmp = work / f"{tag}_cwd", work / f"{tag}_tmp"
    cwd.mkdir()
    tmp.mkdir()
    env = nogpu.env(
        {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("ROCKE_")
            and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
        }
    )
    env.update(
        PYTHONPATH=os.pathsep.join(str(tree / r) for r in prtree.PY_ROOTS),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(tmp),
        MPLBACKEND="Agg",
    )
    why = None
    with open(log, "w") as fh:
        p = subprocess.Popen(
            [python, "-B", "-P", "-s", str(CHILD), str(spec_f), str(res_f)],
            cwd=cwd,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=max(5.0, timeout))
            if rc:
                why = f"the probe process ended with rc={rc} (log {log.name})"
        except subprocess.TimeoutExpired:
            why = f"the time budget ran out ({timeout:.0f} s left for the {tag})"
        finally:
            try:
                os.killpg(
                    p.pid, signal.SIGKILL
                )  # with anything the imported code left running
            except OSError:
                pass
            p.wait()
    got = {}
    if res_f.exists():
        for line in res_f.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            got[rec["module"]] = rec
    return got, why


# --- the report -------------------------------------------------------------------------------------------------------


def frame(frames: list, name: str | None = None) -> list | None:
    """The frame to show: the entry point's own (its line that led there), else the innermost in the tree."""
    own = [f for f in frames or [] if name and f[2] == name]
    return own[-1] if own else (frames[-1] if frames else None)


def _at(f: list | None) -> str:
    return f"{f[0]}:{f[1]}" if f else "?"


def outcome(e: dict) -> str:
    """One entry point's call, in words: what it imported on the way and how it ended."""
    eng = [
        f"imports {x['module']}{'' if x['built'] else ' (not built)'} at {_at(frame(x['site'], e['name']))}"
        for x in e.get("engines") or []
    ]
    if e.get("ok"):
        end = "returns"
    elif e.get("missing"):
        end = f"raises {e['type']}: {e['missing']} is not built"
    elif e.get("stand_in"):
        end = f"calls into the built engine ({e['msg'][:90]})"
    elif e.get("type") == "Timeout":
        end = e["msg"]
    else:
        end = f"stops on the stand-in arguments ({e['type']}: {e['msg'][:70]})"
    return ", then ".join(eng + [end]) if eng else end


def compare(heads: list[dict], H: dict, bases: list[dict], B: dict) -> list[dict]:
    """The leads: head modules that do not import where the base's did, and entry points that need an unbuilt engine
    at the head and not at the base (or are new)."""
    base_of = {b["change"]["old"]: b for b in bases}
    leads = []
    for h in heads:
        hr = H.get(h["module"])
        if not hr:
            continue
        b = base_of.get(h["change"]["old"]) if h["change"]["old"] else None
        br = B.get(b["module"]) if b else None
        imp = hr["import"]
        if not imp["ok"]:
            if br and br["import"]["ok"]:
                leads.append(
                    {
                        "kind": "import",
                        "module": h["module"],
                        "path": h["path"],
                        "head": imp,
                        "base": b["module"],
                    }
                )
            elif not b and imp.get("missing"):
                leads.append(
                    {
                        "kind": "import",
                        "module": h["module"],
                        "path": h["path"],
                        "head": imp,
                        "base": None,
                    }
                )
            continue
        for e in hr["entries"]:
            if not e.get("missing"):
                continue
            be, bm = None, None
            for cand in ([br] if br else []) + [r for r in B.values() if r is not br]:
                be = next((x for x in cand["entries"] if x["name"] == e["name"]), None)
                if be:
                    bm = cand["module"]
                    break
            if be and be.get("missing"):
                continue  # fails at the base too: listed, not a lead
            leads.append(
                {
                    "kind": "entry",
                    "module": h["module"],
                    "path": h["path"],
                    "entry": e,
                    "names": len(hr["names"]),
                    "base_module": bm,
                    "base_entry": be,
                }
            )
    return leads


def describe(lead: dict, K: str) -> str:
    if lead["kind"] == "import":
        imp = lead["head"]
        why = f"{imp['type']}: {imp['missing'] + ' is not built' if imp.get('missing') else imp['msg'][:120]}"
        where = _at(frame(imp.get("frames")))
        base = (
            f"; it imports at the base ({lead['base']})"
            if lead["base"]
            else " (a new module)"
        )
        return f"{lead['module']} does not import at the head: {why} ({where}){base}"
    e, be = lead["entry"], lead.get("base_entry")
    path = (
        lead["path"][len(K) + 1 :] if lead["path"].startswith(K + "/") else lead["path"]
    )
    s = (
        f"{lead['module']}.{e['name']} ({path}:{e['line']}, the cheapest of {lead['names']} with a backend "
        f'parameter): backend="python" {outcome(e)}'
    )
    if not e.get("engines"):
        s += f" (at {_at(frame(e.get('frames'), e['name']))})"
    if be:
        s += f"; at the merge base {lead['base_module']}.{e['name']} {outcome(be)}"
    else:
        s += "; the merge base has no function of that name among the PR's modules"
    return s


def _base_name(f: list | None) -> str:
    return f"{f[0].rsplit('/', 1)[-1]}:{f[1]}" if f else "?"


def short(lead: dict) -> str:
    if lead["kind"] == "import":
        return f"{lead['module']} does not import at the head"
    e, be = lead["entry"], lead.get("base_entry")
    built = [x for x in (be or {}).get("engines") or [] if x["built"]]
    base = (
        f"; at the base it imports {built[0]['module']} first ({_base_name(frame(built[0]['site'], e['name']))})"
        if built
        else "" if be else " (new)"
    )
    return (
        f"{lead['module'].rsplit('.', 1)[-1]}.{e['name']} needs {e['missing']} with backend=\"python\" "
        f"({_base_name(frame(e.get('frames'), e['name']))}){base}"
    )


def verdict(res: dict) -> tuple[str, str]:
    if res.get("error"):
        return "skip", f"NOT RUN: {res['error'][:180]}"
    if not res["heads"] and not res["bases"]:
        return (
            "skip",
            "nothing to probe: the PR changes no Python module under library/ or platform/python/",
        )
    stop = "; ".join(
        f"{t}: {res[t + '_stop']}" for t in ("head", "base") if res.get(t + "_stop")
    )
    if res["leads"]:
        n = len(res["leads"])
        return "fail", f"{n} lead{'s' if n > 1 else ''}: {short(res['leads'][0])}" + (
            f"; +{n - 1} more" if n > 1 else ""
        ) + ("; part NOT RUN" if stop else "")
    if stop:  # a probe cut short never reads clean
        return "skip", f"NOT RUN in part ({stop[:160]})"
    return "pass", (
        f"{res['head_ok']} of {res.get('head_ran', 0)} head modules import, and no backend=\"python\" call "
        f"needs an unbuilt engine where the base's did not"
    )


def render(res: dict, K: str) -> list[str]:
    if res.get("error"):
        return [f"NOT RUN: {res['error']}", " ".join(("IMPORTS",) + verdict(res))]
    if not res["heads"] and not res["bases"]:
        return [" ".join(("IMPORTS",) + verdict(res))]
    new = sum(1 for h in res["heads"] if h["status"] == "A")
    out = [
        f"Python modules the PR adds, changes or moves under library/ and platform/python/ (tests aside): "
        f"{len(res['heads'])} at the head ({new} new), {len(res['bases'])} at the merge base "
        f"{res['mb'][:10]}; imported with {' '.join(res['built'])} standing in as built and no other rocKE engine "
        f"module built, each module's cheapest entry point with a `backend` parameter called with "
        f"backend=\"python\", arch=\"gfx950\" and stand-in arguments ({PER_CALL_S} s each); {res.get('secs', 0)} s"
    ]
    for tag in ("head", "base"):
        if res.get(f"{tag}_stop"):
            out.append(f"NOT RUN for part of the {tag}: {res[f'{tag}_stop']}")
    out += [f"LEAD {describe(x, K)}" for x in res["leads"][:SHOWN]]
    if len(res["leads"]) > SHOWN:
        out.append(f"LEAD +{len(res['leads']) - SHOWN} more (import_probe.json)")
    for tag in ("head", "base"):
        recs = res.get(tag + "_records") or {}
        ok = [r for r in recs.values() if r["import"]["ok"]]
        at_import = [r for r in ok if r["engines"]]
        out.append(
            f"{tag}: {len(ok)} of {len(recs)} modules import"
            + (
                f" ({len(at_import)} import an engine module at import time: "
                f"{', '.join(r['module'] + ' ' + r['engines'][0]['module'] for r in at_import[:3])})"
                if at_import
                else ""
            )
        )
        bad = [r for r in recs.values() if not r["import"]["ok"]]
        for r in bad[:SHOWN]:
            imp = r["import"]
            out.append(
                f"   does not import: {r['module']}: {imp['type']}: "
                f"{imp['missing'] + ' is not built' if imp.get('missing') else imp['msg'][:140]}"
            )
        listed = (
            {
                (x["module"], x["entry"]["name"])
                for x in res["leads"]
                if x["kind"] == "entry"
            }
            if tag == "head"
            else set()
        )
        calls = [
            (r["module"], e)
            for r in ok
            for e in r["entries"]
            if (r["module"], e["name"]) not in listed
        ]
        for mod, e in calls[:SHOWN]:
            out.append(f"   {mod}.{e['name']}: backend=\"python\" {outcome(e)}")
        if len(calls) > SHOWN:
            out.append(f"   +{len(calls) - SHOWN} more calls (import_probe.json)")
    if res.get("scripts"):
        out.append(
            f"scripts, not imported (their top level calls main()): {' '.join(res['scripts'][:SHOWN])}"
        )
    if res.get("capped"):
        out.append(
            f"probed {CAP} modules a side (cap); not probed: {' '.join(res['capped'][:SHOWN])}"
        )
    out.append(" ".join(("IMPORTS",) + verdict(res)))
    return out


# --- the check --------------------------------------------------------------------------------------------------------


def needs(repo: str, head: str, K: str, base: str = prtree.DEVELOP) -> list[str]:
    """The blobs of the clone the probe reads (`--fetch`): both sides of the PR's changes, which git's rename detection
    reads, and the rocKE tree at the head (the merge base's is the same but for those changes).
    """
    mb = prtree.merge_base(repo, base, head)
    return prtree.diff_blobs(repo, mb, head) + prtree.blobs(repo, head, [K])


def run(
    repo: str,
    head: str,
    out: pathlib.Path,
    K: str,
    base: str = prtree.DEVELOP,
    python: str = sys.executable,
    built: tuple[str, ...] = ("rocke_engine",),
    budget: float = BUDGET_S,
) -> dict:
    t0 = time.monotonic()

    def left() -> float:
        return budget - (time.monotonic() - t0)

    res: dict = {
        "head": head,
        "error": None,
        "built": list(built),
        "heads": [],
        "bases": [],
        "leads": [],
        "scripts": [],
        "capped": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    work = None
    try:
        res["mb"] = mb = prtree.merge_base(repo, base, head)
        heads, bases = side_modules(prtree.changes(repo, mb, head, timeout=600), K)
        res["heads"] = [
            {"module": h["module"], "path": h["path"], "status": h["change"]["status"]}
            for h in heads
        ]
        res["bases"] = [
            {"module": b["module"], "path": b["path"], "status": b["change"]["status"]}
            for b in bases
        ]
        if not heads and not bases:
            return res
        work = pathlib.Path(
            tempfile.mkdtemp(
                prefix="import_probe_", dir=os.environ.get("TMPDIR") or None
            )
        ).resolve()
        called: list[str] = []
        records = {}
        for tag, rev, mods in (("head", head, heads), ("base", mb, bases)):
            if not mods:
                records[tag] = {}
                continue
            if left() < 20:
                res[f"{tag}_stop"] = (
                    f"the time budget ({budget:.0f} s) ran out before the {tag}"
                )
                records[tag] = {}
                continue
            root = work / tag
            prtree.extract(repo, rev, [K], root, timeout=max(30.0, left()))
            tree = root / K
            todo = []
            for m in mods:
                f = root / m["path"]
                if f.exists() and is_script(f.read_text(errors="replace")):
                    res["scripts"].append(m["path"][len(K) + 1 :])
                else:
                    todo.append(m)
            res["capped"] += [m["module"] for m in todo[CAP:]]
            spec = {
                "root": str(tree),
                "built": list(built),
                "per_call_s": PER_CALL_S,
                "per_module": 1,
                "also": called if tag == "base" else [],
                "modules": [m["module"] for m in todo[:CAP]],
            }
            got, why = child(
                python, spec, tree, work, tag, left() - 5, out / f"{tag}_probe.log"
            )
            if why:
                missing = [m for m in spec["modules"] if m not in got]
                res[f"{tag}_stop"] = why + (
                    f"; not reached: {' '.join(missing[:6])}" if missing else ""
                )
            records[tag] = got
            res[f"{tag}_ran"] = len(got)
            if tag == "head":
                called = sorted({e["name"] for r in got.values() for e in r["entries"]})
        res["head_records"], res["base_records"] = records.get("head", {}), records.get(
            "base", {}
        )
        res["head_ok"] = sum(
            1 for r in res["head_records"].values() if r["import"]["ok"]
        )
        res["leads"] = compare(heads, res["head_records"], bases, res["base_records"])
    except (
        prtree.NotRun,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        gitops.GitError,
    ) as exc:
        res["error"] = str(exc)[:300]
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
        res["secs"] = round(time.monotonic() - t0, 1)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default=None, help="the probe logs and import_probe.json")
    ap.add_argument("--base", default=prtree.DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter the modules are imported in",
    )
    ap.add_argument(
        "--built",
        default="rocke_engine",
        help="engine modules that stand in as built (comma list)",
    )
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument("--fetch", action="store_true", help=prtree.FETCH_HELP)
    a = ap.parse_args(argv)
    if a.fetch:
        return prtree.fetch_main(
            a.repo, lambda: needs(a.repo, a.head, a.rocke_path, a.base)
        )
    if not a.out:
        ap.error("the following arguments are required: --out")
    out = pathlib.Path(a.out)
    res = run(
        a.repo,
        a.head,
        out,
        a.rocke_path,
        a.base,
        a.python,
        tuple(x for x in a.built.split(",") if x),
        a.budget,
    )
    (out / "import_probe.json").write_text(json.dumps(res, indent=1))
    print("\n".join(render(res, a.rocke_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
