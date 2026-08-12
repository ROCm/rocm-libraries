#!/usr/bin/env python3
"""Issue 5: mutmut survivor records -> per-function triage groups.

Net-new glue for the upcoming mutmut-slice.js orchestrator (this is NOT the
orchestrator). Reimplemented in pure Python (was JS): the reviewer sandbox denies
Node spawning python3 (child_process EPERM), so the enclosing-function AST pass
must run in-process. Python's `ast` runs here directly -- no subprocess.

Input:  survivor records equivalent to SURVIVOR{module, mutant_id, file, line,
        diff?, status?}. `file` is SRC_REL-relative (as mutmut reports it under
        source_paths=["Tensile"], e.g. "Tensile/LibraryIO.py").
Output: groups[] shaped for triage-workflow.js:
        {module, function, source_file, char_dir, test_file, survivors[]}
        plus a separate no_test[] list and conservation stats.

Path contract (settles the Issue-4 caveat): source_file / test_file / char_dir are
ALL SRC_REL-relative. triage-workflow.js reads/writes them as ${WT}/${SRC_REL}/<p>
for host FS access and uses the bare form for the in-container pytest node and the
verify manifest. One source_file value works for BOTH consumers.

Read-only: never edits source, never runs mutmut, never applies mutants.

CLI:    python3 mutmut-results-adapter.py --fixture <records.json> \
            [--src-root projects/hipblaslt/tensilelite] \
            [--default-char-dir <dir-or-comma-list>] [--test-dir-base <dir>]
API:    from mutmut_results_adapter import build_groups  (import by path via importlib)
"""

import ast
import hashlib
import json
import os
import re
import sys

sys.excepthook = sys.__excepthook__  # avoid the distro apport excepthook on errors

DEFAULT_SRC_ROOT = "projects/hipblaslt/tensilelite"
DEFAULT_TEST_DIR_BASE = "Tensile/Tests/unit/characterization/_generated"
MODULE_LEVEL = "<module>"


def enclosing_functions(src_text):
    """Return [(start_line, end_line, qualname)] for every def, decorators included."""
    tree = ast.parse(src_text)
    funcs = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                # FunctionDef.lineno is the 'def' line and excludes decorators, so a
                # survivor on a decorator expression must be pulled into the span too.
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                funcs.append((start, getattr(child, "end_lineno", child.lineno), q))
                visit(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return funcs


def enclosing_for_line(funcs, line):
    best = None
    for s, e, q in funcs:
        if s <= line <= e and (best is None or (e - s) < (best[1] - best[0])):
            best = (s, e, q)
    return best[2] if best else None


def _sanitize(s):
    return re.sub(r"^_+|_+$", "", re.sub(r"[^A-Za-z0-9]+", "_", str(s)))


def test_file_for(test_dir_base, module, fn):
    """Deterministic + collision-free: readable slug + short hash of the exact key."""
    key = "%s::%s" % (module, fn)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    slug = _sanitize("%s_%s" % (module, fn))
    return "%s/test_mut_%s_%s_char.py" % (test_dir_base, slug, h)


def _is_empty_char_dir(v):
    if v is None:
        return True
    if isinstance(v, list):
        return len([x for x in v if x]) == 0
    return str(v).strip() == ""


def pick_char_dir(char_dir_map, module, fn, file, fallback):
    m = char_dir_map or {}
    hit = None
    for cand in (m.get("%s::%s" % (module, fn)), m.get(module), m.get(file)):
        if not _is_empty_char_dir(cand):
            hit = cand
            break
    val = hit if hit is not None else fallback
    if _is_empty_char_dir(val):
        return None
    return [x for x in val if x] if isinstance(val, list) else val


def _is_no_test(status):
    if status is None:
        return False
    s = re.sub(r"[-\s]", "_", str(status).lower())
    return s in ("no_test", "notest", "skipped")


def build_groups(survivors, opts=None):
    """Transform survivor records into triage groups.

    Returns {"groups": [...], "no_test": [...], "stats": {...}}.
    Raises ValueError on a malformed record or an unresolvable char_dir.
    """
    opts = opts or {}
    src_root = opts.get("srcRoot") or DEFAULT_SRC_ROOT
    test_dir_base = opts.get("testDirBase") or DEFAULT_TEST_DIR_BASE
    char_dir_map = opts.get("charDirMap") or {}
    fallback = opts.get("defaultCharDir")

    if not isinstance(survivors, list):
        raise ValueError("survivors must be a list")

    active, no_test = [], []
    for r in survivors:
        if not isinstance(r, dict) or any(
            r.get(k) is None for k in ("module", "mutant_id", "file", "line")
        ):
            raise ValueError(
                "survivor record missing required field {module,mutant_id,file,line}: %r"
                % (r,)
            )
        try:
            int(r["line"])
        except (TypeError, ValueError):
            raise ValueError("survivor record has a non-integer line: %r" % (r,))
        if _is_no_test(r.get("status")):
            no_test.append(
                {
                    "module": r["module"],
                    "mutant_id": r["mutant_id"],
                    "file": r["file"],
                    "line": r["line"],
                }
            )
        else:
            active.append(r)

    funcs_by_file = {}
    groups_by_key = {}
    groups = []
    for r in active:
        f = r["file"]
        if f not in funcs_by_file:
            path = os.path.join(src_root, f)
            try:
                with open(path) as fh:
                    funcs_by_file[f] = enclosing_functions(fh.read())
            except Exception:
                funcs_by_file[f] = (
                    None  # missing/unparseable -> module-level (never crash/hang)
                )
        funcs = funcs_by_file[f]
        fn = (
            enclosing_for_line(funcs, int(r["line"])) if funcs is not None else None
        ) or MODULE_LEVEL
        key = "%s::%s" % (r["module"], fn)
        if key not in groups_by_key:
            cd = pick_char_dir(char_dir_map, r["module"], fn, f, fallback)
            if cd is None:
                raise ValueError(
                    "no char_dir for group %s (provide charDirMap entry or defaultCharDir)"
                    % key
                )
            g = {
                "module": r["module"],
                "function": fn,
                "source_file": f,  # SRC_REL-relative
                "char_dir": cd,  # string or list
                "test_file": test_file_for(test_dir_base, r["module"], fn),
                "survivors": [],
            }
            groups_by_key[key] = g
            groups.append(g)
        groups_by_key[key]["survivors"].append(r["mutant_id"])

    # ---- guarantees (raise, never silently pass) ----
    total_out = sum(len(g["survivors"]) for g in groups)
    if len(active) != total_out:
        raise ValueError(
            "survivor conservation failed: %d active in, %d placed"
            % (len(active), total_out)
        )
    seen = set()
    for g in groups:
        for mid in g["survivors"]:
            if mid in seen:
                raise ValueError("survivor %s placed in more than one group" % mid)
            seen.add(mid)
    tf = set()
    for g in groups:
        if g["test_file"] in tf:
            raise ValueError("test_file collision: %s" % g["test_file"])
        tf.add(g["test_file"])

    return {
        "groups": groups,
        "no_test": no_test,
        "stats": {
            "survivors_in": len(survivors),
            "active_survivors": len(active),
            "no_test": len(no_test),
            "groups": len(groups),
        },
    }


# ------------------------------------------------------------------- CLI
def _parse_args(argv):
    o = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            o["help"] = True
            i += 1
            continue
        if a not in (
            "--fixture",
            "--src-root",
            "--default-char-dir",
            "--test-dir-base",
        ):
            raise ValueError("unknown argument: %s" % a)
        if i + 1 >= len(argv):
            raise ValueError("%s requires a value" % a)
        o[a] = argv[i + 1]
        i += 2
    return o


def main(argv):
    try:
        o = _parse_args(argv)
    except ValueError as e:
        sys.stderr.write("mutmut-results-adapter: %s\n" % e)
        return 2
    if o.get("help") or "--fixture" not in o:
        sys.stderr.write(
            "usage: python3 mutmut-results-adapter.py --fixture <records.json> "
            "[--src-root <dir>] [--default-char-dir <dir|a,b>] [--test-dir-base <dir>]\n"
        )
        return 0 if o.get("help") else 2
    try:
        with open(o["--fixture"]) as fh:
            fixture = json.load(fh)
    except Exception as e:
        sys.stderr.write("mutmut-results-adapter: cannot read fixture: %s\n" % e)
        return 2
    survivors = fixture if isinstance(fixture, list) else fixture.get("survivors", [])
    dcd = o.get("--default-char-dir")
    if dcd is not None and "," in dcd:
        dcd = [x.strip() for x in dcd.split(",") if x.strip()]
    opts = {
        "srcRoot": o.get("--src-root")
        or (fixture.get("srcRoot") if isinstance(fixture, dict) else None),
        "testDirBase": o.get("--test-dir-base")
        or (fixture.get("testDirBase") if isinstance(fixture, dict) else None),
        "charDirMap": fixture.get("charDirMap") if isinstance(fixture, dict) else None,
        "defaultCharDir": (
            dcd
            if dcd is not None
            else (fixture.get("defaultCharDir") if isinstance(fixture, dict) else None)
        ),
    }
    try:
        result = build_groups(survivors, opts)
    except ValueError as e:
        sys.stderr.write("mutmut-results-adapter: %s\n" % e)
        return 1
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
