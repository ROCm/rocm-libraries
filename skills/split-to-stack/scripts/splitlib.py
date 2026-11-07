#!/usr/bin/env python3
"""Shared helpers for the split-to-stack scripts.

Read-only against the repository except where a script says otherwise.
Only the Python standard library and the `git` CLI are used.
"""
import os
import re
import subprocess
import sys

# Diff options pinned for the ledger. They are recorded in the ledger header and the
# scripts refuse a ledger whose header differs. Do not change them between runs.
DIFF_OPTS = [
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "--no-relative",
    "--inter-hunk-context=0",
    "-l0",
    "-M50%",
]
HUNK_OPT = "-U0"
DIFF_OPTS_TEXT = " ".join(DIFF_OPTS + [HUNK_OPT])

LEDGER_COLUMNS = [
    "id",
    "path",
    "old_path",
    "status",
    "hunk",
    "plus",
    "minus",
    "binary",
    "mode",
    "generated",
    "commits",
]
GROUP_COLUMNS = [
    "NN",
    "suffix",
    "type",
    "scope",
    "subject",
    "author",
    "trailers",
    "depends_on",
    "summary",
]
ASSIGN_COLUMNS = [
    "entry",
    "step",
    "nsteps",
    "group",
    "state",
    "purpose",
    "note",
]
TRANSITION_COLUMNS = [
    "id",
    "entry",
    "added_in",
    "removed_in",
    "reason",
    "path",
    "stored",
]
TYPES = [
    "fix",
    "chore",
    "style",
    "test",
    "ci",
    "docs",
    "build",
    "refactor",
    "perf",
    "feat",
]
NULL_SHA = "0" * 40

# Plan states that point to a file under RUN. `stored:` is any text (from history or hand-written);
# `subtractive:` must be the final text with lines removed, which check_plan.py verifies.
FILE_STATE_KINDS = ("stored", "subtractive")

# Ledger statuses whose paths can belong to a generated set. A deleted, renamed or type-changed path that
# matches a set becomes an ordinary F row (it is placed with the code it belongs to).
GENERATED_STATUSES = ("A", "M")

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class SplitError(Exception):
    pass


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


# Environment variables that would redirect git to another repository or index.
# (GIT_OBJECT_DIRECTORY / GIT_ALTERNATE_OBJECT_DIRECTORIES are deliberately kept: selftest --stress
# uses them to keep new objects out of the real repository.)
_STRIP_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_REPLACE_REF_BASE",
)


def clean_env(extra=None):
    """os.environ without repo-redirecting variables; paths are literal, no optional locks."""
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
    for k in [
        k
        for k in env
        if k.startswith("GIT_CONFIG_KEY_") or k.startswith("GIT_CONFIG_VALUE_")
    ]:
        del env[k]
    env["GIT_LITERAL_PATHSPECS"] = "1"
    env["GIT_ATTR_NOSYSTEM"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if extra:
        env.update(extra)
    return env


def git_ok(repo, *args):
    """True if `git -C repo args` exits 0 (clean environment, output discarded)."""
    p = subprocess.run(
        ["git", "-C", repo] + list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=clean_env(),
    )
    return p.returncode == 0


def toplevel(repo):
    """Top-level directory of the work tree containing `repo` (or the git dir for a bare repo)."""
    p = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_env(),
    )
    top = p.stdout.decode().strip()
    if p.returncode == 0 and top:
        return top
    p = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--absolute-git-dir"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_env(),
    )
    if p.returncode != 0:
        raise SplitError("%s is not inside a git repository" % repo)
    return p.stdout.decode().strip()


def git(repo, *args, input=None, env=None, check=True):
    """Run git -C repo ...; return stdout bytes."""
    full_env = clean_env(env)
    p = subprocess.run(
        ["git", "-C", repo] + list(args),
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=full_env,
    )
    if check and p.returncode != 0:
        raise SplitError(
            "git %s failed (%d): %s"
            % (
                " ".join(args[:4]),
                p.returncode,
                p.stderr.decode("utf-8", "replace").strip(),
            )
        )
    return p.stdout


def git_text(repo, *args, **kw):
    return git(repo, *args, **kw).decode("utf-8", "surrogateescape").strip()


def rev_parse(repo, rev):
    return git_text(repo, "rev-parse", "--verify", rev + "^{commit}")


def tree_of(repo, rev):
    return git_text(repo, "rev-parse", "--verify", rev + "^{tree}")


def dec(b):
    return os.fsdecode(b)


def enc(s):
    return os.fsencode(s)


class CatFile:
    """Persistent `git cat-file --batch` for blobs plus cached `git ls-tree` lookups for modes."""

    def __init__(self, repo, env=None):
        self.repo = repo
        self.env = env
        self._ents = {}
        full_env = clean_env(env)
        self.batch = subprocess.Popen(
            ["git", "-C", repo, "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=full_env,
        )

    def blob(self, spec):
        """Bytes of the object named by spec, or None if missing."""
        self.batch.stdin.write(enc(spec) + b"\n")
        self.batch.stdin.flush()
        header = self.batch.stdout.readline().split()
        if not header or header[-1] == b"missing":
            return None
        size = int(header[2])
        data = self.batch.stdout.read(size)
        self.batch.stdout.read(1)
        return data

    def prefetch(self, rev, paths):
        """Look up (mode, sha) for many paths at rev with few git calls."""
        todo = [p for p in dict.fromkeys(paths) if (rev, p) not in self._ents]
        for i in range(0, len(todo), 400):
            chunk = todo[i : i + 400]
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    self.repo,
                    "ls-tree",
                    "-z",
                    "-r",
                    "--full-tree",
                    rev,
                    "--",
                ]
                + [enc(p) for p in chunk],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env(self.env),
            )
            if proc.returncode != 0:
                raise SplitError(
                    "git ls-tree failed: "
                    + proc.stderr.decode("utf-8", "replace").strip()
                )
            out = proc.stdout
            found = {}
            for rec in out.split(b"\0"):
                if not rec:
                    continue
                meta, _, path = rec.partition(b"\t")
                mode, _typ, sha = meta.decode().split()
                found[dec(path)] = (mode, sha)
            for p in chunk:
                self._ents[(rev, p)] = found.get(p)

    def entry(self, rev, path):
        """(mode, sha) of path at rev, or None."""
        if (rev, path) not in self._ents:
            self.prefetch(rev, [path])
        return self._ents[(rev, path)]

    def close(self):
        try:
            self.batch.stdin.close()
            self.batch.wait(timeout=10)
        except Exception:
            self.batch.kill()


def split_lines(data):
    """Split bytes into lines on b'\\n' only, keeping the terminator."""
    if not data:
        return []
    parts = data.split(b"\n")
    out = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def first_unmatched(lines, final):
    """Index of the first element of `lines` that cannot be matched when `lines` is read as an ordered
    subsequence of `final` (exact bytes); None if `lines` is such a subsequence."""
    j = 0
    for i, ln in enumerate(lines):
        while j < len(final) and final[j] != ln:
            j += 1
        if j == len(final):
            return i
        j += 1
    return None


def parse_hunk(line):
    m = HUNK_RE.match(line)
    if not m:
        return None
    a = int(m.group(1))
    b = 1 if m.group(2) is None else int(m.group(2))
    c = int(m.group(3))
    d = 1 if m.group(4) is None else int(m.group(4))
    return (a, b, c, d)


def hunk_text(h):
    a, b, c, d = h
    return "-%d,%d +%d,%d" % (a, b, c, d)


def parse_hunk_text(s):
    m = re.match(r"^-(\d+),(\d+) \+(\d+),(\d+)$", s.strip())
    if not m:
        raise SplitError("bad hunk field: %r" % s)
    return tuple(int(x) for x in m.groups())


def _unesc(v):
    """`-` is an empty field; `\\-` is a literal `-`; a leading backslash is doubled."""
    if v == "-":
        return ""
    if v.startswith("\\"):
        return v[1:]
    return v


def _esc(v):
    if v == "":
        return "-"
    if v == "-" or v.startswith("\\"):
        return "\\" + v
    return v


def read_tsv(path, expect_columns=None):
    """Read a TSV with '#' comment lines. Returns (meta_dict, rows).

    `# key=value` comment lines become meta. A field equal to '-' is an empty string.
    """
    meta = {}
    rows = []
    header = None
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", "surrogateescape")
    except OSError as e:
        raise SplitError("cannot read %s: %s" % (path, e.strerror))
    for ln in raw.split("\n"):
        if not ln.strip():
            continue
        if ln.startswith("#"):
            m = re.match(r"^#\s*([A-Za-z0-9_.-]+)=(.*)$", ln)
            if m:
                meta[m.group(1)] = m.group(2)
            continue
        fields = ln.split("\t")
        if header is None:
            header = fields
            if expect_columns is not None and header != expect_columns:
                raise SplitError(
                    "%s: columns %r != expected %r" % (path, header, expect_columns)
                )
            continue
        if len(fields) != len(header):
            raise SplitError(
                "%s: row has %d fields, header has %d: %r"
                % (path, len(fields), len(header), ln[:120])
            )
        rows.append({k: _unesc(v) for k, v in zip(header, fields)})
    if header is None:
        raise SplitError("%s: no header row" % path)
    return meta, rows


def render_tsv(columns, rows, meta_lines=()):
    out = []
    for m in meta_lines:
        out.append("# " + m + "\n")
    out.append("\t".join(columns) + "\n")
    for r in rows:
        vals = []
        for c in columns:
            v = str(r.get(c, ""))
            if "\t" in v or "\n" in v:
                raise SplitError("tab/newline in field %s: %r" % (c, v[:60]))
            vals.append(_esc(v))
        out.append("\t".join(vals) + "\n")
    return "".join(out).encode("utf-8", "surrogateescape")


def write_tsv(path, columns, rows, meta_lines=()):
    data = render_tsv(columns, rows, meta_lines)
    with open(path, "wb") as f:
        f.write(data)


def load_ledger(run):
    meta, rows = read_tsv(os.path.join(run, "ledger.tsv"), LEDGER_COLUMNS)
    for k in ("base", "tip", "diff-options"):
        if k not in meta:
            raise SplitError("ledger.tsv header is missing '# %s='" % k)
    if meta["diff-options"] != DIFF_OPTS_TEXT:
        raise SplitError(
            "ledger.tsv was generated with different diff options:\n  %s\nexpected:\n  %s"
            % (meta["diff-options"], DIFF_OPTS_TEXT)
        )
    for r in rows:
        r["kind"] = r["id"][0]
        r["hunk_t"] = parse_hunk_text(r["hunk"]) if r["hunk"] else None
    return meta, rows


def load_groups(run):
    _, rows = read_tsv(os.path.join(run, "groups.tsv"), GROUP_COLUMNS)
    for r in rows:
        if not re.match(r"^\d{2}$", r["NN"]):
            raise SplitError(
                "groups.tsv: NN must be two digits (01, 02, ...): %r" % r["NN"]
            )
    rows.sort(key=lambda r: int(r["NN"]))
    return rows


def load_assign(run):
    _, rows = read_tsv(os.path.join(run, "plan_assign.tsv"), ASSIGN_COLUMNS)
    for r in rows:
        for c in ("step", "nsteps"):
            if not re.match(r"^[1-9]\d*$", r[c]):
                raise SplitError(
                    "plan_assign.tsv: %s must be a positive integer: %r (entry %s)"
                    % (c, r[c], r["entry"])
                )
    return rows


AUTHOR_RE = re.compile(r"^[^<>]+ <[^<>\s]+>$")


def split_trailers(s):
    """`|`-separated trailers; a literal `|` is written `\\|`."""
    if not s:
        return []
    return [
        t.replace("\\|", "|").strip() for t in re.split(r"(?<!\\)\|", s) if t.strip()
    ]


def contained(run, rel):
    """Absolute path of `rel` under `run`, or SplitError if it escapes the run directory."""
    root = os.path.realpath(run)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise SplitError("state path escapes the run directory: %s" % rel)
    return full


def load_transitions(run):
    p = os.path.join(run, "transitions.tsv")
    if not os.path.exists(p):
        return []
    _, rows = read_tsv(p, TRANSITION_COLUMNS)
    return rows


def is_whole_row(r):
    """A ledger row whose content is taken whole (not hunk-composed)."""
    if r["kind"] == "G":
        return True
    if r["kind"] == "F":
        if r["status"] in ("A", "D", "R", "T"):
            return True
        return r["binary"] in ("yes", "sub", "whole")
    return False


def is_mode_row(r):
    return r["kind"] == "F" and r["status"] == "M" and r["binary"] == "no"


def expand_steps(ledger_rows, assign_rows, group_index):
    """Map ledger id -> sorted list of (group_idx, state). `@set` rows expand to the set's G rows.

    group_index: dict NN-string -> position in stack order.
    Raises SplitError on unknown ids or groups.
    """
    by_id = {r["id"]: r for r in ledger_rows}
    by_set = {}
    for r in ledger_rows:
        if r["kind"] == "G":
            by_set.setdefault(r["generated"], []).append(r["id"])
    steps = {}
    for a in assign_rows:
        g = a["group"]
        if g not in group_index:
            raise SplitError(
                "plan_assign: unknown group %r for entry %r" % (g, a["entry"])
            )
        targets = []
        if a["entry"].startswith("@"):
            name = a["entry"][1:]
            if name not in by_set:
                raise SplitError("plan_assign: unknown generated set %r" % name)
            targets = by_set[name]
        else:
            if a["entry"] not in by_id:
                raise SplitError("plan_assign: unknown ledger entry %r" % a["entry"])
            targets = [a["entry"]]
        for t in targets:
            steps.setdefault(t, []).append((int(a["step"]), group_index[g], a["state"]))
    out = {}
    for k, v in steps.items():
        v.sort()
        out[k] = [(gi, st) for (_, gi, st) in v]
    return out
