# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The shared-state half of the session's Bash guard (orchestrator.guard; lessons L-164, L-591): commands that would
change what the review's other agents and jobs share, each refused with what to run instead.

- `git config` that writes: the mirror's worktrees share its config, so a write changes every other job's git. A
  setting for one command is `git -c key=value <command>`; reads (`--get`, `--list`, `get`, `list`, one key) pass.
- A PR ref fetched without a leading `+`, or into FETCH_HEAD alone: after a force-push the old ref stays, silently, and
  other jobs move FETCH_HEAD. The fetch is `git fetch origin +pull/N/head:refs/remotes/origin/prN`.
- `git worktree add` anywhere but a review directory's `wt_v*` (the verifier) or `wt_q*` (the code quality reviewer)
  path: the two agents in one worktree edit each other's files. `rk review-prep N` prints both paths. And a worktree
  whose folder `rk pr-tree` did not make first, in the same command: a PR runner reads nothing in the scratch folder but
  the trees made so (worker/perms.py), so PR code could not read it.
- A write into the checks' shared build or its tree (the folders a `checks/shared_build` record names, while its build
  may still be reused: slots.kept_build): the review's agents reuse them read-only. Refused: an archive or build step
  there (ar, ranlib, ninja but its `-t` tools, make, `cmake --build`/`--install`/`-B`), a file write there (rm, mv,
  touch, chmod, `sed -i`, patch; cp, rsync, install and ln into it; tar and unzip extracting into it, tar creating an
  archive there; a `>` or `>>` redirect), and a git command that changes the tree (checkout, reset, clean, apply,
  commit and the like). Reading, running its CTest entries and `ninja -t commands` pass.

A command is read as the shell splits it (shlex; `;`, `&&`, `||`, `|`, `&`, parentheses and new lines end a command).
`bash -c SCRIPT` is read as SCRIPT, and so is the here-document a shell reads its script from (`bash -s <<'EOF'`, the
agents' steps); any other here-document is data. `cd DIR` moves the folder later relative paths are read in, and a
worktree's path is read in git's -C folder. A text shlex cannot split is let through (the outward guard still reads
it).
"""

from __future__ import annotations

import os
import re
import shlex

SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "|&", ";;"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
# Programs that run the rest of the line as the command; the options of theirs that take a value.
WRAPPERS = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "-t"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "nohup": set(),
    "time": set(),
    "command": set(),
    "exec": set(),
    "stdbuf": {"-i", "-o", "-e"},
    "sudo": {"-u", "-g", "-C", "-D"},
    "xargs": {"-I", "-n", "-P", "-d", "-L", "-E", "-s"},
}
SHELLS = {"bash", "sh", "zsh", "dash"}
GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
GIT_CONFIG_READ_FLAGS = {
    "--get",
    "--get-all",
    "--get-regexp",
    "--get-urlmatch",
    "-l",
    "--list",
    "--get-color",
    "--get-colorbool",
}
GIT_CONFIG_WRITE_FLAGS = {
    "--add",
    "--unset",
    "--unset-all",
    "--replace-all",
    "--rename-section",
    "--remove-section",
    "-e",
    "--edit",
}
GIT_TREE_WRITES = {
    "checkout",
    "switch",
    "reset",
    "clean",
    "apply",
    "am",
    "stash",
    "merge",
    "rebase",
    "commit",
    "cherry-pick",
    "revert",
    "restore",
    "rm",
    "mv",
    "pull",
    "add",
}
PR_REF = re.compile(r"^(\+?)(?:refs/)?pull/\d+/(?:head|merge)(?::(.*))?$")
WORKTREE_NAME = re.compile(r"^wt_[vq][\w.@-]*$")
ANY_ARG_WRITERS = {
    "ar",
    "llvm-ar",
    "ranlib",
    "llvm-ranlib",
    "strip",
    "rm",
    "rmdir",
    "touch",
    "truncate",
    "shred",
    "chmod",
    "chown",
    "mv",
    "patch",
    "mkdir",
}
DEST_WRITERS = {"cp", "rsync", "install", "ln"}

GIT_CONFIG = (
    "refused: a git config write (the mirror's worktrees share its config, so it changes every other job's git: "
    "lessons L-164). Give the setting to one command: `git -c KEY=VALUE COMMAND`; reads (`git config --get`, "
    "`--list`) are fine"
)
PR_FETCH = (
    "refused: a PR ref fetched without a leading + (after a force-push the old ref stays, silently) or into FETCH_HEAD "
    "(other jobs move it: lessons L-164). Fetch it forced into a named ref: "
    "`git fetch origin +pull/<N>/head:refs/remotes/origin/pr<N>`, then read that ref"
)
WORKTREE = (
    "refused: a worktree outside the review directory's wt_v*/wt_q* paths (the verifier and the code quality "
    "reviewer in one worktree edit each other's files: lessons L-591). Add it as `<review dir>/wt_v<name>` (the "
    "verifier) or `<review dir>/wt_q<name>` (the code quality reviewer); `rk review-prep N` prints both"
)
TREE_FIRST = (
    "refused: a worktree whose folder `rk pr-tree` did not make first, in the same command (a PR runner reads nothing "
    "in the scratch folder but the trees made so, so PR code could not read this one). Make it one, then add it: "
    "`rk pr-tree PATH && git -C MIRROR worktree add --detach PATH COMMIT`"
)


def shared_write(path: str) -> str:
    return (
        f"refused: this writes in the checks' shared build or its tree ({path}), which the review's agents reuse "
        "read-only (re-running one of its steps, such as the archive step, rebuilds librocke_core.a under the "
        "others: lessons L-164). Read it; run its CTest entries (tools/ctest_read_only.py) and `ninja -t commands` "
        "there through `rk pr-run`; build what you change in your own worktree under the review directory (wt_v* or "
        "wt_q*)"
    )


# A here-document's operator and its end word (`<<EOF`, `<<-'EOF'`, `<< "EOF"`); `<<<` is a here-string.
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)(\w+)\1")
# What stands in the text for a here-document: HERE and its body's index (`heredocs`). A command holds no NUL, so no
# word of its own is one.
HERE = "\0here"


def heredocs(command: str) -> tuple[str, list[str]]:
    """The text with each here-document's operator, end word and body taken out and HERE and the body's index in their
    place, and the bodies in order. A body with no end line runs to the text's end, as the shell reads it.
    """
    out, bodies, ends, body = [], [], [], []
    for line in command.split("\n"):
        if ends:
            if line.strip() == ends[0]:
                bodies.append("\n".join(body))
                ends.pop(0)
                body = []
            else:
                body.append(line)
            continue
        parts, last = [], 0
        for m in HEREDOC.finditer(line):
            parts += [line[last : m.start()], f" {HERE}{len(bodies) + len(ends)} "]
            ends.append(m.group(2))
            last = m.end()
        out.append("".join(parts) + line[last:])
    if ends:
        bodies.append("\n".join(body))
    return "\n".join(out), bodies


def split(command: str) -> list[list[str]] | None:
    """The simple commands of a shell text (here-documents taken out: `heredocs`), each as its words (redirections kept
    as tokens); None when shlex cannot split it."""
    command = command.replace("\\\n", " ")
    lex = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        tokens = list(lex)
    except ValueError:
        return None
    out, cur = [], []
    for t in tokens:
        if t in SEPARATORS or (set(t) <= set(";&|()") and t not in REDIRECTS):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def program(words: list[str]) -> list[str]:
    """The words from the program on: assignments and wrappers (env, nice, timeout, ...) and their options skipped."""
    i = 0
    while i < len(words):
        w = words[i]
        if ASSIGNMENT.match(w):
            i += 1
            continue
        name = os.path.basename(w)
        if name not in WRAPPERS:
            break
        takes = WRAPPERS[name]
        i += 1
        while i < len(words) and (
            words[i].startswith("-") or (name == "env" and ASSIGNMENT.match(words[i]))
        ):
            i += 2 if words[i] in takes else 1
        if name == "timeout" and i < len(words):
            i += 1  # its duration
    return words[i:]


def _norm(path: str, cwd: str | None) -> str | None:
    if not path or "$" in path or path.startswith("~"):
        return None
    if not path.startswith("/"):
        if not cwd:
            return None
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


def _under(path: str | None, roots) -> str | None:
    if not path:
        return None
    for r in roots:
        if path == r or path.startswith(r.rstrip("/") + "/"):
            return r
    return None


def _value(words: list[str], *names: str) -> str | None:
    """The value of the first of `names` among words (`-C dir`, `-Cdir` for one-letter names, `--name=dir`)."""
    for i, w in enumerate(words):
        for n in names:
            if w == n and i + 1 < len(words):
                return words[i + 1]
            if n.startswith("--") and w.startswith(n + "="):
                return w[len(n) + 1 :]
            if (
                len(n) == 2
                and not n.startswith("--")
                and w.startswith(n)
                and len(w) > 2
            ):
                return w[2:]
    return None


def _args(words: list[str], takes: set[str] = frozenset()) -> list[str]:
    """The non-option words after the program (an option in `takes` skips its value too)."""
    out, i = [], 1
    while i < len(words):
        w = words[i]
        if w.startswith("-"):
            i += 2 if w in takes else 1
            continue
        out.append(w)
        i += 1
    return out


def git_words(words: list[str]) -> tuple[str | None, str, list[str]]:
    """git's -C folder, its subcommand and the subcommand's words."""
    i, gdir = 1, None
    while i < len(words) and words[i].startswith("-"):
        w = words[i]
        if w in GIT_VALUE_OPTS:
            if w == "-C" and i + 1 < len(words):
                gdir = words[i + 1]
            i += 2
            continue
        i += 1
    if i >= len(words):
        return gdir, "", []
    return gdir, words[i], words[i + 1 :]


def _at(path: str, cwd: str | None) -> str:
    """`path` as the tree rule compares it: read in `cwd` when it can be (_norm), else its own text (a variable's)."""
    return _norm(path, cwd) or os.path.normpath(path)


def _git(words, cwd, shared, trees=frozenset()) -> str | None:
    gdir, sub, rest = git_words(words)
    if sub == "config":
        plain = [w for w in rest if not w.startswith("-")]
        flags = {w.split("=", 1)[0] for w in rest if w.startswith("-")}
        if plain[:1] in (["get"], ["list"]) or flags & GIT_CONFIG_READ_FLAGS:
            return None
        if len(plain) == 1 and not flags & GIT_CONFIG_WRITE_FLAGS:
            return None  # `git config key`: a read
        if not plain and not flags & GIT_CONFIG_WRITE_FLAGS:
            return None  # nothing to write (git prints its usage)
        return GIT_CONFIG
    if sub == "fetch":
        for w in rest:
            m = PR_REF.match(w)
            if m and (not m.group(1) or not m.group(2)):
                return PR_FETCH
        return None
    where = _norm(gdir, cwd) if gdir else cwd
    if sub == "worktree" and rest[:1] == ["add"]:
        path = next(iter(_args(["add"] + rest[1:], {"-b", "-B", "--reason"})), "")
        path = _at(path, where)  # git reads it in its -C folder
        if not worktree_ok(path):
            return WORKTREE
        return None if path in trees else TREE_FIRST
    if sub in GIT_TREE_WRITES and _under(where, shared):
        return shared_write(where)
    return None


def worktree_ok(path: str) -> bool:
    """A review directory's wt_v*/wt_q* path: its last part names the agent's worktree, and its parent, when the path
    has one, is a review directory (`v<N><round>`: prep.review_dir_pr) or a variable the shell fills in.
    """
    parts = [p for p in path.rstrip("/").split("/") if p and p != "."]
    if not parts or not WORKTREE_NAME.match(parts[-1]):
        return False
    if len(parts) == 1 or "$" in parts[-2]:
        return True
    # A call-time import: the session's guard loads the standard library alone, prep most of the package.
    from . import prep

    return prep.review_dir_pr(parts[-2]) is not None


def _shared(words, cwd, shared) -> str | None:
    """Why this simple command writes in a shared build or tree (None when it does not)."""
    if not shared:
        return None
    for i, w in enumerate(words[:-1]):  # a redirect into it
        if w in REDIRECTS:
            hit = _under(_norm(words[i + 1], cwd), shared)
            if hit:
                return shared_write(hit)
    words = [
        w
        for i, w in enumerate(words)
        if w not in REDIRECTS and (i == 0 or words[i - 1] not in REDIRECTS)
    ]
    name = os.path.basename(words[0])
    args = _args(words)

    def hit(paths) -> str | None:
        for p in paths:
            r = _under(_norm(p, cwd), shared)
            if r:
                return shared_write(r)
        return None

    if name in ANY_ARG_WRITERS:
        return hit(args)
    if name == "sed" and any(
        w == "-i" or w.startswith(("-i", "--in-place")) for w in words[1:]
    ):
        return hit(args[1:])
    if name in DEST_WRITERS:
        dest = _value(words, "-t", "--target-directory")
        return hit([dest] if dest else args[-1:])
    if name in ("ninja", "make", "gmake"):
        if name == "ninja" and _value(words, "-t") is not None:
            return None  # its tools (commands, query, targets, compdb) only read
        if name != "ninja" and {"-n", "--dry-run", "-q", "--question"} & set(words):
            return None
        d = _value(words, "-C", "--directory")
        return hit([d] if d else [cwd or ""])
    if name == "cmake":
        return hit(
            [
                v
                for v in (
                    _value(words, "--build"),
                    _value(words, "--install"),
                    _value(words, "-B"),
                )
                if v
            ]
        )
    if name in ("tar", "bsdtar"):
        opts = "".join(
            w.lstrip("-")
            for w in words[1:]
            if w.startswith("-") and not w.startswith("--")
        )
        if len(words) > 1 and not words[1].startswith("-"):
            opts += words[1]  # the old style: tar xzf FILE
        longs = {w.split("=", 1)[0] for w in words[1:] if w.startswith("--")}
        if "x" in opts or {"--extract", "--get"} & longs:
            d = _value(words, "-C", "--directory")
            return hit([d] if d else [cwd or ""])
        if set("cru") & set(opts) or {"--create", "--append", "--update"} & longs:
            f = _value(words, "-f", "--file")
            if f is None:  # a bundle ending in f: the next word is the archive
                for i, w in enumerate(words[1:-1], 1):
                    bundle = (w.startswith("-") and not w.startswith("--")) or i == 1
                    if bundle and w.lstrip("-").endswith("f"):
                        f = words[i + 1]
                        break
            return hit([f] if f else [])
        return None
    if name == "unzip":
        d = _value(words, "-d")
        return hit([d] if d else [cwd or ""])
    return None


def _redirection(word: str) -> bool:
    """Whether the word is a redirection's operator (`>`, `>>`, `<`, `>&` of `2>&1`, `&>`)."""
    return bool(word) and set(word) <= set("<>&|") and not set(word).isdisjoint("<>")


def script_of(words: list[str], bodies: list[str]) -> str | None:
    """The script a shell runs (`words`: from its name on), when the session's guard can read it: -c's, or the
    here-document it reads its script from (with `-s`, or with no script file named). None for a script file, or with
    neither.
    """
    for i, w in enumerate(words[1:-1], 1):
        if w.startswith("-") and not w.startswith("--") and "c" in w:
            return words[i + 1]
    # its options and operands, without its redirections and here-documents
    rest, here, i = [], [], 1
    while i < len(words):
        w = words[i]
        if w.startswith(HERE):
            here.append(int(w[len(HERE) :]))
        elif _redirection(w):
            if rest and rest[-1].isdigit():
                rest.pop()  # the file descriptor, as in 2>&1
            i += 1  # the operator's file
        else:
            rest.append(w)
        i += 1
    if not here or here[0] >= len(bodies):
        return None
    i, stdin = 0, False
    while i < len(rest) and rest[i][:1] in ("-", "+") and rest[i] not in ("-", "--"):
        w = rest[i]
        short = not w.startswith("--")
        stdin = stdin or (short and "s" in w[1:])
        i += 2 if (short and w[-1] in "oO") or w in ("--rcfile", "--init-file") else 1
    if rest[i : i + 1] == ["--"]:
        i += 1
    return bodies[here[0]] if stdin or rest[i : i + 1] in ([], ["-"]) else None


def check(
    command: str, shared=(), cwd: str | None = None, depth: int = 0, trees=None
) -> str | None:
    """Why this Bash command would change shared state (None when it would not). `shared`: the checks' shared builds
    and trees (shared_paths). `trees`: the folders `rk pr-tree` made earlier in the command, as `_at` reads them.
    """
    shared = [os.path.normpath(s) for s in shared if s]
    trees = set() if trees is None else trees
    text, bodies = heredocs(command or "")
    commands = split(text)
    if commands is None:
        return None
    for words in commands:
        words = program(words)
        if not words:
            continue
        name = os.path.basename(words[0])
        if name == "cd":
            target = words[1] if len(words) > 1 else None
            cwd = _norm(target, cwd) if target else None
            continue
        if name in SHELLS and depth < 3:
            script = script_of(words, bodies)
            why = check(script, shared, cwd, depth + 1, trees) if script else None
            if why:
                return why
            continue
        if name == "rk" and words[1:2] == ["pr-tree"]:
            trees.update(_at(p, cwd) for p in words[2:])
        if name == "git":
            why = _git(words, cwd, shared, trees)
        else:
            why = _shared(words, cwd, shared)
        if why:
            return why
    return None


def shared_paths(workers: str) -> list[str]:
    """The shared builds and their trees the checks named (each `<workers>/pr<N>/checks/shared_build` record: the
    build folder, its tree and the tree's commit, one to a line: slots.shared_build_lines), but a record whose build's
    slot is no longer kept for it (slots.kept_build): that folder is another job's."""
    import glob

    from . import slots

    out = []
    for rec in sorted(
        glob.glob(os.path.join(workers or "", "pr*", "checks", "shared_build"))
    ):
        parts = slots.shared_build_lines(rec)
        if parts and not slots.kept_build(parts[0]):
            continue
        out += [p for p in parts[:2] if p.startswith("/")]
    return out
