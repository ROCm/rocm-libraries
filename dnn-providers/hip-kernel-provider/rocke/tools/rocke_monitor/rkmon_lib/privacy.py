# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The privacy check: no shipped file may carry an internal link or host, a user detail (anyone's name, login or email),
a key or a secret, an internal ticket key, or a team chat or wiki reference: the package is checked in to a public
repository, and every site-specific value comes from the deployment's own settings and secrets, outside the package.

It checks two things:
- identifying values, read at check time from their sources (`sources`) and never written anywhere: this machine's
  user and host names and its git identity, always; the deny list's values; and from each source an option names,
  the operator's login and the names and concrete paths in a live config (--config), the GitHub logins and PR numbers
  a live state knows (--state), the Claude workspace's JIRA and wiki settings and the people its skill and lessons
  name (--workspace), and the git identity of other folders (--git). PR numbers are looked for as bare numbers in
  code and tests (`pr<N>`, `v<N>fu1`, a test's fixture number: in prose a PR number is `#N` or `PR N`, which the
  pattern check below finds in every file, while a bare number in a document is mostly a setting, a port or a size);
- patterns: email addresses, home folders with a user name, AppData and cloud-synced folder paths, token and secret
  formats, ticket keys with a number, PR and issue numbers (rocKE's review rule 1: a PR number points to its author;
  invented examples such as #1111 or #12345 pass), GitHub @handles (invented ones such as @dev-a pass), URLs outside
  the allow-list (a host with a <placeholder> in it is the host its fixed part names: `_public_url`), internal hosts,
  wiki page ids, GPU product and code names (rocKE's AGENTS.md: gfx ids only), paths to documents outside the
  repository, absolute performance results (TFLOP/s, bandwidth, kernel times: rocKE's review rule 2 and AGENTS.md),
  dated narratives (a comment or line that starts with a date and a colon, the repository's "no dated run
  narratives"), attributions (a person's request or answer as the reason, a decision with its date), and the
  monitor's concrete incident ids (INC-<date>-<n>; invented ones on the first of January pass); and, whole files,
  settings files of either kind, deploy or build (one machine's private values), known by their header whatever
  their name (settings.header_line).

Both read each file as it is and, a .json file, with its strings decoded too (`https:\\/\\/<host>\\/page` reads
`https://<host>/page`), a line counting a match as often as the reading that shows it more.

The deny list (`--deny FILE`, which the check requires; kept outside the package and never shipped) gives the values
to look for, so no check needs a monitor's live files. One entry per line, a line that starts with `#` a comment:
- `VALUE`: a value the package must never hold (a login, a name, a host, a path, a key);
- `pr:<n>`: a PR number, looked for as a bare number in code and tests;
- `domain:<name>`: a company domain: every host under it is internal (the public documentation sites aside).
(`deny_values` reads two other forms of line a list may hold.) `--seed-deny FILE` adds the live sources' values to it,
one per line, with the forms of the monitor's folders and files and the team dashboard's address, which the sources do
not give (`site_values`), so later runs need no live sources; a source it cannot read, or one that gives no value,
stops it before it adds anything.

`run` is the check, and the one way to run it (the command line runs it too): it reads the values, scans the files and
the templates, and writes the report with its result. `command` is that command line, run in a package folder so that
the check runs as the folder's own code: the package build, the sync and the container tests run it so.

The check cannot pass while checking nothing: a source named that cannot be read or gives no value, and a path named
that does not exist or has no file in it (each path named is counted on its own), are NOT CHECKED, and NOT CHECKED is a
FAIL.

A path that is a package folder (one with the monitor's code in it: paths.is_package) is checked as a whole package:
- its parts, the docs, the schemas, the skills (agentic/skills/ and .claude/skills/) and the tests (`PARTS`), each
  need a file read, or that part is NOT CHECKED;
- its settings templates, every template the package gives you (settings_templates: the build template, the deploy
  template of each profile and the deploy file's every-key text), are rendered as docker/rkmon prints them and
  scanned like a file, every line of them, and fail on an absolute path and on a private key (the schema's x-class
  "private") whose value is not blank, `<...>` placeholders only or the key's own default, those of the blocks a
  template shows commented out included; their row counts the private keys checked in each, and a template with none
  is NOT CHECKED. The code that runs the check renders them, so they are checked only when that code is the folder's
  own (`command`, run in the folder): over another package folder they are NOT CHECKED, and so is each template that
  cannot be rendered and read back, which the row names with the command that prints it.

The report gives labels and counts only (checks.render): a row for each source named, for the files, for each part
and for the templates of a package folder, one for each kind of identifying value that has values, and one for each
pattern; its result line counts the matches, and the files and the templates they are in. `--where` adds file:line
(template NAME:line for a template) and the label of each match (never the value), to fix them. A FAIL row's fix is
a command (rkmon_lib/checks.py), with the few words around it. A report shows no path, so where the fix runs the
check again (`AGAIN`, in the package folder), a stand-in takes each path's place: DENY_LIST for the deny list, FILE
and FOLDER for another source. The seeding and the package build, which print a source's row to you, give the command
you ran, naming another source (`sources`).

Usage: python3 -B -m rkmon_lib.privacy --deny F [--config F] [--state F] [--workspace DIR] [--git DIR ...]
       [--report F] [--where] PATH...
       python3 -B -m rkmon_lib.privacy [--config F] [--state F] [--workspace DIR] [--git DIR ...] --seed-deny F
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import functools
import getpass
import json
import os
import pathlib
import re
import socket
import subprocess
import sys

from . import checks, container_layout, models, paths, private_files, renames

# Public sites a shipped file may link (besides github.com/ROCm): documentation and schemas, listed one by one.
PUBLIC_SITES = (
    "rocm.docs.amd.com",
    "docs.amd.com",
    "api.github.com",
    "llvm.org",
    "docs.python.org",
    "pre-commit.com",
    "adaptivecards.io",
    "github.blog",
    "docs.github.com",
    "cli.github.com",
    "pypi.org",
    "ollama.com",
    "code.claude.com",
    "docs.anthropic.com",
    "api.anthropic.com",  # rkmon_lib/bindings.py: Anthropic's own API, which Claude Code reaches with no gateway
    "git-scm.com",
    "www.python.org",
    "cmake.org",
    "ninja-build.org",
    "learn.microsoft.com",
    "keyserver.ubuntu.com",  # docker/pkgs.sh: the signing keys of the Ubuntu archives below
    "ppa.launchpadcontent.net",  # docker/pkgs.sh: Python 3.12 and git 2.44 for the older Ubuntu bases
    "stable.repo.amd.com",  # rkmon_lib/build_plan.py: AMD's public ROCm installers (rocm.docs.amd.com points there)
    "json-schema.org",  # schema/*.schema.json: the JSON Schema dialect the settings files' schemas are written in
)
GITHUB_ORG = "ROCm"
GITHUB_PUBLIC = (
    "cli/cli",
)  # the GitHub CLI's official releases, which the image build fetches gh from (deploy.GH_RELEASES)
# Hosts that are never internal: this machine, the reserved example names (RFC 2606, RFC 6761) and the reserved
# documentation addresses (RFC 5737: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24).
LOCAL_HOSTS = re.compile(
    r"^(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|(?:[\w-]+\.)*(?:example\.(?:com|org|net)|"
    r"[\w-]+\.(?:example|invalid|test|localhost))|(?:192\.0\.2|198\.51\.100|203\.0\.113)\."
    r"\d{1,3})(?::\d+)?$",
    re.I,
)
# Product domains that only ever name an internal site: wikis and trackers, document stores, chat, webhooks. Kept as
# (name, suffix) pairs, so this file names no host itself.
INTERNAL_PRODUCTS = tuple(
    a + "." + b
    for a, b in (
        ("atlassian", "net"),
        ("atlassian", "com"),
        ("sharepoint", "com"),
        ("teams.microsoft", "com"),
        ("office", "com"),
        ("office365", "com"),
        ("logic.azure", "com"),
        ("powerplatform", "com"),
        ("slack", "com"),
    )
)
# Ticket-like words that are standard names, never tickets.
SAFE_KEYS = {
    "UTF",
    "SHA",
    "ISO",
    "RFC",
    "IEEE",
    "ECMA",
    "PEP",
    "HTTP",
    "TLS",
    "SSL",
    "AES",
    "MD",
    "CVE",
    "CWE",
    "GFX",
    "LLVM",
    "X86",
    "ARM",
    "POSIX",
    "FP",
    "BF",
    "MI",
    "BASE",
    "LATIN",
    "CP",
    "WIN",
    "IPV",
    "GLM",
    "GPT",
}  # public model families (GLM-5, GPT-4): no tickets
# User names in home paths that are placeholders, not people.
# (rkmon: the container image's own account, docker/Dockerfile.)
PLACEHOLDER_USERS = {
    "user",
    "username",
    "you",
    "me",
    "op",
    "runner",
    "<user>",
    "$user",
    "${user}",
    "name",
    "...",
    *(u.name for u in container_layout.USERS),
}  # `/home/...`: an elided path; the image's own users (rkmon_lib/container_layout.py)
NOT_LOGINS = {
    "body",
    "name",
    "me",
    "you",
    "user",
    "the",
    "us",
    "ci",
    "author",
    "authors",
    "someone",
    "nobody",
    "unknown",
    "none",
    "null",
    "ghost",
    # GitHub's own review and scanning bots, which the code must name to skip or count them: no person
    "copilot",
    "copilot-pull-request-reviewer",
    "github-advanced-security",
    "github-actions",
    "dependabot",
}

L_ME = "the operator's GitHub login (integrations.github_cli.username)"
L_GIT = "the operator's git name and email"
L_LOGINS = (
    "GitHub logins the live state knows (authors, reviewers, approvers, code owners)"
)
L_CFG = "names in the live config (message label, web host and its login, mapped logins and emails, people in notes)"
L_PEOPLE = "people's names in the live skill and lessons"
L_MACHINE = "this machine's user and host names, and the WSL distribution"
L_PATHS = "concrete paths from the live config"
L_WORKSPACE = "the Claude workspace's JIRA and wiki settings (project, epic, component, labels, sites, space key and id)"
L_DENY = "values in the deny list (--deny)"
L_PRNUMS = "PR numbers the live state knows, as bare numbers in code and tests"
P_EMAIL = "email addresses (other than reserved example domains)"
P_HOME = "home folders with a user name"
P_SYNC = "AppData, Temp and cloud-synced folder paths"
P_TOKEN = "token and secret formats"
P_TICKET = "ticket keys with a number (<PROJECT>-<N>)"
P_URL = "URLs outside the allow-list (github.com/ROCm and the listed public sites)"
P_HOST = (
    "internal hosts (the company's domains, wiki, document, chat and webhook sites)"
)
P_WIKI = "wiki page ids"
P_PR = (
    "PR and issue numbers (#<4-6 digits>, PR <N>, PR URLs) other than invented examples"
)
P_GPU = "GPU product and code names (use gfx ids)"
P_DOC = "paths to documents outside the repository"
P_HANDLE = "GitHub @handles (other than invented examples such as @dev-a)"
P_PERF = "absolute performance results (TFLOP/s, bandwidth, kernel times, MFU)"
P_SETTINGS = "settings files, deploy or build (known by their header: one machine's private values)"
P_ARTIFACTS = (
    "internal artifact repositories (an artifact store's host or repository path)"
)
P_DATED = "dated narratives (a line, comment or docstring that starts with MM-DD: or YYYY-MM-DD:, or MM-DD #N:)"
P_ATTRIB = "attributions (a person's request or answer as the reason; a decision with its date)"
P_INCIDENT = "concrete incident ids (INC-YYYY-MM-DD-N; invented ones dated the first of January pass)"
VALUE_LABELS = (
    L_ME,
    L_GIT,
    L_LOGINS,
    L_CFG,
    L_PEOPLE,
    L_MACHINE,
    L_PATHS,
    L_WORKSPACE,
    L_DENY,
    L_PRNUMS,
)
PATTERN_LABELS = (
    P_EMAIL,
    P_HOME,
    P_SYNC,
    P_TOKEN,
    P_TICKET,
    P_PR,
    P_HANDLE,
    P_URL,
    P_HOST,
    P_WIKI,
    P_GPU,
    P_DOC,
    P_PERF,
    P_SETTINGS,
    P_ARTIFACTS,
    P_DATED,
    P_ATTRIB,
    P_INCIDENT,
)

# Token formats, written in pieces so this file never holds one.
TOKENS = re.compile(
    "|".join(
        [
            r"\bgh[pousr]_[A-Za-z0-9]{20,}",
            r"\bgithub_pat_[A-Za-z0-9_]{20,}",
            r"\bATATT[A-Za-z0-9_=-]{20,}",
            r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
            r"\bsk-ant-[A-Za-z0-9_-]{20,}",
            r"\bAKIA[0-9A-Z]{16}\b",
            "-----BEGIN" + r" (?:[A-Z]+ )*PRIVATE KEY-----",
            r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*",
        ]
    )
)
EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
)
HOME = re.compile(
    r"(?:/home/|/Users/|[A-Za-z]:\\Users\\|/mnt/[a-z]/Users/|/[a-z]/Users/)([A-Za-z0-9._$<{-][\w.$<>{}-]*)",
    re.I,
)
SYNC = re.compile(
    r"[\\/]AppData[\\/]|%LOCAL"
    r"APPDATA%|%APP"
    r"DATA%|[\\/]Local[\\/]Temp[\\/]|One"
    r"Drive - |"
    r"One"
    r"Drive---|"
    r"[\\/]OneDrive[\\/]|[\\/]Dropbox[\\/]|[\\/]Google Drive[\\/]|[\\/]Mobile Documents[\\/]|"
    r"[\\/]iCloud Drive[\\/]",
    re.I,
)
TICKET = re.compile(
    r"(?<![\w-])([A-Z][A-Z0-9]{1,15})-(\d+)(?![\w]|\.\d)"
)  # a version after it: no ticket
# The monitor's own incident ids, INC-<date>-<n>, like its lesson (L-<n>) and finding (N<n>) ids: never a ticket.
OWN_IDS = re.compile(r"(?<![\w-])INC-(?:\d{4}-\d{2}-\d{2}-\d+|\d{1,3})(?![\w-])")
# A URL built at run time (a format field, a shell variable, a pattern) names no concrete host or path.
TEMPLATED = re.compile(r"[{}$%<\\+]|:$|\[")
# A placeholder in a host or a port: `<site>`, `<host address>`, `<port>`.
HOST_PLACEHOLDER = re.compile(r"<[^<>\"'\n/\\]+>")
# A port at the end of a host: digits or a placeholder after the colon, or nothing.
PORT = re.compile(r":(?:\d*|<[^<>\"'\n/\\]+>)$")
# A URL: its host, placeholders in it included, then its path. "|" ends a table cell; a backslash before a quote is
# the quote's escape (`\"https://<site>.atlassian.net\"` in a JSON text), not the host's.
URL = re.compile(
    r"\b(?:https?|ftp|wss?)://((?:<[^<>\"'\n/\\]+>|[^/\\\s\"'`)<>\]|]|\\(?![\"']))+)"
    r"([^\s\"'`)<>\]|]*)",
    re.I,
)
HOSTLIKE = re.compile(r"(?<![\w.@/-])((?:[a-z0-9-]+\.)+[a-z]{2,})(?![\w-])", re.I)
PR_HASH = re.compile(
    r"(?<![\w&/:])#(\d{4,6})\b"
)  # not a CSS colour (after ":") nor an entity
PR_WORD = re.compile(r"\bPRs?\s+#?(\d{4,6})\b")
PR_URL = re.compile(r"github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/(\d{4,6})\b")
# Product and code names, kept in halves so this file names none of them itself.
_NAMES = (
    "Inst|inct",
    "Rad|eon",
    "Ry|zen",
    "EP|YC",
    "St|rix",
    "Krack|an",
    "Alde|baran",
    "Arct|urus",
)
GPU = re.compile(
    r"\bMI\d{2,3}[A-Z]{0,2}\b|\b(?:"
    + "|".join(n.replace("|", "") for n in _NAMES)
    + r")\b|"
    r"\bNa"
    r"vi\s?\d{1,2}\b|\b[RC]D"
    r"NA\s?\d\b|\bAqua\s+Van"
    r"jaram\b"
)
DOC = re.compile(
    r"(?:~|/home/[\w.-]+|/mnt/[a-z]|[A-Za-z]:\\)[^\s\"'`)<>]*\.(?:md|pdf|docx?|xlsx?|pptx?|html?)\b|"
    r"[\\/]Down"
    r"loads[\\/]|~/Down"
    r"loads\b|[\\/]Desk"
    r"top[\\/]|[\\/]Docu"
    r"ments[\\/]"
)


# An @handle: not an email (a word before it), a shell or jq form, a path, or a placeholder such as @BASE@.
HANDLE = re.compile(
    r"(?<![\w$.{}/\\@-])@([A-Za-z0-9][A-Za-z0-9-]*(?:/[\w.-]+)?)(?![\w@])"
)
# Words after an @ that are never people: CSS at-rules, jq formats, doc tags, Python decorators and their modules.
SAFE_AT = {
    "media",
    "keyframes",
    "font-face",
    "import",
    "supports",
    "page",
    "charset",
    "namespace",
    "layer",
    "container",
    "tsv",
    "csv",
    "json",
    "sh",
    "base64",
    "base64d",
    "uri",
    "text",
    "html",
    "param",
    "return",
    "returns",
    "raises",
    "throws",
    "type",
    "see",
    "example",
    "brief",
    "note",
    "todo",
    "dataclass",
    "property",
    "staticmethod",
    "classmethod",
    "abstractmethod",
    "override",
    "pytest",
    "mock",
    "unittest",
    "functools",
    "contextlib",
    "contextmanager",
    "wraps",
    "cache",
    "lru_cache",
    "cached_property",
    "patch",
    "localhost",
    "latest",
    "copilot",
    "handle",
    "handles",
    "word",
    "words",
    "mention",
    "mentions",
}
# Invented handles in examples and tests: dev-a, op-login, a single letter, the made-up org acme.
INVENTED_HANDLE = re.compile(
    r"(?:dev|op|user|author|reviewer|someone|bot|login|team)(?:-[\w]+)*|[a-z]|acme(?:/[\w.-]+)?",
    re.I,
)
# Absolute performance results (rocKE's review rule 2, AGENTS.md): throughput, bandwidth, kernel times and MFU.
PERF = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:[KMGTP]?FLOP/?s|[KMGTP]?FLOPS|TOPS|[KMGT]i?B/s|[KMGT]bps|(?:ms|us|µs|μs|ns)\b)|"
    r"\bMFU\b[^.\n]{0,20}?\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?%\s?MFU\b"
)


def _handle_ok(line: str, m: re.Match) -> bool:
    """An @word that names nobody: a safe word, an invented handle, or a decorator alone at the start of a line."""
    word = m.group(1)
    if word.split("/")[0].lower() in SAFE_AT or INVENTED_HANDLE.fullmatch(word):
        return True
    after = line[m.end() : m.end() + 1]
    return not line[: m.start()].strip() and after in ("", "(", ".")


def invented(n: str) -> bool:
    """An obviously invented PR number: one repeated digit (#1111, #99999) or a run of digits (#1234, #12345)."""
    return len(set(n)) == 1 or n in "1234567890" or n in "9876543210"


# An artifact store names only an internal one here, whatever its host (an example host included): the image's
# downloads from one are build arguments from the operator's settings, never in a file of the package. The product
# name in pieces, so this file names none; it is matched inside a host name too.
ARTIFACTS = re.compile("art" + "ifactory", re.I)
WIKI = re.compile(
    r"/pages/\d{5,}|\bpageId=\d+|\bviewpage\.action\b|(?i:\bpage(?:\s+id)?\s+#?\d{8,}\b)"
)

# A dated narrative: a month-day (a year before it optional) and a colon or comma, or a PR, a time or an incident
# and then a colon ("MM-DD #N: ...", "MM-DD HH:MMZ: ..."), at the start of a line, a comment, a docstring, a bullet,
# a table cell or a parenthesis. A time (10:31), an ISO date in data (2026-10-31T..., a key in quotes), a range or
# ratio (10-31), a range after a word ("lines 10-20: ...") and a review table's row key ("Ours (11-01) N1: ...",
# "dev-b (11-04): ...") are not one.
DATED = re.compile(
    r"(?:^|(?<=[\s(\[\"']))(?:20\d\d-)?(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?=[:,](?:\s|$)|\s+(?:#\w+|\d\d:[\dx]{1,2}\w*|INC-[\w-]+)[^:\n]{0,30}?:(?:\s|$))"
)
DATED_BEFORE = (
    "#",
    '"""',
    "'''",
    '"',
    "'",
    "(",
    "[",
    "-",
    "*",
    ">",
    "//",
    ";",
    ".",
    ":",
    "|",
    "\u2014",
    "\u2013",
)
# An attribution: a person's request or answer as the reason for code, or any decision of the user's or the
# operator's with its date. A rule that names who decides ("only on the operator's instruction", "the user's call")
# is a present-tense fact, not an attribution. Written in pieces so this file holds no attribution itself.
_WHO = r"(?:user|operator)"
ATTRIB = re.compile(
    r"\bthe user"
    r"'s (?:requests?|answers?)\b|"
    r"\bthe " + _WHO + r"'s [\w -]{1,40}?,\s*(?:20\d\d-)?(?:0[1-9]|1[0-2])-[0-3]\d\b|"
    r"\((?:the )?" + _WHO + r",\s*(?:20\d\d-)?(?:0[1-9]|1[0-2])-[0-3]\d\b"
)
# The monitor's concrete incident ids, INC-<date>-<n>, name a past run's incident. Generic ones (INC-1, a format
# field) and invented ones dated on the first of January pass.
INCIDENT = re.compile(r"(?<![\w-])INC-\d{4}-(?!01-01-)\d{2}-\d{2}-\d+(?![\w-])")
# Numbers that are settings, ports or sizes, never a PR, whatever the live state has: Ollama's port, a token
# budget, the dashboards' ports. Round thousands are sizes and limits too. `safe_numbers` adds the admin dashboard's
# public port's default.
SAFE_NUMBERS = {
    str(models.OLLAMA_PORT),
    "16000",
    str(container_layout.ADMIN_PORT),
    str(container_layout.TEAM_PORT),
}


@functools.cache
def safe_numbers() -> frozenset:
    """SAFE_NUMBERS and the admin dashboard's public port's default, the deploy schema's (registry.default): read at
    the first call, as the registry reads modules that import this one."""
    from . import registry

    return frozenset(
        SAFE_NUMBERS | {str(registry.default("admin_dashboard.public_port"))}
    )


def _dated_ok(line: str, m: re.Match) -> bool:
    """A month-day with a colon after a word ("lines 10-20: ...") is a range, not a dated narrative."""
    before = line[: m.start()].rstrip()
    return bool(before) and not before.endswith(DATED_BEFORE)


def state_pr_numbers(st: dict) -> set[str]:
    """Every PR number the live state knows: the keys of its PR-keyed maps (prs, conflicts, overlaps, merges, ...)
    and the values of its `pr`, `n` and `number` fields, 4 to 6 digits."""
    out = set()

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and re.fullmatch(r"\d{4,6}", k):
                    out.add(k)
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif (
            key in ("pr", "n", "number")
            and not isinstance(x, bool)
            and isinstance(x, (int, str))
            and re.fullmatch(r"\d{4,6}", str(x))
        ):
            out.add(str(x))

    walk(st)
    return {
        n for n in out if n not in safe_numbers() and int(n) % 1000 and not invented(n)
    }


def pr_numbers_rx(numbers) -> re.Pattern | None:
    """Any of these numbers, bare: not inside a longer number or a hex digest, not after # (the pattern check's)."""
    nums = sorted(set(numbers) - safe_numbers(), key=lambda n: (-len(n), n))
    return (
        re.compile(r"(?<![\dA-Fa-f#])(?:" + "|".join(nums) + r")(?!\d)")
        if nums
        else None
    )


def is_code(path: pathlib.Path, text: str) -> bool:
    """A Python file: .py, or an extensionless file whose first line runs Python."""
    if path.suffix == ".py":
        return True
    first = text.split("\n", 1)[0]
    return not path.suffix and first.startswith("#!") and "python" in first


def _git(args: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _login_ok(x) -> bool:
    return (
        isinstance(x, str)
        and bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", x))
        and x.lower() not in NOT_LOGINS
        and not x.endswith("[bot]")
    )


def state_logins(st: dict) -> set[str]:
    """Every GitHub login in the live state: PR authors, reviewers, approvers, mergers, requesters, code owners."""
    out = set()
    keys = {
        "author",
        "login",
        "merged_by",
        "actor",
        "requester",
        "approver",
        "assignee",
        "user",
    }

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif key in keys and _login_ok(x):
            out.add(x)

    walk(st)
    teams = (st.get("codeowners") or {}).get("teams") or {}
    for members in teams.values():
        out |= {m for m in members if _login_ok(m)}
    for rec in (
        st.get("prs") or {}
    ).values():  # requested reviewers: people, not the team slugs among them
        out |= {
            x
            for x in rec.get("review_requests") or []
            if _login_ok(x)
            and x not in teams
            and not re.search(
                r"reviewers|core$|infra|documentation|-lib$|-platform$", x
            )
        }
    return out


NAME_IN_NOTE = re.compile(
    r"\b([A-Z][a-z]+ [A-Z][a-z]+)(?:'s (?:request|requests|rule|rules|label)| via the user| via the "
    r"operator|, relayed)"
)


def _names(text: str) -> set[str]:
    out = set()
    for full in NAME_IN_NOTE.findall(text):
        out.add(full)
        out |= {w for w in full.split() if len(w) >= 4}
    return out


WORKSPACE_KEYS = (
    "project_key",
    "default_epic",
    "default_component",
    "default_labels",
    "base_url",
    "browse_url",
    "site",
    "space_key",
    "space_id",
    "space",
    "jira_server",
    "jira_server_id",
)


def workspace_settings(ws: pathlib.Path) -> dict[str, str]:
    """The JIRA and wiki settings of the Claude workspace's project_config.md (key -> value)."""
    try:
        text = (ws / ".claude" / "project_config.md").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return {}
    out = {}
    for key, value in re.findall(
        r"^\|\s*`?([a-z_]+)`?\s*\|\s*`?([^|`]+?)`?\s*\|", text, re.M
    ):
        if (
            key in WORKSPACE_KEYS
            and value.strip()
            and value.strip().lower() not in ("none", "-", "n/a")
        ):
            out.setdefault(key, value.strip())
    return out


# The words of the public sites' names: a bare site name such as these says nothing on its own (the site's full host is
# an internal product host, which the host check catches).
PUBLIC_WORDS = {w for h in PUBLIC_SITES for w in h.split(".")} | {
    "rocke",
    "github",
    "jira",
    "wiki",
}


# Two forms of line a deny list may hold beside its entries (`deny_values`); the seeding writes neither.
WITH_TEXT = " => "  # `VALUE => TEXT`: VALUE is looked for
NOT_LOOKED_FOR = " -> "  # `VALUE -> TEXT`: nothing is
DOMAIN = "domain:"
# In a values map: the company domains of the deny list (`domain:` lines), for the host check only.
DOMAINS = "_domains"


def _text(path) -> str | None:
    """A file's text (UTF-8); None when it cannot be read as one (no path, missing, a folder, unreadable, not UTF-8)."""
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError):
        return None


def _deny_lines(path) -> list[str]:
    lines = (_text(path) or "").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]


def deny_values(path) -> set[str]:
    """The deny list's values: one entry per line, a line that starts with `#` a comment; a PR number is a `pr:<n>`
    line, a company domain a `domain:<name>` line. A list may also hold two forms of line the seeding never writes:
    `VALUE => TEXT`, read as VALUE, and `VALUE -> TEXT`, which gives no value and is skipped.
    """
    return {
        x.split(WITH_TEXT, 1)[0].strip()
        for x in _deny_lines(path)
        if NOT_LOOKED_FOR not in x
    }


def seed_deny(path, values: dict, site=()) -> int:
    """Add each live value (`values`, the deny list's own aside) and each site value (`site`: site_values) that the
    deny list does not hold yet to it, one per line, a PR number as `pr:<n>` (the list made mode 600, its folder 700);
    how many were new. A path written from the home folder (`~/...`) leads to nobody: it is not added. Prints no
    value."""
    p = pathlib.Path(path)
    text = _text(p) or ""
    new = {v for v in site if len(v) >= 2}
    for label, vals in values.items():
        if label in (L_DENY, DOMAINS):
            continue  # the deny list's own values
        new |= {"pr:" + v if label == L_PRNUMS else v for v in vals if len(v) >= 2}
    new = sorted(v for v in new - deny_values(p) if not v.startswith("~"))
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(
        os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8"
    ) as f:
        if not text.strip():
            f.write(
                "# The privacy check's deny list: one value per line. Never shipped, never committed.\n"
            )
        elif new and not text.endswith("\n"):
            f.write("\n")  # the list's last line ends before the first value added
        f.writelines(v + "\n" for v in new)
    p.chmod(0o600)
    return len(new)


# A cloud site's host, <site>.atlassian.net: the atlassian CLI names such a site (`--site NAME`).
CLOUD_SITE = re.compile(r"[^./]+\.atlassian\.net", re.IGNORECASE)


def _workspace_values(ws: pathlib.Path) -> set[str]:
    """The JIRA project, epic and site, and the wiki's site, space and ids: a cloud site's host with its name, as
    the atlassian CLI names it (atlassian.site_name)."""
    from . import atlassian

    out = set()
    for key, value in workspace_settings(ws).items():
        if key in ("default_component", "default_labels"):
            continue  # descriptive words (the package templates them), not identifiers
        if key in ("base_url", "browse_url", "site"):
            m = URL.search(value)
            host = m.group(1).split(":")[0] if m else value.split("/")[0]
            out.add(host)
            if CLOUD_SITE.fullmatch(host):
                out.add(atlassian.site_name(host))
        else:
            out.add(value)
    return {x for x in out if len(x) >= 4 and x.lower() not in PUBLIC_WORDS}


def _skill_text(ws: pathlib.Path | None) -> str:
    if not ws:
        return ""
    d = ws / ".claude" / "skills" / "rocke-monitor"
    parts = []
    for f in sorted(d.glob("*.md")) if d.is_dir() else []:
        try:
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def _from_machine(environ) -> dict:
    """Label -> the values this machine gives: its user and host names and its WSL distribution (L_MACHINE), and the git
    identity of the folder the check runs in and of the environment (GITNAME, GITEMAIL; L_GIT).
    """
    users = {
        environ.get("WINUSER", ""),
        environ.get("USER", ""),
        environ.get("LOGNAME", ""),
    }
    try:
        users.add(getpass.getuser())
    except (OSError, KeyError):
        pass
    hosts = {
        environ.get("WINHOST", ""),
        socket.gethostname(),
        environ.get("WSL_DISTRO_NAME", ""),
    }
    try:
        hosts.add(pathlib.Path("/etc/hostname").read_text().strip())
    except OSError:
        pass
    return {
        L_MACHINE: {
            x
            for x in users | hosts
            if x and len(x) >= 3 and x.lower() not in PLACEHOLDER_USERS | {"root"}
        },
        L_GIT: _from_git([None])[L_GIT]
        | {environ.get("GITNAME", "").strip(), environ.get("GITEMAIL", "").strip()},
    }


def _from_git(dirs) -> dict:
    """Label -> the git identity (user.name, user.email) git gives in each folder (None: the one the check runs in), the
    toolkit's own (rkmon) aside (L_GIT)."""
    out = set()
    for d in dirs:
        for key in ("user.name", "user.email"):
            val = _git(["config", key], cwd=str(d) if d else None)
            if val and val not in ("rkmon", "rkmon@localhost"):
                out.add(val)
    return {L_GIT: out}


def _from_config(cfg: dict) -> dict:
    """Label -> the values a live configuration gives: the operator's login (L_ME), the names in it (L_CFG) and its
    concrete paths (L_PATHS)."""
    cfg = renames.migrate(cfg)[0]
    v = {L_ME: set(), L_CFG: set(), L_PATHS: set()}
    login = ((cfg.get("operator") or {}).get("github_login") or "").strip()
    if login:
        v[L_ME].add(login)
    pub = (cfg.get("publish") or {}).get("html") or {}
    for key in ("host", "user"):
        if pub.get(key):
            v[L_CFG].add(pub[key])
    if pub.get("url"):
        m = URL.search(pub["url"])
        if m:
            v[L_CFG].add(m.group(1).split(":")[0])
    if (cfg.get("teams") or {}).get("channel"):
        v[L_CFG].add(cfg["teams"]["channel"])
    for account, email in ((cfg.get("jira") or {}).get("accounts") or {}).items():
        v[L_CFG] |= {account, email}
    v[L_CFG] |= _names(json.dumps(cfg))
    concrete = set()

    def collect(x):
        if isinstance(x, dict):
            for y in x.values():
                collect(y)
        elif isinstance(x, list):
            for y in x:
                collect(y)
        elif (
            isinstance(x, str)
            and (x.startswith(("/", "~/")) or re.match(r"[A-Za-z]:\\", x))
            and len(x) > 5
        ):
            concrete.add(x.rstrip("/"))

    collect(cfg.get("paths") or {})
    collect(
        [
            (cfg.get("weekly_report") or {}).get("dir"),
            (cfg.get("admin_dashboard") or {}).get("access_token_file"),
            (cfg.get("teams") or {}).get("webhook_file"),
            pub.get("key"),
            pub.get("dest"),
            pub.get("site"),
            pub.get("requests"),
        ]
    )
    for x in list(concrete):
        if x.startswith("/mnt/") or re.match(r"[A-Za-z]:\\", x):
            concrete.add(str(pathlib.PurePosixPath(x).parent))
    v[L_PATHS] = {p for p in concrete if p.count("/") >= 2 or not p.startswith("~")}
    return v


def _win(p: str) -> list[str]:
    """A /mnt/<d>/... path's other forms: Windows (C:\\..., C:/...) and Git Bash (/c/...)."""
    m = re.match(r"/mnt/([a-z])/(.*)", p)
    if not m:
        return []
    rest = m.group(2)
    return [
        f"{m.group(1).upper()}:\\" + rest.replace("/", "\\"),
        f"{m.group(1).upper()}:/{rest}",
        f"/{m.group(1)}/{rest}",
    ]


def site_values(
    cfg: dict, workspace: pathlib.Path | None = None, home: str | None = None
) -> set[str]:
    """What the deny list's seeding (`--seed-deny`) adds to a live configuration's values (`_from_config`), which do
    not hold these forms: the monitor's folders (its toolkit, rocKE checkout, working folder and dashboards' folder),
    with their Windows forms on a Windows drive, where the Claude workspace's folder (`workspace`) and the Windows home
    folder count too; its secrets' files, with the home folder spelt out; this machine's home folder (`home`); and the
    team dashboard's address. A path written from the home folder (`~/...`) and the reviewer model's name lead to
    nobody: they are not among them."""
    out = []
    p = cfg.get("paths") or {}
    root = (p.get("root") or p.get("working_dir") or "").rstrip("/")
    home = home or os.path.expanduser("~")
    dash_dir = (
        str(pathlib.PurePosixPath(p["dashboard"]).parent) if p.get("dashboard") else ""
    )
    for folder in (
        root and f"{root}/monitor/rk",
        root and f"{root}/monitor",
        p.get("repo"),
        root,
        dash_dir,
    ):
        if folder:
            out += [folder, *_win(folder)]
    pub = (cfg.get("publish") or {}).get("html") or {}
    for secret_file in (
        (cfg.get("admin_dashboard") or {}).get("access_token_file"),
        (cfg.get("teams") or {}).get("webhook_file"),
        pub.get("key"),
        p.get("secrets_dir"),
    ):
        if secret_file:
            out.append(os.path.expanduser(secret_file))
    if dash_dir.startswith("/mnt/"):
        for folder in (str(workspace) if workspace else "", dash_dir.rsplit("/", 1)[0]):
            if folder:
                out += [folder, *_win(folder)]
    out.append(home)
    if pub.get("url"):
        out += [pub["url"].rstrip("/") + "/", pub["url"].rstrip("/")]
    return {v for v in out if len(v) >= 2 and not v.startswith("~")}


def _from_state(st: dict) -> dict:
    """Label -> the values a live state gives: the GitHub logins it knows (L_LOGINS) and its PR numbers (L_PRNUMS)."""
    return {L_LOGINS: state_logins(st), L_PRNUMS: state_pr_numbers(st)}


def _from_workspace(ws: pathlib.Path) -> dict:
    """Label -> the values a Claude workspace gives: its JIRA and wiki settings (L_WORKSPACE) and the people its
    rocke-monitor skill names (L_PEOPLE)."""
    return {L_WORKSPACE: _workspace_values(ws), L_PEOPLE: _names(_skill_text(ws))}


def _from_deny(path) -> dict:
    """Label -> the values the deny list gives: a `pr:<n>` line's number (L_PRNUMS), a `domain:<name>` line's domain
    (DOMAINS) and every other value (L_DENY)."""
    v = {L_DENY: set(), L_PRNUMS: set(), DOMAINS: set()}
    for x in deny_values(path):
        m = re.fullmatch(r"pr:(\d{4,6})", x)
        if m:
            v[L_PRNUMS].add(m.group(1))
        elif x.startswith(DOMAIN):
            v[DOMAINS].add(x[len(DOMAIN) :].strip().lower())
        else:
            v[L_DENY].add(x)
    return v


def _json_object(path) -> dict | None:
    """The JSON object a file holds; None when it cannot be read as one (see `_text`; not JSON; not an object)."""
    text = _text(path)
    try:
        data = json.loads(text) if text is not None else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# The check's command line after the interpreter (`main`): python -m imports the rkmon_lib of the working folder.
COMMAND = ("-B", "-m", "rkmon_lib.privacy")
COMMAND_LINE = ("python3", *COMMAND)  # the check's command line as you type it
# A fix's stand-ins for the paths only you know: the deny list, a file (private_files.FILE) and a folder.
DENY_LIST = "DENY_LIST"
FILE = private_files.FILE
FOLDER = "FOLDER"
# The check run again in the package folder, as that folder's own code (`command`), the way a report's fix shows it:
# with a stand-in for each path, as a report shows none.
AGAIN = (*COMMAND_LINE, ".", "--deny", DENY_LIST)
# The fix of a deny list that gives no value: seed it from a monitor's files (`main`), or write the values in it.
SEED_FIX = checks.Steps(
    (
        "seed it from a monitor's files:",
        checks.Command(
            (
                *COMMAND_LINE,
                "--config",
                "CONFIG",
                "--state",
                "STATE",
                "--seed-deny",
                FILE,
            )
        ),
        f", {FILE} being the deny list, CONFIG and STATE the monitor's config.json and state.json; or write the "
        "values to look for in it, one a line",
    )
)


def _again(typed, flag: str, given, stand_in: str) -> checks.Command:
    """A source row's fix: the command that names another source with FLAG, STAND_IN standing for it: the command as
    you ran it (TYPED, its program first) with STAND_IN in the place of GIVEN, the path you named there, or FLAG and
    STAND_IN after it (private_files.again). Without TYPED, it is the check run again (`AGAIN`), where the deny list's
    own row puts STAND_IN in DENY_LIST's place.
    """
    if not typed:
        typed, given = AGAIN, DENY_LIST if flag == "--deny" else ""
    return private_files.again(typed, str(given or ""), flag)(stand_in)


# The sources of the values, each named by its option: a row each in the report when it is named.
S_DENY = "the deny list (--deny)"
S_CONFIG = "the live configuration (--config)"
S_STATE = "the live state (--state)"
S_WORKSPACE = "the Claude workspace (--workspace)"
S_GIT = "the git folders (--git)"
S_FILES = "the files"
NOT_READ = "NOT CHECKED: it cannot be read"
NO_VALUE = "NOT CHECKED: it gives no value"


def sources(
    config_path=None,
    state_path=None,
    workspace=None,
    git_dirs=(),
    environ=os.environ,
    deny=None,
    typed=(),
) -> tuple[dict, list]:
    """(label -> the identifying values to look for, a row (checks.Check) for each source named). The values are read
    now: this machine's (`_from_machine`, always), and each source named: the live configuration, the live state, the
    Claude workspace, the git folders and the deny list (`deny`, None when it is not named). A source named that cannot
    be read, or that gives no value, is NOT CHECKED: a FAIL row, which fails the check (`run`). Its fix names another
    source (`_again`): in the command you ran, `typed` (its program first), or, without it, in the check run again with
    a stand-in for each path. A row shows no value, and no path but those in `typed`; no value is written anywhere.
    """
    v = {label: set() for label in VALUE_LABELS}
    v[DOMAINS] = set()
    rows = []

    def add(found: dict) -> int:
        n = 0
        for label, vals in found.items():
            vals = {x for x in vals if x}
            v[label] |= vals
            n += len(vals)
        return n

    def fix(flag: str, given, stand_in: str, what: str) -> checks.Steps:
        """The fix that names another source with FLAG (`_again`), then what STAND_IN stands for."""
        return checks.Steps(
            (_again(typed, flag, given, stand_in), f", {stand_in} being {what}")
        )

    def row(
        name: str, found: dict | None, fix_read: checks.Steps, fix_none: checks.Steps
    ) -> None:
        if found is None:
            rows.append(checks.Check(name, checks.Status.FAIL, NOT_READ, fix_read))
            return
        n = add(found)
        rows.append(
            checks.Check(name, checks.Status.PASS, f"{n} value(s)")
            if n
            else checks.Check(name, checks.Status.FAIL, NO_VALUE, fix_none)
        )

    add(_from_machine(environ))
    if config_path:
        cfg = _json_object(config_path)
        row(
            S_CONFIG,
            None if cfg is None else _from_config(cfg),
            fix(
                "--config",
                config_path,
                FILE,
                "the monitor's config.json (a JSON object), or leave --config out",
            ),
            fix(
                "--config",
                config_path,
                FILE,
                "the monitor's own config.json, the one that names its operator",
            ),
        )
    if state_path:
        st = _json_object(state_path)
        row(
            S_STATE,
            None if st is None else _from_state(st),
            fix(
                "--state",
                state_path,
                FILE,
                "the monitor's state.json (a JSON object), or leave --state out",
            ),
            fix(
                "--state",
                state_path,
                FILE,
                "the monitor's own state.json, the one that knows its PRs",
            ),
        )
    if workspace:
        ws = pathlib.Path(workspace)
        row(
            S_WORKSPACE,
            _from_workspace(ws) if ws.is_dir() else None,
            fix(
                "--workspace",
                workspace,
                FOLDER,
                "the Claude workspace's folder (the one with .claude/ in it), or leave --workspace out",
            ),
            fix(
                "--workspace",
                workspace,
                FOLDER,
                "the Claude workspace whose .claude/project_config.md has the JIRA and wiki settings",
            ),
        )
    if git_dirs:
        missing = [str(d) for d in git_dirs if not pathlib.Path(d).is_dir()]
        row(
            S_GIT,
            None if missing else _from_git([pathlib.Path(d) for d in git_dirs]),
            fix(
                "--git",
                missing[0] if missing else "",
                FOLDER,
                "a folder that exists, or leave that --git out",
            ),
            fix(
                "--git", "", FOLDER, "a git folder whose user.name or user.email is set"
            ),
        )
    if deny is not None:
        row(
            S_DENY,
            _from_deny(deny) if _text(deny) is not None else None,
            fix(
                "--deny",
                deny,
                FILE,
                "a deny list that can be read: one entry a line, kept outside the package",
            ),
            SEED_FIX,
        )
    return {k: {x for x in vals if x} for k, vals in v.items()}, rows


def company_domains(values: dict) -> set[str]:
    """The registrable domains of the live hosts and emails, and the deny list's `domain:` lines: every host under them
    is internal."""
    out = set(values.get(DOMAINS) or ())
    for x in values.get(L_CFG, set()) | values.get(L_GIT, set()):
        host = x.split("@", 1)[1] if "@" in x else x
        if "." in host and re.fullmatch(r"[\w.-]+", host):
            out.add(".".join(host.lower().split(".")[-2:]))
    return out


def _value_rx(label: str, value: str) -> re.Pattern:
    if label == L_PATHS or "/" in value or "\\" in value:
        return re.compile(re.escape(value), re.I)
    if label in (
        L_ME,
        L_GIT,
        L_MACHINE,
        L_DENY,
    ):  # never part of a longer word, even in a slug (C--Users-<name>-...)
        return re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.I
        )
    return re.compile(
        r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])", re.I
    )


def _public_url(host: str, path: str) -> bool:
    """Whether a URL's HOST, with its PATH, names nothing private: this machine, a reserved example name or
    documentation address (LOCAL_HOSTS), a public site (PUBLIC_SITES), or GitHub's own organisation or a public
    repository there (GITHUB_ORG, GITHUB_PUBLIC). A host built at run time (a format field, a shell variable, a
    pattern: TEMPLATED) names none. A host with a <placeholder> in it names nothing when its fixed part names nothing:
    with its placeholders and its port taken out, what remains is nothing or a top-level domain alone (`<host>`,
    `<host>:<port>`, `<company>.com`), or its placeholder is the site's label in an internal product's domain
    (`<site>.atlassian.net`, INTERNAL_PRODUCTS). Any other is the host its fixed part names, and is checked as that
    host: `<site>.example.com` is example.com, `<site>.corp-example.net` is corp-example.net.
    """
    h = host.lower().split("@")[-1]
    if HOST_PLACEHOLDER.search(h):
        site = PORT.sub("", h)
        fixed = HOST_PLACEHOLDER.sub("", site)
        if re.fullmatch(r"(?:\.[a-z]+)?", fixed):
            return True
        label = HOST_PLACEHOLDER.match(site)
        if label and site[label.end() :] in {"." + d for d in INTERNAL_PRODUCTS}:
            return True
        h = fixed.strip(".")
    if LOCAL_HOSTS.match(h) or LOCAL_HOSTS.match(h.rstrip(":")):
        return True
    if TEMPLATED.search(h):  # the host comes from a variable at run time
        return True
    h = h.split(":")[0]
    if h in ("github.com", "www.github.com", "raw.githubusercontent.com"):
        first = (path.split("/")[1:2] or [""])[0]
        two = "/".join(path.split("/")[1:3])
        return (
            first == GITHUB_ORG
            or two in GITHUB_PUBLIC
            or path in ("", "/")
            or bool(TEMPLATED.search(first))
        )
    return h in PUBLIC_SITES


def _scan_text(
    text: str, values: dict, domains: set[str], code: bool = False
) -> list[tuple[int, str]]:
    """(line number, label) for every match in a text. `code`: a Python file, where the live state's PR numbers are
    looked for as bare numbers too."""
    found = []
    compiled = [
        (label, _value_rx(label, val))
        for label, vals in values.items()
        if label not in (L_PRNUMS, DOMAINS)
        for val in sorted(vals)
        if len(val) >= 2
    ]
    prnums = pr_numbers_rx(values.get(L_PRNUMS) or ()) if code else None
    for i, line in enumerate(text.splitlines(), 1):
        for label, rx in compiled:
            if rx.search(line):
                found.append((i, label))
        if prnums is not None and prnums.search(line):
            found.append((i, L_PRNUMS))
        for m in DATED.finditer(line):
            if not _dated_ok(line, m):
                found.append((i, P_DATED))
        if ATTRIB.search(line):
            found.append((i, P_ATTRIB))
        if INCIDENT.search(line):
            found.append((i, P_INCIDENT))
        for m in EMAIL.finditer(line):
            if not LOCAL_HOSTS.match(m.group(1)):
                found.append((i, P_EMAIL))
        for m in HOME.finditer(line):
            if m.group(1).lower() not in PLACEHOLDER_USERS and not m.group(
                1
            ).startswith(("<", "$", "{")):
                found.append((i, P_HOME))
        if SYNC.search(line):
            found.append((i, P_SYNC))
        if TOKENS.search(line):
            found.append((i, P_TOKEN))
        for m in TICKET.finditer(OWN_IDS.sub("", line)):
            if m.group(1) not in SAFE_KEYS:
                found.append((i, P_TICKET))
        for m in URL.finditer(line):
            if not _public_url(m.group(1), m.group(2)):
                found.append((i, P_URL))
        for m in HOSTLIKE.finditer(line):
            h = m.group(1).lower()
            internal = any(
                h == d or h.endswith("." + d) for d in domains | set(INTERNAL_PRODUCTS)
            )
            if internal and h not in PUBLIC_SITES:
                found.append((i, P_HOST))
        if WIKI.search(line):
            found.append((i, P_WIKI))
        if ARTIFACTS.search(line):
            found.append((i, P_ARTIFACTS))
        for rx in (PR_HASH, PR_WORD, PR_URL):
            for m in rx.finditer(line):
                if not invented(m.group(1)):
                    found.append((i, P_PR))
        for m in HANDLE.finditer(line):
            if not _handle_ok(line, m):
                found.append((i, P_HANDLE))
        if PERF.search(line):
            found.append((i, P_PERF))
        if GPU.search(line):
            found.append((i, P_GPU))
        if DOC.search(line):
            found.append((i, P_DOC))
    return found


def _files_under(path) -> list[pathlib.Path]:
    """The files the check reads for a path named: the path itself when it is a file; every file under it, in order,
    when it is a folder (its git data and bytecode caches aside); none when it does not exist.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        return sorted(
            x
            for x in p.rglob("*")
            if x.is_file() and "__pycache__" not in x.parts and ".git" not in x.parts
        )
    return [p] if p.is_file() else []


# A string in a JSON text, its escapes included.
JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def _decoded(m: re.Match) -> str:
    """A JSON string, decoded, in its quotes: its escapes read (`\\/` is `/`); as it is when it cannot be decoded."""
    try:
        return '"' + json.loads(m.group(0)) + '"'
    except ValueError:
        return m.group(0)


def _with_decoded(
    found: list, text: str, values: dict, domains: set, code: bool
) -> list:
    """FOUND, the (line, label) of each match in TEXT, a .json file's, with those its strings show decoded: a URL with
    its slashes escaped (`https:\\/\\/<host>\\/page`) is found as the plain one is, on its line, a string over lines
    included. A line counts a match as often as the reading, as it is or decoded, that shows it more.
    """
    # (the file's line, a line of its text decoded), for each line its decoding changes
    lines = []
    for i, line in enumerate(text.splitlines(), 1):
        decoded = JSON_STRING.sub(_decoded, line)
        if decoded != line:
            lines += [(i, part) for part in decoded.splitlines()]
    if not lines:
        return found
    shown = collections.Counter(
        (lines[j - 1][0], label)
        for j, label in _scan_text(
            "\n".join(part for _i, part in lines), values, domains, code
        )
    )
    have = collections.Counter(found)
    more = [m for m, n in shown.items() for _ in range(n - have[m])]
    return sorted(found + more, key=lambda m: m[0])


def _scan_files(files: list, values: dict) -> list[tuple[str, int, str]]:
    """(file, line, label) for every match in each file, a .json file's strings read decoded too (`_with_decoded`). A
    settings file of either kind, deploy or build (settings.header_line: its header, whatever the file's name), is
    refused as a whole.
    """
    from .settings import header_line

    domains = company_domains(values)
    out = []
    for f in files:
        text = f.read_bytes().decode("utf-8", errors="replace")
        code = is_code(f, text)
        found = _scan_text(text + "\n" + f.name, values, domains, code)
        if f.suffix.lower() == ".json":
            found = _with_decoded(found, text, values, domains, code)
        out += [(str(f), i, label) for i, label in found]
        line = header_line(text)
        if line:
            out.append((str(f), line, P_SETTINGS))
    return out


# --- a package folder's parts and settings templates ------------------------------------------------------------------

S_DOCS = "the docs"
S_SCHEMAS = "the schemas"
S_SKILLS = "the skills"
S_TESTS = "the tests"
S_TEMPLATES = "the settings templates"
# The folders of a package folder that hold its skills: the monitor's (agentic/skills/) and the ones Claude Code reads
# in the package folder itself (.claude/skills/).
SKILL_FOLDERS = (("agentic", "skills"), (".claude", "skills"))
# The parts of a package folder the check reads besides its templates (each a test on a file's path inside the folder):
# a row each, NOT CHECKED when it has no file.
PARTS = (
    (
        S_DOCS,
        lambda rel: rel.suffix == ".md"
        and rel.parts[0] != "tests"
        and rel.parts[:2] not in SKILL_FOLDERS,
    ),
    (S_SCHEMAS, lambda rel: rel.parts[0] == "schema"),
    (S_SKILLS, lambda rel: rel.parts[:2] in SKILL_FOLDERS),
    (S_TESTS, lambda rel: rel.parts[0] == "tests"),
)
P_ABSOLUTE = (
    "absolute paths (a / at the start of a path, a drive's letter, a network share)"
)
P_PRIVATE = "private keys whose value is not blank, <...> placeholders only or the key's default"
TEMPLATE_LABELS = (P_ABSOLUTE, P_PRIVATE)
# An absolute path: a / at the start of a line or after a space, a quote, =, a bracket, a comma, a colon, a semicolon
# or a | (so not a URL's //, nor the / inside a relative path), a drive's letter (C:\, C:/), or a network share (\\h\).
ABSOLUTE = re.compile(
    r"(?:^|(?<=[\s\"'=(\[{,:;|]))/[A-Za-z0-9._~-]|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+\\"
)


def _part_rows(package_dir, files: list) -> list:
    """A row for each part of the package folder (PARTS): how many of `files` are in it, NOT CHECKED when none is."""
    pkg = pathlib.Path(package_dir)
    rels = []
    for f in files:
        try:
            rels.append(pathlib.Path(f).relative_to(pkg))
        except ValueError:
            continue
    out = []
    for name, inside in PARTS:
        n = sum(1 for r in rels if r.parts and inside(r))
        out.append(
            checks.Check(name, checks.Status.PASS, f"{n} file(s) read")
            if n
            else checks.Check(
                name,
                checks.Status.FAIL,
                "NOT CHECKED: the package folder has no file of it",
                checks.Steps(
                    (
                        "in the whole package folder (its schema/, tests/ and agentic/skills/ hold their files),",
                        checks.Command(AGAIN),
                        "checks every part",
                    )
                ),
            )
        )
    return out


def _template_may_show(node: dict, value) -> bool:
    """Whether a template may show `value` for a private key whose schema node is `node`: a blank (an empty text, list
    or mapping, or null), the key's own default, or <...> placeholders only: each text holds a placeholder
    (settings_file.PLACEHOLDER) and no letter or digit outside its placeholders, as "<org>/<repo>" does.
    """
    from .settings_file import PLACEHOLDER

    if value is None or value in ("", [], {}):
        return True
    if "default" in node and value == node["default"]:
        return True
    items = value if isinstance(value, list) else [value]
    return all(
        isinstance(x, str)
        and bool(PLACEHOLDER.search(x))
        and not any(c.isalnum() for c in PLACEHOLDER.sub("", x))
        for x in items
    )


def _private_keys(file_type) -> list:
    """(pattern, nodes) for each key that the schema of `file_type` (a settings_file.FileType) classes private
    (x-class "private"): the pattern matches the key's dotted name, any name in place of a map entry's (`*`), and
    the nodes are the key's own, one for each form a oneOf gives it."""
    from .settings_file import schema_keys

    nodes = {}
    for path, node, _required in schema_keys(file_type.load()):
        if node.get("x-class") == "private":
            nodes.setdefault(path, []).append(node)
    return [
        (re.compile(re.escape(path).replace(r"\*", r"[^.]+")), found)
        for path, found in nodes.items()
    ]


# A key's line in a settings template: its indent, its key (a name, or one <placeholder> that names a map's entry),
# and what follows the colon.
KEY_LINE = re.compile(r"^( *)([A-Za-z0-9_][A-Za-z0-9_.-]*|<[^<>\s]+>):(?:\s+(.*))?$")
# A comment line: its indent and its text after the "# ".
COMMENT_LINE = re.compile(r"^( *)# (.*)$")


def _key_lines(text: str) -> list[tuple[int, str, str]]:
    """(line number, dotted key, what follows its colon) for each key with a value on its line in a settings
    template's TEXT, those of the blocks it shows commented out included. settings_file writes a section its chooser
    does not choose, and a map's example entries, that way: "# " put in at the block's column of each of its lines.
    Such a block is a comment line that opens a mapping (`# KEY:`) and the comment lines after it at the same column,
    each indented deeper behind its "# ": each line reads as it was before its "# ", under the keys above the block.
    A map entry named by a placeholder (`<name>`) is a key like any other."""
    out, above, block = [], [], None
    for n, line in enumerate(text.splitlines(), 1):
        comment = COMMENT_LINE.match(line)
        if comment:
            pad, rest = comment.groups()
            inner = len(rest) - len(rest.lstrip(" "))
            if not (block and block[0] == len(pad) and inner > block[1]):
                opens = KEY_LINE.match(rest)
                if not opens or opens.group(3):
                    block = None
                    continue  # a comment
                where = len(pad) + inner
                block = (len(pad), inner, [(i, k) for i, k in above if i < where])
            line, keys = pad + rest, block[2]
        else:
            block, keys = None, above
        found = KEY_LINE.match(line)
        if not found:
            continue
        indent, key, value = len(found.group(1)), found.group(2), found.group(3)
        while keys and keys[-1][0] >= indent:
            keys.pop()
        if value:
            out.append((n, ".".join([k for _i, k in keys] + [key]), value))
        keys.append((indent, key))
    return out


def _line_value(text: str):
    """The value a key's line gives after its colon (TEXT), as a settings file reads it (answers.parse); TEXT itself
    when it cannot be read."""
    from . import answers

    try:
        return answers.parse("v: " + text)["v"]
    except answers.AnswersError:
        return text


def _templates_printed(broken: list) -> checks.Steps:
    """The fix of the templates that cannot be rendered or read back (BROKEN, settings_templates.Template each): the
    command that prints each one (`Template.command`), which shows its error."""
    many = len(broken) > 1
    parts = []
    for t in broken:
        parts += [","] if parts else []
        parts += [t.command(), f"prints {t.label}" if many else "prints it"]
    return checks.Steps((*parts, f": fix the error {'each' if many else 'it'} shows"))


def _scan_templates(package_dir, values: dict) -> tuple[list, checks.Check]:
    """(each finding in the package's settings templates, as (template LABEL, line, label); the templates' row, which
    counts the private keys checked in each). Each template the package gives you (settings_templates.templates(), its
    text as docker/rkmon prints it) is read as a file is (`_scan_text`), every line of it, and for absolute paths
    (P_ABSOLUTE) and private values (P_PRIVATE: each key its schema classes private, `_private_keys`, shows a value a
    template may show, `_template_may_show`, on its line, `_key_lines`, a block the template shows commented out
    included). A template in which no key is private is NOT CHECKED for private values. The code that runs the check
    renders the templates, so they are NOT CHECKED over a package folder that is not that code's (paths.PACKAGE;
    `command` runs the check as a folder's own code). Each is rendered and read back on its own: one that cannot be is
    NOT CHECKED, and the row names it with the command that prints it (`Template.command`), which shows the error.
    """
    from . import answers, file_types, settings_templates

    pkg = pathlib.Path(package_dir)
    if pkg.resolve() != paths.PACKAGE:
        return [], checks.Check(
            S_TEMPLATES,
            checks.Status.FAIL,
            "NOT CHECKED: the check runs as another folder's code",
            checks.Steps(
                (
                    "run the check as the package folder's own code, in that folder:",
                    checks.Command(AGAIN),
                )
            ),
        )
    # Any error means the templates cannot be listed, so none is checked.
    try:
        listed = settings_templates.templates()
    except Exception as exc:
        return [], checks.Check(
            S_TEMPLATES,
            checks.Status.FAIL,
            f"NOT CHECKED: they cannot be listed ({type(exc).__name__})",
            _templates_printed(
                [
                    settings_templates.Template(file_types.BUILD),
                    settings_templates.Template(file_types.DEPLOY),
                ]
            ),
        )
    # Each is rendered and read back on its own: any error means that one cannot be read, so it is not checked.
    texts, private, broken = {}, {}, {}
    for t in listed:
        try:
            text = t.text()
            answers.parse(text)
            if t.file_type not in private:
                private[t.file_type] = _private_keys(t.file_type)
        except Exception as exc:
            broken[t] = type(exc).__name__
            continue
        texts[t] = text
    domains = company_domains(values)
    out, checked = [], {}
    for t, text in texts.items():
        where = f"template {t.label}"
        out += [(where, i, label) for i, label in _scan_text(text, values, domains)]
        out += [
            (where, i, P_ABSOLUTE)
            for i, line in enumerate(text.split("\n"), 1)
            if ABSOLUTE.search(line)
        ]
        keys = []
        for i, k, value in _key_lines(text):
            nodes = next((n for p, n in private[t.file_type] if p.fullmatch(k)), None)
            if nodes is not None:
                keys.append((i, nodes, _line_value(value)))
        checked[t.label] = len(keys)
        out += [
            (where, i, P_PRIVATE)
            for i, nodes, value in keys
            if not any(_template_may_show(n, value) for n in nodes)
        ]
    counts = ", ".join(f"{label} {n}" for label, n in checked.items())
    rendered = f"{len(texts)} rendered (private keys checked: {counts})"
    if broken:
        return out, checks.Check(
            S_TEMPLATES,
            checks.Status.FAIL,
            f"NOT CHECKED: template(s) {', '.join(t.label for t in broken)} cannot be rendered and read back "
            f"({', '.join(sorted(set(broken.values())))})"
            + (f"; {rendered}" if texts else ""),
            _templates_printed(list(broken)),
        )
    none = [label for label, n in checked.items() if not n]
    if none:
        return out, checks.Check(
            S_TEMPLATES,
            checks.Status.FAIL,
            f"NOT CHECKED: no private key in template(s) {', '.join(none)} (private keys checked: {counts})",
            checks.Steps(
                (
                    f"in its schema ({_schema_folder()}/), give each key whose value can name a person, a site or a "
                    'machine "x-class": "private", then',
                    checks.Command(AGAIN),
                )
            ),
        )
    return out, checks.Check(S_TEMPLATES, checks.Status.PASS, rendered)


def _schema_folder() -> str:
    """The folder of the schemas the templates are made from (file_types.SCHEMAS), in the package folder."""
    from . import file_types

    return file_types.SCHEMAS.relative_to(file_types.PACKAGE).as_posix()


# The check run again, naming each match's place (never the value).
WHERE = checks.Command((*AGAIN, "--where"))


def _where(then: str) -> checks.Steps:
    """A match's fix: the check again with --where, in the package folder, then THEN, what to do at each match."""
    return checks.Steps(
        (
            "in the package folder,",
            WHERE,
            f"gives each match's file or template, line and label (never the value): {then}",
        )
    )


WHERE_FIX = _where("remove the value there, or write a placeholder in its place")
SETTINGS_FIX = _where(
    "move that file out of the package, as a settings file is one machine's"
)


def _template_fix() -> checks.Steps:
    """A template's match's fix: the key's value put right in the schema the template is made from, where the
    profiles' values are too (x-profiles)."""
    return _where(
        f"give that key a blank value, <...> placeholders or its default in its schema ({_schema_folder()}/), or in "
        "its profile there (x-profiles), and write no absolute path"
    )


def _report(
    in_files: list,
    in_templates: list,
    values: dict,
    n_files: int,
    rows=(),
    rendered: bool = False,
) -> tuple[str, bool]:
    """The check's report and whether it passed, from what `run` found in the files (`in_files`) and in the templates
    (`in_templates`), each as (file or template, line, label). Its rows (checks.Check): the rows given (what was read:
    the sources, the files, a package folder's parts and templates), one for each kind of identifying value that has
    values (a kind with none is not reported as checked), one for each pattern, and, when templates were `rendered`, one
    for each of the templates' own checks (TEMPLATE_LABELS); labels and counts only, never a value or a path. It passed
    when no row is a FAIL: a match, or something NOT CHECKED. Its result line counts the matches, and the files and
    the templates they are in, apart.
    """
    findings = in_files + in_templates
    by = {}
    for _f, _i, label in findings:
        by[label] = by.get(label, 0) + 1

    def status(n: int) -> checks.Status:
        return checks.Status.FAIL if n else checks.Status.PASS

    parts = (
        ("What was read", list(rows)),
        (
            "Identifying values",
            [
                checks.Check(
                    label,
                    status(by.get(label, 0)),
                    f"{len(values.get(label, ()))} value(s) checked, {by.get(label, 0)} match(es)",
                    WHERE_FIX,
                )
                for label in VALUE_LABELS
                if values.get(label) or by.get(label)
            ],
        ),
        (
            "Patterns",
            [
                checks.Check(
                    label,
                    status(by.get(label, 0)),
                    f"{by.get(label, 0)} match(es)",
                    SETTINGS_FIX if label == P_SETTINGS else WHERE_FIX,
                )
                for label in PATTERN_LABELS
            ],
        ),
        (
            "Settings templates",
            [
                checks.Check(
                    label,
                    status(by.get(label, 0)),
                    f"{by.get(label, 0)} match(es)",
                    _template_fix(),
                )
                for label in TEMPLATE_LABELS
                if rendered or by.get(label)
            ],
        ),
    )
    lines = [
        f"Privacy check, {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}: {n_files} file(s) scanned.",
        "Identifying values are read at check time and never written here: only labels and counts.",
    ]
    for heading, part in parts:
        if part:
            lines += ["", heading, checks.render(part)]
    failed = [c for _h, part in parts for c in part if c.status == checks.Status.FAIL]
    not_checked = sum(c.status == checks.Status.FAIL for c in rows)
    ok = not failed
    lines += [
        "",
        f"Result: {'PASS' if ok else 'FAIL'} ({len(findings)} match(es) in "
        f"{len({f for f, _i, _l in in_files})} file(s) and "
        f"{len({t for t, _i, _l in in_templates})} template(s)"
        + (f"; {not_checked} NOT CHECKED" if not_checked else "")
        + ")",
    ]
    return "\n".join(lines) + "\n", ok


def _files_row(named: list) -> checks.Check:
    """The files' row, from each path named with the files the check read for it (`named`: (path, files)): NOT CHECKED,
    a FAIL, when no path is named, or when a path named does not exist or holds no file, each path counted on its
    own."""
    missing = sum(not pathlib.Path(p).exists() for p, _files in named)
    empty = sum(pathlib.Path(p).exists() and not files for p, files in named)
    if named and not missing and not empty:
        n = sum(len(files) for _p, files in named)
        return checks.Check(S_FILES, checks.Status.PASS, f"{n} file(s) read")
    if not named:
        why = "no file or folder is named"
    elif missing and empty:
        why = f"of the {len(named)} path(s) named, {missing} do not exist and {empty} hold no file"
    elif missing:
        why = f"{missing} of the {len(named)} path(s) named do not exist"
    else:
        why = f"{empty} of the {len(named)} path(s) named hold no file"
    return checks.Check(
        S_FILES,
        checks.Status.FAIL,
        f"NOT CHECKED: {why}",
        checks.Steps(
            (
                "in the package folder,",
                checks.Command(AGAIN),
                "checks all of it; or name each file and folder to check as it is, one with a file in it",
            )
        ),
    )


def run(
    paths_: list,
    deny,
    config=None,
    state=None,
    workspace=None,
    git_dirs=(),
    environ=os.environ,
) -> tuple[str, bool, list]:
    """The privacy check over every file under paths_, with the deny list it requires: (its report, whether it passed,
    the findings as (file, line, label), a template's as (template NAME, line, label)). It reads the values
    (`sources`) and scans the files (`_scan_files`); for each path that is a package folder (`paths.is_package`), it
    counts the files of its parts (`_part_rows`) and renders and checks its settings templates (`_scan_templates`);
    then it reports (`_report`). It passes only when every source named gave values, every path named exists and holds
    a file, every part of a package folder has files, its templates were rendered, and nothing was found.
    """
    values, rows = sources(config, state, workspace, git_dirs, environ, deny=deny or "")
    named = [(p, _files_under(p)) for p in paths_]
    files = [f for _p, under in named for f in under]
    in_files = _scan_files(files, values)
    rows.append(_files_row(named))
    in_templates = []
    rendered = False
    for p in paths_:
        if paths.is_package(p):
            rows += _part_rows(p, files)
            found, row = _scan_templates(p, values)
            in_templates += found
            rows.append(row)
            rendered = rendered or row.status == checks.Status.PASS
    text, ok = _report(in_files, in_templates, values, len(files), rows, rendered)
    return text, ok, in_files + in_templates


def command(*args) -> list[str]:
    """The check's command line with `args`, its options and then the paths, for subprocess: run it with the package
    folder to check as the working folder, and the check runs as that folder's own code, the one way its settings
    templates are checked (`_scan_templates`)."""
    return [sys.executable, *COMMAND, *(str(a) for a in args)]


def parser() -> argparse.ArgumentParser:
    """The check's command line (`main`): its options, each in one form only, as `--deny FILE` or `--deny=FILE` (an
    option is never abbreviated), and the paths. The sync reads the check's arguments with it too.
    """
    ap = argparse.ArgumentParser(
        prog=" ".join(COMMAND_LINE),
        description="The privacy check (labels and counts only).",
        allow_abbrev=False,
    )
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--config", help="the live config.json (identifying values)")
    ap.add_argument("--state", help="the live state.json (GitHub logins, PR numbers)")
    ap.add_argument(
        "--workspace",
        help="the live Claude workspace (its project settings and rocke-monitor skill)",
    )
    ap.add_argument(
        "--git",
        action="append",
        default=[],
        help="a git folder whose user.name and user.email count",
    )
    ap.add_argument(
        "--deny",
        metavar="FILE",
        help="the deny list (one entry per line): required, except with --seed-deny",
    )
    ap.add_argument(
        "--seed-deny",
        metavar="FILE",
        help="add the live values to this deny list and stop (no check)",
    )
    ap.add_argument("--report", help="write the report here too")
    ap.add_argument(
        "--where",
        action="store_true",
        help="print file:line and label of each match (never the value)",
    )
    return ap


def main(argv=None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.seed_deny:
        if a.paths:
            ap.error(
                "--seed-deny writes the deny list and checks nothing: name no file or folder with it"
            )
        values, rows = sources(
            a.config,
            a.state,
            a.workspace,
            a.git,
            typed=(*COMMAND_LINE, *(sys.argv[1:] if argv is None else argv)),
        )
        if any(r.status == checks.Status.FAIL for r in rows):
            print(checks.render(rows, "problems"))
            print("nothing was added to the deny list")
            return 1
        site = site_values(
            _json_object(a.config) or {},
            pathlib.Path(a.workspace) if a.workspace else None,
        )
        n = seed_deny(a.seed_deny, values, site)
        print(f"{n} value(s) added to the deny list {a.seed_deny} (values not shown)")
        return 0
    if not a.deny:
        ap.error("--deny FILE is required: the deny list of the values to look for")
    if not a.paths:
        ap.error("name the files or folders to scan")
    text, ok, findings = run(a.paths, a.deny, a.config, a.state, a.workspace, a.git)
    print(text, end="")
    if a.where:
        for f, i, label in findings:
            print(f"{f}:{i}: {label}")
    if a.report:
        pathlib.Path(a.report).write_text(text, encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
