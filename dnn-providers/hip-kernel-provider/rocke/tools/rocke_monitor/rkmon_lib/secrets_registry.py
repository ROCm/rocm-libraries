# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every secret and login the package knows, in one table: what each one is, the shape of its value, who makes it,
whose folder keeps it and where, when it is needed and, for a GitHub login, the permissions its token needs.

- `Secret(name, what, shape, made_by, keeper, needed_when=(), permissions=(), key="", variable="", place="",
  files=(), older=(), prototype_use="")`: one entry, checked as it is made (ValueError):
  - `name`: a secret's name (`NAME`: lowercase letters, digits and _), the one `rkmon secret NAME`,
    `rk secret-set NAME` and an answers file's `secrets:` take; a login's is the one `rkmon login NAME` takes
    (`LOGIN_NAME`: - too).
  - `what`: what it is, in plain words, as `rk logins` shows it.
  - `shape`: what its value looks like (`SHAPES`, each described in `SHAPE_WORDS`): `login`, a tool's own login
    (given as a token: one line with no spaces); `https_url`; `ssh_private_key`, the whole key file; `one_line`.
  - `made_by`: who makes it, in plain words; `GENERATED` for a value made here on request (a random token).
  - `keeper`: whose folder keeps it (`KEEPERS`): the monitor's, or the writer's (the user that makes the outward
    writes: GitHub's, the team chat's, the web host's and JIRA's; rkmon_lib/writer.py).
  - `needed_when`: when it is needed, any one of: `ALWAYS`; a capability's name (rkmon_lib/features.py), while it is
    on; `CAPABILITY=VALUE`, while the capability has that value (features.configured: the auto-merge guard's `act`,
    the team dashboard's `publish` or `serve`). Each by the settings (standby changes no need), and for a capability
    that sends for a feature, only while that feature is live: in prototype it sends nothing. Empty:
    nothing needs it, and it is used when it is set.
  - `permissions`: a GitHub login's permissions, as a fine-grained token names them (on <org>/<repo>, but
    gh.ORGANIZATION_PERMISSIONS on the organization): (permission, access) pairs, the access `read` or `write`. Only a
    login has any. The GitHub logins' are what their calls need (gh.permissions, from the list of every GitHub call,
    gh.CALLS).
  - `key`: the config key that holds its place, which the load derives from its `home` (rkmon_lib/derived.py; "" for
    none).
  - `variable`: the environment variable that names another place for it, before its key ("" for none): the read
    login's, the suite's. Nothing the package runs sets one; the container's write login's folder is its `home`,
    container_layout.WRITE_LOGIN_DIR.
  - `place`: the name of its file in <SECRETS_DIR> (a login kept in a folder: of that folder), when that is not NAME
    ("" for NAME).
  - `files`: the files in a login's folder that hold the login (gh's hosts.yml, the atlassian CLI's credentials). Only
    a login has any; a login with none is kept in a file of its own, as a secret is.
  - `older`: its older names and places, each an `Older(name, keeper=MONITOR, copied=True, file=False)`:
    `name`, the name an older package gave it, which is also its file's (a login's: its folder's) name in KEEPER's
    secrets folder (`home`'s rule); `copied`, True when the one-time move copies it to the entry's own place (the
    container's root phase), False when the move only removes it as it ends (`rkmon standby off`); `file`, True for a
    login's older copy kept in a file rather than in the login's folder (only a login kept in a folder has one). The
    ignore lists, adopt's skip list, the move's end and the login walk's states read them here; the move's copies,
    which root makes at each start (docker/rk_root.py), keep the facts they act on in a table of their own, which
    tests/test_root_copies.py holds equal to this registry's.
  - `prototype_use`: what it is used for while each feature it waits for is in prototype (`waits_for`), in words, with
    `{program}` where the name rkmon was run by goes; "" for nothing. While it is not set, the login walk shows it as a
    WARN with the command that sets it, where it shows another entry that waits as OFF: the atlassian
    login, with which rkmon check reads your JIRA project's required fields while tickets are only prepared.
- `entries(cfg)`: the fixed entries, the logins first, then one for each secret a model binding names (config
  `model_bindings.<name>.api_key_secret`, and the first review pass's `worker.api_key_secret`: `FIRST_PASS_KEY`
  after the one-time move), in name order. A binding's key is needed while review automation is on and one of its
  passes runs on a binding that names the key (roles.pass_bindings); with no pass on such a binding
  nothing needs it, and the login walk does not list it. Root's one-time move copies the key by the same rule
  (docker/rk_root.py, `layer_keys`, which tests/test_root_copies.py holds equal to this one). Its older names are the
  names `binding_key_name` gives its name from, but another entry's name, place or older name; the one-time move
  copies each, as a binding names the key. A value that is not a secret's name adds no entry
  (bindings.problems names it), and a binding that names a fixed secret adds its need, by the same rule, to that
  entry. A binding's key's `what` names the deploy file's keys that name it, as you set them (deploy_tree.name:
  `model_bindings.<name>.api_key_secret_name`).
- `binding_key_name(name)`: the name the one-time move copies a model binding's key named NAME to, which
  ends as every binding's key's name does (`BINDING_KEY_SUFFIX`): the one `OLDER_KEY_NAMES` gives it (the local
  worker's key: the first review pass's); else, for a secret's name without that end, that end in the place of a last
  _key (gw_key: gw_api_key; glm: glm_api_key); else NAME. The translation of layout 1's files gives the same
  (from_layout1.api_key_secret_name).
- `entry(cfg, name)`: the entry named NAME among `entries(cfg)`; None when there is none.
- `set_command(entry)`: the command that sets it, as its words after rkmon's name: `login NAME` for a login,
  `secret NAME --generate` for a value made here, and `secret NAME` for any other secret.
- `typed_set_command(entry, environ=None)`: that command as the operator types it where ENVIRON says the text is
  shown (the process's environment when None): rkmon's name on the host, and in the container the name of the wrapper
  that runs it there (doctor.wrapper_name); a Command (rkmon_lib/checks.py). Each text and fix that names the command
  takes it from here.
- `how_to_set(entry)`: the entry and that command, in words for a list of what a feature lacks: `the secret NAME:
  rkmon secret NAME` (a login's: `the login NAME: rkmon login NAME`).
- `set_fix(entry, environ=None)`: the fix that sets it (rkmon_lib/checks.py): that command as the operator types it
  where ENVIRON says the fix is shown (`typed_set_command`), and, for a value you give with `rkmon login` or
  `rkmon secret`, what to give.
- `claude_env_fix(parts)`: the fix of a thing CLAUDE_ENV's shape has that no setting carries, from its parts
  (claude_env_shape.Cannot's FIX: each a text, or an Edit of one of its variables): each Edit an edit of CLAUDE_ENV,
  to what its words say, or taking the variable out; the one part, else Steps. The translation's stops show it.
- `claude_env_or_login_fix(parts)`: that fix as check 16's rows show it while the claude login is not set, which
  names both ways: the file's edit before the first start, or the command that sets the claude login's key. Root's
  one-time move deals with CLAUDE_ENV once, at the first start (docker/rk_root.py), and check 16 cannot tell whether
  it has (rkmon_lib/confirm.py says which fix its rows show).
- `check_value(entry, value)`: one row (rkmon_lib/checks.py): PASS when the value, blank space around it aside, has
  the entry's shape; else FAIL, its fix `set_fix`'s. No part of the value is ever in the row, nor its length. A value
  that is not text is a TypeError.
- `generate(entry)`: a new value for an entry made here (its made_by `GENERATED`): a random token, URL-safe. A
  ValueError for any other entry.
- `home(cfg, entry, environ=None, keeper="")`: its own place, KEEPER's secrets folder /PLACE (its `place`, else its
  name; KEEPER blank: its own `keeper`), where `rkmon secret` puts the secret it is given and `rkmon login` the login
  it makes; absolute. The monitor's secrets folder is <SECRETS_DIR>. In the container (doctor.deployment_image, in
  ENVIRON) the writer's is its own volume, container_layout.WRITE_SECRETS_DIR, which no other user reads; on a host
  the one user keeps both in <SECRETS_DIR>. paths.PathsError when <SECRETS_DIR> is needed and not known.
- `writers_volume(keeper, environ=None)`: whether KEEPER's secrets are the writer's own volume: the writer's, in the
  container (doctor.deployment_image, in ENVIRON; the process's environment when None), which no other user reads
  or writes.
- `path(cfg, entry, environ=None)`: where this configuration keeps it, a secret's file or a login's folder: the place
  its `variable` names in ENVIRON (the process's environment when None), else the place its `key` holds, which the
  load derives from its `home` (rkmon_lib/derived.py; a place a configuration gives is not kept), else, for an entry
  with no key, its `home`; absolute, `~` expanded. In the container a writer's entry has no key: the monitor's
  processes can edit the configuration, so the writer never reads a file it names. The atlassian CLI keeps its login
  there (atlassian.config_dir). The read login's is its own folder, whichever folder gh reads with meanwhile
  (gh.login_dirs: the write login's in the writer, and the login made before the split while there is no read login).
  paths.PathsError when no secrets folder is known and nothing names another place (for an entry with a key: while
  the key is blank, as the load leaves it then).
- `held(place, files=None)`: whether PLACE, an entry's `path`, holds it: each of FILES in a login's folder, or, FILES
  None or empty, a file that is not blank. Its own code is a part of the program a one-off container runs to read the
  secrets volumes before the first start (confirm.HELD_PROBE), so it reads with the standard library alone.
- `held_files(entry, older=None)`: the FILES `held` takes for the entry's place, or for OLDER's: a login's `files`,
  for one kept in a folder; None for one kept in a file (a secret, a login with no `files`, an older name kept in a
  file).
- `older_names(entry)`: its older names but its own name, each once, in its `older` order: the names `rkmon secret`,
  `rkmon login` and `rk secret-set` refuse, naming the entry (`renamed`).
- `older_places(cfg, entry, environ=None)`: where each older name keeps the entry, in its keeper's secrets folder by
  `home`'s rule: [(its Older, that place)], in its `older` order, each place once, but the entry's own place (its
  `home` and its `path`): an older name that keeps its place (a GitHub login's folder) has none. While the one-time
  move has not copied one that is `copied`, a copy there counts as set (state 7). paths.PathsError when no
  secrets folder is known.
- `renamed(cfg, name)`: the entry of `entries(cfg)` among whose older names NAME is, while no entry is named NAME;
  None for any other name. `older_name_refusal(name, new)`: why NAME, an older name of the entry NEW, is refused, in
  words: `NAME is the older name of the secret NEW: rkmon secret NEW sets it` (a login's: `rkmon login NEW`).
- `name_refusal(cfg, name)`: why NAME is refused where a secret's name is asked for (rk secret-set, `rkmon secret`, an
  answers file's `secrets:`), in words; "" for the name of a secret among `entries(cfg)`. A login's name: `NAME is a
  login: rkmon login NAME sets it`; an older name: `older_name_refusal`'s words; any other: `no secret is named NAME`,
  but a name that does not read as one (`LOGIN_NAME`) is not shown, as it may be a value given in its place. CFG None
  is no configuration known (`rkmon secret` before the first confirm): the fixed entries alone, and a name that reads
  as a secret's (`NAME`) that they do not refuse is not refused, as a model binding may name it.
- `retry(config_file=None)`: the RETRY private_files.write takes for a secret's file, CONFIG_FILE being the
  config.json that places it (by default the monitor's own, config.config_path(), found only when a refusal names it):
  the edit that names another secrets folder (`paths.secrets_dir`), the one place of the secrets config.json gives.
  It never names an entry's own place: the load derives that from the secrets folder, and a place config.json gives
  is not kept (rkmon_lib/derived.py). It names no other file (private_files.folder_only): a link in a secret's place
  is removed, the link alone, and the command run again.
- `for_repo(text, repo)`: TEXT for the repository REPO, <org>/<repo>: REPO in the place of `<org>/<repo>`, its owner
  in the place of `<org>` and its name in the place of `<repo>`. rk read-login's refusal of a token that can write
  says READ_TOKEN so (ghlogins.set_read).
- `binding_key_file(cfg, name)`: where a model binding whose `api_key_secret` is NAME reads its key
  (bindings.binding_by_name; the local worker's, the derived key worker.api_key_file): `path` of the entry named
  NAME; "" when NAME is no entry's name, or no secrets folder is known.
- `needed(cfg)`: the entries that something that is on needs, in the order `entries` gives.
- `condition_words(condition)`: a `needed_when` condition in words: `ALWAYS_WORDS` for `ALWAYS`, a capability's
  title (features.CAPABILITIES), lowered as a sentence takes it, and for `CAPABILITY=VALUE` the title and the value in
  brackets ("team dashboard (publish)"). A KeyError for a capability features.py does not have.
- `needed_by(cfg, entry, among=None)`: what needs the entry among what is on, in words, one for each of its
  `needed_when` that is true (`condition_words`), each once; AMONG, when given, only those of them it names (the
  write login's that push: PUSH_NEEDS). Empty when nothing that is on needs it. A configuration that
  rkmon_lib/features.py cannot read yet has nothing on.
- `waits_for(cfg, entry)`: the features the entry waits for while nothing that is on needs it: each one that sends
  (features.SENDS), on and in prototype, whose going live would make one of its `needed_when` true
  (features.as_live), each once, in `needed_when`'s order. The team chat's webhook waits for alerts while they are on
  in prototype with a kind of alert on. Empty while it is needed, and while going live would not need it either. By
  the settings, as `needed`: standby changes nothing.
- `waiting_words(cfg, entry)`: what it waits for, in words: each feature's mode and what ends it
  (features.prototype_words), joined by "; "; "" when it waits for nothing. A row that says an entry is not needed
  names the mode so.
- `walk_entries(cfg)`: the entries the login walk lists (`rk logins`, `rkmon login`, rkmon confirm's logins): each one
  something may need (a `needed_when`), in the order `entries` gives.
- `secret_patterns()`: the names of the files that hold a secret, as globs with no folder in them: each secret's
  file (a login kept in a file is one), a login's `files`, each older name kept in a file, and the files an
  older package kept a secret in that no fixed entry names (`OLDER_SECRET_FILES`), each with its copies (NAME.*); a
  model binding's key by the end every binding's key's name has (`BINDING_KEY_SUFFIX`: the deploy file's
  api_key_secret_name, `BINDING_KEY_NAME`), and SSH and TLS key files (`KEY_FILES`).
- `secret_folders()`: the names of the folders that may hold a secret: a login's folder (its place) and each older
  name kept in a folder, the folder an older package's gh split kept the write login in (`OTHER_LOGIN_FOLDERS`), and
  the secrets folders (the usual one's name, paths' `.config`, and the container's two, container_layout's /secrets
  and /secrets-write).
- `ignore_patterns()`: the names of the files and folders that may hold a private value, wherever they are, as globs
  with no folder in them but a folder's own trailing /: the ignore lists, .gitignore (as they are) and .dockerignore
  (after `**/`, the trailing / dropped), and the scan for stray private files in the package folder
  (deployment.stray_private_files), which matches them on each file's and each folder's name as git does (a
  pattern with a trailing / on a folder's name only). In order: the secrets' files and the older secret files (as
  `secret_patterns` names them, the key files aside), each with its copies (NAME.*); the package's own private files
  (private_files.PACKAGE_FILES: .env, the override, their .prev copies and records.json), each with its kin
  (private_files.own_names: its copies, and the new file a write of it left); your own (`YOUR_FILES`: the settings
  files and the deny list), each with its copies (a name that ends in * has them already); then `secret_folders` and
  the package's own folder of private files (private_files.RECORDS_FOLDER), each NAME/; then the key files
  (`KEY_FILES`) and your network's root certificates (`CERTIFICATE_FILES`).
- `holds_secret(name)`: whether a file or a folder of this name may hold a secret, wherever it is: a name
  `secret_patterns` matches, or one of `secret_folders`.

It loads with the standard library, rkmon_lib/checks.py and what checks.py loads only: rkmon_lib/wrapping.py, and
rkmon_lib/program.py, which wrapping.py loads. Each function imports the modules it reads.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import re
import urllib.parse
from dataclasses import dataclass

from .checks import DELETE, Check, Command, Described, Edit, Status, Steps

NAME = re.compile(r"[a-z0-9][a-z0-9_]*")  # a secret's name
FOLDER_KEY = "paths.secrets_dir"  # the config key that names the secrets folder
ANOTHER_FOLDER = (
    "another folder of yours, outside the package folder, git and cloud-synced folders"
)
LOGIN_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")  # a login's name
LOGIN, HTTPS_URL, SSH_PRIVATE_KEY, ONE_LINE = (
    "login",
    "https_url",
    "ssh_private_key",
    "one_line",
)
SHAPES = (LOGIN, HTTPS_URL, SSH_PRIVATE_KEY, ONE_LINE)
SHAPE_WORDS = {
    LOGIN: "a token on one line, with no spaces",
    HTTPS_URL: "an https:// address on one line",
    SSH_PRIVATE_KEY: "an SSH private key: the whole key file, from its BEGIN line to its END line",
    ONE_LINE: "one line of text",
}
MONITOR, WRITER = "monitor", "writer"
KEEPERS = (MONITOR, WRITER)
ALWAYS = "always"
ALWAYS_WORDS = "the monitor, always"  # what needs an entry needed ALWAYS (needed_by)
GENERATED = "made here on request: a random token"
ACCESS = ("read", "write")
CONDITION = re.compile(r"[a-z][a-z0-9-]*(=[a-z0-9_-]+)?")
CONFIG_KEY = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")  # a dotted config key
VARIABLE = re.compile(r"[A-Z][A-Z0-9_]*")  # an environment variable's name
FILE_NAME = re.compile(
    r"[a-z0-9][a-z0-9._-]*"
)  # a file's or a folder's name, with no folder in it
# The capabilities that write to GitHub; the auto-merge guard only when it acts (when it alerts, it only tells you).
GITHUB_WRITES = (
    "review-posting",
    "auto-merge",
    "auto-merge-guard=act",
    "holds",
    "stability-issues",
    "incident-issues",
    "queue-notices",
    "drift-notices",
    "fix-links",
)
# Of those, the ones that turn auto-merge on or off, which GitHub takes only from a login that may push (`gh pr
# merge`: Contents write, gh.CALLS; gh.probe's can_push); the others need one that may comment (its can_comment).
PUSH_NEEDS = tuple(
    c for c in GITHUB_WRITES if c.split("=")[0] in ("auto-merge", "auto-merge-guard")
)
# A model binding's key, while a pass runs on a binding that names it: the passes are review automation's.
BINDING_NEED = ("review-drafts",)
# gh's two logins, by their side in gh.login_dirs, which reads their folders from here.
GH_SIDES = {"github_write": "write", "github_read": "read"}
# The read login's token, in plain words: github_read's made_by says it, and rk read-login's refusal of a token that can
# write names it as the other token to make, for the repository (ghlogins.set_read, `for_repo`).
READ_TOKEN = (
    "a fine-grained token with <org> as its resource owner that only reads <org>/<repo> and "
    "<org>'s team members"
)
# The Claude session's key, the login `rkmon login claude` sets (rkmon_lib/orchestrator.py reads it).
CLAUDE_LOGIN = "claude"
# The claude login's older name: an older package kept the key in this secret, beside the session's other settings
# (rkmon_lib/claude_env_shape.py reads its shape), and the one-time move copies the key alone.
CLAUDE_ENV = "claude_env"
# A model binding's key is a secret whose name ends so (the deploy file's api_key_secret_name): the ignore lists keep
# each binding's key file out of git and the image by that end alone.
BINDING_KEY_SUFFIX = "_api_key"
BINDING_KEY_NAME = re.compile(NAME.pattern + re.escape(BINDING_KEY_SUFFIX))
# The first review pass's key: the name the one-time move gives the local worker's key (OLDER_KEY_NAMES), which the
# first review pass's binding then names (config `worker.api_key_secret`).
FIRST_PASS_KEY = "first_review_pass_api_key"
# A model binding's key's older names that binding_key_name's rule for the end does not give: the local worker's key is
# the first review pass's.
OLDER_KEY_NAMES = {"worker_api_key": FIRST_PASS_KEY}
KEY_FILES = (
    f"*{BINDING_KEY_SUFFIX}",
    f"*{BINDING_KEY_SUFFIX}.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
)
# The folder an older package's gh split (tools/gh_split_apply.py) kept the write login in: no entry's place, as the
# write login's folder is its key's (gh.write_config_dir), where the split keeps it.
OTHER_LOGIN_FOLDERS = ("gh-write",)
# Your own private files, by the names they usually have: the settings files (rkmon-deploy.yaml, rkmon-build.yaml,
# layout 1's rkmon-settings.yaml, any *settings*.yaml) and the privacy check's deny list (rkmon_lib/privacy.py). They
# belong outside the package folder (rkmon_lib/private_files.py); the ignore lists keep them out of git and the image
# wherever they are.
YOUR_FILES = (
    "rkmon-*.yaml",
    "rkmon-*.yml",
    "rkmon-settings*",
    "*settings*.yaml",
    "*settings*.yml",
    "privacy_deny.txt",
)
# Your network's root certificates, which the build file names outside the package folder (their .pem form is among
# KEY_FILES).
CERTIFICATE_FILES = ("*.crt",)
# The files an older package kept a secret in that no fixed entry has among its older names (an entry's own are its
# `older`): the local worker's key, an older name of the first review pass's key only while a binding names that key
# (OLDER_KEY_NAMES); the local worker's extra headers, which no binding sends (a binding's key header is its auth
# settings); and the JIRA MCP server's headers, an unused feature's. The ignore lists keep them
# out of git and the image, and the one-time move removes them as it ends.
OLDER_SECRET_FILES = ("worker_api_key", "worker_headers", "jira_mcp_headers")
PRIVATE_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
PRIVATE_KEY_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")


@dataclass(frozen=True)
class Older:
    """One older name of an entry (the module's docstring says what each field holds)."""

    name: str
    keeper: str = MONITOR
    copied: bool = True
    file: bool = False

    def __post_init__(self):
        if not FILE_NAME.fullmatch(str(self.name)):
            raise ValueError(f"{self.name!r} is not a file's name with no folder in it")
        if self.keeper not in KEEPERS:
            raise ValueError(
                f"{self.name}: the keeper {self.keeper!r} is not one of {', '.join(KEEPERS)}"
            )


@dataclass(frozen=True)
class Secret:
    """One secret or login (the module's docstring says what each field holds)."""

    name: str
    what: str
    shape: str
    made_by: str
    keeper: str
    needed_when: tuple = ()
    permissions: tuple = ()
    key: str = ""
    variable: str = ""
    place: str = ""
    files: tuple = ()
    older: tuple = ()
    prototype_use: str = ""

    def __post_init__(self):
        object.__setattr__(self, "needed_when", tuple(self.needed_when))
        object.__setattr__(
            self, "permissions", tuple(tuple(p) for p in self.permissions)
        )
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "older", tuple(self.older))
        if not all(isinstance(o, Older) for o in self.older):
            raise ValueError(f"{self.name}: each of its older names is an Older")
        if any(o.file for o in self.older) and not (self.shape == LOGIN and self.files):
            raise ValueError(
                f"{self.name}: only a login kept in a folder has an older name kept in a file"
            )
        if self.shape not in SHAPES:
            raise ValueError(
                f"{self.name}: the shape {self.shape!r} is not one of {', '.join(SHAPES)}"
            )
        login = self.shape == LOGIN
        if not (LOGIN_NAME if login else NAME).fullmatch(str(self.name)):
            raise ValueError(
                f"{self.name!r} is not a {'login' if login else 'secret'}'s name"
            )
        if self.keeper not in KEEPERS:
            raise ValueError(
                f"{self.name}: the keeper {self.keeper!r} is not one of {', '.join(KEEPERS)}"
            )
        if not str(self.what).strip() or not str(self.made_by).strip():
            raise ValueError(f"{self.name}: its what or its made_by is empty")
        bad = [c for c in self.needed_when if not CONDITION.fullmatch(str(c))]
        if bad:
            raise ValueError(
                f"{self.name}: {', '.join(map(repr, bad))} is not a capability's name or CAPABILITY=VALUE"
            )
        if self.permissions and not login:
            raise ValueError(f"{self.name}: only a login has permissions")
        if any(len(p) != 2 or p[1] not in ACCESS for p in self.permissions):
            raise ValueError(
                f"{self.name}: a permission is a (permission, access) pair, its access read or write"
            )
        if self.key and not CONFIG_KEY.fullmatch(str(self.key)):
            raise ValueError(f"{self.name}: {self.key!r} is not a config key")
        if self.variable and not VARIABLE.fullmatch(str(self.variable)):
            raise ValueError(
                f"{self.name}: {self.variable!r} is not an environment variable's name"
            )
        bad = [
            p
            for p in ((self.place,) if self.place else ()) + self.files
            if not FILE_NAME.fullmatch(str(p))
        ]
        if bad:
            raise ValueError(
                f"{self.name}: {', '.join(map(repr, bad))} is not a file's name with no folder in it"
            )
        if self.files and not login:
            raise ValueError(f"{self.name}: only a login has files")


@functools.lru_cache(maxsize=None)
def _fixed() -> tuple:
    """The entries every configuration has."""
    from . import atlassian, gh

    read_token = f"you, on GitHub: {READ_TOKEN}"
    return (
        Secret(
            "github_read",
            "gh's read login: every GitHub read; it must not be able to write",
            LOGIN,
            read_token,
            MONITOR,
            (ALWAYS,),
            gh.permissions(gh.READ),
            key="gh.read_config_dir",
            variable="RKMON_GH_READ_DIR",
            place="gh-read",
            files=(gh.HOSTS_FILE,),
            # Its folder keeps its older name (the one-time move copies neither GitHub login). The read login's token,
            # which an older package kept in a file beside the login, was a second copy: the login is its one copy now.
            older=(
                Older("gh-read"),
                Older("gh_read_token", copied=False, file=True),
                Older("gh-read-token", copied=False, file=True),
            ),
        ),
        Secret(
            "github_write",
            "gh's write login: every GitHub write (reviews, notices, issues, auto-merge)",
            LOGIN,
            "you: gh's own sign-in, or a fine-grained token you make on GitHub with <org> as its resource owner, for "
            "<org>/<repo> and the develop guard's issue repository",
            WRITER,
            GITHUB_WRITES,
            gh.permissions(gh.WRITE),
            key="gh.write_config_dir",
            place="gh",
            files=(gh.HOSTS_FILE,),
            # Its folder keeps its older name. A container from before the read/write split kept it in the monitor's
            # secrets folder: the root step of each start moves it to the writer's (docker/rk_root.py).
            older=(Older("gh"),),
        ),
        Secret(
            "atlassian",
            "the atlassian CLI's login: JIRA tickets, and the Claude session's JIRA reads and writes, all made by the "
            "writer",
            LOGIN,
            "you: the atlassian CLI's own sign-in (its device flow)",
            WRITER,
            atlassian.JIRA_FEATURES,
            files=(atlassian.CREDENTIALS_FILE,),
            # A container from before the writer kept it in the monitor's secrets folder. The move never copies it:
            # Atlassian replaces its refresh token at each refresh, so two copies log each other out.
            older=(Older("atlassian", copied=False),),
            prototype_use="tickets are prepared without it, and with it {program} check reads your JIRA project's "
            "required fields early",
        ),
        Secret(
            CLAUDE_LOGIN,
            "the Claude session's key: an Anthropic API key, or your gateway's token (where it goes and how it is sent "
            "are integrations.claude_code's settings)",
            LOGIN,
            "you: an Anthropic API key, or your Claude gateway's token",
            MONITOR,
            ("headless-claude",),
            # Its file's name is no file's of the package: the ignore lists name it wherever it is, and leave out no
            # file of the package, even where case is ignored (Claude Code's CLAUDE.md, a folder named claude).
            place="claude_key",
            # An older package kept the key in CLAUDE_ENV: the one-time move copies the key alone.
            older=(Older(CLAUDE_ENV),),
        ),
        Secret(
            "admin_access_token",
            "the admin dashboard's access token: every request to it carries the token, so it is needed whenever the "
            "admin dashboard is on",
            ONE_LINE,
            GENERATED,
            MONITOR,
            ("admin-dashboard",),
            key="admin_dashboard.access_token_file",
            # Made anew on each machine: the older token is not copied, and goes once this one is set.
            older=(Older("admin_token", copied=False),),
        ),
        Secret(
            "teams_webhook",
            "the team chat channel's incoming-webhook URL",
            HTTPS_URL,
            "your team chat: the channel's incoming webhook",
            WRITER,
            ("chat-alerts", "needs-you-alerts"),
            key="teams.webhook_file",
            # A container from before the writer kept it in the monitor's secrets folder: the one-time move copies it
            # to the writer's.
            older=(Older("teams_webhook"),),
        ),
        Secret(
            "web_host_ssh_key",
            "the team web host's ssh private key",
            SSH_PRIVATE_KEY,
            "you: an SSH key pair, whose public half the team web host accepts",
            WRITER,
            ("team-dashboard=publish",),
            key="publish.html.key",
            # A container from before the writer kept it in the monitor's secrets folder, under its algorithm's name:
            # the one-time move copies it to the writer's under this one.
            older=(Older("webhost_ed25519"),),
        ),
        Secret(
            "gpu_ssh_key",
            "the remote GPU runner's SSH private key (rkmon_lib/gpu.py)",
            SSH_PRIVATE_KEY,
            "you: an SSH key pair, whose public half the GPU cluster's login host accepts",
            MONITOR,
            ("remote-gpu-tests",),
            key="gpu.ssh_key",
        ),
    )


def _binding_keys(cfg: dict) -> dict:
    """Each secret a model binding names, in name order: its name -> [(a config key that names it, that binding's
    name: bindings.WORKER for the local worker's)]."""
    named = []
    worker = cfg.get("worker")
    if isinstance(worker, dict):
        from .bindings import WORKER

        named.append(("worker.api_key_secret", WORKER, worker.get("api_key_secret")))
    bindings = cfg.get("model_bindings")
    for b, raw in sorted((bindings if isinstance(bindings, dict) else {}).items()):
        if isinstance(raw, dict):
            named.append(
                (f"model_bindings.{b}.api_key_secret", b, raw.get("api_key_secret"))
            )
    out: dict = {}
    for key, binding, value in named:
        name = str(value or "").strip()
        if NAME.fullmatch(name):
            out.setdefault(name, []).append((key, binding))
    return dict(sorted(out.items()))


def binding_key_name(name: str) -> str:
    """The name the one-time move copies a model binding's key named NAME to (the module's docstring)."""
    name = str(name)
    if name in OLDER_KEY_NAMES:
        return OLDER_KEY_NAMES[name]
    if NAME.fullmatch(name) and not name.endswith(BINDING_KEY_SUFFIX):
        return name.removesuffix("_key") + BINDING_KEY_SUFFIX
    return name


def _binding_older(name: str, taken) -> tuple:
    """The older names of a model binding's key named NAME: each name `binding_key_name` gives NAME from, but the names
    in TAKEN (the entries' names, places and older names), as Older the one-time move copies.
    """
    olders = [o for o, new in OLDER_KEY_NAMES.items() if new == name]
    stem = name.removesuffix(BINDING_KEY_SUFFIX)
    if stem != name:
        olders += [f"{stem}_key", stem]
    return tuple(
        Older(o)
        for o in dict.fromkeys(olders)
        if o not in taken and binding_key_name(o) == name
    )


def entries(cfg: dict) -> list[Secret]:
    """The fixed entries, then one for each secret a model binding names (the module's docstring)."""
    out = list(_fixed())
    at = {e.name: i for i, e in enumerate(out)}
    named = _binding_keys(cfg or {})
    taken = {
        *at,
        *named,
        *(e.place for e in out if e.place),
        *(o.name for e in out for o in e.older),
    }
    if named:
        from . import deploy_tree
        from .roles import pass_bindings

        run = pass_bindings(cfg)
    for name, pairs in named.items():
        need = BINDING_NEED if any(b in run for _key, b in pairs) else ()
        if name in at:
            e = out[at[name]]
            out[at[name]] = dataclasses.replace(
                e, needed_when=tuple(dict.fromkeys(e.needed_when + need))
            )
            continue
        # the deploy file's keys that name it, as you set them
        keys = list(dict.fromkeys(deploy_tree.name(key) for key, _b in pairs))
        out.append(
            Secret(
                name,
                f"the API key {' and '.join(keys)} {'names' if len(keys) == 1 else 'name'}",
                ONE_LINE,
                "the model's provider or gateway",
                MONITOR,
                need,
                older=_binding_older(name, taken),
            )
        )
    return out


def entry(cfg: dict, name: str) -> Secret | None:
    """The entry named NAME among `entries(cfg)`; None when there is none."""
    return next((e for e in entries(cfg) if e.name == name), None)


def _problem(shape: str, value: str) -> str:
    """What is wrong with a value for its shape, in words that hold none of it; "" when it has the shape."""
    wrong = f"the value is not {SHAPE_WORDS[shape]}"
    if shape == LOGIN:
        return "" if re.fullmatch(r"\S+", value) else wrong
    if shape == HTTPS_URL:
        try:
            host = urllib.parse.urlsplit(value).hostname
        except ValueError:
            host = None
        return (
            ""
            if value.startswith("https://") and host and not re.search(r"\s", value)
            else wrong
        )
    if shape == SSH_PRIVATE_KEY:
        lines = [line.strip() for line in value.splitlines()]
        return (
            ""
            if len(lines) > 2
            and PRIVATE_KEY_BEGIN.fullmatch(lines[0])
            and PRIVATE_KEY_END.fullmatch(lines[-1])
            else wrong
        )
    return "" if len(value.splitlines()) == 1 else wrong  # ONE_LINE


def set_command(entry: Secret) -> tuple:
    """The command that sets the entry, as its words after rkmon's name (the module's docstring)."""
    if entry.shape == LOGIN:
        return ("login", entry.name)
    return ("secret", entry.name) + (
        ("--generate",) if entry.made_by == GENERATED else ()
    )


def typed_set_command(entry: Secret, environ=None) -> Command:
    """The command that sets the entry, as the operator types it where the text is shown (the module's docstring)."""
    from .doctor import wrapper_name

    return Command(
        (wrapper_name(os.environ if environ is None else environ), *set_command(entry))
    )


def how_to_set(entry: Secret) -> str:
    """The entry and the command that sets it, in words (the module's docstring)."""
    kind = "login" if entry.shape == LOGIN else "secret"
    return f"the {kind} {entry.name}: {typed_set_command(entry)}"


def set_fix(entry: Secret, environ=None):
    """The fix that sets the entry (the module's docstring)."""
    command = typed_set_command(entry, environ)
    if entry.made_by == GENERATED:
        return command
    return Steps((command, f", and give {SHAPE_WORDS[entry.shape]}"))


def claude_env_fix(parts: tuple):
    """The fix of a thing CLAUDE_ENV's shape has that no setting carries, from its PARTS (the module's docstring)."""
    shown = tuple(
        (
            p
            if isinstance(p, str)
            else Edit(p.variable, CLAUDE_ENV, Described(p.words) if p.words else DELETE)
        )
        for p in parts
    )
    return shown[0] if len(shown) == 1 else Steps(shown)


def claude_env_or_login_fix(parts: tuple) -> Steps:
    """claude_env_fix(PARTS) before the first start, or the claude login's command (the module's docstring)."""
    file = claude_env_fix(parts)
    return Steps(
        (
            *(file.parts if isinstance(file, Steps) else (file,)),
            ", before the first start; or set the key with",
            typed_set_command(entry({}, CLAUDE_LOGIN)),
        )
    )


def generate(entry: Secret) -> str:
    """A new value for an entry made here (the module's docstring)."""
    if entry.made_by != GENERATED:
        raise ValueError(
            f"{entry.name} is not made here: its value comes from {entry.made_by}"
        )
    import secrets

    return secrets.token_urlsafe(24)


def check_value(entry: Secret, value: str) -> Check:
    """One row on whether the value has the entry's shape (the module's docstring); never any part of the value."""
    if not isinstance(value, str):
        raise TypeError(f"{entry.name}: a secret's value is text")
    text = value.strip()
    fix = set_fix(entry)
    if not text:
        return Check(entry.name, Status.FAIL, "the value is empty", fix)
    problem = _problem(entry.shape, text)
    if problem:
        return Check(entry.name, Status.FAIL, f"{problem} (it is not shown)", fix)
    return Check(
        entry.name, Status.PASS, f"{SHAPE_WORDS[entry.shape]} (the value is not shown)"
    )


def _absolute(place: str) -> str:
    return os.path.abspath(os.path.expanduser(place))


def writers_volume(keeper: str, environ=None) -> bool:
    """Whether KEEPER's secrets are the writer's own volume (the module's docstring)."""
    from . import doctor

    return keeper == WRITER and doctor.deployment_image(
        os.environ if environ is None else environ
    )


def _in_folder(cfg: dict, entry: Secret, keeper: str, name: str, environ) -> str:
    """NAME in KEEPER's secrets folder, absolute: `home`'s rule, for the entry's own place or an older one's."""
    from . import container_layout, paths

    if writers_volume(keeper, environ):
        return _absolute(f"{container_layout.WRITE_SECRETS_DIR}/{name}")
    try:
        secrets = paths.resolve(cfg).get("secrets_dir") or ""
    except Exception:  # noqa: BLE001 - a configuration with no folders names none
        secrets = ""
    if not secrets:
        raise _unknown(entry)
    return _absolute(f"{secrets}/{name}")


def _unknown(entry: Secret):
    """The paths.PathsError of an entry whose place is not known: no secrets folder is known."""
    from . import paths

    what = "folder" if entry.shape == LOGIN and entry.files else "file"
    return paths.PathsError(
        f"no {what} is known for {entry.name}: config.json names no paths.secrets_dir (set paths.working_dir)"
    )


def home(cfg: dict, entry: Secret, environ=None, keeper: str = "") -> str:
    """Its own place in its keeper's secrets folder (the module's docstring)."""
    return _in_folder(
        cfg, entry, keeper or entry.keeper, entry.place or entry.name, environ
    )


def path(cfg: dict, entry: Secret, environ=None) -> str:
    """Where this configuration keeps the entry: a secret's file, or a login's folder (the module's docstring)."""
    from . import config

    e = os.environ if environ is None else environ
    named = e.get(entry.variable) if entry.variable else ""
    if named:
        return _absolute(named)
    if not entry.key or writers_volume(entry.keeper, e):
        return home(cfg, entry, e)
    # the place the load derives for the key (rkmon_lib/derived.py), blank while no secrets folder is known
    named = str(config.lookup(cfg, entry.key) or "").strip()
    if not named:
        raise _unknown(entry)
    return _absolute(named)


def held(place: str, files=None) -> bool:
    """Whether PLACE holds the entry: each of FILES in a login's folder, else a secret's file (the module's
    docstring)."""
    if not place:
        return False
    p = pathlib.Path(place)
    try:
        if files:
            return all((p / f).is_file() for f in files)
        return p.is_file() and bool(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False


def held_files(entry: Secret, older: Older | None = None) -> list | None:
    """The FILES `held` takes for the entry's place, or for OLDER's (the module's docstring)."""
    if entry.files and not (older is not None and older.file):
        return list(entry.files)
    return None


def older_names(entry: Secret) -> list[str]:
    """Its older names but its own name, each once (the module's docstring)."""
    return [n for n in dict.fromkeys(o.name for o in entry.older) if n != entry.name]


def older_places(cfg: dict, entry: Secret, environ=None) -> list[tuple[Older, str]]:
    """Where each older name keeps the entry, but its own place (the module's docstring)."""
    own = {home(cfg, entry, environ), path(cfg, entry, environ)}
    out: dict = {}
    for o in entry.older:
        p = _in_folder(cfg, entry, o.keeper, o.name, environ)
        if p not in own:
            out.setdefault(p, o)
    return [(o, p) for p, o in out.items()]


def renamed(cfg: dict, name: str) -> Secret | None:
    """The entry NAME is an older name of (the module's docstring)."""
    if entry(cfg, name) is not None:
        return None
    return next((e for e in entries(cfg) if name in older_names(e)), None)


def older_name_refusal(name: str, new: Secret) -> str:
    """Why NAME, an older name of the entry NEW, is refused, in words: NEW's name and the command that sets it."""
    kind = "login" if new.shape == LOGIN else "secret"
    return f"{name} is the older name of the {kind} {new.name}: {typed_set_command(new)} sets it"


def name_refusal(cfg: dict | None, name: str) -> str:
    """Why NAME is refused where a secret's name is asked for, in words (the module's docstring); "" for a secret's
    name, and with no configuration known (CFG None) for one that reads as a secret's which the fixed entries do not
    refuse."""
    found = entry(cfg or {}, name)
    if found is not None:
        return (
            f"{name} is a login: {typed_set_command(found)} sets it"
            if found.shape == LOGIN
            else ""
        )
    new = renamed(cfg or {}, name)
    if new is not None:
        return older_name_refusal(name, new)
    if cfg is None and NAME.fullmatch(name):
        return ""
    if LOGIN_NAME.fullmatch(name):
        return f"no secret is named {name}"
    return "a name given is no secret's, and is not shown: it does not read as a name, and may be a value"


def retry(config_file=None):
    """private_files.write's RETRY for a secret's file: the edit of CONFIG_FILE that names another secrets folder (the
    module's docstring)."""
    from . import config, private_files

    @private_files.folder_only
    def another(path):
        file = str(config_file or config.config_path())
        folder = (
            f"the folder of {path}"
            if str(path) == private_files.FILE
            else ANOTHER_FOLDER
        )
        return Edit(FOLDER_KEY, file, Described(folder))

    return another


def for_repo(text: str, repo: str) -> str:
    """TEXT for the repository REPO, <org>/<repo> (the module's docstring)."""
    org, _, name = str(repo).partition("/")
    return (
        str(text)
        .replace("<org>/<repo>", str(repo))
        .replace("<org>", org)
        .replace("<repo>", name)
    )


def binding_key_file(cfg: dict, name: str) -> str:
    """Where a model binding that names the secret NAME reads its key (the module's docstring)."""
    from . import paths

    found = entry(cfg, name)
    try:
        return path(cfg, found) if found else ""
    except paths.PathsError:
        return ""


def _holds(cfg: dict, condition: str) -> bool:
    """Whether a `needed_when` condition holds, by the settings (features.configured: standby changes no need): the
    capability on, or its value the one named; for one that sends for a feature (features.Capability.feature), only
    while that feature is live, since in prototype it sends nothing."""
    if condition == ALWAYS:
        return True
    from . import features

    capability, _, value = condition.partition("=")
    try:
        feature = features.CAPABILITIES[capability].feature
        if feature and not features.live(cfg, feature):
            return False
        got = features.configured(cfg, capability)
        if not value:
            return got not in (False, "off", None)
        return str(got) == value
    except (
        Exception
    ):  # noqa: BLE001 - a configuration this package cannot read yet: nothing is on
        return False


def needed(cfg: dict) -> list[Secret]:
    """The entries that something that is on needs, in the order `entries` gives."""
    return [e for e in entries(cfg) if any(_holds(cfg, c) for c in e.needed_when)]


def condition_words(condition: str) -> str:
    """A `needed_when` condition in words (the module's docstring)."""
    if condition == ALWAYS:
        return ALWAYS_WORDS
    from . import features

    capability, _, value = condition.partition("=")
    title = features.lowered(features.CAPABILITIES[capability].title)
    return f"{title} ({features.value_words(capability, value)})" if value else title


def needed_by(cfg: dict, entry: Secret, among=None) -> list[str]:
    """What needs the entry among what is on, in words (the module's docstring)."""
    return list(
        dict.fromkeys(
            condition_words(c)
            for c in entry.needed_when
            if (among is None or c in among) and _holds(cfg, c)
        )
    )


def waits_for(cfg: dict, entry: Secret) -> list[str]:
    """The features in prototype the entry waits for, while nothing that is on needs it (the module's docstring)."""
    from . import features

    if any(_holds(cfg, c) for c in entry.needed_when):
        return []
    out = []
    for c in entry.needed_when:
        try:
            feature = features.CAPABILITIES[c.partition("=")[0]].feature
            if (
                feature
                and not features.live(cfg, feature)
                and _holds(features.as_live(cfg, feature), c)
            ):
                out.append(feature)
        except Exception:  # noqa: BLE001 - a configuration not read yet: none
            continue
    return list(dict.fromkeys(out))


def waiting_words(cfg: dict, entry: Secret) -> str:
    """What the entry waits for, in words (the module's docstring)."""
    from . import features

    return "; ".join(features.prototype_words(f) for f in waits_for(cfg, entry))


def walk_entries(cfg: dict) -> list[Secret]:
    """The entries the login walk lists: each one something may need, in the order `entries` gives."""
    return [e for e in entries(cfg) if e.needed_when]


def _with_copies(names) -> list[str]:
    """Each name and its copies' (NAME.*), each once: a name that ends in * has its copies already."""
    out = []
    for name in dict.fromkeys(names):
        out += [name] if name.endswith("*") else [name, f"{name}.*"]
    return out


def _secret_files() -> list[str]:
    """Each secret's file, a login's files (one kept in a file: its file, as a secret's), each older name kept in
    a file, and the older secret files."""
    names = []
    for e in _fixed():
        names += e.files if held_files(e) else [e.place or e.name]
        names += [o.name for o in e.older if held_files(e, o) is None]
    return names + list(OLDER_SECRET_FILES)


def secret_patterns() -> list[str]:
    """The names of the files that hold a secret, as globs with no folder in them (the module's docstring)."""
    return _with_copies(_secret_files()) + list(KEY_FILES)


def secret_folders() -> list[str]:
    """The names of the folders that may hold a secret (the module's docstring)."""
    import posixpath

    from . import container_layout, paths

    names = []
    for e in _fixed():
        if held_files(e):
            names += [e.place or e.name]
            names += [o.name for o in e.older if held_files(e, o)]
    names += list(OTHER_LOGIN_FOLDERS)
    names += [
        posixpath.basename(d)
        for d in (
            paths.UNDER_WORKING["secrets_dir"],
            container_layout.SECRETS_DIR,
            container_layout.WRITE_SECRETS_DIR,
        )
    ]
    return list(dict.fromkeys(names))


def ignore_patterns() -> list[str]:
    """The names of the files and folders that may hold a private value, wherever they are (the module's
    docstring)."""
    import posixpath

    from . import private_files

    package = [posixpath.basename(n) for n in private_files.PACKAGE_FILES]
    folders = secret_folders() + [posixpath.basename(private_files.RECORDS_FOLDER)]
    return (
        _with_copies(_secret_files())
        + [glob for n in dict.fromkeys(package) for glob in private_files.own_names(n)]
        + _with_copies(YOUR_FILES)
        + [f"{d}/" for d in dict.fromkeys(folders)]
        + list(KEY_FILES)
        + list(CERTIFICATE_FILES)
    )


def holds_secret(name: str) -> bool:
    """Whether a file or a folder of this name may hold a secret, wherever it is (the module's docstring)."""
    import fnmatch

    return name in secret_folders() or any(
        fnmatch.fnmatchcase(name, p) for p in secret_patterns()
    )
