# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Settings files (rkmon_lib/settings_file.py), on the small schema in
tests/data/sample.schema.json: the templates and the writer's output, golden; each key's closing sentence, made from
its schema and at the end of its description; a file's profile, read from its title line and written back; plain
values typed by the schema, an all-digit SHA-256 kept as text; each refusal with its fix (the other kind, an earlier
and a later layout, no header, text that does not parse, secrets); the checks; the effective values; the fingerprint;
upgrades; and rewrite, through private_files itself.
"""

import ast
import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import private_files, program, registry, settings_file  # noqa: E402
from rkmon_lib.answers import Plain  # noqa: E402
from rkmon_lib.checks import (
    DELETE,
    Check,
    Command,
    Described,
    Edit,
    Status,
)  # noqa: E402
from rkmon_lib.settings_file import Document, FileType, Refused  # noqa: E402
from test_python_floor import python_files  # noqa: E402

SCHEMA_FILE = ROOT / "tests" / "data" / "sample.schema.json"
SAMPLE = FileType(
    "sample_settings",
    2,
    SCHEMA_FILE,
    template_command="template sample FILE",
    section_command="template section KEY",
)
SHA = "0123456789" * 6 + "0123"  # an all-digit SHA-256
# A token's shape, in pieces, so that no file holds one.
TOKEN = "gh" + "p_" + "A1b2C3d4" * 5
GONE = object()  # a key taken out
# The last words of the fix for the keys missing from a section, once its text is pasted over it.
FILL_IN = 'fill in each key whose comment says "Required when"'


def schema(**top):
    """The test schema as a mapping, with each of TOP's keys set at its top level (None: taken out)."""
    data = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    for k, v in top.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    return data


def other_schema():
    data = schema(**{"x-profiles": None})
    data["properties"]["kind"]["const"] = "other_settings"
    return data


OTHER = FileType(
    "other_settings", 2, other_schema(), template_command="template other FILE"
)

# A small file that fits the schema: reviews on, the queue off.
BODY = {
    "service": {"tag": "1.0.0"},
    "repository": {"name": "my-org/my-repo", "labels": ["sample"]},
    "features": {
        "reviews": {"enabled": True, "mode": "prototype"},
        "queue": {"enabled": False},
    },
    "integrations": {"tracker": {"project": "SAMPLE"}},
}

# BODY as a person writes it.
BODY_YAML = """\
kind: sample_settings
schema_version: 2
service:
  tag: "1.0.0"
repository:
  name: "my-org/my-repo"
  labels: ["sample"]
features:
  reviews:
    enabled: true
    mode: "prototype"
  queue:
    enabled: false
integrations:
  tracker:
    project: "SAMPLE"
"""


def changed(changes, values=None):
    """BODY (or VALUES) with each dotted key set to its value (GONE: taken out)."""
    values = copy.deepcopy(BODY if values is None else values)
    for dotted, value in changes.items():
        *parents, last = dotted.split(".")
        cur = values
        for part in parents:
            cur = cur.setdefault(part, {})
        if value is GONE:
            cur.pop(last, None)
        else:
            cur[last] = value
    return values


def as_json(values, kind="sample_settings", layout=2):
    """A settings file's text, in JSON, which the reader takes as well as YAML."""
    return json.dumps({"kind": kind, "schema_version": layout, **values}, indent=2)


def as_shown(rows):
    """Each check row as (name, status, found, fix), the fix as it is shown (None for none)."""
    return [
        (r.name, r.status, r.found, None if r.fix is None else str(r.fix)) for r in rows
    ]


FULL = """\
# yaml-language-server: $schema=https://example.com/sample/2/sample_settings.schema.json
# Sample tool: sample settings (profile "full": reviews and the queue).
# These settings decide what one sample service does. GUIDE.md walks you through them, and README.md explains each
# one.
# This file is yours: keep it out of git. Never put a password, token or key in it: sample login installs those.
# Put text in double quotes. Replace every <...>, then run: sample check FILE

# The type of settings file. Don't change it.
kind: "sample_settings"
# The layout version of this file. Don't change it: sample migrate updates it when a new tool changes the layout.
schema_version: 2

# Service: the service this file sets up.
service:
  # The service's name. Lowercase letters, digits, - and _. Default: "sample".
  name: "sample"
  # The release to run, such as "1.0.0". Required.
  tag: "1.0.0"
  # The registry to pull the release from, such as "registry.example.com/my-team". Leave blank if you load the release
  # from a file.
  registry: ""

# Host: how much of this host the service may use.
host:
  # The number of CPUs the service may use, from 1 to 8. Default: 4.
  cpus: 4
  # The user id to run as, 1000 or more. Default: null, which means your own user id on this host.
  user_id: null
  # The share of this host's memory the service may use, from 0 to 1. Default: 0.5.
  share: 0.5

# Tool: a tool the service installs.
tool:
  # The tool's version, in quotes, such as "7.10". Leave blank for the newest release.
  version: ""
  # The package's SHA-256 checksum: 64 hex digits, in quotes. Leave blank to skip the check.
  package_sha256: ""

# Repository: the repository the service watches.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The labels that put a pull request in scope. Example: ["my-project"]. Required.
  labels: ["<label>"]
  # Labels that several teams share, each with a folder. Write each one as "label": "folder". Default: none.
  folders: {}

# Features: what the service does. Each feature that posts has a mode:
#   "prototype": it shows you its work, but posts nothing.
#   "live": it posts.
features:
  # Reviews: the service reviews each new pull request.
  reviews:
    # true: review pull requests. false: no reviews. Required.
    enabled: true
    # "prototype": reviews are shown to you, not posted. "live": reviews are posted. Required when reviews are on.
    mode: "prototype"
    # The name at the top of each review. Leave blank to use the service's name.
    reviewer_name: ""
    # How hard the reviewer thinks: "low", "medium" or "high". Default: "high".
    effort: "high"
  # Queue: the merge queue.
  queue:
    # true: run the merge queue. false: no merge queue. Required.
    enabled: true
    # "prototype": the queue is shown to you. "live": the queue merges. Required when the queue is on.
    mode: "prototype"
    # Notices: a comment on each pull request near the front of the queue.
    notices:
      # true: post them. false: none. To turn it on, see GUIDE.md, "Turn on a feature". Required when the queue is on.
      enabled: false

# Integrations: the systems the service talks to.
integrations:
  # Tracker: the issue tracker the reviews link to.
  tracker:
    # The tracker's project key. Required when reviews are on.
    project: "<PROJECT>"
    # The tracker's address. Example: "https://tracker.example.com". Leave blank for no links.
    address: ""

# Bindings: other models the reviews can use, each under a name you choose. Default: none.
bindings: {}
"""

LEAN = """\
# yaml-language-server: $schema=https://example.com/sample/2/sample_settings.schema.json
# Sample tool: sample settings (profile "lean": the queue only).
# These settings decide what one sample service does. GUIDE.md walks you through them, and README.md explains each
# one.
# This service runs the queue by fixed rules. It writes no reviews, so it needs no tracker.
# This file is yours: keep it out of git. Never put a password, token or key in it: sample login installs those.
# Put text in double quotes. Replace every <...>, then run: sample check FILE

# The type of settings file. Don't change it.
kind: "sample_settings"
# The layout version of this file. Don't change it: sample migrate updates it when a new tool changes the layout.
schema_version: 2

# Service: the service this file sets up.
service:
  # The service's name. Lowercase letters, digits, - and _. Default: "sample".
  name: "sample"
  # The release to run, such as "1.0.0". Required.
  tag: "1.0.0"
  # The registry to pull the release from, such as "registry.example.com/my-team". Leave blank if you load the release
  # from a file.
  registry: ""

# Host: how much of this host the service may use.
host:
  # The number of CPUs the service may use, from 1 to 8. Default: 4.
  cpus: 4
  # The user id to run as, 1000 or more. Default: null, which means your own user id on this host.
  user_id: null
  # The share of this host's memory the service may use, from 0 to 1. Default: 0.5.
  share: 0.5

# Tool: a tool the service installs.
tool:
  # The tool's version, in quotes, such as "7.10". Leave blank for the newest release.
  version: ""
  # The package's SHA-256 checksum: 64 hex digits, in quotes. Leave blank to skip the check.
  package_sha256: ""

# Repository: the repository the service watches.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "<org>/<repo>"
  # The labels that put a pull request in scope. Example: ["my-project"]. Required.
  labels: ["<label>"]
  # Labels that several teams share, each with a folder. Write each one as "label": "folder". Default: none.
  folders: {}

# Features: what the service does. Each feature that posts has a mode:
#   "prototype": it shows you its work, but posts nothing.
#   "live": it posts.
features:
  # Reviews: the service reviews each new pull request.
  reviews:
    # true: review pull requests. false: no reviews. To turn it on, see GUIDE.md, "Turn on a feature". Required.
    enabled: false
  # Queue: the merge queue.
  queue:
    # true: run the merge queue. false: no merge queue. Required.
    enabled: true
    # "prototype": the queue is shown to you. "live": the queue merges. Required when the queue is on.
    mode: "prototype"
    # Notices: a comment on each pull request near the front of the queue.
    notices:
      # true: post them. false: none. To turn it on, see GUIDE.md, "Turn on a feature". Required when the queue is on.
      enabled: false

# Integrations: the systems the service talks to.
integrations:
  # Tracker: not used while reviews are off.
  tracker: {}

# Bindings: not used while reviews are off. Default: none.
bindings: {}
"""

# The every-key text: the full profile with no key left out, notices in full; the section post_as does not choose,
# and an example of each form the bindings take, commented out.
EVERY = FULL.replace(
    """\
    notices:
      # true: post them. false: none. To turn it on, see GUIDE.md, "Turn on a feature". Required when the queue is on.
      enabled: false
""",
    """\
    notices:
      # true: post them. false: none. Required when the queue is on.
      enabled: false
      # Minutes between two updates of a notice, 5 or more. Default: 30.
      every_minutes: 30
      # The page the notices link to. Example: "https://dashboards.example.com/my-team/". Required when the queue is
      # live. Otherwise, leave blank to link no page.
      page_url: ""
      # How a notice is posted: "comment" (a comment on the pull request) or "status" (a commit status). Default:
      # "comment".
      post_as: "comment"
      # Comment: the comment's own words.
      comment:
        # A line at the top of each comment. Leave blank for no header.
        header: ""
      # Status: use this block instead of comment when post_as is "status".
      # status:
      #   # The status's name on the pull request. Default: "sample/queue".
      #   context: "sample/queue"
""",
).replace(
    "bindings: {}\n",
    """\
bindings: {}
# A model behind a gateway:
#   <name>:
#     # "gateway": a model behind a gateway. Required.
#     provider: "gateway"
#     # The gateway's address. Required.
#     address: "<url>"
#     # The name of the secret that holds the gateway's key. Leave blank for a gateway that takes no key.
#     api_key_secret_name: ""
# A model on this host:
#   <name>:
#     # "local": a model on this host. Required.
#     provider: "local"
#     # The port the model listens on. Required.
#     port: 8080
""",
)

# A file as someone wrote it: plain and quoted values, a feature that is off with a value of its own, an unused
# section and map that are set, and a map's names that need quotes.
USER = """\
# My settings.
kind: sample_settings
schema_version: 2
service:
  name: web
  tag: "2.1.0"
host:
  cpus: 2
  user_id: 1001
tool:
  version: "7.10"
  package_sha256: "0123456789012345678901234567890123456789012345678901234567890123"
repository:
  name: my-org/my-repo
  labels: [sample, "on"]
  folders:
    needs-ci: projects/sample
    "yes": projects/other
features:
  reviews:
    enabled: false
    mode: live
  queue:
    enabled: true
    mode: live
    notices:
      enabled: true
      every_minutes: 15
      page_url: "https://dashboards.example.com/my-team/"
integrations:
  tracker:
    project: SAMPLE
bindings:
  big:
    provider: gateway
    address: "https://gateway.example.com/v1"
    api_key_secret_name: big_api_key
  small:
    provider: local
    port: 8080
"""

DUMP = """\
# yaml-language-server: $schema=https://example.com/sample/2/sample_settings.schema.json
# Sample tool: sample settings.
# These settings decide what one sample service does. GUIDE.md walks you through them, and README.md explains each
# one.
# This file is yours: keep it out of git. Never put a password, token or key in it: sample login installs those.
# Put text in double quotes. Replace every <...>, then run: sample check FILE

# The type of settings file. Don't change it.
kind: "sample_settings"
# The layout version of this file. Don't change it: sample migrate updates it when a new tool changes the layout.
schema_version: 2

# Service: the service this file sets up.
service:
  # The service's name. Lowercase letters, digits, - and _. Default: "sample".
  name: "web"
  # The release to run, such as "1.0.0". Required.
  tag: "2.1.0"
  # The registry to pull the release from, such as "registry.example.com/my-team". Leave blank if you load the release
  # from a file.
  registry: ""

# Host: how much of this host the service may use.
host:
  # The number of CPUs the service may use, from 1 to 8. Default: 4.
  cpus: 2
  # The user id to run as, 1000 or more. Default: null, which means your own user id on this host.
  user_id: 1001
  # The share of this host's memory the service may use, from 0 to 1. Default: 0.5.
  share: 0.5

# Tool: a tool the service installs.
tool:
  # The tool's version, in quotes, such as "7.10". Leave blank for the newest release.
  version: "7.10"
  # The package's SHA-256 checksum: 64 hex digits, in quotes. Leave blank to skip the check.
  package_sha256: "0123456789012345678901234567890123456789012345678901234567890123"

# Repository: the repository the service watches.
repository:
  # The repository, as <org>/<repo>. Example: "my-org/my-repo". Required.
  name: "my-org/my-repo"
  # The labels that put a pull request in scope. Example: ["my-project"]. Required.
  labels: ["sample", "on"]
  # Labels that several teams share, each with a folder. Write each one as "label": "folder". Default: none.
  folders:
    needs-ci: "projects/sample"
    "yes": "projects/other"

# Features: what the service does. Each feature that posts has a mode:
#   "prototype": it shows you its work, but posts nothing.
#   "live": it posts.
features:
  # Reviews: the service reviews each new pull request.
  reviews:
    # true: review pull requests. false: no reviews. Required.
    enabled: false
    # "prototype": reviews are shown to you, not posted. "live": reviews are posted. Required when reviews are on.
    mode: "live"
    # The name at the top of each review. Leave blank to use the service's name.
    reviewer_name: ""
    # How hard the reviewer thinks: "low", "medium" or "high". Default: "high".
    effort: "high"
  # Queue: the merge queue.
  queue:
    # true: run the merge queue. false: no merge queue. Required.
    enabled: true
    # "prototype": the queue is shown to you. "live": the queue merges. Required when the queue is on.
    mode: "live"
    # Notices: a comment on each pull request near the front of the queue.
    notices:
      # true: post them. false: none. Required when the queue is on.
      enabled: true
      # Minutes between two updates of a notice, 5 or more. Default: 30.
      every_minutes: 15
      # The page the notices link to. Example: "https://dashboards.example.com/my-team/". Required when the queue is
      # live. Otherwise, leave blank to link no page.
      page_url: "https://dashboards.example.com/my-team/"
      # How a notice is posted: "comment" (a comment on the pull request) or "status" (a commit status). Default:
      # "comment".
      post_as: "comment"
      # Comment: the comment's own words.
      comment:
        # A line at the top of each comment. Leave blank for no header.
        header: ""

# Integrations: the systems the service talks to.
integrations:
  # Tracker: the issue tracker the reviews link to.
  tracker:
    # The tracker's project key. Required when reviews are on.
    project: "SAMPLE"
    # The tracker's address. Example: "https://tracker.example.com". Leave blank for no links.
    address: ""

# Bindings: other models the reviews can use, each under a name you choose. Default: none.
bindings:
  big:
    # "gateway": a model behind a gateway. Required.
    provider: "gateway"
    # The gateway's address. Required.
    address: "https://gateway.example.com/v1"
    # The name of the secret that holds the gateway's key. Leave blank for a gateway that takes no key.
    api_key_secret_name: "big_api_key"
  small:
    # "local": a model on this host. Required.
    provider: "local"
    # The port the model listens on. Required.
    port: 8080
"""


class Files(unittest.TestCase):
    """A test with a folder of its own for the files it reads."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = pathlib.Path(tmp.name)
        self.where = settings_file.shown(self.folder / "settings.yaml")
        # The command that reads the file again, which ends a fix that holds no command or edit of its own.
        self.again = str(Command(("rkmon", "check", "--settings", self.where)))

    def file(self, text, name="settings.yaml"):
        path = self.folder / name
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
        return path

    def read(self, text, file_type=SAMPLE, **options):
        return settings_file.read(self.file(text), file_type, **options)

    def refused(self, text, file_type=SAMPLE, **options):
        with self.assertRaises(Refused) as cm:
            self.read(text, file_type, **options)
        return cm.exception

    def assertRefused(self, exc, reason, found, fix, name="sample settings file"):
        """EXC refused for REASON, with one FAIL row: NAME, FOUND and FIX, the fix as it is shown."""
        self.assertEqual(exc.reason, reason)
        self.assertEqual(as_shown(exc.rows), [(name, Status.FAIL, found, fix)])

    def rows(self, values, file_type=SAMPLE):
        """The rows validate gives VALUES, written as a file and read, as_shown."""
        return as_shown(settings_file.validate(self.read(as_json(values), file_type)))


class TestTemplate(Files):
    def test_the_full_profile(self):
        self.assertEqual(settings_file.template(SAMPLE), FULL)
        self.assertEqual(settings_file.template(SAMPLE, "full"), FULL)

    def test_the_lean_profile(self):
        # Its own header paragraph after the first; reviews off, so its switch alone, and the tracker and the
        # bindings, which only reviews read, as {} with the reason.
        self.assertEqual(settings_file.template(SAMPLE, "lean"), LEAN)

    def test_the_every_key_text(self):
        # The full profile's header and values, with no key left out: notices, which is off, in full and with no
        # turn-on sentence; the section post_as does not choose, and the bindings' examples, commented out.
        self.assertEqual(settings_file.template(SAMPLE, every_key=True), EVERY)
        self.assertEqual(settings_file.template(SAMPLE, "full", every_key=True), EVERY)
        # With the lean profile's values: reviews, which is off, and the tracker, which nothing reads, in full.
        lean = settings_file.template(SAMPLE, "lean", every_key=True)
        self.assertIn(
            "    # true: review pull requests. false: no reviews. Required.\n"
            "    enabled: false\n"
            '    # "prototype": reviews are shown to you, not posted. "live": reviews are posted. Required when reviews '
            "are on.\n",
            lean,
        )
        self.assertIn(
            "  # Tracker: the issue tracker the reviews link to.\n  tracker:\n", lean
        )

    def test_no_line_is_longer_than_118_columns(self):
        for text in (FULL, LEAN, DUMP, EVERY):
            self.assertLessEqual(max(len(line) for line in text.splitlines()), 118)

    def test_a_line_to_comment_out_is_written_narrower(self):
        # Each example line is wrapped 2 columns narrower, so that with "# " in front it still fits 118 columns: at
        # 118, the port's comment would take 117 columns, and 119 once commented out.
        data = schema()
        port = data["$defs"]["binding"]["oneOf"][1]["properties"]["port"]
        port["description"] = "abc " * 40 + "Required."
        text = settings_file.template(
            FileType("sample_settings", 2, data), every_key=True
        )
        lines = [line for line in text.splitlines() if line.startswith("#     # abc")]
        self.assertEqual([len(line) for line in lines[:1]], [115])
        self.assertLessEqual(max(len(line) for line in text.splitlines()), 118)

    def test_a_template_reads_back_and_only_its_placeholders_fail(self):
        full = ["repository.name", "repository.labels", "integrations.tracker.project"]
        for profile, every_key, names in (
            ("full", False, full),
            ("lean", False, ["repository.name", "repository.labels"]),
            ("full", True, full),
        ):
            doc = self.read(settings_file.template(SAMPLE, profile, every_key))
            rows = settings_file.validate(doc)
            self.assertEqual(
                [(r.name, r.status) for r in rows], [(n, Status.FAIL) for n in names]
            )
            self.assertTrue(
                all(r.found.startswith("still the template's <") for r in rows)
            )

    def test_the_profiles(self):
        # In the schema's order, the first being the one a template takes when none is named; none for a file type
        # without them.
        self.assertEqual(
            settings_file.profiles(SAMPLE),
            (
                settings_file.Profile("full", "reviews and the queue", (), {}),
                settings_file.Profile(
                    "lean",
                    "the queue only",
                    (
                        "This service runs the queue by fixed rules. It writes no reviews, so it needs no tracker.",
                    ),
                    {"features.reviews.enabled": False},
                ),
            ),
        )
        self.assertEqual(settings_file.profiles(OTHER), ())

    def test_an_unknown_profile(self):
        with self.assertRaisesRegex(
            ValueError, 'no profile "small"; the profiles: full, lean'
        ):
            settings_file.template(SAMPLE, "small")

    def test_a_file_type_without_profiles(self):
        text = settings_file.template(OTHER)
        self.assertEqual(text.splitlines()[1], "# Sample tool: sample settings.")
        self.assertIn('\nkind: "other_settings"\n', text)
        with self.assertRaisesRegex(ValueError, "other settings have no profiles"):
            settings_file.template(OTHER, "full")

    def test_the_modeline_names_only_an_https_address(self):
        for address in ("schema/sample_settings.schema.json", None):
            file_type = FileType("sample_settings", 2, schema(**{"$id": address}))
            first = settings_file.template(file_type).splitlines()[0]
            self.assertEqual(
                first,
                '# Sample tool: sample settings (profile "full": reviews and the queue).',
            )

    def test_a_comment_is_the_description_as_the_schema_has_it(self):
        # The writer makes no closing sentence: it writes the description's own, the turn-on sentence before it, and
        # an unused map's after the reason.
        data = schema()
        data["properties"]["host"]["properties"]["cpus"][
            "description"
        ] = "How many CPUs. Default: four."
        reviews = data["properties"]["features"]["properties"]["reviews"]
        reviews["properties"]["enabled"][
            "description"
        ] = "Whether it reviews. Default: true."
        data["properties"]["bindings"][
            "description"
        ] = "other models. Default: an empty map."
        text = settings_file.template(FileType("sample_settings", 2, data), "lean")
        self.assertIn("\n  # How many CPUs. Default: four.\n  cpus: 4\n", text)
        self.assertIn(
            '\n    # Whether it reviews. To turn it on, see GUIDE.md, "Turn on a feature". Default: true.\n',
            text,
        )
        self.assertIn(
            "\n# Bindings: not used while reviews are off. Default: an empty map.\n",
            text,
        )

    def test_an_unused_section_or_map_without_a_reason_is_left_out(self):
        # x-off-reason is optional. With none (the tracker), or a blank one (the bindings), each is left out while
        # reviews are off, as any other unused key is; integrations, with no other key, is written as {}.
        data = schema()
        del data["properties"]["integrations"]["properties"]["tracker"]["x-off-reason"]
        data["properties"]["bindings"]["x-off-reason"] = " "
        text = settings_file.template(FileType("sample_settings", 2, data), "lean")
        expected = LEAN.replace(
            "integrations:\n"
            "  # Tracker: not used while reviews are off.\n"
            "  tracker: {}\n"
            "\n"
            "# Bindings: not used while reviews are off. Default: none.\n"
            "bindings: {}\n",
            "integrations: {}\n",
        )
        self.assertNotEqual(expected, LEAN)
        self.assertEqual(text, expected)

    def test_a_feature_with_only_its_switch_takes_no_turn_on_sentence(self):
        # The switch alone hides no other key, so its comment is the description alone; reviews' switch, which
        # hides the rest of reviews, still takes the sentence.
        data = schema()
        queue = data["properties"]["features"]["properties"]["queue"]
        notices = queue["properties"]["notices"]["properties"]
        for key in ("every_minutes", "page_url", "post_as", "comment", "status"):
            del notices[key]
        text = settings_file.template(FileType("sample_settings", 2, data), "lean")
        self.assertIn(
            "\n    notices:\n"
            "      # true: post them. false: none. Required when the queue is on.\n"
            "      enabled: false\n",
            text,
        )
        self.assertIn(
            '\n    # true: review pull requests. false: no reviews. To turn it on, see GUIDE.md, "Turn on a feature". '
            "Required.\n    enabled: false\n",
            text,
        )

    def test_every_key_has_a_template_value(self):
        data = schema()
        del data["properties"]["service"]["properties"]["tag"]["x-template"]
        with self.assertRaisesRegex(
            ValueError, "^service.tag: no value for the template"
        ):
            settings_file.template(FileType("sample_settings", 2, data))

    def test_a_keys_template_value(self):
        """Its x-template, else its default, else its const, whatever the value (a blank too), as a copy; a key with
        none of them is refused, naming it."""
        for node, value in (
            ({"x-template": "<x>", "default": "d", "const": "c"}, "<x>"),
            ({"x-template": "", "default": "d"}, ""),
            ({"default": None, "const": "c"}, None),
            ({"const": "c"}, "c"),
        ):
            with self.subTest(node=node):
                self.assertEqual(settings_file.template_value(node, "a.b"), value)
        node = {"default": ["x"]}
        settings_file.template_value(node, "a.b").append("y")
        self.assertEqual(node["default"], ["x"])
        with self.assertRaisesRegex(ValueError, "^a.b: no value for the template"):
            settings_file.template_value({"type": "string"}, "a.b")

    def test_a_description_names_only_the_keys_own_numbers(self):
        data = schema()
        data["properties"]["host"]["properties"]["user_id"][
            "description"
        ] = "At most {maximum}."
        with self.assertRaisesRegex(
            ValueError, r"^host.user_id: its description names \{maximum\}"
        ):
            settings_file.template(FileType("sample_settings", 2, data))

    def test_a_description_names_its_template_value(self):
        """{x-template} is its key's template value, a file type's template values included (they are set when the
        schema loads), written as a scalar; a key with none is refused, naming it."""
        data = schema()
        data["properties"]["service"]["properties"]["tag"][
            "description"
        ] = "Keep {x-template}."
        loaded = FileType(
            "sample_settings", 2, data, template_values=lambda: {"service.tag": "1.2"}
        ).load()
        node = loaded["properties"]["service"]["properties"]["tag"]
        self.assertEqual(settings_file.description(node, "service.tag"), 'Keep "1.2".')
        with self.assertRaisesRegex(
            ValueError, r"^a.b: its description names \{x-template\}, which it does not"
        ):
            settings_file.description({"description": "Keep {x-template}."}, "a.b")

    def test_a_description_names_a_package_default(self):
        """{package:NAME} is the package's default of NAME, read through the registry when the text is made, and
        written as a scalar; a NAME the package's defaults give no value, or a section, is refused, naming the key.
        """
        pct = registry.get(registry.package_defaults(), "max_host_usage_pct")
        self.assertEqual(
            settings_file.description(
                {"description": "Up to {package:max_host_usage_pct}%."}, "a.b"
            ),
            f"Up to {pct}%.",
        )
        node = {"description": "Its size plus {package:a.room_gb}, or {package:b}."}
        made_up = {"a": {"room_gb": 6}, "b": 1.5}
        with mock.patch.object(registry, "package_defaults", return_value=made_up):
            self.assertEqual(
                settings_file.description(node, "host.size"), "Its size plus 6, or 1.5."
            )
            for field in ("{package:c}", "{package:a}"):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError,
                    rf"^host.size: its description names \{{package:[ac]\}}, which is no value of the package's",
                ):
                    settings_file.description({"description": field}, "host.size")


def notices_keys(data):
    """The keys of features.queue.notices in the schema mapping DATA."""
    queue = data["properties"]["features"]["properties"]["queue"]
    return queue["properties"]["notices"]["properties"]


class TestChooser(Files):
    """A key with x-chooses, post_as: the section its value names is written; each other one is left out, unless
    it holds values of its own, and commented out in the every-key text."""

    STATUS = (
        '      # Status: use this block instead of comment when post_as is "status".\n'
        "      status:\n"
        '        # The status\'s name on the pull request. Default: "sample/queue".\n'
    )

    def dumped(self, **notices):
        """The writer's text of BODY with the queue on and notices on, with NOTICES' keys."""
        queue = {"enabled": True, "mode": "prototype", "notices": {"enabled": True}}
        queue["notices"].update(notices)
        doc = self.read(as_json(changed({"features.queue": queue})))
        return settings_file.dump(doc)

    def test_the_section_it_chooses_is_written_and_the_other_left_out(self):
        text = self.dumped(post_as="status", status={"context": "ci/queue"})
        self.assertIn(
            '      post_as: "status"\n' + self.STATUS + '        context: "ci/queue"\n',
            text,
        )
        self.assertNotIn("\n      comment:", text)
        text = self.dumped()
        self.assertIn(
            '      post_as: "comment"\n'
            "      # Comment: the comment's own words.\n"
            "      comment:\n",
            text,
        )
        self.assertNotIn("\n      status:", text)

    def test_one_it_does_not_choose_is_kept_with_values_of_its_own(self):
        text = self.dumped(post_as="status", comment={"header": "Queue"})
        self.assertIn(
            "      comment:\n"
            "        # A line at the top of each comment. Leave blank for no header.\n"
            '        header: "Queue"\n',
            text,
        )
        self.assertIn(self.STATUS, text)

    def test_a_value_that_names_no_section_chooses_none(self):
        # post_as "both" is the checks' to report: each section is written.
        text = self.dumped(post_as="both")
        self.assertIn("\n      comment:\n", text)
        self.assertIn(self.STATUS, text)

    def test_with_every_key_each_other_one_is_commented_out(self):
        data = schema()
        data["x-profiles"]["full"]["values"] = {
            "features.queue.notices.post_as": "status"
        }
        text = settings_file.template(
            FileType("sample_settings", 2, data), every_key=True
        )
        self.assertIn(
            '      post_as: "status"\n'
            "      # Comment: the comment's own words.\n"
            "      # comment:\n"
            "      #   # A line at the top of each comment. Leave blank for no header.\n"
            '      #   header: ""\n'
            + self.STATUS
            + '        context: "sample/queue"\n',
            text,
        )

    def test_what_x_chooses_is_for(self):
        wrong = (
            "features.queue.notices.post_as: x-chooses is true, on a key with an enum whose every value names a "
            "section beside the key"
        )
        changes = (
            lambda keys: keys["post_as"].update({"x-chooses": "yes"}),
            lambda keys: keys["post_as"]["enum"].append("check"),
            lambda keys: keys["post_as"]["enum"].__setitem__(1, "every_minutes"),
            lambda keys: keys["post_as"].pop("enum"),
        )
        for i, change in enumerate(changes):
            data = schema()
            change(notices_keys(data))
            with self.subTest(i=i), self.assertRaises(ValueError) as cm:
                FileType("sample_settings", 2, data).load()
            self.assertEqual(str(cm.exception), wrong)
        data = schema()
        data["properties"]["repository"]["properties"]["labels"]["items"][
            "x-chooses"
        ] = True
        with self.assertRaises(ValueError) as cm:
            FileType("sample_settings", 2, data).load()
        self.assertEqual(
            str(cm.exception),
            "repository.labels[]: x-chooses is for a key of a section",
        )


class TestExamples(Files):
    """With every key, a map whose rule for names has an x-template is followed by its example entries, commented
    out: one for each form its entries take."""

    def every(self, data):
        return settings_file.template(
            FileType("sample_settings", 2, data), every_key=True
        )

    def test_one_form_has_no_heading(self):
        data = schema()
        data["$defs"]["binding"] = data["$defs"]["binding"]["oneOf"][1]
        self.assertTrue(
            self.every(data).endswith(
                "bindings: {}\n"
                "#   <name>:\n"
                '#     # "local": a model on this host. Required.\n'
                '#     provider: "local"\n'
                "#     # The port the model listens on. Required.\n"
                "#     port: 8080\n"
            )
        )

    def test_each_of_several_forms_has_a_description(self):
        data = schema()
        del data["$defs"]["binding"]["oneOf"][0]["description"]
        with self.assertRaises(ValueError) as cm:
            self.every(data)
        self.assertEqual(
            str(cm.exception),
            "bindings: each form of its entries has a description, one sentence",
        )

    def test_a_map_without_an_example_name_has_none(self):
        data = schema()
        del data["properties"]["bindings"]["propertyNames"]["x-template"]
        self.assertTrue(self.every(data).endswith("\nbindings: {}\n"))

    def test_they_follow_the_maps_own_entries(self):
        data = schema()
        data["x-profiles"]["full"]["values"] = {
            "bindings": {"big": {"provider": "local", "port": 9000}}
        }
        self.assertIn(
            "\nbindings:\n"
            "  big:\n"
            '    # "local": a model on this host. Required.\n'
            '    provider: "local"\n'
            "    # The port the model listens on. Required.\n"
            "    port: 9000\n"
            "# A model behind a gateway:\n"
            "#   <name>:\n",
            self.every(data),
        )

    def test_a_name_that_is_one_placeholder_is_written_plain(self):
        bindings = {
            "<name>": {"provider": "local", "port": 8080},
            "<a b>": {"provider": "local", "port": 8081},
        }
        text = settings_file.dump(self.read(as_json(changed({"bindings": bindings}))))
        self.assertIn("\nbindings:\n  <name>:\n", text)
        self.assertIn('\n  "<a b>":\n', text)
        again = settings_file.read(self.file(text, "again.yaml"), SAMPLE)
        self.assertEqual(again.values["bindings"], bindings)


def with_mirror():
    """The test schema as a mapping, with a section after service that the file does not require, which requires a
    key of its own: one you write whole or leave out (like the deploy file's updates section).
    """
    data = schema()
    mirror = {
        "type": "object",
        "title": "Mirror",
        "description": "where the service fetches newer code. Leave this section out to fetch none.",
        "x-class": "public",
        "additionalProperties": False,
        "required": ["address"],
        "properties": {
            "address": {
                "type": "string",
                "pattern": "^https://\\S+$",
                "x-template": "<address>",
                "x-source": "the mirror's address",
                "x-class": "private",
                "description": "The mirror's address. Required.",
            }
        },
    }
    properties = {}
    for key, node in data["properties"].items():
        properties[key] = node
        if key == "service":
            properties["mirror"] = mirror
    data["properties"] = properties
    return data


MIRRORED = FileType(
    "sample_settings",
    2,
    with_mirror(),
    template_command="template sample FILE",
    section_command="template section KEY",
)


class TestWholeOrLeftOut(Files):
    """A section the file does not require, which requires a key of its own (like the deploy file's updates section): a
    template leaves it out, the every-key text and the section command write it, dump writes it when the file has it,
    and its keys are missing only from a file that has it."""

    MIRROR = (
        "# Mirror: where the service fetches newer code. Leave this section out to fetch none.\n"
        "mirror:\n"
        "  # The mirror's address. Required.\n"
        '  address: "<address>"\n'
    )

    def test_a_template_leaves_it_out(self):
        self.assertEqual(settings_file.template(MIRRORED), FULL)
        self.assertEqual(settings_file.template(MIRRORED, "lean"), LEAN)

    def test_the_every_key_text_and_the_section_command_write_it(self):
        self.assertEqual(EVERY.count("\n# Host: "), 1)
        self.assertEqual(
            settings_file.template(MIRRORED, every_key=True),
            EVERY.replace("\n# Host: ", "\n" + self.MIRROR + "\n# Host: "),
        )
        self.assertEqual(settings_file.section(MIRRORED, "mirror"), self.MIRROR)

    def test_dump_writes_it_when_the_file_has_it(self):
        self.assertNotIn("mirror", settings_file.dump(Document(MIRRORED, 2, BODY)))
        for address in ("https://mirror.example.com/sample", "<address>"):
            with self.subTest(address=address):
                values = changed({"mirror": {"address": address}})
                self.assertIn(
                    "\n\n# Mirror: where the service fetches newer code. Leave this section out to fetch none.\n"
                    "mirror:\n"
                    "  # The mirror's address. Required.\n"
                    f'  address: "{address}"\n'
                    "\n# Host: ",
                    settings_file.dump(Document(MIRRORED, 2, values)),
                )

    def test_its_keys_are_missing_only_from_a_file_that_has_it(self):
        self.assertEqual(
            [row[:2] for row in self.rows(BODY, MIRRORED)], [("settings", Status.PASS)]
        )
        self.assertEqual(
            self.rows(changed({"mirror": {}}), MIRRORED),
            [
                (
                    "mirror.address",
                    Status.FAIL,
                    "missing",
                    f"set mirror.address in {self.where} to the mirror's address",
                )
            ],
        )


class TestSections(unittest.TestCase):
    """What the section command prints: one section of the every-key text, in full."""

    def test_the_sections(self):
        # Each section, and the map with a title, in the file's order; no header key, no map's entry.
        self.assertEqual(
            settings_file.sections(SAMPLE),
            [
                ("service", "Service"),
                ("host", "Host"),
                ("tool", "Tool"),
                ("repository", "Repository"),
                ("features", "Features"),
                ("features.reviews", "Reviews"),
                ("features.queue", "Queue"),
                ("features.queue.notices", "Notices"),
                ("features.queue.notices.comment", "Comment"),
                ("features.queue.notices.status", "Status"),
                ("integrations", "Integrations"),
                ("integrations.tracker", "Tracker"),
                ("bindings", "Bindings"),
            ],
        )

    def test_each_is_the_every_key_texts(self):
        # Its comment, then every key, indented as in the file, and nothing of it left out: the line after it in the
        # every-key text is the next key beside it or above it.
        for key, _ in settings_file.sections(SAMPLE):
            text = settings_file.section(SAMPLE, key)
            pad = " " * (2 * key.count("."))
            with self.subTest(key=key):
                if key == "features.queue.notices.status":
                    # The one post_as does not choose: you asked for it, so it is not commented out.
                    self.assertEqual(
                        text, TestChooser.STATUS + '        context: "sample/queue"\n'
                    )
                    continue
                self.assertTrue(text.startswith(pad + "# "))
                start = EVERY.index("\n" + text) + 1
                after = EVERY[start + len(text) :].split("\n", 1)[0]
                self.assertFalse(after.startswith(pad + " "), after)

    def test_one_set_aside_is_written_as_the_every_key_text_writes_it(self):
        """The section post_as does not choose, which the every-key text comments out after its comment, is written as
        it is there before its "# ": a comment line that would fit the file's width, but not with "# " in front of it,
        is wrapped as it is there."""
        data = schema()
        notices = data["properties"]["features"]["properties"]["queue"]["properties"][
            "notices"
        ]
        context = notices["properties"]["status"]["properties"]["context"]
        context["description"] = "a" * 50 + " " + "b" * 56 + ' Default: "sample/queue".'
        sample = FileType(
            "sample_settings", 2, data, section_command="template section KEY"
        )
        text = settings_file.section(sample, "features.queue.notices.status")
        comment, *rest = text.splitlines(keepends=True)
        self.assertEqual(rest[1], "        # " + "a" * 50 + "\n")
        self.assertIn(
            comment + "".join("      # " + line[6:] for line in rest),
            settings_file.template(sample, every_key=True),
        )

    def test_a_key_that_is_no_section(self):
        for key, message in (
            (
                "features.queues",
                "no section features.queues; did you mean features.queue?",
            ),
            ("zzz", "no section zzz; rkmon template section lists the sections"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError) as cm:
                settings_file.section(SAMPLE, key)
            self.assertEqual(str(cm.exception), message)
        with self.assertRaises(ValueError) as cm:
            settings_file.section(FileType("sample_settings", 2, SCHEMA_FILE), "zzz")
        self.assertEqual(str(cm.exception), "no section zzz")


class TestComments(unittest.TestCase):
    """What the README's settings reference prints: each comment of the every-key text, with its key."""

    def test_each_key_in_the_every_key_texts_order(self):
        # Every key, the header's two first; a section and each form of the bindings' entries head other keys; the
        # section post_as does not choose is there, as in the every-key text, where it is commented out.
        self.assertEqual(
            [(c.key, c.groups) for c in settings_file.comments(SAMPLE)],
            [
                ("kind", False),
                ("schema_version", False),
                ("service", True),
                ("service.name", False),
                ("service.tag", False),
                ("service.registry", False),
                ("host", True),
                ("host.cpus", False),
                ("host.user_id", False),
                ("host.share", False),
                ("tool", True),
                ("tool.version", False),
                ("tool.package_sha256", False),
                ("repository", True),
                ("repository.name", False),
                ("repository.labels", False),
                ("repository.folders", False),
                ("features", True),
                ("features.reviews", True),
                ("features.reviews.enabled", False),
                ("features.reviews.mode", False),
                ("features.reviews.reviewer_name", False),
                ("features.reviews.effort", False),
                ("features.queue", True),
                ("features.queue.enabled", False),
                ("features.queue.mode", False),
                ("features.queue.notices", True),
                ("features.queue.notices.enabled", False),
                ("features.queue.notices.every_minutes", False),
                ("features.queue.notices.page_url", False),
                ("features.queue.notices.post_as", False),
                ("features.queue.notices.comment", True),
                ("features.queue.notices.comment.header", False),
                ("features.queue.notices.status", True),
                ("features.queue.notices.status.context", False),
                ("integrations", True),
                ("integrations.tracker", True),
                ("integrations.tracker.project", False),
                ("integrations.tracker.address", False),
                ("bindings", False),
                ("bindings.<name>", True),
                ("bindings.<name>.provider", False),
                ("bindings.<name>.address", False),
                ("bindings.<name>.api_key_secret_name", False),
                ("bindings.<name>", True),
                ("bindings.<name>.provider", False),
                ("bindings.<name>.port", False),
            ],
        )

    def test_each_text_is_the_every_key_texts_before_it_is_wrapped(self):
        # In the every-key text's order, each one's words as it writes them, whatever the lines it wraps them in.
        def words(text):
            return re.sub(r"[\s#]+", " ", text).strip()

        every, at = words(EVERY), 0
        for c in settings_file.comments(SAMPLE):
            with self.subTest(key=c.key):
                at = every.index(words(c.text), at)
        texts = {c.key: c.text for c in settings_file.comments(SAMPLE)}
        # The closing sentence, the numbers from the schema, a title before a key's own words, and the line breaks
        # of a description are as the schema has them.
        self.assertEqual(
            texts["host.cpus"],
            "The number of CPUs the service may use, from 1 to 8. Default: 4.",
        )
        self.assertEqual(
            texts["bindings"],
            "Bindings: other models the reviews can use, each under a name you choose. Default: none.",
        )
        self.assertIn('mode:\n  "prototype": it shows you', texts["features"])
        self.assertEqual(
            texts["bindings.<name>.port"], "The port the model listens on. Required."
        )


class TestClosingSentences(unittest.TestCase):
    def test_the_sentence_each_rule_makes(self):
        for node, required, path, sentence in (
            ({}, True, "kind", "Don't change it."),
            (
                {"x-changed-by": "sample migrate updates it"},
                True,
                "schema_version",
                "Don't change it: sample migrate updates it.",
            ),
            ({"default": 4}, True, "host.cpus", "Required."),
            (
                {"x-required-when": "reviews are on"},
                False,
                "a.b",
                "Required when reviews are on.",
            ),
            (
                {
                    "x-required-when": "the queue is live",
                    "x-blank": "to link no page",
                    "default": "",
                },
                False,
                "a.b",
                "Required when the queue is live. Otherwise, leave blank to link no page.",
            ),
            (
                {"x-blank": "for no links", "default": ""},
                False,
                "a.b",
                "Leave blank for no links.",
            ),
            ({"default": "sample"}, False, "a.b", 'Default: "sample".'),
            (
                {"default": None, "x-default-means": "your own user id"},
                False,
                "a.b",
                "Default: null, which means your own user id.",
            ),
            ({"default": 0.5}, False, "a.b", "Default: 0.5."),
            ({"default": {}}, False, "a.b", "Default: none."),
            ({"default": []}, False, "a.b", "Default: none."),
        ):
            with self.subTest(node=node, required=required):
                self.assertEqual(
                    settings_file.closing_sentence(node, required, path), sentence
                )
        with self.assertRaisesRegex(
            ValueError,
            "^host.cpus: no closing sentence: give it a default, x-blank or x-required-when, or require it$",
        ):
            settings_file.closing_sentence({"x-template": 4}, False, "host.cpus")

    def test_where_the_closing_sentence_starts(self):
        for text, parts in (
            ("The tag. Required.", ("The tag.", "Required.")),
            (
                "The page. Required when the queue is live. Otherwise, leave blank to link no page.",
                (
                    "The page.",
                    "Required when the queue is live. Otherwise, leave blank to link no page.",
                ),
            ),
            (
                "The layout. Don't change it: a migrate updates it.",
                ("The layout.", "Don't change it: a migrate updates it."),
            ),
            (
                "Leave blank for the default. Default: 4.",
                ("Leave blank for the default.", "Default: 4."),
            ),
            ("Required.", ("", "Required.")),
            ("The tag, which is required.", ("The tag, which is required.", "")),
            (
                "The tag. Required. Any release.",
                ("The tag. Required. Any release.", ""),
            ),
            ("", ("", "")),
        ):
            with self.subTest(text=text):
                self.assertEqual(settings_file.closing_of(text), parts)

    def test_a_schemas_keys(self):
        keys = [(p, required) for p, _, required in settings_file.schema_keys(schema())]
        self.assertEqual(len(keys), 33)
        self.assertEqual(
            keys[:5],
            [
                ("kind", True),
                ("schema_version", True),
                ("service.name", False),
                ("service.tag", True),
                ("service.registry", False),
            ],
        )
        # A map's entries are NAME.*, each alternative of a oneOf gives its own keys, and no section is a key.
        self.assertEqual(
            keys[-6:],
            [
                ("bindings", False),
                ("bindings.*.provider", True),
                ("bindings.*.address", True),
                ("bindings.*.api_key_secret_name", False),
                ("bindings.*.provider", True),
                ("bindings.*.port", True),
            ],
        )
        self.assertNotIn("features.reviews", [p for p, _ in keys])
        # A list's items are NAME[].
        listed = {
            "properties": {
                "hosts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {"name": {"type": "string"}},
                    },
                }
            }
        }
        self.assertEqual(
            [(p, required) for p, _, required in settings_file.schema_keys(listed)],
            [("hosts", False), ("hosts[].name", True)],
        )

    def test_the_sample_schema_has_them(self):
        self.assertEqual(settings_file.closing_problems(schema()), [])

    def test_what_is_wrong_with_them(self):
        data = schema()
        service = data["properties"]["service"]["properties"]
        host = data["properties"]["host"]["properties"]
        service["name"]["description"] = "The service's name."
        service["tag"]["description"] = 'The release to run. Default: "1.0.0".'
        service["registry"][
            "description"
        ] = "Leave blank if you load the release from a file."
        host["cpus"]["description"] = "Default: 2 or more. The CPUs. Default: 4."
        host["user_id"]["description"] = ""
        del host["share"]["default"]
        self.assertEqual(
            settings_file.closing_problems(data),
            [
                "service.name: its description does not end with its closing sentence, 'Default: \"sample\".'",
                "service.tag: its description ends with 'Default: \"1.0.0\".', not 'Required.'",
                "service.registry: its description says nothing before its closing sentence",
                "host.cpus: its description has a second closing sentence, before 'Default: 4.'",
                "host.user_id: no description",
                "host.share: no closing sentence: give it a default, x-blank or x-required-when, or require it",
            ],
        )


class TestDump(Files):
    def test_the_writers_output(self):
        # Every string in quotes; the feature that is off written in full, as it has a value of its own; the
        # tracker and the bindings, which nothing reads, written too, so that no value is lost.
        self.assertEqual(settings_file.dump(self.read(USER)), DUMP)

    def test_what_it_writes_reads_back_the_same(self):
        doc = self.read(USER)
        again = settings_file.read(self.file(DUMP, "again.yaml"), SAMPLE)
        self.assertEqual(settings_file.dump(again), DUMP)
        self.assertEqual(settings_file.effective(again), settings_file.effective(doc))

    def test_a_feature_that_is_off_is_its_switch_alone(self):
        values = changed(
            {"features.reviews": {"enabled": False}, "integrations.tracker": GONE}
        )
        doc = self.read(as_json(values))
        text = settings_file.dump(doc)
        self.assertIn(
            "  reviews:\n"
            '    # true: review pull requests. false: no reviews. To turn it on, see GUIDE.md, "Turn on a feature". '
            "Required.\n"
            "    enabled: false\n"
            "  # Queue: the merge queue.\n",
            text,
        )
        self.assertIn(
            "  # Tracker: not used while reviews are off.\n  tracker: {}\n", text
        )

    def test_a_templates_profile_reads_back_and_is_written_back(self):
        for profile in ("full", "lean"):
            with self.subTest(profile=profile):
                text = settings_file.template(SAMPLE, profile)
                doc = self.read(text)
                self.assertEqual(doc.profile, profile)
                self.assertEqual(settings_file.dump(doc), text)

    def test_text_in_quotes_with_escapes(self):
        tricky = 'say "hi" \\ #1\u0085\u2028\ufeff \U0001f600 \u00e9'
        doc = Document(SAMPLE, 2, changed({"service.tag": tricky}))
        text = settings_file.dump(doc)
        self.assertIn(
            '\n  tag: "say \\"hi\\" \\\\ #1\\u0085\\u2028\\ufeff \U0001f600 \u00e9"\n',
            text,
        )
        self.assertEqual(
            settings_file.read(self.file(text), SAMPLE).values["service"]["tag"], tricky
        )

    def test_a_list_of_mappings_with_quoted_keys_reads_back(self):
        # Each item's first key in quotes: a word YAML reads as true, a key with a space before a mapping, and a key
        # with a colon.
        data = schema()
        data["properties"]["routes"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": {
                    "type": ["string", "object"],
                    "additionalProperties": {"type": "string"},
                },
            },
            "default": [],
            "description": "The routes. Default: none.",
        }
        file_type = FileType("sample_settings", 2, data)
        routes = [{"yes": "y", "other": "o"}, {"a b": {"no": "n"}}, {"a: b": "c"}]
        text = settings_file.dump(Document(file_type, 2, changed({"routes": routes})))
        self.assertIn(
            '\nroutes:\n  - "yes": "y"\n    other: "o"\n  - "a b":\n      "no": "n"\n  - "a: b": "c"\n',
            text,
        )
        again = settings_file.read(self.file(text), file_type)
        self.assertEqual(again.values["routes"], routes)
        self.assertEqual(settings_file.dump(again), text)

    def test_a_value_that_takes_anything_at_any_depth(self):
        # A map whose entries take any value (an empty schema, as a ticket's required fields): each plain value read
        # as YAML reads it, at every depth, and written back the same.
        data = schema()
        data["properties"]["fields"] = {
            "type": "object",
            "additionalProperties": {},
            "default": {},
            "description": "The fields. Default: none.",
        }
        file_type = FileType("sample_settings", 2, data)
        text = BODY_YAML + (
            "fields:\n  size: 1\n  team:\n    id: 10\n    names: [a, true, 2.5]\n    note: null\n"
            "    owners:\n      - name: b\n        since: 2\n"
        )
        fields = {
            "size": 1,
            "team": {
                "id": 10,
                "names": ["a", True, 2.5],
                "note": None,
                "owners": [{"name": "b", "since": 2}],
            },
        }
        doc = settings_file.read(self.file(text), file_type)
        self.assertEqual(doc.values["fields"], fields)
        self.assertNotIsInstance(doc.values["fields"]["team"]["names"][0], Plain)
        self.assertNotIsInstance(
            doc.values["fields"]["team"]["owners"][0]["name"], Plain
        )
        out = settings_file.dump(doc)
        self.assertIn(
            '\nfields:\n  size: 1\n  team:\n    id: 10\n    names: ["a", true, 2.5]\n    note: null\n'
            '    owners:\n      - name: "b"\n        since: 2\n',
            out,
        )
        again = settings_file.read(self.file(out, "again.yaml"), file_type)
        self.assertEqual(again.values["fields"], fields)
        self.assertEqual(settings_file.dump(again), out)

    def test_numbers_and_lists(self):
        labels = [f"label-{n:02}" for n in range(12)]
        doc = Document(
            SAMPLE, 2, changed({"host.share": 1e-07, "repository.labels": labels})
        )
        text = settings_file.dump(doc)
        self.assertIn("\n  share: 1.0e-07\n", text)
        # Too long for one line, so in block style.
        self.assertIn('\n  labels:\n    - "label-00"\n    - "label-01"\n', text)
        again = settings_file.read(self.file(text), SAMPLE)
        self.assertEqual(again.values["host"]["share"], 1e-07)
        self.assertEqual(again.values["repository"]["labels"], labels)

    def test_what_the_writer_cannot_write(self):
        # Unwritable, a ValueError naming the value's key apart from why, so a caller can name it another way.
        for changes, path, why in (
            (
                {"host.cpu": 2},
                "host.cpu",
                "no key of sample settings, so the writer has no place for it",
            ),
            (
                {"tool.version": Plain("7.10")},
                "tool.version",
                "7.10 is as written, and not typed yet: validate the file first",
            ),
            (
                {"host.share": float("inf")},
                "host.share",
                "inf cannot be written: a number is finite",
            ),
            (
                {"repository.labels": [["a"]]},
                "repository.labels",
                "the writer writes no list of lists",
            ),
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(settings_file.Unwritable) as cm:
                    settings_file.dump(Document(SAMPLE, 2, changed(changes)))
                self.assertIsInstance(cm.exception, ValueError)
                self.assertEqual(
                    (cm.exception.path, cm.exception.why, str(cm.exception)),
                    (path, why, f"{path}: {why}"),
                )


class TestRead(Files):
    def test_plain_values_are_typed_by_the_schema(self):
        doc = self.read(
            "kind: sample_settings\nschema_version: 2\nservice:\n  name: web\n  tag: 1.0.0\n"
            "host:\n  cpus: 2\n  user_id: null\n  share: 0.25\ntool:\n  version: 7.10\n"
        )
        self.assertEqual(
            doc.values["host"], {"cpus": 2, "user_id": None, "share": 0.25}
        )
        self.assertIs(type(doc.values["host"]["cpus"]), int)
        self.assertIs(type(doc.values["service"]["name"]), str)
        # Text where the schema expects text.
        self.assertIs(type(doc.values["service"]["tag"]), str)
        # 7.10 where text is expected: kept as written, for validate to report with the user's own text.
        self.assertEqual(doc.values["tool"]["version"], Plain("7.10"))
        self.assertIsInstance(doc.values["tool"]["version"], Plain)

    def test_a_number_where_text_is_expected_fails_with_the_users_text(self):
        rows = settings_file.validate(
            self.read(BODY_YAML.replace('tag: "1.0.0"', "tag: 1.10"))
        )
        self.assertEqual(
            as_shown(rows),
            [
                (
                    "service.tag",
                    Status.FAIL,
                    "the number 1.1, as written 1.10",
                    f'write it in quotes: set service.tag in {self.where} to "1.10"',
                )
            ],
        )

    def test_an_all_digit_sha256_stays_text(self):
        quoted = self.read(as_json(changed({"tool.package_sha256": SHA})))
        self.assertIs(type(quoted.values["tool"]["package_sha256"]), str)
        self.assertEqual(quoted.values["tool"]["package_sha256"], SHA)
        self.assertEqual(
            [r.status for r in settings_file.validate(quoted)], [Status.PASS]
        )
        self.assertIn(f'\n  package_sha256: "{SHA}"\n', settings_file.dump(quoted))
        # Unquoted, it is kept as written, never as the number, and validate asks for the quotes.
        plain = self.read(BODY_YAML + f"tool:\n  package_sha256: {SHA}\n")
        self.assertEqual(plain.values["tool"]["package_sha256"], Plain(SHA))
        self.assertEqual(
            as_shown(settings_file.validate(plain)),
            [
                (
                    "tool.package_sha256",
                    Status.FAIL,
                    f"the number {int(SHA)}, as written {SHA}",
                    f'write it in quotes: set tool.package_sha256 in {self.where} to "{SHA}"',
                )
            ],
        )

    def test_yaml_and_json_alike(self):
        self.assertEqual(self.read(BODY_YAML).values, BODY)
        self.assertEqual(self.read(as_json(BODY)).values, BODY)
        # A byte order mark is no part of the text.
        self.assertEqual(self.read("\ufeff" + BODY_YAML).values, BODY)

    def test_an_earlier_layout(self):
        exc = self.refused(as_json(BODY, layout=1))
        found = f"{self.where} is in layout 1, older than this package's layout 2"
        self.assertRefused(
            exc, "older", found, f"rkmon migrate --settings {self.where}"
        )

    def test_a_later_layout(self):
        exc = self.refused(as_json(BODY, layout=3))
        found = f"{self.where} is in layout 3, newer than this package's layout 2"
        fix = "rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one"
        self.assertRefused(exc, "newer", found, fix)
        self.assertRefused(
            self.refused(as_json(BODY, layout=3), older=True), "newer", found, fix
        )

    def test_the_other_kind(self):
        exc = self.refused(as_json(BODY, kind="other_settings"))
        found = f'{self.where} is other settings (kind "other_settings"), not sample settings'
        fix = "name your sample settings file instead, or write one: rkmon template sample FILE"
        self.assertRefused(exc, "kind", found, fix)
        exc = self.refused(as_json(BODY), OTHER)
        found = f'{self.where} is sample settings (kind "sample_settings"), not other settings'
        fix = "name your other settings file instead, or write one: rkmon template other FILE"
        self.assertRefused(exc, "kind", found, fix, "other settings file")

    def test_the_fix_that_starts_from_a_new_file(self):
        """start_over, the one fix for a file no reader takes as it is: a new file from the first file type's
        template, each other's in parentheses, named by its kind's first word, then your values copied into it.
        """
        self.assertEqual(
            str(settings_file.start_over(SAMPLE)),
            "start from a new file, rkmon template sample NEW_FILE, and copy your values into it",
        )
        build = settings_file.FileType(
            "build_settings", 2, {}, template_command="template build FILE"
        )
        self.assertEqual(
            str(settings_file.start_over(SAMPLE, build)),
            "start from a new file, rkmon template sample NEW_FILE (a build file: rkmon template build NEW_FILE), "
            "and copy your values into it",
        )

    def test_no_header(self):
        fix = "start from a new file, rkmon template sample NEW_FILE, and copy your values into it"
        exc = self.refused('service:\n  tag: "1.0.0"\n')
        self.assertRefused(
            exc,
            "header",
            f"{self.where} has no kind line, so it is not sample settings",
            fix,
        )
        # Without a kind line, the refusal keeps the file's first key, for migrate to word.
        self.assertEqual(exc.first_key, "service")
        self.assertEqual(self.refused("{}\n").first_key, "")
        found = f"{self.where} has no schema_version line with a whole number, so its layout is unknown"
        for layout in (
            "",
            'schema_version: "2"\n',
            "schema_version: true\n",
            "schema_version: 0\n",
        ):
            with self.subTest(layout=layout):
                exc = self.refused("kind: sample_settings\n" + layout)
                self.assertRefused(exc, "header", found, fix)
                self.assertIsNone(exc.first_key)

    def test_text_that_cannot_be_read(self):
        exc = self.refused(
            "kind: sample_settings\nschema_version: 2\nservice:\n  tag: [1\n"
        )
        fix = f"correct that line in {self.where}, then run {self.again}"
        self.assertRefused(exc, "text", f"{self.where}, line 4: an unclosed [", fix)
        exc = self.refused("- kind\n- sample_settings\n")
        found = f"{self.where}: the top level is not a mapping of keys to values"
        self.assertRefused(exc, "text", found, fix)
        exc = self.refused(
            b"kind: sample_settings\nschema_version: 2\nservice:\n  name: \xff\n"
        )
        self.assertRefused(
            exc,
            "text",
            f"{self.where} is not UTF-8 text",
            f"save {self.where} as UTF-8 text, then run {self.again}",
        )

    def test_the_command_that_reads_the_file_again(self):
        # It ends each fix that holds no command or edit of its own: check, migrate for a file migrate reads, and
        # doctor for a file type no template writes, whose fix puts back the file the monitor wrote.
        unwritten = FileType("sample_settings", 2, SCHEMA_FILE)
        text = "kind: sample_settings\nschema_version: 2\nservice:\n  tag: [1\n"
        for file_type, options, command in (
            (SAMPLE, {}, f"rkmon check --settings {self.where}"),
            (SAMPLE, {"older": True}, f"rkmon migrate --settings {self.where}"),
            (unwritten, {}, "rkmon doctor"),
        ):
            with self.subTest(command=command):
                exc = self.refused(text, file_type, **options)
                self.assertEqual(
                    str(exc.rows[0].fix),
                    f"correct that line in {self.where}, then run {command}",
                )
        fix = (
            "put back the sample settings file the monitor wrote, then run rkmon doctor"
        )
        for text, reason, found in (
            (
                'service:\n  tag: "1.0.0"\n',
                "header",
                f"{self.where} has no kind line, so it is not sample settings",
            ),
            (
                as_json(BODY, kind="other_settings"),
                "kind",
                f'{self.where} is other settings (kind "other_settings"), not sample settings',
            ),
        ):
            with self.subTest(reason=reason):
                self.assertRefused(self.refused(text, unwritten), reason, found, fix)

    def test_half_a_character_is_refused_naming_the_key(self):
        # A \u escape from \ud800 to \udfff without its other half is no character. Each row names the key, never
        # the value; a key that holds one stays out of the names of the rows under it.
        value = (
            f"the value in {self.where} has a \\u escape for half a character (\\ud800 to \\udfff) without its "
            "other half"
        )
        key = value.replace("the value", "a key", 1)
        fix = (
            f"correct that {{}} in {self.where}: write the character itself, or both of its \\u escapes, then "
            f"run {self.again}"
        )
        for text, rows in (
            (
                BODY_YAML.replace('tag: "1.0.0"', 'tag: "1.0\\ud800"'),
                [("service.tag", value, fix.format("value"))],
            ),
            (
                as_json(changed({"service.tag": "1.0\udc00"})),
                [("service.tag", value, fix.format("value"))],
            ),
            (
                as_json(changed({"features.x\ud800": {"y": "\udfff"}})),
                [
                    ("features", key, fix.format("key")),
                    ("features.y", value, fix.format("value")),
                ],
            ),
            (
                BODY_YAML + '"zz\\udfff": 1\n',
                [("sample settings file", key, fix.format("key"))],
            ),
        ):
            with self.subTest(text=text):
                exc = self.refused(text)
                self.assertEqual(exc.reason, "text")
                self.assertEqual(
                    as_shown(exc.rows), [(n, Status.FAIL, f, x) for n, f, x in rows]
                )
        # Both halves make one character.
        doc = self.read(BODY_YAML.replace('tag: "1.0.0"', 'tag: "1.0\\ud83d\\ude00"'))
        self.assertEqual(doc.values["service"]["tag"], "1.0\U0001f600")

    def test_a_file_that_cannot_be_opened(self):
        with self.assertRaises(FileNotFoundError):
            settings_file.read(self.folder / "none.yaml", SAMPLE)

    def test_an_earlier_layout_for_migrate(self):
        doc = self.read(
            "kind: sample_settings\nschema_version: 1\nservice:\n  image: web\n",
            older=True,
        )
        self.assertEqual((doc.version, doc.path), (1, self.folder / "settings.yaml"))
        # As written: its upgrade types it.
        self.assertEqual(doc.values, {"service": {"image": Plain("web")}})

    def test_the_program_name_in_a_fix(self):
        with mock.patch.object(program, "name", return_value="./rkmon"):
            exc = self.refused(as_json(BODY, layout=1))
            unread = self.refused("- kind\n")
        self.assertEqual(
            str(exc.rows[0].fix), f"./rkmon migrate --settings {self.where}"
        )
        self.assertEqual(
            str(unread.rows[0].fix),
            f"correct that line in {self.where}, then run ./rkmon check --settings {self.where}",
        )

    def test_the_profile_is_the_title_lines(self):
        title = '# Sample tool: sample settings (profile "lean": the queue only).\n'
        modeline = "# yaml-language-server: $schema=https://example.com/s.json\n"
        for head, profile in (
            (title, "lean"),
            (modeline + title, "lean"),
            ("\ufeff" + title, "lean"),
            (title.replace("lean", "small"), None),
            (title.replace("Sample tool", "My tool"), None),
            ("# My settings.\n" + title, None),
            ("", None),
        ):
            with self.subTest(head=head):
                self.assertEqual(self.read(head + BODY_YAML).profile, profile)
        # A file type without profiles: none.
        other = BODY_YAML.replace("kind: sample_settings", "kind: other_settings")
        self.assertIsNone(self.read(title + other, OTHER).profile)
        # JSON has no title line.
        self.assertIsNone(self.read(as_json(BODY)).profile)

    def test_a_text_is_read_as_its_file_is(self):
        # loads: the document read gives the file, with no path; messages name the text "your sample settings", and a
        # command in a fix names it FILE.
        text = (
            '# Sample tool: sample settings (profile "lean": the queue only).\n'
            + BODY_YAML
        )
        doc, file = settings_file.loads(text, SAMPLE), self.read(text)
        self.assertEqual(
            (doc.values, doc.version, doc.profile, doc.path),
            (file.values, file.version, file.profile, None),
        )
        again = "rkmon check --settings FILE"
        for text, reason, name, found, fix in (
            (
                "kind: sample_settings\nschema_version: 2\nservice:\n  tag: [1\n",
                "text",
                "sample settings file",
                "your sample settings, line 4: an unclosed [",
                f"correct that line in your sample settings, then run {again}",
            ),
            (
                BODY_YAML.replace('tag: "1.0.0"', 'tag: "1.0\\ud800"'),
                "text",
                "service.tag",
                "the value in your sample settings has a \\u escape for half a character (\\ud800 to \\udfff) "
                "without its other half",
                "correct that value in your sample settings: write the character itself, or both of its \\u "
                f"escapes, then run {again}",
            ),
            (
                as_json(BODY, layout=1),
                "older",
                "sample settings file",
                "your sample settings is in layout 1, older than this package's layout 2",
                "rkmon migrate --settings FILE",
            ),
            (
                as_json(changed({"service.name": "gh" + "p_" + "A" * 36})),
                "secret",
                "service.name",
                "the value in your sample settings looks like a token or key, and no secret goes in a settings file",
                "delete that value from your sample settings, keep the secret in a file only you can read, and run "
                f"{again}",
            ),
        ):
            with self.subTest(reason=reason, name=name):
                with self.assertRaises(Refused) as cm:
                    settings_file.loads(text, SAMPLE)
                self.assertRefused(cm.exception, reason, found, fix, name)
        # Half a character held as it is, not as a \u escape, is no UTF-8 text.
        with self.assertRaises(Refused) as cm:
            settings_file.loads(BODY_YAML.replace("1.0.0", "1.0\ud800"), SAMPLE)
        self.assertRefused(
            cm.exception,
            "text",
            "your sample settings is not UTF-8 text",
            f"save your sample settings as UTF-8 text, then run {again}",
        )


class TestSecrets(Files):
    def test_a_secrets_section(self):
        for section, entries in (
            ("secret_values", {"a": "x1", "b": "y2"}),
            ("secrets", {"a": "x1"}),
        ):
            with self.subTest(section=section):
                exc = self.refused(as_json(changed({section: entries})))
                count = "2 entries" if len(entries) == 2 else "1 entry"
                found = f"{self.where} has a {section} section ({count}), and no secret goes in a settings file"
                fix = (
                    f"move each value into a file only you can read, delete {section} from {self.where}, and run "
                    "check again; after rkmon confirm, rkmon login lists each secret and installs it"
                )
                self.assertRefused(exc, "secret", found, fix)
                # No value shows: none is in the refusal's words, the file's name, which may hold anything, set aside.
                words = str(exc).replace(self.where, "FILE")
                for value in entries.values():
                    self.assertNotIn(value, words)

    def test_an_earlier_layout_may_name_secret_files(self):
        # Layout 1's secrets section names files, which migrate moves; secret_values is refused in every layout.
        doc = self.read(
            as_json({"secrets": {"github_write": "~/github_write"}}, layout=1),
            older=True,
        )
        self.assertEqual(doc.values, {"secrets": {"github_write": "~/github_write"}})
        exc = self.refused(
            as_json({"secret_values": {"a": "x1"}}, layout=1), older=True
        )
        self.assertEqual(exc.reason, "secret")
        # Run the command that read it again: migrate.
        self.assertIn(", and run migrate again;", str(exc.rows[0].fix))

    def test_a_value_that_looks_like_a_secret(self):
        for key, value, what in (
            ("integrations.tracker.project", TOKEN, "a token or key"),
            ("tool.version", "-----" + "BEGIN CERTIFICATE-----", "a PEM block"),
            ("tool.version", "eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl", "a JWT"),
            (
                "service.registry",
                "https://" + "me:pw@registry.example.com/x",
                "an address with a user or password in it",
            ),
            (
                "integrations.tracker.address",
                "https://tracker.example.com/?api_key=" + "s3cr3t",
                "an address with a credential in its query",
            ),
            ("repository.labels", ["sample", TOKEN], "a token or key"),
        ):
            with self.subTest(what=what, key=key):
                exc = self.refused(as_json(changed({key: value})))
                found = f"the value in {self.where} looks like {what}, and no secret goes in a settings file"
                fix = (
                    f"delete that value from {self.where}, keep the secret in a file only you can read, and run "
                    f"{self.again}"
                )
                self.assertRefused(exc, "secret", found, fix, key)
                for text in value if isinstance(value, list) else [value]:
                    if text != "sample":
                        self.assertNotIn(text, str(exc))

    def test_a_key_that_looks_like_a_secret(self):
        exc = self.refused(as_json(changed({"repository.folders": {TOKEN: TOKEN}})))
        self.assertEqual(exc.reason, "secret")
        self.assertEqual(
            [(r.name, r.found) for r in exc.rows],
            [
                (
                    "repository.folders",
                    f"a key in {self.where} looks like a token or key, and no secret goes in a "
                    "settings file",
                ),
                (
                    "repository.folders",
                    f"the value in {self.where} looks like a token or key, and no secret goes in "
                    "a settings file",
                ),
            ],
        )
        self.assertNotIn(TOKEN, str(exc))

    def test_the_one_check_for_secrets(self):
        """secret_texts, which a read refuses on and the production translation stops on: each text once for each
        path and kind, a hidden key (one that looks like a secret, or holds half a character) left out of the paths
        under it."""
        half = "folder\ud800"
        values = {
            "a": {"b": TOKEN, "c": [TOKEN, "sample", TOKEN]},
            TOKEN: {"d": TOKEN},
            half: {"e": TOKEN},
            "f": "sample",
        }
        what = "a token or key"
        self.assertEqual(
            settings_file.secret_texts(values),
            [
                ("a.b", "value", what),
                ("a.c", "value", what),
                ("", "key", what),
                ("d", "value", what),
                ("e", "value", what),
            ],
        )
        self.assertEqual(settings_file.secret_texts({"f": "sample", "g": [1]}), [])
        for key, hidden in ((TOKEN, True), (half, True), ("folder", False), (7, False)):
            with self.subTest(key=key):
                self.assertIs(settings_file.hidden_key(key), hidden)

    def test_a_secrets_name_must_be_one(self):
        binding = {
            "provider": "gateway",
            "address": "https://gateway.example.com",
            "api_key_secret_name": "Big Key!",
        }
        exc = self.refused(as_json(changed({"bindings": {"big": binding}})))
        found = (
            f"the value in {self.where} is not a model's key's secret name (lowercase letters, digits and _, ending "
            "in _api_key)"
        )
        fix = (
            f"set bindings.big.api_key_secret_name in {self.where} to the name of the secret that keeps the key, such "
            'as "my_model_api_key"; rkmon secret NAME installs the key under that name'
        )
        self.assertRefused(
            exc, "secret", found, fix, "bindings.big.api_key_secret_name"
        )
        self.assertNotIn("Big Key", str(exc))
        # A secret's name that does not end in _api_key is refused too: the ignore lists keep a binding's key file
        # out of git and the image by that end.
        binding["api_key_secret_name"] = "big_key"
        exc = self.refused(as_json(changed({"bindings": {"big": binding}})))
        self.assertRefused(
            exc, "secret", found, fix, "bindings.big.api_key_secret_name"
        )
        for name in ("big_api_key", ""):
            binding["api_key_secret_name"] = name
            self.assertEqual(
                self.read(as_json(changed({"bindings": {"big": binding}}))).version, 2
            )

    def test_a_secrets_name_that_holds_a_placeholder_is_left_to_its_row(self):
        """A secret's name that is still the template's ("<name>_api_key") is read, and validate's placeholder row
        alone says what goes there. The look for a secret still reads it: a secret pasted beside it is refused.
        """
        binding = {
            "provider": "gateway",
            "address": "https://gateway.example.com",
            "api_key_secret_name": "<name>_api_key",
        }
        rows = self.rows(changed({"bindings": {"big": binding}}))
        self.assertEqual(
            [(name, status, found) for name, status, found, _ in rows],
            [
                (
                    "bindings.big.api_key_secret_name",
                    Status.FAIL,
                    "still the template's <name>_api_key",
                )
            ],
        )
        binding["api_key_secret_name"] = f"<name> {TOKEN}"
        exc = self.refused(as_json(changed({"bindings": {"big": binding}})))
        found = f"the value in {self.where} looks like a token or key, and no secret goes in a settings file"
        self.assertEqual(exc.reason, "secret")
        self.assertEqual(
            [(r.name, r.found) for r in exc.rows],
            [("bindings.big.api_key_secret_name", found)],
        )
        self.assertNotIn(TOKEN, str(exc))


class TestValidate(Files):
    def test_a_file_that_fits(self):
        self.assertEqual(
            self.rows(BODY),
            [
                (
                    "settings",
                    Status.PASS,
                    "every key fits the sample settings schema, and no <...> is left",
                    None,
                )
            ],
        )

    def test_wrong_values(self):
        w = self.where
        for changes, rows in (
            (
                {"host.cpus": "4"},
                [
                    (
                        "host.cpus",
                        '"4" is text, in quotes, and this key takes a whole number',
                        f"write it without quotes: set host.cpus in {w} to 4",
                    )
                ],
            ),
            (
                {"host.cpus": 9, "host.user_id": 5, "host.share": 1.5},
                [
                    (
                        "host.cpus",
                        "9 is more than 8",
                        f"set host.cpus in {w} to a number from 1 to 8",
                    ),
                    (
                        "host.user_id",
                        "5 is less than 1000",
                        f"set host.user_id in {w} to 1000 or more",
                    ),
                    (
                        "host.share",
                        "1.5 is more than 1",
                        f"set host.share in {w} to a number from 0 to 1",
                    ),
                ],
            ),
            (
                {"host.cpus": [1]},
                [
                    (
                        "host.cpus",
                        "a list is not a whole number",
                        f"set host.cpus in {w} to a number from 1 to 8",
                    )
                ],
            ),
            (
                {"features.reviews.mode": "draft"},
                [
                    (
                        "features.reviews.mode",
                        '"draft" is not "prototype" or "live"',
                        f'set features.reviews.mode in {w} to "prototype" or "live"',
                    )
                ],
            ),
            (
                {"service.name": "Web!"},
                [
                    (
                        "service.name",
                        '"Web!" is not in the form this key takes',
                        f"set service.name in {w} to a value of this form: The service's name. Lowercase letters, "
                        "digits, - and _.",
                    )
                ],
            ),
            (
                {"repository.labels": []},
                [
                    (
                        "repository.labels",
                        "0 entries, and it takes at least 1",
                        f"set repository.labels in {w} to a list of at least 1 entry",
                    )
                ],
            ),
            (
                {
                    "bindings": {
                        "Big": {"provider": "local", "port": 99999},
                        "x": {"provider": "cloud"},
                    }
                },
                [
                    (
                        "bindings.Big",
                        '"Big" is not a name bindings takes',
                        f"rename bindings.Big in {w} to a name bindings takes (other models the reviews can use, "
                        f"each under a name you choose), then run {self.again}",
                    ),
                    (
                        "bindings.x.provider",
                        '"cloud" is not "gateway" or "local"',
                        f'set bindings.x.provider in {w} to "gateway" or "local"',
                    ),
                ],
            ),
            (
                {"bindings": {"small": {"provider": "local", "port": 99999}}},
                [
                    (
                        "bindings.small.port",
                        "99999 is more than 65535",
                        f"set bindings.small.port in {w} to a number from 1 to 65535",
                    )
                ],
            ),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.rows(changed(changes)),
                    [(n, Status.FAIL, f, x) for n, f, x in rows],
                )

    def test_an_entry_out_of_its_form_has_its_lists_or_maps_form(self):
        """A list's item and a map's entry have no description of their own: the fix quotes the list's or the
        map's, which states the form."""
        data = schema()
        repository = data["properties"]["repository"]["properties"]
        repository["labels"]["items"]["pattern"] = "^[a-z-]+$"
        repository["folders"]["additionalProperties"]["pattern"] = "^[a-z/]+$"
        values = changed(
            {
                "repository.labels": ["sample", "Big Label"],
                "repository.folders": {"shared": "Some Folder"},
            }
        )
        w = self.where
        self.assertEqual(
            self.rows(values, FileType("sample_settings", 2, data)),
            [
                (
                    "repository.labels[1]",
                    Status.FAIL,
                    '"Big Label" is not in the form this key takes',
                    f"set repository.labels[1] in {w} to a value of this form: The labels that put a pull request "
                    'in scope. Example: ["my-project"].',
                ),
                (
                    "repository.folders.shared",
                    Status.FAIL,
                    '"Some Folder" is not in the form this key takes',
                    f"set repository.folders.shared in {w} to a value of this form: Labels that several teams "
                    'share, each with a folder. Write each one as "label": "folder".',
                ),
            ],
        )

    def test_a_value_ending_in_a_line_break_is_out_of_a_form_that_ends_in_dollar(self):
        """A pattern's $ is the end of the value, as JSON Schema reads it, where Python's own $ also matches before a
        final line break: a value, and a map entry's name, that ends in one is out of its form.
        """
        broken = changed(
            {"service.name": "web\n", "repository.folders": {"shared\n": "team"}}
        )
        self.assertEqual(
            [(name, status, found) for name, status, found, _ in self.rows(broken)],
            [
                (
                    "service.name",
                    Status.FAIL,
                    '"web\\n" is not in the form this key takes',
                ),
                (
                    "repository.folders.shared\n",
                    Status.FAIL,
                    '"shared\\n" is not a name repository.folders takes',
                ),
            ],
        )
        whole = changed(
            {"service.name": "web", "repository.folders": {"shared": "team"}}
        )
        self.assertEqual(
            [(name, status) for name, status, _, _ in self.rows(whole)],
            [("settings", Status.PASS)],
        )

    def test_schema_pattern_reads_a_dollar_as_json_schema_does(self):
        """A $ is the end of the value, in a group too; an escaped one, and one in a set, is a dollar."""
        for pattern, fits, refused in (
            ("^[a-z]+$", ["web"], ["web\n", "Web"]),
            ("^(?!x$)", ["x\n", "y"], ["x"]),
            (r"^\$[0-9]+$", ["$5"], ["$5\n", "5"]),
            (r"^a\\$", ["a\\"], ["a\\\n"]),
            ("^[$a]+$", ["$a$"], ["$a$\n"]),
            (r"^[\]$]+$", ["]$"], ["]$\n"]),
            (r"^[^\]$]+$", ["ab", "ab\n"], ["a$", "a]"]),
        ):
            compiled = settings_file.schema_pattern(pattern)
            for value in fits:
                self.assertTrue(compiled.search(value), (pattern, value))
            for value in refused:
                self.assertIsNone(compiled.search(value), (pattern, value))

    def test_schema_pattern_reads_digits_and_word_characters_as_json_schema_does(self):
        """\\d and \\w are ASCII's digits and word characters alone, as JSON Schema's are: a digit or a letter of
        another script is neither, where it is to Python's own."""
        for pattern, fits, refused in (
            # ARABIC-INDIC DIGIT THREE, FULLWIDTH DIGIT ONE
            (r"^\d+$", ["0123456789"], ["\u0663", "1\u0663", "\uff11"]),
            # LATIN SMALL LETTER E WITH ACUTE, CYRILLIC SMALL LETTER ZHE
            (r"^\w+$", ["az_AZ09"], ["\u00e9", "a\u00e9", "\u0436", "\u0663"]),
            (r"^[\w.-]+$", ["a.b-c"], ["caf\u00e9"]),
            (
                r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$",
                ["1-02:00:00"],
                ["1-\u0660\u0662:00:00"],
            ),
        ):
            compiled = settings_file.schema_pattern(pattern)
            for value in fits:
                self.assertTrue(compiled.search(value), (pattern, value))
            for value in refused:
                self.assertIsNone(compiled.search(value), (pattern, value))

    def test_schema_pattern_reads_spaces_as_json_schema_does(self):
        """\\s is JSON Schema's spaces, Unicode's and U+FEFF with ASCII's, in a set too, and \\S every other
        character; where Python's own \\s is ASCII's alone with re.ASCII, and without it takes U+001C to U+001F and
        U+0085 too, but not U+FEFF."""
        # ASCII's; NO-BREAK SPACE, OGHAM SPACE MARK, EN QUAD to HAIR SPACE, LINE SEPARATOR, PARAGRAPH SEPARATOR,
        # NARROW NO-BREAK SPACE, MEDIUM MATHEMATICAL SPACE, IDEOGRAPHIC SPACE and ZERO WIDTH NO-BREAK SPACE.
        codes = [
            0x09,
            0x0A,
            0x0B,
            0x0C,
            0x0D,
            0x20,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
        ]
        codes += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF]
        for space in map(chr, codes):
            with self.subTest(space=hex(ord(space))):
                self.assertIsNone(
                    settings_file.schema_pattern(r"^\S+$").search(f"a{space}b")
                )
                self.assertTrue(settings_file.schema_pattern(r"\s").search(space))
                self.assertTrue(settings_file.schema_pattern(r"^[\s]$").search(space))
                self.assertIsNone(
                    settings_file.schema_pattern(r"^[^/\s]+$").search(f"a{space}")
                )
        # INFORMATION SEPARATOR FOUR to ONE and NEXT LINE, which Python's own takes; MONGOLIAN VOWEL SEPARATOR and
        # ZERO WIDTH SPACE, which are no spaces; a NUL; and a character past U+FFFF.
        for other in map(
            chr, (0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0x180E, 0x200B, 0, 0x1F600)
        ):
            with self.subTest(other=hex(ord(other))):
                self.assertTrue(
                    settings_file.schema_pattern(r"^\S+$").search(f"a{other}b")
                )
                self.assertIsNone(settings_file.schema_pattern(r"\s").search(other))
                self.assertIsNone(settings_file.schema_pattern(r"^[\s]$").search(other))
        # In a set, a - beside \s that is no range is a -: first, last, or after a range.
        for pattern in (r"^[\s-]$", r"^[-\s]$", r"^[a-z-\s]$"):
            compiled = settings_file.schema_pattern(pattern)
            for value in ("-", " ", chr(0x3000)):
                self.assertTrue(compiled.search(value), (pattern, value))
            self.assertIsNone(compiled.search("+"), pattern)

    def test_schema_pattern_reads_a_dot_as_json_schema_does(self):
        """Outside a set, . is any character but a line end (\\n, \\r, U+2028 and U+2029), where Python's own takes
        every character but \\n; in a set, it is a dot."""
        dot = settings_file.schema_pattern("^a.b$")
        for end in map(chr, (0x0A, 0x0D, 0x2028, 0x2029)):
            self.assertIsNone(dot.search(f"a{end}b"), hex(ord(end)))
        for other in map(chr, (0x2E, 0x09, 0, 0x85, 0xA0, 0x1F600)):
            self.assertTrue(dot.search(f"a{other}b"), hex(ord(other)))
        in_a_set = settings_file.schema_pattern("^[.]$")
        self.assertTrue(in_a_set.search("."))
        self.assertIsNone(in_a_set.search("a"))

    def test_schema_pattern_refuses_what_pythons_re_reads_otherwise(self):
        """A pattern that holds what Python's re cannot read as JSON Schema does is refused, naming the pattern and
        what in it: \\S in a set, \\s at an end of a range, a set that starts with ] (JSON Schema's [] holds nothing,
        and its [^] everything), an escape, a group, a count or a quantifier that Python's re reads otherwise, or
        half a character, which JSON Schema reads with its other half; and one Python's re cannot read at all, with
        its reason."""
        half = chr(92) + "ud83d"
        for pattern, what in (
            (r"^[\S]+$", r"\S in a set"),
            (r"^[^/\S]$", r"\S in a set"),
            (r"^[a-\s]$", r"\s at an end of a range"),
            (r"^[\s-z]$", r"\s at an end of a range"),
            ("^[]$]+$", "a set that starts with ]"),
            ("^[^]$]+$", "a set that starts with ]"),
            (r"^a\Z", r"the escape \Z"),
            (r"^(a)\1$", r"the escape \1"),
            (r"^\p{L}+$", r"the escape \p"),
            ("(?P<name>a)", "the group (?P"),
            ("(?i)a", "the group (?i"),
            ("^a{,2}$", "the count {,2}"),
            ("^a*+$", "the quantifier *+"),
            ("^a{2}+$", "the quantifier {2}+"),
            (f"^{half}{chr(92)}ude00$", f"half a character, {half}"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError) as cm:
                    settings_file.schema_pattern(pattern)
                self.assertEqual(
                    str(cm.exception),
                    f"the pattern {pattern} has {what}, which Python's re cannot read as JSON Schema does",
                )
        with self.assertRaises(ValueError) as cm:
            settings_file.schema_pattern("^(a$")
        self.assertEqual(
            str(cm.exception),
            "the pattern ^(a$ is not one Python's re reads: missing ), unterminated subpattern",
        )

    def test_unknown_keys_with_the_closest_known_key(self):
        w = self.where
        for changes, row in (
            (
                {"service.nmae": "web"},
                (
                    "service.nmae",
                    "no such key in sample settings; did you mean service.name?",
                    f"rename service.nmae in {w} to service.name, or delete service.nmae from {w}",
                ),
            ),
            (
                {"cpus": 2},
                (
                    "cpus",
                    "no such key in sample settings; did you mean host.cpus?",
                    f"move cpus in {w} to host.cpus, or delete cpus from {w}",
                ),
            ),
            (
                {"zzz": 2},
                ("zzz", "no such key in sample settings", f"delete zzz from {w}"),
            ),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.rows(changed(changes)), [(row[0], Status.FAIL, row[1], row[2])]
                )

    def test_missing_keys(self):
        w = self.where
        for changes, rows in (
            (
                {"service.tag": GONE},
                [("service.tag", "missing", f"set service.tag in {w} to text")],
            ),
            (
                {"service": GONE},
                [
                    (
                        "service",
                        "missing",
                        f"set service in {w} to its keys, as the template writes them",
                    )
                ],
            ),
            # Required when its feature is on, and it is: one row for the section of the file it is missing from,
            # which the section command prints in full. The printed text writes the reviews on, so no
            # switch is set back.
            (
                {"features.reviews.mode": GONE},
                [
                    (
                        "reviews",
                        "features.reviews is on, and features.reviews.mode is missing",
                        f"run rkmon template section features.reviews, paste its text over features.reviews in {w}, "
                        f"then {FILL_IN}",
                    )
                ],
            ),
            (
                {"integrations.tracker.project": GONE},
                [
                    (
                        "tracker",
                        "features.reviews is on, and integrations.tracker.project is missing",
                        "run rkmon template section integrations.tracker, paste its text over integrations.tracker "
                        f"in {w}, then {FILL_IN}",
                    )
                ],
            ),
            # The feature's switch alone: its section's keys, and the keys of the parts it leaves out.
            (
                {"features.queue.enabled": True},
                [
                    (
                        "queue",
                        "features.queue is on, and features.queue.mode and features.queue.notices.enabled are "
                        "missing",
                        f"run rkmon template section features.queue, paste its text over features.queue in {w}, "
                        f"then {FILL_IN}",
                    )
                ],
            ),
            # A switch in the section that the file has on, and the printed text writes off, is set back to true.
            (
                {"features.queue": {"enabled": True, "notices": {"enabled": True}}},
                [
                    (
                        "queue",
                        "features.queue is on, and features.queue.mode is missing",
                        f"run rkmon template section features.queue, paste its text over features.queue in {w}, "
                        f"then set notices.enabled back to true and {FILL_IN}",
                    )
                ],
            ),
            # No section of the file around it: nothing to paste over. It is set to where its value comes from.
            (
                {"integrations": GONE},
                [
                    (
                        "integrations.tracker.project",
                        "missing, and features.reviews is on",
                        f'set integrations.tracker.project in {w} to your tracker\'s project key, such as "SAMPLE"',
                    )
                ],
            ),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.rows(changed(changes)),
                    [(n, Status.FAIL, f, x) for n, f, x in rows],
                )

    def test_missing_keys_without_a_section_command(self):
        # The build file's form: its template writes every key, so each missing key has its own row.
        w, file_type = self.where, FileType("sample_settings", 2, SCHEMA_FILE)
        for changes, rows in (
            (
                {"features.reviews.mode": GONE},
                [
                    (
                        "features.reviews.mode",
                        "missing, and features.reviews is on",
                        f'set features.reviews.mode in {w} to "prototype" or "live"',
                    )
                ],
            ),
            (
                {"features.queue.enabled": True},
                [
                    (
                        "features.queue.mode",
                        "missing, and features.queue is on",
                        f'set features.queue.mode in {w} to "prototype" or "live"',
                    ),
                    (
                        "features.queue.notices.enabled",
                        "missing, and features.queue is on",
                        f"set features.queue.notices.enabled in {w} to true or false",
                    ),
                ],
            ),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.rows(changed(changes), file_type),
                    [(n, Status.FAIL, f, x) for n, f, x in rows],
                )

    def test_a_key_missing_from_a_maps_entry(self):
        # A map's entry is no section a section command prints, so the key has its own row.
        data = schema()
        key = data["$defs"]["binding"]["oneOf"][0]["properties"]["api_key_secret_name"]
        del key["default"], key["x-blank"]
        key["x-required-when"] = "the reviews are on"
        file_type = FileType(
            "sample_settings", 2, data, section_command="template section KEY"
        )
        values = changed(
            {
                "bindings": {
                    "big": {"provider": "gateway", "address": "https://example.com"}
                }
            }
        )
        self.assertEqual(
            self.rows(values, file_type),
            [
                (
                    "bindings.big.api_key_secret_name",
                    Status.FAIL,
                    "missing, and features.reviews is on",
                    f"set bindings.big.api_key_secret_name in {self.where} to text",
                )
            ],
        )

    def test_the_section_is_named_as_the_program_was_run(self):
        with mock.patch.object(sys, "argv", ["./rkmon"]):
            rows = self.rows(changed({"features.reviews.mode": GONE}))
        self.assertEqual(
            rows[0][3],
            "run ./rkmon template section features.reviews, paste its text over features.reviews in "
            f"{self.where}, then {FILL_IN}",
        )

    def test_the_section_commands_profile_decides_the_switches_to_set_back(self):
        """The section command prints the every-key text for the schema's first profile: a switch that profile writes
        off, which the file has on, is set back to true, named from the section."""
        data = schema()
        profiles = data["x-profiles"]
        # "lean" first: it writes the reviews off.
        data["x-profiles"] = {"lean": profiles["lean"], "full": profiles["full"]}
        file_type = FileType(
            "sample_settings", 2, data, section_command="template section KEY"
        )
        self.assertEqual(
            self.rows(changed({"features.reviews.mode": GONE}), file_type),
            [
                (
                    "reviews",
                    Status.FAIL,
                    "features.reviews is on, and features.reviews.mode is missing",
                    "run rkmon template section features.reviews, paste its text over features.reviews in "
                    f"{self.where}, then set enabled back to true and {FILL_IN}",
                )
            ],
        )

    def test_what_may_be_left_out(self):
        for changes in (
            # Off, so its mode and its parts' switches are unused.
            {"features.reviews": {"enabled": False}, "integrations.tracker": {}},
            # A key with a default, or a blank, means it when it is left out, whatever its feature.
            {
                "features.queue": {
                    "enabled": True,
                    "mode": "live",
                    "notices": {"enabled": True},
                }
            },
            {"host": {}, "tool": {}},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    [r[1] for r in self.rows(changed(changes))], [Status.PASS]
                )

    def test_placeholders_name_where_their_value_comes_from(self):
        w = self.where
        rows = self.rows(
            changed(
                {
                    "repository.name": "<org>/<repo>",
                    "repository.labels": ["sample", "<label>"],
                }
            )
        )
        self.assertEqual(
            rows,
            [
                (
                    "repository.name",
                    Status.FAIL,
                    "still the template's <org>/<repo>",
                    f"set repository.name in {w} to the repository's name, as <org>/<repo>",
                ),
                # No x-source: what it takes.
                (
                    "repository.labels",
                    Status.FAIL,
                    "still the template's <label>",
                    f"set repository.labels in {w} to a list",
                ),
            ],
        )
        # A map's entry named by a placeholder, in a map with no rule for names: a name of your own.
        data = schema()
        del data["properties"]["repository"]["properties"]["folders"]["propertyNames"]
        file_type = FileType(
            "sample_settings", 2, data, template_command=SAMPLE.template_command
        )
        self.assertEqual(
            self.rows(
                changed({"repository.folders": {"<label>": "projects/sample"}}),
                file_type,
            ),
            [
                (
                    "repository.folders",
                    Status.FAIL,
                    "still the template's <label>",
                    f"rename repository.folders.<label> in {w} to a name of your own, then run {self.again}",
                )
            ],
        )

    def test_a_placeholder_has_its_row_alone_not_its_forms_too(self):
        """A value that holds a placeholder is not checked against its key's form (its pattern), which it would fail:
        the placeholder's row alone says what to put there, one row for the key."""
        rows = self.rows(
            changed({"service.name": "<name>", "tool.package_sha256": "<sha256>"})
        )
        self.assertEqual(
            [(name, status, found) for name, status, found, _ in rows],
            [
                ("service.name", Status.FAIL, "still the template's <name>"),
                ("tool.package_sha256", Status.FAIL, "still the template's <sha256>"),
            ],
        )

    def test_a_placeholder_has_its_row_alone_whatever_its_key_takes(self):
        """A value that holds a placeholder is not checked for its type, its value or its bounds either, whatever its
        key takes (a whole number, one of a list of values, true or false, a number), in quotes or not: the
        placeholder's row alone, one row for the key."""
        queue = {"enabled": True, "mode": "prototype", "notices": {"enabled": "<on>"}}
        rows = self.rows(
            changed(
                {
                    "host.cpus": "<cpus>",
                    "host.user_id": "<uid>",
                    "features.reviews.effort": "<effort>",
                    "features.queue": queue,
                }
            )
        )
        self.assertEqual(
            sorted([(name, status, found) for name, status, found, _ in rows]),
            [
                (
                    "features.queue.notices.enabled",
                    Status.FAIL,
                    "still the template's <on>",
                ),
                (
                    "features.reviews.effort",
                    Status.FAIL,
                    "still the template's <effort>",
                ),
                ("host.cpus", Status.FAIL, "still the template's <cpus>"),
                ("host.user_id", Status.FAIL, "still the template's <uid>"),
            ],
        )
        # Unquoted where a number goes, it stays as written (Plain), as any value its key cannot take does.
        doc = self.read(BODY_YAML + "host:\n  share: <share>\n")
        self.assertIsInstance(doc.values["host"]["share"], Plain)
        self.assertEqual(
            [(r.name, r.status, r.found) for r in settings_file.validate(doc)],
            [("host.share", Status.FAIL, "still the template's <share>")],
        )

    def test_a_map_entry_named_by_a_placeholder_has_its_row_alone(self):
        """An entry whose name holds a placeholder has the placeholder's row alone, rename it, and no row beside it
        for the map's rule for names, which the name breaks. What the entry holds is checked as any entry's is: one
        row for each key."""
        w = self.where
        rename = f"to a name of your own, then run {self.again}"
        self.assertEqual(
            self.rows(changed({"repository.folders": {"<label>": "projects/sample"}})),
            [
                (
                    "repository.folders",
                    Status.FAIL,
                    "still the template's <label>",
                    f"rename repository.folders.<label> in {w} {rename}",
                )
            ],
        )
        binding = {
            "provider": "gateway",
            "address": "<address>",
            "api_key_secret_name": "<name>_api_key",
        }
        rows = self.rows(changed({"bindings": {"<name>": binding}}))
        self.assertEqual(
            [(name, status, found) for name, status, found, _ in rows],
            [
                ("bindings", Status.FAIL, "still the template's <name>"),
                (
                    "bindings.<name>.address",
                    Status.FAIL,
                    "still the template's <address>",
                ),
                (
                    "bindings.<name>.api_key_secret_name",
                    Status.FAIL,
                    "still the template's <name>_api_key",
                ),
            ],
        )
        self.assertEqual(rows[0][3], f"rename bindings.<name> in {w} {rename}")
        rows = self.rows(
            changed({"bindings": {"<name>": {"provider": "local", "port": 0}}})
        )
        self.assertEqual(
            [(name, found) for name, _, found, _ in rows],
            [
                ("bindings.<name>.port", "0 is less than 1"),
                ("bindings", "still the template's <name>"),
            ],
        )

    def test_a_value_set_for_a_feature_that_is_off(self):
        w = self.where
        values = changed(
            {
                "features.reviews": {"enabled": False, "mode": "live"},
                "bindings": {"m": {"provider": "local", "port": 8080}},
            }
        )
        fix = f"or turn it on: set features.reviews.enabled in {w} to true"
        self.assertEqual(
            self.rows(values),
            [
                (
                    "settings",
                    Status.PASS,
                    "every key fits the sample settings schema, and no <...> is left",
                    None,
                ),
                (
                    "features.reviews.mode",
                    Status.WARN,
                    "set, and features.reviews is off: nothing reads it",
                    f"delete features.reviews.mode from {w}, {fix}",
                ),
                (
                    "integrations.tracker.project",
                    Status.WARN,
                    "set, and features.reviews is off: nothing reads it",
                    f"delete integrations.tracker.project from {w}, {fix}",
                ),
                (
                    "bindings",
                    Status.WARN,
                    "set, and features.reviews is off: nothing reads it",
                    f"delete bindings from {w}, {fix}",
                ),
            ],
        )
        # A value the template writes anyway is no value of the user's own.
        self.assertEqual(
            len(
                self.rows(
                    changed(
                        {
                            "features.reviews": {"enabled": False, "mode": "prototype"},
                            "integrations.tracker": {},
                        }
                    )
                )
            ),
            1,
        )

    def test_a_value_that_several_features_read(self):
        w = self.where
        data = schema()
        data["properties"]["bindings"]["x-used-by"] = [
            "features.reviews",
            "features.queue",
        ]
        file_type = FileType("sample_settings", 2, data)
        values = changed(
            {
                "features.reviews": {"enabled": False},
                "integrations.tracker": {},
                "bindings": {"m": {"provider": "local", "port": 8080}},
            }
        )
        self.assertEqual(
            self.rows(values, file_type)[1:],
            [
                (
                    "bindings",
                    Status.WARN,
                    "set, and features.reviews and features.queue are off: nothing reads it",
                    f"delete bindings from {w}, or turn one of them on: set features.reviews.enabled in {w} to "
                    f"true or set features.queue.enabled in {w} to true",
                )
            ],
        )

    def test_the_checks_across_keys_once_every_key_fits(self):
        seen = []

        def coherence(doc):
            seen.append(doc.values)
            return [
                Check(
                    "pair",
                    Status.WARN,
                    "x and y differ",
                    Edit("y", "FILE", Described("x")),
                )
            ]

        file_type = FileType("sample_settings", 2, SCHEMA_FILE, coherence=coherence)
        self.assertEqual(
            self.rows(BODY, file_type),
            [
                (
                    "settings",
                    Status.PASS,
                    "every key fits the sample settings schema, and no <...> is left",
                    None,
                ),
                ("pair", Status.WARN, "x and y differ", "set y in FILE to x"),
            ],
        )
        self.assertEqual(seen, [BODY])
        rows = self.rows(changed({"host.cpus": 9}), file_type)
        self.assertEqual(
            rows[-1],
            (
                "coherence",
                Status.SKIP,
                "checked once every key fits the schema and no <...> is left",
                None,
            ),
        )
        self.assertEqual(len(seen), 1)

    def test_misfits_are_validates_rows_for_the_values_the_file_holds(self):
        """misfits: validate's FAIL rows for the values the file holds that their keys do not take, and no other: no
        row for a key the file leaves out, nor for a placeholder."""
        doc = self.read(
            as_json(
                changed(
                    {
                        "service.tag": GONE,  # required
                        "service.name": "Web!",
                        "repository.name": "<org>/<repo>",
                        "repository.labels": [],
                        "features.reviews.mode": GONE,  # required while the reviews are on
                        "host.cpus": 9,
                        "bindings": {
                            "Big": {"provider": "local", "port": 8080},
                            # No provider: no kind of binding takes it.
                            "x": {"port": 8080},
                        },
                        "zzz": 2,
                    }
                )
            )
        )
        held = [
            ("service.name", '"Web!" is not in the form this key takes'),
            ("repository.labels", "0 entries, and it takes at least 1"),
            ("host.cpus", "9 is more than 8"),
            ("bindings.Big", '"Big" is not a name bindings takes'),
            ("bindings.x.provider", "missing"),
            ("zzz", "no such key in sample settings"),
        ]
        misfits = as_shown(settings_file.misfits(doc))
        self.assertEqual(
            [(n, s, f) for n, s, f, _ in misfits],
            [(n, Status.FAIL, f) for n, f in held],
        )
        rows = as_shown(settings_file.validate(doc))
        self.assertEqual([r for r in rows if (r[0], r[2]) in held], misfits)
        # Validate's others: the keys the file leaves out, and the placeholder.
        self.assertEqual(
            [(n, f) for n, _, f, _ in rows if (n, f) not in held],
            [
                ("service.tag", "missing"),
                (
                    "reviews",
                    "features.reviews is on, and features.reviews.mode is missing",
                ),
                ("repository.name", "still the template's <org>/<repo>"),
            ],
        )
        self.assertEqual(settings_file.misfits(self.read(as_json(BODY))), [])


class TestEffective(Files):
    def test_every_key_with_its_default_for_a_blank(self):
        doc = self.read(
            as_json(changed({"service.registry": "", "host.cpus": None, "zzz": 1}))
        )
        self.assertEqual(
            settings_file.effective(doc),
            {
                "service": {"name": "sample", "tag": "1.0.0", "registry": ""},
                "host": {"cpus": 4, "user_id": None, "share": 0.5},
                "tool": {"version": "", "package_sha256": ""},
                "repository": {
                    "name": "my-org/my-repo",
                    "labels": ["sample"],
                    "folders": {},
                },
                "features": {
                    "reviews": {
                        "enabled": True,
                        "mode": "prototype",
                        "reviewer_name": "",
                        "effort": "high",
                    },
                    "queue": {
                        "enabled": False,
                        "mode": None,
                        "notices": {
                            "enabled": None,
                            "every_minutes": 30,
                            "page_url": "",
                            "post_as": "comment",
                            "comment": {"header": ""},
                            "status": {"context": "sample/queue"},
                        },
                    },
                },
                "integrations": {"tracker": {"project": "SAMPLE", "address": ""}},
                "bindings": {},
            },
        )

    def test_empty_text_gives_a_default_that_is_not_blank(self):
        doc = self.read(
            as_json(changed({"service.name": "", "features.reviews.effort": ""}))
        )
        values = settings_file.effective(doc)
        self.assertEqual(values["service"]["name"], "sample")
        self.assertEqual(values["features"]["reviews"]["effort"], "high")

    def test_a_maps_entries_get_their_defaults_too(self):
        big = {"provider": "gateway", "address": "https://gateway.example.com"}
        doc = self.read(as_json(changed({"bindings": {"big": big}})))
        self.assertEqual(
            settings_file.effective(doc)["bindings"],
            {"big": {**big, "api_key_secret_name": ""}},
        )


class TestFingerprint(unittest.TestCase):
    def test_the_sha256_of_canonical_json(self):
        values = {"b": [1, 2.0, "\u00e9"], "a": {"y": None, "x": True}}
        canonical = '{"a":{"x":true,"y":null},"b":[1,2,"\u00e9"]}'
        self.assertEqual(
            settings_file.fingerprint(values),
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            settings_file.fingerprint({}),
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        )

    def test_the_same_values_the_same_fingerprint(self):
        a = settings_file.fingerprint({"x": 1, "y": [1.0, (2, 3)]})
        self.assertEqual(a, settings_file.fingerprint({"y": [1, [2, 3]], "x": 1.0}))
        self.assertNotEqual(a, settings_file.fingerprint({"x": 1, "y": [1, [3, 2]]}))
        self.assertNotEqual(a, settings_file.fingerprint({"x": "1", "y": [1, [2, 3]]}))

    def test_what_it_refuses(self):
        for values, error in (
            ({"x": float("nan")}, ValueError),
            ({"x": float("inf")}, ValueError),
            ({1: "x"}, TypeError),
            ({"x": {1, 2}}, TypeError),
        ):
            with self.subTest(values=values), self.assertRaises(error):
                settings_file.fingerprint(values)


def tag_from_image(values, file_type):
    """Layout 1 to 2 for the tests: service.image is service.tag."""
    service = values.get("service") or {}
    if "image" in service:
        service["tag"] = service.pop("image")
    return values, ["service.image is service.tag"]


UPGRADED = FileType("sample_settings", 2, SCHEMA_FILE, upgrades={1: tag_from_image})


class TestUpgrade(Files):
    OLD = "kind: sample_settings\nschema_version: 1\nservice:\n  image: web\nhost:\n  cpus: 2\n"

    def test_each_step_in_turn(self):
        old = self.read(self.OLD, UPGRADED, older=True)
        new, notes = settings_file.upgrade(old)
        self.assertEqual(
            (new.file_type, new.version, new.path), (UPGRADED, 2, old.path)
        )
        self.assertEqual(new.values, {"service": {"tag": "web"}, "host": {"cpus": 2}})
        # Typed by this layout's schema.
        self.assertIs(type(new.values["host"]["cpus"]), int)
        self.assertEqual(notes, ["service.image is service.tag"])
        self.assertIsNone(new.profile)
        # The bytes read read, carried for rewrite.
        self.assertEqual((old.data, new.data), (self.OLD.encode(), self.OLD.encode()))
        # The document it was given is as it was.
        self.assertEqual(
            old.values,
            {"service": {"image": Plain("web")}, "host": {"cpus": Plain("2")}},
        )

    def test_a_steps_moved_notes_are_the_documents_moves(self):
        def step(values, file_type):
            values["service"]["tag"] = values["service"].pop("image")
            moved = settings_file.Moved("service.image", "service.tag")
            return values, [moved, "host.cpus: as it was"]

        stepped = FileType("sample_settings", 2, SCHEMA_FILE, upgrades={1: step})
        new, notes = settings_file.upgrade(self.read(self.OLD, stepped, older=True))
        self.assertEqual(new.moves, ({"service.tag": "service.image"},))
        # Every note comes back as plain text.
        self.assertEqual(
            notes, ["service.image is now service.tag", "host.cpus: as it was"]
        )
        self.assertEqual([type(n) for n in notes], [str, str])
        # A document read, not upgraded, has none.
        self.assertEqual(self.read(as_json(BODY), SAMPLE).moves, ())

    def test_a_moved_note(self):
        moved = settings_file.Moved("service.image", "service.tag")
        self.assertEqual(
            (moved, moved.old, moved.new),
            ("service.image is now service.tag", "service.image", "service.tag"),
        )
        said = settings_file.Moved("a", "b", "a is now b, by its own words")
        self.assertEqual(
            (said, said.old, said.new), ("a is now b, by its own words", "a", "b")
        )
        # A copy keeps both keys.
        again = copy.deepcopy(said)
        self.assertEqual((again, again.old, again.new), (said, "a", "b"))

    def test_the_key_the_file_had(self):
        # Two steps' moves: the last step's are followed first, each through the nearest move around the key.
        doc = Document(
            SAMPLE, 2, {}, moves=({"a.b": "x", "c": "y.z"}, {"d": "a.b", "d.e": "f"})
        )
        for path, key in (
            ("d.e.g", "f"),
            ("d.h", "x"),
            ("d", "x"),
            ("d[0]", "x"),
            ("c.k", "y.z"),
            ("dd", "dd"),
            ("q.r", "q.r"),
        ):
            with self.subTest(path=path):
                self.assertEqual(settings_file.earlier_key(doc, path), key)

    def test_each_step_is_given_the_file_type_it_upgrades_to(self):
        given = []

        def step(values, file_type):
            given.append(file_type)
            return values, []

        upgraded = FileType("sample_settings", 2, SCHEMA_FILE, upgrades={1: step})
        settings_file.upgrade(Document(upgraded, 1, {}))
        self.assertEqual(given, [upgraded])

    def test_the_profile_is_kept(self):
        title = '# Sample tool: sample settings (profile "lean": the queue only).\n'
        old = self.read(title + self.OLD, UPGRADED, older=True)
        self.assertEqual(old.profile, "lean")
        self.assertEqual(settings_file.upgrade(old)[0].profile, "lean")

    def test_this_layout_comes_back_as_it_is(self):
        doc = self.read(as_json(BODY), UPGRADED)
        self.assertEqual(settings_file.upgrade(doc), (doc, []))

    def test_a_missing_step_and_a_newer_layout(self):
        later = FileType(
            "sample_settings", 3, SCHEMA_FILE, upgrades={2: tag_from_image}
        )
        with self.assertRaisesRegex(ValueError, "no step from layout 1 to layout 2"):
            settings_file.upgrade(Document(later, 1, {}))
        with self.assertRaisesRegex(
            ValueError, "layout 3 is newer than this package's 2"
        ):
            settings_file.upgrade(Document(SAMPLE, 3, {}))

    def test_a_step_that_leaves_a_secret(self):
        leaky = FileType(
            "sample_settings",
            2,
            SCHEMA_FILE,
            upgrades={1: lambda v, f: ({"secret_values": {"a": "b"}}, [])},
        )
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(Document(leaky, 1, {}))
        self.assertEqual(cm.exception.reason, "secret")
        # A document with no path: its fix runs migrate on FILE, yours to fill in.
        shaped = FileType(
            "sample_settings",
            2,
            SCHEMA_FILE,
            upgrades={1: lambda v, f: ({"tool": {"version": TOKEN}}, [])},
            template_command=SAMPLE.template_command,
        )
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(Document(shaped, 1, {}))
        self.assertEqual(
            str(cm.exception.rows[0].fix),
            "delete that value from your sample settings, keep the secret in a file only you can read, and run "
            "rkmon migrate --settings FILE",
        )

    def test_a_step_that_refuses(self):
        # The step words the row; upgrade knows the file, so the fix names it.
        def refusing(values, file_type):
            raise settings_file.StepRefused(
                "service.image",
                "service.image has no place in layout 2",
                lambda where: Edit("service.image", where, DELETE),
            )

        strict = FileType("sample_settings", 2, SCHEMA_FILE, upgrades={1: refusing})
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(self.read(self.OLD, strict, older=True))
        self.assertRefused(
            cm.exception,
            "step",
            "service.image has no place in layout 2",
            f"delete service.image from {self.where}",
            name="service.image",
        )
        # A document with no path: the fix names the kind of file.
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(Document(strict, 1, {}))
        self.assertEqual(
            str(cm.exception.rows[0].fix),
            "delete service.image from your sample settings",
        )


class TestRewrite(Files):
    """rewrite through private_files itself, on a file of yours only in the test's own folder, outside the package,
    git and cloud sync, as migrate's check leaves it."""

    OLD_TEXT = "# My own comment.\n" + TestUpgrade.OLD

    def setUp(self):
        super().setUp()
        self.path = self.file(self.OLD_TEXT)
        self.path.chmod(0o600)
        self.kept = self.folder / "settings.yaml.v1"
        self.new = settings_file.upgrade(
            settings_file.read(self.path, UPGRADED, older=True)
        )[0]
        # The command you ran, naming the file at another path (private_files.write's RETRY).
        self.retry = private_files.again(
            ("rkmon", "migrate", "--settings", self.where), self.where
        )

    def rewrite(self, doc=None, dry_run=False):
        """rewrite's result, or the refusal it raised; it prints nothing."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            try:
                result = settings_file.rewrite(
                    self.path, doc or self.new, self.retry, dry_run=dry_run
                )
            except ValueError as exc:
                result = exc
        self.assertEqual(printed.getvalue(), "")
        return result

    def texts(self):
        """The file's text, and the kept file's (None when there is none)."""
        kept = self.kept.read_text(encoding="utf-8") if self.kept.exists() else None
        return self.path.read_text(encoding="utf-8"), kept

    def modes(self):
        return [stat.S_IMODE(p.stat().st_mode) for p in (self.path, self.kept)]

    def test_the_old_file_kept_then_the_new_one_written(self):
        self.assertEqual(self.rewrite(), self.kept)
        self.assertEqual(self.texts(), (settings_file.dump(self.new), self.OLD_TEXT))
        self.assertEqual(self.modes(), [0o600, 0o600])

    def test_a_kept_file_that_holds_the_file_already_stays(self):
        # As a migrate that stopped part-way leaves it: run again, it goes on, and the kept file is made yours only.
        self.kept.write_text(self.OLD_TEXT, encoding="utf-8")
        self.kept.chmod(0o644)
        self.assertEqual(self.rewrite(), self.kept)
        self.assertEqual(self.texts(), (settings_file.dump(self.new), self.OLD_TEXT))
        self.assertEqual(self.modes(), [0o600, 0o600])

    def test_its_rows_show_each_file_as_migrate_shows_it(self):
        # A link put in the file's place once migrate has read it: write's row names the file as migrate's lines do.
        with mock.patch.dict(os.environ, {"HOME": str(self.folder)}):
            copy = self.file(self.OLD_TEXT, "copy.yaml")
            self.path.unlink()
            self.path.symlink_to(copy)
            refused = self.rewrite()
        self.assertIsInstance(refused, private_files.Refused)
        self.assertEqual(
            [(r.found, str(r.fix)) for r in refused.rows],
            [
                (
                    "~/settings.yaml is a link, not a file",
                    f"name the file it points to: rkmon migrate --settings {os.path.realpath(copy)}",
                )
            ],
        )

    def test_a_kept_file_that_holds_another_is_never_written_over(self):
        self.kept.write_text("kind: sample_settings\n", encoding="utf-8")
        refused = self.rewrite()
        self.assertIsInstance(refused, private_files.Refused)
        self.assertEqual(
            [(r.name, r.status, r.found, str(r.fix)) for r in refused.rows],
            [
                (
                    "sample settings file",
                    Status.FAIL,
                    f"{self.where}.v1 is another file, and a kept file is never written over",
                    f"mv {self.where}.v1 {self.where}.v1.old, then rkmon migrate --settings {self.where}",
                )
            ],
        )
        self.assertEqual(self.texts(), (self.OLD_TEXT, "kind: sample_settings\n"))

    def made_meanwhile(self, function: str, text: str):
        """private_files' FUNCTION, patched to make TEXT the kept file first: a file made there once rewrite has
        looked, before write's own look ("write") or once write has looked ("_put")."""
        real = getattr(private_files, function)

        def first(path, *args, **kwargs):
            if pathlib.Path(path) == self.kept and not self.kept.exists():
                self.kept.write_text(text, encoding="utf-8")
            return real(path, *args, **kwargs)

        return mock.patch.object(private_files, function, first)

    def test_a_kept_file_made_meanwhile_is_never_written_over(self):
        # The refusal for a kept file found there: its fix never names a file of another name.
        for function in ("write", "_put"):
            with self.subTest(made_in=function):
                self.kept.unlink(missing_ok=True)
                with self.made_meanwhile(function, "kind: sample_settings\n"):
                    refused = self.rewrite()
                self.assertIsInstance(refused, private_files.Refused)
                self.assertEqual(
                    [(r.found, str(r.fix)) for r in refused.rows],
                    [
                        (
                            f"{self.where}.v1 is another file, and a kept file is never written over",
                            f"mv {self.where}.v1 {self.where}.v1.old, then rkmon migrate --settings {self.where}",
                        )
                    ],
                )
                self.assertEqual(
                    self.texts(), (self.OLD_TEXT, "kind: sample_settings\n")
                )

    def test_a_kept_file_made_meanwhile_that_holds_the_file_stays(self):
        # As a second migrate of the same file leaves it: rewrite goes on, and the kept file is made yours only.
        with self.made_meanwhile("write", self.OLD_TEXT):
            self.assertEqual(self.rewrite(), self.kept)
        self.assertEqual(self.texts(), (settings_file.dump(self.new), self.OLD_TEXT))
        self.assertEqual(self.modes(), [0o600, 0o600])

    def test_a_file_that_changed_since_it_was_read_is_never_written(self):
        # A change made while migrate runs stays in the file as you made it, and nothing is kept.
        changed = self.OLD_TEXT.replace("cpus: 2", "cpus: 3")
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                self.file(changed)
                refused = self.rewrite(dry_run=dry_run)
                self.assertIsInstance(refused, ValueError)
                self.assertNotIsInstance(refused, (Refused, private_files.Refused))
                self.assertEqual(
                    str(refused), f"{self.where} changed after migrate read it"
                )
                self.assertEqual(self.texts(), (changed, None))
        # A document not read from the file: rewrite cannot tell whether the file changed, so it writes nothing.
        self.file(self.OLD_TEXT)
        refused = self.rewrite(dataclasses.replace(self.new, data=None))
        self.assertIsInstance(refused, ValueError)
        self.assertRegex(str(refused), "rewrite writes a document read from the file")
        self.assertEqual(self.texts(), (self.OLD_TEXT, None))

    def test_the_old_files_header_is_read_as_read_reads_it(self):
        # Each file is one read refuses, given with a document of its bytes, as only a document made otherwise has.
        for text, reason in (
            (TestUpgrade.OLD.replace("sample_settings", "other_settings"), "kind"),
            (TestUpgrade.OLD.replace("kind: sample_settings\n", ""), "header"),
            (
                TestUpgrade.OLD.replace("schema_version: 1", "schema_version: 3"),
                "newer",
            ),
            (b"\xff" + TestUpgrade.OLD.encode("utf-8"), "text"),
        ):
            with self.subTest(reason=reason):
                path = self.file(text)
                refused = self.rewrite(
                    dataclasses.replace(self.new, data=path.read_bytes())
                )
                self.assertIsInstance(refused, Refused)
                self.assertEqual(refused.reason, reason)
                self.assertFalse(self.kept.exists())

    def test_a_file_in_this_layout_is_not_rewritten(self):
        self.file(as_json(BODY))
        refused = self.rewrite(settings_file.read(self.path, SAMPLE))
        self.assertNotIsInstance(refused, Refused)
        self.assertRegex(str(refused), "is not in an earlier layout than 2")
        self.assertEqual(self.texts(), (as_json(BODY), None))

    def test_a_dry_run_writes_nothing_and_refuses_as_the_run_would(self):
        self.assertEqual(self.rewrite(dry_run=True), self.kept)
        self.assertEqual(self.texts(), (self.OLD_TEXT, None))
        # A kept file that holds the file already: the run would go on, and the dry run leaves its permissions.
        self.kept.write_text(self.OLD_TEXT, encoding="utf-8")
        self.kept.chmod(0o644)
        self.assertEqual(self.rewrite(dry_run=True), self.kept)
        self.assertEqual(self.modes(), [0o600, 0o644])
        # One that holds another file: refused, as the run refuses it.
        self.kept.write_text("kind: sample_settings\n", encoding="utf-8")
        refused = self.rewrite(dry_run=True)
        self.assertIsInstance(refused, private_files.Refused)
        self.assertEqual(
            [(r.found, str(r.fix)) for r in refused.rows],
            [
                (
                    f"{self.where}.v1 is another file, and a kept file is never written over",
                    f"mv {self.where}.v1 {self.where}.v1.old, then rkmon migrate --settings {self.where}",
                )
            ],
        )
        self.assertEqual(self.texts(), (self.OLD_TEXT, "kind: sample_settings\n"))
        # The file's header is read as the run reads it.
        self.kept.unlink()
        path = self.file(
            TestUpgrade.OLD.replace("schema_version: 1", "schema_version: 3")
        )
        refused = self.rewrite(
            dataclasses.replace(self.new, data=path.read_bytes()), dry_run=True
        )
        self.assertIsInstance(refused, Refused)
        self.assertEqual(refused.reason, "newer")
        self.assertFalse(self.kept.exists())

    def test_a_dry_run_checks_each_write_as_the_run_does(self):
        # A link put in the file's place once migrate has read it: the dry run refuses it as the run's write does
        # (private_files.write's own dry run), and writes nothing.
        copy = self.file(self.OLD_TEXT, "copy.yaml")
        self.path.unlink()
        self.path.symlink_to(copy)
        rows = []
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                refused = self.rewrite(dry_run=dry_run)
                self.assertIsInstance(refused, private_files.Refused)
                rows.append([(r.found, str(r.fix)) for r in refused.rows])
                if dry_run:
                    self.assertFalse(os.path.lexists(self.kept))
        self.assertEqual(
            rows[0],
            [
                (
                    f"{self.where} is a link, not a file",
                    f"name the file it points to: rkmon migrate --settings {os.path.realpath(copy)}",
                )
            ],
        )
        self.assertEqual(rows[1], rows[0])


class TestFileType(Files):
    def test_its_words(self):
        self.assertEqual(SAMPLE.noun, "sample settings")
        state = FileType("internal_state", 1, {})
        self.assertEqual(
            (state.noun, state.template_command, state.section_command),
            ("internal state", "", ""),
        )

    def test_what_it_refuses(self):
        step = tag_from_image
        for args, options in (
            (("Sample", 2, {}), {}),
            (("", 2, {}), {}),
            (("sample_settings", 0, {}), {}),
            (("sample_settings", True, {}), {}),
            (("sample_settings", "2", {}), {}),
            (("sample_settings", 2, 42), {}),
            (("sample_settings", 2, {}), {"upgrades": {2: step}}),
            (("sample_settings", 2, {}), {"upgrades": {1: "step"}}),
            (("sample_settings", 2, {}), {"coherence": "check"}),
            (("sample_settings", 2, {}), {"template_command": None}),
            (("sample_settings", 2, {}), {"section_command": ("template", "section")}),
            (("sample_settings", 2, {}), {"template_values": {"service.tag": "2.0.0"}}),
        ):
            with self.subTest(args=args, options=options), self.assertRaises(
                ValueError
            ):
                FileType(*args, **options)

    def test_template_values_only_the_package_knows(self):
        # Each is set as its key's x-template when the schema loads: the function is called then, and once.
        calls = []

        def values():
            calls.append(1)
            return {"service.tag": "2.0.0"}

        file_type = FileType("sample_settings", 2, SCHEMA_FILE, template_values=values)
        self.assertEqual(calls, [])
        tag = file_type.load()["properties"]["service"]["properties"]["tag"]
        self.assertEqual(tag["x-template"], "2.0.0")
        self.assertEqual(
            settings_file.template(file_type),
            FULL.replace('tag: "1.0.0"', 'tag: "2.0.0"'),
        )
        self.assertEqual(calls, [1])
        # A key of a section, of its own: no section, no map's entry, no key that is a $ref, which others may share.
        data = schema()
        data["properties"]["service"]["properties"]["tag"] = {"$ref": "#/$defs/tag"}
        data["$defs"]["tag"] = {"type": "string", "description": "The tag. Required."}
        for dotted, words, schema_data in (
            ("service.zzz", "service.zzz is no key of sample settings", SCHEMA_FILE),
            (
                "bindings.big.port",
                "bindings.big.port is no key of sample settings",
                SCHEMA_FILE,
            ),
            (
                "service",
                "service is no key of sample settings of its own: a section, or a $ref",
                SCHEMA_FILE,
            ),
            (
                "service.tag",
                "service.tag is no key of sample settings of its own: a section, or a $ref",
                data,
            ),
        ):
            file_type = FileType(
                "sample_settings", 2, schema_data, template_values=lambda: {dotted: 1}
            )
            with self.subTest(dotted=dotted), self.assertRaises(ValueError) as cm:
                file_type.load()
            self.assertEqual(str(cm.exception), words)

    def test_a_keyword_it_does_not_know(self):
        misspelt = schema()
        mode = misspelt["properties"]["features"]["properties"]["queue"]["properties"][
            "mode"
        ]
        mode["x-requird-when"] = mode.pop("x-required-when")
        with self.assertRaises(ValueError) as cm:
            FileType("sample_settings", 2, misspelt).load()
        self.assertEqual(
            str(cm.exception),
            "features.queue.mode: x-requird-when is no keyword of a schema; did you mean x-required-when?",
        )
        with self.assertRaises(ValueError) as cm:
            FileType("sample_settings", 2, schema(**{"x-colour": "blue"})).load()
        self.assertEqual(
            str(cm.exception), "the top: x-colour is no keyword of a schema"
        )
        # A map's rule for its names is looked at too: its x-template names the map's example entry.
        names = schema()
        rule = names["properties"]["bindings"]["propertyNames"]
        rule["x-tempate"] = rule.pop("x-template")
        with self.assertRaises(ValueError) as cm:
            FileType("sample_settings", 2, names).load()
        self.assertEqual(
            str(cm.exception),
            "bindings (propertyNames): x-tempate is no keyword of a schema; did you mean x-template?",
        )

    def test_the_registrys_keywords_are_known(self):
        data = schema()
        data["properties"]["host"]["x-applies"] = "restart"
        data["properties"]["host"]["properties"]["cpus"]["x-internal"] = "host_cpus"
        self.assertIn("x-class", json.dumps(data))
        FileType("sample_settings", 2, data).load()
        self.assertEqual(
            settings_file.REGISTRY_KEYWORDS, ("x-class", "x-applies", "x-internal")
        )

    def test_the_keywords_are_the_ones_it_reads(self):
        # Each keyword the module's code names is in KEYWORDS, and each of KEYWORDS is read past its own entry.
        source = pathlib.Path(settings_file.__file__).read_text(encoding="utf-8")
        named = set(re.findall(r"[\"'](x-[a-z-]+)[\"']", source))
        known = set(settings_file.KEYWORDS) | set(settings_file.REGISTRY_KEYWORDS)
        self.assertEqual(named - known, set())
        self.assertEqual(len(settings_file.KEYWORDS), 13)
        for keyword in settings_file.KEYWORDS:
            self.assertGreater(source.count(f'"{keyword}"'), 1, keyword)

    def test_the_schema_is_read_at_its_first_use_and_once(self):
        path = self.folder / "sample.schema.json"
        # No such file yet: making the file type reads nothing.
        file_type = FileType("sample_settings", 2, path)
        path.write_text(SCHEMA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        first = file_type.load()
        path.unlink()
        self.assertIs(file_type.load(), first)
        self.assertEqual(settings_file.template(file_type), FULL)

    def test_a_schema_it_cannot_use(self):
        swapped = schema()
        props = swapped["properties"]
        swapped["properties"] = {
            "schema_version": props["schema_version"],
            "kind": props["kind"],
        }
        unused = schema()
        unused["properties"]["bindings"]["x-used-by"] = ["integrations"]
        # x-off-reason on a plain key; on a section that names no feature that reads it; on a map with no title.
        plain = schema()
        plain["properties"]["host"]["properties"]["cpus"].update(
            {"x-used-by": ["features.reviews"], "x-off-reason": "reviews are off"}
        )
        no_readers = schema()
        del no_readers["properties"]["integrations"]["properties"]["tracker"][
            "x-used-by"
        ]
        no_title = schema()
        del no_title["properties"]["bindings"]["title"]
        for file_type, message in (
            (
                FileType("sample_settings", 2, swapped),
                "starts with the keys kind and schema_version",
            ),
            (
                FileType("other_settings", 2, schema()),
                "is for kind 'sample_settings', layout 2, not 'other_settings'",
            ),
            (
                FileType("sample_settings", 3, schema()),
                "is for kind 'sample_settings', layout 2, not 'sample_settings', "
                "layout 3",
            ),
            (
                FileType("sample_settings", 2, unused),
                "^bindings: x-used-by names integrations, which is no section "
                "with an enabled switch",
            ),
            (
                FileType("sample_settings", 2, plain),
                "^host.cpus: x-off-reason is for a section or a map$",
            ),
            (
                FileType("sample_settings", 2, no_readers),
                "^integrations.tracker: a section or a map with x-off-reason names "
                "the features that read it, in x-used-by$",
            ),
            (
                FileType("sample_settings", 2, no_title),
                "^bindings: a section or a map with x-off-reason has a title$",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                file_type.load()

    def test_a_pattern_schema_pattern_refuses(self):
        """A schema with a pattern schema_pattern refuses is refused when it loads, in one line that names the key
        and the pattern: a key's pattern, or a map's rule's for its names."""
        in_a_key = schema()
        in_a_key["properties"]["service"]["properties"]["name"]["pattern"] = r"^[\S]+$"
        in_names = schema()
        in_names["properties"]["bindings"]["propertyNames"][
            "pattern"
        ] = "^(?P<n>[a-z]+)$"
        for data, message in (
            (
                in_a_key,
                r"service.name: the pattern ^[\S]+$ has \S in a set, which Python's re cannot read as JSON Schema "
                "does",
            ),
            (
                in_names,
                "bindings (propertyNames): the pattern ^(?P<n>[a-z]+)$ has the group (?P, which Python's re cannot "
                "read as JSON Schema does",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as cm:
                    FileType("sample_settings", 2, data).load()
                self.assertEqual(str(cm.exception), message)

    def test_a_schema_given_as_a_mapping_is_its_own_copy(self):
        data = schema()
        file_type = FileType("sample_settings", 2, data)
        file_type.load()
        data["title"] = "Changed"
        self.assertEqual(file_type.load()["title"], "Sample tool: sample settings")

    def test_the_layout_another_package_reads(self):
        """layout_in reads another package's copy of the schema, at the schema's place in this package, as load reads
        a schema's header: None when that copy does not say a layout of the kind."""
        rel = SCHEMA_FILE.relative_to(ROOT)

        def copied(change) -> str:
            data = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
            change(data["properties"])
            return json.dumps(data)

        def layout(value):
            return lambda props: props["schema_version"].update(const=value)

        self.assertEqual(SAMPLE.layout_in(ROOT), SAMPLE.version)
        for name, text, expected in (
            ("a later layout", copied(layout(3)), 3),
            ("this layout", copied(layout(2)), 2),
            ("another kind", copied(lambda p: p["kind"].update(const="other")), None),
            ("a layout that is no whole number", copied(layout(True)), None),
            ("a layout as text", copied(layout("3")), None),
            (
                "no fixed layout",
                copied(lambda p: p["schema_version"].pop("const")),
                None,
            ),
            ("no properties", "{}", None),
            ("a list", "[]", None),
            ("no JSON", "kind: sample_settings\n", None),
            ("no schema", None, None),
        ):
            with self.subTest(name):
                package = self.folder / name.replace(" ", "_")
                if text is not None:
                    (package / rel).parent.mkdir(parents=True)
                    (package / rel).write_text(text, encoding="utf-8")
                self.assertEqual(SAMPLE.layout_in(package), expected)
        self.assertIsNone(FileType("sample_settings", 2, schema()).layout_in(ROOT))


class TestPathsAndLayouts(unittest.TestCase):
    def test_shown(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(
            os.environ, {"HOME": home}
        ):
            home = os.path.realpath(home)
            os.environ["HOME"] = home
            self.assertEqual(
                settings_file.shown(os.path.join(home, "a", "b.yaml")), "~/a/b.yaml"
            )
            self.assertEqual(settings_file.shown("~/c.yaml"), "~/c.yaml")
            self.assertEqual(settings_file.shown(home), "~")
            self.assertEqual(settings_file.shown(home + "2/x.yaml"), home + "2/x.yaml")
            self.assertEqual(settings_file.shown("/opt/x.yaml"), "/opt/x.yaml")

    def test_another_layouts_words(self):
        self.assertEqual(
            settings_file.layout_found("~/s.yaml", 1, 2),
            "~/s.yaml is in layout 1, older than this package's layout 2",
        )
        self.assertEqual(
            settings_file.layout_found("~/s.yaml", 3, 2),
            "~/s.yaml is in layout 3, newer than this package's layout 2",
        )
        self.assertEqual(
            settings_file.layout_fix("~/s.yaml", 1, 2),
            Command(("rkmon", "migrate", "--settings", "~/s.yaml")),
        )
        self.assertEqual(
            str(settings_file.layout_fix("~/s.yaml", 3, 2)),
            "rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one",
        )
        # A path with a space is one word.
        self.assertEqual(
            str(settings_file.layout_fix("~/my settings.yaml", 1, 2)),
            "rkmon migrate --settings ~/'my settings.yaml'",
        )


def private_reads(tree) -> list:
    """(line, name), in order, for each name of settings_file that starts with "_" (a __dunder__ aside) and that the
    module TREE reads: imported from it, or an attribute of a name it is bound to, getattr's included.
    """

    def private(name) -> bool:
        return name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )

    def dotted(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            inner = dotted(node.value)
            return f"{inner}.{node.attr}" if inner else ""
        return ""

    bound, found = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {
                a.asname or a.name
                for a in node.names
                if a.name.split(".")[-1] == "settings_file"
            }
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[-1] == "settings_file":
                found += [(node.lineno, a.name) for a in node.names if private(a.name)]
            else:
                bound |= {
                    a.asname or a.name for a in node.names if a.name == "settings_file"
                }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and dotted(node.value) in bound
            and private(node.attr)
        ):
            found.append((node.lineno, node.attr))
        elif (
            isinstance(node, ast.Call)
            and dotted(node.func) == "getattr"
            and len(node.args) > 1
            and dotted(node.args[0]) in bound
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and private(node.args[1].value)
        ):
            found.append((node.lineno, node.args[1].value))
    return sorted(found)


class TestWhatOtherModulesRead(Files):
    """What the package's other modules read of settings_file, the checks across keys among them: the names its
    docstring lists, and no other."""

    def test_no_other_module_reads_a_name_of_it_that_starts_with_an_underscore(self):
        """A module that needs more of settings_file asks for a public name, with its contract; the tests may look
        inside."""
        found = []
        for f in python_files():
            rel = f.relative_to(ROOT).as_posix()
            if rel == "rkmon_lib/settings_file.py" or rel.split("/")[0] == "tests":
                continue
            tree = ast.parse(f.read_bytes(), rel)
            found += [f"{rel}:{line}: {name}" for line, name in private_reads(tree)]
        self.assertEqual(found, [])

    def test_the_reads_it_finds(self):
        for source, reads in (
            ("from . import settings_file\nsettings_file._at(root, 'x')", [(2, "_at")]),
            (
                "from rkmon_lib import settings_file as sf\nsf._Checker(doc)",
                [(2, "_Checker")],
            ),
            (
                "import rkmon_lib.settings_file\nrkmon_lib.settings_file._show(1)",
                [(2, "_show")],
            ),
            ("import rkmon_lib.settings_file as sf\nsf._show(1)", [(2, "_show")]),
            ("from .settings_file import _source, read", [(1, "_source")]),
            ("from rkmon_lib.settings_file import _on", [(1, "_on")]),
            (
                "from . import settings_file\ngetattr(settings_file, '_where')",
                [(2, "_where")],
            ),
            (
                "from . import settings_file\nsettings_file.read, settings_file.__file__",
                [],
            ),
            ("from . import other\nother._at, getattr(other, '_at')", []),
        ):
            with self.subTest(source=source):
                self.assertEqual(private_reads(ast.parse(source)), reads)

    def test_schema_at(self):
        root = SAMPLE.load()
        self.assertEqual(settings_file.schema_at(root, "host.cpus")["maximum"], 8)
        # A map's entry, by its name or by *: the entries' schema, its $ref followed.
        entry = settings_file.schema_at(root, "bindings.*")
        self.assertIs(entry, root["$defs"]["binding"])
        self.assertIs(settings_file.schema_at(root, "bindings.my_model"), entry)
        # No one schema: a key the schema does not have, or one in a section with alternatives (oneOf).
        self.assertIsNone(settings_file.schema_at(root, "host.cores"))
        self.assertIsNone(settings_file.schema_at(root, "bindings.my_model.address"))

    def test_fits(self):
        doc = self.read(as_json(BODY))
        self.assertTrue(settings_file.fits(doc, "host.cpus", 8))
        self.assertFalse(settings_file.fits(doc, "host.cpus", 9))
        self.assertFalse(settings_file.fits(doc, "host.cpus", "4"))
        # A value with a <...> has the placeholder's row alone.
        self.assertTrue(settings_file.fits(doc, "host.cpus", "<cpus>"))
        with self.assertRaises(ValueError) as cm:
            settings_file.fits(doc, "host.cores", 4)
        self.assertEqual(
            str(cm.exception),
            "schema_at finds no schema for host.cores in the sample settings schema",
        )


if __name__ == "__main__":
    unittest.main()
