# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Layout 2's three schemas (schema/*.schema.json, through rkmon_lib/file_types.py), by the rules every settings file
follows and the registry's rules:

- each schema loads, with the deploy and build keys in the order the every-key text and the build template give them,
  and the internal state's keys; a model binding is a union by provider; every pattern is one
  settings_file.schema_pattern reads as JSON Schema does;
- every key carries x-class, x-used-by and x-internal (but a key no internal name feeds today), and every deploy key
  but container.name and the updates section's carries x-applies, no_restart or new_container (which keys
  are no_restart, tests/test_change_plan.py ties to the image test's no-restart stage); each internal name is one
  today's code reads (a build key's: a key of layout 1's build file that rkmon migrate carries);
- the private keys and no others, every private-shaped name marked private, every key public or private and none
  named like a secret, no site in a private key's default or template value, and a secret named only by a binding's
  api_key_secret_name;
- units in names, names and enum values in snake_case (but the base images' own names), booleans only for switches,
  and a mode that is "prototype" or "live";
- no rejected word in a name, a value or any text a person reads;
- x-source on each key whose template value is a placeholder, and a new settings file, in each profile, that fails
  only on those (but the updates section's, which no profile has), once each, and passes once they are filled in,
  the checks across keys included; each profile's values in the file it writes;
- the updates section: in no profile; the section command prints it as the every-key text has it, which, pasted into
  each profile's file with its placeholders filled in, passes; 113 deploy keys, 111 of them in the every-key text,
  and 21 build keys;
- a binding pasted into model_bindings as the template gives it, its name a placeholder too, one row for each key;
- in the every-key text, the model bindings' three forms, a price's entry and the team dashboard's serve block,
  each commented out for you to use;
- x-off-reason on the two sections and the map a template writes as {} when nothing uses them, and on no other key,
  each reason read after "while";
- the build schema's atlassian_cli section naming its tool, and when AMD's public ROCm installer is checked;
- the remote GPU tests' forms in their keys' patterns, the ones rkmon_lib/gpu.py checks, and the web host's, the
  ones the writer checks (publish.target_refusal).
"""

import json
import pathlib
import re
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

import doc_lints  # noqa: E402
import home_repository  # noqa: E402
import settings_template_texts as golden  # noqa: E402
from rkmon_lib import (  # noqa: E402
    answers,
    bindings,
    derived,
    file_types,
    from_layout1,
    gpu,
    image,
    privacy,
    publish,
    secrets_registry,
    settings_file,
    spend,
)
from rkmon_lib.checks import Status  # noqa: E402

FILE_TYPES = {
    "deploy": file_types.DEPLOY,
    "build": file_types.BUILD,
    "state": file_types.INTERNAL_STATE,
}
RA = "features.review_automation"
RGT = f"{RA}.remote_gpu_tests"
DG = "features.merge_queue.develop_guard"
TEAM = "features.dashboards.team"
TICKETS = "features.ticket_creation"
# A value for each key a new settings file has you fill in (a placeholder in the template), as you would write it.
FILLED = {
    "repository.name": "example-org/example-repo",
    "repository.folder": "projects/example",
    "repository.scope_labels": ["example-label"],
    "integrations.github_cli.username": "example-user",
    "updates.repository": "example-org/example-package",
    "updates.branch": "main",
    "atlassian_cli.package_url": "https://example.com/atlassian-cli.tar.gz",
    "atlassian_cli.package_sha256": "0123456789abcdef" * 4,
}


def setUpModule():
    # The repository you fill in stands for the home repository: its repository defaults apply.
    home_repository.stand_in(FILLED["repository.name"], FILLED["repository.folder"])


# The section no profile has, as the section command prints it: its text, from its comment to the blank line
# after it in the golden every-key text.
UPDATES = "updates"
_EVERY_KEY = golden.TEXTS["deploy every key"]
_AT = _EVERY_KEY.index("\n# Updates: ") + 1
UPDATES_TEXT = _EVERY_KEY[_AT : _EVERY_KEY.index("\n\n", _AT) + 1]

# The deploy and build files' keys, in the order the every-key text and the build template give them: the header's
# two, then every key; a map is one key, its entries' keys are not listed.
KEYS = {
    "deploy": (
        "kind",
        "schema_version",
        "container.name",
        "image.registry",
        "image.name",
        "image.tag",
        "updates.repository",
        "updates.branch",
        "repository.name",
        "repository.folder",
        "repository.scope_labels",
        "repository.shared_label_folders",
        "host.resources.cpus",
        "host.resources.memory_gib",
        "host.resources.ramdisk_gib",
        "host.users.monitor_uid",
        "host.users.monitor_gid",
        "monitor.tick_interval_minutes",
        "monitor.full_tick_max_interval_minutes",
        "monitor.parallel_builds",
        f"{RA}.enabled",
        f"{RA}.mode",
        f"{RA}.automated_reviewer_name",
        f"{RA}.skills_folder",
        f"{RA}.orchestrator.model",
        f"{RA}.orchestrator.effort",
        f"{RA}.schedule.push_cooldown_minutes",
        f"{RA}.schedule.follow_up_cooldown_minutes",
        f"{RA}.parallel_reviews",
        f"{RA}.large_prs.in_depth_review_over_lines",
        f"{RA}.large_prs.split_part_lines",
        f"{RA}.passes.verification_pass.model",
        f"{RA}.passes.verification_pass.effort",
        f"{RA}.passes.code_quality_pass.model",
        f"{RA}.passes.code_quality_pass.effort",
        f"{RA}.passes.first_review_pass.model",
        f"{RA}.passes.first_review_pass.effort",
        f"{RA}.passes.ci_triage_pass.model",
        f"{RA}.passes.ci_triage_pass.effort",
        f"{RA}.passes.first_code_quality_pass.model",
        f"{RA}.passes.first_code_quality_pass.effort",
        f"{RA}.passes.notice_draft_pass.model",
        f"{RA}.passes.notice_draft_pass.effort",
        f"{RGT}.enabled",
        f"{RGT}.cluster.login_host",
        f"{RGT}.cluster.ssh_port",
        f"{RGT}.cluster.ssh_user",
        f"{RGT}.cluster.partition",
        f"{RGT}.cluster.accounts",
        f"{RGT}.cluster.qos",
        f"{RGT}.cluster.constraint",
        f"{RGT}.cluster.gpu_types",
        f"{RGT}.cluster.default_target",
        f"{RGT}.cluster.gpus_per_job",
        f"{RGT}.cluster.time_limit",
        f"{RGT}.cluster.excluded_nodes",
        f"{RGT}.node.work_folder",
        f"{RGT}.node.python",
        f"{RGT}.node.setup_command",
        f"{RGT}.node.environment",
        f"{RGT}.node.clone_url",
        f"{RGT}.jobs_per_review",
        f"{RGT}.wait_report_after_minutes",
        f"{RGT}.bad_node_resubmits",
        f"{RA}.spend.enabled",
        f"{RA}.spend.usd_per_million_tokens",
        "features.merge_queue.enabled",
        "features.merge_queue.mode",
        "features.merge_queue.auto_merge.enabled",
        "features.merge_queue.auto_merge_guard.on_your_pull_requests",
        "features.merge_queue.auto_hold.enabled",
        "features.merge_queue.queue_notices.enabled",
        "features.merge_queue.drift_notices.enabled",
        f"{DG}.enabled",
        f"{DG}.issue_repository",
        f"{DG}.ci_red_label",
        f"{DG}.ci_red_extra_labels",
        f"{DG}.stability_label",
        "features.dashboards.admin.enabled",
        "features.dashboards.admin.port",
        f"{TEAM}.enabled",
        f"{TEAM}.mode",
        f"{TEAM}.public_url",
        f"{TEAM}.priority_requests.enabled",
        f"{TEAM}.delivery",
        f"{TEAM}.publish.web_host",
        f"{TEAM}.publish.username",
        f"{TEAM}.publish.site_folder",
        f"{TEAM}.serve.address",
        f"{TEAM}.serve.port",
        "features.alerts.mode",
        "features.alerts.message_label",
        "features.alerts.message_cooldown_minutes",
        "features.alerts.develop_breaks.enabled",
        "features.alerts.needs_you.enabled",
        f"{TICKETS}.mode",
        f"{TICKETS}.project_key",
        f"{TICKETS}.issue_type",
        f"{TICKETS}.epic_key",
        f"{TICKETS}.components",
        f"{TICKETS}.labels",
        f"{TICKETS}.required_fields",
        f"{TICKETS}.follow_up_tickets.enabled",
        f"{TICKETS}.found_issue_tickets.enabled",
        "features.weekly_report.enabled",
        "integrations.github_cli.username",
        "integrations.github_cli.max_requests_per_hour",
        "integrations.atlassian_cli.jira_url",
        "integrations.claude_code.base_url",
        "integrations.claude_code.api_key_sent_as",
        "integrations.claude_code.api_key_header",
        "integrations.claude_code.api_key_scheme",
        "model_bindings",
    ),
    "build": (
        "kind",
        "schema_version",
        "image.registry",
        "image.name",
        "image.tag",
        "operating_system.base_image",
        "operating_system.package_mirror",
        "operating_system.ca_certificates",
        "rocm.version",
        "rocm.package_url",
        "rocm.package_sha256",
        "github_cli.version",
        "github_cli.package_url",
        "github_cli.package_sha256",
        "atlassian_cli.version",
        "atlassian_cli.package_url",
        "atlassian_cli.package_sha256",
        "claude_agent_sdk.version",
        "claude_agent_sdk.package_url",
        "claude_agent_sdk.package_sha256",
        "python_packages.package_index",
    ),
}
# The internal state's keys, with a binding's entry.
STATE_KEYS = {
    "kind",
    "schema_version",
    "repository.codeowner_teams",
    "repository.clone_since",
    "repository.base_branch",
    "model_bindings",
    "model_bindings.*.display_name",
    "model_bindings.*.chat_path",
    "model_bindings.*.models_url",
    "model_bindings.*.probe",
    "team_dashboard.web_host_keys",
    "remote_gpu_tests.login_host_keys",
    "jira.accounts",
    "site.ci_infra_signatures",
    "reviews.automated_reviewer_names_used",
    "reviews.unnamed_are_ours_until",
    "standby.on",
    "standby.reason",
    "standby.since",
    "provisioned.at",
    "provisioned.package_version",
    "provisioned.older_install",
    "provisioned.adopted_at",
}
# The private keys, and no others. In the internal state, a key is private when it names a site, a person or an
# internal system.
PRIVATE = {
    "deploy": {
        "image.registry",
        "updates.repository",
        "repository.name",
        "repository.folder",
        "repository.scope_labels",
        "repository.shared_label_folders",
        f"{RA}.automated_reviewer_name",
        f"{RA}.skills_folder",
        f"{RGT}.cluster.login_host",
        f"{RGT}.cluster.ssh_user",
        f"{RGT}.cluster.partition",
        f"{RGT}.cluster.accounts",
        f"{RGT}.cluster.qos",
        f"{RGT}.cluster.constraint",
        f"{RGT}.cluster.gpu_types",
        f"{RGT}.cluster.excluded_nodes",
        f"{RGT}.node.work_folder",
        f"{RGT}.node.python",
        f"{RGT}.node.setup_command",
        f"{RGT}.node.environment",
        f"{RGT}.node.clone_url",
        f"{RA}.spend.usd_per_million_tokens",
        f"{DG}.issue_repository",
        f"{DG}.ci_red_label",
        f"{DG}.ci_red_extra_labels",
        f"{DG}.stability_label",
        f"{TEAM}.public_url",
        f"{TEAM}.publish.web_host",
        f"{TEAM}.publish.username",
        f"{TEAM}.publish.site_folder",
        f"{TEAM}.serve.address",
        "features.alerts.message_label",
        f"{TICKETS}.project_key",
        f"{TICKETS}.issue_type",
        f"{TICKETS}.epic_key",
        f"{TICKETS}.components",
        f"{TICKETS}.labels",
        f"{TICKETS}.required_fields",
        "integrations.github_cli.username",
        "integrations.atlassian_cli.jira_url",
        "integrations.claude_code.base_url",
        "integrations.claude_code.api_key_header",
        "model_bindings.*.model_id",
        "model_bindings.*.base_url",
        "model_bindings.*.server_url",
        "model_bindings.*.api_key_header",
    },
    "build": {
        "image.registry",
        "operating_system.package_mirror",
        "operating_system.ca_certificates",
        "rocm.package_url",
        "github_cli.package_url",
        "atlassian_cli.package_url",
        "claude_agent_sdk.package_url",
        "python_packages.package_index",
    },
    "state": {
        "repository.codeowner_teams",
        "model_bindings.*.display_name",
        "model_bindings.*.chat_path",
        "model_bindings.*.models_url",
        "model_bindings.*.probe",
        "team_dashboard.web_host_keys",
        "remote_gpu_tests.login_host_keys",
        "jira.accounts",
        "site.ci_infra_signatures",
        "reviews.automated_reviewer_names_used",
    },
}
# The words of a private-shaped name: an address, a host, a registry, a mirror or an index; a user name or an
# account; a label, a folder or a repository; a model's id; a header's name; a cluster's job and node settings.
PRIVATE_WORDS = {
    "url",
    "host",
    "address",
    "registry",
    "mirror",
    "index",
    "certificates",
    "user",
    "username",
    "accounts",
    "label",
    "labels",
    "folder",
    "folders",
    "repository",
    "id",
    "header",
    "partition",
    "qos",
    "constraint",
    "nodes",
    "environment",
    "command",
}
# A private key's default names no site: it is blank, or one of these generic values.
GENERIC = {"127.0.0.1", "rkmon-gpu"}
BLANKS = ("", None, [], {})
# The name of a key that would hold a secret.
SECRET_SHAPED = re.compile(
    r"(?:^|_)(?:token|password|passwd|secret|credentials?|api_key|private_key|ssh_key)$"
)
# The keys no internal name feeds today: keys layout 2 adds. Spend's switch is new
# too, but it feeds the prices (none while it is off), so it names them; and so are the modes of alerts, the team
# dashboard and ticket creation, which feed their features' modes, and the automated reviewer name, which
# feeds the name line 1 gives.
NEW = {
    "deploy": {
        f"{RA}.skills_folder",
    },
    "build": {"claude_agent_sdk.package_url", "claude_agent_sdk.package_sha256"},
    "state": {
        "model_bindings",
        "remote_gpu_tests.login_host_keys",
        "reviews.automated_reviewer_names_used",
        "reviews.unnamed_are_ours_until",
        "standby.on",
        "standby.reason",
        "standby.since",
        "provisioned.at",
        "provisioned.package_version",
        "provisioned.older_install",
        "provisioned.adopted_at",
    },
}
# The deploy keys with no x-applies class: the container's name, which confirm refuses to change while the
# container exists, and the updates section's, which only rkmon package update reads, on the host.
NO_APPLIES = {"container.name", "updates.repository", "updates.branch"}
# The internal names no schema or defaults file holds, which today's code reads: the auto-merge guard's on your own
# PRs, the queue notices' switch, and the modes of alerts, the team dashboard and ticket
# creation, which rkmon_lib/features.py reads by each feature's name (features.mode, features.live).
READ_IN_CODE = {
    "automerge_guard_own",
    "features.merge_queue.queue_notices",
    "features.alerts.mode",
    "features.dashboards.team.mode",
    "features.ticket_creation.mode",
}
# The internal names today's code reads by the schema's x-internal, not by name: the updates section's, at which the
# translation of layout 1's configuration file reads its values (tools/translate_production.py, updates). In layout 2,
# rkmon package update reads that section from your settings file, on the host (NO_APPLIES).
READ_BY_THE_SCHEMA = {"update.repo", "update.branch"}
# The configuration keys a deploy settings file in layout 1 gave config.json, at their defaults (its carry, recorded
# in tests/data/deploy_tree_carry.json): keys today's code reads.
LAYOUT_1_CONFIG = json.loads(
    (ROOT / "tests" / "data" / "deploy_tree_carry.json").read_text(encoding="utf-8")
)["layout_1_config"]["defaults"]
# The two sections and the map a template writes as {} with "TITLE: not used while REASON." when nothing uses
# them; any other key that nothing uses is left out.
NOT_USED = {
    ("deploy", "integrations.atlassian_cli"),
    ("deploy", "integrations.claude_code"),
    ("deploy", "model_bindings"),
}

# The keywords that give a key's values; settings_file.template_value takes a template's value from them.
VALUES = ("x-template", "default", "const")
# The keywords whose text a person reads: in a template's comments, a check's rows, the editor's hover text.
READ_BY_PEOPLE = (
    "title",
    "description",
    "x-header",
    "x-turn-on",
    "x-source",
    "x-blank",
    "x-required-when",
    "x-default-means",
    "x-changed-by",
    "x-off-reason",
)
# The word lists (tests/doc_words.yaml): what no name, value or text may hold.
DOC_WORDS = answers.load(ROOT / doc_lints.WORDS)


class WordList:
    """A word list (`word_list`): `search(text)` finds its words in TEXT outside text in backticks (doc_lints.CODE),
    which quote code, such as production's older names that the translation's report shows as they are; None when
    there is none."""

    def __init__(self, pattern: re.Pattern):
        self.pattern = pattern

    def search(self, text: str):
        return self.pattern.search(doc_lints.CODE.sub(" ", text))


def word_list(name) -> WordList:
    """The list NAME's words and patterns, as the docs' words lint finds them (doc_lints.word_pattern); never in text
    in backticks."""
    return WordList(doc_lints.word_pattern(DOC_WORDS[name]))


# The rejected words (in a name, a value or a text); the words of history and the words found unclear; the closing
# words, which a key's closing sentence says in full.
REJECTED, UNWANTED, CLOSING_WORDS = (
    word_list(name) for name in ("rejected", "unwanted", "closing_words")
)
# A unit's short form at a name's end, as its last word or after a number (cache_write_5m), and a word that starts
# with min or max but is no whole word.
SHORT_UNIT = re.compile(r"(?:^|_|[0-9])(?:s|sec|secs|m|min|mins|h|hr|hrs|gb|mb|kb|ms)$")
MIN_MAX_WORDS = {"min", "max", "minute", "minutes", "minimum", "maximum"}
# A number's unit, as its description gives it, and the end of the name that carries it.
UNIT_IN_TEXT = (
    (re.compile(r"\b(?:in|this many) minutes\b|^Minutes between\b"), "_minutes"),
    (re.compile(r"\b(?:in|this many) seconds\b"), "_seconds"),
    (re.compile(r"\b(?:in|this many) hours\b"), "_hours"),
    (re.compile(r"\b(?:in|this many) days\b"), "_days"),
    (re.compile(r"\bin GiB\b"), "_gib"),
    (re.compile(r"\bchanged lines\b"), "_lines"),
    (re.compile(r"\b(?:per|in an?) minute\b", re.I), "_per_minute"),
    (re.compile(r"\b(?:per|in an?) hour\b", re.I), "_per_hour"),
)
SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
# The one key whose enum values are not snake_case: the bases the image builds on, by the images' own names
# (ubuntu:26.04), as rkmon_lib/image.py's BASES gives them.
BASE_IMAGES = ("build", "operating_system.base_image")


def _deref(root, node):
    while isinstance(node, dict) and "$ref" in node:
        node = root["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def properties(root, node=None, path=""):
    """(path, node, required) for every property of the schema ROOT, at any depth: the keys settings_file.schema_keys
    gives, and the sections, which carry the registry's keywords and names too. A map's entry is NAME.*, and each
    alternative of a oneOf is walked in turn, so a binding's field comes once for each provider that has it.
    """
    node = _deref(root, root if node is None else node)
    needs = node.get("required") or ()
    for name, child in (node.get("properties") or {}).items():
        child = _deref(root, child)
        where = f"{path}.{name}" if path else name
        yield where, child, name in needs
        yield from properties(root, child, where)
    entry = node.get("additionalProperties")
    if isinstance(entry, dict):
        yield from properties(root, entry, f"{path}.*")
    for alternative in node.get("oneOf") or ():
        yield from properties(root, alternative, path)


def patterns(node):
    """Every pattern in NODE, at any depth: a key's, a list item's, a map entry's, a map's rule's for its names, and
    one in $defs."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pattern" and isinstance(value, str):
                yield value
            else:
                yield from patterns(value)
    elif isinstance(node, list):
        for value in node:
            yield from patterns(value)


def types(node):
    t = node.get("type")
    return t if isinstance(t, list) else [t]


def strings(value):
    """Every text in VALUE: VALUE itself when it is a string, else each string, list item and map key in it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)


def has_placeholder(value):
    """Whether VALUE, or a string, list item or map key in it, holds a template's placeholder."""
    return any(settings_file.PLACEHOLDER.search(s) for s in strings(value))


def internal_names(node):
    names = node.get("x-internal", [])
    return names if isinstance(names, list) else [names]


def texts(node):
    """Every text a person reads in a schema node and the nodes under it: the READ_BY_PEOPLE keywords' texts, and each
    profile's what and header (x-profiles), which the writer prints."""
    if isinstance(node, dict):
        for keyword, value in node.items():
            if keyword in READ_BY_PEOPLE:
                yield from value if isinstance(value, list) else [value]
            elif keyword == "x-profiles":
                for profile in (value or {}).values():
                    yield from strings([profile.get("what"), profile.get("header")])
            else:
                yield from texts(value)
    elif isinstance(node, list):
        for item in node:
            yield from texts(item)


def found(mapping, dotted):
    """Whether the dotted path is a key of MAPPING, at any depth."""
    for part in dotted.split("."):
        if not isinstance(mapping, dict) or part not in mapping:
            return False
        mapping = mapping[part]
    return True


# The package's code that reads the configuration, but the registry and deploy_tree: they name each key the schemas
# give them.
CODE = ("rkmon_lib/**/*.py", "worker/*.py", "docker/*.py", "tools/*.py", "rkmon.py")
NAMES_EVERY_KEY = {"rkmon_lib/registry.py", "rkmon_lib/deploy_tree.py"}
_code_texts = []


def read_in_code(dotted):
    """Whether one file of the package's code (CODE) quotes the dotted key, or each of its parts."""
    if not _code_texts:
        for pattern in CODE:
            for p in sorted(ROOT.glob(pattern)):
                if p.relative_to(ROOT).as_posix() not in NAMES_EVERY_KEY:
                    _code_texts.append(p.read_text(encoding="utf-8"))
        _code_texts.append((ROOT / "docker" / "rkmon").read_text(encoding="utf-8"))

    def quoted(text, word):
        return re.search(rf"[\"']{re.escape(word)}[\"']", text) is not None

    return any(
        quoted(text, dotted) or all(quoted(text, part) for part in dotted.split("."))
        for text in _code_texts
    )


def migrate_carries(old):
    """Whether `rkmon migrate` carries OLD, a key of layout 1's build settings file (from_layout1.build): set alone,
    it is moved, merged or refused by a rule of its own, never noted as a key no rule knows.
    """
    section, _, key = old.partition(".")
    try:
        _values, notes = from_layout1.build({section: {key: "1"}}, file_types.BUILD)
    except settings_file.StepRefused:
        return True
    return not any(
        str(note).startswith(f"{old} ") and "not carried" in str(note) for note in notes
    )


def value_at(mapping, dotted):
    """The value at the dotted path of MAPPING."""
    for part in dotted.split("."):
        mapping = mapping[part]
    return mapping


def fill(mapping, dotted, value):
    """Set the dotted path of MAPPING, a key the mapping has, to VALUE."""
    *parents, last = dotted.split(".")
    for part in parents:
        mapping = mapping[part]
    if last not in mapping:
        raise KeyError(dotted)
    mapping[last] = value


def to_fill(file_type):
    """The keys a new file of FILE_TYPE has you fill in (a placeholder in the template), sorted: no map's entry, and
    none of the section no profile has."""
    return sorted(
        p
        for p, n, _ in settings_file.schema_keys(file_type.load())
        if has_placeholder(settings_file.template_value(n, p))
        and ".*" not in p
        and p.split(".")[0] != UPDATES
    )


# The row of the checks across keys while a key fails: they wait (rkmon_lib/settings_file.py, validate).
WAITING = [("coherence", Status.SKIP)]
# A new file's rows that are not PASS once its placeholders are filled in, by its kind: the build file's are its
# check 4's, a WARN for each SHA-256 left blank where nothing checks the download (ROCm's public installer, the SDK
# from the package index).
NOT_CHECKED = {
    "build_settings": [
        ("claude_agent_sdk.package_sha256", Status.WARN),
        ("rocm.package_sha256", Status.WARN),
    ],
    "deploy_settings": [],
}


def open_rows(rows):
    """What ROWS leave to do: the FAIL rows' names, sorted, and each other row that is not PASS, with its status."""
    return (
        sorted(r.name for r in rows if r.status == Status.FAIL),
        [
            (r.name, r.status)
            for r in rows
            if r.status not in (Status.PASS, Status.FAIL)
        ],
    )


def template_entry(root, node):
    """The values the template gives the section the schema NODE describes: each key's template value, and a
    section's own keys in turn."""
    node = _deref(root, node)
    return {
        name: (
            template_entry(root, child)
            if "properties" in _deref(root, child)
            else settings_file.template_value(_deref(root, child), name)
        )
        for name, child in node["properties"].items()
    }


class Reading(dict):
    """A mapping that records each key read from it."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


class TestTheSchemas(unittest.TestCase):
    def test_each_schema_loads_with_its_kind_and_keys(self):
        for name, file_type in FILE_TYPES.items():
            with self.subTest(file=name):
                root = file_type.load()
                self.assertEqual(root["properties"]["kind"]["const"], file_type.kind)
                found_keys = [p for p, _, _ in settings_file.schema_keys(root)]
                if name == "state":
                    self.assertEqual(set(found_keys), STATE_KEYS)
                else:
                    top = [p for p in found_keys if ".*" not in p]
                    self.assertEqual(top, list(KEYS[name]))

    def test_every_pattern_is_one_schema_pattern_reads(self):
        """Every pattern in the schemas, at any depth, is one settings_file.schema_pattern reads as JSON Schema
        does: it refuses none."""
        for name, file_type in FILE_TYPES.items():
            found = list(patterns(file_type.load()))
            self.assertTrue(found, name)
            for pattern in found:
                with self.subTest(file=name, pattern=pattern):
                    settings_file.schema_pattern(pattern)

    def test_a_model_binding_is_a_union_by_provider(self):
        root = file_types.DEPLOY.load()
        bindings_node = root["properties"]["model_bindings"]
        names = settings_file.schema_pattern(bindings_node["propertyNames"]["pattern"])
        for name in ("default", "none"):
            self.assertIsNone(names.search(name))
        self.assertTrue(names.search("gateway"))
        forms = {}
        for alternative in _deref(root, bindings_node["additionalProperties"])["oneOf"]:
            alternative = _deref(root, alternative)
            self.assertIs(alternative["additionalProperties"], False)
            self.assertEqual(alternative["required"][:2], ["provider", "model_id"])
            forms[alternative["properties"]["provider"]["const"]] = alternative
        self.assertEqual(list(forms), ["claude", "openai_compatible", "ollama"])
        self.assertEqual(list(forms["claude"]["properties"]), ["provider", "model_id"])
        limits = {
            form: forms[form]["properties"]["limits"]["properties"]
            for form in ("openai_compatible", "ollama")
        }
        self.assertEqual(
            limits["openai_compatible"]["max_parallel_requests"]["default"], 0
        )
        self.assertEqual(limits["ollama"]["max_parallel_requests"]["default"], 1)
        for form in ("openai_compatible", "ollama"):
            self.assertEqual(
                forms[form]["properties"]["stall_timeout_seconds"]["default"], 300
            )

    def test_every_key_carries_the_registrys_keywords(self):
        for name, file_type in FILE_TYPES.items():
            root = file_type.load()
            paths = {p for p, _, _ in settings_file.schema_keys(root)}
            self.assertLessEqual(NEW[name], paths, name)
            for path, node, _ in settings_file.schema_keys(root):
                with self.subTest(file=name, key=path):
                    self.assertIn(node.get("x-class"), ("public", "private"))
                    self.assertIsInstance(node.get("x-used-by"), list)
                    if path in settings_file.HEADER_KEYS or path in NEW[name]:
                        self.assertNotIn("x-internal", node)
                    else:
                        self.assertTrue(internal_names(node))
                        for internal in internal_names(node):
                            self.assertRegex(internal, r"^\S+$")
                    if name == "deploy" and path not in NO_APPLIES:
                        self.assertIn(
                            node.get("x-applies"), ("no_restart", "new_container")
                        )
                    else:
                        self.assertNotIn("x-applies", node)

    def test_each_internal_name_is_one_todays_code_reads(self):
        price_fields = set()
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in settings_file.schema_keys(file_type.load()):
                for internal in internal_names(node):
                    head, entry, field = internal.partition(".*.")
                    with self.subTest(file=name, key=path, internal=internal):
                        if name == "build":
                            self.assertTrue(migrate_carries(internal))
                        elif entry and head == "prices":
                            price_fields.add(field)
                        elif entry and head == "model_bindings":
                            self.assertIn(field, bindings.binding_defaults())
                        else:
                            self.assertTrue(
                                found(LAYOUT_1_CONFIG, internal)
                                or internal in READ_IN_CODE
                                or (
                                    internal in READ_BY_THE_SCHEMA
                                    and path.startswith("updates.")
                                )
                                or read_in_code(internal)
                            )
        # A price's fields are the ones today's spend reads, each of them.
        prices = Reading({field: 1.0 for field in price_fields})
        spend.usd({"claude-opus-5-5": prices}, {"opus": [1] * 7})
        self.assertEqual(prices.read, price_fields)

    def test_every_property_is_public_or_private_and_none_is_secret(self):
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in properties(file_type.load()):
                with self.subTest(file=name, key=path):
                    self.assertIn(node.get("x-class"), ("public", "private"))
                    self.assertIsNone(SECRET_SHAPED.search(path.rsplit(".", 1)[-1]))

    def test_the_private_keys_and_no_others(self):
        for name, file_type in FILE_TYPES.items():
            classes = {}
            for path, node, _ in properties(file_type.load()):
                classes.setdefault(path, set()).add(node.get("x-class"))
            with self.subTest(file=name):
                self.assertEqual(
                    {p for p, c in classes.items() if "private" in c}, PRIVATE[name]
                )
                self.assertEqual(
                    {p for p, c in classes.items() if len(c) > 1}, set()
                )  # a binding's field has one class, whatever its provider

    def test_every_private_shaped_name_is_private(self):
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in settings_file.schema_keys(file_type.load()):
                words = set(path.rsplit(".", 1)[-1].split("_"))
                if words & PRIVATE_WORDS:
                    with self.subTest(file=name, key=path):
                        self.assertEqual(node.get("x-class"), "private")

    def test_a_private_keys_default_and_template_value_name_no_site(self):
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in settings_file.schema_keys(file_type.load()):
                if node.get("x-class") != "private":
                    continue
                with self.subTest(file=name, key=path):
                    if "default" in node:
                        self.assertTrue(
                            node["default"] in BLANKS or node["default"] in GENERIC
                        )
                    # The template writes what the privacy check lets a template show for a private key, as
                    # its own predicate says: the key's default, a blank or <...> placeholders only.
                    value = settings_file.template_value(node, path)
                    self.assertTrue(privacy._template_may_show(node, value), value)

    def test_only_a_bindings_api_key_secret_name_names_a_secret(self):
        marked = []
        for name, file_type in FILE_TYPES.items():
            root = file_type.load()
            for path, node, _ in properties(root):
                if "x-secret-ref" in node:
                    marked.append((name, path, node))
        self.assertEqual(
            [(n, p) for n, p, _ in marked],
            [("deploy", "model_bindings.*.api_key_secret_name")],
        )
        node = marked[0][2]
        self.assertIs(node["x-secret-ref"], True)
        self.assertEqual(node["x-class"], "public")
        # The registry's rule for a binding's key's secret name (secrets_registry.BINDING_KEY_NAME: a secret's name
        # that ends in _api_key, which the ignore lists match), or blank.
        self.assertEqual(
            node["pattern"], f"^$|^{secrets_registry.BINDING_KEY_NAME.pattern}$"
        )
        form = settings_file.schema_pattern(node["pattern"])
        for value in ("gw_api_key", "key2_api_key", ""):
            self.assertTrue(form.search(value), value)
        for value in (
            "Bad-Name",
            "gw api key",
            "gw-api-key",
            "_key",
            "key2",
            "gw_key",
            "_api_key",
            "ghp_" + "a" * 36 + "!",
        ):
            self.assertIsNone(form.search(value), value)

    def test_units_are_in_the_names(self):
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in properties(file_type.load()):
                last = path.rsplit(".", 1)[-1]
                with self.subTest(file=name, key=path):
                    self.assertIsNone(SHORT_UNIT.search(last))
                    for word in last.split("_"):
                        if word.startswith(("min", "max")):
                            self.assertIn(word, MIN_MAX_WORDS)
                    if {"integer", "number"} & set(types(node)):
                        for words, ending in UNIT_IN_TEXT:
                            if words.search(node.get("description", "")):
                                self.assertTrue(last.endswith(ending), ending)

    def test_names_and_enum_values_are_snake_case(self):
        """Keys and enum values are snake_case, a const as an enum of one value, but for BASE_IMAGES' values:
        the bases the image builds on, by their own names."""
        bases = [base for base, _what, _family in image.BASES]
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in properties(file_type.load()):
                with self.subTest(file=name, key=path):
                    for part in path.split("."):
                        if part != "*":
                            self.assertTrue(SNAKE_CASE.fullmatch(part), part)
                    enum = node.get("enum") or []
                    if (name, path) == BASE_IMAGES:
                        self.assertEqual(enum, bases)
                        continue
                    for value in [*enum, node.get("const")]:
                        if isinstance(value, str):
                            self.assertTrue(SNAKE_CASE.fullmatch(value), value)

    def test_booleans_are_only_switches_and_a_mode_is_prototype_or_live(self):
        for name in ("deploy", "build"):
            for path, node, _ in settings_file.schema_keys(FILE_TYPES[name].load()):
                last = path.rsplit(".", 1)[-1]
                with self.subTest(file=name, key=path):
                    self.assertEqual("boolean" in types(node), last == "enabled")
                    if last == "enabled":
                        self.assertEqual(types(node), ["boolean"])
                    if last == "mode":
                        self.assertEqual(node.get("enum"), ["prototype", "live"])

    def test_no_rejected_word_in_a_name_a_value_or_a_text(self):
        for name, file_type in FILE_TYPES.items():
            root = file_type.load()
            for text in texts(root):
                with self.subTest(file=name, text=text[:70]):
                    for words in (REJECTED, UNWANTED, CLOSING_WORDS):
                        self.assertIsNone(words.search(text))
            for path, node, _ in properties(root):
                values = [node.get("enum") or []] + [
                    node[k] for k in VALUES if k in node
                ]
                for text in [path, *strings(values)]:
                    self.assertIsNone(
                        REJECTED.search(re.sub(r"[_.]", " ", text)),
                        f"{name}: {path}: {text}",
                    )

    def test_the_word_lists_find_whole_words_in_their_case(self):
        """tests/doc_words.yaml's lists find a word or a pattern's match whole, in any case where the list does not
        say match_case, and never in text in backticks."""
        self.assertTrue(REJECTED.search("It Holds one."))
        self.assertIsNone(REJECTED.search("the actor's household"))
        self.assertTrue(UNWANTED.search("new in 2.0"))
        self.assertIsNone(UNWANTED.search("new in 20 places"))
        self.assertTrue(UNWANTED.search("the would-send list"))
        self.assertTrue(CLOSING_WORDS.search("Blank: none."))
        self.assertIsNone(CLOSING_WORDS.search("leave it blank: none."))
        self.assertIsNone(
            REJECTED.search('production\'s `holds true` and `phase "observe"`')
        )
        self.assertTrue(REJECTED.search("`holds true` holds"))
        self.assertTrue(REJECTED.search("an unclosed ` holds"))

    def test_a_key_says_where_its_value_comes_from_when_the_template_has_a_placeholder(
        self,
    ):
        """Each placeholder's row says where the value comes from. Other keys may say it too, for the fixes of the
        checks across keys (tests/test_coherence.py)."""
        for name, file_type in FILE_TYPES.items():
            for path, node, _ in settings_file.schema_keys(file_type.load()):
                with self.subTest(file=name, key=path):
                    value = settings_file.template_value(node, path)
                    if has_placeholder(value):
                        self.assertIn("x-source", node)
                    if "x-source" in node:
                        self.assertTrue(node["x-source"].strip())

    def test_the_not_used_reasons_and_no_others(self):
        """x-off-reason on NOT_USED's sections and map, and on nothing else: the template writes those {} with
        why nothing uses them, and leaves out any other key that nothing uses."""
        found = {
            (name, path): node["x-off-reason"]
            for name, file_type in FILE_TYPES.items()
            for path, node, _ in properties(file_type.load())
            if "x-off-reason" in node
        }
        self.assertEqual(set(found), NOT_USED)
        for where, reason in found.items():
            with self.subTest(key=where):
                # It reads after "while": no word of its own before it, and no full stop.
                self.assertRegex(
                    reason, r"^(?!(?:because|while|since|when)\b)[a-z].*[^.]$"
                )

    def test_the_atlassian_cli_section_says_which_tool_it_is(self):
        """The section's description is the one home of which tool it is and of its commands."""
        text = file_types.BUILD.load()["properties"]["atlassian_cli"]["description"]
        self.assertIn("the atlassian tool, whose commands are", text)

    def test_amds_public_rocm_installer_is_checked_only_with_its_sha256(self):
        """A package_url requires its SHA-256, and AMD's public installer is checked when you set one."""
        rocm = file_types.BUILD.load()["properties"]["rocm"]["properties"]
        closing = {
            key: settings_file.closing_of(rocm[key]["description"])[1]
            for key in ("package_url", "package_sha256")
        }
        self.assertEqual(
            closing,
            {
                "package_url": "Leave blank to use AMD's public installer for this version, which is checked only "
                "when you set package_sha256.",
                "package_sha256": "Required when you set package_url; set it anyway to check AMD's public "
                "installer.",
            },
        )

    def test_gpu_py_takes_each_remote_gpu_tests_form_from_the_schema(self):
        """rkmon_lib/gpu.py's problems checks each remote GPU tests' value by the form of its key in the schema
        (gpu.forms, through settings_file.misfits): the schema refuses a sample, its pattern read as validate reads
        it, just when gpu.py does, in a line that names the key as the file does, with no form of gpu.py's own beside
        it. Whether the default target is one of gpu_types' targets crosses keys.
        """
        root = file_types.DEPLOY.load()
        nodes = {p: n for p, n, _ in properties(root) if p.startswith(f"{RGT}.")}
        # For each map, a name and a value it takes, for the other half of an entry.
        entry = {"gpu.gres": ("gfx942", "mi300x"), "gpu.env": ("NAME", "1")}
        samples = (
            "login.example.com",
            "-oProxyCommand=x",
            "user@host",
            "host:22",
            "9user",
            "_svc",
            "aé",
            "a b",
            "a/b",
            "a:b",
            "p&q|r!s,t",
            "node[01-04]",
            "90",
            "90:00",
            "1-02:00:00",
            "gfx942",
            "GFX942",
            "x\n",
            "a\nb",
            "a\rb",
            "a\x00b",
        )
        compared, looser, stricter = [], set(), set()
        for path, node in nodes.items():
            internal = (internal_names(node) or [""])[0]
            if not internal or internal == "gpu.default_arch":
                continue
            name, value = entry.get(internal, ("", ""))
            parts = (
                ("", node, lambda s: s),
                ("[]", node.get("items") or {}, lambda s: [s]),
                (".*", node.get("additionalProperties"), lambda s: {name: s}),
                (" names", node.get("propertyNames") or {}, lambda s: {s: value}),
            )
            for part, form, written in parts:
                if not isinstance(form, dict) or "pattern" not in form:
                    continue
                what = path.rsplit(".", 1)[-1] + part
                compared.append(what)
                for sample in samples:
                    cfg = {"gpu": {internal.split(".", 1)[1]: written(sample)}}
                    by_gpu = any(p.startswith(f"{path} is ") for p in gpu.problems(cfg))
                    by_schema = not settings_file.schema_pattern(
                        form["pattern"]
                    ).search(sample)
                    if by_schema != by_gpu:
                        (stricter if by_schema else looser).add((what, sample))
        self.assertEqual(
            compared,
            [
                "login_host",
                "ssh_user",
                "partition",
                "accounts[]",
                "qos",
                "constraint",
                "gpu_types.*",
                "gpu_types names",
                "time_limit",
                "excluded_nodes[]",
                "work_folder",
                "python",
                "setup_command",
                "environment.*",
                "environment names",
                "clone_url",
            ],
        )
        self.assertEqual((looser, stricter), (set(), set()))

    def test_the_web_hosts_forms_are_the_ones_the_writer_checks(self):
        """The writer takes from the configuration a plain login and host name (publish.PLAIN_NAME), and the page
        and the requests file the site folder gives with no part that starts with `.`, and none that starts with `/`,
        `~` or `-` (publish.target_refusal): web_host and username are a plain name, and site_folder plain names
        joined by `/`, each blank or the folder with a `/` after it, so `rkmon check` refuses first whatever the
        writer refuses later. No site folder holds the sign that stands for its `/` in the requests file's name
        (derived.site_requests), so two sites never share one."""
        nodes = {
            p.rsplit(".", 1)[1]: n
            for p, n, _ in properties(file_types.DEPLOY.load())
            if p.startswith(f"{TEAM}.publish.")
        }
        plain = publish.PLAIN_NAME.pattern
        self.assertEqual(
            {name: node["pattern"] for name, node in nodes.items()},
            {
                "web_host": f"^(?:{plain})?$",
                "username": f"^(?:{plain})?$",
                "site_folder": f"^(?:{plain}(?:/{plain})*/?)?$",
            },
        )
        site = settings_file.schema_pattern(nodes["site_folder"]["pattern"])
        samples = (
            "public_html/my-team",
            "public_html/my-team/",
            "site",
            "a.b/c_d/e-f",
            "",
            "/var/www",
            "~/site",
            "-e",
            "site/../x",
            ".ssh",
            "site/.profile",
            "a/-b",
            "a//b",
            "/",
            "a b",
            "a:b",
            "a\u00e9",
            "x\n",
            f"a{derived.FOLDER_SIGN}b",
        )
        taken = []
        for sample in samples:
            html = {
                "host": "web.example.com",
                "user": "op",
                "dest": derived.site_page(sample),
                "requests": derived.site_requests(sample),
            }
            cfg = {"publish": {"html": html}}
            by_writer = any(
                publish.target_refusal(cfg, k) for k in ("dest", "requests")
            )
            by_schema = not site.search(sample)
            with self.subTest(sample=sample):
                self.assertTrue(
                    by_schema or not by_writer
                )  # never looser than the writer
            if not by_schema:
                taken.append(sample)
        self.assertEqual(taken, list(samples[:5]))

    def test_a_new_settings_file_fails_only_where_you_fill_in_a_value(self):
        """A new file's FAIL rows, in each profile, are the keys you fill in (a placeholder in the template), one row
        each: the placeholder's, not its form's as well, and the checks across keys wait for them (a SKIP row).
        Filled in, the file passes its checks, those across keys included: the build file with its check 4's
        WARNs alone (NOT_CHECKED).
        """
        templates = [(file_types.BUILD, None)] + [
            (file_types.DEPLOY, p.name)
            for p in settings_file.profiles(file_types.DEPLOY)
        ]
        self.assertEqual(len(templates), 4)
        for file_type, profile in templates:
            with self.subTest(
                file=file_type.kind, profile=profile
            ), tempfile.TemporaryDirectory() as d:
                path = pathlib.Path(d) / "settings.yaml"
                text = settings_file.template(file_type, profile)
                path.write_text(text, encoding="utf-8")
                doc = settings_file.read(path, file_type)
                rows = settings_file.validate(doc)
                keys = to_fill(file_type)
                self.assertEqual(open_rows(rows), (keys, WAITING))
                for row in rows:
                    if row.status == Status.FAIL:
                        self.assertTrue(
                            row.found.startswith("still the template's <"), row
                        )
                for dotted in keys:
                    fill(doc.values, dotted, FILLED[dotted])
                rows = settings_file.validate(doc)
                self.assertEqual(
                    sorted((r.name, r.status) for r in rows if r.status != Status.PASS),
                    NOT_CHECKED[file_type.kind],
                )

    def test_each_profile_sets_its_values(self):
        """Each value a deploy profile sets is in the file its template writes, and the three differ."""
        deploy = file_types.DEPLOY
        profiles = settings_file.profiles(deploy)
        self.assertEqual(
            [p.name for p in profiles], ["full", "review-only", "merge-queue"]
        )
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            for profile in profiles:
                with self.subTest(profile=profile.name):
                    path.write_text(
                        settings_file.template(deploy, profile.name), encoding="utf-8"
                    )
                    values = settings_file.read(path, deploy).values
                    for dotted, value in profile.values.items():
                        self.assertTrue(found(values, dotted), dotted)
                        self.assertEqual(value_at(values, dotted), value)
        self.assertEqual(len({str(sorted(p.values.items())) for p in profiles}), 3)

    def test_an_image_name_with_a_registry_in_it_says_where_the_registry_goes(self):
        """The image's name takes no registry's address (its pattern): in both files, its row's fix says that
        the registry goes in image.registry."""
        for file_type, profile in (
            (file_types.BUILD, None),
            (file_types.DEPLOY, "full"),
        ):
            with self.subTest(file=file_type.kind), tempfile.TemporaryDirectory() as d:
                path = pathlib.Path(d) / "settings.yaml"
                path.write_text(
                    settings_file.template(file_type, profile), encoding="utf-8"
                )
                doc = settings_file.read(path, file_type)
                for dotted in to_fill(file_type):
                    fill(doc.values, dotted, FILLED[dotted])
                fill(doc.values, "image.name", "registry.example.com/team/rkmon")
                (row,) = [
                    r for r in settings_file.validate(doc) if r.status == Status.FAIL
                ]
                self.assertEqual(row.name, "image.name")
                self.assertIn(
                    "without its registry, which goes in image.registry", str(row.fix)
                )

    def test_no_profile_has_the_updates_section(self):
        """The file does not require it, and every profile leaves it out; the every-key text writes it after image."""
        deploy = file_types.DEPLOY
        self.assertNotIn(UPDATES, deploy.load()["required"])
        for profile in settings_file.profiles(deploy):
            with self.subTest(profile=profile.name):
                self.assertNotIn(
                    f"\n{UPDATES}:", settings_file.template(deploy, profile.name)
                )
        self.assertIn(
            f'\n  tag: "{image.package_version(ROOT)}"\n\n{UPDATES_TEXT}\n# Repository: ',
            settings_file.template(deploy, every_key=True),
        )

    def test_the_section_command_prints_the_updates_section(self):
        self.assertEqual(
            settings_file.section(file_types.DEPLOY, UPDATES), UPDATES_TEXT
        )

    def test_the_updates_section_with_its_placeholders_filled_in_passes(self):
        """Pasted into each profile's file after image, as the section command prints it: one row for each of its
        placeholders, beside the profile's own. With its placeholders filled in, and the profile's, the file passes
        the checks that rkmon check runs on it."""
        deploy = file_types.DEPLOY
        keys = [f"{UPDATES}.repository", f"{UPDATES}.branch"]
        filled = UPDATES_TEXT
        for key, placeholder in zip(keys, ("<org>/<repo>", "<branch>")):
            line = '  {}: "{}"\n'.format(key.split(".")[1], "{}")
            self.assertIn(line.format(placeholder), filled)
            filled = filled.replace(line.format(placeholder), line.format(FILLED[key]))
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            for profile in settings_file.profiles(deploy):
                text = settings_file.template(deploy, profile.name)
                at = text.index("\n# Repository: ")
                for section, rows_for in ((UPDATES_TEXT, keys), (filled, [])):
                    with self.subTest(profile=profile.name, filled=not rows_for):
                        path.write_text(
                            text[:at] + "\n" + section + text[at:], encoding="utf-8"
                        )
                        doc = settings_file.read(path, deploy)
                        self.assertEqual(
                            open_rows(settings_file.validate(doc)),
                            (sorted(to_fill(deploy) + rows_for), WAITING),
                        )
                        for dotted in to_fill(deploy):
                            fill(doc.values, dotted, FILLED[dotted])
                self.assertEqual(
                    doc.values[UPDATES],
                    {"repository": FILLED[keys[0]], "branch": FILLED[keys[1]]},
                )
                rows = settings_file.validate(doc)
                self.assertEqual([r for r in rows if r.status != Status.PASS], [])

    def test_the_key_counts(self):
        """113 deploy keys, the header's two included and each named map counted once, 111 of them in the every-key
        text, which comments out the serve block that delivery "publish" does not choose; 21 build keys.
        """
        self.assertEqual(len(KEYS["deploy"]), 113)
        self.assertEqual(len(KEYS["build"]), 21)
        deploy = file_types.DEPLOY
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            path.write_text(
                settings_file.template(deploy, every_key=True), encoding="utf-8"
            )
            values = settings_file.read(path, deploy).values
        values = {"kind": deploy.kind, "schema_version": deploy.version, **values}
        shown = [k for k in KEYS["deploy"] if found(values, k)]
        self.assertEqual(len(shown), 111)
        self.assertEqual(
            [k for k in KEYS["deploy"] if k not in shown],
            [f"{TEAM}.serve.address", f"{TEAM}.serve.port"],
        )

    def test_the_deploy_templates_image_tag_is_the_package_version(self):
        """Not a placeholder: the version this package builds, which a blank tag in the build file means."""
        version = image.package_version(ROOT)
        self.assertTrue(version)
        node = file_types.DEPLOY.load()["properties"]["image"]["properties"]["tag"]
        self.assertEqual(settings_file.template_value(node, "image.tag"), version)
        self.assertIn(
            f'\n  tag: "{version}"\n', settings_file.template(file_types.DEPLOY)
        )

    def test_a_binding_pasted_with_its_placeholders_has_one_row_for_each_key(self):
        """A binding pasted into model_bindings as the template gives it, in each of its forms: named "<name>", with
        its own placeholders inside. Each key has one row, the placeholder's: the entry's name has its rename row,
        and no row beside it for the map's rule for names."""
        deploy = file_types.DEPLOY
        root = deploy.load()
        entries = _deref(
            root, root["properties"]["model_bindings"]["additionalProperties"]
        )
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            path.write_text(settings_file.template(deploy), encoding="utf-8")
            values = settings_file.read(path, deploy).values
            # The template has review automation on, which reads the bindings.
            self.assertIs(values["features"]["review_automation"]["enabled"], True)
            header = {"kind": deploy.kind, "schema_version": deploy.version}
            for form in entries["oneOf"]:
                entry = template_entry(root, form)
                with self.subTest(provider=entry["provider"]):
                    values["model_bindings"] = {"<name>": entry}
                    path.write_text(json.dumps({**header, **values}), encoding="utf-8")
                    rows = [
                        r
                        for r in settings_file.validate(
                            settings_file.read(path, deploy)
                        )
                        if r.name.split(".")[0] == "model_bindings"
                    ]
                    to_fill = [
                        f"model_bindings.<name>.{key}"
                        for key, value in entry.items()
                        if has_placeholder(value)
                    ]
                    self.assertEqual(
                        sorted(r.name for r in rows),
                        sorted(["model_bindings", *to_fill]),
                    )
                    for row in rows:
                        self.assertTrue(
                            row.found.startswith("still the template's <"), row
                        )

    def test_the_every_key_text_comments_out_each_form_and_each_unchosen_block(self):
        """The every-key text, which the section command prints, writes each form of a model binding and a price's
        entry, under the name its map's propertyNames give, and the team dashboard's serve block, which delivery
        "publish" does not choose: each commented out, for you to use."""
        deploy = file_types.DEPLOY
        text = settings_file.section(deploy, "model_bindings")
        self.assertIn("\nmodel_bindings: {}\n", text)
        self.assertEqual(text.count("\n#   <name>:\n"), 3)
        for provider in ("claude", "openai_compatible", "ollama"):
            self.assertIn(f'\n#     provider: "{provider}"\n', text)
        text = settings_file.section(deploy, f"{RA}.spend")
        self.assertIn("\n      usd_per_million_tokens: {}\n", text)
        self.assertIn("\n      #   claude-opus-5-5:\n", text)
        text = settings_file.section(deploy, TEAM)
        self.assertIn('\n      delivery: "publish"\n', text)
        self.assertIn("\n      # serve:\n", text)
        self.assertNotIn("\n      serve:\n", text)

    def test_the_internal_states_template_is_a_state_file_that_passes(self):
        state = file_types.INTERNAL_STATE
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "internal_state.yaml"
            path.write_text(settings_file.template(state), encoding="utf-8")
            rows = settings_file.validate(settings_file.read(path, state))
        self.assertEqual([r for r in rows if r.status == Status.FAIL], [])


if __name__ == "__main__":
    unittest.main()
