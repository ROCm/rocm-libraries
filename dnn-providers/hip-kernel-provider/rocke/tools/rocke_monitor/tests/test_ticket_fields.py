# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fields a new JIRA ticket needs (rkmon_lib/ticket_fields.py; check 6): the issue type a ticket
gets, here and in the filer; the fields a new ticket gets without a setting for each, and those it gets while their
setting lists something or is blank, as the filer sends them; the fields a project's create metadata requires besides those;
check 6's rows, word for word as they print; and the module read on the host's Python, as the floor test reads
what rkmon loads there.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import test_python_floor as floor  # noqa: E402
from rkmon_lib import checks, deployment, registry, settings_file  # noqa: E402
from rkmon_lib import ticket_fields as tf  # noqa: E402
from rkmon_lib import tickets  # noqa: E402
from rkmon_lib.checks import Added, Check, Described, Edit, Steps  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

DEPLOY = "~/rkmon-deploy.yaml"
# An invented project key, built at run time (the privacy check looks for <KEY>-<n>), and an epic in it.
PROJ = "TK" + "T"
EPIC = f"{PROJ}-7"
SITE = "example-site"  # the atlassian CLI's name for the site the reads go to (the fake CLI's own)

# A project's create metadata, as JIRA lists it: its issue types, then the fields of its "Task".
TYPES = [
    {"id": "10001", "name": "Task", "subtask": False},
    {"id": "10002", "name": "Bug", "subtask": False},
    {"id": "10003", "name": "Story", "subtask": False},
    {"id": "10004", "name": "Sub-task", "subtask": True},
]
FILLED_BY_THE_MONITOR = [
    {
        "fieldId": "project",
        "name": "Project",
        "required": True,
        "hasDefaultValue": False,
    },
    {
        "fieldId": "summary",
        "name": "Summary",
        "required": True,
        "hasDefaultValue": False,
    },
    {
        "fieldId": "issuetype",
        "name": "Issue Type",
        "required": True,
        "hasDefaultValue": False,
        "allowedValues": [{"id": "10001", "name": "Task"}],
    },
    {
        "fieldId": "description",
        "name": "Description",
        "required": True,
        "hasDefaultValue": False,
    },
    {
        "fieldId": "priority",
        "name": "Priority",
        "required": True,
        "hasDefaultValue": False,
        "allowedValues": [{"id": "1", "name": "Major"}, {"id": "2", "name": "Minor"}],
    },
]
LABELS = {
    "fieldId": "labels",
    "name": "Labels",
    "required": True,
    "hasDefaultValue": False,
}
COMPONENTS = {
    "fieldId": "components",
    "name": "Components",
    "required": True,
    "hasDefaultValue": False,
    "allowedValues": [
        {"id": "1", "name": "Platform"},
        {"id": "2", "name": "Platform CI"},
    ],
}
SEVERITY = {
    "fieldId": "customfield_10010",
    "name": "Severity",
    "required": True,
    "hasDefaultValue": False,
    "allowedValues": [
        {"id": "1", "value": "Major"},
        {"id": "2", "value": "Moderate"},
        {"id": "3", "value": "Retired", "disabled": True},
    ],
}
TEAM = {
    "key": "customfield_10020",
    "name": "Team",
    "required": True,
    "hasDefaultValue": False,
}
# Required with no default, as some JIRA sites list it, though JIRA fills it itself with the login that files the ticket.
REPORTER = {
    "fieldId": "reporter",
    "name": "Reporter",
    "required": True,
    "hasDefaultValue": False,
}
FIELDS = [
    *FILLED_BY_THE_MONITOR,
    REPORTER,
    LABELS,
    COMPONENTS,
    SEVERITY,
    {
        "fieldId": "customfield_10030",
        "name": "Sprint",
        "required": False,
        "hasDefaultValue": False,
    },
    TEAM,
]
# The settings of features.ticket_creation that give FIELDS' labels and components.
LISTED = {"labels": ["monitor"], "components": ["Platform"]}
# A field a ticket may have, which takes some of a list.
PLATFORMS = {
    "fieldId": "customfield_10040",
    "name": "Platforms",
    "required": False,
    "hasDefaultValue": False,
    "allowedValues": [{"id": "21", "value": "Linux"}, {"id": "22", "value": "Windows"}],
}


class NotRead:
    """FIELDS where the rows must not read them: iterating them fails the test."""

    def __iter__(self):
        raise AssertionError("the fields were read")


class TestTheIssueType(unittest.TestCase):
    def test_a_blank_setting_means_the_default(self):
        self.assertEqual(tf.DEFAULT_ISSUE_TYPE, "Task")
        for value, want in (
            ("", "Task"),
            (None, "Task"),
            ("   ", "Task"),
            ("Story", "Story"),
            ("  Story ", "Story"),
        ):
            with self.subTest(value=value):
                self.assertEqual(tf.issue_type(value), want)

    def test_the_type_s_id_as_jira_writes_its_name(self):
        self.assertEqual(tf.type_id(TYPES, ""), "10001")
        self.assertEqual(tf.type_id(TYPES, " Story "), "10003")
        self.assertEqual(tf.type_id(TYPES, "story"), "")  # case and all
        self.assertEqual(tf.type_id(TYPES, "Epic"), "")
        self.assertEqual(
            tf.type_id(TYPES, "Sub-task"), ""
        )  # a new ticket of it needs a parent
        self.assertEqual(tf.type_id([{"id": 7, "name": "Task"}], None), "7")
        self.assertEqual(tf.type_id([{"name": "Task"}, "Task", None], ""), "")
        self.assertEqual(tf.type_id(None, "Task"), "")

    def test_the_filer_gives_a_draft_without_a_type_the_rule_s_type(self):
        """rkmon_lib/tickets.py's `add` takes the type from the rule here: a default other than "Task" shows it."""
        draft = tickets.parse(
            "---\ntitle: A case is skipped on every runner\npriority: P3\nsummary: one case reports SKIPPED\n---\n"
            "What happens: it is skipped.\n"
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            tf, "DEFAULT_ISSUE_TYPE", "Story"
        ):
            for n, (setting, given, want) in enumerate(
                (
                    ("", "", "Story"),
                    ("  ", "", "Story"),
                    (" Bug ", "", "Bug"),
                    ("Bug", "Epic", "Epic"),
                )
            ):
                with self.subTest(setting=setting, given=given):
                    store = Store(pathlib.Path(tmp) / f"state{n}")
                    store.save(empty_state())
                    cfg = {
                        "jira": {"project": PROJ, "issue_type": setting},
                        "paths": {"working_dir": tmp},
                    }
                    with store.locked(wait=True):
                        st = store.load()
                        added, why, tid = tickets.add(
                            cfg, st, store, dict(draft, type=given)
                        )
                        store.save(st)
                    self.assertTrue(added, why)
                    self.assertEqual(tickets.find(store.load(), tid)["type"], want)


# The flags of `atlassian jira issue create` by the JIRA field each fills; --field and --field-json name theirs.
FLAGS = {
    "--project": "project",
    "--summary": "summary",
    "--type": "issuetype",
    "--priority": "priority",
    "--labels": "labels",
}
JIRA_FILLS = {
    "reporter"
}  # the fields of FILLED that JIRA fills itself: the filer sends none of them


def sent(jira: dict) -> list:
    """The JIRA fields the filer sends for a draft with a priority and no labels or component of its own, added
    (rkmon_lib/tickets.py, `add`) and filed (`file_ticket`) with JIRA as the configuration's jira section.
    """
    draft = tickets.parse(
        "---\ntitle: A case is skipped on every runner\npriority: P2\nsummary: one case reports SKIPPED\n---\n"
        "What happens: it is skipped.\n"
    )
    calls = []

    def run(cmd):
        calls.append(cmd)
        return 0, json.dumps({"key": f"{PROJ}-1"})

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "jira": dict(
                jira,
                project=PROJ,
                url="https://jira.example.com",
                priority_map={"P2": "Major"},
            ),
            "atlassian": {"site": "example-site"},
            "paths": {"working_dir": tmp},
        }
        store = Store(pathlib.Path(tmp) / "state")
        store.save(empty_state())
        with store.locked(wait=True):
            st = store.load()
            added, why, tid = tickets.add(cfg, st, store, draft)
            store.save(st)
        assert added, why
        tickets.file_ticket(cfg, tickets.find(store.load(), tid), "op", run=run)
    [cmd] = calls
    words = cmd[cmd.index("create") + 1 :]
    return [
        value.split("=", 1)[0] if flag in ("--field", "--field-json") else FLAGS[flag]
        for flag, value in zip(words[::2], words[1::2])
    ]


class TestTheFields(unittest.TestCase):
    def test_filled_is_what_the_filer_fills(self):
        """FILLED, less the fields JIRA fills itself, is what the filer sends with an epic and without labels,
        components or required_fields in the settings."""
        self.assertLessEqual(JIRA_FILLS, set(tf.FILLED))
        self.assertEqual(
            sorted(sent({"default_epic": EPIC})), sorted(set(tf.FILLED) - JIRA_FILLS)
        )

    def test_filled_while_blank_is_what_the_filer_sends_while_blank(self):
        """Each field of FILLED_WHILE_BLANK is sent with the new ticket while its setting is blank, and only then:
        with an epic, the filer gives the description after the link to it (tickets.link_epic).
        """
        always = set(tf.FILLED) - JIRA_FILLS
        self.assertEqual(set(sent({})) - always, set(tf.FILLED_WHILE_BLANK))
        for jira, more in (
            ({"default_epic": ""}, {"description"}),
            ({"default_epic": " "}, {"description"}),
            ({"default_epic": EPIC}, set()),
        ):
            with self.subTest(jira=jira):
                self.assertEqual(sorted(sent(jira)), sorted(always | more))

    def test_filled_while_set_is_what_the_filer_sends_while_set(self):
        """Each field of FILLED_WHILE_SET is sent while its setting lists something, and only then."""
        always = set(tf.FILLED) - JIRA_FILLS | set(tf.FILLED_WHILE_BLANK)  # no epic
        both = {"labels": ["monitor"], "component": "Platform, Platform CI"}
        self.assertEqual(set(sent(both)) - always, set(tf.FILLED_WHILE_SET))
        for jira, more in (
            ({"labels": ["monitor"]}, {"labels"}),
            ({"component": "Platform"}, {"components"}),
            ({"labels": [], "component": " "}, set()),
        ):
            with self.subTest(jira=jira):
                self.assertEqual(sorted(sent(jira)), sorted(always | more))

    def test_the_fields_a_new_ticket_needs(self):
        severity = tf.Field("customfield_10010", "Severity", ("Major", "Moderate"))
        team = tf.Field("customfield_10020", "Team")
        self.assertEqual(tf.needed(FIELDS, LISTED), [severity, team])
        self.assertEqual(tf.needed([*FILLED_BY_THE_MONITOR, REPORTER], {}), [])
        self.assertEqual(tf.needed(None, LISTED), [])
        self.assertEqual(
            tf.needed(
                [
                    {"fieldId": "customfield_1", "required": True},  # no name: its id
                    {"name": "No id", "required": True},
                    {"fieldId": "customfield_2", "name": "Maybe", "required": "true"},
                    "customfield_3",
                    {
                        "fieldId": "customfield_4",
                        "name": "Kind",
                        "required": True,
                        "allowedValues": [
                            {"name": "A"},
                            {"value": ""},
                            {"id": "9"},
                            "B",
                            {"value": 2},
                        ],
                    },
                ],
                None,
            ),
            [
                tf.Field("customfield_1", "customfield_1"),
                tf.Field("customfield_4", "Kind", ("A", "2")),
            ],
        )

    def test_the_labels_and_the_components_while_their_setting_lists_none(self):
        labels = tf.Field("labels", "Labels")
        components = tf.Field("components", "Components", ("Platform", "Platform CI"))
        both = [*FILLED_BY_THE_MONITOR, LABELS, COMPONENTS]
        for settings, want in (
            (LISTED, []),
            ({}, [labels, components]),
            (None, [labels, components]),
            ({"labels": ["monitor"]}, [components]),
            ({"labels": ["", " "], "components": ["Platform"]}, [labels]),
            ({"labels": [], "components": []}, [labels, components]),
        ):
            with self.subTest(settings=settings):
                self.assertEqual(tf.needed(both, settings), want)
        self.assertEqual(
            tf.missing(both, {"required_fields": {"labels": ["monitor"]}}),
            [components],
        )

    def test_the_description_while_an_epic_is_set(self):
        description = tf.Field("description", "Description")
        for settings, want in (
            ({"epic_key": EPIC}, [description]),
            ({"epic_key": ""}, []),
            ({"epic_key": "  "}, []),
            ({"epic_key": None}, []),
            ({}, []),
        ):
            with self.subTest(settings=settings):
                self.assertEqual(tf.needed(FILLED_BY_THE_MONITOR, settings), want)
        self.assertEqual(
            tf.missing(
                FILLED_BY_THE_MONITOR,
                {"epic_key": EPIC, "required_fields": {"description": "What happens."}},
            ),
            [],
        )

    def test_the_fields_required_fields_does_not_set(self):
        severity, team = tf.needed(FIELDS, LISTED)
        for given, want in (
            ({}, [severity, team]),
            (None, [severity, team]),
            ("customfield_10010: Major", [severity, team]),
            ({"customfield_10010": {"value": "Major"}}, [team]),
            ({"customfield_10010": "Major", "customfield_10020": "Platform"}, []),
            ({"customfield_10010": "", "customfield_10020": None}, [severity, team]),
            ({"customfield_10010": [], "customfield_10020": {}}, [severity, team]),
            ({"customfield_10010": 0, "customfield_10020": False}, []),
        ):
            with self.subTest(given=given):
                self.assertEqual(
                    tf.missing(FIELDS, dict(LISTED, required_fields=given)), want
                )

    def test_the_names_are_the_registry_s(self):
        """The settings' file keys, as the registry ties each to the name the filer reads it by (jira.*)."""
        for key, name in (
            (tf.ISSUE_TYPE_KEY, "jira.issue_type"),
            (tf.REQUIRED_FIELDS_KEY, "jira.extra_fields"),
            (tf.PROJECT_KEY, "jira.project"),
            (f"{tf.SECTION}.{tf.FILLED_WHILE_SET['labels']}", "jira.labels"),
            (f"{tf.SECTION}.{tf.FILLED_WHILE_SET['components']}", "jira.component"),
            (
                f"{tf.SECTION}.{tf.FILLED_WHILE_BLANK['description']}",
                "jira.default_epic",
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(registry.key(name).file_keys, (key,))

    def test_the_rows_name_is_the_feature_s_title(self):
        """As the other checks of the deploy settings file name a feature's rows (rkmon_lib/coherence.py)."""
        root = json.loads(
            (ROOT / "schema" / "deploy_settings.schema.json").read_text(
                encoding="utf-8"
            )
        )
        title = settings_file.schema_at(root, tf.SECTION)["title"]
        self.assertEqual(tf.NAME, title[:1].lower() + title[1:])


def fail(found: str, fix) -> Check:
    return Check("ticket creation", "FAIL", found, fix)


class TestTheRows(unittest.TestCase):
    def test_a_required_field_s_row_word_for_word(self):
        """Placeholders for the field give its row, as tests/test_checks.py renders it, and its fix, which adds the
        field's line under required_fields; the line has the key in quotes, as the file takes a key that is more than
        letters, digits and _."""
        field = {
            "fieldId": "<field id>",
            "name": "<field name>",
            "required": True,
            "allowedValues": [{"value": "<value>"}, {"value": "<value>"}],
        }
        got = tf.rows(TYPES, [field], {}, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the project requires <field id> ("<field name>") on a new "Task", and\n'
                    "required_fields does not set it",
                    Edit(
                        "features.ticket_creation.required_fields",
                        DEPLOY,
                        Added(('"<field id>": "<one of: <value>, <value>>"',)),
                    ),
                )
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project requires <field id> ("<field name>") on a new "Task", and
        required_fields does not set it
        fix: add under features.ticket_creation.required_fields in ~/rkmon-deploy.yaml:
               "<field id>": "<one of: <value>, <value>>\"""",
        )

    def test_a_long_field_name(self):
        """The line break comes only where the words before it fit on the row's first line: else the row wraps the
        two as one sentence, with no short line between them."""
        field = dict(TEAM, name="Affected platform for the regression analysis")
        got = tf.rows(TYPES, [*FILLED_BY_THE_MONITOR, field], {}, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the project requires customfield_10020 ("Affected platform for the regression analysis") on a '
                    'new "Task", and required_fields does not set it',
                    Edit(
                        "features.ticket_creation.required_fields",
                        DEPLOY,
                        Added(('customfield_10020: "<value>"',)),
                    ),
                )
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project requires customfield_10020 ("Affected platform for the regression analysis") on
        a new "Task", and required_fields does not set it
        fix: add under features.ticket_creation.required_fields in ~/rkmon-deploy.yaml:
               customfield_10020: "<value>\"""",
        )
        # The longest words before the break that keep it fill the row's first line; one letter more, and they flow.
        for name, between in (
            ("Platform the bug was seen on", "\n"),
            ("Platforms the bug was seen on", " "),
        ):
            with self.subTest(name=name):
                got = tf.rows(TYPES, [dict(TEAM, name=name)], {}, DEPLOY)
                self.assertEqual(
                    got[0].found,
                    f'the project requires customfield_10020 ("{name}") on a new "Task", and{between}'
                    "required_fields does not set it",
                )
                first = checks.render(got).splitlines()[0]
                self.assertEqual(len(first) == checks.WIDTH, between == "\n")
        got = tf.rows(
            TYPES,
            [dict(LABELS, name="Labels the team triages its regressions by")],
            {},
            DEPLOY,
        )
        self.assertEqual(
            got[0].found,
            'the project requires labels ("Labels the team triages its regressions by") on a new "Task", and '
            "features.ticket_creation.labels is empty",
        )

    def test_a_row_for_each_field_required_fields_lacks(self):
        settings = dict(
            LISTED, issue_type="Task", required_fields={"customfield_10030": "Sprint 1"}
        )
        got = tf.rows(TYPES, FIELDS, settings, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the project requires customfield_10010 ("Severity") on a new "Task", and\n'
                    "required_fields does not set it",
                    Edit(
                        "features.ticket_creation.required_fields",
                        DEPLOY,
                        Added(('customfield_10010: "<one of: Major, Moderate>"',)),
                    ),
                ),
                fail(
                    'the project requires customfield_10020 ("Team") on a new "Task", and\n'
                    "required_fields does not set it",
                    Edit(
                        "features.ticket_creation.required_fields",
                        DEPLOY,
                        Added(('customfield_10020: "<value>"',)),
                    ),
                ),
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project requires customfield_10010 ("Severity") on a new "Task", and
        required_fields does not set it
        fix: add under features.ticket_creation.required_fields in ~/rkmon-deploy.yaml:
               customfield_10010: "<one of: Major, Moderate>"
  FAIL  ticket creation: the project requires customfield_10020 ("Team") on a new "Task", and
        required_fields does not set it
        fix: add under features.ticket_creation.required_fields in ~/rkmon-deploy.yaml:
               customfield_10020: "<value>\"""",
        )
        self.assertEqual(
            tf.rows(
                TYPES,
                FIELDS,
                dict(LISTED, required_fields={"customfield_10010": "Major"}),
                DEPLOY,
            ),
            got[1:],
        )

    def test_the_fix_s_line_as_json_and_back(self):
        """rk doctor keeps its rows as JSON (checks.fix_to_json): the fix that adds a field's line reads back the
        same."""
        (row,) = tf.rows(TYPES, [*FILLED_BY_THE_MONITOR, SEVERITY], {}, DEPLOY)
        data = checks.fix_to_json(row.fix)
        self.assertEqual(
            data,
            {
                "edit": ["features.ticket_creation.required_fields", DEPLOY],
                "added": ['customfield_10010: "<one of: Major, Moderate>"'],
            },
        )
        self.assertEqual(checks.fix_from_json(json.loads(json.dumps(data))), row.fix)

    def test_jira_s_texts_show_what_does_not_print_escaped(self):
        """A field's name and the values JIRA allows, as a row and its fix show them: each character that does not
        print as its JSON escape, so a terminal is given no control character of JIRA's.
        """
        field = dict(
            SEVERITY,
            name="Severity\x9b2J",
            allowedValues=[
                {"id": "1", "value": "Major\x1b[2J"},
                {"id": "2", "value": "Moderate\u2028"},
            ],
        )
        text = checks.render(
            tf.rows(TYPES, [*FILLED_BY_THE_MONITOR, field], {}, DEPLOY)
        )
        self.assertIn('customfield_10010 ("Severity\\u009b2J")', text)
        self.assertIn(
            'customfield_10010: "<one of: Major\\u001b[2J, Moderate\\u2028>"', text
        )
        self.assertTrue(all(c.isprintable() for c in text.replace("\n", "")), text)

    def test_a_row_for_each_list_setting_that_lists_none(self):
        """The labels and the components, which the filer sends only while their setting lists something: the fix
        sets that setting."""
        got = tf.rows(
            TYPES,
            [*FILLED_BY_THE_MONITOR, LABELS, COMPONENTS],
            {"labels": [" "]},
            DEPLOY,
        )
        self.assertEqual(
            got,
            [
                fail(
                    'the project requires labels ("Labels") on a new "Task", and\n'
                    "features.ticket_creation.labels is empty",
                    Edit("features.ticket_creation.labels", DEPLOY, ["<value>"]),
                ),
                fail(
                    'the project requires components ("Components") on a new "Task", and\n'
                    "features.ticket_creation.components is empty",
                    Edit(
                        "features.ticket_creation.components",
                        DEPLOY,
                        ["<one of: Platform, Platform CI>"],
                    ),
                ),
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project requires labels ("Labels") on a new "Task", and
        features.ticket_creation.labels is empty
        fix: set features.ticket_creation.labels in ~/rkmon-deploy.yaml to ["<value>"]
  FAIL  ticket creation: the project requires components ("Components") on a new "Task", and
        features.ticket_creation.components is empty
        fix: set features.ticket_creation.components in ~/rkmon-deploy.yaml to ["<one of: Platform, Platform CI>"]""",
        )
        self.assertEqual(
            tf.rows(
                TYPES,
                [*FILLED_BY_THE_MONITOR, LABELS, COMPONENTS],
                {
                    "components": ["Platform"],
                    "required_fields": {"labels": ["monitor"]},
                },
                DEPLOY,
            ),
            [
                Check(
                    "ticket creation",
                    "PASS",
                    'the project requires labels ("Labels") on a new "Task", and required_fields sets it',
                )
            ],
        )

    def test_a_required_description_while_an_epic_is_set(self):
        """With an epic, the filer gives the description after the link to it, so JIRA refuses a new ticket of a
        project that requires one: the fix clears epic_key, or has the project not require it.
        """
        got = tf.rows(TYPES, FILLED_BY_THE_MONITOR, {"epic_key": EPIC}, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the project requires description ("Description") on a new "Task", and\n'
                    "features.ticket_creation.epic_key is set: the monitor then gives the description after the link "
                    "to the epic, as one given before it can come out garbled",
                    Steps(
                        (
                            Edit("features.ticket_creation.epic_key", DEPLOY, ""),
                            ", or have the project not require a description when a ticket is created",
                        )
                    ),
                )
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project requires description ("Description") on a new "Task", and
        features.ticket_creation.epic_key is set: the monitor then gives the description after the link to the epic,
        as one given before it can come out garbled
        fix: set features.ticket_creation.epic_key in ~/rkmon-deploy.yaml to "", or have the project not require a
             description when a ticket is created""",
        )
        for settings in (
            {"epic_key": " "},
            {"epic_key": EPIC, "required_fields": {"description": "What happens."}},
        ):
            with self.subTest(settings=settings):
                self.assertEqual(
                    tf.rows(TYPES, FILLED_BY_THE_MONITOR, settings, DEPLOY)[0].status,
                    "PASS",
                )

    def test_one_pass_row_when_required_fields_gives_them_all(self):
        for fields, settings, found in (
            (
                FILLED_BY_THE_MONITOR,
                {},
                'the project requires no field on a new "Task" that the monitor does not fill itself',
            ),
            (
                [*FILLED_BY_THE_MONITOR, LABELS, COMPONENTS],
                LISTED,
                'the project requires no field on a new "Task" that the monitor does not fill itself',
            ),
            (
                [*FILLED_BY_THE_MONITOR, TEAM],
                {"required_fields": {"customfield_10020": "Platform"}},
                'the project requires customfield_10020 ("Team") on a new "Task", and required_fields sets it',
            ),
            (
                FIELDS,
                dict(
                    LISTED,
                    required_fields={
                        "customfield_10010": "Major",
                        "customfield_10020": "Platform",
                    },
                ),
                'the project requires customfield_10010 ("Severity") and customfield_10020 ("Team") on a new '
                '"Task", and required_fields sets them',
            ),
        ):
            with self.subTest(found=found):
                self.assertEqual(
                    tf.rows(TYPES, fields, settings, DEPLOY),
                    [Check("ticket creation", "PASS", found)],
                )

    def test_a_value_jira_does_not_allow(self):
        """A required_fields value that is none of the values JIRA allows for the field, required or not: one row per
        field, naming each such value, whose fix gives the values JIRA allows."""
        settings = dict(
            LISTED,
            required_fields={
                "customfield_10010": "Retired",  # disabled
                "customfield_10020": "Platform",  # Team takes any value
                "customfield_10040": ["Linux", {"value": "BSD"}, {"id": "22"}, "macOS"],
            },
        )
        got = tf.rows(TYPES, [*FIELDS, PLATFORMS], settings, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'required_fields sets customfield_10010 ("Severity") to "Retired", which the project does not '
                    'allow on a new "Task"',
                    Edit(
                        "features.ticket_creation.required_fields.customfield_10010",
                        DEPLOY,
                        "<one of: Major, Moderate>",
                    ),
                ),
                fail(
                    'required_fields sets customfield_10040 ("Platforms") to "BSD" and "macOS", which the project '
                    'does not allow on a new "Task"',
                    Edit(
                        "features.ticket_creation.required_fields.customfield_10040",
                        DEPLOY,
                        "<one of: Linux, Windows>",
                    ),
                ),
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: required_fields sets customfield_10010 ("Severity") to "Retired", which the project does not
        allow on a new "Task"
        fix: set features.ticket_creation.required_fields.customfield_10010 in ~/rkmon-deploy.yaml to
             "<one of: Major, Moderate>"
  FAIL  ticket creation: required_fields sets customfield_10040 ("Platforms") to "BSD" and "macOS", which the project
        does not allow on a new "Task"
        fix: set features.ticket_creation.required_fields.customfield_10040 in ~/rkmon-deploy.yaml to
             "<one of: Linux, Windows>\"""",
        )

    def test_the_values_jira_takes(self):
        """A value JIRA allows, by its value, its name or its id, as a word, a mapping or a list of those."""
        found = (
            'the project requires customfield_10010 ("Severity") and customfield_10020 ("Team") on a new "Task", '
            "and required_fields sets them"
        )
        for value in (
            "Major",
            "1",
            1,
            {"value": "Moderate"},
            {"id": "2"},
            {"name": "Major", "id": ""},
            ["Major", {"id": "2"}],
        ):
            with self.subTest(value=value):
                settings = dict(
                    LISTED,
                    required_fields={
                        "customfield_10010": value,
                        "customfield_10020": "Platform",
                        "customfield_10040": [{"id": "21"}, "Windows"],
                    },
                )
                self.assertEqual(
                    tf.rows(TYPES, [*FIELDS, PLATFORMS], settings, DEPLOY),
                    [Check("ticket creation", "PASS", found)],
                )

    def test_a_field_the_type_does_not_have(self):
        """JIRA refuses a new ticket that sets a field its type does not have, blank or not: the fix deletes it."""
        settings = dict(
            LISTED,
            required_fields={
                "customfield_10010": "Major",
                "customfield_10099": "x",
                "customfield_10020": "Platform",
                "customfield_10098": "",
            },
        )
        got = tf.rows(TYPES, FIELDS, settings, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'required_fields sets customfield_10099, a field the project does not have on a new "Task"',
                    Edit(
                        "features.ticket_creation.required_fields.customfield_10099",
                        DEPLOY,
                        checks.DELETE,
                    ),
                ),
                fail(
                    'required_fields sets customfield_10098, a field the project does not have on a new "Task"',
                    Edit(
                        "features.ticket_creation.required_fields.customfield_10098",
                        DEPLOY,
                        checks.DELETE,
                    ),
                ),
            ],
        )
        self.assertEqual(
            checks.render(got[:1]),
            """\
  FAIL  ticket creation: required_fields sets customfield_10099, a field the project does not have on a new "Task"
        fix: delete features.ticket_creation.required_fields.customfield_10099 from ~/rkmon-deploy.yaml""",
        )

    def test_the_rows_in_the_fields_order_then_the_fields_the_type_lacks(self):
        settings = {
            "components": ["Platform"],
            "required_fields": {
                "customfield_10099": "x",
                "customfield_10010": "Retired",
            },
        }
        self.assertEqual(
            [c.found.split(",")[0] for c in tf.rows(TYPES, FIELDS, settings, DEPLOY)],
            [
                'the project requires labels ("Labels") on a new "Task"',
                'required_fields sets customfield_10010 ("Severity") to "Retired"',
                'the project requires customfield_10020 ("Team") on a new "Task"',
                "required_fields sets customfield_10099",
            ],
        )

    def test_an_issue_type_the_project_lacks(self):
        """The row names every type the project has; its fix, each type a new ticket can be of."""
        self.assertEqual(
            tf.rows(TYPES, NotRead(), {"issue_type": "task"}, DEPLOY),
            [
                fail(
                    'the project has no issue type "task", which features.ticket_creation.issue_type names; it has '
                    '"Task", "Bug", "Story" and "Sub-task"',
                    Edit(
                        "features.ticket_creation.issue_type",
                        DEPLOY,
                        Described('one of "Task", "Bug" or "Story"'),
                    ),
                )
            ],
        )
        got = tf.rows(TYPES[1:], NotRead(), {"issue_type": " "}, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the project has no issue type "Task", which a blank features.ticket_creation.issue_type means; '
                    'it has "Bug", "Story" and "Sub-task"',
                    Edit(
                        "features.ticket_creation.issue_type",
                        DEPLOY,
                        Described('one of "Bug" or "Story"'),
                    ),
                )
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the project has no issue type "Task", which a blank features.ticket_creation.issue_type
        means; it has "Bug", "Story" and "Sub-task"
        fix: set features.ticket_creation.issue_type in ~/rkmon-deploy.yaml to one of "Bug" or "Story\"""",
        )

    def test_a_sub_task_type(self):
        """A new ticket of a sub-task type needs a parent, which the monitor never gives: the fix names each type a
        new ticket can be of, or the one."""
        got = tf.rows(TYPES, NotRead(), {"issue_type": "Sub-task"}, DEPLOY)
        self.assertEqual(
            got,
            [
                fail(
                    'the issue type "Sub-task", which features.ticket_creation.issue_type names, is a sub-task type: '
                    "a new ticket of it needs a parent, which the monitor never gives",
                    Edit(
                        "features.ticket_creation.issue_type",
                        DEPLOY,
                        Described('one of "Task", "Bug" or "Story"'),
                    ),
                )
            ],
        )
        self.assertEqual(
            checks.render(got),
            """\
  FAIL  ticket creation: the issue type "Sub-task", which features.ticket_creation.issue_type names, is a sub-task
        type: a new ticket of it needs a parent, which the monitor never gives
        fix: set features.ticket_creation.issue_type in ~/rkmon-deploy.yaml to one of "Task", "Bug" or "Story\"""",
        )
        one = [
            {"id": "1", "name": "Bug"},
            {"id": "2", "name": "Sub-task", "subtask": True},
        ]
        self.assertEqual(
            tf.rows(one, NotRead(), {"issue_type": "Sub-task"}, DEPLOY),
            [
                fail(
                    'the issue type "Sub-task", which features.ticket_creation.issue_type names, is a sub-task type: '
                    "a new ticket of it needs a parent, which the monitor never gives",
                    Edit("features.ticket_creation.issue_type", DEPLOY, "Bug"),
                )
            ],
        )

    def test_a_project_with_only_sub_task_types_for_the_login(self):
        types = [
            {"id": "5", "name": "Sub-task", "subtask": True},
            {"id": "6", "name": "Sub-bug", "subtask": True},
        ]
        self.assertEqual(
            tf.rows(types, NotRead(), {"issue_type": "Sub-task"}, DEPLOY),
            [
                fail(
                    'JIRA lists only sub-task types in the project for the atlassian login ("Sub-task" and '
                    '"Sub-bug"): a new ticket of one needs a parent, which the monitor never gives',
                    Edit(
                        "features.ticket_creation.project_key",
                        DEPLOY,
                        Described(
                            "a project the atlassian login may create tickets in"
                        ),
                    ),
                )
            ],
        )

    def test_a_project_with_no_issue_type_for_the_login(self):
        """JIRA lists the types the login may create tickets of: none, where it may not create tickets (JIRA's Create
        issues permission). The fix gives the login that permission, or names another project, then runs the check
        again."""
        want = [
            fail(
                "the atlassian login may not create tickets in this project: JIRA lists no issue type for it",
                Steps(
                    (
                        "give the atlassian login JIRA's Create issues permission in the project, or",
                        Edit(
                            "features.ticket_creation.project_key",
                            DEPLOY,
                            Described("a project where it has that permission"),
                        ),
                        "; then run",
                        *deployment.check_again(DEPLOY),
                    )
                ),
            )
        ]
        for types in ([], None, [{"name": "Task"}, {"id": "1", "name": ""}]):
            with self.subTest(types=types):
                self.assertEqual(tf.rows(types, NotRead(), {}, DEPLOY), want)
        self.assertEqual(
            checks.render(want),
            """\
  FAIL  ticket creation: the atlassian login may not create tickets in this project: JIRA lists no issue type for it
        fix: give the atlassian login JIRA's Create issues permission in the project, or set
             features.ticket_creation.project_key in ~/rkmon-deploy.yaml to a project where it has that permission;
             then run rkmon check --settings ~/rkmon-deploy.yaml""",
        )

    def test_fields_that_are_not_a_list(self):
        """FIELDS that is not a list, as JIRA gives them, is a FAIL, as for TYPES, never a PASS that read nothing: None,
        a mapping by field id (whose keys alone a loop would read) or a word."""
        want = [
            fail(
                'JIRA gave no list of the fields of a new "Task", so which ones the project requires is not known',
                Steps(("once JIRA lists them, run", *deployment.check_again(DEPLOY))),
            )
        ]
        by_id = {
            f.get("fieldId") or f["key"]: f for f in [*FILLED_BY_THE_MONITOR, TEAM]
        }
        for fields in (None, by_id, "summary"):
            with self.subTest(fields=fields):
                self.assertEqual(tf.rows(TYPES, fields, {}, DEPLOY), want)
        self.assertEqual(
            checks.render(want),
            """\
  FAIL  ticket creation: JIRA gave no list of the fields of a new "Task", so which ones the project requires is not
        known
        fix: once JIRA lists them, run rkmon check --settings ~/rkmon-deploy.yaml""",
        )
        self.assertEqual(
            tf.rows(TYPES, tuple(FILLED_BY_THE_MONITOR), {}, DEPLOY),
            tf.rows(TYPES, FILLED_BY_THE_MONITOR, {}, DEPLOY),
        )

    def test_the_steps_that_check_again_are_the_caller_s(self):
        """The steps that run the check again, in the row for no issue type and the row for no list of fields: the
        caller's AGAIN (rk doctor's: doctor.ticket_rows), else rkmon check on the file (deployment.check_again).
        """
        again = (checks.Command(("rkmon-x", "confirm")), "and")
        for kwargs, then in (
            ({}, deployment.check_again(DEPLOY)),
            ({"again": again}, again),
        ):
            with self.subTest(kwargs=kwargs):
                no_type = tf.rows([], NotRead(), {}, DEPLOY, **kwargs)[0].fix
                no_fields = tf.rows(TYPES, None, {}, DEPLOY, **kwargs)[0].fix
                self.assertEqual(no_type.parts[2:], ("; then run", *then))
                self.assertEqual(no_fields.parts, ("once JIRA lists them, run", *then))


class TestOnTheHostsPython(unittest.TestCase):
    def test_it_reads_on_the_host_s_python(self):
        """The floor test's reading (tests/test_python_floor.py) of what rkmon loads on the host, from this module,
        which check 6 runs there: the module and each of the package's files it imports.
        """
        with mock.patch.object(floor, "loaded_by_name", return_value={"ticket_fields"}):
            files = floor.host_modules()
        self.assertIn(floor.LIB / "ticket_fields.py", files)
        bad = [
            f"{f.name}:{line}: {why}"
            for f in files
            for line, why in floor.host_problems(f.read_text(encoding="utf-8"))
        ]
        self.assertEqual(bad, [], f"not read by Python {deployment.HOST_PYTHON}")


# features.ticket_creation in the configuration (ticket_fields.settings_of), on and in prototype, for project PROJ, on
# the site SITE, by its address as the deploy settings file gives it.
def config(mode: str = "prototype", **jira) -> dict:
    return {
        "features": {"ticket_creation": {"enabled": True, "mode": mode}},
        "jira": {"project": PROJ, **jira},
        "atlassian": {"site": f"https://{SITE}.atlassian.net"},
    }


class Reads:
    """A stand-in for the read through the writer: answers each path from ANSWERS (by the path without its query) and
    logs the paths asked, and, through `run`, the CLI's words."""

    def __init__(self, answers: dict):
        self.answers, self.paths, self.words = answers, [], []

    def __call__(self, path: str) -> tuple:
        self.paths.append(path)
        answer = self.answers.get(path.partition("?")[0])
        return (
            (0, "a line before the answer\n" + json.dumps(answer))
            if answer is not None
            else (1, "HTTP 404")
        )

    def run(self, words: list) -> tuple:
        """The CLI's WORDS (ticket_fields.checked's RUN): logged, then answered by their path."""
        self.words.append(words)
        return self(words[words.index("--path") + 1])


META = f"/rest/api/3/issue/createmeta/{PROJ}/issuetypes"
FIRST = "?startAt=0&maxResults=50"  # the query of the first read of a list


class TestTheListRows(unittest.TestCase):
    def test_labels_or_components_set_while_the_type_has_no_such_field(self):
        """Either list set while a new ticket of the type has no such field: a FAIL whose fix leaves it empty."""
        for key in ("labels", "components"):
            with self.subTest(key=key):
                fields = [
                    f
                    for f in FIELDS
                    if f.get("fieldId") not in ("labels", "components")
                ]
                got = tf.rows(TYPES, fields, {key: ["Platform"]}, DEPLOY)
                self.assertIn(
                    fail(
                        f'features.ticket_creation.{key} is set, and the project has no {key} field on a new "Task"',
                        Edit(f"features.ticket_creation.{key}", DEPLOY, []),
                    ),
                    got,
                )

    def test_a_component_the_field_does_not_allow_names_the_ones_it_allows(self):
        got = tf.rows(
            TYPES, FIELDS, {**LISTED, "components": ["Platform", "Web"]}, DEPLOY
        )
        allowed = '"Platform" and "Platform CI"'
        self.assertIn(
            fail(
                'features.ticket_creation.components names "Web", which the project does not allow on a new "Task"; '
                f"it allows {allowed}",
                Edit(
                    "features.ticket_creation.components",
                    DEPLOY,
                    Described(f"a list of {allowed}"),
                ),
            ),
            got,
        )
        free = tf.rows(TYPES, FIELDS, {**LISTED, "labels": ["any text"]}, DEPLOY)
        self.assertFalse(
            [r for r in free if "ticket_creation.labels" in r.found]
        )  # free text: not checked

    def test_no_list_of_types_is_one_fail(self):
        got = tf.checked(config(), DEPLOY, Reads({}).run, lambda: True, "rkmon")[0]
        self.assertEqual([r.status for r in got], [checks.Status.FAIL])
        self.assertIn("JIRA gave no list of the project's issue types", got[0].found)
        then = ("once JIRA lists them, run",)
        self.assertEqual(got[0].fix.parts, (*then, *deployment.check_again(DEPLOY)))
        again = (checks.Command(("rkmon-x", "confirm")),)
        got = tf.checked(
            config(), DEPLOY, Reads({}).run, lambda: True, "rkmon", again=again
        )[0]
        self.assertEqual(got[0].fix.parts, (*then, *again))  # the caller's steps


class TestTheRead(unittest.TestCase):
    def test_two_reads_cloud_and_server(self):
        """The types, then the fields of the type issue_type names, as Cloud ("issueTypes", "fields") and Server
        ("values") list them, each a read the writer lists, and none a write."""
        from rkmon_lib import writer

        for types_key, fields_key in (("issueTypes", "fields"), ("values", "values")):
            with self.subTest(types_key=types_key):
                reads = Reads(
                    {META: {types_key: TYPES}, f"{META}/10002": {fields_key: FIELDS}}
                )
                self.assertEqual(tf.read(reads, PROJ, "Bug"), (TYPES, FIELDS))
                self.assertEqual(
                    reads.paths,
                    [
                        f"{META}{FIRST}",
                        f"{META}/10002{FIRST}",
                    ],
                )
                for path in reads.paths:
                    call = writer.atlassian_call(tf.rest_words(path, SITE))
                    self.assertEqual((call.writes, call.site), (False, SITE))

    def test_pages_and_a_read_that_gives_nothing(self):
        many = [{"fieldId": f"customfield_{n}", "name": str(n)} for n in range(60)]
        pages = iter(
            [{"fields": many[:50], "total": 60}, {"fields": many[50:], "total": 60}]
        )
        self.assertEqual(
            tf._listed(
                lambda path: (0, "warning: [slow] {retry}\n" + json.dumps(next(pages))),
                META,
            ),
            many,
        )
        reads = Reads({})
        self.assertEqual(tf.read(reads, PROJ, ""), (None, None))
        self.assertEqual(len(reads.paths), 1)  # no type: its fields are not read

    def test_a_short_page_while_jira_says_more(self):
        """A page shorter than asked for ends the list only while JIRA says nothing of more: its "isLast", else its
        "total" (a number), says whether more come; a page with no item ends it."""
        many = [{"fieldId": f"customfield_{n}", "name": str(n)} for n in range(30)]

        def listed(pages):
            pages, asked = iter(pages), []

            def ask(path):
                asked.append(path)
                return 0, json.dumps(next(pages))

            return tf._listed(ask, META), len(asked)

        total = [{"fields": many[:20], "total": 30}, {"fields": many[20:], "total": 30}]
        self.assertEqual(listed(total), (many, 2))
        last = [
            {"values": many[:20], "isLast": False},
            {"values": many[20:], "isLast": True},
        ]
        self.assertEqual(listed(last), (many, 2))
        self.assertEqual(listed([{"fields": many[:20]}]), (many[:20], 1))
        empty = [{"fields": many[:20], "total": 99}, {"fields": [], "total": 99}]
        self.assertEqual(listed(empty), (many[:20], 2))
        self.assertEqual(listed([{"fields": many[:20], "total": "30"}]), (many[:20], 1))


class TestChecked(unittest.TestCase):
    def test_off_without_the_login_and_with_it(self):
        def never(words):
            raise AssertionError("read without the login, or without a site")

        self.assertEqual(
            tf.checked(
                {"jira": {"project": PROJ}}, DEPLOY, never, lambda: True, "rkmon"
            )[:2],
            ([], []),
        )
        no_site = {**config("live"), "atlassian": {"site": ""}}
        self.assertEqual(
            tf.checked(no_site, DEPLOY, never, lambda: True, "rkmon")[:2], ([], [])
        )  # its key's own row (features: jira_url is blank)
        skip = tf.checked(config(), DEPLOY, never, lambda: False, "rkmon")[0]
        self.assertEqual(
            skip,
            [
                Check(
                    "ticket creation",
                    "SKIP",
                    "the atlassian login is not in yet: checked again at rkmon confirm and by rkmon doctor, once "
                    "rkmon login atlassian has run",
                )
            ],
        )

        def unasked():
            raise AssertionError("asked whether the deployment can have the login")

        self.assertEqual(
            tf.checked(config(), DEPLOY, never, lambda: False, "rkmon", unasked)[0],
            skip,
        )  # in prototype, a SKIP whatever the deployment can have
        live = tf.checked(config("live"), DEPLOY, never, lambda: False, "rkmon")[0]
        self.assertEqual(
            [(r.status, r.fix) for r in live],
            [("FAIL", checks.Command(("rkmon", "login", "atlassian")))],
        )
        # Before the deployment can have the login (rkmon check and rkmon confirm before deploy step 5): a SKIP.
        early = tf.checked(
            config("live"), DEPLOY, never, lambda: False, "rkmon", lambda: False
        )[0]
        self.assertEqual(early, skip)
        # The login cannot be asked yet: its caller's SKIP, which says why and when it is; nothing read.
        why = "the atlassian login is read through the image x:1, which is not on this host yet: checked again later"
        for mode in ("prototype", "live"):
            self.assertEqual(
                tf.checked(config(mode), DEPLOY, never, lambda: why, "rkmon", unasked)[
                    :2
                ],
                ([Check("ticket creation", "SKIP", why)], []),
            )
        reads = Reads(
            {META: {"issueTypes": TYPES}, f"{META}/10001": {"fields": FIELDS}}
        )
        got, offered, _rows_for = tf.checked(
            config(component=["Platform"], labels=["monitor"]),
            DEPLOY,
            reads.run,
            lambda: True,
            "x",
        )
        self.assertEqual(got, tf.rows(TYPES, FIELDS, {**LISTED}, DEPLOY))
        self.assertEqual(
            [f.id for f in offered], [f.id for f in tf.missing(FIELDS, LISTED)]
        )
        # each read names the site, by the name the address gives (atlassian.jira_site)
        self.assertEqual([w[:2] for w in reads.words], [["--site", SITE]] * 2)

    def test_the_rows_for_another_file_from_the_read_in_hand(self):
        """Checked.rows_for: check 6's rows for the configuration another file makes (the deploy settings file as a
        yes left it), from the lists the read gave, with no second read; with nothing read, the same rows.
        """
        reads = Reads(
            {META: {"issueTypes": TYPES}, f"{META}/10001": {"fields": FIELDS}}
        )
        listed = config(component=["Platform"], labels=["monitor"])
        found = tf.checked(listed, DEPLOY, reads.run, lambda: True, "x")
        self.assertEqual([r.status for r in found.rows], ["FAIL", "FAIL"])
        given = {"customfield_10010": "Major", "customfield_10020": "Core"}
        after = {**listed, "jira": {**listed["jira"], "extra_fields": given}}
        self.assertEqual(
            found.rows_for(after),
            tf.rows(TYPES, FIELDS, {**LISTED, "required_fields": given}, DEPLOY),
        )
        self.assertEqual([r.status for r in found.rows_for(after)], ["PASS"])
        self.assertEqual(found.rows_for(listed), found.rows)
        self.assertEqual(len(reads.words), 2)  # the two reads of the check, no more
        skip = tf.checked(config(), DEPLOY, Reads({}).run, lambda: False, "x")
        self.assertEqual(skip.rows_for(after), skip.rows)

    def test_each_setting_from_the_key_its_x_internal_names(self):
        """settings_of gives each setting rows reads the value of the configuration key under `jira` that the deploy
        settings schema's x-internal names for it."""
        schema = json.loads(
            (ROOT / "schema" / "deploy_settings.schema.json").read_text(
                encoding="utf-8"
            )
        )

        def find(node):
            if isinstance(node, dict):
                props = node.get("properties") or {}
                if "required_fields" in props and "issue_type" in props:
                    return props
                return next((p for p in map(find, node.values()) if p), None)
            if isinstance(node, list):
                return next((p for p in map(find, node) if p), None)
            return None

        props = find(schema)
        for key in (
            "issue_type",
            "required_fields",
            "labels",
            "components",
            "epic_key",
        ):
            internal = props[key]["x-internal"]
            self.assertTrue(internal.startswith("jira."), key)
            value = object()
            cfg = {"jira": {internal[len("jira.") :]: value}}
            self.assertIs(tf.settings_of(cfg)[key], value, key)
        # One home: the schema's x-internal as deploy_tree reads it, no copy of it here.
        from rkmon_lib import deploy_tree

        moved = tuple(
            (path, "jira.tags" if path.endswith(".labels") else internal)
            for path, internal in deploy_tree.feeds()
        )
        with mock.patch.object(deploy_tree, "feeds", return_value=moved):
            cfg = {"jira": {"tags": ["x"], "labels": ["y"]}}
            self.assertEqual(tf.settings_of(cfg)["labels"], ["x"])

    def test_rk_doctor_reads_through_the_fake_atlassian_cli_and_files_no_ticket(self):
        """rk doctor's row (doctor.ticket_rows) runs the atlassian CLI (here the fake one, logging its calls) through
        atlassian's _run: the two reads, and no other call."""
        import os
        import shutil

        from rkmon_lib import atlassian, doctor

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            shutil.copy(ROOT / "tests" / "fakes" / "fake_atlassian", base / "atlassian")
            (base / "atlassian").chmod(0o755)
            meta = base / "meta.json"
            meta.write_text(
                json.dumps(
                    {"types": {"issueTypes": TYPES}, "fields": {"fields": FIELDS}}
                )
            )
            env = {
                "PATH": f"{base}:{os.environ.get('PATH', '')}",
                "FAKE_ATLASSIAN_LOG": str(base / "log"),
                "FAKE_ATLASSIAN_CREATEMETA": str(meta),
            }
            with mock.patch.dict(os.environ, env), mock.patch.object(
                atlassian, "login_here", return_value=True
            ), mock.patch.object(atlassian, "env", return_value={}), mock.patch.object(
                atlassian, "status", return_value={"logged_in": True}
            ):
                got = doctor.ticket_rows(
                    config(component=["Platform"], labels=["monitor"])
                )
            calls = [json.loads(x) for x in (base / "log").read_text().splitlines()]
        self.assertEqual([r.name for r in got], ["ticket creation"] * len(got))
        self.assertEqual(
            calls,
            [
                tf.rest_words(f"{META}{FIRST}", SITE),
                tf.rest_words(f"{META}/10001{FIRST}", SITE),
            ],
        )

    def test_rk_doctor_s_fix_takes_the_change_from_the_host(self):
        """rk doctor (doctor.ticket_rows) runs in the container, where you type neither rk nor rkmon check: a fix that
        runs the check again names the container's commands from the host, `<wrapper> confirm` and `<wrapper> apply`.
        """
        import os

        from rkmon_lib import atlassian, doctor

        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": "1", "RKMON_WRAPPER": "rkmon-x"}
        ), mock.patch.object(
            atlassian, "status", return_value={"logged_in": True}
        ), mock.patch.object(
            atlassian, "_run", return_value=(1, "HTTP 404")
        ):
            got = doctor.ticket_rows(config())
        self.assertEqual(
            [str(r.fix) for r in got],
            ["once JIRA lists them, run rkmon-x confirm and rkmon-x apply"],
        )
        self.assertEqual(
            [c for r in got for c in r.fix.commands],
            ["rkmon-x confirm", "rkmon-x apply"],
        )


def wrapper() -> tuple:
    """docker/rkmon, loaded in this process: (the module, a Wrapper of it)."""
    import importlib.machinery
    import importlib.util

    path = str(ROOT / "docker" / "rkmon")
    loader = importlib.machinery.SourceFileLoader("rkmon_wrapper", path)
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("rkmon_wrapper", loader)
    )
    loader.exec_module(mod)
    return mod, mod.Wrapper(path)


class TestThroughTheWriter(unittest.TestCase):
    def test_the_host_asks_the_writer_and_makes_no_docker_object_before_its_volume(
        self,
    ):
        """rkmon's read (docker/rkmon ticket_rows: writer_holds, writer_read) asks the writer, as its own user, in the
        running container: its logins, then `rk atlassian rest` on the site the settings name, the writer's listed
        read. Before the first start, without the volume the writer's logins go to, it asks docker nothing (a one-off
        would make that volume).
        """
        from rkmon_lib import container_layout

        _mod, w = wrapper()
        asked = []

        def ask(*words, timeout=60, variables=None):
            asked.append(words)
            if "writer-secrets" in words:
                return 0, json.dumps({"atlassian": {"set": True}})
            fields = any(f"{META}/10001?" in word for word in words)
            return 0, json.dumps(
                {"fields": FIELDS} if fields else {"issueTypes": TYPES}
            )

        exec_as = (
            "compose",
            "exec",
            "-T",
            "--user",
            container_layout.WRITER.name,
            "rkmon",
            w.entry,
        )
        with mock.patch.object(w, "ask_docker", side_effect=ask), mock.patch.object(
            w, "running", return_value=True
        ), mock.patch.object(w, "asked_quietly", return_value=True):
            rows = w.ticket_rows(config(), DEPLOY)
        self.assertEqual([r.name for r in rows], ["ticket creation"] * len(rows))

        def read(path: str) -> tuple:
            return (*exec_as, "rk", "atlassian", *tf.rest_words(path, SITE))

        self.assertEqual(
            asked,
            [
                (*exec_as, "rk", "writer-secrets", "--json"),
                read(f"{META}{FIRST}"),
                read(f"{META}/10001{FIRST}"),
            ],
        )
        asked.clear()
        with mock.patch.object(w, "ask_docker", side_effect=ask), mock.patch.object(
            w, "running", return_value=False
        ), mock.patch.object(
            w, "secrets_volumes", return_value=None
        ), mock.patch.object(
            w, "asked_quietly", return_value=True
        ):
            self.assertIs(w.writer_login("atlassian"), False)
        self.assertEqual(asked, [])

    def test_no_fail_before_the_deployment_can_have_the_login(self):
        """rkmon check and rkmon confirm (docker/rkmon ticket_rows), ticket creation live and the login not in: a
        SKIP while this host has no volume for the writer's logins (writer_volume: none, or only the monitor's) or
        docker compose would refuse (asked_quietly: no settings confirmed yet), then a FAIL whose fix is `rkmon login
        atlassian`.
        """
        from rkmon_lib import container_layout as cl

        _mod, w = wrapper()
        fixes = {
            "SKIP": None,
            "FAIL": checks.Command((w.program, "login", "atlassian")),
        }
        for quiet, volumes, status in (
            (True, None, "SKIP"),
            (True, {cl.SECRETS_DIR: "example_secrets"}, "SKIP"),
            (False, {cl.WRITE_SECRETS_DIR: "example_secrets-write"}, "SKIP"),
            (True, {cl.WRITE_SECRETS_DIR: "example_secrets-write"}, "FAIL"),
        ):
            with self.subTest(quiet=quiet, volumes=volumes), mock.patch.object(
                w, "writer_login", return_value=False
            ), mock.patch.object(
                w, "asked_quietly", return_value=quiet
            ), mock.patch.object(
                w, "secrets_volumes", return_value=volumes
            ):
                rows = w.ticket_rows(config("live"), DEPLOY)
            self.assertEqual(
                [(r.status, r.fix) for r in rows], [(status, fixes[status])]
            )

    def test_the_writer_is_asked_only_where_that_says_nothing(self):
        """Check 6's question of the writer (docker/rkmon writer_login). Before the first confirm, or while docker
        compose would refuse (asked_quietly), docker is asked nothing and nothing is said: no login yet, check 6's SKIP.
        With the volume of the writer's logins and no running container: the image a one-off reads through missing,
        or --dry-run, which makes no one-off container, is a SKIP that says why the writer cannot be asked and when it
        is (deploy step 4; the run without --dry-run), in prototype and live alike, never a missing login.
        """
        import contextlib
        import io

        from rkmon_lib import container_layout as cl
        from rkmon_lib import records, steps

        _mod, w = wrapper()

        def never(*args, **kwargs):
            raise AssertionError(f"docker asked: {args}")

        said = io.StringIO()
        with mock.patch.object(w, "confirmed", return_value=None), mock.patch.object(
            w, "ask_docker", side_effect=never
        ), mock.patch.object(
            w, "running", side_effect=never
        ), contextlib.redirect_stdout(
            said
        ), contextlib.redirect_stderr(
            said
        ):
            self.assertIs(w.writer_login("atlassian"), False)
            rows = w.ticket_rows(config("live"), DEPLOY)
        self.assertEqual(said.getvalue(), "")
        self.assertEqual([r.status for r in rows], ["SKIP"])
        self.assertIn("the atlassian login is not in yet", rows[0].found)

        image = records.Image(registry="", name="example/rkmon", tag="2")
        confirmed = mock.Mock(image=image)
        step = steps.step_command(steps.DEPLOY_STEPS[3])
        for dry, labels, words in (
            (0, None, f"the image example/rkmon:2, which is not on this host yet"),
            (0, None, f"once {step} has put it here"),
            (1, {}, "--dry-run does not start: checked when you run the command"),
        ):
            for mode in ("prototype", "live"):
                with self.subTest(dry=dry, words=words, mode=mode), mock.patch.object(
                    w, "asked_quietly", return_value=True
                ), mock.patch.object(
                    w, "running", return_value=False
                ), mock.patch.object(
                    w,
                    "secrets_volumes",
                    return_value={cl.WRITE_SECRETS_DIR: "example_secrets-write"},
                ), mock.patch.object(
                    w, "image_labels", return_value=labels
                ), mock.patch.object(
                    w, "image_ref", return_value="example/rkmon:2"
                ), mock.patch.object(
                    w, "confirmed", return_value=confirmed
                ), mock.patch.object(
                    w, "first_tick_recorded", return_value=False
                ), mock.patch.object(
                    w, "writer_quiet", side_effect=never
                ):
                    w.dry = dry
                    rows = w.ticket_rows(config(mode), DEPLOY)
                self.assertEqual([r.status for r in rows], ["SKIP"])
                self.assertIn(words, rows[0].found)
        w.dry = 0
        with mock.patch.object(
            w, "asked_quietly", return_value=True
        ), mock.patch.object(w, "running", return_value=True), mock.patch.object(
            w, "writer_holds", return_value=True
        ) as holds:
            self.assertIs(w.writer_login("atlassian"), True)
        holds.assert_called_once_with("atlassian")

    def test_no_terminal_no_question_and_the_file_as_it_was(self):
        """rkmon check with no terminal, or on one with --dry-run (docker/rkmon offer_lines), check 6's read having
        found a field missing: no question, nothing read from the input, and the deploy settings file left as it was,
        with no FILE.bak (--dry-run changes nothing).
        """
        from rkmon_lib import file_types

        read = []

        class NoInput:
            def readline(self):
                read.append("readline")
                raise OSError("no terminal")

        for terminal, dry in ((False, 0), (True, 1)):
            _mod, w = wrapper()
            w.dry = dry
            field = tf.Field("customfield_10010", "Severity", ("High",))
            w.ticket = tf.Checked([], [field], lambda cfg: [])
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
                text = settings_file.template(file_types.DEPLOY)
                path.write_text(text, encoding="utf-8")
                path.chmod(0o600)
                with mock.patch.object(
                    w, "interactive", return_value=terminal
                ), mock.patch.object(sys, "stdin", NoInput()):
                    outcome = w.offer_lines(str(path), lambda f: None, [])
                self.assertEqual((outcome, read), (None, []))
                self.assertEqual(path.read_bytes(), text.encode())
                self.assertFalse(path.with_name(path.name + ".bak").exists())

    def test_a_write_that_fails_is_an_error_and_the_file_as_it_was(self):
        """A yes whose write fails (a full disk): ERROR, exit 1, which says the file is as it was, then the command
        again; the file as it was."""
        import contextlib
        import io

        from rkmon_lib import file_types, private_files

        class Typed:
            def __init__(self, lines):
                self.lines = iter(lines)

            def readline(self):
                return next(self.lines, "")

        mod, w = wrapper()
        field = tf.Field("customfield_10010", "Severity", ("High", "Low"))
        w.ticket = tf.Checked([], [field], lambda cfg: [])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
            text = settings_file.template(file_types.DEPLOY)
            path.write_text(text, encoding="utf-8")
            path.chmod(0o600)
            w.typed = ["check", "--settings", str(path)]
            full, out = OSError(28, "No space left on device"), io.StringIO()
            with mock.patch.object(
                w, "interactive", return_value=True
            ), mock.patch.object(
                sys, "stdin", Typed(["2\n", "y\n"])
            ), mock.patch.object(
                private_files, "write", side_effect=full
            ), contextlib.redirect_stdout(
                out
            ):
                with self.assertRaises(mod.Stopped) as stopped:
                    w.offer_lines(str(path), lambda f: None, [])
            self.assertEqual(stopped.exception.code, 1)
            words = " ".join(out.getvalue().split())
            self.assertIn(
                f"ERROR: the lines could not be added to {settings_file.shown(path)} (No space left on device); the "
                f"file is as it was next: {w.program} check --settings {path}",
                words,
            )
            self.assertEqual(path.read_bytes(), text.encode())

    def test_the_input_s_end_puts_the_outcome_on_a_line_of_its_own(self):
        """The input's end at check 6's question (docker/rkmon offer_lines), at its value or at its yes: nothing
        written, and the outcome starts on a line of its own, not on the prompt's."""
        import contextlib
        import io

        from rkmon_lib import file_types

        class Typed:
            def __init__(self, lines):
                self.lines = iter(lines)

            def readline(self):
                return next(self.lines, "")

        field = tf.Field("customfield_10010", "Severity", ("High", "Low"))
        fix = Edit(f"{tf.REQUIRED_FIELDS_KEY}.customfield_10010", DEPLOY, "High")
        row = Check("ticket creation", "FAIL", "required_fields does not set it", fix)
        for typed, prompt in ((), "which one? "), (("1\n",), "[y/N] "):
            mod, w = wrapper()
            w.ticket = tf.Checked([row], [field], lambda cfg: [])
            with self.subTest(prompt), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
                text = settings_file.template(file_types.DEPLOY)
                path.write_text(text, encoding="utf-8")
                path.chmod(0o600)
                w.typed = ["check", "--settings", str(path)]
                out = io.StringIO()
                with mock.patch.object(
                    w, "interactive", return_value=True
                ), mock.patch.object(
                    sys, "stdin", Typed(typed)
                ), contextlib.redirect_stdout(
                    out
                ):
                    rc = w.check_outcome(
                        [row],
                        False,
                        between=lambda: w.offer_lines(str(path), lambda f: None, [row]),
                    )
                self.assertEqual(rc, 4)
                self.assertIn(
                    f"{prompt}\nFAILED: 1 of 1 checks failed; nothing changed",
                    out.getvalue(),
                )
                self.assertEqual(path.read_bytes(), text.encode())

    def test_after_a_yes_the_outcome_of_the_file_as_written(self):
        """A yes at check 6's question (docker/rkmon check_outcome, offer_lines): the file as written is read again
        and check 6 checked on it from the read in hand, with no second read; the line after the answer says what was
        added and its effect, once; then the outcome of the file as it now is: PASSED, exit 0, or, with
        another check failing, FAILED for that one, exit 4, nothing else changed. A file that cannot be read again
        after the write: ERROR, exit 1, naming FILE.bak.
        """
        import contextlib
        import io

        from rkmon_lib import file_types

        class Typed:
            def __init__(self, lines):
                self.lines = iter(lines)

            def readline(self):
                return next(self.lines, "")

        fields = [*FILLED_BY_THE_MONITOR, REPORTER, SEVERITY]
        docker = Check("docker", "PASS", "29.0.0")
        taken = Check(
            "ports", "FAIL", "the port is taken", checks.Command(("ss", "-ltn"))
        )
        for others, rc, outcome in (
            ([docker], 0, "PASSED: 2 checks"),
            (
                [docker, taken],
                4,
                "FAILED: 1 of 3 checks failed; nothing else changed (fix each FAIL above)",
            ),
            ([docker], 1, "ERROR: the lines were added to"),
        ):
            reads = Reads(
                {META: {"issueTypes": TYPES}, f"{META}/10001": {"fields": fields}}
            )
            with self.subTest(outcome), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
                path.write_text(
                    settings_file.template(file_types.DEPLOY), encoding="utf-8"
                )
                path.chmod(0o600)
                mod, w = wrapper()
                w.typed = ["check", "--settings", str(path)]
                w.ticket = tf.checked(config(), str(path), reads.run, lambda: True, "x")
                self.assertEqual([r.status for r in w.ticket.rows], ["FAIL"])
                found, out = [*others, *w.ticket.rows], io.StringIO()
                gone = OSError(2, "No such file or directory") if rc == 1 else None
                with mock.patch.object(
                    w, "interactive", return_value=True
                ), mock.patch.object(
                    sys, "stdin", Typed(["1\n", "y\n"])
                ), mock.patch.object(
                    deployment, "read", side_effect=gone, wraps=deployment.read
                ), contextlib.redirect_stdout(
                    out
                ):
                    try:
                        got = w.check_outcome(
                            found,
                            False,
                            between=lambda: w.offer_lines(
                                str(path), lambda f: None, found
                            ),
                        )
                    except mod.Stopped as stopped:
                        got = stopped.code
                words = " ".join(out.getvalue().split())
                self.assertEqual(got, rc, words)
                self.assertIn('customfield_10010: "Major"', path.read_text())
                self.assertEqual(len(reads.words), 2)  # no second read from JIRA
                shown, kept = settings_file.shown(path), settings_file.shown(
                    f"{path}.bak"
                )
                added = (
                    "added 1 line under features.ticket_creation.required_fields, so the ticket creation check "
                    f"passes; the file as it was is {kept}"
                )
                if rc == 1:
                    self.assertIn(
                        f"{outcome} {shown}, but the file cannot be read again; the file as it was is {kept} "
                        f"next: {w.program} check --settings {path}",
                        words,
                    )
                    self.assertNotIn("added 1 line", words)
                    continue
                self.assertEqual(words.count("added 1 line"), 1, words)
                self.assertIn(f"{added} {outcome}", words)


class TestCheckSHelp(unittest.TestCase):
    def test_it_names_check_6_s_read_and_its_one_write(self):
        """`rkmon help check`: the command changes nothing but the ticket fields JIRA requires, on your
        yes; its details name check 6's read, which makes no ticket, the other commands that run it, its SKIP and FAIL
        without the atlassian login, its question and that one write, with FILE.bak and when nothing is written; its
        exit codes, the error of a write that failed."""
        from rkmon_lib import commands, program

        row = next(r for r in commands.TABLE if r.name == "check")
        text = " ".join(commands.help_text(row).split())
        for words in (
            "check: check this machine and your settings; change nothing but the ticket fields JIRA requires, added "
            "on your yes.",
            "the ticket creation check reads, with the atlassian login, which fields your JIRA project requires on a "
            "new ticket (a read: no ticket is made); "
            f"{program.name()} confirm and {program.name()} doctor run it too.",
            "Without the atlassian login it reads nothing: its row is a SKIP until the deployment can have the login "
            "(deploy step 5), then a FAIL while ticket creation is live.",
            "With the login, on a terminal, it then asks for a value for each required field your file lacks, and "
            "offers to add their lines under "
            f"features.ticket_creation.required_fields: the one write {program.name()} check makes, and only on your "
            "yes",
            "the file as it was is kept beside it as FILE.bak; without a terminal, on a no, or with --dry-run, "
            "nothing is written",
            "Exit: 0 every check passed; 1 an error, such as lines you said yes to that could not be written; 2",
        ):
            self.assertIn(words, text)


class TestBeforeTheFirstConfirm(unittest.TestCase):
    """rkmon check and the first rkmon confirm, run against the stand-in docker (tests/fakes/fake_docker) in a copy of
    the package (tests/deploy_folder.py), with ticket creation live."""

    def setUp(self):
        import shutil

        import deploy_folder
        import sandbox

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        pkg = deploy_folder.copy(d / "pkg")
        self.w = str(pkg / "docker" / "rkmon")
        bin_dir = pathlib.Path(sandbox.safe_path(d))
        shutil.copy(ROOT / "tests" / "fakes" / "fake_docker", bin_dir / "docker")
        (bin_dir / "docker").chmod(0o755)
        self.home = d / "home"
        self.home.mkdir()
        self.settings = self.home / "rkmon-deploy.yaml"
        tickets_on = {
            "mode": "live",
            "project_key": PROJ,
            "follow_up_tickets": {"enabled": True},
        }
        deploy_folder.write_settings(
            self.settings,
            sandbox.free_port(),
            features={"ticket_creation": tickets_on},
            integrations={"atlassian_cli": {"jira_url": "https://jira.example.com"}},
        )
        self.log, answers = d / "docker.log", d / "answers.json"
        host = [
            [r"^version", "29.0.0", 0],
            [r"compose version", "2.40.0", 0],
            [r"DockerRootDir", str(d), 0],
            [r"image inspect", "", 1],
        ]
        answers.write_text(json.dumps(host), encoding="utf-8")
        self.env = {
            "PATH": str(bin_dir),
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_DOCKER_ANSWERS": str(answers),
            # this host's memory and CPUs, as the tests set them, whatever the machine
            **deploy_folder.HOST_SIZE,
        }

    def rkmon(self, *args: str) -> tuple:
        import subprocess

        p = subprocess.run(
            [self.w, *args],
            cwd=self.home,
            env=self.env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        calls = (
            self.log.read_text(encoding="utf-8").splitlines()
            if self.log.exists()
            else []
        )
        self.log.unlink(missing_ok=True)
        return p.returncode, p.stdout + p.stderr, [c.split("\t", 1)[1] for c in calls]

    def test_check_6_asks_docker_nothing_and_says_no_refusal(self):
        """Before the first confirm there is no login to read and no deployment for docker compose to ask about: check
        6 is a SKIP (rkmon check shows it; the first confirm's summary shows only the rows that need you), docker
        compose is asked nothing for it, and the command ends with its one outcome, no refusal said before it.
        """
        skip = "SKIP  ticket creation: the atlassian login is not in yet"
        for args, done, row in (
            (("check", "--settings", str(self.settings)), ("PASSED", "FAILED"), skip),
            (
                ("confirm", "--settings", str(self.settings), "--yes"),
                ("CONFIRMED",),
                "",
            ),
        ):
            with self.subTest(command=args[0]):
                rc, out, calls = self.rkmon(*args)
                self.assertIn(row, out)
                self.assertNotIn("REFUSED", out)
                outcomes = [
                    line
                    for line in out.splitlines()
                    if line.startswith(("PASSED", "FAILED", "REFUSED", "CONFIRMED"))
                ]
                self.assertEqual(len(outcomes), 1, out)
                self.assertTrue(outcomes[0].startswith(done), out)
                compose = [c.split()[1] for c in calls if c.startswith("compose ")]
                asked = [word for word in compose if word != "version"]
                # rkmon check asks docker compose nothing; the confirm, once it is recorded, only whether the
                # deployment runs (the state's next line), and neither reads through the writer
                self.assertEqual(
                    asked, [] if args[0] == "check" else ["ps"] * len(asked)
                )


SETTINGS_TEXT = """\
features:
  ticket_creation:
    mode: "prototype"
    # Other fields your JIRA project requires on a new ticket, by field id.
    required_fields: {}
    follow_up_tickets:
      enabled: false
"""


class TestTheWrite(unittest.TestCase):
    def test_the_lines_in_place_of_braces_or_after_the_last_entry(self):
        entry = [("customfield_10010", "High")]
        self.assertEqual(
            tf.added(SETTINGS_TEXT, entry),
            SETTINGS_TEXT.replace(
                "required_fields: {}\n",
                'required_fields:\n      customfield_10010: "High"\n',
            ),
        )
        listed = SETTINGS_TEXT.replace(
            "required_fields: {}\n",
            'required_fields:\n      a_field: "x"\n    # its end\n',
        )
        self.assertEqual(
            tf.added(listed, entry),
            listed.replace(
                'a_field: "x"\n', 'a_field: "x"\n      customfield_10010: "High"\n'
            ),
        )
        self.assertIsNone(tf.added(SETTINGS_TEXT.replace("{}", '{a: "x"}'), entry))
        self.assertIsNone(tf.added("features: {}\n", entry))
        left_out = "".join(
            line for line in SETTINGS_TEXT.splitlines(True) if "required" not in line
        )
        self.assertEqual(
            tf.added(left_out, entry),
            left_out + '    required_fields:\n      customfield_10010: "High"\n',
        )

    PASSED = Check("ticket creation", "PASS", "required_fields sets it")

    def ask(self, typed, text=None, after=None, earlier=None, field=None):
        """check 6's question (tf.offer) for FIELD (else Severity, High or Low) on a settings file holding TEXT, the
        answers TYPED (None: the input's end), check 6 checked again on the file as written giving AFTER (else one PASS
        row), and a FILE.bak holding EARLIER there before (None: none): (check 6's rows for the file as written, None
        when nothing was written; the file's bytes; its bytes before; FILE.bak's bytes, None without it; FILE.bak's
        mode; what it said). Its prompts, in self.prompts.
        """
        from rkmon_lib import file_types

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
            bak = path.with_name(path.name + ".bak")
            text = text or settings_file.template(file_types.DEPLOY)
            path.write_bytes(text.encode())
            path.chmod(0o600)
            if earlier is not None:
                bak.write_bytes(earlier.encode())
                bak.chmod(0o600)
            answers, said, self.prompts = iter(typed), [], []
            field = field or tf.Field("customfield_10010", "Severity", ("High", "Low"))

            def recheck():
                self.assertNotEqual(path.read_bytes(), text.encode(), "not written")
                return after if after is not None else [self.PASSED]

            def answer(prompt: str):
                self.prompts.append(
                    prompt.replace(settings_file.shown(bak), "FILE.bak")
                )
                return next(answers, None)

            found = tf.offer(
                [field],
                path,
                lambda f: None,
                ROOT,
                answer,
                said.append,
                recheck,
            )
            kept = bak.read_bytes() if bak.exists() else None
            mode = oct(bak.stat().st_mode & 0o777) if bak.exists() else None
            return found, path.read_bytes(), text.encode(), kept, mode, said

    def test_a_yes_writes_only_those_lines_and_keeps_the_file_as_it_was(self):
        found, data, old, kept, mode, said = self.ask(["3", "2", "y"])
        self.assertEqual(found, [self.PASSED])
        self.assertEqual(kept, old)
        self.assertEqual(mode, "0o600")
        new = [
            "    required_fields:",
            '      customfield_10010: "Low"',
        ]  # the template leaves the key out
        lines = data.decode().splitlines()
        self.assertEqual(
            [line for line in lines if line not in new], old.decode().splitlines()
        )
        self.assertEqual(lines[lines.index(new[0]) + 1], new[1])
        self.assertEqual(said[0], "a number from 1 to 2")
        self.assertNotIn("earlier", self.prompts[-1])  # no FILE.bak was there
        self.assertRegex(
            " ".join(said[-1].split()),
            r"^added 1 line under features\.ticket_creation\.required_fields, so the ticket creation check passes; "
            r"the file as it was is \S+/rkmon-deploy\.yaml\.bak$",
        )
        self.assertLessEqual(max(map(len, said[-1].splitlines())), checks.WIDTH)

    def test_the_line_after_a_yes_says_what_check_6_still_fails_on(self):
        """Check 6 checked again on the file as written, failing on rows the lines do not fix (the components): the
        line after the answer says so, once."""
        other = Check(
            "ticket creation",
            "FAIL",
            "components the field does not allow",
            Edit("features.ticket_creation.components", DEPLOY, "[]"),
        )
        found, _data, _old, _kept, _mode, said = self.ask(["2", "y"], after=[other])
        self.assertEqual(found, [other])
        self.assertRegex(
            " ".join(said[-1].split()),
            r"^added 1 line under features\.ticket_creation\.required_fields; the ticket creation check still fails "
            r"on its other FAIL rows above; the file as it was is \S+/rkmon-deploy\.yaml\.bak$",
        )

    def test_the_value_itself_as_jira_writes_it(self):
        """The question takes one of JIRA's values typed as JIRA writes it, as it takes its number; anything else is
        asked again, for a number from 1 to N."""
        for typed, again in (
            (["Low", "y"], []),
            (["low", "Low", "y"], ["a number from 1 to 2"]),
        ):
            with self.subTest(typed=typed):
                found, data, _old, _kept, _mode, said = self.ask(typed)
                self.assertEqual(found, [self.PASSED])
                self.assertIn(b'      customfield_10010: "Low"\n', data)
                self.assertEqual(said[:-1], again)

    def test_jira_s_texts_in_the_question_show_what_does_not_print_escaped(self):
        """The question shows the field's name and the values JIRA allows as a row does, each character that does
        not print as its JSON escape; the line added gives the value itself, in that escape, which the file's reader
        reads back."""
        field = tf.Field("customfield_10010", "Severity\x1b[2J", ("High\x9b", "Low"))
        found, data, _old, _kept, _mode, _said = self.ask(["1", "y"], field=field)
        self.assertEqual(found, [self.PASSED])
        self.assertEqual(
            self.prompts[0], "Severity\\u001b[2J: 1 High\\u009b, 2 Low; which one? "
        )
        self.assertIn('\n    customfield_10010: "High\\u009b"\n', self.prompts[1])
        self.assertTrue(
            all(c.isprintable() or c == "\n" for c in "".join(self.prompts))
        )
        self.assertIn(b'      customfield_10010: "High\\u009b"\n', data)

    def test_an_earlier_file_bak_is_said_to_be_replaced(self):
        """A FILE.bak there already: the question says the file as it is now replaces it; a yes keeps the file as it
        was there, in place of the earlier one."""
        found, _data, old, kept, _mode, _said = self.ask(
            ["2", "y"], earlier="an earlier copy\n"
        )
        self.assertEqual((found, kept), ([self.PASSED], old))
        self.assertTrue(
            self.prompts[-1].endswith(
                '\n    customfield_10010: "Low"\n'
                "The earlier FILE.bak is replaced by the file as it is now.\n[y/N] "
            ),
            self.prompts[-1],
        )
        found, _data, old, kept, _mode, _said = self.ask(
            ["2", "n"], earlier="an earlier copy\n"
        )
        self.assertEqual((found, kept), (None, b"an earlier copy\n"))

    def test_a_no_or_no_terminal_writes_nothing(self):
        for typed in (["1", "n"], ["1"], []):
            with self.subTest(typed=typed):
                found, data, old, kept, _mode, _said = self.ask(typed)
                self.assertEqual((found, data, kept), (None, old, None))

    @staticmethod
    def kept_text(eol: str = "\n", end: bool = True) -> str:
        """The deploy template with required_fields: {} and a comment on its line, a comment with a tab and trailing
        spaces above it, and a blank line at the end; its line ends EOL, the last one left out without END.
        """
        from rkmon_lib import file_types

        text = settings_file.template(file_types.DEPLOY).replace(
            "    follow_up_tickets:\n",
            "    #\tby field id, as JIRA names them  \n"
            "    required_fields: {}  # none yet\n"
            "    follow_up_tickets:\n",
            1,
        )
        text = (text + "\n").replace("\n", eol)
        return text if end else text[: -len(eol)]

    def test_every_byte_as_it_was_but_the_lines_added(self):
        """A yes changes the file's bytes only by the lines it adds, with the file's line ends, the comment on the
        key's line kept; FILE.bak holds the bytes as they were (nothing else changes).
        """
        for eol in ("\n", "\r\n"):
            for end in (True, False):
                with self.subTest(eol=eol, end=end):
                    text = self.kept_text(eol, end)
                    found, data, old, kept, _mode, _said = self.ask(["2", "y"], text)
                    want = text.replace(
                        f"    required_fields: {{}}  # none yet{eol}",
                        f"    required_fields: # none yet{eol}"
                        f'      customfield_10010: "Low"{eol}',
                    )
                    self.assertEqual((found, kept), ([self.PASSED], old))
                    self.assertEqual(data, want.encode())

    def test_a_left_out_key_at_the_indent_of_the_section_s_keys(self):
        """required_fields left out of a section whose first line is a comment at another indent: the key comes at the
        indent of the section's keys, as the lines under a key that is there do."""
        entry = [("customfield_10010", "High")]
        text = 'features:\n  ticket_creation:\n      # by field id\n    mode: "prototype"\n'
        self.assertEqual(
            tf.added(text, entry),
            text + '    required_fields:\n      customfield_10010: "High"\n',
        )

    def test_braces_with_a_space_and_a_comment_after_a_tab(self):
        """required_fields: { } takes the lines in place of its braces, as {} does; a comment after a tab is a
        comment, as one after a space is, and stays on the key's line."""
        entry = [("customfield_10010", "High")]
        head = "features:\n  ticket_creation:\n"
        for given, kept in (
            ("required_fields: { }", "required_fields:"),
            ("required_fields: {}\t# none yet", "required_fields: # none yet"),
            ("required_fields:\t# none yet", "required_fields:\t# none yet"),
        ):
            with self.subTest(given=given):
                self.assertEqual(
                    tf.added(f"{head}    {given}\n", entry),
                    f'{head}    {kept}\n      customfield_10010: "High"\n',
                )
        text = self.kept_text().replace(
            "required_fields: {}  # none yet", "required_fields: { }\t# none yet"
        )
        found, data, old, kept, _mode, _said = self.ask(["2", "y"], text)
        want = text.replace(
            "    required_fields: { }\t# none yet\n",
            '    required_fields: # none yet\n      customfield_10010: "Low"\n',
        )
        self.assertEqual((found, kept), ([self.PASSED], old))
        self.assertEqual(data, want.encode())

    def test_the_last_line_without_its_end(self):
        """The key's line, or its last entry, as the file's last line with no line end: the lines added after it
        take the file's line end."""
        entry = [("customfield_10010", "High")]
        for eol in ("\n", "\r\n"):
            with self.subTest(eol=eol):
                head = f"features:{eol}  ticket_creation:{eol}"
                self.assertEqual(
                    tf.added(head + "    required_fields: {}", entry),
                    f'{head}    required_fields:{eol}      customfield_10010: "High"{eol}',
                )
                self.assertEqual(
                    tf.added(f'{head}    required_fields:{eol}      a: "x"', entry),
                    f'{head}    required_fields:{eol}      a: "x"{eol}'
                    f'      customfield_10010: "High"{eol}',
                )

    def test_the_file_as_it_was_when_its_write_fails(self):
        """FILE.bak is written first: a failure between the two writes leaves the file as it was, and FILE.bak with
        it."""
        from rkmon_lib import private_files

        real, calls = private_files.write, []

        def write(path, *args, **kwargs):
            calls.append(path)
            if len(calls) == 2:
                raise OSError(28, "No space left on device")
            return real(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
            old = self.kept_text().encode()
            path.write_bytes(old)
            path.chmod(0o600)
            with mock.patch.object(private_files, "write", write):
                with self.assertRaises(OSError):
                    tf.write_lines(
                        path, [("customfield_10010", "Low")], lambda f: None, ROOT
                    )
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(path.with_name(path.name + ".bak").read_bytes(), old)

    def test_a_new_text_that_reads_otherwise_is_refused(self):
        """The new text is read before it is written: one that reads otherwise than the file with the lines added,
        or that the reader refuses (whose refusal would name a line the file does not have), is refused, the lines to
        be added by hand, and nothing is written."""
        entry = [("customfield_10010", "Low")]
        old = self.kept_text()
        new = tf.added(old, entry)
        last = "    enabled: false\n\n# Integrations"  # the weekly report's switch
        follow = "    follow_up_tickets:\n"
        for why, otherwise in (
            ("reads otherwise", new.replace(last, last.replace("false", "true"))),
            ("a key twice", new.replace(follow, "    required_fields: {}\n" + follow)),
        ):
            self.assertNotEqual(otherwise, new)
            with self.subTest(why), tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "rkmon-deploy.yaml"
                path.write_bytes(old.encode())
                path.chmod(0o600)
                with mock.patch.object(tf, "added", return_value=otherwise):
                    with self.assertRaisesRegex(ValueError, "add them by hand"):
                        tf.write_lines(path, entry, lambda f: None, ROOT)
                self.assertEqual(path.read_bytes(), old.encode())
                self.assertFalse(path.with_name(path.name + ".bak").exists())

    def test_file_bak_a_link_or_a_folder(self):
        """FILE.bak a link or a folder: the write is refused with a fix of its own, which moves FILE.bak aside, then
        runs the check again; nothing is written, and the file the link points to is left as it was.
        """
        from rkmon_lib import private_files

        field = tf.Field("customfield_10010", "Severity", ("High", "Low"))
        for shape in ("link", "folder"):
            with self.subTest(shape), tempfile.TemporaryDirectory() as tmp:
                d = pathlib.Path(tmp)
                path, bak, other = (
                    d / "rkmon-deploy.yaml",
                    d / "rkmon-deploy.yaml.bak",
                    d / "elsewhere.txt",
                )
                old = self.kept_text().encode()
                path.write_bytes(old)
                path.chmod(0o600)
                other.write_text("kept\n", encoding="utf-8")
                if shape == "link":
                    bak.symlink_to(other)
                else:
                    bak.mkdir()
                argv = ("rkmon", "check", "--settings", str(path))
                retry = private_files.again(argv, str(path))
                typed, said = iter(["2", "y"]), []

                def recheck():
                    raise AssertionError("checked again with nothing written")

                found = tf.offer(
                    [field],
                    path,
                    retry,
                    ROOT,
                    lambda prompt: next(typed, None),
                    said.append,
                    recheck,
                )
                self.assertEqual((found, path.read_bytes()), (None, old))
                self.assertEqual(other.read_text(encoding="utf-8"), "kept\n")
                words = " ".join(said[-1].split())
                what = "a link" if shape == "link" else "not a regular file"
                kept, shown = settings_file.shown(bak), settings_file.shown(path)
                self.assertIn(
                    f"FAIL private file: {kept} is {what}: the file as it was is kept there",
                    words,
                )
                self.assertIn(
                    f"fix: mv {kept} {kept}.old, then rkmon check --settings {shown}",
                    words,
                )

    def test_a_quoted_key(self):
        """required_fields with its key in quotes takes the lines in place of its {}, as the key as it is does."""
        text = self.kept_text().replace(
            "    required_fields: {}", '    "required_fields": {}'
        )
        found, data, old, kept, _mode, _said = self.ask(["2", "y"], text)
        want = text.replace(
            '    "required_fields": {}  # none yet\n',
            '    "required_fields": # none yet\n      customfield_10010: "Low"\n',
        )
        self.assertEqual((found, kept), ([self.PASSED], old))
        self.assertEqual(data, want.encode())

    def test_a_field_listed_twice_is_asked_for_once(self):
        """A field JIRA lists twice (a page read twice) is asked for once, as its row says it once."""
        field = {
            "fieldId": "customfield_10010",
            "name": "Severity",
            "required": True,
            "allowedValues": [{"id": "1", "value": "High"}],
        }
        self.assertEqual(
            tf.offered([field, dict(field)], {}),
            [tf.Field("customfield_10010", "Severity", ("High",))],
        )


if __name__ == "__main__":
    unittest.main()
