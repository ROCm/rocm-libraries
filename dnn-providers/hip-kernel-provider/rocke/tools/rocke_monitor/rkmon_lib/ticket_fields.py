# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fields a new JIRA ticket needs (features.ticket_creation, check 6): the issue type it gets,
the fields it gets without a setting for each, and check 6's rows, which say whether the settings give a new ticket
every field the project requires, each in a form the project takes.

- DEFAULT_ISSUE_TYPE: the issue type a new ticket gets while features.ticket_creation.issue_type is blank.
- `issue_type(value) -> str`: the issue type a new ticket gets, from that setting's VALUE (jira.issue_type): VALUE
  without the spaces around it, else DEFAULT_ISSUE_TYPE. rkmon_lib/tickets.py files each ticket with it.
- FILLED: the fields a new ticket gets without a setting for each, by JIRA's field id: those the filer sends
  (rkmon_lib/tickets.py, `file_ticket`: the project, the summary, the issue type and the draft's priority, which
  every draft has: `check` refuses one without it), and the one JIRA always fills itself (the reporter).
- FILLED_WHILE_SET: the fields the filer sends only while a setting of features.ticket_creation lists something,
  field id -> that setting's key: the labels and the components.
- FILLED_WHILE_BLANK: the fields the filer sends with a new ticket only while a setting of features.ticket_creation is
  blank, field id -> that setting's key: the description, while epic_key is blank. With an epic, the filer gives the
  description after the link to it (tickets.link_epic), as one given before the link can come out garbled.
- `type_id(types, setting) -> str`: the id of the issue type SETTING names (`issue_type`) among the project's TYPES,
  the type whose FIELDS check 6 reads: "" when the project has no type of that name, or it is a sub-task type, whose
  tickets need a parent the monitor never gives. A name matches as JIRA writes it, case and all.
- `Field(id, name, allowed)`: a field of a new ticket: its id, its name as JIRA shows it, and the values JIRA allows
  for it (() when it takes any value).
- `needed(fields, settings) -> list[Field]`: the fields of FIELDS a new ticket needs required_fields for: each one
  JIRA requires and gives no value of its own that is not in FILLED, nor in FILLED_WHILE_SET while its setting lists
  something, nor in FILLED_WHILE_BLANK while its setting is blank; in FIELDS' order.
- `missing(fields, settings) -> list[Field]`: those of them required_fields does not set; a blank value sets nothing.
- `blank(value) -> bool`: whether a required_fields VALUE sets nothing: "", null, [] or {}. The filer sends no field
  for it (rkmon_lib/tickets.py, `file_ticket`).
- `rows(types, fields, settings, where, again) -> list[Check]`: check 6's rows, below. AGAIN: the steps a fix ends
  with to run the check again: its caller's (rk doctor's are its own), else rkmon check on WHERE
  (deployment.check_again).
- `read(ask, project, setting) -> (types, fields)`: check 6's read of the project's create metadata, JIRA's lists as
  they come (Server's "values", Cloud's "issueTypes" and "fields"), page by page, None where the read gave nothing;
  ASK(path) reads PATH with the login. No ticket is made.
- `checked(cfg, where, run, login, wrapper, login_possible, again) -> Checked`: check 6 with its read, for
  `rkmon check`, `rkmon confirm` and rk doctor, over the configuration (`settings_of`: the schema's x-internal): its
  rows, the fields its question asks for (`offered`), and its rows for another configuration from the lists in hand.
  RUN(words) runs the atlassian CLI's `rest` on the configured site (`rest_words`, atlassian.jira_site) with the
  login: through the writer, its one keeper (docker/rkmon writer_read; in the container, atlassian's `_run`).
- `offer(fields, path, retry, package_dir, answer, say, recheck) -> rows or None`: the question on a terminal, a value
  for each field, then whether to add the lines (`lines`); a yes writes them in place of required_fields' {} or after
  its last entry, or with the key where the file leaves it out (`added`), every other line and comment kept, and
  keeps the file as it was as FILE.bak
  (`write_lines`, `kept`); then check 6 is checked again on the file as written (RECHECK), and one line says what was
  added and whether check 6 now passes.

TYPES: the project's issue types as JIRA's create metadata lists them for the atlassian login
(`…/issue/createmeta/<project>/issuetypes`), each a mapping with its "id", its "name" and, for a sub-task type,
"subtask" true. FIELDS: the list of the fields of the type `type_id` gives (`…/issuetypes/<id>`), each a mapping with
its id ("fieldId", else "key"), its "name", whether JIRA "required" it, whether JIRA gives it a value when a new
ticket has none ("hasDefaultValue") and, for a field that takes one of a list, its "allowedValues", each with its
"value" or "name" and its "id" (one that is "disabled" is left out). SETTINGS: features.ticket_creation's values by
key: issue_type; required_fields, field id -> value (a value, a mapping with its "value", "name" or "id", or a list
of those); labels and components, lists; epic_key. WHERE: the deploy settings file, as a fix names it.

The rows, each named "ticket creation":
- No type a new ticket can be of. JIRA lists none for the atlassian login where it may not create tickets in the
  project: one FAIL, 'the atlassian login may not create tickets in this project: JIRA lists no issue type for it',
  whose fix gives the login JIRA's Create issues permission in the project, or sets
  features.ticket_creation.project_key to a project where it has that permission, then runs the check again
  (AGAIN). JIRA lists only sub-task types: one FAIL, whose fix sets
  features.ticket_creation.project_key.
- The project has no issue type SETTING names, or it is a sub-task type: one FAIL (the first names every type the
  project has), whose fix sets features.ticket_creation.issue_type to a type that is not a sub-task type. FIELDS is
  not read in these cases.
- FIELDS that is not a list (None, as for a read that gave nothing): one FAIL, 'JIRA gave no list of the fields of a
  new "<type>", so which ones the project requires is not known', whose fix runs the check again once JIRA lists them.
- Else, for each field of FIELDS, in their order:
  - one `missing` gives: a FAIL, 'the project requires <field id> ("<field name>") on a new "<type>", and
    required_fields does not set it', with a line break before "required_fields", where the
    words before it fit on the row's first line (else the row wraps the two as one sentence), whose fix adds under
    features.ticket_creation.required_fields the line `<field id>: "<one of: <value>, <value>>"`, the values JIRA
    allows, or `<field id>: "<value>"` for a field that takes any (checks.Added). For a field of
    FILLED_WHILE_SET, the row ends 'features.ticket_creation.<its setting> is empty' instead, and its fix sets that
    setting to a list of that value; for the description, 'features.ticket_creation.epic_key is set', with the
    reason, and its fix clears epic_key or has the project not require a description when a ticket is created.
  - one required_fields sets to a value JIRA does not allow: a FAIL, 'required_fields sets <field id> ("<field
    name>") to "<value>", which the project does not allow on a new "<type>"', whose fix sets
    features.ticket_creation.required_fields.<field id> to the values JIRA allows, as above.
- Then, for the labels and the components: a list set while a new ticket of the type has no such field, a FAIL whose
  fix leaves the list empty; components the field does not allow, a FAIL naming the ones it allows. Labels are free
  text in JIRA, so their values are not checked.
- Then, for each field required_fields sets that a new ticket of the type does not have, which JIRA refuses: a FAIL,
  'required_fields sets <field id>, a field the project does not have on a new "<type>"', whose fix deletes it.
- Without any of those, one PASS row, which names the fields `needed` gives, if any.
A text of JIRA's (a type's or a field's name, a field's id, a value it allows), as a row, the line a fix adds or the
question shows it, has each character that does not print, a control character among them, as its JSON escape
(\\u001b): JIRA's lists are not the operator's, and a terminal takes such a character as a command.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs check 6 there.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from collections.abc import Callable, Mapping
from typing import NamedTuple

from . import deployment, wrapping
from .checks import (
    DELETE,
    NAME_COLUMN,
    WIDTH,
    Added,
    Check,
    Command,
    Described,
    Edit,
    Status,
    Steps,
)

DEFAULT_ISSUE_TYPE = "Task"
# The fields a new ticket gets without a setting for each, by JIRA's field id: those the filer sends, then the one JIRA
# fills itself.
FILLED = (
    "project",
    "summary",
    "issuetype",
    # The draft's own: a ticket's priority differs from ticket to ticket, so no setting gives one, and the skill's
    # rules have the session give each draft one, by its impact (agentic/skills/rocke-monitor/rules.md); a draft
    # without one is refused (rkmon_lib/tickets.py, `check`).
    "priority",
    # JIRA fills it itself: a new ticket that names no reporter gets the login that files it.
    "reporter",
)
# The fields the filer sends only while a setting of the section lists something: field id -> the setting's key.
FILLED_WHILE_SET = {"labels": "labels", "components": "components"}
# The fields the filer sends with a new ticket only while a setting of the section is blank: field id -> the setting's
# key. With an epic, the description comes after the link to it, as one given before the link can come out garbled.
FILLED_WHILE_BLANK = {"description": "epic_key"}
NAME = "ticket creation"  # check 6's rows' name
SECTION = "features.ticket_creation"
ISSUE_TYPE_KEY = SECTION + ".issue_type"
REQUIRED_FIELDS_KEY = SECTION + ".required_fields"
PROJECT_KEY = SECTION + ".project_key"


class Field(NamedTuple):
    id: str
    name: str
    allowed: tuple = ()


def issue_type(value) -> str:
    """The issue type a new ticket gets: see the module's docstring."""
    return str(value or "").strip() or DEFAULT_ISSUE_TYPE


def _types(types) -> list:
    """The issue types of TYPES that have a name and an id."""
    return [
        t
        for t in types or ()
        if isinstance(t, Mapping) and t.get("name") and t.get("id") not in (None, "")
    ]


def _sub_task(t: Mapping) -> bool:
    return t.get("subtask") is True


def _named(types, setting):
    """The issue type of TYPES that SETTING names, None when there is none."""
    name = issue_type(setting)
    return next((t for t in _types(types) if t["name"] == name), None)


def type_id(types, setting) -> str:
    """The id of the issue type the setting names, when a new ticket can be of it: see the module's docstring."""
    t = _named(types, setting)
    return str(t["id"]) if t is not None and not _sub_task(t) else ""


def _section(settings) -> Mapping:
    return settings if isinstance(settings, Mapping) else {}


def _required_fields(settings) -> Mapping:
    given = _section(settings).get("required_fields")
    return given if isinstance(given, Mapping) else {}


def blank(value) -> bool:
    """Whether a required_fields VALUE sets nothing: see the module's docstring."""
    return value in (None, "", [], {})


def _lists(value) -> bool:
    """Whether a list setting (the labels, the components) lists something: a word that is not blank."""
    items = value if isinstance(value, (list, tuple)) else [value]
    return any(isinstance(x, str) and x.strip() for x in items)


def _id(field: Mapping) -> str:
    return str(field.get("fieldId") or field.get("key") or "")


def _allowed(field: Mapping) -> tuple:
    """The values JIRA allows for FIELD, as a row shows them: the "value" or the "name" of each that is not
    disabled."""
    values = []
    for v in field.get("allowedValues") or ():
        if isinstance(v, Mapping) and not v.get("disabled"):
            shown = v.get("value") or v.get("name")
            if shown not in (None, ""):
                values.append(str(shown))
    return tuple(values)


def _takes(field: Mapping) -> set:
    """Each word JIRA takes for FIELD: the value, the name and the id of each allowed value that is not disabled; none
    for a field that takes any value."""
    words = set()
    for v in field.get("allowedValues") or ():
        if isinstance(v, Mapping) and not v.get("disabled"):
            words |= {
                str(v[k]) for k in ("value", "name", "id") if v.get(k) not in (None, "")
            }
    return words


def _words(value) -> list:
    """What a required_fields VALUE says, as words: for a list, what each item says; for a mapping, its "value", its
    "name" or its "id"; else the value itself."""
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, Mapping):
            item = next(
                (
                    item[k]
                    for k in ("value", "name", "id")
                    if item.get(k) not in (None, "")
                ),
                "",
            )
        out.append(str(item))
    return out


def _refused(field: Mapping, value) -> list:
    """The words of VALUE that JIRA does not take for FIELD: none for a blank value, or a field that takes any
    value."""
    takes = _takes(field)
    if blank(value) or not takes:
        return []
    return [w for w in _words(value) if w not in takes]


def _field(field: Mapping) -> Field:
    fid = _id(field)
    return Field(fid, str(field.get("name") or fid), _allowed(field))


def _sent(fid: str, section: Mapping) -> bool:
    """Whether the filer sends the field FID with a new ticket under the settings SECTION (a blank setting as
    tickets.epic reads one)."""
    if fid in FILLED_WHILE_SET:
        return _lists(section.get(FILLED_WHILE_SET[fid]))
    if fid in FILLED_WHILE_BLANK:
        return not str(section.get(FILLED_WHILE_BLANK[fid]) or "").strip()
    return fid in FILLED


def needed(fields, settings) -> list:
    """The fields a new ticket needs required_fields for: see the module's docstring."""
    section = _section(settings)
    out = []
    for f in fields or ():
        if not isinstance(f, Mapping):
            continue
        fid = _id(f)
        if (
            fid
            and not _sent(fid, section)
            and f.get("required") is True
            and f.get("hasDefaultValue") is not True
        ):
            out.append(_field(f))
    return out


def missing(fields, settings) -> list:
    """The fields a new ticket needs that required_fields does not set: see the module's docstring."""
    given = _required_fields(settings)
    return [f for f in needed(fields, settings) if blank(given.get(f.id))]


def _printable(text) -> str:
    """TEXT with each character that does not print (str.isprintable: a control character, a line break, a format
    character) as its JSON escape (\\u001b, \\t), which the deploy settings file's reader takes in a quoted value.
    """
    return "".join(c if c.isprintable() else json.dumps(c)[1:-1] for c in str(text))


def _quoted(text: str) -> str:
    return _printable(json.dumps(text, ensure_ascii=False))


def _shown(field: Field) -> str:
    """A field as a row shows it: its id, and its name as JIRA shows it."""
    return f"{_printable(field.id)} ({_quoted(field.name)})"


def _placeholder(field: Field) -> str:
    """The value a fix shows for FIELD: the values JIRA allows, or any value."""
    return f"<one of: {', '.join(field.allowed)}>" if field.allowed else "<value>"


def _fail(found: str, fix) -> Check:
    return Check(NAME, Status.FAIL, found, fix)


def _again(where: str, again) -> tuple:
    """The steps that run check 6 again: AGAIN, its caller's, else rkmon check on WHERE (deployment.check_again)."""
    return deployment.check_again(where) if again is None else again


def _type_rows(types, setting, where: str, again: tuple) -> list:
    """The rows for the issue type: a FAIL when a new ticket cannot be of the type SETTING names, else none. AGAIN:
    the steps that run the check again."""
    known = _types(types)
    if not known:
        project = Described("a project where it has that permission")
        return [
            _fail(
                "the atlassian login may not create tickets in this project: JIRA lists no issue type for it",
                Steps(
                    (
                        "give the atlassian login JIRA's Create issues permission in the project, or",
                        Edit(PROJECT_KEY, where, project),
                        "; then run",
                        *again,
                    )
                ),
            )
        ]
    choices = [str(t["name"]) for t in known if not _sub_task(t)]
    has = wrapping.joined(_quoted(str(t["name"])) for t in known)
    if not choices:
        found = (
            f"JIRA lists only sub-task types in the project for the atlassian login ({has}): a new ticket of one "
            "needs a parent, which the monitor never gives"
        )
        project = Described("a project the atlassian login may create tickets in")
        return [_fail(found, Edit(PROJECT_KEY, where, project))]
    fix = Edit(
        ISSUE_TYPE_KEY,
        where,
        (
            choices[0]
            if len(choices) == 1
            else Described(
                "one of " + wrapping.joined((_quoted(c) for c in choices), "or")
            )
        ),
    )
    name = _quoted(issue_type(setting))
    which = (
        f"which {ISSUE_TYPE_KEY} names"
        if str(setting or "").strip()
        else f"which a blank {ISSUE_TYPE_KEY} means"
    )
    named = _named(known, setting)
    if named is None:
        return [
            _fail(f"the project has no issue type {name}, {which}; it has {has}", fix)
        ]
    if _sub_task(named):
        return [
            _fail(
                f"the issue type {name}, {which}, is a sub-task type: a new ticket of it needs a parent, which the "
                "monitor never gives",
                fix,
            )
        ]
    return []


def _found(first: str, rest: str) -> str:
    """A row's found text: FIRST, then REST on a line of its own where FIRST fits on the row's first line
    (checks.render); else the two as one sentence, which the row wraps with no short line between
    them."""
    fits = len(f"{NAME}: {first}") <= WIDTH - NAME_COLUMN
    return first + ("\n" if fits else " ") + rest


def _lacking(field: Field, name: str, where: str) -> Check:
    """The FAIL for a field the project requires on a new NAME that the settings do not give."""
    requires = f"the project requires {_shown(field)} on a new {name}, and"
    key = FILLED_WHILE_SET.get(field.id)
    if key:
        return _fail(
            _found(requires, f"{SECTION}.{key} is empty"),
            Edit(f"{SECTION}.{key}", where, [_placeholder(field)]),
        )
    key = FILLED_WHILE_BLANK.get(field.id)
    if key:
        return _fail(
            _found(
                requires,
                f"{SECTION}.{key} is set: the monitor then gives the description after the link to the epic, as one "
                "given before it can come out garbled",
            ),
            Steps(
                (
                    Edit(f"{SECTION}.{key}", where, ""),
                    ", or have the project not require a description when a ticket is created",
                )
            ),
        )
    return _fail(
        _found(requires, "required_fields does not set it"),
        Edit(
            REQUIRED_FIELDS_KEY,
            where,
            Added(tuple(lines([(field.id, _placeholder(field))]))),
        ),
    )


def _list_rows(fields, section: Mapping, name: str, where: str) -> list:
    """The rows for the labels and the components: a list set while a new NAME has no such field, and components the
    field does not allow (labels are free text in JIRA, so their values are not checked).
    """
    by_id = {_id(f): f for f in fields if isinstance(f, Mapping)}
    out = []
    for fid, key in FILLED_WHILE_SET.items():
        value, setting = section.get(key), f"{SECTION}.{key}"
        if not _lists(value):
            continue
        field = by_id.get(fid)
        if field is None:
            found = (
                f"{setting} is set, and the project has no {fid} field on a new {name}"
            )
            out.append(_fail(found, Edit(setting, where, [])))
            continue
        refused = _refused(field, value) if fid == "components" else []
        allowed = _allowed(field)
        if refused:
            names = wrapping.joined(_quoted(a) for a in allowed)
            out.append(
                _fail(
                    f"{setting} names {wrapping.joined(_quoted(w) for w in refused)}, which the project does not "
                    f"allow on a new {name}; it allows {names}",
                    Edit(setting, where, Described(f"a list of {names}")),
                )
            )
    return out


def rows(types, fields, settings, where: str, again: tuple | None = None) -> list:
    """Check 6's rows: see the module's docstring. AGAIN: the steps that run the check again (`_again`)."""
    again = _again(where, again)
    section = _section(settings)
    out = _type_rows(types, section.get("issue_type"), where, again)
    if out:
        return out
    name = _quoted(issue_type(section.get("issue_type")))
    if not isinstance(fields, (list, tuple)):
        return [
            _fail(
                f"JIRA gave no list of the fields of a new {name}, so which ones the project requires is not known",
                Steps(("once JIRA lists them, run", *again)),
            )
        ]
    given = _required_fields(section)
    lacking = {f.id: f for f in missing(fields, section)}
    seen = set()
    for raw in fields:
        fid = _id(raw) if isinstance(raw, Mapping) else ""
        if not fid or fid in seen:
            continue
        seen.add(fid)
        if fid in lacking:
            out.append(_lacking(lacking[fid], name, where))
            continue
        refused = _refused(raw, given.get(fid))
        if refused:
            f = _field(raw)
            out.append(
                _fail(
                    f"required_fields sets {_shown(f)} to {wrapping.joined(_quoted(w) for w in refused)}, which the "
                    f"project does not allow on a new {name}",
                    Edit(f"{REQUIRED_FIELDS_KEY}.{fid}", where, _placeholder(f)),
                )
            )
    out += _list_rows(fields, section, name, where)
    out += [
        _fail(
            f"required_fields sets {fid}, a field the project does not have on a new {name}",
            Edit(f"{REQUIRED_FIELDS_KEY}.{fid}", where, DELETE),
        )
        for fid in given
        if str(fid) not in seen
    ]
    if out:
        return out
    by_required_fields = needed(fields, section)
    if not by_required_fields:
        found = f"the project requires no field on a new {name} that the monitor does not fill itself"
    else:
        found = (
            f"the project requires {wrapping.joined(_shown(f) for f in by_required_fields)} on a new {name}, and "
            f"required_fields sets {'it' if len(by_required_fields) == 1 else 'them'}"
        )
    return [Check(NAME, Status.PASS, found)]


# --- the read, the question and the write (check 6) -------------------------------------------------------------------

TYPES_PATH = "/rest/api/3/issue/createmeta/{project}/issuetypes"
# The list a create-metadata answer gives, by the key it comes under: Server's "values"; Cloud's "issueTypes" for the
# issue types and "fields" for a type's fields.
LISTS = ("values", "issueTypes", "fields")
PAGE = 50  # the items one read asks for (the writer lists the reads with startAt and maxResults: rkmon_lib/writer.py)
PAGES = 20  # the reads of one list at most


def rest_words(path: str, site: str) -> list:
    """The atlassian CLI's words for the read of PATH on SITE (atlassian.jira_site): `rest`, one of the commands the
    writer lists."""
    words = ["rest", "--product", "jira", "--method", "GET", "--path", path]
    return ["--site", site, *words]


def _answer(text: str):
    """The JSON value TEXT gives from the first { or [ that starts one, each tried in turn, after any lines before it
    (a warning that holds a bracket among them); None when it gives none."""
    text = str(text or "")
    for i, c in enumerate(text):
        if c in "{[":
            try:
                return json.JSONDecoder().raw_decode(text, i)[0]
            except ValueError:
                continue
    return None


def _listed(ask, path: str):
    """The list the reads of PATH give, PAGE items a read asked for, PAGES reads at most; None when a read gives no
    list. A page with no item ends the list; else JIRA's word on whether more come, when it gives one: its "isLast",
    else its "total" (a number); without either, a page shorter than asked for ends it.
    """
    items = []
    for _ in range(PAGES):
        rc, text = ask(f"{path}?startAt={len(items)}&maxResults={PAGE}")
        answer = _answer(text) if rc == 0 else None
        if isinstance(answer, list):
            return items + answer
        got = next(
            (
                answer[k]
                for k in LISTS
                if isinstance(answer, Mapping) and isinstance(answer.get(k), list)
            ),
            None,
        )
        if got is None:
            return None
        items += got
        last, total = answer.get("isLast"), answer.get("total")
        if isinstance(last, bool):
            more = not last
        elif isinstance(total, int) and not isinstance(total, bool):
            more = len(items) < total
        else:
            more = len(got) >= PAGE
        if not got or not more:
            break
    return items


def read(ask, project: str, setting) -> tuple:
    """Check 6's read: (TYPES, FIELDS) for `rows`, JIRA's lists as they come, None where the read gave nothing.
    ASK(path) reads PATH (`rest_words`) with the atlassian login: (exit code, answer). TYPES first; FIELDS, the fields
    of the type SETTING names (`type_id`), only when a new ticket can be of it, else None.
    """
    path = TYPES_PATH.format(project=project)
    types = _listed(ask, path)
    tid = type_id(types, setting) if types is not None else ""
    return types, (_listed(ask, f"{path}/{tid}") if tid else None)


def settings_of(cfg: Mapping) -> dict:
    """SETTINGS for `rows` from the configuration CFG: each setting of features.ticket_creation that feeds a
    configuration key under `jira` (the deploy settings schema's x-internal: deploy_tree.feeds), by its key, with that
    key's value."""
    from . import deploy_tree

    j, head = cfg.get("jira") or {}, SECTION + "."
    return {
        path[len(head) :]: j.get(internal[len("jira.") :])
        for path, internal in deploy_tree.feeds()
        if path.startswith(head) and internal.startswith("jira.")
    }


def offered(fields, settings) -> list:
    """The fields of `missing` whose lines go under required_fields, each once (JIRA may list one twice, as `rows`
    says each once): not the labels, the components or the description, which a setting of their own gives.
    """
    out = {}
    for f in missing(fields, settings):
        if f.id not in {**FILLED_WHILE_SET, **FILLED_WHILE_BLANK}:
            out.setdefault(f.id, f)
    return list(out.values())


class Checked(NamedTuple):
    """Check 6 with its read (`checked`): its rows; the fields `offer` asks for; and ROWS_FOR(cfg), its rows for the
    configuration another file makes (the deploy settings file as a yes left it), from the lists the read gave.
    """

    rows: list
    offered: list
    rows_for: Callable


def _without_read(found: list) -> Checked:
    """Check 6's rows FOUND with nothing read: no field to ask for, and the same rows for any configuration."""
    return Checked(found, [], lambda cfg: found)


def checked(
    cfg: Mapping,
    where: str,
    run,
    login,
    wrapper: str,
    login_possible=lambda: True,
    again: tuple | None = None,
) -> Checked:
    """Check 6 with its read, over the configuration CFG: its rows, the fields `offer` asks for, and its rows for
    another configuration from the lists in hand, no second read (Checked). No row while ticket creation is off or
    names no project or site (their keys' own rows). LOGIN() says whether the atlassian login is in (True or False),
    or why that cannot be asked yet, the words of a SKIP that name when it is: that SKIP is then the one row, and
    nothing is read. Without the login, nothing is read, and one row: SKIP while ticket creation is in prototype,
    and while LOGIN_POSSIBLE() says the deployment cannot have the login yet (rkmon check and rkmon
    confirm before deploy step 5; rk doctor's deployment can); else FAIL, its fix `WRAPPER login atlassian`. With the
    login, `read`, each read RUN(words), the CLI's words on the site (`rest_words`, atlassian.jira_site): (exit code,
    answer); then `rows`, WHERE naming the file in a fix and AGAIN running the check again, as rows takes it; a read
    that gave no list of types is one FAIL, whose fix runs the check again once JIRA lists them.
    """
    from . import atlassian, features

    project = str((cfg.get("jira") or {}).get("project") or "").strip()
    site = atlassian.jira_site(cfg)
    if not features.enabled(cfg, "ticket_creation") or not project or not site:
        return _without_read([])
    held = login()
    if isinstance(held, str):  # cannot be asked yet: why, and when it is
        return _without_read([Check(NAME, Status.SKIP, held)])
    if not held:
        login_fix = atlassian.login_fix(wrapper=wrapper)
        if not (features.live(cfg, "ticket_creation") and login_possible()):
            confirm, doctor = Command((wrapper, "confirm")), Command(
                (wrapper, "doctor")
            )
            words = (
                f"the atlassian login is not in yet: checked again at {confirm} and by {doctor}, once {login_fix} "
                "has run"
            )
            return _without_read([Check(NAME, Status.SKIP, words)])
        found = "the atlassian login is not in, so JIRA's required fields cannot be read, and no ticket can be filed"
        return _without_read([_fail(found, login_fix)])
    settings = settings_of(cfg)
    types, fields = read(
        lambda path: run(rest_words(path, site)), project, settings["issue_type"]
    )
    again = _again(where, again)
    if types is None:
        found = "JIRA gave no list of the project's issue types, so whether a new ticket can be filed is not known"
        return _without_read(
            [_fail(found, Steps(("once JIRA lists them, run", *again)))]
        )

    def rows_for(other: Mapping) -> list:
        return rows(types, fields, settings_of(other), where, again)

    return Checked(rows_for(cfg), offered(fields, settings), rows_for)


def _key(fid: str) -> str:
    return fid if re.fullmatch(r"[A-Za-z0-9_]+", fid) else _quoted(fid)


def lines(entries) -> list:
    """The lines under required_fields for ENTRIES, (field id, value) pairs, without their indent."""
    return [f"{_key(fid)}: {_quoted(value)}" for fid, value in entries]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _rows(text: str) -> list:
    """TEXT's lines, each with its own line end ("\\n" or "\\r\\n"), split at "\\n" alone, as YAML splits them."""
    return re.findall(r"[^\n]*\n|[^\n]+\Z", text)


def _eol(text: str) -> str:
    """The line end TEXT's lines take: "\\r\\n" when a line of it ends so, else "\\n"."""
    return "\r\n" if "\r\n" in text else "\n"


def _end(rows: list, at: int, indent: int) -> int:
    """The index after the last line of the block of the key on line AT, at INDENT: its lines indented further, with
    the comments among them; a blank line or a comment after its last line goes with what follows.
    """
    end = at + 1
    for i in range(at + 1, len(rows)):
        if rows[i].strip():
            if _indent(rows[i]) <= indent:
                break
            end = i + 1
    return end


def _child(rows: list, start: int, stop: int):
    """The indent of the first line among ROWS[START:STOP] that is neither blank nor a comment, the indent the keys
    there take; None when there is no such line."""
    return next(
        (
            _indent(r)
            for r in rows[start:stop]
            if r.strip() and not r.lstrip().startswith("#")
        ),
        None,
    )


def _key_line(rows: list, key: str, start: int, stop: int):
    """(the index of KEY's line among ROWS[START:STOP], its indent): a key at the indent of the first key there, as it
    is or in quotes; None when there is no such key."""
    child, k = _child(rows, start, stop), re.escape(key)
    for i in range(start, stop):
        if _indent(rows[i]) == child and re.match(
            rf"""({k}|"{k}"|'{k}')\s*:(\s|$)""", rows[i].lstrip(" ")
        ):
            return i, child
    return None


def added(text: str, entries) -> str | None:
    """TEXT, a deploy settings file's, with the lines of ENTRIES (`lines`) under features.ticket_creation's
    required_fields: in place of its {} (or { }), or after its last entry, or with the key after the section's last
    line where the file leaves the key out, at the indent of the section's keys; every other line and comment as it
    was, each with its line end, and the lines added with the file's. A comment on the key's line, after a space or a
    tab, stays there. None when the file gives the key otherwise (a {...} that lists something), or has no such
    section: its lines are added by hand.
    """
    rows = _rows(text)
    at, indent, stop = -1, -1, len(rows)
    keys = REQUIRED_FIELDS_KEY.split(".")
    for key in keys:
        found = _key_line(rows, key, at + 1, stop)
        if (
            found is None and key == keys[-1] and at >= 0
        ):  # the key left out: it comes with its lines
            eol = _eol(text)
            child = _child(rows, at + 1, stop)
            pad = " " * (indent + 2 if child is None else child)
            new = [f"{pad}{key}:{eol}"] + [
                f"{pad}  {line}{eol}" for line in lines(entries)
            ]
            if stop == len(rows) and not rows[-1].endswith("\n"):
                rows[-1] += eol
            return "".join(rows[:stop] + new + rows[stop:])
        if found is None:
            return None
        at, indent = found
        stop = _end(rows, at, indent)
    body = rows[at].rstrip("\r\n")
    eol = rows[at][len(body) :] or _eol(text)
    head, _, rest = body.partition(":")
    mark = re.search(r"\s#", rest)  # a comment: a # after a space or a tab
    value = (rest[: mark.start()] if mark else rest).strip()
    braces = re.fullmatch(r"\{\s*\}", value) is not None
    if value and not braces:
        return None
    child = _child(rows, at + 1, stop)
    pad = " " * (child if child is not None else indent + 2)
    new = [pad + line + eol for line in lines(entries)]
    if braces:
        rows[at] = f"{head}: #{rest[mark.end() :]}{eol}" if mark else f"{head}:{eol}"
        stop = at + 1
    if stop == len(rows) and not rows[-1].endswith("\n"):
        rows[-1] += eol
    return "".join(rows[:stop] + new + rows[stop:])


def kept(path) -> pathlib.Path:
    """FILE.bak beside the deploy settings file PATH: where `write_lines` keeps the file as it was."""
    p = pathlib.Path(path)
    return p.with_name(p.name + ".bak")


def write_lines(path, entries, retry, package_dir) -> pathlib.Path:
    """The one write `rkmon check` makes: the lines of ENTRIES added to the deploy settings file PATH (`added`),
    and the file as it was kept beside it as FILE.bak, yours only, each through private_files (a new file at 600,
    synced, then renamed; RETRY and PACKAGE_DIR as private_files.write takes them). It sits beside
    settings_file.rewrite, rkmon migrate's write, which writes a whole new layout and keeps FILE.v<N>: this one
    changes those lines alone, every other byte of the file kept, its line ends too, and FILE.bak holds the bytes
    read. FILE.bak is written first: FILE changes only once it is kept. FILE.bak's path. ValueError, nothing written,
    when the file is no UTF-8 text, the lines cannot be placed, or the file would then read otherwise than with them
    added. private_files.Refused, nothing written, when FILE.bak is a link or no regular file: its fix moves it aside,
    then runs the check again (RETRY on PATH)."""
    from . import file_types, private_files, settings_file

    p, shown, deploy = pathlib.Path(path), settings_file.shown, file_types.DEPLOY
    by_hand = f"the lines cannot be added to {shown(p)} here: add them by hand"
    try:
        old = p.read_bytes().decode("utf-8")  # no line end translated
    except UnicodeDecodeError:
        raise ValueError(by_hand) from None
    new = added(old, entries)
    want = settings_file.loads(old, deploy).values
    section = want.setdefault("features", {}).setdefault("ticket_creation", {})
    section["required_fields"] = {
        **(section.get("required_fields") or {}),
        **dict(entries),
    }
    try:  # the reader's own refusal of the new text would name a line your file does not have
        got = None if new is None else settings_file.loads(new, deploy).values
    except settings_file.Refused:
        got = None
    if got != want:
        raise ValueError(by_hand)
    bak = kept(p)
    if os.path.islink(bak) or (os.path.lexists(bak) and not os.path.isfile(bak)):
        what = "a link" if os.path.islink(bak) else "not a regular file"
        found = f"{shown(bak)} is {what}: the file as it was is kept there, as a file of its own"
        move = Command(("mv", shown(bak), shown(private_files.aside(bak))))
        fix = Steps((move, ", then", retry(shown(p))))
        raise private_files.Refused([Check(private_files.ROW, Status.FAIL, found, fix)])
    private_files.write(
        bak, old, package_dir, replace=True, retry=retry, shown=shown(bak)
    )
    private_files.write(p, new, package_dir, replace=True, retry=retry, shown=shown(p))
    return bak


def _value(field: Field, answer, say):
    """FIELD's value as typed at ANSWER's prompt, asked again until it is one: the number of one of the values JIRA
    allows, or one of them as JIRA writes it; for a field that takes any, the text. None at the input's end. The
    prompt shows the field's name and the values as a row does (`_printable`).
    """
    choices = ", ".join(f"{n} {_printable(v)}" for n, v in enumerate(field.allowed, 1))
    name = _printable(field.name)
    while True:
        typed = answer(
            f"{name}: {choices}; which one? " if field.allowed else f"{name}: "
        )
        if typed is None:
            return None
        typed = typed.strip()
        if not field.allowed and typed:
            return typed
        if field.allowed and typed.isdigit() and 1 <= int(typed) <= len(field.allowed):
            return field.allowed[int(typed) - 1]
        if typed in field.allowed:
            return typed
        say(
            f"a number from 1 to {len(field.allowed)}"
            if field.allowed
            else "a value, please"
        )


def offer(fields, path, retry, package_dir, answer, say, recheck) -> list | None:
    """On a terminal, check 6's question: a value for each of FIELDS (`offered`), then whether to add their lines to the
    deploy settings file PATH, which says so when a FILE.bak there already is replaced; a yes writes them
    (`write_lines`), then RECHECK() gives check 6's rows for the file as written (Checked.rows_for: no second read),
    and one line says what was added and whether check 6 now passes. ANSWER(prompt): the line typed, None at the
    input's end, which writes nothing; SAY(text) prints. Check 6's rows for the file as written; None when nothing was
    written.
    """
    from . import private_files, settings_file

    entries = []
    for field in fields:
        value = _value(field, answer, say)
        if value is None:
            return None
        entries.append((field.id, value))
    earlier = kept(path)
    question = [
        f"Add these lines to {settings_file.shown(path)}?",
        *("    " + line for line in lines(entries)),
    ]
    if os.path.isfile(earlier) and not os.path.islink(earlier):
        question.append(
            f"The earlier {settings_file.shown(earlier)} is replaced by the file as it is now."
        )
    typed = answer("\n".join(question) + "\n[y/N] ")
    if (typed or "").strip().lower() not in ("y", "yes"):
        return None
    try:
        bak = write_lines(path, entries, retry, package_dir)
    except (ValueError, private_files.Refused) as exc:
        say(str(exc))
        return None
    found = recheck()
    if any(r.status == Status.FAIL for r in found):
        effect = f"; the {NAME} check still fails on its other FAIL rows above"
    else:
        effect = f", so the {NAME} check passes"
    counted = wrapping.counted(len(entries), "line")
    done = f"added {counted} under {REQUIRED_FIELDS_KEY}{effect}; the file as it was is {settings_file.shown(bak)}"
    say("\n".join(wrapping.wrap(done, WIDTH)))
    return found
