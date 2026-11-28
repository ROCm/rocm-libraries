# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Settings files: one module reads, checks, upgrades and writes every settings file.

A file type is one header kind and its schema; a document is one file's values.

- `FileType(kind, version, schema, upgrades=None, coherence=None, template_command="", section_command="",
  template_values=None)`: KIND, the header's `kind` ("deploy_settings"); VERSION, its layout, the integer
  `schema_version`; SCHEMA, the JSON Schema, as the path of its file or as a mapping, read at its first use and once
  only (`load()`), and refused (ValueError) unless its first two keys are `kind` and `schema_version`, fixed to KIND
  and VERSION; UPGRADES, for each earlier layout N, the step from N to N + 1, `step(values, file_type) -> (values,
  notes)`, pure and offline, FILE_TYPE being this file type, the one `upgrade` upgrades to, whose schema and
  commands a step may name; a note that says a key moved is a `Moved`; a step raises `StepRefused` for a value it
  may neither carry nor drop; COHERENCE, `coherence(doc) -> list[Check]`, the checks across keys. TEMPLATE_COMMAND
  is the command that writes a new file of the kind, and SECTION_COMMAND the one that prints a section of it in
  full, each as `rkmon help` shows it after the program's name ("template deploy [--profile PROFILE] FILE",
  "template section KEY"; "" for none). TEMPLATE_VALUES, `template_values() -> {dotted key: value}`, gives the
  template values that only the package knows, such as the deploy file's image.tag, the package's version:
  `load()` sets each as its key's `x-template`, and refuses (ValueError) a key that is no key of a section, or one
  that is a $ref. Its `noun` is the kind in words ("deploy settings"); `layout_in(package)`, the layout of the kind
  that the package in the folder PACKAGE reads: what its own copy of SCHEMA, at SCHEMA's place in this package, fixes,
  as `load()` reads a schema's header; None when it holds none of the kind there that says, or SCHEMA is a mapping.
- `Document(file_type, version, values, path=None, profile=None, data=None, moves=())`: VALUES are the file's keys,
  the header's two left out; PROFILE, the profile the file's title line names, or None; DATA, the bytes `read`
  read from PATH, which `upgrade` carries, so `rewrite` can tell whether the file changed since (None for a document
  made otherwise); MOVES, for a document `upgrade` made, each step's moves in turn, `{new key: earlier key}`, from
  the step's `Moved` notes (() for any other).
- `read(path, file_type, older=False) -> Document`: the file (YAML in the package reader's subset, or JSON:
  rkmon_lib/answers.py), refused (`Refused`, nothing read) with the fix when it is no UTF-8 text, does not parse, or
  has a key or a value that holds half a character (a \\u escape from \\ud800 to \\udfff without its other half: a row
  named by its key, never showing the value); when it has no header, or the other `kind`; when its layout is newer, or
  older unless OLDER (`rkmon migrate` reads an older file this way, for `upgrade`); when it has a `secret_values`
  section, or a `secrets` section in this layout, naming the section and how many entries it has; when a value
  looks like a secret (a token or key, a PEM block, a JWT, a password or user in an address, a credential in an
  address's query), or a key the schema marks `x-secret-ref` holds something other than a model's key's secret name
  (secrets_registry.BINDING_KEY_NAME: lowercase letters, digits and _, ending in _api_key), never echoing the value;
  a secret's name that holds a placeholder is left to `validate`'s placeholder row, though still looked at for a
  secret. In this layout, each plain value is typed by the schema: plain text where text is expected is text; a
  plain number, true, false or null is one where the schema takes one; under a key whose schema takes any value (`{}`,
  as a free-form map's entries), at any depth, it is what YAML reads. A plain value the schema cannot take as written,
  such as `7.10` where text is expected, stays as written (answers.Plain), for `validate` to report. Its PROFILE is
  NAME when its title line (its first line, or the one after the modeline) starts with the schema's title, or with
  that title with EARLIER_PRODUCT_NAME in place of PRODUCT_NAME, as a title line an earlier package wrote has it, and
  `(profile "NAME":`, and `profiles` gives a profile NAME; else None. A file that cannot be opened raises its OSError.
- `loads(text, file_type) -> Document`: TEXT, a file's text, read as `read` reads a file in this layout and refused
  alike, as a document with no path: messages name it "your NOUN" ("your deploy settings"), and a command in a fix
  names it FILE. Half a character that TEXT holds as it is, not as a \\u escape, is no UTF-8 text.
  tools/translate_production.py reads back this way the text it writes, so its check is the check of the file.
- `upgrade(doc) -> (Document, notes)`: an older document in this layout, through each step in turn; then typed and
  checked for secrets as `read` does, with the document's path and profile. A document in this layout comes back as
  it is, with no notes. A step's `StepRefused` is refused (`Refused`, reason "step") with its one row, the fix
  naming the document's file. The steps' `Moved` notes are the document's MOVES; every note comes back as plain
  text.
- `validate(doc) -> list[Check]`: the schema (types, values, forms, bounds, unknown keys with the closest known key,
  and missing keys: a required one, of a section the file has, or one Required when its feature is on, while that
  feature is on, unless it has a default or a blank; none in a section nothing reads), then the placeholders (`<...>`
  left in a value: FAIL, with where the value comes from; left in a map entry's name: FAIL, rename it; what holds one
  has that row alone: no row for a value's type, value, form or bounds, whatever its key's type, nor for a name the
  map's rule for names would refuse), each a FAIL row named by its key, or one PASS row, "settings", when there is
  none. The fix for a value out of its key's form quotes the key's description, which states the form, without its
  closing sentence (for a list's item or a map's entry, the list's or the map's). The keys missing while their
  feature is on share one row per section: the innermost section of the file around them, the row named by the
  section's own key (`alerts` for features.alerts), its fix `run PROGRAM SECTION_COMMAND, paste its text over SECTION
  in FILE, then set SWITCHES back to true and fill in each key whose comment says "Required when"`: the section
  command prints the section in full, and SWITCHES are the switches in it that the file has on and that text writes
  otherwise, named from the section (`develop_breaks.enabled`); with none, the fix leaves out `set SWITCHES back to
  true and` (check 13). A key has a row of its own when the file type has no section command, when no section of
  the file is around it, or when that section is a map's entry or a list's item. Then a WARN row for each value set
  for a feature that is off; then the file type's coherence checks, which run only once there is no FAIL (else a SKIP
  row says so).
- `misfits(doc) -> list[Check]`: validate's FAIL rows for the values the document holds that their keys do not take
  (a type, a value, a form, a bound, a list's length, a map entry's name, a key the schema does not know, and a
  section's discriminating key that no alternative of its oneOf takes, missing or not), and no other: none for a
  key the document leaves out, nor for a placeholder. `rkmon migrate` reports each one.
- `effective(doc) -> dict`: every key of the schema, nested: the file's value, or the key's default where the file
  leaves it out or blank ("" or null); None for a key with neither. Keys the schema does not know are left out.
- `keys_set(doc) -> list[str]`: the keys the file sets, dotted, in the schema's order: each key it gives a value that
  is not blank, a map one key. A section is no key, so an empty one sets none. `effective` gives every other key of
  the schema its default.
- `blank(value) -> bool`: whether a key's VALUE in a file leaves it blank, so that `effective` gives its default: null
  or "" (None stands for a key the file leaves out).
- `canonical(values)`: VALUES as canonical JSON holds them: the same mappings and lists, a whole float as an
  integer; NaN and infinity refused (ValueError), and a key that is no text or a value that is none of JSON's
  (TypeError).
- `fingerprint(values) -> str`: "sha256:" and the SHA-256 of VALUES as canonical JSON: `canonical(values)`,
  keys sorted, no spaces, UTF-8. Its caller chooses what goes in.
- `template(file_type, profile=None, every_key=False) -> str`: the file a template command writes, generated from the
  schema: the header, then every key with its comment and its template value. PROFILE names one of the
  schema's profiles (`x-profiles`; None: the first); a file type without profiles takes none. EVERY_KEY: the
  every-key text, with the profile's header and values: no key left out and no feature folded to its switch,
  with each map's example entries and each section a key does not choose commented out.
- `profiles(file_type) -> tuple`: the schema's profiles (`x-profiles`), in its order, the first being the one a
  template takes when none is named; () for a file type without profiles. Each is a `Profile(name, what, header,
  values)`: its name; what it is, the words after its name in the title line; its own header paragraphs; and its
  values, each dotted key with the value the profile's template sets for it.
- `sections(file_type) -> list`: `(KEY, title)` for each section a section command prints, in the file's order: each
  section, and each map with a title, that sections' keys reach (no map's entry, no list's item), the header's two
  keys left out. KEY is its name with its parents' names before it, joined by dots (`features.alerts`).
- `section(file_type, key) -> str`: what the section command prints for KEY, one `sections` gives: the section as the
  every-key text writes it, its comment first, indented as in the file; what the every-key text comments out (a
  section a key does not choose, and what is in it) is written as it is there before its "# ", not commented out,
  since you asked for it. Any other KEY is a ValueError that names the closest one, or else the command that lists
  them.
- `comments(file_type) -> list`: each comment the every-key text writes above a key, in the file's order, the header's
  two keys first: a `Comment(key, text, groups)`. KEY is the key's name with its parents' names before it, joined by
  dots, a map's example entry named as the every-key text names it (`bindings.<name>.provider`); TEXT, the comment's
  words before they are wrapped; GROUPS, True for a comment that heads other keys (a section's "Title: description",
  and the heading of each form a map's entries take), False for a key's own. The README's settings reference prints
  these, so it says what the templates say.
- `template_value(node, path)`: a key's template value, from NODE, its schema with a $ref followed (as `schema_keys`
  gives it): its `x-template`, else its `default`, else its `const`, as a copy; a ValueError naming PATH when it has
  none of them.
- `dump(doc) -> str`: the one writer: the template's layout and comments, with the document's values in place of the
  template's. Its header is the template's for the document's profile, or names none when PROFILE is None. A value
  it cannot write, under a key the schema does not know, as written (answers.Plain), or a number that is not finite,
  is `Unwritable`, naming its key.
- `profile_of(doc) -> (name, why)`: the profile whose title line fits DOC's values, or None, and why, as
  `rkmon migrate`'s report says it. The profiles are those `profiles` gives, each starting from the template's
  values with its own `values` laid over them, as `template` writes it, and they differ in the features that start
  on. The switches that tell them apart are the `enabled` keys of the outer features, those no other feature is
  around (a feature's own parts, as the merge queue's drift notices, go with it), that two profiles start
  differently. A profile matches when DOC has each of those switches as the profile starts it; where DOC has no
  value, or a blank one, the template's counts, as the writer writes it. The first profile that matches, in the
  schema's order, is named; with none, or when the file type has no profiles, None.
- `rewrite(path, doc, retry, dry_run=False) -> pathlib.Path`: `rkmon migrate`'s write, once migrate has checked the
  file at PATH as every reader of a private file does (`private_files.checked`): the file, as it is, kept as
  PATH.v<N> (N its layout), then `dump(doc)` as PATH, each through rkmon_lib/private_files.py (a new file, yours
  only, synced, then renamed into place), with RETRY as `private_files.write` takes it: the function from a path to
  the command that names that file, for a refusal's fix; their rows show each file as migrate's lines do (`shown`).
  The file is read again first: when its bytes are not DOC's DATA, the ones `read` read (it changed since, or DOC
  was not read from it), a ValueError, nothing written, so a change made while migrate runs is never lost. Its text
  and header are read as `read` reads them, so a file that `read` refuses for its text, its header, its kind or a
  newer layout is refused (`Refused`) with nothing written; a ValueError when its layout is not older than DOC's. A
  PATH.v<N> that holds the file already, as a migrate that stopped part-way leaves it, stays, made yours only; one
  that holds anything else, there before or made there while rewrite writes, is never written over:
  `private_files.Refused`, nothing written, its fix the mv that moves it aside (`private_files.aside`), then RETRY
  naming PATH. A refusal is raised with its rows, never printed: the caller shows it. The kept file's path. With
  DRY_RUN, every check runs, each write's through `private_files.write`'s own dry run, and the same refusals are
  raised, but nothing is written and no permissions change: a PATH.v<N> that holds anything but the file is refused,
  and the path it would have is returned.
- `Refused`: a refused read (a ValueError), with the rows that say why and the rule that refused it, for a file in
  another layout the two layouts, and for a file without a kind line its first key; whom its fixes speak to: see
  its docstring.
- `StepRefused(key, found, fix)`: an upgrade step's refusal (a ValueError): KEY names its row, FOUND says what the
  earlier file holds, and FIX(where) is the fix for the file shown as WHERE, which only `upgrade` knows.
- `Moved(old, new, text="")`: an upgrade step's note that a key moved: a string, TEXT or else "OLD is now NEW", that
  keeps OLD, the key in the earlier layout, and NEW, its key in the next.
- `Unwritable(path, why)`: a value the writer cannot write (a ValueError): PATH, its key path, and WHY.
- `earlier_key(doc, path) -> str`: the key of the file `upgrade` upgraded DOC from that PATH, a key path of DOC, is
  or lies under: through DOC's MOVES, the last step's first, the nearest move whose new key is PATH or a key around
  it gives its earlier key; PATH itself when no move reaches it. `rkmon migrate` names it for a value the writer
  cannot write.
- `shown(path)`: a path as messages show it: `~/…` under the home folder.
- `move_aside(path) -> Command`: the command that moves the file at PATH out of the way, beside it
  (private_files.aside), both as `shown` shows them.
- `aside_fix(path, mend=False, then=()) -> Steps`: the fix of the file at PATH that cannot be read as what it
  is: move it aside (`move_aside`), then THEN, more steps' parts; with MEND, mending it at the line and column
  its error names comes first.
- `secret_texts(values) -> list[SecretText(path, kind, what)]`: the one check for secrets in settings, which `read`
  and `upgrade` refuse and tools/translate_production.py stops on: each text in VALUES that looks like a secret,
  once for each PATH and KIND ("value" or "key"), WHAT being what `secret_shape(text)` says it looks like ("a token
  or key", "a PEM block", "a JWT", or an address with a user, a password or a credential in it). A hidden key stays
  out of the paths under it: `hidden_key(key)`, a key that looks like a secret or holds half a character, which no
  path a message shows holds.
- `schema_pattern(text) -> re.Pattern`: a schema's `pattern` TEXT, compiled for a search, means what it means to
  JSON Schema (ECMA-262's regular expressions); one that holds what Python's `re` cannot read so is refused
  (ValueError), naming the pattern and what in it. validate reads each pattern so.
- `layout_found(where, layout, current)` and `layout_fix(where, layout, current)`: what a refusal says of a
  file in another layout, and its fix: `rkmon migrate --settings FILE` for an older file, a newer package (`rkmon
  package update --from DIR`) for a newer one.
- `start_over(*file_types) -> Steps`: the one fix for a file no reader takes as it is, as it has no header or the
  header of the layout before layout 1: `start from a new file, PROG TEMPLATE NEW_FILE, and copy your values
  into it`, TEMPLATE the template command of the first of FILE_TYPES, each other's after it in parentheses, named by
  its kind's first word (`(a build file: PROG template build NEW_FILE)`); NEW_FILE stands for the new file. `read`
  gives it, of the file's own kind, for a file of a kind a template writes that has no kind line or no layout;
  rkmon_lib/migrate.py and docker/rkmon's build settings give it for the files they refuse so.

Fixes. Every row's fix is a command, an edit, or steps of them (checks.Command, Edit and Steps), each command after
the name rkmon was run by (rkmon_lib/program.py). An edit sets the key: to its value when one value fits, else to
where its value comes from (its `x-source`) or what it takes; or it deletes the key. A fix that holds no command and
no edit of its own (save the file again, correct a line, rename a key) ends with the command that reads the file
again: `PROG check --settings FILE`, or `PROG migrate --settings FILE` for a file migrate reads (OLDER); `PROG
doctor` for a file type no template writes, the monitor's own.

The schema and a file's values as validate reads them, for the checks across keys (rkmon_lib/coherence.py). ROOT is a
file type's schema (`FileType.load()`), and VALUES a file's values as read (`Document.values`). No other module reads
a name of this one that starts with "_", the tests aside: what one needs gets a public name, with its contract here.

- `schema_at(root, path)`: the schema of the key at PATH, a dotted path of keys (a map's entry by its name, or `*`
  for any entry), its $ref followed; None when no one schema is the key's: the schema has no such key, or the key
  lies in a section with alternatives (oneOf).
- `is_feature(node) -> bool`: whether NODE, a key's schema as `schema_at` gives it, is a feature's: a section with an
  `enabled` switch.
- `feature_on(root, values, feature) -> bool`: whether the feature at FEATURE, a dotted path, is on in VALUES: its
  switch true there, and each feature around it on.
- `readers(root, path) -> list`: the features that read the key at PATH: those around it (outermost first; a switch is
  not inside its own feature), then those its `x-used-by` names (its own, else the nearest around it).
- `in_use(root, values, path) -> (used, off)`: whether the key at PATH is in use in VALUES, as What the writer writes
  says, and OFF, the features that are off for it: the outermost feature around it that is off, else the features its
  `x-used-by` names (its own, else the nearest around it) when each of them is off; [] when the key is in use.
- `required_by(root, values, path, node) -> list`: the features that are on in VALUES and need the key at PATH, NODE
  its schema with an `x-required-when`: with each feature around the key on, those its `x-used-by` names that are on,
  else the nearest feature around it; [] when a feature around it is off. [] as well when no feature decides it, and
  a check across keys does: a key with no feature around it and no `x-used-by`, or with a default or a blank
  (`x-blank`), which a key left out means.
- `fits(doc, key, value) -> bool`: whether VALUE fits the schema of the key at KEY, as validate checks a value there
  in DOC: validate gives it no row (a value with a <...> fits: its row is the placeholder's). A KEY that `schema_at`
  finds no schema for is a ValueError.
- `where_of(doc) -> str`: DOC's file as messages and edits name it: `shown` of its path, or "your NOUN" (its file
  type's noun) for a document with no path (`loads`).
- `read_again(file_type, path, older=False) -> Command`: the command a fix runs once the file at PATH is put right,
  which reads it again (Fixes; OLDER: a file migrate reads); FILE stays as it is when PATH is None.
- `source_of(node) -> Described`: where the value of the key whose schema is NODE comes from, as an edit's value says
  it (Fixes): its `x-source`, else what the key takes.
- `shown_value(value) -> str`: VALUE as messages show it: as JSON writes it (text in double quotes), a value as
  written (answers.Plain) as it is, and a list or a mapping by what it is ("a list", "a mapping").

Closing sentences. Each key's description ends with exactly one, made from the key's schema;
tools/closing_sentences.py writes it there, and a test checks each file type's schema with `closing_problems`:

- `closing_sentence(node, required, path) -> str`: the key's one closing sentence: "Don't change it." for the
  header's two keys (PATH `kind` or `schema_version`; with `x-changed-by` after a colon), "Required." (REQUIRED: its
  section lists it in `required`), "Required when CONDITION." (`x-required-when`, with " Otherwise, leave blank
  BLANK." when it has `x-blank`), "Leave blank BLANK." (`x-blank`), or its default's (`default`, with
  `x-default-means`: default_sentence). A key with none of these is refused (ValueError), naming it.
- `default_sentence(default, path, means="") -> str`: the closing sentence of a key whose default is DEFAULT:
  "Default: VALUE.", VALUE as the writer writes a scalar, or a map of them as `{"NAME": VALUE, ...}`, its entries
  in their order (Unwritable, naming PATH, for one it cannot write), and ", which means MEANS" before the period
  when MEANS is given; "Default: none." for an empty list or map. The registry's package keys close with it too
  (registry.closing_problems).
- `closing_of(text, start=CLOSING_START, whole=CLOSING) -> (body, closing)`: a description split before its closing
  sentence: at the last place where "Required", "Leave blank", "Default:" or "Don't change it" starts the text or
  follows a space, and the rest of the text is one closing sentence of those forms; CLOSING is "" when there is none.
  START and WHOLE give other forms in their place: the registry's (registry.CLOSING_START and CLOSING).
- `schema_keys(schema)`: `(path, node, required)` for every key of the schema mapping SCHEMA: each property, at any
  depth (a map's entries are `NAME.*`, a list's items `NAME[]`), but a section and a oneOf of sections, which group
  other keys and have a title instead. NODE is the schema's own mapping (a $ref followed), so a tool may change its
  description.
- `map_keys(schema) -> frozenset`: the paths of SCHEMA's maps (a key whose value is a mapping of names, such as
  repository.shared_label_folders) outside any map's entry: the keys a reader takes whole, as registry.leaves does.
- `closing_problems(schema) -> list[str]`: "PATH: what is wrong" for each key that has no rule for its closing
  sentence or no description, or whose description does not end with that sentence, says nothing before it, or has
  a second one; [] when there is none.
- `description(node, path) -> str`: a key's description as every text shows it (a template's comment, a fix, a
  document), closing sentence and all, each field in it written as the writer writes a scalar: `{minimum}`,
  `{maximum}`, `{default}` and `{x-template}` are its key's own (the last its template value, which a file type's
  template values set when the schema loads), and `{package:NAME}` the package's default of the dotted NAME
  (config.defaults.json, read through rkmon_lib/registry.py); a ValueError naming PATH when it names a value its key
  does not have, or a NAME the package's defaults give no value (a section is none).

What the writer writes:

- The header: the schema's `$id` as a modeline when it is an https address (never a path); the title (`title`, and
  for a profile, `(profile "NAME": WHAT)`); the paragraphs of `x-header`, with a profile's own paragraphs after the
  first; a blank line; then `kind` and `schema_version`. Each top-level key after them follows a blank line.
- Every key has its comment above it, at most WIDTH columns, wrapped without breaking a word or a quoted phrase. A
  section's comment is "Title: description". Any other key's is its description as the schema has it, closing
  sentence and all (after "Title: " when it has a title). `{minimum}`, `{maximum}`, `{default}` and `{x-template}`
  in a description are the key's own, and `{package:NAME}` a package default (`description`).
- In a template, a key's value is its template value (`template_value`), or the one a profile's `values`
  (dotted key -> value) set for it. A key with no template value is refused (ValueError), naming it.
- A section its parent does not require, which requires keys of its own (the deploy file's `updates`), is written
  whole or left out: a template leaves it out, `dump` writes it when the document has it, and the every-key text
  writes it with its keys' template values.
- A feature is a section with an `enabled` switch; it is on when that switch is true and each feature around it is
  on. A key is unused while a feature around it is off, or while every feature its `x-used-by` names (its own, else
  the nearest around it) is off. A feature that is off is written as its switch alone; when the feature has keys
  besides its switch, which the switch alone hides, the switch's comment takes the schema's `x-turn-on` sentence
  before its closing. An unused section or map with `x-off-reason` is written as `{}` with "Title: not used while
  REASON." (and a map's closing sentence); any other unused key, section or map is left out. A feature that is off
  is written in full, every key, when a value in it differs from its template value, and an unused section, map or
  key is written when its value does, so `dump` never drops a value.
- A key with `x-chooses` chooses one of the sections beside it, the one its value names: each other one is left
  out, unless it holds values of its own. In the every-key text, each other one is written after its comment,
  commented out ("# " in front of each line), for you to use instead.
- In the every-key text, a map whose `propertyNames` has an `x-template` is followed by its example entries,
  commented out: one entry, named by that x-template, with its template values, for each form the map's entries take
  (each oneOf alternative after its description, as a heading that ends with a colon). A line written to be commented
  out fits WIDTH with its "# ". A map entry's name that is one placeholder (`<name>`) is written plain.
- Every string in double quotes (JSON's escapes, and \\u escapes for the characters YAML does not print); integers;
  floats with a point; true, false and null; a list of scalars in flow style when it fits on its line, else in block
  style, as is a list of mappings; a map in block style, `{}` when empty, with its entries' keys commented from the
  entries' schema; under a key that takes any value, every value it holds, at any depth. What it cannot write as its
  schema says, it refuses (ValueError) naming the key: a value the schema does not know, a plain value not yet typed,
  a float that is not finite, a list of lists.

The schema subset: `type`, `const`, `enum`, `pattern`, `minimum`, `maximum`, `minItems`, `items`, `properties`,
`additionalProperties` (a section without it takes no other key), `propertyNames`, `required`, `default`, `title`
(every section has one), `description`, local `$ref`, and `oneOf` whose alternatives each fix one discriminating key
with `const`. Its own keywords are KEYWORDS, each with what it says; the registry's are REGISTRY_KEYWORDS,
which this module does not read. A schema with any other `x-` keyword is refused when it loads (ValueError), naming
the keyword and where it is, and so is one with a pattern `schema_pattern` refuses, naming its key.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import copy
import difflib
import functools
import hashlib
import json
import math
import os
import pathlib
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

from . import answers, private_files, program, secrets_registry, steps, wrapping
from .answers import Plain
from .checks import (
    DELETE,
    FIX_LABEL,
    Check,
    Command,
    Described,
    Edit,
    Fix,
    Status,
    Steps,
)
from .paths import PACKAGE
from .privacy import TOKENS

WIDTH = 118  # a comment line at most
HEADER_KEYS = ("kind", "schema_version")
# The commands a fix names, as `rkmon help` shows them after the program's name: the one that checks a settings file
# (each path's step 2), the one that upgrades an older file, and the one that installs a newer package.
CHECK = steps.DEPLOY_STEPS[1].command
MIGRATE = "migrate --settings FILE"
PACKAGE_UPDATE = "package update --from DIR"
NEW_FILE = "NEW_FILE"  # the new file a fix that starts over names (start_over)
# The schemas' own keywords: each one this module reads, and what it says.
KEYWORDS = {
    "x-header": "at the top: the paragraphs of the file's header comment",
    "x-profiles": "at the top: the profiles a template takes, each with what it is, its paragraphs and its values",
    "x-turn-on": "at the top: the sentence before the closing one of an off feature's switch that hides other keys",
    "x-template": "a key's template value, in place of its default; in propertyNames, a map's example entry's name",
    "x-chooses": "on a key with an enum: each value names a section beside the key, the one in use",
    "x-blank": 'what a blank value does: "Leave blank BLANK."',
    "x-required-when": 'when the key is required: "Required when CONDITION."',
    "x-default-means": 'what the default means: "Default: VALUE, which means MEANS."',
    "x-changed-by": 'what changes a header key: "Don\'t change it: WHAT."',
    "x-source": 'where the key\'s value comes from, as a fix says it after "set KEY in FILE to"',
    "x-used-by": "the features that read the key",
    "x-off-reason": 'on a section or a map with x-used-by: written unused as {} with "Title: not used while REASON."',
    "x-secret-ref": "the key names the secret that keeps a model's key",
}
# The registry's keywords, which the registry reads through this name; this module does not.
REGISTRY_KEYWORDS = ("x-class", "x-applies", "x-internal")
SECRET_SECTIONS = ("secret_values", "secrets")  # refused, naming the section
# No secret in a settings file, as a refusal or a note says it; secret_steps says where the secrets go instead.
NO_SECRET = "no secret goes in a settings file"
# The keywords that restrict a value; a schema with none of them takes any value (`_anything`).
VALUE_KEYWORDS = (
    "type",
    "properties",
    "additionalProperties",
    "items",
    "oneOf",
    "const",
    "enum",
    "$ref",
)
# A template's placeholder, such as <org>/<repo>.
PLACEHOLDER = re.compile(r"<[^<>\n]+>")
# Where a closing sentence may start: its first words, at the start of the text or after a space.
CLOSING_START = re.compile(
    r"(?:^|(?<=\s))(?:Required\.|Required when |Leave blank |Default: |Don't change it[.:])"
)
# A closing sentence, whole: one of closing_sentence's forms.
CLOSING = re.compile(
    r"Required\.|(?:Required when|Leave blank|Default:) .+\.|Don't change it(?:: .+)?\.",
    re.S,
)
PEM = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+")
ADDRESS = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*)://([^\s/?#\"'<>]*)([^\s\"'<>]*)")
# A query's names that carry a credential, written as the query names them (case and - aside).
CREDENTIAL_NAMES = {
    "token",
    "access_token",
    "auth_token",
    "private_token",
    "api_key",
    "apikey",
    "key",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "sig",
    "signature",
    "auth",
    "credential",
    "credentials",
    "x_amz_signature",
    "x_amz_credential",
    "x_amz_security_token",
}
CREDENTIAL_ENDS = ("_token", "_key", "_secret", "_password")
# Characters JSON leaves as they are and YAML does not print, or reads as a line break: written as \u escapes.
UNPRINTED = re.compile("[\u007f-\u009f\u2028\u2029\ufeff\ufffe\uffff\ud800-\udfff]")
# Half a character (a lone surrogate): what a \u escape from \ud800 to \udfff gives without its other half.
LONE_SURROGATE = re.compile("[\ud800-\udfff]")
# What schema_pattern writes for Python's re. JSON Schema's \s, ECMA-262's white space and line ends, as the members
# of a set (Python's own \s is ASCII's spaces alone with re.ASCII, and without it takes U+001C to U+001F and U+0085,
# but not U+FEFF); and JSON Schema's . outside a set, any character but a line end.
_SPACES = r"\t-\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_NO_LINE_END = r"[^\n\r\u2028\u2029]"
# The escapes of a letter or a digit that JSON Schema and Python's re (with re.ASCII) read alike, \0 before no digit
# (\01 is an octal number to Python's re); and an escape of half a character, which JSON Schema reads together with
# its other half, as one character.
_ALIKE = re.compile(r"\\(?:[bBdDfnrtvwW]|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|0(?![0-9]))")
_HALF = re.compile(r"\\u[dD][89a-fA-F][0-9A-Fa-f]{2}")
# The start of a group, and a count, that both read alike; and the count that is Python's re's alone ({,m}).
_GROUP = re.compile(r"\((?:\?(?::|=|!|<=|<!))?")
_COUNT = re.compile(r"\{\d+(?:,\d*)?\}")
_PYTHONS_COUNT = re.compile(r"\{,\d*\}")
TYPE_WORDS = {
    "string": "text",
    "integer": "a whole number",
    "number": "a number",
    "boolean": "true or false",
    "array": "a list",
    "object": "a mapping",
    "null": "null",
}
_MISSING = object()


class Refused(ValueError):
    """A refused read, nothing read: `rows` are the FAIL rows that say why, each with its fix; `reason` says which
    rule refused it ("text", "header", "kind", "older", "newer" or "secret"; "step" for an upgrade step's refusal);
    `layouts`, for a file in another layout ("older", "newer"), is (its layout, the one this package reads), else ();
    `first_key`, for a file without a kind line ("header"), is its first key ("" when it has none), else None.

    The fixes speak to the owner of a settings file: they say what to change in it, or which command writes a new
    one. A reader of the monitor's internal state, which no human edits, shows none of them: it words its own fix
    from the reason, `rkmon doctor`."""

    def __init__(
        self,
        rows: list,
        reason: str,
        layouts: tuple = (),
        first_key: str | None = None,
    ):
        self.rows = list(rows)
        self.reason = reason
        self.layouts = tuple(layouts)
        self.first_key = first_key
        super().__init__(
            "; ".join(f"{r.found} ({FIX_LABEL}{r.fix})" for r in self.rows)
        )


class StepRefused(ValueError):
    """An upgrade step's refusal: the earlier file holds a value the step may neither carry nor drop. KEY names the
    row, FOUND says what was found, and FIX is a function from WHERE, the file as messages show it, to the row's fix
    (a checks.Command, Edit or Steps)."""

    def __init__(self, key: str, found: str, fix: Callable):
        self.key, self.found, self.fix = key, found, fix
        super().__init__(found)


class Moved(str):
    """An upgrade step's note that a key moved: a string, as every note is (TEXT, else "OLD is now NEW"), that keeps
    OLD, the key in the earlier layout, and NEW, its key in the next, for `upgrade` to record (`earlier_key`).
    """

    def __new__(cls, old: str, new: str, text: str = ""):
        note = super().__new__(cls, text or f"{old} is now {new}")
        note.old, note.new = old, new
        return note

    def __getnewargs__(self):
        return self.old, self.new, str(self)


class Unwritable(ValueError):
    """A value the writer cannot write: PATH, its key path, and WHY."""

    def __init__(self, path: str, why: str):
        self.path, self.why = path, why
        super().__init__(f"{path}: {why}")


@dataclass(frozen=True, eq=False)
class FileType:
    """One header kind: its layout, its schema, the steps from each earlier layout, and its checks across keys."""

    kind: str
    version: int
    schema: object
    upgrades: Mapping = None
    coherence: Callable | None = None
    template_command: str = ""
    section_command: str = ""
    template_values: Callable | None = None
    _loaded: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.kind
        ):
            raise ValueError(f"a kind is a snake_case name, not {self.kind!r}")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError(
                f"{self.kind}: a layout is a whole number from 1, not {self.version!r}"
            )
        if not isinstance(self.schema, (str, os.PathLike, Mapping)):
            raise ValueError(f"{self.kind}: the schema is a file's path or a mapping")
        steps = dict(self.upgrades or {})
        for n, step in steps.items():
            if (
                isinstance(n, bool)
                or not isinstance(n, int)
                or not 1 <= n < self.version
                or not callable(step)
            ):
                raise ValueError(
                    f"{self.kind}: an upgrade step is a function, from a layout before {self.version}"
                )
        object.__setattr__(self, "upgrades", steps)
        if self.coherence is not None and not callable(self.coherence):
            raise ValueError(f"{self.kind}: coherence is a function of the document")
        for command in (self.template_command, self.section_command):
            if not isinstance(command, str):
                raise ValueError(
                    f"{self.kind}: a command is its words, as {program.name()} help shows them after the program's "
                    "name"
                )
        if self.template_values is not None and not callable(self.template_values):
            raise ValueError(
                f"{self.kind}: template values come from a function, called when the schema loads"
            )

    @property
    def noun(self) -> str:
        """The kind in words: "deploy settings"."""
        return self.kind.replace("_", " ")

    def load(self) -> dict:
        """The schema, read at its first use and once only, checked against the kind and the layout, with the
        template values only the package knows."""
        if "schema" not in self._loaded:
            if isinstance(self.schema, Mapping):
                data = copy.deepcopy(dict(self.schema))
            else:
                data = json.loads(pathlib.Path(self.schema).read_text(encoding="utf-8"))
            _check_schema(data, self)
            values = self.template_values() if self.template_values else {}
            for dotted, value in values.items():
                _key_node(data, dotted, self.noun)["x-template"] = copy.deepcopy(value)
            self._loaded["schema"] = data
        return self._loaded["schema"]

    def layout_in(self, package) -> int | None:
        """The layout of the kind that the package in the folder PACKAGE reads: see the module's docstring."""
        try:  # a schema given as a mapping has no place: pathlib refuses it (TypeError)
            rel = pathlib.Path(self.schema).relative_to(PACKAGE)
            root = json.loads((pathlib.Path(package) / rel).read_text(encoding="utf-8"))
            kind, version = _schema_header(root)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        whole = isinstance(version, int) and not isinstance(version, bool)
        return version if kind == self.kind and whole else None


@dataclass(frozen=True)
class Document:
    """One settings file's values (the header's two keys left out), in a layout of its file type."""

    file_type: FileType
    version: int
    values: dict
    path: pathlib.Path | None = None
    profile: str | None = None
    data: bytes | None = field(default=None, repr=False, compare=False)
    moves: tuple = field(default=(), repr=False, compare=False)


class Profile(NamedTuple):
    """One of a schema's profiles (`x-profiles`): see `profiles` in the module's docstring."""

    name: str
    what: str
    header: tuple
    values: dict


class Comment(NamedTuple):
    """One comment of the every-key text: see `comments` in the module's docstring."""

    key: str
    text: str
    groups: bool


# --- paths and words ---------------------------------------------------------------------------------------------


def shown(path) -> str:
    """PATH as messages show it: `~/…` when it is under the home folder, else absolute."""
    p = os.path.abspath(os.path.expanduser(os.fspath(path)))
    home = os.path.expanduser("~")
    if home and home != "~":
        home = os.path.abspath(home)
        if p == home:
            return "~"
        if p.startswith(home.rstrip(os.sep) + os.sep):
            return "~/" + p[len(home.rstrip(os.sep)) + 1 :].replace(os.sep, "/")
    return p


def move_aside(path) -> Command:
    """The command that moves the file at PATH out of the way: see the module's docstring."""
    return Command(("mv", shown(path), shown(private_files.aside(path))))


def aside_fix(path, mend: bool = False, then: tuple = ()) -> Steps:
    """The fix of the file at PATH that cannot be read as what it is: see the module's docstring."""
    first = ("mend it at the line and column above, or",) if mend else ()
    return Steps((*first, "move it aside:", move_aside(path), *then))


def layout_found(where: str, layout: int, current: int) -> str:
    """What a refusal says of a file in another layout than this package's."""
    side = "older" if layout < current else "newer"
    return f"{where} is in layout {layout}, {side} than this package's layout {current}"


def layout_fix(where: str, layout: int, current: int) -> Fix:
    """The fix for a file in another layout: migrate an older one; a newer one needs a newer package."""
    if layout < current:
        return steps.command_line(MIGRATE, FILE=where)
    return Steps(
        (
            steps.command_line(PACKAGE_UPDATE),
            ", with DIR the package that wrote your file, or a newer one",
        )
    )


def _title(root: dict, file_type: FileType) -> str:
    """What the file is, as its title line says it: the schema's title, else the kind in words."""
    return str(root.get("title") or file_type.noun).strip()


def where_of(doc: Document) -> str:
    """DOC's file as messages and edits name it: see the module's docstring."""
    return _place(doc.path, doc.file_type)


def _place(path, file_type: FileType) -> str:
    """The file at PATH as messages name it: `shown(path)`, or "your NOUN" for a document or a text with no path."""
    return shown(path) if path is not None else f"your {file_type.noun}"


def _join(path: str, key) -> str:
    return f"{path}.{key}" if path else str(key)


# --- the schema ----------------------------------------------------------------------------------------------------


def _deref(root: dict, node):
    """NODE with its local $ref followed ("#/$defs/NAME")."""
    for _ in range(64):
        if not isinstance(node, Mapping) or "$ref" not in node:
            return node
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError(f"only a local $ref is followed, not {ref!r}")
        target = root
        try:
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            raise ValueError(f"$ref {ref} names nothing in the schema") from None
        node = target
    raise ValueError("a $ref loop in the schema")


def _schema_header(root) -> tuple:
    """(kind, layout): the values the header's two keys fix in the schema ROOT (_MISSING for one that fixes none)."""
    props = root["properties"]
    return tuple(_fixed(_deref(root, props[k])) for k in HEADER_KEYS)


def _fixed(node) -> object:
    """The one value NODE allows (`const`, or a one-value `enum`), or _MISSING."""
    if not isinstance(node, Mapping):
        return _MISSING
    if "const" in node:
        return node["const"]
    if isinstance(node.get("enum"), list) and len(node["enum"]) == 1:
        return node["enum"][0]
    return _MISSING


def _discriminator(root: dict, alternatives: list) -> str:
    """The key every alternative of a oneOf fixes to a value of its own, or ""."""
    names = None
    for alt in alternatives:
        props = alt.get("properties") or {}
        fixed = {k for k, v in props.items() if _fixed(_deref(root, v)) is not _MISSING}
        names = fixed if names is None else names & fixed
    for name in sorted(names or ()):
        return name
    return ""


def _node(root: dict, node, value=_MISSING):
    """NODE as it applies to VALUE: its $ref followed, and for a oneOf, the alternative VALUE's discriminating key
    chooses (the oneOf itself when none does)."""
    node = _deref(root, node)
    if isinstance(node, Mapping) and "oneOf" in node:
        alternatives = [_deref(root, a) for a in node["oneOf"]]
        key = _discriminator(root, alternatives)
        if key and isinstance(value, Mapping) and key in value:
            for alt in alternatives:
                if _fixed(_deref(root, alt["properties"][key])) == value[key]:
                    return alt
    return node


def _is_section(node) -> bool:
    return isinstance(node, Mapping) and isinstance(node.get("properties"), Mapping)


def _section_key(root: dict, path: str) -> bool:
    """Whether PATH is a section's KEY as a section command takes it: each part a property of the one before, so no
    map's entry and no list's item."""
    node = root
    for part in path.split("."):
        props = _deref(root, node).get("properties") or {}
        if part not in props:
            return False
        node = props[part]
    return True


def _is_map(node) -> bool:
    return (
        isinstance(node, Mapping)
        and not _is_section(node)
        and (
            isinstance(node.get("additionalProperties"), Mapping)
            or "object" in _types(node)
        )
    )


def is_feature(node) -> bool:
    """Whether NODE, a key's schema, is a feature's: see the module's docstring."""
    return _is_section(node) and "enabled" in node["properties"]


def _types(node) -> list:
    t = node.get("type") if isinstance(node, Mapping) else None
    return list(t) if isinstance(t, list) else [t] if t else []


def _anything(node) -> bool:
    """Whether NODE takes any value: a schema with no keyword that says what (`{}`, as a free-form map's entries). What
    a value of it holds, at any depth, takes anything too."""
    return isinstance(node, Mapping) and not any(k in node for k in VALUE_KEYWORDS)


def _child(root: dict, node, key):
    """The schema of KEY in a mapping that NODE describes, or None when NODE takes no such key."""
    node = _deref(root, node)
    if not isinstance(node, Mapping):
        return None
    props = node.get("properties")
    if isinstance(props, Mapping) and key in props:
        return props[key]
    extra = node.get("additionalProperties")
    return extra if isinstance(extra, Mapping) else None


def schema_at(root: dict, path: str):
    """The schema of the key at PATH: see the module's docstring."""
    node = root
    for part in path.split("."):
        node = _child(root, node, part)
        if node is None:
            return None
    return _deref(root, node)


def _key_node(root: dict, dotted: str, noun: str) -> dict:
    """The schema's own mapping of the key at DOTTED, a key of a section (each part a property of the one before),
    for a change to it alone; ValueError for any other path, or a key that is a $ref, which others may share.
    """
    node = root
    for part in dotted.split("."):
        node = _deref(root, node)
        props = node.get("properties") if isinstance(node, Mapping) else None
        if not isinstance(props, Mapping) or part not in props:
            raise ValueError(f"{dotted} is no key of {noun}")
        node = props[part]
    if not isinstance(node, Mapping) or "$ref" in node or _is_section(node):
        raise ValueError(
            f"{dotted} is no key of {noun} of its own: a section, or a $ref"
        )
    return node


def _schema_nodes(root: dict, node, path: str, depth: int = 0, required=None):
    """(path, node, required) for NODE and every node under it, $refs followed: properties, each with REQUIRED True
    when its section lists it in `required`, else False; map entries ("NAME.*"), list items ("NAME[]") and oneOf
    alternatives, with None."""
    node = _deref(root, node)
    if not isinstance(node, Mapping) or depth > 32:
        return
    yield path, node, required
    needs = node.get("required") or ()
    for k, child in (node.get("properties") or {}).items():
        yield from _schema_nodes(root, child, _join(path, k), depth + 1, k in needs)
    if isinstance(node.get("additionalProperties"), Mapping):
        yield from _schema_nodes(
            root, node["additionalProperties"], _join(path, "*"), depth + 1
        )
    if isinstance(node.get("items"), Mapping):
        yield from _schema_nodes(root, node["items"], path + "[]", depth + 1)
    for alt in node.get("oneOf") or ():
        yield from _schema_nodes(root, alt, path, depth + 1)


def _schema_parts(root: dict):
    """(where, part) for each node of ROOT (`_schema_nodes`) and each map's rule for its names (propertyNames), WHERE
    naming the part as a refusal does."""
    for path, node, _ in _schema_nodes(root, root, ""):
        where = path or "the top"
        yield where, node
        names = node.get("propertyNames")
        if isinstance(names, Mapping):
            yield f"{where} (propertyNames)", names


def _check_keywords(root: dict) -> None:
    """ValueError for an x- keyword that is neither KEYWORDS' nor REGISTRY_KEYWORDS', naming it and where it is: in
    a node, or in a map's rule for its names (propertyNames)."""
    known = (*KEYWORDS, *REGISTRY_KEYWORDS)
    for place, part in _schema_parts(root):
        for key in part:
            if isinstance(key, str) and key.startswith("x-") and key not in known:
                close = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
                raise ValueError(
                    f"{place}: {key} is no keyword of a schema"
                    + (f"; did you mean {close[0]}?" if close else "")
                )


def _check_patterns(root: dict) -> None:
    """ValueError for a pattern `schema_pattern` refuses, naming its key and what it says: in a node, or in a map's
    rule for its names (propertyNames)."""
    for place, part in _schema_parts(root):
        if "pattern" in part:
            try:
                schema_pattern(part["pattern"])
            except ValueError as exc:
                raise ValueError(f"{place}: {exc}") from None


def _check_choosers(root: dict) -> None:
    """ValueError unless each x-chooses is true, on a key of a section, with an enum whose every value names a
    section beside the key."""
    choosers = set()
    for path, node, _ in _schema_nodes(root, root, ""):
        props = node.get("properties") if _is_section(node) else {}
        for k, child in props.items():
            child = _deref(root, child)
            if not isinstance(child, Mapping) or "x-chooses" not in child:
                continue
            choosers.add(id(child))
            choices = child.get("enum")
            if child["x-chooses"] is not True or not (
                isinstance(choices, list)
                and choices
                and all(
                    isinstance(c, str)
                    and c in props
                    and _is_section(_deref(root, props[c]))
                    for c in choices
                )
            ):
                raise ValueError(
                    f"{_join(path, k)}: x-chooses is true, on a key with an enum whose every value names a section "
                    "beside the key"
                )
    for path, node, _ in _schema_nodes(root, root, ""):
        if "x-chooses" in node and id(node) not in choosers:
            raise ValueError(f"{path}: x-chooses is for a key of a section")


def _check_schema(root, file_type: FileType) -> None:
    """A schema this module can use for FILE_TYPE, or ValueError: its header's two keys first, fixed to the kind and
    the layout; no x- keyword but KEYWORDS and REGISTRY_KEYWORDS; every x-used-by naming a feature; x-off-reason only
    on a section or a map with x-used-by and a title, which the writer's "Title: not used while REASON." names;
    x-chooses as `_check_choosers` says; every pattern one `schema_pattern` reads.
    """
    props = root.get("properties") if isinstance(root, Mapping) else None
    if not isinstance(props, Mapping) or list(props)[:2] != list(HEADER_KEYS):
        raise ValueError(
            f"the {file_type.noun} schema starts with the keys kind and schema_version"
        )
    kind, version = _schema_header(root)
    if kind != file_type.kind or version != file_type.version:
        raise ValueError(
            f"the {file_type.noun} schema is for kind {kind!r}, layout {version!r}, not {file_type.kind!r}, "
            f"layout {file_type.version}"
        )
    _check_keywords(root)
    _check_choosers(root)
    _check_patterns(root)
    for path, node, _ in _schema_nodes(root, root, ""):
        for feature in node.get("x-used-by") or ():
            if not is_feature(schema_at(root, str(feature))):
                raise ValueError(
                    f"{path}: x-used-by names {feature}, which is no section with an enabled switch"
                )
        if not str(node.get("x-off-reason") or "").strip():
            continue  # no reason, or a blank one: the writer leaves the key out while it is unused
        if not (_is_section(node) or _is_map(node)):
            raise ValueError(f"{path}: x-off-reason is for a section or a map")
        if not node.get("x-used-by"):
            raise ValueError(
                f"{path}: a section or a map with x-off-reason names the features that read it, in x-used-by"
            )
        if not str(node.get("title") or "").strip():
            raise ValueError(
                f"{path}: a section or a map with x-off-reason has a title"
            )


def _readers(root: dict, path: str) -> tuple:
    """The features around the key at PATH (outermost first; a switch is not inside its own feature), and the
    features its x-used-by names (its own, else the nearest around it)."""
    parts = path.split(".")
    node, around, used_by = root, [], []
    for i, part in enumerate(parts):
        if i and is_feature(node) and not (i == len(parts) - 1 and part == "enabled"):
            around.append(".".join(parts[:i]))
        node = _child(root, node, part)
        if node is None:
            break
        node = _deref(root, node)
        if isinstance(node, Mapping) and "x-used-by" in node:
            used_by = [str(x) for x in node["x-used-by"]]
    return around, used_by


def readers(root: dict, path: str) -> list:
    """The features that read the key at PATH: see the module's docstring."""
    around, used_by = _readers(root, path)
    return list(dict.fromkeys(around + used_by))


def feature_on(root: dict, values, feature: str) -> bool:
    """Whether the feature at FEATURE is on in VALUES: see the module's docstring."""
    node, value = root, values
    for part in feature.split("."):
        node = _deref(root, _child(root, node, part))
        value = value.get(part) if isinstance(value, Mapping) else None
        if is_feature(node) and not (
            isinstance(value, Mapping) and value.get("enabled") is True
        ):
            return False
    return True


def _switches_on(root: dict, node, value, path: str) -> list:
    """The switches at PATH and under it that VALUE, the value there, has on (`enabled` true), as dotted paths."""
    node = _node(root, node, value)
    if not _is_section(node) or not isinstance(value, Mapping):
        return []
    found = (
        [_join(path, "enabled")]
        if is_feature(node) and value.get("enabled") is True
        else []
    )
    for k, v in value.items():
        if k != "enabled" and k in node["properties"]:
            found += _switches_on(root, node["properties"][k], v, _join(path, k))
    return found


def in_use(root: dict, values, path: str) -> tuple:
    """Whether the key at PATH is in use in VALUES, and the features off for it: see the module's docstring."""
    around, used_by = _readers(root, path)
    off = [f for f in around if not feature_on(root, values, f)]
    if off:
        return False, off[:1]
    if used_by and not any(feature_on(root, values, f) for f in used_by):
        return False, used_by
    return True, []


def required_by(root: dict, values, path: str, node) -> list:
    """The features that are on in VALUES and need the key at PATH: see the module's docstring."""
    if "default" in node or "x-blank" in node:
        return []
    around, used_by = _readers(root, path)
    if not around and not used_by:
        return []
    if any(not feature_on(root, values, f) for f in around):
        return []
    if used_by:
        return [f for f in used_by if feature_on(root, values, f)]
    return around[-1:]


# --- reading -------------------------------------------------------------------------------------------------------


def _whole(value):
    """A layout's number: an integer, or a plain number written as one; else None."""
    if isinstance(value, Plain):
        value = answers.typed(str(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def read_again(file_type: FileType, path, older: bool = False) -> Command:
    """The command that reads the file at PATH again, which ends a fix: see the module's docstring."""
    if not file_type.template_command:
        return Command((program.name(), "doctor"))
    words = {} if path is None else {"FILE": shown(path)}
    return steps.command_line(MIGRATE if older else CHECK, **words)


def _new_file(file_type: FileType) -> Command:
    """The command that writes a new file of FILE_TYPE: its template command."""
    return steps.command_line(file_type.template_command)


def start_over(*file_types: FileType) -> Steps:
    """The fix for a file no reader takes as it is, which starts from a new file: see the module's docstring."""

    def template(file_type: FileType) -> Command:
        return steps.command_line(file_type.template_command, FILE=NEW_FILE)

    first, *others = file_types
    words = ["start from a new file,", template(first)]
    for other in others:
        words += [f"(a {other.kind.split('_')[0]} file:", template(other), ")"]
    return Steps((*words, ", and copy your values into it"))


def _start_over(file_type: FileType, again: Command) -> Steps:
    """The fix for a file whose header cannot be read: a new file of its kind (`start_over`), for a kind a template
    writes; else the file the monitor wrote, put back."""
    if file_type.template_command:
        return start_over(file_type)
    return Steps(
        (f"put back the {file_type.noun} file the monitor wrote, then run", again)
    )


def _parse(data: bytes, where: str, name: str, again: Command) -> tuple:
    """(DATA as text, its values); Refused ("text") when DATA is no UTF-8 text or does not parse, and when a key or
    a value holds half a character (LONE_SURROGATE), each a row named by its key. Each fix ends with AGAIN.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        row = Check(
            name,
            Status.FAIL,
            f"{where} is not UTF-8 text",
            Steps((f"save {where} as UTF-8 text, then run", again)),
        )
        raise Refused([row], "text") from None
    try:
        values = answers.loads(
            text[1:] if text.startswith("\ufeff") else text, where, plain=True
        )
    except answers.AnswersError as exc:
        found = str(exc) if str(exc).startswith(where) else f"{where}, {exc}"
        fix = Steps((f"correct that line in {where}, then run", again))
        raise Refused([Check(name, Status.FAIL, found, fix)], "text") from None
    halves = {}
    for path, string, what in _strings(values, ""):
        if LONE_SURROGATE.search(string) and (path, what) not in halves:
            halves[path, what] = Check(
                path or name,
                Status.FAIL,
                f"{'the value' if what == 'value' else 'a key'} in {where} has a \\u escape for half a character "
                "(\\ud800 to \\udfff) without its other half",
                Steps(
                    (
                        f"correct that {what} in {where}: write the character itself, or both of its \\u escapes, "
                        "then run",
                        again,
                    )
                ),
            )
    if halves:
        raise Refused(list(halves.values()), "text")
    return text, values


def _header(
    values: dict,
    where: str,
    file_type: FileType,
    name: str,
    again: Command,
    older: bool = False,
    file: str | None = None,
) -> int:
    """The file's layout, once its header names FILE_TYPE's kind and FILE_TYPE's layout, or an older layout when OLDER;
    Refused otherwise. FILE is the file as a command names it (None: WHERE)."""
    kind = values.get("kind")
    if kind is None:
        found = f"{where} has no kind line, so it is not {file_type.noun}"
        fix = _start_over(file_type, again)
        first = str(next(iter(values), ""))
        raise Refused([Check(name, Status.FAIL, found, fix)], "header", first_key=first)
    if str(kind) != file_type.kind:
        found = f'{where} is {str(kind).replace("_", " ")} (kind "{kind}"), not {file_type.noun}'
        fix = (
            Steps(
                (
                    f"name your {file_type.noun} file instead, or write one:",
                    _new_file(file_type),
                )
            )
            if file_type.template_command
            else _start_over(file_type, again)
        )
        raise Refused([Check(name, Status.FAIL, found, fix)], "kind")
    layout = _whole(values.get("schema_version"))
    if layout is None or layout < 1:
        found = f"{where} has no schema_version line with a whole number, so its layout is unknown"
        fix = _start_over(file_type, again)
        raise Refused([Check(name, Status.FAIL, found, fix)], "header")
    if layout > file_type.version or (layout < file_type.version and not older):
        found = layout_found(where, layout, file_type.version)
        fix = layout_fix(where if file is None else file, layout, file_type.version)
        reason = "newer" if layout > file_type.version else "older"
        layouts = (layout, file_type.version)
        raise Refused([Check(name, Status.FAIL, found, fix)], reason, layouts)
    return layout


def secret_shape(text: str) -> str:
    """What TEXT looks like when it looks like a secret ("a token or key", ...), else ""."""
    if TOKENS.search(text):
        return "a token or key"
    if PEM.search(text):
        return "a PEM block"
    if JWT.search(text):
        return "a JWT"
    for m in ADDRESS.finditer(text):
        scheme, authority, rest = m.group(1).lower(), m.group(2), m.group(3)
        user = authority.rpartition("@")[0] if "@" in authority else ""
        if ":" in user or (user and scheme in ("http", "https")):
            return "an address with a user or password in it"
        query = urllib.parse.urlsplit("x://h" + rest)
        for part in (query.query, query.fragment):
            for qname, _ in urllib.parse.parse_qsl(part, keep_blank_values=True):
                q = qname.lower().replace("-", "_")
                if q in CREDENTIAL_NAMES or q.endswith(CREDENTIAL_ENDS):
                    return "an address with a credential in its query"
    return ""


def hidden_key(key) -> bool:
    """Whether a mapping's KEY stays out of every path a message shows: it looks like a secret (`secret_shape`), or
    holds half a character."""
    text = str(key)
    return bool(secret_shape(text) or LONE_SURROGATE.search(text))


def _strings(value, path: str):
    """(path, text, what) for every text in VALUE: "value" for a value or a list's item, at its key's path; "key"
    for a key, at its mapping's path. A hidden key (`hidden_key`) stays out of the paths under it.
    """
    if isinstance(value, str):
        yield path, value, "value"
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield path, str(k), "key"
            yield from _strings(v, path if hidden_key(k) else _join(path, k))
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v, path)


class SecretText(NamedTuple):
    """A text that looks like a secret: where it is, "value" or "key", and what it looks like (`secret_texts`)."""

    path: str
    kind: str
    what: str


def secret_texts(values) -> list:
    """Each text in VALUES that looks like a secret (`secret_shape`), once for each path and kind, in their order: a
    value or a list's item at its key's path, a key at its mapping's path ("" at the top), a hidden key (`hidden_key`)
    left out of the paths under it. Every text is looked at, placeholders and all."""
    found = {}
    for path, text, kind in _strings(values, ""):
        what = secret_shape(text)
        if what and (path, kind) not in found:
            found[path, kind] = SecretText(path, kind, what)
    return list(found.values())


def secret_steps() -> tuple:
    """Where a settings file's secrets go instead, as a fix's parts (checks.Steps) or a note ends: "after rkmon
    confirm, rkmon login lists each secret and installs it"."""
    return (
        "after",
        steps.command_line(steps.command_name(steps.DEPLOY_STEPS[2].command)),
        ",",
        steps.command_line(steps.command_name(steps.DEPLOY_STEPS[4].command)),
        "lists each secret and installs it",
    )


def _refuse_secrets(
    values: dict, root, where: str, current: bool, name: str, again: Command
) -> None:
    """Refused for a secrets section, a secret-shaped value (every value is looked at, placeholders and all), or a
    secret's name that is none, unless it holds a placeholder: validate's row reports that. AGAIN is the command
    that reads the file again."""
    rows = []
    for section in SECRET_SECTIONS:
        if section in values and (section == "secret_values" or current):
            held = values[section]
            n = (
                len(held)
                if isinstance(held, (Mapping, list))
                else int(held is not None)
            )
            # the command's name alone ("run migrate again"): the line after the fix runs it
            run = steps.command_name(" ".join(again.argv[1:]))
            fix = Steps(
                (
                    f"move each value into a file only you can read, delete {section} from {where}, and run {run} "
                    "again;",
                    *secret_steps(),
                )
            )
            rows.append(
                Check(
                    name,
                    Status.FAIL,
                    f"{where} has a {section} section ({wrapping.counted(n, 'entry', 'entries')}), and {NO_SECRET}",
                    fix,
                )
            )
    if rows:
        raise Refused(rows, "secret")
    flagged = [
        Check(
            s.path or name,
            Status.FAIL,
            f"{'the value' if s.kind == 'value' else 'a key'} in {where} looks like {s.what}, and {NO_SECRET}",
            Steps(
                (
                    f"delete that {s.kind} from {where}, keep the secret in a file only you can read, and run",
                    again,
                )
            ),
        )
        for s in secret_texts(
            {k: v for k, v in values.items() if k not in SECRET_SECTIONS}
        )
    ]
    if flagged:
        raise Refused(flagged, "secret")
    for path, node, value in _values_with_nodes(root, root, values, ""):
        if not node.get("x-secret-ref") or value in ("", None):
            continue
        if isinstance(value, str) and PLACEHOLDER.search(value):
            continue  # the placeholder's row says what goes there (validate)
        if not (
            isinstance(value, str)
            and secrets_registry.BINDING_KEY_NAME.fullmatch(value)
        ):
            source = Described(
                'the name of the secret that keeps the key, such as "my_model_api_key"'
            )
            fix = Steps(
                (
                    Edit(path, where, source),
                    ";",
                    Command((program.name(), "secret", "NAME")),
                    "installs the key under that name",
                )
            )
            found = (
                f"the value in {where} is not a model's key's secret name (lowercase letters, digits and _, ending "
                f"in {secrets_registry.BINDING_KEY_SUFFIX})"
            )
            rows.append(Check(path, Status.FAIL, found, fix))
    if rows:
        raise Refused(rows, "secret")


def _values_with_nodes(root: dict, node, value, path: str):
    """(path, node, value) for VALUE and each value inside it that the schema describes."""
    node = _node(root, node, value)
    if not isinstance(node, Mapping):
        return
    if path:
        yield path, node, value
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not path and k in HEADER_KEYS:
                continue
            child = _child(root, node, k)
            if child is not None:
                yield from _values_with_nodes(root, child, v, _join(path, k))
    elif isinstance(value, list) and isinstance(node.get("items"), Mapping):
        for i, v in enumerate(value):
            yield from _values_with_nodes(root, node["items"], v, f"{path}[{i}]")


def _plain(node, text: Plain):
    """A plain value typed by its schema: the value it reads as, where the schema takes it; else as written."""
    types = _types(node)
    value = answers.typed(str(text))
    if not types:
        return value
    if isinstance(value, bool):
        return value if "boolean" in types else text
    if value is None:
        return None if "null" in types else text
    if isinstance(value, int):
        return value if "integer" in types or "number" in types else text
    if isinstance(value, float):
        return value if "number" in types else text
    return str(text) if "string" in types else text


def _typed(root: dict, node, value):
    """VALUE with each plain value typed by its schema (a value the schema does not describe stays as read)."""
    if node is None:
        return value
    node = _node(root, node, value)
    anything = _anything(node)
    if isinstance(value, Mapping):
        return {
            k: _typed(root, node if anything else _child(root, node, k), v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        items = (
            node
            if anything
            else node.get("items") if isinstance(node, Mapping) else None
        )
        return [
            _typed(root, items if isinstance(items, Mapping) else None, v)
            for v in value
        ]
    if isinstance(value, Plain):
        return _plain(node, value)
    return value


# The product's name in the schemas' titles, which every title line the writer writes has; and the name in its place
# in a title line an earlier package wrote, which names its profile all the same.
PRODUCT_NAME, EARLIER_PRODUCT_NAME = "rocKE monitor", "rocKE PR monitor"


def _title_profile(text: str, file_type: FileType) -> str | None:
    """The profile the title line of TEXT names, when `profiles` gives it for FILE_TYPE: see the module's docstring."""
    lines = text.lstrip("\ufeff").splitlines()[:2]
    lines = [line for line in lines if not line.startswith("# yaml-language-server:")]
    m = re.match(r'# (.+?) \(profile "([^"]+)":', lines[0]) if lines else None
    if not m:
        return None
    title = _title(file_type.load(), file_type)
    titles = {title, title.replace(PRODUCT_NAME, EARLIER_PRODUCT_NAME, 1)}
    if m.group(1) in titles and m.group(2) in {p.name for p in profiles(file_type)}:
        return m.group(2)
    return None


def _document(data: bytes, file_type: FileType, path, older: bool) -> Document:
    """DATA, the bytes of a file of FILE_TYPE at PATH (None for a text with no path), read as `read` reads a file. The
    document keeps DATA as the bytes read from PATH, which `rewrite` compares with the file's; a text with no path
    keeps none."""
    where, name = _place(path, file_type), f"{file_type.noun} file"
    again = read_again(file_type, path, older)
    text, values = _parse(data, where, name, again)
    # a text with no path is FILE in a command, as in AGAIN
    file = None if path is not None else "FILE"
    layout = _header(values, where, file_type, name, again, older, file)
    body = {k: v for k, v in values.items() if k not in HEADER_KEYS}
    current = layout == file_type.version
    root = file_type.load() if current else None
    _refuse_secrets(body, root, where, current, name, again)
    values = _typed(root, root, body) if current else body
    return Document(
        file_type,
        layout,
        values,
        path,
        _title_profile(text, file_type),
        data if path is not None else None,
    )


def read(path, file_type: FileType, older: bool = False) -> Document:
    """The settings file at PATH, as a document of FILE_TYPE: see the module's docstring."""
    p = pathlib.Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return _document(p.read_bytes(), file_type, p, older)


def loads(text: str, file_type: FileType) -> Document:
    """TEXT, a file's text, as a document of FILE_TYPE with no path: see the module's docstring."""
    # half a character, which a str may hold, is no UTF-8: refused as the file's bytes would be
    return _document(text.encode("utf-8", "surrogatepass"), file_type, None, False)


def upgrade(doc: Document) -> tuple:
    """(the document in its file type's layout, the steps' notes): see the module's docstring."""
    file_type = doc.file_type
    if doc.version > file_type.version:
        raise ValueError(
            f"{file_type.noun}: layout {doc.version} is newer than this package's {file_type.version}"
        )
    if doc.version == file_type.version:
        return doc, []
    values, notes, moves = copy.deepcopy(doc.values), [], []
    for n in range(doc.version, file_type.version):
        step = file_type.upgrades.get(n)
        if step is None:
            raise ValueError(
                f"{file_type.noun}: no step from layout {n} to layout {n + 1}"
            )
        try:
            values, more = step(values, file_type)
        except StepRefused as exc:
            row = Check(exc.key, Status.FAIL, exc.found, exc.fix(where_of(doc)))
            raise Refused([row], "step") from None
        moves.append({x.new: x.old for x in more if isinstance(x, Moved)})
        notes += [str(x) for x in more]
    values = {k: v for k, v in values.items() if k not in HEADER_KEYS}
    root = file_type.load()
    again = read_again(file_type, doc.path, older=True)
    _refuse_secrets(values, root, where_of(doc), True, f"{file_type.noun} file", again)
    return (
        Document(
            file_type,
            file_type.version,
            _typed(root, root, values),
            doc.path,
            doc.profile,
            doc.data,
            tuple(moves),
        ),
        notes,
    )


def earlier_key(doc: Document, path: str) -> str:
    """The key of the file `upgrade` upgraded DOC from that PATH, a key path of DOC, is or lies under: see the module's
    docstring."""
    key = path
    for moves in reversed(doc.moves):
        around = [
            new for new in moves if key == new or key.startswith((f"{new}.", f"{new}["))
        ]
        if around:
            key = moves[max(around, key=len)]
    return key


# --- checking ------------------------------------------------------------------------------------------------------


def _is(value, t: str) -> bool:
    if t == "string":
        return isinstance(value, str) and not isinstance(value, Plain)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, Mapping)
    if t == "null":
        return value is None
    return True


def shown_value(value) -> str:
    """VALUE as messages show it: see the module's docstring."""
    if isinstance(value, list):
        return "a list"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, Plain):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _choices(node) -> str:
    return wrapping.joined([shown_value(v) for v in node.get("enum") or ()], "or")


def _type_words(node) -> str:
    types = _types(node)
    return (
        wrapping.joined([TYPE_WORDS.get(t, t) for t in types], "or")
        if types
        else "a value"
    )


def _what(node) -> str:
    """What a key takes, in words: its values, its range, or its type; for a section, its keys."""
    node = node if isinstance(node, Mapping) else {}
    if _is_section(node):
        return "its keys, as the template writes them"
    if node.get("enum"):
        return _choices(node)
    low, high = node.get("minimum"), node.get("maximum")
    if low is not None and high is not None:
        return f"a number from {low} to {high}"
    return _type_words(node)


# A description's fields (`description`): a value of its key's own, or a package default, {package:NAME}.
PACKAGE_FIELD = "package:"
FIELDS = re.compile(
    r"\{(minimum|maximum|default|x-template|package:[a-z0-9_]+(?:\.[a-z0-9_]+)*)\}"
)


def description(node, path: str) -> str:
    """A key's description as every text shows it: see the module's docstring."""

    def value(m):
        name = m.group(1)
        if name.startswith(PACKAGE_FIELD):
            # imported here: the registry reads the state schema's descriptions through this function
            from . import registry

            v = registry.get(registry.package_defaults(), name[len(PACKAGE_FIELD) :])
            if v is registry.MISSING or isinstance(v, Mapping):
                raise ValueError(
                    f"{path}: its description names {{{name}}}, which is no value of the package's defaults"
                )
            return _scalar(v, path)
        if name not in node:
            raise ValueError(
                f"{path}: its description names {{{name}}}, which it does not have"
            )
        return _scalar(node[name], path)

    return FIELDS.sub(value, str(node.get("description") or "").strip())


def _described(node, path: str) -> str:
    """A key's description without its closing sentence: what a fix quotes of it."""
    return closing_of(description(node, path))[0]


def source_of(node) -> Described:
    """Where the value of the key whose schema is NODE comes from: see the module's docstring."""
    node = node if isinstance(node, Mapping) else {}
    return Described(str(node.get("x-source") or "").strip() or _what(node))


@functools.lru_cache(maxsize=None)
def schema_pattern(text: str) -> re.Pattern:
    """A schema's pattern TEXT as JSON Schema reads it: see the module's docstring."""

    def refuse(what: str):
        raise ValueError(
            f"the pattern {text} has {what}, which Python's re cannot read as JSON Schema does"
        )

    def escape(i: int) -> str:
        """The escape at I: one both read alike, or an escaped sign, which is the sign to both. Python's re refuses
        a hex escape without its digits itself."""
        if _HALF.match(text, i):
            refuse(f"half a character, {text[i : i + 6]}")
        alike, sign = _ALIKE.match(text, i), text[i + 1 : i + 2]
        if not alike and sign.isascii() and sign.isalnum() and sign not in "ux":
            refuse(f"the escape {text[i : i + 2]}")
        return alike.group() if alike else text[i : i + 2]

    def member(i: int) -> tuple:
        """The member of a set at I, as Python's re reads it alike, and where the next one starts."""
        if text.startswith("\\S", i):
            refuse("\\S in a set")
        if text.startswith("\\s", i):
            return _SPACES, i + 2
        part = escape(i) if text[i] == "\\" else text[i]
        return part, i + len(part)

    out, i = [], 0
    while i < len(text):
        start, c = i, text[i]
        if text.startswith(("\\s", "\\S"), i):
            part = ("[" if text[i + 1] == "s" else "[^") + _SPACES + "]"
            i += 2
        elif c == "\\":
            part = escape(i)
            i += len(part)
        elif c == "[":
            # A set ends at its first ] that is not escaped: JSON Schema's [] holds nothing, and its [^] everything.
            i += 2 if text.startswith("[^", i) else 1
            if text.startswith("]", i):
                refuse("a set that starts with ]")
            parts = [text[start:i]]
            while i < len(text) and text[i] != "]":
                first, i = member(i)
                if text.startswith("-", i) and text[i + 1 : i + 2] not in ("", "]"):
                    last, i = member(i + 1)
                    if _SPACES in (first, last):
                        refuse("\\s at an end of a range")
                    first = f"{first}-{last}"
                parts.append(first)
            part = "".join(parts) + text[i : i + 1]  # its ], unless it has none
            i += 1
        elif c == "(":
            part = _GROUP.match(text, i).group()
            if part == "(" and text.startswith("(?", i):
                refuse(f"the group {text[i : i + 3]}")
            i += len(part)
        elif _PYTHONS_COUNT.match(text, i):
            refuse(f"the count {_PYTHONS_COUNT.match(text, i).group()}")
        else:
            count = _COUNT.match(text, i)
            part = count.group() if count else {".": _NO_LINE_END, "$": r"\Z"}.get(c, c)
            i = count.end() if count else i + 1
            if (count or c in "*+?") and text.startswith("+", i):
                refuse(f"the quantifier {text[start:i]}+")
        out.append(part)
    try:
        return re.compile("".join(out), re.ASCII)
    except re.error as exc:
        raise ValueError(
            f"the pattern {text} is not one Python's re reads: {exc.msg}"
        ) from None


class _Checker:
    """The schema's rows for one document (see validate)."""

    def __init__(self, doc: Document, report_missing: bool = True):
        self.root = doc.file_type.load()
        self.file_type = doc.file_type
        self.noun = doc.file_type.noun
        self.section_command = doc.file_type.section_command
        self.values = doc.values
        self.where = where_of(doc)
        self.again = read_again(doc.file_type, doc.path)
        # False: rows for the values the document has alone (misfits), none for a key it leaves out
        self.report_missing = report_missing
        self.rows = []
        # section -> (the features that are on, the keys they need that are missing): see missing
        self.sections = {}

    def fail(self, path: str, found: str, fix: Fix) -> None:
        self.rows.append(Check(path, Status.FAIL, found, fix))

    def edit(self, path: str, value) -> Edit:
        """The edit that sets the key at PATH in the file to VALUE (DELETE: deletes it)."""
        return Edit(path, self.where, value)

    def value(self, node, value, path: str, form: str = "") -> None:
        """The rows of VALUE, the value at PATH, by its schema NODE. FORM: the words for its form where NODE has no
        description (a list's item or a map's entry: the list's or the map's description).
        """
        node = _node(self.root, node, value)
        if not isinstance(node, Mapping):
            return
        if isinstance(value, str) and PLACEHOLDER.search(value):
            return  # the placeholder's row alone (_placeholder_rows), whatever the key takes
        if "oneOf" in node:
            self.union(node, value, path)
            return
        if isinstance(value, Plain):
            self.plain(node, value, path)
            return
        types = _types(node)
        if types and not any(_is(value, t) for t in types):
            self.wrong_type(node, value, path)
            return
        fixed = node.get("const", _MISSING)
        if fixed is not _MISSING and value != fixed:
            self.fail(
                path,
                f"{shown_value(value)} is not {shown_value(fixed)}",
                self.edit(path, fixed),
            )
        elif "enum" in node and value not in node["enum"]:
            self.fail(
                path,
                f"{shown_value(value)} is not {_choices(node)}",
                self.edit(path, Described(_choices(node))),
            )
        elif (
            isinstance(value, str)
            and "pattern" in node
            and not schema_pattern(node["pattern"]).search(value)
        ):
            self.fail(
                path,
                f"{shown_value(value)} is not in the form this key takes",
                self.edit(
                    path,
                    Described(
                        f"a value of this form: {_described(node, path) or form}"
                    ),
                ),
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self.bounds(node, value, path)
        elif isinstance(value, list):
            least = node.get("minItems")
            if isinstance(least, int) and len(value) < least:
                at_least = wrapping.counted(least, "entry", "entries")
                self.fail(
                    path,
                    f"{wrapping.counted(len(value), 'entry', 'entries')}, and it takes at least {least}",
                    self.edit(path, Described(f"a list of at least {at_least}")),
                )
            items = node.get("items")
            for i, v in enumerate(value):
                if isinstance(items, Mapping):
                    self.value(items, v, f"{path}[{i}]", _described(node, path))
        elif isinstance(value, Mapping):
            self.mapping(node, value, path)

    def union(self, node, value, path: str) -> None:
        alternatives = [_deref(self.root, a) for a in node["oneOf"]]
        key = _discriminator(self.root, alternatives)
        if not key:
            raise ValueError(
                f"{path}: a oneOf needs a key each alternative fixes with const"
            )
        allowed = [
            _fixed(_deref(self.root, a["properties"][key])) for a in alternatives
        ]
        shown_ = wrapping.joined([shown_value(a) for a in allowed], "or")
        got = value.get(key, _MISSING) if isinstance(value, Mapping) else _MISSING
        found = "missing" if got is _MISSING else f"{shown_value(got)} is not {shown_}"
        self.fail(
            _join(path, key), found, self.edit(_join(path, key), Described(shown_))
        )

    def plain(self, node, value: Plain, path: str) -> None:
        read_as = answers.typed(str(value))
        if "string" in _types(node) and not isinstance(read_as, str):
            if isinstance(read_as, bool) or read_as is None:
                what = f"the value {json.dumps(read_as)}"
            else:
                what = f"the number {_scalar(read_as, path)}"
            self.fail(
                path,
                f"{what}, as written {value}",
                Steps(("write it in quotes:", self.edit(path, str(value)))),
            )
            return
        self.fail(
            path,
            f"{value} is not {_type_words(node)}",
            self.edit(path, Described(_what(node))),
        )

    def wrong_type(self, node, value, path: str) -> None:
        types = _types(node)
        if isinstance(value, str) and ("integer" in types or "number" in types):
            number = answers.typed(value)
            if (
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and _is(number, types[0])
            ):
                self.fail(
                    path,
                    f"{shown_value(value)} is text, in quotes, and this key takes {_type_words(node)}",
                    Steps(("write it without quotes:", self.edit(path, number))),
                )
                return
        self.fail(
            path,
            f"{shown_value(value)} is not {_type_words(node)}",
            self.edit(path, Described(_what(node))),
        )

    def bounds(self, node, value, path: str) -> None:
        low, high = node.get("minimum"), node.get("maximum")
        if (low is None or value >= low) and (high is None or value <= high):
            return
        if low is not None and high is not None:
            what = f"a number from {low} to {high}"
        else:
            what = f"{low} or more" if low is not None else f"{high} or less"
        side = (
            f"less than {low}"
            if low is not None and value < low
            else f"more than {high}"
        )
        self.fail(
            path, f"{_scalar(value, path)} is {side}", self.edit(path, Described(what))
        )

    def mapping(self, node, value, path: str, present: str | None = None) -> None:
        """The keys of VALUE, the mapping at PATH, then the keys it misses. PRESENT is the innermost section of the
        file around PATH when the file leaves that section out (VALUE is then {}); None when VALUE is the
        file's own."""
        props = node.get("properties") or {}
        extra = node.get("additionalProperties", not props)
        names = (node.get("propertyNames") or {}).get("pattern")
        for k, v in value.items():
            key = _join(path, k)
            if not path and k in HEADER_KEYS:
                continue
            if k in props:
                self.value(props[k], v, key)
            elif extra is False:
                self.unknown(props, key, k)
            elif (
                names
                and not _placeholder_name(node, k)
                and not schema_pattern(names).search(str(k))
            ):
                takes = f"a name {path} takes ({_described(node, path).rstrip('.')})"
                self.fail(
                    key,
                    f"{shown_value(str(k))} is not a name {path} takes",
                    Steps(
                        (
                            f"rename {key} in {self.where} to {takes}, then run",
                            self.again,
                        )
                    ),
                )
            elif isinstance(extra, Mapping):
                self.value(extra, v, key, _described(node, path))
        if not self.report_missing:
            return
        if path and not in_use(self.root, self.values, path)[0]:
            return  # nothing reads an unused section, so it needs none of its keys
        own = present is None
        present = path if own else present
        # A section the file leaves out, which no section around it requires, needs none of its required keys: one
        # that requires keys of its own is written whole or left out.
        required = [
            k
            for k in (node.get("required") or () if own else ())
            if path or k not in HEADER_KEYS
        ]
        for k in required:
            if k not in value:
                key = _join(path, k)
                self.fail(key, "missing", self.edit(key, source_of(props.get(k))))
        for k, child in props.items():
            if k in value or k in required or (not path and k in HEADER_KEYS):
                continue
            key, child = _join(path, k), _deref(self.root, child)
            if child.get("x-required-when"):
                on = required_by(self.root, self.values, key, child)
                if on:
                    self.missing(key, child, on, present)
            elif _is_section(child):
                self.mapping(child, {}, key, present)
        if own and path in self.sections:
            self.section_row(path, *self.sections.pop(path))

    def missing(self, key: str, node, on: list, present: str) -> None:
        """KEY is missing while ON, the features that need it, are on. It joins the row of PRESENT, the innermost
        section of the file around it, which the section command prints in full; it has a row of its own when
        there is no section command, no such section, or one that command cannot print (a map's entry or a
        list's item)."""
        if self.section_command and present and _section_key(self.root, present):
            features, keys = self.sections.setdefault(present, ([], []))
            features.extend(f for f in on if f not in features)
            keys.append(key)
            return
        self.fail(
            key,
            f"missing, and {wrapping.joined(on)} {wrapping.are(on)} on",
            self.edit(key, source_of(node)),
        )

    def section_row(self, section: str, on: list, keys: list) -> None:
        """The one row of the keys missing from SECTION while ON are on: print the section and paste it over; then
        set back to true each switch in it that the file has on and the pasted text writes otherwise, and fill in
        the keys that are Required when."""
        pasted = _section_writer(self.file_type).values
        back = [
            switch[len(section) + 1 :]
            for switch in _switches_on(
                self.root,
                schema_at(self.root, section),
                _value_at(self.values, section),
                section,
            )
            if _value_at(pasted, switch) is not True
        ]
        then = f"set {wrapping.joined(back)} back to true and " if back else ""
        fix = Steps(
            (
                "run",
                steps.command_line(self.section_command, KEY=section),
                f", paste its text over {section} in {self.where}, then {then}fill in each key whose comment"
                ' says "Required when"',
            )
        )
        self.fail(
            section.rsplit(".", 1)[-1],
            f"{wrapping.joined(on)} {wrapping.are(on)} on, and {wrapping.joined(keys)} {wrapping.are(keys)} missing",
            fix,
        )

    def unknown(self, props, key: str, name) -> None:
        """A key the schema does not know, with the closest known key: beside it, else anywhere in the file."""
        parent = key[: -len(str(name))].rstrip(".")
        near = difflib.get_close_matches(str(name), list(props), n=1, cutoff=0.6)
        if near:
            closest, verb = _join(parent, near[0]), "rename"
        else:
            known = {}
            for p, _schema, _ in _schema_nodes(self.root, self.root, ""):
                if p and p not in HEADER_KEYS and "*" not in p and "[]" not in p:
                    known.setdefault(p.rsplit(".", 1)[-1], p)
            near = difflib.get_close_matches(str(name), list(known), n=1, cutoff=0.6)
            closest, verb = (known[near[0]] if near else ""), "move"
        delete = self.edit(key, DELETE)
        if closest:
            found = f"no such key in {self.noun}; did you mean {closest}?"
            words = f"{verb} {key} in {self.where} to {closest}, or"
            self.fail(key, found, Steps((words, delete)))
        else:
            self.fail(key, f"no such key in {self.noun}", delete)


def _placeholder_name(node, name) -> bool:
    """Whether NAME, a key of the mapping NODE describes, is a map entry's name that holds a placeholder: the
    placeholder's row is its one row, with no row for the map's rule for names (propertyNames) beside it.
    """
    return _is_map(node) and bool(PLACEHOLDER.search(str(name)))


def _placeholder_rows(root: dict, values, where: str, again: Command) -> list:
    """A FAIL row for each <...> left in VALUES: a value's fix sets it to where its value comes from; a map's
    entry name's renames it, then runs AGAIN."""
    rows = []

    def row(path: str, text: str, fix: Fix) -> None:
        rows.append(Check(path, Status.FAIL, f"still the template's {text}", fix))

    def visit(node, value, path: str) -> None:
        node = _node(root, node, value)
        if not isinstance(node, Mapping):
            return
        if isinstance(value, str) and PLACEHOLDER.search(value):
            row(path, value, Edit(path, where, source_of(node)))
        elif isinstance(value, list):
            items = node.get("items")
            for v in value:
                if isinstance(v, str) and PLACEHOLDER.search(v):
                    row(path, v, Edit(path, where, source_of(node)))
                elif isinstance(items, Mapping):
                    visit(items, v, path)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                if not path and k in HEADER_KEYS:
                    continue
                if _placeholder_name(node, k):
                    words = f"rename {_join(path, k)} in {where} to a name of your own, then run"
                    row(path, str(k), Steps((words, again)))
                child = _child(root, node, k)
                if child is not None:
                    visit(child, v, _join(path, k))

    visit(root, values, "")
    return rows


def _differs(value, base) -> bool:
    return value is not _MISSING and (base is _MISSING or value != base)


def _unused_rows(root: dict, template_values: dict, values: dict, where: str) -> list:
    """A WARN row for each value set for a feature that is off: delete it, or turn the feature on."""
    rows = []

    def visit(node, value, base, path: str) -> None:
        node = _node(root, node, value)
        used, off = in_use(root, values, path) if path else (True, [])
        if _is_section(node) and isinstance(value, Mapping):
            for k, child in node["properties"].items():
                if k in value and not (not path and k in HEADER_KEYS):
                    below = (
                        base.get(k, _MISSING) if isinstance(base, Mapping) else _MISSING
                    )
                    visit(child, value[k], below, _join(path, k))
            return
        if not used and _differs(value, base):
            them = "it" if len(off) == 1 else "one of them"
            parts = [Edit(path, where, DELETE), f", or turn {them} on:"]
            for i, feature in enumerate(off):
                parts += ["or"] if i else []
                parts.append(Edit(f"{feature}.enabled", where, True))
            found = f"set, and {wrapping.joined(off)} {wrapping.are(off)} off: nothing reads it"
            rows.append(Check(path, Status.WARN, found, Steps(parts)))

    visit(root, values, template_values, "")
    return rows


def validate(doc: Document) -> list:
    """The document's check rows: see the module's docstring."""
    checker = _Checker(doc)
    checker.value(checker.root, doc.values, "")
    rows = checker.rows + _placeholder_rows(
        checker.root, doc.values, checker.where, checker.again
    )
    failed = any(r.status == Status.FAIL for r in rows)
    if not failed:
        rows = [
            Check(
                "settings",
                Status.PASS,
                f"every key fits the {doc.file_type.noun} schema, and no <...> is left",
            )
        ]
    rows += _unused_rows(
        checker.root, _template_values(checker.root), doc.values, checker.where
    )
    if doc.file_type.coherence is not None:
        if failed:
            rows.append(
                Check(
                    "coherence",
                    Status.SKIP,
                    "checked once every key fits the schema and no <...> is left",
                )
            )
        else:
            rows += list(doc.file_type.coherence(doc))
    return rows


def misfits(doc: Document) -> list:
    """Validate's rows for the values DOC holds that their keys do not take: see the module's docstring."""
    checker = _Checker(doc, report_missing=False)
    checker.value(checker.root, doc.values, "")
    return checker.rows


def fits(doc: Document, key: str, value) -> bool:
    """Whether VALUE fits the schema of the key at KEY: see the module's docstring."""
    checker = _Checker(doc)
    node = schema_at(checker.root, key)
    if node is None:
        raise ValueError(
            f"schema_at finds no schema for {key} in the {doc.file_type.noun} schema"
        )
    checker.value(node, value, key)
    return not checker.rows


# --- values --------------------------------------------------------------------------------------------------------


def blank(value) -> bool:
    """Whether a key's VALUE leaves it blank: see the module's docstring."""
    return (
        value is _MISSING or value is None or (isinstance(value, str) and value == "")
    )


def _effective(root: dict, node, value, path: str):
    node = _node(root, node, value)
    if _is_section(node):
        value = value if isinstance(value, Mapping) else {}
        return {
            k: _effective(root, child, value.get(k, _MISSING), _join(path, k))
            for k, child in node["properties"].items()
            if path or k not in HEADER_KEYS
        }
    if blank(value):
        if isinstance(node, Mapping) and "default" in node:
            return copy.deepcopy(node["default"])
        return None if value is _MISSING else value
    extra = node.get("additionalProperties") if isinstance(node, Mapping) else None
    if isinstance(value, Mapping) and isinstance(extra, Mapping):
        return {k: _effective(root, extra, v, _join(path, k)) for k, v in value.items()}
    return copy.deepcopy(value)


def effective(doc: Document) -> dict:
    """Every key's effective value: see the module's docstring."""
    root = doc.file_type.load()
    return _effective(root, root, doc.values, "")


def _keys_set(root: dict, node, value, path: str):
    node = _node(root, node, value)
    if _is_section(node):
        value = value if isinstance(value, Mapping) else {}
        for k, child in node["properties"].items():
            if path or k not in HEADER_KEYS:
                yield from _keys_set(
                    root, child, value.get(k, _MISSING), _join(path, k)
                )
    elif not blank(value):
        yield path


def keys_set(doc: Document) -> list:
    """The keys the file sets: see the module's docstring."""
    root = doc.file_type.load()
    return list(_keys_set(root, root, doc.values, ""))


def canonical(value):
    """VALUE as canonical JSON holds it: see the module's docstring."""
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON's keys are text, not {k!r}")
            out[str(k)] = canonical(v)
        return out
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON takes no NaN or infinity")
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    raise TypeError(f"canonical JSON takes JSON's values, not {type(value).__name__}")


def fingerprint(values) -> str:
    """ "sha256:" and the SHA-256 of VALUES as canonical JSON."""
    text = json.dumps(
        canonical(values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- closing sentences ---------------------------------------------------------------------------------------------


def closing_sentence(node, required: bool, path: str) -> str:
    """A key's one closing sentence, made from its schema: see the module's docstring."""
    if path in HEADER_KEYS:
        why = str(node.get("x-changed-by") or "").strip()
        return f"Don't change it: {why}." if why else "Don't change it."
    if required:
        return "Required."
    blank = str(node.get("x-blank") or "").strip()
    when = str(node.get("x-required-when") or "").strip()
    if when:
        return f"Required when {when}." + (
            f" Otherwise, leave blank {blank}." if blank else ""
        )
    if blank:
        return f"Leave blank {blank}."
    if "default" in node:
        means = str(node.get("x-default-means") or "").strip()
        return default_sentence(node["default"], path, means)
    raise ValueError(
        f"{path}: no closing sentence: give it a default, x-blank or x-required-when, or require it"
    )


def default_sentence(default, path: str, means: str = "") -> str:
    """The closing sentence of a key whose default is DEFAULT: see the module's docstring."""
    if isinstance(default, (list, Mapping)) and not default:
        return "Default: none."
    if isinstance(default, Mapping):
        shown = ", ".join(
            f"{_quoted(str(k))}: {_scalar(v, f'{path}.{k}')}"
            for k, v in default.items()
        )
        shown = "{" + shown + "}"
    else:
        shown = _scalar(default, path)
    return f"Default: {shown}" + (f", which means {means}" if means else "") + "."


def closing_of(text: str, start=CLOSING_START, whole=CLOSING) -> tuple:
    """(the words before TEXT's closing sentence, that sentence): see the module's docstring."""
    text = str(text).strip()
    for m in reversed(list(start.finditer(text))):
        if whole.fullmatch(text, m.start()):
            return text[: m.start()].rstrip(), text[m.start() :]
    return text, ""


def _groups(root: dict, node) -> bool:
    """Whether NODE groups other keys, as a section or a oneOf of sections does: it has a title, not a closing
    sentence."""
    if _is_section(node):
        return True
    alternatives = node.get("oneOf") if isinstance(node, Mapping) else None
    return bool(alternatives) and all(
        _is_section(_deref(root, a)) for a in alternatives
    )


def schema_keys(schema: dict):
    """(path, node, required) for every key of the schema mapping SCHEMA: see the module's docstring."""
    for path, node, required in _schema_nodes(schema, schema, ""):
        if required is not None and not _groups(schema, node):
            yield path, node, required


def map_keys(schema: dict) -> frozenset:
    """The paths of the schema mapping SCHEMA's maps: see the module's docstring."""
    return frozenset(
        path
        for path, node, _required in schema_keys(schema)
        if "*" not in path and _is_map(node)
    )


def closing_problems(schema: dict) -> list:
    """What is wrong with each key's closing sentence in the schema mapping SCHEMA: see the module's docstring."""
    problems = []
    for path, node, required in schema_keys(schema):
        try:
            sentence = closing_sentence(node, required, path)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        body, closing = closing_of(node.get("description") or "")
        if not body and not closing:
            problems.append(f"{path}: no description")
        elif closing != sentence:
            problems.append(
                f"{path}: its description ends with {closing!r}, not {sentence!r}"
                if closing
                else f"{path}: its description does not end with its closing sentence, {sentence!r}"
            )
        elif not body:
            problems.append(
                f"{path}: its description says nothing before its closing sentence"
            )
        elif CLOSING_START.search(body):
            problems.append(
                f"{path}: its description has a second closing sentence, before {sentence!r}"
            )
    return problems


# --- writing -------------------------------------------------------------------------------------------------------


def _float(value: float, path: str) -> str:
    if not math.isfinite(value):
        raise Unwritable(path, f"{value} cannot be written: a number is finite")
    text = repr(value)
    mantissa, e, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return mantissa + e + exponent


def _quoted(text: str) -> str:
    return UNPRINTED.sub(
        lambda m: f"\\u{ord(m.group(0)):04x}", json.dumps(text, ensure_ascii=False)
    )


def _scalar(value, path: str) -> str:
    """A scalar as the writer writes it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Plain):
        raise Unwritable(
            path, f"{value} is as written, and not typed yet: validate the file first"
        )
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return _float(value, path)
    if isinstance(value, str):
        return _quoted(value)
    if isinstance(value, (list, Mapping)) and not value:
        return "[]" if isinstance(value, list) else "{}"
    if isinstance(value, list) and all(
        not isinstance(v, (list, Mapping)) for v in value
    ):
        return "[" + ", ".join(_scalar(v, path) for v in value) + "]"
    raise Unwritable(path, f"the writer cannot write {type(value).__name__} here")


def _map_key(name) -> str:
    """A map entry's name as the writer writes it: plain when it is a word YAML reads as text, or one placeholder
    (`<name>`, as a template's example entry has it); else in double quotes."""
    if not isinstance(name, str):
        raise ValueError(f"a map's names are text, not {name!r}")
    if re.fullmatch(r"<[A-Za-z0-9_.-]+>", name):
        return name
    plain = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) and name.lower() not in (
        "y",
        "n",
        "yes",
        "no",
        "on",
        "off",
        "true",
        "false",
        "null",
    )
    return name if plain else _quoted(name)


def template_value(node, path: str):
    """A key's template value, from its schema NODE: see the module's docstring."""
    for key in ("x-template", "default", "const"):
        if key in node:
            return copy.deepcopy(node[key])
    raise ValueError(
        f"{path}: no value for the template: give it a default or an x-template"
    )


def _whole_or_left_out(root: dict, node, key: str) -> bool:
    """Whether the key KEY of the section NODE is a section NODE does not require, which requires keys of its own: a
    section you write whole or leave out (see the module's docstring)."""
    child = _deref(root, node["properties"][key])
    return (
        _is_section(child)
        and bool(child.get("required"))
        and key not in (node.get("required") or ())
    )


def _template_values(root: dict, node=None, path: str = "", every_key: bool = False):
    """The value the template writes for the node at PATH: for a section, a mapping of its keys', but a section
    written whole or left out, which only the every-key text (EVERY_KEY) writes."""
    node = _deref(root, root if node is None else node)
    if _is_section(node):
        return {
            k: _template_values(root, child, _join(path, k), every_key)
            for k, child in node["properties"].items()
            if (path or k not in HEADER_KEYS)
            and (every_key or not _whole_or_left_out(root, node, k))
        }
    return template_value(node, path)


def _merge(root: dict, node, base, value):
    """VALUE laid over BASE, section by section; any other value replaces BASE's."""
    if value is _MISSING:
        return copy.deepcopy(base)
    node = _node(root, node, value)
    if _is_section(node) and isinstance(base, Mapping) and isinstance(value, Mapping):
        out = copy.deepcopy(dict(base))
        for k, v in value.items():
            child = _child(root, node, k)
            out[k] = (
                _merge(root, child, base.get(k, _MISSING), v)
                if k in base
                else copy.deepcopy(v)
            )
        return out
    return copy.deepcopy(value)


def _put(values: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = values
    for part in parts[:-1]:
        cur = cur.get(part) if isinstance(cur, dict) else None
    if not isinstance(cur, dict) or parts[-1] not in cur:
        raise ValueError(f"a profile sets {dotted}, which is no key of the template")
    cur[parts[-1]] = copy.deepcopy(value)


def _profiles(root: dict) -> tuple:
    return tuple(
        Profile(
            str(name),
            str(p.get("what") or ""),
            tuple(str(h) for h in p.get("header") or ()),
            dict(p.get("values") or {}),
        )
        for name, p in (root.get("x-profiles") or {}).items()
    )


def profiles(file_type: FileType) -> tuple:
    """The profiles of FILE_TYPE's schema, in order: see the module's docstring."""
    return _profiles(file_type.load())


def _profile(root: dict, file_type: FileType, name) -> Profile | None:
    known = {p.name: p for p in _profiles(root)}
    if not known:
        if name is not None:
            raise ValueError(f"{file_type.noun} have no profiles")
        return None
    name = next(iter(known)) if name is None else name
    if name not in known:
        raise ValueError(
            f'{file_type.noun} have no profile "{name}"; the profiles: {", ".join(known)}'
        )
    return known[name]


def _value_at(values, dotted: str):
    """The value at the dotted path DOTTED in VALUES, or None where there is none."""
    for part in dotted.split("."):
        if not isinstance(values, Mapping) or part not in values:
            return None
        values = values[part]
    return values


def _switches_shown(switches, values: dict) -> str:
    return wrapping.joined([f"{s} {json.dumps(values[s])}" for s in switches])


def profile_of(doc: Document) -> tuple:
    """(the profile DOC's switches match, or None; why): see the module's docstring."""
    file_type = doc.file_type
    known = profiles(file_type)
    if not known:
        return (
            None,
            f"the title line names no profile, as {file_type.noun} have no profiles",
        )
    root = file_type.load()
    features = dict.fromkeys(
        path
        for path, node, _ in _schema_nodes(root, root, "")
        if path and "*" not in path and "[]" not in path and is_feature(node)
    )
    # The switches of the outer features: a feature inside another is one of its parts.
    switches = [
        _join(path, "enabled")
        for path in features
        if not any(path.startswith(f"{outer}.") for outer in features)
    ]
    base = _template_values(root)
    starts = {}
    for profile in known:
        values = copy.deepcopy(base)
        for dotted, value in profile.values.items():
            _put(values, dotted, value)
        starts[profile.name] = {s: _value_at(values, s) for s in switches}
    differ = [
        s for s in switches if len({json.dumps(v[s]) for v in starts.values()}) > 1
    ]
    has = {}
    for s in differ:
        value = _value_at(doc.values, s)
        has[s] = _value_at(base, s) if value in (None, "") else value
    for profile in known:
        start = starts[profile.name]
        if all(has[s] == start[s] for s in differ):
            what = profile.what.strip()
            why = f'the title line names the profile "{profile.name}"' + (
                f" ({what})" if what else ""
            )
            if differ:
                why += f", which starts with {_switches_shown(differ, start)}, as your file has them"
            return profile.name, why
    return None, (
        f"the title line names no profile, as none starts with {_switches_shown(differ, has)}, as your file has "
        "them"
    )


def _writable(root: dict, node, value, path: str, noun: str) -> None:
    """`Unwritable`, naming the key, for what the writer cannot place: a key the schema does not know, a plain
    value."""
    node = _node(root, node, value)
    if isinstance(value, Plain):
        raise Unwritable(
            path, f"{value} is as written, and not typed yet: validate the file first"
        )
    anything = _anything(node)
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not path and k in HEADER_KEYS:
                continue
            child = node if anything else _child(root, node, k)
            if child is None:
                raise Unwritable(
                    _join(path, k),
                    f"no key of {noun}, so the writer has no place for it",
                )
            _writable(root, child, v, _join(path, k), noun)
    elif isinstance(value, list) and isinstance(node, Mapping):
        items = node if anything else node.get("items")
        for i, v in enumerate(value if isinstance(items, Mapping) else ()):
            _writable(root, items, v, f"{path}[{i}]", noun)


class _Writer:
    """One file's text: the schema's layout and comments, with VALUES; TEMPLATE_VALUES say what differs. EVERY_KEY:
    every key, none left out or folded (the every-key text)."""

    def __init__(
        self,
        file_type: FileType,
        template_values: dict,
        values: dict,
        every_key: bool = False,
    ):
        self.file_type = file_type
        self.root = file_type.load()
        self.base = template_values
        self.values = values
        self.every_key = every_key
        self.width = WIDTH  # narrower while lines are written to be commented out
        self.turn_on = str(self.root.get("x-turn-on") or "").strip()
        self.said = None  # a list: each comment above a key is kept in it, as `comments` gives them

    def comment(self, text: str, pad: str) -> list:
        return wrapping.wrap(text, self.width, pad + "# ", pad + "# ")

    def above(self, path: str, text: str, pad: str, groups: bool = False) -> list:
        """The comment above the key at PATH: TEXT's lines at PAD, kept in `said` while that is a list. GROUPS: it heads
        other keys (a section's, or a form's heading)."""
        if self.said is not None:
            self.said.append(Comment(path, text, groups))
        return self.comment(text, pad)

    def narrower(self, write, *args) -> list:
        """The lines WRITE(*ARGS) gives, written 2 columns narrower: as they are before they are commented out."""
        self.width -= 2
        try:
            return write(*args)
        finally:
            self.width += 2

    def commented(self, indent: int, write, *args) -> list:
        """The lines WRITE(*ARGS) gives, commented out: "# " put in at column INDENT of each, which each line was
        written 2 columns narrower to leave room for."""
        lines = self.narrower(write, *args)
        pad = " " * indent
        if any(not line.startswith(pad) for line in lines):
            raise ValueError(f"lines to comment out at column {indent} start there")
        return [f"{pad}# {line[indent:]}" for line in lines]

    def title(self, node, path: str) -> str:
        title = str(node.get("title") or "").strip()
        if not title:
            raise ValueError(
                f"{path}: a section, and a key written as not used, have a title"
            )
        return title

    def section_text(self, node, path: str) -> str:
        desc = description(node, path)
        return (
            f"{self.title(node, path)}: {desc}"
            if desc
            else f"{self.title(node, path)}."
        )

    def key_text(self, node, path: str, turn_on: bool = False) -> str:
        """A key's comment: its description as the schema has it, after "Title: " when it has a title; TURN_ON puts
        the schema's x-turn-on sentence before the closing sentence."""
        desc = description(node, path)
        if node.get("title"):
            desc = f"{node['title']}: {desc}"
        if not desc:
            raise ValueError(f"{path}: every key has a description")
        if turn_on and self.turn_on:
            body, closing = closing_of(desc)
            desc = " ".join(part for part in (body, self.turn_on, closing) if part)
        return desc

    def text(self, profile: Profile | None) -> str:
        lines = self.header(profile)
        for k, node in self.root["properties"].items():
            if k in HEADER_KEYS:
                node = _deref(self.root, node)
                value = self.file_type.kind if k == "kind" else self.file_type.version
                part = self.above(k, self.key_text(node, k), "") + [
                    f"{k}: {_scalar(value, k)}"
                ]
            else:
                part = self.node(k, node, self.values.get(k, _MISSING), k, 0)
            if part:
                if k != HEADER_KEYS[1]:
                    lines.append("")
                lines += part
        return "\n".join(lines) + "\n"

    def header(self, profile: Profile | None) -> list:
        lines = []
        address = self.root.get("$id")
        if isinstance(address, str) and address.startswith("https://"):
            lines.append(f"# yaml-language-server: $schema={address}")
        title = _title(self.root, self.file_type)
        title = (
            f'{title} (profile "{profile.name}": {profile.what}).'
            if profile
            else f"{title}."
        )
        paragraphs = [str(p) for p in self.root.get("x-header") or ()]
        if profile and profile.header:
            paragraphs = (
                paragraphs[:1] + [str(p) for p in profile.header] + paragraphs[1:]
            )
        for paragraph in [title] + paragraphs:
            lines += self.comment(paragraph, "")
        return lines

    def node(
        self,
        key: str,
        node,
        value,
        path: str,
        indent: int,
        full: bool = False,
    ) -> list:
        """The lines of the key at PATH: its comment and its value; for a section, its keys'. FULL: inside a section
        written in full (unused, or a feature that is off, with values of its own), so no key is left out.
        """
        node = _node(self.root, node, value)
        if value is _MISSING:
            return (
                []
            )  # a section written whole or left out, which these values leave out
        pad = " " * indent
        base = self.base_at(path)
        full = full or self.every_key
        if not full and not in_use(self.root, self.values, path)[0]:
            # on a section or a map only (_check_schema)
            reason = str(node.get("x-off-reason") or "").strip()
            if _differs(value, base):
                full = True  # unused, with values of its own: every key, so that none is lost
            elif reason:
                closing = (
                    "" if _is_section(node) else closing_of(description(node, path))[1]
                )
                text = f"{self.title(node, path)}: not used while {reason}. {closing}"
                return self.above(path, text.rstrip(), pad, _is_section(node)) + [
                    f"{pad}{key}: {{}}"
                ]
            else:
                return []
        if not _is_section(node):
            return self.above(path, self.key_text(node, path), pad) + self.value(
                key, node, value, path, indent
            )
        return self.above(
            path, self.section_text(node, path), pad, True
        ) + self.section(key, node, value, path, indent, full)

    def section(
        self, key: str, node, value, path: str, indent: int, full: bool
    ) -> list:
        """A section's lines after its comment: KEY, then its keys'. A feature that is off is its switch alone, unless
        it holds values of its own or every key is written; a section its chooser does not choose, see `other`.
        """
        pad = " " * indent
        base = self.base_at(path)
        if (
            not self.every_key
            and is_feature(node)
            and isinstance(value, Mapping)
            and value.get("enabled") is False
        ):
            rest = {k: v for k, v in value.items() if k != "enabled"}
            below = (
                {k: v for k, v in base.items() if k != "enabled"}
                if isinstance(base, Mapping)
                else {}
            )
            if not any(_differs(v, below.get(k, _MISSING)) for k, v in rest.items()):
                switch = _deref(self.root, node["properties"]["enabled"])
                hides = any(k != "enabled" for k in node["properties"])
                text = self.key_text(switch, _join(path, "enabled"), turn_on=hides)
                return (
                    [f"{pad}{key}:"]
                    + self.above(_join(path, "enabled"), text, pad + "  ")
                    + [f"{pad}  enabled: false"]
                )
            full = True  # off, with values of its own: every key, so that none is lost
        lines = []
        value = value if isinstance(value, Mapping) else {}
        unchosen = self.unchosen(node, value, path)
        for k, child in node["properties"].items():
            write = self.other if k in unchosen else self.node
            lines += write(
                k, child, value.get(k, _MISSING), _join(path, k), indent + 2, full
            )
        return [f"{pad}{key}:"] + lines if lines else [f"{pad}{key}: {{}}"]

    def unchosen(self, node, value, path: str) -> set:
        """The sections of the section NODE that a key with x-chooses beside them does not choose: the values of its
        enum but its own (its template value where VALUE leaves it out); none while its value is none of them.
        """
        out = set()
        for k, child in node["properties"].items():
            child = _deref(self.root, child)
            if isinstance(child, Mapping) and child.get("x-chooses") is True:
                chosen = value.get(k, self.base_at(_join(path, k)))
                if chosen in child["enum"]:
                    out |= {c for c in child["enum"] if c != chosen}
        return out

    def set_aside(self, path: str) -> list:
        """The sections along PATH, from the top to the key's own, that a key beside them does not choose: the
        every-key text comments out each one, after its comment (`other`)."""
        out, node, value, at = [], self.root, self.values, ""
        for part in path.split("."):
            here = value if isinstance(value, Mapping) else {}
            if part in self.unchosen(node, here, at):
                out.append(_join(at, part))
            node = _deref(self.root, node["properties"][part])
            value, at = here.get(part, _MISSING), _join(at, part)
        return out

    def other(
        self,
        key: str,
        node,
        value,
        path: str,
        indent: int,
        full: bool = False,
    ) -> list:
        """A section its chooser does not choose: with every key, its comment, then the section commented out,
        for you to use instead; else left out, unless it holds values of its own."""
        node = _node(self.root, node, value)
        if self.every_key:
            pad = " " * indent
            return self.above(
                path, self.section_text(node, path), pad, True
            ) + self.commented(
                indent, self.section, key, node, value, path, indent, True
            )
        if _differs(value, self.base_at(path)):
            return self.node(key, node, value, path, indent, full)
        return []

    def examples(self, node, path: str, indent: int) -> list:
        """With every key, a map's example entries, commented out: one for each form its entries take (each oneOf
        alternative, after its description as a heading), named by its propertyNames' x-template, with the template's
        values; [] for a map without that name, or while not every key is written."""
        names = node.get("propertyNames") if isinstance(node, Mapping) else None
        if not (
            self.every_key and isinstance(names, Mapping) and "x-template" in names
        ):
            return []
        name = str(names["x-template"])
        entry = _deref(self.root, node.get("additionalProperties"))
        forms = (
            [_deref(self.root, a) for a in entry["oneOf"]]
            if isinstance(entry, Mapping) and "oneOf" in entry
            else [entry]
        )
        pad, where, lines = " " * indent, _join(path, name), []
        for form in forms:
            if len(forms) > 1:
                heading = str(form.get("description") or "").strip()
                if not heading.endswith("."):
                    raise ValueError(
                        f"{path}: each form of its entries has a description, one sentence"
                    )
                lines += self.above(where, heading[:-1] + ":", pad, True)
            values = _template_values(self.root, form, where)
            lines += self.commented(
                indent, self.entry, _map_key(name), form, values, where, indent + 2
            )
        return lines

    def base_at(self, path: str):
        cur = self.base
        for part in path.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return _MISSING
            cur = cur[part]
        return cur

    def value(self, key: str, node, value, path: str, indent: int) -> list:
        pad = " " * indent
        if isinstance(value, Mapping):
            out = [f"{pad}{key}:"] if value else [f"{pad}{key}: {{}}"]
            entry = (
                node.get("additionalProperties") if isinstance(node, Mapping) else None
            )
            for name, v in value.items():
                out += self.entry(
                    _map_key(name),
                    entry if isinstance(entry, Mapping) else None,
                    v,
                    _join(path, name),
                    indent + 2,
                )
            return out + self.examples(node, path, indent)
        if isinstance(value, list) and value:
            if any(isinstance(v, list) for v in value):
                raise Unwritable(path, "the writer writes no list of lists")
            if not any(isinstance(v, Mapping) for v in value):
                line = f"{pad}{key}: {_scalar(value, path)}"
                if len(line) <= self.width:
                    return [line]
                return [f"{pad}{key}:"] + [
                    f"{pad}  - {_scalar(v, path)}" for v in value
                ]
            out = [f"{pad}{key}:"]
            items = node.get("items") if isinstance(node, Mapping) else None
            for v in value:
                out += self.item(
                    items if isinstance(items, Mapping) else None, v, path, indent + 2
                )
            return out
        return [f"{pad}{key}: {_scalar(value, path)}"]

    def item(self, node, value, path: str, indent: int) -> list:
        """A list's item in block style: a mapping's first key on the dash's line, the others two columns in."""
        pad = " " * indent
        if not isinstance(value, Mapping) or not value:
            return [f"{pad}- {_scalar(value, path)}"]
        node = _node(self.root, node, value) if node is not None else None
        lines = []
        for name, v in value.items():
            child = _child(self.root, node, name) if node is not None else None
            lines += self.value(
                _map_key(name),
                _deref(self.root, child),
                v,
                _join(path, name),
                indent + 2,
            )
        return [f"{pad}- {lines[0][indent + 2:]}"] + lines[1:]

    def entry(self, name: str, node, value, path: str, indent: int) -> list:
        """A map's entry: its keys commented from the entry's schema, in the schema's order."""
        pad = " " * indent
        node = _node(self.root, node, value) if node is not None else None
        if not (_is_section(node) and isinstance(value, Mapping)):
            return self.value(name, node, value, path, indent)
        if not value:
            return [f"{pad}{name}: {{}}"]
        lines = [f"{pad}{name}:"]
        for k, child in node["properties"].items():
            if k not in value:
                continue
            child = _node(self.root, child, value[k])
            if _is_section(child):
                lines += self.above(
                    _join(path, k),
                    self.section_text(child, _join(path, k)),
                    pad + "  ",
                    True,
                )
                lines += self.entry(k, child, value[k], _join(path, k), indent + 2)
            else:
                lines += self.above(
                    _join(path, k), self.key_text(child, _join(path, k)), pad + "  "
                )
                lines += self.value(k, child, value[k], _join(path, k), indent + 2)
        return lines


def _template_writer(file_type: FileType, profile, every_key: bool) -> tuple:
    """The writer of a template for PROFILE (None: the first, if any), and that profile."""
    root = file_type.load()
    chosen = _profile(root, file_type, profile)
    base = _template_values(root, every_key=every_key)
    values = copy.deepcopy(base)
    for dotted, value in (chosen.values if chosen else {}).items():
        _put(values, dotted, value)
    return _Writer(file_type, base, values, every_key), chosen


def _section_writer(file_type: FileType) -> _Writer:
    """The writer of what a section command prints: the every-key text, for the schema's first profile."""
    return _template_writer(file_type, None, True)[0]


def template(
    file_type: FileType, profile: str | None = None, every_key: bool = False
) -> str:
    """The template's text: see the module's docstring."""
    writer, chosen = _template_writer(file_type, profile, every_key)
    return writer.text(chosen)


def sections(file_type: FileType) -> list:
    """(KEY, title) for each section a section command prints: see the module's docstring."""
    root = file_type.load()
    found = []

    def walk(node, path: str) -> None:
        for k, child in _deref(root, node)["properties"].items():
            child = _deref(root, child)
            key = _join(path, k)
            if (path or k not in HEADER_KEYS) and (
                _is_section(child) or (_is_map(child) and child.get("title"))
            ):
                found.append((key, str(child.get("title") or "").strip()))
            if _is_section(child):
                walk(child, key)

    walk(root, "")
    return found


def section(file_type: FileType, key: str) -> str:
    """The text a section command prints for KEY: see the module's docstring."""
    known = [k for k, _ in sections(file_type)]
    if key not in known:
        close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
        if close:
            raise ValueError(f"no section {key}; did you mean {close[0]}?")
        command = steps.command_name(file_type.section_command)
        listed = (
            f"; {steps.command_line(command)} lists the sections" if command else ""
        )
        raise ValueError(f"no section {key}{listed}")
    writer = _section_writer(file_type)
    value = writer.values
    for part in key.split("."):
        value = value.get(part, _MISSING) if isinstance(value, Mapping) else _MISSING
    name, indent = key.rsplit(".", 1)[-1], 2 * key.count(".")
    node = _node(writer.root, schema_at(writer.root, key), value)
    aside = writer.set_aside(key)
    # What the every-key text comments out is written as it is written there before its "# ": 2 columns narrower
    # for each section around the key that is set aside, and the key's own section's keys 2 more when it is one.
    writer.width -= 2 * len([at for at in aside if at != key])
    if key in aside:
        lines = writer.comment(
            writer.section_text(node, key), " " * indent
        ) + writer.narrower(writer.section, name, node, value, key, indent, True)
    else:
        lines = writer.node(name, node, value, key, indent)
    return "\n".join(lines) + "\n"


def comments(file_type: FileType) -> list:
    """Each comment of the every-key text, with its key, in the file's order: see the module's docstring."""
    writer, chosen = _template_writer(file_type, None, True)
    writer.said = []
    writer.text(chosen)
    return writer.said


def dump(doc: Document) -> str:
    """The document's text, as the one writer writes it: see the module's docstring."""
    root = doc.file_type.load()
    _writable(root, root, doc.values, "", doc.file_type.noun)
    base = _template_values(root)
    profile = _profile(root, doc.file_type, doc.profile) if doc.profile else None
    return _Writer(doc.file_type, base, _merge(root, root, base, doc.values)).text(
        profile
    )


def _holds(path: pathlib.Path, data: bytes) -> bool:
    """Whether PATH is a regular file, not a link, that holds DATA."""
    return path.is_file() and not path.is_symlink() and path.read_bytes() == data


def _kept_taken(kept: pathlib.Path, where: str, name: str, retry):
    """The refusal for a kept file's place that holds another file: move that one aside, then migrate again."""
    found = f"{shown(kept)} is another file, and a kept file is never written over"
    fix = Steps((move_aside(kept), ", then", retry(where)))
    return private_files.Refused([Check(name, Status.FAIL, found, fix)])


def rewrite(path, doc: Document, retry, dry_run: bool = False) -> pathlib.Path:
    """rkmon migrate's write: see the module's docstring."""
    p = pathlib.Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    where, name = shown(p), f"{doc.file_type.noun} file"
    again = read_again(doc.file_type, p, older=True)
    data = p.read_bytes()
    if doc.data is None:
        raise ValueError(
            f"{where}: rewrite writes a document read from the file, and this one was not"
        )
    if data != doc.data:
        raise ValueError(f"{where} changed after migrate read it")
    old, values = _parse(data, where, name, again)
    layout = _header(values, where, doc.file_type, name, again, older=True)
    if layout >= doc.version:
        raise ValueError(
            f"{where} is not in an earlier layout than {doc.version}: there is nothing to keep"
        )
    text = dump(doc)
    kept = p.with_name(f"{p.name}.v{layout}")
    if not os.path.lexists(kept):
        try:
            private_files.write(
                kept, old, PACKAGE, retry=retry, dry_run=dry_run, shown=shown(kept)
            )
        except private_files.Refused:
            if not os.path.lexists(kept):
                raise  # refused for another reason than a file made there meanwhile
    # Another file, found there or made there meanwhile, is never written over; a dry run made none.
    if (os.path.lexists(kept) or not dry_run) and not _holds(kept, data):
        raise _kept_taken(kept, where, name, retry)
    if not dry_run:
        private_files.tighten(kept)
    private_files.write(
        p, text, PACKAGE, replace=True, retry=retry, dry_run=dry_run, shown=where
    )
    return kept
