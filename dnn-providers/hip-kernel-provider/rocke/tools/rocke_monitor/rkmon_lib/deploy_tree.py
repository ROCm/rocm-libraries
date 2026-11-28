# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deploy settings file (file_types.DEPLOY: schema/deploy_settings.schema.json) and the configuration today's
monitor runs (config.json): the carry from the file into the configuration, through which the monitor runs the
delivered layer (rkmon_lib/runtime_config.py); then the template of the earlier layout, the tests'.

The carry is generated from the schema: each key names, in its `x-internal`, the configuration
key or keys it feeds (`feeds()`). A configuration key that one file key feeds takes that key's value as it is: among
them the keys merged into a key of the file that today's code still reads, each in that key's x-internal after the
key's own: jira.url, team_dashboard.url and tick_interval_min. The
others are made by rules (`rules()`), each named by the file keys it reads. A rule the schema cannot say (_JOINED)
reads the keys that feed its configuration key, or those that feed the configuration keys it names (`_fed`), and
takes their values in the schema's order; rules() refuses a rule that does not read every key that feeds its
configuration key:

- deployment.image: image.name and image.tag joined, NAME:TAG; a blank tag with the default name gives "".
- ramdisk.enabled and ramdisk.size_gb: host.resources.ramdisk_gib, on when it is above 0.
- features.merge_queue.auto_merge: "enabled" when auto_merge.enabled is true, else "hints_only".
- automerge_guard_own: true when the auto-merge guard treats your pull requests as everyone else's ("same_as_others").
- incident_issue.labels: the develop guard's CI-red label, then its extra labels; a blank CI-red label keeps its
  place while there are extra labels, so that no extra label reads as the CI-red one.
- jira.component: the ticket components as one text, ", " between them.
- publish.html.requests: while priority requests are on, the requests file the site folder gives
  (derived.site_requests); else none, "".
- update.repo: the GitHub address of updates.repository, https://github.com/<org>/<repo>.
- atlassian.enabled: true when a type of ticket is on: the keys that feed ticket creation's switch.
- The switch of alerts and of ticket creation: on while a switch that feeds it is on, whatever the mode: alerts while
  either kind is (features.alerts.develop_breaks.enabled, features.alerts.needs_you.enabled), ticket creation while a
  type of ticket is. Each kind's own switch is carried as it is too, and the kind's sender reads both
  (rkmon_lib/features.py). The mode of each of the five features that send is carried as it is: in
  prototype a feature does its work, and what it would send stops at the one outward gate (rkmon_lib/post.py).
- prices: the spend prices by model id, each price under config.json's name, while spend is on (spend.enabled, which
  feeds prices too); none while it is off.
- A pass's model: an optional pass's (one whose model defaults to "default") as it is; a pass that runs in Claude
  Code by the name Claude Code takes: the model_id of a claude binding by that name, else the name (opus, sonnet
  and haiku are the built-in models, each sent by its id: BUILTIN, model_id). A pass's effort: "" for "default".
- model_bindings: every binding but the one named `worker`, with config.json's names for its provider and for how its
  key is sent (PROVIDER, AUTH_MODE), and no blank api_key_header (config.json's header is then Authorization); and a
  claude binding for each built-in model an optional pass names that the file does not bind.
- worker.*: the binding named `worker`, today's local worker, under config.json's worker names (WORKER_FIELDS), its
  provider as the worker's backend (WORKER_BACKEND).
- worker.think, the local worker's effort, merged into the first review pass's: the effort
  of the keys that feed reviewer and effort.first_pass, the first review pass's model and effort, while the pass runs
  on the binding named `worker` and the effort is one the worker sends (OTHER_EFFORTS), else none, so the
  package's own stands: unset, and the worker's requests carry no effort. claude.effort stays the
  package's base effort, since the file names each Claude pass's own.

NOT_CARRIED names each key today's container does not take, and why; every other key feeds a configuration key or is
read by a rule (a test checks it). integrations.claude_code's api_key_sent_as, api_key_header and api_key_scheme
feed claude.api_key_sent_as, claude.api_key_header and claude.api_key_scheme, which say how the Claude session sends
its key (orchestrator.session_variables).

- `to_config(values)`: a file's values (settings_file.Document.values) -> the configuration they make: every
  configuration key the file feeds or a rule makes, from the file's values and its keys' defaults
  (settings_file.effective); a key with neither is left out, and so is one whose rule cannot take the file's value
  (the file's checks name that value).
- `internal_defaults()`: the configuration keys the file feeds, at the file's defaults, the registry's defaults of
  the deploy keys (rkmon_lib/registry.py): from each key's default, else its template value when that holds no
  placeholder, else its type's blank; but atlassian.enabled, which a file answers.
- `read(path) -> (configuration, rows)`: a deploy settings file through settings_file.read and settings_file.validate:
  the configuration it makes and its FAIL and WARN rows, with check 7's PASS row, which says where the team dashboard
  in prototype is served (coherence.TEAM_ROW). A file the reader refuses raises settings_file.Refused, with
  its rows: a file in the layout before layout 1 with migrate's row, a new file as its fix (`migrate.before_layout_1`);
  one that cannot be opened, its OSError.
- `file_key(internal)`: the file key that names a configuration key in messages and documents: the key that feeds
  it, or the section around the keys its rule reads; for a key inside one the file feeds (a map's entry, a key under
  a whole mapping), that key's, with the rest kept. None for a key the file does not feed. `name(internal)`: its
  file key, else the key itself. `held()`: each configuration key the file feeds or a rule makes, with its file key.
- `value_schema(internal)`: the schema of the values a configuration key takes from the file: the key's that feeds it
  as it is, or its rule's own; None when neither says.
- `node(key)`: (the schema path, the schema node) of a file key: a key's (a map's entry by its name or as NAME.*) or a
  section's (rkmon_lib/schema.py node); ("", None) for a key the schema does not have.
- `description(key)`: a file key's description in the schema, as settings_file.description gives it, from its node;
  "" when it has none.
- `entries(map_key)`: the keys of each entry of the file's map MAP_KEY (after MAP_KEY.*.) -> the keys of the
  configuration entry each feeds (after INTERNAL.*.), from their x-internal: a binding's fields (model_bindings), a
  model's prices.
- `worker_fields()`: each key the binding named `worker` carries: its name in config.json's bindings -> its name
  under config.json's `worker` (WORKER_FIELDS).
- `parts()`: the file's sections, its top-level keys but the header's.
- `builtin()`: the built-in models, BUILTIN; `model_id(name)`: the id a Claude model is sent by; `family(model)`: the
  Claude model family a model's name or id holds (FAMILIES), the name the Agent tool's `model` option takes.
  OTHER_EFFORTS: the efforts a model that is not Claude's takes, which rkmon_lib/coherence.py checks a pass's effort
  against and rkmon_lib/bindings.py a binding's.
- `get(data, dotted)` and `put(data, dotted, value)`: a dotted key's value in nested mappings (MISSING when it has
  none), and its setting, the mappings on the way made.
- `image_parts(ref) -> (name, tag)`: the image REF, <name>:<tag>, split at its last ":" unless a "/" follows it, as a
  ":" before the last "/" is a registry's port: host:5000/name is (host:5000/name, "") and host:5000/name:tag
  (host:5000/name, tag). ("", "") for a blank REF or one with no ":" at all, which layout 1 reads as no image of its
  own (deployment.image_ref). The joined rule of deployment.image above makes the reference this splits.

The template of the earlier layout (schema/layout1/, through rkmon_lib/schema.py), the tests': they check the
guides' samples against it, which show that layout until the guides are written for layout 2. `template`, with
`schema_line` and `deploy_schema`; the templates the package gives you are layout 2's (rkmon_lib/settings_templates.py).

docker/rkmon runs this on the host: the standard library only.
"""

from __future__ import annotations

import copy
import functools
import inspect
from typing import Callable, NamedTuple

from . import derived, image, schema, settings_file
from .checks import Status

FILE = "your deploy settings file"  # the file in a fix's words, where no path of it is at hand
DEFAULT_MODEL = (
    "default"  # an optional pass's model and effort: the pass in charge does its work
)
WORKER = "worker"  # the model binding today's container runs as its local worker (config.json's `worker`)
# The built-in models: each name, and the id every call sends it by, the latest Claude model's (model_id).
BUILTIN = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
}
# The Claude model families (`family`): the built-in models' names and fable. They are the only names the Agent tool's
# `model` option takes, and the model guard (rkmon_lib/spend_scan.py) compares a call's model with its pass's by them.
FAMILIES = (*BUILTIN, "fable")
# The efforts a model that is not Claude's takes (a model_bindings entry whose provider is openai_compatible or
# ollama): what its requests send as their reasoning effort.
OTHER_EFFORTS = ("low", "medium", "high")
# config.json's values where a binding's differ: its provider, and how its key is sent (worker/llm_client.py).
PROVIDER = {"openai_compatible": "openai-compatible"}
AUTH_MODE = {"header_and_bearer": "both"}
# The local worker's names for a binding's configuration keys (config.json's worker), and its backend for each
# provider.
WORKER_FIELDS = {"provider": "backend", "concurrency": "max_parallel"}
WORKER_BACKEND = {"ollama": "ollama", "openai-compatible": "openai"}
# A section of secrets' values, which no file holds: settings_file refuses it, and the template of the earlier layout
# leaves it out.
SECRET_VALUES = "secret_values"

# Each key today's container does not take, and why.
NOT_CARRIED = {
    "features.review_automation.skills_folder": "the host reads the folder: the first start and each apply put its "
    "files in the settings volume beside the layer, and the session's render lays them over the package's",
}


class _Missing:
    pass


MISSING = _Missing()


def get(d, dotted: str):
    """The value at a dotted key of nested mappings, MISSING when there is none."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def put(d: dict, dotted: str, value) -> None:
    """Set a dotted key of nested mappings, making the mappings on the way (a value there that is no mapping is
    replaced)."""
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _file_type():
    from . import file_types  # which reads this module first

    return file_types.DEPLOY


def builtin() -> dict:
    """The built-in models: name -> the latest Claude model's id."""
    return dict(BUILTIN)


def model_id(name: str) -> str:
    """The id a Claude model is sent by: a built-in model's name gives its id (BUILTIN), a suffix in brackets kept, so
    opus[1m] is sent as claude-opus-5-5[1m]; any other name, an id or another of Claude Code's aliases, as it is.
    """
    name = str(name or "").strip()
    base = base_id(name)
    return BUILTIN[base] + name[len(base) :] if base in BUILTIN else name


def base_id(model: str) -> str:
    """A model's name or id without its suffix in brackets (Claude Code's context window): claude-opus-5-5[1m]'s is
    claude-opus-5-5."""
    return str(model or "").strip().partition("[")[0]


def family(model: str) -> str:
    """The Claude model family (FAMILIES) a model's name or id holds, ignoring case (a gateway reports its own
    spelling): claude-opus-5-5[1m]'s is opus, the name the Agent tool's `model` option takes for it. "" for a model of
    none."""
    m = str(model or "").lower()
    return next((f for f in FAMILIES if f in m), "")


def parts() -> tuple:
    """The file's sections: its top-level keys, but the header's."""
    keys = _file_type().load()["properties"]
    return tuple(k for k in keys if k not in settings_file.HEADER_KEYS)


# --- the schema's keys ------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _keys() -> dict:
    """Each key of the file (a map's entries as NAME.*) -> its schema, the first form's where forms share a key."""
    out = {}
    for path, node, _required in settings_file.schema_keys(_file_type().load()):
        out.setdefault(path, node)
    return out


@functools.lru_cache(maxsize=None)
def feeds() -> tuple:
    """(file key, configuration key) for each configuration key a key's `x-internal` names, in the schema's order."""
    out = []
    for path, node, _required in settings_file.schema_keys(_file_type().load()):
        internal = node.get("x-internal")
        for key in [internal] if isinstance(internal, str) else internal or ():
            if (path, key) not in out:
                out.append((path, key))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _fed() -> dict:
    """Each configuration key the file feeds -> the file keys that feed it, those NOT_CARRIED left out."""
    out = {}
    for path, internal in feeds():
        if path not in NOT_CARRIED:
            out.setdefault(internal, []).append(path)
    return {k: tuple(v) for k, v in out.items()}


def entries(map_key: str) -> dict:
    """The keys of each entry of the map MAP_KEY -> the configuration entry's keys they feed: see the module's
    docstring."""
    out = {}
    for path, internal in feeds():
        if path.startswith(map_key + ".*.") and ".*." in internal:
            out[path[len(map_key) + 3 :]] = internal.split(".*.", 1)[1]
    return out


def _common(paths) -> str:
    """The longest dotted prefix the keys share ("" for none)."""
    out = []
    for group in zip(*(p.split(".") for p in paths)):
        if len(set(group)) != 1:
            break
        out.append(group[0])
    return ".".join(out)


# --- the rules --------------------------------------------------------------------------------------------------------


class Rule(NamedTuple):
    """How one configuration key is made: READS, the file keys whose values make it (the section around them names
    it); VALUE, the function from the file's effective values to the key's value (MISSING: no value); NODE, the
    schema of its values, None when nothing says."""

    reads: tuple
    value: Callable
    node: dict | None = None


class _Joined(NamedTuple):
    """A rule the schema cannot say: VALUE, the function from the file's effective values and the file keys the rule
    reads, one argument each in the schema's order, to the key's value; NODE, the schema of its values; FED_FROM, the
    configuration keys whose feeding keys (`_fed`) the rule reads, when they are not its own key's.
    """

    value: Callable
    node: dict | None = None
    fed_from: tuple = ()


_BOOLEAN = {"type": "boolean"}


def _as_is(v, path: str):
    return copy.deepcopy(get(v, path))


def _any_on(v, *switches) -> bool:
    return any(get(v, k) is True for k in switches)


def image_parts(ref) -> tuple:
    """(name, tag) of the image REF: see the module's docstring."""
    text = str(ref or "").strip()
    if ":" not in text:
        return "", ""
    name, _, tag = text.rpartition(":")
    # a ":" before the last "/" is a registry's port: no tag
    return (text, "") if "/" in tag else (name, tag)


def _image_ref(v, name: str, tag: str) -> str:
    t = str(get(v, tag) or "").strip()
    n = str(get(v, name) or "").strip() or image.DEFAULT_NAME
    # A blank tag is container.name's (deployment.image_ref); the default name with it: no value at all.
    return f"{n}:{t}" if t else "" if n == image.DEFAULT_NAME else f"{n}:"


def _github_address(v, repository: str):
    repo = get(v, repository)
    return derived.github_address(repo) if isinstance(repo, str) and repo else repo


def _ramdisk_gb(v, size_gib: str):
    try:
        size = float(get(v, size_gib) or 0)
    except (TypeError, ValueError):
        size = 0.0
    return int(size) if size == int(size) else size


def _prices(v, switch: str, prices: str) -> dict:
    if get(v, switch) is not True:
        return {}
    names = entries(prices)
    return {
        model: {names[k]: p for k, p in fields.items() if k in names}
        for model, fields in (get(v, prices) or {}).items()
    }


def _incident_labels(v, label: str, extra_labels: str) -> list:
    first, extra = str(get(v, label) or "").strip(), list(get(v, extra_labels) or [])
    return [first] + extra if first or extra else []


def _requests_file(v, on: str, site: str) -> str:
    return derived.site_requests(get(v, site)) if get(v, on) is True else ""


def _worker_think(v, model: str, effort: str):
    m, e = get(v, model), get(v, effort)
    return e if m == WORKER and e in OTHER_EFFORTS else MISSING


# The rules the schema cannot say, by configuration key: see the module's docstring.
_JOINED = {
    "deployment.image": _Joined(_image_ref),
    "ramdisk.enabled": _Joined(lambda v, size: _ramdisk_gb(v, size) > 0, _BOOLEAN),
    "ramdisk.size_gb": _Joined(_ramdisk_gb),
    "features.merge_queue.auto_merge": _Joined(
        lambda v, on: "enabled" if get(v, on) is True else "hints_only",
        {"type": "string", "enum": ["enabled", "hints_only"]},
    ),
    "automerge_guard_own": _Joined(
        lambda v, own: get(v, own) == "same_as_others", _BOOLEAN
    ),
    "incident_issue.labels": _Joined(_incident_labels),
    "update.repo": _Joined(_github_address),
    "jira.component": _Joined(lambda v, parts: ", ".join(get(v, parts) or [])),
    "publish.html.requests": _Joined(
        _requests_file,
        {"type": "string"},
        ("publish.html.requests", "publish.html.site"),
    ),
    "atlassian.enabled": _Joined(
        _any_on, _BOOLEAN, ("features.ticket_creation.enabled",)
    ),
    "features.alerts.enabled": _Joined(_any_on, _BOOLEAN),
    "features.ticket_creation.enabled": _Joined(_any_on, _BOOLEAN),
    "prices": _Joined(_prices),
    "worker.think": _Joined(_worker_think, None, ("reviewer", "effort.first_pass")),
}


def _binding(entry: dict) -> dict:
    """A binding's configuration keys (config.json's model_bindings.<name>) from its effective values."""
    out = {}
    for key, internal in entries("model_bindings").items():
        value = get(entry, key)
        if value is MISSING or (internal == "auth_header" and not value):
            continue
        if internal == "provider":
            value = PROVIDER.get(value, value)
        elif internal == "auth_mode":
            value = AUTH_MODE.get(value, value)
        out[internal] = copy.deepcopy(value)
    return out


def _bindings(v) -> dict:
    mb = get(v, "model_bindings") or {}
    out = {n: _binding(b) for n, b in mb.items() if n != WORKER and isinstance(b, dict)}
    for path, node in _model_keys():
        m = get(v, path)
        if node.get("default") == DEFAULT_MODEL and m in BUILTIN and m not in mb:
            out[m] = {"provider": "claude", "model": m}
    return out


def _worker(v, internal: str):
    b = (get(v, "model_bindings") or {}).get(WORKER)
    if not isinstance(b, dict):
        return MISSING
    value = _binding(b).get(internal, MISSING)
    return WORKER_BACKEND.get(value, value) if internal == "provider" else value


def _optional_model(v, path: str) -> str:
    return str(get(v, path) or "").strip() or DEFAULT_MODEL


def _claude_model(v, path: str) -> str:
    m = str(get(v, path) or "").strip()
    b = (get(v, "model_bindings") or {}).get(m)
    if isinstance(b, dict) and b.get("provider") == "claude" and b.get("model_id"):
        return str(b["model_id"])
    return m


def _effort(v, path: str):
    e = get(v, path)
    return "" if e == DEFAULT_MODEL else e


def _joined(v, value: Callable, reads: tuple):
    return value(v, *reads)


@functools.lru_cache(maxsize=None)
def _model_keys() -> tuple:
    """(model key, its schema) for each pass: a key NAME.model the file feeds, beside its NAME.effort."""
    keys = _keys()
    return tuple(
        (path, node)
        for path, node in keys.items()
        if path.endswith(".model")
        and node.get("x-internal")
        and path[: -len("model")] + "effort" in keys
    )


def _feeding(path: str) -> str:
    """The one configuration key a file key feeds."""
    (internal,) = [i for p, i in feeds() if p == path]
    return internal


@functools.lru_cache(maxsize=None)
def rules() -> dict:
    """Configuration key -> its Rule: see the module's docstring. ValueError when the schema feeds a key from several
    file keys and no rule joins them; when a rule does not read every key that feeds its configuration key; or when a
    joined rule reads no key, or its function does not take the keys it reads.
    """
    fed, keys = _fed(), _keys()
    out = {}
    for internal, paths in fed.items():
        if len(paths) == 1 and "*" not in internal:
            out[internal] = Rule(
                paths, functools.partial(_as_is, path=paths[0]), keys[paths[0]]
            )
    for path, node in _model_keys():
        model = (
            _optional_model if node.get("default") == DEFAULT_MODEL else _claude_model
        )
        out[_feeding(path)] = Rule((path,), functools.partial(model, path=path))
        effort = path[: -len("model")] + "effort"
        out[_feeding(effort)] = Rule((effort,), functools.partial(_effort, path=effort))
    out["model_bindings"] = Rule(("model_bindings",), _bindings)
    by_worker_key, fields = {}, worker_fields()
    for key, internal in entries("model_bindings").items():
        worker_key = f"worker.{fields[internal]}"
        by_worker_key.setdefault(worker_key, ([], internal))[0].append(
            f"model_bindings.{WORKER}.{key}"
        )
    for worker_key, (reads, internal) in by_worker_key.items():
        out[worker_key] = Rule(
            tuple(reads), functools.partial(_worker, internal=internal)
        )
    for internal, joined in _JOINED.items():
        sources = joined.fed_from or (internal,)
        reads = tuple(path for key in sources for path in fed.get(key, ()))
        if not reads:
            raise ValueError(
                f"the rule for {internal} reads no key: nothing feeds {', '.join(sources)}"
            )
        try:
            inspect.signature(joined.value).bind(None, *reads)
        except TypeError:
            raise ValueError(
                f"the rule for {internal} does not take the keys it reads: {', '.join(reads)}"
            ) from None
        out[internal] = Rule(
            reads,
            functools.partial(_joined, value=joined.value, reads=reads),
            joined.node,
        )
    for internal, paths in fed.items():
        if "*" in internal:
            continue
        if internal not in out:
            raise ValueError(
                f"{internal} is fed by {', '.join(paths)}, and no rule joins them"
            )
        unread = [path for path in paths if path not in out[internal].reads]
        if unread:
            raise ValueError(
                f"{internal} is fed by {', '.join(unread)}, which its rule does not read"
            )
    return out


# --- the file -> the configuration ------------------------------------------------------------------------------------


def to_config(values: dict) -> dict:
    """The configuration a file's values make: see the module's docstring."""
    doc = settings_file.Document(_file_type(), _file_type().version, values)
    eff = settings_file.effective(doc)
    cfg = {}
    for internal, rule in rules().items():
        try:
            value = rule.value(eff)
        except (TypeError, ValueError, AttributeError):
            continue  # a value the file's checks refuse
        if value is not None and value is not MISSING:
            put(cfg, internal, value)
    return cfg


def _blank_of(node: dict):
    t = node.get("type")
    t = t[0] if isinstance(t, list) else t
    return {"string": "", "array": [], "object": {}, "boolean": False}.get(t)


def _default_values() -> dict:
    """Every key's value in a file that leaves nothing to fill: its default, else its template value when that holds
    no placeholder, else its type's blank."""
    out = {}
    for path, node in _keys().items():
        if "*" in path or "[]" in path or path in settings_file.HEADER_KEYS:
            continue
        if "default" in node:
            value = node["default"]
        else:
            try:
                value = settings_file.template_value(node, path)
            except ValueError:
                value = _blank_of(node)
            if settings_file.PLACEHOLDER.search(str(value)):
                value = _blank_of(node)
        put(out, path, copy.deepcopy(value))
    return out


@functools.lru_cache(maxsize=None)
def _internal_defaults() -> dict:
    cfg = to_config(_default_values())
    (cfg.get("atlassian") or {}).pop("enabled", None)
    return cfg


def internal_defaults() -> dict:
    """The configuration keys the file feeds, at the file's defaults: see the module's docstring."""
    return copy.deepcopy(_internal_defaults())


def read(path) -> tuple:
    """A deploy settings file -> (the configuration it makes, its FAIL and WARN rows, and check 7's PASS row): see the
    module's docstring."""
    from . import coherence  # which reads this module first

    try:
        doc = settings_file.read(path, _file_type())
    except settings_file.Refused as exc:
        if exc.reason != "header":
            raise
        from . import migrate  # which reads this module first, through file_types

        # a file in the layout before layout 1: migrate's row, named as the reader names its own
        row = migrate.before_layout_1(
            exc.first_key, settings_file.shown(path), exc.rows[0].name
        )
        if row is None:
            raise
        raise settings_file.Refused([row], "header", first_key=exc.first_key) from None
    rows = [
        r
        for r in settings_file.validate(doc)
        if r.status in (Status.FAIL, Status.WARN) or r.name == coherence.TEAM_ROW
    ]
    return to_config(doc.values), rows


# --- names ------------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _names() -> dict:
    """Configuration key -> its file key, a map's entries in both as NAME.*."""
    out = {internal: _common(rule.reads) for internal, rule in rules().items()}
    for internal, paths in _fed().items():
        if "*" in internal:
            out[internal] = _common(paths)
    return out


def held() -> dict:
    """Each configuration key the file feeds or a rule makes -> its file key."""
    return {k: v for k, v in _names().items() if "*" not in k}


def _entry(pattern: str, internal: str) -> str | None:
    """The map entry's name when INTERNAL is PATTERN with a name in place of its *, else None."""
    want, have = pattern.split("."), internal.split(".")
    if len(want) != len(have):
        return None
    entry = None
    for w, h in zip(want, have):
        if w == "*":
            entry = h
        elif w != h:
            return None
    return entry


def file_key(internal: str) -> str | None:
    """The file key that names a configuration key: see the module's docstring."""
    names = _names()
    if internal in names and "*" not in internal:
        return names[internal]
    for pattern, key in names.items():
        entry = _entry(pattern, internal) if "*" in pattern else None
        if entry is not None:
            return key.replace("*", entry, 1)
    for k, key in names.items():
        if "*" not in k and internal.startswith(k + "."):
            return key + internal[len(k) :]
    return None


def name(internal: str) -> str:
    """How messages, help texts and documents name a configuration key: its file key, else its own name."""
    return file_key(internal) or internal


def value_schema(internal: str) -> dict | None:
    """The schema of the values a configuration key takes from the file: see the module's docstring."""
    rule = rules().get(internal)
    return rule.node if rule else None


def node(key: str) -> tuple:
    """(the schema path, the schema node) of the file key KEY: a map entry's key is found under the map's NAME.*;
    ("", None) for a key the schema does not have."""
    keys = _keys()
    parts = key.split(".")
    for path in [key] + [
        ".".join(parts[:i] + ["*"] + parts[i + 1 :]) for i in range(len(parts))
    ]:
        if path in keys:
            return path, keys[path]
    found = schema.node(_file_type().load(), key)
    return (key, found) if found is not None else ("", None)


def description(key: str) -> str:
    """A file key's description in the schema: see the module's docstring."""
    path, found = node(key)
    return settings_file.description(found or {}, path or key)


def worker_fields() -> dict:
    """Each key the binding named `worker` carries: its name in config.json's bindings -> its name under `worker`."""
    return {
        internal: WORKER_FIELDS.get(internal, internal)
        for internal in entries("model_bindings").values()
    }


# --- the template of the earlier layout -------------------------------------------------------------------------------

# The parts of a file of the earlier layout beside its tree, which its template leaves out.
OTHER_PARTS = (
    "secrets",
    "setup",
    "exported_at",
    "package_version",
    "source",
)


def deploy_schema() -> dict:
    return schema.load("deploy")


def schema_line(package_dir=None) -> str:
    """The first line of the layout 1 template (`template`, the tests'): the schema editors read
    (yaml-language-server), by its absolute path in the package folder."""
    return f"# yaml-language-server: $schema={schema.path('deploy', package_dir)}"


HEADER_LINES = (
    "rocKE monitor deploy settings (QUICK_DEPLOY_GUIDE.md). "
    "Keep this file outside the package folder and outside git, readable by you only (chmod 600 FILE). No secret, "
    "token or key goes here: rkmon login and rkmon secret set them. Fill in each <...>; a key you leave out takes "
    "its default. Then: rkmon confirm --settings FILE.",
)


def _head(header, package_dir) -> list:
    import textwrap

    out = [schema_line(package_dir)]
    for h in header:
        out += [
            "# " + x if x else "#" for x in (textwrap.wrap(h, schema.WIDTH - 2) or [""])
        ]
    return out + schema.header_keys(deploy_schema())


def template(version: str = "", package_dir=None, values: dict | None = None) -> str:
    """The deploy settings template in layout 1, the tests': the guides' samples they check. The schema line, the
    header, then every tree key in the schema's order with its template value (`values`
    over them) and its description, model_bindings with its examples, and `advanced`. The templates the package gives
    you are layout 2's (rkmon_lib/settings_templates.py).
    """
    vals = {"container.image.tag": version or "local", **(values or {})}
    return schema.template(
        deploy_schema(),
        _head(HEADER_LINES, package_dir),
        vals,
        skip=("kind", "schema_version", SECRET_VALUES, *OTHER_PARTS),
    )
