# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The features and what they run: the deploy settings' features (config.json `features`, which the deploy settings
file's feed: rkmon_lib/deploy_tree.py; review automation, the merge queue, the dashboards, the alerts and ticket
creation, each of the five that send (SENDS) in mode prototype or live) decide every capability the code acts on
(CAPABILITIES), one rule each.

- A feature's switches: `switch(cfg, "review_automation.mode")`; a key the configuration leaves out takes the file's
  default (schema/deploy_settings.schema.json, through deploy_tree.internal_defaults).
- prototype, live and standby: a capability's rule reads its switches, never a mode, so a feature in
  prototype does all its work (the dashboards and the files show it); what it would send stops at the one outward gate
  (rkmon_lib/post.py outward_gate), which keeps it as an unsent item. In live it sends: review automation posts the
  reviews and overlap notices; the merge queue arms auto-merge (auto_merge enabled), holds (auto_hold), posts queue
  notices (queue_notices) and drift notices (drift_notices), and, with develop_guard, files develop issues and incidents
  and links their fixes; the alerts go to the team chat; the team dashboard's page is pushed to the web host (served
  here, it is served in either mode, and in prototype on this host's own address alone: `team_host`); ticket creation
  files the tickets. Standby is the whole monitor's (`standby`): while it is on, no feature sends, whatever its mode. A
  capability's `feature` is the feature it sends for (SENDS); `held(cfg, feature)` says why that feature's sends do not
  go out now (`held_words`, in words; for a feature that is not live, `prototype_words`), and `sends` that they do;
  `as_live(cfg, feature)` is the settings with the feature live, to read what going live needs
  (secrets_registry.waits_for). Review posting alone is on only while its feature sends (`sending_only`): while it is
  off, each verified review is kept unposted (post.keeps).
- A capability that is off never acts and never fails: the tick does not print its action lines, its commands refuse
  with a plain message naming the setting that decides it and exit 0 (EXIT_OFF), and `rk doctor` shows one "off"
  line. One that is on but lacks what it needs (a setting, a secret, a connection) is skipped: the tick reports it once
  (until what is missing changes), `rk features` and `rk doctor` say exactly what is missing and the command that
  fixes it, and its commands refuse with exit 5 (EXIT_NOT_SET_UP).
- `coherence`: the combinations that cannot run (FAIL, with the fix) or that do less than they seem to (WARN):
  `rkmon confirm` shows them all, and config.problems refuses the structural ones.
- `components`: which of the deployment's parts start (the web server, the Claude session, the review driver, ticks run
  by fixed rules, the merge queue, the develop guard); `rk features --runs PART` answers with its exit code
  (docker/entrypoint.sh).

`summary` is the explicit account `rk features` prints: each feature's switches, then every capability, its state,
what it is configured with (secrets only as "set"), and what is missing.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from . import container_layout

EXIT_OFF, EXIT_NOT_SET_UP = 0, 5
FEATURE_NAMES = (
    "review_automation",
    "merge_queue",
    "dashboards",
    "alerts",
    "ticket_creation",
)
PREFIX = "features."  # the configuration keys of the switches
# The features that send, each in its words: what a capability sends for (Capability.feature)
SENDS = {
    "review_automation": "review automation",
    "merge_queue": "the merge queue",
    "alerts": "alerts",
    "dashboards.team": "the team dashboard",
    "ticket_creation": "ticket creation",
}


def _defaults() -> dict:
    from . import deploy_tree

    return deploy_tree.internal_defaults()["features"]


def switch(cfg: dict, dotted: str):
    """A feature's switch (config `features.<dotted>`), else the schema's default."""
    from .config import lookup

    feats = (
        (cfg or {}).get("features")
        if isinstance((cfg or {}).get("features"), dict)
        else {}
    )
    v = lookup(feats, dotted)
    return lookup(_defaults(), dotted) if v is None else v


def enabled(cfg: dict, feature: str) -> bool:
    """A feature on: review_automation, merge_queue, alerts, ticket_creation, dashboards.admin, dashboards.team."""
    return switch(cfg, f"{feature}.enabled") is True


def live(cfg: dict, feature: str) -> bool:
    """A feature that sends (SENDS) on and in mode live, by the settings: it sends, unless the monitor is in standby
    (`sends`)."""
    return enabled(cfg, feature) and switch(cfg, f"{feature}.mode") == "live"


def mode(cfg: dict, feature: str) -> str:
    return str(switch(cfg, f"{feature}.mode") or "prototype")


def team_host(cfg: dict, address: str) -> str:
    """The host address the container publishes the team dashboard's port on, for the settings CFG (the monitor's
    configuration, or the deploy settings' values: each holds `features`) and the address they give it, ADDRESS
    (serve.address; the monitor's team_dashboard.publish_host): in prototype (`mode`), this host's own,
    container_layout.LOOPBACK, whatever ADDRESS says (check 7); else ADDRESS. docker/.env's
    RKMON_TEAM_HOST (deploy_plan.env_values), check 7's words (coherence) and the container's own check of docker/.env
    (webserve.publish_env) each read it here."""
    if mode(cfg, "dashboards.team") == "prototype":
        return container_layout.LOOPBACK
    return address


# The internal state's file -> (what its stat said when it was read, standby.on)
_STANDBY_READ: dict = {}


def standby(home=None) -> bool:
    """Whether the whole monitor is in standby: `standby.on` in the internal state of the monitor's home
    HOME (config.home() by default), read through rkmon_lib/state_store.py again whenever the file changed, so that a
    process that runs for long sees standby begin and end. An internal state that cannot be read counts as standby:
    nothing goes out on a state the monitor cannot read. So does a home whose one-time move is pending
    (one_time_move.pending: your settings delivered beside the config.json of an older install, until the move ends),
    and one that holds the mark of an adoption (adopt.marked: from before rkmon adopt moves anything until
    `rkmon standby off`), whatever its internal state says, and with none: nothing goes out before the move is made,
    or when it fails, nor before standby off ends an adoption, finished or not.
    """
    from . import adopt, config, one_time_move, state_store

    home = config.home() if home is None else home
    if one_time_move.pending(home) or adopt.marked(home):
        return True
    p = state_store.path(home)
    try:
        s = os.stat(p)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    seen = (s.st_ino, s.st_mtime_ns, s.st_size)
    last = _STANDBY_READ.get(str(p))
    if last and last[0] == seen:
        return last[1]
    try:
        on_ = (state_store.read(home).get("standby") or {}).get("on") is True
    except state_store.StateError:
        on_ = True
    _STANDBY_READ[str(p)] = (seen, on_)
    return on_


def held(cfg: dict, feature: str) -> str:
    """Why a send of FEATURE (review_automation, merge_queue, alerts, dashboards.team, ticket_creation) does not go out
    now, in state_store's words: "standby" while the monitor is in standby, whatever the modes say; "prototype" while
    the feature is not live; "" when it goes out."""
    from . import state_store

    if standby():
        return state_store.STANDBY
    return "" if live(cfg, feature) else state_store.PROTOTYPE


def sends(cfg: dict, feature: str) -> bool:
    """FEATURE's sends go out now: it is live, and the monitor is not in standby."""
    return not held(cfg, feature)


def held_words(cfg: dict, feature: str) -> str:
    """Why FEATURE's sends do not go out now (`held`), in words that name the mode or standby and what ends it; "" when
    they go out."""
    from . import state_store, steps

    why = held(cfg, feature)
    if why == state_store.STANDBY:
        return (
            "the monitor is in standby: nothing goes out until standby ends "
            f"({steps.command_line('standby off')})"
        )
    return prototype_words(feature) if why else ""


def prototype_words(feature: str) -> str:
    """FEATURE in prototype, in words that name its mode and what ends it: why its sends do not go out (`held_words`),
    and what a login or secret its sends need waits for (secrets_registry.waiting_words).
    """
    from . import deploy_tree

    key = deploy_tree.name(f"{PREFIX}{feature}.mode")
    return f"{key} is prototype: nothing goes out until it is live"


def as_live(cfg: dict, feature: str) -> dict:
    """CFG with FEATURE in mode live (`features.<feature>.mode`) and every other setting as it is, as a copy: CFG is
    not changed. What going live needs reads it (secrets_registry.waits_for)."""
    feats = (cfg or {}).get("features")
    feats = dict(feats) if isinstance(feats, dict) else {}
    node = feats
    *parents, last = feature.split(".")
    for part in parents:
        node[part] = dict(node[part]) if isinstance(node.get(part), dict) else {}
        node = node[part]
    node[last] = {
        **(node[last] if isinstance(node.get(last), dict) else {}),
        "mode": "live",
    }
    return {**(cfg or {}), "features": feats}


def _in_container() -> bool:
    return os.environ.get("RKMON_CONTAINER") == "1"


def _advanced_on(cfg: dict, dotted: str) -> bool:
    from .config import lookup

    return lookup(cfg or {}, dotted) is True


# --- what each capability needs ---------------------------------------------------------------------------------------


class Need(str):
    """One thing a capability lacks, in words (a str: missing() returns them), and what sets it up (fix): a
    configuration key the operator sets (`Need.setting`), a login or secret its own command sets (`Need.secret`), a
    part of the image, which no setting gives (`Need.image_part`), or what another fix sets up (`Need.fixed_by`).
    Plain words carry none.
    """

    key = value = file_key = lacks = ""
    entry = given = None

    @classmethod
    def setting(cls, internal: str, what: str, key: str = "") -> Need:
        """The configuration key INTERNAL, named as the deploy settings file names it (KEY where a key of each
        provider's binding feeds INTERNAL, else deploy_tree.name), whose value is WHAT.
        """
        from .deploy_tree import name

        need = cls(f"{key or name(internal)} ({what})")
        need.key, need.value, need.file_key = internal, what, key
        return need

    @classmethod
    def secret(cls, entry, what: str, given=None) -> Need:
        """The login or secret ENTRY (rkmon_lib/secrets_registry.py), whose value is WHAT: WHAT and how to set it.
        GIVEN, ENVIRON -> a checks fix, sets it where its own command alone does not say how (the atlassian CLI's
        login, a device flow: atlassian.login_fix)."""
        from . import secrets_registry as sr

        need = cls(f"{what} ({sr.how_to_set(entry)})")
        need.entry, need.given = entry, given
        return need

    @classmethod
    def image_part(cls, what: str) -> Need:
        """WHAT (in words), which the container's configuration takes from the image (rkmon_lib/derived.py)."""
        need = cls(f"{what} (from the image)")
        need.lacks = what
        return need

    @classmethod
    def fixed_by(cls, words: str, given) -> Need:
        """WORDS, which GIVEN, ENVIRON -> a checks fix, sets up: a host's own key, recorded by the logins' walk
        (host_keys.record_fix); a tool every image of this package carries, by running one (doctor.image_fix).
        """
        need = cls(words)
        need.given = given
        return need

    def fix_parts(self, environ=os.environ) -> tuple:
        """Its fix's parts, as the operator types them on the host: its own fix (GIVEN's), the key's edit in the
        deploy settings file (doctor.settings_parts), the entry's own command (secrets_registry.set_fix), or the
        check of the image that lacks the part (doctor.image_lacks_fix); for plain words, the checks run again.
        """
        from . import doctor, secrets_registry as sr
        from .checks import Described, Steps

        if self.given is not None:
            fix = self.given(environ)
        elif self.entry is not None:
            fix = sr.set_fix(self.entry, environ)
        elif self.lacks:
            fix = doctor.image_lacks_fix(self.lacks, environ)
        elif self.key:
            return doctor.settings_parts(
                self.key, Described(self.value), environ, key=self.file_key
            )
        else:
            fix = doctor.check_again(environ)
        return fix.parts if isinstance(fix, Steps) else (fix,)


def _secret_found(cfg: dict, name: str) -> dict:
    """Whether the secret NAME is set where this configuration keeps it, and where (logins.found: in the container
    the writer answers for the secrets it keeps)."""
    from . import logins, secrets_registry as sr

    return logins.found(cfg, sr.entry(cfg, name))


def _needed(cfg: dict, name: str) -> bool:
    """Whether something that is on needs the secret or login NAME, by the settings (secrets_registry.needed_by): one
    that a capability needs for its sends, only while the capability's feature is live.
    """
    from . import secrets_registry as sr

    entry = sr.entry(cfg, name)
    return bool(entry and sr.needed_by(cfg, entry))


def _secret_need(cfg: dict, name: str, what: str) -> list[str]:
    """[] when nothing that is on needs the secret NAME (`_needed`), or it is set (`_secret_found`); else WHAT and how
    to set it (`Need.secret`), or why it is not known."""
    from . import secrets_registry as sr

    if not _needed(cfg, name):
        return []
    got = _secret_found(cfg, name)
    if got["set"]:
        return []
    if got["set"] is None:
        return [f"{what} (not known whether it is set: {got['where']})"]
    return [Need.secret(sr.entry(cfg, name), what)]


def _secret_word(cfg: dict, name: str) -> str:
    """The secret NAME as `configured_with` shows it: set, missing, or not known and why (`_secret_found`)."""
    from . import logins

    got = _secret_found(cfg, name)
    return logins.said(got) + ("" if got["set"] is not None else f" ({got['where']})")


def _gh(cfg: dict) -> list[str]:
    """What the monitor's writes on GitHub need: the login it posts as (a setting), and the gh CLI the toolkit resolves
    (paths.gh, else gh on PATH), the image's (its logins: rk doctor checks them).
    """
    import shutil
    from . import config, gh

    out = (
        []
        if config.operator_login(cfg)
        else [
            Need.setting(
                "operator.github_login", "the GitHub login the monitor posts as"
            )
        ]
    )
    binary = gh.settings(cfg)[0]
    if not (shutil.which(binary) or os.path.isfile(binary)):
        out.append(Need.image_part("the gh CLI"))
    return out


def _claude(cfg: dict) -> list[str]:
    """The Claude session, in the container: Claude Code and the session's key, the claude login (rk orchestrate keeps
    the session; the restart policy brings it back). Its Claude workspace is the container's own
    (derived.claude_workspace), never a setting.
    """
    import shutil

    out = (
        []
        if shutil.which("claude")
        else ["Claude Code installed on this machine (claude on PATH), logged in"]
    )
    from . import orchestrator

    return out + _secret_need(cfg, orchestrator.LOGIN, "the Claude session's key")


def _jira_project(cfg: dict) -> list[str]:
    """Ticket creation's own settings: the JIRA project (rkmon confirm reads it before any CLI is here), a setting."""
    if ((cfg.get("jira") or {}).get("project") or "").strip():
        return []
    return [Need.setting("jira.project", "the JIRA project new tickets go to")]


def _jira(cfg: dict) -> list[str]:
    """Ticket creation: the JIRA project, and the atlassian CLI it files through, here, and logged in while its login
    is needed (`_needed`: ticket creation is live)."""
    from . import atlassian

    return _jira_project(cfg) + atlassian.missing(cfg, login=_needed(cfg, "atlassian"))


def _chat(cfg: dict) -> list[str]:
    """The team chat: its webhook, which decides the channel; the message label may be blank."""
    return _secret_need(cfg, "teams_webhook", "the channel's incoming-webhook URL")


def _team_dashboard_settings(cfg: dict) -> list[str]:
    """The team dashboard's settings: served here, none; published to the web host, the host, the login and the page's
    place there, publish.html.dest, which the site folder gives (derived.site_page): the site folder is named when
    neither is set, since rkmon confirm reads the settings before any derived key is here.
    """
    from . import derived, publish

    if publish.mode(cfg) == "serve":
        return []
    p = (cfg.get("publish") or {}).get("html") or {}
    page = p.get("dest") or derived.site_page(p.get("site"))
    return [f"publish.html.{k}" for k in ("host", "user") if not p.get(k)] + (
        [] if page else ["publish.html.site"]
    )


# The team dashboard's publish settings (_team_dashboard_settings), each in words.
_PUBLISH_WORDS = {
    "publish.html.host": "the team web host",
    "publish.html.user": "your login on the web host",
    "publish.html.site": "the folder the web host serves the page from",
}


def _team_dashboard(cfg: dict) -> list[str]:
    """What the team dashboard needs: its settings, and published to the web host, the key while it is needed
    (`_secret_need`: the team dashboard is live)."""
    from . import publish

    out = [Need.setting(k, _PUBLISH_WORDS[k]) for k in _team_dashboard_settings(cfg)]
    if publish.mode(cfg) == "serve":
        return out
    return out + _secret_need(cfg, "web_host_ssh_key", "the web host's ssh key")


def _model(cfg: dict) -> list[str]:
    """The first pass's model only (features.review_automation.passes.first_review_pass.model): nothing for the
    default (the verifier does it); a binding's address, model and key (never a value); the local worker's
    (triage.worker_gaps: its endpoint and model, and a model the start's checks did not refuse for the first pass).
    """
    from . import bindings, config, triage

    if bindings.no_draft(cfg):
        return []
    r = bindings.reviewer(cfg)
    if r["kind"] == "missing":
        return [
            Need.setting(
                "reviewer", f"a model model_bindings defines; {r['choice']} is not one"
            )
        ]
    if r["kind"] == "binding":
        gap = triage.binding_gap(bindings.client_settings(cfg, "drafter"))
        return [gap] if gap else []
    return triage.worker_gaps(config.worker(cfg))


def _gpu(cfg: dict) -> list[str]:
    """Remote GPU tests: the cluster's values (config `gpu`), the SSH key and the ssh client (rkmon_lib/gpu.py)."""
    from . import gpu

    return gpu.missing(cfg)


def _admin(cfg: dict) -> list[str]:
    """The access token, which every request carries (webserve.access_refusal: the service does not start without
    it). The page is published on the host's loopback alone (deploy_plan.env_values), so no address is needed.
    """
    from . import secrets_registry as sr, webserve

    if webserve.access_token(cfg):
        return []
    return [
        Need.secret(
            sr.entry(cfg, webserve.ACCESS_SECRET),
            "the access token, which every request carries",
        )
    ]


def _checks(cfg: dict) -> list[str]:
    """What the develop guard lacks of the checks venv and ROCm's LLVM tools (paths.rocm_llvm_bin): both are the
    image's (rkmon_lib/derived.py), so what it lacks is one part of the image (Need.image_part), and its fix the image's
    check.
    """
    from . import doctor, paths

    p = paths.resolve(cfg)
    venv = bool(p["venv"]) and (pathlib.Path(p["venv"]) / "bin" / "python").exists()
    llvm = bool(p["rocm_llvm_bin"])
    lacks = [
        what
        for what, there in (("the checks venv", venv), (doctor.LLVM_TOOLS, llvm))
        if not there
    ]
    return [Need.image_part(" and ".join(lacks))] if lacks else []


# --- the capabilities -------------------------------------------------------------------------------------------------


def _ra(c):
    return enabled(c, "review_automation")


def _ra_live(c):
    return live(c, "review_automation")


def _mq(c):
    return enabled(c, "merge_queue")


def _mq_live(c):
    return live(c, "merge_queue")


def _guard(c):
    return _mq(c) and switch(c, "merge_queue.develop_guard") is True


def _tickets(trigger: str):
    return (
        lambda c: enabled(c, "ticket_creation")
        and switch(c, f"ticket_creation.triggers.{trigger}") is True
    )


def _alert(kind: str):
    return lambda c: enabled(c, "alerts") and switch(c, f"alerts.{kind}") is True


# The names a title may start with, which a sentence keeps as they are (`lowered`).
NAMES = ("Claude",)


def lowered(title: str) -> str:
    """A capability's title inside a sentence: its first word lowercased, unless it is an acronym (GPU, JIRA) or a
    name (NAMES)."""
    first = title.split(" ", 1)[0]
    keep = (first.isupper() and len(first) > 1) or first in NAMES
    return title if keep else title[:1].lower() + title[1:]


# A capability's value in words, where the value alone does not say what the capability does then
# (secrets_registry.condition_words, the logins' needs): (capability, value) -> words.
VALUE_WORDS = {("auto-merge-guard", "act"): "turning auto-merge off"}


def value_words(name: str, value: str) -> str:
    """The value VALUE of capability NAME in words (VALUE_WORDS), else VALUE as it is."""
    return VALUE_WORDS.get((name, value), value)


@dataclass(frozen=True)
class Capability:
    """One thing the monitor does, and the switches that decide it: `rule` (cfg -> bool), `when` (the switches it
    runs under, in words, as the deploy settings name them)."""

    name: str
    title: str
    what: str  # two plain sentences
    needs: str  # what it needs, in words
    rule: object
    when: str
    check: str = ""  # the name of this module's function cfg -> list of what is missing
    # The section of the deploy settings file that holds its settings, by its name in doctor.SECTION_SETTINGS: one
    # edit of it sets every setting it lacks (needs_fix). "" for none: each setting it lacks is edited on its own.
    section: str = ""
    values: tuple = (False, True)
    feature: str = ""  # the feature it sends for (SENDS); "" for one that sends nothing
    sending_only: bool = False  # on only while its feature sends (`sends`)


CAPABILITIES = {
    c.name: c
    for c in (
        Capability(
            "review-drafts",
            "Reviews",
            "Each new or changed PR in scope is reviewed: the first review pass (features.review_automation.passes."
            "first_review_pass), then the verification pass, which checks every claim (features.review_automation."
            "passes.verification_pass), several reviews at once (features.review_automation.parallel_reviews at "
            "first, then adjusted to how busy this host is). The result is kept on this machine.",
            "nothing with no first-pass drafter (the default: the verifier reviews alone); a model's endpoint and key "
            "when a model drafts",
            _ra,
            "features.review_automation.enabled is true",
            check="_model",
            section="reviewer",
        ),
        Capability(
            "review-posting",
            "Review posting",
            "Each verified review is posted on its PR as one comment, by the GitHub login the monitor posts as. In "
            "prototype, each stays on this machine as an unposted review.",
            "integrations.github_cli.username, and gh rights to comment",
            _ra,
            "features.review_automation is enabled and live",
            check="_gh",
            feature="review_automation",
            sending_only=True,
        ),
        Capability(
            "overlap-management",
            "Overlap management",
            "Duplicate or conflicting work across open PRs is found and one owner is recorded for each shared piece. "
            "While review automation is live, the PRs that should drop their copy get an overlap notice.",
            "comment rights (live)",
            _ra,
            "features.review_automation.enabled is true (its notices go out while it is live)",
            check="_gh",
            feature="review_automation",
        ),
        Capability(
            "remote-gpu-tests",
            "Remote GPU tests",
            "The GPU-only test cases of each review run on a GPU node of a Slurm cluster over SSH: queued, never "
            "holding the review, whose GPU tests row is updated in place when they finish. Without it, GPU-only cases "
            "skip here.",
            "the cluster's settings (features.review_automation.remote_gpu_tests.cluster) and the SSH key for its "
            "login host (a secret)",
            lambda c: _ra(c) and _advanced_on(c, "gpu.enabled"),
            "features.review_automation is enabled and features.review_automation.remote_gpu_tests.enabled is true",
            check="_gpu",
            section="gpu",
        ),
        Capability(
            "merge-plan",
            "Merge plan",
            "The landing order, the gates and the merge queue view, computed every tick. It only reads GitHub.",
            "the repository and the monitor's clone",
            _mq,
            "features.merge_queue.enabled is true",
        ),
        Capability(
            "priority-requests",
            "Priority requests",
            "Priority requests are taken in at all times: from the admin dashboard, from rkmon rk prioritize, and "
            "from the team dashboard's request box. You approve or decline each one.",
            "nothing",
            _mq,
            "features.merge_queue.enabled is true",
        ),
        Capability(
            "auto-merge",
            "Auto-merge",
            "A PR you cleared (rkmon rk clear N) gets GitHub's own auto-merge once it is validated, so GitHub "
            "merges it only when every gate it enforces passes. It never merges directly, never uses admin rights, "
            "never forces.",
            "your login's merge rights",
            lambda c: _mq(c) and switch(c, "merge_queue.auto_merge") == "enabled",
            "features.merge_queue is enabled and its auto_merge.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "auto-merge-guard",
            "Auto-merge guard",
            "When someone else's auto-merge is on and one of our gates fails, the auto-merge guard alerts you "
            "(prototype) or comments in plain words and turns that auto-merge off (live). It never turns off an "
            "auto-merge you turned on yourself.",
            "comment and merge rights (live)",
            _mq,
            "features.merge_queue.enabled is true (it comments and turns auto-merge off while the merge queue is "
            "live)",
            check="_gh",
            values=("off", "alert", "act"),
            feature="merge_queue",
        ),
        Capability(
            "holds",
            "Auto-hold",
            "A PR whose gate failure is confirmed as its own is held, and a change request from your account keeps it "
            "from merging until it is fixed and you re-approve it.",
            "review rights",
            lambda c: _mq(c) and switch(c, "merge_queue.auto_hold") is True,
            "features.merge_queue is enabled and its auto_hold.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "queue-notices",
            "Queue notices",
            "PRs near the front of the merge queue get one status comment, kept current, saying where they stand.",
            "comment rights",
            lambda c: _mq(c) and switch(c, "merge_queue.queue_notices") is True,
            "features.merge_queue is enabled and its queue_notices.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "drift-notices",
            "Drift notices",
            "When today's develop breaks a PR's own tests, its author gets a notice naming the develop commit that "
            "changed the behaviour.",
            "comment rights",
            lambda c: _mq(c) and switch(c, "merge_queue.drift_notices") is True,
            "features.merge_queue is enabled and its drift_notices.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "develop-checks",
            "Develop guard",
            "After every merge into develop, develop is checked (native build, goldens, byte identity, ABI), and a "
            "break is traced to its PR.",
            "the checks venv and ROCm's LLVM tools",
            _guard,
            "features.merge_queue is enabled and its develop_guard.enabled is true",
            check="_checks",
        ),
        Capability(
            "stability-issues",
            "Develop issues",
            "A develop break that CI does not show (a test no CI lane runs) is filed as a GitHub issue and assigned to "
            "the breaking PR's author.",
            "gh rights to open issues",
            _guard,
            "features.merge_queue is enabled and its develop_guard.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "incident-issues",
            "Incidents",
            "A CI-red develop break is filed as a GitHub issue, and its incident runs: culprit, revert offer, fix "
            "tracking.",
            "gh rights to open issues and comment",
            _guard,
            "features.merge_queue is enabled and its develop_guard.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "fix-links",
            "Fix links",
            "An open PR that fixes a develop issue is linked to it once verified, with a comment, and goes to the front "
            "of the queues.",
            "gh rights to comment",
            _guard,
            "features.merge_queue is enabled and its develop_guard.enabled is true",
            check="_gh",
            feature="merge_queue",
        ),
        Capability(
            "admin-dashboard",
            "Admin dashboard",
            'Your page, served by the monitor (where it listens and who can reach it: README.md, "The admin '
            'dashboard"). It is the one place to approve or decline priority requests, the tickets waiting for '
            "you and the in-depth reviews of large PRs.",
            "the access token (a secret), which every request carries",
            lambda c: enabled(c, "dashboards.admin"),
            "features.dashboards.admin.enabled is true",
            check="_admin",
        ),
        Capability(
            "team-dashboard",
            "Team dashboard",
            "The dashboard for other teams, with its request box and no admin controls: published on a team web host, "
            "or served by this monitor on a port of its own (features.dashboards.team.delivery).",
            "publish: its public_url and its publish block (web_host, username, site_folder) and, in live, the web "
            "host's SSH key (a secret); serve: its serve block (address and port)",
            lambda c: enabled(c, "dashboards.team"),
            "features.dashboards.team.enabled is true",
            check="_team_dashboard",
            section="publish",
            values=("off", "publish", "serve"),
            feature="dashboards.team",
        ),
        Capability(
            "chat-alerts",
            "Develop break alerts",
            "When develop's CI is red and no fix is planned, the team chat channel gets a short message.",
            "in live, the channel's webhook URL (a secret)",
            _alert("develop_breaks"),
            "features.alerts.develop_breaks.enabled is true",
            check="_chat",
            feature="alerts",
        ),
        Capability(
            "needs-you-alerts",
            "Needs-you alerts",
            "The decisions on the dashboard's Needs-you list (a merge to propose, a stale hold, a priority request, a "
            "PR ready for your approval, an in-depth review to approve, and the like) go once each to the team chat "
            "channel.",
            "in live, the channel's webhook URL (a secret)",
            _alert("needs_you"),
            "features.alerts.needs_you.enabled is true",
            check="_chat",
            feature="alerts",
        ),
        Capability(
            "follow-up-tickets",
            "Follow-up tickets",
            "When a PR merges with our significant review items still open, each gets a follow-up ticket for the PR's "
            "author.",
            "features.ticket_creation.project_key and the atlassian CLI (integrations.atlassian_cli.jira_url; in "
            "live, logged in)",
            _tickets("review_follow_up"),
            "features.ticket_creation.follow_up_tickets.enabled is true",
            check="_jira",
            section="jira",
            feature="ticket_creation",
        ),
        Capability(
            "found-issue-tickets",
            "Out-of-scope issue tickets",
            "An issue found outside any PR's review gets a proposed ticket, which waits on the admin dashboard's "
            "Tickets card: your Approve files it, your Decline drops it.",
            "features.ticket_creation.project_key and the atlassian CLI (integrations.atlassian_cli.jira_url; in "
            "live, logged in)",
            _tickets("out_of_scope_issue"),
            "features.ticket_creation.found_issue_tickets.enabled is true",
            check="_jira",
            section="jira",
            feature="ticket_creation",
        ),
        Capability(
            "weekly-report",
            "Weekly report",
            "Each week the monitor writes a statistics report and the raw material for a lessons-learned review.",
            "nothing",
            lambda c: _advanced_on(c, "weekly_report.enabled"),
            "features.weekly_report.enabled is true",
        ),
        Capability(
            "headless-claude",
            "Claude session",
            "Claude's part of each tick runs unattended: in the container the orchestrator's long-lived session "
            "(rk orchestrate), which runs while review automation is enabled.",
            "Claude Code and its login (the claude login, the session's key)",
            lambda c: _in_container() and _ra(c),
            "features.review_automation.enabled is true (in the container)",
            check="_claude",
        ),
    )
}
# What a capability's tick lines start with: not printed while it is off or not set up.
LINES = {
    "QUEUE_NOTICE": "queue-notices",
    "AUTOMERGE_RISK": "auto-merge-guard",
    "!! AUTOMERGE_RISK": "auto-merge-guard",
    "AUTOMERGE_JUMP": "auto-merge-guard",
    "!! AUTOMERGE_JUMP": "auto-merge-guard",
    "ARM_READY": "auto-merge",
    "TRIAGE": "holds",
    "HOLD_STALE": "holds",
    "!! HOLD_REVIEW_NEEDED": "holds",
    "HOLD_REVIEW_NEEDED": "holds",
    "HOLD_FIXED": "holds",
    "HOLD_REAPPROVED": "holds",
    "!! HOLD_DISMISSED": "holds",
    "HOLD_DISMISSED": "holds",
    "DRIFT_FAIL": "drift-notices",
    "FIX_CANDIDATE": "fix-links",
    "OVERLAP_NOTICE_DUE": "overlap-management",
    "WEEKLY REPORT": "weekly-report",
    "LESSONS REVIEW": "weekly-report",
    "REVERT_OFFER": "incident-issues",
    "!! CRITICAL REVERT_OFFER": "incident-issues",
    "GPU_": "remote-gpu-tests",
    "(gpu": "remote-gpu-tests",
    "FOLLOWUP_SCAN": "follow-up-tickets",
    "!! FOLLOWUP_SCAN": "follow-up-tickets",
    "(followup scan": "follow-up-tickets",
}


def value(cfg: dict, name: str):
    """A capability's value: on or off (True, False), one that is `sending_only` off while its feature does not send;
    the auto-merge guard's off, alert (the merge queue in prototype, or the monitor in standby) or act (the merge queue
    sends); the team dashboard's off, or where it goes: publish (a team web host) or serve (this monitor).
    """
    return _value(cfg, name, lambda feature: sends(cfg, feature))


def configured(cfg: dict, name: str):
    """A capability's value by the settings alone, as `value` gives it with no standby: each feature that sends taken
    as sending while it is live (what it needs, secrets_registry's needed_when: standby passes, the needs stay).
    """
    return _value(cfg, name, lambda feature: live(cfg, feature))


def _value(cfg: dict, name: str, sending) -> object:
    c = CAPABILITIES[name]
    on_ = bool(c.rule(cfg)) and (not c.sending_only or sending(c.feature))
    if name == "auto-merge-guard":
        return "off" if not on_ else ("act" if sending(c.feature) else "alert")
    if name == "team-dashboard":
        from . import publish

        return publish.mode(cfg) if on_ else "off"
    return on_


def on(cfg: dict, name: str) -> bool:
    return value(cfg, name) not in (False, "off", None)


def why_off(cfg: dict, name: str) -> str:
    """The setting that keeps a capability off, in the deploy settings' words, or for one that is `sending_only` and
    on by its switches, the mode or standby (held_words); "" when it is on."""
    if on(cfg, name):
        return ""
    c = CAPABILITIES[name]
    if c.sending_only and c.rule(cfg):
        return held_words(cfg, c.feature)
    return f"it runs while {c.when}"


def missing(cfg: dict, name: str) -> list[str]:
    """What an enabled capability lacks (always empty for one that is off)."""
    c = CAPABILITIES[name]
    if not on(cfg, name) or not c.check:
        return []
    if name == "auto-merge-guard" and configured(cfg, name) != "act":
        return []
    try:
        return list(globals()[c.check](cfg))
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a check that cannot run is a missing dependency, never a traceback
        return [f"could not check: {exc}"]


def active(cfg: dict, name: str) -> bool:
    """On, and nothing it needs is missing."""
    return on(cfg, name) and not missing(cfg, name)


def fix(cfg: dict, name: str, environ=os.environ):
    """What sets up what capability NAME lacks (missing), as the operator types it on the host (a checks fix):
    needs_fix's."""
    return needs_fix(missing(cfg, name), name, environ)


def needs_fix(needs: list, name: str, environ=os.environ):
    """What sets up NEEDS, what capability NAME lacks, as the operator types it on the host (a checks fix): each by its
    own fix (Need), in turn. First the logins and secrets, each by its command: a deployment whose settings need one
    that is not set applies no change (state 7). Then the settings: the edit of the capability's section
    of the deploy settings file (doctor.section_parts) for those it holds, and each other one's own edit. Then the
    rest: the image, a host's own key, the checks again for plain words. With nothing lacking, the checks run again.
    """
    from . import deploy_tree, doctor
    from .checks import Steps

    needs = [n if isinstance(n, Need) else Need(n) for n in needs]
    first = [n for n in needs if n.entry is not None]
    settings = [n for n in needs if n.entry is None and n.key]
    section = CAPABILITIES[name].section
    held = doctor.SECTION_SETTINGS.get(section, ()) if section else ()

    def holds(n: Need) -> bool:
        key = n.file_key or deploy_tree.name(n.key)
        return any(key == k or key.startswith(k + ".") for k in held)

    inside = [n for n in settings if holds(n)]
    groups = [n.fix_parts(environ) for n in first]
    if inside:
        groups.append(doctor.section_parts(section, environ))
    groups += [n.fix_parts(environ) for n in settings if n not in inside]
    groups += [n.fix_parts(environ) for n in needs if n.entry is None and not n.key]
    parts, said = [], set()
    for group in groups:
        words = tuple(map(str, group))
        if words not in said:
            said.add(words)
            parts += (["; then"] if parts else []) + list(group)
    return Steps(tuple(parts or (doctor.check_again(environ),)))


def refusal(cfg: dict, name: str) -> tuple[int, str] | None:
    """For a command that acts for a capability: None when it may act, else (exit code, plain message)."""
    c = CAPABILITIES[name]
    if not on(cfg, name):
        return EXIT_OFF, (
            f"OFF: {lowered(c.title)} is off, so nothing was done: {why_off(cfg, name)}."
        )
    gone = missing(cfg, name)
    if gone:
        return EXIT_NOT_SET_UP, (
            f"NOT SET UP: {lowered(c.title)} is on but needs {'; '.join(gone)}. Nothing was done; fix: {fix(cfg, name)}"
        )
    return None


def filter_lines(cfg: dict, lines: list[str]) -> list[str]:
    """The tick's lines without those of capabilities that are off or not set up."""
    out = []
    for line in lines:
        name = next((n for p, n in LINES.items() if line.startswith(p)), None)
        if name is None or active(cfg, name):
            out.append(line)
    return out


def tick_reports(cfg: dict, st: dict) -> list[str]:
    """One line per capability that is on but lacks something, printed once until what is missing changes."""
    seen = st.setdefault("features_reported", {})
    out = []
    for name, c in CAPABILITIES.items():
        gone = missing(cfg, name)
        key = "; ".join(gone)
        if gone and seen.get(name) != key:
            out.append(
                f"FEATURE_SKIPPED {lowered(c.title)}: on but needs {key} (fix: {fix(cfg, name)})"
            )
        if gone:
            seen[name] = key
        else:
            seen.pop(name, None)
    return out


def guard_alert_words(cfg: dict) -> str:
    """Why the auto-merge guard alerts and does not act while it is on and set up, in words: the merge queue sends
    nothing, and why (held_words: its mode, or standby)."""
    return (
        "the auto-merge guard only alerts while the merge queue sends nothing "
        f"({held_words(cfg, 'merge_queue')})"
    )


def guard_mode(cfg: dict) -> str:
    """The auto-merge guard in effect: off, alert, or act (act only while it is set up)."""
    v = value(cfg, "auto-merge-guard")
    if v != "act":
        return v
    return v if active(cfg, "auto-merge-guard") else "alert"


def arms(cfg: dict) -> bool:
    """A validated merge order is armed (GitHub's auto-merge for the PRs you cleared), not only proposed: auto-merge
    on (the tick's line is skipped while it is not set up: filter_lines)."""
    return on(cfg, "auto-merge")


# --- the switches' own problems and the combinations ------------------------------------------------------------------


def problems(cfg: dict) -> list[str]:
    """Bad switches in config `features` (values outside their choices, by the values the deploy settings file gives
    each: deploy_tree.value_schema; keys that are no switch), then the combinations that cannot run (coherence's
    structural FAILs)."""
    from . import deploy_tree, schema

    out = []
    feats = cfg.get("features")
    if feats not in (None, {}) and not isinstance(feats, dict):
        return ["features must be a mapping"]
    switches = {k[len(PREFIX) :] for k in deploy_tree.held() if k.startswith(PREFIX)}
    for dotted in _leaves(feats or {}):
        if dotted not in switches:
            out.append(
                f"features.{dotted} is not a feature switch; the switches are "
                + ", ".join(f"features.{s}" for s in sorted(switches))
            )
            continue
        node = deploy_tree.value_schema(PREFIX + dotted) or {}
        out += [
            line
            for _k, line in schema.validate(
                node, switch(cfg, dotted), f"features.{dotted}."
            )
        ]
    return out + [
        what
        for status, what, _fix, structural in coherence(cfg)
        if status == "FAIL" and structural
    ]


def _leaves(d: dict, prefix: str = "") -> list:
    out = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out += _leaves(v, prefix + k + ".")
        else:
            out.append(prefix + k)
    return out


def coherence(cfg: dict, file: str = "") -> list[tuple[str, str, object, bool]]:
    """Every combination of the features and their settings that cannot run (FAIL) or does less than it seems to
    (WARN), as (status, what, the fix, structural): the fix a checks fix (rkmon_lib/checks.py) whose edits are of FILE,
    your deploy settings file in words by default (deploy_tree.FILE), each key by that file's name for it
    (deploy_tree.name), and a switch turned off at each key of the file that turns it on (deploy_tree.feeds); a
    structural FAIL is a combination config.problems refuses; the others are settings left blank, which rkmon confirm
    refuses before anything runs (and the capability reports NOT SET UP at run time)."""
    from . import deploy_tree, settings_file
    from .checks import Described, Edit, Steps

    file = file or deploy_tree.FILE

    def edit(internal: str, value) -> Edit:
        return Edit(deploy_tree.name(internal), file, value)

    def joined(edits) -> tuple:
        # the edits, ", and" between them
        return tuple(
            part
            for i, one in enumerate(edits)
            for part in ((", and",) if i else ()) + (one,)
        )

    def off(internal: str) -> tuple:
        # each key of the file that turns the switch on, set to false: one for most, each type's for alerts and tickets
        keys = [k for k, fed in deploy_tree.feeds() if fed == internal]
        return joined(
            Edit(key, file, False) for key in keys or [deploy_tree.name(internal)]
        )

    def source(internal: str) -> Edit:
        # INTERNAL set to where its value comes from, the x-source of its key in the file (settings_file.source_of)
        return edit(
            internal,
            settings_file.source_of(deploy_tree.node(deploy_tree.name(internal))[1]),
        )

    def either(*parts) -> Steps:
        return Steps(parts)

    admin_on = edit("features.dashboards.admin.enabled", True)
    review_on = edit("features.review_automation.enabled", True)
    out = []
    ra, mq = enabled(cfg, "review_automation"), enabled(cfg, "merge_queue")
    admin, team = enabled(cfg, "dashboards.admin"), enabled(cfg, "dashboards.team")
    tickets, alerts = enabled(cfg, "ticket_creation"), enabled(cfg, "alerts")
    follow = (
        tickets and switch(cfg, "ticket_creation.triggers.review_follow_up") is True
    )
    found = (
        tickets and switch(cfg, "ticket_creation.triggers.out_of_scope_issue") is True
    )
    if team and not admin:
        out.append(
            (
                "FAIL",
                "features.dashboards.team is enabled but features.dashboards.admin is not: the team dashboard is "
                "served or published with the admin one",
                either(admin_on, ", or", *off("features.dashboards.team.enabled")),
                True,
            )
        )
    if found and not admin:
        out.append(
            (
                "FAIL",
                f"{deploy_tree.name('features.ticket_creation.triggers.out_of_scope_issue')} is true but "
                "features.dashboards.admin is off: its proposed tickets wait for your Approve on the admin dashboard",
                either(
                    admin_on,
                    ", or",
                    *off("features.ticket_creation.triggers.out_of_scope_issue"),
                ),
                True,
            )
        )
    if follow and not ra:
        out.append(
            (
                "FAIL",
                f"{deploy_tree.name('features.ticket_creation.triggers.review_follow_up')} is true but "
                "features.review_automation is off: a follow-up ticket carries our review's open items",
                either(
                    review_on,
                    ", or",
                    *off("features.ticket_creation.triggers.review_follow_up"),
                ),
                True,
            )
        )
    if _advanced_on(cfg, "gpu.enabled") and not ra:
        out.append(
            (
                "FAIL",
                f"{deploy_tree.name('gpu.enabled')} is true but features.review_automation is off: the remote GPU "
                "tests are a review's GPU-only cases",
                either(review_on, ", or", *off("gpu.enabled")),
                True,
            )
        )
    if tickets:
        from . import atlassian

        if not atlassian.in_use(cfg):
            out.append(
                (
                    "FAIL",
                    f"features.ticket_creation is enabled but {deploy_tree.name('atlassian.site')} is blank: "
                    "tickets are filed through the atlassian CLI, which needs the site",
                    either(
                        edit("atlassian.site", Described("your JIRA site's address")),
                        ", or",
                        *off("features.ticket_creation.enabled"),
                    ),
                    False,
                )
            )
        if _jira_project(cfg):
            out.append(
                (
                    "FAIL",
                    f"features.ticket_creation is enabled but {deploy_tree.name('jira.project')} is blank",
                    either(
                        edit("jira.project", Described("the JIRA project's key")),
                        ", or",
                        *off("features.ticket_creation.enabled"),
                    ),
                    False,
                )
            )
    if team:
        gone = _team_dashboard_settings(cfg)
        if gone:
            # the fix sets each blank key, serves the page, or leaves the dashboard off
            out.append(
                (
                    "FAIL",
                    "features.dashboards.team is enabled (delivery publish) but these are blank: "
                    + ", ".join(deploy_tree.name(k) for k in gone),
                    either(
                        *joined(source(internal) for internal in gone),
                        ", or",
                        edit("team_dashboard.mode", "serve"),
                        ", or",
                        *off("features.dashboards.team.enabled"),
                    ),
                    False,
                )
            )
    if mq and switch(cfg, "merge_queue.auto_merge") == "enabled" and not _mq_live(cfg):
        out.append(
            (
                "WARN",
                "features.merge_queue.auto_merge is enabled, but the merge queue is in prototype: nothing is armed "
                "until it is live",
                either(
                    "fine while you watch the plan;",
                    edit("features.merge_queue.mode", "live"),
                    "when it should arm",
                ),
                False,
            )
        )
    if mq and _mq_live(cfg) and ra and not _ra_live(cfg):
        out.append(
            (
                "WARN",
                "features.merge_queue is live, but review automation is in prototype: the queue gates on unposted "
                "reviews their authors never saw",
                either(
                    edit("features.review_automation.mode", "live"),
                    ", or",
                    # the merge queue in prototype, the mode that does not gate
                    edit("features.merge_queue.mode", "prototype"),
                ),
                False,
            )
        )
    if follow and ra and not _ra_live(cfg):
        out.append(
            (
                "WARN",
                f"{deploy_tree.name('features.ticket_creation.triggers.review_follow_up')} is true, but review "
                "automation is in prototype: nothing is posted, so no merged PR has our open items to follow up",
                either(
                    "fine until review automation goes live:",
                    edit("features.review_automation.mode", "live"),
                    "when its reviews should post",
                ),
                False,
            )
        )
    if not ra and not mq:
        out.append(
            (
                "WARN",
                "features.review_automation and features.merge_queue are both off: the monitor only reads GitHub",
                either(review_on, ", or", edit("features.merge_queue.enabled", True)),
                False,
            )
        )
    if alerts and not (_guard(cfg) or admin):
        out.append(
            (
                "WARN",
                "features.alerts is enabled, but neither the develop guard nor the admin dashboard runs: there is "
                "nothing to send",
                either(
                    edit("features.merge_queue.develop_guard", True),
                    ", or",
                    admin_on,
                    ", or",
                    *off("features.alerts.enabled"),
                ),
                False,
            )
        )
    return out


# --- what runs --------------------------------------------------------------------------------------------------------


def uses_claude(cfg: dict) -> bool:
    """A Claude session runs: review automation (the verifier is Claude), or the container's Claude session."""
    return on(cfg, "review-drafts") or on(cfg, "headless-claude")


def rules_only(cfg: dict) -> bool:
    """Ticks are run by fixed rules (rk rules-tick): the merge queue runs and no Claude session does."""
    return on(cfg, "merge-plan") and not uses_claude(cfg)


COMPONENTS = ("web", "claude", "reviews", "rules", "merge-queue", "develop-checks")


def components(cfg: dict) -> dict:
    """Which of the deployment's parts start: the web server, the Claude session (the orchestrator), the review
    driver, ticks run by fixed rules, the merge queue, the develop guard."""
    return {
        "web": on(cfg, "admin-dashboard"),
        "claude": uses_claude(cfg),
        "reviews": on(cfg, "review-drafts"),
        "rules": rules_only(cfg),
        "merge-queue": on(cfg, "merge-plan"),
        "develop-checks": on(cfg, "develop-checks"),
    }


def runs_line(cfg: dict) -> str:
    """What starts, in one line (rk features, rk status)."""
    c = components(cfg)
    return "starts: " + ", ".join(
        [
            "the web server" if c["web"] else "no web server",
            "the Claude session" if c["claude"] else "no Claude session",
            "the review driver" if c["reviews"] else "no review driver",
            (
                "ticks run by fixed rules (rk rules-tick)"
                if c["rules"]
                else "ticks run by Claude"
            ),
        ]
    )


def headline_parts(cfg: dict) -> list[str]:
    """Each feature's state, in the words deploy steps 2 and 3 show: "review automation prototype", "merge queue
    off", "admin dashboard on", "team dashboard live", "alerts off", "ticket creation prototype": a feature that sends
    (SENDS) by its mode while it is on; while the merge queue is on, its auto_merge switch after it ("auto_merge
    hints_only")."""

    def feat(name: str, words: str) -> str:
        return f"{words} {mode(cfg, name) if enabled(cfg, name) else 'off'}"

    def switched(name: str, words: str) -> str:
        return f"{words} {'on' if enabled(cfg, name) else 'off'}"

    auto = (
        [f"auto_merge {switch(cfg, 'merge_queue.auto_merge')}"]
        if enabled(cfg, "merge_queue")
        else []
    )
    return [
        feat("review_automation", "review automation"),
        feat("merge_queue", "merge queue"),
        *auto,
        switched("dashboards.admin", "admin dashboard"),
        feat("dashboards.team", "team dashboard"),
        feat("alerts", "alerts"),
        feat("ticket_creation", "ticket creation"),
    ]


def headline(cfg: dict) -> str:
    """The features in one line, their parts (headline_parts) joined by " · ": review automation prototype · merge
    queue off · admin dashboard on · team dashboard off · alerts off · ticket creation off.
    """
    return " · ".join(headline_parts(cfg))


NONE_ON = "no feature is on"  # what on_line says, and rkmon status's first line, when no feature is on


def on_words(cfg: dict) -> list[str]:
    """Each feature that is on, with its mode where it has one (SENDS): "review automation prototype", "merge queue
    live", "the admin dashboard", "alerts prototype"; none when none is. rkmon status's first line joins them with
    " · ".
    """
    return [
        f"{what} {mode(cfg, f)}" if f in SENDS else what
        for f, what in (
            ("review_automation", "review automation"),
            ("merge_queue", "merge queue"),
            ("dashboards.admin", "the admin dashboard"),
            ("dashboards.team", "the team dashboard"),
            ("alerts", "alerts"),
            ("ticket_creation", "ticket creation"),
        )
        if enabled(cfg, f)
    ]


def on_line(cfg: dict) -> str:
    """Each feature that is on (on_words) in one line, as deploy step 8's done line names them: "review
    automation prototype, merge queue live, the admin dashboard"; NONE_ON when none is.
    """
    return ", ".join(on_words(cfg)) or NONE_ON


# --- the explicit account (`rk features`) -----------------------------------------------------------------------------


def configured_with(cfg: dict, name: str) -> str:
    """What a capability is configured with, in a few words: values shown, secrets only as set or missing."""
    from . import bindings, config, doctor, gpu, paths, publish, registry, webserve
    from . import breakage
    from .post import message_label

    p = paths.resolve(cfg)
    w = config.worker(cfg)
    j = cfg.get("jira") or {}
    pub = (cfg.get("publish") or {}).get("html") or {}
    login = config.operator_login(cfg) or "(integrations.github_cli.username not set)"
    return {
        "review-drafts": (
            "first pass "
            + bindings.reviewer_line(cfg)
            + (
                ", API key set"
                if bindings.reviewer_choice(cfg) == bindings.WORKER
                and not bindings.no_draft(cfg)
                and w.get("api_key_file")
                else ""
            )
            + f"; line 1: {bindings.first_line(cfg) or '(not set yet)'}"
            + f"; parallel reviews {config.pipes_line(cfg)}"
        ),
        "merge-plan": f"{cfg.get('repo') or '(repository.name not set)'}, rocKE at "
        f"{paths.repository_folder(cfg) or '(repository.folder not set)'}",
        "admin-dashboard": (
            f"{webserve.home_url(cfg)}, access token "
            f"{'set' if webserve.access_token(cfg) else 'missing'}"
            + (
                f"; reachable beyond this machine on {webserve.exposure(cfg)}"
                if webserve.exposure(cfg)
                else "; this machine only"
            )
        ),
        "priority-requests": f"the admin dashboard and {doctor.rk_command('prioritize')}"
        + (
            ", and the team dashboard's request box"
            if active(cfg, "team-dashboard")
            else ""
        ),
        "develop-checks": (
            f"venv {p['venv'] or '(not set)'}, LLVM tools {p['rocm_llvm_bin'] or '(not set)'}"
        ),
        "review-posting": f"posts as {login}",
        "auto-merge": "GitHub auto-merge only, for PRs you cleared",
        "auto-merge-guard": f"{value(cfg, 'auto-merge-guard')}",
        "holds": f"change requests as {login}",
        "overlap-management": "owner decisions recorded locally; notices posted as the operator",
        "stability-issues": f"label {breakage.labels(cfg)[breakage.STABILITY] or 'not set'}",
        "incident-issues": f"label {breakage.labels(cfg)[breakage.RED] or 'not set'}",
        "chat-alerts": (
            f"message label {message_label(cfg) or 'none'}, webhook "
            f"{_secret_word(cfg, 'teams_webhook')}"
        ),
        "needs-you-alerts": (
            f"message label {message_label(cfg) or 'none'}, one message of a kind per "
            f"{registry.value(cfg, 'teams.message_cooldown_min')} min at most"
        ),
        "follow-up-tickets": f"JIRA project {j.get('project') or '(not set)'}"
        + (f", epic {j['default_epic']}" if j.get("default_epic") else ""),
        "found-issue-tickets": (
            f"JIRA project {j.get('project') or '(not set)'}"
            + (f", epic {j['default_epic']}" if j.get("default_epic") else "")
            + "; proposed tickets wait on the admin dashboard's Tickets card; filed through the atlassian CLI (site "
            f"{(cfg.get('atlassian') or {}).get('site') or 'not set'})"
        ),
        "team-dashboard": (
            f"served here at {webserve.team_url(cfg)} (read-only, with its request box)"
            if publish.mode(cfg) == "serve"
            else f"{pub.get('user') or '?'}@{pub.get('host') or '(host not set)'}, page {pub.get('url') or '?'}, "
            f"key {_secret_word(cfg, 'web_host_ssh_key')}"
        ),
        "queue-notices": "status comments as the operator",
        "drift-notices": "notices as the operator",
        "fix-links": "comments as the operator",
        "weekly-report": f"to {p['weekly_dir'] or '(not set)'}",
        "remote-gpu-tests": gpu.configured_with(cfg),
        "headless-claude": (
            f"the container's orchestrator, a long-lived Claude session (rk orchestrate; {doctor.wrapper()} "
            "logs shows it)"
        ),
    }[name]


def feature_lines(cfg: dict) -> list[str]:
    """Each feature's switches, one line each, as the deploy settings name them."""
    out = []
    for f in ("review_automation", "merge_queue"):
        if not enabled(cfg, f):
            out.append(f"- `features.{f}`: off")
            continue
        extra = (
            f", auto_merge {switch(cfg, 'merge_queue.auto_merge')}, auto_hold "
            f"{str(switch(cfg, 'merge_queue.auto_hold')).lower()}, develop_guard "
            f"{str(switch(cfg, 'merge_queue.develop_guard')).lower()}"
            if f == "merge_queue"
            else ""
        )
        out.append(f"- `features.{f}`: on, {mode(cfg, f)}{extra}")
    out.append(
        f"- `features.dashboards.admin`: {'on' if enabled(cfg, 'dashboards.admin') else 'off'}"
    )
    out.append(
        f"- `features.dashboards.team`: on, {mode(cfg, 'dashboards.team')}"
        if enabled(cfg, "dashboards.team")
        else "- `features.dashboards.team`: off"
    )
    if enabled(cfg, "alerts"):
        kinds = ", ".join(
            f"{k} {str(switch(cfg, f'alerts.{k}') is True).lower()}"
            for k in ("develop_breaks", "needs_you")
        )
        out.append(f"- `features.alerts`: on, {mode(cfg, 'alerts')}, {kinds}")
    else:
        out.append("- `features.alerts`: off")
    if enabled(cfg, "ticket_creation"):
        on_ = [
            t
            for t in ("review_follow_up", "out_of_scope_issue")
            if switch(cfg, f"ticket_creation.triggers.{t}") is True
        ]
        out.append(
            f"- `features.ticket_creation`: on, {mode(cfg, 'ticket_creation')}, triggers "
            f"{', '.join(on_) or 'none'}"
        )
    else:
        out.append("- `features.ticket_creation`: off")
    return out


def summary(cfg: dict) -> str:
    """The features' switches, what starts, then every capability: its state, what it is configured with, and what an
    enabled one still lacks, with the fix; then the two plain lists. Markdown, for `rk features` (rkmon_lib/summary.py).
    """
    rows, good, bad = [], [], []
    for name, c in CAPABILITIES.items():
        v = value(cfg, name)
        gone = missing(cfg, name)
        if not on(cfg, name):
            state = "off"
        else:
            state = ("on" if v is True else str(v)) + (" (incomplete)" if gone else "")
        need = f"needs {'; '.join(gone)}; fix: {fix(cfg, name)}" if gone else ""
        rows.append(
            f"| {c.title} | {state} | {c.when} | {configured_with(cfg, name) if on(cfg, name) else '—'} | "
            f"{need or '—'} |"
        )
        (good if on(cfg, name) and not gone else bad).append(
            f"- {c.title}"
            + (
                f" — on, but needs {'; '.join(gone)} (fix: {fix(cfg, name)})"
                if gone
                else "" if on(cfg, name) else " — off"
            )
        )
    return (
        "\n".join(
            [
                "**Features**",
                "",
                *feature_lines(cfg),
                "",
                runs_line(cfg),
                "",
                "| Capability | State | Runs while | Configured with | Missing |",
                "|---|---|---|---|---|",
                *rows,
                "",
                "**Configured and on**",
                "",
                *(good or ["- none"]),
                "",
                "**Not configured or off**",
                "",
                *(bad or ["- none"]),
            ]
        )
        + "\n"
    )
