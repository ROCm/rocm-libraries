# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The image test's adopt stage (tests/container/image.py): `rkmon adopt`, then `rkmon start`, as a
person moves an earlier monitor's state into a new deployment, on the image under test: before the first start, and
again once the deployment has run and stopped. Each start after an adopt makes
the one-time move of the config.json the copy brought, and the monitor runs in standby until `rkmon standby off`.

The deployment is the deploy path's (tests/container/stage_deploy_path.py, DeployPath: the guide's steps on a fresh
machine, each checked, its results under this stage's name), walked to step 5, the logins, with review automation live,
as an adoption may bring it, and so with a write login, whose token the stand-ins take as GitHub takes one that may
write (tests/fakes/fake_net.py): from the start after an adopt to standby off, standby alone holds what the monitor
sends. Its objects are named after the scope <names>-adopt. The earlier monitor is two copies of its /work, as `docker
cp NAME:/work DIR` makes one, in the stage's temporary folder, each made by `earlier_work` with invented values only:
the package's synthetic production configuration (tests/data/production_config.json) with a value of the copy's own, so
that the two differ, and a state with PR records, one of them in flight, a secret's file and a lock, beside the
lessons, a post's copy, a posted review, and a day's and the spend's files.

In order, each command run as the guide's person types it (DeployPath.rkmon):
- `rkmon adopt --from COPY --dry-run` before the first adopt: PASSED, and its next line gives the command without
  --dry-run; and it changed nothing: `rkmon next` says what it said before, and what the deployment holds is as it was
  (Adopt.footprint).
- `rkmon adopt --from COPY --yes` before the first start: ADOPTED, which says that the deployment starts in standby;
  its apply's lines say the move is due again, and that this deployment's own copies are set aside; its next line gives
  the start.
- The start that line gives: STARTED, in standby, and its next line gives doctor. In the container, as the monitor
  reads it (FACTS): the move made once, of the config.json the copy brought (its record open, of that file's bytes, and
  its report in the start's log); reviews.unnamed_are_ours_until at the move's time; the monitor in standby for the
  adoption (standby.on, adopt's reason) and the adoption's mark finished; the copy's PR records in place, and its
  secret's file left behind.
- doctor and first-tick, as the next lines give them, after which the next line gives standby off.
- `rkmon stop --drain`, as the deploy path stops the monitor (DeployPath.stop_drained).
- adopt again, from the second copy, and the start its next line gives: as before, but that the move made now is a
  move of its own, made after the first, of the second copy's config.json, with that copy's records in place; then
  doctor and first-tick again to the standby off line.
- A send of the monitor's own, as its pipeline makes one: `rk notice`, an overlap notice on a pull request (NOTICE).
  Standby holds it: rk notice says it was kept as an unsent item (HELD), and the state folder keeps it, for standby.
- `rkmon standby off`: RELEASED, after the move's end has renamed config.json, and its next line is the done line. In
  the container: config.json.migrated is the second copy's config.json, read-only; config.json is gone; the move's
  record ended at the time reviews.unnamed_are_ours_until now holds; standby is off and the mark gone. standby off
  again: NOTHING TO DO, as the monitor is not in standby.
- The same send again: it goes out, and rk notice gives the comment's URL (POSTED).
Each check that the next ones need ends the stage when it fails: each adopt, the start after it, and the move that start
makes. Then what the deployment did (DeployPath.after): no write reached the stand-ins before standby off, and after it
the notice did, with the write login's token; and the end (DeployPath.clean).
"""

import hashlib
import json
import os
import pathlib
import re
import time

import hostrun
import stage_deploy_path
from stage_deploy_path import (
    MOVE_WAIT_S,
    OUTCOMES,
    PROGRAM,
    RELEASED,
    REPOSITORY,
    STANDBY_WORDS,
    STARTED_IN_STANDBY,
    WORK_VOLUME,
    command_label,
    github_token,
    next_command,
    next_line,
    outcome,
    polled,
    says,
    step_timeout,
)

STAGE = "adopt"
# The package's synthetic production configuration: production's shape and switches, invented values.
PRODUCTION = ("tests", "data", "production_config.json")
# The secret's file in the earlier monitor's state folder, which adopt leaves behind, and its invented value.
SECRET = ("teams_webhook", "https://hooks.example.com/workflows/invented\n")
LAST_TICK = "2026-11-20T08:00:00Z"  # the earlier monitor's last tick
# The two copies of the earlier monitor's /work, in the order the stage adopts them: each one's own drift_batch, so that
# their config.json differ, and its PRs, each one's status (pr_record).
COPIES = (
    {"drift_batch": 97, "prs": {"101": "merged", "202": "closed"}},
    {"drift_batch": 98, "prs": {"101": "merged", "202": "closed", "303": "merged"}},
)
ADOPTED = (
    r"ADOPTED: the earlier monitor's state, its logins and secrets left behind; the deployment starts "
    + STANDBY_WORDS
)
NOT_IN_STANDBY = r"NOTHING TO DO: the monitor is not in standby"
DRY_RUN = r"PASSED: the plan can be applied"
# The next line once doctor and first-tick have passed in standby: the state's, which ends standby.
STANDBY_OFF = f"next: {PROGRAM} standby off"
# The monitor's home and the config.json the move splits, in the container.
CONFIG_JSON = "/work/monitor/config.json"
DUE_AGAIN = f"the one-time move is due again: the next start splits {CONFIG_JSON} by owner, once"
SET_ASIDE = r"APPLIED: this deployment's own copies are in /work/adopt-backup-\S+"
RENAMED = f"renamed {CONFIG_JSON} config.json.migrated, read-only"
# What the monitor's home says of the move, standby and the adoption, as JSON: python3 -c, as the monitor, in the
# folder of the code it runs (DeployPath.as_monitor).
FACTS = (
    "import hashlib, json, pathlib\n"
    "from rkmon_lib import adopt, config, container_layout, one_time_move, provision, registry, state_store\n"
    "home = pathlib.Path(container_layout.MONITOR_HOME)\n"
    "def sha(p):\n"
    "    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ''\n"
    "def get(name):\n"
    "    v = registry.get(state, name)\n"
    "    return None if v is registry.MISSING else v\n"
    "state = state_store.read(home)\n"
    "rec = one_time_move.record(home)\n"
    "migrated, mark = home / one_time_move.MIGRATED, adopt.mark_file(home)\n"
    "log = pathlib.Path(container_layout.WORK_DIR, 'logs', 'start.log')\n"
    "folder = config.state_dir(home)\n"
    "print(json.dumps({\n"
    "    'standby': get('standby.on'),\n"
    "    'reason': get('standby.reason'),\n"
    "    'until': get(one_time_move.UNTIL),\n"
    "    'adopted_at': get(provision.ADOPTED_AT),\n"
    "    'record': {k: rec.get(k) for k in ('moved_at', 'ended_at', 'config_json_sha256')} if rec else None,\n"
    "    'config_json': sha(one_time_move.config_file(home)),\n"
    "    'migrated': sha(migrated),\n"
    "    'migrated_mode': migrated.stat().st_mode & 0o777 if migrated.is_file() else None,\n"
    "    'mark': json.loads(mark.read_text()) if mark.is_file() else None,\n"
    "    'prs': sorted(json.loads((folder / 'state.json').read_text()).get('prs') or {}),\n"
    "    'state_folder': sorted(p.name for p in folder.iterdir()),\n"
    "    'moved': sum(x.startswith('moved: ') for x in (log.read_text().splitlines() if log.is_file() else [])),\n"
    "}))\n"
)
# A send the monitor makes as its own pipeline makes one (outward.OWN_VAR): `rk notice`, an overlap notice on a pull
# request of the stand-ins', the same one in standby and once standby is off. NOTICE writes its text, whose line 1 is
# the one rk notice wants (post.notice_line), in NOTICE_FILE, the monitor's, and prints the text as JSON. In standby rk
# notice holds it, kept as an unsent item (HELD); after standby off it goes out (POSTED).
NOTICE_PR, NOTICE_KEY = 120, "adopt-stage"
NOTICE_FILE = "/tmp/adopt-stage-notice.md"
NOTICE = (
    "import json, pathlib, sys\n"
    "from rkmon_lib import config, post\n"
    "line1 = post.notice_line(config.load(check=False), 'overlap')\n"
    "text = line1 + '\\n\\nAn invented overlap notice of the image test.\\n'\n"
    "pathlib.Path(sys.argv[1]).write_text(text, encoding='utf-8')\n"
    "print(json.dumps(text))\n"
)
HELD = (
    rf"UNSENT: an overlap notice for pull request {NOTICE_PR}, kept as an unsent item: the monitor is in standby: "
    r"nothing goes out until standby ends \(.+ standby off\)"
)
POSTED = rf"posted: https://github\.com/{re.escape(REPOSITORY)}/pull/{NOTICE_PR}#issuecomment-\d+"
# The unsent items the monitor's state folder keeps (state_store.unsent_items), as JSON: each one's type, what it was
# for, why it was kept and its content.
KEPT = (
    "import json\n"
    "from rkmon_lib import config, container_layout, state_store\n"
    "items = state_store.unsent_items(config.state_dir(container_layout.MONITOR_HOME))\n"
    "keys = ('type', 'for', 'why', 'content')\n"
    "print(json.dumps([{k: i.get(k) for k in keys} for i in items]))\n"
)


def pr_record(key: str, status: str) -> dict:
    """The record of the PR whose number is KEY, STATUS, as the monitor's poll writes one (rkmon_lib/events.py
    diff_and_apply), every value invented: the monitor's tick reads each record whole.
    """
    from rkmon_lib import events  # noqa: E402

    n = int(key)
    return {
        **dict.fromkeys(events.PR_FIELDS),
        "n": n,
        "title": f"An invented change {n}",
        "author": "invented-author",
        "draft": False,
        "labels": [],
        "base": "develop",
        "head_ref": f"change-{n}",
        "head": key[0] * 40,
        "size": 10 * n,
        "files": [],
        "reviews": [],
        "first_seen": LAST_TICK,
        "head_seen_at": LAST_TICK,
        "status": status,
        "ours": None,
        "gates": None,
    }


def earlier_work(folder: pathlib.Path, pkg: str, n: int) -> str:
    """The Nth of COPIES of an earlier monitor's /work, made in FOLDER, a folder that does not exist yet, as `docker cp
    NAME:/work FOLDER` copies one (the module's docstring), every value invented, its state.json as the monitor's store
    writes one (rkmon_lib/store.py): the sha256 of its config.json.
    """
    from rkmon_lib import store  # noqa: E402

    copy = COPIES[n]
    config = json.loads(pathlib.Path(pkg, *PRODUCTION).read_text(encoding="utf-8"))
    config["drift_batch"] = copy["drift_batch"]
    state = {
        **store.empty_state(),
        "enabled": True,
        "prs": {key: pr_record(key, status) for key, status in copy["prs"].items()},
        "last_tick": LAST_TICK,
        # work in flight, which adopt clears
        "review_ready": {sorted(copy["prs"])[0]: {}},
    }
    name, value = SECRET
    files = {
        "monitor/config.json": json.dumps(config, indent=2) + "\n",
        "monitor/state/state.json": json.dumps(state, indent=1),
        f"monitor/state/{name}": value,
        "monitor/state/lock": "",
        "lessons/notes/review.md": "a lesson\n",
        "scratch/post/pr101_fr.md": "a post\n",
        "PR_101_review.md": "a posted review\n",
        "rocke_daily/2026-11-20.md": "a day\n",
        "rocke_spend/2026-11-20.json": "{}\n",
    }
    folder.mkdir(mode=0o700)
    for rel, text in files.items():
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return hashlib.sha256((folder / "monitor" / "config.json").read_bytes()).hexdigest()


def moved_findings(facts: dict, sha: str, prs: list, before: str = "") -> list:
    """How FACTS, the monitor's home's (FACTS), differ from a move made after BEFORE (a time; "" for none) of the
    config.json whose sha256 is SHA, its PRS in place, as the start after an adopt makes it: the move's record open, of
    that file's bytes, made after BEFORE; reviews.unnamed_are_ours_until at its time; the monitor in standby for the
    adoption, whose mark has finished; the PR records, and the secret's file left behind. [] when none.
    """
    from rkmon_lib import adopt  # noqa: E402

    rec = facts.get("record") or {}
    found = []
    if not rec.get("moved_at") or rec.get("ended_at") or rec["moved_at"] <= before:
        found.append(
            f"no open record of a move made after {before or 'the adopt'}: {rec}"
        )
    if (
        not sha
        or rec.get("config_json_sha256") != sha
        or facts.get("config_json") != sha
    ):
        found.append(
            "the move's record and config.json are not of the copy's config.json"
        )
    if facts.get("until") != rec.get("moved_at"):
        found.append(
            f"reviews.unnamed_are_ours_until is {facts.get('until')!r}, not the move's time"
        )
    if facts.get("standby") is not True or facts.get("reason") != adopt.REASON:
        found.append(
            f"standby {facts.get('standby')!r} for {facts.get('reason')!r}, not on for the adoption"
        )
    if not facts.get("adopted_at") or not (facts.get("mark") or {}).get("applied_at"):
        found.append(f"the adoption is not recorded as finished: {facts.get('mark')}")
    if facts.get("prs") != sorted(prs):
        found.append(
            f"the PR records are {facts.get('prs')}, not the copy's {sorted(prs)}"
        )
    if SECRET[0] in (facts.get("state_folder") or []):
        found.append(f"the copy's {SECRET[0]} came along")
    return found


def dry_run_findings(rc: int, out: str, copy: str, before: tuple, after: tuple) -> list:
    """How `rkmon adopt --from COPY --dry-run`, which exited RC saying OUT, differs from a dry run that changed nothing:
    PASSED, its next line the command without --dry-run, and BEFORE and AFTER the same, each (`rkmon next`'s output,
    what the deployment holds: Adopt.footprint's lines), read before and after it. [] when none.
    """
    found = []
    if rc != 0 or not outcome(out, DRY_RUN):
        found.append(f"exit {rc}, outcome {outcome(out, DRY_RUN)!r}")
    if next_command(next_line(out))[1] != ["adopt", "--from", copy]:
        found.append(f"its next line is {next_line(out)!r}")
    if before[0] != after[0]:
        found.append("rkmon next says otherwise after it")
    found += [
        f"{'gone' if x in before[1] else 'new'}: {x}"
        for x in sorted(set(before[1]) ^ set(after[1]))
    ]
    return found


def last_line(out: str) -> str:
    """OUT's last line that is not blank, stripped, or ""."""
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def held_findings(rc: int, out: str, kept: list, text: str) -> list:
    """How `rk notice`, the monitor's own send in standby, which exited RC saying OUT, differs from a send standby
    holds: exit 0, its last line HELD, and among KEPT, the unsent items (KEPT), the notice whose text is TEXT, a comment
    for the pull request, kept for standby. [] when none."""
    from rkmon_lib import state_store  # noqa: E402

    found = []
    if rc != 0 or not re.fullmatch(HELD, last_line(out)):
        found.append(f"rk notice: exit {rc}, its last line {last_line(out)!r}")
    want = {
        "type": state_store.COMMENT,
        "for": f"pull request {NOTICE_PR}",
        "why": state_store.STANDBY,
        "content": text,
    }
    if not any({k: i.get(k) for k in want} == want for i in kept):
        found.append(
            f"no unsent item of the notice for standby among {len(kept)}: "
            + json.dumps([{k: i.get(k) for k in ("type", "for", "why")} for i in kept])
        )
    return found


def released_findings(facts: dict, sha: str) -> list:
    """How FACTS differ from the end of the move of the config.json whose sha256 is SHA, as standby off ends it:
    config.json.migrated that file, read-only, and config.json gone; the move's record ended at the time
    reviews.unnamed_are_ours_until says; standby off, and the mark gone. [] when none.
    """
    rec = facts.get("record") or {}
    found = []
    if facts.get("migrated") != sha or facts.get("config_json"):
        found.append(
            "config.json.migrated is not the copy's config.json, or config.json is still there"
        )
    mode = facts.get("migrated_mode")
    if mode is None or mode & 0o222:
        found.append(
            f"config.json.migrated's permissions are {mode if mode is None else oct(mode)}, not read-only"
        )
    if not rec.get("ended_at") or facts.get("until") != rec.get("ended_at"):
        found.append(
            f"the move's record ended at {rec.get('ended_at')!r}, until is {facts.get('until')!r}"
        )
    if facts.get("standby") is not False or facts.get("mark") is not None:
        found.append(f"standby {facts.get('standby')!r}, the mark {facts.get('mark')}")
    return found


class Adopt(stage_deploy_path.DeployPath):
    """One run of the stage: the deploy path's (DeployPath) to step 5, with review automation live, then the
    adoptions."""

    def __init__(
        self, engine: str, image: str, pkg: str, names: str, run_opts: list, res
    ):
        super().__init__(engine, image, pkg, names, run_opts, res, STAGE)
        self.write_token = github_token()  # the write login's

    # --- the deploy path, with review automation live ----------------------------------------------------------------
    def fakes_config(self) -> dict:
        """The deploy path's stand-ins (DeployPath.fakes_config), with a write login: its token."""
        config = super().fakes_config()
        config["github"]["write_token"] = self.write_token
        return config

    def fill_in(self) -> None:
        """The deploy path's values (DeployPath.fill_in), with review automation live, as an adoption may bring it: from
        the start after an adopt to standby off, standby alone holds what it sends."""
        from rkmon_lib import file_types, settings_file  # noqa: E402

        super().fill_in()
        doc = settings_file.read(str(self.file), file_types.DEPLOY)
        doc.values["features"]["review_automation"]["mode"] = "live"
        self.file.write_text(settings_file.dump(doc), encoding="utf-8")

    def values(self) -> dict:
        """The deploy path's (DeployPath.values), and the write login's token, which review automation live needs."""
        return {**super().values(), "github_write": ("token", self.write_token + "\n")}

    # --- what the stage reads ----------------------------------------------------------------------------------------
    def facts(self, ready=lambda facts: True) -> dict:
        """What the monitor's home says (FACTS), once READY is true of it, or MOVE_WAIT_S have passed: the container's
        own start makes the move, which may finish after the start has said STARTED."""
        return polled(
            lambda: json.loads(
                self.as_monitor("python3", "-c", FACTS).splitlines()[-1]
            ),
            ready,
            MOVE_WAIT_S,
        )

    def footprint(self) -> list:
        """What the deployment holds, while no container of it runs, as lines to compare, sorted: its compose project's
        volumes and containers, by the label docker compose gives them, each container with its state; each file and
        link in its work volume, by kind, size, time and path, read by a container of the stage's own, with no network,
        that mounts the volume read-only and lists it; and the files of this folder's records and docker/.env, by path,
        size and time.
        The users' ids record is left out (docker/rk_ids.py RECORD): the start of any container of the deployment, a
        one-off's too, writes it when the volume has none yet.
        """
        from rkmon_lib import container_layout, deployment, identity  # noqa: E402

        label = f"label={identity.PROJECT}={self.scope}"
        volumes = self.must(
            "volume", "ls", "--filter", label, "--format", "{{.Name}}"
        ).split()
        containers = self.must(
            "ps", "-a", "--filter", label, "--format", "{{.Names}} {{.State}}"
        ).splitlines()
        lines = [f"volume {v}" for v in volumes] + [
            f"container {c}" for c in containers
        ]
        work = f"{self.scope}_{WORK_VOLUME}"
        if work in volumes:
            where, record = container_layout.WORK_DIR, deployment.rk_ids().RECORD
            listed = self.must(
                "run",
                "--rm",
                "--name",
                f"{self.scope}-files",
                "--network",
                "none",
                "--user",
                "0:0",
                "--mount",
                f"type=volume,source={work},target={where},readonly,volume-nocopy",
                "--entrypoint",
                "find",
                self.derived,
                where,
                "!",
                "-type",
                "d",
                "-printf",
                r"%y %s %T@ %p\n",
            )
            lines += [
                f"work {x}" for x in listed.splitlines() if not x.endswith(f" {record}")
            ]
        docker = self.copy / "docker"
        for p in [*(docker / ".rkmon").rglob("*"), docker / ".env"]:
            if p.is_file():
                s = p.stat()
                lines.append(
                    f"host {p.relative_to(self.copy)} {s.st_size} {s.st_mtime_ns}"
                )
        return sorted(lines)

    # --- the person's commands ---------------------------------------------------------------------------------------
    def dry_run(self, copy: pathlib.Path) -> None:
        """`rkmon adopt --from COPY --dry-run`, checked (dry_run_findings): `rkmon next` and what the deployment holds
        (footprint), read before and after it."""
        before = self.rkmon("next")[1], self.footprint()
        rc, out = self.rkmon("adopt", "--from", str(copy), "--dry-run")
        after = self.rkmon("next")[1], self.footprint()
        found = dry_run_findings(rc, out, str(copy), before, after)
        if found:
            hostrun.indented("\n".join(found))
        self.result(
            not found,
            "rkmon adopt --dry-run before the first adopt passes, its next line gives the command without --dry-run, "
            "and it changes nothing: rkmon next says what it said before, and the deployment's volumes, containers, "
            "work volume's files and this folder's records are as they were",
        )

    def adopt(self, copy: pathlib.Path, again: bool) -> str:
        """`rkmon adopt --from COPY --yes`, checked: its output, "" when the check fails."""
        rc, out = self.rkmon("adopt", "--from", str(copy), "--yes")
        joined = " ".join(out.split())
        ok = (
            rc == 0
            and bool(outcome(out, ADOPTED))
            and says(out, DUE_AGAIN)
            and re.search(SET_ASIDE, joined) is not None
            and next_command(next_line(out))[1] == ["start"]
        )
        when = (
            "again, once the deployment has run and stopped,"
            if again
            else "before the first start"
        )
        self.result(
            ok,
            f"rkmon adopt {when} adopts the copy, says that the deployment starts in standby and that the one-time move "
            "is due again, sets this deployment's own copies aside, and its next line gives the start",
        )
        return out if ok else ""

    def start(self, out: str, again: bool) -> str:
        """The start OUT's next line gives, checked: its output, "" when the check fails."""
        rc, out = self.follow(out)
        ok = (
            rc == 0
            and bool(outcome(out, STARTED_IN_STANDBY))
            and next_command(next_line(out))[0] == 7
        )
        if not ok:
            hostrun.indented(
                f"start: exit {rc}, outcome {outcome(out, STARTED_IN_STANDBY)!r}, next line {next_line(out)!r}"
            )
        self.result(
            ok,
            f"the start after {'that' if again else 'the'} adopt starts the monitor in standby, and its next line "
            "gives doctor",
        )
        return out if ok else ""

    def moved(self, sha: str, prs: dict, before: str = "") -> str:
        """The move the start made after BEFORE (a time; "" for none), checked in the container (moved_findings): its
        time, "" when the check fails."""
        facts = self.facts(
            lambda f: ((f.get("record") or {}).get("moved_at") or "") > before
        )
        found = moved_findings(facts, sha, list(prs), before)
        if not found and facts.get("moved") != (2 if before else 1):
            found.append(f"the start's log has {facts.get('moved')} reports of a move")
        if found:
            hostrun.indented("\n".join(found))
        self.result(
            not found,
            f"the start made the one-time move{' again' if before else ''} of the config.json the copy brought, once, "
            "the monitor in standby for the adoption, the copy's PR records in place and its secret left behind",
        )
        return "" if found else facts["record"]["moved_at"]

    def to_standby_off(self, out: str) -> None:
        """From OUT, the start's output, each next line followed while it gives deploy step 7's or step 8's command,
        each exiting 0 with its outcome line; after which the next line gives standby off (STANDBY_OFF).
        """
        said, ok = [], True
        for n in (7, 8):
            if next_command(next_line(out))[0] != n:
                continue
            rc, out = self.follow(out, timeout=step_timeout(n))
            found = outcome(out, OUTCOMES[n])
            said.append(
                f"step {n}: exit {rc}, outcome {found!r}, next line {next_line(out)!r}"
            )
            ok = ok and rc == 0 and bool(found)
        ok = ok and next_line(out).startswith(STANDBY_OFF)
        if not ok:
            hostrun.indented("\n".join(said + [f"last next line: {next_line(out)!r}"]))
        self.result(
            ok,
            "doctor and first-tick, as the next lines give them, pass in standby, after which the next line gives "
            "standby off",
        )

    def notice(self) -> tuple:
        """`rk notice` in the deployment's container (DeployPath.inside) as the monitor's own send (NOTICE), marked as
        it starts: (its exit code, its output, the notice's text)."""
        from rkmon_lib import outward  # noqa: E402

        text = json.loads(
            self.as_monitor("python3", "-c", NOTICE, NOTICE_FILE).splitlines()[-1]
        )
        argv = ["rk", "notice", str(NOTICE_PR), NOTICE_FILE, "--kind", "overlap"]
        argv += ["--key", NOTICE_KEY]
        self.marks.append((time.time(), "rk notice, the monitor's own send"))
        rc, out = self.inside(*argv, env={outward.OWN_VAR: "1"})
        hostrun.indented(f"$ {' '.join(argv)}\n{out}")
        return rc, out, text

    def held(self) -> None:
        """Before standby off, the monitor's own send (notice), checked (held_findings): held, kept as an unsent item.
        That nothing of it reached the stand-ins, after() checks (requests_findings)."""
        rc, out, text = self.notice()
        kept = json.loads(self.as_monitor("python3", "-c", KEPT).splitlines()[-1])
        found = held_findings(rc, out, kept, text)
        if found:
            hostrun.indented("\n".join(found))
        self.result(
            not found,
            "with review automation live, the monitor's own send in standby, an overlap notice, is held: rk notice "
            "says so, and keeps it as an unsent item for standby",
        )

    def sent(self) -> None:
        """After standby off, the same send (notice): it goes out (POSTED), which after() sees in the stand-ins' log as
        the write login's (requests_findings)."""
        rc, out, _text = self.notice()
        ok = rc == 0 and re.fullmatch(POSTED, last_line(out)) is not None
        if not ok:
            hostrun.indented(f"rk notice: exit {rc}, its last line {last_line(out)!r}")
        self.result(
            ok,
            "after standby off, the same send goes out: rk notice says posted, with the comment's URL",
        )

    def released(self, sha: str) -> None:
        """`rkmon standby off`, then again, checked (released_findings)."""
        rc, out = self.rkmon("standby", "off")
        ok = (
            rc == 0
            and bool(outcome(out, RELEASED))
            and says(out, RENAMED)
            and next_line(out).startswith(stage_deploy_path.DONE)
        )
        found = released_findings(self.facts(lambda f: f.get("standby") is False), sha)
        if found or not ok:
            hostrun.indented(
                "\n".join(
                    [f"standby off: exit {rc}, next line {next_line(out)!r}", *found]
                )
            )
        self.result(
            ok and not found,
            "rkmon standby off ends the move and standby: config.json is config.json.migrated, read-only, the move's "
            "record ended at the time reviews.unnamed_are_ours_until says, the mark is gone, and the next line is "
            "the done line",
        )
        rc, out = self.rkmon("standby", "off")
        self.result(
            rc == 0 and bool(outcome(out, NOT_IN_STANDBY)),
            "standby off again has nothing to do: the monitor is not in standby",
        )

    # --- the checks --------------------------------------------------------------------------------------------------
    def adoptions(self) -> None:
        """The stage's checks once the deploy path has passed step 5, in the module docstring's order."""
        moved_at = ""
        for n, again in ((0, False), (1, True)):
            copy = self.top / f"earlier-{n + 1}"
            sha = earlier_work(copy, self.pkg, n)
            if not again:
                self.dry_run(copy)
            out = self.adopt(copy, again)
            out = out and self.start(out, again)
            if not out:
                return
            moved_at = self.moved(sha, COPIES[n]["prs"], moved_at)
            if not moved_at:
                return
            self.to_standby_off(out)
            if not again:
                self.stop_drained()
        self.held()
        self.released(sha)
        self.sent()


def adopt_stage(
    engine: str, image: str, pkg: str, names: str, run_opts: list, res
) -> None:
    """The stage (the module's docstring): its results through RES."""
    if os.name == "nt":
        print(f"SKIP  {STAGE}: rkmon runs on a Linux or macOS host, not on Windows")
        return
    t = Adopt(engine, image, pkg, names, run_opts, res)
    try:
        t.prepare()
        if (
            t.path(last=5) == 5
        ):  # the logins are set: the deployment waits for its first start
            t.adoptions()
        t.after(sends=command_label(("standby", "off")))
    except (RuntimeError, OSError, ValueError) as exc:
        hostrun.indented(str(exc))
        res.result("FAIL", f"{STAGE}: the stage's own setup: {exc.__class__.__name__}")
    finally:
        t.clean()
