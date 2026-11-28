# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's orchestrator (rkmon_lib/orchestrator.py), against a fake Claude Agent SDK: the guard that keeps every
outward action in rk, the session's environment (integrations.claude_code's settings and the claude login's key),
what it lacks before it runs, the settings file it writes before the session starts, the allow list (and no MCP
server), and one session: resume, the ticks while idle, a task notification and its nudge, the usage it records, and
a second start that resumes the session. With the tick triage (a fake rk and a fake model): the orchestrator's own
ticks, the lines the session gets and those it does not, a line already handed over, a triage that does not answer, a
busy session, and the review driver. Then the hooks that attribute tokens to reviews, the effort, and the model roles.
No real Claude and no real model is called."""

import asyncio
import json
import os
import pathlib
import shlex
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import (  # noqa: E402
    claude_env_shape,
    config,
    control,
    first_pass,
    orchestrator,
    outward,
    paths,
    post,
    roles,
    settings_file,
    state_store,
    tickfilter,
    triage,
    usage,
)
from rkmon_lib.store import Store  # noqa: E402

KEY = "test-" + "key-not-real"
OTHER_KEY = "test-" + "other-key-not-real"
HEADERS = "X-Gateway-Sub: " + "sub-not-real"


class Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class AssistantMessage(Msg):
    pass


class ResultMessage(Msg):
    pass


class TaskNotificationMessage(Msg):
    pass


class TaskStartedMessage(Msg):
    pass


class FakeSDK:
    """claude_agent_sdk as the orchestrator uses it: options, hook matchers, and a client whose answers are scripted
    per prompt (`default` answers any prompt the script does not name)."""

    def __init__(self):
        self.queries, self.options, self.script, self.default = [], [], {}, []
        self.queue: asyncio.Queue | None = None
        sdk = self

        class ClaudeAgentOptions:
            def __init__(self, **kw):
                self.__dict__.update(kw)
                sdk.options.append(self)

        class HookMatcher:
            def __init__(self, matcher=None, hooks=None, timeout=None):
                self.matcher, self.hooks, self.timeout = matcher, hooks or [], timeout

        class ClaudeSDKClient:
            def __init__(self, options=None):
                self.options = options

            async def __aenter__(self):
                sdk.queue = asyncio.Queue()
                return self

            async def __aexit__(self, *exc):
                return False

            async def query(self, prompt):
                sdk.queries.append(prompt)
                for m in sdk.script.get(prompt, sdk.default):
                    await sdk.queue.put(m)

            async def receive_messages(self):
                while True:
                    yield await sdk.queue.get()

        self.ClaudeAgentOptions, self.HookMatcher, self.ClaudeSDKClient = (
            ClaudeAgentOptions,
            HookMatcher,
            ClaudeSDKClient,
        )


def turn(session="sess-1", origin=None, n=3, tokens=(100, 20, 1000, 50)):
    return [
        AssistantMessage(content=[]),
        ResultMessage(
            session_id=session,
            origin=origin,
            num_turns=n,
            subtype="success",
            duration_ms=5,
            usage={
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
                "cache_read_input_tokens": tokens[2],
                "cache_creation_input_tokens": tokens[3],
            },
        ),
    ]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        raw = config.load_defaults()
        raw.pop("_doc", None)
        raw.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
            claude_workspace=str(self.d / "ws"),
        )
        raw["paths"] = dict(raw["paths"], working_dir=str(self.d / "work"))
        raw["claude"] = dict(
            raw["claude"],
            gateway="https://gateway.example.com/anthropic",
            api_key_header="X-Gateway-Sub",
        )  # integrations.claude_code's settings
        self.cfg = config.normalize(raw)
        skill = self.d / "ws" / ".claude" / "skills" / "rocke-monitor"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: rocke-monitor\n---\n", encoding="utf-8"
        )
        self.secrets = pathlib.Path(self.cfg["paths"]["secrets_dir"])
        self.secrets.mkdir(parents=True, mode=0o700)
        self.write_key(KEY)

    def write_key(self, key):
        """The claude login: the session's key alone, as `rkmon login claude` writes it."""
        p = self.secrets / "claude_key"
        p.write_text(f"{key}\n", encoding="utf-8")
        p.chmod(0o600)


class TestStateFolder(Base):
    def test_the_state_files_are_in_the_one_state_folder(self):
        """The session's state and usage, the tick triage's record and the first passes' are kept in
        config.toolkit_state_dir, the state folder whose name paths.STATE alone holds, and their readers look there.
        """
        with mock.patch.object(paths, "STATE", "kept"):
            orchestrator.save_state(self.cfg, {"n": 1})
            orchestrator.record(self.cfg, {"cost_usd": 0.5})
            triage.record(self.cfg, {"rc": 0, "claude": 2})
            first_pass.record(self.cfg, {"role": "ci_triage", "status": "ok"})
            kept = pathlib.Path(self.cfg["paths"]["toolkit"]) / "kept"
            self.assertEqual(
                sorted(p.name for p in kept.iterdir()),
                sorted(
                    (orchestrator.STATE, orchestrator.USAGE, triage.LOG, first_pass.LOG)
                ),
            )
            self.assertEqual(orchestrator.load_state(self.cfg), {"n": 1})
            self.assertEqual(len(usage.claude_rows(self.cfg)), 1)
            self.assertIn(
                "ci_triage_pass: last run", "\n".join(roles.table_lines(self.cfg))
            )

    def test_the_session_state_is_written_whole_at_the_states_shared_mode(self):
        """The session's state is written whole, at the state's shared mode whatever the umask, with nothing left
        beside it."""
        umask = os.umask(0o077)
        self.addCleanup(os.umask, umask)
        orchestrator.save_state(self.cfg, {"n": 1})
        folder = config.toolkit_state_dir(self.cfg)
        self.assertEqual(
            stat.S_IMODE((folder / orchestrator.STATE).stat().st_mode),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
        self.assertEqual([p.name for p in folder.iterdir()], [orchestrator.STATE])


class TestGuard(unittest.TestCase):
    def test_outward_writes_that_are_not_rk_are_refused(self):
        for cmd in (
            "gh pr comment 7 --body-file x.md",
            "gh pr review 7 --approve",
            "gh pr merge 7 --auto",
            "gh issue create -t x -b y",
            "gh api -X POST repos/o/r/issues/7/comments -f body=x",
            "gh api repos/o/r/issues/7/comments --method PATCH",
            "gh api repos/o/r/pulls/7/reviews -f event=X",
            "cd /work/repo && git push origin HEAD",
            "git -C /work/repo push",
            "curl -X POST https://x.example/y",
            "curl -d x=1 https://x.example/y",
            "curl --json '{}' https://x.example",
            "wget --post-data=a https://x.example",
            "ssh login.example.com squeue",
            "scp a b:/c",
            "sbatch --wrap 'x'",
            "gh auth token",
            "gh workflow run ci.yml",
            "gh run rerun 5",
        ):
            self.assertIsNotNone(orchestrator.guard(cmd), cmd)
        self.assertIn(
            "goes through an rk command", orchestrator.guard("gh pr comment 7 -b x")
        )

    def test_reads_local_work_and_rk_pass(self):
        for cmd in (
            "gh api repos/o/r/pulls/7",
            "gh api repos/o/r/pulls/7/comments --paginate --jq '.[].body'",
            "gh pr view 7 --json headRefOid",
            "gh run view 5 --log-failed",
            "git -C /work/repo log -3",
            "git fetch origin develop",
            "curl -fsSL https://example.com/x",
            "/work/monitor/rk post 7 x.md",
            "rk post 7 /work/scratch/post/pr7_fr.md --premises spec.json",
            "python3 -m pytest -q tests",
            "atlassian --site s jira issue get " + "TK" + "-1",
            "/work/monitor/rk atlassian --site s --json jira issue get " + "TK" + "-1",
            "cmake --build build -j4",
            "rk gpu-submit 7",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)

    def test_atlassian_writes_only_with_a_ticket_feature(self):
        cmd = "atlassian --site s jira issue create --project P --summary x"
        self.assertIn(
            "an atlassian CLI write (follow-up tickets are off or not set up)",
            orchestrator.guard(cmd),
        )
        self.assertIn(
            "a found issue's ticket is proposed with rk ticket-draft",
            orchestrator.guard(cmd),
        )
        self.assertIsNone(orchestrator.guard(cmd, tickets=True))
        self.assertIsNotNone(
            orchestrator.guard("gh pr comment 1 -b x", tickets=True)
        )  # tickets open JIRA only
        # The session's JIRA calls are rk's (`rk atlassian`, run by the writer): rk weighs a write's feature on each
        # call (rkmon.py write_gate), so the guard lets them through whatever the feature was at the session's start.
        for write in (
            "/work/monitor/rk atlassian --site s jira issue create --project P --summary x",
            "/work/monitor/rk atlassian --site s jira issue remote-link "
            + "TK"
            + "-1 --url https://example.com/x",
        ):
            self.assertIsNone(orchestrator.guard(write), write)
            self.assertIsNone(orchestrator.guard(write, tickets=True), write)
        # The bare CLI's write beside an rk atlassian read is still refused.
        mixed = (
            "rk atlassian jira issue list --jql 'text ~ \"add\"' && "
            "atlassian --site s jira issue create --project P --summary x"
        )
        self.assertIn(
            "(follow-up tickets are off or not set up)", orchestrator.guard(mixed)
        )

    def test_a_jira_read_whose_jql_names_a_write_word_passes(self):
        for jql in (
            'project = P AND text ~ "retry add"',
            'summary ~ "flaky update test"',
            'summary ~ "broken link"',
        ):
            for listed in ("list", "search"):
                cmd = shlex.join(
                    ["/work/monitor/rk", "atlassian", "--site", "s", "--json"]
                    + ["jira", "issue", listed, "--jql", jql]
                )
                self.assertIsNone(orchestrator.guard(cmd), cmd)

    def test_the_commands_only_the_operator_or_rk_runs_are_refused_however_spelled(
        self,
    ):
        names = outward.names(session=False)
        self.assertEqual(
            set(names),
            {
                "ticket-file",
                "ticket-decline",
                "review-approve",
                "teams-send",
                "web-push",
                "web-requests",
                "teams-test",
                "secret-set",
                "host-key",
                "web-host-install",
                "read-login",
                "standby-off",
            },
        )
        for name in names:
            for spelling in (
                "rk {} 7",
                "/work/monitor/rk {} 7",
                "cd /work/monitor && ./rk {} 7",
                "sh /opt/rocke_monitor/rk --home /work/monitor {} 7",
                "python3 -B /opt/rocke_monitor/rkmon.py {} 7",
                "sudo -n -u rkwrite /opt/rocke_monitor/docker/rk_write.py {} 7",
                "rk '{}' 7",
                "echo x | rk {}",
            ):
                cmd = spelling.format(name)
                with self.subTest(cmd=cmd):
                    self.assertIn(
                        f"refused: rk {name} is for a person to run, or for rk itself, never for the "
                        "session: ",
                        orchestrator.guard(cmd, tickets=True) or "",
                    )
        # The session's own: a develop break's message to the team chat (with its cooldown and its ledger line), a
        # draft, the writer's probes, a GitHub write through the writer's entry point, and a log read.
        for cmd in (
            "rk teams-breakage 7",
            "/work/monitor/rk teams-breakage 7 --force",
            "rk ticket-draft /work/scratch/post/t.md",
            "rk writer-secrets --json",
            "rk gh-logins --side write --json",
            "sudo -n -u rkwrite /opt/rocke_monitor/docker/rk_write.py post 7 /work/scratch/post/pr7.md",
            "cat /work/monitor/logs/rk.log | grep web-push",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)


class TestSettings(Base):
    def test_the_session_runs_gh_logged_in_and_has_no_atlassian_login(self):
        """The session's own gh calls (the verifier's, the lane's) use the read login in the secrets folder, as rk's
        own children do. Nothing of the atlassian CLI's login reaches it, however the CLI is set up: its JIRA calls are
        `rk atlassian`, which the writer runs with that login."""
        cfg = dict(
            self.cfg,
            atlassian={
                "site": "example-site",
                "proxy_url": "https://proxy.example.com",
            },
        )
        env = orchestrator.session_env(cfg)
        self.assertEqual(env["GH_CONFIG_DIR"], f"{self.secrets}/gh-read")
        self.assertEqual(env["GH_CONFIG_DIR"], config.env(cfg)["GH_CONFIG_DIR"])
        self.assertEqual([k for k in env if k.startswith("ATLASSIAN_")], [])
        self.assertNotIn(
            "ATLASSIAN_CONFIG_DIR", config.env(cfg)
        )  # the jobs' environment never names it

    def test_its_writes_are_the_monitors_own(self):
        """The session's environment marks its writes as the monitor's own (outward.OWN_VAR): the outward gate keeps
        one it stops as an unsent item, where it refuses one a person typed.
        """
        env = orchestrator.session_env(self.cfg)
        self.assertEqual(env[outward.OWN_VAR], "1")
        with mock.patch.dict(os.environ, {outward.OWN_VAR: env[outward.OWN_VAR]}):
            self.assertTrue(outward.own())

    def test_doctor_checks_gh_as_the_session_runs_it(self):
        """One row, gh's, even with the atlassian CLI set up: the session has no atlassian login to check (the CLI's
        own row checks the login `rk atlassian` uses)."""
        from rkmon_lib import doctor

        cfg = dict(self.cfg, atlassian={"site": "example-site"})
        seen = []

        def run(cmd, **kw):
            seen.append(
                (
                    cmd[0],
                    kw["env"].get("GH_CONFIG_DIR"),
                    kw["env"].get("ATLASSIAN_CONFIG_DIR"),
                )
            )
            return mock.Mock(
                returncode=0,
                stdout="github.com\n  Logged in to github.com account read-login (keyring)",
                stderr="",
            )

        with mock.patch("shutil.which", lambda name, path=None: f"/usr/bin/{name}"):
            checks = doctor.session_login_checks(cfg, run=run)
        self.assertEqual(
            [(c.name, c.status) for c in checks], [("session login: gh", "PASS")]
        )
        self.assertTrue(
            checks[0].found.startswith("read-login, as the session runs it"),
            checks[0].found,
        )
        self.assertEqual(seen, [("gh", f"{self.secrets}/gh-read", None)])
        self.assertNotIn(KEY, " ".join(c.found for c in checks))
        with mock.patch("shutil.which", lambda name, path=None: f"/usr/bin/{name}"):
            (row,) = doctor.session_login_checks(
                cfg,
                run=lambda cmd, **kw: mock.Mock(
                    returncode=1, stdout="", stderr="not logged in"
                ),
            )
        self.assertEqual(row.status, "FAIL")
        self.assertIn(f"GH_CONFIG_DIR {self.secrets}/gh-read", row.found)
        self.assertIn("rkmon login github_read", str(row.fix))
        # An answer that names no login is never a PASS.
        with mock.patch("shutil.which", lambda name, path=None: f"/usr/bin/{name}"):
            (row,) = doctor.session_login_checks(
                cfg,
                run=lambda cmd, **kw: mock.Mock(returncode=0, stdout="", stderr=""),
            )
        self.assertEqual(row.status, "WARN")

    def test_the_sessions_environment(self):
        """The session's variables are those integrations.claude_code's settings and the claude login's key make;
        the nonessential switch is always on, and the prompt cache's lifetimes are
        settings-file keys. An older claude_env beside the login is never read: not its key, its headers, its switch
        nor its model names, and with no login the session gets no key from it."""
        env = orchestrator.session_env(self.cfg)
        self.assertEqual(
            {k: v for k, v in env.items() if k.startswith("ANTHROPIC_")},
            {
                "ANTHROPIC_BASE_URL": "https://gateway.example.com/anthropic",
                "ANTHROPIC_API_KEY": KEY,
                "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Sub: {KEY}",
            },
        )
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        # The prompt cache's lifetimes are settings-file keys (session_settings), never the environment's.
        self.assertFalse([k for k in env if "PROMPT_CACHE_TTL" in k])
        self.assertEqual(
            {
                k: v
                for k, v in orchestrator.session_settings(self.cfg).items()
                if "Cache" in k
            },
            {"promptCacheTtl": "1h", "subagentPromptCacheTtl": "5m"},
        )
        older = self.secrets / "claude_env"
        older.write_text(
            json.dumps(
                {
                    "ANTHROPIC_API_KEY": OTHER_KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": HEADERS,
                    "ANTHROPIC_MODEL": "example-model-1",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0",
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(orchestrator.session_env(self.cfg), env)
        (self.secrets / "claude_key").unlink()
        env = orchestrator.session_env(self.cfg)
        self.assertEqual(
            {k: v for k, v in env.items() if k.startswith("ANTHROPIC_")},
            {"ANTHROPIC_BASE_URL": "https://gateway.example.com/anthropic"},
        )
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertNotIn(OTHER_KEY, json.dumps(env))

    def test_each_row_of_the_table_reaches_the_session(self):
        """Each way the settings send the key, with the key from the claude login: the session's environment
        carries the variables session_variables makes of the settings and that key, and the switch's 1.
        """
        for sent_as, header, scheme in (
            ("header", "", ""),
            ("bearer", "", ""),
            ("header_and_bearer", "", ""),
            ("header", "X-Gateway-Key", "Token"),
            ("header_and_bearer", "X-Gateway-Key", ""),
        ):
            cfg = dict(
                self.cfg,
                claude=dict(
                    self.cfg["claude"],
                    api_key_sent_as=sent_as,
                    api_key_header=header,
                    api_key_scheme=scheme,
                ),
            )
            with self.subTest(sent_as=sent_as, header=header, scheme=scheme):
                env = orchestrator.session_env(cfg)
                self.assertEqual(
                    {k: v for k, v in env.items() if k.startswith("ANTHROPIC_")},
                    orchestrator.session_variables(cfg, KEY),
                )
                self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")

    def test_what_it_lacks(self):
        from rkmon_lib import agentic, doctor

        self.assertEqual(orchestrator.check(self.cfg), [])
        # No operator login: the skill is not rendered, and the fix is the deploy settings file's, then confirm and
        # apply (doctor.apply_commands).
        w = doctor.wrapper_name()
        self.assertEqual(
            orchestrator.check(dict(self.cfg, operator={"github_login": ""})),
            [
                f"{agentic.LOGIN_MISSING} (set it in your deploy settings file, then {w} confirm and {w} apply)"
            ],
        )
        self.assertEqual(
            orchestrator.describe(self.cfg),
            f"the claude login: set ({self.secrets / 'claude_key'}; not shown): the session gets "
            "ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_CUSTOM_HEADERS",
        )
        # An older claude_env is no key of the session's: the one-time move copies it to the claude login.
        (self.secrets / "claude_key").unlink()
        (self.secrets / "claude_env").write_text(
            json.dumps({"ANTHROPIC_API_KEY": OTHER_KEY}), encoding="utf-8"
        )
        off = dict(
            self.cfg,
            features={"review_automation": {"enabled": False}},
            claude_workspace="",
        )
        missing = orchestrator.check(off)
        self.assertEqual(len(missing), 3, missing)
        self.assertIn("review automation is disabled", missing[0])
        self.assertIn("claude_workspace is blank", missing[1])
        self.assertIn("the Claude session has no key (the login claude: ", missing[2])
        self.assertEqual(
            orchestrator.describe(self.cfg),
            f"the claude login: not set ({self.secrets / 'claude_key'})",
        )
        self.assertNotIn(OTHER_KEY, " ".join(missing))

    def test_rk_orchestrate_writes_the_sessions_settings_before_it_starts(self):
        """In the container rk orchestrate writes the session's settings file before the session starts, and nothing
        else writes it there; on a host the workspace is a person's, and `--check`
        starts nothing and writes nothing."""
        from rkmon_lib import agentic, doctor

        target = self.d / "ws" / ".claude" / "settings.json"
        seen = []

        async def run(cfg, stop=None, say=None):
            seen.append(
                json.loads(target.read_text(encoding="utf-8"))
                if target.exists()
                else None
            )
            return 0

        for image, check_only, want in (
            (False, False, [None]),
            (True, True, []),
            (True, False, [orchestrator.session_settings(self.cfg)]),
        ):
            with self.subTest(image=image, check_only=check_only):
                target.unlink(missing_ok=True)
                seen.clear()
                with mock.patch.object(orchestrator, "run", run), mock.patch.object(
                    agentic, "refresh", lambda cfg: []
                ), mock.patch.object(
                    doctor, "deployment_image", lambda environ=None: image
                ), mock.patch(
                    "builtins.print"
                ):
                    self.assertEqual(
                        orchestrator.main(self.cfg, check_only=check_only), 0
                    )
                self.assertEqual(seen, want)
                self.assertEqual(target.exists(), image and not check_only)

    def test_the_allow_list_and_no_jira_server(self):
        """The allow list holds the local tools, the skill and the two agent types. The session has no MCP server: the
        JIRA MCP server is an unused feature, gone with its keys, so no file an older package left in the
        secrets folder gives the session one, nor its tools."""
        tools = orchestrator.allowed_tools(self.cfg)
        for want in (
            "Bash",
            "Agent(rocke-review-verifier)",
            "Skill(rocke-monitor)",
            f"Edit(/{self.cfg['paths']['tmp']}/**)",
            f"Edit(/{self.cfg['paths']['lessons']}/**)",
        ):
            self.assertIn(want, tools)
        (self.secrets / "jira_mcp_headers").write_text(
            "Authorization: Basic " + "mcp-header-31\n", encoding="utf-8"
        )
        on = dict(self.cfg, jira={"project": "TK"})
        with mock.patch.object(orchestrator, "tickets_on", return_value=True):
            self.assertEqual(
                [t for t in orchestrator.allowed_tools(on) if t.startswith("mcp__")],
                [],
            )
            opts = orchestrator.options(on, FakeSDK(), None)
        self.assertFalse(getattr(opts, "mcp_servers", None))
        self.assertFalse(
            [
                w
                for w in ("mcp-header-31", "jira_mcp_headers")
                if w in json.dumps(vars(opts), default=str)
            ]
        )

    def test_each_folder_the_session_writes_in_gets_one_edit_rule_on_its_absolute_path(
        self,
    ):
        """Claude Code reads a file rule's path after `//` as absolute; after one `/` it starts at the settings file's
        folder, so such a rule allows nothing in the work folders. An Edit rule covers Write too, and a Write rule of
        its own is refused all the same. Each folder the skill and its agents write files in gets one rule: Edit, on
        its absolute path, with `//`: the work folders, the weekly reports' folder (the weekly lessons review's drafts)
        and the site lessons' folder (the site lessons, which that review edits)."""
        p = paths.resolve(self.cfg)
        files = [
            t
            for t in orchestrator.allowed_tools(self.cfg)
            if t.startswith(("Write(", "Edit("))
        ]
        self.assertEqual([t for t in files if t.startswith("Write(")], [])
        for rule in files:
            self.assertRegex(rule, r"^Edit\(//[^/].*/\*\*\)$")
        folders = [
            p[key]
            for key in (
                "tmp",
                "scratch",
                "probes",
                "workers",
                "lessons",
                "posts",
                "cache",
                "weekly_dir",
            )
        ] + [os.path.dirname(p["site_lessons"])]
        self.assertTrue(all(f.startswith("/") for f in folders), folders)
        self.assertEqual(files, [f"Edit(/{f}/**)" for f in folders])

    def test_a_work_folder_that_is_not_absolute_is_refused(self):
        """A relative folder would make a rule that allows nothing where the monitor writes: refused, by its name."""
        cfg = dict(self.cfg, paths=dict(self.cfg["paths"], probes="work/probes"))
        with self.assertRaises(ValueError) as cm:
            orchestrator.allowed_tools(cfg)
        self.assertIn(
            "paths.probes is 'work/probes', not an absolute path", str(cm.exception)
        )

    def test_standby_stops_the_sessions_jira_writes_at_their_next_call(self):
        """Whether the session may write to JIRA past rk (tickets_on) is read at each call: standby, begun while the
        session runs, stops the bare atlassian CLI's writes at their next call, and the refusal names standby, or, with
        ticket creation in prototype, its mode.
        """
        cfg = dict(
            self.cfg,
            features=dict(
                self.cfg["features"],
                ticket_creation={"enabled": True, "mode": "live"},
            ),
        )
        home = self.d / "home"
        create = {
            "tool_input": {
                "command": "atlassian --site s jira issue create --project P --summary x"
            }
        }
        active = (
            orchestrator.features.active
        )  # follow-up tickets set up, as the writer's login would have them

        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}), mock.patch.object(
            orchestrator.features,
            "active",
            lambda cfg, name: name == "follow-up-tickets" or active(cfg, name),
        ):
            opts = orchestrator.options(cfg, FakeSDK(), None)
            bash = opts.hooks["PreToolUse"][0].hooks[0]

            def ask():
                return asyncio.run(bash(create, "tu-1", None))

            self.assertEqual(ask(), {})
            state_store.set(home, "standby.on", True)
            denied = [ask()["hookSpecificOutput"]]
            state_store.set(home, "standby.on", False)
            cfg["features"]["ticket_creation"]["mode"] = "prototype"
            denied += [ask()["hookSpecificOutput"]]
        self.assertEqual([d["permissionDecision"] for d in denied], ["deny", "deny"])
        for d, why in zip(
            denied,
            [
                "the monitor is in standby: nothing goes out until standby ends",
                "features.ticket_creation.mode is prototype: nothing goes out until it is live",
            ],
        ):
            self.assertIn(f"({why}", d["permissionDecisionReason"])

    def test_the_hook_s_refusal_names_the_operator_by_login(self):
        """Through the session's hook, a command only a person or rk itself runs is refused naming the person by the
        operator's GitHub login (config.operator_login), as the skill files name them; the guard given no login says
        a person."""
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(self.d / "home")}):
            opts = orchestrator.options(self.cfg, FakeSDK(), None)
            bash = opts.hooks["PreToolUse"][0].hooks[0]
            said = asyncio.run(
                bash({"tool_input": {"command": "rk teams-test"}}, "tu-1", None)
            )
        said = said["hookSpecificOutput"]
        self.assertEqual(said["permissionDecision"], "deny")
        self.assertTrue(
            said["permissionDecisionReason"].startswith(
                "refused: rk teams-test is for op-login to run, or for rk itself, never for the session: "
            ),
            said["permissionDecisionReason"],
        )
        self.assertTrue(
            orchestrator.guard("rk teams-test").startswith(
                "refused: rk teams-test is for a person to run, or for rk itself, never for the session: "
            )
        )


class TestSessionVariables(unittest.TestCase):
    """The session's variables from integrations.claude_code's settings and the key: each way the key is sent, the
    address, a blank key, and the way back to each shape of claude_env. It reads no file and no
    environment."""

    @staticmethod
    def cfg(sent_as="header", header="", scheme="", gateway=""):
        return {
            "claude": {
                "gateway": gateway,
                "api_key_sent_as": sent_as,
                "api_key_header": header,
                "api_key_scheme": scheme,
            }
        }

    def test_each_row_of_the_table(self):
        rows = [
            (("header", ""), {"ANTHROPIC_API_KEY": KEY}),
            (("bearer", ""), {"ANTHROPIC_AUTH_TOKEN": KEY}),
            (("bearer", "X-Gateway-Key"), {"ANTHROPIC_AUTH_TOKEN": KEY}),
            (
                ("header_and_bearer", ""),
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY},
            ),
            (
                ("header", "X-Gateway-Key"),
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
                },
            ),
            (
                ("header_and_bearer", "X-Gateway-Key"),
                {
                    "ANTHROPIC_AUTH_TOKEN": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
                },
            ),
        ]
        for (sent_as, header), want in rows:
            with self.subTest(sent_as=sent_as, header=header):
                self.assertEqual(
                    orchestrator.session_variables(self.cfg(sent_as, header), KEY),
                    want,
                )

    def test_the_word_the_address_and_a_blank_key(self):
        def made(key=KEY, **kw):
            return orchestrator.session_variables(self.cfg(**kw), key)

        self.assertEqual(
            made(header="X-Gateway-Key", scheme="Token")["ANTHROPIC_CUSTOM_HEADERS"],
            f"X-Gateway-Key: Token {KEY}",
        )
        # Anthropic's own header takes the key alone: no word without a named header.
        self.assertEqual(made(scheme="Token"), {"ANTHROPIC_API_KEY": KEY})
        self.assertEqual(
            made(gateway=" https://llm-gateway.example.com/anthropic "),
            {
                "ANTHROPIC_BASE_URL": "https://llm-gateway.example.com/anthropic",
                "ANTHROPIC_API_KEY": KEY,
            },
        )
        self.assertEqual(
            made("", gateway="https://llm-gateway.example.com", header="X-Key"),
            {"ANTHROPIC_BASE_URL": "https://llm-gateway.example.com"},
        )
        # No settings at all: the key in Anthropic's own header, to Anthropic's own API.
        self.assertEqual(
            orchestrator.session_variables({}, KEY), {"ANTHROPIC_API_KEY": KEY}
        )
        # The traffic switch is no setting's: SESSION_ENV's 1 holds for every session.
        self.assertNotIn(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            made(sent_as="header_and_bearer", header="X-Key"),
        )
        self.assertEqual(
            orchestrator.SESSION_ENV["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1"
        )

    def test_each_claude_env_shape_comes_back(self):
        """claude_env's shape (rkmon_lib/claude_env_shape.py) -> integrations.claude_code's settings -> the session's
        variables gives back each shape's key and header variables and its address: the headers production sends today.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "older_secret"
            for env in (
                {"ANTHROPIC_API_KEY": KEY},
                {"ANTHROPIC_AUTH_TOKEN": KEY},
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY},
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
                },
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}",
                },
                {
                    "ANTHROPIC_AUTH_TOKEN": KEY,
                    "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Bearer {KEY}",
                    "ANTHROPIC_BASE_URL": "https://llm-gateway.example.com/anthropic",
                },
            ):
                with self.subTest(env=sorted(env)):
                    path.write_text(json.dumps(env), encoding="utf-8")
                    made, cannot = claude_env_shape.settings(
                        claude_env_shape.read(path, settings_file.secret_shape)
                    )
                    self.assertEqual(cannot, [])
                    cfg = self.cfg(
                        made["api_key_sent_as"],
                        made["api_key_header"],
                        made["api_key_scheme"],
                        made["base_url"],
                    )
                    self.assertEqual(orchestrator.session_variables(cfg, KEY), env)

    def test_it_reads_no_file_and_no_environment(self):
        def refused(*_a, **_kw):
            raise AssertionError("session_variables read a file")

        with mock.patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": OTHER_KEY, "ANTHROPIC_BASE_URL": "x"}
        ), mock.patch("builtins.open", refused), mock.patch.object(
            pathlib.Path, "read_text", refused
        ):
            self.assertEqual(
                orchestrator.session_variables(self.cfg(header="X-Key"), "<key>"),
                {
                    "ANTHROPIC_API_KEY": "<key>",
                    "ANTHROPIC_CUSTOM_HEADERS": "X-Key: <key>",
                },
            )


class SessionBase(Base):
    def run_session(self, sdk, steps, **kw):
        """Run the orchestrator with a fake clock; `steps` are (seconds to advance, action) run one per poll. `kw` go
        to orchestrator.run (a fake rk, the triage's fake model); without one, a fake rk that prints nothing, so no test
        ever runs the real rk."""
        kw.setdefault("rk", FakeRk())
        clock = [1000.0]
        self.seed_last_tick(clock[0])
        stop = asyncio.Event()
        said = []

        async def go():
            task = asyncio.create_task(
                orchestrator.run(
                    self.cfg,
                    sdk=sdk,
                    stop=stop,
                    poll_s=0.01,
                    say=said.append,
                    clock=lambda: clock[0],
                    **kw,
                )
            )
            await asyncio.sleep(0.05)
            for advance, action in steps:
                clock[0] += advance
                if action:
                    await action()
                await asyncio.sleep(0.1)
            stop.set()
            return await task

        return asyncio.run(go()), said

    def seed_last_tick(self, at: float) -> None:
        """A tick on record at `at` (the fake clock's start: a tick just ran), unless the test recorded one itself; the
        orchestrator's tick clock starts from it."""
        import datetime as dt

        f = pathlib.Path(self.cfg["paths"]["toolkit"]) / "state" / "state.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        st.setdefault(
            "last_tick",
            dt.datetime.fromtimestamp(at, dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        )
        f.write_text(json.dumps(st), encoding="utf-8")


class TestSession(SessionBase):
    def test_resume_ticks_a_notification_its_nudge_and_the_usage(self):
        sdk = FakeSDK()
        sdk.script = {
            "/rocke-monitor resume": turn(),
            "/rocke-monitor tick": turn(tokens=(10, 2, 500, 0)),
            orchestrator.NUDGE: turn(tokens=(5, 1, 100, 0)),
        }

        async def notify():
            await sdk.queue.put(
                TaskNotificationMessage(
                    task_id="t1",
                    status="completed",
                    tool_use_id="u1",
                    usage={"total_tokens": 4321},
                    summary="verified #7",
                )
            )

        rk = FakeRk(tick=[(0, "TICK idle: nothing changed since the last full tick\n")])
        code, said = self.run_session(
            sdk,
            [(60, None), (600, None), (1, notify), (orchestrator.NUDGE_S + 1, None)],
            rk=rk,
        )
        self.assertEqual(code, 0)
        # The rule filter (the default triage) read the orchestrator's own tick: an idle tick needs no turn.
        self.assertEqual(sdk.queries, ["/rocke-monitor resume", orchestrator.NUDGE])
        self.assertEqual(rk.calls[0], ["tick"])
        opts = sdk.options[0]
        self.assertEqual(
            (opts.permission_mode, opts.setting_sources, opts.resume, opts.cwd),
            ("dontAsk", ["project"], None, str(self.d / "ws")),
        )
        self.assertEqual(opts.env["ANTHROPIC_API_KEY"], KEY)
        self.assertIn("WebFetch", opts.disallowed_tools)
        self.assertEqual((opts.model, opts.effort), ("claude-opus-5-5", "high"))
        self.assertEqual(
            [m.matcher for m in opts.hooks["PreToolUse"]], ["Bash", "Agent|Task"]
        )
        self.assertEqual(len(opts.hooks["SubagentStop"]), 1)
        st = json.loads(
            (
                pathlib.Path(self.cfg["paths"]["toolkit"])
                / "state"
                / "orchestrator.json"
            ).read_text()
        )
        self.assertEqual(st["session_id"], "sess-1")
        recs = [
            json.loads(x)
            for x in (
                pathlib.Path(self.cfg["paths"]["toolkit"])
                / "state"
                / "claude_usage.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        self.assertEqual([u["kind"] for u in recs], ["resume", "task", "nudge"])
        self.assertEqual((recs[0]["input"], recs[0]["cache_read"]), (100, 1000))
        self.assertEqual(recs[1]["total_tokens"], 4321)
        log = "\n".join(said)
        self.assertNotIn(KEY, log + json.dumps(recs))
        self.assertIn(
            "the tick triage: the rule filter (rk tick-filter, no model)", log
        )
        # The hook refuses an outward write by any agent, and lets a read through.
        hook = opts.hooks["PreToolUse"][0].hooks[0]
        deny = asyncio.run(
            hook(
                {
                    "tool_input": {"command": "gh pr comment 7 -b x"},
                    "agent_type": "rocke-review-verifier",
                },
                "u9",
                None,
            )
        )
        self.assertEqual(deny["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(
            asyncio.run(
                hook(
                    {"tool_input": {"command": "gh api repos/o/r/pulls/7"}}, "u10", None
                )
            ),
            {},
        )
        # rk usage: tokens only
        report = "\n".join(
            usage.report(
                self.cfg, Store(pathlib.Path(self.cfg["paths"]["toolkit"]) / "state"), 7
            )
        )
        self.assertIn(
            "Claude 2 turns (1 nudge, 1 resume); 105 / 21 / 1100 / 50; subagents 4321",
            report,
        )
        self.assertIn("1 command(s) refused by the session's guard", report)
        self.assertNotIn("cost", report)
        # A second start resumes the session.
        sdk2 = FakeSDK()
        sdk2.script = {"/rocke-monitor resume": turn()}
        self.run_session(sdk2, [(1, None)])
        self.assertEqual(sdk2.options[0].resume, "sess-1")

    def test_a_restart_ticks_at_once_when_the_last_tick_on_record_is_due(self):
        """The tick clock starts from state.json's last tick, not from the (re)start: a restart 25 minutes after the
        last tick ticks at once, one 1 minute after it waits out the rest of claude.tick_min, and with no tick on
        record the first runs at once."""
        idle_tick = [(0, "TICK idle: nothing changed since the last full tick\n")]
        for ago, steps, want in (
            (25 * 60, [(1, None)], [["tick"]]),
            (60, [(1, None)], []),
            (60, [(1, None), (540, None)], [["tick"]]),
            (None, [(1, None)], [["tick"]]),
        ):
            with self.subTest(ago=ago, steps=len(steps)):
                if ago is None:
                    f = pathlib.Path(self.cfg["paths"]["toolkit"]) / "state"
                    f.mkdir(parents=True, exist_ok=True)
                    (f / "state.json").write_text(
                        json.dumps({"last_tick": None}), encoding="utf-8"
                    )
                else:
                    (
                        pathlib.Path(self.cfg["paths"]["toolkit"])
                        / "state"
                        / "state.json"
                    ).unlink(missing_ok=True)
                    self.seed_last_tick(1000.0 - ago)
                sdk = FakeSDK()
                sdk.script = {"/rocke-monitor resume": turn()}
                rk = FakeRk(tick=list(idle_tick))
                self.run_session(sdk, steps, rk=rk)
                self.assertEqual(rk.calls, want)
        self.assertEqual(orchestrator.recorded_tick(self.cfg, 5.0), 0.0)
        # A tick on record at 5000 s: a clock behind it counts it as now.
        (
            pathlib.Path(self.cfg["paths"]["toolkit"]) / "state" / "state.json"
        ).write_text(
            json.dumps({"last_tick": "1970-01-01T01:23:20Z"}), encoding="utf-8"
        )
        self.assertEqual(orchestrator.recorded_tick(self.cfg, 1000.0), 1000.0)
        self.assertEqual(orchestrator.recorded_tick(self.cfg, 9000.0), 5000.0)

    def test_no_tick_while_a_turn_runs(self):
        sdk = FakeSDK()
        sdk.script = {
            "/rocke-monitor resume": [AssistantMessage(content=[])]
        }  # a long turn: no result yet
        self.run_session(sdk, [(700, None), (700, None)])
        self.assertEqual(sdk.queries, ["/rocke-monitor resume"])

    def test_rk_tick_now_brings_the_next_tick_forward(self):
        """A tick just ran, so the next one is claude.tick_min away; rk tick-now's request starts it at the next look,
        and the request is gone."""
        state = pathlib.Path(self.cfg["paths"]["toolkit"]) / "state"
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        rk = FakeRk(tick=[(0, "TICK idle: nothing changed since the last full tick\n")])

        async def ask():
            (state / control.TICK_NOW).write_text("2026-11-23T10:00:00Z\n")

        code, said = self.run_session(sdk, [(60, None), (1, ask), (1, None)], rk=rk)
        self.assertEqual(code, 0)
        self.assertEqual(rk.calls, [["tick"]])
        self.assertFalse((state / control.TICK_NOW).exists())
        self.assertIn(
            "ORCH rk tick-now asked for a tick: it is brought forward", "\n".join(said)
        )

    GONE = "0a1b2c3d-0000-4000-8000-000000000000"  # a session recorded on another host

    def gone_sdk(self, where: str, error: str, new_error: str = "") -> FakeSDK:
        """A FakeSDK whose resume fails with ERROR, as the CLI's error shows: at the connect (WHERE "connect") or as
        the reader's (WHERE "reader"). A new session works, and its id is sess-new; with NEW_ERROR, its connect fails
        with that error."""
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn(session="sess-new")}

        class Client(sdk.ClaudeSDKClient):
            async def __aenter__(self):
                await super().__aenter__()
                if self.options.resume and where == "connect":
                    raise RuntimeError(error)
                if not self.options.resume and new_error:
                    raise RuntimeError(new_error)
                return self

            async def receive_messages(self):
                if self.options.resume and where == "reader":
                    raise RuntimeError(error)
                async for m in super().receive_messages():
                    yield m

        sdk.ClaudeSDKClient = Client
        return sdk

    def test_a_recorded_session_not_on_this_host_gives_way_to_a_new_one(self):
        """The recorded session was made on another host, so the CLI does not find it, at the connect or as the
        reader's error: a new session starts at once, gets the resume, and is recorded, and one line says so.
        """
        error = (
            "Claude Code returned an error result: No conversation found with session ID: "
            f"{self.GONE} (exit code: 1)"
        )
        for where, sent in (("connect", 1), ("reader", 2)):
            with self.subTest(where=where):
                orchestrator.save_state(self.cfg, {"session_id": self.GONE})
                sdk = self.gone_sdk(where, error)
                code, said = self.run_session(sdk, [(1, None), (1, None)])
                self.assertEqual(code, 0)
                self.assertEqual([o.resume for o in sdk.options], [self.GONE, None])
                self.assertEqual(sdk.queries, ["/rocke-monitor resume"] * sent)
                lines = [x for x in said if "not found" in x]
                self.assertEqual(len(lines), 1, said)
                self.assertFalse([x for x in said if "the session ended" in x], said)
                self.assertTrue(
                    lines[0].endswith(
                        " ORCH the recorded session 0a1b2c3d was not found on this host: a new session starts"
                    ),
                    lines,
                )
                self.assertEqual(
                    orchestrator.load_state(self.cfg)["session_id"], "sess-new"
                )

    def test_any_other_error_of_a_resume_keeps_its_way(self):
        """Another error at the connect ends the orchestrator with it; another error of the reader ends the session
        (exit 1, and the log says why). Neither starts a new session, and the recorded one stays.
        """
        error = "Claude Code returned an error result: the gateway answered 502 (exit code: 1)"
        orchestrator.save_state(self.cfg, {"session_id": self.GONE})
        sdk = self.gone_sdk("connect", error)
        with self.assertRaisesRegex(RuntimeError, "the gateway answered 502"):
            self.run_session(sdk, [(1, None)])
        self.assertEqual([o.resume for o in sdk.options], [self.GONE])
        sdk = self.gone_sdk("reader", error)
        code, said = self.run_session(sdk, [(1, None)])
        self.assertEqual(code, 1)
        self.assertIn(f"ORCH the session ended: {error}", "\n".join(said))
        self.assertEqual([o.resume for o in sdk.options], [self.GONE])
        self.assertEqual(orchestrator.load_state(self.cfg)["session_id"], self.GONE)

    def test_a_new_session_that_fails_leaves_no_session_on_record(self):
        """The recorded session was not found, and the new session fails at its connect: the orchestrator ends with
        that error, and the record names no session, so the next start begins a new one, not the one this host does
        not have."""
        gone = (
            "Claude Code returned an error result: No conversation found with session ID: "
            f"{self.GONE} (exit code: 1)"
        )
        orchestrator.save_state(self.cfg, {"session_id": self.GONE})
        sdk = self.gone_sdk("connect", gone, new_error="the gateway answered 502")
        with self.assertRaisesRegex(RuntimeError, "the gateway answered 502"):
            self.run_session(sdk, [(1, None)])
        self.assertEqual([o.resume for o in sdk.options], [self.GONE, None])
        self.assertIsNone(orchestrator.load_state(self.cfg)["session_id"])


TICK_OUT = """TICK 2026-11-05T10:20:00Z monitor ON reviews=live merge-queue=prototype auto-merge=hints_only
develop 1234567890 CI green; last green 1234567890 (ci)
EVENTS: none
QUEUE: empty (no PR passes G1-G4, G8, G9)
ANALYSIS_START #12 (G-ready, waited 3 min): early gate
PROPOSE #12 → #13: validated on develop 1234567890; tell the user to merge in this order
!! DISARM_NEEDED #4: its head moved. Run `rk disarm 4`
REVIEW_READY #7 (follow-up): worker done, checks pass. Inputs: /w/tmp/v7fu1
SOMETHING_NEW #8: a line no rule knows
BOARD (changed): show it to the user
| Part | State |
|---|---|
| Monitor | ON |
time: poll 3s
"""
VERDICT = {
    "summary": "rocke-monitor 10:20Z · develop 1234567 green · 0 events · queue: empty · next: #7",
    "unsure": [
        {
            "line": "SOMETHING_NEW #8: a line no rule knows",
            "class": "quiet",
            "why": "progress",
        }
    ],
    "claude": [
        {
            "line": "CI_AFTER_POST #9 abc1234: build-linux",
            "why": "failed after our post",
        }
    ],
    "user": [],
    "done": ["rk review-prep 7: /w/tmp/v7fu1"],
}


class FakeRk:
    """rk as the orchestrator runs it: each command's outputs in turn (the last one repeats), every call recorded."""

    def __init__(self, **outputs):
        self.outputs, self.calls = {
            k.replace("_", "-"): list(v) for k, v in outputs.items()
        }, []

    async def __call__(self, args, timeout):
        self.calls.append(list(args))
        out = self.outputs.get(args[0]) or [(0, "")]
        return out.pop(0) if len(out) > 1 else out[0]


def model(verdict=None):
    """The tick triage's model: one JSON answer."""
    return lambda messages, tools: {"content": json.dumps(verdict or VERDICT)}


class TestTriagedTicks(SessionBase):
    """The tick triage on (models.tick: the reviewer model, set up): the orchestrator's own ticks."""

    def setUp(self):
        super().setUp()
        self.cfg = dict(
            self.cfg,
            worker={
                "backend": "openai",
                "url": "https://models.example.invalid/v1",
                "model": "m-1",
            },
            models=dict(self.cfg["models"], tick="worker"),
        )
        self.state = pathlib.Path(self.cfg["paths"]["toolkit"]) / "state"

    @staticmethod
    def acts(sdk):
        return [q for q in sdk.queries if q.startswith("/rocke-monitor tick given=")]

    def test_the_orchestrator_ticks_and_the_session_gets_only_the_lines_that_need_claude(
        self,
    ):
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        sdk.default = turn(tokens=(50, 5, 500, 0))
        rk = FakeRk(tick=[(0, TICK_OUT)])
        code, said = self.run_session(
            sdk, [(60, None), (600, None), (0, None), (0, None)], rk=rk, chat=model()
        )
        self.assertEqual(code, 0)
        self.assertEqual(rk.calls, [["tick"]])
        self.assertNotIn("/rocke-monitor tick", sdk.queries)  # no plain tick
        (act,) = self.acts(sdk)
        head, *lines = act.splitlines()
        path = pathlib.Path(head.split("given=", 1)[1])
        self.assertEqual(path.parent, self.state / orchestrator.TICKS)
        self.assertEqual(
            lines,
            [
                "The orchestrator ran rk tick and the tick triage did the rest. Handle these lines as "
                "the tick procedure says:",
                "!! DISARM_NEEDED #4: its head moved. Run `rk disarm 4`",
                "REVIEW_READY #7 (follow-up): worker done, checks pass. Inputs: /w/tmp/v7fu1",
                "CI_AFTER_POST #9 abc1234: build-linux",
            ],
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("## The tick's output", text)
        self.assertIn("- did: rk review-prep 7: /w/tmp/v7fu1", text)
        self.assertIn("SOMETHING_NEW #8", text)
        log = "\n".join(said)
        self.assertIn("the tick triage: the local review loop's model m-1", log)
        self.assertIn(VERDICT["summary"], log)
        self.assertIn("  PROPOSE #12", log)
        self.assertIn("  !! DISARM_NEEDED #4", log)
        self.assertIn(
            "  for the session: !! DISARM_NEEDED, REVIEW_READY #7, CI_AFTER_POST #9",
            log,
        )
        self.assertIn("| Monitor | ON |", log)  # the board: it changed
        self.assertNotIn("ANALYSIS_START", log)
        (rec,) = [
            json.loads(x) for x in (self.state / triage.LOG).read_text().splitlines()
        ]
        self.assertEqual(
            (rec["rc"], rec["claude"], rec["down"], rec["classes"]["claude"]),
            (0, 3, "", 2),
        )
        turns = [
            json.loads(x)
            for x in (self.state / "claude_usage.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [(t["kind"], t.get("prs")) for t in turns], [("resume", None), ("act", [7])]
        )

    def test_a_line_already_handed_over_waits_an_hour_unless_it_is_urgent(self):
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        sdk.default = turn()
        rk = FakeRk(tick=[(0, TICK_OUT)])
        steps = [
            (600, None),
            (0, None),
            (600, None),
            (0, None),
            (orchestrator.REPEAT_MIN * 60, None),
            (0, None),
        ]
        self.run_session(sdk, steps, rk=rk, chat=model(dict(VERDICT, claude=[])))
        acts = self.acts(sdk)
        self.assertEqual((len(rk.calls), len(acts)), (3, 3))
        self.assertEqual([len(a.splitlines()) - 2 for a in acts], [2, 1, 2])
        self.assertIn("!! DISARM_NEEDED #4", acts[1])
        self.assertNotIn("REVIEW_READY", acts[1])

    def test_a_triage_that_does_not_answer_gives_the_session_the_whole_tick_then_the_plain_tick(
        self,
    ):
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn(), "/rocke-monitor tick": turn()}
        sdk.default = turn()

        def down(messages, tools):
            raise ConnectionError("connection refused")

        rk = FakeRk(tick=[(0, TICK_OUT)])
        steps = [
            (600, None),
            (0, None),
            (600, None),
            (0, None),
            (triage.DOWN_MIN * 60, None),
            (0, None),
        ]
        _code, said = self.run_session(sdk, steps, rk=rk, chat=down)
        acts = self.acts(sdk)
        self.assertIn(
            "the tick triage did not answer: handle the whole tick in that file as the tick procedure says, "
            "from step 2",
            acts[0],
        )
        self.assertEqual(
            sdk.queries[2], "/rocke-monitor tick"
        )  # the plain tick while the triage counts as down
        self.assertEqual(len(rk.calls), 2)  # then the orchestrator ticks again
        self.assertIn(
            "the tick triage did not answer (ConnectionError: connection refused)",
            "\n".join(said),
        )

    def test_a_busy_session_gets_the_lines_when_its_turn_ends(self):
        sdk = FakeSDK()
        sdk.script = {
            "/rocke-monitor resume": [AssistantMessage(content=[])]
        }  # a long turn: no result yet
        sdk.default = turn()
        rk = FakeRk(tick=[(0, TICK_OUT)])
        before = []

        async def finish():
            before.extend(sdk.queries)
            await sdk.queue.put(turn()[1])

        self.run_session(
            sdk, [(600, None), (0, None), (0, finish), (0, None)], rk=rk, chat=model()
        )
        self.assertEqual(rk.calls, [["tick"]])  # the tick ran during the long turn
        self.assertEqual(
            before, ["/rocke-monitor resume"]
        )  # its lines waited for the turn's end
        self.assertEqual(len(self.acts(sdk)), 1)

    def test_the_review_driver_runs_from_here_and_its_ready_lines_reach_the_session(
        self,
    ):
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "state.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8"
        )
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        sdk.default = turn()
        ready = "REVIEW_READY #5 (first-round): worker done, checks pass. Inputs: /w/tmp/v5fr"
        rk = FakeRk(
            review_driver=[
                (0, f"10:01Z REVIEW_START #5 (first-round): started\n{ready}\n"),
                (
                    0,
                    "DRIVER: nothing to start now (backlog empty); the ticks take over\n",
                ),
            ]
        )
        self.run_session(sdk, [(0, None), (0, None)], rk=rk, chat=model())
        self.assertEqual(
            rk.calls, [["review-driver"], ["review-driver"]]
        )  # again at once after REVIEW_READY only
        (act,) = self.acts(sdk)
        self.assertEqual(act.splitlines()[2:], [ready])
        self.assertTrue(act.splitlines()[0].endswith("-driver.md"))
        # Off, or with review automation disabled, the driver never starts.
        (self.state / "state.json").write_text(
            json.dumps({"enabled": False}), encoding="utf-8"
        )
        self.assertFalse(orchestrator.driver_ok(self.cfg))
        (self.state / "state.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8"
        )
        self.assertFalse(
            orchestrator.driver_ok(
                dict(self.cfg, features={"review_automation": {"enabled": False}})
            )
        )

    def ready_state(self, *entries, skip=()):
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "state.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "review_ready": list(entries),
                    "review_skip": list(skip),
                }
            ),
            encoding="utf-8",
        )

    def keep_unposted(self, n: int) -> None:
        """PR n's verified review kept unposted, as `rk post` keeps it while posting is off (post.save_draft)."""
        f = self.state / "state.json"
        st = dict(json.loads(f.read_text(encoding="utf-8")), prs={})
        verified = self.state / f"pr{n}_verified.md"
        verified.write_text("the verified review\n", encoding="utf-8")
        post.save_draft(self.cfg, st, Store(self.state), n, str(verified))
        f.write_text(json.dumps(st), encoding="utf-8")

    def test_a_ready_review_no_verifier_holds_is_handed_after_the_start(self):
        """An orchestrator killed after its driver started: the review became ready in the driver it left running, whose
        REVIEW_READY reached no session. The new session gets the line, verbatim, once its resume turn is over; a
        review its resume turn started a verifier for is not handed; a review it got stays with it for REPEAT_MIN.
        """
        ready = "REVIEW_READY #14 (follow-up): worker done, checks pass. Inputs: /w/workers/pr14/review_agent.md"
        self.ready_state(
            {
                "pr": 14,
                "kind": "follow-up",
                "head": "a" * 40,
                "at": "1970-01-01T00:00:00Z",
                "line": ready,
            },
            {
                "pr": 15,
                "kind": "first-round",
                "head": "b" * 40,
                "at": "1970-01-01T00:00:00Z",
            },
        )
        sdk = FakeSDK()
        sdk.script = {
            "/rocke-monitor resume": [AssistantMessage(content=[])]
        }  # the resume turn runs until resume_verifies_15 ends it
        sdk.default = turn()
        rk = FakeRk(
            review_driver=[(0, "DRIVER: another review driver is already running\n")],
            tick=[(0, "TICK idle: nothing changed since the last full tick\n")],
        )

        async def resume_verifies_15():
            self.assertEqual(self.acts(sdk), [])  # nothing while the resume turn runs
            opts = sdk.options[0]
            await opts.hooks["PreToolUse"][1].hooks[0](
                {"tool_input": {"prompt": "Brief: PR #15, inputs in /work/tmp/v15fr"}},
                "tu-15",
                None,
            )
            await sdk.queue.put(TaskStartedMessage(task_id="t15", tool_use_id="tu-15"))
            await sdk.queue.put(turn()[1])

        _code, said = self.run_session(
            sdk,
            [(0, resume_verifies_15), (0, None), (0, None), (600, None), (0, None)],
            rk=rk,
            chat=model(),
        )
        (act,) = self.acts(sdk)  # once: the tick 10 minutes later hands nothing again
        head, *lines = act.splitlines()
        self.assertTrue(head.endswith("-ready.md"), head)
        self.assertEqual(lines[-1], ready)
        self.assertNotIn("#15", act)  # its verifier runs
        self.assertIn("READY again for the session: REVIEW_READY #14", "\n".join(said))
        self.assertIn(["tick"], rk.calls)

    def test_which_ready_reviews_a_verifier_holds(self):
        self.ready_state(
            {
                "pr": 21,
                "kind": "follow-up",
                "head": "c" * 40,
                "at": "1970-01-01T01:00:00Z",
            },
            {
                "pr": 22,
                "kind": "follow-up",
                "head": "d" * 40,
                "at": "1970-01-01T01:00:00Z",
            },
            {
                "pr": 23,
                "kind": "follow-up",
                "head": "e" * 40,
                "at": "1970-01-01T00:00:00Z",
            },
            {
                "pr": 24,
                "kind": "follow-up",
                "head": "f" * 40,
                "at": "1970-01-01T01:00:00Z",
            },  # verified, then kept unposted below (posting off): it waits on no verifier
            {
                "pr": 25,
                "kind": "first-round",
                "head": "9" * 40,
                "at": "1970-01-01T01:00:00Z",
            },  # skipped (rk review-skip): it waits on no verifier
            skip=[25],
        )
        self.keep_unposted(24)
        clock = [1000.0]
        s = orchestrator.Session(
            self.cfg, clock=lambda: clock[0], say=lambda line: None
        )
        s.agent_started(
            "tu-21", 21
        )  # an earlier round's verifier, still running: it holds the PR
        s.on_message(None, TaskStartedMessage(task_id="t21", tool_use_id="tu-21"))
        s.agent_started(
            "tu-22", 22
        )  # an earlier round's, in the background, ended before the review became ready
        s.on_message(None, TaskStartedMessage(task_id="t22", tool_use_id="tu-22"))
        s.on_message(
            None,
            TaskNotificationMessage(
                task_id="t22", tool_use_id="tu-22", status="completed"
            ),
        )
        s.agent_started(
            "tu-23", 23
        )  # ran in the foreground for this review: its turn ended
        s.on_message(None, turn()[1])
        lines = orchestrator.unheld_ready(self.cfg, s)
        self.assertEqual([x.split()[1] for x in lines], ["#22"])
        self.assertIn(
            "awaiting verification at dddddddddd since 1970-01-01T01:00:00Z", lines[0]
        )
        self.assertIn("pr22/review_agent.md", lines[0])
        # Off, nothing is handed.
        (self.state / "state.json").write_text(
            json.dumps({"enabled": False, "review_ready": [{"pr": 22, "at": "x"}]}),
            encoding="utf-8",
        )
        self.assertEqual(orchestrator.rehand(self.cfg, s), [])

    def test_a_split_reviews_running_parts_keep_it_from_being_handed_back(self):
        """A split review's part verifiers, dispatched with the description and the brief `rk review-split` prints for
        each, are the review's verifiers: its REVIEW_READY line is not handed back while they run in the background,
        after the session's turn is over. An agent whose description and brief name no PR counts for no review.
        """
        from rkmon_lib import split

        ready = "REVIEW_READY #31 (first round): checks done. Inputs: /w/workers/pr31/review_agent.md"
        self.ready_state(
            {
                "pr": 31,
                "kind": "first-round",
                "head": "a" * 40,
                "at": "1970-01-01T00:00:00Z",
                "line": ready,
            }
        )
        s = orchestrator.Session(self.cfg, clock=lambda: 5000.0, say=lambda line: None)
        hook = (
            orchestrator.options(self.cfg, FakeSDK(), None, session=s)
            .hooks["PreToolUse"][1]
            .hooks[0]
        )

        def start(tu: str, description: str, prompt: str) -> None:
            tool_input = {
                "description": description,
                "subagent_type": "rocke-review-verifier",
                "prompt": prompt,
            }
            asyncio.run(hook({"tool_input": tool_input}, tu, None))
            s.on_message(None, TaskStartedMessage(task_id=tu, tool_use_id=tu))

        start("tu-0", "part 1 verifier", "Review the part file given above.")
        s.on_message(None, turn()[1])
        self.assertEqual(orchestrator.unheld_ready(self.cfg, s), [ready])
        out = pathlib.Path("/work/tmp/v31fr")
        for k in (1, 2, 3):
            description, brief = split.agent_call(31, "fr", out, k, 3)
            start(f"tu-{k}", description, f"A part review. The brief: {brief}")
        s.on_message(None, turn()[1])
        self.assertEqual(orchestrator.unheld_ready(self.cfg, s), [])

    def test_the_drivers_recheck_and_drop_lines_reach_the_session(self):
        """As the live tick loop's driver filter (rk tick-filter --driver) hands them over."""
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "state.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8"
        )
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        sdk.default = turn()
        recheck = "RECHECK_DONE #6: the checks re-ran at abc1234: /w/workers/pr6/checks/checks.md"
        drop = "REVIEW_DROP #8: closed or head moved before verification; will re-pick"
        rk = FakeRk(
            review_driver=[
                (0, f"10:01Z REVIEW #6: checks running\n{recheck}\n{drop}\n"),
                (
                    0,
                    "DRIVER: nothing to start now (backlog empty); the ticks take over\n",
                ),
            ]
        )
        self.run_session(sdk, [(0, None), (0, None)], rk=rk, chat=model())
        (act,) = self.acts(sdk)
        self.assertIn(recheck, act.splitlines())
        self.assertIn(drop, act.splitlines())
        self.assertNotIn("DRIVER_EXIT", act)

    def test_the_tick_follows_tick_timeout_min(self):
        self.assertEqual(orchestrator.tick_timeout_s(self.cfg), 20 * 60)
        self.assertEqual(
            orchestrator.tick_timeout_s(dict(self.cfg, tick_timeout_min=7)), 7 * 60
        )
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        sdk.default = turn()
        seen = []

        async def slow(args, timeout):
            seen.append(timeout)
            raise asyncio.TimeoutError

        self.cfg = dict(self.cfg, tick_timeout_min=7)
        _code, said = self.run_session(
            sdk, [(600, None), (0, None)], rk=slow, chat=model()
        )
        self.assertEqual(seen[:1], [420])
        self.assertIn(
            "rk tick ran over 7 min (tick_timeout_min) and was ended", "\n".join(said)
        )

    def a_tick_of_one_line(self, line: str) -> None:
        """A tick that did nothing: its line said, no triage, no tick file and nothing for the session."""
        sdk = FakeSDK()
        sdk.script = {"/rocke-monitor resume": turn()}
        rk = FakeRk(tick=[(0, line)])
        asked = []
        _code, said = self.run_session(
            sdk,
            [(600, None), (0, None)],
            rk=rk,
            chat=lambda m, t: asked.append(m) or {"content": "{}"},
        )
        self.assertEqual((rk.calls, asked, self.acts(sdk)), ([["tick"]], [], []))
        self.assertIn(line.strip(), "\n".join(said))
        self.assertFalse((self.state / orchestrator.TICKS).exists())

    def test_a_tick_that_did_nothing_is_one_line(self):
        self.a_tick_of_one_line(
            "TICK 2026-11-05T10:20:00Z monitor OFF: nothing done (paused until `rk resume`)\n"
        )

    def test_a_tick_skipped_while_another_runs_is_one_line(self):
        """In the words rk tick prints when it finds another tick running (tickfilter.TICK_RUNNING)."""
        self.a_tick_of_one_line(
            f"TICK 2026-11-05T10:20:00Z {tickfilter.TICK_RUNNING}; skipped\n"
        )

    def test_check_names_the_triage_and_the_effort(self):
        with mock.patch("builtins.print") as out:
            self.assertEqual(
                orchestrator.main(
                    dict(self.cfg, effort={"orchestrator": "max"}), check_only=True
                ),
                0,
            )
        self.assertIn(
            "effort max, a tick every 10 min; the tick triage: the local review loop's model m-1",
            out.call_args[0][0],
        )


class TestUsageHooks(Base):
    """The records that give each review its Claude tokens: the Agent brief's PR and round, the task's start and
    notification, the subagent's transcript, and the turn that answers the notification.
    """

    def test_a_subagents_tokens_reach_its_review(self):
        sdk = FakeSDK()
        s = orchestrator.Session(self.cfg, clock=lambda: 5000.0, say=lambda line: None)
        opts = orchestrator.options(self.cfg, sdk, None, session=s)
        brief = opts.hooks["PreToolUse"][1].hooks[0]
        stop = opts.hooks["SubagentStop"][0].hooks[0]
        self.assertEqual(
            asyncio.run(
                brief(
                    {
                        "tool_input": {
                            "description": "Verify the follow-up",
                            "subagent_type": "rocke-review-verifier",
                            "prompt": "Brief: PR #12, inputs in /work/tmp/v12fu2/facts.txt",
                        }
                    },
                    "tu-1",
                    None,
                )
            ),
            {},
        )
        self.assertEqual(s.task_prs, {"tu-1": 12})
        s.on_message(sdk, TaskStartedMessage(task_id="a1", tool_use_id="tu-1"))
        transcript = self.d / "agent-a1.jsonl"
        parts = [
            {
                "type": "assistant",
                "message": {
                    "id": "m1",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 100,
                    },
                },
            },
            {
                "type": "assistant",
                "message": {
                    "id": "m1",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 40,
                        "cache_read_input_tokens": 100,
                    },
                },
            },
            {"type": "user", "message": {"role": "user", "content": "x"}},
            {
                "type": "assistant",
                "message": {
                    "id": "m2",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 700,
                    },
                },
            },
        ]
        transcript.write_text(
            "".join(json.dumps(p) + "\n" for p in parts) + "{broken\n", encoding="utf-8"
        )
        self.assertEqual(
            asyncio.run(
                stop(
                    {
                        "agent_id": "a1",
                        "agent_type": "rocke-review-verifier",
                        "agent_transcript_path": str(transcript),
                    },
                    None,
                    None,
                )
            ),
            {},
        )
        s.on_message(
            sdk,
            TaskNotificationMessage(
                task_id="a1",
                status="completed",
                tool_use_id="tu-1",
                usage={"total_tokens": 999},
                summary="verified #12",
            ),
        )
        s.on_message(sdk, turn(origin={"kind": "task-notification"})[1])
        recs = [
            json.loads(x)
            for x in (
                pathlib.Path(self.cfg["paths"]["toolkit"])
                / "state"
                / "claude_usage.jsonl"
            )
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [r["kind"] for r in recs],
            ["agent", "task_start", "subagent", "task", "task-notification"],
        )
        self.assertEqual(
            (recs[0]["pr"], recs[0]["round"], recs[0]["subagent_type"]),
            (12, "fu2", "rocke-review-verifier"),
        )
        self.assertEqual(
            {
                k: recs[2][k]
                for k in ("input", "output", "cache_read", "cache_write", "messages")
            },
            {
                "input": 3,
                "output": 45,
                "cache_read": 100,
                "cache_write": 700,
                "messages": 2,
            },
        )
        self.assertEqual(recs[4]["prs"], [12])
        (run,) = usage.subagents(recs)
        self.assertEqual((run["pr"], run["round"], run["tokens"]), (12, "fu2", 848))

    def test_brief_pr(self):
        for text, want in (
            ("inputs in /w/tmp/v123fr/facts.txt", (123, "fr")),
            ("v7fu3", (7, "fu3")),
            ("Verify PR #45, first round", (45, "fr")),
            ("the follow-up 2 of PR 46", (46, "fu2")),
            ("Quality lane for #47 (fu1)", (47, "fu1")),
            ("#48, a follow-up", (48, "fu")),
            ("no number here", (None, "")),
            ("", (None, "")),
        ):
            self.assertEqual(orchestrator.brief_pr(text), want, text)
        self.assertEqual(
            orchestrator.transcript_usage(str(self.d / "missing.jsonl")), {}
        )
        self.assertEqual(orchestrator.transcript_usage(None), {})

    def test_the_prompt_for_waiting_lines(self):
        p = orchestrator.act_prompt(
            {
                "A #1": {"src": "/s/t1.md"},
                "(the whole tick in /s/t2.md)": {"src": "/s/t2.md", "whole": True},
                "B #2": {"src": "/s/t1.md"},
            }
        )
        self.assertEqual(
            p.splitlines(),
            [
                "/rocke-monitor tick given=/s/t1.md,/s/t2.md",
                "The orchestrator ran rk tick and the tick triage did not answer: handle the whole tick in that file as "
                "the tick procedure says, from step 2.",
                "Then handle these lines too as the tick procedure says:",
                "A #1",
                "B #2",
            ],
        )


class TestClaudeSettings(Base):
    def test_effort_and_the_sdk_without_it(self):
        sdk = FakeSDK()
        self.assertEqual(orchestrator.options(self.cfg, sdk, None).effort, "high")
        self.assertEqual(
            orchestrator.options(
                dict(
                    self.cfg,
                    claude={"effort": "low"},
                    effort={"orchestrator": "xhigh"},
                ),
                sdk,
                None,
            ).effort,
            "xhigh",
        )  # features.review_automation.orchestrator.effort, whatever the base of the other roles says

        class Old(FakeSDK):
            def __init__(self):
                super().__init__()
                base = self.ClaudeAgentOptions

                def options(**kw):
                    if "effort" in kw:
                        raise TypeError(
                            "__init__() got an unexpected keyword argument 'effort'"
                        )
                    return base(**kw)

                self.ClaudeAgentOptions = options

        self.assertFalse(hasattr(orchestrator.options(self.cfg, Old(), None), "effort"))

    def test_the_model_roles(self):
        self.assertEqual(
            config.role_effort(self.cfg, "orchestrator"),
            ("high", "features.review_automation.orchestrator.effort"),
        )
        self.assertEqual(
            config.role_effort(dict(self.cfg, claude={"effort": "MAX"}), "audit"),
            ("max", "claude.effort"),
        )  # a role with no effort of its own: the base
        self.assertEqual(
            [
                config.claude_model(self.cfg, r)
                for r in ("orchestrator", "audit", "deep")
            ],
            ["opus", "opus", "opus"],
        )
        cfg = dict(
            self.cfg,
            orchestrator={"model": "sonnet"},
            models=dict(self.cfg["models"], audit="claude", deep="claude-opus-5-5[1m]"),
        )
        self.assertEqual(
            [
                config.claude_model(cfg, r)
                for r in ("orchestrator", "audit", "deep", "verify")
            ],
            ["sonnet", "sonnet", "claude-opus-5-5[1m]", "sonnet"],
        )
        self.assertEqual(
            config.model_for(cfg, "deep"),
            {"by": "claude", "model": "claude-opus-5-5[1m]"},
        )
        self.assertEqual(
            config.model_for(self.cfg, "tick")["by"], "none"
        )  # the default: the rule filter
        self.assertEqual(
            config.model_for(dict(self.cfg, models={"tick": "worker"}), "tick")["by"],
            "worker",
        )
        self.assertEqual(
            config.model_for(dict(self.cfg, models={"tick": ""}), "tick")["by"], "none"
        )
        bad = dict(
            self.cfg,
            models={"audit": "two words", "deep": 5, "tick": "worker"},
            claude={"effort": "extreme"},
        )
        problems = "\n".join(config.problems(bad))
        self.assertIn(
            "features.review_automation.passes.verification_pass.model is 'two words'; use a model as Claude Code "
            "names it",
            problems,
        )
        self.assertIn(
            "features.review_automation.passes.code_quality_pass.model is 5", problems
        )
        self.assertIn(
            "claude.effort is 'extreme'; it must be one of low, medium, high, xhigh, max",
            problems,
        )
        self.assertNotIn("models.tick", problems)


if __name__ == "__main__":
    unittest.main()
