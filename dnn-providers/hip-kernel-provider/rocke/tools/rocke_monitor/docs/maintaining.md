<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Maintaining the package

For the people who change the package: what its folder has, the checks to run before you push a change, its
generated parts, how to build the package and its tarball, and how to publish it. [README.md](../README.md) is the
reference for running the monitor, and [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md) builds an image from the
package.

## The package folder

| Path | What it is |
|---|---|
| `rkmon.py`, `rkmon_lib/` | the toolkit: the monitor's command, `rk` (`rk --help` lists every command), and the modules `rkmon` runs on the host |
| `rk` | the monitor's command, in POSIX sh: it runs `rkmon.py` with python3 |
| `docker/` | the container: `rkmon`, the wrapper of docker that the host runs; the image's recipe (`Dockerfile`) and its build helpers (`pkgs.sh`, `param.sh`, `fetch.sh`, `masked.sh`, the ROCm, atlassian and Claude Agent SDK installers, `claude_cli.py`); the entry point (`entrypoint.sh`) and the steps it runs as root (`rk_ids.py`, `rk_layout.py`, `rk_root.py`); the write broker (`rk_write.py`), the PR-code launcher (`rk_prcode.py`), and the container's `gh`, which counts each request to GitHub against `integrations.github_cli.max_requests_per_hour` (`rk_gh.py`); the compose file, and an example of the optional host mounts (`compose.local.example.yaml`) |
| `schema/` | the JSON Schemas of the build and deploy settings files and of the internal state; `layout1/`, layout 1's deploy settings schema and its `config.defaults.json`, which `rkmon migrate` and the tests read (`rkmon_lib/schema.py`, `rkmon_lib/from_layout1.py`) |
| `config.defaults.json`, `config.json` | every key of the configuration, with its default; `config.json` is blank, and the build writes it blank whatever the tree has |
| `tools/` | the job scripts, the review driver, the admin dashboard's starter (`rkweb.py`), the web host's server, a running container's settings as a deploy settings file, and the round trip that checks it (`translate_production.py`, `round_trip.py`), and the maintainers' tools this guide names |
| `worker/` | the tools the monitor's jobs run outside the toolkit ([worker/README.md](../worker/README.md) lists each, with what calls it); each start of the container copies them to the working folder's `tools/` (`rk worker-tools --install`) |
| `repository/` | the repository defaults (`rkmon_lib/repository_defaults.py`): what belongs to one repository rather than to the monitor, in `<org>__<repo>/`. They are the values it gives package keys, such as its CI job names and its test tiers, for the folders of the repository they were written for (`defaults.json`), and its repository profile: the first review pass's settings and rules for it (`drafter.json`, `drafter_rules.md`). Any other repository or folder takes the generic values and the generic profile, and `rk doctor` and `rkmon check` warn. The package ships its home repository's alone |
| `requirements/` | the checks' Python environments, each installed from its lock ([requirements/README.md](../requirements/README.md)) |
| `agentic/` | the Claude Code files: the `/rocke-monitor` skill, the reviewer team members and the review book, as templates the install fills in ([agentic/README.md](../agentic/README.md), and [Changing the agentic files](#changing-the-agentic-files)) |
| `.claude/` | the setup skill, `/rkmon-setup`, which a deployer's own Claude Code session runs; the image leaves it out. Git takes each skill the docs' index lists (DOCS in `tools/generated.py`) even in a repository that leaves out every `.claude/` folder, and leaves out the rest: a skill of your own, and Claude Code's local files, such as `settings.local.json` (`.gitignore`) |
| `README.md`, `QUICK_BUILD_GUIDE.md`, `QUICK_DEPLOY_GUIDE.md`, `QUICK_USER_GUIDE.md`, `docs/` | the docs ([the README's list](../README.md#documentation)) |
| `tests/`, `conftest.py` | the offline tests and the fakes they use; `tests/container/`, the container tests. `conftest.py` keeps pytest, run from rocKE or the repository's root, from collecting them (`RKMON_PYTEST=1` lets it) |
| `.gitignore`, `.dockerignore` | what git leaves out, and what the image's build context leaves out |
| `VERSION` | the package's version: the day of its release |

## Before you push a change

No CI job or hook runs this package's tests or its privacy check: run them yourself, in the package folder.

### The tests

`python3 -B rkmon.py selftest` (or `./rk selftest`) runs the suite, offline and with no secrets. It runs in a home
folder of its own (HOME, with the XDG folders in it), with a TMPDIR and a monitor home (RKMON_HOME) of their own
there, each new and empty, all removed after: a test that leaves anything there, or that writes in the package folder
(an ignored file too), fails the run, which names each entry. The package build, an update's check of the code it
installs and the image's test run the suite the same way (`rkmon_lib/selftest.py`).
`python3 -B rkmon.py selftest tests.test_selftest` runs one module the same way (or one class or test, by its
unittest name). `--jobs N` runs the whole suite's modules N at a time, the largest first, each in a home folder
of its own.

None of your logins', keys', gateways', git's or the monitor's settings reach the tests, nor your Windows host's name
(`TEST_ENV_DROP` in `rkmon_lib/selftest.py`). Three variables do reach the tests, so they can find their tools: the
tests that run the checks venv's pytest take that venv from RKMON_VENV, the probes' torch test its python from
RKMON_TORCH_PY, and the header test its black from RKMON_BLACK
(`RKMON_VENV=WORKING_DIR/venv python3 -B rkmon.py selftest`). Without them, those tests are skipped.

### The privacy check

```sh
python3 -B -m rkmon_lib.privacy --deny DENY_LIST .
```

It fails on internal links and hosts, anyone's name, login, email or handle, tokens, ticket keys, PR numbers (`#N`,
`PR N` and PR links in every file; in code and tests also a bare number the deny list gives as a PR), wiki
references, GPU product names, absolute performance numbers, dated narratives (a line or comment that starts with a
date and a colon), attributions (a person's request or answer as the reason, a decision with its date), the
monitor's concrete incident ids, and a settings file of either type, build or deploy, whatever its name. Its report
gives labels and counts only; `--where` adds the file, the line and the label of each match, never its value.

It cannot pass while checking nothing: a source it is given that it cannot read or that gives no value, and a path
that does not exist or has no file, are NOT CHECKED, which fails it. Over the package folder it also needs a file read
in each of its docs, schemas, skills and tests, and renders every settings template (the build template, and the
deploy template of each profile) as the folder's own code, so run it in the folder: a template fails on an absolute
path, and on a private key whose value is not blank, a `<...>` placeholder or the key's own default.

The deny list (`--deny`, required; kept outside the package and never committed) has the values to look for, one a
line: a value (a login, a name, a host, a path), `pr:N` (a PR number, looked for as a bare number in code and tests)
or `domain:NAME` (a company domain: every host under it is internal); a line that starts with `#` is a comment.
`--seed-deny DENY_LIST`, with `--config`, `--state` and `--workspace` naming a monitor's configuration, its state and
its Claude workspace, adds their values to the list once; the build then reads none of their files. The package build
and the sync run the check with that list before their tests, and `tests/test_privacy.py` scans the package's own
files for the build machine's names and the generic patterns. `rkmon_lib/privacy.py` says the rest.

`tests/test_design_refs.py` fails on a line that points at a numbered section of another document, such as `design N.N`
or `(N.N)`, instead of saying the thing itself; a doc's own sections pass written `section N.N` or `§N`, and a version
passes next to its program's name.

### Formatting and the headers

The package's Python files, the extensionless entry points with a python `#!` line among them, are formatted with
black, at the release the repository's pre-commit hooks pin and with its default settings, so the package's folder in
the repository and this one have the same files. `RKMON_BLACK=BLACK python3 -B rkmon.py selftest tests.test_headers`
checks the formatting, and rocKE's two-line header on every file that can have a comment;
`tools/make_package.py --black BLACK` formats the built folder.

### The docs' lints

```sh
python3 -B tools/doc_lints.py [--lint NAME]... FILE...
```

The lints read the docs, the schemas' texts, the skill files and the messages the code prints: a fact written outside
its home, a key no settings file has, a word the docs keep out, a command another one replaced, a quoted output the code
does not print, a link to nothing, and a placeholder the README's glossary lacks. Each problem names its file, its line
and what to write instead. It exits 0 when there is none, 1 when there is one, and 2 on a usage error; `--lint NAME`
runs the lints it names. Their lists are `tests/doc_fact_homes.yaml`, `tests/doc_keys.yaml`, `tests/doc_words.yaml`,
`tests/doc_old_commands.yaml` and `tests/doc_placeholders.yaml`. `tests/doc_lint_files.txt` lists the files the suite
lints whole (`tests/test_doc_lints.py`): every Markdown file of the package, which the suite checks, and another file
once it passes. The tool's docstring says what each lint reads.

### The container tests

Optional, with Docker or Podman. The checks of tarball.py run with no network; for image.py's stages,
`rkmon image test --help` says the network each runs on. Building the images fetches their packages.

- `tests/container/image.py` tests an image of the package, in stages: `rkmon image test --help` lists them, in
  order, with what each checks. Several of them share one deployment inside the image, made from
  `tests/container/deploy.yaml` as the deploy path makes one; `tests/container/stages.py` names them and says how.
  `--settings FILE` builds the image first, `--no-build --image IMAGE` tests one built already, and `--stages` runs
  some; `rkmon image test --settings FILE` runs it on the image your build settings name. To see a stage catch a
  bug you plant on purpose, plant it where the stage's code comes from: the stages that deploy the image with
  `rkmon` run the code of the package folder the test runs from (`--package DIR`), which `rkmon start` and
  `rkmon apply` install over the image's own, and the others run the image's.
- For a release, `tests/container/tarball.py --tarball FILE --dist DIR` unpacks the released tarball on Ubuntu 22.04
  and 24.04 and Debian, with Python 3 and no other tool, as a user who is not root: exactly the built folder's files,
  with no owner of the build machine, and the privacy check on what it unpacked.

## The generated parts

Some of the package's text is made from a source: the docs' generated sections and the generated files.
`tools/generated.py` lists them (its SECTIONS and FILES), each with its source. They are committed with the code, so a
clone of the package's repository builds, deploys and updates with nothing else. Every build, and
`tests/test_generated.py`, stop when one is out of step with its source, naming it and the command that writes it
again:

```sh
python3 -B tools/make_package.py --refresh-generated
```

It runs alone and builds nothing: it writes each generated section and file whose text differs from what its source
makes, for you to review and commit. Change the source, never the generated text. A generated section sits between
two marker comments, the first naming the section, its source and that command; `tools/generated.py` describes them.

`VERSION` is the exception: no source makes it. It is the day of the package's release, and only a release build
writes it ([Building the package and its tarball](#building-the-package-and-its-tarball)).

## Building the package and its tarball

```sh
python3 -B tools/make_package.py --out DIR --deny DENY_LIST [--black BLACK] [--no-tests] [--new-version]
```

It makes the package from a clone of its repository: the files git lists there (tracked, or untracked and not ignored,
but a Python environment's), with a blank `config.json`; it refuses a folder where git tracks no file. A package
update (`rkmon package update`) copies a package by the same rule (`rkmon_lib/package_files.py`). The build's
steps, in order:

1. the generated parts, compared with their sources; with `--new-version`, VERSION, today's date, in the built folder;
2. black, when `--black` names it;
3. rocKE's header check on every file that can have a comment;
4. the privacy check over every file and every settings template, as the built folder's own code: its report is
   `DIR/privacy_report.txt`, labels and counts only, and a failure stops the build;
5. the package's tests, inside the built folder, on a PATH and a bytecode folder of the build's own, so the caller's
   PATH, venv or `python` makes no difference (`--no-tests` skips them);
6. with `--new-version`, the built folder's VERSION into the tree, for you to review and commit: a release;
7. the tarball.

The folder is `DIR/rocke_monitor/`, and the tarball `DIR/rocke_monitor-<VERSION>.tar.gz`, named by the built
folder's VERSION (YYYY.MM.DD), as the image built from it is tagged, with the folder in it as `rocke_monitor/`.
Unpacked in a new folder, it is the package [QUICK_BUILD_GUIDE.md](../QUICK_BUILD_GUIDE.md) starts from.
`--install-into SANDBOX_CHECKOUT` copies the folder over the package in a folder below the top of a checkout with no
git remote, found as `rkmon package update` finds the package in a fetched commit, and shows
`git status --porcelain --ignored` there; it never commits or pushes. The docstring of `tools/make_package.py` says
what the build reads.

## Changing the agentic files

The agentic files are the package's own: change them where they are, in `agentic/`, and commit them; the build copies
them as they are. `tests/test_agentic.py` checks them: they name no `rk` command, tick line or `tools/` script the
package lacks (except the names in its EXEMPT_NAMES, each with its reason), and keep none of a live skill's own fill-in
tokens (`<paths.root>`, `<wsl_distro>`, `<wsl_user>`) and no reference to `.claude/project_config.md`: a deployment's
settings are the monitor's configuration, and the install fills the placeholders from it. They name none of the
host install's services or their switches, no user's systemd unit, and nothing of WSL (its keep-alive, `wsl.exe`).
Each word in capitals in angle brackets in them is a placeholder (`agentic.PLACEHOLDERS`) or one of the example
words the test lists: a misspelt placeholder, which the install leaves as written, fails it.

The team members' rows of the workspace's dispatch table, `agentic/team_members/dispatch_rows.md`, are a generated
section ([The generated parts](#the-generated-parts)): each row is made from its team member file's front matter (its
role, name, context and domain). Change the front matter, then run the refresh.

## Publishing the package

Make every change in the package. A change made in a monitor's own checkout of the toolkit (a live toolkit) comes in
by hand, one commit at a time. The package has a repository of its own, with a history of its own, which is published
from the package and never from a live toolkit:

```sh
PACKAGE_CHECKOUT/tools/publish_package.py PACKAGE_REPO -- --deny DENY_LIST
```

`tools/publish_package.py` publishes the committed HEAD of the checkout it lives in. It refuses a work tree with
changes, and a HEAD whose `config.json` is not the blank configuration (a live toolkit's tree has a machine's settings
there). Then it runs the sync's steps below in the package repository, and commits "rocKE monitor package:
publish COMMIT" as `rkmon <rkmon@localhost>`. It exits 0 when it published or found nothing to publish, 1 on a
refusal or a failed step, and 2 on a usage error.

## Syncing from a live toolkit

`tools/sync_from_live.py` brings a reviewed branch of a live toolkit's checkout into the package repository as a
file-level squash: one commit per sync, never a merge, since a live toolkit's history has its machine's
configuration. The publish above runs the same steps on the package's own commit. From the package repository's top:

```sh
tools/sync_from_live.py LIVE_CHECKOUT SYNC_BRANCH -- --deny DENY_LIST
```

It refuses a work tree with changes, and a target that shares the live checkout's repository. It replaces the
repository's tracked files with the branch's tracked tree and a blank `config.json`, runs the privacy check over the
files and over the staged change (the lines it adds), runs the tests, and commits as `rkmon <rkmon@localhost>`. When a
step fails it commits nothing, and leaves the staged sync for you to inspect (`git reset --hard` drops it).
